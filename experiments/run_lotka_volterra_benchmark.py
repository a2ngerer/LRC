"""Train one (wiring, cell, seed) on the predator-prey next-step task.

Sequence-rollout regression: the RNN is unrolled over the full trajectory
(tf.keras.layers.RNN carries the hidden state), so unlike the neural_ode T=1
Euler harness the cNCP delayed-state edges actually receive gradient. See
src/tasks/lotka_volterra/datasets.py for the why.

Two evaluations are stored per run:
  * teacher-forced  -- predict step k+1 from the TRUE state at step k;
  * closed-loop     -- feed only the initial state, then feed the model's own
                       predictions back (true test of whether the dynamics were
                       learned; this is what the phase-space overlay shows).

De-normalised ground-truth / teacher-forced / closed-loop trajectories for the
test split are written into the JSON so plot_lotka_volterra.py needs no retrain.

Usage:
    uv run python experiments/run_lotka_volterra_benchmark.py \
        --wiring cncp --cell cfc_lrc --seed 0 --epochs 300
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import tensorflow as tf

from src.wirings import param_counts
from src.tasks.lotka_volterra.datasets import load_lotka_volterra, reference_frame
from src.tasks.lotka_volterra.model import build_lotka_volterra_model, LV_WIRINGS
from src.tasks.active_sensing.model import TBT_MODES
from src.benchmark.tracking import keras_callbacks, track


def closed_loop_rollout(model, data, needs_loc: bool = False) -> np.ndarray:
    """Autoregressively roll the model from each test initial state.

    Returns the generated NORMALISED trajectory (N, seq_len+1, 2): column 0 is
    the given initial state, columns 1.. are the model's own predictions fed
    back one step at a time. Re-running the RNN over the growing history each
    step reproduces training-time state reconstruction (state at position k is a
    function of inputs 0..k), so the last output is the genuine one-step-ahead
    prediction given that history.

    For the tbt_cncp* wirings (needs_loc) the reference-frame code is recomputed
    from the model's own predicted state at every step -- the location is a
    deterministic function of the state, so closed-loop stays self-consistent.
    """
    n_steps = data.seq_len
    dt_col = data.test_t                       # (N, seq_len, 1) constant dt
    history = [data.test_x[:, 0, :].astype(np.float32)]   # (N, 2) initial state
    for _ in range(n_steps):
        length = len(history)
        seq = np.stack(history, axis=1)                    # (N, length, 2)
        t_seq = dt_col[:, :length, :]                      # (N, length, 1)
        if needs_loc:
            loc = reference_frame(seq)                     # (N, length, L)
            pred = model((tf.constant(seq), tf.constant(t_seq),
                          tf.constant(loc)), training=False).numpy()
        else:
            pred = model((tf.constant(seq), tf.constant(t_seq)),
                         training=False).numpy()           # (N, length, 2)
        history.append(pred[:, -1, :].astype(np.float32))  # next state
    return np.stack(history, axis=1)                       # (N, seq_len+1, 2)


def _round(a, nd=5):
    return np.round(np.asarray(a, dtype=np.float64), nd).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiring", required=True, choices=LV_WIRINGS)
    ap.add_argument("--cell", default="cfc_lrc")
    ap.add_argument("--system", default="periodic_predator_prey",
                    help="neural_ode 2D system (e.g. periodic_predator_prey, duffing)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-trajectories", type=int, default=80)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--outdir", default="results/lotka_volterra")
    ap.add_argument("--elastance-type", default="asymmetric",
                    help="forwarded to LRC-family cells (RQ2/RQ5 mechanism)")
    ap.add_argument("--verbose", type=int, default=0)
    ap.add_argument("--wandb", action="store_true",
                    help="log this run to Weights & Biases (needs the "
                         "'tracking' extra; use WANDB_MODE=offline on cluster)")
    args = ap.parse_args()

    cell_kwargs = {}
    if args.cell in ("cfc_lrc", "lrc", "lrc_ar"):
        cell_kwargs["elastance_type"] = args.elastance_type

    needs_loc = args.wiring in TBT_MODES
    data = load_lotka_volterra(n_trajectories=args.n_trajectories,
                               seq_len=args.seq_len, system=args.system)
    print(f"[{args.system}] train {data.train_x.shape[0]} / test "
          f"{data.test_x.shape[0]} trajectories, seq_len {data.seq_len}, "
          f"dt {data.dt:.4f}, loc_dim {data.loc_dim}")

    model = build_lotka_volterra_model(
        args.wiring, args.cell, size=args.size, seed=args.seed,
        feature_size=data.feature_size, loc_dim=data.loc_dim, lr=args.lr,
        **cell_kwargs)
    pc = param_counts(model)
    params = pc['params_effective']
    print(f"{args.wiring}/{args.cell} seed {args.seed}: {params} params "
          f"(needs_loc={needs_loc})")

    # tbt_cncp* wirings take a 3rd input (the reference-frame code); the topology
    # arms (dense/ncp/cncp) take only (state, time).
    train_in = ((data.train_x, data.train_t, data.train_loc) if needs_loc
                else (data.train_x, data.train_t))
    test_in = ((data.test_x, data.test_t, data.test_loc) if needs_loc
               else (data.test_x, data.test_t))

    wandb_config = {
        "task": "lotka_volterra", "system": args.system,
        "cell": args.cell, "wiring": args.wiring, "seed": args.seed,
        "size": args.size, "lr": args.lr, "batch_size": args.batch_size,
        "epochs": args.epochs, **pc,
    }
    tag = f"{args.cell}_{args.wiring}_seed{args.seed}"
    with track(args.wandb, group=args.cell, job_type=args.wiring, name=tag,
               config=wandb_config) as run_:
        hist = model.fit(
            train_in, data.train_y,
            batch_size=args.batch_size, epochs=args.epochs,
            validation_data=(test_in, data.test_y),
            verbose=args.verbose,
            callbacks=keras_callbacks(run_))

    # Teacher-forced: MSE/MAE in normalised space + de-normalised prediction.
    tf_mse, tf_mae = model.evaluate(test_in, data.test_y, verbose=0)
    tf_pred_norm = model.predict(test_in, verbose=0)                    # (N,T,2)

    # Closed-loop rollout from the initial state only.
    gen_norm = closed_loop_rollout(model, data, needs_loc=needs_loc)    # (N,T+1,2)
    truth_norm = (data.test_traj - data.mean) / data.std
    cl_mse = float(np.mean((gen_norm[:, 1:] - truth_norm[:, 1:]) ** 2))
    # Per-step closed-loop MSE (error growth vs rollout horizon) for the plot.
    cl_mse_per_step = np.mean((gen_norm[:, 1:] - truth_norm[:, 1:]) ** 2,
                              axis=(0, 2))                               # (T,)

    # Persistence baseline (predict next == current), teacher-forced, normalised.
    persistence_mse = float(np.mean((data.test_x - data.test_y) ** 2))

    # De-normalised trajectories aligned with the ground truth for plotting.
    tf_traj = np.concatenate(
        [data.test_traj[:, :1, :], data.denormalise(tf_pred_norm)], axis=1)
    cl_traj = data.denormalise(gen_norm)

    print(f"  teacher-forced MSE {float(tf_mse):.5f} | closed-loop MSE "
          f"{cl_mse:.5f} | persistence {persistence_mse:.5f}")

    os.makedirs(args.outdir, exist_ok=True)
    record = {
        "wiring": args.wiring, "cell": args.cell, "seed": args.seed,
        "system": args.system,
        "size": args.size, "epochs": args.epochs, "lr": args.lr,
        **pc, "dt": data.dt, "seq_len": data.seq_len,
        "n_test": int(data.test_x.shape[0]),
        "teacher_forced_mse": float(tf_mse), "teacher_forced_mae": float(tf_mae),
        "closed_loop_mse": cl_mse, "persistence_mse": persistence_mse,
        "closed_loop_mse_per_step": _round(cl_mse_per_step, 6),
        "val_loss_curve": _round(hist.history["val_loss"], 6),
        # Trajectories (de-normalised, raw prey/predator) for the overlay plots.
        "ground_truth_traj": _round(data.test_traj),
        "teacher_forced_traj": _round(tf_traj),
        "closed_loop_traj": _round(cl_traj),
    }
    fname = f"{args.cell}_{args.wiring}_seed{args.seed}.json"
    path = os.path.join(args.outdir, fname)
    with open(path, "w") as f:
        json.dump(record, f)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
