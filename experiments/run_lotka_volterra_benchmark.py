"""Train one (wiring, cell, seed) on the predator-prey next-step task.

Sequence-rollout regression: the RNN is unrolled over the full trajectory
(tf.keras.layers.RNN carries the hidden state), so unlike the neural_ode T=1
Euler harness the cNCP delayed-state edges actually receive gradient. See
src/tasks/lotka_volterra/datasets.py for the why.

Two evaluations are stored per run, each on the val and the test split
(keys val_* / test_*, normalised units; unprefixed keys repeat the test values
for older readers):
  * teacher-forced  -- predict step k+1 from the TRUE state at step k;
  * closed-loop     -- feed only the initial state, then feed the model's own
                       predictions back (true test of whether the dynamics were
                       learned; this is what the phase-space overlay shows).
Model selection (HPO) reads the val keys only. best_val_epoch is the epoch with
the lowest validation loss; best_test_teacher_forced_mse is the test loss at
that epoch, never a minimum over the test curve.

Rollout endpoints (#20): rollout_metrics_val / rollout_metrics_test hold the
src/evaluation/rollout_metrics dict of the closed-loop rollouts (VPT(0.5) as the
primary endpoint, NRMSE@16/32/128, orbit error, invariant drift, divergence
class rates); norm_stats lets experiments/rollout_metrics.py recompute it.

Forcing (#25): --forcing teacher (default) trains the plain model on the true
state at every step, as before. --forcing mixed trains the same weights through
a ForcedRolloutCell, input alpha * y_t + (1 - alpha) * yhat_{t-1}, with alpha
following AlphaSchedule (1 -> --alpha-min over the first half of the epochs;
the per-epoch values are stored as alpha_curve; loss and val_loss_curve are at
that epoch's alpha). Teacher-forced metrics always use the plain model; the
closed-loop rollout of every protocol is the wrapper at alpha = 0 (one O(T)
pass), closed_loop_rollout stays the reference for the tbt_cncp* and
cncp2_multi wirings. Mixed-forcing results get a `_mixed` filename suffix.

Data: 48/16/16 whole trajectories at the default 80, drawn with data seed
DATA_SEED_BASE + --seed (+ --data-seed-offset for tuning runs), so all arms of
one replicate share a draw and every replicate gets a fresh one.

De-normalised ground-truth / teacher-forced / closed-loop trajectories for the
test split are written into the JSON so plot_lotka_volterra.py needs no retrain.

Capacity: --param-budget sizes the arm on its own width knob to the EFFECTIVE
parameter budget (masked-off weights excluded); --size fixes the knob. The run
records size, both counts and the deviation from the budget (flagged above 5 %).

Usage:
    uv run python experiments/run_lotka_volterra_benchmark.py \
        --wiring cncp --cell cfc_lrc --seed 0 --epochs 300 --param-budget 4000
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import tensorflow as tf

from src.wirings import (CorticalColumnCell, CorticalColumnV2Cell,
                         MultiColumnV2Cell, NODE_ORDER, NODE_ORDER_V2,
                         param_counts, size_for_budget)

DEFAULT_SIZE = 64
BUDGET_TOLERANCE = 0.05
from src.tasks.lotka_volterra.datasets import (DATA_SEED_BASE,
                                               load_lotka_volterra,
                                               reference_frame)
from src.tasks.lotka_volterra.model import (
    build_lotka_volterra_model, forced_rollout_model, FORCEABLE_WIRINGS,
    LV_WIRINGS)
from src.neurons.forced_rollout_cell import AlphaSchedule
from src.tasks.person_activity.model import (
    backbone_override, backbone_suffix, effective_backbone_layers,
    effective_self_recurrent, add_multi_column_args, multi_column_config,
    multi_column_suffix, ode_unfolds_override)
from src.tasks.active_sensing.model import TBT_MODES
from src.benchmark.tracking import keras_callbacks, track
from src.benchmark.checkpoints import save_weights
from src.evaluation.rollout_metrics import make_norm_stats, rollout_metrics


def closed_loop_rollout(model, data, needs_loc: bool = False,
                        split: str = "test") -> np.ndarray:
    """Autoregressively roll the model from each initial state of ``split``
    ('val' or 'test').

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
    dt_col = getattr(data, f"{split}_t")       # (N, seq_len, 1) constant dt
    history = [getattr(data, f"{split}_x")[:, 0, :].astype(np.float32)]  # (N, 2)
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


def forced_closed_loop(view, data, split: str = "test") -> np.ndarray:
    """closed_loop_rollout via a forced_rollout_model view at alpha = 0: one
    pass over the true inputs, of which only y_0 enters (same (N, seq_len+1,
    2) normalised layout)."""
    x = getattr(data, f"{split}_x")
    pred = view((tf.constant(x), tf.constant(getattr(data, f"{split}_t"))),
                training=False).numpy()
    return np.concatenate([x[:, :1], pred], axis=1).astype(np.float32)


def split_inputs(data, split, needs_loc):
    """Keras inputs of one split; tbt_cncp* wirings take the reference-frame code
    as a 3rd input, the topology arms (dense/ncp/cncp) only (state, time)."""
    x, t = getattr(data, f"{split}_x"), getattr(data, f"{split}_t")
    return (x, t, getattr(data, f"{split}_loc")) if needs_loc else (x, t)


def evaluate_split(model, data, split, needs_loc, view=None):
    """Teacher-forced and closed-loop metrics on 'val' or 'test' (normalised
    units; NRMSE = sqrt of the pooled MSE). The rollout runs the alpha = 0
    ``view`` (forced_rollout_model) when given, else closed_loop_rollout.
    Returns (metrics, rollout)."""
    y = getattr(data, f"{split}_y")
    tf_mse, tf_mae = model.evaluate(split_inputs(data, split, needs_loc), y,
                                    verbose=0)
    gen_norm = (closed_loop_rollout(model, data, needs_loc, split)
                if view is None else forced_closed_loop(view, data, split))
    truth_norm = (getattr(data, f"{split}_traj") - data.mean) / data.std
    sq_err = (gen_norm[:, 1:] - truth_norm[:, 1:]) ** 2
    cl_mse = float(np.mean(sq_err))
    return {
        "teacher_forced_mse": float(tf_mse), "teacher_forced_mae": float(tf_mae),
        "teacher_forced_nrmse": float(np.sqrt(tf_mse)),
        "closed_loop_mse": cl_mse, "closed_loop_nrmse": float(np.sqrt(cl_mse)),
        # Per-step closed-loop MSE (error growth vs rollout horizon) for the plot.
        "closed_loop_mse_per_step": _round(np.mean(sq_err, axis=(0, 2)), 6),
        # Persistence baseline (predict next == current), teacher-forced.
        "persistence_mse": float(np.mean(
            (getattr(data, f"{split}_x") - y) ** 2)),
    }, gen_norm


def result_filename(args, multi=None):
    """Result JSON name; a set --backbone-layers adds `_bb{N}` (V2 vs V2s),
    cncp2_multi adds `_K{K}_{partition}_{coupling}_{readout}`, --forcing mixed
    adds `_mixed`."""
    return (f"{args.cell}_{args.wiring}_seed{args.seed}"
            f"{backbone_suffix(args.backbone_layers)}"
            f"{multi_column_suffix(multi or {})}"
            f"{'_mixed' if getattr(args, 'forcing', None) == 'mixed' else ''}"
            ".json")


class _StateTap(tf.keras.layers.AbstractRNNCell):
    """Read-only wrapper: runs a built cell and emits all its new states.

    Wrapped in a fresh RNN layer it returns the per-step states of the
    original (shared-weight) cell; it creates no variables and changes no state
    of the trained model.
    """

    def __init__(self, cell):
        super().__init__()
        self.cell = cell

    @property
    def state_size(self):
        return self.cell.state_size

    @property
    def output_size(self):
        return int(np.sum(np.atleast_1d(self.cell.state_size)))

    def build(self, input_shape):
        self.built = True

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return self.cell.get_initial_state(inputs=inputs, batch_size=batch_size,
                                           dtype=dtype)

    def call(self, inputs, states):
        _, new = self.cell(inputs, states)
        new = list(new) if isinstance(new, (list, tuple)) else [new]
        return tf.concat(new, axis=-1), new


def _state_names(cell):
    if isinstance(cell, MultiColumnV2Cell):
        return [f"{node}{k}" for k in range(cell.K) for node in NODE_ORDER_V2]
    if isinstance(cell, CorticalColumnV2Cell):
        return list(NODE_ORDER_V2)
    if isinstance(cell, CorticalColumnCell):          # also tbt_cncp*
        return list(NODE_ORDER)
    return [f"state{i}" for i in range(len(np.atleast_1d(cell.state_size)))]


class StabilityProbe(tf.keras.callbacks.Callback):
    """--probe-stability (spec 2026-10-21 B.5, R6): per-state L2 norms of the
    model's RNN states on one fixed training batch at init, after exactly one
    optimiser step and after the first epoch, plus the epoch-1 train loss.
    ``held_out`` batches (runner: val and test) are probed at the same points
    under suffixed keys (norms_init_val, ...); 'finite' covers all of them.

    Each norm is the L2 norm over the unit axis, max over batch and time,
    keyed "<rnn layer>/<state>" (cncp: the eight nodes, cncp2: G/S/D/R, ncp
    and dense layers: state0..). The probe only runs forward passes with
    training=False through read-only taps, so it does not alter the training
    trajectory beyond the extra forward computation.
    """

    def __init__(self, model, batch_inputs, held_out=None):
        super().__init__()
        self._taps = []
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.RNN):
                feed = tf.keras.Model(model.inputs, layer.input)
                tap = tf.keras.layers.RNN(_StateTap(layer.cell),
                                          return_sequences=True)
                self._taps.append((layer.name, _state_names(layer.cell),
                                   np.atleast_1d(layer.cell.state_size),
                                   feed, tap))
        # key suffix -> batch: "" is the training batch (spec R6); held_out
        # adds e.g. "_val"/"_test" (norms_init_val, ...) on held-out batches.
        self._batches = {"": batch_inputs,
                         **{f"_{k}": v for k, v in (held_out or {}).items()}}
        self.result = {}

    def _record(self, key):
        for suffix, batch in self._batches.items():
            self.result[key + suffix] = self._norms(batch)

    def _norms(self, batch):
        out = {}
        for name, states, sizes, feed, tap in self._taps:
            seq = tap(feed(batch, training=False), training=False)
            for state, part in zip(states, tf.split(seq, sizes.tolist(), -1)):
                out[f"{name}/{state}"] = float(tf.reduce_max(
                    tf.norm(part, axis=-1)))
        return out

    def on_train_begin(self, logs=None):
        self._record("norms_init")

    def on_train_batch_end(self, batch, logs=None):
        if "norms_step1" not in self.result:
            self._record("norms_step1")

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0:
            self._record("norms_epoch1")
            self.result["tf_loss_epoch1"] = float(logs["loss"])
            values = [v for k, norms in self.result.items()
                      if k.startswith("norms_") for v in norms.values()]
            self.result["finite"] = bool(np.all(np.isfinite(
                values + [self.result["tf_loss_epoch1"]])))


def _round(a, nd=5):
    return np.round(np.asarray(a, dtype=np.float64), nd).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiring", required=True, choices=LV_WIRINGS)
    ap.add_argument("--cell", default="cfc_lrc")
    ap.add_argument("--system", default="periodic_predator_prey",
                    help="neural_ode 2D system (e.g. periodic_predator_prey, duffing)")
    ap.add_argument("--seed", type=int, default=0)
    data_seed = ap.add_mutually_exclusive_group()
    data_seed.add_argument("--data-seed", type=int, default=None,
                           help="trajectory draw + split seed (default "
                                "DATA_SEED_BASE + offset + --seed)")
    data_seed.add_argument("--data-seed-offset", type=int, nargs="?",
                           const=1000, default=0,
                           help="shift of the derived data seed; tuning runs "
                                "pass it (bare flag = 1000) so their draws are "
                                "disjoint from the final replicates' (offset 0)")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--forcing", choices=("teacher", "mixed"), default="teacher",
                    help="training protocol (#25): teacher = true state at "
                         "every step (unchanged); mixed = alpha * y_t + "
                         "(1 - alpha) * yhat_{t-1}, alpha 1 -> --alpha-min "
                         "over the first half of the epochs (FORCEABLE_WIRINGS)")
    ap.add_argument("--alpha-min", type=float, default=0.2,
                    help="final alpha of --forcing mixed")
    cap = ap.add_mutually_exclusive_group()
    cap.add_argument("--size", type=int, default=None,
                     help="the arm's width knob (see build_lotka_volterra_model); "
                          f"default {DEFAULT_SIZE} when no budget is given")
    cap.add_argument("--param-budget", type=int, default=None,
                     help="size the arm to this EFFECTIVE parameter budget "
                          "(closest size on the arm's own knob)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clipnorm", type=float, default=None,
                    help="clip every gradient tensor to this L2 norm (the "
                         "optimizer's clipnorm); default: no clipping")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-trajectories", type=int, default=80)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--outdir", default="results/lotka_volterra")
    ap.add_argument("--elastance-type", default="asymmetric",
                    help="forwarded to LRC-family cells (RQ2/RQ5 mechanism)")
    ap.add_argument("--ode-unfolds", type=int, default=6,
                    help="explicit Euler sub-steps per input step for the "
                         "numerical ODE cells (ltc, lrc, lrc_pm, mm_ltc, "
                         "mm_lrc); ignored for closed-form and discrete cells")
    ap.add_argument("--dt-scale", type=float, default=1.0,
                    help="multiply the elapsed time by this factor before it "
                         "reaches any cell; 1.0 = unchanged")
    ap.add_argument("--backbone-layers", type=int, default=None,
                    help="CfC-family backbone depth for EVERY wiring (overrides "
                         "the default 1 of every wiring; 0 = no backbone); "
                         "default: unchanged per-wiring behaviour")
    ap.add_argument("--head-activation", default="none", choices=("none", "tanh"),
                    help="activation on the ff1/ff2 heads of the CfC-family cells "
                         "(bb0 attribution control); default: linear heads")
    ap.add_argument("--no-self-recurrence", dest="self_recurrent",
                    action="store_false",
                    help="ncp x CfC family: drop the diagonal self-recurrence "
                         "of the NCP layers (issue #23 stability pilot)")
    ap.add_argument("--verbose", type=int, default=0)
    ap.add_argument("--probe-stability", action="store_true",
                    help="record per-state norms at init, after one step and "
                         "after epoch 1, plus the epoch-1 train loss "
                         "(result key 'stability', spec 2026-10-21 R6)")
    ap.add_argument("--no-save-weights", dest="save_weights",
                    action="store_false",
                    help="do not write <stem>.weights.h5 next to the result JSON")
    add_multi_column_args(ap)
    ap.add_argument("--wandb", action="store_true",
                    help="log this run to Weights & Biases (needs the "
                         "'tracking' extra; use WANDB_MODE=offline on cluster)")
    args = ap.parse_args()
    if args.size is None and args.param_budget is None:
        args.size = DEFAULT_SIZE
    mixed = args.forcing == "mixed"
    if mixed and args.wiring not in FORCEABLE_WIRINGS:
        ap.error(f"--forcing mixed needs one of {FORCEABLE_WIRINGS}")
    if not 0.0 <= args.alpha_min <= 1.0:
        ap.error("--alpha-min must lie in [0, 1]")

    cell_kwargs = {}
    if args.cell in ("cfc_lrc", "lrc", "lrc_ar"):
        cell_kwargs["elastance_type"] = args.elastance_type
    cell_kwargs.update(backbone_override(args.cell, args.backbone_layers))
    if args.head_activation != "none":     # non-CfC cells reject the kwarg
        cell_kwargs["head_activation"] = args.head_activation
    cell_kwargs.update(ode_unfolds_override(args.cell, args.ode_unfolds))
    bb_effective = effective_backbone_layers(args.wiring, args.cell, cell_kwargs)
    self_rec = effective_self_recurrent(args.wiring, args.cell,
                                        args.self_recurrent)
    multi = multi_column_config(args)

    needs_loc = args.wiring in TBT_MODES
    data_seed = (args.data_seed if args.data_seed is not None
                 else DATA_SEED_BASE + args.data_seed_offset + args.seed)
    data = load_lotka_volterra(n_trajectories=args.n_trajectories,
                               seq_len=args.seq_len, seed=data_seed,
                               system=args.system)
    split = [len(data.train_x), len(data.val_x), len(data.test_x)]
    print(f"[{args.system}] train/val/test {split} trajectories (data seed "
          f"{data_seed}), seq_len {data.seq_len}, dt {data.dt:.4f}, "
          f"loc_dim {data.loc_dim}")

    def build(size):
        return build_lotka_volterra_model(
            args.wiring, args.cell, size=size, seed=args.seed,
            feature_size=data.feature_size, loc_dim=data.loc_dim, lr=args.lr,
            self_recurrent=args.self_recurrent, dt_scale=args.dt_scale,
            **multi, **cell_kwargs)

    # Capacity: a fixed width knob, or the knob closest to the effective
    # parameter budget for THIS arm.
    budget_dev = None
    if args.param_budget is not None:
        args.size, n_eff = size_for_budget(build, args.param_budget)
        budget_dev = abs(n_eff - args.param_budget) / args.param_budget
        flag = "  ABOVE TOLERANCE" if budget_dev > BUDGET_TOLERANCE else ""
        print(f"budget {args.param_budget}: size={args.size} "
              f"params_effective={n_eff} (dev {budget_dev:.1%}){flag}")
    model = build(args.size)
    model.optimizer.clipnorm = args.clipnorm      # None = no clipping
    pc = param_counts(model)
    params = pc['params_effective']
    print(f"{args.wiring}/{args.cell} seed {args.seed}: size {args.size}, "
          f"{params} effective params (raw {pc['params_raw']}, "
          f"needs_loc={needs_loc}, backbone_layers requested "
          f"{args.backbone_layers} effective {bb_effective})")

    train_in, val_in, test_in = (split_inputs(data, s, needs_loc)
                                 for s in ("train", "val", "test"))

    wandb_config = {
        "task": "lotka_volterra", "system": args.system,
        "cell": args.cell, "wiring": args.wiring, "seed": args.seed,
        "size": args.size, "param_budget": args.param_budget,
        "budget_deviation": budget_dev, "lr": args.lr,
        "clipnorm": args.clipnorm,
        "batch_size": args.batch_size, "epochs": args.epochs,
        "backbone_layers": args.backbone_layers, "self_recurrent": self_rec,
        "head_activation": args.head_activation, "data_seed": data_seed,
        "forcing": args.forcing, **pc, **multi,
        "ode_unfolds": cell_kwargs.get("ode_unfolds"), "dt_scale": args.dt_scale,
    }
    tag = result_filename(args, multi)[:-len(".json")]
    probe = None
    if args.probe_stability:
        # The first batch_size sequences of each split: train (spec R6), val
        # and test.
        tr_b, val_b, te_b = (tuple(a[:args.batch_size] for a in inputs)
                             for inputs in (train_in, val_in, test_in))
        probe = StabilityProbe(model, tr_b, {"val": val_b, "test": te_b})
    # Test teacher-forced loss per epoch, read only at the best-val epoch.
    test_curve = []
    test_tap = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: test_curve.append(
            model.evaluate(test_in, data.test_y, verbose=0)[0]))
    # --forcing mixed trains a weight-sharing view of `model`; everything else
    # (teacher-forced metrics, probe, weights) keeps reading the plain model.
    schedule = None
    train_model = model
    if mixed:
        alpha = tf.Variable(1.0, trainable=False, name="alpha")
        schedule = AlphaSchedule(alpha, args.epochs, args.alpha_min)
        train_model = forced_rollout_model(model, alpha)
    with track(args.wandb, group=args.cell, job_type=args.wiring, name=tag,
               config=wandb_config) as run_:
        hist = train_model.fit(
            train_in, data.train_y,
            batch_size=args.batch_size, epochs=args.epochs,
            validation_data=(val_in, data.val_y),
            verbose=args.verbose,
            callbacks=([schedule] if schedule else []) + keras_callbacks(run_)
            + [test_tap] + ([probe] if probe else []))
    val_curve = hist.history["val_loss"]
    best = int(np.argmin(np.nan_to_num(val_curve, nan=np.inf)))  # NaN never wins

    view = (forced_rollout_model(model, tf.Variable(0.0, trainable=False))
            if args.wiring in FORCEABLE_WIRINGS else None)
    val_metrics, val_gen = evaluate_split(model, data, "val", needs_loc, view)
    test_metrics, gen_norm = evaluate_split(model, data, "test", needs_loc,
                                            view)
    tf_pred_norm = model.predict(test_in, verbose=0)                    # (N,T,2)

    # De-normalised trajectories aligned with the ground truth for plotting.
    tf_traj = np.concatenate(
        [data.test_traj[:, :1, :], data.denormalise(tf_pred_norm)], axis=1)
    cl_traj = data.denormalise(gen_norm)
    norm_stats = make_norm_stats(data.train_traj, data.mean, data.std)
    rollout = {
        "val": rollout_metrics(data.val_traj, data.denormalise(val_gen),
                               norm_stats, args.system),
        "test": rollout_metrics(data.test_traj, cl_traj, norm_stats,
                                args.system)}

    for name, m in (("val", val_metrics), ("test", test_metrics)):
        print(f"  {name}: teacher-forced MSE {m['teacher_forced_mse']:.5f} | "
              f"closed-loop MSE {m['closed_loop_mse']:.5f} | persistence "
              f"{m['persistence_mse']:.5f} | VPT median "
              f"{rollout[name]['vpt_median']:g}/{data.seq_len}")

    os.makedirs(args.outdir, exist_ok=True)
    record = {
        "wiring": args.wiring, "cell": args.cell, "seed": args.seed,
        "system": args.system,
        "size": args.size, "param_budget": args.param_budget,
        "budget_deviation": budget_dev, "epochs": args.epochs, "lr": args.lr,
        "clipnorm": args.clipnorm,
        "backbone_layers": args.backbone_layers,
        "backbone_layers_effective": bb_effective, "self_recurrent": self_rec,
        "head_activation": args.head_activation,
        **pc, **multi, "dt": data.dt, "seq_len": data.seq_len,
        "ode_unfolds": cell_kwargs.get("ode_unfolds"), "dt_scale": args.dt_scale,
        "data_seed": data_seed, "split": split, "n_test": split[2],
        **{f"val_{k}": v for k, v in val_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
        **test_metrics,       # unprefixed = test (plot_lotka_volterra.py, #20)
        "best_val_epoch": best + 1,      # 1-based; curve index best_val_epoch-1
        "best_val_teacher_forced_mse": float(val_curve[best]),
        "best_test_teacher_forced_mse": float(test_curve[best]),
        "val_loss_curve": _round(val_curve, 6),
        "norm_stats": norm_stats,
        **{f"rollout_metrics_{s}": m for s, m in rollout.items()},
        # Trajectories (de-normalised, raw prey/predator) for the overlay plots.
        "ground_truth_traj": _round(data.test_traj),
        "teacher_forced_traj": _round(tf_traj),
        "closed_loop_traj": _round(cl_traj),
        "training_diverged": not np.isfinite(hist.history["loss"][-1]),
        "forcing": args.forcing,
        "alpha_min": args.alpha_min if mixed else None,
        "alpha_curve": _round(schedule.curve, 6) if mixed else None,
        "config": {"effective": {
            "task": "lotka_volterra", "system": args.system,
            "n_trajectories": args.n_trajectories, "seq_len": args.seq_len,
            "data_seed": data_seed, "split": split,
            "param_budget": args.param_budget, "backbone_layers": bb_effective,
            "forcing": args.forcing,
            "alpha_min": args.alpha_min if mixed else None,
            "build": dict(wiring=args.wiring, cell=args.cell, size=args.size,
                          seed=args.seed, feature_size=data.feature_size,
                          loc_dim=data.loc_dim, lr=args.lr,
                          self_recurrent=args.self_recurrent,
                          dt_scale=args.dt_scale, **multi),
            "cell_kwargs": cell_kwargs}},
    }
    if probe is not None:
        record["stability"] = probe.result
    path = os.path.join(args.outdir, result_filename(args, multi))
    record["weights_path"] = save_weights(model, path, args.save_weights)
    with open(path, "w") as f:
        json.dump(record, f)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
