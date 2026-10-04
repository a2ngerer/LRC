"""Optuna HPO for one (cell, wiring) on the Lotka-Volterra rollout task, every
trial tracked in Weights & Biases.

Thesis methodology demo (Farsang 2026-07-02): the arms dense / NCP / cNCP are
compared PARAMETER-MATCHED. The network width is therefore NOT a search
dimension; instead ``size`` is derived per (cell, wiring) so every arm lands
within tolerance of a fixed parameter budget (``size_for_param_budget``). Optuna
then tunes only the *soft* hyper-parameters (learning rate, batch size) with a
TPE sampler and a MedianPruner, so the cross-wiring comparison stays fair (equal
capacity) while each arm still gets its own well-tuned optimiser settings.

Data hygiene: HPO selects on a validation split carved out of the TRAIN
trajectories (never the test split), fixing the val==test leakage the plain
runner had. The held-out test split is touched only once, for the final report
metric of the best trial.

Storage: one Optuna JournalStorage file per study
(``outputs/optuna/<study>.log``) so a study is resumable and could be shared by
several workers (``load_if_exists=True``).

wandb: each trial is one run (group=cell, job_type=wiring). On the compute node
set ``WANDB_MODE=offline`` and ``wandb sync`` the run dirs from the login node.

Usage (one study)::

    uv run python experiments/run_lotka_volterra_hpo.py \
        --cell cfc_lrc --wiring cncp --n-trials 25 --epochs 150 \
        --param-budget 4000 --wandb
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import optuna
import tensorflow as tf
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna_integration.tfkeras import TFKerasPruningCallback

from src.tasks.lotka_volterra.datasets import load_lotka_volterra, reference_frame
from src.tasks.lotka_volterra.model import build_lotka_volterra_model, LV_WIRINGS
from src.tasks.active_sensing.model import TBT_MODES
from src.benchmark.tracking import keras_callbacks, track
from src.wirings import match_param_budget, param_counts
from src.benchmark.registry import cell_kwargs as registry_cell_kwargs
from run_lotka_volterra_benchmark import closed_loop_rollout


# --------------------------------------------------------------------------- #
# Parameter matching
# --------------------------------------------------------------------------- #
def _param_counts(wiring, cell, size, feature_size, loc_dim, cell_kwargs):
    """Build a throwaway model and return ``param_counts`` (effective + raw).

    For the sparse 'ncp' wiring the masked-off synapse weights exist as
    variables but are multiplied by zero on every step, so they are excluded
    (docs/ncp-wiring-fix-2026-09-17.md); for every other wiring this is the
    plain trainable count.
    """
    model = build_lotka_volterra_model(
        wiring, cell, size=size, seed=0, feature_size=feature_size,
        loc_dim=loc_dim, lr=1e-3, **cell_kwargs)
    n = param_counts(model)
    del model
    tf.keras.backend.clear_session()
    return n


def size_for_param_budget(wiring, cell, target, feature_size, loc_dim,
                          cell_kwargs, lo=4, hi=192, step=1):
    """Pick the ``size`` whose model's EFFECTIVE count is closest to ``target``.

    Each arm (dense / ncp / cncp / tbt_cncp*) is sized on its own width knob,
    scanned in unit steps (``match_param_budget``: the count is monotone in
    ``size``, so the scan stops at the first size above the target).
    Returns ``(size, params)``.
    """
    return match_param_budget(
        lambda size: _param_counts(wiring, cell, size, feature_size, loc_dim,
                                   cell_kwargs)['params_effective'],
        target, range(lo, hi + 1, step))


# --------------------------------------------------------------------------- #
# Data (train / val / test)
# --------------------------------------------------------------------------- #
def _pack_inputs(x, t, loc, needs_loc):
    return (x, t, loc) if needs_loc else (x, t)


def carve_val(data, needs_loc, val_fraction, seed):
    """Split the TRAIN trajectories into an HPO-train and an HPO-val part.

    The test split is left untouched. Returns a dict with keras-ready
    (inputs, targets) tuples for ``tr`` (fit), ``val`` (Optuna objective) and
    ``full`` (train+val, used to refit the winning trial before the test eval).
    """
    n = data.train_x.shape[0]
    perm = np.random.default_rng(seed).permutation(n)
    n_val = max(1, int(round(val_fraction * n)))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    def subset(idx):
        loc = data.train_loc[idx] if needs_loc else None
        return _pack_inputs(data.train_x[idx], data.train_t[idx], loc,
                            needs_loc), data.train_y[idx]

    full = _pack_inputs(data.train_x, data.train_t,
                        data.train_loc if needs_loc else None, needs_loc)
    return {"tr": subset(tr_idx), "val": subset(val_idx),
            "full": (full, data.train_y)}


# --------------------------------------------------------------------------- #
# Objective
# --------------------------------------------------------------------------- #
def make_objective(cfg, data, splits, needs_loc, size, param_count,
                   cell_kwargs):
    (tr_in, tr_y), (val_in, val_y) = splits["tr"], splits["val"]

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        model = build_lotka_volterra_model(
            cfg["wiring"], cfg["cell"], size=size, seed=cfg["seed"],
            feature_size=data.feature_size, loc_dim=data.loc_dim, lr=lr,
            **cell_kwargs)

        run_cfg = {
            "task": "lotka_volterra", "system": cfg["system"],
            "cell": cfg["cell"], "wiring": cfg["wiring"], "seed": cfg["seed"],
            "size": size, "params": param_count,
            "param_budget": cfg["param_budget"],
            "lr": lr, "batch_size": batch_size, "epochs": cfg["epochs"],
            "trial": trial.number,
        }
        name = f"{cfg['cell']}_{cfg['wiring']}_t{trial.number}"
        best_val = float("inf")
        with track(cfg["wandb"], group=cfg["cell"], job_type=cfg["wiring"],
                   name=name, config=run_cfg) as run_:
            callbacks = keras_callbacks(run_)
            callbacks.append(TFKerasPruningCallback(trial, "val_loss"))
            hist = model.fit(
                tr_in, tr_y, validation_data=(val_in, val_y),
                batch_size=batch_size, epochs=cfg["epochs"], verbose=0,
                callbacks=callbacks)
            best_val = float(np.min(hist.history["val_loss"]))
            if run_ is not None:
                run_.summary["best_val_loss"] = best_val
        tf.keras.backend.clear_session()
        return best_val

    return objective


# --------------------------------------------------------------------------- #
# Final refit + test evaluation of the best trial
# --------------------------------------------------------------------------- #
def evaluate_best(cfg, data, splits, needs_loc, size, param_count, cell_kwargs,
                  best_params):
    """Refit at the best hyper-parameters on train+val, evaluate on test once."""
    (full_in, full_y) = splits["full"]
    test_in = _pack_inputs(data.test_x, data.test_t,
                           data.test_loc if needs_loc else None, needs_loc)

    model = build_lotka_volterra_model(
        cfg["wiring"], cfg["cell"], size=size, seed=cfg["seed"],
        feature_size=data.feature_size, loc_dim=data.loc_dim,
        lr=best_params["lr"], **cell_kwargs)

    run_cfg = {
        "task": "lotka_volterra", "system": cfg["system"], "cell": cfg["cell"],
        "wiring": cfg["wiring"], "seed": cfg["seed"], "size": size,
        "params": param_count, "param_budget": cfg["param_budget"],
        "role": "best_refit", **best_params, "epochs": cfg["epochs"],
    }
    name = f"{cfg['cell']}_{cfg['wiring']}_best"
    with track(cfg["wandb"], group=cfg["cell"], job_type=cfg["wiring"],
               name=name, config=run_cfg) as run_:
        model.fit(full_in, full_y, validation_data=(test_in, data.test_y),
                  batch_size=best_params["batch_size"], epochs=cfg["epochs"],
                  verbose=0, callbacks=keras_callbacks(run_))
        tf_mse, tf_mae = model.evaluate(test_in, data.test_y, verbose=0)
        gen_norm = closed_loop_rollout(model, data, needs_loc=needs_loc)
        truth_norm = (data.test_traj - data.mean) / data.std
        cl_mse = float(np.mean((gen_norm[:, 1:] - truth_norm[:, 1:]) ** 2))
        persistence_mse = float(np.mean((data.test_x - data.test_y) ** 2))
        if run_ is not None:
            run_.summary.update({"test_teacher_forced_mse": float(tf_mse),
                                 "test_closed_loop_mse": cl_mse,
                                 "persistence_mse": persistence_mse})
    tf.keras.backend.clear_session()
    return {"test_teacher_forced_mse": float(tf_mse),
            "test_teacher_forced_mae": float(tf_mae),
            "test_closed_loop_mse": cl_mse,
            "persistence_mse": persistence_mse}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="cfc_lrc")
    ap.add_argument("--wiring", required=True, choices=LV_WIRINGS)
    ap.add_argument("--system", default="periodic_predator_prey")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-trials", type=int, default=25)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--param-budget", type=int, default=4000,
                    help="target trainable parameter count all wirings match")
    ap.add_argument("--tol", type=float, default=0.15,
                    help="warn if the matched param count deviates more than this")
    ap.add_argument("--n-trajectories", type=int, default=80)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--outdir", default="results/lotka_volterra_hpo")
    ap.add_argument("--storage-dir", default="outputs/optuna")
    ap.add_argument("--elastance-type", default="asymmetric")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    # Registry is the single source of truth for variant kwargs -- crucially
    # cfc_lrc_outer -> elastance_gate='outer' (M. Farsang review variant). Setting
    # kwargs by hand here silently dropped that, so cfc_lrc_outer fell back to the
    # inner gate and an inner-vs-outer HPO compared inner against inner. Bare
    # lrc/lrc_ar are not registered, so they keep the CLI-driven elastance_type.
    cell_kwargs = registry_cell_kwargs(args.cell)
    if args.cell in ("cfc_lrc", "lrc", "lrc_ar") and "elastance_type" not in cell_kwargs:
        cell_kwargs["elastance_type"] = args.elastance_type

    needs_loc = args.wiring in TBT_MODES
    data = load_lotka_volterra(n_trajectories=args.n_trajectories,
                               seq_len=args.seq_len, system=args.system)

    size, param_count = size_for_param_budget(
        args.wiring, args.cell, args.param_budget, data.feature_size,
        data.loc_dim, cell_kwargs)
    dev = abs(param_count - args.param_budget) / args.param_budget
    flag = "  <-- OUT OF TOLERANCE" if dev > args.tol else ""
    print(f"[{args.cell}/{args.wiring}] param-matched size={size} -> "
          f"{param_count} params (budget {args.param_budget}, dev {dev:.1%}){flag}")

    splits = carve_val(data, needs_loc, args.val_fraction, seed=args.seed)

    cfg = {"cell": args.cell, "wiring": args.wiring, "system": args.system,
           "seed": args.seed, "epochs": args.epochs,
           "param_budget": args.param_budget, "wandb": args.wandb}

    os.makedirs(args.storage_dir, exist_ok=True)
    # Seed IS part of the study name: every (cell, wiring, seed) is its own
    # INDEPENDENT study (own TPE seed, own val split, own trial pool). Without the
    # seed suffix, seeds sharing a (cell, wiring) collapse -- via load_if_exists --
    # into one distributed study, and the per-seed spread stops being a real
    # replication. (For deliberate multi-worker sharing, run several processes
    # with the SAME seed against the same journal instead.)
    study_name = f"lv_{args.system}_{args.cell}_{args.wiring}_seed{args.seed}"
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(
            os.path.join(args.storage_dir, f"{study_name}.log")))
    study = optuna.create_study(
        study_name=study_name, storage=storage, load_if_exists=True,
        direction="minimize", sampler=TPESampler(seed=args.seed),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=15))

    objective = make_objective(cfg, data, splits, needs_loc, size, param_count,
                               cell_kwargs)
    study.optimize(objective, n_trials=args.n_trials, catch=(Exception,))

    n_done = sum(t.state == optuna.trial.TrialState.COMPLETE
                 for t in study.trials)
    n_pruned = sum(t.state == optuna.trial.TrialState.PRUNED
                   for t in study.trials)
    print(f"[{args.cell}/{args.wiring}] {n_done} complete / {n_pruned} pruned, "
          f"best val {study.best_value:.5f} @ {study.best_params}")

    test_metrics = evaluate_best(cfg, data, splits, needs_loc, size,
                                 param_count, cell_kwargs, study.best_params)
    print(f"[{args.cell}/{args.wiring}] TEST teacher-forced "
          f"{test_metrics['test_teacher_forced_mse']:.5f} | closed-loop "
          f"{test_metrics['test_closed_loop_mse']:.5f} | persistence "
          f"{test_metrics['persistence_mse']:.5f}")

    record = {
        "cell": args.cell, "wiring": args.wiring, "system": args.system,
        "seed": args.seed, "param_budget": args.param_budget,
        "matched_size": size, "matched_params": param_count,
        **_param_counts(args.wiring, args.cell, size, data.feature_size,
                        data.loc_dim, cell_kwargs),
        "param_dev": dev, "n_trials": args.n_trials, "epochs": args.epochs,
        "n_complete": n_done, "n_pruned": n_pruned,
        "best_val_loss": float(study.best_value),
        "best_params": study.best_params,
        **test_metrics,
        "trials": [
            {"number": t.number, "state": t.state.name, "value": t.value,
             "params": t.params}
            for t in study.trials
        ],
    }
    os.makedirs(args.outdir, exist_ok=True)
    # System is part of the filename: a multi-system sweep sharing one outdir would
    # otherwise overwrite {cell}_{wiring}_seed{seed}.json across systems.
    path = os.path.join(
        args.outdir,
        f"{args.system}_{args.cell}_{args.wiring}_seed{args.seed}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
