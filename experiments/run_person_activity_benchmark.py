"""Person-activity classification benchmark runner: one run = (wiring, cell, seed).

Trains the UCI person-activity sequence-classification task (32 timesteps,
7 features, 7 classes, irregular sampling) with a selectable wiring ('cncp',
'ncp', 'dense', 'dense3') around a selectable cell, so cNCP vs NCP can be
compared apples-to-apples (same inputs, same Dense head, same optimizer; see
src/tasks/person_activity/model.py for the fairness invariant).

Capacity: pass --param-budget to size the arm on its own width knob to the
EFFECTIVE parameter budget (masked-off weights excluded); --size fixes the
knob directly. The run records size, effective and raw counts and the
deviation from the budget (flagged above 5 % as budget_above_tolerance);
cncp scans fractional sizes (finer lamina knob, scaled_lamina_units).

Split: --split grouped (default) trains on 15 whole recordings, validates on
5 and tests on 5 (one per person each, --split-seed); --split random_window is
the original window-level split (literature anchor only, val = test).
Metrics: per-step macro-F1 (primary) and accuracy; the epoch with the best
val macro-F1 is selected and the test values are read at that epoch.

Elapsed time: the discrete cells (gru, lstm) ignore the time input, so with
--dt-feature on (default) they get dt as an extra input feature (issue #27).
--dt-scale multiplies dt before it reaches any cell or that feature (default
1.0 = unchanged); the numerical ODE cells (ltc, lrc, lrc_pm, ctrnn, mm_ltc,
mm_lrc) take --ode-unfolds explicit Euler sub-steps per input step (default
6). Both are recorded in the result JSON (issues #28, #56).

Mirrors the run + JSON-result-file structure of run_mujoco_benchmark.py /
run_campaign.py: one JSON per run under --outdir, containing the config,
the parameter counts, val/test macro-F1 and accuracy at the best-val epoch,
the per-epoch val curves and the test confusion matrix per test recording.

Examples
--------
    uv run python experiments/run_person_activity_benchmark.py \
        --wiring cncp --cell cfc_lrc --seed 0 --epochs 50 --param-budget 30000
    uv run python experiments/run_person_activity_benchmark.py \
        --wiring ncp --cell cfc_lrc --seed 0 --epochs 3   # smoke test
"""
import argparse
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone

import numpy as np

# Allow running as a plain script (``src`` is the installed package).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmark.registry import cell_kwargs as registry_cell_kwargs  # noqa: E402
from src.evaluation.metrics import confusion_matrix, macro_f1  # noqa: E402
from src.tasks.person_activity.datasets import SPLITS  # noqa: E402
from src.tasks.person_activity.model import (  # noqa: E402
    WIRINGS, backbone_override, backbone_suffix, budget_sizes,
    dt_feature_applies,
    effective_backbone_layers, effective_self_recurrent, add_multi_column_args,
    multi_column_config, multi_column_suffix, ode_unfolds_override,
    effective_ode_unfolds, DEFAULT_ODE_UNFOLDS)

DEFAULT_SIZE = 64
BUDGET_TOLERANCE = 0.05

def result_tag(args, multi=None):
    """Run tag (log, wandb name, result JSON stem); --backbone-layers adds
    `_bb{N}`, cncp2_multi adds `_K{K}_{partition}_{coupling}_{readout}`."""
    return (f"{args.cell}_{args.wiring}_seed{args.seed}"
            f"{backbone_suffix(args.backbone_layers)}"
            f"{multi_column_suffix(multi or {})}")


def split_confusions(model, data, part, batch_size):
    """{recording id: confusion matrix} of the per-step predictions on one
    split ('val' or 'test')."""
    x, t, y, rec = (getattr(data, f"{part}_{k}") for k in ("x", "t", "y", "rec"))
    pred = model.predict((x, t), batch_size=batch_size, verbose=0).argmax(-1)
    return {str(r): confusion_matrix(y[rec == r], pred[rec == r],
                                     data.num_classes)
            for r in np.unique(rec)}


def accuracy(cm):
    return float(np.trace(cm) / cm.sum())


def run(args):
    import tensorflow as tf
    from src.tasks.person_activity import (build_person_activity_model,
                                           load_person_activity)
    from src.benchmark.tracking import keras_callbacks, track
    from src.benchmark.checkpoints import save_weights
    from src.wirings import param_counts, size_for_budget

    data = load_person_activity(data_path=args.data_path, split=args.split,
                                split_seed=args.split_seed)
    print(f"data ({data.split}, split_seed={data.split_seed}, "
          f"{data.n_recordings} recordings, {data.n_negative_dt} negative dt "
          f"clamped): train {data.train_x.shape} val {data.val_x.shape} "
          f"test {data.test_x.shape} features={data.feature_size} "
          f"classes={data.num_classes}", flush=True)

    # Majority-class share of the test labels: the trivial-accuracy floor a
    # trained model must clear.
    counts = np.bincount(data.test_y.flatten(), minlength=data.num_classes)
    majority_baseline = float(counts.max() / counts.sum())

    # Per-cell constructor kwargs from the registry (elastance_type for the
    # LRC family, elastance_gate for cfc_lrc_outer); ode_unfolds added below
    # for the numerical cells (issue #28).
    cell_kwargs = registry_cell_kwargs(args.cell, args.elastance_type)
    cell_kwargs.update(backbone_override(args.cell, args.backbone_layers))
    if args.head_activation != "none":     # non-CfC cells reject the kwarg
        cell_kwargs["head_activation"] = args.head_activation
    cell_kwargs.update(ode_unfolds_override(args.cell, args.ode_unfolds))
    ode_unfolds = cell_kwargs.get("ode_unfolds")   # None: closed-form/discrete
    bb_effective = effective_backbone_layers(args.wiring, args.cell, cell_kwargs)
    self_rec = effective_self_recurrent(args.wiring, args.cell,
                                        args.self_recurrent)
    multi = multi_column_config(args)
    dt_feature = dt_feature_applies(args.cell, args.dt_feature == "on")

    def build(size):
        return build_person_activity_model(
            args.wiring, args.cell, size=size, seed=args.seed,
            num_classes=data.num_classes, feature_size=data.feature_size,
            seq_len=data.seq_len, lr=args.lr, dt_feature=dt_feature,
            dt_scale=args.dt_scale, self_recurrent=args.self_recurrent,
            **multi, **cell_kwargs)

    # Capacity: either a fixed width knob or the knob that lands closest to the
    # effective parameter budget for THIS arm.
    budget_dev, over_budget = None, False
    if args.param_budget is not None:
        args.size, n_eff = size_for_budget(build, args.param_budget,
                                           budget_sizes(args.wiring))
        budget_dev = abs(n_eff - args.param_budget) / args.param_budget
        over_budget = budget_dev > BUDGET_TOLERANCE
        flag = "  ABOVE TOLERANCE" if over_budget else ""
        print(f"budget {args.param_budget}: size={args.size} "
              f"params_effective={n_eff} (dev {budget_dev:.1%}){flag}",
              flush=True)
    model = build(args.size)
    model.optimizer.global_clipnorm = args.clipnorm   # None = no clipping
    pc = param_counts(model)
    params = pc['params_effective']

    tag = result_tag(args, multi)
    print(f"=== TRAIN {tag} size={args.size} params={params} "
          f"(raw {pc['params_raw']}) epochs={args.epochs} lr={args.lr} "
          f"batch={args.batch_size} backbone_layers={args.backbone_layers} "
          f"(effective {bb_effective}) dt_feature={dt_feature} "
          f"ode_unfolds={ode_unfolds} dt_scale={args.dt_scale} ===",
          flush=True)

    wandb_config = {
        "task": "person_activity",
        "cell": args.cell,
        "wiring": args.wiring,
        "seed": args.seed,
        "size": args.size,
        "param_budget": args.param_budget,
        "budget_deviation": budget_dev,
        "budget_above_tolerance": over_budget,
        "lr": args.lr,
        "clipnorm": args.clipnorm,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "backbone_layers": args.backbone_layers,
        "head_activation": args.head_activation,
        "self_recurrent": self_rec,
        "dt_feature": dt_feature,
        "ode_unfolds": ode_unfolds,
        "dt_scale": args.dt_scale,
        "split": data.split,
        "split_seed": data.split_seed,
        **pc,
        **multi,
    }

    # After every epoch: val macro-F1 / accuracy (the val curve selects the
    # epoch) and the per-recording test confusion matrices (read at that
    # epoch only). val_macro_f1 also goes to the Keras logs (history, wandb).
    val_f1, val_acc, test_cms = [], [], []

    def evaluate(epoch, logs):
        cm = sum(split_confusions(model, data, "val", args.batch_size).values())
        val_f1.append(macro_f1(cm))
        val_acc.append(accuracy(cm))
        test_cms.append(split_confusions(model, data, "test", args.batch_size))
        logs["val_macro_f1"] = val_f1[-1]

    diag = None
    if args.diagnostics:
        from src.evaluation.gradient_flow import RQ4Diagnostics
        b = args.batch_size
        diag = RQ4Diagnostics(model, (data.train_x[:b], data.train_t[:b]),
                              data.train_y[:b])

    t0 = time.time()
    with track(args.wandb, group=args.cell, job_type=args.wiring, name=tag,
               config=wandb_config) as run_:
        hist = model.fit(
            x=(data.train_x, data.train_t),
            y=data.train_y,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_data=((data.val_x, data.val_t), data.val_y),
            verbose=2,
            callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=evaluate),
                       *keras_callbacks(run_)] + ([diag] if diag else []),
        )
    wall = time.time() - t0

    train_acc = [float(a)
                 for a in hist.history["sparse_categorical_accuracy"]]
    best = int(np.argmax(val_f1))
    test_cm = sum(test_cms[best].values())

    result = {
        "wiring": args.wiring,
        "cell": args.cell,
        "seed": args.seed,
        "epochs": args.epochs,
        "size": args.size,
        "param_budget": args.param_budget,
        "budget_deviation": budget_dev,
        "budget_above_tolerance": over_budget,
        "lr": args.lr,
        "clipnorm": args.clipnorm,
        "clip_mode": None if args.clipnorm is None else "global",
        "batch_size": args.batch_size,
        "backbone_layers": args.backbone_layers,
        "backbone_layers_effective": bb_effective,
        "head_activation": args.head_activation,
        "self_recurrent": self_rec,
        "dt_feature": dt_feature,
        "ode_unfolds": ode_unfolds,
        "dt_scale": args.dt_scale,
        **pc,
        **multi,
        "cell_kwargs": cell_kwargs,
        "split": data.split,
        "split_seed": data.split_seed,
        "recordings": data.recordings,
        "n_recordings": data.n_recordings,
        "n_negative_dt": data.n_negative_dt,
        "train_sequences": int(data.train_x.shape[0]),
        "val_sequences": int(data.val_x.shape[0]),
        "test_sequences": int(data.test_x.shape[0]),
        "majority_baseline": majority_baseline,
        # Selection on val macro-F1, test read at that epoch (1-based; curve
        # index best_val_epoch - 1). random_window: val == test, so this is
        # selection on test (literature protocol).
        "best_val_epoch": best + 1,
        "val_macro_f1": val_f1[best],
        "val_accuracy": val_acc[best],
        "test_macro_f1": macro_f1(test_cm),
        # same epoch, one score per test recording (hierarchical bootstrap
        # over seeds x test recordings in the aggregation)
        "test_macro_f1_per_recording": {r: macro_f1(cm) for r, cm in
                                        test_cms[best].items()},
        "test_accuracy": accuracy(test_cm),
        # rows = true, columns = predicted; summing a resample of recordings
        # gives its pooled confusion (recording-level bootstrap).
        "test_confusion_by_recording": {r: cm.tolist() for r, cm in
                                        test_cms[best].items()},
        "val_macro_f1_curve": val_f1,
        "val_accuracy_curve": val_acc,
        "train_accuracy_curve": train_acc,
        "wallclock_s": round(wall, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "training_diverged": not np.isfinite(hist.history["loss"][-1]),
        "config": {"effective": {
            "task": "person_activity", "param_budget": args.param_budget,
            "backbone_layers": bb_effective,
            "ode_unfolds": effective_ode_unfolds(args.cell, cell_kwargs),
            "build": dict(wiring=args.wiring, cell=args.cell, size=args.size,
                          seed=args.seed, num_classes=data.num_classes,
                          feature_size=data.feature_size,
                          seq_len=data.seq_len, lr=args.lr,
                          dt_feature=dt_feature, dt_scale=args.dt_scale,
                          self_recurrent=args.self_recurrent, **multi),
            "elastance_type": cell_kwargs.get("elastance_type"),
            "cell_kwargs": cell_kwargs}},
    }

    if diag is not None:
        result["diagnostics"] = diag.result

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, f"{tag}.json")
    result["weights_path"] = save_weights(model, out_path, args.save_weights)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[{tag}] done in {wall:.0f}s  best_val_epoch={best + 1}  "
          f"val_f1={val_f1[best]:.4f}  test_f1={result['test_macro_f1']:.4f}  "
          f"test_acc={result['test_accuracy']:.4f}  "
          f"baseline={majority_baseline:.4f}  -> {out_path}", flush=True)
    return result


def main():
    p = argparse.ArgumentParser(
        description="Person-activity classification benchmark "
                    "(wiring x cell x seed)")
    p.add_argument("--wiring", required=True, choices=list(WIRINGS))
    p.add_argument("--cell", required=True,
                   help="cell key, e.g. cfc_lrc, gru, lrc, cfc, lstm (cncp "
                        "takes single-state cells only)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=50)
    cap = p.add_mutually_exclusive_group()
    cap.add_argument("--size", type=int, default=None,
                     help="the arm's width knob (see build_person_activity_model); "
                          f"default {DEFAULT_SIZE} when no budget is given")
    cap.add_argument("--param-budget", type=int, default=None,
                     help="size the arm to this EFFECTIVE parameter budget "
                          "(closest size on the arm's own knob)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--clipnorm", type=float, default=None,
                   help="clip the global gradient norm to this value (the "
                        "optimizer's global_clipnorm); default: no clipping")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--backbone-layers", type=int, default=None,
                   help="CfC-family backbone depth for EVERY wiring (overrides "
                        "the default 1 of every wiring; 0 = no backbone); "
                        "default: unchanged per-wiring behaviour")
    p.add_argument("--head-activation", default="none", choices=("none", "tanh"),
                   help="activation on the ff1/ff2 heads of the CfC-family cells "
                        "(bb0 attribution control); default: linear heads")
    p.add_argument("--no-self-recurrence", dest="self_recurrent",
                   action="store_false",
                   help="ncp x CfC family: drop the diagonal self-recurrence "
                        "of the NCP layers (issue #23 stability pilot)")
    p.add_argument("--dt-feature", default="on", choices=("on", "off"),
                   help="discrete cells (gru, lstm): concatenate the elapsed "
                        "time dt (the tensor the continuous-time cells "
                        "integrate) to the feature input of every layer that "
                        "receives features; no-op for continuous-time cells. "
                        "Person Activity only: the LV / Duffing runners have "
                        "no such flag, dt is constant there.")
    p.add_argument("--dt-scale", type=float, default=1.0,
                   help="multiply the elapsed time by this factor before it "
                        "reaches any cell (and the dt feature); task-level "
                        "time unit, 1.0 = unchanged")
    p.add_argument("--ode-unfolds", type=int, default=DEFAULT_ODE_UNFOLDS,
                   help="explicit Euler sub-steps per input step for the "
                        "numerical ODE cells (ltc, lrc, lrc_pm, ctrnn, "
                        "mm_ltc, mm_lrc); ignored for closed-form and discrete "
                        "cells")
    p.add_argument("--elastance-type", default=None,
                   help="override the registry elastance_type of the "
                        "LRC-family cells; default: the registry value")
    p.add_argument("--outdir", default="results/person_activity")
    p.add_argument("--data-path", default=None,
                   help="override the ConfLongDemo_JSI.txt location")
    p.add_argument("--split", default="grouped", choices=SPLITS,
                   help="grouped: whole recordings to train/val/test (15/5/5, "
                        "one val and one test recording per person); "
                        "random_window: original window-level split, val = "
                        "test (literature anchor only)")
    p.add_argument("--split-seed", type=int, default=0,
                   help="seed of the grouped recording assignment (keep it "
                        "fixed across a suite; ignored for random_window)")
    p.add_argument("--wandb", action="store_true",
                   help="log this run to Weights & Biases (needs the "
                        "'tracking' extra; use WANDB_MODE=offline on cluster)")
    p.add_argument("--diagnostics", action="store_true",
                   help="RQ4 diagnostics on one fixed training batch per epoch "
                        "(result key 'diagnostics', see src/evaluation/"
                        "gradient_flow.py RQ4Diagnostics, #30)")
    p.add_argument("--no-save-weights", dest="save_weights",
                   action="store_false",
                   help="do not write <stem>.weights.h5 next to the result JSON")
    add_multi_column_args(p)
    args = p.parse_args()
    if args.size is None and args.param_budget is None:
        args.size = DEFAULT_SIZE
    run(args)


if __name__ == "__main__":
    main()
