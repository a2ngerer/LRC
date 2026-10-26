"""Person-activity classification benchmark runner: one run = (wiring, cell, seed).

Trains the UCI person-activity sequence-classification task (32 timesteps,
7 features, 7 classes, irregular sampling) with a selectable wiring ('cncp',
'ncp', 'dense', 'dense3') around a selectable cell, so cNCP vs NCP can be
compared apples-to-apples (same inputs, same Dense head, same optimizer; see
src/tasks/person_activity/model.py for the fairness invariant).

Capacity: pass --param-budget to size the arm on its own width knob to the
EFFECTIVE parameter budget (masked-off weights excluded); --size fixes the
knob directly. The run records size, effective and raw counts and the
deviation from the budget (flagged above 5 %).

Mirrors the run + JSON-result-file structure of run_mujoco_benchmark.py /
run_campaign.py: one JSON per run under --outdir, containing the config,
the parameter counts, final/best test accuracy and the per-epoch
val-accuracy curve.

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

from src.tasks.person_activity.model import (  # noqa: E402
    WIRINGS, backbone_override, backbone_suffix, effective_backbone_layers,
    effective_self_recurrent, add_multi_column_args, multi_column_config,
    multi_column_suffix)

DEFAULT_SIZE = 64
BUDGET_TOLERANCE = 0.05

# Per-cell constructor kwargs, mirroring CELL_KWARGS in run_benchmark.py:
# the numerical LRC family needs elastance_type='asymmetric' (its 'interp'
# default disables the elastance mechanism); cfc_lrc already defaults to
# 'asymmetric', listed here for explicitness.
CELL_KWARGS = {
    "lrc": dict(elastance_type="asymmetric"),
    "lrc_pm": dict(elastance_type="asymmetric"),
    "cfc_lrc": dict(elastance_type="asymmetric"),
}


def result_tag(args, multi=None):
    """Run tag (log, wandb name, result JSON stem); --backbone-layers adds
    `_bb{N}`, cncp2_multi adds `_K{K}_{partition}_{coupling}_{readout}`."""
    return (f"{args.cell}_{args.wiring}_seed{args.seed}"
            f"{backbone_suffix(args.backbone_layers)}"
            f"{multi_column_suffix(multi or {})}")


def run(args):
    from src.tasks.person_activity import (build_person_activity_model,
                                           load_person_activity)
    from src.benchmark.tracking import keras_callbacks, track
    from src.wirings import param_counts, size_for_budget

    data = load_person_activity(data_path=args.data_path)
    print(f"data: train {data.train_x.shape} test {data.test_x.shape} "
          f"features={data.feature_size} classes={data.num_classes}",
          flush=True)

    # Majority-class share of the test labels: the trivial-accuracy floor a
    # trained model must clear.
    counts = np.bincount(data.test_y.flatten(), minlength=data.num_classes)
    majority_baseline = float(counts.max() / counts.sum())

    cell_kwargs = dict(CELL_KWARGS.get(args.cell, {}))
    cell_kwargs.update(backbone_override(args.cell, args.backbone_layers))
    if args.head_activation != "none":     # non-CfC cells reject the kwarg
        cell_kwargs["head_activation"] = args.head_activation
    bb_effective = effective_backbone_layers(args.wiring, args.cell, cell_kwargs)
    self_rec = effective_self_recurrent(args.wiring, args.cell,
                                        args.self_recurrent)
    multi = multi_column_config(args)

    def build(size):
        return build_person_activity_model(
            args.wiring, args.cell, size=size, seed=args.seed,
            num_classes=data.num_classes, feature_size=data.feature_size,
            seq_len=data.seq_len, lr=args.lr,
            self_recurrent=args.self_recurrent, **multi, **cell_kwargs)

    # Capacity: either a fixed width knob or the knob that lands closest to the
    # effective parameter budget for THIS arm.
    budget_dev = None
    if args.param_budget is not None:
        args.size, n_eff = size_for_budget(build, args.param_budget)
        budget_dev = abs(n_eff - args.param_budget) / args.param_budget
        flag = "  ABOVE TOLERANCE" if budget_dev > BUDGET_TOLERANCE else ""
        print(f"budget {args.param_budget}: size={args.size} "
              f"params_effective={n_eff} (dev {budget_dev:.1%}){flag}",
              flush=True)
    model = build(args.size)
    pc = param_counts(model)
    params = pc['params_effective']

    tag = result_tag(args, multi)
    print(f"=== TRAIN {tag} size={args.size} params={params} "
          f"(raw {pc['params_raw']}) epochs={args.epochs} lr={args.lr} "
          f"batch={args.batch_size} backbone_layers={args.backbone_layers} "
          f"(effective {bb_effective}) ===", flush=True)

    wandb_config = {
        "task": "person_activity",
        "cell": args.cell,
        "wiring": args.wiring,
        "seed": args.seed,
        "size": args.size,
        "param_budget": args.param_budget,
        "budget_deviation": budget_dev,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "backbone_layers": args.backbone_layers,
        "head_activation": args.head_activation,
        "self_recurrent": self_rec,
        **pc,
        **multi,
    }

    t0 = time.time()
    with track(args.wandb, group=args.cell, job_type=args.wiring, name=tag,
               config=wandb_config) as run_:
        hist = model.fit(
            x=(data.train_x, data.train_t),
            y=data.train_y,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_data=((data.test_x, data.test_t), data.test_y),
            verbose=2,
            callbacks=keras_callbacks(run_),
        )
    wall = time.time() - t0

    val_acc = [float(a)
               for a in hist.history["val_sparse_categorical_accuracy"]]
    train_acc = [float(a)
                 for a in hist.history["sparse_categorical_accuracy"]]

    result = {
        "wiring": args.wiring,
        "cell": args.cell,
        "seed": args.seed,
        "epochs": args.epochs,
        "size": args.size,
        "param_budget": args.param_budget,
        "budget_deviation": budget_dev,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "backbone_layers": args.backbone_layers,
        "backbone_layers_effective": bb_effective,
        "head_activation": args.head_activation,
        "self_recurrent": self_rec,
        **pc,
        **multi,
        "cell_kwargs": cell_kwargs,
        "train_sequences": int(data.train_x.shape[0]),
        "test_sequences": int(data.test_x.shape[0]),
        "majority_baseline": majority_baseline,
        # validation split == test split (as in the original runner), so
        # val accuracy IS test accuracy.
        "test_accuracy": val_acc[-1],
        "best_test_accuracy": max(val_acc),
        "val_accuracy_curve": val_acc,
        "train_accuracy_curve": train_acc,
        "wallclock_s": round(wall, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
    }

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, f"{tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[{tag}] done in {wall:.0f}s  final={val_acc[-1]:.4f}  "
          f"best={max(val_acc):.4f}  baseline={majority_baseline:.4f}  "
          f"-> {out_path}", flush=True)
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
    p.add_argument("--outdir", default="results/person_activity")
    p.add_argument("--data-path", default=None,
                   help="override the ConfLongDemo_JSI.txt location")
    p.add_argument("--wandb", action="store_true",
                   help="log this run to Weights & Biases (needs the "
                        "'tracking' extra; use WANDB_MODE=offline on cluster)")
    add_multi_column_args(p)
    args = p.parse_args()
    if args.size is None and args.param_budget is None:
        args.size = DEFAULT_SIZE
    run(args)


if __name__ == "__main__":
    main()
