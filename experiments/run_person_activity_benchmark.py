"""Person-activity classification benchmark runner: one run = (wiring, cell, seed).

Trains the UCI person-activity sequence-classification task (32 timesteps,
7 features, 7 classes, irregular sampling) with a selectable sparse wiring
('cncp', 'ncp', 'dense') around a selectable single-state cell, so cNCP vs
NCP can be compared apples-to-apples (same inputs, same Dense head, same
optimizer; see src/tasks/person_activity/model.py for the fairness
invariant and the size matching).

Mirrors the run + JSON-result-file structure of run_mujoco_benchmark.py /
run_campaign.py: one JSON per run under --outdir, containing the config,
the parameter count, final/best test accuracy and the per-epoch
val-accuracy curve.

Examples
--------
    uv run python experiments/run_person_activity_benchmark.py \
        --wiring cncp --cell cfc_lrc --seed 0 --epochs 50
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

# Per-cell constructor kwargs, mirroring CELL_KWARGS in run_benchmark.py:
# the numerical LRC family needs elastance_type='asymmetric' (its 'interp'
# default disables the elastance mechanism); cfc_lrc already defaults to
# 'asymmetric', listed here for explicitness.
CELL_KWARGS = {
    "lrc": dict(elastance_type="asymmetric"),
    "lrc_pm": dict(elastance_type="asymmetric"),
    "cfc_lrc": dict(elastance_type="asymmetric"),
}


def run(args):
    from src.tasks.person_activity import (build_person_activity_model,
                                           load_person_activity)
    from src.benchmark.tracking import keras_callbacks, track
    from src.wirings import param_counts

    data = load_person_activity(data_path=args.data_path)
    print(f"data: train {data.train_x.shape} test {data.test_x.shape} "
          f"features={data.feature_size} classes={data.num_classes}",
          flush=True)

    # Majority-class share of the test labels: the trivial-accuracy floor a
    # trained model must clear.
    counts = np.bincount(data.test_y.flatten(), minlength=data.num_classes)
    majority_baseline = float(counts.max() / counts.sum())

    cell_kwargs = dict(CELL_KWARGS.get(args.cell, {}))
    model = build_person_activity_model(
        args.wiring, args.cell, size=args.size, seed=args.seed,
        num_classes=data.num_classes, feature_size=data.feature_size,
        seq_len=data.seq_len, lr=args.lr, **cell_kwargs)
    pc = param_counts(model)
    params = pc['params_effective']

    tag = f"{args.cell}_{args.wiring}_seed{args.seed}"
    print(f"=== TRAIN {tag} size={args.size} params={params} "
          f"epochs={args.epochs} lr={args.lr} batch={args.batch_size} ===",
          flush=True)

    wandb_config = {
        "task": "person_activity",
        "cell": args.cell,
        "wiring": args.wiring,
        "seed": args.seed,
        "size": args.size,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        **pc,
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
        "lr": args.lr,
        "batch_size": args.batch_size,
        **pc,
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
    p.add_argument("--wiring", required=True,
                   choices=["cncp", "ncp", "dense"])
    p.add_argument("--cell", required=True,
                   help="single-state cell key, e.g. cfc_lrc, gru, lrc, cfc")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--size", type=int, default=64,
                   help="shared width knob (see build_person_activity_model)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--outdir", default="results/person_activity")
    p.add_argument("--data-path", default=None,
                   help="override the ConfLongDemo_JSI.txt location")
    p.add_argument("--wandb", action="store_true",
                   help="log this run to Weights & Biases (needs the "
                        "'tracking' extra; use WANDB_MODE=offline on cluster)")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
