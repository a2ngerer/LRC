"""Train one (wiring, train-fraction, seed) on person-activity and record the
test accuracy (Iteration 8: DATA EFFICIENCY as an inductive-bias test).

Headline: does the sparse bio-inspired cNCP wiring generalise better from LESS
data than a dense reference at MATCHED parameters? A structural prior should help
most in the low-data regime (a steeper learning curve) and wash out at full data.
This is a property no augmentation trick addresses -- pure sample complexity.

For a fixed seed every wiring sees the SAME random training subset (fair), while
the seed also seeds weight init; sweeping seeds gives CIs on the learning curve.

Usage:
    uv run python experiments/run_data_efficiency_benchmark.py \
        --wiring cncp --frac 0.1 --seed 0 --epochs 60
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from src.wirings import param_counts
from src.tasks.person_activity.datasets import load_person_activity
from src.tasks.person_activity.model import build_person_activity_model, WIRINGS


def per_step_accuracy(model, x, t, y):
    return float((model.predict([x, t], verbose=0).argmax(-1) == y).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiring", default="cncp", choices=WIRINGS)
    ap.add_argument("--frac", type=float, default=1.0,
                    help="fraction of the training set to use")
    ap.add_argument("--cell", default="cfc_lrc")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-train", type=int, default=64,
                    help="floor on the subset size so tiny fracs still train")
    ap.add_argument("--outdir", default="results/data_efficiency")
    ap.add_argument("--elastance-type", default="asymmetric")
    ap.add_argument("--verbose", type=int, default=0)
    args = ap.parse_args()

    cell_kwargs = {}
    if args.cell in ("cfc_lrc", "lrc", "lrc_ar"):
        cell_kwargs["elastance_type"] = args.elastance_type

    d = load_person_activity(seq_len=args.seq_len)
    F, C = d.feature_size, d.num_classes
    N = d.train_x.shape[0]
    # Same subset for every wiring at a given seed (fair), varies across seeds.
    perm = np.random.default_rng(args.seed).permutation(N)
    k = max(args.min_train, int(round(args.frac * N)))
    k = min(k, N)
    idx = perm[:k]
    print(f"data-eff {args.wiring} frac={args.frac} -> train {k}/{N}, "
          f"test {d.test_x.shape[0]}, F={F} C={C}")

    model = build_person_activity_model(
        args.wiring, args.cell, size=args.size, seed=args.seed, num_classes=C,
        feature_size=F, seq_len=args.seq_len, lr=args.lr, **cell_kwargs)
    model.fit([d.train_x[idx], d.train_t[idx]], d.train_y[idx],
              validation_data=([d.test_x, d.test_t], d.test_y),
              epochs=args.epochs, batch_size=args.batch_size,
              verbose=args.verbose)
    pc = param_counts(model)
    params = pc['params_effective']
    acc = per_step_accuracy(model, d.test_x, d.test_t, d.test_y)
    print(f"  params {params} | train {k} | test acc {acc:.3f}")

    os.makedirs(args.outdir, exist_ok=True)
    record = {
        "task": "person_activity", "wiring": args.wiring, "cell": args.cell,
        "size": args.size, "seed": args.seed, "epochs": args.epochs,
        **pc, "frac": args.frac, "n_train": k, "n_total": N,
        "num_classes": C, "test_accuracy": acc,
    }
    fname = f"{args.cell}_{args.wiring}_f{args.frac}_seed{args.seed}.json"
    path = os.path.join(args.outdir, fname)
    with open(path, "w") as f:
        json.dump(record, f)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
