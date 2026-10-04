"""Train one (wiring, policy, seed) on the ACTIVE-glimpse object task and record
the accuracy-vs-glimpse curve (Iteration 5: sensorimotor active sensing).

Headline: policy=active (the column steers its own glimpses via the L5 motor
hub) vs policy=random (passive random glimpses) -- does steering reach accuracy
in fewer glimpses? And tbt_cncp (dedicated L5 motor) vs dense (hidden-state
motor) -- does the cortical motor hub help?

Usage:
    uv run python experiments/run_active_glimpse_benchmark.py \
        --wiring tbt_cncp --policy active --seed 0 --epochs 40
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from src.wirings import param_counts
from src.tasks.active_sensing.active_glimpse import (build_active_glimpse_model,
                                                     load_active_glimpse,
                                                     ACTIVE_GLIMPSE_WIRINGS)


def accuracy_curve(model, img, pos, y):
    logits = model.predict([img, pos], verbose=0)          # (N,T,C)
    pred = logits.argmax(-1)
    return [float((pred[:, t] == y).mean()) for t in range(pred.shape[1])]


def _round(a, nd=4):
    return np.round(np.asarray(a, dtype=np.float64), nd).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiring", default="tbt_cncp", choices=ACTIVE_GLIMPSE_WIRINGS)
    ap.add_argument("--policy", default="active", choices=["active", "random"])
    ap.add_argument("--cell", default="cfc_lrc")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-per-class", type=int, default=220)
    ap.add_argument("--n-classes", type=int, default=8)
    ap.add_argument("--grid", type=int, default=24)
    ap.add_argument("--seq-len", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--explore", type=float, default=0.3,
                    help="train-time epsilon-greedy exploration for active policy")
    ap.add_argument("--outdir", default="results/active_glimpse")
    ap.add_argument("--elastance-type", default="asymmetric")
    ap.add_argument("--verbose", type=int, default=0)
    args = ap.parse_args()

    cell_kwargs = {}
    if args.cell in ("cfc_lrc", "lrc", "lrc_ar"):
        cell_kwargs["elastance_type"] = args.elastance_type

    d = load_active_glimpse(n_per_class=args.n_per_class, n_classes=args.n_classes,
                            grid=args.grid, seq_len=args.seq_len, seed=args.seed)
    T = d["seq_len"]
    print(f"active-glimpse {args.wiring}/{args.policy}: train "
          f"{d['train_img'].shape[0]} / test {d['test_img'].shape[0]}, "
          f"{args.n_classes} classes, {T} glimpses")

    model = build_active_glimpse_model(
        args.wiring, args.policy, cell=args.cell, size=args.size, grid=args.grid,
        k=d["k"], seq_len=T, n_classes=args.n_classes, lr=args.lr,
        explore=args.explore, seed=args.seed, **cell_kwargs)

    def yseq(y):
        return np.repeat(y[:, None], T, axis=1).astype(np.int32)

    model.fit([d["train_img"], d["train_pos"]], yseq(d["train_y"]),
              validation_data=([d["test_img"], d["test_pos"]], yseq(d["test_y"])),
              batch_size=args.batch_size, epochs=args.epochs, verbose=args.verbose)
    pc = param_counts(model)
    params = pc['params_effective']   # built after the first forward pass

    curve = accuracy_curve(model, d["test_img"], d["test_pos"], d["test_y"])
    baseline = 1.0 / args.n_classes
    thr = args.threshold
    reach = next((t + 1 for t, a in enumerate(curve) if a >= thr), None)
    print(f"  final acc {curve[-1]:.3f} | baseline {baseline:.3f} | "
          f"glimpses to {thr}: {reach}")

    os.makedirs(args.outdir, exist_ok=True)
    record = {
        "wiring": args.wiring, "policy": args.policy, "cell": args.cell,
        "seed": args.seed, "size": args.size, "epochs": args.epochs,
        **pc, "seq_len": T, "n_classes": args.n_classes,
        "majority_baseline": baseline, "accuracy_curve": _round(curve),
        "final_accuracy": float(curve[-1]), "threshold": thr,
        "glimpses_to_thr": reach, "explore": args.explore,
    }
    fname = f"{args.cell}_{args.wiring}_{args.policy}_seed{args.seed}.json"
    path = os.path.join(args.outdir, fname)
    with open(path, "w") as f:
        json.dump(record, f)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
