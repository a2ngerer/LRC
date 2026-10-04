"""Train a K-column voting tbt_cNCP on active-sensing and record the accuracy-
vs-glimpses-per-column curve (TBT ingredient 3 / hypothesis H2: do K columns
that vote reach a given accuracy in fewer glimpses than one column?).

Usage:
    uv run python experiments/run_voting_benchmark.py --n-columns 3 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from src.wirings import param_counts
from src.tasks.active_sensing import load_active_sensing, build_voting_model


def accuracy_curve(model, patch, time, loc, y):
    logits = model.predict([patch, time, loc], verbose=0)   # (N,T,C)
    pred = logits.argmax(-1)
    return [float((pred[:, t] == y).mean()) for t in range(pred.shape[1])]


def _round(a, nd=4):
    return np.round(np.asarray(a, dtype=np.float64), nd).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-columns", type=int, required=True)
    ap.add_argument("--cell", default="cfc_lrc")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-per-class", type=int, default=300)
    ap.add_argument("--seq-len", type=int, default=12)
    ap.add_argument("--outdir", default="results/active_sensing_voting")
    ap.add_argument("--location-mode", default="concat",
                    choices=["concat", "film", "gate"],
                    help="how the reference frame reaches each column")
    ap.add_argument("--elastance-type", default="asymmetric")
    ap.add_argument("--verbose", type=int, default=0)
    args = ap.parse_args()

    cell_kwargs = {}
    if args.cell in ("cfc_lrc", "lrc", "lrc_ar"):
        cell_kwargs["elastance_type"] = args.elastance_type

    K = args.n_columns
    data = load_active_sensing(n_per_class=args.n_per_class, seq_len=args.seq_len,
                               n_columns=K)
    T = data.seq_len
    print(f"K={K}: train {data.train_patch.shape[0]} / test "
          f"{data.test_patch.shape[0]} episodes, {T} glimpses/column")

    model = build_voting_model(
        K, args.cell, size=args.size, seed=args.seed,
        num_classes=data.num_classes, patch_dim=data.patch_dim,
        loc_dim=data.loc_dim, lr=args.lr, location_mode=args.location_mode,
        **cell_kwargs)
    pc = param_counts(model)
    params = pc['params_effective']
    print(f"vote_k{K}/{args.cell}/{args.location_mode} seed {args.seed}: "
          f"{params} params")

    def yseq(y):
        return np.repeat(y[:, None], T, axis=1).astype(np.int32)

    # For K==1 the loader squeezes the column axis (N,T,P); the voting model
    # always wants a column axis (N,T,K,P), so add it back when missing.
    def kax(a):
        return a if a.ndim == 4 else a[:, :, None, :]

    tr_patch, tr_loc = kax(data.train_patch), kax(data.train_loc)
    te_patch, te_loc = kax(data.test_patch), kax(data.test_loc)

    ins = [tr_patch, data.train_time, tr_loc]
    val = ([te_patch, data.test_time, te_loc], yseq(data.test_y))
    model.fit(ins, yseq(data.train_y), validation_data=val,
              batch_size=args.batch_size, epochs=args.epochs,
              verbose=args.verbose)

    curve = accuracy_curve(model, te_patch, data.test_time, te_loc,
                           data.test_y)
    _, counts = np.unique(data.test_y, return_counts=True)
    baseline = float(counts.max() / counts.sum())
    thr = 0.7
    reach = next((t + 1 for t, a in enumerate(curve) if a >= thr), None)
    print(f"  final acc {curve[-1]:.3f} | acc@1 {curve[0]:.3f} | "
          f"glimpses to {thr}: {reach}")

    os.makedirs(args.outdir, exist_ok=True)
    record = {
        "wiring": f"vote_k{K}_{args.location_mode}", "n_columns": K,
        "location_mode": args.location_mode, "cell": args.cell,
        "seed": args.seed, "size": args.size, "epochs": args.epochs,
        **pc, "seq_len": T, "num_classes": data.num_classes,
        "majority_baseline": baseline, "accuracy_curve": _round(curve),
        "final_accuracy": float(curve[-1]),
        "glimpses_to_0.7": reach,
    }
    path = os.path.join(
        args.outdir,
        f"{args.cell}_votek{K}_{args.location_mode}_seed{args.seed}.json")
    with open(path, "w") as f:
        json.dump(record, f)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
