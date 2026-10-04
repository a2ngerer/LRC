"""Train one (wiring, cell, seed) on the active-sensing object task and record
the accuracy-vs-number-of-glimpses curve -- the Thousand-Brains sample-
efficiency signature (concept: docs/superpowers/specs/2026-07-04-tbt-cncp).

The headline comparison is tbt_cncp vs tbt_cncp_noloc (same graph, location
signal on/off): does the reference frame let the column recognise the object in
fewer glimpses? ncp / dense (fed concat(patch, location)) are topology baselines.

Usage:
    uv run python experiments/run_active_sensing_benchmark.py \
        --wiring tbt_cncp --cell cfc_lrc --seed 0 --epochs 60
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from src.wirings import param_counts
from src.tasks.active_sensing.datasets import load_active_sensing
from src.tasks.active_sensing.model import (build_active_sensing_model,
                                            ACTIVE_WIRINGS)


def accuracy_curve(model, patch, time, loc, y):
    """Per-step test accuracy: acc after 1, 2, ... glimpses."""
    out = model.predict([patch, time, loc], verbose=0)
    logits = out[0] if isinstance(out, (list, tuple)) else out   # (N,T,C)
    pred = logits.argmax(-1)                                      # (N,T)
    return [float((pred[:, t] == y).mean()) for t in range(pred.shape[1])]


def _round(a, nd=4):
    return np.round(np.asarray(a, dtype=np.float64), nd).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiring", required=True, choices=ACTIVE_WIRINGS)
    ap.add_argument("--cell", default="cfc_lrc")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-per-class", type=int, default=360)
    ap.add_argument("--seq-len", type=int, default=12)
    ap.add_argument("--no-prediction", action="store_true",
                    help="disable the self-supervised next-patch head")
    ap.add_argument("--outdir", default="results/active_sensing")
    ap.add_argument("--elastance-type", default="asymmetric")
    ap.add_argument("--verbose", type=int, default=0)
    args = ap.parse_args()

    cell_kwargs = {}
    if args.cell in ("cfc_lrc", "lrc", "lrc_ar"):
        cell_kwargs["elastance_type"] = args.elastance_type

    data = load_active_sensing(n_per_class=args.n_per_class,
                               seq_len=args.seq_len)
    occ = load_active_sensing(n_per_class=args.n_per_class,
                              seq_len=args.seq_len, occlude=True)
    T = data.seq_len
    print(f"train {data.train_patch.shape[0]} / test {data.test_patch.shape[0]} "
          f"episodes, {T} glimpses, {data.num_classes} classes")

    use_pred = not args.no_prediction
    model = build_active_sensing_model(
        args.wiring, args.cell, size=args.size, seed=args.seed,
        num_classes=data.num_classes, patch_dim=data.patch_dim,
        loc_dim=data.loc_dim, lr=args.lr, use_prediction=use_pred, **cell_kwargs)
    pc = param_counts(model)
    params = pc['params_effective']
    print(f"{args.wiring}/{args.cell} seed {args.seed}: {params} params")

    def targets(y, nxt):
        yseq = np.repeat(y[:, None], T, axis=1).astype(np.int32)
        return {"logits": yseq, "pred": nxt} if use_pred else yseq

    ins = [data.train_patch, data.train_time, data.train_loc]
    val_ins = [data.test_patch, data.test_time, data.test_loc]
    model.fit(ins, targets(data.train_y, data.train_next),
              validation_data=(val_ins, targets(data.test_y, data.test_next)),
              batch_size=args.batch_size, epochs=args.epochs,
              verbose=args.verbose)

    curve = accuracy_curve(model, data.test_patch, data.test_time,
                           data.test_loc, data.test_y)
    occ_curve = accuracy_curve(model, occ.test_patch, occ.test_time,
                               occ.test_loc, occ.test_y)
    # Graded occlusion-robustness sweep (Iteration 3): re-evaluate the SAME
    # trained model at increasing occlusion. frac 0.0 == clean (reuses `data`),
    # frac 0.5 == the binary `occ` stressor. The reference frame should let the
    # location-carrying wirings degrade more gracefully as more is blanked.
    occ_fracs = [0.0, 0.25, 0.5, 0.75]
    occ_accs = []
    for frac in occ_fracs:
        od = data if frac == 0.0 else load_active_sensing(
            n_per_class=args.n_per_class, seq_len=args.seq_len,
            occlude_frac=frac)
        oc = accuracy_curve(model, od.test_patch, od.test_time, od.test_loc,
                            od.test_y)
        occ_accs.append(float(oc[-1]))
    _, counts = np.unique(data.test_y, return_counts=True)
    baseline = float(counts.max() / counts.sum())
    print(f"  final acc {curve[-1]:.3f} (occluded {occ_curve[-1]:.3f}) | "
          f"baseline {baseline:.3f} | acc@1glimpse {curve[0]:.3f}")
    print(f"  occlusion sweep {list(zip(occ_fracs, [round(a,3) for a in occ_accs]))}")

    os.makedirs(args.outdir, exist_ok=True)
    record = {
        "wiring": args.wiring, "cell": args.cell, "seed": args.seed,
        "size": args.size, "epochs": args.epochs, **pc,
        "seq_len": T, "num_classes": data.num_classes,
        "use_prediction": use_pred, "majority_baseline": baseline,
        "accuracy_curve": _round(curve), "final_accuracy": float(curve[-1]),
        "occluded_curve": _round(occ_curve),
        "occluded_final": float(occ_curve[-1]),
        "occlusion_fracs": occ_fracs,
        "occlusion_accs": _round(occ_accs),
    }
    fname = f"{args.cell}_{args.wiring}_seed{args.seed}.json"
    path = os.path.join(args.outdir, fname)
    with open(path, "w") as f:
        json.dump(record, f)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
