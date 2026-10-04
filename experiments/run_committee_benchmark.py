"""Train one (wiring, K, seed) partial-view voting committee on person-activity
and record clean accuracy + a test-time sensor-dropout curve (Iteration 7).

Headline comparisons (all at matched parameters, cncp size=48 ~ dense size=64):
  - K=1 (single monolith, full input) vs K>1 (committee, partitioned views):
    does splitting the input across K weight-shared columns + voting help,
    especially as test-time sensor dropout rises (graceful degradation, H1)?
  - cncp committee (cortical columns, L2/3 vote) vs dense committee (plain cells,
    hidden-state vote): is the cortical column a better weak learner (H3)?

Usage:
    uv run python experiments/run_committee_benchmark.py \
        --wiring cncp --n-columns 4 --size 48 --seed 0 --epochs 40
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from src.wirings import param_counts
from src.tasks.person_activity.datasets import load_person_activity
from src.tasks.committee.views import make_views, drop_features, add_noise
from src.tasks.committee.model import build_committee_model, COMMITTEE_WIRINGS


def per_step_accuracy(model, views, t, y):
    logits = model.predict([views, t], verbose=0)          # (N,T,C)
    return float((logits.argmax(-1) == y).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiring", default="cncp", choices=COMMITTEE_WIRINGS)
    ap.add_argument("--n-columns", type=int, default=4)
    ap.add_argument("--size", type=int, default=48,
                    help="cncp 48 ~ dense 64 are parameter-matched (~26k)")
    ap.add_argument("--cell", default="cfc_lrc")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--view-mode", default="partition",
                    choices=("partition", "noisy"))
    ap.add_argument("--overlap", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--view-seed", type=int, default=1,
                    help="fixes the feature partition (same masks train+test)")
    ap.add_argument("--train-drop", type=float, default=0.0,
                    help="train-time channel-dropout rate (robustness baseline); "
                         "K=1 with >0 is the augmented monolith control")
    ap.add_argument("--train-noise", type=float, default=0.0,
                    help="train-time Gaussian-noise sigma (the noise-augmented "
                         "monolith = home-advantage upper bound for noise)")
    ap.add_argument("--test-corruption", default="dropout",
                    choices=("dropout", "noise"),
                    help="test-time corruption family for the degradation sweep "
                         "(7d: 'noise' is unanticipated by a dropout-aug monolith)")
    ap.add_argument("--noise-sigmas", default="0.0,0.25,0.5,1.0",
                    help="test-time Gaussian-noise sigmas (for --test-corruption "
                         "noise)")
    ap.add_argument("--drop-fracs", default="0.0,0.15,0.3,0.45",
                    help="test-time channel-dropout fractions (sensor failure)")
    ap.add_argument("--drop-seeds", default="0,1,2,3,4",
                    help="average each dropout frac over these channel-choice "
                         "seeds (F is small, so WHICH channel fails dominates)")
    ap.add_argument("--outdir", default="results/committee")
    ap.add_argument("--elastance-type", default="asymmetric")
    ap.add_argument("--verbose", type=int, default=0)
    args = ap.parse_args()

    cell_kwargs = {}
    if args.cell in ("cfc_lrc", "lrc", "lrc_ar"):
        cell_kwargs["elastance_type"] = args.elastance_type

    d = load_person_activity(seq_len=args.seq_len)
    F, C = d.feature_size, d.num_classes

    def views(X):
        v, _ = make_views(X, args.n_columns, mode=args.view_mode,
                          overlap=args.overlap, noise=args.noise,
                          seed=args.view_seed)
        return v

    print(f"committee {args.wiring} K={args.n_columns} size={args.size}: "
          f"train {d.train_x.shape[0]} / test {d.test_x.shape[0]}, "
          f"F={F} C={C} mode={args.view_mode}")

    model = build_committee_model(
        args.wiring, args.n_columns, cell=args.cell, size=args.size,
        feature_size=F, seq_len=args.seq_len, num_classes=C, lr=args.lr,
        seed=args.seed, train_drop=args.train_drop,
        train_noise=args.train_noise, **cell_kwargs)
    model.fit([views(d.train_x), d.train_t], d.train_y,
              validation_data=([views(d.test_x), d.test_t], d.test_y),
              epochs=args.epochs, batch_size=args.batch_size,
              verbose=args.verbose)
    pc = param_counts(model)
    params = pc['params_effective']

    # Test-time corruption sweep: channel dropout OR (7d) additive Gaussian noise.
    if args.test_corruption == "noise":
        levels = [float(x) for x in args.noise_sigmas.split(",")]
        corrupt = add_noise
    else:
        levels = [float(x) for x in args.drop_fracs.split(",")]
        corrupt = drop_features
    drop_seeds = [int(s) for s in args.drop_seeds.split(",")]
    accs, stds = [], []
    for lv in levels:
        per_seed = [per_step_accuracy(model, views(corrupt(
            d.test_x, lv, seed=s)), d.test_t, d.test_y) for s in drop_seeds]
        accs.append(float(np.mean(per_seed)))
        stds.append(float(np.std(per_seed)))
    clean = accs[0]
    print(f"  params {params} | clean {clean:.3f} | {args.test_corruption} "
          f"{[f'{lv}:{a:.3f}' for lv, a in zip(levels, accs)]}")

    os.makedirs(args.outdir, exist_ok=True)
    record = {
        "task": "person_activity", "wiring": args.wiring,
        "n_columns": args.n_columns, "size": args.size, "cell": args.cell,
        "seed": args.seed, "epochs": args.epochs, **pc,
        "feature_size": F, "num_classes": C, "view_mode": args.view_mode,
        "overlap": args.overlap, "train_drop": args.train_drop,
        "train_noise": args.train_noise, "test_corruption": args.test_corruption,
        "clean_accuracy": clean,
        "drop_fracs": levels, "drop_accs": [round(a, 4) for a in accs],
        "drop_stds": [round(s, 4) for s in stds], "drop_seeds": drop_seeds,
    }
    td = f"_td{args.train_drop}" if args.train_drop > 0.0 else ""
    tn = f"_tn{args.train_noise}" if args.train_noise > 0.0 else ""
    tc = "_tcnoise" if args.test_corruption == "noise" else ""
    fname = (f"{args.cell}_{args.wiring}_K{args.n_columns}_s{args.size}_"
             f"{args.view_mode}{td}{tn}{tc}_seed{args.seed}.json")
    path = os.path.join(args.outdir, fname)
    with open(path, "w") as f:
        json.dump(record, f)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
