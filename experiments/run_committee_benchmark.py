"""Train one (wiring, K, seed) partial-view voting committee on person-activity
and record clean accuracy + a test-time sensor-dropout curve (Iteration 7).

Headline comparisons (all arms at matched EFFECTIVE parameters; size the arms
with --param-budget or pass a per-arm --size):
  - learned consensus vs the controls at the SAME K and views: --vote-lambda 0
    (columns independent, readout averages), --consensus median, and
    --view-average mean|median (one cell on the coordinate-wise mean or lower
    median of the K views, the input-averaging monolith); K=1 is the single
    monolith on the full input.
  - cncp committee (cortical columns, L2/3 vote) vs ncp committee (command-slice
    vote, motor readout) vs dense committee (hidden-state vote, or a leading
    subspace with --dense-vote-units): is the column type or the consensus the
    source of a gain?

Every corruption level also records accuracy and macro-F1 per window position
t = 1..seq_len (fresh state per window): how fast the committee reaches its
final decision (normalised area under the macro-F1 curve, steps_to_90).

Usage:
    uv run python experiments/run_committee_benchmark.py \
        --wiring cncp --n-columns 4 --size 48 --seed 0 --epochs 40
    uv run python experiments/run_committee_benchmark.py \
        --wiring ncp --n-columns 4 --size 48 --vote-lambda 0 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np

from src.evaluation.metrics import confusion_matrix, macro_f1
from src.wirings import param_counts, size_for_budget
from src.tasks.person_activity.datasets import load_person_activity
from src.tasks.committee.views import (make_views, average_views,
                                       drop_features, add_noise)
from src.tasks.committee.model import build_committee_model, COMMITTEE_WIRINGS
from src.tasks.person_activity.model import (
    backbone_override, backbone_suffix, effective_backbone_layers)


DEFAULT_SIZE = 48
BUDGET_TOLERANCE = 0.05


def position_metrics(pred, y, num_classes):
    """Per window position t: accuracy (T,) and macro-F1 (T,) of the
    predicted classes pred (N,T) against y (N,T); plus the window macro-F1
    pooled over all positions."""
    cms = np.stack([confusion_matrix(y[:, t], pred[:, t], num_classes)
                    for t in range(y.shape[1])])            # (T,C,C)
    acc = np.trace(cms, axis1=1, axis2=2) / cms.sum(axis=(1, 2))
    return acc, np.array([macro_f1(c) for c in cms]), macro_f1(cms.sum(axis=0))


def steps_to(curve, frac=0.9):
    """First window position (1-based) where the curve reaches frac of its
    window-final value."""
    curve = np.asarray(curve)
    return int(np.argmax(curve >= frac * curve[-1])) + 1


def result_filename(args):
    """Result JSON name; a set --backbone-layers adds `_bb{N}` (V2 vs V2s)."""
    td = f"_td{args.train_drop}" if args.train_drop > 0.0 else ""
    tn = f"_tn{args.train_noise}" if args.train_noise > 0.0 else ""
    tc = "_tcnoise" if args.test_corruption == "noise" else ""
    lam = "" if args.vote_lambda is None else f"_lam{args.vote_lambda:g}"
    med = "_median" if args.consensus == "median" else ""
    dvu = f"_dvu{args.dense_vote_units}" if args.dense_vote_units else ""
    avg = {"off": "", "mean": "_viewavg",
           "median": "_viewmed"}[args.view_average]
    return (f"{args.cell}_{args.wiring}_K{args.n_columns}_s{args.size}_"
            f"{args.view_mode}{lam}{med}{dvu}{avg}{td}{tn}{tc}_seed{args.seed}"
            f"{backbone_suffix(args.backbone_layers)}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiring", default="cncp", choices=COMMITTEE_WIRINGS)
    ap.add_argument("--n-columns", type=int, default=4)
    cap = ap.add_mutually_exclusive_group()
    cap.add_argument("--size", type=int, default=None,
                     help="the arm's width knob (see build_committee_model); "
                          f"default {DEFAULT_SIZE} when no budget is given")
    cap.add_argument("--param-budget", type=int, default=None,
                     help="size the arm to this EFFECTIVE parameter budget "
                          "(closest size on the arm's own knob; weight sharing "
                          "makes the count independent of K)")
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
    ap.add_argument("--vote-lambda", type=float, default=None,
                    help="fixed voting strength in [0, 1] (default: learned); "
                         "0 = columns independent, the consensus control")
    ap.add_argument("--consensus", default="mean", choices=("mean", "median"),
                    help="consensus and readout statistic across columns")
    ap.add_argument("--dense-vote-units", type=int, default=None,
                    help="dense only: vote on / read out the leading n units "
                         "(the subspace-voting control for the cncp L2/3 vote)")
    ap.add_argument("--view-average", default="off", nargs="?", const="mean",
                    choices=("off", "mean", "median"),
                    help="input-averaging monolith: collapse the K views to "
                         "their coordinate-wise mean or lower median, then run "
                         "a single column (K=1 model); bare flag = mean")
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
    ap.add_argument("--backbone-layers", type=int, default=None,
                    help="CfC-family backbone depth for EVERY wiring (overrides "
                         "the default 1 of every wiring; 0 = no backbone); "
                         "default: unchanged per-wiring behaviour")
    ap.add_argument("--column-loop", action="store_true",
                    help="cncp/ncp/dense: step the K columns in the per-column "
                         "loop instead of the folded batch (same function, "
                         "slower; the #32 equivalence canary)")
    ap.add_argument("--verbose", type=int, default=0)
    args = ap.parse_args()
    if args.size is None and args.param_budget is None:
        args.size = DEFAULT_SIZE
    if args.wiring == "cncp2_multi" and args.vote_lambda is not None:
        # Independent columns read out by mean/median, no state vote.
        warnings.warn("--vote-lambda does not apply to cncp2_multi; ignored")
        args.vote_lambda = None
    if args.view_average == "median" and args.view_mode == "partition":
        warnings.warn("--view-average median on partition views: a feature "
                      "owned by fewer than half of the columns has median 0; "
                      "use --view-mode noisy")

    cell_kwargs = {}
    if args.cell in ("cfc_lrc", "lrc", "lrc_ar"):
        cell_kwargs["elastance_type"] = args.elastance_type
    cell_kwargs.update(backbone_override(args.cell, args.backbone_layers))
    bb_effective = effective_backbone_layers(args.wiring, args.cell, cell_kwargs)

    d = load_person_activity(seq_len=args.seq_len)
    F, C = d.feature_size, d.num_classes

    def views(X):
        v, _ = make_views(X, args.n_columns, mode=args.view_mode,
                          overlap=args.overlap, noise=args.noise,
                          seed=args.view_seed)
        if args.view_average != "off":   # (N,T,K,F) -> (N,T,1,F): one view
            v = average_views(v, args.view_average)
        return v

    # The model's column count: K, or 1 for the input-averaging monolith.
    n_cols = 1 if args.view_average != "off" else args.n_columns

    print(f"committee {args.wiring} K={args.n_columns} size={args.size} "
          f"lambda={'learned' if args.vote_lambda is None else args.vote_lambda} "
          f"consensus={args.consensus}"
          f" view-average={args.view_average} "
          f"backbone_layers={args.backbone_layers} (effective {bb_effective}): "
          f"train {d.train_x.shape[0]} / test {d.test_x.shape[0]}, "
          f"F={F} C={C} mode={args.view_mode}")

    def build(size):
        return build_committee_model(
            args.wiring, n_cols, cell=args.cell, size=size,
            feature_size=F, seq_len=args.seq_len, num_classes=C, lr=args.lr,
            seed=args.seed, vote_lambda=args.vote_lambda,
            consensus=args.consensus, dense_vote_units=args.dense_vote_units,
            train_drop=args.train_drop, train_noise=args.train_noise,
            column_loop=args.column_loop, **cell_kwargs)

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
    accs, stds, f1s, pos_acc, pos_f1 = [], [], [], [], []
    for lv in levels:
        # (S,T) per-position accuracy and macro-F1, (S,) window macro-F1
        a, f, w = (np.array(m) for m in zip(*[position_metrics(
            model.predict([views(corrupt(d.test_x, lv, seed=s)), d.test_t],
                          verbose=0).argmax(-1), d.test_y, C)
            for s in drop_seeds]))
        accs.append(float(a.mean()))           # window accuracy, seed mean
        stds.append(float(a.mean(axis=1).std()))
        f1s.append(float(w.mean()))
        pos_acc.append(a.mean(axis=0))
        pos_f1.append(f.mean(axis=0))
    clean = accs[0]
    steps90 = [steps_to(c) for c in pos_f1]
    print(f"  params {params} | clean {clean:.3f} | {args.test_corruption} "
          f"{[f'{lv}:{a:.3f}' for lv, a in zip(levels, accs)]} | "
          f"steps_to_90 {steps90}")

    os.makedirs(args.outdir, exist_ok=True)
    record = {
        "task": "person_activity", "wiring": args.wiring,
        "n_columns": args.n_columns, "size": args.size,
        "param_budget": args.param_budget, "budget_deviation": budget_dev,
        "cell": args.cell, "seed": args.seed, "epochs": args.epochs, **pc,
        "feature_size": F, "num_classes": C, "view_mode": args.view_mode,
        "overlap": args.overlap, "train_drop": args.train_drop,
        "train_noise": args.train_noise, "test_corruption": args.test_corruption,
        "vote_lambda": args.vote_lambda, "consensus": args.consensus,
        "dense_vote_units": args.dense_vote_units,
        "view_average": args.view_average, "model_columns": n_cols,
        "backbone_layers": args.backbone_layers,
        "backbone_layers_effective": bb_effective,
        **({"coupling": "none", "readout": args.consensus}
           if args.wiring == "cncp2_multi" else {}),
        "clean_accuracy": clean,
        "drop_fracs": levels, "drop_accs": [round(a, 4) for a in accs],
        "drop_stds": [round(s, 4) for s in stds], "drop_seeds": drop_seeds,
        "drop_macro_f1": [round(f, 4) for f in f1s],
        # one length-seq_len list per corruption level (seed mean); the area
        # is normalised (mean over positions), steps are 1-based positions
        "per_position_accuracy": [[round(float(v), 4) for v in c]
                                  for c in pos_acc],
        "per_position_macro_f1": [[round(float(v), 4) for v in c]
                                  for c in pos_f1],
        "per_position_macro_f1_auc": [round(float(c.mean()), 4)
                                      for c in pos_f1],
        "steps_to_90": steps90,
    }
    path = os.path.join(args.outdir, result_filename(args))
    with open(path, "w") as f:
        json.dump(record, f)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
