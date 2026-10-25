"""Recompute the #20 rollout metrics from existing LV/Duffing result JSONs.

Scores the stored test-split trajectories (ground_truth_traj vs
closed_loop_traj, de-normalised) with src/evaluation/rollout_metrics and writes
one CSV row per result JSON. Files written since #20 carry the runner's
norm_stats (train z-score and range); older files (e.g. the 2026-10-21 pilot)
do not, so their stats are fitted on the stored test ground truth instead
(column norm_stats_source = test_proxy), which shifts the thresholds slightly.

Usage:
    uv run python experiments/rollout_metrics.py \
        'results/01_lotka_volterra/**/*.json' 'results/02_duffing/**/*.json' \
        --out rollout_metrics.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from glob import glob

from src.evaluation.rollout_metrics import make_norm_stats, rollout_metrics

ID_KEYS = ("system", "cell", "wiring", "seed", "param_budget",
           "backbone_layers", "epochs")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("globs", nargs="+", help="result JSON globs (** recurses)")
    ap.add_argument("--out", default="rollout_metrics.csv")
    args = ap.parse_args(argv)

    rows = []
    for path in sorted({p for g in args.globs for p in glob(g, recursive=True)}):
        with open(path) as f:
            r = json.load(f)
        if "closed_loop_traj" not in r:      # not a rollout run (configs, HPO, ...)
            continue
        stats, source = r.get("norm_stats"), "stored"
        if stats is None:
            stats, source = make_norm_stats(r["ground_truth_traj"]), "test_proxy"
        metrics = rollout_metrics(r["ground_truth_traj"], r["closed_loop_traj"],
                                  stats, r.get("system", "periodic_predator_prey"))
        rows.append({"path": path, **{k: r.get(k) for k in ID_KEYS},
                     "norm_stats_source": source, **metrics})
    if not rows:
        sys.exit("no result JSON with closed_loop_traj matched")
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
