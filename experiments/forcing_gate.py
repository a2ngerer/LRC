"""Rank report of the forcing pilot (issue #25 pilot, per-fit report #69).

Descriptive only (protocol freeze F1-2026-10-26, section 6): mixed forcing is
the primary protocol by that freeze; this report attaches no threshold and no
decision to the numbers below.

Reads the pilot's LV runner JSONs ({gru, cfc, cfc_lrc} x {dense3, ncp} x
seeds, each under --forcing teacher and mixed, per system) and reports:

  * fit score per endpoint
      k0           -- median VPT(0.5) over the fit's test trajectories of the
                      stored closed-loop rollout (free run from the initial
                      state, src/evaluation/rollout_metrics);
      multi_anchor -- rollout_metrics_test.vpt_multi_median (#62), only when
                      every fit of the system carries it (the feat-25 pilot
                      JSONs do after `run_lotka_volterra_benchmark.py
                      --eval-only`);
    a fit with training_diverged scores -inf (worst rank);
  * arm score: median of the fit scores over seeds (per-fit aggregation);
  * Kendall tau_b with its standard error for the n arms between the teacher
    and the mixed ranking per endpoint, and between the k0 and the
    multi-anchor ranking per forcing.

The #25 gate pooled seeds x trajectories into one median per arm;
arm_score(pooled=True) keeps that aggregation only so the historical values
can be reproduced (tests/experiments/test_lv_forcing.py).

Usage:
    uv run python experiments/forcing_gate.py \
        '<lv run>/results/*.json' '<duffing run>/results/*.json' [--out rank.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from glob import glob

import numpy as np

from src.evaluation.rollout_metrics import rollout_metrics
from src.evaluation.stats import kendall_tau_b, kendall_tau_b_se

CELLS = ("gru", "cfc", "cfc_lrc")
WIRINGS = ("dense3", "ncp")
ARMS = tuple((c, w) for c in CELLS for w in WIRINGS)
FORCINGS = ("teacher", "mixed")
ENDPOINTS = ("k0", "multi_anchor")


def trajectory_vpt(r):
    """Per-test-trajectory VPT(0.5) of one result JSON; -inf for a fit whose
    training diverged."""
    gt, cl = np.asarray(r["ground_truth_traj"]), np.asarray(r["closed_loop_traj"])
    if r.get("training_diverged"):
        return np.full(len(gt), -np.inf)
    return np.array([rollout_metrics(gt[i:i + 1], cl[i:i + 1], r["norm_stats"],
                                     r["system"])["vpt_median"]
                     for i in range(len(gt))])


def arm_score(fits, endpoint="k0", pooled=False):
    """Score of one arm from its fits [result dict, one per seed]: median over
    the per-fit scores, or (pooled=True, k0 only, the #25 gate) the median
    over all seeds x trajectories."""
    if endpoint == "multi_anchor":
        per_fit = [-np.inf if r.get("training_diverged")
                   else r["rollout_metrics_test"]["vpt_multi_median"]
                   for r in fits]
        return float(np.median(per_fit))
    vpts = [trajectory_vpt(r) for r in fits]
    return float(np.median(np.concatenate(vpts) if pooled
                           else [np.median(v) for v in vpts]))


def collect(records):
    """records -> {(system, forcing, cell, wiring): [fit per seed]}; raises
    ValueError if a pilot fit is missing."""
    fits, seeds = {}, {}
    for r in records:
        key = (r["system"], r.get("forcing", "teacher"), r["cell"], r["wiring"])
        if key[1] in FORCINGS and key[2:] in ARMS:
            fits.setdefault(key, {})[r["seed"]] = r
            seeds.setdefault(r["system"], set()).add(r["seed"])
    missing = [f"{s}/{f}/{c}_{w}_seed{k}" for s in seeds for f in FORCINGS
               for c, w in ARMS for k in sorted(seeds[s])
               if k not in fits.get((s, f, c, w), {})]
    if missing:
        raise ValueError(f"{len(missing)} pilot fits missing: {missing}")
    return {key: [by_seed[k] for k in sorted(by_seed)]
            for key, by_seed in fits.items()}


def report(records, pooled=False):
    """records: result dicts -> {system: rank report} (see module docstring);
    pooled=True scores k0 the #25 way (reproduction only)."""
    fits = collect(records)
    out = {}
    for s in sorted({k[0] for k in fits}):
        arm_fits = {(f, a): fits[(s, f, *a)] for f in FORCINGS for a in ARMS}
        endpoints = ENDPOINTS if all(
            "vpt_multi_median" in r.get("rollout_metrics_test", {})
            for fs in arm_fits.values() for r in fs) else ENDPOINTS[:1]
        score = {f"{f}@{e}": [arm_score(arm_fits[(f, a)], e, pooled)
                              for a in ARMS]
                 for e in endpoints for f in FORCINGS}
        pairs = [(f"teacher@{e}", f"mixed@{e}") for e in endpoints]
        if len(endpoints) == 2:
            pairs += [(f"{f}@k0", f"{f}@multi_anchor") for f in FORCINGS]
        out[s] = {"arms": [f"{c}/{w}" for c, w in ARMS],
                  "n_seeds": len(arm_fits[(FORCINGS[0], ARMS[0])]),
                  "aggregation": "pooled" if pooled else "per_fit",
                  "median_vpt": score,
                  "tau_b": {f"{a} vs {b}": kendall_tau_b(score[a], score[b])
                            for a, b in pairs},
                  "tau_b_se": kendall_tau_b_se(len(ARMS))}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("globs", nargs="+", help="result JSON globs (** recurses)")
    ap.add_argument("--out", default=None, help="also write the report as JSON")
    args = ap.parse_args(argv)
    paths = sorted({p for g in args.globs for p in glob(g, recursive=True)})
    records = []
    for p in paths:
        with open(p) as f:
            r = json.load(f)
        if "closed_loop_traj" in r:
            records.append(r)
    try:
        rep = report(records)
    except ValueError as e:
        sys.exit(str(e))
    for s, r in rep.items():
        cols = list(r["median_vpt"])
        print(f"== {s} (arm = median over {r['n_seeds']} per-fit VPTs)")
        print("  " + " " * 16 + "".join(f"{c:>22s}" for c in cols))
        for i, arm in enumerate(r["arms"]):
            print(f"  {arm:16s}" + "".join(f"{r['median_vpt'][c][i]:22.1f}"
                                           for c in cols))
        for pair, tau in r["tau_b"].items():
            print(f"  tau_b {pair}: {tau:+.3f} (SE {r['tau_b_se']:.2f}, "
                  f"n={len(r['arms'])})")
        if len(cols) == 2:
            print("  multi_anchor: n/a (no rollout_metrics_test.vpt_multi_median;"
                  " re-evaluate with run_lotka_volterra_benchmark.py --eval-only)")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
