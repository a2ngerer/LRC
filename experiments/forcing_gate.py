"""Training-protocol gate of the rollout tasks (issue #25, pre-registered).

Reads the pilot's LV runner JSONs ({gru, cfc, cfc_lrc} x {dense3, ncp} x
seeds, each under --forcing teacher and mixed, per system) and applies the
pre-registered rule:

  * arm score = median VPT(0.5) over seeds x test trajectories (pooled), from
    the stored de-normalised test rollouts (src/evaluation/rollout_metrics);
    a fit with training_diverged scores -inf on every trajectory (worst rank);
  * per system: Kendall tau_b between the teacher and the mixed arm scores
    (tau_b from src.evaluation.stats, issue #29);
  * per system: sign of the paired median difference VPT(gru/dense3) -
    VPT(cfc_lrc/dense3) over (seed, test trajectory) pairs; a flip is a
    strictly opposite sign under the two protocols (a zero is reported, not a
    flip);
  * mixed forcing becomes primary if tau_b < 0.6 or a sign flips on either
    system; otherwise teacher forcing stays primary.

Without #29 the tau_b part reports "n/a"; the decision is then final only if a
sign flip already decides it.

Usage:
    uv run python experiments/forcing_gate.py \
        '<lv run>/results/*.json' '<duffing run>/results/*.json' [--out gate.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from glob import glob

import numpy as np

from src.evaluation.rollout_metrics import rollout_metrics

CELLS = ("gru", "cfc", "cfc_lrc")
WIRINGS = ("dense3", "ncp")
ARMS = tuple((c, w) for c in CELLS for w in WIRINGS)
FORCINGS = ("teacher", "mixed")
PAIR = (("gru", "dense3"), ("cfc_lrc", "dense3"))
TAU_MIN = 0.6


def trajectory_vpt(r):
    """Per-test-trajectory VPT(0.5) of one result JSON; -inf for a fit whose
    training diverged."""
    gt, cl = np.asarray(r["ground_truth_traj"]), np.asarray(r["closed_loop_traj"])
    if r.get("training_diverged"):
        return np.full(len(gt), -np.inf)
    return np.array([rollout_metrics(gt[i:i + 1], cl[i:i + 1], r["norm_stats"],
                                     r["system"])["vpt_median"]
                     for i in range(len(gt))])


def kendall_tau_b(a, b):
    """tau_b from #29's src.evaluation.stats; None until that is merged."""
    try:
        from src.evaluation.stats import kendall_tau_b as tau
    except ImportError:
        return None
    return tau(a, b)


def gate(records):
    """records: result dicts -> {system: report} and the decision."""
    vpt, seeds = {}, {}
    for r in records:
        key = (r["system"], r.get("forcing", "teacher"), r["cell"], r["wiring"])
        if key[1] in FORCINGS and key[2:] in ARMS:
            vpt.setdefault(key, {})[r["seed"]] = trajectory_vpt(r)
            seeds.setdefault(r["system"], set()).add(r["seed"])
    missing = [f"{s}/{f}/{c}_{w}_seed{k}" for s in seeds for f in FORCINGS
               for c, w in ARMS for k in sorted(seeds[s])
               if k not in vpt.get((s, f, c, w), {})]
    if missing:
        raise ValueError(f"{len(missing)} pilot fits missing: {missing}")

    report = {}
    for s in sorted(seeds):
        pooled = {(f, a): np.stack([vpt[(s, f, *a)][k] for k in sorted(seeds[s])])
                  for f in FORCINGS for a in ARMS}
        score = {f: [float(np.median(pooled[(f, a)])) for a in ARMS]
                 for f in FORCINGS}
        with np.errstate(invalid="ignore"):            # -inf - -inf: both failed
            diff = {f: float(np.nanmedian(pooled[(f, PAIR[0])]
                                          - pooled[(f, PAIR[1])]))
                    for f in FORCINGS}
        sign = {f: int(np.sign(np.nan_to_num(d))) for f, d in diff.items()}
        tau = kendall_tau_b(score["teacher"], score["mixed"])
        report[s] = {"arms": [f"{c}/{w}" for c, w in ARMS],
                     "median_vpt": score, "tau_b": tau,
                     "pair_median_diff": diff, "pair_sign": sign,
                     "sign_flip": sign["teacher"] * sign["mixed"] < 0}
    flip = any(r["sign_flip"] for r in report.values())
    taus = [r["tau_b"] for r in report.values()]
    if flip or any(t is not None and t < TAU_MIN for t in taus):
        decision = "mixed"
    elif None in taus:
        decision = "undecided (tau_b needs #29)"
    else:
        decision = "teacher"
    return report, decision


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
        report, decision = gate(records)
    except ValueError as e:
        sys.exit(str(e))
    for s, r in report.items():
        print(f"== {s}")
        for i, arm in enumerate(r["arms"]):
            print(f"  {arm:16s} median VPT teacher {r['median_vpt']['teacher'][i]:7.1f}"
                  f"  mixed {r['median_vpt']['mixed'][i]:7.1f}")
        tau = "n/a (needs #29)" if r["tau_b"] is None else f"{r['tau_b']:.3f}"
        print(f"  tau_b {tau} (switch if < {TAU_MIN})")
        print(f"  {PAIR[0][0]}-{PAIR[1][0]} on dense3, paired median VPT diff: "
              f"teacher {r['pair_median_diff']['teacher']:+g}, mixed "
              f"{r['pair_median_diff']['mixed']:+g} -> sign flip "
              f"{r['sign_flip']}")
    print(f"DECISION: primary protocol = {decision}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"report": report, "decision": decision}, f, indent=1)


if __name__ == "__main__":
    main()
