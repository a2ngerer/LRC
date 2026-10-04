"""Visualise the active-glimpse sensorimotor benchmark (Iteration 5).

Reads results/active_glimpse/*.json and plots, per (wiring, policy):
  1. accuracy vs glimpse (mean +/- std over seeds); line COLOR = wiring,
     line STYLE = policy (active solid, random dashed) -- does steering reach
     accuracy in fewer glimpses?
  2. glimpses-to-threshold, grouped bars per wiring split by policy.

Usage:
    uv run python experiments/plot_active_glimpse.py [--indir results/active_glimpse]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"tbt_cncp": "#d1495b", "dense": "#55A868", "gru": "#4C72B0"}
LABELS = {"tbt_cncp": "tbt_cNCP (L5-Motor)", "dense": "dense", "gru": "GRU"}
STYLE = {"active": "-", "random": "--"}
HATCH = {"active": "", "random": "//"}


def load(indir: Path):
    groups = defaultdict(list)
    for p in sorted(indir.glob("*.json")):
        try:
            r = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        groups[(r["wiring"], r["policy"])].append(r)
    return groups


def reach(curve, thr):
    return next((t + 1 for t, v in enumerate(curve) if v >= thr), len(curve) + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/active_glimpse")
    ap.add_argument("--out", default=None)
    ap.add_argument("--threshold", type=float, default=0.7)
    args = ap.parse_args()
    indir = Path(args.indir)
    out = Path(args.out) if args.out else indir / "active_glimpse.png"
    groups = load(indir)
    if not groups:
        raise SystemExit(f"no result JSONs in {indir}")

    wirings = [w for w in ("tbt_cncp", "dense", "gru")
               if any(ww == w for ww, _ in groups)]
    policies = [p for p in ("active", "random")
                if any(pp == p for _, pp in groups)]
    baseline = float(next(iter(groups.values()))[0]["majority_baseline"])

    fig, (axc, axb) = plt.subplots(1, 2, figsize=(13, 5.3))

    print(f"\nbaseline {baseline:.3f}  threshold {args.threshold}")
    print(f"{'wiring':<12}{'policy':<9}{'final acc':>15}{'glimpses->thr':>16}")
    print("-" * 52)
    reach_tab = {}
    for w in wirings:
        for p in policies:
            rs = groups.get((w, p))
            if not rs:
                continue
            arr = [np.asarray(r["accuracy_curve"], float) for r in rs]
            L = min(len(a) for a in arr)
            cur = np.stack([a[:L] for a in arr], axis=0)
            x = np.arange(1, L + 1)
            mean, std = cur.mean(0), cur.std(0)
            axc.plot(x, mean, color=COLORS.get(w, "#555"), lw=2.4,
                     ls=STYLE.get(p, "-"), marker="o", markersize=3,
                     label=f"{LABELS.get(w, w)} {p}")
            axc.fill_between(x, mean - std, mean + std,
                             color=COLORS.get(w, "#555"), alpha=0.10)
            rr = np.array([reach(a, args.threshold) for a in arr], float)
            reach_tab[(w, p)] = (rr.mean(), rr.std())
            fin = np.array([r["final_accuracy"] for r in rs])
            print(f"{w:<12}{p:<9}{fin.mean():>9.3f}+/-{fin.std():.3f}"
                  f"{rr.mean():>11.1f}")

    axc.axhline(args.threshold, ls=":", color="gray", lw=1)
    axc.axhline(baseline, ls="--", color="lightgray", lw=1,
                label=f"Baseline {baseline:.2f}")
    axc.set_xlabel("Anzahl Glimpses")
    axc.set_ylabel("Objekt-Klassifikations-Accuracy")
    axc.set_title("Aktives Abtasten: Accuracy vs. #Glimpses (Stil = Policy)")
    axc.legend(loc="lower right", fontsize=8)
    axc.grid(alpha=0.3)

    width = 0.8 / max(1, len(policies))
    xw = np.arange(len(wirings))
    for j, p in enumerate(policies):
        mm = [reach_tab.get((w, p), (np.nan, 0))[0] for w in wirings]
        ss = [reach_tab.get((w, p), (np.nan, 0))[1] for w in wirings]
        off = (j - (len(policies) - 1) / 2) * width
        axb.bar(xw + off, mm, width, yerr=ss, capsize=4, label=p, alpha=0.9,
                edgecolor="white", hatch=HATCH.get(p),
                color=[COLORS.get(w, "#555") for w in wirings])
        for xi, v in zip(xw + off, mm):
            if not np.isnan(v):
                axb.text(xi, v, f"{v:.1f}", ha="center", va="bottom",
                         fontsize=8, fontweight="bold")
    axb.set_xticks(xw)
    axb.set_xticklabels([LABELS.get(w, w) for w in wirings], fontsize=8)
    axb.set_ylabel(f"Glimpses bis Accuracy {args.threshold:.1f}")
    axb.set_title("Sample-Effizienz: aktiv vs. zufällig")
    axb.legend(loc="upper right", fontsize=9)
    axb.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
