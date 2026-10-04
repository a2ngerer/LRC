"""Visualise the K-column voting benchmark, comparing location mechanisms.

Reads results/active_sensing_voting/*.json and plots, for each (K, location_mode):
  1. accuracy vs glimpses-per-column (mean +/- std over seeds), line STYLE =
     location mechanism (FiLM solid, concat dashed), line COLOR = K;
  2. glimpses-to-threshold, grouped bars per K split by location mode -- the
     TBT sample-efficiency claim AND the Iteration-2 question: does FiLM voting
     reach the accuracy threshold in fewer glimpses than concat voting?

Usage:
    uv run python experiments/plot_voting.py [--indir results/active_sensing_voting]
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

CMAP = plt.get_cmap("viridis")
MODE_STYLE = {"film": "-", "concat": "--", "gate": ":"}
MODE_LABEL = {"film": "FiLM", "concat": "concat", "gate": "Gate"}
MODE_HATCH = {"film": "", "concat": "//", "gate": ".."}


def load(indir: Path):
    groups = defaultdict(list)
    for p in sorted(indir.glob("*.json")):
        try:
            r = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        mode = r.get("location_mode", "concat")
        groups[(int(r["n_columns"]), mode)].append(r)
    return groups


def reach(curve, thr):
    # glimpses to first cross the threshold; "never" penalised as T+1
    return next((t + 1 for t, v in enumerate(curve) if v >= thr), len(curve) + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/active_sensing_voting")
    ap.add_argument("--out", default=None)
    ap.add_argument("--threshold", type=float, default=0.7)
    args = ap.parse_args()
    indir = Path(args.indir)
    out = Path(args.out) if args.out else indir / "voting_sample_efficiency.png"
    groups = load(indir)
    if not groups:
        raise SystemExit(f"no result JSONs in {indir}")

    ks = sorted({k for k, _ in groups})
    modes = [m for m in ("concat", "film", "gate")
             if any(mm == m for _, mm in groups)]
    baseline = float(next(iter(groups.values()))[0]["majority_baseline"])
    kcolor = {k: CMAP(0.15 + 0.7 * i / max(1, len(ks) - 1))
              for i, k in enumerate(ks)}

    fig, (axc, axb) = plt.subplots(1, 2, figsize=(13, 5.3))

    print(f"\nmajority baseline {baseline:.3f}  threshold {args.threshold}")
    print(f"{'K':<4}{'mode':<9}{'final acc':>15}{'glimpses->thr':>16}")
    print("-" * 44)
    reach_tab = {}   # (k, mode) -> (mean, std)
    for k in ks:
        for m in modes:
            rs = groups.get((k, m))
            if not rs:
                continue
            arr = [np.asarray(r["accuracy_curve"], float) for r in rs]
            L = min(len(a) for a in arr)
            cur = np.stack([a[:L] for a in arr], axis=0)
            x = np.arange(1, L + 1)
            mean, std = cur.mean(0), cur.std(0)
            axc.plot(x, mean, color=kcolor[k], lw=2.2,
                     ls=MODE_STYLE.get(m, "-"), marker="o", markersize=3,
                     label=f"K={k} {MODE_LABEL.get(m, m)}")
            axc.fill_between(x, mean - std, mean + std, color=kcolor[k],
                             alpha=0.10)
            rr = np.array([reach(a, args.threshold) for a in arr], float)
            reach_tab[(k, m)] = (rr.mean(), rr.std())
            fin = np.array([r["final_accuracy"] for r in rs])
            print(f"{k:<4}{m:<9}{fin.mean():>9.3f}+/-{fin.std():.3f}"
                  f"{rr.mean():>11.1f}")

    axc.axhline(args.threshold, ls=":", color="gray", lw=1)
    axc.axhline(baseline, ls="--", color="lightgray", lw=1)
    axc.set_xlabel("Glimpses pro Säule")
    axc.set_ylabel("Objekt-Klassifikations-Accuracy")
    axc.set_title("Voting: Accuracy vs. #Glimpses (Stil = Mechanismus, Farbe = K)")
    axc.legend(loc="lower right", fontsize=8, ncol=max(1, len(modes)))
    axc.grid(alpha=0.3)

    # grouped bars: glimpses-to-threshold, K on x, one bar per location mode
    width = 0.8 / max(1, len(modes))
    xk = np.arange(len(ks))
    for j, m in enumerate(modes):
        mm = [reach_tab.get((k, m), (np.nan, 0))[0] for k in ks]
        ss = [reach_tab.get((k, m), (np.nan, 0))[1] for k in ks]
        off = (j - (len(modes) - 1) / 2) * width
        axb.bar(xk + off, mm, width, yerr=ss, capsize=4,
                label=MODE_LABEL.get(m, m), alpha=0.9, edgecolor="white",
                hatch=MODE_HATCH.get(m))
        for xi, v in zip(xk + off, mm):
            if not np.isnan(v):
                axb.text(xi, v, f"{v:.1f}", ha="center", va="bottom",
                         fontsize=8, fontweight="bold")
    axb.set_xticks(xk)
    axb.set_xticklabels([str(k) for k in ks])
    axb.set_xlabel("Anzahl Säulen K")
    axb.set_ylabel(f"Glimpses bis Accuracy {args.threshold:.1f}")
    axb.set_title("Sample-Effizienz: FiLM- vs concat-Voting")
    axb.legend(loc="upper right", fontsize=9)
    axb.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
