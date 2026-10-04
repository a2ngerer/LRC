"""Visualise the tbt_cNCP active-sensing benchmark.

Reads the per-run JSONs from run_active_sensing_benchmark.py and renders the
headline Thousand-Brains view:

  1. accuracy vs number of glimpses (mean +/- std over seeds), one line per
     wiring -- the sample-efficiency curve; tbt_cncp vs its no-location ablation
     is the H1 test;
  2. final accuracy (clean vs occluded) per wiring, with the majority baseline.

Usage:
    uv run python experiments/plot_active_sensing.py [--indir results/active_sensing]
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

COLORS = {"tbt_cncp": "#3fb7a6", "tbt_cncp_film": "#d1495b",
          "tbt_cncp_concat": "#e0a53f",
          "tbt_cncp_both": "#b98cdd", "tbt_cncp_noloc": "#9aa0a6",
          "ncp": "#4C72B0", "dense": "#55A868"}
LABELS = {"tbt_cncp": "tbt_cNCP (Gate)",
          "tbt_cncp_film": "tbt_cNCP (FiLM)",
          "tbt_cncp_concat": "tbt_cNCP (concat)",
          "tbt_cncp_both": "tbt_cNCP (Gate+concat)",
          "tbt_cncp_noloc": "tbt_cNCP ohne Location",
          "ncp": "NCP", "dense": "dense"}
ORDER = ["tbt_cncp", "tbt_cncp_film", "tbt_cncp_concat", "tbt_cncp_both",
         "tbt_cncp_noloc", "ncp", "dense"]


def load(indir: Path):
    groups = defaultdict(list)
    for p in sorted(indir.glob("*.json")):
        try:
            r = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        groups[r["wiring"]].append(r)
    return groups


def stack(rs, key):
    arr = [np.asarray(r[key], float) for r in rs]
    L = min(len(a) for a in arr)
    return np.stack([a[:L] for a in arr], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/active_sensing")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    indir = Path(args.indir)
    out = Path(args.out) if args.out else indir / "tbt_cncp_active_sensing.png"
    groups = load(indir)
    if not groups:
        raise SystemExit(f"no result JSONs in {indir}")

    wirings = [w for w in ORDER if w in groups]
    baseline = float(next(iter(groups.values()))[0]["majority_baseline"])

    has_occ_sweep = any("occlusion_accs" in r for rs in groups.values()
                        for r in rs)
    ncol = 3 if has_occ_sweep else 2
    fig, axes = plt.subplots(1, ncol, figsize=(7 * ncol, 5.5))
    axc, axb = axes[0], axes[1]
    axo = axes[2] if has_occ_sweep else None

    # --- 1. accuracy vs glimpses ---
    print(f"\nmajority baseline {baseline:.3f}")
    print(f"{'wiring':<24}{'final acc':>16}{'occluded':>14}")
    print("-" * 54)
    for w in wirings:
        rs = groups[w]
        cur = stack(rs, "accuracy_curve")
        x = np.arange(1, cur.shape[1] + 1)
        mean, std = cur.mean(0), cur.std(0)
        lw = 2.8 if w.startswith("tbt") else 1.8
        axc.plot(x, mean, color=COLORS[w], lw=lw, label=LABELS[w],
                 marker="o", markersize=3)
        axc.fill_between(x, mean - std, mean + std, color=COLORS[w], alpha=0.12)
        fin = np.array([r["final_accuracy"] for r in rs])
        occ = np.array([r["occluded_final"] for r in rs])
        print(f"{LABELS[w]:<24}{fin.mean():>10.3f}+/-{fin.std():.3f}"
              f"{occ.mean():>9.3f} ({len(rs)})")
    axc.axhline(baseline, ls="--", color="gray", lw=1,
                label=f"Baseline {baseline:.2f}")
    axc.set_xlabel("Anzahl Glimpses (Sensationen)")
    axc.set_ylabel("Objekt-Klassifikations-Accuracy")
    axc.set_title("Sample-Effizienz: Accuracy vs. #Glimpses")
    axc.legend(loc="lower right", fontsize=9)
    axc.grid(alpha=0.3)

    # --- 2. final accuracy clean vs occluded ---
    x = np.arange(len(wirings))
    width = 0.38
    finm = [np.mean([r["final_accuracy"] for r in groups[w]]) for w in wirings]
    fins = [np.std([r["final_accuracy"] for r in groups[w]]) for w in wirings]
    occm = [np.mean([r["occluded_final"] for r in groups[w]]) for w in wirings]
    occs = [np.std([r["occluded_final"] for r in groups[w]]) for w in wirings]
    axb.bar(x - width / 2, finm, width, yerr=fins, capsize=4, label="clean",
            color=[COLORS[w] for w in wirings], alpha=0.9, edgecolor="white")
    axb.bar(x + width / 2, occm, width, yerr=occs, capsize=4, label="occluded",
            color=[COLORS[w] for w in wirings], alpha=0.45, edgecolor="white")
    axb.axhline(baseline, ls="--", color="gray", lw=1)
    axb.set_xticks(x)
    axb.set_xticklabels([LABELS[w] for w in wirings], rotation=18, ha="right",
                        fontsize=8)
    axb.set_ylabel("finale Accuracy (nach allen Glimpses)")
    axb.set_title("Final: clean vs occluded")
    axb.legend(loc="upper right", fontsize=9)
    axb.grid(axis="y", alpha=0.3)

    # --- 3. graded occlusion robustness (Iteration 3) ---
    if axo is not None:
        print(f"\n{'wiring':<24}" + "".join(f"occ@{f:>4}" for f in
              next(r for rs in groups.values() for r in rs
                   if "occlusion_fracs" in r)["occlusion_fracs"]))
        for w in wirings:
            rs = [r for r in groups[w] if "occlusion_accs" in r]
            if not rs:
                continue
            fr = np.asarray(rs[0]["occlusion_fracs"], float)
            acc = np.stack([np.asarray(r["occlusion_accs"], float)
                            for r in rs], 0)
            m, s = acc.mean(0), acc.std(0)
            lw = 2.8 if w.startswith("tbt") else 1.8
            axo.plot(fr, m, color=COLORS[w], lw=lw, marker="o", markersize=4,
                     label=LABELS[w])
            axo.fill_between(fr, m - s, m + s, color=COLORS[w], alpha=0.12)
            print(f"{LABELS[w]:<24}" + "".join(f"{v:>7.3f}" for v in m))
        axo.axhline(baseline, ls="--", color="gray", lw=1,
                    label=f"Baseline {baseline:.2f}")
        axo.set_xlabel("Verdeckungsgrad (Objekt-Anteil von links)")
        axo.set_ylabel("finale Accuracy")
        axo.set_title("Occlusion-Robustheit: Accuracy vs. Verdeckung")
        axo.legend(loc="lower left", fontsize=8)
        axo.grid(alpha=0.3)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
