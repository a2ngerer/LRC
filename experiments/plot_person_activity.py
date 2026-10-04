"""Visualise the cNCP-vs-NCP Person-Activity benchmark.

Reads the per-run result JSONs written by run_person_activity_benchmark.py
(one file per cell x wiring x seed), aggregates over seeds, and produces:

  1. a grouped bar chart of final test accuracy (mean +/- std over seeds),
     grouped by cell, ncp vs cncp side by side, with the majority-class
     baseline drawn in; and
  2. per-cell validation-accuracy learning curves (mean over seeds, +/- std
     band), ncp vs cncp.

It also prints a compact text summary table. Read-only w.r.t. the results.

Usage:
    uv run python experiments/plot_person_activity.py \
        [--indir results/person_activity] [--out results/person_activity/cncp_vs_ncp.png]
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

WIRING_COLORS = {"ncp": "#4C72B0", "cncp": "#C44E52", "dense": "#55A868"}
WIRING_LABELS = {"ncp": "NCP", "cncp": "cNCP", "dense": "dense"}


def load_runs(indir: Path) -> list[dict]:
    runs = []
    for p in sorted(indir.glob("*.json")):
        try:
            runs.append(json.loads(p.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    return runs


def aggregate(runs: list[dict]) -> dict:
    """Group by (cell, wiring) -> dict of aggregated arrays over seeds."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in runs:
        groups[(r["cell"], r["wiring"])].append(r)

    agg = {}
    for (cell, wiring), rs in groups.items():
        best = np.array([r.get("best_test_accuracy", r["test_accuracy"]) for r in rs])
        final = np.array([r["test_accuracy"] for r in rs])
        curves = [np.asarray(r["val_accuracy_curve"], dtype=float) for r in rs]
        L = min(len(c) for c in curves)
        curve_stack = np.stack([c[:L] for c in curves], axis=0)
        agg[(cell, wiring)] = {
            "n": len(rs),
            "seeds": sorted(r["seed"] for r in rs),
            "best_all": best.tolist(),
            "best_mean": float(best.mean()), "best_std": float(best.std()),
            "final_mean": float(final.mean()), "final_std": float(final.std()),
            "params": int(rs[0].get("params_effective", rs[0].get("params", 0))),
            "curve_mean": curve_stack.mean(axis=0),
            "curve_std": curve_stack.std(axis=0),
            "baseline": float(rs[0].get("majority_baseline", np.nan)),
        }
    return agg


def print_summary(agg: dict) -> None:
    cells = sorted({c for c, _ in agg})
    wirings = [w for w in ("ncp", "cncp", "dense") if any(wi == w for _, wi in agg)]
    baseline = next((v["baseline"] for v in agg.values()), float("nan"))
    print(f"\nMajority-class baseline: {baseline:.4f}\n")
    header = f"{'cell':<10} " + " ".join(f"{WIRING_LABELS[w]:>18}" for w in wirings) + f"  {'cNCP-NCP':>10}"
    print(header)
    print("-" * len(header))
    for cell in cells:
        row = f"{cell:<10} "
        cells_acc = {}
        for w in wirings:
            v = agg.get((cell, w))
            if v:
                cells_acc[w] = v["best_mean"]
                row += f" {v['best_mean']:.4f}+/-{v['best_std']:.3f} ({v['n']})"
            else:
                row += f" {'--':>18}"
        if "cncp" in cells_acc and "ncp" in cells_acc:
            delta = cells_acc["cncp"] - cells_acc["ncp"]
            row += f"  {delta:+.4f}"
        print(row)
    print("\nParams (mean over runs):")
    for cell in cells:
        parts = [f"{WIRING_LABELS[w]} {agg[(cell, w)]['params']}" for w in wirings if (cell, w) in agg]
        print(f"  {cell:<10} " + " | ".join(parts))


def make_plot(agg: dict, out: Path) -> None:
    cells = sorted({c for c, _ in agg})
    wirings = [w for w in ("ncp", "cncp", "dense") if any(wi == w for _, wi in agg)]
    baseline = next((v["baseline"] for v in agg.values()), float("nan"))

    fig, (ax_bar, ax_curve) = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- 1. grouped bar chart -------------------------------------------------
    x = np.arange(len(cells))
    width = 0.8 / max(1, len(wirings))
    all_pts = [v for c in cells for w in wirings if (c, w) in agg
               for v in agg[(c, w)]["best_all"]]
    for i, w in enumerate(wirings):
        means = [agg[(c, w)]["best_mean"] if (c, w) in agg else 0.0 for c in cells]
        stds = [agg[(c, w)]["best_std"] if (c, w) in agg else 0.0 for c in cells]
        offs = (i - (len(wirings) - 1) / 2) * width
        bars = ax_bar.bar(x + offs, means, width, yerr=stds, capsize=4,
                          label=WIRING_LABELS[w], color=WIRING_COLORS[w],
                          alpha=0.85, edgecolor="white")
        # Overlay the individual seed values -- honest at n=3, where an error bar
        # alone hides how much the seeds actually spread.
        for j, c in enumerate(cells):
            if (c, w) in agg:
                vals = agg[(c, w)]["best_all"]
                ax_bar.scatter([x[j] + offs] * len(vals), vals, color="black",
                               s=16, zorder=5, alpha=0.75)
        for b, m in zip(bars, means):
            if m > 0:
                ax_bar.text(b.get_x() + b.get_width() / 2, m + 0.0012, f"{m:.3f}",
                            ha="center", va="bottom", fontsize=8, fontweight="bold")
    # Zoom the y-axis to the relevant band so the small wiring gap is visible;
    # the majority baseline (~0.37) sits far below, noted in text instead of a line.
    if all_pts:
        ax_bar.set_ylim(min(all_pts) - 0.02, max(all_pts) + 0.015)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(cells)
    ax_bar.set_ylabel("test accuracy (bars = mean, dots = seeds)")
    ax_bar.set_title("Person Activity: cNCP vs NCP wiring")
    ax_bar.legend(loc="lower right", fontsize=9)
    ax_bar.grid(axis="y", alpha=0.3)
    if np.isfinite(baseline):
        ax_bar.text(0.02, 0.03, f"majority baseline {baseline:.3f} (far below)",
                    transform=ax_bar.transAxes, fontsize=8, color="gray",
                    style="italic")

    # --- 2. learning curves ---------------------------------------------------
    styles = {"cfc_lrc": "-", "gru": "--", "cfc": ":", "lrc": "-.", "ltc": (0, (3, 1, 1, 1))}
    for (cell, w), v in sorted(agg.items()):
        epochs = np.arange(1, len(v["curve_mean"]) + 1)
        ax_curve.plot(epochs, v["curve_mean"], styles.get(cell, "-"),
                      color=WIRING_COLORS[w], lw=1.8,
                      label=f"{WIRING_LABELS[w]} / {cell}")
        ax_curve.fill_between(epochs, v["curve_mean"] - v["curve_std"],
                              v["curve_mean"] + v["curve_std"],
                              color=WIRING_COLORS[w], alpha=0.12)
    if np.isfinite(baseline):
        ax_curve.axhline(baseline, ls="--", color="gray", lw=1)
    ax_curve.set_xlabel("epoch")
    ax_curve.set_ylabel("validation accuracy")
    ax_curve.set_title("Learning curves (mean over seeds)")
    ax_curve.legend(loc="lower right", fontsize=8)
    ax_curve.grid(alpha=0.3)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"\nWrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/person_activity")
    ap.add_argument("--out", default="results/person_activity/cncp_vs_ncp.png")
    args = ap.parse_args()

    indir = Path(args.indir)
    runs = load_runs(indir)
    if not runs:
        raise SystemExit(f"no result JSONs found in {indir}")
    agg = aggregate(runs)
    print(f"Loaded {len(runs)} runs across {len(agg)} (cell, wiring) groups.")
    print_summary(agg)
    make_plot(agg, Path(args.out))


if __name__ == "__main__":
    main()
