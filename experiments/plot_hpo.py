"""Render figures from the Lotka-Volterra Optuna+wandb HPO result JSONs.

Reads every ``*.json`` written by run_lotka_volterra_hpo.py in ``--indir`` and
writes three PNGs:

  benchmark_results.png  test teacher-forced + closed-loop MSE per (cell, wiring),
                         mean +/- std over seeds -- the headline comparison.
  optuna_hpo.png         per (cell, wiring, seed=0) Optuna search: best-so-far
                         validation loss vs trial, pruned trials marked -- shows
                         what the HPO layer adds beyond a single training run.
  tuned_hyperparams.png  the tuned learning rate per arm + pruning efficiency.

Distinct from plot_lotka_volterra.py, which overlays predator-prey trajectories
from the plain runner (a different JSON schema).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from statistics import mean, pstdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WIRING_ORDER = ["dense", "ncp", "cncp"]
CELL_COLORS = {"cfc_lrc": "#d62728", "cfc": "#1f77b4", "gru": "#2ca02c"}


def load(indir):
    rows = []
    for p in sorted(glob.glob(os.path.join(indir, "*.json"))):
        with open(p) as f:
            rows.append(json.load(f))
    return rows


def _cells(rows):
    return sorted({r["cell"] for r in rows})


def fig_benchmark(rows, out):
    cells = _cells(rows)
    grp = defaultdict(list)
    for r in rows:
        grp[(r["cell"], r["wiring"])].append(r)
    persistence = mean([r["persistence_mse"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    metrics = [("test_teacher_forced_mse", "Teacher-forced MSE (one-step)", False),
               ("test_closed_loop_mse", "Closed-loop MSE (free rollout)", True)]
    x = np.arange(len(WIRING_ORDER))
    width = 0.8 / len(cells)
    for ax, (key, title, show_pers) in zip(axes, metrics):
        for i, cell in enumerate(cells):
            means, stds = [], []
            for w in WIRING_ORDER:
                vals = [r[key] for r in grp.get((cell, w), [])]
                means.append(mean(vals) if vals else 0.0)
                stds.append(pstdev(vals) if len(vals) > 1 else 0.0)
            ax.bar(x + i * width, means, width, yerr=stds, capsize=3,
                   label=cell, color=CELL_COLORS.get(cell), alpha=0.88)
        if show_pers:
            ax.axhline(persistence, ls="--", color="black", lw=1,
                       label=f"persistence ({persistence:.3f})")
        ax.set_title(title)
        ax.set_xticks(x + width * (len(cells) - 1) / 2)
        ax.set_xticklabels(WIRING_ORDER)
        ax.set_xlabel("wiring")
        ax.set_ylabel("MSE (normalised)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Lotka-Volterra (periodic predator-prey) — parameter-matched, "
                 "Optuna-tuned, mean±std over 3 seeds", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _running_best(trials):
    """(x, best-so-far) over COMPLETED trials ordered by number.

    Pruning is read from the trial ``state`` field, not from ``value``: a pruned
    trial keeps its last intermediate value (not None), so the best-so-far line
    must consider only ``state == "COMPLETE"`` trials.
    """
    xs, ys, best = [], [], None
    for t in sorted(trials, key=lambda t: t["number"]):
        if t["state"] != "COMPLETE":
            continue
        v = t["value"]
        best = v if best is None else min(best, v)
        xs.append(t["number"])
        ys.append(best)
    return xs, ys


def fig_optuna(rows, out):
    cells = _cells(rows)
    by = {(r["cell"], r["wiring"], r["seed"]): r for r in rows}
    fig, axes = plt.subplots(len(cells), len(WIRING_ORDER),
                             figsize=(13, 3.2 * len(cells)), squeeze=False)
    for i, cell in enumerate(cells):
        for j, w in enumerate(WIRING_ORDER):
            ax = axes[i][j]
            r = by.get((cell, w, 0))
            if r is None:
                ax.set_visible(False)
                continue
            trials = r["trials"]
            xs, ys = _running_best(trials)
            comp = [(t["number"], t["value"]) for t in trials
                    if t["state"] == "COMPLETE"]
            pru = [(t["number"], t["value"]) for t in trials
                   if t["state"] == "PRUNED" and t["value"] is not None]
            if comp:
                cx, cy = zip(*comp)
                ax.scatter(cx, cy, s=16, color="#888", alpha=0.7,
                           label="completed")
            if pru:
                px, py = zip(*pru)
                ax.scatter(px, py, s=20, marker="x", color="orange", alpha=0.8,
                           label="pruned (stopped early)")
            ax.plot(xs, ys, color=CELL_COLORS.get(cell), lw=2,
                    label="best so far")
            ax.set_yscale("log")
            ax.set_title(f"{cell} / {w}  ({len(comp)} done, {len(pru)} pruned)",
                         fontsize=9)
            if i == len(cells) - 1:
                ax.set_xlabel("trial #")
            if j == 0:
                ax.set_ylabel("val loss (log)")
            ax.grid(alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.suptitle("Optuna HPO search (seed 0): validation loss per trial, "
                 "best-so-far, pruned trials = orange | TPE + MedianPruner",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_hyperparams(rows, out):
    cells = _cells(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    x = np.arange(len(WIRING_ORDER))
    width = 0.8 / len(cells)
    for i, cell in enumerate(cells):
        lrs_m, lrs_s = [], []
        for w in WIRING_ORDER:
            lrs = [r["best_params"]["lr"] for r in rows
                   if r["cell"] == cell and r["wiring"] == w]
            lrs_m.append(mean(lrs) if lrs else 0.0)
            lrs_s.append(pstdev(lrs) if len(lrs) > 1 else 0.0)
        ax.bar(x + i * width, lrs_m, width, yerr=lrs_s, capsize=3,
               label=cell, color=CELL_COLORS.get(cell), alpha=0.88)
    ax.set_title("Optuna-tuned learning rate (mean±std over seeds)")
    ax.set_xticks(x + width * (len(cells) - 1) / 2)
    ax.set_xticklabels(WIRING_ORDER)
    ax.set_xlabel("wiring")
    ax.set_ylabel("best learning rate")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    frac_by_cell = defaultdict(list)
    for r in rows:
        tot = r["n_complete"] + r["n_pruned"]
        if tot:
            frac_by_cell[r["cell"]].append(r["n_pruned"] / tot)
    labels = list(frac_by_cell)
    fracs = [100 * mean(frac_by_cell[c]) for c in labels]
    ax.bar(labels, fracs, color=[CELL_COLORS.get(c) for c in labels], alpha=0.88)
    ax.set_title("MedianPruner efficiency: % of trials pruned early")
    ax.set_ylabel("% trials pruned")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("What Optuna surfaces: per-arm tuned hyper-parameters + pruning",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/lotka_volterra_hpo")
    args = ap.parse_args()
    rows = load(args.indir)
    if not rows:
        print(f"no result JSONs under {args.indir}")
        return
    fig_benchmark(rows, os.path.join(args.indir, "benchmark_results.png"))
    fig_optuna(rows, os.path.join(args.indir, "optuna_hpo.png"))
    fig_hyperparams(rows, os.path.join(args.indir, "tuned_hyperparams.png"))
    print(f"wrote 3 figures to {args.indir}/ "
          f"(benchmark_results, optuna_hpo, tuned_hyperparams).png")


if __name__ == "__main__":
    main()
