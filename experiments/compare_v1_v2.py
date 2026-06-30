#!/usr/bin/env python
"""Build v1-only / v2-only NRMSE overviews and side-by-side v1-vs-v2 figures.

Prerequisite: render the per-version figures first, e.g.

    uv run python experiments/plot_results.py --runs results/runs     --out results/figures_v1
    uv run python experiments/plot_results.py --runs results/runs_v2  --out results/figures_v2

Then:

    uv run python experiments/compare_v1_v2.py

This adds an ``nrmse_overview.png`` to each version's figure dir and writes a
horizontal ``v1 | v2`` montage for every figure present in BOTH dirs
(loss_<system>, phase_<system>, nrmse_overview) into --out. gradflow_* are
per-cell and the cell sets differ between versions, so they are not paired.
"""
import argparse
import collections
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


def mean_nrmse_by_variant(runs_dir):
    """(cell[+clip], wiring) -> list of full-trajectory NRMSE over systems/seeds."""
    vals = collections.defaultdict(list)
    for f in glob.glob(os.path.join(runs_dir, "*.json")):
        with open(f) as fh:
            r = json.load(fh)
        run = r["run"]
        n = r.get("evaluation", {}).get("nrmse")
        if n is None or not np.isfinite(n):
            continue
        clip = run.get("clip_norm") or 0
        variant = run["cell"] + ("+clip" if clip else "")
        vals[(variant, run["wiring"])].append(n)
    return vals


def nrmse_overview(runs_dir, out_png, title):
    vals = mean_nrmse_by_variant(runs_dir)
    variants = sorted({v for (v, _w) in vals})
    if not variants:
        return False
    dense = [np.nanmean(vals.get((v, "dense"), [np.nan])) for v in variants]
    ncp = [np.nanmean(vals.get((v, "ncp"), [np.nan])) for v in variants]
    order = np.argsort(np.nan_to_num(dense, nan=9e9))
    variants = [variants[i] for i in order]
    dense = [dense[i] for i in order]
    ncp = [ncp[i] for i in order]
    x = np.arange(len(variants))
    w = 0.4
    fig, ax = plt.subplots(figsize=(max(7.0, 1.05 * len(variants)), 5.0))
    bars = list(ax.bar(x - w / 2, dense, w, label="dense", color="#2b6cb0"))
    bars += list(ax.bar(x + w / 2, ncp, w, label="ncp", color="#c05621"))
    for b in bars:
        h = b.get_height()
        if np.isfinite(h):
            ax.text(b.get_x() + b.get_width() / 2, h, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=45, ha="right")
    ax.set_ylabel("mean NRMSE over systems (lower = better)")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=135)
    plt.close(fig)
    return True


def stitch(v1_png, v2_png, out_png, name):
    img1, img2 = mpimg.imread(v1_png), mpimg.imread(v2_png)
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    for ax, img, tag in zip(axes, (img1, img2),
                            ("v1  (baseline cells)", "v2  (fixed cells)")):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(tag, fontsize=13)
    fig.suptitle(name, fontsize=15)
    plt.tight_layout()
    plt.savefig(out_png, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="v1-vs-v2 figure comparison montages")
    p.add_argument("--v1-runs", default="results/runs")
    p.add_argument("--v2-runs", default="results/runs_v2")
    p.add_argument("--v1-figs", default="results/figures_v1")
    p.add_argument("--v2-figs", default="results/figures_v2")
    p.add_argument("--out", default="results/figures_v1_vs_v2")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    nrmse_overview(args.v1_runs, os.path.join(args.v1_figs, "nrmse_overview.png"),
                   "v1 benchmark - mean NRMSE per cell")
    nrmse_overview(args.v2_runs, os.path.join(args.v2_figs, "nrmse_overview.png"),
                   "v2 benchmark - mean NRMSE per cell")

    v1_files = {os.path.basename(f) for f in glob.glob(os.path.join(args.v1_figs, "*.png"))}
    v2_files = {os.path.basename(f) for f in glob.glob(os.path.join(args.v2_figs, "*.png"))}
    common = sorted(v1_files & v2_files)
    for name in common:
        stitch(os.path.join(args.v1_figs, name), os.path.join(args.v2_figs, name),
               os.path.join(args.out, name), name[:-4])

    print(f"v1 figs: {len(v1_files)} | v2 figs: {len(v2_files)} | side-by-side: {len(common)}")
    print("side-by-side:", ", ".join(common))


if __name__ == "__main__":
    main()
