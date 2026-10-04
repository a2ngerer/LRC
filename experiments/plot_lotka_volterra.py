"""Visualise the predator-prey (Lotka-Volterra) cNCP-vs-NCP rollout benchmark.

Reads the per-run JSONs written by run_lotka_volterra_benchmark.py and renders
three interpretable views comparing model output against ground truth:

  1. PHASE SPACE (headline): the (prey, predator) orbit -- ground truth vs each
     wiring's CLOSED-LOOP rollout. A model that learned the dynamics traces the
     limit cycle; one that did not spirals in or out.
  2. TIME SERIES: prey and predator vs time, ground truth vs closed-loop rollout.
  3. MSE bars + learning curves: teacher-forced and closed-loop test MSE per
     wiring (mean +/- std over seeds), with the persistence baseline drawn in.

Overlays use one seed (default 0); the MSE panel aggregates over all seeds.

Usage:
    uv run python experiments/plot_lotka_volterra.py \
        [--indir results/lotka_volterra] [--cell cfc_lrc] [--seed 0]
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

WIRING_COLORS = {"dense": "#55A868", "ncp": "#4C72B0", "cncp": "#C44E52",
                 "tbt_cncp_concat": "#E0A53F", "tbt_cncp_noloc": "#9AA0A6",
                 "tbt_cncp": "#8172B3", "tbt_cncp_both": "#CCB974",
                 "tbt_cncp_film": "#D1495B"}
WIRING_LABELS = {"dense": "dense", "ncp": "NCP", "cncp": "cNCP",
                 "tbt_cncp_concat": "tbt_cNCP", "tbt_cncp_noloc": "tbt_cNCP (no-loc)",
                 "tbt_cncp": "tbt_cNCP (gate)",
                 "tbt_cncp_both": "tbt_cNCP (gate+concat)",
                 "tbt_cncp_film": "tbt_cNCP (FiLM)"}
# Preference order for the bar / summary panels; only wirings actually present in
# the result dir are shown.
WIRING_ORDER = ("dense", "ncp", "cncp", "tbt_cncp_noloc", "tbt_cncp",
                "tbt_cncp_film", "tbt_cncp_concat", "tbt_cncp_both")
# Phase / time-series overlays: the most informative arms, to keep the orbit
# plots readable (the rest live in the MSE / horizon panels). Includes FiLM so
# the affine-gate rollout can be compared with concat directly.
OVERLAY_WIRINGS = ("ncp", "tbt_cncp_concat", "tbt_cncp_film")


# Per-system axis / title labels so the same plotter serves predator-prey and
# duffing (whose two state dims are position / velocity, not two species).
_SYS_AXES = {
    "periodic_predator_prey": ("prey", "predator", "Predator-prey"),
    "duffing": ("Position x", "Geschwindigkeit v", "Duffing-Oszillator"),
}


def sys_axes(run: dict) -> tuple[str, str, str]:
    system = run.get("system", "periodic_predator_prey")
    return _SYS_AXES.get(system, ("state y0", "state y1", system))


def load_runs(indir: Path) -> list[dict]:
    runs = []
    for p in sorted(indir.glob("*.json")):
        try:
            runs.append(json.loads(p.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    return runs


def pick(runs, cell, wiring, seed):
    for r in runs:
        if r["cell"] == cell and r["wiring"] == wiring and r["seed"] == seed:
            return r
    return None


def example_indices(n_test, n_examples):
    n = min(n_examples, n_test)
    return np.linspace(0, n_test - 1, n, dtype=int)


def plot_phase(runs, cell, seed, out, n_examples=4):
    truth_run = next((pick(runs, cell, w, seed) for w in OVERLAY_WIRINGS
                      if pick(runs, cell, w, seed)), None)
    if truth_run is None:
        return
    gt = np.asarray(truth_run["ground_truth_traj"])         # (N, T+1, 2)
    xl, yl, title = sys_axes(truth_run)
    idx = example_indices(gt.shape[0], n_examples)
    fig, axes = plt.subplots(1, len(idx), figsize=(4.2 * len(idx), 4.2))
    axes = np.atleast_1d(axes)
    for ax, si in zip(axes, idx):
        ax.plot(gt[si, :, 0], gt[si, :, 1], color="black", lw=2.4,
                label="ground truth", alpha=0.85)
        for w in OVERLAY_WIRINGS:
            r = pick(runs, cell, w, seed)
            if r is None:
                continue
            cl = np.asarray(r["closed_loop_traj"])
            ax.plot(cl[si, :, 0], cl[si, :, 1], color=WIRING_COLORS[w],
                    lw=1.6, ls="--", label=f"{WIRING_LABELS[w]} rollout",
                    alpha=0.9)
        ax.scatter([gt[si, 0, 0]], [gt[si, 0, 1]], color="black", s=45,
                   zorder=5, marker="o", label="start")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(f"test trajectory {si}")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"{title} phase space — ground truth vs closed-loop "
                 f"rollout ({cell})", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"Wrote {out}")


def plot_timeseries(runs, cell, seed, out, n_examples=4):
    truth_run = next((pick(runs, cell, w, seed) for w in OVERLAY_WIRINGS
                      if pick(runs, cell, w, seed)), None)
    if truth_run is None:
        return
    gt = np.asarray(truth_run["ground_truth_traj"])
    dt = truth_run["dt"]
    xl, yl, title = sys_axes(truth_run)
    t = np.arange(gt.shape[1]) * dt
    idx = example_indices(gt.shape[0], n_examples)
    fig, axes = plt.subplots(len(idx), 2, figsize=(12, 2.4 * len(idx)),
                             sharex=True)
    axes = np.atleast_2d(axes)
    for row, si in enumerate(idx):
        for col, (dim, name) in enumerate(((0, xl), (1, yl))):
            ax = axes[row, col]
            ax.plot(t, gt[si, :, dim], color="black", lw=2.2,
                    label="ground truth", alpha=0.85)
            for w in OVERLAY_WIRINGS:
                r = pick(runs, cell, w, seed)
                if r is None:
                    continue
                cl = np.asarray(r["closed_loop_traj"])
                ax.plot(t, cl[si, :, dim], color=WIRING_COLORS[w], lw=1.5,
                        ls="--", label=f"{WIRING_LABELS[w]}", alpha=0.9)
            ax.grid(alpha=0.3)
            if row == 0:
                ax.set_title(name)
            if col == 0:
                ax.set_ylabel(f"traj {si}")
    axes[0, 1].legend(loc="upper right", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("time")
    fig.suptitle(f"{title} time series — ground truth vs closed-loop "
                 f"rollout ({cell})", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"Wrote {out}")


def aggregate_mse(runs):
    groups = defaultdict(list)
    for r in runs:
        groups[(r["cell"], r["wiring"])].append(r)
    agg = {}
    for key, rs in groups.items():
        tf_mse = np.array([r["teacher_forced_mse"] for r in rs])
        cl_mse = np.array([r["closed_loop_mse"] for r in rs])
        agg[key] = {
            "n": len(rs),
            "tf_mean": float(tf_mse.mean()), "tf_std": float(tf_mse.std()),
            "cl_mean": float(cl_mse.mean()), "cl_std": float(cl_mse.std()),
            "persistence": float(rs[0]["persistence_mse"]),
            "params": int(rs[0].get("params_effective", rs[0].get("params", 0))),
            "curves": [np.asarray(r["val_loss_curve"], float) for r in rs],
        }
    return agg


def plot_mse(runs, cell, out):
    agg = aggregate_mse(runs)
    wirings = [w for w in WIRING_ORDER if (cell, w) in agg]
    if not wirings:
        return
    persistence = agg[(cell, wirings[0])]["persistence"]

    fig, (ax_bar, ax_curve) = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(2)   # teacher-forced, closed-loop
    width = 0.8 / max(1, len(wirings))
    for i, w in enumerate(wirings):
        a = agg[(cell, w)]
        means = [a["tf_mean"], a["cl_mean"]]
        stds = [a["tf_std"], a["cl_std"]]
        offs = (i - (len(wirings) - 1) / 2) * width
        ax_bar.bar(x + offs, means, width, yerr=stds, capsize=4,
                   label=f"{WIRING_LABELS[w]} ({a['params']}p)",
                   color=WIRING_COLORS[w], alpha=0.85, edgecolor="white")
        for xi, m in zip(x + offs, means):
            ax_bar.text(xi, m, f"{m:.3f}", ha="center", va="bottom", fontsize=7)
    ax_bar.axhline(persistence, ls="--", color="gray", lw=1,
                   label=f"persistence {persistence:.3f}")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(["teacher-forced", "closed-loop"])
    ax_bar.set_ylabel("test MSE (normalised, mean +/- std over seeds)")
    ax_bar.set_title(f"Predator-prey next-step MSE ({cell})")
    ax_bar.legend(loc="upper left", fontsize=8)
    ax_bar.grid(axis="y", alpha=0.3)

    for w in wirings:
        curves = agg[(cell, w)]["curves"]
        L = min(len(c) for c in curves)
        stack = np.stack([c[:L] for c in curves], axis=0)
        epochs = np.arange(1, L + 1)
        ax_curve.plot(epochs, stack.mean(0), color=WIRING_COLORS[w], lw=1.8,
                      label=WIRING_LABELS[w])
        ax_curve.fill_between(epochs, stack.mean(0) - stack.std(0),
                              stack.mean(0) + stack.std(0),
                              color=WIRING_COLORS[w], alpha=0.12)
    ax_curve.set_yscale("log")
    ax_curve.set_xlabel("epoch")
    ax_curve.set_ylabel("validation MSE (log)")
    ax_curve.set_title("Learning curves (mean over seeds)")
    ax_curve.legend(loc="upper right", fontsize=8)
    ax_curve.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"Wrote {out}")


def print_summary(runs, cell):
    agg = aggregate_mse(runs)
    wirings = [w for w in WIRING_ORDER if (cell, w) in agg]
    if not wirings:
        return
    print(f"\ncell {cell} | persistence baseline "
          f"{agg[(cell, wirings[0])]['persistence']:.4f}")
    print(f"{'wiring':<8} {'params':>8} {'TF-MSE':>16} {'CL-MSE':>16}")
    print("-" * 52)
    for w in wirings:
        a = agg[(cell, w)]
        print(f"{WIRING_LABELS[w]:<8} {a['params']:>8} "
              f"{a['tf_mean']:.4f}+/-{a['tf_std']:.3f}  "
              f"{a['cl_mean']:.4f}+/-{a['cl_std']:.3f} ({a['n']})")


def plot_horizon(runs, cell, out):
    """Closed-loop MSE vs rollout horizon: which wiring stays accurate longest.

    The headline stability view -- teacher-forced MSE hides compounding error,
    but a wiring that only learned a local one-step map diverges fast in
    closed loop; one that captured the dynamics keeps the per-step error low
    for many more steps.
    """
    groups = defaultdict(list)
    for r in runs:
        if r["cell"] != cell:
            continue
        per = r.get("closed_loop_mse_per_step")
        if per:
            groups[r["wiring"]].append(np.asarray(per, float))
    wirings = [w for w in WIRING_ORDER if w in groups]
    if not wirings:
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for w in wirings:
        curves = groups[w]
        L = min(len(c) for c in curves)
        stack = np.stack([c[:L] for c in curves], axis=0)
        steps = np.arange(1, L + 1)
        m, s = stack.mean(0), stack.std(0)
        lw = 2.6 if w.startswith("tbt") or w == "cncp" else 1.8
        ax.plot(steps, m, color=WIRING_COLORS.get(w, "#555"), lw=lw,
                label=WIRING_LABELS.get(w, w))
        ax.fill_between(steps, np.maximum(m - s, 1e-9), m + s,
                        color=WIRING_COLORS.get(w, "#555"), alpha=0.12)
    ax.set_yscale("log")
    ax.set_xlabel("Rollout-Schritt (Vorhersage-Horizont)")
    ax.set_ylabel("closed-loop MSE pro Schritt (log, mean +/- std)")
    ax.set_title(f"Fehlerwachstum im closed-loop Rollout ({cell})")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"Wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/lotka_volterra")
    ap.add_argument("--outdir", default=None,
                    help="defaults to --indir")
    ap.add_argument("--cell", default="cfc_lrc")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-examples", type=int, default=4)
    args = ap.parse_args()

    indir = Path(args.indir)
    outdir = Path(args.outdir) if args.outdir else indir
    outdir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(indir)
    if not runs:
        raise SystemExit(f"no result JSONs found in {indir}")
    print(f"Loaded {len(runs)} runs from {indir}")
    print_summary(runs, args.cell)

    plot_phase(runs, args.cell, args.seed, outdir / "lotka_volterra_phase.png",
               n_examples=args.n_examples)
    plot_timeseries(runs, args.cell, args.seed,
                    outdir / "lotka_volterra_timeseries.png",
                    n_examples=args.n_examples)
    plot_mse(runs, args.cell, outdir / "lotka_volterra_mse.png")
    plot_horizon(runs, args.cell, outdir / "lotka_volterra_horizon.png")


if __name__ == "__main__":
    main()
