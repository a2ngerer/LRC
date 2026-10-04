"""Plot the partial-view voting-committee benchmark (Iteration 7).

Left panel: sensor-dropout degradation curves -- test accuracy vs the fraction of
channels dropped, one line per (wiring, K), mean +/- 95% CI over model seeds. The
committee's hypothesised win is a flatter curve (graceful degradation) than the
K=1 monolith, at matched parameters.

Right panel: the accuracy/robustness trade-off vs committee size K -- clean
accuracy (solid) and mean accuracy under dropout (dashed) per wiring, so the
frontier (a bit of clean accuracy traded for robustness) is visible.

Usage:
    uv run python experiments/plot_committee.py --indir results/committee
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

WIRING_COLORS = {"cncp": "#d1495b", "dense": "#4C72B0"}
WIRING_LABELS = {"cncp": "cNCP committee", "dense": "dense committee"}
K_STYLE = {1: ":", 2: "--", 4: "-"}          # K=1 monolith dotted, committee solid


def _ci95(vals):
    v = np.asarray(vals, dtype=np.float64)
    n = len(v)
    m = float(v.mean())
    if n < 2:
        return m, 0.0
    return m, 1.96 * float(v.std(ddof=1)) / np.sqrt(n)


def load(indir):
    """(wiring, K) -> {'fracs':[...], 'acc_by_frac':[[per-seed],...], 'clean':[...]}"""
    groups = defaultdict(lambda: {"fracs": None, "acc": defaultdict(list),
                                  "clean": [], "params": []})
    for f in sorted(glob.glob(os.path.join(indir, "*.json"))):
        d = json.load(open(f))
        key = (d["wiring"], int(d["n_columns"]))
        g = groups[key]
        g["fracs"] = d["drop_fracs"]
        for fr, a in zip(d["drop_fracs"], d["drop_accs"]):
            g["acc"][fr].append(a)
        g["clean"].append(d["clean_accuracy"])
        g["params"].append(d.get("params_effective", d.get("params", 0)))
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/committee")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    groups = load(args.indir)
    if not groups:
        print(f"no result JSONs in {args.indir}")
        return

    fig, (axd, axf) = plt.subplots(1, 2, figsize=(13, 5))

    # --- left: dropout degradation curves ---
    for (wiring, K) in sorted(groups):
        g = groups[(wiring, K)]
        fracs = g["fracs"]
        means, his = [], []
        for fr in fracs:
            m, h = _ci95(g["acc"][fr])
            means.append(m); his.append(h)
        means, his = np.array(means), np.array(his)
        axd.plot(fracs, means, K_STYLE.get(K, "-"),
                 color=WIRING_COLORS.get(wiring, "gray"),
                 label=f"{WIRING_LABELS.get(wiring, wiring)} K={K}", lw=2)
        axd.fill_between(fracs, means - his, means + his, alpha=0.15,
                         color=WIRING_COLORS.get(wiring, "gray"))
    axd.set_xlabel("fraction of channels dropped (sensor failure)")
    axd.set_ylabel("test accuracy (per step)")
    axd.set_title("Graceful degradation under sensor dropout")
    axd.grid(alpha=0.3)
    axd.legend(fontsize=8)

    # --- right: accuracy/robustness vs K ---
    for wiring in sorted({w for (w, _) in groups}):
        Ks = sorted(K for (w, K) in groups if w == wiring)
        clean = [np.mean(groups[(wiring, K)]["clean"]) for K in Ks]
        # mean accuracy across the dropped fracs (>0) = robustness summary
        robust = []
        for K in Ks:
            g = groups[(wiring, K)]
            drop_fr = [fr for fr in g["fracs"] if fr > 0.0]
            robust.append(np.mean([np.mean(g["acc"][fr]) for fr in drop_fr]))
        c = WIRING_COLORS.get(wiring, "gray")
        axf.plot(Ks, clean, "-o", color=c,
                 label=f"{WIRING_LABELS.get(wiring, wiring)} clean")
        axf.plot(Ks, robust, "--s", color=c, alpha=0.7,
                 label=f"{WIRING_LABELS.get(wiring, wiring)} under dropout")
    axf.set_xlabel("committee size K")
    axf.set_ylabel("test accuracy")
    axf.set_title("Clean vs dropout accuracy across K (matched params)")
    axf.grid(alpha=0.3)
    axf.legend(fontsize=8)

    fig.suptitle("Partial-view voting committee on person-activity "
                 "(Iteration 7)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = args.out or os.path.join(args.indir, "committee.png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
