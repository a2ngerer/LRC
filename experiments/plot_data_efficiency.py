"""Plot the data-efficiency benchmark (Iteration 8): learning curves.

Test accuracy vs the fraction of training data used, one line per wiring, mean
+/- 95% CI over seeds, log-scaled x. The inductive-bias claim is a STEEPER curve
for the sparse cNCP wiring -- an advantage in the low-data regime that washes out
as data grows, at matched parameters.

Usage:
    uv run python experiments/plot_data_efficiency.py --indir results/data_efficiency
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

WIRING_COLORS = {"cncp": "#d1495b", "ncp": "#edae49", "dense": "#4C72B0"}
WIRING_LABELS = {"cncp": "cNCP (sparse cortical)", "ncp": "NCP",
                 "dense": "dense"}
ORDER = ("cncp", "ncp", "dense")


def _ci95(vals):
    v = np.asarray(vals, dtype=np.float64)
    n = len(v)
    m = float(v.mean())
    if n < 2:
        return m, 0.0
    return m, 1.96 * float(v.std(ddof=1)) / np.sqrt(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/data_efficiency")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # wiring -> frac -> [test accs over seeds]
    data = defaultdict(lambda: defaultdict(list))
    params = {}
    for f in sorted(glob.glob(os.path.join(args.indir, "*.json"))):
        d = json.load(open(f))
        data[d["wiring"]][d["frac"]].append(d["test_accuracy"])
        params[d["wiring"]] = d.get("params_effective", d.get("params", 0))
    if not data:
        print(f"no result JSONs in {args.indir}")
        return

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for wiring in ORDER:
        if wiring not in data:
            continue
        fracs = sorted(data[wiring])
        means, his = [], []
        for fr in fracs:
            m, h = _ci95(data[wiring][fr])
            means.append(m); his.append(h)
        means, his = np.array(means), np.array(his)
        c = WIRING_COLORS.get(wiring, "gray")
        lbl = f"{WIRING_LABELS.get(wiring, wiring)} ({params.get(wiring, 0)/1000:.0f}k)"
        ax.plot(fracs, means, "-o", color=c, label=lbl, lw=2)
        ax.fill_between(fracs, means - his, means + his, alpha=0.15, color=c)
    ax.set_xscale("log")
    ax.set_xlabel("fraction of training data (log scale)")
    ax.set_ylabel("test accuracy (per step)")
    ax.set_title("Data efficiency on person-activity (matched params)\n"
                 "steeper curve = stronger inductive bias in the low-data regime")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    out = args.out or os.path.join(args.indir, "data_efficiency.png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
