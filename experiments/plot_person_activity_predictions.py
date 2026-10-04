"""Graphical performance view for the Person-Activity benchmark.

Person Activity is sequence CLASSIFICATION (7 activity classes, per timestep),
not trajectory regression -- so the classification analogue of an ODE
trajectory overlay is used here:

  1. predicted-vs-true activity over time, overlaid on example test sequences
     (true = solid black step line; NCP / cNCP predictions as dashed/dotted
     step lines), and
  2. per-class confusion matrices for NCP vs cNCP.

The runner (run_person_activity_benchmark.py) stores only accuracy JSONs, not
model weights, so this script trains cfc_lrc for both wirings, predicts on the
test split, and plots. Fewer epochs than the benchmark is fine for a
qualitative prediction picture.

Usage:
    uv run python experiments/plot_person_activity_predictions.py [--epochs 30] [--seed 0]
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.tasks.person_activity.datasets import load_person_activity
from src.tasks.person_activity.model import build_person_activity_model

CLASS_NAMES = ["lying", "sitting", "standing-up", "walking",
               "falling", "on-all-fours", "sitting-ground"]
WIRING_COLORS = {"ncp": "#4C72B0", "cncp": "#C44E52"}


def train_and_predict(wiring, cell, data, epochs, seed):
    model = build_person_activity_model(
        wiring=wiring, cell=cell, size=64, seed=seed,
        num_classes=data.num_classes, feature_size=data.feature_size,
        seq_len=data.seq_len, elastance_type="asymmetric")
    model.fit((data.train_x, data.train_t), data.train_y, batch_size=128,
              epochs=epochs,
              validation_data=((data.test_x, data.test_t), data.test_y),
              verbose=0)
    logits = model.predict((data.test_x, data.test_t), verbose=0)  # (N,T,C)
    pred = logits.argmax(-1).astype(np.int32)                       # (N,T)
    acc = float((pred == data.test_y).mean())
    return pred, acc


def confusion(true, pred, n):
    cm = np.zeros((n, n), dtype=float)
    for t, p in zip(true.ravel(), pred.ravel()):
        cm[int(t), int(p)] += 1
    return cm


def plot_confusion(data, results, out):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    for ax, wiring in zip(axes, ("ncp", "cncp")):
        pred, acc = results[wiring]
        cm = confusion(data.test_y, pred, data.num_classes)
        cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(data.num_classes))
        ax.set_yticks(range(data.num_classes))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(CLASS_NAMES, fontsize=8)
        ax.set_xlabel("predicted class")
        ax.set_ylabel("true class")
        ax.set_title(f"{'cNCP' if wiring == 'cncp' else 'NCP'} "
                     f"— per-step accuracy {acc:.3f}")
        for i in range(data.num_classes):
            for j in range(data.num_classes):
                if cmn[i, j] > 0.005:
                    ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                            color="white" if cmn[i, j] > 0.5 else "black",
                            fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Person Activity — confusion matrix (row-normalised), cfc_lrc: "
                 "cNCP vs NCP", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"Wrote {out}")


def plot_overlays(data, results, out, n_examples=6, seed=0):
    # Pick diverse example sequences: prefer ones with several distinct classes,
    # so the overlay actually shows transitions the model must track.
    rng = np.random.RandomState(seed)
    n_distinct = np.array([len(np.unique(y)) for y in data.test_y])
    candidates = np.where(n_distinct >= 3)[0]
    if len(candidates) < n_examples:
        candidates = np.arange(len(data.test_y))
    idx = rng.choice(candidates, n_examples, replace=False)

    fig, axes = plt.subplots(n_examples, 1, figsize=(11, 2.0 * n_examples),
                             sharex=True)
    t_axis = np.arange(data.seq_len)
    for ax, si in zip(axes, idx):
        ax.step(t_axis, data.test_y[si], where="mid", color="black", lw=2.6,
                label="true", alpha=0.85)
        ax.step(t_axis, results["ncp"][0][si], where="mid",
                color=WIRING_COLORS["ncp"], lw=1.6, ls="--", label="NCP pred",
                alpha=0.9)
        ax.step(t_axis, results["cncp"][0][si], where="mid",
                color=WIRING_COLORS["cncp"], lw=1.6, ls=":", label="cNCP pred",
                alpha=0.9)
        acc_ncp = float((results["ncp"][0][si] == data.test_y[si]).mean())
        acc_cncp = float((results["cncp"][0][si] == data.test_y[si]).mean())
        ax.set_yticks(range(data.num_classes))
        ax.set_yticklabels(CLASS_NAMES, fontsize=6)
        ax.set_ylim(-0.5, data.num_classes - 0.5)
        ax.grid(alpha=0.3)
        ax.set_ylabel(f"seq {si}\nNCP {acc_ncp:.2f} / cNCP {acc_cncp:.2f}",
                      fontsize=7)
    axes[0].legend(loc="upper right", fontsize=8, ncol=3)
    axes[-1].set_xlabel("timestep")
    fig.suptitle("Person Activity — predicted vs true activity over time "
                 "(example test sequences, cfc_lrc)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"Wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="results/person_activity_cluster")
    args = ap.parse_args()

    data = load_person_activity()
    print(f"test sequences: {len(data.test_y)}, seq_len {data.seq_len}, "
          f"classes {data.num_classes}")

    results = {}
    for wiring in ("ncp", "cncp"):
        print(f"training cfc_lrc / {wiring} for {args.epochs} epochs ...")
        pred, acc = train_and_predict(wiring, "cfc_lrc", data, args.epochs,
                                      args.seed)
        results[wiring] = (pred, acc)
        print(f"  {wiring} per-step test accuracy: {acc:.4f}")

    plot_confusion(data, results, f"{args.outdir}/predictions_confusion.png")
    plot_overlays(data, results, f"{args.outdir}/predictions_overlay.png",
                  seed=args.seed)


if __name__ == "__main__":
    main()
