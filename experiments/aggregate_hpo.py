"""Aggregate the per-(cell, wiring) Optuna HPO result JSONs into one table.

Reads every ``*.json`` written by run_lotka_volterra_hpo.py in ``--indir`` and
emits a Markdown table (best validation loss + held-out test metrics + the tuned
hyper-parameters + the parameter-matched size), sorted by test closed-loop MSE.
The table is printed and written to ``<indir>/summary.md`` so it can be pasted
straight into the meeting notes for Farsang.

Distinct from experiments/aggregate_results.py, which consumes the
run_benchmark.py schema (nrmse, Wilcoxon/Cohen's d); this one consumes the HPO
schema (best_params, trials, val/test MSE).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from statistics import mean, pstdev


def _per_study_table(rows):
    header = ("| cell | wiring | params | size | best val | test TF-MSE | "
              "test CL-MSE | persistence | best lr | best bs | seed | "
              "trials (done/pruned) |")
    sep = "|" + "|".join(["---"] * 12) + "|"
    lines = [header, sep]
    for r in sorted(rows, key=lambda r: r.get("test_closed_loop_mse", float("inf"))):
        bp = r.get("best_params", {})
        lines.append(
            f"| {r['cell']} | {r['wiring']} | {r.get('params_effective', r.get('matched_params','?'))} "
            f"| {r.get('matched_size','?')} | {r.get('best_val_loss',float('nan')):.5f} "
            f"| {r.get('test_teacher_forced_mse',float('nan')):.5f} "
            f"| {r.get('test_closed_loop_mse',float('nan')):.5f} "
            f"| {r.get('persistence_mse',float('nan')):.5f} "
            f"| {bp.get('lr',float('nan')):.2e} | {bp.get('batch_size','?')} "
            f"| {r.get('seed','?')} "
            f"| {r.get('n_complete','?')}/{r.get('n_pruned','?')} |")
    return lines


def _grouped_table(rows):
    """Mean +/- std of the test metrics per (cell, wiring), across seeds."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["cell"], r["wiring"])].append(r)

    def ms(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return "n/a"
        return f"{mean(vals):.5f} +/- {pstdev(vals):.5f}"

    header = ("| cell | wiring | params | test TF-MSE (mean+/-std) | "
              "test CL-MSE (mean+/-std) | n seeds |")
    sep = "|" + "|".join(["---"] * 6) + "|"
    lines = [header, sep]
    ordered = sorted(groups.items(),
                     key=lambda kv: mean([r.get("test_closed_loop_mse", float("inf"))
                                          for r in kv[1]]))
    for (cell, wiring), rs in ordered:
        params = rs[0].get("params_effective", rs[0].get("matched_params", "?"))
        tf_ms = ms([r.get("test_teacher_forced_mse") for r in rs])
        cl_ms = ms([r.get("test_closed_loop_mse") for r in rs])
        lines.append(f"| {cell} | {wiring} | {params} | {tf_ms} | {cl_ms} | {len(rs)} |")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/lotka_volterra_hpo")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.indir, "*.json")))
    rows = []
    for path in files:
        with open(path) as f:
            rows.append(json.load(f))

    if not rows:
        print(f"no result JSONs under {args.indir}")
        return

    lines = [f"# Lotka-Volterra HPO summary ({rows[0].get('system','?')})", "",
             f"Parameter budget: {rows[0].get('param_budget','?')} "
             f"(all wirings matched within tolerance). {len(rows)} studies, "
             f"{rows[0].get('n_trials','?')} trials each, tuned via Optuna "
             f"(TPE + MedianPruner), every trial tracked in wandb.", "",
             "## Per (cell, wiring), mean +/- std over seeds", ""]
    lines += _grouped_table(rows)
    lines += ["", "## Per study (one Optuna study per cell x wiring x seed)", ""]
    lines += _per_study_table(rows)

    table = "\n".join(lines)
    print(table)
    out = os.path.join(args.indir, "summary.md")
    with open(out, "w") as f:
        f.write(table + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
