"""Eval-only path ablation of saved cNCP fits at fixed weights (issue #136).

Per result JSON in --results (cncp records of periodic_predator_prey, duffing
and Person Activity; protocol F3 runs it on the cfc_lrc/cncp finals) the
trained model is rebuilt from its weights and config.effective
(rebuild_from_result, #18) and evaluated without training under each
ablation in ABLATIONS, switched by set_cncp_ablation (src/wirings/cncp.py):

  none     the plain trained model
  delayed  the delayed feedback edges L5ET -> L5IT and L6CC -> L2/3 cut
  apical   the gain factor 1 + g * sigmoid(apical) set to 1 at L2/3 and L5ET
  relay    the relay exit edge Thal -> L4 cut (the relay state still runs)
  all      the three together

--out gets one CSV row per (fit, ablation) with the primary endpoint in
`metric` (named in `endpoint`):
  * rollouts: test_vpt_multi_median, the clean-test multi-anchor VPT median
    on the ForcedRolloutCell wrapper (eval_robustness.rollout_set, the code
    path of its rebuild check), plus hnorm_mean / hnorm_max, the mean and max
    of ||S_t||_2 (S = the core cell's state) over the test sequences and steps
    of the alpha = 0 free run from y_0 (rq4_diagnostics.OneStep);
  * Person Activity: test_macro_f1, the pooled test macro-F1 of the saved
    best-validation weights (eval_robustness.noisy_macro_f1 at sigma 0; fold
    records load their fold).
Every row also carries metric_stored (the record's value of the endpoint),
rebuild_rel_dev (ablation none only: relative deviation of the rebuilt
metric from metric_stored, warned above eval_robustness.REBUILD_TOL),
config_effective_sha256 and the source protocol_version.

Read-only on the sources: the script writes the CSV only, refuses an --out
inside a source folder and never restamps a record.

Skipped with a message: records of other wirings or tasks (cncp_ff has none
of the three paths).

Usage:
    uv run python experiments/eval_cncp_ablation.py \
        --results 'results/<run>/**/cfc_lrc_cncp_seed*.json' --out ablation.csv
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.aggregate_results import _record_task, arm_label  # noqa: E402
from experiments.eval_robustness import (SYSTEMS, noisy_macro_f1,  # noqa: E402
                                         rebuild_dev, rollout_set)
from experiments.rq4_diagnostics import OneStep  # noqa: E402
from experiments.run_lotka_volterra_benchmark import (  # noqa: E402
    load_effective_data)
from src.benchmark.checkpoints import config_hash, rebuild_from_result  # noqa: E402
from src.evaluation.rollout_metrics import TAU_ORB  # noqa: E402
from src.tasks.person_activity.datasets import load_person_activity  # noqa: E402
from src.wirings.cncp import set_cncp_ablation  # noqa: E402

ABLATIONS = {"none": {}, "delayed": {"delayed": True},
             "apical": {"apical": True}, "relay": {"relay": True},
             "all": {"delayed": True, "apical": True, "relay": True}}
WIRINGS = ("cncp",)


def state_norms(model, data):
    """(mean, max) of ||S_t||_2 over the test sequences and steps of the
    alpha = 0 free run from y_0, S = the core cell's state."""
    one = OneStep(model)                       # alpha 0: the free run
    step = tf.function(one.step)               # traced with the current flags
    x, t = data.test_x, data.test_t
    z = tf.zeros([len(x), one.dim + one.cell.output_size])
    started = tf.zeros([len(x), 1])
    norms = []
    for k in range(x.shape[1]):
        _, z = step(tf.constant(x[:, k]), tf.constant(t[:, k]), z, started)
        started = tf.ones_like(started)
        norms.append(np.linalg.norm(z.numpy()[:, :one.dim], axis=-1))
    norms = np.stack(norms)
    return float(norms.mean()), float(norms.max())


def rollout_evaluator(record):
    """(endpoint, stored, evaluate(model) -> columns) of a rollout fit."""
    eff = record["config"]["effective"]
    data = load_effective_data(eff)
    m = record["rollout_metrics_test"]
    args = (m["anchors"], m["anchor_horizon"], m["anchor_theta"],
            eff.get("tau_orb", TAU_ORB))

    def evaluate(model):
        mean, mx = state_norms(model, data)
        return {"metric": rollout_set(model, data, data.test_traj, *args)[
                    "vpt_multi_median"],
                "hnorm_mean": mean, "hnorm_max": mx}
    return "test_vpt_multi_median", m["vpt_multi_median"], evaluate


def pa_evaluator(record, data_path=None):
    """(endpoint, stored, evaluate(model) -> columns) of a Person Activity
    fit."""
    data = load_person_activity(
        seq_len=record["config"]["effective"]["build"]["seq_len"],
        data_path=data_path, split=record["split"],
        split_seed=record["split_seed"],
        split_fold=record.get("split_fold"))
    bs = record.get("batch_size", 128)
    return "test_macro_f1", record["test_macro_f1"], lambda model: {
        "metric": noisy_macro_f1(model, data, 0.0, bs)}


def evaluate(path, data_path=None):
    """The CSV rows (one per ablation) of the result JSON at ``path``, or []
    if unsupported."""
    with open(path) as f:
        record = json.load(f)
    task = _record_task(record) if isinstance(record, dict) else None
    if not ((task == "pa" or (task == "lv" and record["system"] in SYSTEMS))
            and record["wiring"] in WIRINGS):
        print(f"skip {path}: no cNCP path ablation for this record")
        return []
    if not record.get("weights_path"):
        raise SystemExit(f"{path}: no weights_path, cannot rebuild the fit")
    endpoint, stored, run = (pa_evaluator(record, data_path) if task == "pa"
                             else rollout_evaluator(record))
    eff = record["config"]["effective"]
    base = {"file": path, "task": task,
            "system": "pa" if task == "pa" else record["system"],
            "arm": arm_label(record), "cell": record["cell"],
            "wiring": record["wiring"], "seed": record["seed"],
            "split_fold": record.get("split_fold"),
            "config_effective_sha256": config_hash(eff),
            "protocol_version": eff.get("protocol_version")}
    model = rebuild_from_result(path)
    rows = []
    for name, flags in ABLATIONS.items():
        set_cncp_ablation(model, **flags)
        cols = run(model)
        rows.append({**base, "ablation": name, "endpoint": endpoint, **cols,
                     "metric_stored": stored,
                     "rebuild_rel_dev": (rebuild_dev(path, cols["metric"],
                                                     stored)
                                         if name == "none" else np.nan)})
        print(f"ablation {path} [{name}]: {endpoint} {cols['metric']:.4g}")
    set_cncp_ablation(model)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", required=True, nargs="+",
                    help="cncp result JSONs or globs (quote them; ** "
                         "recurses), weights next to them")
    ap.add_argument("--out", default="cncp_ablation.csv",
                    help="CSV with one row per (fit, ablation); never inside "
                         "a source folder")
    ap.add_argument("--data-path", default=None,
                    help="Person Activity: ConfLongDemo_JSI.txt (default: "
                         "the loader's data/person path)")
    args = ap.parse_args(argv)
    paths = sorted({p for g in args.results
                    for p in glob.glob(g, recursive=True)})
    out_dir = os.path.dirname(os.path.abspath(args.out))
    sources = {os.path.dirname(os.path.abspath(p)) for p in paths}
    if out_dir in sources:
        raise SystemExit(f"--out {args.out} lies in a source folder; the "
                         "ablation never writes next to the records")
    rows = [r for p in paths for r in evaluate(p, args.data_path)]
    if not rows:
        raise SystemExit(f"no supported result JSON matches {args.results!r}")
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {len(rows)} rows ({len(rows) // len(ABLATIONS)} fits) to "
          f"{args.out}")


if __name__ == "__main__":
    main()
