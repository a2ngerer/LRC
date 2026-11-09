"""Eval-only sign check of the gate time coefficient t_a (issue #98, protocol
F2 section C-1).

The closed-form gate is sigmoid(t_a * dt + t_b); no implementation constrains
the sign of t_a. t_a is the output of the dense head `time_a`
(src/neurons/cfc_cell.py, src/neurons/cfc_lrc_cell.py), so it depends on the
input. Per result JSON matched by --results (cells CELLS, runners of the
rollout tasks, Person Activity and the committee) the fit is rebuilt from its
weights and config.effective (rebuild_from_result) and run, CPU only and
without training, on the test windows of its task:

  * rollouts (lv: Lotka-Volterra, Duffing, multi-timescale): the teacher-forced
    test inputs of the runner (split_inputs on the fit's redrawn data split);
  * Person Activity (pa): the test windows of the fit's split (the fold of
    config.effective.split_fold, #134; the rebuilt recordings must equal the
    record's);
  * committee (co): the runner's test views without faults (m = 0) of the
    first test seed, with the fit's noise_sigma and view transform.

Every call of every CfC/CfC-LRC cell's gate heads is recorded, so the count
covers all units of all cells, all time steps and all test windows. The same
count on the untrained model of the same seed (the builder seeds the init
from build.seed; rebuild_from_result(untrained=True), a same-seed init draw
on this device) is the null reference, about 0.5 for a sign-symmetric init.
Rollout fits are read under teacher forcing, not in the closed loop.

--out gets one row per fit: file, task, system, arm, cell, wiring, seed,
config_effective_sha256, n_values (t_a values per model), neg_share_trained,
neg_share_init and neg_share_diff (trained - init). --summary gets one row per
task, system and arm: n_seeds, the medians of the three shares, the seed
bootstrap interval (diff_ci_lo, diff_ci_hi; bootstrap_median_ci, 10000
resamples, 95 %) of the median difference and sign_mostly_negative, the
reading of protocol F2 C-1 (median trained share > 0.5, median difference > 0
and an interval that excludes 0). Descriptive, no test.

arm: aggregate_results.arm_label on rollouts and Person Activity; on the
committee the result file stem without _seed<N> (its arms differ in vote and
view knobs the label does not carry).

Usage:
    uv run python experiments/check_gate_sign.py \
        --results 'results/<run>/**/*.json' --out gate_sign.csv \
        --summary gate_sign_summary.csv
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.aggregate_results import _record_task, arm_label  # noqa: E402
from experiments.run_committee_benchmark import (  # noqa: E402
    _average, _rng, view_transform)
from experiments.run_lotka_volterra_benchmark import (  # noqa: E402
    TBT_MODES, load_effective_data, split_inputs)
from src.benchmark.checkpoints import config_hash, rebuild_from_result  # noqa: E402
from src.evaluation.stats import bootstrap_median_ci  # noqa: E402
from src.neurons.cfc_cell import CfC_Cell  # noqa: E402
from src.neurons.cfc_lrc_cell import CfC_LRC_Cell  # noqa: E402
from src.tasks.committee.views import corrupt_views  # noqa: E402
from src.tasks.person_activity.datasets import load_person_activity  # noqa: E402

CELLS = ("cfc", "cfc_lrc", "cfc_lrc_outer")
BATCH = 256


def negative_share(model, inputs, batch_size=BATCH):
    """(share of negative t_a, number of t_a values) of ``model`` on the
    input tuple ``inputs`` over all gate-head calls of its CfC/CfC-LRC cells
    (every unit, time step and window). Runs eagerly, so the Keras RNN loop
    calls the cells once per step. Keras' backend.rnn calls the step once
    more before the loop (output_time_zero, shape inference) on step 0's
    input and state: per forward pass the first call of each cell is that
    duplicate; it is checked equal to the second and left out."""
    cells = [m for m in model.submodules
             if isinstance(m, (CfC_Cell, CfC_LRC_Cell))]
    if not cells:
        raise SystemExit("no CfC/CfC-LRC cell in the model")
    calls = {id(c): [] for c in cells}

    def record(original, sink):
        def time_constants(head):
            t_a, t_b = original(head)
            sink.append(t_a.numpy())    # fails loudly outside eager mode
            return t_a, t_b
        return time_constants

    for cell in cells:
        cell._time_constants = record(cell._time_constants, calls[id(cell)])
    neg = total = 0
    for i in range(0, len(inputs[0]), batch_size):
        model(tuple(a[i:i + batch_size] for a in inputs), training=False)
        for sink in calls.values():
            if (len(sink) != inputs[0].shape[1] + 1
                    or not np.array_equal(sink[0], sink[1])):
                raise RuntimeError("expected one gate call per time step "
                                   "plus the step-0 duplicate of backend.rnn")
            a = np.concatenate([c.ravel() for c in sink[1:]])
            neg, total = neg + int((a < 0).sum()), total + a.size
            sink.clear()
    return neg / total, total


def test_inputs(record, task, data_path=None):
    """The model inputs of the fit's test windows (module docstring)."""
    eff = record["config"]["effective"]
    if task == "lv":
        return split_inputs(load_effective_data(eff), "test",
                            record["wiring"] in TBT_MODES)
    d = load_person_activity(seq_len=eff["build"]["seq_len"],
                             data_path=data_path, split=eff["split"],
                             split_seed=eff["split_seed"],
                             split_fold=eff.get("split_fold"))
    got, want = (None if r is None else {k: sorted(v) for k, v in r.items()}
                 for r in (d.recordings, record.get("recordings")))
    if got != want:                     # e.g. a split key not passed through
        raise SystemExit(f"rebuilt recordings {got} != the record's {want}")
    if task == "pa":
        return d.test_x, d.test_t
    v, seed = eff["views"], eff["test_seeds"][0]
    rng = _rng(seed, 0)
    transform = view_transform(v["mode"], d.feature_size, v["n_columns"],
                               v["overlap"], v["noise"], v["seed"])
    views = corrupt_views(d.test_x, v["n_columns"], 0,
                          np.asarray(record["noise_sigma"], np.float32), rng)
    return _average(transform(views, rng), v["average"]), d.test_t


def evaluate(path, data_path=None):
    """The --out row of the result JSON at ``path``, or None if skipped."""
    with open(path) as f:
        record = json.load(f)
    task = _record_task(record) if isinstance(record, dict) else None
    if task is None or record.get("cell") not in CELLS:
        print(f"skip {path}: not a {'/'.join(CELLS)} result record")
        return None
    if not record.get("weights_path"):
        raise SystemExit(f"{path}: no weights_path, cannot rebuild the fit")
    inputs = test_inputs(record, task, data_path)
    trained, n = negative_share(rebuild_from_result(path), inputs)
    init, n_init = negative_share(rebuild_from_result(path, untrained=True),
                                  inputs)
    assert n == n_init, (n, n_init)
    arm = (re.sub(r"_seed\d+", "", os.path.basename(path)[:-len(".json")])
           if task == "co" else arm_label(record))
    print(f"t_a {path}: negative share trained {trained:.4f}, "
          f"init {init:.4f} ({n} values)")
    return {"file": path, "task": task,
            "system": record["system"] if task == "lv" else task,
            "arm": arm, "cell": record["cell"], "wiring": record["wiring"],
            "seed": record["seed"],
            "config_effective_sha256": config_hash(
                record["config"]["effective"]),
            "n_values": n, "neg_share_trained": trained,
            "neg_share_init": init, "neg_share_diff": trained - init}


def summarise(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per task, system and arm (module docstring)."""
    out = []
    for (task, system, arm), g in rows.groupby(["task", "system", "arm"],
                                               sort=True):
        diff = g["neg_share_diff"].to_numpy(float)
        lo, hi = bootstrap_median_ci(diff)
        med = {c: float(np.median(g[c])) for c in
               ("neg_share_trained", "neg_share_init", "neg_share_diff")}
        out.append({"task": task, "system": system, "arm": arm,
                    "n_seeds": len(g),
                    **{f"median_{c}": v for c, v in med.items()},
                    "diff_ci_lo": lo, "diff_ci_hi": hi,
                    "sign_mostly_negative": bool(
                        med["neg_share_trained"] > 0.5
                        and med["neg_share_diff"] > 0 and lo > 0)})
    return pd.DataFrame(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", required=True,
                    help="glob of result JSONs (quote it; ** recurses), "
                         "weights next to them")
    ap.add_argument("--out", default="gate_sign.csv",
                    help="CSV with one row per fit")
    ap.add_argument("--summary", default="gate_sign_summary.csv",
                    help="CSV with one row per task, system and arm")
    ap.add_argument("--data-path", default=None,
                    help="Person Activity and committee: ConfLongDemo_JSI.txt "
                         "(default: the loader's data/person path)")
    args = ap.parse_args(argv)
    try:                                            # CPU only
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError:            # TF already initialised (e.g. in tests)
        if tf.config.list_logical_devices("GPU"):
            raise SystemExit("GPU already initialised; run on the CPU")
    paths = sorted(glob.glob(args.results, recursive=True))
    rows = [r for r in (evaluate(p, args.data_path) for p in paths) if r]
    if not rows:
        raise SystemExit(f"no {'/'.join(CELLS)} result JSON matches "
                         f"{args.results!r}")
    rows = pd.DataFrame(rows)
    rows.to_csv(args.out, index=False)
    summarise(rows).to_csv(args.summary, index=False)
    print(f"wrote {len(rows)} fits ({len(paths) - len(rows)} skipped) to "
          f"{args.out}, the per-arm summary to {args.summary}")


if __name__ == "__main__":
    main()
