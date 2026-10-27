"""Eval-only robustness endpoint of the wiring hypothesis H-c (issue #34).

Per result JSON matched by --results the trained model is rebuilt from its
weights and config.effective (rebuild_from_result, #18) and evaluated without
training; --out gets one CSV row per fit.

Person Activity (run_person_activity_benchmark.py JSONs): pooled test
macro-F1 under Gaussian noise on the x, y, z coordinates through the
committee corruption code (noise_sigma, corrupt_views with one view and no
faults). sigma in SIGMAS is in units of the training-split std of each
coordinate; N_DRAWS draws from the fixed NOISE_SEED (draw d:
default_rng([NOISE_SEED, d]), the same noisy inputs for every arm) are
averaged. sigma 0 is the clean reference of the same weights: the runner
saves the last epoch's weights but reads test_macro_f1 at the best
validation epoch, so the two agree only when best_val_epoch == epochs.
Columns f1_sigma_<sigma>.

Rollouts (run_lotka_volterra_benchmark.py JSONs of SYSTEMS on
FORCEABLE_WIRINGS): two fresh sets of N_ROLLOUT trajectories, drawn through
sample_ood_trajectories (the code path of the runner's --ic-spread) at the
fit's dt from seed data_seed + SEED_OFFSET (shared by every arm of a
replicate, disjoint from every training draw) and normalised with the fit's
training statistics:
  * ood:  initial-condition spread OOD_SPREAD (0.8 instead of 0.4), the fit's
          seq_len, anchors, horizon and theta;
  * long: the fit's spread and anchors, LONG_HORIZON (256) free-run steps
          after each anchor (descriptive).
Each set gets the runner's rollout dict on the trained ForcedRolloutCell
wrapper at alpha = 0 (#25, as #65): rollout_metrics of the free run from y_0
(VPT, NRMSE@16/32/128, orbit error, invariant drift, outcome class rates) and
anchor_metrics (multi-anchor VPT, #62). Columns ood_<key> / long_<key>
(nested keys joined by '_').

Every row: file, task, system ('pa' on Person Activity), arm (the label of
aggregate_results.arm_label), cell, wiring, seed, config_effective_sha256 and
metric_robust, the pre-registered decision value (protocol freeze
F1-2026-10-26, decision n): f1_sigma_0.3 on Person Activity,
ood_vpt_multi_median on the rollouts. `aggregate_results.py --contrasts
--robustness CSV` runs the H-c contrasts on it as the second Bonferroni family
(m = 2 per task).

Skipped with a message: committee and other JSONs, rollout systems without an
OOD set (multitimescale_lib), wirings without a ForcedRolloutCell view.

Usage:
    uv run python experiments/eval_robustness.py \
        --results 'results/<run>/**/*.json' --out robustness.csv
"""
import argparse
import glob
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.aggregate_results import _record_task, arm_label  # noqa: E402
from experiments.run_lotka_volterra_benchmark import (  # noqa: E402
    anchored_rollouts, forced_closed_loop, load_effective_data)
from src.benchmark.checkpoints import config_hash, rebuild_from_result  # noqa: E402
from src.evaluation.metrics import confusion_matrix, macro_f1  # noqa: E402
from src.evaluation.rollout_metrics import (TAU_ORB, anchor_metrics,  # noqa: E402
                                            make_norm_stats, rollout_metrics)
from src.tasks.committee.views import corrupt_views, noise_sigma  # noqa: E402
from src.tasks.lotka_volterra.datasets import sample_ood_trajectories  # noqa: E402
from src.tasks.lotka_volterra.model import (  # noqa: E402
    FORCEABLE_WIRINGS, forced_rollout_model)
from src.tasks.person_activity.datasets import load_person_activity  # noqa: E402

SIGMAS = (0.0, 0.1, 0.3)
SIGMA_DECISION = 0.3
N_DRAWS = 5
NOISE_SEED = 34
N_ROLLOUT = 16
OOD_SPREAD = 0.8
LONG_HORIZON = 256
SEED_OFFSET = 10 ** 6
SYSTEMS = ("periodic_predator_prey", "duffing")


def noisy_macro_f1(model, data, sigma, batch_size=128):
    """Pooled test macro-F1 under coordinate noise ``sigma`` (x the
    training std), mean over N_DRAWS draws (one draw at sigma 0)."""
    scale = noise_sigma(data.train_x, sigma)
    f1 = []
    for d in range(N_DRAWS if sigma else 1):
        x = corrupt_views(data.test_x, 1, 0.0, scale,
                          np.random.default_rng([NOISE_SEED, d]))[:, :, 0]
        pred = model.predict((x, data.test_t), batch_size=batch_size,
                             verbose=0).argmax(-1)
        f1.append(macro_f1(confusion_matrix(data.test_y, pred,
                                            data.num_classes)))
    return float(np.mean(f1))


def rollout_set(model, data, trajs, anchors, horizon, theta, tau_orb):
    """The runner's rollout dict (rollout_metrics + anchor_metrics) of raw
    trajectories (n, L + 1, 2) on the alpha = 0 forced wrapper, in the fit's
    normalisation (``data``: the fit's LotkaVolterraData)."""
    x = ((trajs[:, :-1] - data.mean) / data.std).astype(np.float32)
    fresh = SimpleNamespace(test_x=x, test_t=np.full(
        (*x.shape[:2], 1), data.dt, np.float32))
    gen = forced_closed_loop(forced_rollout_model(
        model, tf.Variable(0.0, trainable=False)), fresh, "test")
    preds = anchored_rollouts(model, fresh, "test", False, anchors, horizon,
                              forced_rollout_model(model))
    norm_stats = make_norm_stats(data.train_traj, data.mean, data.std)
    return {**rollout_metrics(trajs, data.denormalise(gen), norm_stats,
                              data.system, tau_orb),
            **anchor_metrics(trajs, {a: data.denormalise(p)
                                     for a, p in preds.items()},
                             norm_stats, theta)}


def rollout_columns(path, record):
    """ood_* / long_* columns and metric_robust of a rollout fit."""
    eff = record["config"]["effective"]
    data = load_effective_data(eff)
    model = rebuild_from_result(path)
    m = record["rollout_metrics_test"]
    anchors, horizon, theta = (m["anchors"], m["anchor_horizon"],
                               m["anchor_theta"])
    t0, t1 = data.t_span

    def draw(spread, steps):    # the fit's dt over `steps` steps
        return sample_ood_trajectories(
            N_ROLLOUT, eff["data_seed"] + SEED_OFFSET, spread, steps,
            data.system, (t0, t0 + (t1 - t0) * steps / data.seq_len))

    tau_orb = eff.get("tau_orb", TAU_ORB)
    sets = {"ood": rollout_set(model, data, draw(OOD_SPREAD, data.seq_len),
                               anchors, horizon, theta, tau_orb),
            "long": rollout_set(model, data,
                                draw(data.spread, max(anchors) + LONG_HORIZON),
                                anchors, LONG_HORIZON, theta, tau_orb)}
    return {**pd.json_normalize(sets, sep="_").iloc[0].to_dict(),
            "metric_robust": sets["ood"]["vpt_multi_median"]}


def pa_columns(path, record, data_path=None):
    """f1_sigma_* columns and metric_robust of a Person Activity fit."""
    model = rebuild_from_result(path)
    data = load_person_activity(
        seq_len=record["config"]["effective"]["build"]["seq_len"],
        data_path=data_path, split=record["split"],
        split_seed=record["split_seed"])
    cols = {f"f1_sigma_{s:g}": noisy_macro_f1(model, data, s,
                                              record.get("batch_size", 128))
            for s in SIGMAS}
    return {**cols, "metric_robust": cols[f"f1_sigma_{SIGMA_DECISION:g}"]}


def evaluate(path, data_path=None):
    """The CSV row of the result JSON at ``path``, or None if unsupported."""
    with open(path) as f:
        record = json.load(f)
    task = _record_task(record) if isinstance(record, dict) else None
    if task == "pa":
        system = "pa"
    elif (task == "lv" and record["system"] in SYSTEMS
          and record["wiring"] in FORCEABLE_WIRINGS):
        system = record["system"]
    else:
        print(f"skip {path}: no robustness endpoint for this record")
        return None
    if not record.get("weights_path"):
        raise SystemExit(f"{path}: no weights_path, cannot rebuild the fit")
    cols = (pa_columns(path, record, data_path) if task == "pa"
            else rollout_columns(path, record))
    print(f"robustness {path}: metric_robust {cols['metric_robust']:.4g}")
    return {"file": path, "task": task, "system": system,
            "arm": arm_label(record), "cell": record["cell"],
            "wiring": record["wiring"], "seed": record["seed"],
            "config_effective_sha256": config_hash(
                record["config"]["effective"]), **cols}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", required=True,
                    help="glob of LV and PA result JSONs (quote it; ** "
                         "recurses), weights next to them")
    ap.add_argument("--out", default="robustness.csv",
                    help="CSV with one row per evaluated fit")
    ap.add_argument("--data-path", default=None,
                    help="Person Activity: ConfLongDemo_JSI.txt (default: "
                         "the loader's data/person path)")
    args = ap.parse_args(argv)
    paths = sorted(glob.glob(args.results, recursive=True))
    rows = [r for r in (evaluate(p, args.data_path) for p in paths) if r]
    if not rows:
        raise SystemExit(f"no supported result JSON matches {args.results!r}")
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {len(rows)} rows ({len(paths) - len(rows)} skipped) to "
          f"{args.out}")


if __name__ == "__main__":
    main()
