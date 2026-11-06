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
averaged. sigma 0 is the clean reference: the runner saves the weights of
the best validation epoch, at which it reads test_macro_f1, so f1_sigma_0
reproduces it. Columns f1_sigma_<sigma>.

--per-recording-confusions (#96, default off, existing columns unchanged)
adds the columns confusions_sigma_<sigma>: a JSON object {recording id:
[N_DRAWS confusion matrices of that recording's windows, rows = true]} per
sigma (one matrix at sigma 0), the input of the crossed seeds x recordings
bootstrap of the robustness family in aggregate_results.py. The scalar
f1_sigma_* come from the same predictions through the same code path and
equal the macro-F1 of the summed per-recording matrices (checked).
--verify-against CSV [CSV ...] compares f1_sigma_* and metric_robust of every
PA row bit-identically with the stored rows of earlier robustness CSVs
(matched on system, arm, seed; a row without a stored match or with any
difference raises SystemExit). Person Activity ignores --seed-streams, so a
rerun with either value reproduces the stored scalars.

Rollouts (run_lotka_volterra_benchmark.py JSONs of SYSTEMS on
FORCEABLE_WIRINGS): two fresh sets of N_ROLLOUT trajectories, drawn through
sample_ood_trajectories (the code path of the runner's --ic-spread) at the
fit's dt and normalised with the fit's training statistics. Each set has its
own seed (rollout_seeds), shared by every arm of a replicate and disjoint
from every training draw. --seed-streams picks the version (#101): v1
(default, the streams of the evaluated F1/F2 records) seeds ood with
data_seed + SEED_OFFSET and long with data_seed + LONG_SEED_OFFSET, so the
long seed of data_seed d is the ood seed of d + 1, i.e. of the next
replicate; v2 keeps the ood seed and moves long to data_seed +
LONG_SEED_OFFSET_V2, disjoint from every ood seed.
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
aggregate_results.arm_label), cell, wiring, seed, config_effective_sha256,
metric_robust, the pre-registered decision value (protocol freeze
F1-2026-10-26, decision n): f1_sigma_0.3 on Person Activity,
ood_vpt_multi_median on the rollouts; diverged_robust, the divergence flag of
that evaluation (rollouts: every OOD rollout unbounded or a non-finite one,
as load_sequence_runs on the clean test set; Person Activity: non-finite
f1_sigma_0.3); rebuild_rel_dev, the relative deviation of the rebuilt fit's
clean primary metric (f1_sigma_0, the test set's vpt_multi_median) from the
stored one, printed as a warning above REBUILD_TOL (decision s; e.g. a Person
Activity JSON written before the runner saved the best-val epoch's weights).
`aggregate_results.py --contrasts --robustness CSV` runs the H-c contrasts on
metric_robust and diverged_robust as the second Bonferroni family (m = 2 per
task).

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
LONG_SEED_OFFSET = SEED_OFFSET + 1
# ponytail: v2 is disjoint while data seeds span < SEED_OFFSET (in use:
# offsets 0/1000 plus seeds < 1000); widen the offsets if that ever changes
LONG_SEED_OFFSET_V2 = 2 * SEED_OFFSET
SEED_STREAMS = ("v1", "v2")
SYSTEMS = ("periodic_predator_prey", "duffing")
REBUILD_TOL = 1e-3


def noisy_macro_f1(model, data, sigma, batch_size=128, by_recording=None):
    """Pooled test macro-F1 under coordinate noise ``sigma`` (x the
    training std), mean over N_DRAWS draws (one draw at sigma 0). A dict as
    ``by_recording`` is filled with {recording id: [confusion matrix per
    draw]} (lists) of the same predictions."""
    scale = noise_sigma(data.train_x, sigma)
    f1 = []
    for d in range(N_DRAWS if sigma else 1):
        x = corrupt_views(data.test_x, 1, 0.0, scale,
                          np.random.default_rng([NOISE_SEED, d]))[:, :, 0]
        pred = model.predict((x, data.test_t), batch_size=batch_size,
                             verbose=0).argmax(-1)
        f1.append(macro_f1(confusion_matrix(data.test_y, pred,
                                            data.num_classes)))
        if by_recording is not None:
            for r in np.unique(data.test_rec):
                m = data.test_rec == r
                by_recording.setdefault(str(r), []).append(confusion_matrix(
                    data.test_y[m], pred[m], data.num_classes).tolist())
    return float(np.mean(f1))


def rebuild_dev(path, recomputed, stored):
    """Relative deviation of the rebuilt fit's clean primary metric from the
    stored one; a warning above REBUILD_TOL."""
    dev = float(abs(recomputed - stored) / max(abs(stored), 1e-12))
    if not dev <= REBUILD_TOL:
        print(f"WARNING {path}: the rebuilt fit's clean metric "
              f"{recomputed:.6g} deviates from the stored {stored:.6g} by "
              f"{dev:.3g} (> {REBUILD_TOL:g}); weights of another epoch?")
    return dev


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


def rollout_seeds(data_seed, streams="v1"):
    """(ood, long) draw seeds of a fit with ``data_seed`` (#101)."""
    long = {"v1": LONG_SEED_OFFSET, "v2": LONG_SEED_OFFSET_V2}[streams]
    return data_seed + SEED_OFFSET, data_seed + long


def rollout_columns(path, record, streams="v1"):
    """ood_* / long_* columns, metric_robust, diverged_robust and
    rebuild_rel_dev of a rollout fit."""
    eff = record["config"]["effective"]
    data = load_effective_data(eff)
    model = rebuild_from_result(path)
    m = record["rollout_metrics_test"]
    anchors, horizon, theta = (m["anchors"], m["anchor_horizon"],
                               m["anchor_theta"])
    t0, t1 = data.t_span

    ood_seed, long_seed = rollout_seeds(eff["data_seed"], streams)

    def draw(seed, spread, steps):    # the fit's dt over `steps` steps
        return sample_ood_trajectories(
            N_ROLLOUT, seed, spread, steps,
            data.system, (t0, t0 + (t1 - t0) * steps / data.seq_len))

    tau_orb = eff.get("tau_orb", TAU_ORB)
    sets = {"ood": rollout_set(model, data,
                               draw(ood_seed, OOD_SPREAD, data.seq_len),
                               anchors, horizon, theta, tau_orb),
            "long": rollout_set(model, data,
                                draw(long_seed, data.spread,
                                     max(anchors) + LONG_HORIZON),
                                anchors, LONG_HORIZON, theta, tau_orb)}
    clean = rollout_set(model, data, data.test_traj, anchors, horizon, theta,
                        tau_orb)["vpt_multi_median"]
    ood = sets["ood"]
    return {**pd.json_normalize(sets, sep="_").iloc[0].to_dict(),
            "metric_robust": ood["vpt_multi_median"],
            "diverged_robust": bool(ood["rate_unbounded"] == 1
                                    or ood["non_finite"]),
            "rebuild_rel_dev": rebuild_dev(path, clean,
                                           m["vpt_multi_median"])}


def pa_columns(path, record, data_path=None, confusions=False):
    """f1_sigma_* columns, metric_robust, diverged_robust and
    rebuild_rel_dev of a Person Activity fit; confusions=True adds the
    confusions_sigma_* columns (JSON, see the module docstring)."""
    model = rebuild_from_result(path)
    data = load_person_activity(
        seq_len=record["config"]["effective"]["build"]["seq_len"],
        data_path=data_path, split=record["split"],
        split_seed=record["split_seed"])
    cols, conf = {}, {}
    for s in SIGMAS:
        by_rec = {} if confusions else None
        cols[f"f1_sigma_{s:g}"] = noisy_macro_f1(
            model, data, s, record.get("batch_size", 128), by_rec)
        if confusions:
            # the pooled matrices of the draws reproduce the scalar exactly
            # (integer counts): the crossed bootstrap starts from this value
            pooled = np.sum([c for c in by_rec.values()], axis=0)
            if float(np.mean([macro_f1(m) for m in pooled])) \
                    != cols[f"f1_sigma_{s:g}"]:
                raise SystemExit(f"{path}: the per-recording confusions do "
                                 f"not reproduce f1_sigma_{s:g}")
            conf[f"confusions_sigma_{s:g}"] = json.dumps(by_rec)
    f1 = cols[f"f1_sigma_{SIGMA_DECISION:g}"]
    return {**cols, **conf, "metric_robust": f1,
            "diverged_robust": not np.isfinite(f1),
            "rebuild_rel_dev": rebuild_dev(path, cols["f1_sigma_0"],
                                           record["test_macro_f1"])}


def evaluate(path, data_path=None, streams="v1", confusions=False):
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
    cols = (pa_columns(path, record, data_path, confusions) if task == "pa"
            else rollout_columns(path, record, streams))
    print(f"robustness {path}: metric_robust {cols['metric_robust']:.4g}")
    return {"file": path, "task": task, "system": system,
            "arm": arm_label(record), "cell": record["cell"],
            "wiring": record["wiring"], "seed": record["seed"],
            "config_effective_sha256": config_hash(
                record["config"]["effective"]), **cols}


VERIFIED = ("f1_sigma_0", "f1_sigma_0.1", "f1_sigma_0.3", "metric_robust")


def verify_against(rows, csv_paths):
    """SystemExit unless every Person Activity row has exactly one stored row
    (same system, arm, seed) in ``csv_paths`` whose VERIFIED columns equal
    its own bit for bit (#96)."""
    key = ["system", "arm", "seed"]
    stored = pd.concat([pd.read_csv(p, float_precision="round_trip")
                        for p in csv_paths], ignore_index=True)
    stored = stored.set_index(key)
    bad = []
    for r in rows:
        if r["task"] != "pa":
            continue
        k = (r["system"], r["arm"], r["seed"])
        match = stored.loc[[k]] if k in stored.index else stored.iloc[:0]
        if len(match) != 1:
            bad.append(f"{k}: {len(match)} stored rows")
            continue
        bad += [f"{k}: {c} {r[c]!r} != stored {match[c].iloc[0]!r}"
                for c in VERIFIED if r[c] != match[c].iloc[0]]
    if bad:
        raise SystemExit("--verify-against: " + "; ".join(bad))
    print(f"verified {sum(r['task'] == 'pa' for r in rows)} Person Activity "
          "row(s) bit-identical with the stored CSVs")


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
    ap.add_argument("--seed-streams", choices=SEED_STREAMS, default="v1",
                    help="rollout draw seeds (#101): v1 = the streams of the "
                         "evaluated records (long of replicate s = ood of "
                         "s + 1), v2 = disjoint ood and long streams; "
                         "Person Activity ignores it")
    ap.add_argument("--per-recording-confusions", action="store_true",
                    help="Person Activity: add the per-recording, per-draw "
                         "confusion matrices (columns confusions_sigma_*) "
                         "for the crossed bootstrap (#96)")
    ap.add_argument("--verify-against", nargs="+", metavar="CSV",
                    help="earlier robustness CSVs (paths or globs): every "
                         "Person Activity row must reproduce its stored "
                         "f1_sigma_* and metric_robust bit-identically")
    args = ap.parse_args(argv)
    paths = sorted(glob.glob(args.results, recursive=True))
    rows = [r for r in (evaluate(p, args.data_path, args.seed_streams,
                                 args.per_recording_confusions)
                        for p in paths) if r]
    if not rows:
        raise SystemExit(f"no supported result JSON matches {args.results!r}")
    if args.verify_against:
        stored = sorted(p for g in args.verify_against
                        for p in glob.glob(g, recursive=True))
        if not stored:
            raise SystemExit(f"no CSV matches {args.verify_against}")
        verify_against(rows, stored)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {len(rows)} rows ({len(paths) - len(rows)} skipped) to "
          f"{args.out}")


if __name__ == "__main__":
    main()
