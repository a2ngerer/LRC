"""S0 smoke and S1 pilot gates of the frozen suite (#72, #75-#77; protocol
freeze F1-2026-10-26, validation ladder S0 and S1).

--stage s0 reads every fit JSON below the given run folders
(<run>/results/hpo/<arm>/lr*_clip*_seed*/*.json, run_grid_hpo.py under
cluster/submit_suite.sh) and checks the S0 criteria:

  arms              every arm-task of the smoke YAMLs (--configs, default the
                    93 of configs/suite/smoke_{lv,duffing,mt,pa}.yaml, expanded
                    by suite_manifest) has exactly one fit; fits of other arms
                    (e.g. the consensus smoke) are listed and checked, not
                    failed
  config            config.effective carries ode_unfolds, dt_scale, forcing
                    and anchors (rollout tasks), and the fit a weights_path
                    (top level, where the runners write it); the committee
                    runner has no dt_scale (closed-form cells)
  protocol_version  set on every fit, one value across all fits
  metrics           rollouts: rollout_metrics_{val,test} with
                    vpt_multi_median, vpt_anchor_{8,16,24,32},
                    nrmse_32_anchor_8, the outcome rates and non_finite false;
                    Person Activity and committee: the validation and test
                    primary metric finite
  nan               no NaN or Inf anywhere in a fit JSON; the inv_drift_*
                    fields of multitimescale_lib are exempt (listed), and a
                    NaN rollout_metrics_{val,test}.inv_drift_* of a block
                    with n_inv_drift_undefined > 0 is listed, not failed (the
                    summary drops undefined trajectories, so NaN means all
                    were undefined; F1 decision i: descriptive metric)
  budget            budget_above_tolerance true only on gru/dense3 and
                    lstm/dense3
  elastance         elastance_type 'asymmetric' on every cell whose registry
                    kwargs carry an elastance (src/benchmark/registry.py:
                    lrc, mm_lrc, cfc_lrc, cfc_lrc_outer, cfc_mm_lrc; mm_ltc
                    and cfc_mm_ltc have none and record None)
  ode_unfolds       6 on every solver cell (ODE_UNFOLD_CELLS: ctrnn, lrc,
                    mm_lrc, ltc, mm_ltc)
  determinism       --determinism (determinism_probe.json, #70; missing or
                    unreadable: FAIL "not measured"): |Delta| of the two
                    "on" fits <= 1e-3 on lv and pa; a runtime ratio on/off
                    above 1.3 is reported as DECISION "op-determinism off
                    for the suite" (F1 decision l), not as a failure
  floors            with --floors (aggregate_results.py --floors, one or more
                    floors.json): a floor line with a finite val and test
                    floor per rollout system and data seed of the fits

--stage s1 reads every fit JSON below the pilot run folders (finals in
results/, the consensus pilot under results/hpo/<arm>/lr*_clip*_seed100/; a
rollout fit with config.effective.dt_scale 1 is the dt_scale control) and
evaluates the S1 go criteria with the frozen thresholds:

  fits   every fit of the six pilot YAMLs present once (--configs, default
         configs/suite/pilot_{lv,duffing,mt,pa,lv_dt1,co}.yaml, 61 fits)
  config config.effective.deterministic true on every fit: the suite runs
         with op-determinism on (F1 decision l after the S0 probe, #76)
  floors a floor line per rollout system and data seed (--floors)
  G1     LV, Duffing: test vpt_multi_median of cfc_lrc/ncp above
         max(persistence, AR(1)) of its data seed + 2 steps in >= 2 of 3 seeds;
         below it LV fails (stop before S2) and Duffing is the F1 section 7
         DECISION (floor-bound, stays in the plan) unless a seed is missing,
         non-finite or without a floor line (FAIL); multitimescale_lib:
         cfc/ncp >= 2 x persistence in >= 2 of 3 seeds and rate_unbounded 0
         on every cfc/ncp fit (test split)
  G0-E   per task, an (arm, seed) qualifies when the validation metric's mean
         over the last tenth of the epochs beats its value at two thirds of the
         epochs by more than 5 % relative (rollouts val_loss_curve, lower is
         better; Person Activity val_macro_f1_curve, higher is better);
         DECISION epochs x1.5 (rollouts) or x2 (Person Activity) when
         ceil(2n/3) of the task's n pilot arms have >= 2 qualifying seeds
         (F1 decision m: LV, Duffing 4 of 5, MT, PA 2 of 2)
  G2     Person Activity: median test_macro_f1 of cfc_lrc/ncp >= 0.60, every
         lrc/ncp fit finite (training_diverged false, finite macro-F1);
         wallclock_s of both arms listed. Reported, not gated (#80): each
         fit's test macro-F1 against the one-class value of the test split
         (from its test_confusion_by_recording), and an arm at chance (at
         most 0.01 above that value in >= 2 of 3 seeds) is listed as
         "at chance" in the summary
  G3     descriptive: median VPT of the dt_scale=1 control next to the main arm
  GK     consensus pilot, per arm (ncp_cons, ncp_lam0, avg_in, aug_mono) the
         fit with the best val_macro_f1_mean (the HPO key; grid order on
         ties): validation macro-F1 at m=0 (best epoch) >= 0.60; test
         macro-F1 degrades over the fault grid: no step m -> m' rises by
         more than one paired bootstrap SE over the test recordings
         (test_confusion_by_recording_by_fault, BOOT_N resamples) and
         MF1(m_max) < MF1(0) (without the confusions: strict, non-increasing);
         ncp_cons vote_lambda_final in (0.05, 0.95). The committee JSON
         stores no per-batch fault-count sequence or hash, so the fault
         schedule is not compared (said so in the report)

A FAIL row carries its consequence, which names its source: F1 section 7,
the readings pre-registered in F1 section 10.5, or an operational rule.

The report (--out, Markdown, also printed) lists every check as PASS, FAIL or
DECISION with the offending arms; the exit code is 0 only when no check
fails.

Usage:
    uv run python experiments/suite_gate.py --stage s0 \\
        --runs results/00_canary/<date>_s0-smoke-{lv,duffing,mt,pa,co} \\
        --determinism <probe run>/results/determinism_probe.json \\
        --floors <agg>/{lv,duffing,mt}/floors.json --out S0_GATE.md
    uv run python experiments/suite_gate.py --stage s1 \\
        --runs <the six pilot run folders> \\
        --floors <agg>/{lv,duffing,mt}/floors.json --out S1_GATE.md
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections import Counter
from functools import lru_cache
from glob import glob

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.suite_manifest import (LV_DEFAULT_SYSTEM, grid,  # noqa: E402
                                        jobs, load)
from src.benchmark.registry import cell_kwargs  # noqa: E402
from src.evaluation.metrics import (CHANCE_MARGIN, at_chance,  # noqa: E402
                                    macro_f1, one_class_macro_f1)
from src.evaluation.rollout_metrics import ANCHORS  # noqa: E402

SMOKE = [os.path.join(ROOT, "configs", "suite", f"smoke_{t}.yaml")
         for t in ("lv", "duffing", "mt", "pa")]
TASKS = {"lotka_volterra": "lv", "person_activity": "pa", "committee": "co"}
CONFIG_KEYS = {"lv": ("ode_unfolds", "dt_scale", "forcing", "anchors",
                      "weights_path"),
               "pa": ("ode_unfolds", "dt_scale", "weights_path"),
               "co": ("ode_unfolds", "weights_path")}
TOP_LEVEL = "weights_path"         # written by the runners next to config
ROLLOUT_KEYS = ("vpt_multi_median", *(f"vpt_anchor_{a}" for a in ANCHORS),
                "nrmse_32_anchor_8", "rate_ok", "rate_bounded_wrong",
                "rate_unbounded")
PRIMARY = {"pa": ("val_macro_f1", "test_macro_f1"),
           "co": ("val_macro_f1_mean", "test_macro_f1_mean_grid")}
OVER_BUDGET = {("gru", "dense3"), ("lstm", "dense3")}   # known, F1 S0
ODE_UNFOLDS = 6                                          # F1 decision a
MAX_RATIO, MAX_DELTA = 1.3, 1e-3                         # F1 decision l
EXEMPT_SYSTEM, EXEMPT_PREFIX = "multitimescale_lib", "inv_drift_"
DRIFT_BLOCKS = ("rollout_metrics_val", "rollout_metrics_test")
UNDEFINED = "n_inv_drift_undefined"


def _eff(rec):
    return (rec.get("config") or {}).get("effective") or {}


def _field(rec, key):
    """(present, value) of key in config.effective; only weights_path falls
    back to the top level of the record."""
    for d in (_eff(rec), rec) if key == TOP_LEVEL else (_eff(rec),):
        if key in d:
            return True, d[key]
    return False, None


def _finite(v):
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(v))


@lru_cache(maxsize=None)
def _solver(cell):
    """A numerical ODE cell (Euler sub-steps), as the runners decide it."""
    from src.tasks.person_activity.model import effective_ode_unfolds
    try:
        return effective_ode_unfolds(cell, {}) is not None
    except (KeyError, TypeError):           # no or an unknown cell key
        return False


def _task_key(task, system=None):
    return (system or LV_DEFAULT_SYSTEM) if task == "lv" else task


def _label(arm):
    return "/".join(arm)


def expected_arms(configs):
    """{(task key, arm name): (cell, wiring)} of the campaign YAMLs."""
    out = {}
    for path in configs:
        cfg = load(path)
        for (name, cell, wiring, _), _, _ in jobs(cfg):
            out[(_task_key(cfg["task"], cfg.get("system")), name)] = (
                cell, wiring)
    return out


def read_fits(runs):
    """One dict per fit JSON below the run folders: arm (task key, folder
    name under hpo/), task (lv | pa | co | None), path, rec."""
    fits = []
    for run in runs:
        for path in sorted(glob(os.path.join(run, "**", "hpo", "*",
                                             "lr*_clip*_seed*", "*.json"),
                                recursive=True)):
            with open(path, encoding="utf-8") as f:
                rec = json.load(f)
            task = TASKS.get(_eff(rec).get("task"))
            key = _task_key(task, rec.get("system")) if task else "unknown"
            fits.append({"arm": (key, path.split(os.sep)[-3]), "task": task,
                         "path": path, "rec": rec})
    return fits


def _non_finite(x, path=""):
    """Sorted dotted paths ([] for list items) of the NaN/Inf floats."""
    if isinstance(x, float):
        return [] if math.isfinite(x) else [path]
    if isinstance(x, dict):
        items = ((f"{path}.{k}" if path else str(k), v) for k, v in x.items())
    elif isinstance(x, list):
        items = ((f"{path}[]", v) for v in x)
    else:
        return []
    return sorted({p for sub, v in items for p in _non_finite(v, sub)})


def _undefined_drift(rec, path):
    """path is a NaN rollout_metrics_{val,test}.inv_drift_* summary of a
    block with n_inv_drift_undefined > 0 (the summary drops undefined
    trajectories, so NaN means every one was undefined)."""
    block, _, key = path.partition(".")
    m = rec.get(block) if block in DRIFT_BLOCKS else None
    return (isinstance(m, dict) and key.startswith(EXEMPT_PREFIX)
            and isinstance(m.get(key), float) and math.isnan(m[key])
            and _finite(m.get(UNDEFINED)) and m[UNDEFINED] > 0)


def check_fit(fit):
    """{check: [problems]} of one fit, its exempt NaN paths and its listed
    undefined inv_drift_* blocks ("<block> (n_inv_drift_undefined n):
    <keys>")."""
    rec, task = fit["rec"], fit["task"]
    eff = _eff(rec)
    cell, wiring = rec.get("cell"), rec.get("wiring")
    out = {k: [] for k in ("config", "protocol_version", "metrics", "nan",
                           "budget", "elastance", "ode_unfolds")}
    if task is None:
        out["config"].append("config.effective.task missing or unknown")
    for k in CONFIG_KEYS.get(task, ()):
        present, v = _field(rec, k)
        if not present or (v is None and k != "ode_unfolds"):
            out["config"].append(f"{k} missing")
    if _field(rec, "protocol_version")[1] is None:
        out["protocol_version"].append("protocol_version missing")
    if task == "lv":
        for split in ("val", "test"):
            m = rec.get(f"rollout_metrics_{split}")
            if not isinstance(m, dict):
                out["metrics"].append(f"rollout_metrics_{split} missing")
                continue
            miss = [k for k in ROLLOUT_KEYS if k not in m]
            if miss:
                out["metrics"].append(f"rollout_metrics_{split} lacks "
                                      f"{', '.join(miss)}")
            if m.get("non_finite") is not False:
                out["metrics"].append(f"rollout_metrics_{split}.non_finite "
                                      f"{m.get('non_finite')!r}")
    for k in PRIMARY.get(task, ()):
        if not _finite(rec.get(k)):
            out["metrics"].append(f"{k} {rec.get(k)!r}")
    exempt, undefined = [], {}
    for p in _non_finite(rec):
        if (rec.get("system") == EXEMPT_SYSTEM
                and p.rsplit(".", 1)[-1].startswith(EXEMPT_PREFIX)):
            exempt.append(p)
        elif _undefined_drift(rec, p):
            block, _, key = p.partition(".")
            undefined.setdefault(block, []).append(key)
        else:
            out["nan"].append(p)
    listed = [f"{b} ({UNDEFINED} {rec[b][UNDEFINED]}): {', '.join(ks)}"
              for b, ks in undefined.items()]
    over = rec.get("budget_above_tolerance")
    if over is None:
        out["budget"].append("budget_above_tolerance missing")
    elif over and (cell, wiring) not in OVER_BUDGET:
        out["budget"].append(f"above tolerance (budget_deviation "
                             f"{rec.get('budget_deviation')})")
    if ("elastance_type" in cell_kwargs(cell)
            and eff.get("elastance_type") != "asymmetric"):
        out["elastance"].append(
            f"elastance_type {eff.get('elastance_type')!r}")
    if _solver(cell) and eff.get("ode_unfolds") != ODE_UNFOLDS:
        out["ode_unfolds"].append(f"ode_unfolds {eff.get('ode_unfolds')!r}")
    return out, exempt, listed


DESCRIPTION = {
    "config": "config.effective: ode_unfolds, dt_scale, forcing, anchors "
              "(rollouts); weights_path",
    "protocol_version": "protocol_version on every fit, one value",
    "metrics": "rollout keys, outcome rates, non_finite false; PA/committee "
               "primary metric finite",
    "nan": "no NaN/Inf (multitimescale_lib inv_drift_* exempt, all-undefined "
           "inv_drift_* listed)",
    "budget": "budget_above_tolerance only on gru/dense3, lstm/dense3",
    "elastance": "elastance_type 'asymmetric' on the elastance cells",
    "ode_unfolds": f"ode_unfolds {ODE_UNFOLDS} on the solver cells",
}


def gate(fits, expected, determinism=None, floors=None):
    """[(check, description, [offending entries], decision or None)], {arm
    label: exempt paths}, {arm label: listed undefined inv_drift_* blocks},
    [arm labels outside the expected list]. determinism None: not
    measured (FAIL)."""
    n = Counter(f["arm"] for f in fits)
    missing = [a for a in expected if not n[a]]
    results = [("arms", f"{len(expected) - len(missing)}/{len(expected)} "
                f"arm-tasks present, one fit each",
                [f"{_label(a)}: no fit" for a in missing]
                + [f"{_label(a)}: {k} fits" for a, k in sorted(n.items())
                   if k > 1], None)]
    per_check, exempt, listed = {k: [] for k in DESCRIPTION}, {}, {}
    for fit in fits:
        problems, ex, li = check_fit(fit)
        for k, msgs in problems.items():
            per_check[k] += [f"{_label(fit['arm'])}: {m}" for m in msgs]
        if ex:
            exempt[_label(fit["arm"])] = ex
        if li:
            listed[_label(fit["arm"])] = li
    versions = sorted({str(v) for v in (
        _field(f["rec"], "protocol_version")[1] for f in fits)
        if v is not None})
    if len(versions) > 1:
        per_check["protocol_version"].append(
            f"several values: {', '.join(versions)}")
    for k, desc in DESCRIPTION.items():
        if k == "protocol_version" and len(versions) == 1:
            desc += f" ({versions[0]})"
        results.append((k, desc, per_check[k], None))
    results.append(check_determinism(determinism))
    if floors is not None:
        results.append(check_floors(fits, floors))
    unexpected = sorted(_label(a) for a in n if a not in expected)
    return results, exempt, listed, unexpected


def check_determinism(probe):
    """Runtime ratio and on-pair |Delta| of determinism_probe.json (#70);
    probe None or not a JSON object: not measured. A ratio above MAX_RATIO
    is F1 decision l (op-determinism off for the suite), not a failure."""
    desc = (f"|Delta on| <= {MAX_DELTA:g} on lv and pa; ratio on/off > "
            f"{MAX_RATIO}: op-determinism off (decision)")
    if not isinstance(probe, dict):
        return ("determinism", desc, ["not measured (no readable "
                                      "--determinism probe file)"], None)
    bad, seen, slow = [], [], []
    for task in ("lv", "pa"):
        s = (probe.get("tasks") or {}).get(task)
        if s is None:
            bad.append(f"{task}: not in the probe")
            continue
        ratio, delta = s.get("ratio_on_off"), s.get("max_abs_delta_on")
        seen.append(f"{task} ratio {ratio:.3g}, |Delta| {delta:.3g}"
                    if _finite(ratio) and _finite(delta) else task)
        if not _finite(ratio):
            bad.append(f"{task}: runtime ratio on/off {ratio!r}")
        elif ratio > MAX_RATIO:
            slow.append(f"{task} {ratio:.3g}")
        if not (_finite(delta) and delta <= MAX_DELTA):
            bad.append(f"{task}: |Delta| of the on fits {delta} > "
                       f"{MAX_DELTA:g}")
    decision = (f"op-determinism off for the suite (F1 decision l): runtime "
                f"ratio on/off above {MAX_RATIO} on {', '.join(slow)}; seed "
                f"noise is reported as a nuisance component" if slow else None)
    return ("determinism", f"{desc} ({'; '.join(seen)})", bad, decision)


def check_floors(fits, floors):
    """A floor line with finite val and test floors per (system, data seed)
    of the rollout fits; floors: the loaded floors.json files."""
    lines = {(d.get("system"), d.get("data_seed")): d
             for fl in floors for d in fl.get("floors", [])}
    want = sorted({(f["rec"].get("system"), _field(f["rec"], "data_seed")[1])
                   for f in fits if f["task"] == "lv"}, key=str)
    bad = []
    for system, seed in want:
        d = lines.get((system, seed))
        if d is None:
            bad.append(f"{system} data seed {seed}: no floor line")
            continue
        for split in ("val", "test"):
            if not _finite((d.get(split) or {}).get("floor")):
                bad.append(f"{system} data seed {seed}: no finite {split} "
                           f"floor")
    return ("floors", f"floor line per system and data seed, val and test "
            f"({len(want)} draws)", bad, None)


def report(results, exempt, listed, unexpected, runs, configs, n_fits):
    ok = all(not bad for _, _, bad, _ in results)
    lines = [f"# S0 smoke gate: {'PASS' if ok else 'FAIL'}\n",
             f"Runs: {', '.join(runs)}  ",
             f"Fit JSONs: {n_fits}; expected arm-tasks from "
             f"{', '.join(os.path.basename(c) for c in configs)}\n",
             "| Check | Result | Criterion | Offending |",
             "|-------|--------|-----------|-----------|"]
    for k, desc, bad, decision in results:
        status = "FAIL" if bad else "DECISION" if decision else "PASS"
        cell = desc.replace("|", "\\|")      # |Delta| keeps the columns
        lines.append(f"| {k} | {status} | {cell} | {len(bad)} |")
    for k, _, bad, decision in results:
        if bad:
            lines += ["", f"## FAIL: {k}", ""] + [f"- {b}" for b in bad]
        if decision:
            lines += ["", f"## DECISION: {k}", "", f"- {decision}"]
    if exempt:
        lines += ["", f"## Exempt ({EXEMPT_SYSTEM} {EXEMPT_PREFIX}*, listed, "
                  "not failed)", ""]
        lines += [f"- {a}: {', '.join(p)}" for a, p in sorted(exempt.items())]
    if listed:
        lines += ["", f"## Listed ({EXEMPT_PREFIX}* undefined on every "
                  "trajectory, descriptive, not failed)", ""]
        lines += [f"- {a}: {b}" for a, bs in sorted(listed.items())
                  for b in bs]
    if unexpected:
        lines += ["", "## Fits outside the expected arm list (checked, not "
                  "counted)", ""] + [f"- {a}" for a in unexpected]
    return "\n".join(lines) + "\n", ok


# --- S1 pilot gate (#75) ----------------------------------------------------

PILOT = [os.path.join(ROOT, "configs", "suite", f"pilot_{t}.yaml")
         for t in ("lv", "duffing", "mt", "pa", "lv_dt1", "co")]
DUF, MT = "duffing", "multitimescale_lib"
CONTROL = "dt_scale 1"           # tag of the dt_scale control fits (F1 h)
FLOOR_MARGIN, MT_FACTOR, MIN_SEEDS = 2, 2, 2          # G1
TAIL, MARK, GAIN = 0.1, 2 / 3, 0.05                    # G0-E (F1 decision m)
CURVES = {"lv": ("val_loss_curve", False), "pa": ("val_macro_f1_curve", True)}
EPOCH_FACTOR = {"lv": 1.5, "pa": 2}
MIN_F1, LAMBDA = 0.60, (0.05, 0.95)                    # G2, GK
GK_ARMS = ("ncp_cons", "ncp_lam0", "avg_in", "aug_mono")
CM_KEY = "test_confusion_by_recording_by_fault"        # GK bootstrap units
BOOT_N, BOOT_SEED = 10_000, 0                          # GK (F1 section 10.5)
CONSEQUENCE = {   # on FAIL (G1 duffing: the DECISION); each names its source
    "fits": "complete the pilot runs (cluster/submit_suite.sh --resume, one "
            "run folder per pilot YAML) and rerun the gate (operational "
            "rule, #75)",
    "config": "rerun the offending fits with --deterministic (extra_args of "
              "the pilot YAML) and rerun the gate; operational rule (#76)",
    "floors": "write the floor lines (aggregate_results.py --floors over the "
              "pilot runs) and rerun the gate (operational rule, #75)",
    "G1 lv": "stop: find the cause (endpoint, floors, dt_scale) before S2; "
             "H-c and RQ1 are affected (operational rule, F1 section 10.5)",
    "G1 duffing": "Duffing stays in the plan: H-b/H-c Duffing contrasts "
                  "floor-bound, H-c conjunction over LV and PA (F1 section 7)",
    "G1 unevaluated": "a missing or non-finite fit or floor line decides no "
                      "floor: find the cause and rerun the gate (operational "
                      "rule, #77)",
    "G1 mt": "spread 0.25 or 256 steps, repeat once (F1 section 7)",
    "G0-E": "no validation curve: the epoch gate cannot be decided "
            "(operational rule, #75)",
    "G2": "stop, find the cause before S2 (F1 section 7)",
    "GK": "architecture check (width, backbone) before the consensus study "
          "(F1 section 7)",
    "GK arm": "arm check (lr grid, head) before the consensus study (F1 "
              "section 10.5)",
    "GK monotone": "protocol check (fault mask, noise scale) before the "
                   "consensus study (F1 section 10.5)",
    "GK lambda": "vote_init check of one fit (F1 section 7)",
}


def _name(key):
    return "lv" if key == LV_DEFAULT_SYSTEM else key


def _num(v, default=math.nan):
    return v if _finite(v) else default


def _median(xs):
    return statistics.median(xs) if xs else math.nan


def read_pilot_fits(runs):
    """One dict per fit JSON below the run folders (as read_fits, plus seed
    and tag): arm (task key, cell_wiring; an hpo fit: its hpo/<arm> folder),
    tag CONTROL for a rollout fit at dt_scale 1, 'lr <lr>' for an hpo fit,
    else ''. JSONs without config.effective.task (hpo_selection.json, wandb
    files) are left out."""
    fits = []
    for run in runs:
        for path in sorted(glob(os.path.join(run, "**", "*.json"),
                                recursive=True)):
            with open(path, encoding="utf-8") as f:
                rec = json.load(f)
            task = (TASKS.get(_eff(rec).get("task"))
                    if isinstance(rec, dict) else None)
            if task is None:
                continue
            parts = path.split(os.sep)
            hpo = len(parts) > 3 and parts[-4] == "hpo"
            arm = (parts[-3] if hpo
                   else f"{rec.get('cell')}_{rec.get('wiring')}")
            tag = (f"lr {rec.get('lr'):g}" if hpo else CONTROL
                   if task == "lv" and _eff(rec).get("dt_scale") == 1 else "")
            fits.append({"arm": (_task_key(task, rec.get("system")), arm),
                         "task": task, "seed": rec.get("seed"), "tag": tag,
                         "path": path, "rec": rec})
    return fits


def pilot_fits(configs):
    """{(task key, arm, seed, tag)} of the pilot YAMLs: one per final line,
    one per stage-1 grid lr on the first tuning seed per hpo line."""
    out = set()
    for path in configs:
        cfg = load(path)
        key = _task_key(cfg["task"], cfg.get("system"))
        tag = CONTROL if cfg["task"] == "lv" and cfg["dt_scale"] == 1 else ""
        for (name, _, _, flags), _, seed in jobs(cfg):
            if seed is None:
                out |= {(key, name, cfg["seeds"][0], f"lr {lr:g}")
                        for lr in grid(cfg, flags)[0]}
            else:
                out.add((key, name, seed, tag))
    return out


def _pick(fits, key, arm=None, tag=""):
    """The fits of one task key (and arm) with this tag (None: any)."""
    return sorted((f for f in fits if f["arm"][0] == key
                   and arm in (None, f["arm"][1]) and tag in (None, f["tag"])),
                  key=lambda f: (f["arm"], f["seed"], f["path"]))


def _vpt(rec):
    """Test vpt_multi_median; non-finite or missing is 0 (the worst value)."""
    return _num((rec.get("rollout_metrics_test") or {}).get(
        "vpt_multi_median"), 0.0)


def _row(gate, task, criterion, summary, bad=(), decision=None,
         consequence=None, detail=()):
    return {"gate": gate, "task": task, "criterion": criterion,
            "summary": summary, "bad": list(bad), "decision": decision,
            "consequence": consequence, "detail": list(detail)}


def check_pilot_fits(fits, expected):
    have, bad = Counter((*f["arm"], f["seed"], f["tag"]) for f in fits), []
    for k, a, s, t in sorted(expected, key=str):
        n = have[k, a, s, t]
        if n != 1:
            bad.append(f"{_name(k)}/{a} seed {s}{f' ({t})' if t else ''}: "
                       + (f"{n} fits" if n else "no fit"))
    return _row("fits", "all", "every fit of the pilot YAMLs present, once",
                f"{sum(1 for e in expected if have[e])}/{len(expected)} "
                f"present", bad, consequence=CONSEQUENCE["fits"])


def check_pilot_config(fits):
    """config.effective.deterministic true on every pilot fit (#76)."""
    bad = []
    for f in fits:
        v = _eff(f["rec"]).get("deterministic")
        if v is not True:
            tag = f" ({f['tag']})" if f["tag"] else ""
            bad.append(f"{_name(f['arm'][0])}/{f['arm'][1]} seed {f['seed']}"
                       f"{tag}: deterministic {v!r}")
    return _row("config", "all", "config.effective.deterministic true on "
                "every fit (op-determinism on, F1 decision l)",
                f"{len(fits) - len(bad)}/{len(fits)} fits deterministic", bad,
                consequence=CONSEQUENCE["config"])


def gate_floors(fits, lines, expected):
    """G1: cfc_lrc/ncp above max(persistence, AR(1)) + 2 (LV, Duffing);
    cfc/ncp at least 2 x persistence and never unbounded (multitimescale_lib).
    lines: {(system, data seed): the test block of its floors.json draw};
    expected: pilot_fits(). Too few passing seeds fail LV; on Duffing they
    are the F1 section 7 DECISION when every expected seed has a fit with a
    finite VPT and a floor line, else FAIL (#77)."""
    rows = []
    for key in (LV_DEFAULT_SYSTEM, DUF, MT):
        mt = key == MT
        arm = "cfc_ncp" if mt else "cfc_lrc_ncp"
        fs, n_ok, bad, detail = _pick(fits, key, arm), 0, [], []
        unevaluated = [f"seed {s}: no fit" for s in sorted(
            {s for k, a, s, t in expected if (k, a, t) == (key, arm, "")}
            - {f["seed"] for f in fs}, key=str)]
        for f in fs:
            rec, s = f["rec"], f["seed"]
            ds = _eff(rec).get("data_seed")
            line = lines.get((rec.get("system"), ds))
            v = (rec.get("rollout_metrics_test") or {}).get("vpt_multi_median")
            if not _finite(v):
                unevaluated.append(f"seed {s}: test vpt_multi_median {v!r}")
            if line is None:
                detail.append(f"seed {s} (data seed {ds}): no floor line")
                unevaluated.append(detail[-1])
                continue
            if mt:
                p = line["persistence"]["vpt_multi_median"]
                bar, why = MT_FACTOR * p, f"{MT_FACTOR} x persistence {p:g}"
                ok = _vpt(rec) >= bar
                unb = (rec.get("rollout_metrics_test") or {}).get(
                    "rate_unbounded")
                if unb != 0:
                    bad.append(f"seed {s}: rate_unbounded {unb!r}")
            else:
                bar = line["floor"] + FLOOR_MARGIN
                why = f"floor {line['floor']:g} + {FLOOR_MARGIN}"
                ok = _vpt(rec) > bar
            n_ok += ok
            detail.append(f"seed {s} (data seed {ds}): VPT {_vpt(rec):g} vs "
                          f"{bar:g} ({why}) -> {'pass' if ok else 'no'}")
        decision = None
        consequence = CONSEQUENCE["G1 mt" if mt else f"G1 {_name(key)}"]
        if n_ok < MIN_SEEDS and key == DUF and fs and not unevaluated:
            decision = consequence
        elif n_ok < MIN_SEEDS:
            bad[:0] = [f"{n_ok} of {len(fs)} seeds pass (needs "
                       f"{MIN_SEEDS})"] + ([] if mt else unevaluated)
            if key == DUF:
                consequence = CONSEQUENCE["G1 unevaluated"]
        crit = (f"test VPT of cfc/ncp >= {MT_FACTOR} x persistence in >= "
                f"{MIN_SEEDS} of 3 seeds; rate_unbounded 0 on every fit" if mt
                else f"test VPT of cfc_lrc/ncp > max(persistence, AR(1)) + "
                     f"{FLOOR_MARGIN} in >= {MIN_SEEDS} of 3 seeds")
        rows.append(_row("G1", _name(key), crit,
                         f"{n_ok}/{len(fs)} seeds pass", bad, decision,
                         consequence, detail))
    return rows


def gate_epochs(fits, expected):
    """G0-E per task over its pilot arms (the dt_scale control left out):
    DECISION when ceil(2n/3) of the task's n pilot arms in the pilot YAMLs
    (expected: pilot_fits()) have >= 2 qualifying seeds (F1 decision m, read
    as a two-thirds share in F1 section 10.5)."""
    rows = []
    for key in (LV_DEFAULT_SYSTEM, DUF, MT, "pa"):
        kind = "pa" if key == "pa" else "lv"
        curve_key, higher = CURVES[kind]
        qualifying, bad, detail, epochs = {}, [], [], []
        for f in _pick(fits, key):
            label = f"{f['arm'][1]} seed {f['seed']}"
            c = f["rec"].get(curve_key)
            if not isinstance(c, list) or len(c) < 3:
                bad.append(f"{label}: no {curve_key}")
                continue
            n, k = len(c), max(1, round(len(c) * TAIL))
            mark, tail = c[round(n * MARK) - 1], sum(c[-k:]) / k
            gain = (tail - mark if higher else mark - tail) / (
                abs(mark) or math.nan)
            q = gain > GAIN                         # NaN never qualifies
            epochs.append(n)
            qualifying.setdefault(f["arm"][1], []).append(q)
            detail.append(f"{label}: {mark:.4g} at epoch {round(n * MARK)}, "
                          f"{tail:.4g} over the last {k} ({gain:+.1%})"
                          f"{' qualifies' if q else ''}")
        arms = [a for a, qs in qualifying.items() if sum(qs) >= MIN_SEEDS]
        n_arms = len({a for k, a, _, t in expected if k == key and t == ""})
        need = math.ceil(2 * n_arms / 3)
        share = f"{len(arms)} of {n_arms} arms qualify, needs {need}"
        decision, summary = None, f"epochs unchanged ({share})"
        if arms and len(arms) >= need:
            e, x = max(epochs), EPOCH_FACTOR[kind]
            span = f"epochs x{x:g} ({e} to {round(e * x)}"
            decision = f"{span}): {', '.join(arms)} qualify"
            summary = f"{span}; {share})"
        rows.append(_row(
            "G0-E", _name(key),
            f"{curve_key} ({'higher' if higher else 'lower'} is better): "
            f"mean over the last tenth beats the value at 2/3 of the epochs "
            f"by > {GAIN:.0%}; ceil(2n/3) of the n pilot arms with >= "
            f"{MIN_SEEDS} qualifying seeds change the epochs (needs {need} "
            f"of {n_arms})",
            summary, bad, decision, CONSEQUENCE["G0-E"], detail))
    return rows


def _vs_chance(rec):
    """(one-class macro-F1 of the test split, test macro-F1 at chance), the
    value from the per-recording test confusions (row sums = the test
    labels); (None, False) without them (#80)."""
    cms = rec.get("test_confusion_by_recording")
    if not cms:
        return None, False
    one = one_class_macro_f1(np.sum(list(cms.values()), axis=0).sum(axis=1))
    return one, at_chance(_num(rec.get("test_macro_f1")), one)


def gate_pa(fits):
    """G2: cfc_lrc/ncp median test macro-F1, lrc/ncp finite, wall times; the
    test macro-F1 of both arms against the one-class value, reported only
    (#80, the S1 record stays as evaluated)."""
    main, solver = _pick(fits, "pa", "cfc_lrc_ncp"), _pick(fits, "pa",
                                                           "lrc_ncp")
    med = _median([_num(f["rec"].get("test_macro_f1"), -math.inf)
                   for f in main])
    bad = [] if med >= MIN_F1 else [
        f"median test macro-F1 of cfc_lrc/ncp {med:.3f} < {MIN_F1:.2f}"]
    bad += [] if solver else ["lrc/ncp: no fit"]
    for f in solver:
        r = f["rec"]
        why = [k for k in PRIMARY["pa"] if not _finite(r.get(k))] + (
            [] if r.get("training_diverged") is False
            else [f"training_diverged {r.get('training_diverged')!r}"])
        if why:
            bad.append(f"lrc/ncp seed {f['seed']}: {', '.join(why)}")
    vs = {f["path"]: _vs_chance(f["rec"]) for f in main + solver}

    def versus(f):
        one, hit = vs[f["path"]]
        return ("one-class n/a (no test confusions)" if one is None else
                f"one-class {one:.4f} + {CHANCE_MARGIN:g}: "
                f"{'at chance' if hit else 'above chance'}")
    detail = [f"{f['arm'][1]} seed {f['seed']}: test macro-F1 "
              f"{_num(f['rec'].get('test_macro_f1')):.3f}, wallclock_s "
              f"{f['rec'].get('wallclock_s')}; {versus(f)}"
              for f in main + solver]
    chance = [f"{name} ({n} of {len(fs)} seeds)"
              for name, fs in (("cfc_lrc/ncp", main), ("lrc/ncp", solver))
              if (n := sum(vs[f["path"]][1] for f in fs)) >= MIN_SEEDS]
    one = sorted({f"{vs[f['path']][0]:.4f}" for f in solver
                  if vs[f["path"]][0] is not None}) or ["n/a"]
    f1 = [round(_num(f["rec"].get("test_macro_f1")), 4) for f in solver]
    return [_row("G2", "pa", f"median test macro-F1 of cfc_lrc/ncp >= "
                 f"{MIN_F1:.2f}; every lrc/ncp fit finite (wallclock_s "
                 f"listed)",
                 f"median {med:.3f}; lrc/ncp test macro-F1 {f1} vs one-class "
                 f"{'/'.join(one)} (test split); at chance (<= one-class + "
                 f"{CHANCE_MARGIN:g} in >= {MIN_SEEDS} of 3 seeds): "
                 f"{', '.join(chance) or 'none'}; lrc/ncp wallclock_s "
                 f"{[f['rec'].get('wallclock_s') for f in solver]}", bad,
                 consequence=CONSEQUENCE["G2"], detail=detail)]


def gate_dt_scale(fits):
    """G3, descriptive: the dt_scale=1 control next to the main arm."""
    arms = {t: _pick(fits, LV_DEFAULT_SYSTEM, "cfc_lrc_ncp", t)
            for t in (CONTROL, "")}
    med = {t: _median([_vpt(f["rec"]) for f in fs]) for t, fs in arms.items()}
    detail = [f"{'dt_scale 1' if t else 'main arm'} seed {f['seed']}: VPT "
              f"{_vpt(f['rec']):g} (dt_scale {_eff(f['rec']).get('dt_scale')})"
              for t, fs in arms.items() for f in fs]
    return [_row("G3", "lv", "descriptive: median test VPT of cfc_lrc/ncp at "
                 "dt_scale 1 next to the main arm (dt_scale auto)",
                 f"median VPT {med[CONTROL]:g} (dt_scale 1, "
                 f"{len(arms[CONTROL])} fits) vs {med['']:g} (main, "
                 f"{len(arms[''])} fits)", detail=detail)]


def _steps(cm_rec, f1):
    """Macro-F1 per fault count from the per-recording test confusions summed
    over the recordings, and per adjacent step m -> m' the pair (MF1(m') -
    MF1(m), SE): SE is the standard deviation of that difference over BOOT_N
    paired resamples of the recording ids (with replacement, BOOT_SEED; each
    resample sums the confusions of the drawn recordings at m and at m').
    cm_rec: one {recording: confusion} per fault count; ValueError when the
    full sums do not reproduce f1 (test_macro_f1_by_fault)."""
    recs = sorted(cm_rec[0])
    cms = np.array([[c[r] for r in recs] for c in cm_rec], dtype=np.int64)
    mf1 = [macro_f1(c.sum(axis=0)) for c in cms]
    if len(mf1) != len(f1) or not np.allclose(mf1, f1, rtol=0, atol=1e-9):
        raise ValueError(f"summed confusions give {mf1}, not {f1}")
    draws = np.random.default_rng(BOOT_SEED).integers(
        len(recs), size=(BOOT_N, len(recs)))
    counts = (draws[..., None] == np.arange(len(recs))).sum(axis=1)
    boot = np.array([[macro_f1(c) for c in s]      # (resample, fault count)
                     for s in np.einsum("br,frij->bfij", counts, cms)])
    se = np.diff(boot, axis=1).std(axis=0, ddof=1)
    return mf1, [(b - a, float(s)) for a, b, s in zip(mf1, mf1[1:], se)]


def _degradation(r, grid_f1):
    """GK degradation over the fault grid (F1 section 10.5): ([violations],
    detail text). A step m -> m' violates when MF1(m') - MF1(m) exceeds one
    paired bootstrap SE (_steps); MF1(m_max) < MF1(0) is required. Without
    usable CM_KEY confusions: the strict check (non-increasing)."""
    fc = r.get("fault_counts") or list(range(len(grid_f1)))
    pairs = [f"{a:g}->{b:g}" for a, b in zip(fc, fc[1:])]
    try:
        vals, steps = _steps(r[CM_KEY], grid_f1)
    except (KeyError, IndexError, TypeError, ValueError) as e:
        why = f"no {CM_KEY}" if CM_KEY not in r else f"{CM_KEY} unusable: {e}"
        vals = grid_f1
        if not (vals and all(_finite(v) for v in vals)):
            return [f"test macro-F1 by fault {vals} not finite"], (
                f"{why}; strict check")
        text = f"{why}; strict check (non-increasing, no tolerance)"
        bad = [f"{p} rises {b - a:+.3f} (strict)"
               for p, a, b in zip(pairs, vals, vals[1:]) if not b <= a]
    else:
        text = (f"steps MF1(m') - MF1(m) with paired bootstrap SE ("
                f"{len(r[CM_KEY][0])} test recordings, {BOOT_N} resamples): "
                + ", ".join(f"{p} {d:+.3f} (SE {s:.2g})"
                            for p, (d, s) in zip(pairs, steps)))
        bad = [f"{p} rises {d:+.3f} > SE {s:.2g}"
               for p, (d, s) in zip(pairs, steps) if not d <= s]
    if not vals[-1] < vals[0]:
        bad.append(f"m={fc[-1]:g} {vals[-1]:.3f} not below m=0 {vals[0]:.3f}")
    return bad, text


def gate_consensus(fits):
    """GK on the best-lr fit of each consensus pilot arm."""
    bad, detail, consequences, corruption = [], [], [], set()
    for arm in GK_ARMS:
        fs = _pick(fits, "co", arm, None)
        if not fs:
            bad.append(f"{arm}: no fit")
            consequences.append(CONSEQUENCE["fits"])
            continue
        r = min(fs, key=lambda f: (-_num(f["rec"].get("val_macro_f1_mean"),
                                         -math.inf), f["rec"]["lr"]))["rec"]
        corruption.add(r.get("corruption_seed"))
        try:
            m0 = r["val_macro_f1_grid_curve"][r["best_val_epoch"] - 1][
                r["fault_counts"].index(0)]
        except (KeyError, IndexError, TypeError, ValueError):
            m0 = math.nan
        grid_f1 = r.get("test_macro_f1_by_fault") or []
        lam = _num(r.get("vote_lambda_final"))
        violated, text = _degradation(r, grid_f1)
        detail.append(
            f"{arm}: best lr {r['lr']:g} (val_macro_f1_mean "
            f"{_num(r.get('val_macro_f1_mean')):.3f}), val macro-F1 at m=0 "
            f"{_num(m0):.3f}, test macro-F1 by fault "
            f"{[round(v, 3) for v in grid_f1]}"
            + (f", vote_lambda_final {lam:.3f}" if arm == "ncp_cons" else "")
            + f"; {text}")
        if not _num(m0) >= MIN_F1:
            bad.append(f"{arm}: validation macro-F1 at m=0 {_num(m0):.3f} < "
                       f"{MIN_F1:.2f} at its best lr {r['lr']:g}")
            consequences.append(
                CONSEQUENCE["GK" if arm == "ncp_cons" else "GK arm"])
        if violated:
            bad.append(f"{arm}: test macro-F1 does not degrade over the "
                       f"fault grid: {'; '.join(violated)}")
            consequences.append(CONSEQUENCE["GK monotone"])
        if arm == "ncp_cons" and not LAMBDA[0] < lam < LAMBDA[1]:
            bad.append(f"ncp_cons: vote_lambda_final {lam:.3f} outside "
                       f"{LAMBDA}")
            consequences.append(CONSEQUENCE["GK lambda"])
    detail.append(f"fault schedule: the committee JSON stores no per-batch "
                  f"fault-count sequence or hash (fault_counts is the grid; "
                  f"corruption_seed {sorted(corruption, key=str)}), not compared")
    return [_row("GK", "co", f"best lr per arm (val_macro_f1_mean): "
                 f"validation macro-F1 at m=0 >= {MIN_F1:.2f}; test macro-F1 "
                 f"degrades over the fault grid (no step up by more than one "
                 f"paired bootstrap SE over the test recordings, MF1(m_max) "
                 f"< MF1(0)); ncp_cons vote_lambda_final in {LAMBDA}",
                 f"{len(GK_ARMS) - len({b.split(':')[0] for b in bad})}/"
                 f"{len(GK_ARMS)} arms pass", bad,
                 consequence="; ".join(dict.fromkeys(consequences)),
                 detail=detail)]


def gate_s1(fits, expected, floors):
    """The S1 rows: fits, config, floors, G1, G0-E, G2, G3, GK."""
    _, desc, bad, _ = check_floors(fits, floors)
    lines = {(d.get("system"), d.get("data_seed")): d.get("test") or {}
             for fl in floors for d in fl.get("floors", [])}
    return [check_pilot_fits(fits, expected), check_pilot_config(fits),
            _row("floors", "rollouts", desc, f"{len(bad)} missing", bad,
                 consequence=CONSEQUENCE["floors"]),
            *gate_floors(fits, lines, expected), *gate_epochs(fits, expected),
            *gate_pa(fits),
            *gate_dt_scale(fits), *gate_consensus(fits)]


def _status(row):
    return "FAIL" if row["bad"] else "DECISION" if row["decision"] else "PASS"


def report_s1(rows, runs, configs, n_fits):
    ok = not any(r["bad"] for r in rows)

    def cell(text):
        return text.replace("|", "\\|")
    lines = [f"# S1 pilot gate: {'PASS' if ok else 'FAIL'}\n",
             f"Runs: {', '.join(runs)}  ",
             f"Fit JSONs: {n_fits}; expected fits from "
             f"{', '.join(os.path.basename(c) for c in configs)}\n",
             "| Gate | Task | Result | Criterion | Summary |",
             "|------|------|--------|-----------|---------|"]
    lines += [f"| {r['gate']} | {r['task']} | {_status(r)} | "
              f"{cell(r['criterion'])} | {cell(r['summary'])} |" for r in rows]
    for r in rows:
        if not (r["bad"] or r["decision"] or r["detail"]):
            continue
        lines += ["", f"## {_status(r)}: {r['gate']} {r['task']}", ""]
        lines += [f"- {b}" for b in r["bad"]]
        if r["bad"]:
            lines.append(f"- Consequence: {r['consequence']}")
        if r["decision"]:
            lines.append(f"- DECISION: {r['decision']}")
        lines += [f"- {d}" for d in r["detail"]]
    return "\n".join(lines) + "\n", ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", required=True, choices=["s0", "s1"])
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run folders (s0: searched for "
                         "hpo/<arm>/lr*_clip*_seed*/; s1: every fit JSON)")
    ap.add_argument("--configs", nargs="+", default=None,
                    help="campaign YAMLs of the expected arm-tasks (default: "
                         "s0 the four smoke YAMLs, 93 arm-tasks; s1 the six "
                         "pilot YAMLs, 61 fits)")
    ap.add_argument("--determinism", default=None,
                    help="determinism_probe.json of the S0 probe (without "
                         "it the determinism check fails: not measured)")
    ap.add_argument("--floors", nargs="+", default=None,
                    help="floors.json of aggregate_results.py --floors")
    ap.add_argument("--out", required=True, help="Markdown report")
    a = ap.parse_args(argv)
    for p in a.runs:
        if not os.path.isdir(p):
            ap.error(f"no run folder {p}")

    def read(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def probe(path):            # missing or unreadable: not measured (FAIL)
        try:
            return read(path) if path else None
        except (OSError, ValueError):
            return None
    if a.stage == "s1":
        if not a.floors:
            ap.error("--stage s1 needs --floors")
        configs = a.configs or PILOT
        fits = read_pilot_fits(a.runs)
        text, ok = report_s1(gate_s1(fits, pilot_fits(configs),
                                     [read(p) for p in a.floors]),
                             a.runs, configs, len(fits))
    else:
        configs = a.configs or SMOKE
        fits = read_fits(a.runs)
        results, exempt, listed, unexpected = gate(
            fits, expected_arms(configs), probe(a.determinism),
            [read(p) for p in a.floors] if a.floors else None)
        text, ok = report(results, exempt, listed, unexpected, a.runs,
                          configs, len(fits))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
