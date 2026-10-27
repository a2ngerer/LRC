"""S0 smoke gate of the frozen suite (#72; protocol freeze F1-2026-10-26,
validation ladder S0).

Reads every fit JSON below the given run folders
(<run>/results/hpo/<arm>/lr*_clip*_seed*/*.json, run_grid_hpo.py under
cluster/submit_suite.sh) and checks the S0 criteria:

  arms              every arm-task of the smoke YAMLs (--configs, default the
                    93 of configs/suite/smoke_{lv,duffing,mt,pa}.yaml, expanded
                    by suite_manifest) has exactly one fit; fits of other arms
                    (e.g. the consensus smoke) are listed and checked, not
                    failed
  config            config.effective (or the top level, where the runners
                    write weights_path) carries ode_unfolds, dt_scale, forcing
                    (rollout tasks) and a weights_path; the committee runner
                    has no dt_scale (closed-form cells)
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
  determinism       with --determinism (determinism_probe.json, #70): runtime
                    ratio on/off <= 1.3 and |Delta| of the two "on" fits
                    <= 1e-3 on lv and pa
  floors            with --floors (aggregate_results.py --floors, one or more
                    floors.json): a floor line with a finite val and test
                    floor per rollout system and data seed of the fits

The report (--out, Markdown, also printed) lists every check as PASS/FAIL with
the offending arms; the exit code is 0 only when every check passes.

Usage:
    uv run python experiments/suite_gate.py --stage s0 \\
        --runs results/00_canary/<date>_s0-smoke-{lv,duffing,mt,pa,co} \\
        --determinism <probe run>/results/determinism_probe.json \\
        --floors <agg>/{lv,duffing,mt}/floors.json --out S0_GATE.md
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from functools import lru_cache
from glob import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.suite_manifest import (LV_DEFAULT_SYSTEM, jobs,  # noqa: E402
                                        load)
from src.benchmark.registry import cell_kwargs  # noqa: E402
from src.evaluation.rollout_metrics import ANCHORS  # noqa: E402

SMOKE = [os.path.join(ROOT, "configs", "suite", f"smoke_{t}.yaml")
         for t in ("lv", "duffing", "mt", "pa")]
TASKS = {"lotka_volterra": "lv", "person_activity": "pa", "committee": "co"}
CONFIG_KEYS = {"lv": ("ode_unfolds", "dt_scale", "forcing", "weights_path"),
               "pa": ("ode_unfolds", "dt_scale", "weights_path"),
               "co": ("ode_unfolds", "weights_path")}
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
    """(present, value) of key in config.effective, else at the top level."""
    for d in (_eff(rec), rec):
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
    "config": "config.effective: ode_unfolds, dt_scale, forcing (rollouts), "
              "weights_path",
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
    """[(check, description, [offending entries])], {arm label: exempt
    paths}, {arm label: listed undefined inv_drift_* blocks}, [arm labels
    outside the expected list]."""
    n = Counter(f["arm"] for f in fits)
    missing = [a for a in expected if not n[a]]
    results = [("arms", f"{len(expected) - len(missing)}/{len(expected)} "
                f"arm-tasks present, one fit each",
                [f"{_label(a)}: no fit" for a in missing]
                + [f"{_label(a)}: {k} fits" for a, k in sorted(n.items())
                   if k > 1])]
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
        results.append((k, desc, per_check[k]))
    if determinism is not None:
        results.append(check_determinism(determinism))
    if floors is not None:
        results.append(check_floors(fits, floors))
    unexpected = sorted(_label(a) for a in n if a not in expected)
    return results, exempt, listed, unexpected


def check_determinism(probe):
    """Runtime ratio and on-pair |Delta| of determinism_probe.json (#70)."""
    bad, seen = [], []
    for task in ("lv", "pa"):
        s = (probe.get("tasks") or {}).get(task)
        if s is None:
            bad.append(f"{task}: not in the probe")
            continue
        ratio, delta = s.get("ratio_on_off"), s.get("max_abs_delta_on")
        seen.append(f"{task} ratio {ratio:.3g}, |Delta| {delta:.3g}"
                    if _finite(ratio) and _finite(delta) else task)
        if not (_finite(ratio) and ratio <= MAX_RATIO):
            bad.append(f"{task}: runtime ratio on/off {ratio} > {MAX_RATIO}")
        if not (_finite(delta) and delta <= MAX_DELTA):
            bad.append(f"{task}: |Delta| of the on fits {delta} > "
                       f"{MAX_DELTA:g}")
    return ("determinism", f"ratio <= {MAX_RATIO}, |Delta on| <= "
            f"{MAX_DELTA:g} on lv and pa ({'; '.join(seen)})", bad)


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
            f"({len(want)} draws)", bad)


def report(results, exempt, listed, unexpected, runs, configs, n_fits):
    ok = all(not bad for _, _, bad in results)
    lines = [f"# S0 smoke gate: {'PASS' if ok else 'FAIL'}\n",
             f"Runs: {', '.join(runs)}  ",
             f"Fit JSONs: {n_fits}; expected arm-tasks from "
             f"{', '.join(os.path.basename(c) for c in configs)}\n",
             "| Check | Result | Criterion | Offending |",
             "|-------|--------|-----------|-----------|"]
    lines += [f"| {k} | {'FAIL' if bad else 'PASS'} | {desc} | {len(bad)} |"
              for k, desc, bad in results]
    for k, _, bad in results:
        if bad:
            lines += ["", f"## FAIL: {k}", ""] + [f"- {b}" for b in bad]
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", required=True, choices=["s0"])
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run folders (searched for "
                         "hpo/<arm>/lr*_clip*_seed*/)")
    ap.add_argument("--configs", nargs="+", default=SMOKE,
                    help="campaign YAMLs of the expected arm-tasks (default: "
                         "the four smoke YAMLs, 93 arm-tasks)")
    ap.add_argument("--determinism", default=None,
                    help="determinism_probe.json of the S0 probe")
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
    fits = read_fits(a.runs)
    results, exempt, listed, unexpected = gate(
        fits, expected_arms(a.configs),
        read(a.determinism) if a.determinism else None,
        [read(p) for p in a.floors] if a.floors else None)
    text, ok = report(results, exempt, listed, unexpected, a.runs, a.configs,
                      len(fits))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
