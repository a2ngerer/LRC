"""S0 op-determinism probe (#70, protocol freeze F1-2026-10-26, decision l).

Runs cfc_lrc/ncp on Lotka-Volterra and on Person Activity three times each at
one seed, in the order on, on, off (runner flag --deterministic), one fit after
the other so no two fits share the device. Per task it prints

  * the runtime ratio on/off: mean training wall time (result key wallclock_s)
    of the two "on" fits over the "off" fit;
  * |Delta| of each validation selection metric (run_grid_hpo.SELECT: LV
    rollout_metrics_val.vpt_median and .nrmse_32_median, PA val_macro_f1)
    between the two "on" fits, which must stay within --tol (1e-3).

Decision rule of the protocol: ratio <= --max-ratio (1.3) on both tasks ->
--deterministic for the whole suite, otherwise off and seed noise is reported
as a nuisance component. The summary goes to <outdir>/determinism_probe.json;
the exit code is 1 when an "on" pair is not reproducible. A fit whose result
JSON exists is reused, so a requeued job only runs what is missing.

Usage:
    uv run python experiments/determinism_probe.py --outdir results/s0_determinism
    uv run python experiments/determinism_probe.py --outdir /tmp/smoke \\
        --lv-args "--epochs 1" --pa-args "--epochs 1"            # smoke
"""
import argparse
import json
import math
import os
import shlex
import subprocess
import sys
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_grid_hpo import RUNNERS, SELECT, _metric  # noqa: E402

CELL, WIRING = "cfc_lrc", "ncp"
# Protocol lengths and budgets; --lv-args / --pa-args are appended, so a later
# --epochs overrides these (argparse keeps the last value).
TASK_ARGS = {"lv": "--epochs 300 --param-budget 4000",
             "pa": "--epochs 50 --param-budget 30000"}
RUNS = (("on1", True), ("on2", True), ("off", False))


def fit(task, run, deterministic, a):
    """Run (or reuse) one fit into <outdir>/<task>/<run>/; return its record."""
    d = os.path.join(a.outdir, task, run)
    if not glob(os.path.join(d, "*.json")):
        cmd = ["uv", "run", "python", RUNNERS[task], "--cell", CELL,
               "--wiring", WIRING, "--seed", str(a.seed), "--outdir", d,
               "--no-save-weights", *shlex.split(TASK_ARGS[task]),
               *shlex.split(getattr(a, f"{task}_args")),
               *(["--deterministic"] if deterministic else [])]
        print("+", shlex.join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    (path,) = glob(os.path.join(d, "*.json"))     # one result per fit folder
    with open(path) as f:
        rec = json.load(f)
    if rec["config"]["effective"].get("deterministic") is not deterministic:
        raise SystemExit(f"{path}: config.effective.deterministic is not "
                         f"{deterministic}")
    return rec, path


def _delta(x, y):
    # two identical divergences (NaN, NaN) reproduce; one NaN does not
    return 0.0 if math.isnan(x) and math.isnan(y) else abs(x - y)


def summarize(task, recs, tol):
    """recs: run name -> (record, path). Ratio and on-pair deltas of one task."""
    wall = {r: rec["wallclock_s"] for r, (rec, _) in recs.items()}
    metrics = {}
    for key in filter(None, SELECT[task]):
        v = {r: _metric(rec, key, path) for r, (rec, path) in recs.items()}
        metrics[key] = {**v, "abs_delta_on": _delta(v["on1"], v["on2"])}
    worst = max(m["abs_delta_on"] for m in metrics.values())
    return {"wallclock_s": wall,
            "ratio_on_off": (wall["on1"] + wall["on2"]) / 2 / wall["off"],
            "metrics": metrics, "max_abs_delta_on": worst,
            "reproducible": worst <= tol}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tasks", nargs="+", choices=sorted(TASK_ARGS),
                    default=sorted(TASK_ARGS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lv-args", default="",
                    help="extra LV runner flags, e.g. '--forcing mixed'")
    ap.add_argument("--pa-args", default="", help="extra PA runner flags")
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="max |Delta| of a validation metric between the two "
                         "'on' fits")
    ap.add_argument("--max-ratio", type=float, default=1.3,
                    help="runtime ratio on/off up to which determinism is "
                         "enabled for the suite")
    a = ap.parse_args(argv)

    tasks = {t: summarize(t, {r: fit(t, r, det, a) for r, det in RUNS}, a.tol)
             for t in a.tasks}
    summary = {
        "cell": CELL, "wiring": WIRING, "seed": a.seed, "tol": a.tol,
        "max_ratio": a.max_ratio, "tasks": tasks,
        "reproducible": all(s["reproducible"] for s in tasks.values()),
        "enable_for_suite": all(s["ratio_on_off"] <= a.max_ratio
                                for s in tasks.values())}
    for t, s in tasks.items():
        w = s["wallclock_s"]
        print(f"[{t}] runtime on/off {s['ratio_on_off']:.3f} (on {w['on1']:g}s, "
              f"{w['on2']:g}s; off {w['off']:g}s)")
        for key, m in s["metrics"].items():
            print(f"[{t}]   {key}: on {m['on1']:.6g} / {m['on2']:.6g}, off "
                  f"{m['off']:.6g}, |Delta on| {m['abs_delta_on']:.3g}")
        print(f"[{t}] reproducible (|Delta| <= {a.tol:g}): {s['reproducible']}")
    print(f"op-determinism for the suite: "
          f"{'ON' if summary['enable_for_suite'] else 'OFF'} (ratio <= "
          f"{a.max_ratio:g} on {', '.join(tasks)})")
    os.makedirs(a.outdir, exist_ok=True)
    path = os.path.join(a.outdir, "determinism_probe.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {path}")
    if not summary["reproducible"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
