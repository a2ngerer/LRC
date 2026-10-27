"""Final-suite manifest: one campaign YAML -> one JSONL line per SLURM task (#52).

Campaign YAML (configs/suite/*.yaml):

    task: lv                    # lv | pa
    system: duffing             # lv only; the runner default if unset
    stage: hpo                  # hpo | final
    cells: [cfc_lrc, gru]
    wirings: [dense3, ncp]
    param_budget: 4000          # optional
    seeds: [0, 1]               # hpo: the two tuning seeds; final: the replicates
    epochs: 300                 # optional, the runner default if unset
    ode_unfolds: 6              # required: Euler sub-steps of the solver cells
    dt_scale: auto              # required: auto (1 / the task's dt) or a number
                                # both go to every runner call; per_cell
                                # overrides them for a control arm (protocol
                                # freeze F1-2026-10-26 a, h)
    extra_args: []              # appended to every call (run_grid_hpo.py
                                # forwards what it does not know to the runner)
    per_cell:                   # optional flags for the arms of one cell,
      lrc: {ode_unfolds: 1}     # key -> --key-with-dashes value (true = bare flag)
    hpo_dir: results/01_lotka_volterra/<run>/results/hpo   # final only: the
                                # fetched HPO run (cluster/fetch_run.sh)
    cluster: {runs_per_gpu: 4, mem: 32G, walltime: "12:00:00", cpus: 12,
              array_throttle: 8}                    # optional, see below
    heavy:                      # optional: arms with one of these cells or
      cells: [lrc]              # wirings run as their own array job with
      wirings: [cncp]           # cluster resources merged over `cluster`
      cluster: {runs_per_gpu: 1, mem: 48G, walltime: "48:00:00"}

Arms (cell, wiring) are ordered cell-major in YAML order. Stage hpo gives one
line per arm: one experiments/run_grid_hpo.py call into results/hpo/<cell>_<wiring>/.
Stage final gives one line per (arm, seed): one task-runner fit into results/
(flat, as experiments/aggregate_results.py reads it) with the lr and clipnorm
of <hpo_dir>/<cell>_<wiring>/hpo_selection.json.

A line is {"id", "group", "argv", "outdir", "done"}; outdir and done are
relative to the run folder's results/. A line whose results/<done> exists (the
fit's result JSON, or the arm's hpo_selection.json) is done and skipped, so a
requeue or a re-submit after a walltime kill only runs what is missing.

Usage:
    uv run python experiments/suite_manifest.py CONFIG --list [--group G]
    uv run python experiments/suite_manifest.py CONFIG --count [--group G]
    # cluster/submit_suite.sh: manifest_<group>.jsonl per group + a shell plan
    uv run python experiments/suite_manifest.py CONFIG --write DIR [--wandb]
    # one SLURM bundle member (cluster/suite_array.sbatch): skip if done, else run
    uv run python experiments/suite_manifest.py --run MANIFEST --index I --results DIR
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys

import yaml

RUNNERS = {"lv": "experiments/run_lotka_volterra_benchmark.py",
           "pa": "experiments/run_person_activity_benchmark.py"}
KEYS = {"task", "system", "stage", "cells", "wirings", "param_budget", "seeds",
        "epochs", "ode_unfolds", "dt_scale", "extra_args", "per_cell", "hpo_dir",
        "cluster", "heavy", "description"}
CLUSTER_KEYS = ("runs_per_gpu", "mem", "walltime", "cpus", "array_throttle")


def load(path):
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    unknown = set(cfg) - KEYS
    if unknown:
        raise SystemExit(f"{path}: unknown keys {sorted(unknown)}")
    for k in ("task", "stage", "cells", "wirings", "seeds"):
        if not cfg.get(k):
            raise SystemExit(f"{path}: '{k}' is required")
    # Written by the campaign, never left to a runner or cell default.
    for k in ("ode_unfolds", "dt_scale"):
        if cfg.get(k) is None:
            raise SystemExit(f"{path}: '{k}' is required")
    if not (isinstance(cfg["ode_unfolds"], int) and cfg["ode_unfolds"] >= 1):
        raise SystemExit(f"{path}: ode_unfolds must be an integer >= 1")
    if cfg["dt_scale"] != "auto" and not (
            isinstance(cfg["dt_scale"], (int, float)) and cfg["dt_scale"] > 0):
        raise SystemExit(f"{path}: dt_scale must be auto or a positive number")
    if cfg["task"] not in RUNNERS:
        raise SystemExit(f"{path}: task must be lv or pa")
    if cfg["stage"] not in ("hpo", "final"):
        raise SystemExit(f"{path}: stage must be hpo or final")
    if cfg.get("system") and cfg["task"] != "lv":
        raise SystemExit(f"{path}: system applies to task lv only")
    if cfg["stage"] == "hpo" and len(set(cfg["seeds"])) != 2:
        raise SystemExit(f"{path}: stage hpo needs two different tuning seeds")
    if cfg["stage"] == "final" and not cfg.get("hpo_dir"):
        raise SystemExit(f"{path}: stage final needs hpo_dir")
    return cfg


def bench(cfg):
    """Benchmarks-tree folder of the run (same mapping as submit_grid_hpo.sh)."""
    if cfg["task"] == "pa":
        return "03_person_activity"
    return "02_duffing" if cfg.get("system") == "duffing" else "01_lotka_volterra"


def _flags(mapping):
    out = []
    for k, v in mapping.items():
        flag = "--" + k.replace("_", "-")
        if v is True:
            out.append(flag)
        elif isinstance(v, list):
            out += [flag, *map(str, v)]
        else:
            out += [flag, str(v)]
    return out


def _flag_value(argv, flag):
    """Value of the last `flag` in argv, or None."""
    vals = [argv[i + 1] for i, a in enumerate(argv[:-1]) if a == flag]
    return vals[-1] if vals else None


def result_name(task, cell, wiring, seed, argv):
    """Result JSON name the task runner writes for this fit (LV result_filename,
    PA result_tag; tests/experiments/test_suite_manifest.py checks the match)."""
    if wiring == "cncp2_multi":
        # ponytail: the multi-column suffix needs the runner's column config;
        # add it here when cncp2_multi enters the final suite.
        raise SystemExit("stage final does not support cncp2_multi")
    bb = _flag_value(argv, "--backbone-layers")
    mixed = task == "lv" and _flag_value(argv, "--forcing") == "mixed"
    return (f"{cell}_{wiring}_seed{seed}{f'_bb{int(bb)}' if bb else ''}"
            f"{'_mixed' if mixed else ''}.json")


def selection(cfg, cell, wiring):
    """The arm's hpo_selection.json, checked against the campaign."""
    path = os.path.join(cfg["hpo_dir"], f"{cell}_{wiring}", "hpo_selection.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        sel = json.load(f)
    want = {"task": cfg["task"], "cell": cell, "wiring": wiring,
            "system": cfg.get("system"), "param_budget": cfg.get("param_budget")}
    for k, v in want.items():
        if v is not None and sel.get(k) != v:
            raise SystemExit(f"{path}: {k} {sel.get(k)!r}, campaign has {v!r}")
    return sel


def expand(cfg, wandb=False):
    """The manifest lines in submission order (see the module docstring)."""
    task, system = cfg["task"], cfg.get("system")
    heavy = cfg.get("heavy") or {}
    common = []
    if cfg.get("param_budget") is not None:
        common += ["--param-budget", str(cfg["param_budget"])]
    if cfg.get("epochs") is not None:
        common += ["--epochs", str(cfg["epochs"])]
    common += ["--ode-unfolds", str(cfg["ode_unfolds"]),
               "--dt-scale", str(cfg["dt_scale"])]
    lines, missing = [], []
    for cell in cfg["cells"]:
        for wiring in cfg["wirings"]:
            group = ("heavy" if cell in heavy.get("cells", ())
                     or wiring in heavy.get("wirings", ()) else "default")
            tail = [*common, *_flags((cfg.get("per_cell") or {}).get(cell, {})),
                    *map(str, cfg.get("extra_args") or []),
                    *(["--wandb"] if wandb else [])]
            arm = f"{cell}_{wiring}"
            if cfg["stage"] == "hpo":
                argv = ["uv", "run", "python", "experiments/run_grid_hpo.py",
                        "--task", task, "--cell", cell, "--wiring", wiring,
                        *(["--system", system] if system else []),
                        "--tuning-seeds", *map(str, cfg["seeds"]), *tail]
                lines.append({"id": f"hpo_{arm}", "group": group, "argv": argv,
                              "outdir": f"hpo/{arm}",
                              "done": f"hpo/{arm}/hpo_selection.json"})
                continue
            sel = selection(cfg, cell, wiring)
            if sel is None:
                missing.append(arm)
                continue
            for seed in cfg["seeds"]:
                argv = ["uv", "run", "python", RUNNERS[task], "--cell", cell,
                        "--wiring", wiring, "--seed", str(seed),
                        *(["--system", system] if system else []),
                        "--lr", f"{sel['lr']:g}",
                        *(["--clipnorm", f"{sel['clipnorm']:g}"]
                          if sel["clipnorm"] is not None else []), *tail]
                name = result_name(task, cell, wiring, seed, argv)
                lines.append({"id": name[:-len(".json")], "group": group,
                              "argv": argv, "outdir": ".", "done": name})
    if missing:
        raise SystemExit(f"no hpo_selection.json under {cfg['hpo_dir']} for "
                         f"{', '.join(missing)}")
    return lines


def groups(lines):
    """{group: its lines}, groups in first-appearance order."""
    out = {}
    for line in lines:
        out.setdefault(line["group"], []).append(line)
    return out


def is_done(line, results):
    return os.path.exists(os.path.join(results, line["done"]))


def pending(lines, results):
    """The lines still to run: the resume filter."""
    return [line for line in lines if not is_done(line, results)]


def resources(cfg, group):
    res = dict(cfg.get("cluster") or {})
    if group == "heavy":
        res.update((cfg.get("heavy") or {}).get("cluster") or {})
    return {k: res[k] for k in CLUSTER_KEYS if k in res}


def write(cfg, lines, outdir):
    """manifest_<group>.jsonl per group; returns the shell plan for
    cluster/submit_suite.sh (unset resources fall back to cluster/config.sh)."""
    by_group = groups(lines)
    plan = [f"SUITE_BENCH={bench(cfg)}", f"SUITE_STAGE={cfg['stage']}",
            f"SUITE_COUNT={len(lines)}",
            f"SUITE_GROUPS={shlex.quote(' '.join(by_group))}"]
    for g, glines in by_group.items():
        with open(os.path.join(outdir, f"manifest_{g}.jsonl"), "w") as f:
            f.writelines(json.dumps(line) + "\n" for line in glines)
        plan.append(f"SUITE_{g}_N={len(glines)}")
        plan += [f"SUITE_{g}_{k.upper()}={shlex.quote(str(v))}"
                 for k, v in resources(cfg, g).items()]
    return "\n".join(plan)


def run(manifest, index, results):
    """One SLURM bundle member: skip a done line, else exec its command."""
    with open(manifest) as f:
        line = json.loads(f.readlines()[index])
    if is_done(line, results):
        print(f"[skip] {line['id']}: {line['done']} exists", flush=True)
        return
    argv = [*line["argv"], "--outdir",
            os.path.normpath(os.path.join(results, line["outdir"]))]
    print(f"[run ] {line['id']}\n+ {shlex.join(argv)}", flush=True)
    os.execvp(argv[0], argv)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config", nargs="?")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true")
    mode.add_argument("--count", action="store_true")
    mode.add_argument("--write", metavar="DIR")
    mode.add_argument("--run", metavar="MANIFEST")
    ap.add_argument("--group", default=None, help="--list/--count: one group")
    ap.add_argument("--wandb", action="store_true",
                    help="append --wandb to every call (offline runs)")
    ap.add_argument("--index", type=int, help="--run: manifest line")
    ap.add_argument("--results", help="--run: the run folder's results/")
    a = ap.parse_args(argv)
    if a.run:
        if a.index is None or not a.results:
            ap.error("--run needs --index and --results")
        return run(a.run, a.index, a.results)
    if not a.config:
        ap.error("CONFIG is required")
    cfg = load(a.config)
    lines = expand(cfg, wandb=a.wandb)
    if a.write:
        print(write(cfg, lines, a.write))
        return
    by_group = groups(lines)
    if a.group:
        by_group = {a.group: by_group.get(a.group, [])}
    if a.count:
        print(sum(map(len, by_group.values())))
        return
    for g, glines in by_group.items():
        for i, line in enumerate(glines):
            cmd = shlex.join([*line["argv"], "--outdir",
                              os.path.join("<results>", line["outdir"])])
            print(f"{g}[{i}]\t{line['id']}\t{cmd}")


if __name__ == "__main__":
    sys.exit(main())
