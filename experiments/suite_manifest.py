"""Final-suite manifest: one campaign YAML -> one JSONL line per SLURM task (#52).

Campaign YAML (configs/suite/*.yaml; the frozen protocol F1-2026-10-26 in
configs/suite/{lv,duffing,mt,pa,co}_*.yaml, #71):

    task: lv                    # lv | pa | co (committee, consensus study)
    system: duffing             # lv only; the runner default if unset
    stage: hpo                  # hpo | final
    cells: [cfc_lrc, gru]       # cells x wirings, or a mapping cell -> its
    wirings: [dense3, ncp]      # wirings ({gru: [dense, ncp], mlp: [dense]})
    arms:                       # or named arms (co): name -> cell, wiring and
      ncp_lam0: {cell: cfc_lrc, wiring: ncp, vote_lambda: 0}   # runner flags
    param_budget: 4000          # optional
    seeds: [0, 1]               # hpo: the two tuning seeds; final: the replicates
    extra_seeds:                # final only: more seeds for some arms, emitted
      seeds: [5, 6, 7, 8, 9]    # after every base line (the ten-seed arms)
      arms: [cfc_lrc_ncp]
    config_from:                # arms without HPO: skipped in stage hpo; stage
      ltc_ncp: lrc_ncp          # final reads the named reference arm's selection
    epochs: 300                 # optional, the runner default if unset
    ode_unfolds: 6              # lv, pa required: Euler sub-steps of the solver cells
    dt_scale: auto              # lv, pa required: auto (1 / the task's dt) or a number
                                # both go to every runner call; per_cell
                                # overrides them for a control arm (protocol
                                # freeze F1-2026-10-26 a, h). lv also gets the
                                # multi-anchor endpoint (--anchors 8,16,24,32
                                # --horizon 96) and the system's --ic-spread.
    extra_args: []              # appended to every call (run_grid_hpo.py
                                # forwards what it does not know to the runner)
    per_cell:                   # optional flags for the arms of one cell, after
      lrc: {ode_unfolds: 1}     # extra_args: key -> --key-with-dashes value
                                # (true = bare flag); likewise a named arm's keys
    hpo_dir: results/01_lotka_volterra/<run>/results/hpo   # final only: the
                                # fetched HPO run (cluster/fetch_run.sh)
    cluster: {runs_per_gpu: 4, mem: 32G, walltime: "12:00:00", cpus: 12,
              array_throttle: 8}                    # optional, see below
    heavy:                      # optional: arms with one of these cells,
      cells: [lrc]              # wirings or arm names run as their own array
      wirings: [cncp]           # job with cluster resources merged over
      arms: [ctrnn_ncp]         # `cluster`
      cluster: {runs_per_gpu: 1, mem: 48G, walltime: "48:00:00"}

Arms are ordered cell-major in YAML order and named <cell>_<wiring> (or by
their `arms` key). Stage hpo gives one line per arm: one
experiments/run_grid_hpo.py call into results/hpo/<arm>/. Stage final gives one
line per (arm, seed): one task-runner fit into results/ (flat, as
experiments/aggregate_results.py reads it) with the lr and clipnorm of
<hpo_dir>/<arm or its config_from reference>/hpo_selection.json (co: also its
final_epochs, the epoch guard of #55).

A line is {"id", "group", "argv", "outdir", "done"}; outdir and done are
relative to the run folder's results/. A line whose results/<done> exists (the
fit's result JSON, or the arm's hpo_selection.json; co: a glob, the runner's
name carries the budget-derived size) is done and skipped, so a requeue or a
re-submit after a walltime kill only runs what is missing.

--count prints the planned fits (an hpo line counts HPO_FITS of its task) of one
or more YAMLs and needs no hpo_selection.json; the frozen fit counts per task:
    uv run python experiments/suite_manifest.py --count configs/suite/lv_*.yaml

Usage:
    uv run python experiments/suite_manifest.py CONFIG --list [--group G]
    uv run python experiments/suite_manifest.py --count CONFIG... [--group G]
    # cluster/submit_suite.sh: manifest_<group>.jsonl per group + a shell plan
    uv run python experiments/suite_manifest.py CONFIG --write DIR [--wandb]
    # one SLURM bundle member (cluster/suite_array.sbatch): skip if done, else run
    uv run python experiments/suite_manifest.py --run MANIFEST --index I --results DIR
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shlex
import sys

import yaml

RUNNERS = {"lv": "experiments/run_lotka_volterra_benchmark.py",
           "pa": "experiments/run_person_activity_benchmark.py",
           "co": "experiments/run_committee_benchmark.py"}
# Planned fits of one hpo line (protocol freeze F1-2026-10-26, section 5): lv
# and pa 4 lr x 2 clip on seed 100, top 2 on seed 101, +1 expected for the edge
# rule; co 4 lr (one fixed clip) + top 2, edge extensions not planned.
HPO_FITS = {"lv": 11, "pa": 11, "co": 6}
LV_DEFAULT_SYSTEM = "periodic_predator_prey"   # run_lotka_volterra_benchmark.py
KEYS = {"task", "system", "stage", "cells", "wirings", "arms", "param_budget",
        "seeds", "extra_seeds", "config_from", "epochs", "ode_unfolds",
        "dt_scale", "extra_args", "per_cell", "hpo_dir", "cluster", "heavy",
        "description"}
CLUSTER_KEYS = ("runs_per_gpu", "mem", "walltime", "cpus", "array_throttle")


def load(path):
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    unknown = set(cfg) - KEYS
    if unknown:
        raise SystemExit(f"{path}: unknown keys {sorted(unknown)}")
    for k in ("task", "stage", "seeds"):
        if not cfg.get(k):
            raise SystemExit(f"{path}: '{k}' is required")
    if cfg["task"] not in RUNNERS:
        raise SystemExit(f"{path}: task must be one of {', '.join(RUNNERS)}")
    if bool(cfg.get("arms")) == bool(cfg.get("cells")):
        raise SystemExit(f"{path}: give either 'cells' or 'arms'")
    if any(not (a or {}).get("cell") or not (a or {}).get("wiring")
           for a in (cfg.get("arms") or {}).values()):
        raise SystemExit(f"{path}: every named arm needs cell and wiring")
    if isinstance(cfg.get("cells"), list) and not cfg.get("wirings"):
        raise SystemExit(f"{path}: 'wirings' is required with a list of cells")
    # Written by the campaign, never left to a runner or cell default (the
    # committee runner has neither flag: its cells are closed-form).
    for k in ("ode_unfolds", "dt_scale") if cfg["task"] != "co" else ():
        if cfg.get(k) is None:
            raise SystemExit(f"{path}: '{k}' is required")
    if cfg.get("ode_unfolds") is not None and not (
            isinstance(cfg["ode_unfolds"], int) and cfg["ode_unfolds"] >= 1):
        raise SystemExit(f"{path}: ode_unfolds must be an integer >= 1")
    if cfg.get("dt_scale") not in (None, "auto") and not (
            isinstance(cfg["dt_scale"], (int, float)) and cfg["dt_scale"] > 0):
        raise SystemExit(f"{path}: dt_scale must be auto or a positive number")
    if cfg["stage"] not in ("hpo", "final"):
        raise SystemExit(f"{path}: stage must be hpo or final")
    if cfg.get("system") and cfg["task"] != "lv":
        raise SystemExit(f"{path}: system applies to task lv only")
    if cfg["stage"] == "hpo" and len(set(cfg["seeds"])) != 2:
        raise SystemExit(f"{path}: stage hpo needs two different tuning seeds")
    if cfg["stage"] == "final" and not cfg.get("hpo_dir"):
        raise SystemExit(f"{path}: stage final needs hpo_dir")
    if cfg["stage"] == "hpo" and cfg.get("extra_seeds"):
        raise SystemExit(f"{path}: extra_seeds applies to stage final only")
    names = [a[0] for a in arms(cfg)]
    refs = cfg.get("config_from") or {}
    for key, listed in (("extra_seeds.arms", (cfg.get("extra_seeds") or {}).get("arms", [])),
                        ("heavy.arms", (cfg.get("heavy") or {}).get("arms", [])),
                        ("config_from", [*refs, *refs.values()])):
        if set(listed) - set(names):
            raise SystemExit(f"{path}: {key} names unknown arms "
                             f"{sorted(set(listed) - set(names))}")
    if set(refs) & set(refs.values()):
        raise SystemExit(f"{path}: a config_from reference has no HPO itself")
    return cfg


def arms(cfg):
    """(name, cell, wiring, runner flags) of every arm in YAML order."""
    if cfg.get("arms"):
        return [(name, a["cell"], a["wiring"],
                 {k: v for k, v in a.items() if k not in ("cell", "wiring")})
                for name, a in cfg["arms"].items()]
    cells = cfg["cells"]
    if not isinstance(cells, dict):
        cells = {c: cfg["wirings"] for c in cells}
    per_cell = cfg.get("per_cell") or {}
    return [(f"{c}_{w}", c, w, per_cell.get(c, {}))
            for c, wirings in cells.items() for w in wirings]


def bench(cfg):
    """Benchmarks-tree folder of the run (same mapping as submit_grid_hpo.sh)."""
    if cfg["task"] == "pa":
        return "03_person_activity"
    if cfg["task"] == "co":
        return "04_column_voting"
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
    PA result_tag, committee result_filename as a glob over the size;
    tests/experiments/test_suite_manifest.py checks the match)."""
    if wiring == "cncp2_multi":
        # ponytail: the multi-column suffix needs the runner's column config;
        # add it here when cncp2_multi enters the final suite.
        raise SystemExit("stage final does not support cncp2_multi")
    bb = _flag_value(argv, "--backbone-layers")
    bb = f"_bb{int(bb)}" if bb else ""
    if task == "co":
        # ponytail: the suite flags only; the exploratory July flags
        # (--train-drop, --test-corruption, ...) are not modelled.
        def v(flag, default=None):
            val = _flag_value(argv, flag)
            return default if val is None else val
        lam, dvu, clip = (v("--vote-lambda"), v("--dense-vote-units"),
                          v("--clipnorm"))
        avg = {"off": "", "mean": "_viewavg", "median": "_viewmed",
               "fault_mean": "_viewfm"}[v("--view-average", "off")]
        return (f"{cell}_{wiring}_K{v('--n-columns', 4)}_s*_"
                f"{v('--view-mode', 'identical')}"
                f"{'' if lam is None else f'_lam{float(lam):g}'}"
                f"{'_median' if v('--consensus') == 'median' else ''}"
                f"{f'_dvu{dvu}' if dvu else ''}{avg}"
                f"{'_trainclean' if v('--train-corruption') == 'off' else ''}"
                f"_lr{float(v('--lr', 1e-3)):g}"
                f"_clip{'off' if clip is None else f'{float(clip):g}'}"
                f"_cs{v('--corruption-seed', 0)}_seed{seed}{bb}.json")
    mixed = task == "lv" and _flag_value(argv, "--forcing") == "mixed"
    return f"{cell}_{wiring}_seed{seed}{bb}{'_mixed' if mixed else ''}.json"


def selection(cfg, name, cell, wiring):
    """The hpo_selection.json of arm `name`, checked against the campaign."""
    path = os.path.join(cfg["hpo_dir"], name, "hpo_selection.json")
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


def _group(cfg, name, cell, wiring):
    heavy = cfg.get("heavy") or {}
    return ("heavy" if cell in heavy.get("cells", ())
            or wiring in heavy.get("wirings", ())
            or name in heavy.get("arms", ()) else "default")


def jobs(cfg):
    """(arm, group, seed) per manifest line in submission order; seed None for
    an hpo line. Stage hpo skips the config_from arms; stage final emits the
    extra seeds after every base line."""
    refs = cfg.get("config_from") or {}
    extra = cfg.get("extra_seeds") or {}
    out = []
    for arm in arms(cfg):
        g = _group(cfg, *arm[:3])
        if cfg["stage"] == "hpo":
            out += [] if arm[0] in refs else [(arm, g, None)]
        else:
            out += [(arm, g, s) for s in cfg["seeds"]]
    if cfg["stage"] == "final":
        out += [(arm, _group(cfg, *arm[:3]), s) for arm in arms(cfg)
                if arm[0] in extra.get("arms", ()) for s in extra["seeds"]]
    return out


def count(cfg, group=None):
    """Planned fits (an hpo line: HPO_FITS of the task); no selection needed."""
    return sum(HPO_FITS[cfg["task"]] if seed is None else 1
               for _, g, seed in jobs(cfg) if group in (None, g))


def _common(cfg):
    """Flags of every runner call of the campaign (before the per-arm ones)."""
    out = []
    if cfg.get("param_budget") is not None:
        out += ["--param-budget", str(cfg["param_budget"])]
    if cfg.get("epochs") is not None:
        out += ["--epochs", str(cfg["epochs"])]
    if cfg["task"] == "co":
        return out
    out += ["--ode-unfolds", str(cfg["ode_unfolds"]),
            "--dt-scale", str(cfg["dt_scale"])]
    if cfg["task"] == "lv":
        # The frozen rollout endpoint and the system's spread, written out
        # rather than left to the runner defaults (F1-2026-10-26 c, g, r).
        from src.evaluation.rollout_metrics import ANCHOR_HORIZON, ANCHORS
        from src.tasks.lotka_volterra.datasets import _resolve
        spread = _resolve(cfg.get("system") or LV_DEFAULT_SYSTEM, None, None)[1]
        out += ["--anchors", ",".join(map(str, ANCHORS)),
                "--horizon", str(ANCHOR_HORIZON), "--ic-spread", f"{spread:g}"]
    return out


def expand(cfg, wandb=False):
    """The manifest lines in submission order (see the module docstring)."""
    task, system = cfg["task"], cfg.get("system")
    refs, common = cfg.get("config_from") or {}, _common(cfg)
    by_name = {a[0]: a for a in arms(cfg)}
    lines, missing, sels = [], [], {}
    for (name, cell, wiring, flags), group, seed in jobs(cfg):
        # the arm's own flags last: they override the campaign's
        tail = [*common, *map(str, cfg.get("extra_args") or []), *_flags(flags),
                *(["--wandb"] if wandb else [])]
        if seed is None:
            argv = ["uv", "run", "python", "experiments/run_grid_hpo.py",
                    "--task", task, "--cell", cell, "--wiring", wiring,
                    *(["--system", system] if system else []),
                    "--tuning-seeds", *map(str, cfg["seeds"]), *tail]
            lines.append({"id": f"hpo_{name}", "group": group, "argv": argv,
                          "outdir": f"hpo/{name}",
                          "done": f"hpo/{name}/hpo_selection.json"})
            continue
        ref = refs.get(name, name)
        if ref not in sels:
            sels[ref] = selection(cfg, ref, *by_name[ref][1:3])
        sel = sels[ref]
        if sel is None:
            if ref not in missing:
                missing.append(ref)
            continue
        argv = ["uv", "run", "python", RUNNERS[task], "--cell", cell,
                "--wiring", wiring, "--seed", str(seed),
                *(["--system", system] if system else []),
                "--lr", f"{sel['lr']:g}",
                *(["--clipnorm", f"{sel['clipnorm']:g}"]
                  if sel["clipnorm"] is not None else []),
                *(["--epochs", str(sel["final_epochs"])]
                  if sel.get("final_epochs") else []), *tail]
        name_ = result_name(task, cell, wiring, seed, argv)
        lines.append({"id": f"{name}_seed{seed}", "group": group,
                      "argv": argv, "outdir": ".", "done": name_})
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
    return bool(glob.glob(os.path.join(glob.escape(results), line["done"])))


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
    ap.add_argument("config", nargs="*", help="one YAML; --count: one or more")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true")
    mode.add_argument("--count", action="store_true",
                      help="planned fits (no hpo_selection.json needed)")
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
    if not a.config or (len(a.config) > 1 and not a.count):
        ap.error("one CONFIG is required (--count takes several)")
    if a.count:
        print(sum(count(load(c), a.group) for c in a.config))
        return
    cfg = load(a.config[0])
    lines = expand(cfg, wandb=a.wandb)
    if a.write:
        print(write(cfg, lines, a.write))
        return
    by_group = groups(lines)
    if a.group:
        by_group = {a.group: by_group.get(a.group, [])}
    for g, glines in by_group.items():
        for i, line in enumerate(glines):
            cmd = shlex.join([*line["argv"], "--outdir",
                              os.path.join("<results>", line["outdir"])])
            print(f"{g}[{i}]\t{line['id']}\t{cmd}")


if __name__ == "__main__":
    sys.exit(main())
