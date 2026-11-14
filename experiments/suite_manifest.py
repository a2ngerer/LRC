"""Final-suite manifest: one campaign YAML -> one JSONL line per SLURM task (#52).

Campaign YAML (configs/suite/*.yaml; the frozen protocol F1-2026-10-26 in
configs/suite/{lv,duffing,mt,pa,co}_*.yaml, #71):

    task: lv                    # lv | pa | co (committee, consensus study)
    system: duffing             # lv only; the runner default if unset
    stage: hpo                  # hpo | final | eval (see below)
    cells: [cfc_lrc, gru]       # cells x wirings, or a mapping cell -> its
    wirings: [dense3, ncp]      # wirings ({gru: [dense, ncp], mlp: [dense]})
    arms:                       # or named arms (co): name -> cell, wiring and
      ncp_lam0: {cell: cfc_lrc, wiring: ncp, vote_lambda: 0}   # runner flags
    param_budget: 4000          # optional
    seeds: [0, 1]               # hpo: the two tuning seeds (one: stage 1 only,
                                # the S0 smoke); final: the replicates
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
    fixed: {lr: 0.001, clipnorm: null}   # final only, instead of hpo_dir: one
                                # lr and clipnorm (null: off) for every arm, no
                                # hpo_selection.json (the S1 pilots)
    selection_from:             # final only, with hpo_dir (#105): an arm with
      cfc_lrc_eps1_ncp: {arm: cfc_lrc_ncp, cell: cfc_lrc, wiring: ncp}
                                # a new cell or wiring and no HPO of its own
                                # reads <hpo_dir>/<arm>/hpo_selection.json of a
                                # reference arm that is not in the YAML (an F1
                                # HPO run); the selection is checked against
                                # the reference's cell and wiring
    cluster: {runs_per_gpu: 4, mem: 32G, walltime: "12:00:00", cpus: 12,
              array_throttle: 8}                    # optional, see below
    heavy:                      # optional: arms with one of these cells,
      cells: [lrc]              # wirings or arm names run as their own array
      wirings: [cncp]           # job with cluster resources merged over
      arms: [ctrnn_ncp]         # `cluster`
      cluster: {runs_per_gpu: 1, mem: 48G, walltime: "48:00:00"}

Stage eval (#105) trains nothing: it re-evaluates stored result JSONs.

    task: lv                    # lv | pa
    system: duffing             # lv: the system of every source
    stage: eval
    sources:                    # the records of these final-stage YAMLs
      - config: configs/suite/lv_final.yaml
        records_dir: results/01_lotka_volterra/<run>/results
        arms: [cfc_lrc_ncp]     # optional: default every arm of the source
        seeds: [0, 1]           # optional: default every seed of the arm
    robustness: true            # optional (the control evals, #99): lv gets a
    seed_streams: v1            # second line per record, see below; the rollout
                                # draw seeds (#101) are written out, no default
    extra_args: []              # pa only: appended to every eval_robustness.py
                                # call (#96: --per-recording-confusions,
                                # --verify-against <stored CSVs>)
    cluster: {runs_per_gpu: 10}

One line per record, named with result_name from the source YAML: lv runs
run_lotka_volterra_benchmark.py --eval-only <record> --anchors 8,16,24,32
--horizon 96 (the updated JSON goes to results/<source stem>/), pa runs
eval_robustness.py --results <record> --out results/<source stem>/<record
stem>.csv (the rebuild check; no --outdir; the line carries that path as
`out`). id and done are <source stem>/<record>, so equal record names of two
sources never collide; the records_dir folders are read-only inputs (a cluster
path, read at run time only). --count gives the line count.

With `robustness: true` (and the required `seed_streams`, v1 for the records of
F1/F2) the robustness evaluation runs on the same records: an lv source gets a
second line per record, eval_robustness.py --results <record> --seed-streams v1
--out results/<source stem>/<record stem>.csv (its rebuild_rel_dev is the
rebuild fidelity, its metric_robust the robustness row); a pa line is that
call already and only gains --seed-streams (ignored on Person Activity). The
CSVs go to `aggregate_results.py --contrasts --robustness`.

Arms are ordered cell-major in YAML order and named <cell>_<wiring> (or by
their `arms` key). Stage hpo gives one line per arm: one
experiments/run_grid_hpo.py call into results/hpo/<arm>/. Stage final gives one
line per (arm, seed): one task-runner fit into results/ (flat, as
experiments/aggregate_results.py reads it) with the lr and clipnorm of
<hpo_dir>/<arm or its config_from reference>/hpo_selection.json (co: also its
final_epochs, the epoch guard of #55), or of `fixed`.

A line is {"id", "group", "argv", "outdir", "done"}; outdir and done are
relative to the run folder's results/. A line whose results/<done> exists (the
fit's result JSON, or the arm's hpo_selection.json; co: a glob, the runner's
name carries the budget-derived size) is done and skipped, so a requeue or a
re-submit after a walltime kill only runs what is missing. Result names carry
neither the optimizer nor the epoch count, so expand() raises when two lines of
one campaign resolve to the same `done` (#138); arms that differ only there go
into campaigns with their own run folders. A committee arm takes
`corruption_seed: seed` (--corruption-seed seed): the draws follow the line's
seed and the name carries `_cs<seed>`.

--count prints the planned fits (an hpo line counts HPO_FITS of its task; with
one tuning seed its stage-1 grid, --grid-lr x --grid-clip, default GRID_LR x
GRID_CLIP of run_grid_hpo.py; an edge extension is not counted) of one or more
YAMLs and needs no hpo_selection.json; the frozen fit counts per task:
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

from experiments.run_grid_hpo import GRID_CLIP, GRID_LR

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
        "dt_scale", "extra_args", "per_cell", "hpo_dir", "fixed", "cluster",
        "heavy", "description", "selection_from", "sources", "robustness",
        "seed_streams"}
EVAL_KEYS = {"task", "system", "stage", "sources", "cluster", "description",
             "robustness", "seed_streams", "extra_args"}
# eval_robustness.SEED_STREAMS (#101), not imported: that module loads
# TensorFlow; a test pins the equality
SEED_STREAMS = ("v1", "v2")
ROBUSTNESS_RUNNER = "experiments/eval_robustness.py"
EVAL_RUNNERS = {"lv": RUNNERS["lv"], "pa": ROBUSTNESS_RUNNER}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLUSTER_KEYS = ("runs_per_gpu", "mem", "walltime", "cpus", "array_throttle")


def _positive(v):
    # a YAML 1e-3 without a dot is a string (PyYAML): write 0.001 or 1.0e-3
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0


def load(path):
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    unknown = set(cfg) - KEYS
    if unknown:
        raise SystemExit(f"{path}: unknown keys {sorted(unknown)}")
    if cfg.get("stage") == "eval":
        return _load_eval(path, cfg)
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
        raise SystemExit(f"{path}: stage must be hpo, final or eval")
    if cfg.get("sources"):
        raise SystemExit(f"{path}: sources applies to stage eval only")
    if cfg.get("system") and cfg["task"] != "lv":
        raise SystemExit(f"{path}: system applies to task lv only")
    if cfg["stage"] == "hpo" and (len(cfg["seeds"]) > 2 or len(
            set(cfg["seeds"])) != len(cfg["seeds"])):
        raise SystemExit(f"{path}: stage hpo takes one tuning seed or two "
                         "different tuning seeds")
    if cfg["stage"] == "final" and bool(cfg.get("hpo_dir")) == bool(
            cfg.get("fixed")):
        raise SystemExit(f"{path}: stage final needs exactly one of hpo_dir "
                         "and fixed")
    if cfg["stage"] == "hpo" and cfg.get("fixed"):
        raise SystemExit(f"{path}: fixed applies to stage final only")
    fixed = cfg.get("fixed")
    if fixed and not (isinstance(fixed, dict)
                      and set(fixed) == {"lr", "clipnorm"}
                      and _positive(fixed["lr"])
                      and (fixed["clipnorm"] is None
                           or _positive(fixed["clipnorm"]))):
        raise SystemExit(f"{path}: fixed must be {{lr: <number > 0>, "
                         "clipnorm: <number > 0 or null>}")
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
    sel_from = cfg.get("selection_from") or {}
    if sel_from:
        if cfg["stage"] != "final" or cfg.get("fixed"):
            raise SystemExit(f"{path}: selection_from needs stage final with "
                             "hpo_dir, not fixed")
        if set(sel_from) - set(names):
            raise SystemExit(f"{path}: selection_from names unknown arms "
                             f"{sorted(set(sel_from) - set(names))}")
        if set(sel_from) & set(refs):
            raise SystemExit(f"{path}: an arm takes config_from or "
                             "selection_from, not both")
        if any(not isinstance(r, dict) or set(r) != {"arm", "cell", "wiring"}
               for r in sel_from.values()):
            raise SystemExit(f"{path}: selection_from maps an arm to "
                             "{arm, cell, wiring}")
    return cfg


def _source_path(config):
    return config if os.path.isabs(config) else os.path.join(ROOT, config)


def _load_eval(path, cfg):
    """stage eval: task (+ system for a non-default lv system) and sources;
    every source a final-stage YAML of the same task and system."""
    extra = set(cfg) - EVAL_KEYS
    if extra:
        raise SystemExit(f"{path}: stage eval does not take {sorted(extra)}")
    if cfg.get("task") not in EVAL_RUNNERS:
        raise SystemExit(f"{path}: stage eval needs task lv or pa")
    if cfg.get("system") and cfg["task"] != "lv":
        raise SystemExit(f"{path}: system applies to task lv only")
    if cfg.get("extra_args") and cfg["task"] != "pa":
        raise SystemExit(f"{path}: extra_args applies to task pa only")
    if not cfg.get("sources"):
        raise SystemExit(f"{path}: 'sources' is required")
    if cfg.get("robustness") not in (None, True, False):
        raise SystemExit(f"{path}: robustness must be true or false")
    if cfg.get("robustness") and cfg.get("seed_streams") not in SEED_STREAMS:
        raise SystemExit(f"{path}: robustness needs seed_streams, one of "
                         f"{', '.join(SEED_STREAMS)} (written out, no default)")
    if cfg.get("seed_streams") and not cfg.get("robustness"):
        raise SystemExit(f"{path}: seed_streams applies with robustness: true")
    for src in cfg["sources"]:
        if not isinstance(src, dict) or not {"config", "records_dir"} <= set(
                src) <= {"config", "records_dir", "arms", "seeds"}:
            raise SystemExit(f"{path}: a source is {{config, records_dir, "
                             "arms (optional), seeds (optional)}")
        scfg = load(_source_path(src["config"]))
        if scfg["stage"] != "final" or (scfg["task"], scfg.get("system")) != (
                cfg["task"], cfg.get("system")):
            raise SystemExit(f"{path}: source {src['config']} is not a final "
                             "stage of the same task and system")
        have = {(a[0], s) for a, _, s in jobs(scfg)}
        unknown = set(src.get("arms") or []) - {n for n, _ in have}
        if unknown:
            raise SystemExit(f"{path}: source {src['config']} has no arms "
                             f"{sorted(unknown)}")
        names = src.get("arms") or sorted({n for n, _ in have})
        want = {(n, s) for n in names
                for s in src.get("seeds") or [t for m, t in have if m == n]}
        if not want or want - have:
            raise SystemExit(f"{path}: source {src['config']} has no fit "
                             f"{sorted(want - have) or names}")
    return cfg


def _eval_jobs(scfg, src):
    """(arm, seed) of the source records an eval source names, in the
    source's job order (arms and seeds filter it)."""
    names, seeds = src.get("arms"), src.get("seeds")
    return [(arm, s) for arm, _, s in jobs(scfg)
            if (names is None or arm[0] in names)
            and (seeds is None or s in seeds)]


def expand_eval(cfg):
    """The lines of a stage eval YAML (see the module docstring)."""
    from src.evaluation.rollout_metrics import ANCHOR_HORIZON, ANCHORS
    task, lines = cfg["task"], []
    for src in cfg["sources"]:
        scfg = load(_source_path(src["config"]))
        stem = os.path.splitext(os.path.basename(src["config"]))[0]
        common = [*_common(scfg), *map(str, scfg.get("extra_args") or [])]
        for (_, cell, wiring, flags), seed in _eval_jobs(scfg, src):
            rec = result_name(task, cell, wiring, seed,
                              [*common, *_flags(flags)])
            record = os.path.join(src["records_dir"], rec)
            argv = ["uv", "run", "python", EVAL_RUNNERS[task]]
            csv = f"{stem}/{os.path.splitext(rec)[0]}.csv"
            line = {"id": f"{stem}/{rec}", "group": "default", "outdir": stem}
            streams = (["--seed-streams", cfg["seed_streams"]]
                       if cfg.get("robustness") else [])
            if task == "lv":
                argv += ["--eval-only", record, "--anchors",
                         ",".join(map(str, ANCHORS)),
                         "--horizon", str(ANCHOR_HORIZON)]
                line["done"] = f"{stem}/{rec}"
            else:
                argv += ["--results", record, *streams,
                         *map(str, cfg.get("extra_args") or [])]
                line["done"] = line["out"] = csv
            lines.append({**line, "argv": argv})
            if task == "lv" and cfg.get("robustness"):
                lines.append({"id": csv, "group": "default", "outdir": stem,
                              "done": csv, "out": csv, "argv": [
                                  "uv", "run", "python", ROBUSTNESS_RUNNER,
                                  "--results", record, *streams]})
    dup = len(lines) - len({ln["done"] for ln in lines})
    if dup:
        raise SystemExit(f"stage eval: {dup} records are listed twice")
    return lines


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
        cs = v("--corruption-seed", 0)      # the word `seed`: the line's seed
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
                f"_cs{seed if cs == 'seed' else cs}_seed{seed}{bb}.json")
    mixed = task == "lv" and _flag_value(argv, "--forcing") == "mixed"
    fold = _flag_value(argv, "--split-fold")      # PA fold rotation (#134)
    fold = "" if fold is None else f"_fold{int(fold)}"
    return (f"{cell}_{wiring}_seed{seed}{bb}{'_mixed' if mixed else ''}"
            f"{fold}.json")


def selection(cfg, name, cell, wiring):
    """The hpo_selection.json of arm `name`, checked against the campaign;
    `fixed` instead when the campaign sets it."""
    if cfg.get("fixed"):
        return cfg["fixed"]
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


def grid(cfg, flags):
    """(lrs, clips) of the stage-1 grid of one hpo line: --grid-lr and
    --grid-clip of the campaign or the arm, else run_grid_hpo.py's GRID_LR x
    GRID_CLIP (clips as given: 'off' or a number)."""
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--grid-lr", type=float, nargs="+", default=list(GRID_LR))
    ap.add_argument("--grid-clip", nargs="+", default=list(GRID_CLIP))
    g, _ = ap.parse_known_args(
        [*map(str, cfg.get("extra_args") or []), *_flags(flags)])
    return g.grid_lr, g.grid_clip


def _hpo_fits(cfg, flags):
    """Planned fits of one hpo line: HPO_FITS of the task; one tuning seed runs
    stage 1 only (run_grid_hpo.py), one fit per grid() point."""
    if len(cfg["seeds"]) == 2:
        return HPO_FITS[cfg["task"]]
    lrs, clips = grid(cfg, flags)
    return len(lrs) * len(clips)


def count(cfg, group=None):
    """Planned fits (an hpo line: _hpo_fits); no selection needed. Stage
    eval: its lines (evaluations, not fits)."""
    if cfg["stage"] == "eval":
        return len(expand_eval(cfg)) if group in (None, "default") else 0
    return sum(_hpo_fits(cfg, arm[3]) if seed is None else 1
               for arm, g, seed in jobs(cfg) if group in (None, g))


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
    if cfg["stage"] == "eval":
        return expand_eval(cfg)
    task, system = cfg["task"], cfg.get("system")
    refs, common = cfg.get("config_from") or {}, _common(cfg)
    sel_from = cfg.get("selection_from") or {}
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
        if name in sel_from:
            r = sel_from[name]
            ref, ref_arm = r["arm"], (r["cell"], r["wiring"])
        else:
            ref = refs.get(name, name)
            ref_arm = tuple(by_name[ref][1:3])
        if (ref, ref_arm) not in sels:
            sels[ref, ref_arm] = selection(cfg, ref, *ref_arm)
        sel = sels[ref, ref_arm]
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
    _reject_collisions(lines)
    return lines


def _reject_collisions(lines):
    """Result names carry neither the optimizer nor the epoch count, so two
    lines of one campaign that differ only there write (or skip as done) the
    same file in the run folder: refuse, naming both lines."""
    seen = {}
    for line in lines:
        first = seen.setdefault(line["done"], line)
        if first is not line:
            raise SystemExit(
                f"{first['id']} and {line['id']} both resolve to the result "
                f"{line['done']} in one run folder (result names carry no "
                "optimizer or epoch count); split the arms into campaigns "
                "with their own run folders")


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


def _out_flag(line, results):
    """--outdir <results>/<outdir>; a pa eval line: --out <results>/<out>."""
    if "out" in line:
        return ["--out", os.path.join(results, line["out"])]
    return ["--outdir", os.path.join(results, line["outdir"])]


def run(manifest, index, results):
    """One SLURM bundle member: skip a done line, else exec its command."""
    with open(manifest) as f:
        line = json.loads(f.readlines()[index])
    if is_done(line, results):
        print(f"[skip] {line['id']}: {line['done']} exists", flush=True)
        return
    argv = [*line["argv"], *map(os.path.normpath, _out_flag(line, results))]
    if "out" in line:       # eval_robustness.py does not create the folder
        os.makedirs(os.path.dirname(argv[-1]), exist_ok=True)
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
            cmd = shlex.join([*line["argv"], *_out_flag(line, "<results>")])
            print(f"{g}[{i}]\t{line['id']}\t{cmd}")


if __name__ == "__main__":
    sys.exit(main())
