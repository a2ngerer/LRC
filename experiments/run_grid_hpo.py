"""Fixed-grid HPO for the sequence tasks (Chapter 5 protocol, issue #22).

Grid lr {1e-4, 1e-3, 3e-3, 1e-2} x gradient clipping {off, 1.0}. Stage 1 fits
every config on tuning seed s1; stage 2 fits the top 2 by the validation primary
metric on s2; the config with the better mean over both seeds wins. Edge rule
(pre-registered): a winner at lr 1e-4 or 1e-2 extends the grid by one point
(3e-5 / 3e-2), stage 1 runs for the new point only and the selection is
repeated once. No pruning, no early stopping: every fit trains the runner's
fixed epoch count. Unknown flags (e.g. --epochs 2) go to the runner unchanged.

Selection key from the runner JSON (higher wins): LV/Duffing
rollout_metrics_val.vpt_median, ties broken by the lower
rollout_metrics_val.nrmse_32_median; PA val_macro_f1. --select-on KEY
[--minimize] overrides it for diagnostics only. NaN ranks last.

Every fit is one subprocess call of the task runner (uv run python
experiments/run_*_benchmark.py) into its own folder
<outdir>/lr<lr>_clip<clip>_seed<seed>/ (the LV result name has neither lr nor
data seed). A folder that already holds its result JSON is reused, so a rerun
resumes. LV tuning fits pass --data-seed-offset (draws disjoint from the final
replicates).

<outdir>/hpo_selection.json: task, system (from the runner JSON; None for PA),
cell, wiring, the chosen lr and clipnorm, the selection keys, tuning seeds and
their data seeds, the searched grid, grid_extended, lr_at_grid_edge (the winner
lies at an edge of the final grid), ranking (the stage-2 configs best first:
lr, clipnorm, score = mean over the tuning seeds, per-seed scores) and
candidates (every fit with its metrics and result path).

Usage:
    uv run python experiments/run_grid_hpo.py --task lv --cell cfc_lrc \
        --wiring dense3 --param-budget 4000 --tuning-seeds 0 1 --outdir D
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
from functools import reduce
from glob import glob

GRID_LR = (1e-4, 1e-3, 3e-3, 1e-2)
GRID_CLIP = (None, 1.0)
EDGE_EXTENSION = {1e-4: 3e-5, 1e-2: 3e-2}
TOP_K = 2
RUNNERS = {"lv": "experiments/run_lotka_volterra_benchmark.py",
           "pa": "experiments/run_person_activity_benchmark.py"}
# task -> (primary key, higher wins), tie-break key (lower wins)
SELECT = {"lv": ("rollout_metrics_val.vpt_median",
                 "rollout_metrics_val.nrmse_32_median"),
          "pa": ("val_macro_f1", None)}


def _clip(s):
    return None if s == "off" else float(s)


def _metric(record, key, path):
    try:
        return float(reduce(lambda d, k: d[k], key.split("."), record))
    except KeyError:
        raise SystemExit(f"{path}: no selection key {key!r}") from None


class GridHPO:
    def __init__(self, args, runner_args):
        self.a, self.runner_args = args, runner_args
        self.select_on, self.tie_break = SELECT[args.task]
        if args.select_on:
            self.select_on, self.tie_break = args.select_on, None
        self.fits = {}                          # (lr, clip, seed) -> fit

    def fit(self, lr, clip, seed):
        """Run one candidate (or reuse its result JSON); return its metrics."""
        key = (lr, clip, seed)
        if key in self.fits:
            return self.fits[key]
        a = self.a
        clip_tag = "off" if clip is None else f"{clip:g}"
        d = os.path.join(a.outdir, f"lr{lr:g}_clip{clip_tag}_seed{seed}")
        if not glob(os.path.join(d, "*.json")):
            cmd = [*shlex.split(a.runner or f"uv run python {RUNNERS[a.task]}"),
                   "--cell", a.cell, "--wiring", a.wiring, "--seed", str(seed),
                   "--lr", f"{lr:g}", "--outdir", d]
            if clip is not None:
                cmd += ["--clipnorm", f"{clip:g}"]
            if a.param_budget is not None:
                cmd += ["--param-budget", str(a.param_budget)]
            if a.task == "lv":
                cmd += ["--data-seed-offset"] + (
                    ["--system", a.system] if a.system else [])
            cmd += self.runner_args
            print("+", shlex.join(cmd), flush=True)
            subprocess.run(cmd, check=True)
        (path,) = glob(os.path.join(d, "*.json"))  # one result per fit folder
        with open(path) as f:
            rec = json.load(f)
        self.fits[key] = {
            "lr": lr, "clipnorm": clip, "seed": seed,
            "data_seed": rec.get("data_seed", rec.get("split_seed")),
            "system": rec.get("system"),
            "score": _metric(rec, self.select_on, path),
            "tie_break": (_metric(rec, self.tie_break, path)
                          if self.tie_break else None),
            "result": os.path.relpath(path, a.outdir)}
        return self.fits[key]

    def summary(self, lr, clip, seeds):
        """Mean score / tie-break of one config over the given seeds."""
        fits = [self.fit(lr, clip, s) for s in seeds]

        def mean(k):
            return (None if fits[0][k] is None
                    else sum(f[k] for f in fits) / len(fits))
        return {"lr": lr, "clipnorm": clip, "score": mean("score"),
                "tie_break": mean("tie_break"),
                "scores": {str(f["seed"]): f["score"] for f in fits}}

    def _rank(self, rows):
        # best first; NaN ranks last; stable, so exact ties keep grid order
        sign = 1 if self.a.minimize else -1

        def key(r):
            s, t = r["score"], r["tie_break"]
            return (math.inf if math.isnan(s) else sign * s,
                    math.inf if t is None or math.isnan(t) else t)
        return sorted(rows, key=key)

    def select(self, grid):
        """Stage 1 on s1 over the grid, stage 2 (top 2) ranked by the mean
        over s1 and s2."""
        s1, s2 = self.a.tuning_seeds
        stage1 = self._rank([self.summary(lr, c, [s1])
                             for lr in grid for c in self.a.grid_clip])
        return self._rank([self.summary(r["lr"], r["clipnorm"], [s1, s2])
                           for r in stage1[:TOP_K]])

    def run(self):
        grid = sorted(self.a.grid_lr)
        ranking = self.select(grid)
        best = ranking[0]["lr"]
        extended = best in EDGE_EXTENSION and best in (grid[0], grid[-1])
        if extended:                             # edge rule, once
            grid = sorted(grid + [EDGE_EXTENSION[best]])
            ranking = self.select(grid)
        a, fits = self.a, list(self.fits.values())
        return {
            "task": a.task, "system": fits[0]["system"], "cell": a.cell,
            "wiring": a.wiring, "param_budget": a.param_budget,
            "lr": ranking[0]["lr"], "clipnorm": ranking[0]["clipnorm"],
            "select_on": self.select_on, "maximize": not a.minimize,
            "tie_break": self.tie_break,
            "tuning_seeds": a.tuning_seeds,
            "data_seeds": [next(f["data_seed"] for f in fits if f["seed"] == s)
                           for s in a.tuning_seeds],
            "grid_lr": grid, "grid_clip": a.grid_clip,
            "grid_extended": extended,
            "lr_at_grid_edge": ranking[0]["lr"] in (grid[0], grid[-1]),
            "ranking": ranking, "candidates": fits,
            "runner_args": self.runner_args,
        }


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False, description=__doc__.split(
        "\n\n")[0], epilog="Unknown flags are forwarded to the task runner.")
    ap.add_argument("--task", required=True, choices=sorted(RUNNERS))
    ap.add_argument("--cell", required=True)
    ap.add_argument("--wiring", required=True)
    ap.add_argument("--system", default=None,
                    help="lv only: neural_ode system (runner default if unset)")
    ap.add_argument("--param-budget", type=int, default=None)
    ap.add_argument("--tuning-seeds", type=int, nargs=2, required=True,
                    metavar=("S1", "S2"))
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--grid-lr", type=float, nargs="+", default=list(GRID_LR))
    ap.add_argument("--grid-clip", type=_clip, nargs="+",
                    default=list(GRID_CLIP), help="clipnorm values, 'off' = none")
    ap.add_argument("--select-on", default=None,
                    help="diagnostics only: dotted runner-JSON key to select on")
    ap.add_argument("--minimize", action="store_true",
                    help="with --select-on: lower wins")
    ap.add_argument("--runner", default=None,
                    help="runner command (default: uv run python "
                         "experiments/run_<task>_benchmark.py; tests: a fake)")
    args, runner_args = ap.parse_known_args(argv)
    if args.system and args.task != "lv":
        ap.error("--system applies to --task lv only")
    if args.tuning_seeds[0] == args.tuning_seeds[1]:
        ap.error("--tuning-seeds needs two different seeds")
    if args.minimize and not args.select_on:
        ap.error("--minimize needs --select-on")

    os.makedirs(args.outdir, exist_ok=True)
    sel = GridHPO(args, runner_args).run()
    path = os.path.join(args.outdir, "hpo_selection.json")
    with open(path, "w") as f:
        json.dump(sel, f, indent=2)
    best = sel["ranking"][0]
    print(f"chosen lr {best['lr']:g} clipnorm {best['clipnorm']} "
          f"({sel['select_on']} mean {best['score']:g}); grid_extended "
          f"{sel['grid_extended']}, lr_at_grid_edge {sel['lr_at_grid_edge']}"
          f" -> {path}")


if __name__ == "__main__":
    main()
