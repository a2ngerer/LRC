"""Fixed-grid HPO for the sequence tasks (Chapter 5 protocol, issue #22).

Grid lr {1e-4, 1e-3, 3e-3, 1e-2} x gradient clipping {off, 1.0}. Stage 1 fits
every config on tuning seed s1; stage 2 fits the top 2 by the validation primary
metric on s2; the config with the better mean over both seeds wins. Edge rule
(pre-registered): a winner at lr 1e-4 or 1e-2 extends the grid by one point
(3e-5 / 3e-2), stage 1 runs for the new point only and the selection is
repeated once. No pruning, no early stopping: every fit trains the runner's
fixed epoch count. Unknown flags (e.g. --epochs 2) go to the runner unchanged.

Selection key from the runner JSON (higher wins): LV/Duffing
rollout_metrics_val.vpt_multi_median (multi-anchor VPT, #62), ties broken by
the lower rollout_metrics_val.nrmse_32_anchor_8, then grid order; PA
val_macro_f1 (grid order on ties); committee (co) val_macro_f1_mean, the mean
validation macro-F1 over the fault grid. --select-on KEY [--minimize] overrides
the primary key for diagnostics only; the task's tie-break stays. NaN ranks
last.

Consensus study (co, protocol freeze F1-2026-10-26, issue #55): learning rate
only, one fixed --grid-clip value. Epoch guard: if best_val_epoch >= 36 in more
than half of the fits, hpo_selection.json sets epochs_guard_triggered and
final_epochs 60 (else final_epochs = the fits' epochs). Transfer K=3 -> K=30:
    run_grid_hpo.py --transfer-guard K3/hpo_selection.json K30_SEED0.json
exits 1 when the K=30 fit's m=0 validation macro-F1 (at its best epoch) lies
more than 5 pp below the K=3 selection's (mean over its tuning seeds); the
fallback is a three-point lr grid at K=30 with one tuning seed
(--tuning-seeds 100 runs stage 1 only).

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

Person Activity chance guard (protocol freeze F1-2026-10-26, issue #80):
every PA candidate carries one_class_macro_f1, the macro-F1 of always
predicting the majority class of the validation split (from the loader's class
counts on the split and split_seed of its result JSON, --data-path as passed to
the runner), and at_chance (its val_macro_f1 <= that value + 0.01). When every
stage-1 candidate is at chance, the arm never left the one-class plateau: the
selection sets not_learned true and names the plateau in reason; there is no
edge extension and no stage 2, and lr / clipnorm are the stage-1 ranking's
first entry (exact ties in grid order). Else not_learned is false, reason null.

Usage:
    uv run python experiments/run_grid_hpo.py --task lv --cell cfc_lrc \
        --wiring dense3 --param-budget 4000 --tuning-seeds 100 101 --outdir D
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
from functools import lru_cache, reduce
from glob import glob

from src.evaluation.metrics import CHANCE_MARGIN, at_chance, one_class_macro_f1

GRID_LR = (1e-4, 1e-3, 3e-3, 1e-2)
GRID_CLIP = (None, 1.0)
EDGE_EXTENSION = {1e-4: 3e-5, 1e-2: 3e-2}
TOP_K = 2
FINAL_SEEDS = range(10)                 # the final replicates; tuning stays out
RUNNERS = {"lv": "experiments/run_lotka_volterra_benchmark.py",
           "pa": "experiments/run_person_activity_benchmark.py",
           "co": "experiments/run_committee_benchmark.py"}
# task -> (primary key, higher wins), tie-break key (lower wins)
SELECT = {"lv": ("rollout_metrics_val.vpt_multi_median",
                 "rollout_metrics_val.nrmse_32_anchor_8"),
          "pa": ("val_macro_f1", None),
          "co": ("val_macro_f1_mean", None)}
# co epoch guard: best_val_epoch >= LATE in more than half the fits -> finals
# run FINAL_EPOCHS (of the default 40)
EPOCHS_GUARD_LATE, EPOCHS_GUARD_FINAL = 36, 60
TRANSFER_GUARD = 0.05            # co K=3 -> K=30: m=0 val macro-F1 tolerance


def _clip(s):
    return None if s == "off" else float(s)


def _metric(record, key, path):
    try:
        return float(reduce(lambda d, k: d[k], key.split("."), record))
    except KeyError:
        raise SystemExit(f"{path}: no selection key {key!r}") from None


@lru_cache(maxsize=None)
def pa_one_class_val(split, split_seed, data_path=None):
    """One-class macro-F1 of the Person Activity validation split (#80)."""
    import numpy as np
    from src.tasks.person_activity.datasets import load_person_activity
    data = load_person_activity(data_path=data_path, split=split,
                                split_seed=split_seed)
    return one_class_macro_f1(np.bincount(data.val_y.ravel(),
                                          minlength=data.num_classes))


class GridHPO:
    def __init__(self, args, runner_args):
        self.a, self.runner_args = args, runner_args
        self.select_on, self.tie_break = SELECT[args.task]
        if args.select_on:
            self.select_on = args.select_on
        self.fits = {}                          # (lr, clip, seed) -> fit
        dp = argparse.ArgumentParser(add_help=False)
        dp.add_argument("--data-path", default=None)
        self.data_path = dp.parse_known_args(runner_args)[0].data_path

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
            "epochs": rec.get("epochs"),
            "best_val_epoch": rec.get("best_val_epoch"),
            "score": _metric(rec, self.select_on, path),
            "tie_break": (_metric(rec, self.tie_break, path)
                          if self.tie_break else None),
            "result": os.path.relpath(path, a.outdir)}
        if a.task == "pa":                      # chance guard (#80)
            one = pa_one_class_val(rec.get("split"), rec.get("split_seed"),
                                   self.data_path)
            self.fits[key].update(one_class_macro_f1=one, at_chance=at_chance(
                rec["val_macro_f1"], one))
        return self.fits[key]

    def plateau(self):
        """PA: every fit so far at chance (checked after stage 1)."""
        return self.a.task == "pa" and all(f["at_chance"]
                                           for f in self.fits.values())

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
        over s1 and s2; a single tuning seed or a PA plateau (every stage-1
        fit at chance) ranks stage 1 only."""
        s1, *s2 = self.a.tuning_seeds
        stage1 = self._rank([self.summary(lr, c, [s1])
                             for lr in grid for c in self.a.grid_clip])
        if not s2 or self.plateau():
            return stage1
        return self._rank([self.summary(r["lr"], r["clipnorm"], [s1, *s2])
                           for r in stage1[:TOP_K]])

    def run(self):
        grid = sorted(self.a.grid_lr)
        ranking = self.select(grid)
        best, plateau = ranking[0]["lr"], self.plateau()
        extended = (not plateau and best in EDGE_EXTENSION
                    and best in (grid[0], grid[-1]))
        if extended:                             # edge rule, once
            grid = sorted(grid + [EDGE_EXTENSION[best]])
            ranking = self.select(grid)
        a, fits = self.a, list(self.fits.values())
        guard = {}
        if a.task == "co":
            late = sum(f["best_val_epoch"] >= EPOCHS_GUARD_LATE for f in fits)
            hit = late > len(fits) / 2
            guard = {"epochs_guard_triggered": hit,
                     "final_epochs": (EPOCHS_GUARD_FINAL if hit
                                      else fits[0]["epochs"])}
        elif a.task == "pa":
            one = fits[0]["one_class_macro_f1"]
            guard = {"not_learned": plateau, "reason": (
                "one-class plateau: every stage-1 candidate at chance "
                f"(val_macro_f1 <= one-class {one:.4f} + {CHANCE_MARGIN:g}); "
                "no edge extension, no stage 2" if plateau else None)}
        return {
            "task": a.task, "system": fits[0]["system"], "cell": a.cell,
            "wiring": a.wiring, "param_budget": a.param_budget,
            "lr": ranking[0]["lr"], "clipnorm": ranking[0]["clipnorm"],
            "select_on": self.select_on, "maximize": not a.minimize,
            "tie_break": self.tie_break,
            "tuning_seeds": a.tuning_seeds,
            "data_seeds": [next((f["data_seed"] for f in fits
                                 if f["seed"] == s), None)  # None: not fit
                           for s in a.tuning_seeds],
            "grid_lr": grid, "grid_clip": a.grid_clip,
            "grid_extended": extended,
            "lr_at_grid_edge": ranking[0]["lr"] in (grid[0], grid[-1]),
            "ranking": ranking, "candidates": fits,
            "runner_args": self.runner_args,
            **guard,
        }


def _val_m0(path):
    """m = 0 validation macro-F1 of a committee result at its best epoch."""
    with open(path) as f:
        rec = json.load(f)
    return rec["val_macro_f1_grid_curve"][rec["best_val_epoch"] - 1][
        rec["fault_counts"].index(0)]


def transfer_guard(selection_path, k30_path, tol=TRANSFER_GUARD):
    """co K=3 -> K=30: the K=30 fit with the K=3 configuration passes when
    its m=0 validation macro-F1 is at most tol below the K=3 value (mean over
    the selected configuration's tuning fits)."""
    with open(selection_path) as f:
        sel = json.load(f)
    best = sel["ranking"][0]
    k3 = [_val_m0(os.path.join(os.path.dirname(selection_path), c["result"]))
          for c in sel["candidates"]
          if c["lr"] == best["lr"] and c["clipnorm"] == best["clipnorm"]]
    k3, k30 = sum(k3) / len(k3), _val_m0(k30_path)
    return {"k3_val_macro_f1_m0": k3, "k30_val_macro_f1_m0": k30,
            "tolerance": tol, "passed": k30 >= k3 - tol}


def main(argv=None):
    guard = argparse.ArgumentParser(add_help=False)
    guard.add_argument("--transfer-guard", nargs=2,
                       metavar=("K3_SELECTION", "K30_RESULT"))
    g, _ = guard.parse_known_args(argv)
    if g.transfer_guard:
        res = transfer_guard(*g.transfer_guard)
        print(json.dumps(res))
        return 0 if res["passed"] else 1

    ap = argparse.ArgumentParser(allow_abbrev=False, description=__doc__.split(
        "\n\n")[0], epilog="Unknown flags are forwarded to the task runner.")
    ap.add_argument("--task", required=True, choices=sorted(RUNNERS))
    ap.add_argument("--cell", required=True)
    ap.add_argument("--wiring", required=True)
    ap.add_argument("--system", default=None,
                    help="lv only: neural_ode system (runner default if unset)")
    ap.add_argument("--param-budget", type=int, default=None)
    ap.add_argument("--tuning-seeds", type=int, nargs="+", required=True,
                    help="S1 [S2]: stage 1 on S1, top 2 confirmed on S2; one "
                         "seed ranks stage 1 only")
    ap.add_argument("--allow-seed-overlap", action="store_true",
                    help="allow tuning seeds in the final seed range 0..9")
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
    if len(args.tuning_seeds) > 2 or len(set(args.tuning_seeds)) < len(
            args.tuning_seeds):
        ap.error("--tuning-seeds takes one seed or two different seeds")
    if args.task == "co" and len(args.grid_clip) != 1:
        ap.error("--task co tunes the learning rate only: pass one fixed "
                 "--grid-clip value")
    overlap = [s for s in args.tuning_seeds if s in FINAL_SEEDS]
    if overlap and not args.allow_seed_overlap:
        ap.error(f"--tuning-seeds {overlap} lie in the final seed range 0..9 "
                 "(e.g. use 100 101) or pass --allow-seed-overlap")
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
          + (f"; NOT LEARNED: {sel['reason']}" if sel.get("not_learned")
             else "") + f" -> {path}")


if __name__ == "__main__":
    raise SystemExit(main())
