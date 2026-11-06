# experiments/aggregate_results.py
"""Aggregate benchmark run JSONs into thesis-ready tables and statistics.

Reads the per-run JSONs produced by experiments/run_benchmark.py and writes:
  - results/summary.csv     — one row per run (long format)
  - results/summary.md      — mean +/- std tables and statistical tests

Statistics (per thesis protocol, Chapter 5):
  - Wilcoxon signed-rank tests, alpha = 0.05, paired over (system, seed), on the
    finite pairs only (the excluded count is printed); Bonferroni over the
    pre-registered family size (BONFERRONI_FAMILIES) or, for exploratory
    blocks, over the block's tests
  - paired Cohen's d_z (mean(diff) / std(diff)) on the same finite pairs
  - P(A>B) over all pairs (diverged runs lose) with an exact Clopper-Pearson
    interval, an effect size next to the Wilcoxon, not a decision rule

Optional sections:
  --hpo-dir DIR    HPO fragility per arm x task from every hpo_selection.json
                   (lr at the grid edge, grid extended, sign of each core
                   contrast under the second-best configuration of both arms)
  --pa-runs DIR    Person Activity arm means with a percentile bootstrap over
                   seeds (--bootstrap-units seed) or over seeds and test
                   recordings (--bootstrap-units seed,recording)
  --rollout-runs DIR  rollout outcome-class shares per arm (test split) with
                   exact Clopper-Pearson intervals, at tau_orb and at the 0.30
                   sensitivity, plus one non-finite label per arm
  --co-runs DIR    H-d consensus-study family (committee runner JSONs, #55):
                   one-sided paired Wilcoxon over seeds at K=30, Bonferroni
                   over the family of four, plus the attribution flag
  --floors         with --rollout-runs: persistence and AR(1) floor lines
                   per task, split and data seed on the multi-anchor endpoint,
                   the floor marking per arm and the floor_bound flag per core
                   contrast, written to <out>/floors.json (#67)
  --contrasts      with --rollout-runs and/or --pa-runs (#57): the runner
                   JSONs of both tasks (load_sequence_runs), every entry of
                   CORE_CONTRASTS as a paired block over the final seeds
                   (core_contrast_blocks: exact Wilcoxon without ties or
                   zeros, Bonferroni within the family, n_pairs < 8
                   undecidable, d_z, P(A>B), median difference with a 10,000
                   resample bootstrap interval, crossed over seeds x test
                   recordings on Person Activity, divergence rates, the
                   floor_bound flag of --floors or --floors-json and, with
                   --hpo-dir, the tuning-seed sign stability; on Person
                   Activity the chance marking of #80), the H-c
                   conjunction and every other arm pair descriptively;
                   <out>/contrasts.csv and <out>/sequence_runs.csv
  --robustness CSV  with --contrasts: the eval-only robustness rows of
                   experiments/eval_robustness.py (#34); their metric_robust
                   (diverged_robust as the divergence flag) runs the H-c
                   contrasts once more as the second family, the other
                   robustness endpoints get a descriptive per-arm table
  --f2-families SPEC.yaml  the seven tests of the four F2 families (protocol
                   F2 sections 2.0 to 2.4, F2_FAMILIES): per arm the label,
                   the expected protocol_version and an explicit list of run
                   folders or record files (template:
                   configs/aggregate/f2_families.yaml); paired Wilcoxon,
                   Bonferroni within the family and F2-wide (sensitivity),
                   exact sign-flip permutation p, floor marking per arm
                   wherever it lives, reading PASS/FAIL/floor-bound/
                   undecidable; <out>/f2_families.csv

Usage:
    uv run python experiments/aggregate_results.py [--runs results/runs] [--out results]
        [--hpo-dir DIR ...] [--pa-runs DIR [--bootstrap-units seed,recording]]
        [--rollout-runs DIR [--floors]] [--co-runs DIR]
        [--contrasts [--floors-json PATH] [--robustness CSV ...]]
        [--f2-families SPEC.yaml]

    # the frozen suite: ROLLOUTS holds the final run folders of every rollout
    # system (searched recursively), PA the Person Activity final run
    uv run python experiments/aggregate_results.py --runs none --out OUT \
        --rollout-runs ROLLOUTS --pa-runs PA --floors --contrasts \
        --hpo-dir HPO_LV HPO_DUFFING HPO_MT HPO_PA
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from glob import escape, glob, has_magic
from itertools import combinations

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata, wilcoxon

# Allow running as a plain script (``src`` is the installed package).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.metrics import (CHANCE_MARGIN, at_chance,  # noqa: E402
                                    one_class_macro_f1)
from src.evaluation.rollout_metrics import (SENSITIVITY_KEY,  # noqa: E402
                                            TAU_ORB_SENSITIVITY, floor_lines)
from src.evaluation.stats import (bootstrap_mean_ci,  # noqa: E402
                                  bootstrap_median_ci, clopper_pearson,
                                  crossed_median_diff_ci, p_a_gt_b)

ALPHA = 0.05
METRIC = 'nrmse'   # primary comparison metric (proposal: MSE/NRMSE)
# Metric orientation for P(A>B) and the worst value: 'harness' = load_runs
# (errors), 'lv' = multi-anchor VPT, 'pa' = test macro-F1 (load_sequence_runs),
# 'ppo' = eval_return_mean (load_ppo_runs).
HIGHER_IS_BETTER = {'harness': False, 'lv': True, 'pa': True, 'ppo': True}
FINAL_SEEDS = range(10)   # final replicates, as in run_grid_hpo.py; tuning seeds lie outside
GAMMA = 0.75       # P(A>B) reference line, printed for orientation only
MIN_PAIRS = 8      # fewer complete seed pairs: the contrast is undecidable
N_BOOT = 10_000    # bootstrap resamples of the median paired difference

# Pre-registered core contrasts (protocol freeze F1-2026-10-26, decisions f,
# g, n and section 6): family -> [(task, arm A, arm B)], arm = (cell,
# wiring), A the arm the hypothesis predicts to be better, task = rollout
# system name or 'pa'.
CORE_CONTRASTS = {
    'H-a': [('multitimescale_lib', ('cfc', 'ncp'), ('cfc_static', 'ncp')),
            ('duffing', ('cfc', 'ncp'), ('cfc_static', 'ncp')),
            ('pa', ('cfc', 'dense'), ('cfc_static', 'dense'))],
    'H-b': [('duffing', ('cfc_lrc', 'ncp'), ('lrc', 'ncp')),
            ('pa', ('cfc_lrc', 'ncp'), ('lrc', 'ncp'))],
    'H-c': [(task, ('cfc_lrc', wiring), ('cfc_lrc', 'dense3'))
            for task in ('periodic_predator_prey', 'duffing', 'pa')
            for wiring in ('ncp', 'cncp')],
    'RQ1': [(task, ('cfc_lrc', 'ncp'), ('gru', 'ncp'))
            for task in ('periodic_predator_prey', 'duffing', 'pa')],
}
# The H-c contrasts once more on the eval-only robustness endpoint (#34), a
# second family; evaluated on the column 'metric_robust' when it is loaded.
ROBUSTNESS_FAMILY = 'H-c robustness'
# The other robustness endpoints, summarised per arm in the report only.
ROBUST_DESCRIPTIVE = ('metric_robust', 'f1_sigma_0', 'f1_sigma_0.1',
                      'ood_nrmse_128_median', 'ood_rate_unbounded',
                      'long_vpt_multi_median', 'long_nrmse_128_median',
                      'long_rate_unbounded')
# Bonferroni family sizes, the one place they are defined.
BONFERRONI_FAMILIES = {
    'H-a': 3,              # multitimescale_lib/ncp, duffing/ncp, pa/dense
    'H-b': 2,              # duffing, pa
    'H-c': 2,              # per task: ncp and cncp vs dense3 (conjunction over tasks)
    ROBUSTNESS_FAMILY: 2,  # per task, as H-c
    'RQ1': 3,              # cfc_lrc/ncp vs gru/ncp per task
    # {ncp, cncp} x {part i, part ii} at K=30 (consensus_family below)
    'H-d': 4,
}

# H-d (protocol freeze F1-2026-10-26, #55), one-sided (consensus better), at
# K=30: part i learned consensus vs input averaging, test macro-F1 at m=15;
# part ii learned vs lambda=0 on the normalised per-position macro-F1 AUC,
# per seed the mean over m in {0, 15}. Attribution flag (no family test): the
# part-i effect counts as consensus only if learned lambda also beats
# lambda=0 on the test macro-F1 at m=15 (one-sided, alpha unadjusted).
HD_K, HD_M, HD_AUC_M = 30, 15, (0, 15)
HD_WIRINGS = ('ncp', 'cncp')


def load_runs(runs_dirs) -> pd.DataFrame:
    """Load run JSONs from one or more directories into a long DataFrame.

    Runs get a variant suffix on the cell label so every downstream
    groupby/statistic treats each experimental condition as its own variant:
      '+clip'        active gradient clip            (v2)
      '+unfolds<n>'  non-default ODE solver substeps (v3 solver-fidelity arm)
      '+bt<n>'       non-default training horizon     (v3 training-horizon arm)
      '+<regime>'    v5 generalization stressor       (noise/extrapolation/ood_init)
    v1/v2 labels ('<cell>', '<cell>+clip') are unchanged, since their configs
    carry batch_time=16 and no ode_unfolds. v5-vs-clean is paired by loading both
    results/runs_v4 and results/runs_v5 (e.g. 'ltc' vs 'ltc+noise').
    """
    if isinstance(runs_dirs, str):
        runs_dirs = [runs_dirs]
    rows = []
    for runs_dir in runs_dirs:
        for path in sorted(glob(os.path.join(runs_dir, '*.json'))):
            with open(path, encoding='utf-8') as f:
                r = json.load(f)
            cfg = r.get('config', {})
            clip = float(cfg.get('clip_norm', 0.0))
            unfolds = cfg.get('ode_unfolds')
            batch_time = cfg.get('batch_time', 16)
            stress = cfg.get('stress') or r['run'].get('stress')   # v5 stress regime
            # v6a wiring graph + v6b dose-response level (seed 42 / None -> no suffix).
            wseed = cfg.get('ncp_wiring_seed') or r['run'].get('ncp_wiring_seed')
            stress_level = next(
                (v for v in (cfg.get('stress_level'),
                             r['run'].get('stress_noise_level'),
                             r['run'].get('stress_train_fraction'),
                             r['run'].get('stress_ood_scale')) if v is not None),
                None)
            suffix = ''
            if stress:
                suffix += f'+{stress}'
            if wseed and int(wseed) != 42:
                suffix += f'+w{int(wseed)}'
            if stress_level is not None:
                suffix += f'+lvl{stress_level}'
            if clip:
                suffix += '+clip'
            if unfolds and int(unfolds) != 6:
                suffix += f'+unfolds{int(unfolds)}'
            if batch_time and int(batch_time) != 16:
                suffix += f'+bt{int(batch_time)}'
            cell = r['run']['cell'] + suffix
            rows.append({
                'cell': cell,
                'wiring': r['run']['wiring'],
                'system': r['run']['system'],
                'seed': r['run']['seed'],
                'clip_norm': clip,
                'stress': stress,
                'ncp_wiring_seed': int(wseed) if wseed else None,
                'stress_level': stress_level,
                'ode_unfolds': int(unfolds) if unfolds else None,
                'batch_time': int(batch_time) if batch_time else None,
                'final_loss': r['training']['final_loss'],
                'mse': r['evaluation']['mse'],
                'nrmse': r['evaluation']['nrmse'],
                'duration_s': r['training']['duration_s'],
                # budget sizing outside tolerance (#59); absent in older runs
                'budget_above_tolerance': bool(r.get('budget_above_tolerance')),
                'file': os.path.basename(path),
            })
    return pd.DataFrame(rows)


def summary_tables(df: pd.DataFrame) -> str:
    lines = ['## Mean +/- std over seeds (NRMSE, full-trajectory rollout)\n']
    pivot = df.groupby(['system', 'cell', 'wiring'])[METRIC].agg(['mean', 'std', 'count'])
    for system in sorted(df['system'].unique()):
        lines.append(f'### {system}\n')
        lines.append('| Cell | Dense | NCP |')
        lines.append('|------|-------|-----|')
        for cell in sorted(df['cell'].unique()):
            row = [f'| {cell}']
            for wiring in ['dense', 'ncp']:
                try:
                    m = pivot.loc[(system, cell, wiring)]
                    row.append(f"{m['mean']:.4f} +/- {m['std']:.4f} (n={int(m['count'])})")
                except KeyError:
                    row.append('—')
            lines.append(' | '.join(row) + ' |')
        lines.append('')

    lines.append('## Aggregated over all systems (mean NRMSE)\n')
    lines.append('| Cell | Dense | NCP |')
    lines.append('|------|-------|-----|')
    agg = df.groupby(['cell', 'wiring'])[METRIC].mean()
    for cell in sorted(df['cell'].unique()):
        row = [f'| {cell}']
        for wiring in ['dense', 'ncp']:
            row.append(f'{agg.get((cell, wiring), float("nan")):.4f}')
        lines.append(' | '.join(row) + ' |')
    lines.append('')
    return '\n'.join(lines)


def _paired_vectors(df: pd.DataFrame, key_a: dict, key_b: dict):
    """Metric vectors and divergence flags for two configs, paired on (system,
    seed). A run is diverged if its metric is non-finite or a loader set the
    optional 'diverged' column (e.g. the rollout class 'unbounded')."""
    df = df.assign(_div=~np.isfinite(df[METRIC]) | df.get('diverged', False))
    a = df.loc[(df[list(key_a)] == pd.Series(key_a)).all(axis=1)]
    b = df.loc[(df[list(key_b)] == pd.Series(key_b)).all(axis=1)]
    merged = a.merge(b, on=['system', 'seed'], suffixes=('_a', '_b'))
    return (merged[f'{METRIC}_a'].to_numpy(), merged[f'{METRIC}_b'].to_numpy(),
            merged['_div_a'].to_numpy(bool), merged['_div_b'].to_numpy(bool))


def _above_budget(df: pd.DataFrame, *keys) -> bool:
    """True if any run of the given configs carries budget_above_tolerance."""
    if 'budget_above_tolerance' not in df:
        return False
    return any(df.loc[(df[list(k)] == pd.Series(k)).all(axis=1),
                      'budget_above_tolerance'].any() for k in keys)


def wilcoxon_test(x, y, alternative='two-sided'):
    """(p, method) of the paired Wilcoxon signed-rank test: the exact null
    distribution without ties or zero differences, else scipy's normal
    approximation (zeros dropped); (None, None) if every difference is zero."""
    d = np.asarray(x, float) - np.asarray(y, float)
    nz = d[d != 0]
    if nz.size == 0:
        return None, None
    method = ('exact' if nz.size == d.size
              and np.unique(np.abs(nz)).size == nz.size else 'approx')
    return float(wilcoxon(x, y, alternative=alternative,
                          method=method).pvalue), method


def _cohens_d_paired(x: np.ndarray, y: np.ndarray) -> float:
    diff = x - y
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else float('nan')


def _test_block(df, pairs, title, family=None) -> str:
    """One comparison family; Bonferroni over the pre-registered family size
    (BONFERRONI_FAMILIES[family]) or, without a family, over the block's tests.

    Wilcoxon and d_z use the finite pairs only; P(A>B) uses every pair. In a
    pre-registered (confirmatory) family, a contrast with an arm sized outside
    the budget tolerance (budget_above_tolerance, #59) is excluded."""
    n_tests = BONFERRONI_FAMILIES[family] if family else len(pairs)
    alpha_adj = ALPHA / n_tests
    higher = HIGHER_IS_BETTER['harness']
    sign = 1 if higher else -1
    lines = [f'## {title} (Wilcoxon signed-rank, alpha={ALPHA}, metric={METRIC}, '
             f'Bonferroni m={n_tests})\n']
    lines.append('| Comparison | n pairs | excluded | median diff | p raw '
                 '| p Bonferroni | alpha adj | significant | Cohen\'s d_z '
                 '| P(A>B) [95% CI] |')
    lines.append('|------------|---------|----------|-------------|-------'
                 '|--------------|-----------|-------------|--------------'
                 '|-----------------|')
    for label, key_a, key_b in pairs:
        if family and _above_budget(df, key_a, key_b):
            lines.append(f'| {label} | — | — | — | — | — | {alpha_adj:.4g} '
                         f'| excluded: arm above budget tolerance | — | — |')
            continue
        x, y, dx, dy = _paired_vectors(df, key_a, key_b)
        if len(x) < 5:
            lines.append(f'| {label} | {len(x)} | — | — | — | — | {alpha_adj:.4g} '
                         f'| too few pairs | — | — |')
            continue
        pw, lo, hi = p_a_gt_b(sign * x, sign * y, dx, dy)
        win = f'{pw:.2f} [{lo:.3f}, {hi:.3f}]'
        finite = np.isfinite(x) & np.isfinite(y)
        excl = int((~finite).sum())
        x, y = x[finite], y[finite]
        if len(x) < 5:
            lines.append(f'| {label} | {len(x) + excl} | {excl} | — | — | — '
                         f'| {alpha_adj:.4g} | too few finite pairs | — | {win} |')
            continue
        p, _ = wilcoxon_test(x, y)
        if p is None:        # all differences zero
            lines.append(f'| {label} | {len(x) + excl} | {excl} | 0 | 1.0 | 1.0 '
                         f'| {alpha_adj:.4g} | no | 0 | {win} |')
            continue
        p_adj = min(1.0, p * n_tests)
        d = _cohens_d_paired(x, y)
        sig = 'YES' if p_adj < ALPHA else 'no'
        lines.append(f'| {label} | {len(x) + excl} | {excl} | {np.median(x - y):+.4f} '
                     f'| {p:.4g} | {p_adj:.4g} | {alpha_adj:.4g} | {sig} | {d:+.3f} '
                     f'| {win} |')
    lines.append('')
    lines.append(f'P(A>B): paired win fraction of A ({"higher" if higher else "lower"} '
                 f'{METRIC} wins, ties 0.5, a diverged run loses, two diverged runs tie), '
                 f'exact Clopper-Pearson 95% CI; reference gamma = {GAMMA} for '
                 f'orientation only (the decision is the Wilcoxon).\n')
    return '\n'.join(lines)


def statistics(df: pd.DataFrame) -> str:
    cells = sorted(df['cell'].unique())
    out = []

    # Cell vs cell, within each wiring
    for wiring in ['dense', 'ncp']:
        pairs = [
            (f'{a} vs {b} ({wiring})',
             {'cell': a, 'wiring': wiring}, {'cell': b, 'wiring': wiring})
            for a, b in combinations(cells, 2)
        ]
        out.append(_test_block(df, pairs, f'Cell comparisons — {wiring} wiring (RQ1)'))

    # Dense vs NCP, within each cell
    pairs = [
        (f'{c}: dense vs ncp', {'cell': c, 'wiring': 'dense'}, {'cell': c, 'wiring': 'ncp'})
        for c in cells
    ]
    out.append(_test_block(df, pairs, 'Wiring effect per cell (RQ3)'))
    return '\n'.join(out)


def load_hpo_selections(hpo_dirs) -> dict:
    """{(task, cell, wiring): selection} from every hpo_selection.json below
    hpo_dirs (one folder or a list, e.g. the HPO runs of several tasks).

    Keys read: task, system (rollouts; used as the task label), cell, wiring,
    lr_at_grid_edge, grid_extended, and ranking = the stage-2 configs best
    first, each with its lr, its validation selection 'score' and its
    per-tuning-seed 'scores' (sign_stable_tuning).
    """
    sel = {}
    paths = [path for d in ([hpo_dirs] if isinstance(hpo_dirs, str) else hpo_dirs)
             for path in sorted(glob(os.path.join(d, '**', 'hpo_selection.json'),
                                     recursive=True))]
    for path in paths:
        with open(path, encoding='utf-8') as f:
            r = json.load(f)
        sel[(r.get('system') or r['task'], r['cell'], r['wiring'])] = r
    return sel


def hpo_fragility(hpo_dirs) -> str:
    """Grid-edge flags per arm x task and, per core contrast, whether the sign
    of the validation-score difference A - B under the best configurations
    holds under the second-best configuration of both arms."""
    sel = load_hpo_selections(hpo_dirs)
    lines = ['## HPO fragility (hpo_selection.json)\n',
             '| Task | Arm | lr | lr_at_grid_edge | grid_extended |',
             '|------|-----|----|-----------------|---------------|']
    for (task, cell, wiring), r in sorted(sel.items()):
        lines.append(f"| {task} | {cell}/{wiring} | {r['ranking'][0]['lr']:g} "
                     f"| {r['lr_at_grid_edge']} | {r['grid_extended']} |")
    lines += ['', '| Family | Task | A | B | sign best | sign 2nd | sign_stable_2nd |',
              '|--------|------|---|---|-----------|----------|-----------------|']
    for family, contrasts in CORE_CONTRASTS.items():
        for task, a, b in contrasts:
            ra, rb = sel.get((task, *a)), sel.get((task, *b))
            if ra is None or rb is None:
                continue
            s1, s2 = (np.sign(ra['ranking'][i]['score'] - rb['ranking'][i]['score'])
                      for i in (0, 1))
            lines.append(f"| {family} | {task} | {'/'.join(a)} | {'/'.join(b)} "
                         f"| {s1:+.0f} | {s2:+.0f} | {'yes' if s1 == s2 else 'NO'} |")
    lines.append('')
    return '\n'.join(lines)


def pa_arm_means(pa_dir, units='seed') -> str:
    """Person Activity arm means (test macro-F1) with a percentile bootstrap
    over seeds, or over seeds and test recordings ('seed,recording', reads
    test_macro_f1_per_recording). Files without the key are skipped."""
    by_rec = 'recording' in units
    key = 'test_macro_f1_per_recording' if by_rec else 'test_macro_f1'
    arms, skipped = defaultdict(list), 0
    for path in sorted(glob(os.path.join(pa_dir, '*.json'))):
        with open(path, encoding='utf-8') as f:
            r = json.load(f)
        if key not in r:
            skipped += 1
            continue
        v = r[key]
        # arm = result tag without the seed (keeps backbone/multi-column suffixes)
        arm = os.path.basename(path)[:-len('.json')].replace(f"_seed{r['seed']}", '')
        arms[arm].append([v[k] for k in sorted(v)] if isinstance(v, dict) else v)
    lines = [f'## Person Activity arm means (test macro-F1, bootstrap over {units})\n',
             '| Arm | n seeds | mean | 95% CI |', '|-----|---------|------|--------|']
    for arm, rows in sorted(arms.items()):
        mean, lo, hi = bootstrap_mean_ci(rows, resample_cols=by_rec)
        lines.append(f'| {arm} | {len(rows)} | {mean:.4f} | [{lo:.4f}, {hi:.4f}] |')
    lines.append(f'\nSkipped {skipped} result file(s) without {key}.\n' if skipped else '')
    return '\n'.join(lines)


OUTCOME_CLASSES = ('ok', 'bounded_wrong', 'unbounded')


def _rollout_arm(path, r, rollout_dir):
    """(arm, folder, tag) of a rollout fit: tag = file name without
    _seed<N>, folder relative to rollout_dir, arm = '<folder>/<tag>' ('.'
    omitted)."""
    tag = os.path.basename(path)[:-len('.json')].replace(f"_seed{r['seed']}", '')
    rel = os.path.relpath(os.path.dirname(path), rollout_dir)
    return (tag if rel == '.' else f'{rel}/{tag}'), rel, tag


def rollout_outcome_classes(rollout_dir) -> str:
    """Outcome-class shares per rollout arm on the test split, pooled over the
    trajectories of all fits, with exact Clopper-Pearson 95 % intervals; one
    row at the stored tau_orb and one at the 0.30 sensitivity. Non-finite is a
    fit property: one label per arm with the count of non-finite fits.

    An arm is (system, folder relative to rollout_dir, file name without
    _seed<N>, tau_orb): the file name carries neither lr, clipnorm nor the
    control knobs (ode_unfolds, dt_scale), so fits from different run or HPO
    folders are never pooled. Fits on a seed outside FINAL_SEEDS (HPO tuning
    fits, e.g. hpo/<arm>/lr<lr>_clip<clip>_seed100/) are left out, as are
    files without the #63 keys (older schema, hpo_selection.json)."""
    arms, skipped, tuning = defaultdict(list), 0, 0
    for path in sorted(glob(os.path.join(rollout_dir, '**', '*.json'),
                            recursive=True)):
        with open(path, encoding='utf-8') as f:
            r = json.load(f)
        m = r.get('rollout_metrics_test') if isinstance(r, dict) else None
        if (not m or SENSITIVITY_KEY not in m or r.get('seed') is None
                or r.get('system') is None):
            skipped += 1
            continue
        if r['seed'] not in FINAL_SEEDS:
            tuning += 1
            continue
        arms[(r['system'], _rollout_arm(path, r, rollout_dir)[0],
              m['tau_orb'])].append(m)
    lines = ['## Rollout outcome classes (test trajectories, share [95% '
             'Clopper-Pearson CI])\n',
             '| System | Arm | fits | trajectories | tau_orb | '
             + ' | '.join(OUTCOME_CLASSES) + ' | non-finite fits |',
             '|--------|-----|------|--------------|---------|'
             + '----|' * len(OUTCOME_CLASSES) + '-----------------|']
    for (system, arm, tau), ms in sorted(arms.items()):
        n = [m['n_trajectories'] for m in ms]
        n_nf = sum(bool(m['non_finite']) for m in ms)
        lo, hi = clopper_pearson(n_nf, len(ms))
        nf = (f'at least 1 of {len(ms)} fits non-finite ({n_nf}/{len(ms)} '
              f'[{lo:.2f}, {hi:.2f}])' if n_nf else '—')
        for t, rates in ((tau, ms), (TAU_ORB_SENSITIVITY,
                                     [m[SENSITIVITY_KEY] for m in ms])):
            shares = []
            for c in OUTCOME_CLASSES:
                k = sum(round(rt[f'rate_{c}'] * ni) for rt, ni in zip(rates, n))
                lo, hi = clopper_pearson(k, sum(n))
                shares.append(f'{k / sum(n):.2f} [{lo:.2f}, {hi:.2f}]')
            lines.append(f'| {system} | {arm} | {len(ms)} | {sum(n)} | {t:g} | '
                         + ' | '.join(shares) + f' | {nf} |')
    lines.append(f'\nSkipped {skipped} file(s) without outcome classes.\n'
                 if skipped else '')
    lines.append(f'\nLeft out {tuning} tuning-seed fit(s) (seed outside '
                 f'{FINAL_SEEDS.start}..{FINAL_SEEDS.stop - 1}).\n' if tuning else '')
    return '\n'.join(lines)


# The data draw of a rollout fit (load_lotka_volterra arguments, as in the
# runner's --eval-only) plus the multi-anchor settings its endpoint used.
FLOOR_DATA_KEYS = ('system', 'data_seed', 'n_trajectories', 'seq_len', 'split',
                   't_span', 'ic_spread')
FLOOR_SPLIT = 'test'   # the split of the floor marking (F1 S5 floors check)


def not_learned(n_below, n_fits) -> bool:
    """Seed rule (protocol freeze F1-2026-10-26, decision f, with the
    erratum F1 section 10 (15)): an arm counts as not learned if a fit lies
    below the floor in more than a fifth of its fits, i.e. in at least 2 of
    5 or at least 3 of 10 seeds."""
    return n_below * 5 > n_fits


def fit_floor(r, draws):
    """(test vpt_multi_median, test floor of its own data draw) of a rollout
    fit record, the per-fit part of the floor marking (below: not vpt >=
    floor, so NaN counts as below); the draw's floor lines are computed on
    first use into draws ({key: draw + floor_lines}). None for a record
    without the #62 multi-anchor keys or config.effective."""
    from src.tasks.lotka_volterra.datasets import load_lotka_volterra

    m = r.get('rollout_metrics_test') if isinstance(r, dict) else None
    eff = (r.get('config') or {}).get('effective') if m else None
    if not eff or 'vpt_multi_median' not in m:
        return None
    draw = {k: eff.get(k) for k in FLOOR_DATA_KEYS}
    draw.update(anchors=m['anchors'], anchor_horizon=m['anchor_horizon'],
                anchor_theta=m['anchor_theta'])
    key = json.dumps(draw, sort_keys=True)
    if key not in draws:
        data = load_lotka_volterra(
            n_trajectories=draw['n_trajectories'], seq_len=draw['seq_len'],
            split=draw['split'], seed=draw['data_seed'],
            system=draw['system'], t_span=draw['t_span'],
            spread=draw['ic_spread'])
        draws[key] = {**draw, **floor_lines(
            data, draw['anchors'], draw['anchor_horizon'],
            draw['anchor_theta'])}
    return m['vpt_multi_median'], draws[key][FLOOR_SPLIT]['floor']


def compute_floors(rollout_dir) -> dict:
    """floors.json content for the rollout fits below rollout_dir.

    floors: per data draw (FLOOR_DATA_KEYS + anchors, horizon, theta of the
    fits' rollout_metrics_test) the persistence and AR(1) anchor_metrics on
    val and test (src/evaluation/rollout_metrics.floor_lines), the draws of
    tuning seeds included (the S0 smoke fits run on the tuning seed, #72).
    arms: per final-seed (system, arm) the fits whose test vpt_multi_median lies below the floor of
    their own data draw (NaN counts as below) and above_floor by the seed rule
    (not_learned). contrasts: every rollout entry of CORE_CONTRASTS whose two
    arms lie in the same folder with the same file-name suffix (forcing,
    backbone), floor_bound if either arm is not above floor. Files without
    the #62 multi-anchor keys or config.effective are left out."""
    fits, draws = [], {}
    for path in sorted(glob(os.path.join(rollout_dir, '**', '*.json'),
                            recursive=True)):
        with open(path, encoding='utf-8') as f:
            r = json.load(f)
        fl = fit_floor(r, draws)
        if fl is None:
            continue
        if r.get('seed') not in FINAL_SEEDS:
            continue                # a tuning draw: floor line, no marking
        vpt, floor = fl
        arm, folder, tag = _rollout_arm(path, r, rollout_dir)
        fits.append({'system': r['system'], 'arm': arm, 'folder': folder,
                     'cell': r['cell'], 'wiring': r['wiring'],
                     'suffix': tag[len(f"{r['cell']}_{r['wiring']}"):],
                     'seed': r['seed'],
                     'below': not vpt >= floor})
    arms = {}
    for fit in fits:
        a = arms.setdefault((fit['system'], fit['arm']), {
            **{k: fit[k] for k in ('system', 'arm', 'folder', 'cell', 'wiring',
                                   'suffix')},
            'split': FLOOR_SPLIT, 'seeds': [], 'seeds_below': []})
        a['seeds'].append(fit['seed'])
        if fit['below']:
            a['seeds_below'].append(fit['seed'])
    for a in arms.values():
        a['above_floor'] = not not_learned(len(a['seeds_below']),
                                           len(a['seeds']))
    by_key = {(a['system'], a['folder'], a['suffix'], a['cell'],
               a['wiring']): a for a in arms.values()}
    contrasts = []
    for family, entries in CORE_CONTRASTS.items():
        for task, ca, cb in entries:
            for (system, folder, suffix, *cw), a in sorted(by_key.items()):
                b = by_key.get((system, folder, suffix, *cb))
                if system != task or tuple(cw) != ca or b is None:
                    continue
                contrasts.append({
                    'family': family, 'task': task, 'a': list(ca),
                    'b': list(cb), 'arm_a': a['arm'], 'arm_b': b['arm'],
                    'floor_bound': not (a['above_floor']
                                        and b['above_floor'])})
    return {'split': FLOOR_SPLIT, 'floors': list(draws.values()),
            'arms': sorted(arms.values(),
                           key=lambda a: (a['system'], a['arm'])),
            'contrasts': contrasts}


def floor_report(floors) -> str:
    """Markdown of compute_floors: floors per data draw on the marking split,
    the arm marking and the floor_bound contrasts."""
    sp = floors['split']
    lines = [f'## Floor lines (multi-anchor VPT median, {sp} split)\n',
             '| System | data seed | persistence | AR(1) | floor |',
             '|--------|-----------|-------------|-------|-------|']
    for d in sorted(floors['floors'],
                    key=lambda d: (d['system'], d['data_seed'])):
        lines.append(f"| {d['system']} | {d['data_seed']} "
                     f"| {d[sp]['persistence']['vpt_multi_median']:g} "
                     f"| {d[sp]['ar1']['vpt_multi_median']:g} "
                     f"| {d[sp]['floor']:g} |")
    lines += ['', '| System | Arm | fits | seeds below floor | above floor |',
              '|--------|-----|------|-------------------|-------------|']
    for a in floors['arms']:
        lines.append(f"| {a['system']} | {a['arm']} | {len(a['seeds'])} "
                     f"| {a['seeds_below'] or '—'} "
                     f"| {'yes' if a['above_floor'] else 'NO (not learned)'} |")
    lines += ['', '| Family | Task | A | B | floor_bound |',
              '|--------|------|---|---|-------------|']
    for c in floors['contrasts']:
        lines.append(f"| {c['family']} | {c['task']} | {c['arm_a']} "
                     f"| {c['arm_b']} | {'YES' if c['floor_bound'] else 'no'} |")
    lines.append('\nSeed rule: not learned if below max(persistence, AR(1)) '
                 'in more than one of five seeds (two of ten).\n')
    return '\n'.join(lines)


# The sequence-runner suite (#57): one row per fit of
# experiments/run_lotka_volterra_benchmark.py ('lv', every rollout system) or
# experiments/run_person_activity_benchmark.py ('pa').
WORST = {'lv': 0.0, 'pa': 0.0}   # VPT 0, macro-F1 0: a diverged fit's value


def _record_task(r):
    """'lv' | 'pa' | 'co' | None: the runner that wrote a result record."""
    if 'rollout_metrics_test' in r:
        return 'lv'
    if 'n_columns' in r:
        return 'co'
    return 'pa' if 'test_macro_f1' in r else None


# Epochs of the F1 finals per task (protocol F2 section 3.2, point 1), keyed
# by config.effective.system, Person Activity by its config.effective.task.
F1_EPOCHS = {'periodic_predator_prey': 300, 'duffing': 450,
             'multitimescale_lib': 450, 'person_activity': 50}


def _control_tag(r):
    """The descriptive control arms of the protocol freeze F1-2026-10-26
    (decisions a, d, g, h): ode_unfolds 1, dt_scale 1, a fixed width instead
    of the budget (cfc_static_w; also the width-matched dense3 of protocol
    F2 section 2.2, #85), and teacher forcing on the rollouts. On a record
    stamped with an F2 protocol_version also epochs=N when its epochs differ
    from the F1 value of its task (the doubled-budget arms, protocol F2
    section 3.2, point 7); F1 records never get it (e.g. pilots)."""
    eff = (r.get('config') or {}).get('effective') or {}
    tags = []
    if eff.get('ode_unfolds') == 1:
        tags.append('k=1')
    if r.get('dt_scale') == 1:
        tags.append('dt_scale=1')
    if r.get('param_budget') is None:
        tags.append('fixed width')
    if 'rollout_metrics_test' in r and r.get('forcing') != 'mixed':
        tags.append(f"forcing={r.get('forcing')}")
    if str(eff.get('protocol_version') or '').startswith('F2-'):
        f1 = F1_EPOCHS.get('person_activity' if eff.get('task')
                           == 'person_activity' else eff.get('system'))
        if f1 is not None and eff.get('epochs') != f1:
            tags.append(f"epochs={eff.get('epochs')}")
    return ', '.join(tags)


def arm_label(r):
    """'cell/wiring' of a runner record, a control arm with its
    _control_tag in brackets (the arm column of load_sequence_runs and of
    experiments/eval_robustness.py)."""
    tag = _control_tag(r)
    return f"{r['cell']}/{r['wiring']}" + (f' [{tag}]' if tag else '')


def _sequence_row(path, r, task) -> dict:
    """The load_sequence_runs row of one runner record of the task."""
    if task == 'lv':
        m = r['rollout_metrics_test']
        if 'vpt_multi_median' not in m:
            raise SystemExit(f'{path}: no rollout_metrics_test.'
                             'vpt_multi_median (pre-#62 runner JSON)')
        metric, system = m['vpt_multi_median'], r['system']
        diverged = m.get('rate_unbounded') == 1 or bool(
            m.get('non_finite'))
        extra = {k: v for k, v in m.items() if k.startswith(
            ('vpt_anchor_', 'nrmse_32_anchor_'))}
        extra.update({f'vpt_theta_{th}': v for th, v in
                      (m.get('theta_sensitivity') or {}).items()})
        extra['rate_unbounded'] = m.get('rate_unbounded')
    else:
        metric, system, diverged = r['test_macro_f1'], 'pa', False
        cms = r.get('test_confusion_by_recording')
        extra = {'confusion': (np.array([cms[k] for k in sorted(cms)],
                                        float) if cms else None)}
    metric = float('nan') if metric is None else float(metric)
    return {
        'cell': r['cell'], 'wiring': r['wiring'], 'task': task,
        'system': system, 'seed': r['seed'],
        'stage': 'final' if r['seed'] in FINAL_SEEDS else 'hpo',
        'arm': arm_label(r), 'control': bool(_control_tag(r)),
        'metric': metric,
        'diverged': bool(diverged or not np.isfinite(metric)),
        **{k: r.get(k) for k in ('best_val_epoch', 'epochs',
                                 'params_effective',
                                 'budget_deviation', 'lr',
                                 'clipnorm')},
        'budget_above_tolerance': bool(r.get('budget_above_tolerance')),
        **extra, 'file': path}


def load_sequence_runs(runs_dirs, task) -> pd.DataFrame:
    """One row per fit of the task's runner JSONs below runs_dirs (searched
    recursively; other runners' JSONs and non-fit files are skipped).

    Columns: cell, wiring, task, system ('pa' on Person Activity), seed,
    stage ('final' for seeds in FINAL_SEEDS, else 'hpo'), arm ('cell/wiring',
    a control arm with its tag in brackets, so it never pairs as a
    confirmatory arm), control, metric (lv: rollout_metrics_test.
    vpt_multi_median, pa: test_macro_f1), diverged (lv:
    rate_unbounded == 1 or non_finite; any task: non-finite metric),
    best_val_epoch, epochs, params_effective, budget_deviation,
    budget_above_tolerance, lr, clipnorm, file; lv also the secondary
    columns vpt_anchor_<a>, nrmse_32_anchor_<a>, vpt_theta_<theta> and
    rate_unbounded; pa the test confusion matrices (recordings sorted) for
    the crossed bootstrap. An old single-step-harness JSON, a pre-#62 rollout
    JSON and two final fits of one arm and seed raise SystemExit."""
    rows = []
    for d in [runs_dirs] if isinstance(runs_dirs, str) else runs_dirs:
        for path in sorted(glob(os.path.join(d, '**', '*.json'),
                                recursive=True)):
            with open(path, encoding='utf-8') as f:
                r = json.load(f)
            if not isinstance(r, dict):
                continue
            if 'run' in r and 'evaluation' in r:
                raise SystemExit(
                    f'{path}: old single-step harness record (run/evaluation, '
                    'NRMSE); load_sequence_runs reads the Lotka-Volterra and '
                    'Person Activity runner JSONs, load_runs the old harness')
            if _record_task(r) != task:
                continue
            rows.append(_sequence_row(path, r, task))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    final = df[df['stage'] == 'final']
    dup = final[final.duplicated(['system', 'arm', 'seed'], keep=False)]
    if not dup.empty:
        raise SystemExit('two final fits of one arm and seed (load the run '
                         'folders of one protocol, or give the control its '
                         'own run): ' + ', '.join(dup['file']))
    return df


def merge_robustness(df, csv_paths) -> pd.DataFrame:
    """df (load_sequence_runs) with the columns of the eval-only robustness
    CSVs of experiments/eval_robustness.py (#34): metric_robust,
    diverged_robust (the divergence flag of that evaluation, which replaces
    the clean one in the robustness family), rebuild_rel_dev and the
    descriptive f1_sigma_* / ood_* / long_* columns, matched on (system,
    arm, seed) of the final fits; a fit without a row stays NaN and is left
    out of the robustness family. A CSV without diverged_robust, two rows of
    one final fit and a row whose config_effective_sha256 is not that of the
    loaded fit's config.effective raise SystemExit."""
    from src.benchmark.checkpoints import config_hash   # imports TensorFlow
    rob = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
    if 'diverged_robust' not in rob:
        raise SystemExit('robustness CSV without diverged_robust: rerun '
                         'experiments/eval_robustness.py')
    key = ['system', 'arm', 'seed']
    cols = ['metric_robust', 'diverged_robust', 'config_effective_sha256',
            *(c for c in rob if c.startswith(('f1_sigma_', 'ood_', 'long_',
                                              'rebuild_rel_dev')))]
    rob = rob.loc[rob['seed'].isin(FINAL_SEEDS), key + cols]
    dup = rob[rob.duplicated(key, keep=False)]
    if not dup.empty:
        raise SystemExit('two robustness rows of one final fit: '
                         + ', '.join(map(str, dup[key].values.tolist())))
    df = df.merge(rob, on=key, how='left')
    for path, sha in df.loc[df['diverged_robust'].notna(),
                            ['file', 'config_effective_sha256']].values:
        with open(path, encoding='utf-8') as f:
            if config_hash(json.load(f)['config']['effective']) != sha:
                raise SystemExit(f'{path}: its robustness row has another '
                                 'config_effective_sha256 (a CSV of another '
                                 'run folder?)')
    return df.drop(columns='config_effective_sha256')


def pa_chance_marking(df) -> dict:
    """{arm: {'one_class', 'seeds_at_chance', 'fits'}} of the final Person
    Activity fits (#80, the analogue of decision f): a fit is at chance when
    its test macro-F1 is at most CHANCE_MARGIN above the one-class value of
    its test split (at_chance; below it and non-finite included), that value
    from its test confusions (row sums = the loader's test class counts).
    None for an arm with a fit without confusions."""
    f = df[(df['stage'] == 'final') & (df['system'] == 'pa')] \
        if not df.empty else df
    out = {}
    for arm, g in f.groupby('arm'):
        if not all(isinstance(c, np.ndarray) for c in g['confusion']):
            out[arm] = None
            continue
        one = [one_class_macro_f1(c.sum(axis=(0, 2))) for c in g['confusion']]
        out[arm] = {'one_class': sorted(set(one)), 'fits': len(g),
                    'seeds_at_chance': sorted(int(s) for s, m, o in zip(
                        g['seed'], g['metric'], one) if at_chance(m, o))}
    return out


def _chance_bound(chance, arm_a, arm_b):
    """floor_bound of a Person Activity contrast: an arm not learned by the
    seed rule (not_learned on its seeds at chance); None without both
    markings."""
    marks = [chance.get(arm_a), chance.get(arm_b)]
    if None in marks:
        return None
    return any(not_learned(len(m['seeds_at_chance']), m['fits'])
               for m in marks)


def _paired(df, system, arm_a, arm_b):
    """The final fits of two arms of one task merged on seed (suffixes _a,
    _b), with the paired value 'v' (a diverged fit at the worst value)."""
    f = df[(df['stage'] == 'final') & (df['system'] == system)]
    f = f.assign(v=np.where(f['diverged'], f['task'].map(WORST), f['metric']))
    return f[f['arm'] == arm_a].merge(f[f['arm'] == arm_b], on='seed',
                                      suffixes=('_a', '_b')).sort_values('seed')


def _rate(k, n):
    lo, hi = clopper_pearson(k, n)
    return f'{k}/{n} [{lo:.2f}, {hi:.2f}]'


def _pair_stats(p, task, m=1, alternative='two-sided') -> dict:
    """Statistics of one paired block (a _paired frame, n >= 1): Wilcoxon
    (exact without ties or zeros; undecidable below MIN_PAIRS pairs), d_z,
    P(A>B) with its Clopper-Pearson interval, the median difference with its
    bootstrap interval (Person Activity: crossed seeds x test recordings when
    every fit has its confusion matrices and none diverged; else seeds), the
    divergence rates of both arms."""
    x, y = p['v_a'].to_numpy(float), p['v_b'].to_numpy(float)
    sign = 1 if HIGHER_IS_BETTER[task] else -1
    pw, lo, hi = p_a_gt_b(sign * p['metric_a'].to_numpy(float),
                          sign * p['metric_b'].to_numpy(float),
                          p['diverged_a'].to_numpy(bool),
                          p['diverged_b'].to_numpy(bool))
    crossed = (task == 'pa' and not (p['diverged_a'] | p['diverged_b']).any()
               and all(isinstance(c, np.ndarray) for c in
                       (*p['confusion_a'], *p['confusion_b']))
               and len({c.shape for c in (*p['confusion_a'],
                                          *p['confusion_b'])}) == 1)
    ci = (crossed_median_diff_ci(np.stack(p['confusion_a']),
                                 np.stack(p['confusion_b']), N_BOOT)
          if crossed else bootstrap_median_ci(x - y, N_BOOT))
    n = len(x)
    p_raw, method = wilcoxon_test(x, y, alternative) if n else (None, None)
    p_adj = None if p_raw is None else min(1.0, p_raw * m)
    status = 'undecidable' if n < MIN_PAIRS else 'tested'
    return {'n_pairs': n, 'status': status, 'method': method,
            'p_raw': p_raw, 'p_adj': p_adj,
            'significant': (status == 'tested' and p_adj is not None
                            and p_adj < ALPHA),
            'd_z': _cohens_d_paired(x, y) if n > 1 else float('nan'),
            'p_a_gt_b': pw, 'p_a_gt_b_lo': lo, 'p_a_gt_b_hi': hi,
            'median_diff': float(np.median(x - y)),
            'median_diff_lo': ci[0], 'median_diff_hi': ci[1],
            'ci_units': 'seed x recording' if crossed else 'seed',
            'diverged_a': _rate(int(p['diverged_a'].sum()), n),
            'diverged_b': _rate(int(p['diverged_b'].sum()), n)}


def _sign_stable_tuning(hpo, task, a, b):
    """Whether the validation score difference A - B of the two selected
    configurations has the same sign on both tuning seeds (None without
    both selections or both per-seed scores)."""
    ra, rb = hpo.get((task, *a)), hpo.get((task, *b))
    if not ra or not rb:
        return None
    sa, sb = ra['ranking'][0].get('scores', {}), rb['ranking'][0].get(
        'scores', {})
    seeds = [s for s in sa if s in sb]
    if len(seeds) < 2:
        return None
    s1, s2 = (np.sign(sa[s] - sb[s]) for s in seeds[:2])
    return bool(s1 == s2)


def core_contrast_blocks(df, floors=None, hpo=None) -> pd.DataFrame:
    """Every entry of CORE_CONTRASTS as a paired block over the final seeds,
    Bonferroni over its family size (BONFERRONI_FAMILIES), two-sided. The
    H-c entries run once more as ROBUSTNESS_FAMILY on the column
    'metric_robust' when df carries it (#34, merge_robustness): fits
    without an eval-only row are left out, the interval is over seeds (the
    test confusion matrices belong to the clean metric) and the divergence
    flag is diverged_robust (or a non-finite metric_robust), not the clean
    one: a fit diverged on the robustness set enters at the worst value.

    floors: compute_floors output (floors.json, #67); a rollout block takes
    the floor_bound flag of its (family, task, A, B) entry (None without
    floors or without an entry); a Person Activity block takes the chance
    marking of its arms (pa_chance_marking, #80). hpo: load_hpo_selections
    output, for sign_stable_tuning. A block with an arm sized outside the
    budget tolerance (#59) is excluded; one without fits is missing."""
    hpo = hpo or {}
    fb = {}
    for c in (floors or {}).get('contrasts', []):
        k = (c['family'], c['task'], tuple(c['a']), tuple(c['b']))
        fb[k] = fb.get(k, False) or c['floor_bound']
    chance = pa_chance_marking(df)
    families = [(f, e, f, 'metric') for f, e in CORE_CONTRASTS.items()]
    if 'metric_robust' in df:
        families.append((ROBUSTNESS_FAMILY, CORE_CONTRASTS['H-c'], 'H-c',
                         'metric_robust'))
    rows = []
    for family, entries, floor_family, col in families:
        m = BONFERRONI_FAMILIES[family]
        for task, a, b in entries:
            arm_a, arm_b = '/'.join(a), '/'.join(b)
            row = {'family': family, 'task': task, 'a': arm_a, 'b': arm_b,
                   'm': m, 'endpoint': col, 'n_pairs': 0,
                   'significant': False,
                   'floor_bound': (_chance_bound(chance, arm_a, arm_b)
                                   if task == 'pa' else
                                   fb.get((floor_family, task, a, b))),
                   'sign_stable_tuning': _sign_stable_tuning(hpo, task, a, b)}
            d = df
            if col != 'metric':     # the fits with an eval-only row
                d = df[df['diverged_robust'].notna()]
                d = d.assign(metric=d[col], confusion=None, diverged=(
                    d['diverged_robust'].astype(bool)
                    | ~np.isfinite(d[col].astype(float))))
            p = (_paired(d, task, arm_a, arm_b)
                 if not d.empty else pd.DataFrame())
            if p.empty:
                row['status'] = 'missing'
            elif (p['budget_above_tolerance_a'] | p['budget_above_tolerance_b']
                  ).any():
                row.update(n_pairs=len(p),
                           status='excluded: arm above budget tolerance')
            else:
                row.update(_pair_stats(p, p['task_a'].iloc[0], m))
            rows.append(row)
    return pd.DataFrame(rows)


def hc_conjunction(blocks) -> dict:
    """{A arm: supported} for H-c (decision n): supported if the contrast is
    significant on every task whose block is not floor-bound (a missing
    floor marking counts as passed)."""
    hc = blocks[blocks['family'] == 'H-c']
    out = {}
    for arm, g in hc.groupby('a'):
        g = g[g['floor_bound'] != True]   # noqa: E712 (None counts as passed)
        out[arm] = bool(len(g)) and bool(g['significant'].all())
    return out


def descriptive_pairs(df) -> pd.DataFrame:
    """Every other pair of arms within a task (control arms included): n
    pairs, median difference and P(A>B); no test, no significance claim."""
    core = {(t, '/'.join(a), '/'.join(b)) for e in CORE_CONTRASTS.values()
            for t, a, b in e}
    f = df[df['stage'] == 'final'] if not df.empty else df
    rows = []
    for system in sorted(f['system'].unique()) if not f.empty else ():
        for a, b in combinations(sorted(f.loc[f['system'] == system,
                                              'arm'].unique()), 2):
            if (system, a, b) in core or (system, b, a) in core:
                continue
            p = _paired(df, system, a, b)
            if p.empty:
                continue
            x, y = p['v_a'].to_numpy(float), p['v_b'].to_numpy(float)
            sign = 1 if HIGHER_IS_BETTER[p['task_a'].iloc[0]] else -1
            pw, lo, hi = p_a_gt_b(sign * p['metric_a'].to_numpy(float),
                                  sign * p['metric_b'].to_numpy(float),
                                  p['diverged_a'].to_numpy(bool),
                                  p['diverged_b'].to_numpy(bool))
            rows.append({'task': system, 'a': a, 'b': b, 'n_pairs': len(p),
                         'median_diff': float(np.median(x - y)),
                         'p_a_gt_b': pw, 'p_a_gt_b_lo': lo, 'p_a_gt_b_hi': hi})
    return pd.DataFrame(rows)


def _fmt(v, spec='.4g'):
    return '—' if v is None or (isinstance(v, float) and np.isnan(v)) \
        else format(v, spec)


def contrast_report(df, blocks, desc) -> str:
    """Markdown of the arm table, the contrast blocks, the H-c conjunction
    and the descriptive pairs."""
    f = df[df['stage'] == 'final']
    lines = ['## Sequence-suite arms (final seeds; metric: multi-anchor VPT '
             'median on the rollouts, test macro-F1 on Person Activity)\n',
             '| Task | Arm | fits | median metric | diverged [95% CP] '
             '| budget above tolerance |',
             '|------|-----|------|---------------|-------------------'
             '|------------------------|']
    for (system, arm), g in f.groupby(['system', 'arm']):
        lines.append(f"| {system} | {arm} | {len(g)} "
                     f"| {_fmt(float(np.median(g['metric'])))} "
                     f"| {_rate(int(g['diverged'].sum()), len(g))} "
                     f"| {'YES' if g['budget_above_tolerance'].any() else 'no'} |")
    lines += ['', f'## Pre-registered contrasts (paired over seeds, two-sided '
              f'Wilcoxon, alpha={ALPHA}, Bonferroni within the family; '
              f'n_pairs < {MIN_PAIRS}: undecidable)\n',
              '| Family | m | Task | A | B | n pairs | status | method | p raw '
              '| p adj | significant | d_z | P(A>B) [95% CI] '
              '| median diff [95% CI] | diverged A | diverged B '
              '| floor_bound | sign_stable_tuning |',
              '|--------|---|------|---|---|---------|--------|--------|-------'
              '|-------|-------------|-----|-----------------'
              '|----------------------|------------|------------'
              '|-------------|--------------------|']
    for b in blocks.to_dict('records'):
        if b['status'] in ('tested', 'undecidable'):
            stats = (f"| {b['n_pairs']} | {b['status']} | {b['method'] or '—'} "
                     f"| {_fmt(b['p_raw'])} | {_fmt(b['p_adj'])} "
                     f"| {'YES' if b['significant'] else 'no'} "
                     f"| {_fmt(b['d_z'], '+.3f')} "
                     f"| {b['p_a_gt_b']:.2f} [{b['p_a_gt_b_lo']:.3f}, "
                     f"{b['p_a_gt_b_hi']:.3f}] "
                     f"| {b['median_diff']:+.4g} [{b['median_diff_lo']:+.4g}, "
                     f"{b['median_diff_hi']:+.4g}] ({b['ci_units']}) "
                     f"| {b['diverged_a']} | {b['diverged_b']} ")
        else:
            stats = (f"| {b['n_pairs']} | {b['status']} | — | — | — | — | — "
                     "| — | — | — | — ")
        lines.append(f"| {b['family']} | {b['m']} | {b['task']} | {b['a']} "
                     f"| {b['b']} {stats}| {_fmt(b['floor_bound'], '')} "
                     f"| {_fmt(b['sign_stable_tuning'], '')} |")
    lines += ['', 'H-c conjunction (significant on every task that is not '
              'floor-bound):']
    lines += [f'- {a} vs cfc_lrc/dense3: **{"supported" if s else "not supported"}**'
              for a, s in hc_conjunction(blocks).items()]
    lines += ['', 'A diverged fit (non-finite, or every test rollout '
              'unbounded) enters the pairing with the worst value (VPT 0, '
              'macro-F1 0); in the H-c robustness family divergence is that '
              'of the robustness set (diverged_robust: every OOD rollout '
              'unbounded or a non-finite one, non-finite noisy macro-F1). '
              'Median difference: 10,000-resample percentile '
              'bootstrap. sign_stable_tuning: the sign of A - B on the '
              'validation scores of the selected configurations agrees '
              'between the two tuning seeds.\n']
    chance = pa_chance_marking(df)
    if chance:
        lines += [f'## Person Activity chance marking (#80: test macro-F1 '
                  f'at most {CHANCE_MARGIN:g} above the one-class value of '
                  'the test split; floor_bound of the contrasts)\n',
                  '| Task | Arm | fits | one-class | seeds at chance '
                  '| above chance |',
                  '|------|-----|------|-----------|-----------------'
                  '|--------------|']
        for arm, m in sorted(chance.items()):
            if m is None:
                lines.append(f'| pa | {arm} | — | — | — | no test confusions |')
                continue
            nl = not_learned(len(m['seeds_at_chance']), m['fits'])
            lines.append(
                f"| pa | {arm} | {m['fits']} "
                f"| {'/'.join(f'{v:.4f}' for v in m['one_class'])} "
                f"| {m['seeds_at_chance'] or '—'} "
                f"| {'NO (not learned)' if nl else 'yes'} |")
        lines.append('\nSeed rule (decision f): not learned if at chance in '
                     'more than one of five seeds (two of ten).\n')
    lines += ['## Descriptive pairs (no significance claim)\n',
              '| Task | A | B | n pairs | median diff | P(A>B) [95% CI] |',
              '|------|---|---|---------|-------------|-----------------|']
    for r in desc.to_dict('records'):
        lines.append(f"| {r['task']} | {r['a']} | {r['b']} | {r['n_pairs']} "
                     f"| {r['median_diff']:+.4g} | {r['p_a_gt_b']:.2f} "
                     f"[{r['p_a_gt_b_lo']:.3f}, {r['p_a_gt_b_hi']:.3f}] |")
    if 'diverged_robust' in f:
        cols = [c for c in ROBUST_DESCRIPTIVE if c in f]
        lines += ['', '## Eval-only robustness endpoints (#34; final seeds '
                  'with a robustness row, median per arm, descriptive)\n',
                  '| Task | Arm | fits | diverged_robust | '
                  + ' | '.join(cols) + ' |',
                  '|' + '---|' * (len(cols) + 4)]
        r = f[f['diverged_robust'].notna()]
        for (system, arm), g in r.groupby(['system', 'arm']):
            lines.append(
                f"| {system} | {arm} | {len(g)} "
                f"| {int(g['diverged_robust'].astype(bool).sum())} | "
                + ' | '.join(_fmt(g[c].median()) for c in cols) + ' |')
    lines.append('')
    return '\n'.join(lines)


def _co_arm(r):
    """'cons' | 'lam0' | 'avg_in' | None for a committee result record of the
    suite protocol (identical views, training corruption on)."""
    if r.get('view_mode') != 'identical' or r.get('train_corruption') != 'on':
        return None
    if r.get('view_average') == 'mean':
        return 'avg_in'
    if r.get('view_average', 'off') != 'off' or r.get('consensus') != 'mean':
        return None
    if r.get('vote_lambda') is None:
        return 'cons'
    return 'lam0' if r['vote_lambda'] == 0 else None


def load_consensus_runs(co_dir) -> dict:
    """{(cell, wiring, arm): {seed: (macro-F1 at HD_M, AUC mean over
    HD_AUC_M)}} from the K=HD_K committee JSONs with the fault-grid endpoints;
    the input-averaging monolith (any wiring) is also keyed (cell, None,
    'avg_in'). An arm with a run sized outside the budget tolerance (#59) is
    replaced by None (it enters no confirmatory contrast)."""
    runs, above = defaultdict(dict), set()
    for path in sorted(glob(os.path.join(co_dir, '*.json'))):
        with open(path, encoding='utf-8') as f:
            r = json.load(f)
        arm = _co_arm(r)
        if (arm is None or r.get('n_columns') != HD_K
                or 'test_macro_f1_by_fault' not in r):
            continue
        m = r['fault_counts']
        auc = np.mean([r['per_position_macro_f1_auc'][m.index(k)]
                       for k in HD_AUC_M])
        key = (r['cell'], r['wiring'], arm)
        runs[key][r['seed']] = (r['test_macro_f1_by_fault'][m.index(HD_M)],
                                float(auc))
        if r.get('budget_above_tolerance'):
            above.add(key)
    runs = {k: (None if k in above else v) for k, v in runs.items()}
    for cell, wiring, arm in list(runs):
        if arm == 'avg_in':
            runs.setdefault((cell, None, 'avg_in'), runs[(cell, wiring, arm)])
    return runs


def _one_sided(a, b, j):
    """(n pairs, median diff, one-sided p, Wilcoxon method) of a > b on
    component j, paired by seed; p None below MIN_PAIRS pairs (undecidable)
    or with all differences zero."""
    seeds = sorted(set(a) & set(b))
    x = np.array([a[s][j] for s in seeds])
    y = np.array([b[s][j] for s in seeds])
    med = float(np.median(x - y)) if seeds else None
    if len(seeds) < MIN_PAIRS:
        return len(seeds), med, None, None
    return (len(seeds), med, *wilcoxon_test(x, y, alternative='greater'))


def consensus_family(co_dir) -> str:
    """The H-d family of four (one-sided, Bonferroni) and the attribution
    flag per wiring and cell."""
    runs = load_consensus_runs(co_dir)
    n_tests = BONFERRONI_FAMILIES['H-d']
    lines = [f'## H-d consensus family (K={HD_K}, one-sided Wilcoxon over '
             f'seeds, Bonferroni m={n_tests}, alpha={ALPHA})\n',
             '| Cell | Wiring | Part | Contrast | n pairs | median diff '
             '| p one-sided | p Bonferroni | method | significant |',
             '|------|--------|------|----------|---------|-------------'
             '|-------------|--------------|--------|-------------|']
    flags = []
    for cell in sorted({c for c, _, _ in runs}):
        for w in HD_WIRINGS:
            cons = runs.get((cell, w, 'cons'))
            avg = runs.get((cell, w, 'avg_in'), runs.get(
                (cell, None, 'avg_in')))
            lam0 = runs.get((cell, w, 'lam0'))
            for part, label, b, j in (
                    ('i', f'consensus vs avg_in, macro-F1 m={HD_M}', avg, 0),
                    ('ii', 'consensus vs lambda=0, per-position AUC '
                           f'm in {set(HD_AUC_M)}', lam0, 1)):
                if cons is None or b is None:
                    lines.append(f'| {cell} | {w} | {part} | {label} | 0 '
                                 '| — | — | — | — | missing arm or arm above '
                                 'budget tolerance |')
                    continue
                n, med, p, method = _one_sided(cons, b, j)
                if p is None:
                    why = ('undecidable' if n < MIN_PAIRS
                           else 'all differences zero')
                    lines.append(f'| {cell} | {w} | {part} | {label} | {n} '
                                 f'| {med if med is None else f"{med:+.4f}"} '
                                 f'| — | — | — | {why} |')
                    continue
                p_adj = min(1.0, p * n_tests)
                lines.append(f'| {cell} | {w} | {part} | {label} | {n} '
                             f'| {med:+.4f} | {p:.4g} | {p_adj:.4g} '
                             f'| {method} | {"YES" if p_adj < ALPHA else "no"} |')
            if cons is not None and lam0 is not None:
                n, med, p, _ = _one_sided(cons, lam0, 0)
                attr = ('consensus' if p is not None and p < ALPHA
                        else 'late fusion')
                flags.append(f'- {cell}/{w}: learned lambda vs lambda=0 at '
                             f'm={HD_M}, n={n}, p one-sided '
                             f'{"—" if p is None else f"{p:.4g}"} -> '
                             f'attribution: **{attr}**')
    lines += ['', 'Attribution (a part-i effect counts as consensus only if '
              'learned lambda also beats lambda=0 at m=15, one-sided, '
              f'alpha={ALPHA}; no family test):', *flags, '']
    return '\n'.join(lines)


# --- F2 test families (protocol F2 sections 2.0 to 2.4 and 6, #107) --------

# family -> {test: (task, arm A label, arm B label)}, delta = A - B, higher is
# better on every endpoint (VPT-multi, test macro-F1, eval_return_mean); the
# family size m is the number of its tests, the one place it is defined.
# Labels as arm_label returns them (protocol F2 section 3.2, point 7); the
# width-matched dense3 (#85) runs with a fixed --size and no param_budget.
F2_FAMILIES = {
    'Rewiring': {
        'rewiring_lv': ('periodic_predator_prey', 'cfc_lrc/ncp',
                        'cfc_lrc/ncp_random'),
        'rewiring_pa': ('pa', 'cfc_lrc/ncp', 'cfc_lrc/ncp_random')},
    'Width': {
        'width_lv': ('periodic_predator_prey', 'cfc_lrc/ncp',
                     'cfc_lrc/dense3 [fixed width]'),
        'width_pa': ('pa', 'cfc_lrc/ncp', 'cfc_lrc/dense3 [fixed width]')},
    'Budget': {
        'budget_lv': ('periodic_predator_prey', 'cfc_lrc/ncp [epochs=600]',
                      'cfc_lrc/ncp'),
        'budget_duffing': ('duffing', 'cfc_lrc/ncp [epochs=900]',
                           'cfc_lrc/ncp')},
    'PPO': {'ppo_halfcheetah': ('HalfCheetah-v5', 'cfc_lrc', 'gru')},
}
F2_M = sum(len(tests) for tests in F2_FAMILIES.values())   # F2-wide m (7)
F2_TESTS = {test: (family, *t) for family, tests in F2_FAMILIES.items()
            for test, t in tests.items()}
F2_PPO_TASKS = {t[0] for t in F2_FAMILIES['PPO'].values()}
F2_ARM_KEYS = {'label', 'protocol_version', 'paths'}
PPO_FIELDS = ('model', 'seed', 'task', 'profile', 'eval_return_mean',
              'eval_return_std', 'untrained_return_mean',
              'untrained_return_std', 'params_effective', 'budget_deviation',
              'budget_above_tolerance')
EXCLUDED = 'excluded: arm above budget tolerance'   # as core_contrast_blocks


def load_f2_spec(path) -> dict:
    """The --f2-families spec: {test: {'a': arm, 'b': arm}} for exactly the
    tests of F2_FAMILIES, arm = {label (as in F2_FAMILIES), protocol_version
    (a string or a list of strings; null for PPO, whose records carry no
    stamp), paths (run folders or record files, no patterns)}. Unknown or
    missing keys, a pattern in a path and a path that does not exist raise
    SystemExit."""
    with open(path, encoding='utf-8') as f:
        spec = yaml.safe_load(f)
    if not isinstance(spec, dict) or set(spec) != set(F2_TESTS):
        got = set(spec) if isinstance(spec, dict) else set()
        raise SystemExit(f'{path}: the spec must hold exactly the tests '
                         f'{sorted(F2_TESTS)} (missing '
                         f'{sorted(set(F2_TESTS) - got)}, unknown '
                         f'{sorted(got - set(F2_TESTS))})')
    for test, arms in spec.items():
        task = F2_TESTS[test][1]
        if not isinstance(arms, dict) or set(arms) != {'a', 'b'}:
            raise SystemExit(f'{path}: {test} needs exactly the keys a, b')
        for side, label in zip('ab', F2_TESTS[test][2:]):
            arm = arms[side]
            if not isinstance(arm, dict) or set(arm) != F2_ARM_KEYS:
                raise SystemExit(f'{path}: {test}.{side} needs exactly the '
                                 f'keys {sorted(F2_ARM_KEYS)}')
            if arm['label'] != label:
                raise SystemExit(f'{path}: {test}.{side}: label '
                                 f'{arm["label"]!r}, expected {label!r}')
            pv = arm['protocol_version']
            if task in F2_PPO_TASKS:
                ok = pv is None
            else:
                pv = [pv] if isinstance(pv, str) else pv
                ok = (isinstance(pv, list) and pv
                      and all(isinstance(v, str) for v in pv))
            if not ok:
                raise SystemExit(f'{path}: {test}.{side}: protocol_version '
                                 'must be a string or a list of strings '
                                 '(null on PPO)')
            arm['protocol_version'] = pv
            paths = arm['paths']
            if (not isinstance(paths, list) or not paths
                    or not all(isinstance(p, str) for p in paths)):
                raise SystemExit(f'{path}: {test}.{side}: paths must be a '
                                 'non-empty list of run folders or files')
            for p in paths:
                if has_magic(p):
                    raise SystemExit(f'{path}: {test}.{side}: pattern in '
                                     f'path {p!r}; list the folders or files')
                if not os.path.exists(p):
                    raise SystemExit(f'{path}: {test}.{side}: no such path '
                                     f'{p!r}')
    return spec


def _record_files(paths):
    """The JSON files of a path list: a file as is, a folder searched
    recursively (inside the folder only)."""
    for p in paths:
        if os.path.isdir(p):
            yield from sorted(glob(os.path.join(escape(p), '**', '*.json'),
                                   recursive=True))
        else:
            yield p


def load_ppo_runs(paths, task='HalfCheetah-v5'):
    """(DataFrame, n skipped) of the run_mujoco_benchmark.py records (#61)
    below paths: one row per record with profile 'final' and the task, the
    PPO_FIELDS plus arm (= model), metric (eval_return_mean), diverged (a
    non-finite metric) and file; records of another profile or task are
    skipped and counted. Two kept records of one (model, seed) or a kept
    record without one of the PPO_FIELDS raise SystemExit."""
    rows, skipped = [], 0
    for path in _record_files(paths):
        with open(path, encoding='utf-8') as f:
            r = json.load(f)
        if not isinstance(r, dict) or 'model' not in r:
            continue
        if r.get('profile') != 'final' or r.get('task') != task:
            skipped += 1
            continue
        missing = [k for k in PPO_FIELDS if k not in r]
        if missing:
            raise SystemExit(f'{path}: PPO record of arm {r["model"]} '
                             f'without {missing}')
        metric = float('nan') if r['eval_return_mean'] is None \
            else float(r['eval_return_mean'])
        rows.append({**{k: r[k] for k in PPO_FIELDS}, 'arm': r['model'],
                     'metric': metric, 'diverged': not np.isfinite(metric),
                     'file': path})
    df = pd.DataFrame(rows)
    if not df.empty:
        dup = df[df.duplicated(['model', 'seed'], keep=False)]
        if not dup.empty:
            raise SystemExit('two PPO records of one (model, seed): '
                             + ', '.join(dup['file']))
    return df, skipped


def _width_matched(r) -> bool:
    """The width-matched dense3 of protocol F2 section 2.2: #85 records it
    as dense3 with a fixed --size and no param_budget (no flag of its own),
    the one arm exempt from the budget-tolerance rule (F1 section 36)."""
    return r.get('wiring') == 'dense3' and r.get('param_budget') is None


def _f2_arm(test, side, task, arm, draws):
    """Final fits of one F2 arm: the records in its paths whose label is the
    arm label, each with the declared protocol_version (else SystemExit
    naming the file), plus data_key, width_matched and below (the per-fit
    floor marking; a fit without one raises SystemExit naming the arm)."""
    label, versions = arm['label'], arm['protocol_version']
    name = f'{test}.{side} ({label})'
    if task in F2_PPO_TASKS:
        df, skipped = load_ppo_runs(arm['paths'], task)
        df = df[df['arm'] == label] if not df.empty else df
        if df.empty:
            raise SystemExit(f'{name}: no final record in its paths')
        df = df.assign(task='ppo', width_matched=False, data_key=None,
                       budget_above_tolerance=df['budget_above_tolerance']
                       .astype(bool),
                       below=[not m > u + sd for m, u, sd in zip(
                           df['metric'], df['untrained_return_mean'],
                           df['untrained_return_std'])])
        return df, skipped
    runner = 'pa' if task == 'pa' else 'lv'
    rows = []
    for path in _record_files(arm['paths']):
        with open(path, encoding='utf-8') as f:
            r = json.load(f)
        if (not isinstance(r, dict) or _record_task(r) != runner
                or (runner == 'lv' and r.get('system') != task)
                or r.get('seed') not in FINAL_SEEDS or arm_label(r) != label):
            continue
        eff = (r.get('config') or {}).get('effective') or {}
        if eff.get('protocol_version') not in versions:
            raise SystemExit(f'{path}: protocol_version '
                             f'{eff.get("protocol_version")!r}, but {name} '
                             f'lists {versions}')
        row = _sequence_row(path, r, runner)
        if runner == 'lv':
            m = r['rollout_metrics_test']
            fl = (None if any(k not in m for k in (
                'anchors', 'anchor_horizon', 'anchor_theta')) or any(
                eff.get(k) is None for k in FLOOR_DATA_KEYS)
                else fit_floor(r, draws))
            if fl is None or fl[1] is None or not np.isfinite(fl[1]):
                raise SystemExit(f'{name}: no floor marking for {path}')
            below = not fl[0] >= fl[1]
        else:
            if row['confusion'] is None:
                raise SystemExit(f'{name}: no test confusion matrices (chance '
                                 f'marking) in {path}')
            below = at_chance(row['metric'], one_class_macro_f1(
                row['confusion'].sum(axis=(0, 2))))
        rows.append({**row, 'width_matched': _width_matched(r),
                     'data_key': eff.get('split_seed' if runner == 'pa'
                                         else 'data_seed'),
                     'below': bool(below)})
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f'{name}: no final fit in its paths')
    dup = df[df.duplicated('seed', keep=False)]
    if not dup.empty:
        raise SystemExit(f'two final fits of {name} and one seed: '
                         + ', '.join(dup['file']))
    return df, 0


def _signed_ranks(x, y):
    """Signs and mid-ranks of |d| of the nonzero paired differences x - y."""
    d = np.asarray(x, float) - np.asarray(y, float)
    nz = d[d != 0]
    return np.sign(nz), rankdata(np.abs(nz))


def permutation_p(x, y) -> float:
    """Exact two-sided sign-flip p of the signed-rank statistic: W+ (sum of
    the mid-ranks of |d| over the k nonzero differences with d > 0) against
    all 2^k sign patterns, two-sided by the distance of W+ from its mean
    sum(ranks) / 2. Without ties and zeros it equals the exact Wilcoxon p;
    1.0 without a nonzero difference."""
    sign, ranks = _signed_ranks(x, y)
    if not ranks.size:
        return 1.0
    # ponytail: enumerates 2^k patterns, fine for k <= 20 (seeds 0-9)
    flips = (np.arange(2 ** ranks.size)[:, None] >> np.arange(ranks.size)) & 1
    mu = ranks.sum() / 2
    dev = abs(ranks[sign > 0].sum() - mu)
    return float(np.mean(np.abs(flips @ ranks - mu) >= dev - 1e-9))


def _f2_test(family, test, task, m, a, b) -> dict:
    """One F2 test row from the final fits of arm A and arm B."""
    row = {'family': family, 'test': test, 'task': task,
           'a': a['arm'].iloc[0], 'b': b['arm'].iloc[0], 'm': m}
    marks = {}
    for side, d in (('A', a), ('B', b)):
        k, n = int(d['below'].sum()), len(d)
        row[f'floor_{side.lower()}'] = f'{k}/{n}'
        marks[side] = not_learned(k, n)
    row['floor_bound'] = marks['A'] or marks['B']
    row['floor_bound_arms'] = ','.join(s for s in 'AB' if marks[s]) or None
    kind = a['task'].iloc[0]
    if kind == 'ppo':     # worst value: smallest finite return of the test
        fin = pd.concat([a['metric'], b['metric']])
        fin = fin[np.isfinite(fin)]
        worst = float(fin.min()) if len(fin) else float('nan')
    else:
        worst = WORST[kind]
    a, b = (d.assign(v=np.where(d['diverged'], worst, d['metric']))
            for d in (a, b))
    p = a.merge(b, on='seed', suffixes=('_a', '_b')).sort_values('seed')
    if kind != 'ppo':
        bad = p[p['data_key_a'].isna() | (p['data_key_a'] != p['data_key_b'])]
        if not bad.empty:
            raise SystemExit(f'{test}: arms A and B differ in the data key of '
                             f'config.effective on seed(s) '
                             f'{bad["seed"].tolist()}')
    above = any((d['budget_above_tolerance'] & ~d['width_matched']).any()
                for d in (a, b))
    n = len(p)
    row.update(n_pairs=n, n_nonfinite=int(p['diverged_a'].sum()
                                          + p['diverged_b'].sum()))
    if n:
        st = _pair_stats(p, kind, m)
        x, y = p['v_a'].to_numpy(float), p['v_b'].to_numpy(float)
        row.update({k: st[k] for k in (
            'd_z', 'p_a_gt_b', 'p_a_gt_b_lo', 'p_a_gt_b_hi', 'median_diff',
            'median_diff_lo', 'median_diff_hi', 'ci_units')},
            n_positive=int((x - y > 0).sum()),
            ci_includes_zero=(bool(st['median_diff_lo'] <= 0
                                   <= st['median_diff_hi'])
                              if task == 'pa' else None))
    if above:
        row.update(status=EXCLUDED, reading=None)
        return row
    row['status'] = 'tested' if n >= MIN_PAIRS else 'undecidable'
    if not n:
        row['reading'] = 'undecidable'
        return row
    p_raw, method = st['p_raw'], st['method']
    if p_raw is None:       # every paired difference zero
        p_raw = 1.0
    p_perm = permutation_p(x, y)
    sign, ranks = _signed_ranks(x, y)
    w = float((sign * ranks).sum())          # W+ - W-
    tested = row['status'] == 'tested'
    row.update(method=method, p_raw=p_raw, p_adj=min(1.0, m * p_raw),
               p_adj_f2=min(1.0, F2_M * p_raw), p_perm=p_perm,
               p_perm_adj=min(1.0, m * p_perm),
               w_plus_minus_w_minus=w)
    row['significant'] = tested and row['p_adj'] < ALPHA
    row['significant_f2'] = tested and row['p_adj_f2'] < ALPHA
    row['only_under_family_cut'] = (row['significant']
                                    and not row['significant_f2'])
    row['verdict_flip'] = (row['p_adj'] < ALPHA) != (row['p_perm_adj'] < ALPHA)
    row['reading'] = ('undecidable' if not tested else 'floor-bound'
                      if row['floor_bound'] else 'PASS'
                      if row['p_adj'] < ALPHA else 'FAIL')
    if row['reading'] == 'PASS':
        row['direction'] = 'A>B' if w > 0 else 'A<B'
        row['median_sign_differs'] = bool(np.sign(row['median_diff'])
                                          != np.sign(w))
    return row


F2_COLUMNS = ['family', 'test', 'task', 'a', 'b', 'm', 'n_pairs', 'status',
              'reading', 'direction', 'method', 'p_raw', 'p_adj', 'p_adj_f2',
              'p_perm', 'p_perm_adj', 'significant', 'significant_f2',
              'only_under_family_cut', 'verdict_flip', 'median_sign_differs',
              'ci_includes_zero', 'w_plus_minus_w_minus', 'median_diff',
              'median_diff_lo', 'median_diff_hi', 'ci_units', 'd_z',
              'p_a_gt_b', 'p_a_gt_b_lo', 'p_a_gt_b_hi', 'n_positive',
              'n_nonfinite', 'floor_a', 'floor_b', 'floor_bound',
              'floor_bound_arms', 'n_skipped_records']


def f2_family_blocks(spec) -> pd.DataFrame:
    """One row per test of F2_FAMILIES (columns F2_COLUMNS) from a
    load_f2_spec spec: paired over the final seeds, two-sided, delta = A - B;
    p of wilcoxon_test (the deciding p), adjusted within the family (m) and
    F2-wide (F2_M, sensitivity), the exact permutation p next to it; the
    floor marking per arm from its own records; the budget-tolerance
    exclusion of every arm except the width-matched one."""
    draws, rows = {}, []
    for family, tests in F2_FAMILIES.items():
        for test, (task, _, _) in tests.items():
            (a, skip_a), (b, skip_b) = (
                _f2_arm(test, side, task, spec[test][side], draws)
                for side in 'ab')
            row = _f2_test(family, test, task, len(tests), a, b)
            if task in F2_PPO_TASKS:
                row['n_skipped_records'] = skip_a + skip_b
            rows.append(row)
    return pd.DataFrame(rows).reindex(columns=F2_COLUMNS)


def f2_report(blocks) -> str:
    """Markdown of f2_family_blocks."""
    lines = [f'## F2 test families (protocol F2 sections 2.0 to 2.4; paired '
             f'over seeds, two-sided Wilcoxon, delta = A - B, Bonferroni '
             f'within the family, F2-wide m={F2_M} as sensitivity; '
             f'n_pairs < {MIN_PAIRS}: undecidable)\n',
             '| Family | m | Test | Task | A | B | n | status | reading '
             '| method | p | p adj | p adj F2 | p perm | p perm adj | flags '
             '| median delta [95% CI] | d_z | P(A>B) [95% CI] | seeds delta>0 '
             '| floor A | floor B | floor_bound | non-finite |',
             '|' + '---|' * 24]
    flags = (('only_under_family_cut', 'only under the family cut'),
             ('verdict_flip', 'verdict flips with p perm'),
             ('median_sign_differs', 'median sign differs'),
             ('ci_includes_zero', 'interval includes 0'))

    def yes(v):
        return isinstance(v, (bool, np.bool_)) and bool(v)

    for b in blocks.to_dict('records'):
        fl = '; '.join([*([b['direction']] if isinstance(b['direction'], str)
                          else []), *(t for k, t in flags if yes(b[k]))])
        med = ('—' if pd.isna(b['median_diff']) else
               f"{b['median_diff']:+.4g} [{b['median_diff_lo']:+.4g}, "
               f"{b['median_diff_hi']:+.4g}] ({b['ci_units']})")
        win = ('—' if pd.isna(b['p_a_gt_b']) else
               f"{b['p_a_gt_b']:.2f} [{b['p_a_gt_b_lo']:.3f}, "
               f"{b['p_a_gt_b_hi']:.3f}]")
        text = [b[k] if isinstance(b[k], str) else '—'
                for k in ('reading', 'method')]
        lines.append(
            f"| {b['family']} | {b['m']} | {b['test']} | {b['task']} "
            f"| {b['a']} | {b['b']} | {b['n_pairs']} | {b['status']} "
            f"| {text[0]} | {text[1]} | "
            + ' | '.join(_fmt(b[k]) for k in (
                'p_raw', 'p_adj', 'p_adj_f2', 'p_perm', 'p_perm_adj'))
            + f" | {fl or '—'} | {med} | {_fmt(b['d_z'], '+.3f')} | {win} "
            f"| {_fmt(b['n_positive'], '.0f')} | {b['floor_a']} "
            f"| {b['floor_b']} | "
            + (f"YES ({b['floor_bound_arms']})" if yes(b['floor_bound'])
               else 'no') + f" | {b['n_nonfinite']} |")
    skipped = blocks['n_skipped_records'].dropna()
    lines += ['', 'reading: undecidable before floor-bound before PASS/FAIL '
              f'(adj. p < {ALPHA}); the deciding p is that of wilcoxon_test '
              '(exact without ties or zeros, else the normal approximation), '
              'p perm the exact sign-flip test over the mid-ranks. floor A/B: '
              'seeds below the floor (rollouts: max(persistence, AR(1)) of '
              'the data draw; Person Activity: at most 0.01 above the '
              'one-class value; PPO: at most untrained mean + std), not '
              'learned in more than a fifth of the seeds. A non-finite or '
              'unbounded fit enters at the worst value (PPO: the smallest '
              'finite eval_return_mean of the test).']
    if len(skipped):
        lines.append(f'PPO records of another profile or task skipped: '
                     f'{int(skipped.sum())}.')
    lines.append('')
    return '\n'.join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', default=['results/runs'])
    p.add_argument('--out', default='results')
    p.add_argument('--cells', default=None,
                   help='comma-separated cell-variant subset, e.g. '
                        '"ltc,mm_ltc,ltc+clip"')
    p.add_argument('--hpo-dir', nargs='+', default=None,
                   help='folder(s) searched recursively for '
                        'hpo_selection.json (HPO-fragility flags; with '
                        '--contrasts also sign_stable_tuning)')
    p.add_argument('--pa-runs', default=None,
                   help='Person Activity result JSON folder (arm means with '
                        'bootstrap interval)')
    p.add_argument('--rollout-runs', default=None,
                   help='rollout result JSON folder, searched recursively '
                        '(outcome-class shares per arm; each subfolder is '
                        'its own arm, tuning seeds are left out)')
    p.add_argument('--floors', action='store_true',
                   help='with --rollout-runs: floor lines, floor marking and '
                        'floor_bound contrasts into <out>/floors.json')
    p.add_argument('--contrasts', action='store_true',
                   help='with --rollout-runs and/or --pa-runs: the sequence '
                        'loader, the pre-registered contrast blocks and the '
                        'descriptive pairs (contrasts.csv, sequence_runs.csv)')
    p.add_argument('--floors-json', default=None,
                   help='with --contrasts and without --floors: a floors.json '
                        'whose floor_bound flags the contrast blocks carry')
    p.add_argument('--robustness', nargs='+', default=None, metavar='CSV',
                   help='with --contrasts: eval-only robustness CSV(s) of '
                        'experiments/eval_robustness.py; their metric_robust '
                        'runs the H-c contrasts as the second family (#34)')
    p.add_argument('--co-runs', default=None,
                   help='committee result JSON folder (H-d consensus family)')
    p.add_argument('--f2-families', default=None, metavar='SPEC.yaml',
                   help='the F2 test families from a spec of explicit path '
                        'lists per arm (configs/aggregate/f2_families.yaml); '
                        'writes <out>/f2_families.csv')
    p.add_argument('--bootstrap-units', default='seed',
                   choices=('seed', 'seed,recording'),
                   help='resampling units for the --pa-runs arm means')
    args = p.parse_args(argv)
    if args.floors and not args.rollout_runs:
        p.error('--floors needs --rollout-runs')
    if args.contrasts and not (args.rollout_runs or args.pa_runs):
        p.error('--contrasts needs --rollout-runs and/or --pa-runs')
    if args.robustness and not args.contrasts:
        p.error('--robustness needs --contrasts')

    df = load_runs(args.runs)
    if args.cells and not df.empty:
        df = df[df['cell'].isin(args.cells.split(','))]
    if df.empty and not (args.hpo_dir or args.pa_runs or args.rollout_runs
                         or args.co_runs or args.f2_families):
        print(f'No run JSONs found in {args.runs}', file=sys.stderr)
        return 1

    os.makedirs(args.out, exist_ok=True)
    parts = ['# Benchmark Summary\n']
    if not df.empty:
        df.to_csv(os.path.join(args.out, 'summary.csv'), index=False)
        parts += [
            f'Runs: {len(df)} | Cells: {sorted(df.cell.unique())} | '
            f'Wirings: {sorted(df.wiring.unique())} | Seeds/config: '
            f'{df.groupby(["cell", "wiring", "system"]).size().min()}-'
            f'{df.groupby(["cell", "wiring", "system"]).size().max()}\n',
            summary_tables(df),
            statistics(df),
        ]
    if args.hpo_dir:
        parts.append(hpo_fragility(args.hpo_dir))
    if args.pa_runs:
        parts.append(pa_arm_means(args.pa_runs, args.bootstrap_units))
    if args.rollout_runs:
        parts.append(rollout_outcome_classes(args.rollout_runs))
    floors = None
    if args.floors:
        floors = compute_floors(args.rollout_runs)
        with open(os.path.join(args.out, 'floors.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(floors, f, indent=1)
        parts.append(floor_report(floors))
    elif args.floors_json:
        with open(args.floors_json, encoding='utf-8') as f:
            floors = json.load(f)
    if args.contrasts:
        seq = pd.concat([load_sequence_runs(d, t) for d, t in (
            (args.rollout_runs, 'lv'), (args.pa_runs, 'pa')) if d],
            ignore_index=True)
        if seq.empty:
            print('--contrasts: no sequence-runner JSONs found',
                  file=sys.stderr)
            return 1
        if args.robustness:
            seq = merge_robustness(seq, args.robustness)
        blocks = core_contrast_blocks(
            seq, floors, load_hpo_selections(args.hpo_dir)
            if args.hpo_dir else None)
        blocks.to_csv(os.path.join(args.out, 'contrasts.csv'), index=False)
        seq.drop(columns='confusion', errors='ignore').to_csv(
            os.path.join(args.out, 'sequence_runs.csv'), index=False)
        parts.append(contrast_report(seq, blocks, descriptive_pairs(seq)))
    if args.co_runs:
        parts.append(consensus_family(args.co_runs))
    if args.f2_families:
        f2 = f2_family_blocks(load_f2_spec(args.f2_families))
        f2.to_csv(os.path.join(args.out, 'f2_families.csv'), index=False)
        parts.append(f2_report(f2))
    report = '\n'.join(parts)
    md_path = os.path.join(args.out, 'summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(report)
    print(f'\nWritten: {md_path}' + (' and summary.csv' if not df.empty else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
