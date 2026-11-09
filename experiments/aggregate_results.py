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
                   recordings (--bootstrap-units seed,recording); searched
                   recursively (subfolder = '<folder>/<tag>' arm), with a
                   cross-mean column where the records carry per-recording F1
  --rollout-runs DIR  rollout outcome-class shares per arm (test split) with
                   exact Clopper-Pearson intervals, at tau_orb and at the 0.30
                   sensitivity, plus one non-finite label per arm
  --co-runs DIR    H-d consensus-study family (committee runner JSONs, #55):
                   one-sided paired Wilcoxon over seeds at K=30, Bonferroni
                   over the family of four, plus the attribution flag;
                   <out>/consensus_family.csv adds per test the median
                   difference with its crossed seeds x recordings and seed
                   bootstrap intervals, d_z and P(A>B),
                   <out>/consensus_arms.csv the grid mean and steps_to_90
                   per arm with seed intervals (#94)
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
                   <out>/contrasts.csv and <out>/sequence_runs.csv;
                   <out>/rq5_paired.csv adds the descriptive RQ5 pairs
                   (cfc_lrc vs cfc, cfc_lrc vs cfc_lrc_outer) per task and
                   wiring: median paired difference with a seed-bootstrap
                   interval, P(A>B), positive seeds and n, no p-value (#97)
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
  --descriptive SPEC.yaml  the descriptive F2 blocks (protocol F2 sections 5
                   and 2.3, DESCRIPTIVE_BLOCKS: B1-B5, BA-2, C-2, the 2x
                   budget pairs) from explicit path lists per arm (template:
                   configs/aggregate/descriptive_blocks.yaml): per-arm
                   median, paired difference with its bootstrap interval,
                   the reading of each block (three-seed rule, five-seed
                   direction rule, B5 plateau exit, budget direction), no
                   test; <out>/descriptive_blocks.csv
  --f3-families SPEC.yaml  the F3 test families (protocol F3, F3_FAMILIES:
                   test kind, side, endpoint, floor rule, learning-success
                   rule, match fields and status per test) from a spec of
                   an explicit seed list and explicit path lists per arm
                   (template: configs/aggregate/f3_families.yaml); paired
                   Wilcoxon with the exact permutation p, or the exact
                   McNemar on the learning success; Bonferroni within the
                   family and F3-wide (sensitivity), records outside the
                   seed list counted, reading incomplete/floor-bound/
                   PASS/FAIL; <out>/f3_families.csv

Usage:
    uv run python experiments/aggregate_results.py [--runs results/runs] [--out results]
        [--hpo-dir DIR ...] [--pa-runs DIR [--bootstrap-units seed,recording]]
        [--rollout-runs DIR [--floors]] [--co-runs DIR]
        [--contrasts [--floors-json PATH] [--robustness CSV ...]]
        [--f2-families SPEC.yaml] [--descriptive SPEC.yaml]
        [--f3-families SPEC.yaml]

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
from scipy.stats import binomtest, rankdata, wilcoxon

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
# The per-recording, per-noise-draw test confusions of the decision sigma
# (eval_robustness.py --per-recording-confusions, #96), the input of the
# crossed bootstrap of the PA robustness rows.
ROBUST_CONFUSIONS = 'confusions_sigma_0.3'
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

# Descriptive F2 blocks (protocol F2 sections 5.1 and 5.2 and the
# descriptive part of 2.3, #109): no test, each reading fixed by the
# protocol. block -> task (rollout system or 'pa'), final seeds, reading kind
# and pairings {a, b} with delta = a - b on the paired value of _paired (b
# None: no reference arm); budget pairings also name the same pair at 1x
# (a_1x, b_1x, from the F1 path list); note, caveat and config_note are
# fixed text columns of the row. Arm labels as arm_label returns them.
B5_EXIT_THRESHOLD = 0.0851   # 0.0751 + 0.01: val macro-F1 leaves the plateau
B5_EPOCHS = 10               # ... by epoch 10
ESCAPE_LOSS = 0.5            # escape epoch: first val_loss_curve entry below
CONVERGED = 0.95             # flag: best_val_epoch >= 0.95 x its own epochs
C8_NOTE = ('Epochs and learning-rate schedule change together; the result '
           'is not attributed to either factor.')
C7_CAVEAT = ('The initial plateau of lrc/ncp can mask a retention effect, '
             'and the effective gain of the base arm is shifted relative to '
             'the init arm; a result on lrc/ncp is therefore not attributed '
             'to the time-scale initialisation alone.')
DIRECTION_TEXT = 'hint, not a test'
_LV, _EPS1 = 'periodic_predator_prey', 'cfc_lrc_eps1/ncp'


def _budget_pairs(epochs):
    """RQ1 and both H-c legs at 2x (descriptive part of protocol F2 2.3)
    next to the same pair at 1x, delta in the direction of the F1 rows."""
    x2 = f' [epochs={epochs}]'
    return [{'a': a + x2, 'b': b + x2, 'a_1x': a, 'b_1x': b}
            for a, b in (('cfc_lrc/ncp', 'gru/ncp'),
                         ('cfc_lrc/ncp', 'cfc_lrc/dense3'),
                         ('cfc_lrc/cncp', 'cfc_lrc/dense3'))]


DESCRIPTIVE_BLOCKS = {
    'B1': {'task': 'duffing', 'seeds': range(3), 'reading': 'floor + 2',
           'pairs': [{'a': 'lrc_init/ncp', 'b': 'lrc/ncp'},
                     {'a': 'lrc_cal_duffing/ncp', 'b': 'lrc/ncp'}]},
    'B2': {'task': _LV, 'seeds': range(10), 'reading': 'descriptive',
           'pairs': [{'a': a, 'b': b}
                     for a in ('lrc_init/ncp', 'lrc_cal_lv/ncp')
                     for b in ('lrc/ncp', 'cfc_lrc/ncp')]},
    'B3': {'task': _LV, 'seeds': range(3), 'reading': 'above reference',
           'pairs': [{'a': f'{arm} [epochs=900]', 'b': arm, 'note': C8_NOTE}
                     for arm in ('lrc/ncp', 'lrc/dense', 'lrc_init/ncp',
                                 'lrc_cal_lv/ncp')]},
    'B4': {'task': _LV, 'seeds': range(3), 'reading': 'above reference',
           'pairs': [{'a': 'lrc_tau/ncp', 'b': 'lrc/ncp',
                      'caveat': C7_CAVEAT},
                     {'a': 'lrc_init_tau/ncp', 'b': 'lrc_init/ncp'}]},
    'B5': {'task': 'pa', 'seeds': range(3), 'reading': 'plateau exit',
           'pairs': [{'a': 'ltc_minel/ncp', 'b': None}]},
    'BA-2': {'task': 'pa', 'seeds': range(5), 'reading': 'direction',
             'pairs': [{'a': 'cfc_lrc/cncp', 'b': 'cfc_lrc/cncp_random'}]},
    'C-2': {'task': 'duffing', 'seeds': range(5), 'reading': 'direction',
            'pairs': [
                {'a': 'cfc_lrc/ncp', 'b': _EPS1, 'config_note': (
                    'same width, isolates the elastance; the eps = 1 arm '
                    'runs with the cfc_lrc/ncp config (tuning advantage of '
                    'cfc_lrc/ncp)')},
                {'a': _EPS1, 'b': 'cfc/ncp', 'config_note': (
                    'same cell form, width differs; cfc/ncp runs with its '
                    'own tuned config, the eps = 1 arm with the cfc_lrc/ncp '
                    'config')}]},
    'budget_2x_lv': {'task': _LV, 'seeds': range(10),
                     'reading': 'budget direction',
                     'pairs': _budget_pairs(600)},
    'budget_2x_duffing': {'task': 'duffing', 'seeds': range(10),
                          'reading': 'budget direction',
                          'pairs': _budget_pairs(900)},
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


def mcnemar_exact(success_a, success_b):
    """(p, n10, n01) of the exact two-sided McNemar test on paired binary
    outcomes: n10 pairs where only A succeeds, n01 where only B does; p of
    the binomial test of min(n10, n01) in n10 + n01 at 1/2, 1.0 without a
    discordant pair."""
    a, b = np.asarray(success_a, bool), np.asarray(success_b, bool)
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    if not n10 + n01:
        return 1.0, n10, n01
    return float(binomtest(min(n10, n01), n10 + n01, 0.5).pvalue), n10, n01


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
    test_macro_f1_per_recording). Result files are searched recursively; as in
    rollout_outcome_classes an arm is '<folder relative to pa_dir>/<file name
    without _seed<N>>' ('.' omitted), fits on a seed outside FINAL_SEEDS are
    left out and files without the key are skipped. If any record carries the
    per-recording key and the units are 'seed', a last column holds the
    cross-mean (mean over seeds and recordings) with its crossed interval
    ('—' for an arm whose seeds lack the key or differ in recordings)."""
    by_rec = 'recording' in units
    key = 'test_macro_f1_per_recording' if by_rec else 'test_macro_f1'
    arms, crossed, skipped, tuning = defaultdict(list), defaultdict(list), 0, 0
    for path in sorted(glob(os.path.join(pa_dir, '**', '*.json'),
                            recursive=True)):
        with open(path, encoding='utf-8') as f:
            r = json.load(f)
        if not isinstance(r, dict) or key not in r:
            skipped += 1
            continue
        if r.get('seed') not in FINAL_SEEDS:
            tuning += 1
            continue
        v = r[key]
        arm = _rollout_arm(path, r, pa_dir)[0]
        arms[arm].append([v[k] for k in sorted(v)] if isinstance(v, dict) else v)
        rec = r.get('test_macro_f1_per_recording')
        crossed[arm].append(rec if isinstance(rec, dict) and rec else None)
    cross = not by_rec and any(None not in c for c in crossed.values())
    lines = [f'## Person Activity arm means (test macro-F1, bootstrap over {units})\n',
             '| Arm | n seeds | mean | 95% CI |'
             + (' cross-mean [95% CI, seed x recording] |' if cross else ''),
             '|-----|---------|------|--------|' + ('------|' if cross else '')]
    for arm, rows in sorted(arms.items()):
        mean, lo, hi = bootstrap_mean_ci(rows, resample_cols=by_rec)
        line = f'| {arm} | {len(rows)} | {mean:.4f} | [{lo:.4f}, {hi:.4f}] |'
        if cross:
            recs = crossed[arm]
            if None in recs or len({tuple(sorted(x)) for x in recs}) > 1:
                line += ' — |'     # a seed without, or another recording set
            else:
                m, lo, hi = bootstrap_mean_ci(
                    [[x[k] for k in sorted(x)] for x in recs],
                    resample_cols=True)
                line += f' {m:.4f} [{lo:.4f}, {hi:.4f}] |'
        lines.append(line)
    lines.append(f'\nSkipped {skipped} result file(s) without {key}.\n' if skipped else '')
    if tuning:
        lines.append(f'\nLeft out {tuning} tuning-seed fit(s) (seed outside '
                     f'{FINAL_SEEDS.start}..{FINAL_SEEDS.stop - 1}).\n')
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
    stamped with an F2 or F3 protocol_version also epochs=N when its epochs
    differ from the F1 value of its task (the doubled-budget arms, protocol
    F2 section 3.2, point 7); F1 records never get it (e.g. pilots). Last
    the optimizer of config.effective.build when it is not the protocol
    default (optimizer=<name>, opt_epsilon=<value>; the optimizer arms of
    protocol F3 family A, #131)."""
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
    if str(eff.get('protocol_version') or '').startswith(('F2-', 'F3-')):
        f1 = F1_EPOCHS.get('person_activity' if eff.get('task')
                           == 'person_activity' else eff.get('system'))
        if f1 is not None and eff.get('epochs') != f1:
            tags.append(f"epochs={eff.get('epochs')}")
    build = eff.get('build') or {}
    if build.get('optimizer', 'rmsprop') != 'rmsprop':
        tags.append(f"optimizer={build['optimizer']}")
    if build.get('opt_epsilon') is not None:
        tags.append(f"opt_epsilon={build['opt_epsilon']}")
    return ', '.join(tags)


def arm_label(r):
    """'cell/wiring' of a runner record, a control arm with its
    _control_tag in brackets (the arm column of load_sequence_runs and of
    experiments/eval_robustness.py)."""
    tag = _control_tag(r)
    return f"{r['cell']}/{r['wiring']}" + (f' [{tag}]' if tag else '')


def _sequence_row(path, r, task, curves=False, fields=()) -> dict:
    """The load_sequence_runs row of one runner record of the task; with
    curves also the per-epoch validation curve (val_loss_curve on the
    rollouts, val_macro_f1_curve on Person Activity) and training_diverged,
    appended after the other columns; fields: top-level record keys appended
    last (the endpoint field of the F3 path, #131), a record without one
    raises SystemExit naming the file."""
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
    row = {
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
    if curves:
        key = 'val_loss_curve' if task == 'lv' else 'val_macro_f1_curve'
        row.update({key: r.get(key),
                    'training_diverged': r.get('training_diverged')})
    for k in fields:
        if k not in r:
            raise SystemExit(f'{path}: no {k} in the record')
        row[k] = r[k]
    return row


def load_sequence_runs(runs_dirs, task, curves=False) -> pd.DataFrame:
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
    the crossed bootstrap. curves=True (main with --descriptive) appends
    val_loss_curve (lv) or val_macro_f1_curve (pa) and training_diverged;
    the default frame is the one before #109. An old single-step-harness
    JSON, a pre-#62 rollout JSON and two final fits of one arm and seed
    raise SystemExit."""
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
            rows.append(_sequence_row(path, r, task, curves))
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
    out of the robustness family. A confusions_sigma_0.3 column (#96) becomes
    the column confusion_robust: (recordings sorted, draws, C, C) per fit,
    None for a fit without. A CSV without diverged_robust, two rows of one
    final fit, a row whose config_effective_sha256 is not that of the loaded
    fit's config.effective and fits with different test recordings raise
    SystemExit."""
    from src.benchmark.checkpoints import config_hash   # imports TensorFlow
    rob = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
    if 'diverged_robust' not in rob:
        raise SystemExit('robustness CSV without diverged_robust: rerun '
                         'experiments/eval_robustness.py')
    key = ['system', 'arm', 'seed']
    cols = ['metric_robust', 'diverged_robust', 'config_effective_sha256',
            *(c for c in rob if c.startswith(('f1_sigma_', 'ood_', 'long_',
                                              'rebuild_rel_dev'))
              or c == ROBUST_CONFUSIONS)]
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
    if ROBUST_CONFUSIONS in df:
        by_rec = [json.loads(v) if isinstance(v, str) else None
                  for v in df.pop(ROBUST_CONFUSIONS)]
        names = {tuple(sorted(c)) for c in by_rec if c}
        if len(names) > 1:
            raise SystemExit('robustness CSV: fits differ in their test '
                             f'recordings: {sorted(names)}')
        df['confusion_robust'] = pd.Series(
            [np.array([c[k] for k in sorted(c)], float) if c else None
             for c in by_rec], index=df.index, dtype=object)
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


def _median_diff_ci(p, task):
    """((lo, hi), units) of the bootstrap interval of the median paired
    difference v_a - v_b of a _paired frame: crossed over seeds x test
    recordings on Person Activity when every fit has its confusion matrices
    (one shape) and none diverged, else over seeds."""
    crossed = (task == 'pa' and not (p['diverged_a'] | p['diverged_b']).any()
               and all(isinstance(c, np.ndarray) for c in
                       (*p['confusion_a'], *p['confusion_b']))
               and len({c.shape for c in (*p['confusion_a'],
                                          *p['confusion_b'])}) == 1)
    if crossed:
        return (crossed_median_diff_ci(np.stack(p['confusion_a']),
                                       np.stack(p['confusion_b']), N_BOOT),
                'seed x recording')
    return (bootstrap_median_ci(p['v_a'].to_numpy(float)
                                - p['v_b'].to_numpy(float), N_BOOT), 'seed')


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
    ci, units = _median_diff_ci(p, task)
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
            'ci_units': units,
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
    without an eval-only row are left out, the interval is over seeds, or
    on Person Activity crossed over seeds x test recordings when every fit
    has its per-recording noisy confusions (confusion_robust, #96; the clean
    test confusions belong to the clean metric) and none diverged, with the
    seed interval next to it (median_diff_seed_lo/hi), and the divergence
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
                d = d.assign(metric=d[col], confusion=d.get(
                    'confusion_robust'), diverged=(
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
                if family == ROBUSTNESS_FAMILY:
                    row['median_diff_seed_lo'], row['median_diff_seed_hi'] = (
                        bootstrap_median_ci(p['v_a'].to_numpy(float)
                                            - p['v_b'].to_numpy(float),
                                            N_BOOT))
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


# RQ5 (descriptive by protocol): cfc_lrc against cfc and inner against
# outer gate, per task and wiring (#97).
RQ5_PAIRS = (('cfc_lrc', 'cfc'), ('cfc_lrc', 'cfc_lrc_outer'))
RQ5_COLUMNS = ['task', 'wiring', 'comparison', 'a', 'b', 'n_pairs',
               'n_positive', 'median_diff', 'median_diff_lo',
               'median_diff_hi', 'p_a_gt_b', 'p_a_gt_b_lo', 'p_a_gt_b_hi',
               'budget_above_tolerance']


def rq5_paired(df) -> pd.DataFrame:
    """The RQ5 comparisons (RQ5_PAIRS) as paired-by-seed differences of the
    final fits per task and wiring (plain arms only, controls never pair):
    n complete pairs, the number of seeds with v_a - v_b > 0, the median
    difference with its seed-bootstrap interval, P(A>B) with its
    Clopper-Pearson interval, and whether an arm sits above the budget
    tolerance (kept in, #59). No p-value, no significance column: the
    comparison is descriptive. A pairing without complete pairs has no row."""
    f = df[df['stage'] == 'final'] if not df.empty else df
    rows = []
    for system in sorted(f['system'].unique()) if not f.empty else ():
        for wiring in sorted(f.loc[f['system'] == system, 'wiring'].unique()):
            for ca, cb in RQ5_PAIRS:
                a, b = f'{ca}/{wiring}', f'{cb}/{wiring}'
                p = _paired(df, system, a, b)
                if p.empty:
                    continue
                d = p['v_a'].to_numpy(float) - p['v_b'].to_numpy(float)
                sign = 1 if HIGHER_IS_BETTER[p['task_a'].iloc[0]] else -1
                pw, lo, hi = p_a_gt_b(sign * p['metric_a'].to_numpy(float),
                                      sign * p['metric_b'].to_numpy(float),
                                      p['diverged_a'].to_numpy(bool),
                                      p['diverged_b'].to_numpy(bool))
                ci = bootstrap_median_ci(d, N_BOOT)
                rows.append({
                    'task': system, 'wiring': wiring,
                    'comparison': f'{ca} vs {cb}', 'a': a, 'b': b,
                    'n_pairs': len(d), 'n_positive': int((d > 0).sum()),
                    'median_diff': float(np.median(d)),
                    'median_diff_lo': ci[0], 'median_diff_hi': ci[1],
                    'p_a_gt_b': pw, 'p_a_gt_b_lo': lo, 'p_a_gt_b_hi': hi,
                    'budget_above_tolerance': bool(
                        (p['budget_above_tolerance_a']
                         | p['budget_above_tolerance_b']).any())})
    return pd.DataFrame(rows, columns=RQ5_COLUMNS)


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
                     f"{b['median_diff_hi']:+.4g}] ({b['ci_units']})"
                     + (f"; seed [{b['median_diff_seed_lo']:+.4g}, "
                        f"{b['median_diff_seed_hi']:+.4g}]"
                        if b['ci_units'] != 'seed'
                        and pd.notna(b.get('median_diff_seed_lo')) else '')
                     + f" | {b['diverged_a']} | {b['diverged_b']} ")
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
        ctl = r[r['control']]
        if 'rebuild_rel_dev' in ctl and not ctl.empty:
            lines += ['', '## Control arms: rebuild fidelity and robustness '
                      '(#99; descriptive, the controls are no family '
                      'members)\n',
                      '| Task | Arm | fits | max rebuild_rel_dev | clean '
                      'median | metric_robust median | diverged_robust |',
                      '|------|-----|------|---------------------|--------'
                      '-------|----------------------|-----------------|']
            for (system, arm), g in ctl.groupby(['system', 'arm']):
                lines.append(
                    f"| {system} | {arm} | {len(g)} "
                    f"| {_fmt(g['rebuild_rel_dev'].max())} "
                    f"| {_fmt(g['metric'].median())} "
                    f"| {_fmt(g['metric_robust'].median())} "
                    f"| {int(g['diverged_robust'].astype(bool).sum())} |")
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


def _co_arm_any(r):
    """_co_arm plus the descriptive committee arms of the suite: 'med'
    (median consensus) and 'fault_mean' (fault-aware input mean)."""
    arm = _co_arm(r)
    if (arm or r.get('view_mode') != 'identical'
            or r.get('train_corruption') != 'on'):
        return arm
    if r.get('view_average') == 'fault_mean':
        return 'fault_mean'
    if (r.get('view_average', 'off') == 'off' and r.get('vote_lambda') is None
            and r.get('consensus') == 'median'):
        return 'med'
    return None


def load_consensus_records(co_dir):
    """({(cell, wiring, arm): {seed: record}}, keys with a run above the
    budget tolerance) of the K=HD_K committee JSONs with the fault-grid
    endpoints, every arm of _co_arm_any."""
    recs, above = defaultdict(dict), set()
    for path in sorted(glob(os.path.join(co_dir, '*.json'))):
        with open(path, encoding='utf-8') as f:
            r = json.load(f)
        arm = _co_arm_any(r)
        if (arm is None or r.get('n_columns') != HD_K
                or 'test_macro_f1_by_fault' not in r):
            continue
        key = (r['cell'], r['wiring'], arm)
        recs[key][r['seed']] = r
        if r.get('budget_above_tolerance'):
            above.add(key)
    return recs, above


def load_consensus_runs(co_dir) -> dict:
    """{(cell, wiring, arm): {seed: (macro-F1 at HD_M, AUC mean over
    HD_AUC_M, record)}} of the three H-d arms (_co_arm) from
    load_consensus_records; the input-averaging monolith (any wiring) is also
    keyed (cell, None, 'avg_in'). An arm with a run sized outside the budget
    tolerance (#59) is replaced by None (it enters no confirmatory
    contrast)."""
    recs, above = load_consensus_records(co_dir)
    runs = {}
    for key, by_seed in recs.items():
        if key[2] not in ('cons', 'lam0', 'avg_in'):
            continue
        vals = {}
        for seed, r in by_seed.items():
            m = r['fault_counts']
            auc = np.mean([r['per_position_macro_f1_auc'][m.index(k)]
                           for k in HD_AUC_M])
            vals[seed] = (r['test_macro_f1_by_fault'][m.index(HD_M)],
                          float(auc), r)
        runs[key] = None if key in above else vals
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


def _hd_contrasts(runs, cell, w):
    """(part, label, arm B runs, component) of the two H-d tests of one cell
    and wiring; arm A is (cell, w, 'cons')."""
    avg = runs.get((cell, w, 'avg_in'), runs.get((cell, None, 'avg_in')))
    return (('i', f'consensus vs avg_in, macro-F1 m={HD_M}', avg, 0),
            ('ii', 'consensus vs lambda=0, per-position AUC '
                   f'm in {set(HD_AUC_M)}', runs.get((cell, w, 'lam0')), 1))


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
            lam0 = runs.get((cell, w, 'lam0'))
            for part, label, b, j in _hd_contrasts(runs, cell, w):
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


def _hd_confusions(r):
    """(sorted recording names, (recordings, C, C) test confusions at m=HD_M)
    of a committee record, (None, None) without per-recording confusions."""
    by_fault = r.get('test_confusion_by_recording_by_fault')
    if not by_fault:
        return None, None
    c = by_fault[r['fault_counts'].index(HD_M)]
    recs = sorted(c)
    return tuple(recs), np.array([c[k] for k in recs], float)


def _hd_frame(a, b, j):
    """A _paired-style frame (v, diverged, confusion per side) of the
    seeds both H-d arms share; confusions only on part i (j = 0), whose
    endpoint is the pooled macro-F1 at m=HD_M. SystemExit if two fits differ
    in their test recordings."""
    rows, names = [], set()
    for s in sorted(set(a) & set(b)):
        row = {'seed': s}
        for side, arm in (('a', a), ('b', b)):
            r = arm[s][2]
            recs, conf = _hd_confusions(r) if j == 0 else (None, None)
            if recs is not None:
                names.add(recs)
            row.update({f'v_{side}': arm[s][j],
                        f'diverged_{side}': bool(r.get('training_diverged'))
                        or not np.isfinite(arm[s][j]),
                        f'confusion_{side}': conf})
        rows.append(row)
    if len(names) > 1:
        raise SystemExit('H-d: committee fits differ in their test '
                         f'recordings: {sorted(names)}')
    return pd.DataFrame(rows)


HD_COLUMNS = ['cell', 'wiring', 'part', 'contrast', 'n_pairs', 'median_diff',
              'p_one_sided', 'p_bonferroni', 'method', 'significant',
              'status', 'median_diff_lo', 'median_diff_hi', 'ci_crossed_note',
              'median_diff_seed_lo', 'median_diff_seed_hi', 'd_z',
              'p_a_gt_b', 'p_a_gt_b_lo', 'p_a_gt_b_hi', 'n_positive',
              'n_diverged']


def consensus_intervals(co_dir) -> pd.DataFrame:
    """One row per H-d test of consensus_family (columns HD_COLUMNS): its
    n pairs, median difference, one-sided p, Bonferroni p, method and
    significance as there (the same _one_sided), plus the median paired
    difference A - B with the crossed seeds x test recordings bootstrap
    interval (median_diff_lo/hi; part i only, from the per-recording
    confusions at m=HD_M of every paired fit, diverged ones included since
    H-d pairs the raw values; else empty with ci_crossed_note) and the seed
    bootstrap interval next to it, d_z, P(A>B) with its Clopper-Pearson
    interval (a diverged or non-finite fit loses) and the seeds with
    A - B > 0 (#94)."""
    runs = load_consensus_runs(co_dir)
    n_tests, rows = BONFERRONI_FAMILIES['H-d'], []
    for cell in sorted({c for c, _, _ in runs}):
        for w in HD_WIRINGS:
            cons = runs.get((cell, w, 'cons'))
            for part, label, b, j in _hd_contrasts(runs, cell, w):
                row = {'cell': cell, 'wiring': w, 'part': part,
                       'contrast': label, 'n_pairs': 0}
                rows.append(row)
                if cons is None or b is None:
                    row['status'] = ('missing arm or arm above budget '
                                     'tolerance')
                    continue
                n, med, p, method = _one_sided(cons, b, j)
                row.update(n_pairs=n, median_diff=med)
                if p is None:
                    row['status'] = ('undecidable' if n < MIN_PAIRS
                                     else 'all differences zero')
                else:
                    p_adj = min(1.0, p * n_tests)
                    row.update(p_one_sided=p, p_bonferroni=p_adj,
                               method=method, significant=p_adj < ALPHA,
                               status='tested')
                if not n:
                    continue
                f = _hd_frame(cons, b, j)
                x, y = f['v_a'].to_numpy(float), f['v_b'].to_numpy(float)
                conf = [*f['confusion_a'], *f['confusion_b']]
                if j:
                    row['ci_crossed_note'] = ('no per-recording endpoint '
                                              '(per-position AUC)')
                elif (any(c is None for c in conf)
                      or len({c.shape for c in conf}) > 1):
                    row['ci_crossed_note'] = ('per-recording confusions '
                                              'missing or ragged')
                else:
                    row['median_diff_lo'], row['median_diff_hi'] = (
                        crossed_median_diff_ci(np.stack(f['confusion_a']),
                                               np.stack(f['confusion_b']),
                                               N_BOOT))
                lo, hi = bootstrap_median_ci(x - y, N_BOOT)
                pw, plo, phi = p_a_gt_b(x, y, f['diverged_a'].to_numpy(bool),
                                        f['diverged_b'].to_numpy(bool))
                row.update(median_diff_seed_lo=lo, median_diff_seed_hi=hi,
                           d_z=(_cohens_d_paired(x, y) if n > 1
                                else float('nan')),
                           p_a_gt_b=pw, p_a_gt_b_lo=plo, p_a_gt_b_hi=phi,
                           n_positive=int((x - y > 0).sum()),
                           n_diverged=int(f['diverged_a'].sum()
                                          + f['diverged_b'].sum()))
    return pd.DataFrame(rows).reindex(columns=HD_COLUMNS).astype(
        {'n_positive': 'Int64', 'n_diverged': 'Int64'})


def consensus_arm_endpoints(co_dir) -> pd.DataFrame:
    """The secondary endpoints of the consensus study (F1 protocol, section
    4) per K=HD_K arm of load_consensus_records: the test macro-F1 mean over
    the fault grid (endpoint grid_mean) and steps_to_90 per fault count m,
    each as the median over seeds with its seed bootstrap interval; an arm
    above the budget tolerance stays in, flagged (#94)."""
    recs, above = load_consensus_records(co_dir)
    rows = []
    for key in sorted(recs, key=str):
        by_seed = [recs[key][s] for s in sorted(recs[key])]
        values = {('grid_mean', None): [float(np.mean(
            r['test_macro_f1_by_fault'])) for r in by_seed]}
        for r in by_seed:
            for m, v in zip(r['fault_counts'], r.get('steps_to_90') or []):
                values.setdefault(('steps_to_90', m), []).append(float(v))
        for (endpoint, m), v in values.items():
            lo, hi = bootstrap_median_ci(v, N_BOOT)
            rows.append({'cell': key[0], 'wiring': key[1], 'arm': key[2],
                         'endpoint': endpoint, 'm': m, 'n_seeds': len(v),
                         'median': float(np.median(v)), 'lo': lo, 'hi': hi,
                         'budget_above_tolerance': key in above})
    return pd.DataFrame(rows, columns=[
        'cell', 'wiring', 'arm', 'endpoint', 'm', 'n_seeds', 'median', 'lo',
        'hi', 'budget_above_tolerance']).astype({'m': 'Int64'})


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


def _check_spec_arm(path, where, arm, ppo, empty_ok=False):
    """Check one arm of a spec in place (keys F2_ARM_KEYS, protocol_version
    a string or a list of strings, null on PPO, normalised to a list; paths
    a list of existing run folders or record files without patterns, empty
    only with empty_ok); SystemExit naming path and where."""
    if not isinstance(arm, dict) or set(arm) != F2_ARM_KEYS:
        raise SystemExit(f'{path}: {where} needs exactly the '
                         f'keys {sorted(F2_ARM_KEYS)}')
    pv = arm['protocol_version']
    if ppo:
        ok = pv is None
    else:
        pv = [pv] if isinstance(pv, str) else pv
        ok = (isinstance(pv, list) and pv
              and all(isinstance(v, str) for v in pv))
    if not ok:
        raise SystemExit(f'{path}: {where}: protocol_version '
                         'must be a string or a list of strings '
                         '(null on PPO)')
    arm['protocol_version'] = pv
    paths = arm['paths']
    if (not isinstance(paths, list) or not (paths or empty_ok)
            or not all(isinstance(p, str) for p in paths)):
        raise SystemExit(f'{path}: {where}: paths must be a '
                         'non-empty list of run folders or files')
    for p in paths:
        if has_magic(p):
            raise SystemExit(f'{path}: {where}: pattern in '
                             f'path {p!r}; list the folders or files')
        if not os.path.exists(p):
            raise SystemExit(f'{path}: {where}: no such path '
                             f'{p!r}')


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
            _check_spec_arm(path, f'{test}.{side}', arm, task in F2_PPO_TASKS)
            if arm['label'] != label:
                raise SystemExit(f'{path}: {test}.{side}: label '
                                 f'{arm["label"]!r}, expected {label!r}')
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
    """(DataFrame, skipped files) of the run_mujoco_benchmark.py records
    (#61) below paths: one row per record with profile 'final', the task
    and a final seed, the PPO_FIELDS plus arm (= model), metric
    (eval_return_mean), diverged (a non-finite metric) and file; records of
    another profile or task are skipped and listed. Two kept records of one
    (model, seed) or a kept record without one of the PPO_FIELDS (or with
    null untrained values) raise SystemExit."""
    rows, skipped = [], []
    for path in _record_files(paths):
        with open(path, encoding='utf-8') as f:
            r = json.load(f)
        if not isinstance(r, dict) or 'model' not in r:
            continue
        if r.get('profile') != 'final' or r.get('task') != task:
            skipped.append(path)
            continue
        if r.get('seed') not in FINAL_SEEDS:
            continue
        missing = [k for k in PPO_FIELDS if k not in r] + [
            k for k in ('untrained_return_mean', 'untrained_return_std')
            if k in r and r[k] is None]
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


def _check_protocol(path, eff, versions, name):
    """SystemExit naming the file unless config.effective.protocol_version
    of a record is one its arm lists (F2 and F3 path lists)."""
    if eff.get('protocol_version') not in versions:
        raise SystemExit(f'{path}: protocol_version '
                         f'{eff.get("protocol_version")!r}, but {name} '
                         f'lists {versions}')


def _floor_mark(path, r, eff, row, name, draws):
    """(below, floor) of one _sequence_row fit: a rollout fit below the
    floor of its own data draw (floor = that floor line), a Person Activity
    fit at chance (row metric against the one-class value of its test
    confusions; floor None). A fit without a marking raises SystemExit
    naming the arm."""
    if row['task'] == 'lv':
        m = r['rollout_metrics_test']
        fl = (None if any(k not in m for k in (
            'anchors', 'anchor_horizon', 'anchor_theta')) or any(
            eff.get(k) is None for k in FLOOR_DATA_KEYS)
            else fit_floor(r, draws))
        if fl is None or fl[1] is None or not np.isfinite(fl[1]):
            raise SystemExit(f'{name}: no floor marking for {path}')
        return not fl[0] >= fl[1], fl[1]
    if row['confusion'] is None:
        raise SystemExit(f'{name}: no test confusion matrices (chance '
                         f'marking) in {path}')
    return at_chance(row['metric'], one_class_macro_f1(
        row['confusion'].sum(axis=(0, 2)))), None


def _f2_arm(test, side, task, arm, draws, curves=False, empty_ok=False):
    """Final fits of one F2 arm: the records in its paths whose label is the
    arm label, each with the declared protocol_version (else SystemExit
    naming the file), plus data_key, width_matched, below (the per-fit
    floor marking; a fit without one raises SystemExit naming the arm),
    floor (the rollout floor of the fit's data draw, None on Person
    Activity) and protocol_version. curves: the validation curves of
    _sequence_row; empty_ok: an arm without a final fit is an empty frame
    instead of SystemExit (sequence runners only)."""
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
        _check_protocol(path, eff, versions, name)
        row = _sequence_row(path, r, runner, curves)
        below, floor = _floor_mark(path, r, eff, row, name, draws)
        rows.append({**row, 'width_matched': _width_matched(r),
                     'data_key': eff.get('split_seed' if runner == 'pa'
                                         else 'data_seed'),
                     'below': bool(below), 'floor': floor,
                     'protocol_version': eff.get('protocol_version')})
    df = pd.DataFrame(rows)
    if df.empty and empty_ok:
        return df, []
    if df.empty:
        raise SystemExit(f'{name}: no final fit in its paths')
    dup = df[df.duplicated('seed', keep=False)]
    if not dup.empty:
        raise SystemExit(f'two final fits of {name} and one seed: '
                         + ', '.join(dup['file']))
    return df, []


def _signed_ranks(x, y):
    """Signs and mid-ranks of |d| of the nonzero paired differences x - y."""
    d = np.asarray(x, float) - np.asarray(y, float)
    nz = d[d != 0]
    return np.sign(nz), rankdata(np.abs(nz))


def permutation_p(x, y, alternative='two-sided') -> float:
    """Exact sign-flip p of the signed-rank statistic: W+ (sum of the
    mid-ranks of |d| over the k nonzero differences with d > 0) against all
    2^k sign patterns; two-sided by the distance of W+ from its mean
    sum(ranks) / 2, 'greater' as the share of patterns with W+ >= the
    observed W+. Without ties and zeros it equals the exact Wilcoxon p;
    1.0 without a nonzero difference. The null distribution is counted
    exactly over the integer sums 2 W+ (mid-ranks are multiples of 1/2), so
    k = 25 needs a few kB instead of the 2^k x k pattern matrix."""
    if alternative not in ('two-sided', 'greater'):
        raise ValueError(f'alternative {alternative!r}')
    sign, ranks = _signed_ranks(x, y)
    if not ranks.size:
        return 1.0
    counts = np.zeros(int(np.rint(2 * ranks.sum())) + 1, np.int64)
    counts[0] = 1
    for r2 in np.rint(2 * ranks).astype(int):
        counts[r2:] = counts[r2:] + counts[:-r2]   # RHS built before the write
    w = np.arange(counts.size) / 2               # every reachable W+
    obs = ranks[sign > 0].sum()
    if alternative == 'greater':
        hit = w >= obs - 1e-9
    else:
        mu = ranks.sum() / 2
        hit = np.abs(w - mu) >= abs(obs - mu) - 1e-9
    return float(counts[hit].sum() / 2 ** ranks.size)


def _check_data_key(p, name):
    """SystemExit unless both fits of every pair of a merged frame share the
    data key of config.effective (data_seed, Person Activity split_seed)."""
    bad = p[p['data_key_a'].isna() | (p['data_key_a'] != p['data_key_b'])]
    if not bad.empty:
        raise SystemExit(f'{name}: arms A and B differ in the data key of '
                         f'config.effective on seed(s) '
                         f'{bad["seed"].tolist()}')


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
        _check_data_key(p, test)
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
                row['n_skipped_records'] = len(set(skip_a) | set(skip_b))
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


# --- descriptive F2 blocks (protocol F2 sections 5 and 2.3, #109) ----------

DESC_COLUMNS = [
    'block', 'task', 'reading', 'a', 'b', 'status', 'not_run', 'origin', 'n',
    'median_a', 'median_a_lo', 'median_a_hi', 'median_b', 'median_b_lo',
    'median_b_hi', 'median_diff', 'median_diff_lo', 'median_diff_hi',
    'ci_units', 'n_positive', 'interval_excludes_0', 'per_seed',
    'n_satisfied', 'summary', 'direction', 'median_sign', 'majority',
    'median_diff_1x', 'median_sign_1x', 'n_positive_1x', 'n_1x',
    'majority_1x', 'vpt8_median_a', 'vpt8_median_diff', 'vpt8_n_positive',
    'seeds_below_floor_a', 'not_learned_a', 'n_flagged_a', 'n_flagged_b',
    'comparator_flagged', 'best_val_epoch_a', 'best_val_epoch_b',
    'escape_epoch_a', 'text', 'note', 'caveat', 'config_note']
THREE_SEED = {3: 'stable', 2: 'mixed'}    # else 'not supported'


def _block_labels(block):
    """The arm labels a block reads, in table order."""
    out = []
    for pair in DESCRIPTIVE_BLOCKS[block]['pairs']:
        for k in ('a', 'b', 'a_1x', 'b_1x'):
            if pair.get(k) and pair[k] not in out:
                out.append(pair[k])
    return out


def load_descriptive_spec(path) -> dict:
    """The --descriptive spec: {block: [arm, ...]} for exactly the blocks of
    DESCRIPTIVE_BLOCKS, one arm per label the block reads (_block_labels),
    each checked as in load_f2_spec; paths may be empty (an arm that did not
    run, its rows become 'not run'). SystemExit as in load_f2_spec."""
    with open(path, encoding='utf-8') as f:
        spec = yaml.safe_load(f)
    blocks = set(DESCRIPTIVE_BLOCKS)
    if not isinstance(spec, dict) or set(spec) != blocks:
        got = set(spec) if isinstance(spec, dict) else set()
        raise SystemExit(f'{path}: the spec must hold exactly the blocks '
                         f'{sorted(blocks)} (missing {sorted(blocks - got)}, '
                         f'unknown {sorted(got - blocks)})')
    for block, arms in spec.items():
        need = _block_labels(block)
        if not isinstance(arms, list):
            raise SystemExit(f'{path}: {block} must be a list of arms')
        for i, arm in enumerate(arms):
            _check_spec_arm(path, f'{block}[{i}]', arm, False, empty_ok=True)
        got = [arm['label'] for arm in arms]
        if sorted(got) != sorted(need):
            raise SystemExit(f'{path}: {block} needs exactly the arms {need}, '
                             f'got {got}')
    return spec


def _seed_runs(seeds):
    """'0-4, 7' for the seeds 0, 1, 2, 3, 4, 7."""
    seeds, out = sorted(int(s) for s in seeds), []
    for s in seeds:
        if out and s == out[-1][1] + 1:
            out[-1][1] = s
        else:
            out.append([s, s])
    return ', '.join(str(a) if a == b else f'{a}-{b}' for a, b in out)


def _by_seed(seeds, values):
    return '; '.join(f'{int(s)}: {v}' for s, v in zip(seeds, values))


def _origin(label, df):
    """'label: <protocol_version> seeds <runs>[, ...]' of an arm's fits."""
    parts = [f'{pv} seeds {_seed_runs(g["seed"])}'
             for pv, g in df.groupby('protocol_version', sort=True)]
    return f'{label}: ' + ', '.join(parts)


def _escape_epoch(curve):
    """First epoch (1-based) with a validation loss below ESCAPE_LOSS,
    'none' if it never falls below."""
    return next((i + 1 for i, v in enumerate(curve or [])
                 if v is not None and v < ESCAPE_LOSS), 'none')


def b5_seed_reading(curve, diverged):
    """'not readable' | 'satisfied' | 'not satisfied' of one P2b fit
    (protocol F2 5.1, B5): not readable if training diverged or any value
    of epochs 1-B5_EPOCHS of val_macro_f1_curve is non-finite (or missing);
    satisfied if their maximum is strictly above B5_EXIT_THRESHOLD."""
    c = np.asarray([np.nan if v is None else v
                    for v in (curve or [])[:B5_EPOCHS]], float)
    if diverged or c.size < B5_EPOCHS or not np.isfinite(c).all():
        return 'not readable'
    return 'satisfied' if c.max() > B5_EXIT_THRESHOLD else 'not satisfied'


def b5_verdict(readings) -> str:
    """Arm verdict of B5 from {seed: b5_seed_reading}."""
    readable = {s: r for s, r in readings.items() if r != 'not readable'}
    if len(readable) < 2:
        return 'not readable'
    k = sum(r == 'satisfied' for r in readable.values())
    verdict = ('drive parametrisation discriminates' if k == len(readable)
               else 'init gain not sufficient' if k == 0
               else 'mixed, no statement')
    if len(readable) < 3:
        verdict += f' (readable seeds {_seed_runs(readable)} only)'
    return verdict


def direction5(d, n_seeds) -> str:
    """Five-seed direction rule (protocol F2 5.2, BA-2): a direction only if
    every seed of the block has a complete pair and all differences share
    one strict sign; 4 of 5 or a zero difference is no direction."""
    d = np.asarray(d, float)
    if len(d) == n_seeds and (d > 0).all():
        return 'delta > 0'
    if len(d) == n_seeds and (d < 0).all():
        return 'delta < 0'
    return 'no direction'


def _majority(d):
    d = np.asarray(d, float)
    return ('delta > 0' if (d > 0).sum() > len(d) / 2 else 'delta < 0'
            if (d < 0).sum() > len(d) / 2 else 'none')


def _flagged(g):
    """Fits whose best validation epoch is >= CONVERGED x their epochs."""
    return int((g['best_val_epoch'].astype(float)
                >= CONVERGED * g['epochs'].astype(float)).sum())


def _valued(g):
    """g with the paired value v (a diverged fit at the worst value)."""
    return g.assign(v=np.where(g['diverged'], g['task'].map(WORST),
                               g['metric'])).sort_values('seed')


def _desc_row(block, spec_block, pair, arms) -> dict:
    """One descriptive_blocks row: block, arm pairing and every column of
    its reading (DESC_COLUMNS)."""
    task, reading = spec_block['task'], spec_block['reading']
    seeds = list(spec_block['seeds'])
    labels = [pair[k] for k in ('a', 'b', 'a_1x', 'b_1x') if pair.get(k)]
    row = {'block': block, 'task': task, 'reading': reading,
           'a': pair['a'], 'b': pair['b'],
           **{k: pair.get(k) for k in ('note', 'caveat', 'config_note')}}
    if reading == 'direction':
        row['text'] = DIRECTION_TEXT
    missing = [lb for lb in labels if arms[lb].empty]
    if missing:
        row.update(status='not run', not_run=', '.join(missing))
        return row
    if reading not in ('floor + 2', 'above reference', 'plateau exit'):
        # five-seed blocks: fewer than MIN_PAIRS pairs by design
        row['status'] = ('undecidable' if len(seeds) < MIN_PAIRS
                         else 'descriptive')
    row['origin'] = '; '.join(_origin(lb, arms[lb]) for lb in labels)
    a = _valued(arms[pair['a']])
    lo, hi = bootstrap_median_ci(a['v'], N_BOOT)
    row.update(median_a=float(np.median(a['v'])), median_a_lo=lo,
               median_a_hi=hi, n_flagged_a=_flagged(a),
               best_val_epoch_a=_by_seed(a['seed'], a['best_val_epoch']),
               seeds_below_floor_a=_seed_runs(a.loc[a['below'], 'seed'])
               or 'none',
               not_learned_a=not_learned(int(a['below'].sum()), len(a)))
    if task != 'pa':
        row['escape_epoch_a'] = _by_seed(
            a['seed'], map(_escape_epoch, a['val_loss_curve']))
    if reading == 'floor + 2':
        ok = a['v'] > a['floor'].astype(float) + 2
        row['per_seed'] = _by_seed(a['seed'], np.where(ok, 'yes', 'no'))
        row['n_satisfied'] = int(ok.sum())
        row['summary'] = THREE_SEED.get(int(ok.sum()), 'not supported')
    if reading == 'plateau exit':
        r = {s: b5_seed_reading(c, dv) for s, c, dv in zip(
            a['seed'], a['val_macro_f1_curve'], a['training_diverged'])}
        row.update(per_seed=_by_seed(r, r.values()),
                   n_satisfied=sum(v == 'satisfied' for v in r.values()),
                   summary=b5_verdict(r))
    if reading == 'descriptive' and 'vpt_anchor_8' in a:
        row['vpt8_median_a'] = float(np.median(a['vpt_anchor_8']))
    if pair['b'] is None:
        return row
    b = _valued(arms[pair['b']])
    lo, hi = bootstrap_median_ci(b['v'], N_BOOT)
    row.update(median_b=float(np.median(b['v'])), median_b_lo=lo,
               median_b_hi=hi, n_flagged_b=_flagged(b),
               best_val_epoch_b=_by_seed(b['seed'], b['best_val_epoch']))
    p = _paired(pd.concat([arms[pair['a']], arms[pair['b']]],
                          ignore_index=True), task, pair['a'], pair['b'])
    _check_data_key(p, f"{block} {pair['a']} - {pair['b']}")
    d = (p['v_a'] - p['v_b']).to_numpy(float)
    row.update(n=len(p), n_positive=int((d > 0).sum()))
    if len(p):
        (lo, hi), units = _median_diff_ci(p, p['task_a'].iloc[0])
        row.update(median_diff=float(np.median(d)), median_diff_lo=lo,
                   median_diff_hi=hi, ci_units=units)
    if reading == 'descriptive':
        row['interval_excludes_0'] = (bool(not lo <= 0 <= hi) if len(p)
                                      else None)
    if reading == 'above reference':
        k = int((d > 0).sum())
        row.update(per_seed=_by_seed(p['seed'], np.where(d > 0, 'yes', 'no')),
                   n_satisfied=k,
                   summary=THREE_SEED.get(k, 'not supported'))
        if 'vpt_anchor_8_a' in p:
            d8 = (p['vpt_anchor_8_a'] - p['vpt_anchor_8_b']).to_numpy(float)
            row.update(vpt8_median_diff=float(np.median(d8)) if len(d8)
                       else None, vpt8_n_positive=int((d8 > 0).sum()))
    if reading == 'direction':
        row['direction'] = direction5(d, len(seeds))
    if reading == 'budget direction':
        p1 = _paired(pd.concat([arms[pair['a_1x']], arms[pair['b_1x']]],
                               ignore_index=True), task, pair['a_1x'],
                     pair['b_1x'])
        _check_data_key(p1, f"{block} {pair['a_1x']} - {pair['b_1x']}")
        d1 = (p1['v_a'] - p1['v_b']).to_numpy(float)
        row.update(median_sign=float(np.sign(np.median(d))) if len(d)
                   else None, majority=_majority(d),
                   median_diff_1x=float(np.median(d1)) if len(d1) else None,
                   median_sign_1x=float(np.sign(np.median(d1))) if len(d1)
                   else None, n_positive_1x=int((d1 > 0).sum()),
                   n_1x=len(d1), majority_1x=_majority(d1),
                   comparator_flagged=row['n_flagged_b'] > 0)
        row['direction'] = (None if not (len(d) and len(d1)) else
                            'unchanged' if (row['median_sign'],
                                            row['majority']) == (
                                row['median_sign_1x'], row['majority_1x'])
                            else 'changed')
    return row


def descriptive_blocks(spec) -> pd.DataFrame:
    """One row per block, arm and pairing of DESCRIPTIVE_BLOCKS (columns
    DESC_COLUMNS) from a load_descriptive_spec spec: the final fits of the
    block's seeds from each arm's own path list (protocol_version and floor
    checks of _f2_arm), the per-arm median with a seed-bootstrap interval,
    the paired difference a - b (median, 10,000-resample interval as in
    _pair_stats, seeds with delta > 0, n; both fits of a pair on one data
    key) and the reading of the block. An arm listed with paths [] makes
    its rows 'not run'; listed paths without a final fit of the arm on the
    block seeds raise SystemExit. No test, no p."""
    draws, rows = {}, []
    for block, sb in DESCRIPTIVE_BLOCKS.items():
        arms = {}
        for i, arm in enumerate(spec[block]):
            df, _ = _f2_arm(block, i, sb['task'], arm, draws, curves=True,
                            empty_ok=not arm['paths'])
            if not df.empty:
                df = df[df['seed'].isin(sb['seeds'])]
                if df.empty:
                    raise SystemExit(
                        f"{block}[{i}] ({arm['label']}): no final fit on the "
                        f"block seeds {_seed_runs(sb['seeds'])} in its paths")
            arms[arm['label']] = df
        rows += [_desc_row(block, sb, pair, arms) for pair in sb['pairs']]
    return pd.DataFrame(rows).reindex(columns=DESC_COLUMNS)


def descriptive_report(blocks) -> str:
    """Markdown of descriptive_blocks: one table per block with the columns
    that carry a value in that block."""
    lines = ['## Descriptive F2 blocks (protocol F2 sections 5 and 2.3; no '
             'test, no p; delta = a - b, a diverged fit at the worst value; '
             'median intervals: 10,000-resample bootstrap)\n']
    for block, g in blocks.groupby('block', sort=False):
        cols = [c for c in DESC_COLUMNS if c != 'block' and g[c].notna().any()]
        lines += [f'### {block}\n', '| ' + ' | '.join(cols) + ' |',
                  '|' + '---|' * len(cols)]
        for r in g.to_dict('records'):
            lines.append('| ' + ' | '.join(
                _fmt(r[c]) if isinstance(r[c], float) else
                '—' if r[c] is None else str(r[c]) for c in cols) + ' |')
        lines.append('')
    lines.append(f'Three-seed rule: stable 3 of 3, mixed 2 of 3, not '
                 f'supported at most 1 of 3. B5 exit threshold '
                 f'{B5_EXIT_THRESHOLD:g} (val macro-F1, epochs '
                 f'1-{B5_EPOCHS}). '
                 f'Convergence flag: best validation epoch >= {CONVERGED:g} '
                 f'x the epochs of the record. Escape epoch: first epoch with '
                 f'validation loss < {ESCAPE_LOSS:g}.\n')
    return '\n'.join(lines)


# --- F3 test families (protocol F3, draft, #131) ----------------------------

# family -> {'status', 'tests': {test: entry}}: the one place where family
# membership, test kind, side, endpoint, floor rule, learning-success rule,
# match fields and status are defined. delta = A - B, higher is better on
# every endpoint. Entry keys:
#   task      rollout system or 'pa'
#   a, b      arm labels as arm_label returns them
#   kind      'wilcoxon' (paired signed-rank) | 'mcnemar' (exact, on the
#             learning success of the fits)
#   side      'two-sided' | 'greater'
#   endpoint  rollouts: 'vpt_multi_median'; Person Activity: the record
#             field, 'test_macro_f1' (best-validation epoch) or
#             'test_macro_f1_last' (last epoch, #138); no default
#   floor     'exclusive' (F1 rule: a floor-bound row reads floor-bound) |
#             'informational' (marking reported, reading stays PASS/FAIL)
#   success   optional learning success per fit (learning_success):
#             ('threshold', t) = endpoint >= t; ('floor+', d) = VPT-multi
#             median >= the fit's own floor + d
#   match     optional fields (of lr, clipnorm, epochs, param_budget) equal
#             within every seed pair, else SystemExit; without it reported
#   status    optional, overrides the family's 'required' | 'conditional'
#             | 'optional'
# Values follow the F3 draft; a change before the freeze is a registry PR.
# Families B (#132) and E (#133) plug in here.
F3_CONFIG_FIELDS = ('lr', 'clipnorm', 'epochs', 'param_budget')
_LEGACY = ' [optimizer=rmsprop_legacy]'
_VPT = 'vpt_multi_median'


def _f3(task, a, b, endpoint, floor, kind='wilcoxon', side='two-sided',
        **extra):
    return {'task': task, 'a': a, 'b': b, 'kind': kind, 'side': side,
            'endpoint': endpoint, 'floor': floor, **extra}


F3_FAMILIES = {
    # A: one fixed recipe for all arms (no arm-wise tuning), learning
    # success at the end of the budget (draft 4.1, E8)
    'A': {'status': 'required', 'tests': {
        'T-A1': _f3('pa', 'lrc/ncp' + _LEGACY, 'lrc/ncp', 'test_macro_f1_last',
                    'informational', kind='mcnemar',
                    success=('threshold', 0.30), match=F3_CONFIG_FIELDS),
        'T-A2': _f3('pa', 'lrc_cal/ncp' + _LEGACY, 'lrc/ncp' + _LEGACY,
                    'test_macro_f1_last', 'informational',
                    match=F3_CONFIG_FIELDS)}},
    'C': {'status': 'required', 'tests': {
        'T-C1': _f3(_LV, 'cfc_lrc/ncp', 'gru/ncp', _VPT, 'exclusive',
                    match=('epochs', 'param_budget')),
        'T-C2': _f3(_LV, 'cfc_lrc/ncp', 'lstm/ncp', _VPT, 'exclusive',
                    match=('epochs', 'param_budget'))}},
    'D': {'status': 'required', 'tests': {
        'T-D1': _f3('pa', 'cfc_lrc/ncp', _EPS1, 'test_macro_f1', 'exclusive'),
        'T-D2': _f3('pa', 'cfc_lrc/ncp', 'cfc_lrc_outer/ncp', 'test_macro_f1',
                    'exclusive')}},
    'F': {'status': 'required', 'tests': {
        'T-F1': _f3('duffing', 'lrc_init/ncp', 'lrc/ncp', _VPT,
                    'informational', success=('floor+', 2.0),
                    match=('epochs', 'param_budget')),
        'T-F2': _f3('duffing', 'lrc_cal_duffing/ncp', 'lrc/ncp', _VPT,
                    'informational', success=('floor+', 2.0),
                    match=('epochs', 'param_budget'))}},
    'G': {'status': 'conditional', 'tests': {
        'T-G1': _f3(_LV, 'cfc_lrc/cncp', 'cfc_lrc/cncp_ff', _VPT,
                    'exclusive'),
        'T-G2': _f3('pa', 'cfc_lrc/cncp', 'cfc_lrc/cncp_ff', 'test_macro_f1',
                    'exclusive')}},
    'H': {'status': 'optional', 'tests': {
        'T-H1': _f3(_LV, 'cfc_mm_lrc/ncp', 'cfc_lrc/ncp', _VPT, 'exclusive',
                    side='greater'),
        'T-H2': _f3('pa', 'cfc_mm_lrc/ncp', 'cfc_lrc/ncp', 'test_macro_f1',
                    'exclusive', side='greater')}},
}
F3_TESTS = {test: {**t, 'family': family,
                   'status': t.get('status', f['status'])}
            for family, f in F3_FAMILIES.items()
            for test, t in f['tests'].items()}
F3_TEST_KEYS = {'seeds', 'a', 'b'}


def load_f3_spec(path) -> dict:
    """The --f3-families spec: {test: {'seeds': [...], 'a': arm, 'b': arm}}
    for the tests of F3_TESTS; every required test, the other tests of one
    family and status only all together or not at all (a conditional or
    optional family is left out as a whole, a lone optional test alone).
    seeds: the explicit final seeds of the test (distinct integers); arm as
    in load_f2_spec (keys F2_ARM_KEYS, label as in the registry), with F3
    stamps only. Unknown or missing keys, a pattern, a missing path and a
    non-F3 protocol_version raise SystemExit."""
    with open(path, encoding='utf-8') as f:
        spec = yaml.safe_load(f)
    if not isinstance(spec, dict):
        raise SystemExit(f'{path}: the spec must map tests to entries')
    unknown = sorted(set(spec) - set(F3_TESTS))
    missing = [t for t, e in F3_TESTS.items()
               if e['status'] == 'required' and t not in spec]
    if unknown or missing:
        raise SystemExit(f'{path}: unknown tests {unknown}, missing required '
                         f'tests {missing}')
    groups = defaultdict(list)
    for t, e in F3_TESTS.items():
        if e['status'] != 'required':
            groups[(e['family'], e['status'])].append(t)
    for (family, status), tests in groups.items():
        got = [t for t in tests if t in spec]
        if got and got != tests:
            raise SystemExit(f'{path}: the {status} tests {tests} of family '
                             f'{family} are left out only together '
                             f'(missing {sorted(set(tests) - set(got))})')
    for test, entry in spec.items():
        if not isinstance(entry, dict) or set(entry) != F3_TEST_KEYS:
            raise SystemExit(f'{path}: {test} needs exactly the keys '
                             f'{sorted(F3_TEST_KEYS)}')
        seeds = entry['seeds']
        if (not isinstance(seeds, list) or not seeds
                or not all(type(s) is int for s in seeds)
                or len(set(seeds)) != len(seeds)):
            raise SystemExit(f'{path}: {test}.seeds must be a non-empty list '
                             'of distinct integers')
        for side in 'ab':
            arm = entry[side]
            _check_spec_arm(path, f'{test}.{side}', arm, False)
            if arm['label'] != F3_TESTS[test][side]:
                raise SystemExit(f'{path}: {test}.{side}: label '
                                 f'{arm["label"]!r}, expected '
                                 f'{F3_TESTS[test][side]!r}')
            bad = [v for v in arm['protocol_version']
                   if not v.startswith('F3-')]
            if bad:
                raise SystemExit(f'{path}: {test}.{side}: protocol_version '
                                 f'{bad} is not an F3 stamp')
    return spec


def learning_success(v, diverged, threshold) -> bool:
    """Learning success of one fit (protocol F3, E8): not diverged and
    v >= threshold (a non-finite v fails). The descriptive B1 block of F2
    uses a strict > instead."""
    return bool(not diverged and v >= threshold)


def _f3_arm(test, side, entry, arm, seeds, draws, key=('seed',)):
    """(fits, outside) of one F3 arm. Every record in its paths with the
    arm label must carry a protocol_version the arm lists, whatever its
    seed (_check_protocol); a record whose seed is not in the test's seed
    list is not used, its seed goes to outside. fits: one _sequence_row per
    used record with the endpoint as metric (Person Activity: the registry
    field; a diverged fit = non-finite endpoint; the crossed interval only
    on the best-validation field its confusions belong to), below and floor
    (_floor_mark on that metric), success (with a registry success rule),
    data_key, width_matched, param_budget and protocol_version; an empty
    frame when no listed seed has a record. Two records of one key raise
    SystemExit."""
    task, endpoint = entry['task'], entry['endpoint']
    runner = 'pa' if task == 'pa' else 'lv'
    label, versions = arm['label'], arm['protocol_version']
    name = f'{test}.{side} ({label})'
    rows, outside = [], []
    for path in _record_files(arm['paths']):
        with open(path, encoding='utf-8') as f:
            r = json.load(f)
        if (not isinstance(r, dict) or _record_task(r) != runner
                or (runner == 'lv' and r.get('system') != task)
                or arm_label(r) != label):
            continue
        eff = (r.get('config') or {}).get('effective') or {}
        _check_protocol(path, eff, versions, name)
        if r.get('seed') not in seeds:
            outside.append(r.get('seed'))
            continue
        row = _sequence_row(path, r, runner,
                            fields=(endpoint,) if runner == 'pa' else ())
        if runner == 'pa':
            v = row[endpoint]
            v = float('nan') if v is None else float(v)
            row.update(metric=v, diverged=not np.isfinite(v))
        below, floor = _floor_mark(path, r, eff, row, name, draws)
        if runner == 'pa' and endpoint != 'test_macro_f1':
            row['confusion'] = None     # confusions of the best-val epoch
        if 'success' in entry:
            rule, t = entry['success']
            row['success'] = learning_success(
                row['metric'], row['diverged'],
                t if rule == 'threshold' else floor + t)
        rows.append({**row, 'width_matched': _width_matched(r),
                     'data_key': eff.get('split_seed' if runner == 'pa'
                                         else 'data_seed'),
                     'below': bool(below), 'floor': floor,
                     'param_budget': r.get('param_budget'),
                     'protocol_version': eff.get('protocol_version')})
    df = pd.DataFrame(rows)
    if not df.empty:
        dup = df[df.duplicated(list(key), keep=False)]
        if not dup.empty:
            raise SystemExit(f'two records of {name} and one '
                             f'{"/".join(key)}: ' + ', '.join(dup['file']))
    return df, outside


def _config_values(d, field):
    """'v1;v2' of the distinct values of a config field over an arm's fits,
    None for an empty arm."""
    return ';'.join(sorted({str(v) for v in d[field]})) if len(d) else None


def _check_match(p, fields, name):
    """SystemExit naming the seed(s) unless both fits of every pair of a
    merged frame have equal values in each of fields (None == None)."""
    for f in fields:
        bad = [s for s, x, y in zip(p['seed'], p[f'{f}_a'], p[f'{f}_b'])
               if not (x == y or (pd.isna(x) and pd.isna(y)))]
        if bad:
            raise SystemExit(f'{name}: arms A and B differ in {f} on seed(s) '
                             f'{bad}')


def _f3_test(test, entry, m, m_f3, a, b, seeds, outside, key=('seed',)):
    """One F3 test row (F3_COLUMNS) from the fits of arm A and arm B on the
    test's seed list, paired on key (seed; family E pairs on fold and seed,
    #133)."""
    kind, side = entry['kind'], entry['side']
    row = {'family': entry['family'], 'test': test,
           'registry_status': entry['status'], 'task': entry['task'],
           'endpoint': entry['endpoint'], 'side': side, 'a': entry['a'],
           'b': entry['b'], 'm': m, 'm_f3': m_f3, 'n_expected': len(seeds),
           'n_outside_list': sum(len(o) for o in outside.values()),
           'outside_list': '; '.join(f'{s}: seeds {_seed_runs(set(o))}'
                                     for s, o in outside.items() if o)
           or None, 'floor_rule': entry['floor']}
    marks = {}
    for s, d in (('a', a), ('b', b)):
        k, n = (int(d['below'].sum()), len(d)) if len(d) else (0, 0)
        row[f'floor_{s}'] = f'{k}/{n}'
        marks[s] = not_learned(k, n)
        for f in F3_CONFIG_FIELDS:
            row[f'{f}_{s}'] = _config_values(d, f)
    row['floor_bound'] = marks['a'] or marks['b']
    row['floor_bound_arms'] = ','.join(
        s.upper() for s in 'ab' if marks[s]) or None
    above = any(len(d) and (d['budget_above_tolerance']
                            & ~d['width_matched']).any() for d in (a, b))
    if len(a) and len(b):
        worst = WORST['pa' if entry['task'] == 'pa' else 'lv']
        a, b = (d.assign(v=np.where(d['diverged'], worst, d['metric']))
                for d in (a, b))
        p = a.merge(b, on=list(key), suffixes=('_a', '_b')).sort_values(
            list(key))
        _check_data_key(p, test)
        _check_match(p, entry.get('match', ()), test)
    else:
        p = pd.DataFrame()
    n = len(p)
    row.update(n_pairs=n, n_nonfinite=int(p['diverged_a'].sum()
                                          + p['diverged_b'].sum()) if n
               else 0)
    if n:
        st = _pair_stats(p, p['task_a'].iloc[0], m, side)
        x, y = p['v_a'].to_numpy(float), p['v_b'].to_numpy(float)
        row.update({k: st[k] for k in (
            'd_z', 'p_a_gt_b', 'p_a_gt_b_lo', 'p_a_gt_b_hi', 'median_diff',
            'median_diff_lo', 'median_diff_hi', 'ci_units')},
            n_positive=int((x - y > 0).sum()),
            ci_includes_zero=(bool(st['median_diff_lo'] <= 0
                                   <= st['median_diff_hi'])
                              if entry['task'] == 'pa' else None))
        if 'success' in entry:
            sa, sb = p['success_a'].to_numpy(bool), p['success_b'].to_numpy(
                bool)
            _, n10, n01 = mcnemar_exact(sa, sb)
            row.update(n10=n10, n01=n01, rate_a=_rate(int(sa.sum()), n),
                       rate_b=_rate(int(sb.sum()), n),
                       rate_diff=(int(sa.sum()) - int(sb.sum())) / n)
    if above:
        row.update(status=EXCLUDED, reading=None)
        return row
    row['status'] = 'tested'
    complete = n == len(seeds)
    if not n:
        row['reading'] = 'incomplete'
        return row
    if kind == 'mcnemar':
        p_raw, n10, n01 = mcnemar_exact(sa, sb)
        row.update(method='mcnemar-exact')
        effect = n10 - n01
    else:
        p_raw, method = st['p_raw'], st['method']
        if p_raw is None:       # every paired difference zero
            p_raw = 1.0
        sign, ranks = _signed_ranks(x, y)
        effect = float((sign * ranks).sum())          # W+ - W-
        p_perm = permutation_p(x, y, side)
        row.update(method=method, p_perm=p_perm,
                   p_perm_adj=min(1.0, m * p_perm),
                   w_plus_minus_w_minus=effect)
    row.update(p_raw=p_raw, p_adj=min(1.0, m * p_raw),
               p_adj_f3=min(1.0, m_f3 * p_raw))
    row['significant'] = complete and row['p_adj'] < ALPHA
    row['significant_f3'] = complete and row['p_adj_f3'] < ALPHA
    row['only_under_family_cut'] = (row['significant']
                                    and not row['significant_f3'])
    if kind != 'mcnemar':
        row['verdict_flip'] = ((row['p_adj'] < ALPHA)
                               != (row['p_perm_adj'] < ALPHA))
    row['reading'] = ('incomplete' if not complete else 'floor-bound'
                      if entry['floor'] == 'exclusive' and row['floor_bound']
                      else 'PASS' if row['p_adj'] < ALPHA else 'FAIL')
    if row['reading'] == 'PASS':
        row['direction'] = 'A>B' if effect > 0 else 'A<B'
        if kind != 'mcnemar':
            row['median_sign_differs'] = bool(np.sign(row['median_diff'])
                                              != np.sign(effect))
    return row


F3_COLUMNS = ['family', 'test', 'registry_status', 'task', 'endpoint',
              'side', 'a', 'b', 'm', 'm_f3', 'n_expected', 'n_pairs',
              'n_outside_list', 'outside_list', 'status', 'reading',
              'direction', 'method', 'p_raw', 'p_adj', 'p_adj_f3', 'p_perm',
              'p_perm_adj', 'significant', 'significant_f3',
              'only_under_family_cut', 'verdict_flip', 'median_sign_differs',
              'ci_includes_zero', 'w_plus_minus_w_minus', 'median_diff',
              'median_diff_lo', 'median_diff_hi', 'ci_units', 'd_z',
              'p_a_gt_b', 'p_a_gt_b_lo', 'p_a_gt_b_hi', 'n_positive',
              'n_nonfinite', 'n10', 'n01', 'rate_a', 'rate_b', 'rate_diff',
              'floor_rule', 'floor_a', 'floor_b', 'floor_bound',
              'floor_bound_arms',
              *(f'{f}_{s}' for f in F3_CONFIG_FIELDS for s in 'ab')]


def f3_family_blocks(spec) -> pd.DataFrame:
    """One row per test of a load_f3_spec spec, in registry order (columns
    F3_COLUMNS): paired over the test's seed list with its own merge, delta
    = A - B, the registry side; Wilcoxon (exact without ties or zeros, else
    the normal approximation) with the exact permutation p next to it, or
    the exact McNemar on the learning success; adjusted within the family
    (m = its tests in the spec) and F3-wide (m_f3 = tests in the spec,
    sensitivity); reading incomplete (fewer complete pairs than listed
    seeds; statistics still reported) before the floor rule of the registry
    before PASS/FAIL; the budget-tolerance exclusion of every arm except
    the width-matched one."""
    draws, rows = {}, []
    for test, entry in F3_TESTS.items():
        if test not in spec:
            continue
        seeds = spec[test]['seeds']
        m = sum(t in spec for t in F3_FAMILIES[entry['family']]['tests'])
        (a, out_a), (b, out_b) = (
            _f3_arm(test, side, entry, spec[test][side], seeds, draws)
            for side in 'ab')
        rows.append(_f3_test(test, entry, m, len(spec), a, b, seeds,
                             {'a': out_a, 'b': out_b}))
    return pd.DataFrame(rows).reindex(columns=F3_COLUMNS)


def f3_report(blocks) -> str:
    """Markdown of f3_family_blocks."""
    m_f3 = int(blocks['m_f3'].iloc[0]) if len(blocks) else 0
    lines = [f'## F3 test families (protocol F3; paired over the seed list '
             f'of each test, delta = A - B, Bonferroni within the family, '
             f'F3-wide m={m_f3} as sensitivity)\n',
             '| Family | m | Test | Task | endpoint | side | A | B | n/listed '
             '| status | reading | method | p | p adj | p adj F3 | p perm '
             '| p perm adj | flags | median delta [95% CI] | d_z '
             '| P(A>B) [95% CI] | seeds delta>0 | success A | success B '
             '| floor rule | floor A | floor B | floor_bound | non-finite |',
             '|' + '---|' * 29]
    flags = (('only_under_family_cut', 'only under the family cut'),
             ('verdict_flip', 'verdict flips with p perm'),
             ('median_sign_differs', 'median sign differs'),
             ('ci_includes_zero', 'interval includes 0'))

    def yes(v):
        return isinstance(v, (bool, np.bool_)) and bool(v)

    def txt(v):
        return v if isinstance(v, str) else '—'

    for b in blocks.to_dict('records'):
        fl = '; '.join([*([b['direction']] if isinstance(b['direction'], str)
                          else []), *(t for k, t in flags if yes(b[k]))])
        med = ('—' if pd.isna(b['median_diff']) else
               f"{b['median_diff']:+.4g} [{b['median_diff_lo']:+.4g}, "
               f"{b['median_diff_hi']:+.4g}] ({b['ci_units']})")
        win = ('—' if pd.isna(b['p_a_gt_b']) else
               f"{b['p_a_gt_b']:.2f} [{b['p_a_gt_b_lo']:.3f}, "
               f"{b['p_a_gt_b_hi']:.3f}]")
        lines.append(
            f"| {b['family']} | {b['m']} | {b['test']} | {b['task']} "
            f"| {b['endpoint']} | {b['side']} | {b['a']} | {b['b']} "
            f"| {b['n_pairs']}/{b['n_expected']} | {b['status']} "
            f"| {txt(b['reading'])} | {txt(b['method'])} | "
            + ' | '.join(_fmt(b[k]) for k in (
                'p_raw', 'p_adj', 'p_adj_f3', 'p_perm', 'p_perm_adj'))
            + f" | {fl or '—'} | {med} | {_fmt(b['d_z'], '+.3f')} | {win} "
            f"| {_fmt(b['n_positive'], '.0f')} | {txt(b['rate_a'])} "
            f"| {txt(b['rate_b'])} | {b['floor_rule']} | {b['floor_a']} "
            f"| {b['floor_b']} | "
            + (f"YES ({b['floor_bound_arms']})" if yes(b['floor_bound'])
               else 'no') + f" | {b['n_nonfinite']} |")
    lines += ['', '| Test | ' + ' | '.join(
        f'{f} A | {f} B' for f in F3_CONFIG_FIELDS) + ' |',
        '|' + '---|' * (1 + 2 * len(F3_CONFIG_FIELDS))]
    for b in blocks.to_dict('records'):
        lines.append(f"| {b['test']} | " + ' | '.join(
            txt(b[f'{f}_{s}']) for f in F3_CONFIG_FIELDS for s in 'ab')
            + ' |')
    outside = blocks[blocks['n_outside_list'] > 0]
    lines += ['', 'Records with the arm label outside the seed list of the '
              'test (not used): ' + ('; '.join(
                  f"{t} {o} ({n})" for t, o, n in zip(
                      outside['test'], outside['outside_list'],
                      outside['n_outside_list'])) or 'none') + '.',
              '', 'reading: incomplete (fewer complete pairs than listed '
              'seeds) before floor-bound (floor rule exclusive only) before '
              f'PASS/FAIL (adj. p < {ALPHA}); the deciding p is that of '
              'wilcoxon_test (exact without ties or zeros, else the normal '
              'approximation) or the exact McNemar on the learning success, '
              'p perm the exact sign-flip test over the mid-ranks with the '
              'same side. success A/B: fits with learning success (Person '
              'Activity: endpoint >= the registry threshold; rollouts: '
              'VPT-multi median >= the fit\'s own floor + 2; a diverged or '
              'non-finite fit fails). floor A/B: seeds below the floor, not '
              'learned in more than a fifth of the seeds. A non-finite or '
              'unbounded fit enters at the worst value.', '']
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
                        'bootstrap interval), searched recursively')
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
    p.add_argument('--descriptive', default=None, metavar='SPEC.yaml',
                   help='the descriptive F2 blocks (DESCRIPTIVE_BLOCKS) from '
                        'a spec of explicit path lists per arm '
                        '(configs/aggregate/descriptive_blocks.yaml); writes '
                        '<out>/descriptive_blocks.csv')
    p.add_argument('--f3-families', default=None, metavar='SPEC.yaml',
                   help='the F3 test families (F3_FAMILIES) from a spec of '
                        'seed lists and explicit path lists per arm '
                        '(configs/aggregate/f3_families.yaml); writes '
                        '<out>/f3_families.csv')
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
                         or args.co_runs or args.f2_families
                         or args.descriptive or args.f3_families):
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
        seq = pd.concat([load_sequence_runs(d, t, curves=bool(
            args.descriptive)) for d, t in (
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
        seq.drop(columns=['confusion', 'confusion_robust'],
                 errors='ignore').to_csv(
            os.path.join(args.out, 'sequence_runs.csv'), index=False)
        parts.append(contrast_report(seq, blocks, descriptive_pairs(seq)))
        rq5_paired(seq).to_csv(os.path.join(args.out, 'rq5_paired.csv'),
                               index=False)
    if args.co_runs:
        parts.append(consensus_family(args.co_runs))
        consensus_intervals(args.co_runs).to_csv(
            os.path.join(args.out, 'consensus_family.csv'), index=False)
        consensus_arm_endpoints(args.co_runs).to_csv(
            os.path.join(args.out, 'consensus_arms.csv'), index=False)
    if args.f2_families:
        f2 = f2_family_blocks(load_f2_spec(args.f2_families))
        f2.to_csv(os.path.join(args.out, 'f2_families.csv'), index=False)
        parts.append(f2_report(f2))
    if args.descriptive:
        desc = descriptive_blocks(load_descriptive_spec(args.descriptive))
        desc.to_csv(os.path.join(args.out, 'descriptive_blocks.csv'),
                    index=False)
        parts.append(descriptive_report(desc))
    if args.f3_families:
        f3 = f3_family_blocks(load_f3_spec(args.f3_families))
        f3.to_csv(os.path.join(args.out, 'f3_families.csv'), index=False)
        parts.append(f3_report(f3))
    report = '\n'.join(parts)
    md_path = os.path.join(args.out, 'summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(report)
    print(f'\nWritten: {md_path}' + (' and summary.csv' if not df.empty else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
