# experiments/analyze_v5.py
"""Benchmark v5 generalization-stress analysis.

Reads the clean v4 matrix (results/runs_v4) and the v5 stress matrix
(results/runs_v5) and answers the v5 questions the generic all-pairs
aggregate_results.py does not target:

  1. Master table: median / mean / divergence-rate (NRMSE>1) per cell x wiring,
     for clean (v4) and each stress regime (v5: noise, extrapolation, ood_init).
  2. Degradation: paired clean->stress change per cell on ncp (does the cell's
     accuracy survive the stressor?).
  3. Bio-vs-classical per regime on ncp: does the v4 bio advantage over
     gru/lstm/ctrnn hold under each stressor -- the real, non-clean-fit Q2?

Paired Wilcoxon over (system, seed), paired Cohen's d, Bonferroni per family.
Negative median diff / d = the first cell is better (lower NRMSE).

Usage:
    uv run python experiments/analyze_v5.py
"""
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

from experiments.aggregate_results import load_runs

REGIMES = ['noise', 'extrapolation', 'ood_init']
CELLS = ['ltc', 'cfc', 'mm_ltc', 'cfc_mm_ltc',
         'lrc', 'cfc_lrc', 'mm_lrc', 'cfc_mm_lrc',
         'gru', 'lstm', 'ctrnn']
BIO_CLOSED = ['cfc', 'cfc_lrc', 'cfc_mm_lrc', 'cfc_mm_ltc']   # the v4 champions
CLASSICAL = ['gru', 'lstm', 'ctrnn']
ALPHA = 0.05
DIV = 1.0   # NRMSE > 1 == divergence


def variant(cell, regime):
    return cell if regime == 'clean' else f'{cell}+{regime}'


def cell_stats(df, cellvar, wiring):
    s = df[(df.cell == cellvar) & (df.wiring == wiring)]['nrmse'].to_numpy()
    if len(s) == 0:
        return None
    return dict(n=len(s), median=float(np.median(s)), mean=float(s.mean()),
                div=100.0 * float(np.mean(s > DIV)))


def paired(df, va, vb, wiring):
    a = df[(df.cell == va) & (df.wiring == wiring)][['system', 'seed', 'nrmse']]
    b = df[(df.cell == vb) & (df.wiring == wiring)][['system', 'seed', 'nrmse']]
    m = a.merge(b, on=['system', 'seed'], suffixes=('_a', '_b'))
    return m['nrmse_a'].to_numpy(), m['nrmse_b'].to_numpy()


def wilcox(x, y, m=1):
    if len(x) < 5:
        return None
    try:
        _, p = wilcoxon(x, y)
    except ValueError:        # all differences zero
        return dict(n=len(x), median_diff=0.0, p=1.0, p_adj=1.0, d=0.0)
    diff = x - y
    sd = diff.std(ddof=1)
    d = float(diff.mean() / sd) if sd > 0 else 0.0
    return dict(n=len(x), median_diff=float(np.median(diff)), p=float(p),
                p_adj=float(min(1.0, p * m)), d=d)


def master_table(df):
    lines = ['## 1. Master table — median / mean / divergence% (NRMSE>1)\n']
    for regime in ['clean'] + REGIMES:
        lines.append(f'### {regime}\n')
        lines.append('| Cell | dense med | dense div% | ncp med | ncp mean | ncp div% |')
        lines.append('|------|-----------|------------|---------|----------|----------|')
        for cell in CELLS:
            d = cell_stats(df, variant(cell, regime), 'dense')
            n = cell_stats(df, variant(cell, regime), 'ncp')
            if d is None or n is None:
                lines.append(f'| {cell} | — | — | — | — | — |')
                continue
            lines.append(f"| {cell} | {d['median']:.3f} | {d['div']:.0f} | "
                         f"{n['median']:.3f} | {n['mean']:.2f} | {n['div']:.0f} |")
        lines.append('')
    return '\n'.join(lines)


def degradation_table(df, wiring='ncp'):
    lines = [f'## 2. Degradation clean->stress on {wiring} '
             f'(paired Wilcoxon vs clean, Bonferroni m={len(CELLS)})\n']
    lines.append('| Cell | clean med (div%) | noise med (Δ, sig) | '
                 'extrapolation med (Δ, sig) | ood_init med (Δ, sig) |')
    lines.append('|------|------------------|--------------------|'
                 '----------------------------|------------------------|')
    for cell in CELLS:
        clean = cell_stats(df, cell, wiring)
        row = [f'| {cell} | {clean["median"]:.3f} ({clean["div"]:.0f}%)']
        for regime in REGIMES:
            st = cell_stats(df, variant(cell, regime), wiring)
            x, y = paired(df, variant(cell, regime), cell, wiring)  # stress - clean
            w = wilcox(x, y, m=len(CELLS))
            sig = '—'
            if w is not None:
                sig = 'sig' if w['p_adj'] < ALPHA else 'ns'
            delta = st['median'] - clean['median']
            row.append(f'{st["median"]:.3f} ({delta:+.3f} {st["div"]:.0f}%, {sig})')
        lines.append(' | '.join(row) + ' |')
    lines.append('')
    return '\n'.join(lines)


def bio_vs_classical(df, wiring='ncp'):
    m = len(BIO_CLOSED) * len(CLASSICAL)
    lines = [f'## 3. Bio (closed-form) vs classical per regime on {wiring} '
             f'(paired Wilcoxon, Bonferroni m={m}, neg = bio better)\n']
    for regime in REGIMES:
        lines.append(f'### {regime}\n')
        lines.append('| Bio | vs | median diff | p_adj | significant | Cohen d |')
        lines.append('|-----|----|-------------|-------|-------------|---------|')
        for bio in BIO_CLOSED:
            for clf in CLASSICAL:
                x, y = paired(df, variant(bio, regime), variant(clf, regime), wiring)
                w = wilcox(x, y, m=m)
                if w is None:
                    continue
                sig = 'YES' if w['p_adj'] < ALPHA else 'no'
                lines.append(f"| {bio} | {clf} | {w['median_diff']:+.4f} | "
                             f"{w['p_adj']:.3g} | {sig} | {w['d']:+.2f} |")
        lines.append('')
    return '\n'.join(lines)


def robustness_ranking(df, wiring='ncp'):
    lines = [f'## 4. Robustness ranking per regime on {wiring} (by mean NRMSE)\n']
    for regime in ['clean'] + REGIMES:
        rows = []
        for cell in CELLS:
            s = cell_stats(df, variant(cell, regime), wiring)
            if s:
                rows.append((cell, s['mean'], s['median'], s['div']))
        rows.sort(key=lambda r: r[1])
        lines.append(f'### {regime}\n')
        lines.append('| rank | cell | mean | median | div% |')
        lines.append('|------|------|------|--------|------|')
        for i, (cell, mean, med, div) in enumerate(rows, 1):
            lines.append(f'| {i} | {cell} | {mean:.3f} | {med:.3f} | {div:.0f} |')
        lines.append('')
    return '\n'.join(lines)


def main(argv=None):
    runs = ['results/runs_v4', 'results/runs_v5']
    df = load_runs(runs)
    if df.empty:
        print(f'No runs in {runs}', file=sys.stderr)
        return 1
    report = '\n'.join([
        '# Benchmark v5 — Generalization Stress Analysis\n',
        f'Clean baseline: v4 ({(df.stress.isna()).sum()} runs). '
        f'Stress: v5 ({df.stress.notna().sum()} runs, '
        f'{sorted(df.stress.dropna().unique())}).\n',
        'Metric: full-trajectory NRMSE. Divergence = share NRMSE>1. '
        'n=30 per cell x wiring x regime (5 seeds x 6 systems).\n',
        master_table(df),
        degradation_table(df),
        bio_vs_classical(df),
        robustness_ranking(df),
    ])
    out = 'results/aggregate_v5/v5_analysis.md'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        f.write(report)
    print(report)
    print(f'\nWritten: {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
