# experiments/aggregate_results.py
"""Aggregate benchmark run JSONs into thesis-ready tables and statistics.

Reads the per-run JSONs produced by experiments/run_benchmark.py and writes:
  - results/summary.csv     — one row per run (long format)
  - results/summary.md      — mean +/- std tables and statistical tests

Statistics (per thesis protocol, see research-questions-hypotheses):
  - Wilcoxon signed-rank tests, alpha = 0.05, paired over (system, seed)
  - paired Cohen's d as effect size (mean(diff) / std(diff))

Usage:
    uv run python experiments/aggregate_results.py [--runs results/runs] [--out results]
"""
import argparse
import json
import os
import sys
from glob import glob
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ALPHA = 0.05
METRIC = 'nrmse'   # primary comparison metric (proposal: MSE/NRMSE)


def load_runs(runs_dir: str) -> pd.DataFrame:
    rows = []
    for path in sorted(glob(os.path.join(runs_dir, '*.json'))):
        with open(path, encoding='utf-8') as f:
            r = json.load(f)
        rows.append({
            'cell': r['run']['cell'],
            'wiring': r['run']['wiring'],
            'system': r['run']['system'],
            'seed': r['run']['seed'],
            'final_loss': r['training']['final_loss'],
            'mse': r['evaluation']['mse'],
            'nrmse': r['evaluation']['nrmse'],
            'duration_s': r['training']['duration_s'],
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
    """Metric vectors for two configs, paired on (system, seed)."""
    a = df.loc[(df[list(key_a)] == pd.Series(key_a)).all(axis=1)]
    b = df.loc[(df[list(key_b)] == pd.Series(key_b)).all(axis=1)]
    merged = a.merge(b, on=['system', 'seed'], suffixes=('_a', '_b'))
    return merged[f'{METRIC}_a'].to_numpy(), merged[f'{METRIC}_b'].to_numpy()


def _cohens_d_paired(x: np.ndarray, y: np.ndarray) -> float:
    diff = x - y
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else float('nan')


def _test_block(df, pairs, title) -> str:
    lines = [f'## {title} (Wilcoxon signed-rank, alpha={ALPHA}, metric={METRIC})\n']
    lines.append('| Comparison | n pairs | median diff | p-value | significant | Cohen\'s d |')
    lines.append('|------------|---------|-------------|---------|-------------|-----------|')
    for label, key_a, key_b in pairs:
        x, y = _paired_vectors(df, key_a, key_b)
        if len(x) < 5:
            lines.append(f'| {label} | {len(x)} | — | — | too few pairs | — |')
            continue
        try:
            stat, p = wilcoxon(x, y)
        except ValueError:   # all differences zero
            lines.append(f'| {label} | {len(x)} | 0 | 1.0 | no | 0 |')
            continue
        d = _cohens_d_paired(x, y)
        sig = 'YES' if p < ALPHA else 'no'
        lines.append(f'| {label} | {len(x)} | {np.median(x - y):+.4f} '
                     f'| {p:.4g} | {sig} | {d:+.3f} |')
    lines.append('')
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--runs', default='results/runs')
    p.add_argument('--out', default='results')
    args = p.parse_args(argv)

    df = load_runs(args.runs)
    if df.empty:
        print(f'No run JSONs found in {args.runs}', file=sys.stderr)
        return 1

    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, 'summary.csv'), index=False)

    report = '\n'.join([
        '# Benchmark Summary\n',
        f'Runs: {len(df)} | Cells: {sorted(df.cell.unique())} | '
        f'Wirings: {sorted(df.wiring.unique())} | Seeds/config: '
        f'{df.groupby(["cell", "wiring", "system"]).size().min()}-'
        f'{df.groupby(["cell", "wiring", "system"]).size().max()}\n',
        summary_tables(df),
        statistics(df),
    ])
    md_path = os.path.join(args.out, 'summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(report)
    print(f'\nWritten: {md_path} and summary.csv')
    return 0


if __name__ == '__main__':
    sys.exit(main())
