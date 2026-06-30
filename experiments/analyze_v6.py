# experiments/analyze_v6.py
"""Benchmark v6 follow-up analysis (graph robustness + dose-response).

The v5 analysis (analyze_v5.py) established that on the ncp wiring, under the
*noise* stressor, the closed-form bio cells (cfc family) beat the classical
RNN cells (gru/lstm/ctrnn). v6 stress-tests that result along two axes the v5
single-point design could not rule out:

  v6a  multi-wiring-seed graph robustness
       Was the v5 noise ordering an artifact of the single ncp wiring graph
       (ncp_wiring_seed=42), or does it survive other random graphs?
       5 graphs: ncp_wiring_seed in {42 (=v5 anchor), 7, 13, 21, 99}.
       -> Kendall's W / Friedman concordance of the 8-cell ranking across the
          5 graphs, plus per-(bio,classical)-pair sign test over the 5 graphs.

  v6b  stress-level dose-response
       Was the v5 ordering a genuine monotone dose-response, or a floor (stress
       too weak, everyone fine) / ceiling (too strong, everyone diverges)
       artifact? 3 stressors, 4 ordered severity levels each, on ncp seed 42.
       -> per (cell, stressor) monotonic-trend test (Page's L, implemented by
          hand and cross-checked against scipy.stats.page_trend_test) plus a
          floor/ceiling diagnostic and a bio<classical check at every level.

cell_base is derived by stripping ALL '+...' suffixes (regime, +w<seed>,
+lvl<val>): cell.split('+')[0]. The v5 plot regex only strips a trailing
'+<regime>' and would silently merge variants here -- do NOT reuse it.

Metric: full-trajectory NRMSE (range-normalized). Divergence = NRMSE > 1.
Paired statistics over (system, seed); n=30 = 5 seeds x 6 systems.

Usage:
    uv run python experiments/analyze_v6.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, norm, rankdata, wilcoxon

try:
    # SciPy >= 1.7 provides Page's L; used only to cross-check the hand impl.
    from scipy.stats import page_trend_test
except ImportError:  # pragma: no cover
    page_trend_test = None

from experiments.aggregate_results import load_runs

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CELLS_V6 = ['cfc', 'cfc_lrc', 'cfc_mm_lrc', 'cfc_mm_ltc',
            'gru', 'lstm', 'ctrnn', 'lrc']
BIO = ['cfc', 'cfc_lrc', 'cfc_mm_lrc', 'cfc_mm_ltc']   # closed-form bio champions
CLASSICAL = ['gru', 'lstm', 'ctrnn']
GRAPHS = [42, 7, 13, 21, 99]          # 42 = v5 noise anchor, rest = v6a
ALPHA = 0.05
DIV = 1.0                              # NRMSE > 1 == divergence

# v6b dose-response: ordered by INCREASING severity. Each tuple is
# (severity_label, selector). Selector is resolved against the ncp/seed-42 frame.
#   'clean'      -> v4 rows (stress.isna()), seeds 0-4 for n=30 comparability
#   'mid'        -> v5 rows (stress==regime AND stress_level.isna())
#   float        -> v6b rows (stress==regime AND stress_level==value)
DOSE = {
    # noise std: 0.00 < 0.05 < 0.10(=v5 mid) < 0.20
    'noise': [('0.00', 'clean'), ('0.05', 0.05), ('0.10', 'mid'), ('0.20', 0.20)],
    # ood_init scale: 0.00 < 0.10 < 0.20(=v5 mid) < 0.40
    'ood_init': [('0.00', 'clean'), ('0.10', 0.10), ('0.20', 'mid'), ('0.40', 0.40)],
    # extrapolation train-fraction: LOWER frac = HARDER, so severity-order is
    # reversed fraction: frac 1.00(clean) > 0.70 > 0.50(=v5 mid) > 0.30.
    'extrapolation': [('frac1.00', 'clean'), ('frac0.70', 0.70),
                      ('frac0.50', 'mid'), ('frac0.30', 0.30)],
}


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------
def add_cell_base(df: pd.DataFrame) -> pd.DataFrame:
    """Strip ALL '+...' suffixes to recover the base cell type.

    e.g. 'cfc', 'cfc+noise', 'cfc+noise+w7', 'cfc+noise+lvl0.2' -> 'cfc'.
    """
    df = df.copy()
    df['cell_base'] = df['cell'].str.split('+').str[0]
    return df


def wilcox(x: np.ndarray, y: np.ndarray, m: int = 1) -> dict:
    """Paired Wilcoxon signed-rank with paired Cohen's d. neg diff = x better."""
    if len(x) < 5:
        return None
    diff = x - y
    if np.all(diff == 0):
        return dict(n=len(x), median_diff=0.0, p=1.0, p_adj=1.0, d=0.0)
    try:
        _, p = wilcoxon(x, y)
    except ValueError:
        return dict(n=len(x), median_diff=0.0, p=1.0, p_adj=1.0, d=0.0)
    sd = diff.std(ddof=1)
    d = float(diff.mean() / sd) if sd > 0 else 0.0
    return dict(n=len(x), median_diff=float(np.median(diff)), p=float(p),
                p_adj=float(min(1.0, p * m)), d=d)


# ---------------------------------------------------------------------------
# v6a: multi-wiring-seed graph robustness
# ---------------------------------------------------------------------------
def v6a_frame(df: pd.DataFrame) -> pd.DataFrame:
    """ncp + noise + single-level (stress_level NaN), the 8 v6 cells, 5 graphs."""
    return df[(df.wiring == 'ncp') & (df.stress == 'noise')
              & (df.stress_level.isna()) & (df.cell_base.isin(CELLS_V6))].copy()


def v6a_per_graph_means(sub: pd.DataFrame) -> pd.DataFrame:
    """8 cells x 5 graphs matrix of mean NRMSE (rows=cell, cols=graph)."""
    g = (sub.groupby(['cell_base', 'ncp_wiring_seed'])['nrmse']
         .mean().unstack('ncp_wiring_seed'))
    return g.reindex(index=CELLS_V6, columns=GRAPHS)


def kendalls_w(rank_matrix: np.ndarray) -> float:
    """Kendall's W of m raters (graphs) ranking n subjects (cells).

    rank_matrix: shape (m_graphs, n_cells), each row a ranking of the cells.
    W in [0,1]; 1 = perfect agreement of the cell ordering across graphs.
    """
    m, n = rank_matrix.shape
    rank_sums = rank_matrix.sum(axis=0)                 # per cell, summed over graphs
    s = np.sum((rank_sums - rank_sums.mean()) ** 2)
    return float(12.0 * s / (m ** 2 * (n ** 3 - n)))


def v6a_concordance(means: pd.DataFrame):
    """Friedman + Kendall's W on the 8-cell ranking across the 5 graphs.

    Treats graphs as blocks and cells as treatments. The per-graph mean NRMSE
    of each cell is ranked (low NRMSE = rank 1 = best). High W => the cell
    ordering is the same regardless of which random ncp graph was drawn.
    """
    rank_per_graph = np.vstack([rankdata(means[g].to_numpy()) for g in GRAPHS])
    w = kendalls_w(rank_per_graph)
    fr_stat, fr_p = friedmanchisquare(*[means[g].to_numpy() for g in GRAPHS])
    return dict(W=w, friedman_stat=float(fr_stat), friedman_p=float(fr_p),
                rank_per_graph=rank_per_graph)


def v6a_pair_signtest(sub: pd.DataFrame):
    """Per (bio, classical) pair: per-graph paired Wilcoxon + 5-graph sign test.

    For each of the 5 graphs the pair is compared paired on (system, seed).
    'bio wins' = bio median NRMSE < classical (median_diff < 0). The count of
    wins over 5 graphs gives a sign test; one-sided unanimity p = 0.5^5
    = 0.03125.
    """
    results = []
    for bio in BIO:
        for clf in CLASSICAL:
            wins = sig_wins = 0
            per_graph = []
            for g in GRAPHS:
                a = sub[(sub.cell_base == bio) & (sub.ncp_wiring_seed == g)][
                    ['system', 'seed', 'nrmse']]
                b = sub[(sub.cell_base == clf) & (sub.ncp_wiring_seed == g)][
                    ['system', 'seed', 'nrmse']]
                mrg = a.merge(b, on=['system', 'seed'], suffixes=('_a', '_b'))
                w = wilcox(mrg['nrmse_a'].to_numpy(), mrg['nrmse_b'].to_numpy())
                bio_wins = w['median_diff'] < 0
                wins += int(bio_wins)
                if bio_wins and w['p'] < ALPHA:
                    sig_wins += 1
                per_graph.append((g, w))
            results.append(dict(bio=bio, clf=clf, wins=wins, n_graphs=len(GRAPHS),
                                sig_wins=sig_wins, p_unanimous=0.5 ** len(GRAPHS),
                                per_graph=per_graph))
    return results


def v6a_report(df: pd.DataFrame):
    sub = v6a_frame(df)
    means = v6a_per_graph_means(sub)
    counts = sub.groupby(['cell_base', 'ncp_wiring_seed']).size()
    n_ok = (counts == 30).all() and len(counts) == len(CELLS_V6) * len(GRAPHS)

    lines = ['## v6a — multi-wiring-seed graph robustness (ncp, noise regime)\n']
    lines.append(f'Rows: {len(sub)} (expect {len(CELLS_V6)*len(GRAPHS)*30} = '
                 f'8 cells x 5 graphs x 30). Per-group n all 30: {bool(n_ok)}.\n')

    # (a) per-graph mean-NRMSE table
    lines.append('### (a) Per-graph mean NRMSE and ranking concordance\n')
    lines.append('| cell | ' + ' | '.join(f'g{g}' for g in GRAPHS) + ' | mean |')
    lines.append('|------|' + '|'.join(['------'] * (len(GRAPHS) + 1)) + '|')
    for cell in CELLS_V6:
        vals = [means.loc[cell, g] for g in GRAPHS]
        lines.append(f'| {cell} | ' + ' | '.join(f'{v:.3f}' for v in vals)
                     + f' | {np.mean(vals):.3f} |')
    lines.append('')

    conc = v6a_concordance(means)
    lines.append('Ranking per graph (1 = best/lowest NRMSE):\n')
    lines.append('| cell | ' + ' | '.join(f'g{g}' for g in GRAPHS) + ' |')
    lines.append('|------|' + '|'.join(['---'] * len(GRAPHS)) + '|')
    for i, cell in enumerate(CELLS_V6):
        ranks = conc['rank_per_graph'][:, i]
        lines.append(f'| {cell} | ' + ' | '.join(f'{int(r)}' for r in ranks) + ' |')
    lines.append('')
    lines.append(f"Kendall's W = {conc['W']:.3f}  "
                 f"(1 = identical cell ordering across all 5 graphs).  "
                 f"Friedman chi2 = {conc['friedman_stat']:.2f}, "
                 f"p = {conc['friedman_p']:.3g} "
                 f"(low p = cells differ; W measures ordering agreement).\n")

    # (b) per-pair sign test across graphs
    pairs = v6a_pair_signtest(sub)
    lines.append('### (b) Bio-vs-classical robustness across the 5 graphs\n')
    lines.append('Per pair: bio-wins / 5 graphs (bio median NRMSE < classical), '
                 'how many of those wins are individually significant (paired '
                 'Wilcoxon p<0.05), and the one-sided unanimity p (0.5^5 = '
                 '0.03125).\n')
    lines.append('| bio | classical | bio wins | sig wins | unanimous? | p_unan |')
    lines.append('|-----|-----------|----------|----------|------------|--------|')
    robust, flips = [], []
    for r in pairs:
        unanimous = r['wins'] == r['n_graphs']
        tag = 'YES (5/5)' if unanimous else ('flips' if 0 < r['wins'] < 5 else 'NO (0/5)')
        if unanimous:
            robust.append((r['bio'], r['clf']))
        elif r['wins'] < 5:
            flips.append((r['bio'], r['clf'], r['wins']))
        lines.append(f"| {r['bio']} | {r['clf']} | {r['wins']}/5 | "
                     f"{r['sig_wins']}/5 | {tag} | {r['p_unanimous']:.4f} |")
    lines.append('')
    lines.append(f"Robust 5/5 bio>classical pairs: {len(robust)}/12.")
    if flips:
        flip_s = ', '.join(f'{b} vs {c} ({w}/5)' for b, c, w in flips)
        lines.append(f"Pairs that flip on >=1 graph: {flip_s}.")
    else:
        lines.append("No pair flips on any graph.")
    lines.append('')

    summary = dict(n_rows=len(sub), n_ok=bool(n_ok), W=conc['W'],
                   friedman_p=conc['friedman_p'], n_robust=len(robust),
                   flips=flips, means=means, pairs=pairs)
    return '\n'.join(lines), summary


# ---------------------------------------------------------------------------
# v6b: stress-level dose-response
# ---------------------------------------------------------------------------
def v6b_level_vector(df, cell, regime, selector):
    """Paired frame (system, seed, nrmse) for one dose level.

    All selectors are restricted to ncp wiring and ncp_wiring_seed 42 so the
    curve is on a single graph; clean is further restricted to seeds 0-4 to
    match the n=30 of the stress points (v4 has tail seeds 5-9 on only two
    systems, which would imbalance the per-system mean).
    """
    base = df[(df.wiring == 'ncp') & (df.cell_base == cell)]
    if selector == 'clean':
        v = base[(base.stress.isna()) & (base.ncp_wiring_seed == 42)
                 & (base.seed.isin([0, 1, 2, 3, 4]))]
    elif selector == 'mid':
        v = base[(base.stress == regime) & (base.stress_level.isna())
                 & (base.ncp_wiring_seed == 42)]
    else:  # explicit v6b level
        v = base[(base.stress == regime) & (base.stress_level == selector)
                 & (base.ncp_wiring_seed == 42)]
    return v[['system', 'seed', 'nrmse']]


def pages_l(level_vectors):
    """Page's L trend test by hand, for a predicted increasing order of levels.

    level_vectors: list of np arrays, one per level, IN HYPOTHESIZED ASCENDING
    order (severity rises -> NRMSE predicted to rise). Subjects = paired
    (system, seed) blocks; levels = treatments.

    Within each subject (block) the k level values are ranked 1..k (low=1).
    Rj = sum of ranks of level j over all blocks. L = sum_j j * Rj.
    Returns (L, z, p_onesided) with the large-sample normal approximation
        E[L] = b*k*(k+1)^2 / 4
        Var[L] = b*k^2*(k+1)*(k^2-1) / 144
    High L (positive z) supports the monotone increase.
    """
    mat = np.vstack(level_vectors).T          # shape (b_blocks, k_levels)
    b, k = mat.shape
    ranks = np.vstack([rankdata(row) for row in mat])   # rank within each block
    Rj = ranks.sum(axis=0)
    j = np.arange(1, k + 1)
    L = float(np.sum(j * Rj))
    EL = b * k * (k + 1) ** 2 / 4.0
    VL = b * k ** 2 * (k + 1) * (k ** 2 - 1) / 144.0
    z = (L - EL) / np.sqrt(VL)
    p = float(norm.sf(z))                     # one-sided upper tail
    return dict(L=L, z=float(z), p=p, b=b, k=k, Rj=Rj.tolist())


def v6b_report(df: pd.DataFrame):
    lines = ['## v6b — stress-level dose-response (ncp, wiring seed 42)\n']
    lines.append('Clean anchor: v4 ncp seed-42 rows restricted to seeds 0-4 '
                 '(n=30, balanced over the 6 systems), paired to the stress '
                 'points on (system, seed). Levels ordered by INCREASING '
                 'severity. Page\'s L tests a monotone NRMSE increase along '
                 'that order (hand impl, cross-checked vs scipy).\n')

    all_curves = {}
    trend_summ = []
    for regime, levels in DOSE.items():
        labels = [lab for lab, _ in levels]
        lines.append(f'### {regime}\n')
        lines.append('| cell | ' + ' | '.join(labels)
                     + ' | Page L | z | p | monotone? |')
        lines.append('|------|' + '|'.join(['---'] * (len(labels) + 4)) + '|')
        curve_means, curve_div = {}, {}
        for cell in CELLS_V6:
            frames = [v6b_level_vector(df, cell, regime, sel) for _, sel in levels]
            merged = frames[0].rename(columns={'nrmse': 'l0'})
            for i, fr in enumerate(frames[1:], 1):
                merged = merged.merge(fr.rename(columns={'nrmse': f'l{i}'}),
                                      on=['system', 'seed'])
            vecs, means_row, div_row = [], [], []
            for i in range(len(levels)):
                col = merged[f'l{i}'].to_numpy()
                vecs.append(col)
                means_row.append(float(col.mean()))
                div_row.append(100.0 * float(np.mean(col > DIV)))
            curve_means[cell] = means_row
            curve_div[cell] = div_row
            page = pages_l(vecs)
            mono = 'YES' if page['p'] < ALPHA else 'no'
            cells_str = ' | '.join(f'{m:.3f}' for m in means_row)
            lines.append(f'| {cell} | {cells_str} | {page["L"]:.0f} | '
                         f'{page["z"]:+.2f} | {page["p"]:.2g} | {mono} |')
            trend_summ.append(dict(regime=regime, cell=cell, z=page['z'],
                                   p=page['p'], n_blocks=page['b'],
                                   means=means_row, div=div_row))
        lines.append('')

        # bio<classical at every level + floor/ceiling diagnostic
        lines.append(f'**bio < classical per level ({regime})** '
                     '(mean over the 4 bio vs mean over the 3 classical):\n')
        lines.append('| level | bio mean | classical mean | bio<clf? | '
                     'max div% (all cells) | min mean (all cells) |')
        lines.append('|-------|----------|----------------|----------|'
                     '----------------------|----------------------|')
        bio_holds_all = True
        floor_levels, ceiling_levels = [], []
        for li, lab in enumerate(labels):
            bio_m = np.mean([curve_means[c][li] for c in BIO])
            clf_m = np.mean([curve_means[c][li] for c in CLASSICAL])
            holds = bio_m < clf_m
            bio_holds_all &= holds
            max_div = max(curve_div[c][li] for c in CELLS_V6)
            min_mean = min(curve_means[c][li] for c in CELLS_V6)
            worst_mean = max(curve_means[c][li] for c in CELLS_V6)
            best_div = min(curve_div[c][li] for c in CELLS_V6)
            if worst_mean < 0.10:           # floor: even worst cell is fine
                floor_levels.append(lab)
            if best_div >= 50.0:            # ceiling: even best cell diverges
                ceiling_levels.append(lab)
            lines.append(f'| {lab} | {bio_m:.3f} | {clf_m:.3f} | '
                         f'{"YES" if holds else "NO"} | {max_div:.0f} | '
                         f'{min_mean:.3f} |')
        lines.append('')
        floor_s = ', '.join(floor_levels) if floor_levels else 'none'
        ceil_s = ', '.join(ceiling_levels) if ceiling_levels else 'none'
        lines.append(f'bio<classical holds at EVERY level: {bool(bio_holds_all)}. '
                     f'Floor levels (worst cell still mean<0.10): {floor_s}. '
                     f'Ceiling levels (best cell >=50% divergence): {ceil_s}.\n')
        all_curves[regime] = dict(means=curve_means, div=curve_div, labels=labels,
                                  bio_holds_all=bool(bio_holds_all),
                                  floor=floor_levels, ceiling=ceiling_levels)

    n_mono = sum(1 for t in trend_summ if t['p'] < ALPHA)
    lines.append('### Trend summary\n')
    lines.append(f'Monotone-increase (Page L, p<0.05): {n_mono}/{len(trend_summ)} '
                 '(cell x stressor) curves.\n')
    return '\n'.join(lines), dict(trend=trend_summ, curves=all_curves,
                                  n_mono=n_mono, n_curves=len(trend_summ))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    runs = ['results/runs_v4', 'results/runs_v5', 'results/runs_v6a', 'results/runs_v6b']
    df = load_runs(runs)
    if df.empty:
        print(f'No runs in {runs}', file=sys.stderr)
        return 1
    df = add_cell_base(df)

    v6a_md, v6a_sum = v6a_report(df)
    v6b_md, v6b_sum = v6b_report(df)

    header = '\n'.join([
        '# Benchmark v6 — Graph-Robustness & Dose-Response Analysis\n',
        f'Sources: runs_v4 (clean), runs_v5 (single-level stress), '
        f'runs_v6a (multi-wiring), runs_v6b (dose-response). '
        f'Total rows loaded: {len(df)}.\n',
        'Metric: full-trajectory NRMSE (range-normalized). '
        'Divergence = NRMSE > 1. Paired on (system, seed), n=30 = 5 seeds x '
        '6 systems. cell_base = cell.split("+")[0] (strips regime, +w<seed>, '
        '+lvl<val>).\n',
        f'BIO = {BIO}; CLASSICAL = {CLASSICAL}.\n',
    ])
    report = '\n'.join([header, v6a_md, v6b_md])

    out = 'results/aggregate_v6/v6_analysis.md'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        f.write(report)
    print(report)
    print(f'\nWritten: {out}')

    print('\n=== SUMMARY ===')
    print(f"v6a: rows={v6a_sum['n_rows']} n_ok={v6a_sum['n_ok']} "
          f"W={v6a_sum['W']:.3f} friedman_p={v6a_sum['friedman_p']:.3g} "
          f"robust_pairs={v6a_sum['n_robust']}/12 flips={v6a_sum['flips']}")
    print(f"v6b: monotone={v6b_sum['n_mono']}/{v6b_sum['n_curves']} curves")
    return 0


if __name__ == '__main__':
    sys.exit(main())
