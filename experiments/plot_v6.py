# experiments/plot_v6.py
"""Publication-quality figures for the v6 benchmark analyses.

Loads runs from four result directories:
  results/runs_v4   -- clean baseline (880 runs)
  results/runs_v5   -- single-level stress (1980 runs)
  results/runs_v6a  -- multi-wiring-seed graph robustness (960 runs)
  results/runs_v6b  -- dose-response across stress levels (1440 runs)

Produces figures into results/figures_v6/:

  fig1_v6a_ranking_lines.png   -- per-graph mean NRMSE lines (8 cells x 5 graphs)
  fig2_v6a_rank_heatmap.png    -- cell x graph rank heatmap (1=best)
  fig3_v6b_noise.png           -- dose-response: noise stressor
  fig4_v6b_extrapolation.png   -- dose-response: extrapolation stressor
  fig5_v6b_ood_init.png        -- dose-response: ood_init stressor

Usage (from repo root):
    uv run python experiments/plot_v6.py [--out results/figures_v6]
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import friedmanchisquare, wilcoxon

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

CELLS_V6 = ['cfc', 'cfc_lrc', 'cfc_mm_lrc', 'cfc_mm_ltc', 'gru', 'lstm', 'ctrnn', 'lrc']

BIO = ['cfc', 'cfc_lrc', 'cfc_mm_lrc', 'cfc_mm_ltc']      # closed-form bio champions
CLASSICAL = ['gru', 'lstm', 'ctrnn']
NUMERICAL = ['lrc']                                          # numerical-only (no cf)

CELL_ORDER_V6 = ['cfc', 'cfc_lrc', 'cfc_mm_lrc', 'cfc_mm_ltc', 'gru', 'lstm', 'ctrnn', 'lrc']

# Family color palette (matches plot_v5.py)
CELL_COLOR = {
    'cfc':        '#4C72B0',
    'cfc_mm_ltc': '#2D5A8E',
    'cfc_lrc':    '#DD8452',
    'cfc_mm_lrc': '#B5622E',
    'gru':        '#55A868',
    'lstm':       '#3D8050',
    'ctrnn':      '#6EC28A',
    'lrc':        '#C44E52',
}

FAMILY_COLORS = {
    'bio':       '#4C72B0',
    'classical': '#55A868',
    'lrc':       '#C44E52',
}

WIRING_SEEDS = [42, 7, 13, 21, 99]    # seed 42 = v5 reused anchor
WIRING_SEED_LABELS = {42: 'w42 (v5 anchor)', 7: 'w7', 13: 'w13', 21: 'w21', 99: 'w99'}

DIVERGENCE_THRESH = 1.0
CLIP_NRMSE = 4.0
DPI = 200

# Dose-response severity levels per stressor (x-axis ordered by severity)
# severity 0 = easiest (clean), 3 = hardest
DOSE_LEVELS = {
    'noise': {
        'levels':   [0.00, 0.05, 0.10, 0.20],
        'labels':   ['0.00\n(clean)', '0.05', '0.10\n(v5)', '0.20'],
        'xlabel':   'Noise sigma (fraction of signal std)',
    },
    'ood_init': {
        'levels':   [0.00, 0.10, 0.20, 0.40],
        'labels':   ['0.00\n(clean)', '0.10', '0.20\n(v5)', '0.40'],
        'xlabel':   'OOD init scale (fraction offset)',
    },
    'extrapolation': {
        # frac of training data; LOWER = HARDER -> flip so x goes easy -> hard
        'levels':   [1.00, 0.70, 0.50, 0.30],
        'labels':   ['1.00\n(clean)', '0.70', '0.50\n(v5)', '0.30'],
        'xlabel':   'Train fraction (lower = harder)',
    },
}


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_data() -> pd.DataFrame:
    """Load all four result directories and derive cell_base by stripping ALL suffixes."""
    from experiments.aggregate_results import load_runs
    df = load_runs(['results/runs_v4', 'results/runs_v5', 'results/runs_v6a', 'results/runs_v6b'])
    # Correct strip: split on '+', take first element only.
    # The v5 regex only stripped the regime suffix; here we correctly remove
    # +w<seed> and +lvl<val> as well.
    df['cell_base'] = df['cell'].str.split('+').str[0]
    return df


def balanced_mean(sub: pd.DataFrame) -> float:
    """Equal-weighted mean of per-system means (avoids over-weighting stiff systems)."""
    if len(sub) == 0:
        return float('nan')
    return float(sub.groupby('system')['nrmse'].mean().mean())


# --------------------------------------------------------------------------- #
# Statistical helpers
# --------------------------------------------------------------------------- #

def kendalls_w(ranks: np.ndarray) -> float:
    """Kendall's W concordance coefficient.

    ranks: shape (n_judges, n_subjects) -- here (n_graphs, n_cells).
    Returns W in [0, 1], where 1 = perfect agreement.
    """
    n_judges, n_subjects = ranks.shape
    # Sum of ranks per subject
    Rj = ranks.sum(axis=0)
    mean_Rj = Rj.mean()
    S = np.sum((Rj - mean_Rj) ** 2)
    W = 12 * S / (n_judges ** 2 * (n_subjects ** 3 - n_subjects))
    return float(W)


def pages_l_test(data: np.ndarray) -> tuple:
    """Page's L test for ordered alternatives.

    data: shape (n, k) -- n subjects (cells), k ordered conditions (dose levels).
    Returns (L, p_value) using the normal approximation.
    L = sum_{j=1}^{k} j * R_j  where R_j is the column rank-sum (ranks within each row).
    """
    n, k = data.shape
    # Rank each row
    from scipy.stats import rankdata, norm
    row_ranks = np.array([rankdata(row) for row in data])
    Rj = row_ranks.sum(axis=0)                  # column rank sums
    L = float(np.sum((np.arange(1, k + 1)) * Rj))
    # Normal approximation (Page 1963)
    mu_L = n * k * (k + 1) ** 2 / 4
    sigma2_L = n * k ** 2 * (k + 1) * (k - 1) * (k + 1) / 144
    z = (L - mu_L) / np.sqrt(sigma2_L)
    p = 1 - norm.cdf(z)                          # one-sided: H1 = increasing trend
    return L, p


# --------------------------------------------------------------------------- #
# v6a: multi-wiring-seed graph robustness
# --------------------------------------------------------------------------- #

def build_v6a(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to v6a analysis: ncp wiring, noise stress, no stress_level, CELLS_V6.

    Returns a DataFrame with one row per (cell_base, ncp_wiring_seed, system, seed).
    The 5 wiring seeds are [42, 7, 13, 21, 99].
    """
    mask = (
        (df['wiring'] == 'ncp') &
        (df['stress'] == 'noise') &
        (df['stress_level'].isna()) &
        (df['cell_base'].isin(CELLS_V6))
    )
    return df[mask].copy()


def compute_v6a_means(v6a: pd.DataFrame) -> pd.DataFrame:
    """Compute per-system-balanced mean NRMSE for each (cell_base, ncp_wiring_seed)."""
    records = []
    for wseed in WIRING_SEEDS:
        for cell in CELLS_V6:
            sub = v6a[(v6a['ncp_wiring_seed'] == wseed) & (v6a['cell_base'] == cell)]
            records.append({
                'cell_base': cell,
                'ncp_wiring_seed': wseed,
                'mean_nrmse': balanced_mean(sub),
            })
    return pd.DataFrame(records)


def v6a_statistics(v6a: pd.DataFrame, means_df: pd.DataFrame) -> dict:
    """Kendall's W + Friedman test for rank concordance; pairwise bio vs classical."""
    from scipy.stats import binom
    # Build rank matrix: shape (n_graphs=5, n_cells=8)
    pivot = means_df.pivot(index='ncp_wiring_seed', columns='cell_base', values='mean_nrmse')
    pivot = pivot[CELLS_V6]   # enforce column order
    rank_matrix = pivot.apply(lambda row: pd.Series(row.rank().values), axis=1).values.astype(float)

    W = kendalls_w(rank_matrix)
    # Friedman: graphs as blocks -> transpose so cells are subjects per test
    try:
        _, p_friedman = friedmanchisquare(*[pivot[c].values for c in CELLS_V6])
    except Exception:
        p_friedman = float('nan')

    # Pairwise bio vs classical: per-graph Wilcoxon, then sign test (count wins)
    pairwise = {}
    for bio_cell in BIO:
        for cls_cell in CLASSICAL:
            wins = 0
            ps = []
            for wseed in WIRING_SEEDS:
                sub_bio = v6a[(v6a['ncp_wiring_seed'] == wseed) & (v6a['cell_base'] == bio_cell)]
                sub_cls = v6a[(v6a['ncp_wiring_seed'] == wseed) & (v6a['cell_base'] == cls_cell)]
                merged = sub_bio.merge(sub_cls, on=['system', 'seed'], suffixes=('_b', '_c'))
                if len(merged) < 3:
                    ps.append(float('nan'))
                    continue
                bd = merged['nrmse_b'].values
                cd = merged['nrmse_c'].values
                try:
                    # two-sided Wilcoxon (project standard, matches analyze_v6) AND
                    # direction-aware: a "win" requires bio to be BOTH significantly
                    # different (p<0.05) AND the better one (median of paired diff
                    # < 0). A bare mean/p comparison is direction-blind and would
                    # wrongly credit graphs where the classical cell wins.
                    _, p = wilcoxon(bd, cd)
                    ps.append(p)
                    if np.median(bd - cd) < 0 and p < 0.05:
                        wins += 1
                except ValueError:
                    ps.append(float('nan'))
            pairwise[(bio_cell, cls_cell)] = {'wins': wins, 'p_per_graph': ps}

    return {'W': W, 'p_friedman': p_friedman, 'pairwise': pairwise, 'rank_matrix': rank_matrix, 'pivot': pivot}


def fig_v6a_ranking_lines(means_df: pd.DataFrame, stats: dict, outdir: str) -> str:
    """Line plot: per-graph mean NRMSE for 8 cells across the 5 wiring seeds.

    One subplot per wiring seed; x-axis = cells ordered by rank within that graph.
    Additionally: an overlay subplot showing all 5 graphs on one axis.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Left: one line per cell, x = wiring seed ---
    ax = axes[0]
    x_ticks = np.arange(len(WIRING_SEEDS))
    x_labels = [WIRING_SEED_LABELS[s] for s in WIRING_SEEDS]

    for cell in CELL_ORDER_V6:
        row = means_df[means_df['cell_base'] == cell].set_index('ncp_wiring_seed')
        y = [row.loc[ws, 'mean_nrmse'] if ws in row.index else float('nan') for ws in WIRING_SEEDS]
        y_clipped = [min(v, CLIP_NRMSE) if not np.isnan(v) else float('nan') for v in y]
        is_bio = cell in BIO
        color = CELL_COLOR[cell]
        ax.plot(
            x_ticks, y_clipped,
            marker='o' if is_bio else 's',
            linewidth=2.2 if is_bio else 1.4,
            linestyle='-' if is_bio else '--',
            color=color,
            markersize=7 if is_bio else 5,
            label=cell,
            alpha=0.9,
            zorder=3 if is_bio else 2,
        )

    ax.axhline(DIVERGENCE_THRESH, color='red', linestyle=':', linewidth=1.0,
               zorder=0, label='divergence (NRMSE=1)')
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel('Mean NRMSE (per-system balanced, capped at 4.0)', fontsize=9)
    ax.set_xlabel('NCP wiring graph seed', fontsize=9)
    ax.set_ylim(0, None)
    ax.grid(alpha=0.2)
    ax.set_title('v6a: Mean NRMSE per cell across wiring graphs\n(noise regime, NCP only)', fontsize=10)
    ax.legend(fontsize=8, frameon=False, ncol=2, loc='upper right')

    W = stats['W']
    # Kendall's W is the rank-concordance statistic that matters here. Friedman is
    # deliberately NOT shown: on only 5 graph blocks it is underpowered and is not
    # the relevant test for "is the cell ordering graph-invariant" (W is).
    ax.text(0.02, 0.98, f"Kendall's W = {W:.3f}\n(rank concordance, 5 graphs)",
            transform=ax.transAxes, va='top', ha='left', fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # --- Right: pairwise bio vs classical win summary (dot matrix) ---
    ax2 = axes[1]
    bio_cells = BIO
    cls_cells = CLASSICAL
    pairwise = stats['pairwise']

    matrix = np.zeros((len(bio_cells), len(cls_cells)))
    for ri, bio_cell in enumerate(bio_cells):
        for ci, cls_cell in enumerate(cls_cells):
            matrix[ri, ci] = pairwise[(bio_cell, cls_cell)]['wins']

    im = ax2.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=5, aspect='auto')
    ax2.set_xticks(range(len(cls_cells)))
    ax2.set_xticklabels(cls_cells, fontsize=9)
    ax2.set_yticks(range(len(bio_cells)))
    ax2.set_yticklabels(bio_cells, fontsize=9)
    ax2.set_xlabel('Classical cell', fontsize=9)
    ax2.set_ylabel('Bio champion', fontsize=9)

    for ri in range(len(bio_cells)):
        for ci in range(len(cls_cells)):
            wins = int(matrix[ri, ci])
            ax2.text(ci, ri, f'{wins}/5', ha='center', va='center',
                     fontsize=11, fontweight='bold',
                     color='white' if wins >= 4 or wins <= 1 else 'black')

    cbar = fig.colorbar(im, ax=ax2, shrink=0.7, pad=0.02)
    cbar.set_label('# graphs where bio beats classical (of 5)', fontsize=8)
    ax2.set_title('Bio vs classical win count across graphs\n(Wilcoxon paired p<0.05, n=30 per graph)',
                  fontsize=10)

    fig.suptitle(
        'v6a: Wiring-graph robustness -- does the bio-vs-classical ordering survive '
        'other NCP wiring seeds?\n'
        'Left: mean NRMSE trajectories across 5 wiring graphs. '
        'Right: #graphs where each bio cell significantly outperforms each classical cell.',
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    path = os.path.join(outdir, 'fig1_v6a_ranking_lines.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  {path}')
    return path


def fig_v6a_rank_heatmap(means_df: pd.DataFrame, stats: dict, outdir: str) -> str:
    """Rank heatmap: cells (rows) x wiring-graph seeds (cols), value = rank (1=best).

    Low rank (=1, dark green) = best. High rank (=8, red) = worst.
    Stable rank across columns => graph-invariant ordering.
    """
    pivot = stats['pivot']   # shape (5 graphs x 8 cells)
    rank_df = pivot.rank(axis=1, ascending=True).astype(int)   # rank per graph

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(rank_df.values.T, cmap='RdYlGn_r', vmin=1, vmax=8,
                   aspect='auto', interpolation='nearest')

    ax.set_xticks(range(len(WIRING_SEEDS)))
    ax.set_xticklabels([WIRING_SEED_LABELS[s] for s in pivot.index], fontsize=9)
    ax.set_yticks(range(len(CELLS_V6)))
    ax.set_yticklabels(CELLS_V6, fontsize=9)
    ax.set_xlabel('NCP wiring graph seed', fontsize=9)
    ax.set_ylabel('Cell', fontsize=9)

    # Bold y-tick labels for bio champions
    for tick, cell in zip(ax.get_yticklabels(), CELLS_V6):
        if cell in BIO:
            tick.set_fontweight('bold')

    # Annotate each cell
    for ri, wseed in enumerate(pivot.index):
        for ci, cell in enumerate(CELLS_V6):
            rank_val = rank_df.loc[wseed, cell]
            ax.text(ri, ci, str(rank_val), ha='center', va='center',
                    fontsize=11, fontweight='bold',
                    color='white' if rank_val >= 6 or rank_val <= 2 else 'black')

    # Horizontal separator between bio and classical
    ax.axhline(len(BIO) - 0.5, color='black', linewidth=1.5)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Rank (1 = lowest mean NRMSE = best)', fontsize=8)

    W = stats['W']
    ax.set_title(
        f'v6a: Cell ranking across NCP wiring graphs (noise regime)\n'
        f"Kendall's W = {W:.3f}  "
        f'  (W>0.7 = strong concordance; bold rows = bio champions)',
        fontsize=10,
    )
    fig.tight_layout()
    path = os.path.join(outdir, 'fig2_v6a_rank_heatmap.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  {path}')
    return path


# --------------------------------------------------------------------------- #
# v6b: dose-response
# --------------------------------------------------------------------------- #

def build_v6b_curve(df: pd.DataFrame, stressor: str, cell: str) -> tuple:
    """Build the 4-point dose-response curve (severity levels, mean NRMSE).

    Uses per-system-balanced means throughout.
    Clean anchor: v4 rows, seeds 0-4, ncp wiring (n=30, 5 per system).
    Mid level (v5): stress=stressor, stress_level=NaN, ncp_wiring_seed=42.
    v6b new levels: stress=stressor, stress_level=value, ncp_wiring_seed=42.
    """
    dose_info = DOSE_LEVELS[stressor]
    severity_levels = dose_info['levels']   # ordered easy -> hard

    # V5 mid-level anchors (stress_level=NaN)
    v5_mid = {
        'noise': 0.10,
        'ood_init': 0.20,
        'extrapolation': 0.50,
    }
    mid_val = v5_mid[stressor]

    nrmses = []
    for sev in severity_levels:
        if sev == severity_levels[0]:
            # Clean anchor (easiest level: sigma=0 / ood_scale=0 / train_frac=1.0)
            sub = df[
                (df['stress'].isna()) &
                (df['wiring'] == 'ncp') &
                (df['cell_base'] == cell) &
                (df['seed'] < 5)
            ]
        elif np.isclose(sev, mid_val):
            # V5 mid-level anchor: stress set but stress_level is NaN
            sub = df[
                (df['stress'] == stressor) &
                (df['stress_level'].isna()) &
                (df['wiring'] == 'ncp') &
                (df['ncp_wiring_seed'] == 42) &
                (df['cell_base'] == cell)
            ]
        else:
            # v6b new level
            sub = df[
                (df['stress'] == stressor) &
                (df['stress_level'].notna()) &
                (np.isclose(df['stress_level'].fillna(-99), sev)) &
                (df['wiring'] == 'ncp') &
                (df['ncp_wiring_seed'] == 42) &
                (df['cell_base'] == cell)
            ]
        nrmses.append(balanced_mean(sub))
    return severity_levels, nrmses


def v6b_statistics(df: pd.DataFrame, stressor: str) -> dict:
    """Page's L test for monotone trend + bio<classical check at every level."""
    dose_info = DOSE_LEVELS[stressor]
    severity_levels = dose_info['levels']
    n_levels = len(severity_levels)

    cell_nrmses = {}
    for cell in CELLS_V6:
        _, nrmses = build_v6b_curve(df, stressor, cell)
        cell_nrmses[cell] = nrmses

    # Page's L across ALL cells: data shape (n_cells, n_levels)
    data_matrix = np.array([cell_nrmses[c] for c in CELLS_V6])
    L, p_L = pages_l_test(data_matrix)

    # Floor/ceiling detection (matches the analysis definition, NOT the all-cell
    # mean -- the mean hides that lstm sits at ~0.24 while lrc inflates it). floor =
    # a level where even the WORST non-lrc cell is fine (< 0.15) so nobody is
    # stressed; ceiling = a level where even the BEST cell diverges (> 0.5). lrc is
    # excluded from the floor test: it is structurally weak at every level incl.
    # clean, so it would mask a genuine floor. Per the analysis, neither occurs.
    level_means = np.nanmean(data_matrix, axis=0)
    non_lrc = [c for c in CELLS_V6 if c != 'lrc']
    floor_level = None
    ceiling_level = None
    for i, sev in enumerate(severity_levels):
        worst_non_lrc = np.nanmax([cell_nrmses[c][i] for c in non_lrc])
        best_all = np.nanmin([cell_nrmses[c][i] for c in CELLS_V6])
        if worst_non_lrc < 0.15:   # everyone (bar lrc) is fine -> floor
            floor_level = sev
        if best_all > 0.5:         # even the best diverges -> ceiling
            ceiling_level = sev

    # Bio < classical at every level?
    bio_wins_per_level = []
    for i in range(n_levels):
        bio_vals = [cell_nrmses[c][i] for c in BIO if not np.isnan(cell_nrmses[c][i])]
        cls_vals = [cell_nrmses[c][i] for c in CLASSICAL if not np.isnan(cell_nrmses[c][i])]
        if bio_vals and cls_vals:
            bio_wins_per_level.append(bool(np.mean(bio_vals) < np.mean(cls_vals)))
        else:
            bio_wins_per_level.append(None)

    return {
        'L': L, 'p_L': p_L,
        'cell_nrmses': cell_nrmses,
        'level_means': level_means,
        'floor_level': floor_level,
        'ceiling_level': ceiling_level,
        'bio_wins_per_level': bio_wins_per_level,
    }


def fig_v6b_dose_response(df: pd.DataFrame, stressor: str, stats: dict, outdir: str,
                           fig_name: str) -> str:
    """Line plot: mean NRMSE vs dose level (severity) for all 8 cells.

    Bio champions: thicker lines, circle markers.
    Classical: thinner, square markers.
    lrc: dashed, diamond.
    x-axis ordered by severity (easy -> hard).
    """
    dose_info = DOSE_LEVELS[stressor]
    severity_levels = dose_info['levels']
    xlabels = dose_info['labels']
    xlabel = dose_info['xlabel']

    fig, ax = plt.subplots(figsize=(10, 6))

    x_pos = np.arange(len(severity_levels))

    for cell in CELL_ORDER_V6:
        _, nrmses = build_v6b_curve(df, stressor, cell)
        nrmses_clipped = [min(v, CLIP_NRMSE) if not np.isnan(v) else float('nan') for v in nrmses]
        is_bio = cell in BIO
        is_classical = cell in CLASSICAL
        color = CELL_COLOR[cell]

        ax.plot(
            x_pos, nrmses_clipped,
            marker='o' if is_bio else ('s' if is_classical else 'D'),
            linewidth=2.5 if is_bio else 1.5,
            linestyle='-' if is_bio else ('--' if is_classical else ':'),
            color=color,
            markersize=8 if is_bio else 6,
            label=cell,
            alpha=0.92 if is_bio else 0.80,
            zorder=3 if is_bio else 2,
        )

    ax.axhline(DIVERGENCE_THRESH, color='red', linestyle=':', linewidth=1.2,
               zorder=0, label='divergence (NRMSE=1)')

    # Shade the v5 mid-level x position
    mid_idx = 2   # 3rd severity level is the v5 anchor
    y_top_shade = max(
        max(min(v, CLIP_NRMSE) for vals in [
            [min(nrmses_clipped, default=0) for nrmses_clipped in
             [[min(v2, CLIP_NRMSE) if not np.isnan(v2) else 0 for v2 in build_v6b_curve(df, stressor, c)[1]]
              for c in CELLS_V6]
            ]
        ] for v in vals),
        DIVERGENCE_THRESH * 1.1
    ) if False else 0.02   # just place v5 label at bottom
    ax.axvspan(mid_idx - 0.3, mid_idx + 0.3, alpha=0.08, color='gray', zorder=0)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('Mean NRMSE (per-system balanced, capped at 4.0)', fontsize=9)
    ax.set_ylim(0, None)
    ax.grid(alpha=0.2)

    # Add v5 anchor label after axes are set
    y_lim = ax.get_ylim()
    ax.text(mid_idx, y_lim[0] + (y_lim[1] - y_lim[0]) * 0.01,
            'v5 anchor', ha='center', va='bottom', fontsize=7, color='0.5', style='italic')

    # Statistical annotation
    L = stats['L']
    p_L = stats['p_L']
    bio_wins = stats['bio_wins_per_level']
    floor_lvl = stats['floor_level']
    ceiling_lvl = stats['ceiling_level']

    bio_str = 'bio<cls: ' + ', '.join(
        ('Y' if w else ('N' if w is False else '?')) for w in bio_wins
    )
    floor_str = f'floor at {floor_lvl}' if floor_lvl is not None else ''
    ceiling_str = f'ceiling at {ceiling_lvl}' if ceiling_lvl is not None else ''
    fc_str = ' | '.join(s for s in [floor_str, ceiling_str] if s) or 'no floor/ceiling'

    stats_text = (
        f"Page's L = {L:.1f}, p = {p_L:.3g}\n"
        f"{bio_str}\n"
        f"{fc_str}"
    )
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            va='top', ha='right', fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.75))

    # Legend
    legend_handles = []
    for cell in CELL_ORDER_V6:
        is_bio = cell in BIO
        is_classical = cell in CLASSICAL
        marker = 'o' if is_bio else ('s' if is_classical else 'D')
        ls = '-' if is_bio else ('--' if is_classical else ':')
        lw = 2.5 if is_bio else 1.5
        legend_handles.append(
            Line2D([], [], color=CELL_COLOR[cell], marker=marker,
                   linestyle=ls, linewidth=lw, markersize=6, label=cell)
        )
    legend_handles.append(
        Line2D([], [], color='red', linestyle=':', linewidth=1.2, label='divergence')
    )
    ax.legend(handles=legend_handles, fontsize=8, frameon=False, ncol=2,
              loc='upper left', bbox_to_anchor=(0.0, 1.0))

    stressor_label = {'noise': 'Noise (sigma)', 'ood_init': 'OOD Init (offset scale)',
                      'extrapolation': 'Extrapolation (train fraction)'}[stressor]
    fig.suptitle(
        f'v6b: Dose-response -- {stressor_label} (NCP wiring, wiring seed 42)\n'
        f'x-axis ordered easy -> hard; v5 anchor = grey band; n=30 per level per cell; '
        f'clean from v4 seeds 0-4 (n=30 per cell)',
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    path = os.path.join(outdir, fig_name)
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  {path}')
    return path


# --------------------------------------------------------------------------- #
# Statistics report
# --------------------------------------------------------------------------- #

def print_statistics_report(df: pd.DataFrame, v6a: pd.DataFrame, means_df: pd.DataFrame,
                             v6a_stats: dict) -> None:
    print('\n' + '=' * 70)
    print('STATISTICS REPORT -- v6')
    print('=' * 70)

    # v6a: Kendall's W / Friedman
    W = v6a_stats['W']
    p_f = v6a_stats['p_friedman']
    print(f"\n[v6a] Kendall's W (rank concordance across 5 wiring graphs): {W:.4f}")
    # Friedman on 5 graph-blocks is underpowered and orientation-sensitive; it is
    # NOT the relevant test (W is) and is printed only for completeness, not shown
    # on the figures. Do not cite this p-value.
    print(f"[v6a] Friedman chi2 p-value (underpowered, not the relevant test): {p_f:.4g}")
    if W >= 0.7:
        print("  -> Strong concordance: ordering is graph-invariant")
    elif W >= 0.5:
        print("  -> Moderate concordance")
    else:
        print("  -> Weak concordance: ordering sensitive to wiring graph")

    print("\n[v6a] Bio vs classical pairwise sign test (wins out of 5 graphs):")
    pairwise = v6a_stats['pairwise']
    from scipy.stats import binom
    for bio_cell in BIO:
        for cls_cell in CLASSICAL:
            data = pairwise[(bio_cell, cls_cell)]
            wins = data['wins']
            ps = data['p_per_graph']
            p_vals_str = ', '.join(f'{p:.3f}' if not np.isnan(p) else 'NaN' for p in ps)
            # Sign test: Pr(X >= wins | p=0.5, n=5), one-sided
            p_sign = float(binom.sf(wins - 1, 5, 0.5))
            unanimous = wins == 5
            flag = ' ***ROBUST (5/5)' if unanimous else (' **4/5' if wins == 4 else '')
            print(f"  {bio_cell} vs {cls_cell}: {wins}/5 wins  p_sign={p_sign:.4f}{flag}")
            print(f"    per-graph Wilcoxon p: [{p_vals_str}]")

    # v6b: Page's L
    print("\n[v6b] Page's L monotone trend test (all 8 cells pooled):")
    for stressor in ['noise', 'ood_init', 'extrapolation']:
        stats = v6b_statistics(df, stressor)
        L = stats['L']
        p_L = stats['p_L']
        bio_wins = stats['bio_wins_per_level']
        print(f"  {stressor}: L={L:.1f}, p={p_L:.4g}, bio<cls at each level: {bio_wins}")
        if stats['floor_level'] is not None:
            print(f"    FLOOR detected at level {stats['floor_level']}")
        if stats['ceiling_level'] is not None:
            print(f"    CEILING detected at level {stats['ceiling_level']}")

    print()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description='Plot v6 benchmark figures (graph robustness + dose-response).'
    )
    p.add_argument('--out', default='results/figures_v6',
                   help='Output directory for PNG figures.')
    args = p.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)

    print('Loading all runs from results/runs_v4, runs_v5, runs_v6a, runs_v6b ...')
    df = load_data()
    print(f'  Total rows: {len(df)}')
    print(f'  Cells (base): {sorted(df["cell_base"].unique())}')
    print(f'  Stress levels: {sorted(df["stress_level"].dropna().unique())}')
    print(f'  Wiring seeds:  {sorted(df["ncp_wiring_seed"].dropna().unique())}')

    # ------------------------------------------------------------------ v6a --
    print('\n--- v6a: multi-wiring-seed graph robustness ---')
    v6a = build_v6a(df)
    print(f'  v6a rows: {len(v6a)} | wiring_seeds: {sorted(v6a["ncp_wiring_seed"].unique())}')
    means_df = compute_v6a_means(v6a)
    v6a_stats = v6a_statistics(v6a, means_df)

    produced = []
    print('Generating figures:')
    produced.append(fig_v6a_ranking_lines(means_df, v6a_stats, args.out))
    produced.append(fig_v6a_rank_heatmap(means_df, v6a_stats, args.out))

    # ------------------------------------------------------------------ v6b --
    print('\n--- v6b: dose-response ---')
    for stressor, figname in [
        ('noise',         'fig3_v6b_noise.png'),
        ('extrapolation', 'fig4_v6b_extrapolation.png'),
        ('ood_init',      'fig5_v6b_ood_init.png'),
    ]:
        stats_s = v6b_statistics(df, stressor)
        produced.append(fig_v6b_dose_response(df, stressor, stats_s, args.out, figname))

    # Print full statistics report
    print_statistics_report(df, v6a, means_df, v6a_stats)

    print(f'Done. {len(produced)} figures in {args.out}/')
    return 0


if __name__ == '__main__':
    sys.exit(main())
