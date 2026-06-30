# experiments/plot_v5.py
"""Publication-quality figures for the v5 generalization-stress benchmark.

Loads clean baseline runs (v4, results/runs_v4/) and stress-test runs
(v5, results/runs_v5/) and produces four figures into results/figures_v5/:

  fig1_nrmse_regimes_ncp.png     -- 4-panel boxplots, one per regime, NCP only,
                                     x = 11 cells ordered by family.
  fig2_degradation_ncp.png       -- Per-cell degradation overlay: mean NRMSE across
                                     clean + 3 stress regimes (grouped bars).
  fig3_divergence_heatmap_ncp.png -- Heatmap: cells x regimes, value = % runs NRMSE>1.
  fig4_bio_vs_classical_ncp.png  -- Mean NRMSE per regime: bio champions vs classical
                                     cells (line chart, NCP wiring).

Axis / outlier policy:
  - NRMSE values > CLIP_NRMSE (4.0) are clipped for visualisation purposes;
    this is annotated in each figure title.  The divergence threshold (NRMSE>1)
    uses the unclipped values to compute rates correctly.
  - Log y-scale is used for boxplots; linear scale for heatmap and grouped bars.

Usage (from repo root, requires the uv venv):
    uv run python experiments/plot_v5.py [--v4 results/runs_v4] [--v5 results/runs_v5]
                                         [--out results/figures_v5]
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
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Cell ordering: LTC family | LRC family | classical gated | vanilla CT
CELL_ORDER = [
    'ltc', 'mm_ltc', 'cfc_mm_ltc', 'cfc',          # LTC family
    'lrc', 'mm_lrc', 'cfc_mm_lrc', 'cfc_lrc',       # LRC family
    'gru', 'lstm',                                    # classical gated
    'ctrnn',                                          # vanilla continuous-time
]

# Family membership for color coding
FAMILY = {
    'ltc':       'ltc_family',
    'mm_ltc':    'ltc_family',
    'cfc_mm_ltc': 'ltc_family',
    'cfc':       'ltc_family',
    'lrc':       'lrc_family',
    'mm_lrc':    'lrc_family',
    'cfc_mm_lrc': 'lrc_family',
    'cfc_lrc':   'lrc_family',
    'gru':       'classical',
    'lstm':      'classical',
    'ctrnn':     'classical',
}

# "Closed-form bio" champions (filled markers / darker shade in degradation plot)
BIO_CHAMPIONS = {'cfc', 'cfc_lrc', 'cfc_mm_lrc', 'cfc_mm_ltc'}

# Palette (tab10-inspired, family-consistent)
FAMILY_COLORS = {
    'ltc_family': '#4C72B0',   # blue
    'lrc_family': '#DD8452',   # orange
    'classical':  '#55A868',   # green
}

REGIME_ORDER = ['clean', 'noise', 'extrapolation', 'ood_init']
REGIME_LABELS = {
    'clean':         'Clean\n(baseline v4)',
    'noise':         'Noise\n(sigma=10% std)',
    'extrapolation': 'Extrapolation\n(50% train)',
    'ood_init':      'OOD init\n(y0 +/-20%)',
}

DIVERGENCE_THRESH = 1.0   # NRMSE above this = forward rollout failure
CLIP_NRMSE = 4.0          # visual cap for log-axis plots (unclipped for div. rate)

DPI = 200   # output resolution


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_data(v4_dir: str, v5_dir: str) -> pd.DataFrame:
    """Load v4 (clean) and v5 (stress) runs into a unified long DataFrame.

    Adds a 'regime' column: 'clean' for v4 rows, otherwise the stress name.
    The 'cell_base' column strips any +suffix so groupby on base cell works.
    """
    from experiments.aggregate_results import load_runs
    df = load_runs([v4_dir, v5_dir])

    # Derive regime from the stress column (None/NaN -> 'clean')
    df['regime'] = df['stress'].fillna('clean')

    # Base cell name: strip the '+{regime}' suffix that load_runs appends
    df['cell_base'] = df['cell'].str.replace(
        r'\+(noise|extrapolation|ood_init)$', '', regex=True
    )
    return df


def ncp_df(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to NCP wiring and known cells in CELL_ORDER."""
    mask = (df['wiring'] == 'ncp') & (df['cell_base'].isin(CELL_ORDER))
    return df[mask].copy()


def balanced_mean(sub: pd.DataFrame) -> float:
    """Per-system equal-weighted mean of NRMSE (mean of per-system means).

    The clean (v4) data is n=40 per cell on ncp: duffing & periodic_predator_prey
    carry 10 seeds each (the v4 tail seeds 5-9), the other 4 systems 5 each. A flat
    .mean() therefore overweights those two stiff, high-NRMSE systems and inflates
    the clean baseline, which would understate the clean->stress degradation in a
    headline comparison figure. The stress (v5) data is n=30, balanced 5-per-system,
    so averaging per-system means first puts every system on equal footing and makes
    clean and stress bars directly comparable.
    """
    if len(sub) == 0:
        return float('nan')
    return float(sub.groupby('system')['nrmse'].mean().mean())


# --------------------------------------------------------------------------- #
# Figure 1: Boxplots per regime (4 panels, NCP)
# --------------------------------------------------------------------------- #

def fig_nrmse_regimes_ncp(df_ncp: pd.DataFrame, outdir: str) -> str:
    """4-panel figure: one box plot per regime, cells on x-axis (NCP wiring).

    Y-axis uses log scale; values above CLIP_NRMSE are capped for display.
    The red dashed line marks the divergence threshold (NRMSE=1).
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.5), sharey=True)

    # Shared y limits: slightly below min positive, slightly above cap
    y_lo = max(df_ncp['nrmse'].clip(upper=CLIP_NRMSE).min() * 0.5, 1e-4)
    y_hi = CLIP_NRMSE * 1.3

    cell_colors = [FAMILY_COLORS[FAMILY[c]] for c in CELL_ORDER]
    x_positions = np.arange(len(CELL_ORDER))

    for ax, regime in zip(axes, REGIME_ORDER):
        sub = df_ncp[df_ncp['regime'] == regime]
        box_data = []
        for cell in CELL_ORDER:
            vals = sub.loc[sub['cell_base'] == cell, 'nrmse'].values
            # Clip for display; keep NaN bucket if no runs
            clipped = np.clip(vals, a_min=None, a_max=CLIP_NRMSE) if len(vals) else np.array([])
            box_data.append(clipped)

        bp = ax.boxplot(
            box_data,
            positions=x_positions,
            widths=0.55,
            patch_artist=True,
            showfliers=True,
            flierprops=dict(marker='.', markersize=3, alpha=0.5, linestyle='none'),
            medianprops=dict(color='black', linewidth=1.4),
            whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0),
        )
        for patch, col in zip(bp['boxes'], cell_colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.72)

        # Mark bio champions with a thicker box edge
        for patch, cell in zip(bp['boxes'], CELL_ORDER):
            if cell in BIO_CHAMPIONS:
                patch.set_linewidth(2.0)
                patch.set_edgecolor('black')

        ax.axhline(DIVERGENCE_THRESH, color='red', linestyle='--',
                   linewidth=1.0, zorder=0, label='divergence (NRMSE=1)')
        ax.set_yscale('log')
        ax.set_ylim(y_lo, y_hi)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(CELL_ORDER, rotation=40, ha='right', fontsize=8)
        ax.set_title(REGIME_LABELS[regime], fontsize=10)
        ax.grid(axis='y', alpha=0.2, which='both')
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    axes[0].set_ylabel('NRMSE (log, capped at 4.0)', fontsize=10)

    # Legend: family colors + bio champion marker + divergence line
    legend_handles = [
        Patch(facecolor=FAMILY_COLORS['ltc_family'], alpha=0.72, label='LTC family'),
        Patch(facecolor=FAMILY_COLORS['lrc_family'], alpha=0.72, label='LRC family'),
        Patch(facecolor=FAMILY_COLORS['classical'],  alpha=0.72, label='Classical gated / CT'),
        Patch(facecolor='white', edgecolor='black', linewidth=2.0, label='Bio champion (cfc*)'),
        Line2D([], [], color='red', linestyle='--', label='Divergence (NRMSE=1)'),
    ]
    fig.legend(handles=legend_handles, loc='lower center',
               bbox_to_anchor=(0.5, -0.02), ncol=5, fontsize=9, frameon=False)

    fig.suptitle(
        'v5 Generalization Stress -- NRMSE per regime (NCP wiring, 11 cells)\n'
        'clean n=40/cell (v4: +tail seeds 5-9 on duffing & periodic_pp) | stress n=30 (seeds 0-4 x 6 systems)\n'
        'Outliers capped at NRMSE=4 for display; unclipped values used for divergence rate.',
        fontsize=11, y=1.01,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    path = os.path.join(outdir, 'fig1_nrmse_regimes_ncp.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  {path}')
    return path


# --------------------------------------------------------------------------- #
# Figure 2: Degradation overlay (grouped bars, NCP)
# --------------------------------------------------------------------------- #

def fig_degradation_ncp(df_ncp: pd.DataFrame, outdir: str) -> str:
    """Grouped bar chart: mean NRMSE per cell (x) across all four regimes (hue).

    One bar group per cell, four bars per group (one per regime).
    Bio champions are shown with a bold border. Y-axis linear (mean NRMSE).
    """
    regime_colors = {
        'clean':         '#4C72B0',
        'noise':         '#DD8452',
        'extrapolation': '#C44E52',
        'ood_init':      '#8172B2',
    }

    n_cells = len(CELL_ORDER)
    n_regimes = len(REGIME_ORDER)
    group_width = 0.8
    bar_width = group_width / n_regimes
    x = np.arange(n_cells)

    fig, ax = plt.subplots(figsize=(14, 5.5))

    for ri, regime in enumerate(REGIME_ORDER):
        sub = df_ncp[df_ncp['regime'] == regime]
        means = []
        for cell in CELL_ORDER:
            # Per-system balanced (not flat): clean v4 is n=40 stiff-overweighted,
            # stress v5 is n=30 balanced -- balance both so the comparison is fair.
            means.append(balanced_mean(sub.loc[sub['cell_base'] == cell]))

        offsets = x + (ri - n_regimes / 2 + 0.5) * bar_width
        bars = ax.bar(
            offsets, means, width=bar_width * 0.92,
            color=regime_colors[regime], alpha=0.85,
            label=regime.replace('_', ' '),
        )

        # Annotate with the numeric mean if bar is small enough to read
        for bar, mean_val in zip(bars, means):
            if not np.isnan(mean_val) and mean_val < 0.5:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f'{mean_val:.2f}',
                    ha='center', va='bottom', fontsize=5.5, rotation=90, color='0.3',
                )

    ax.axhline(DIVERGENCE_THRESH, color='red', linestyle='--',
               linewidth=1.0, label='divergence (NRMSE=1)', zorder=0)

    # Bold ticks for bio champions
    ax.set_xticks(x)
    ax.set_xticklabels(CELL_ORDER, rotation=38, ha='right', fontsize=9)
    for tick, cell in zip(ax.get_xticklabels(), CELL_ORDER):
        if cell in BIO_CHAMPIONS:
            tick.set_fontweight('bold')

    ax.set_ylabel('Mean NRMSE (linear)', fontsize=10)
    ax.set_xlabel('Cell type (bold = bio champion)', fontsize=9)
    ax.grid(axis='y', alpha=0.2)
    ax.set_ylim(0, None)
    ax.legend(fontsize=9, frameon=False, ncol=5, loc='upper left')

    # Family separator lines
    ltc_end = CELL_ORDER.index('cfc') + 0.5
    lrc_end = CELL_ORDER.index('cfc_lrc') + 0.5
    ax.axvline(ltc_end, color='0.6', linestyle=':', linewidth=0.8)
    ax.axvline(lrc_end, color='0.6', linestyle=':', linewidth=0.8)

    # Family labels -- position after drawing so y_top is accurate
    y_top = ax.get_ylim()[1]
    ax.text(ltc_end / 2, y_top * 0.97, 'LTC family', ha='center',
            fontsize=8, color='0.4', style='italic')
    ax.text((ltc_end + lrc_end) / 2, y_top * 0.97, 'LRC family', ha='center',
            fontsize=8, color='0.4', style='italic')
    ax.text((lrc_end + n_cells - 1) / 2, y_top * 0.97, 'Classical', ha='center',
            fontsize=8, color='0.4', style='italic')

    fig.suptitle(
        'Degradation overlay -- per-system-balanced mean NRMSE per regime (NCP wiring)\n'
        'clean (n=40) & stress (n=30) equal-weighted per system for a fair comparison; '
        'bold x-labels = closed-form bio champions (cfc*)',
        fontsize=11,
    )
    fig.tight_layout()
    path = os.path.join(outdir, 'fig2_degradation_ncp.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  {path}')
    return path


# --------------------------------------------------------------------------- #
# Figure 3: Divergence-rate heatmap (NCP)
# --------------------------------------------------------------------------- #

def fig_divergence_heatmap_ncp(df_ncp: pd.DataFrame, outdir: str) -> str:
    """Heatmap: cells (rows) x regimes (cols), value = % runs NRMSE>1 (NCP).

    Uses unclipped NRMSE values. Divergence threshold = 1.0.
    Color scale: white (0%) -> red (100%).
    """
    n_cells = len(CELL_ORDER)
    n_regimes = len(REGIME_ORDER)
    matrix = np.full((n_cells, n_regimes), np.nan)

    for ci, cell in enumerate(CELL_ORDER):
        for ri, regime in enumerate(REGIME_ORDER):
            sub = df_ncp[(df_ncp['cell_base'] == cell) & (df_ncp['regime'] == regime)]
            if len(sub) == 0:
                continue
            rate = 100.0 * (sub['nrmse'] > DIVERGENCE_THRESH).mean()
            matrix[ci, ri] = rate

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, aspect='auto', cmap='Reds', vmin=0, vmax=100,
                   interpolation='nearest')

    # Cell and regime annotations
    ax.set_yticks(range(n_cells))
    ax.set_yticklabels(CELL_ORDER, fontsize=9)
    ax.set_xticks(range(n_regimes))
    ax.set_xticklabels(
        [REGIME_LABELS[r].replace('\n', ' ') for r in REGIME_ORDER],
        fontsize=9, rotation=20, ha='right',
    )

    # Bold y-tick labels for bio champions
    for tick, cell in zip(ax.get_yticklabels(), CELL_ORDER):
        if cell in BIO_CHAMPIONS:
            tick.set_fontweight('bold')

    # Annotate each cell with the percentage
    for ci in range(n_cells):
        for ri in range(n_regimes):
            val = matrix[ci, ri]
            if np.isnan(val):
                continue
            text_color = 'white' if val > 55 else 'black'
            ax.text(ri, ci, f'{val:.0f}%', ha='center', va='center',
                    fontsize=8.5, color=text_color)

    # Family separator lines (horizontal, between cell groups)
    ltc_boundary = CELL_ORDER.index('cfc') + 0.5
    lrc_boundary = CELL_ORDER.index('cfc_lrc') + 0.5
    for y in [ltc_boundary, lrc_boundary]:
        ax.axhline(y, color='black', linewidth=1.5)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('Divergence rate (% runs NRMSE > 1)', fontsize=9)

    ax.set_title(
        'Forward-rollout divergence rate (NCP wiring)\n'
        'Bold rows = bio champions (cfc*). Black lines = family boundaries.',
        fontsize=10,
    )
    fig.tight_layout()
    path = os.path.join(outdir, 'fig3_divergence_heatmap_ncp.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  {path}')
    return path


# --------------------------------------------------------------------------- #
# Figure 4: Bio champions vs classical -- mean NRMSE per regime (NCP)
# --------------------------------------------------------------------------- #

def fig_bio_vs_classical_ncp(df_ncp: pd.DataFrame, outdir: str) -> str:
    """Line chart: bio champions vs classical cells, mean NRMSE across regimes.

    Each cell is one line; x-axis = regime ordered by expected difficulty.
    The y-axis is linear (mean NRMSE); log would compress the interesting
    region where bio and classical diverge.

    Complements Fig 1 (distribution) with a per-cell trajectory view so
    reviewers can trace which cells degrade and which stay flat.
    """
    # Group cells into three families for visual separation
    bio_cells = [c for c in CELL_ORDER if c in BIO_CHAMPIONS]
    classical_cells = ['gru', 'lstm', 'ctrnn']
    numerical_cells = ['ltc', 'mm_ltc', 'lrc', 'mm_lrc']

    group_specs = [
        (bio_cells,       'Bio champions (cfc*)', '#4C72B0', '-',  'o'),
        (classical_cells, 'Classical gated / CT', '#55A868', '--', 's'),
        (numerical_cells, 'Numerical LTC/LRC',    '#DD8452', ':',  '^'),
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for cells_in_group, _group_label, base_color, ls, marker in group_specs:
        # Slightly vary shade within group to keep individual lines distinguishable
        n = len(cells_in_group)
        shades = np.linspace(0.55, 1.0, n) if n > 1 else [0.85]

        for cell, shade in zip(cells_in_group, shades):
            means = []
            for regime in REGIME_ORDER:
                sub = df_ncp[(df_ncp['cell_base'] == cell) & (df_ncp['regime'] == regime)]
                means.append(balanced_mean(sub))   # per-system balanced (see fig2)

            # Blend base color toward white by shade factor
            rgb = matplotlib.colors.to_rgb(base_color)
            color = tuple(min(1.0, c * shade + (1 - shade) * 0.95) for c in rgb)

            ax.plot(
                range(len(REGIME_ORDER)), means,
                marker=marker, linestyle=ls, color=color,
                linewidth=1.8 if cell in BIO_CHAMPIONS else 1.2,
                markersize=7 if cell in BIO_CHAMPIONS else 5,
                label=cell,
                alpha=0.9,
                zorder=3 if cell in BIO_CHAMPIONS else 2,
            )

    ax.axhline(DIVERGENCE_THRESH, color='red', linestyle='--',
               linewidth=1.0, label='divergence (NRMSE=1)', zorder=0)
    ax.set_xticks(range(len(REGIME_ORDER)))
    ax.set_xticklabels(
        [REGIME_LABELS[r].replace('\n', ' ') for r in REGIME_ORDER],
        fontsize=10,
    )
    ax.set_ylabel('Mean NRMSE (linear)', fontsize=10)
    ax.grid(alpha=0.2)
    ax.set_ylim(0, None)
    ax.legend(fontsize=8.5, frameon=False, ncol=3, loc='upper left')

    fig.suptitle(
        'Bio champions vs classical cells -- mean NRMSE trajectory across regimes\n'
        '(NCP wiring; regime order = increasing expected difficulty)',
        fontsize=11,
    )
    fig.tight_layout()
    path = os.path.join(outdir, 'fig4_bio_vs_classical_ncp.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  {path}')
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description='Plot v5 generalization-stress benchmark figures.'
    )
    p.add_argument('--v4', default='results/runs_v4',
                   help='Directory with clean baseline (v4) run JSONs.')
    p.add_argument('--v5', default='results/runs_v5',
                   help='Directory with stress-test (v5) run JSONs.')
    p.add_argument('--out', default='results/figures_v5',
                   help='Output directory for PNG figures.')
    args = p.parse_args(argv)

    # Make output directory
    os.makedirs(args.out, exist_ok=True)

    # Load data
    print(f'Loading runs from {args.v4} and {args.v5} ...')
    df = load_data(args.v4, args.v5)
    print(f'  Total rows: {len(df)} | regimes: {sorted(df.regime.unique())} '
          f'| wirings: {sorted(df.wiring.unique())}')

    # NCP subset (primary scientific focus)
    ncp = ncp_df(df)
    print(f'  NCP rows: {len(ncp)}')

    # Summary: count per cell x regime
    counts = ncp.groupby(['cell_base', 'regime']).size().unstack(fill_value=0)
    print('\nRun counts per cell x regime (NCP):')
    print(counts.to_string())
    print()

    # Produce figures
    print('Generating figures:')
    produced = []
    produced.append(fig_nrmse_regimes_ncp(ncp, args.out))
    produced.append(fig_degradation_ncp(ncp, args.out))
    produced.append(fig_divergence_heatmap_ncp(ncp, args.out))
    produced.append(fig_bio_vs_classical_ncp(ncp, args.out))

    print(f'\nDone. {len(produced)} figures in {args.out}/')
    return 0


if __name__ == '__main__':
    sys.exit(main())
