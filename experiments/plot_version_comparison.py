# experiments/plot_version_comparison.py
"""Intermediate v1 vs v2 vs v3 comparison figures with shared scales.

Tracks benchmark progress across the three generations:
  v1 (results/runs)     original cells (ltc, lrc, gru, lstm), baseline.
  v2 (results/runs_v2)  vanishing-gradient fixes (mm_ltc, mm_lrc, cfc) + clip axis.
  v3 (results/runs_v3)  solver-fidelity (ode_unfolds=24) + horizon (batch_time=64)
                        interventions on the stiff cells, plus extra seeds.

Each version is compared on its BASELINE arm (clip_norm=0, ode_unfolds=6,
batch_time=16) so the version axis is not confounded by v2's clip or v3's
solver/horizon interventions; those interventions get their own panel.

Figures (all written to --out):
  nrmse_by_system.png   2x3 system panels, per-cell NRMSE box plots hued by
                        version, SHARED log-y across all panels.
  divergence_rate.png   fraction of runs with NRMSE>1 per cell, dense vs ncp,
                        grouped by version, SHARED y (0-100%).
  v3_intervention.png   stiff cells x stiff systems: pooled baseline vs
                        +unfolds24 (+bt64 when present), SHARED log-y.

Usage:
    uv run python experiments/plot_version_comparison.py --out ~/Downloads
"""
import argparse
import json
import os
import sys
from datetime import datetime
from glob import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

VERSIONS = [('v1', 'results/runs'), ('v2', 'results/runs_v2'), ('v3', 'results/runs_v3')]
VCOLORS = {'v1': '#4C72B0', 'v2': '#DD8452', 'v3': '#55A868'}
DIVERGENCE = 1.0  # NRMSE > 1 == forward rollout failure
STIFF_CELLS = ['ltc', 'mm_ltc']
STIFF_SYS = ['duffing', 'periodic_predator_prey']


def _arm(clip, unfolds, bt):
    if clip:
        return 'clip'
    if unfolds != 6:
        return f'unfolds{unfolds}'
    if bt != 16:
        return f'bt{bt}'
    return 'base'


def load_rows():
    """Flatten every run JSON into a tagged record."""
    rows = []
    for ver, d in VERSIONS:
        for path in sorted(glob(os.path.join(d, '*.json'))):
            try:
                j = json.load(open(path, encoding='utf-8'))
            except Exception:
                continue
            r = j.get('run', {})
            c = j.get('config', {})
            e = j.get('evaluation', {})
            t = j.get('training', {})
            nrmse = e.get('nrmse')
            if nrmse is None or not np.isfinite(nrmse):
                continue
            clip = float(r.get('clip_norm', c.get('clip_norm', 0.0)) or 0.0)
            unfolds = int(r.get('ode_unfolds', c.get('ode_unfolds', 6)) or 6)
            bt = int(c.get('batch_time', 16) or 16)
            rows.append(dict(
                version=ver, cell=r.get('cell'), wiring=r.get('wiring'),
                system=r.get('system'), seed=r.get('seed'),
                clip=clip, unfolds=unfolds, bt=bt, arm=_arm(clip, unfolds, bt),
                nrmse=float(nrmse), final_loss=t.get('final_loss'),
            ))
    return rows


def _shared_log_ylim(values, pad=1.3):
    vals = [v for v in values if v is not None and v > 0]
    lo, hi = min(vals), max(vals)
    return lo / pad, hi * pad


def fig_nrmse_by_system(rows, outdir, v3n):
    """2x3 grid; per system, per-cell NRMSE box plots grouped by version."""
    base = [r for r in rows if r['arm'] == 'base']
    systems = sorted({r['system'] for r in base})
    cells = ['ltc', 'lrc', 'mm_ltc', 'mm_lrc', 'cfc', 'gru', 'lstm']
    cells = [c for c in cells if any(r['cell'] == c for r in base)]
    ylim = _shared_log_ylim([r['nrmse'] for r in base])

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
    for ax, system in zip(axes.flat, systems):
        positions, data, colors, centers, labels = [], [], [], [], []
        x = 0.0
        present_vers = [v for v, _ in VERSIONS
                        if any(r['version'] == v and r['system'] == system for r in base)]
        width = 0.8 / max(len(present_vers), 1)
        for cell in cells:
            group_start = x
            drawn = 0
            for vi, ver in enumerate(present_vers):
                vals = [r['nrmse'] for r in base
                        if r['system'] == system and r['cell'] == cell and r['version'] == ver]
                if not vals:
                    continue
                positions.append(x + vi * width)
                data.append(vals)
                colors.append(VCOLORS[ver])
                drawn += 1
            if drawn:
                centers.append(group_start + (len(present_vers) - 1) * width / 2)
                labels.append(cell)
                x += 1.0
        if data:
            bp = ax.boxplot(data, positions=positions, widths=width * 0.9,
                            patch_artist=True, showfliers=True,
                            flierprops=dict(marker='.', markersize=3, alpha=0.5),
                            medianprops=dict(color='black', linewidth=1.2))
            for patch, col in zip(bp['boxes'], colors):
                patch.set_facecolor(col)
                patch.set_alpha(0.75)
        ax.axhline(DIVERGENCE, color='red', linestyle=':', linewidth=1.0, zorder=0)
        ax.set_yscale('log')
        ax.set_ylim(*ylim)
        ax.set_xticks(centers)
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
        ax.set_title(system, fontsize=10)
        ax.grid(axis='y', alpha=0.25)
    for ax in axes[:, 0]:
        ax.set_ylabel('NRMSE (log)')
    handles = [Patch(facecolor=VCOLORS[v], alpha=0.75, label=v) for v, _ in VERSIONS]
    handles.append(plt.Line2D([], [], color='red', linestyle=':', label='divergence (NRMSE=1)'))
    fig.suptitle(f'NRMSE per system — v1 vs v2 vs v3 (baseline arm, shared log-y) '
                 f'· v3 partial {v3n}/720', y=0.995, fontsize=12)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.955),
               ncol=4, fontsize=9, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(outdir, 'nrmse_by_system.png')
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_divergence_rate(rows, outdir, v3n):
    """Per-cell divergence rate (NRMSE>1), dense vs ncp, grouped by version."""
    base = [r for r in rows if r['arm'] == 'base']
    cells = ['ltc', 'lrc', 'mm_ltc', 'mm_lrc', 'cfc', 'gru', 'lstm']
    cells = [c for c in cells if any(r['cell'] == c for r in base)]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    for ax, wiring in zip(axes, ['dense', 'ncp']):
        present_vers = [v for v, _ in VERSIONS]
        width = 0.8 / len(present_vers)
        centers, labels = [], []
        for ci, cell in enumerate(cells):
            for vi, ver in enumerate(present_vers):
                vals = [r['nrmse'] for r in base
                        if r['cell'] == cell and r['wiring'] == wiring and r['version'] == ver]
                if not vals:
                    continue
                rate = 100.0 * sum(1 for v in vals if v > DIVERGENCE) / len(vals)
                pos = ci + vi * width
                ax.bar(pos, rate, width=width * 0.9, color=VCOLORS[ver], alpha=0.85)
                ax.text(pos, rate + 1.5, f'{int(round(rate))}', ha='center',
                        fontsize=6.5, color='0.3')
            centers.append(ci + (len(present_vers) - 1) * width / 2)
            labels.append(cell)
        ax.set_xticks(centers)
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
        ax.set_title(wiring, fontsize=11)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.25)
    axes[0].set_ylabel('Divergence rate (% runs with NRMSE>1)')
    handles = [Patch(facecolor=VCOLORS[v], alpha=0.85, label=v) for v, _ in VERSIONS]
    fig.suptitle(f'Forward-rollout divergence — v1 vs v2 vs v3 (baseline arm) '
                 f'· v3 partial {v3n}/720', y=0.99, fontsize=12)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.92),
               ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    path = os.path.join(outdir, 'divergence_rate.png')
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_v3_intervention(rows, outdir, v3n):
    """Stiff cells x stiff systems: pooled baseline vs +unfolds24 (+bt64)."""
    # Pool baseline across all versions (gives up to 15 seeds), interventions from v3.
    base = [r for r in rows if r['arm'] == 'base'
            and r['cell'] in STIFF_CELLS and r['system'] in STIFF_SYS]
    arms = ['base', 'unfolds24', 'bt64']
    arm_color = {'base': '#888888', 'unfolds24': '#55A868', 'bt64': '#C44E52'}
    arm_label = {'base': 'baseline', 'unfolds24': '+unfolds24', 'bt64': '+bt64'}
    pool = base + [r for r in rows if r['arm'] in ('unfolds24', 'bt64')
                   and r['cell'] in STIFF_CELLS and r['system'] in STIFF_SYS]
    ylim = _shared_log_ylim([r['nrmse'] for r in pool]) if pool else (1e-3, 1e1)

    combos = [(c, w) for c in STIFF_CELLS for w in ['dense', 'ncp']]
    fig, axes = plt.subplots(len(combos), len(STIFF_SYS),
                             figsize=(11, 12), sharey=True, squeeze=False)
    for ri, (cell, wiring) in enumerate(combos):
        for ci, system in enumerate(STIFF_SYS):
            ax = axes[ri][ci]
            data, positions, colors, ticks, labels = [], [], [], [], []
            for ai, arm in enumerate(arms):
                vals = [r['nrmse'] for r in pool
                        if r['cell'] == cell and r['wiring'] == wiring
                        and r['system'] == system and r['arm'] == arm]
                if not vals:
                    continue
                data.append(vals)
                positions.append(ai)
                colors.append(arm_color[arm])
                ticks.append(ai)
                labels.append(f'{arm_label[arm]}\nn={len(vals)}')
            if data:
                bp = ax.boxplot(data, positions=positions, widths=0.6,
                                patch_artist=True, showfliers=True,
                                flierprops=dict(marker='.', markersize=4, alpha=0.6),
                                medianprops=dict(color='black', linewidth=1.2))
                for patch, col in zip(bp['boxes'], colors):
                    patch.set_facecolor(col)
                    patch.set_alpha(0.8)
                for pos, vals in zip(positions, data):
                    ax.scatter([pos] * len(vals), vals, s=10, color='black',
                               alpha=0.35, zorder=3)
            ax.axhline(DIVERGENCE, color='red', linestyle=':', linewidth=1.0)
            ax.set_yscale('log')
            ax.set_ylim(*ylim)
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_xlim(-0.6, len(arms) - 0.4)
            if ci == 0:
                ax.set_ylabel(f'{cell} / {wiring}\nNRMSE (log)', fontsize=9)
            if ri == 0:
                ax.set_title(system, fontsize=10)
            ax.grid(axis='y', alpha=0.25)
    fig.suptitle(f'v3 intervention effect — stiff cells x stiff systems '
                 f'(baseline pooled all versions) · v3 partial {v3n}/720',
                 y=0.997, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(outdir, 'v3_intervention.png')
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--out', default=os.path.expanduser('~/Downloads'))
    args = p.parse_args(argv)
    outdir = os.path.expanduser(args.out)
    os.makedirs(outdir, exist_ok=True)

    rows = load_rows()
    if not rows:
        print('No run JSONs found.', file=sys.stderr)
        return 1
    v3n = sum(1 for r in rows if r['version'] == 'v3')
    stamp = datetime.now().strftime('%Y%m%d-%H%M')
    print(f'Loaded {len(rows)} runs (v3 partial {v3n}/720). Writing to {outdir}')

    paths = [
        fig_nrmse_by_system(rows, outdir, v3n),
        fig_divergence_rate(rows, outdir, v3n),
        fig_v3_intervention(rows, outdir, v3n),
    ]
    # Stamp a copy so successive intermediate snapshots do not overwrite.
    import shutil
    for path in paths:
        base, ext = os.path.splitext(path)
        try:
            shutil.copy(path, f'{base}_{stamp}{ext}')
        except Exception:
            pass
        print(f'  {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
