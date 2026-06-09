# experiments/plot_results.py
"""Generate figures from benchmark run JSONs.

Per ODE system:
  - loss_{system}.png      — training curves (mean over seeds, one line per cell x wiring)
  - phase_{system}.png     — phase portraits: predicted vs. true trajectory (seed 0)
Per cell (default: all cells with gradient logs):
  - gradflow_{cell}_{system}.png — per-layer gradient norms, dense vs ncp (RQ4)

Usage:
    uv run python experiments/plot_results.py [--runs results/runs] [--out results/figures]
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from glob import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.tasks.neural_ode.datasets import generate_dataset

STYLES = {'dense': '-', 'ncp': '--'}


def load_runs(runs_dir):
    runs = []
    for path in sorted(glob(os.path.join(runs_dir, '*.json'))):
        with open(path, encoding='utf-8') as f:
            runs.append(json.load(f))
    return runs


def _group(runs):
    by_key = defaultdict(list)
    for r in runs:
        k = (r['run']['system'], r['run']['cell'], r['run']['wiring'])
        by_key[k].append(r)
    return by_key


def plot_loss_curves(runs, outdir):
    by_key = _group(runs)
    systems = sorted({k[0] for k in by_key})
    cells = sorted({k[1] for k in by_key})
    cmap = plt.get_cmap('tab10')
    colors = {c: cmap(i) for i, c in enumerate(cells)}

    for system in systems:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for cell in cells:
            for wiring in ['dense', 'ncp']:
                group = by_key.get((system, cell, wiring))
                if not group:
                    continue
                histories = [r['training']['loss_history'] for r in group]
                n = min(len(h) for h in histories)
                mean = np.mean([h[:n] for h in histories], axis=0)
                ax.plot(mean, STYLES[wiring], color=colors[cell],
                        label=f'{cell} / {wiring}', linewidth=1.2)
        ax.set_yscale('log')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Training loss (MSE)')
        ax.set_title(f'Training curves — {system}')
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        path = os.path.join(outdir, f'loss_{system}.png')
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f'  {path}')


def plot_phase_portraits(runs, outdir, seed=0):
    by_key = _group(runs)
    systems = sorted({k[0] for k in by_key})
    cells = sorted({k[1] for k in by_key})

    for system in systems:
        data_size = next(iter(by_key[(system, cells[0], 'dense')]),
                         {'config': {'data_size': 1000}})['config']['data_size']
        _, y_true = generate_dataset(system, data_size=data_size)

        fig, axes = plt.subplots(2, len(cells), figsize=(3 * len(cells), 6),
                                 squeeze=False)
        for col, cell in enumerate(cells):
            for row, wiring in enumerate(['dense', 'ncp']):
                ax = axes[row][col]
                ax.plot(y_true[:, 0], y_true[:, 1], color='0.6', linewidth=1.0,
                        label='true')
                group = by_key.get((system, cell, wiring), [])
                run = next((r for r in group if r['run']['seed'] == seed), None)
                if run is not None:
                    pred = np.array(run['evaluation']['trajectory_pred'])
                    ax.plot(pred[:, 0], pred[:, 1], color='C3', linewidth=1.0,
                            label='predicted')
                    ax.set_title(f"{cell}/{wiring}  NRMSE={run['evaluation']['nrmse']:.3f}",
                                 fontsize=8)
                else:
                    ax.set_title(f'{cell}/{wiring} (no run)', fontsize=8)
                ax.tick_params(labelsize=6)
        axes[0][0].legend(fontsize=6)
        fig.suptitle(f'Phase portraits — {system} (seed {seed})')
        fig.tight_layout()
        path = os.path.join(outdir, f'phase_{system}.png')
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f'  {path}')


def plot_gradient_flow(runs, outdir):
    by_key = _group(runs)
    systems = sorted({k[0] for k in by_key})
    cells = sorted({k[1] for k in by_key})

    for system in systems:
        for cell in cells:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
            has_data = False
            for ax, wiring in zip(axes, ['dense', 'ncp']):
                group = by_key.get((system, cell, wiring), [])
                gf_runs = [r for r in group if r['gradient_flow']['iterations']]
                ax.set_title(f'{wiring}')
                if not gf_runs:
                    continue
                has_data = True
                layer_names = sorted(gf_runs[0]['gradient_flow']['layer_norms'])
                for name in layer_names:
                    series = [r['gradient_flow']['layer_norms'].get(name) for r in gf_runs]
                    series = [s for s in series if s]
                    if not series:
                        continue
                    n = min(len(s) for s in series)
                    mean = np.mean([s[:n] for s in series], axis=0)
                    iters = gf_runs[0]['gradient_flow']['iterations'][:n]
                    ax.plot(iters, mean, label=name, linewidth=1.0)
                ax.set_yscale('log')
                ax.set_xlabel('Iteration')
                ax.legend(fontsize=6)
            if not has_data:
                plt.close(fig)
                continue
            axes[0].set_ylabel('Layer gradient norm')
            fig.suptitle(f'Gradient flow — {cell} on {system} (mean over seeds)')
            fig.tight_layout()
            path = os.path.join(outdir, f'gradflow_{cell}_{system}.png')
            fig.savefig(path, dpi=200)
            plt.close(fig)
            print(f'  {path}')


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--runs', default='results/runs')
    p.add_argument('--out', default='results/figures')
    p.add_argument('--seed', type=int, default=0, help='seed used for phase portraits')
    p.add_argument('--skip-gradflow', action='store_true')
    args = p.parse_args(argv)

    runs = load_runs(args.runs)
    if not runs:
        print(f'No run JSONs found in {args.runs}', file=sys.stderr)
        return 1
    os.makedirs(args.out, exist_ok=True)

    print('Loss curves:')
    plot_loss_curves(runs, args.out)
    print('Phase portraits:')
    plot_phase_portraits(runs, args.out, seed=args.seed)
    if not args.skip_gradflow:
        print('Gradient flow (RQ4):')
        plot_gradient_flow(runs, args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
