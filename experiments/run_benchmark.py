# experiments/run_benchmark.py
"""Cluster-ready benchmark runner for the thesis matrix.

Thesis matrix: {ltc, lrc, gru, lstm} x {dense, ncp} x 6 ODE systems x N seeds.

One invocation = one training run = one JSON result file. This maps directly
onto a SLURM array job (see cluster/benchmark_array.sbatch):

    # list all run specs with their indices
    uv run python experiments/run_benchmark.py --list

    # number of specs (for --array=0-$((N-1)))
    uv run python experiments/run_benchmark.py --count

    # run spec i (SLURM array task)
    uv run python experiments/run_benchmark.py --index $SLURM_ARRAY_TASK_ID

    # explicit single run
    uv run python experiments/run_benchmark.py --cell ltc --wiring ncp \
        --system spiral --seed 0

    # full matrix sequentially (local, slow)
    uv run python experiments/run_benchmark.py --all
"""
import argparse
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from itertools import product

import numpy as np
import tensorflow as tf

from src.models import make_dense_model, make_ncp_model
from src.tasks.neural_ode.datasets import generate_dataset
from src.tasks.neural_ode.ode_model import SequentialODEFunc
from src.tasks.neural_ode.solver import euler_odeint
from src.tasks.neural_ode.trainer import train
from src.evaluation import GradientFlowTracker, mse, nrmse
from src.utils import set_global_seed

CELLS = ['ltc', 'lrc', 'gru', 'lstm']
WIRINGS = ['dense', 'ncp']
SYSTEMS = [
    'spiral',
    'duffing',
    'periodic_sinusoidal',
    'periodic_predator_prey',
    'limited_predator_prey',
    'nonlinear_predator_prey',
]
SEEDS = [0, 1, 2, 3, 4]

# Benchmark v2: vanishing-gradient-fixed cells (see
# docs/superpowers/specs/2026-06-10-benchmark-v2-fixed-cells-design.md).
CELLS_V2 = ['mm_ltc', 'mm_lrc', 'cfc']
# Clip threshold for the v2 clip-axis runs. 1.0 is the common recurrent-RL
# default; the exact value is an experiment parameter, not a tuned constant.
V2_CLIP_NORM = 1.0

DENSE_UNITS = 16
# inter=16, command=8 chosen to be comparable to Dense units=16.
# motor_neurons=2 matches the 2-dimensional ODE output.
NCP_CONFIG = dict(inter_neurons=16, command_neurons=8, motor_neurons=2)
# Fixed wiring seed: all training seeds share the same NCP graph, so seed
# variance measures init/batch randomness, not wiring randomness.
NCP_WIRING_SEED = 42

DEFAULTS = dict(n_iters=2000, batch_size=16, batch_time=16, lr=1e-3,
                loss='mse', grad_log_every=25, data_size=1000)

# Per-cell constructor overrides forwarded to the cell (and every NCP layer).
# LRC defaults to elastance_type="interp", which falls through to a constant
# elastance (no liquid capacitance) -- effectively a saturated LTC. "asymmetric"
# activates the input/state-dependent liquid elastance that defines the LRC model
# and that RQ2/RQ5 isolate. Other cells take no extra kwargs.
CELL_KWARGS = {
    'lrc': dict(elastance_type='asymmetric'),
    'mm_lrc': dict(elastance_type='asymmetric'),
}


def build_specs(cells, wirings, systems, seeds, clip_norm=0.0):
    """Deterministic run-spec list; index order is the SLURM array contract."""
    return [
        {'cell': c, 'wiring': w, 'system': sy, 'seed': se, 'clip_norm': clip_norm}
        for c, w, sy, se in product(cells, wirings, systems, seeds)
    ]


def build_specs_v2(systems=SYSTEMS, seeds=SEEDS):
    """v2 matrix: fixed cells x {no clip, clip} + problem cells x clip.

    Order (= SLURM array contract for v2):
      [0,180):   CELLS_V2, clip off
      [180,360): CELLS_V2, clip V2_CLIP_NORM
      [360,480): ltc/lrc,  clip V2_CLIP_NORM (isolates the optimizer fix)
    """
    return (
        build_specs(CELLS_V2, WIRINGS, systems, seeds, clip_norm=0.0)
        + build_specs(CELLS_V2, WIRINGS, systems, seeds, clip_norm=V2_CLIP_NORM)
        + build_specs(['ltc', 'lrc'], WIRINGS, systems, seeds, clip_norm=V2_CLIP_NORM)
    )


def build_model(cell: str, wiring: str) -> SequentialODEFunc:
    cell_kwargs = CELL_KWARGS.get(cell, {})
    if wiring == 'dense':
        net = make_dense_model(cell, units=DENSE_UNITS, output_neurons=2, **cell_kwargs)
    else:
        net = make_ncp_model(cell, seed=NCP_WIRING_SEED, **NCP_CONFIG, **cell_kwargs)
    return SequentialODEFunc(net)


def evaluate_full_trajectory(model, t, y):
    """Roll out the trained model from y[0] over the full time grid."""
    y0 = tf.constant(y[0][np.newaxis, np.newaxis, :], dtype=tf.float32)
    t_tf = tf.constant(t, dtype=tf.float32)
    pred = euler_odeint(model, y0, t_tf)            # (T, 1, 1, 2)
    pred_traj = pred.numpy().reshape(len(t), -1)    # (T, 2)
    return {
        'mse': mse(y, pred_traj),
        'nrmse': nrmse(y, pred_traj),
        'trajectory_pred': pred_traj.astype(float).tolist(),
    }


def run_one(spec: dict, cfg: dict) -> dict:
    rng = set_global_seed(spec['seed'], deterministic_ops=cfg['deterministic'])
    t, y = generate_dataset(spec['system'], data_size=cfg['data_size'])
    model = build_model(spec['cell'], spec['wiring'])
    tracker = GradientFlowTracker(log_every=cfg['grad_log_every'])

    clip_norm = float(spec.get('clip_norm', 0.0))
    t0 = time.time()
    losses = train(
        model, t, y,
        n_iters=cfg['n_iters'], batch_size=cfg['batch_size'],
        batch_time=cfg['batch_time'], lr=cfg['lr'], loss=cfg['loss'],
        rng=rng, gradient_tracker=tracker,
        clip_norm=clip_norm or None,
    )
    duration = time.time() - t0

    evaluation = evaluate_full_trajectory(model, t, y)

    return {
        'schema_version': 2,
        'run': spec,
        'config': {
            **{k: cfg[k] for k in ('n_iters', 'batch_size', 'batch_time',
                                   'lr', 'loss', 'data_size', 'deterministic')},
            'dense_units': DENSE_UNITS,
            'ncp': NCP_CONFIG,
            'ncp_wiring_seed': NCP_WIRING_SEED,
            'cell_kwargs': CELL_KWARGS.get(spec['cell'], {}),
            'clip_norm': clip_norm,
        },
        'env': {
            'tensorflow': tf.__version__,
            'gpus': [d.name for d in tf.config.list_physical_devices('GPU')],
            'hostname': socket.gethostname(),
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        },
        'training': {
            'loss_history': [float(x) for x in losses],
            'initial_loss': float(losses[0]),
            'final_loss': float(losses[-1]),
            'duration_s': float(duration),
        },
        'gradient_flow': tracker.as_dict(),
        'evaluation': evaluation,
    }


def result_filename(spec: dict) -> str:
    base = f"{spec['cell']}_{spec['wiring']}_{spec['system']}_seed{spec['seed']}"
    if spec.get('clip_norm'):
        base += f"_clip{spec['clip_norm']}"
    return base + '.json'


def save_result(result: dict, outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, result_filename(result['run']))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)
    return path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Thesis benchmark runner (cell x wiring x system x seed)')
    sel = p.add_argument_group('run selection')
    sel.add_argument('--list', action='store_true', help='print all run specs and exit')
    sel.add_argument('--count', action='store_true', help='print number of run specs and exit')
    sel.add_argument('--index', type=int, default=None, help='run spec by index (SLURM array)')
    sel.add_argument('--all', action='store_true', help='run all specs sequentially')
    sel.add_argument('--profile', choices=['v1', 'v2'], default='v1',
                     help='v1: thesis matrix (240 runs, results/runs). '
                          'v2: fixed cells + clip axis (480 runs, results/runs_v2)')
    sel.add_argument('--clip-norm', type=float, default=0.0,
                     help='gradient clip threshold for explicit single runs '
                          '(0 = off); profile runs take it from the spec')
    sel.add_argument('--cell', choices=CELLS + CELLS_V2, default=None)
    sel.add_argument('--wiring', choices=WIRINGS, default=None)
    sel.add_argument('--system', choices=SYSTEMS, default=None)
    sel.add_argument('--seed', type=int, default=None)

    mat = p.add_argument_group('matrix filters (apply before indexing)')
    mat.add_argument('--cells', default=None, help='comma-separated cell subset')
    mat.add_argument('--systems', default=None, help='comma-separated system subset')
    mat.add_argument('--wirings', default=None, help='comma-separated wiring subset')
    mat.add_argument('--seeds', default=None, help='comma-separated seeds')

    tr = p.add_argument_group('training config')
    tr.add_argument('--iters', type=int, default=DEFAULTS['n_iters'])
    tr.add_argument('--batch-size', type=int, default=DEFAULTS['batch_size'])
    tr.add_argument('--batch-time', type=int, default=DEFAULTS['batch_time'])
    tr.add_argument('--lr', type=float, default=DEFAULTS['lr'])
    tr.add_argument('--loss', choices=['mse', 'mae'], default=DEFAULTS['loss'])
    tr.add_argument('--grad-log-every', type=int, default=DEFAULTS['grad_log_every'],
                    help='gradient-norm logging interval; 0 disables (RQ4)')
    tr.add_argument('--data-size', type=int, default=DEFAULTS['data_size'])
    tr.add_argument('--deterministic', action='store_true',
                    help='enable TF op determinism (bit-exact, slower)')
    tr.add_argument('--outdir', default=None,
                    help='default: results/runs (v1) / results/runs_v2 (v2)')
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.profile == 'v2':
        specs = build_specs_v2()
    else:
        specs = build_specs(CELLS, WIRINGS, SYSTEMS, SEEDS)
    if args.cells:
        keep = set(args.cells.split(','))
        specs = [s for s in specs if s['cell'] in keep]
    if args.wirings:
        keep = set(args.wirings.split(','))
        specs = [s for s in specs if s['wiring'] in keep]
    if args.systems:
        keep = set(args.systems.split(','))
        specs = [s for s in specs if s['system'] in keep]
    if args.seeds:
        keep = {int(s) for s in args.seeds.split(',') if s != ''}
        specs = [s for s in specs if s['seed'] in keep]
    if args.outdir is None:
        args.outdir = 'results/runs_v2' if args.profile == 'v2' else 'results/runs'

    if args.count:
        print(len(specs))
        return 0
    if args.list:
        for i, s in enumerate(specs):
            print(f"{i:4d}  {s['cell']:<7} {s['wiring']:<6} {s['system']:<26} "
                  f"seed={s['seed']} clip={s['clip_norm']}")
        return 0

    cfg = dict(
        n_iters=args.iters, batch_size=args.batch_size, batch_time=args.batch_time,
        lr=args.lr, loss=args.loss, grad_log_every=args.grad_log_every,
        data_size=args.data_size, deterministic=args.deterministic,
    )

    if args.index is not None:
        if not (0 <= args.index < len(specs)):
            print(f'Index {args.index} out of range [0, {len(specs)})', file=sys.stderr)
            return 1
        todo = [specs[args.index]]
    elif args.all:
        todo = specs
    elif all(v is not None for v in (args.cell, args.wiring, args.system, args.seed)):
        todo = [{'cell': args.cell, 'wiring': args.wiring,
                 'system': args.system, 'seed': args.seed,
                 'clip_norm': args.clip_norm}]
    else:
        print('Select runs via --index, --all, or --cell/--wiring/--system/--seed '
              '(or use --list/--count).', file=sys.stderr)
        return 1

    for i, spec in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {spec['cell']} x {spec['wiring']} x "
              f"{spec['system']} (seed {spec['seed']}) ...")
        result = run_one(spec, cfg)
        path = save_result(result, args.outdir)
        ev = result['evaluation']
        print(f"  done in {result['training']['duration_s']:.1f}s — "
              f"final_loss={result['training']['final_loss']:.6f} "
              f"traj MSE={ev['mse']:.6f} NRMSE={ev['nrmse']:.4f} -> {path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
