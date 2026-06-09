# experiments/benchmark_neural_ode.py
import json
import os
import sys
import time
from datetime import datetime

import tensorflow as tf

from src.models import make_dense_model, make_ncp_model
from src.tasks.neural_ode.datasets import generate_dataset
from src.tasks.neural_ode.ode_model import SequentialODEFunc
from src.tasks.neural_ode.trainer import train

# Thesis matrix (decision 2026-06-09): ltc, lrc, gru, lstm x dense, ncp.
# STC is not finalized yet; ctrnn/lrc_ar remain available via make_model but
# are not part of the benchmark matrix.
# Note: experiments/run_benchmark.py is the cluster runner (multi-seed,
# per-run JSON); this script remains as a quick sequential local runner.
COMBINATIONS = [
    ('ltc',  'dense'),
    ('ltc',  'ncp'),
    ('lrc',  'dense'),
    ('lrc',  'ncp'),
    ('gru',  'dense'),
    ('gru',  'ncp'),
    ('lstm', 'dense'),
    ('lstm', 'ncp'),
]

SYSTEMS = [
    'spiral',
    'duffing',
    'periodic_sinusoidal',
    'periodic_predator_prey',
    'limited_predator_prey',
    'nonlinear_predator_prey',
]

# Dense config per neuron.
# lrc_ar: units must equal input features (2) due to ODE state constraint.
DENSE_UNITS = {
    'lrc':    16,
    'lrc_ar': 2,
    'ctrnn':  16,
    'lstm':   16,
    'ltc':    16,
    'gru':    16,
}

# NCP config — inter=16, command=8 chosen to be comparable to Dense units=16.
# motor_neurons=2 matches the 2-dimensional ODE output.
NCP_CONFIG = dict(inter_neurons=16, command_neurons=8, motor_neurons=2)

TRAIN_CONFIG = dict(n_iters=2000, batch_size=16, batch_time=16, lr=1e-3)


def build_model(neuron: str, wiring: str) -> SequentialODEFunc:
    """Return the appropriate model for (neuron, wiring).

    Wrapped in SequentialODEFunc: euler_odeint calls func(t, y), which would
    misroute y into Sequential's training flag otherwise.
    """
    if wiring == 'dense':
        net = make_dense_model(neuron, units=DENSE_UNITS[neuron], output_neurons=2)
    else:
        net = make_ncp_model(neuron, **NCP_CONFIG)
    return SequentialODEFunc(net)


def run_one(neuron: str, wiring: str, system: str) -> dict:
    """Train one (neuron, wiring) combination on one ODE system.

    Returns:
        dict with keys: neuron, wiring, system, initial_loss, final_loss,
                        decrease_pct, duration_s, loss_history.
    """
    t, y = generate_dataset(system)
    model = build_model(neuron, wiring)
    t0 = time.time()
    losses = train(model, t, y, **TRAIN_CONFIG)
    duration = time.time() - t0
    initial = losses[0]
    final = losses[-1]
    pct = ((initial - final) / initial * 100) if initial != 0.0 else 0.0
    return {
        'neuron': neuron,
        'wiring': wiring,
        'system': system,
        'initial_loss': float(initial),
        'final_loss': float(final),
        'decrease_pct': float(pct),
        'duration_s': float(duration),
        'loss_history': [float(x) for x in losses],
    }


def _save_json(results: list, timestamp: str) -> None:
    """Save raw results to results/neural_ode_benchmark_<timestamp>.json."""
    os.makedirs('results', exist_ok=True)
    path = f'results/neural_ode_benchmark_{timestamp}.json'
    payload = {
        'timestamp': datetime.utcnow().isoformat(),
        'config': {
            'train': TRAIN_CONFIG,
            'dense_units': DENSE_UNITS,
            'ncp': NCP_CONFIG,
        },
        'runs': results,
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f'\nResults saved to {path}')


def _save_markdown(results: list, timestamp: str) -> None:
    """Save Markdown summary table to results/neural_ode_benchmark_<timestamp>.md."""
    os.makedirs('results', exist_ok=True)
    path = f'results/neural_ode_benchmark_{timestamp}.md'
    date_str = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    lines = [
        '# Neural ODE Benchmark\n',
        f'**Date:** {date_str}  ',
        f'**Config:** n_iters={TRAIN_CONFIG["n_iters"]}, '
        f'batch_size={TRAIN_CONFIG["batch_size"]}, '
        f'batch_time={TRAIN_CONFIG["batch_time"]}, '
        f'lr={TRAIN_CONFIG["lr"]}\n',
        '| Neuron | Wiring | System | Initial Loss | Final Loss | Decrease |',
        '|--------|--------|--------|-------------|-----------|---------|',
    ]
    for r in results:
        lines.append(
            f'| {r["neuron"]:<6} | {r["wiring"]:<6} | {r["system"]:<26} '
            f'| {r["initial_loss"]:.4f}       | {r["final_loss"]:.4f}     '
            f'| {r["decrease_pct"]:.1f}%    |'
        )
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'Markdown saved to {path}')


def _print_summary(results: list) -> None:
    """Print the same summary table to stdout."""
    print('\n=== Neural ODE Benchmark — Summary ===\n')
    print(f'  {"Neuron":<8} {"Wiring":<7} {"System":<26} '
          f'{"Init Loss":>10} {"Final Loss":>10} {"Decrease":>9}')
    print(f'  {"-"*8} {"-"*7} {"-"*26} {"-"*10} {"-"*10} {"-"*9}')
    for r in results:
        print(
            f'  {r["neuron"]:<8} {r["wiring"]:<7} {r["system"]:<26} '
            f'{r["initial_loss"]:>10.4f} {r["final_loss"]:>10.4f} '
            f'{r["decrease_pct"]:>8.1f}%'
        )


def main() -> int:
    """Run all 42 combinations. Save JSON + Markdown. Return exit code 0."""
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    total = len(COMBINATIONS) * len(SYSTEMS)
    results = []
    idx = 0

    for neuron, wiring in COMBINATIONS:
        for system in SYSTEMS:
            idx += 1
            print(f'\n[{idx}/{total}] {neuron} × {wiring} × {system} ...')
            result = run_one(neuron, wiring, system)
            results.append(result)
            print(
                f'  Done in {result["duration_s"]:.1f}s — '
                f'{result["initial_loss"]:.4f} → {result["final_loss"]:.4f} '
                f'({result["decrease_pct"]:.1f}% ↓)'
            )

    _save_json(results, timestamp)
    _save_markdown(results, timestamp)
    _print_summary(results)
    return 0


if __name__ == '__main__':
    sys.exit(main())
