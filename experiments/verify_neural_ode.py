import argparse
import json
import os
import sys
from datetime import datetime

from src.tasks.neural_ode.datasets import generate_dataset
from src.tasks.neural_ode.ode_model import ODEFuncModel
from src.tasks.neural_ode.trainer import train

ALL_SYSTEMS = [
    'spiral',
    'duffing',
    'periodic_sinusoidal',
    'periodic_predator_prey',
    'limited_predator_prey',
    'nonlinear_predator_prey',
]

DEFAULT_CONFIG = {
    'neuron': 'lrc_ar',
    'wiring': 'dense',
    'units': 16,
    'niters': 2000,
    'batch_size': 16,
    'batch_time': 16,
    'lr': 1e-3,
}


def run_verification(systems=ALL_SYSTEMS, n_iters=None, config=None):
    """Run training on each ODE system and return results dict.

    Args:
        systems:  list of system name strings to run
        n_iters:  overrides config['niters'] if provided
        config:   dict of hyperparameters; defaults to DEFAULT_CONFIG

    Returns:
        dict with keys 'timestamp', 'config', 'systems'.
        Each entry in 'systems' has 'initial_loss', 'final_loss', 'loss_history'.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    if n_iters is not None:
        cfg['niters'] = n_iters

    results = {
        'timestamp': datetime.utcnow().isoformat(),
        'config': cfg,
        'systems': {},
    }

    for name in systems:
        print(f'\n=== {name} ===')
        t, y = generate_dataset(name)
        model = ODEFuncModel(cfg['neuron'], cfg['wiring'], cfg['units'], features=2)
        losses = train(model, t, y,
                       n_iters=cfg['niters'],
                       batch_size=cfg['batch_size'],
                       batch_time=cfg['batch_time'],
                       lr=cfg['lr'])
        initial = losses[0]
        final = losses[-1]
        pct = (initial - final) / initial * 100
        print(f'{name}: {initial:.4f} → {final:.4f} ({pct:.1f}% decrease)')
        results['systems'][name] = {
            'initial_loss': initial,
            'final_loss': final,
            'loss_history': losses,
        }

    return results


def check_convergence(results):
    """Return True if all systems have final_loss < 0.5 * initial_loss."""
    all_converged = True
    for name, data in results['systems'].items():
        if data['final_loss'] >= 0.5 * data['initial_loss']:
            print(f'FAIL: {name} did not converge '
                  f'(initial={data["initial_loss"]:.4f}, '
                  f'final={data["final_loss"]:.4f})')
            all_converged = False
    return all_converged


def save_results(results, path='results/neural_ode_lrc_baseline.json'):
    dir_part = os.path.dirname(path)
    if dir_part:
        os.makedirs(dir_part, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {path}')


def main():
    parser = argparse.ArgumentParser(
        description='Verify LRC_AR + Dense on all 6 Neural ODE systems')
    parser.add_argument('--output', default='results/neural_ode_lrc_baseline.json',
                        help='Output JSON path')
    args = parser.parse_args()

    results = run_verification()
    save_results(results, args.output)

    if check_convergence(results):
        print('\nAll systems converged. ✓')
        sys.exit(0)
    else:
        print('\nSome systems failed to converge.')
        sys.exit(1)


if __name__ == '__main__':
    main()
