# Design: Verify LRC + Dense on All 6 ODE Systems (Phase 1 / Step 6)

**Date:** 2026-03-10
**Branch:** `phase1/step6-verify-lrc-results`
**Status:** Approved

## Goal

Run the trained LRC_AR + DenseWiring model on all 6 ODE systems for 2000 iterations (matching the original `neuralODE/run_ode.py`) and confirm that loss decreases meaningfully on each. Save a structured JSON baseline to `results/` for Phase 3 reference.

## Verification Criterion

Qualitative: `final_loss < 0.5 * initial_loss` for all 6 systems. No comparison to exact numerical values from the original code.

## File Layout

```
experiments/
    verify_neural_ode.py          # verification runner
results/
    neural_ode_lrc_baseline.json  # generated output (gitignored)
tests/experiments/
    __init__.py
    test_verify_script.py         # smoke test
```

## Component Specifications

### experiments/verify_neural_ode.py

Single public function + CLI entry point:

```python
import argparse
import json
import os
from datetime import datetime

import numpy as np

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
        dict with keys:
            'timestamp': ISO 8601 string
            'config':    hyperparameter dict used
            'systems':   {name: {'initial_loss': float,
                                  'final_loss': float,
                                  'loss_history': list[float]}}
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
    for name, data in results['systems'].items():
        if data['final_loss'] >= 0.5 * data['initial_loss']:
            print(f'FAIL: {name} did not converge '
                  f'(initial={data["initial_loss"]:.4f}, '
                  f'final={data["final_loss"]:.4f})')
            return False
    return True


def save_results(results, path='results/neural_ode_lrc_baseline.json'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
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
        exit(0)
    else:
        print('\nSome systems failed to converge.')
        exit(1)


if __name__ == '__main__':
    main()
```

**Usage:**
```bash
uv run python experiments/verify_neural_ode.py
uv run python experiments/verify_neural_ode.py --output results/custom.json
```

### results/neural_ode_lrc_baseline.json (generated)

```json
{
  "timestamp": "2026-03-10T...",
  "config": {
    "neuron": "lrc_ar",
    "wiring": "dense",
    "units": 16,
    "niters": 2000,
    "batch_size": 16,
    "batch_time": 16,
    "lr": 0.001
  },
  "systems": {
    "spiral": {
      "initial_loss": 0.342,
      "final_loss": 0.018,
      "loss_history": [0.342, ...]
    },
    ...
  }
}
```

### tests/experiments/test_verify_script.py

Two tests:

1. **Output shape test**: calls `run_verification(systems=['spiral'], n_iters=5)`, asserts:
   - result has keys `timestamp`, `config`, `systems`
   - `result['systems']['spiral']` has keys `initial_loss`, `final_loss`, `loss_history`
   - `len(result['systems']['spiral']['loss_history']) == 5`
   - all loss values are finite positive floats

2. **Convergence check — passes**: construct synthetic results dict with `final_loss = 0.4 * initial_loss` for two systems, assert `check_convergence(results) is True`

3. **Convergence check — fails**: construct synthetic results dict with `final_loss = 0.6 * initial_loss` for one system, assert `check_convergence(results) is False`

## Success Criterion

```bash
uv run python experiments/verify_neural_ode.py
```

Exits 0, prints per-system summary with visible loss decrease, saves `results/neural_ode_lrc_baseline.json`.

## Out of Scope

- Visualization (loss curves, phase portraits) → Phase 3 evaluation module
- Running all 4 neuron types or both wirings → Phase 3 benchmarking
- Statistical significance testing
- GPU device placement
- Random seed control
- Checkpointing
