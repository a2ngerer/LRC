# Design: Neural ODE Benchmark — All Cell × Wiring Combinations (Phase 3 / Step 10)

**Date:** 2026-03-13
**Branch:** `phase3/step10-neural-ode-benchmark`
**Status:** Approved

## Goal

Run all 7 valid cell × wiring combinations on all 6 Neural ODE systems (42 training runs total,
2000 iterations each). Save raw results as JSON and a Markdown summary table. Do NOT execute
the training during development — the script is written and tested structurally; the user
runs it manually before merging.

## Combination Matrix

7 valid combinations (lrc_ar + NCP excluded — documented incompatibility from step 9):

| Neuron | Dense | NCP |
|--------|-------|-----|
| lrc    | ✅    | ✅  |
| lrc_ar | ✅    | ❌ (excluded) |
| ctrnn  | ✅    | ✅  |
| lstm   | ✅    | ✅  |

6 ODE systems: `spiral`, `duffing`, `periodic_sinusoidal`, `periodic_predator_prey`,
`limited_predator_prey`, `nonlinear_predator_prey`.

Total: 7 × 6 = **42 training runs**.

## File Layout

```
experiments/
    benchmark_neural_ode.py     # new: main benchmark script
tests/experiments/
    test_benchmark_neural_ode.py  # new: 5 structural tests (no training)
results/                        # gitignored — created at runtime
    neural_ode_benchmark_<timestamp>.json
    neural_ode_benchmark_<timestamp>.md
```

## Component Specifications

### `experiments/benchmark_neural_ode.py`

**Imports:**

```python
import sys
import time
from datetime import datetime
import tensorflow as tf
from src.tasks.neural_ode.datasets import generate_dataset
from src.tasks.neural_ode.trainer import train
from src.models import make_dense_model, make_ncp_model
```

**Constants:**

```python
COMBINATIONS = [
    ('lrc',    'dense'),
    ('lrc',    'ncp'),
    ('lrc_ar', 'dense'),
    ('ctrnn',  'dense'),
    ('ctrnn',  'ncp'),
    ('lstm',   'dense'),
    ('lstm',   'ncp'),
]
# ('lrc_ar', 'ncp') excluded — lrc_ar passes raw inputs as v_pre into _sigmoid
# where mu/sigma have shape (units,units). At the NCP inter layer input_dim=2
# but inter_neurons=16, so 2 != 16 raises a shape error.

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
}

# NCP config — inter=16, command=8 chosen to be comparable to Dense units=16.
# motor_neurons=2 matches the 2-dimensional ODE output.
NCP_CONFIG = dict(inter_neurons=16, command_neurons=8, motor_neurons=2)

TRAIN_CONFIG = dict(n_iters=2000, batch_size=16, batch_time=16, lr=1e-3)
```

**Model builder:**

```python
def build_model(neuron: str, wiring: str) -> tf.keras.Sequential:
    """Return the appropriate model for (neuron, wiring)."""
    if wiring == 'dense':
        return make_dense_model(neuron, units=DENSE_UNITS[neuron], output_neurons=2)
    else:
        return make_ncp_model(neuron, **NCP_CONFIG)
```

**Single-run helper:**

```python
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
```

**`main()` function:**

```python
def main() -> int:
    """Run all 42 combinations. Save JSON + Markdown. Return exit code."""
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
```

**Output helpers** (private functions):

- `_save_json(results, timestamp)` — saves `results/neural_ode_benchmark_<timestamp>.json`.
  Creates `results/` directory if it does not exist (`os.makedirs(exist_ok=True)`).
  Uses `json.dump(..., indent=2, ensure_ascii=False)` with UTF-8 encoding.
  File I/O exceptions propagate to caller (no silent swallowing).
  ```json
  {
    "timestamp": "2026-03-13T14:32:10",
    "config": { "train": TRAIN_CONFIG, "dense_units": DENSE_UNITS, "ncp": NCP_CONFIG },
    "runs": [ <list of run dicts> ]
  }
  ```

- `_save_markdown(results, timestamp)` — saves `results/neural_ode_benchmark_<timestamp>.md`.
  Filename uses condensed format (`%Y%m%d_%H%M%S`). Header shows human-readable UTC timestamp.
  Duration is shown in the console progress output only, NOT in the Markdown table.
  Decimal precision: losses to 4 decimal places, decrease to 1 decimal place.
  Creates `results/` directory if it does not exist (`os.makedirs(exist_ok=True)`).
  ```markdown
  # Neural ODE Benchmark

  **Date:** 2026-03-13 14:32:10 UTC
  **Config:** n_iters=2000, batch_size=16, batch_time=16, lr=1e-3

  | Neuron | Wiring | System | Initial Loss | Final Loss | Decrease |
  |--------|--------|--------|-------------|-----------|---------|
  | lrc    | dense  | spiral | 0.1234      | 0.0312    | 74.7%   |
  ...
  ```

- `_print_summary(results)` — prints the same Markdown table to stdout after all runs.

**Entry point:**
```python
if __name__ == '__main__':
    sys.exit(main())
```

---

### `tests/experiments/test_benchmark_neural_ode.py`

Thin structural tests — no training, no model building. Runs in the normal pytest suite.

```python
from experiments.benchmark_neural_ode import (
    COMBINATIONS, SYSTEMS, DENSE_UNITS, NCP_CONFIG, TRAIN_CONFIG,
)


def test_combinations_count():
    """Exactly 7 valid combinations (lrc_ar+ncp excluded)."""
    assert len(COMBINATIONS) == 7


def test_lrc_ar_ncp_excluded():
    """lrc_ar + NCP must not appear (architectural incompatibility)."""
    assert ('lrc_ar', 'ncp') not in COMBINATIONS


def test_systems_count():
    assert len(SYSTEMS) == 6


def test_dense_units_covers_all_neurons():
    neurons_in_combinations = {n for n, _ in COMBINATIONS}
    for n in neurons_in_combinations:
        assert n in DENSE_UNITS


def test_ncp_motor_neurons_equals_ode_features():
    """motor_neurons must equal ODE output dimension (2)."""
    assert NCP_CONFIG['motor_neurons'] == 2


def test_systems_names():
    """System names match the datasets available in generate_dataset."""
    expected = {
        'spiral', 'duffing', 'periodic_sinusoidal',
        'periodic_predator_prey', 'limited_predator_prey',
        'nonlinear_predator_prey',
    }
    assert set(SYSTEMS) == expected
```

---

## Success Criterion

```bash
uv run pytest tests/experiments/test_benchmark_neural_ode.py -v
```

5/5 structural tests pass. All pre-existing tests continue to pass.

The benchmark script itself (`uv run python experiments/benchmark_neural_ode.py`) is
**not executed during development** — the user runs it manually to verify results before
merging to main.
