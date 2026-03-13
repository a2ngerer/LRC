# Neural ODE Benchmark Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `experiments/benchmark_neural_ode.py` that runs all 7 valid cell × wiring combinations on all 6 ODE systems (42 runs, 2000 iters each) and saves JSON + Markdown results — but does NOT execute the training during development.

**Architecture:** Single new script using existing `train()`, `generate_dataset()`, `make_dense_model`, and `make_ncp_model`. Thin structural tests validate constants only (no training). The benchmark script is written and tested structurally; the user runs it manually.

**Tech Stack:** Python 3.11, TensorFlow 2.15, `uv run` for all commands.

---

## Chunk 1: Structural Tests + Benchmark Script

### Task 1: Write the 6 structural pytest tests

**Files:**
- Create: `tests/experiments/test_benchmark_neural_ode.py`

`tests/experiments/__init__.py` already exists. All 6 tests fail at import time until the benchmark script is created.

- [ ] **Step 1: Write the failing test file**

```python
# tests/experiments/test_benchmark_neural_ode.py
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


def test_systems_names():
    """System names match datasets available in generate_dataset."""
    expected = {
        'spiral', 'duffing', 'periodic_sinusoidal',
        'periodic_predator_prey', 'limited_predator_prey',
        'nonlinear_predator_prey',
    }
    assert set(SYSTEMS) == expected


def test_dense_units_covers_all_neurons():
    neurons_in_combinations = {n for n, _ in COMBINATIONS}
    for n in neurons_in_combinations:
        assert n in DENSE_UNITS


def test_ncp_motor_neurons_equals_ode_features():
    """motor_neurons must equal ODE output dimension (2)."""
    assert NCP_CONFIG['motor_neurons'] == 2
```

- [ ] **Step 2: Run to verify all 6 tests fail (import error)**

```bash
uv run pytest tests/experiments/test_benchmark_neural_ode.py -v
```

Expected: `ERROR — ModuleNotFoundError: No module named 'experiments.benchmark_neural_ode'`

---

### Task 2: Implement `experiments/benchmark_neural_ode.py`

**Files:**
- Create: `experiments/benchmark_neural_ode.py`

Implement constants, `build_model`, `run_one`, the three private output helpers, and `main()` in one file. The structural tests pass as soon as the constants are defined correctly.

- [ ] **Step 1: Write the complete script**

```python
# experiments/benchmark_neural_ode.py
import json
import os
import sys
import time
from datetime import datetime

import tensorflow as tf

from src.models import make_dense_model, make_ncp_model
from src.tasks.neural_ode.datasets import generate_dataset
from src.tasks.neural_ode.trainer import train

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


def build_model(neuron: str, wiring: str) -> tf.keras.Sequential:
    """Return the appropriate model for (neuron, wiring)."""
    if wiring == 'dense':
        return make_dense_model(neuron, units=DENSE_UNITS[neuron], output_neurons=2)
    else:
        return make_ncp_model(neuron, **NCP_CONFIG)


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
```

- [ ] **Step 2: Run structural tests — all 6 must pass**

```bash
uv run pytest tests/experiments/test_benchmark_neural_ode.py -v
```

Expected:
```
PASSED test_combinations_count
PASSED test_lrc_ar_ncp_excluded
PASSED test_systems_count
PASSED test_systems_names
PASSED test_dense_units_covers_all_neurons
PASSED test_ncp_motor_neurons_equals_ode_features
6 passed
```

- [ ] **Step 3: Commit**

```bash
git add experiments/benchmark_neural_ode.py tests/experiments/test_benchmark_neural_ode.py
git commit -m "feat: add Neural ODE benchmark script for all cell x wiring combinations (phase3/step10)"
```

---

### Task 3: Verify full test suite

**Files:** (no changes — verification only)

- [ ] **Step 1: Run the full pytest suite**

```bash
uv run pytest --tb=short -q
```

Expected: all tests pass (6 new + all pre-existing), 0 failures.

> ⚠️ **Do NOT run `uv run python experiments/benchmark_neural_ode.py`** — the actual
> training (42 × 2000 iterations) is intentionally not executed during development.
> The user will run it manually to review results before merging to main.

- [ ] **Step 2: Commit (only if Task 3 Step 1 required any fixes)**

If you needed to fix something: commit those fixes with a descriptive message.
Otherwise no commit needed here.
