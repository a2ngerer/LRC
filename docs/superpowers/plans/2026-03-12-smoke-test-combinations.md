# Smoke Test: All Cell × Wiring Combinations Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a standalone script that builds every cell × wiring combination, runs 3 gradient steps each, and reports 7 PASS + 1 XFAIL (lrc_ar + NCP) with exit code 0.

**Architecture:** Two new files: a pytest test file with 5 structural (import-only) tests that validates the configuration constants, and a standalone experiment script with the `run_combination` helper and `main()` that exercises all 8 combinations. TDD order: write and pass the structural tests first (they drive the shape of the constants), then implement the training logic.

**Tech Stack:** Python 3.11, TensorFlow 2.15, `uv run` for all commands. Existing factories `make_dense_model`/`make_ncp_model` from `src.models`.

---

## Chunk 1: Structural Tests + Smoke Test Script

### Task 1: Write the 5 structural pytest tests

**Files:**
- Create: `tests/experiments/test_smoke_combinations.py`

Context: `tests/experiments/__init__.py` already exists. The test file imports constants from `experiments/smoke_test_combinations.py` which does not exist yet — all 5 tests will fail at import time.

- [ ] **Step 1: Write the failing test file**

```python
# tests/experiments/test_smoke_combinations.py
from experiments.smoke_test_combinations import (
    NEURONS, WIRINGS, EXPECTED_FAIL, DENSE_UNITS, NCP_CONFIG,
)


def test_combination_matrix_size():
    """Matrix covers all planned neurons × wirings."""
    assert len(NEURONS) == 4
    assert len(WIRINGS) == 2


def test_expected_fail_is_subset_of_combinations():
    all_combos = {(n, w) for n in NEURONS for w in WIRINGS}
    assert EXPECTED_FAIL.issubset(all_combos)


def test_dense_units_covers_all_neurons():
    for n in NEURONS:
        assert n in DENSE_UNITS


def test_lrc_ar_dense_units_equals_spiral_features():
    """lrc_ar must have units == input features (2 for spiral)."""
    assert DENSE_UNITS['lrc_ar'] == 2


def test_ncp_motor_neurons_equals_spiral_features():
    """motor_neurons == output features == 2 for spiral."""
    assert NCP_CONFIG['motor_neurons'] == 2
```

- [ ] **Step 2: Run to verify all 5 tests fail (import error)**

```bash
uv run pytest tests/experiments/test_smoke_combinations.py -v
```

Expected: ERROR — `ModuleNotFoundError: No module named 'experiments.smoke_test_combinations'`

---

### Task 2: Implement `experiments/smoke_test_combinations.py`

**Files:**
- Create: `experiments/smoke_test_combinations.py`

Implement the constants, `run_combination`, and `main()` in one file. The structural tests only import constants — they pass as soon as the constants are defined correctly. The end-to-end script test (`uv run python experiments/smoke_test_combinations.py`) is the integration check.

- [ ] **Step 1: Write the complete script**

```python
# experiments/smoke_test_combinations.py
import sys
import tensorflow as tf
from src.models import make_dense_model, make_ncp_model

NEURONS = ['lrc', 'lrc_ar', 'ctrnn', 'lstm']
WIRINGS = ['dense', 'ncp']

# Dense config per neuron.
# lrc_ar uses raw input as the ODE state (v_pre = inputs), so mu/sigma weight
# matrices are (units×units). The broadcast in _sigmoid requires input_dim==units.
# For dense wiring with spiral data (features=2), lrc_ar must have units=2.
DENSE_UNITS = {
    'lrc':    8,
    'lrc_ar': 2,
    'ctrnn':  8,
    'lstm':   8,
}

# NCP config — same for all neurons.
# lrc_ar + NCP fails: the inter layer receives features=2 (raw input) but
# inter_neurons=8, so the (units,units)=(8,8) mu/sigma matrices cause a shape
# error in _sigmoid. This is an expected, documented incompatibility.
NCP_CONFIG = dict(inter_neurons=8, command_neurons=6, motor_neurons=2)

# Known-incompatible pairs. These are expected to fail — not bugs.
EXPECTED_FAIL = {('lrc_ar', 'ncp')}


def run_combination(neuron: str, wiring: str, n_iters: int = 3) -> tuple[bool, Exception | None]:
    """Build a model for (neuron, wiring) and run n_iters gradient steps.

    Uses synthetic random data — no real ODE dataset needed.

    Returns:
        (success, error) where error is None on success.
    """
    try:
        x = tf.random.normal((2, 10, 2))
        y = tf.random.normal((2, 10, 2))

        if wiring == 'dense':
            model = make_dense_model(neuron, units=DENSE_UNITS[neuron], output_neurons=2)
        else:
            model = make_ncp_model(neuron, **NCP_CONFIG)

        optimizer = tf.keras.optimizers.Adam()
        for _ in range(n_iters):
            with tf.GradientTape() as tape:
                pred = model(x, training=True)
                loss = tf.reduce_mean(tf.square(pred - y))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return True, None
    except Exception as e:
        return False, e


def main() -> int:
    """Run all combinations. Return exit code (0=success, 1=failure)."""
    results = {}
    for neuron in NEURONS:
        for wiring in WIRINGS:
            ok, err = run_combination(neuron, wiring)
            results[(neuron, wiring)] = (ok, err)

    # Print summary table
    print('\n=== Smoke Test: Cell × Wiring Combinations ===\n')
    print(f'  {"Neuron":<8} {"Wiring":<8} Status')
    print(f'  {"-------":<8} {"-------":<8} ------')

    n_pass = 0
    n_xfail = 0
    exit_code = 0

    for neuron in NEURONS:
        for wiring in WIRINGS:
            ok, err = results[(neuron, wiring)]
            key = (neuron, wiring)
            is_xfail = key in EXPECTED_FAIL

            if is_xfail:
                if ok:
                    # Expected to fail but passed — unexpected success
                    status = '❌ XPASS (unexpected pass — should have failed)'
                    exit_code = 1
                else:
                    status = '⚠️  XFAIL (expected — lrc_ar input constraint)'
                    n_xfail += 1
            else:
                if ok:
                    status = '✅ PASS'
                    n_pass += 1
                else:
                    status = f'❌ FAIL — {type(err).__name__}: {err}'
                    exit_code = 1

            print(f'  {neuron:<8} {wiring:<8} {status}')

    total_expected_pass = len(NEURONS) * len(WIRINGS) - len(EXPECTED_FAIL)
    print(f'\nResult: {n_pass} passed, {n_xfail} expected failure(s). '
          f'Exit {exit_code}.')

    return exit_code


if __name__ == '__main__':
    sys.exit(main())
```

- [ ] **Step 2: Run structural tests — all 5 must pass**

```bash
uv run pytest tests/experiments/test_smoke_combinations.py -v
```

Expected output:
```
PASSED tests/experiments/test_smoke_combinations.py::test_combination_matrix_size
PASSED tests/experiments/test_smoke_combinations.py::test_expected_fail_is_subset_of_combinations
PASSED tests/experiments/test_smoke_combinations.py::test_dense_units_covers_all_neurons
PASSED tests/experiments/test_smoke_combinations.py::test_lrc_ar_dense_units_equals_spiral_features
PASSED tests/experiments/test_smoke_combinations.py::test_ncp_motor_neurons_equals_spiral_features
5 passed
```

- [ ] **Step 3: Commit**

```bash
git add experiments/smoke_test_combinations.py tests/experiments/test_smoke_combinations.py
git commit -m "feat: add smoke test for all cell x wiring combinations (phase2/step9)"
```

---

### Task 3: Verify end-to-end smoke test run

**Files:** (no changes — verification only)

- [ ] **Step 1: Run the full pytest suite — all pre-existing tests must still pass**

```bash
uv run pytest --tb=short -q
```

Expected: all tests pass (5 new + all pre-existing), 0 failures.

- [ ] **Step 2: Run the smoke test script end-to-end**

```bash
uv run python experiments/smoke_test_combinations.py
echo "Exit code: $?"
```

Expected output:
```
=== Smoke Test: Cell × Wiring Combinations ===

  Neuron   Wiring   Status
  -------  -------  ------
  lrc      dense    ✅ PASS
  lrc      ncp      ✅ PASS
  lrc_ar   dense    ✅ PASS
  lrc_ar   ncp      ⚠️  XFAIL (expected — lrc_ar input constraint)
  ctrnn    dense    ✅ PASS
  ctrnn    ncp      ✅ PASS
  lstm     dense    ✅ PASS
  lstm     ncp      ✅ PASS

Result: 7 passed, 1 expected failure(s). Exit 0.
Exit code: 0
```

Exit code must be 0.

- [ ] **Step 3: Commit verification (no code changes)**

No commit needed for this task — verification only. If you had to fix something, commit those fixes with a descriptive message before marking done.
