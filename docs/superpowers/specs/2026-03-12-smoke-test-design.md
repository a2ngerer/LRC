# Design: Integration Smoke Test — All Cell × Wiring Combinations (Phase 2 / Step 9)

**Date:** 2026-03-12
**Branch:** `phase2/step9-smoke-test`
**Status:** Approved

## Goal

Verify that all available cell × wiring combinations can build, run a forward pass, and complete 3 gradient steps without crashing. Document the one known-incompatible combination (`lrc_ar + NCP`) as an expected failure.

## Architecture Matrix (current state)

| Neuron   | Dense     | NCP               |
|----------|-----------|-------------------|
| lrc      | ✅ PASS   | ✅ PASS           |
| lrc_ar   | ✅ PASS   | ⚠️ XFAIL          |
| ctrnn    | ✅ PASS   | ✅ PASS           |
| lstm     | ✅ PASS   | ✅ PASS           |

STC is excluded (blocked pending supervisor meeting).

## File Layout

```
experiments/
    smoke_test_combinations.py   # new: main smoke test script
tests/experiments/
    test_smoke_combinations.py   # new: thin structural test (no training)
```

## Component Specifications

### `experiments/smoke_test_combinations.py`

**Constants:**

```python
NEURONS = ['lrc', 'lrc_ar', 'ctrnn', 'lstm']
WIRINGS = ['dense', 'ncp']

# Dense config per neuron (units must equal input features for lrc_ar)
DENSE_UNITS = {
    'lrc':    8,
    'lrc_ar': 2,   # spiral features=2; lrc_ar requires input_dim == units
    'ctrnn':  8,
    'lstm':   8,
}

# NCP config — same for all neurons; lrc_ar fails (inter_neurons=8 ≠ features=2)
NCP_CONFIG = dict(inter_neurons=8, command_neurons=6, motor_neurons=2)

# Known-incompatible combinations: expected to fail, not a bug
EXPECTED_FAIL = {('lrc_ar', 'ncp')}
```

**Training helper:**

```python
def run_combination(neuron: str, wiring: str, n_iters: int = 3) -> tuple[bool, Exception | None]:
    """Build a model for (neuron, wiring) and run n_iters gradient steps.

    Returns:
        (success, error) where error is None on success.
    """
    # Input: random (batch=2, timesteps=10, features=2), target same shape
    # Build model
    # Run n_iters steps with Adam + MSE loss using tf.GradientTape
    # Return (True, None) on success, (False, exception) on failure
```

Model construction:
- Dense: `make_dense_model(neuron, units=DENSE_UNITS[neuron], output_neurons=2)`
- NCP: `make_ncp_model(neuron, **NCP_CONFIG)`

Training loop per iteration:
```python
with tf.GradientTape() as tape:
    pred = model(x, training=True)
    loss = tf.reduce_mean(tf.square(pred - y))
grads = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

**`main()` function:**

```python
def main() -> int:
    """Run all combinations. Return exit code (0=success, 1=failure)."""
    results = {}
    for neuron in NEURONS:
        for wiring in WIRINGS:
            ok, err = run_combination(neuron, wiring)
            results[(neuron, wiring)] = (ok, err)

    # Print summary table + per-combination status
    # Determine exit code:
    #   0 if all expected-pass combos passed AND all xfail combos actually failed
    #   1 if any expected-pass combo failed OR any xfail combo unexpectedly passed
```

**Output format:**
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

Result: 7 passed, 1 expected failure. Exit 0.
```

If an expected-pass combo fails, it shows the exception:
```
  lrc      dense    ❌ FAIL — ValueError: ...
```

Exit code 1 if any unexpected failure or unexpected success (xfail that passed).

**Entry point:**
```python
if __name__ == '__main__':
    sys.exit(main())
```

---

### `tests/experiments/test_smoke_combinations.py`

Thin structural test — no training, runs in the normal pytest suite.

```python
from experiments.smoke_test_combinations import (
    NEURONS, WIRINGS, EXPECTED_FAIL, DENSE_UNITS, NCP_CONFIG, run_combination
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

---

## Success Criterion

```bash
uv run python experiments/smoke_test_combinations.py
```

Exit code 0. Output shows 7 PASS + 1 XFAIL.

```bash
uv run pytest tests/experiments/test_smoke_combinations.py -v
```

5/5 structural tests pass. Full suite: 59 existing + 5 new = **64/64**.
