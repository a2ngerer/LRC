# Neural ODE Tasks Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the 6 Neural ODE training tasks from `neuralODE/run_ode.py` into `src/tasks/neural_ode/`, replacing broken `tfdiffeq` with `scipy` (data generation) and a self-contained TF Euler integrator, using `ODEFuncModel` that wraps `make_model`.

**Architecture:** `datasets.py` generates trajectories via `scipy.integrate.solve_ivp`; `ode_model.py` wraps `make_model` into an ODE function (Dense → RNN → Dense); `solver.py` provides a 15-line Euler integrator; `trainer.py` provides `get_batch` + `train` loop using `GradientTape`; `experiments/run_neural_ode.py` is the CLI entry point reading a YAML config.

**Tech Stack:** uv, TensorFlow 2.15, scipy, pyyaml, pytest. All deps already in `pyproject.toml`. Run commands with `uv run`.

**Spec:** `docs/superpowers/specs/2026-03-10-neural-ode-tasks-design.md`

---

## Chunk 1: Branch + Package Scaffolding + Datasets

### Task 1: Create feature branch

**Files:**
- (none — branch setup only)

- [ ] **Step 1: Create feature branch from main**

```bash
git checkout main && git pull
git checkout -b phase1/step5-port-neural-ode-tasks
```

Expected: now on branch `phase1/step5-port-neural-ode-tasks`

---

### Task 2: Package scaffolding — create all __init__.py files

**Files:**
- Create: `src/tasks/__init__.py`
- Create: `src/tasks/neural_ode/__init__.py`
- Create: `experiments/__init__.py`
- Create: `tests/tasks/__init__.py`
- Create: `tests/tasks/neural_ode/__init__.py`

- [ ] **Step 1: Create all empty __init__.py files**

Create each of the following as empty files (0 bytes):

- `src/tasks/__init__.py`
- `src/tasks/neural_ode/__init__.py`
- `experiments/__init__.py`
- `tests/tasks/__init__.py`
- `tests/tasks/neural_ode/__init__.py`

- [ ] **Step 2: Verify structure**

```bash
find src/tasks tests/tasks experiments -name "*.py" | sort
```

Expected output:
```
experiments/__init__.py
src/tasks/__init__.py
src/tasks/neural_ode/__init__.py
tests/tasks/__init__.py
tests/tasks/neural_ode/__init__.py
```

- [ ] **Step 3: Commit scaffolding**

```bash
git add src/tasks/ tests/tasks/ experiments/
git commit -m "feat(tasks): add package scaffolding for neural_ode tasks"
```

---

### Task 3: datasets.py — tests then implementation (TDD)

**Files:**
- Create: `tests/tasks/neural_ode/test_datasets.py`
- Create: `src/tasks/neural_ode/datasets.py`

- [ ] **Step 1: Create test file**

Create `tests/tasks/neural_ode/test_datasets.py` with this exact content:

```python
import pytest
import numpy as np
from src.tasks.neural_ode.datasets import generate_dataset


@pytest.mark.parametrize("name", [
    'spiral',
    'duffing',
    'periodic_sinusoidal',
    'periodic_predator_prey',
    'limited_predator_prey',
    'nonlinear_predator_prey',
])
def test_generate_dataset_shape_and_no_nan(name):
    t, y = generate_dataset(name)
    assert t.shape == (1000,), f"Expected t.shape (1000,), got {t.shape}"
    assert y.shape == (1000, 2), f"Expected y.shape (1000, 2), got {y.shape}"
    assert not np.any(np.isnan(t)), "t contains NaN"
    assert not np.any(np.isnan(y)), "y contains NaN"


def test_generate_dataset_invalid_name():
    with pytest.raises(ValueError):
        generate_dataset('invalid_name')
```

- [ ] **Step 2: Run tests — verify they FAIL**

```bash
uv run pytest tests/tasks/neural_ode/test_datasets.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` — `src.tasks.neural_ode.datasets` does not exist yet.

- [ ] **Step 3: Create `src/tasks/neural_ode/datasets.py`**

```python
import numpy as np
from scipy.integrate import solve_ivp

_SYSTEMS = {
    'spiral': {'t_span': (0, 25), 'y0': [0.5, 0.01]},
    'duffing': {'t_span': (0, 25), 'y0': [-1, 1]},
    'periodic_sinusoidal': {'t_span': (0, 10), 'y0': [1, 1]},
    'periodic_predator_prey': {'t_span': (0, 10), 'y0': [1, 1]},
    'limited_predator_prey': {'t_span': (0, 20), 'y0': [1, 1]},
    'nonlinear_predator_prey': {'t_span': (0, 20), 'y0': [2, 1]},
}

_A_SPIRAL = np.array([[-0.1, 3.0], [-3.0, -0.1]])
_A_NONLINEAR = 0.33


def _spiral(t, y):
    return y @ _A_SPIRAL


def _duffing(t, y):
    return [y[1], y[0] - y[0] ** 3]


def _periodic_sinusoidal(t, y):
    r = np.sqrt(y[0] ** 2 + y[1] ** 2)
    return [y[0] * (1 - r) - y[1], y[0] + y[1] * (1 - r)]


def _periodic_predator_prey(t, y):
    return [1.5 * y[0] - 1.0 * y[0] * y[1], -3.0 * y[1] + 1.0 * y[0] * y[1]]


def _limited_predator_prey(t, y):
    return [y[0] * (1 - y[0]) - y[0] * y[1], -y[1] + 2.0 * y[0] * y[1]]


def _nonlinear_predator_prey(t, y):
    return [
        y[0] * (1 - y[0]) + _A_NONLINEAR * y[0] * y[1],
        y[1] * (1 - y[1]) + y[0] * y[1],
    ]


_ODE_FUNCS = {
    'spiral': _spiral,
    'duffing': _duffing,
    'periodic_sinusoidal': _periodic_sinusoidal,
    'periodic_predator_prey': _periodic_predator_prey,
    'limited_predator_prey': _limited_predator_prey,
    'nonlinear_predator_prey': _nonlinear_predator_prey,
}


def generate_dataset(name: str, data_size: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Generate a trajectory for the named ODE system.

    Args:
        name: one of 'spiral', 'duffing', 'periodic_sinusoidal',
              'periodic_predator_prey', 'limited_predator_prey',
              'nonlinear_predator_prey'
        data_size: number of time points

    Returns:
        t: np.ndarray, shape (data_size,)
        y: np.ndarray, shape (data_size, 2)

    Raises:
        ValueError: if name is not recognized
    """
    if name not in _SYSTEMS:
        raise ValueError(f"Unknown system '{name}'. Valid: {list(_SYSTEMS.keys())}")

    params = _SYSTEMS[name]
    t_eval = np.linspace(params['t_span'][0], params['t_span'][1], data_size)

    sol = solve_ivp(
        _ODE_FUNCS[name],
        params['t_span'],
        params['y0'],
        method='DOP853',
        t_eval=t_eval,
        rtol=1e-3,
        atol=1e-6,
    )

    t = sol.t        # (data_size,)
    y = sol.y.T      # (data_size, 2) — solve_ivp returns (2, data_size), so transpose

    return t, y
```

- [ ] **Step 4: Run tests — verify they PASS**

```bash
uv run pytest tests/tasks/neural_ode/test_datasets.py -v
```

Expected: all 7 tests PASS (6 parametrized + 1 invalid name).

Troubleshooting:
- NaN values in a system: check the ODE equations and initial conditions match the original `neuralODE/run_ode.py` exactly.
- Shape wrong: `sol.y` has shape `(2, data_size)`, so `sol.y.T` gives `(data_size, 2)`. Verify the transpose.
- `nonlinear_predator_prey` produces NaN: initial condition `[2, 1]` is aggressive — verify `t_span=(0, 20)`.

- [ ] **Step 5: Commit**

```bash
git add src/tasks/neural_ode/datasets.py tests/tasks/neural_ode/test_datasets.py
git commit -m "feat(tasks/neural_ode): add generate_dataset for 6 ODE systems via scipy"
```

---

## Chunk 2: ODE Model + Solver

### Task 4: ode_model.py — tests then implementation (TDD)

**Files:**
- Create: `tests/tasks/neural_ode/test_ode_model.py`
- Create: `src/tasks/neural_ode/ode_model.py`

- [ ] **Step 1: Create test file**

Create `tests/tasks/neural_ode/test_ode_model.py` with this exact content:

```python
import tensorflow as tf
from src.tasks.neural_ode.ode_model import ODEFuncModel


def test_ode_model_forward_pass_shape():
    """ODEFuncModel maps (batch, 1, features) → (batch, 1, features)."""
    model = ODEFuncModel('lrc_ar', 'dense', 4, features=2)
    t = tf.constant(0.0)
    state = tf.zeros([1, 1, 2])
    output = model(t, state)
    assert output.shape == (1, 1, 2)


def test_ode_model_batch_size_independence():
    """Output shape scales correctly with batch size."""
    model = ODEFuncModel('lrc_ar', 'dense', 4, features=2)
    t = tf.constant(0.0)
    for batch in [1, 3]:
        state = tf.zeros([batch, 1, 2])
        output = model(t, state)
        assert output.shape == (batch, 1, 2), f"batch={batch}: expected ({batch}, 1, 2), got {output.shape}"
```

- [ ] **Step 2: Run tests — verify they FAIL**

```bash
uv run pytest tests/tasks/neural_ode/test_ode_model.py -v
```

Expected: `ImportError` — `src.tasks.neural_ode.ode_model` does not exist yet.

- [ ] **Step 3: Create `src/tasks/neural_ode/ode_model.py`**

```python
import tensorflow as tf
from tensorflow.keras.layers import Dense
from src.models import make_model


class ODEFuncModel(tf.keras.Model):
    def __init__(self, neuron_type, wiring_type, units, features, **cell_kwargs):
        """ODE function model: Dense(units) → RNN core → Dense(features).

        Wraps make_model as the RNN core and adds input/output projections
        so the model maps state (batch, 1, features) → derivative (batch, 1, features).

        Args:
            neuron_type: passed to make_model ('lrc_ar' for ODE tasks)
            wiring_type: passed to make_model ('dense')
            units:       RNN cell width
            features:    input/output feature dimension (2 for all 6 ODE systems)
            **cell_kwargs: forwarded to make_model → cell constructor
        """
        super().__init__()
        self.dense_in = Dense(units)
        self.rnn = make_model(neuron_type, wiring_type, units, **cell_kwargs)
        self.dense_out = Dense(features)

    def call(self, t, state):
        # t is unused (autonomous ODE) but kept for euler_odeint compatibility
        # state: (batch, 1, features)
        h = self.dense_in(state)    # (batch, 1, units)
        dh = self.rnn(h)            # (batch, 1, units)
        dy = self.dense_out(dh)     # (batch, 1, features)
        return dy
```

- [ ] **Step 4: Run tests — verify they PASS**

```bash
uv run pytest tests/tasks/neural_ode/test_ode_model.py -v
```

Expected: both tests PASS.

Troubleshooting:
- Shape error `(batch, 1, units)` mismatch into `rnn`: `make_model('lrc_ar', 'dense', 4)` with input `(batch, 1, units)` requires `features == units` for `LRC_AR_Cell` (autoregressive). Since `dense_in` outputs `units` features and the cell has `units` units, this is satisfied.
- `call` not receiving `t` as keyword: use positional args `model(t, state)` in tests, not `model(t=0.0, y=...)`.

- [ ] **Step 5: Commit**

```bash
git add src/tasks/neural_ode/ode_model.py tests/tasks/neural_ode/test_ode_model.py
git commit -m "feat(tasks/neural_ode): add ODEFuncModel wrapping make_model"
```

---

### Task 5: solver.py — tests then implementation (TDD)

**Files:**
- Create: `tests/tasks/neural_ode/test_solver.py`
- Create: `src/tasks/neural_ode/solver.py`

- [ ] **Step 1: Create test file**

Create `tests/tasks/neural_ode/test_solver.py` with this exact content:

```python
import numpy as np
import tensorflow as tf
from src.tasks.neural_ode.solver import euler_odeint


def test_euler_odeint_zero_ode():
    """dy/dt = 0 → state is constant at all time steps."""
    func = lambda t, y: tf.zeros_like(y)
    y0 = tf.constant([[[1.0, 2.0]]])  # (1, 1, 2)
    t = tf.constant([0.0, 1.0, 2.0])  # T=3

    result = euler_odeint(func, y0, t)

    assert result.shape == (3, 1, 1, 2), f"Expected (3, 1, 1, 2), got {result.shape}"
    for step in range(3):
        np.testing.assert_allclose(
            result[step].numpy(), y0.numpy(),
            err_msg=f"Step {step}: state changed but dy/dt=0"
        )


def test_euler_odeint_linear_ode():
    """dy/dt = y → exponential growth, values should increase monotonically."""
    func = lambda t, y: y
    y0 = tf.constant([[[1.0, 1.0]]])  # (1, 1, 2), positive initial state
    t = tf.constant([0.0, 0.1, 0.2, 0.3])  # T=4

    result = euler_odeint(func, y0, t)

    assert result.shape == (4, 1, 1, 2), f"Expected (4, 1, 1, 2), got {result.shape}"
    vals = result[:, 0, 0, 0].numpy()
    assert all(
        vals[i] <= vals[i + 1] for i in range(len(vals) - 1)
    ), f"Expected monotonically increasing values, got {vals}"
```

- [ ] **Step 2: Run tests — verify they FAIL**

```bash
uv run pytest tests/tasks/neural_ode/test_solver.py -v
```

Expected: `ImportError` — `src.tasks.neural_ode.solver` does not exist yet.

- [ ] **Step 3: Create `src/tasks/neural_ode/solver.py`**

```python
import tensorflow as tf


def euler_odeint(func, y0, t):
    """Euler ODE integrator.

    Args:
        func: callable(t, y) -> dy where t is a 0-D tensor and dy has the same shape as y
        y0:   initial state, shape (batch, 1, features)
        t:    time points, shape (T,) — numpy array or TF tensor with static shape

    Returns:
        Tensor of shape (T, batch, 1, features)

    Note: Uses Python list + tf.stack (eager mode only; no @tf.function).
    """
    ys = [y0]
    for i in range(t.shape[0] - 1):
        dt = t[i + 1] - t[i]
        dy = func(t[i], ys[-1])
        ys.append(ys[-1] + tf.cast(dt, ys[-1].dtype) * dy)
    return tf.stack(ys, axis=0)
```

- [ ] **Step 4: Run tests — verify they PASS**

```bash
uv run pytest tests/tasks/neural_ode/test_solver.py -v
```

Expected: both tests PASS.

Troubleshooting:
- Shape error: verify `tf.stack(ys, axis=0)` stacks along the first axis. With `ys = [y0, y1, ..., yT]` where each `yi` has shape `(batch, 1, features)`, the result is `(T, batch, 1, features)`.
- `t.shape[0]` AttributeError: if `t` is a Python list, wrap it first: `t = tf.constant(t)` in the test.
- Monotonicity test fails: with `dt=0.1` and `dy=y`, `y_next = y + 0.1*y = 1.1*y` — should be increasing. If it fails, verify `tf.cast(dt, ys[-1].dtype)` is not producing zero.

- [ ] **Step 5: Commit**

```bash
git add src/tasks/neural_ode/solver.py tests/tasks/neural_ode/test_solver.py
git commit -m "feat(tasks/neural_ode): add euler_odeint solver"
```

---

## Chunk 3: Trainer + Experiments

### Task 6: trainer.py — tests then implementation (TDD)

**Files:**
- Create: `tests/tasks/neural_ode/test_trainer.py`
- Create: `src/tasks/neural_ode/trainer.py`

- [ ] **Step 1: Create test file**

Create `tests/tasks/neural_ode/test_trainer.py` with this exact content:

```python
import numpy as np
import tensorflow as tf
from src.tasks.neural_ode.datasets import generate_dataset
from src.tasks.neural_ode.ode_model import ODEFuncModel
from src.tasks.neural_ode.solver import euler_odeint
from src.tasks.neural_ode.trainer import get_batch, train


def test_get_batch_shapes():
    """get_batch returns correct shapes for all three outputs."""
    t, y = generate_dataset('spiral', data_size=50)
    y0, t_batch, y_batch = get_batch(t, y, batch_size=4, batch_time=10)
    assert y0.shape == (4, 1, 2), f"Expected (4, 1, 2), got {y0.shape}"
    assert t_batch.shape == (10,), f"Expected (10,), got {t_batch.shape}"
    assert y_batch.shape == (10, 4, 1, 2), f"Expected (10, 4, 1, 2), got {y_batch.shape}"


def test_train_completes_without_error():
    """train runs 5 iterations without raising."""
    t, y = generate_dataset('spiral', data_size=50)
    model = ODEFuncModel('lrc_ar', 'dense', 4, features=2)
    train(model, t, y, n_iters=5, batch_size=4, batch_time=10)


def test_train_returns_loss_list():
    """train returns a list of floats with one entry per iteration."""
    t, y = generate_dataset('spiral', data_size=50)
    model = ODEFuncModel('lrc_ar', 'dense', 4, features=2)
    losses = train(model, t, y, n_iters=5, batch_size=4, batch_time=10)
    assert isinstance(losses, list), f"Expected list, got {type(losses)}"
    assert len(losses) == 5, f"Expected 5 losses, got {len(losses)}"
    assert all(isinstance(l, float) for l in losses), "All losses must be float"


def test_train_gradient_flow():
    """GradientTape captures gradients through euler_odeint."""
    t, y = generate_dataset('spiral', data_size=50)
    model = ODEFuncModel('lrc_ar', 'dense', 4, features=2)

    y0_batch, t_batch, y_batch = get_batch(t, y, batch_size=2, batch_time=5)
    t_tf = tf.constant(t_batch, dtype=tf.float32)
    y0_tf = tf.constant(y0_batch, dtype=tf.float32)
    y_true = tf.constant(y_batch, dtype=tf.float32)

    # Build model before tape so variables exist
    model(tf.constant(0.0), y0_tf)

    with tf.GradientTape() as tape:
        pred = euler_odeint(model, y0_tf, t_tf)
        loss = tf.reduce_mean(tf.abs(pred - y_true))

    grads = tape.gradient(loss, model.trainable_variables)
    assert len(model.trainable_variables) > 0, "Model should have trainable variables"
    assert any(
        g is not None and tf.reduce_any(g != 0.0).numpy()
        for g in grads
    ), "At least one gradient should be non-zero"
```

- [ ] **Step 2: Run tests — verify they FAIL**

```bash
uv run pytest tests/tasks/neural_ode/test_trainer.py -v
```

Expected: `ImportError` — `src.tasks.neural_ode.trainer` does not exist yet.

- [ ] **Step 3: Create `src/tasks/neural_ode/trainer.py`**

```python
import numpy as np
import tensorflow as tf
from .solver import euler_odeint


def get_batch(t, y, batch_size, batch_time):
    """Sample a random batch of sub-sequences.

    Args:
        t:          time array, shape (data_size,)
        y:          trajectory array, shape (data_size, 2)
        batch_size: number of random start points
        batch_time: length of each sub-sequence

    Returns:
        y0:       shape (batch_size, 1, 2)          — initial states
        t_batch:  shape (batch_time,)               — fixed time window t[:batch_time]
        y_batch:  shape (batch_time, batch_size, 1, 2) — trajectory for each batch element
    """
    data_size = len(t)
    s = np.random.choice(data_size - batch_time, batch_size, replace=False)

    y0 = y[s][:, np.newaxis, :]                           # (batch_size, 1, 2)
    t_batch = t[:batch_time]                               # (batch_time,)
    y_batch = np.stack(
        [y[s_i:s_i + batch_time, np.newaxis, :] for s_i in s], axis=1
    )  # (batch_time, batch_size, 1, 2)

    return y0, t_batch, y_batch


def train(model, t, y, n_iters, batch_size=16, batch_time=16, lr=1e-3):
    """Training loop.

    Args:
        model:      ODEFuncModel instance
        t:          time array from generate_dataset, shape (data_size,)
        y:          trajectory array from generate_dataset, shape (data_size, 2)
        n_iters:    number of training iterations
        batch_size: random start points per batch
        batch_time: time steps per batch
        lr:         Adam learning rate

    Returns:
        list of float losses, one per iteration
    """
    optimizer = tf.keras.optimizers.Adam(lr)
    losses = []

    for itr in range(1, n_iters + 1):
        y0_batch, t_batch, y_batch = get_batch(t, y, batch_size, batch_time)
        t_tf = tf.constant(t_batch, dtype=tf.float32)
        y0_tf = tf.constant(y0_batch, dtype=tf.float32)
        y_true = tf.constant(y_batch, dtype=tf.float32)

        with tf.GradientTape() as tape:
            # pred: (batch_time, batch_size, 1, 2)
            # true: (batch_time, batch_size, 1, 2)
            pred = euler_odeint(model, y0_tf, t_tf)
            loss = tf.reduce_mean(tf.abs(pred - y_true))

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        loss_val = float(loss.numpy())
        losses.append(loss_val)

        if itr % 10 == 0:
            print(f'Iter {itr:04d} | Loss {loss_val:.6f}')

    return losses
```

- [ ] **Step 4: Run tests — verify they PASS**

```bash
uv run pytest tests/tasks/neural_ode/test_trainer.py -v
```

Expected: all 4 tests PASS.

Troubleshooting:
- `get_batch` shape wrong: `y[s]` with `s` as array gives `(batch_size, 2)`. Then `[:, np.newaxis, :]` gives `(batch_size, 1, 2)`. If shape is wrong, print intermediate shapes.
- Gradient flow test: all gradients are `None` — this means variables weren't watched. Ensure `model(tf.constant(0.0), y0_tf)` is called before the tape to build the model. The tape with `watch_accessed_variables=True` (default) will watch all trainable variables accessed inside it.
- All gradients are zero: unlikely for random init, but if it happens, check that `euler_odeint` is called inside the `GradientTape` context (not outside it).

- [ ] **Step 5: Commit**

```bash
git add src/tasks/neural_ode/trainer.py tests/tasks/neural_ode/test_trainer.py
git commit -m "feat(tasks/neural_ode): add get_batch and train loop"
```

---

### Task 7: experiments — YAML config + CLI entry point

**Files:**
- Create: `experiments/configs/neural_ode_lrc_spiral.yaml`
- Create: `experiments/run_neural_ode.py`

Note: No TDD for the entry point — correctness verified by running it directly.

- [ ] **Step 1: Create YAML config**

Create `experiments/configs/neural_ode_lrc_spiral.yaml`:

```yaml
neuron: lrc_ar
wiring: dense
data: spiral
units: 16
niters: 100
batch_size: 16
batch_time: 16
lr: 0.001
```

- [ ] **Step 2: Create `experiments/run_neural_ode.py`**

```python
import argparse
import yaml
from src.tasks.neural_ode.datasets import generate_dataset
from src.tasks.neural_ode.ode_model import ODEFuncModel
from src.tasks.neural_ode.trainer import train


def main():
    parser = argparse.ArgumentParser(description='Train Neural ODE model')
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Training Neural ODE | data={config['data']} neuron={config['neuron']} units={config['units']}")

    t, y = generate_dataset(config['data'])
    model = ODEFuncModel(config['neuron'], config['wiring'], config['units'], features=2)

    losses = train(
        model, t, y,
        n_iters=config['niters'],
        batch_size=config['batch_size'],
        batch_time=config['batch_time'],
        lr=config['lr'],
    )

    print(f'Training complete. Final loss: {losses[-1]:.6f}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 3: Run the success criterion**

```bash
uv run python experiments/run_neural_ode.py --config experiments/configs/neural_ode_lrc_spiral.yaml
```

Expected: prints header line, then `Iter 0010 | Loss X.XXXXXX` every 10 iterations, ends with `Training complete. Final loss: X.XXXXXX`. Exits with code 0.

Troubleshooting:
- `ModuleNotFoundError: No module named 'src'`: run from the repo root (`/Users/angeral/Repositories/master_thesis/code`) — the package is installed in editable mode via `pyproject.toml`.
- KeyError in config: verify YAML keys match exactly (`niters`, `batch_size`, `batch_time`, `lr`, `data`, `neuron`, `wiring`, `units`).

- [ ] **Step 4: Commit**

```bash
git add experiments/configs/ experiments/run_neural_ode.py
git commit -m "feat(experiments): add run_neural_ode.py CLI with spiral config"
```

---

## Chunk 4: Full Suite + Merge + Work Log

### Task 8: Full test suite + branch merge

**Files:**
- (none — verification and git operations only)

- [ ] **Step 1: Run full test suite**

```bash
uv run pytest -v
```

Expected: all tests PASS (previous neurons + wirings + models + new neural_ode tasks = at least 34 tests).

If any test fails, fix before proceeding.

- [ ] **Step 2: Merge branch to main**

```bash
git checkout main
git merge --no-ff phase1/step5-port-neural-ode-tasks -m "merge: phase1/step5-port-neural-ode-tasks — Neural ODE tasks with scipy + Euler solver"
```

- [ ] **Step 3: Push branch and main to origin**

```bash
git push origin phase1/step5-port-neural-ode-tasks
git push origin main
```

---

### Task 9: Update Obsidian work log and project status

**Files:**
- Modify: `Thesis/work-documentation.md` (Obsidian vault) — prepend new entry at top
- Modify: `Meta/claude-project-status.md` (Obsidian vault) — mark step 5 done, add step 6

- [ ] **Step 1: Prepend to Obsidian work log**

Read `Thesis/work-documentation.md` and prepend this entry at the top (before the existing Step 4 entry):

```markdown
## 2026-03-10 — Phase 1 / Step 5: Neural ODE Tasks

**Branch:** `phase1/step5-port-neural-ode-tasks` (merged ✅)

### What was done
- Created `src/tasks/neural_ode/` package with 4 modules
- `datasets.py`: 6 ODE systems via `scipy.integrate.solve_ivp` (DOP853), replacing broken `tfdiffeq`
- `ode_model.py`: `ODEFuncModel` — Dense(units) → make_model RNN core → Dense(features)
- `solver.py`: `euler_odeint` — 15-line TF Euler integrator (equivalent to original `method='euler'`)
- `trainer.py`: `get_batch` + `train` loop using `GradientTape` + Adam
- `experiments/run_neural_ode.py`: CLI entry point reading YAML config
- 13 new tests (6 datasets + 2 ode_model + 2 solver + 4 trainer)

### Files changed
- `src/tasks/__init__.py` (new)
- `src/tasks/neural_ode/__init__.py` (new)
- `src/tasks/neural_ode/datasets.py` (new)
- `src/tasks/neural_ode/ode_model.py` (new)
- `src/tasks/neural_ode/solver.py` (new)
- `src/tasks/neural_ode/trainer.py` (new)
- `experiments/__init__.py` (new)
- `experiments/configs/neural_ode_lrc_spiral.yaml` (new)
- `experiments/run_neural_ode.py` (new)
- `tests/tasks/__init__.py` (new)
- `tests/tasks/neural_ode/__init__.py` (new)
- `tests/tasks/neural_ode/test_datasets.py` (new)
- `tests/tasks/neural_ode/test_ode_model.py` (new)
- `tests/tasks/neural_ode/test_solver.py` (new)
- `tests/tasks/neural_ode/test_trainer.py` (new)

### Next
`phase1/step6-verify-lrc-results` — verify LRC + Dense reproduces original paper results

---
```

- [ ] **Step 2: Update `Meta/claude-project-status.md`**

Mark `phase1/step5-port-neural-ode-tasks` as done (`[x]`) and add `phase1/step6-verify-lrc-results` as next pending step.
