# Design: Neural ODE Tasks (Phase 1 / Step 5)

**Date:** 2026-03-10
**Branch:** `phase1/step5-port-neural-ode-tasks`
**Status:** Approved

## Goal

Port the 6 Neural ODE training tasks from `neuralODE/run_ode.py` into the modular `src/` structure. Replace the broken `tfdiffeq` dependency with `scipy` (data generation) and a self-contained TF Euler integrator (prediction). Use `ODEFuncModel` — which wraps `make_model` — as the ODE function.

## tfdiffeq Decision

`tfdiffeq` is not on PyPI; the GitHub package (`titu1994/tfdiffeq`) fails to import on modern matplotlib because `viz_utils.py` calls `plt.style.use('seaborn-paper')` (removed in matplotlib ≥ 3.6). Rather than patch an external package:

- **Data generation** → `scipy.integrate.solve_ivp` (already in `pyproject.toml`)
- **ODE integration for prediction** → self-written `euler_odeint` (~15 lines TF)

The original code already used `method='euler'` for all LRC experiments, so behaviour is identical.

## File Layout

```
src/tasks/neural_ode/
    __init__.py         # empty
    datasets.py         # generate_dataset(name, data_size) → (t, y) via scipy
    ode_model.py        # ODEFuncModel — wraps make_model as ODE function
    solver.py           # euler_odeint(func, y0, t) — Euler ODE integrator
    trainer.py          # get_batch + train loop
experiments/
    __init__.py         # empty
    configs/
        neural_ode_lrc_spiral.yaml
    run_neural_ode.py   # CLI entry point
tests/tasks/
    __init__.py
    neural_ode/
        __init__.py
        test_datasets.py
        test_ode_model.py
        test_solver.py
        test_trainer.py
```

## Component Specifications

### datasets.py

Single public function:

```python
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
```

All 6 systems return 2D trajectories (features=2). Uses `scipy.integrate.solve_ivp` internally. System parameters (initial conditions, time span, ODE coefficients) match `neuralODE/run_ode.py` exactly.

| System | t_span | y0 |
|--------|--------|----|
| spiral | [0, 25] | [0.5, 0.01] |
| duffing | [0, 25] | [-1, 1] |
| periodic_sinusoidal | [0, 10] | [1, 1] |
| periodic_predator_prey | [0, 10] | [1, 1] |
| limited_predator_prey | [0, 20] | [1, 1] |
| nonlinear_predator_prey | [0, 20] | [2, 1] |

### ode_model.py

```python
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
        self.dense_in = Dense(units)
        self.rnn = make_model(neuron_type, wiring_type, units, **cell_kwargs)
        self.dense_out = Dense(features)

    def call(self, t, y):
        # y: (batch, 1, features)
        h = self.dense_in(y)    # (batch, 1, units)
        dh = self.rnn(h)        # (batch, 1, units)
        dy = self.dense_out(dh) # (batch, 1, features)
        return dy
```

Note: `t` is accepted but unused (autonomous ODE). Kept for compatibility with the `euler_odeint` calling convention.

### solver.py

```python
def euler_odeint(func, y0, t):
    """Euler ODE integrator.

    Args:
        func: callable(t_scalar, y) -> dy/dt, same shape as y
        y0:   initial state, shape (batch, 1, features)
        t:    time points as TF tensor, shape (T,)

    Returns:
        Tensor of shape (T, batch, 1, features)
    """
    ys = [y0]
    for i in tf.range(len(t) - 1):
        dt = t[i + 1] - t[i]
        dy = func(t[i], ys[-1])
        ys.append(ys[-1] + tf.cast(dt, ys[-1].dtype) * dy)
    return tf.stack(ys, axis=0)
```

### trainer.py

```python
def get_batch(t, y, batch_size, batch_time):
    """Sample a random batch of sub-sequences.

    Args:
        t:          time array, shape (data_size,)
        y:          trajectory array, shape (data_size, 2)
        batch_size: number of random start points
        batch_time: length of each sub-sequence

    Returns:
        y0:       shape (batch_size, 1, 2)   — initial states
        t_batch:  shape (batch_time,)         — time points
        y_batch:  shape (batch_time, batch_size, 1, 2) — true trajectory
    """

def train(model, t, y, n_iters, batch_size=16, batch_time=16, lr=1e-3):
    """Training loop.

    Args:
        model:      ODEFuncModel instance
        t:          time array from generate_dataset
        y:          trajectory array from generate_dataset
        n_iters:    number of training iterations
        batch_size: random start points per batch
        batch_time: time steps per batch
        lr:         Adam learning rate

    Returns:
        list of loss values (one per iteration)
    """
    # Uses GradientTape + Adam
    # Loss: mean absolute error between euler_odeint(model, y0, t_batch) and y_batch
    # Prints loss every 10 iterations
```

### experiments/configs/neural_ode_lrc_spiral.yaml

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

### experiments/run_neural_ode.py

```python
# Usage: uv run python experiments/run_neural_ode.py --config experiments/configs/neural_ode_lrc_spiral.yaml
```

- Parses `--config` argument
- Loads YAML with `pyyaml`
- Calls `generate_dataset(config['data'])`
- Builds `ODEFuncModel(config['neuron'], config['wiring'], config['units'], features=2)`
- Calls `train(model, t, y, config['niters'], ...)`

## Tests

### test_datasets.py (6 tests, one per system)

Parametrize over all 6 system names. For each: assert `t.shape == (1000,)`, `y.shape == (1000, 2)`, no NaN values.

### test_ode_model.py (2 tests)

1. Forward pass: `ODEFuncModel('lrc_ar', 'dense', 4, 2)(t=0.0, y=tf.zeros([1, 1, 2]))` → output shape `(1, 1, 2)`
2. Returns same shape regardless of batch size (batch=3)

### test_solver.py (2 tests)

1. Zero ODE: `dy/dt = 0` → output equals y0 at all time steps; shape `(T, batch, 1, features)` correct
2. Linear ODE: `dy/dt = y` → output values increase monotonically (sanity check)

### test_trainer.py (3 tests)

1. `get_batch` output shapes correct
2. `train` completes 5 iterations on spiral without error
3. `train` returns list of float losses of length == n_iters

## Success Criterion

```bash
uv run python experiments/run_neural_ode.py --config experiments/configs/neural_ode_lrc_spiral.yaml
```

Runs 100 iterations, prints loss every 10 steps, exits cleanly.

## Out of Scope

- Visualization (phase portraits, training curves) → Step 6 / evaluation module
- YAML validation / error messages for bad configs
- Multiple ODE system runs in one script
- GPU device placement
- Checkpointing / model saving
