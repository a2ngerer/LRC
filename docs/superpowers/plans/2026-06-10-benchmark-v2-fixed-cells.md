# Benchmark v2 (Vanishing-Gradient-Fixed Cells) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add vanishing-gradient-fixed cell variants (`cfc`, `mm_ltc`, `mm_lrc`) plus a gradient-clipping axis to the thesis benchmark, runnable as a `--profile v2` matrix alongside the unchanged v1 matrix, with a written comparison guide.

**Architecture:** Additive extension on branch `phase3/step12-benchmark-v2-fixed-cells` (spec: `docs/superpowers/specs/2026-06-10-benchmark-v2-fixed-cells-design.md`). Two new cell files subclassing `BaseCell`; `clip_norm` flows spec → runner → trainer; aggregation/plots get multi-dir loading with a `+clip` variant label so all downstream grouping code works unchanged.

**Tech Stack:** TensorFlow 2 / Keras (AbstractRNNCell), pytest, pandas/scipy (aggregation), uv, SLURM (dataLAB).

**Conventions:** All commands run from `code/`. Always `uv run`, never bare `python`. Branch is already created and checked out.

---

### Task 1: CfC_Cell

Closed-form continuous-time cell after Hasani et al. 2022 (arXiv:2106.13898), default ("gated") mode: shared backbone over `[input, state]`, four heads, sigmoid time-interpolation between two learned regimes. No ODE solver.

**Files:**
- Create: `src/neurons/cfc_cell.py`
- Modify: `src/neurons/__init__.py`, `src/models/rnn_model.py`
- Test: `tests/neurons/test_cells.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/neurons/test_cells.py`:

```python
# --- CfC_Cell ---

def test_cfc_cell_is_subclass_of_basecell():
    from src.neurons import CfC_Cell
    assert issubclass(CfC_Cell, BaseCell)


def test_cfc_cell_state_size():
    from src.neurons import CfC_Cell
    cell = CfC_Cell(units=32)
    assert cell.state_size == 32


def test_cfc_forward_pass_shape():
    from src.neurons import CfC_Cell
    units, batch, input_dim = 4, 2, 3
    cell = CfC_Cell(units=units)
    inputs = tf.zeros([batch, input_dim])
    state = [tf.zeros([batch, units])]
    output, new_state = cell(inputs, state)
    assert output.shape == (batch, units)
    assert new_state[0].shape == (batch, units)


def test_cfc_irregular_sampling():
    """CfC uses elapsed_time: different dt -> different next state."""
    from src.neurons import CfC_Cell
    tf.random.set_seed(0)
    cell = CfC_Cell(units=4)
    x = tf.random.normal([2, 3])
    state = [tf.random.normal([2, 4])]
    _, s1 = cell((x, 1.0), state)
    _, s2 = cell((x, 0.1), state)
    assert not tf.reduce_all(tf.abs(s1[0] - s2[0]) < 1e-7)


def test_cfc_state_stays_finite():
    """Closed-form update is saturated -> no blow-up over many steps."""
    from src.neurons import CfC_Cell
    tf.random.set_seed(0)
    cell = CfC_Cell(units=4)
    x = tf.random.normal([2, 3]) * 10.0
    state = [tf.zeros([2, 4])]
    for _ in range(100):
        _, state = cell(x, state)
    assert bool(tf.reduce_all(tf.math.is_finite(state[0])))


def test_cfc_make_model_and_gradient_flow():
    from src.models import make_dense_model
    tf.random.set_seed(0)
    model = make_dense_model('cfc', units=8, output_neurons=2)
    x = tf.random.normal((2, 10, 3))
    with tf.GradientTape() as tape:
        y = model(x)
        loss = tf.reduce_mean(tf.square(y))
    grads = tape.gradient(loss, model.trainable_variables)
    assert y.shape == (2, 10, 2)
    assert all(g is not None for g in grads)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/neurons/test_cells.py -k cfc -v`
Expected: FAIL/ERROR with `ImportError: cannot import name 'CfC_Cell'`

- [ ] **Step 3: Implement the cell**

Create `src/neurons/cfc_cell.py`:

```python
# Follows the CfC formulation by Hasani, Lechner et al. (2022),
# "Closed-form continuous-time neural networks", Nature Machine Intelligence
# 4(11), arXiv:2106.13898, and the reference implementation
# https://github.com/mlech26l/ncps/blob/master/ncps/tf/cfc_cell.py
# (default/gated mode). Verification note (thesis): cross-check gate signs
# against the paper PDF before citing equations.

import tensorflow as tf
from .base_cell import BaseCell


def _lecun_tanh(x):
    """Activation used by the CfC reference backbone."""
    return 1.7159 * tf.math.tanh(0.666 * x)


class CfC_Cell(BaseCell):
    def __init__(self, units, backbone_units=None, backbone_layers=1,
                 dt=1.0, **kwargs):
        """
        Closed-form Continuous-time (CfC) <https://arxiv.org/abs/2106.13898> cell.

        Approximates the LTC ODE solution in closed form -- no ODE solver,
        no unfolding. The hidden state is a sigmoid time-gated interpolation
        between two learned regimes:

            h(t) = sigma(t_a * t + t_b) interpolating ff1 <-> ff2

        which replaces the exponential decay of the exact solution and
        thereby avoids its vanishing-gradient factor (paper, Sec. 3).

        Args:
            units:           hidden state size
            backbone_units:  width of the shared backbone (default: units)
            backbone_layers: number of backbone Dense layers (default 1)
            dt:              default elapsed time for regularly sampled mode
        """
        super().__init__(units, **kwargs)
        self._backbone_units = backbone_units or units
        self._backbone_layers = backbone_layers
        self._dt = dt

    def build(self, input_shape):
        # Nested tuple -> first item is the feature tensor shape (same
        # convention as LTC_Cell.build).
        if isinstance(input_shape[0], (tuple, tf.TensorShape)):
            input_dim = input_shape[0][-1]
        else:
            input_dim = input_shape[-1]
        self.input_dim = input_dim

        width = input_dim + self.units
        self._backbone = []
        for i in range(self._backbone_layers):
            layer = tf.keras.layers.Dense(
                self._backbone_units, activation=_lecun_tanh,
                name=f'backbone_{i}',
            )
            layer.build((None, width))
            width = self._backbone_units
            self._backbone.append(layer)

        def _head(name):
            head = tf.keras.layers.Dense(self.units, name=name)
            head.build((None, width))
            return head

        self._ff1 = _head('ff1')
        self._ff2 = _head('ff2')
        self._time_a = _head('time_a')
        self._time_b = _head('time_b')
        self.built = True

    def call(self, inputs, states):
        if isinstance(inputs, (tuple, list)):
            # Irregularly sampled mode
            inputs, elapsed_time = inputs
        else:
            # Regularly sampled mode
            elapsed_time = self._dt

        x = tf.concat([inputs, states[0]], axis=-1)
        for layer in self._backbone:
            x = layer(x)

        ff1 = self._ff1(x)
        ff2 = self._ff2(x)
        t_a = self._time_a(x)
        t_b = self._time_b(x)
        t_interp = tf.nn.sigmoid(t_a * elapsed_time + t_b)
        new_state = ff1 * (1.0 - t_interp) + t_interp * ff2
        return new_state, [new_state]
```

Register the export in `src/neurons/__init__.py` (needed by the test imports):

```python
from .base_cell import BaseCell
from .lrc_cell import LRC_Cell
from .lrc_ar_cell import LRC_AR_Cell
from .ctrnn_cell import CTRNN_Cell
from .lstm_cell import LSTM_Cell
from .ltc_cell import LTC_Cell
from .gru_cell import GRU_Cell
from .cfc_cell import CfC_Cell

__all__ = ["BaseCell", "LRC_Cell", "LRC_AR_Cell", "CTRNN_Cell", "LSTM_Cell",
           "LTC_Cell", "GRU_Cell", "CfC_Cell"]
```

And add the registry entry in `src/models/rnn_model.py` (needed by
`test_cfc_make_model_and_gradient_flow`) — extend the import and registry:

```python
from src.neurons import (LRC_Cell, LRC_AR_Cell, CTRNN_Cell, LSTM_Cell,
                         LTC_Cell, GRU_Cell, CfC_Cell)

_CELL_REGISTRY = {
    "lrc":    LRC_Cell,
    "lrc_ar": LRC_AR_Cell,
    "ctrnn": CTRNN_Cell,
    "lstm": LSTM_Cell,
    "ltc": LTC_Cell,
    "gru": GRU_Cell,
    "cfc": CfC_Cell,
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/neurons/test_cells.py -k cfc -v`
Expected: 6 PASS

- [ ] **Step 5: Run the full cell test file (no regressions)**

Run: `uv run pytest tests/neurons/ -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/neurons/cfc_cell.py src/neurons/__init__.py src/models/rnn_model.py tests/neurons/test_cells.py
git commit -m "feat(neurons): CfC cell (closed-form continuous-time, gated mode)"
```

---

### Task 2: MixedMemoryCell (MM_LTC, MM_LRC)

ODE-LSTM pattern (Lechner & Hasani 2020, arXiv:2006.04418): LSTM gates own the
memory path `c` (constant error propagation); the wrapped ODE cell evolves the
hidden state continuously. The ODE cell's state *is* the LSTM hidden state —
no third state tensor.

**Files:**
- Create: `src/neurons/mixed_memory_cell.py`
- Modify: `src/neurons/__init__.py`, `src/models/rnn_model.py`
- Test: `tests/neurons/test_cells.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/neurons/test_cells.py`:

```python
# --- MixedMemoryCell (MM_LTC, MM_LRC) ---

def test_mm_cells_are_subclasses_of_basecell():
    from src.neurons import MM_LTC_Cell, MM_LRC_Cell, MixedMemoryCell
    assert issubclass(MixedMemoryCell, BaseCell)
    assert issubclass(MM_LTC_Cell, MixedMemoryCell)
    assert issubclass(MM_LRC_Cell, MixedMemoryCell)


def test_mm_state_size_is_h_and_c():
    from src.neurons import MM_LTC_Cell
    cell = MM_LTC_Cell(units=8)
    assert cell.state_size == [8, 8]


def test_mm_ltc_forward_pass_shape():
    from src.neurons import MM_LTC_Cell
    units, batch, input_dim = 4, 2, 3
    cell = MM_LTC_Cell(units=units)
    inputs = tf.zeros([batch, input_dim])
    state = [tf.zeros([batch, units]), tf.zeros([batch, units])]
    output, new_state = cell(inputs, state)
    assert output.shape == (batch, units)
    assert len(new_state) == 2
    assert new_state[0].shape == (batch, units)
    assert new_state[1].shape == (batch, units)


def test_mm_lrc_forwards_elastance_kwarg():
    from src.neurons import MM_LRC_Cell
    cell = MM_LRC_Cell(units=4, elastance_type='asymmetric')
    inputs = tf.zeros([2, 3])
    state = [tf.zeros([2, 4]), tf.zeros([2, 4])]
    output, _ = cell(inputs, state)
    assert cell._inner._elastance_type == 'asymmetric'
    assert output.shape == (2, 4)


def test_mm_ltc_irregular_sampling():
    """elapsed_time reaches the inner ODE cell: different dt -> different state."""
    from src.neurons import MM_LTC_Cell
    tf.random.set_seed(0)
    cell = MM_LTC_Cell(units=4)
    x = tf.random.normal([2, 3])
    state = [tf.random.normal([2, 4]), tf.random.normal([2, 4])]
    _, s1 = cell((x, 1.0), state)
    _, s2 = cell((x, 0.1), state)
    assert not tf.reduce_all(tf.abs(s1[0] - s2[0]) < 1e-7)


def test_mm_memory_path_isolated_from_ode():
    """The c path is pure LSTM gating: same x and (h, c) but different dt
    must yield the *same* new c (only h goes through the ODE)."""
    from src.neurons import MM_LTC_Cell
    tf.random.set_seed(0)
    cell = MM_LTC_Cell(units=4)
    x = tf.random.normal([2, 3])
    state = [tf.random.normal([2, 4]), tf.random.normal([2, 4])]
    _, s1 = cell((x, 1.0), state)
    _, s2 = cell((x, 0.1), state)
    assert bool(tf.reduce_all(tf.abs(s1[1] - s2[1]) < 1e-7))


def test_mm_make_models_and_gradient_flow():
    from src.models import make_dense_model
    for key in ('mm_ltc', 'mm_lrc'):
        tf.random.set_seed(0)
        model = make_dense_model(key, units=8, output_neurons=2)
        x = tf.random.normal((2, 10, 3))
        with tf.GradientTape() as tape:
            y = model(x)
            loss = tf.reduce_mean(tf.square(y))
        grads = tape.gradient(loss, model.trainable_variables)
        assert y.shape == (2, 10, 2)
        assert all(g is not None for g in grads), key
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/neurons/test_cells.py -k mm -v`
Expected: FAIL/ERROR with `ImportError: cannot import name 'MM_LTC_Cell'`

- [ ] **Step 3: Implement the wrapper**

Create `src/neurons/mixed_memory_cell.py`:

```python
# Follows the ODE-LSTM (mixed memory) pattern by Lechner & Hasani (2020),
# "Learning Long-Term Dependencies in Irregularly-Sampled Time Series",
# arXiv:2006.04418, reference implementation
# https://github.com/mlech26l/ode-lstms (ODELSTMCell): an LSTM owns the
# memory path c (additive update -> constant error propagation), while a
# continuous-time cell evolves the hidden state h. Here the inner ODE cell
# is pluggable (LTC or LRC) instead of a fixed CT-RNN.

import tensorflow as tf
from .base_cell import BaseCell
from .ltc_cell import LTC_Cell
from .lrc_cell import LRC_Cell


class MixedMemoryCell(BaseCell):
    """Mixed-memory wrapper: LSTM gating + inner continuous-time cell.

    Step (ODE-LSTM pattern):
      1. LSTM gates compute candidate hidden state h_cand and new memory c
         from (x, [h, c]).
      2. The inner ODE cell evolves h_cand with input x over elapsed_time;
         its state slot is h_cand, so the ODE state IS the hidden state.
      3. Output is the inner cell's (mapped) output; recurrent state is
         [evolved h, new c].

    Gradients along the memory path c never pass through the ODE solver's
    Jacobians -- this is the architectural fix for the vanishing/exploding
    gradient of BPTT through the ODE (Lechner & Hasani 2020, Theorem 1/2).

    State: [h, c], both (batch, units).
    """

    def __init__(self, units, inner_cell_cls, **inner_kwargs):
        super().__init__(units)
        self._inner_cell_cls = inner_cell_cls
        self._inner_kwargs = inner_kwargs

    @property
    def state_size(self):
        return [self.units, self.units]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        dtype = dtype or tf.float32
        return [
            tf.zeros([batch_size, self.units], dtype=dtype),  # h
            tf.zeros([batch_size, self.units], dtype=dtype),  # c
        ]

    def build(self, input_shape):
        self._lstm = tf.keras.layers.LSTMCell(self.units)
        self._lstm.build(input_shape)
        self._inner = self._inner_cell_cls(units=self.units, **self._inner_kwargs)
        self._inner.build(input_shape)
        self.built = True

    def call(self, inputs, states):
        if isinstance(inputs, (tuple, list)):
            x, elapsed_time = inputs
            inner_inputs = (x, elapsed_time)
        else:
            x = inputs
            inner_inputs = inputs
        h, c = states

        _, lstm_states = self._lstm(x, [h, c])
        h_cand, new_c = lstm_states

        output, inner_states = self._inner(inner_inputs, [h_cand])
        new_h = inner_states[0]
        return output, [new_h, new_c]


class MM_LTC_Cell(MixedMemoryCell):
    """Mixed-memory LTC: LSTM memory path + LTC continuous-time dynamics."""

    def __init__(self, units, **kwargs):
        super().__init__(units, LTC_Cell, **kwargs)


class MM_LRC_Cell(MixedMemoryCell):
    """Mixed-memory LRC: LSTM memory path + LRC continuous-time dynamics."""

    def __init__(self, units, **kwargs):
        super().__init__(units, LRC_Cell, **kwargs)
```

Update `src/neurons/__init__.py`:

```python
from .base_cell import BaseCell
from .lrc_cell import LRC_Cell
from .lrc_ar_cell import LRC_AR_Cell
from .ctrnn_cell import CTRNN_Cell
from .lstm_cell import LSTM_Cell
from .ltc_cell import LTC_Cell
from .gru_cell import GRU_Cell
from .cfc_cell import CfC_Cell
from .mixed_memory_cell import MixedMemoryCell, MM_LTC_Cell, MM_LRC_Cell

__all__ = ["BaseCell", "LRC_Cell", "LRC_AR_Cell", "CTRNN_Cell", "LSTM_Cell",
           "LTC_Cell", "GRU_Cell", "CfC_Cell", "MixedMemoryCell",
           "MM_LTC_Cell", "MM_LRC_Cell"]
```

Update `src/models/rnn_model.py` import and registry:

```python
from src.neurons import (LRC_Cell, LRC_AR_Cell, CTRNN_Cell, LSTM_Cell,
                         LTC_Cell, GRU_Cell, CfC_Cell, MM_LTC_Cell,
                         MM_LRC_Cell)

_CELL_REGISTRY = {
    "lrc":    LRC_Cell,
    "lrc_ar": LRC_AR_Cell,
    "ctrnn": CTRNN_Cell,
    "lstm": LSTM_Cell,
    "ltc": LTC_Cell,
    "gru": GRU_Cell,
    "cfc": CfC_Cell,
    "mm_ltc": MM_LTC_Cell,
    "mm_lrc": MM_LRC_Cell,
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/neurons/test_cells.py -k mm -v`
Expected: 7 PASS

- [ ] **Step 5: Verify NCP wiring compatibility (multi-state cell in 3-layer stack)**

Run:
```bash
uv run python - <<'EOF'
import tensorflow as tf
from src.models import make_ncp_model
for key in ('cfc', 'mm_ltc', 'mm_lrc'):
    m = make_ncp_model(key, inter_neurons=8, command_neurons=6, motor_neurons=2)
    y = m(tf.random.normal((2, 10, 2)))
    print(key, y.shape)
EOF
```
Expected: three lines `<key> (2, 10, 2)`, no exception.

- [ ] **Step 6: Run full test suite (no regressions)**

Run: `uv run pytest tests/ -q`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add src/neurons/mixed_memory_cell.py src/neurons/__init__.py src/models/rnn_model.py tests/neurons/test_cells.py
git commit -m "feat(neurons): mixed-memory cells MM_LTC/MM_LRC (ODE-LSTM pattern)"
```

---

### Task 3: Smoke-test coverage for the new cells

**Files:**
- Modify: `experiments/smoke_test_combinations.py:6-20`
- Test: `tests/experiments/test_smoke_combinations.py`

- [ ] **Step 1: Update the failing test first**

In `tests/experiments/test_smoke_combinations.py`, replace
`test_combination_matrix_size` with:

```python
def test_combination_matrix_size():
    """Matrix covers all planned neurons x wirings.

    Benchmark matrix v1: ltc, lrc, gru, lstm x dense, ncp.
    Benchmark matrix v2 adds: cfc, mm_ltc, mm_lrc.
    ctrnn and lrc_ar are kept as additional smoke-tested cells.
    """
    assert len(NEURONS) == 9
    for n in ['ltc', 'lrc', 'gru', 'lstm', 'cfc', 'mm_ltc', 'mm_lrc']:
        assert n in NEURONS
    assert len(WIRINGS) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/experiments/test_smoke_combinations.py -v`
Expected: `test_combination_matrix_size` FAILS (`len(NEURONS) == 6`)

- [ ] **Step 3: Update the smoke script**

In `experiments/smoke_test_combinations.py` change:

```python
NEURONS = ['lrc', 'lrc_ar', 'ctrnn', 'lstm', 'ltc', 'gru', 'cfc', 'mm_ltc', 'mm_lrc']
```

and extend `DENSE_UNITS`:

```python
DENSE_UNITS = {
    'lrc':    8,
    'lrc_ar': 2,
    'ctrnn':  8,
    'lstm':   8,
    'ltc':    8,
    'gru':    8,
    'cfc':    8,
    'mm_ltc': 8,
    'mm_lrc': 8,
}
```

(`EXPECTED_FAIL` stays `{('lrc_ar', 'ncp')}` — the new cells work in both wirings.)

- [ ] **Step 4: Run tests, then the smoke script itself**

Run: `uv run pytest tests/experiments/test_smoke_combinations.py -v`
Expected: all PASS

Run: `uv run python experiments/smoke_test_combinations.py`
Expected: exit code 0; 17 PASS + 1 XFAIL (9 neurons x 2 wirings)

- [ ] **Step 5: Commit**

```bash
git add experiments/smoke_test_combinations.py tests/experiments/test_smoke_combinations.py
git commit -m "test(experiments): smoke-test cfc and mixed-memory cells in both wirings"
```

---

### Task 4: Gradient clipping in trainer + tracker

**Files:**
- Modify: `src/tasks/neural_ode/trainer.py:42-90`
- Modify: `src/evaluation/gradient_flow.py`
- Test: Create `tests/tasks/test_trainer_clipping.py` (check first with
  `ls tests/` whether a `tests/tasks/` package exists; if not, also create an
  empty `tests/tasks/__init__.py`)

- [ ] **Step 1: Write the failing tests**

Create `tests/tasks/test_trainer_clipping.py`:

```python
import numpy as np
import tensorflow as tf

from src.models import make_dense_model
from src.tasks.neural_ode.datasets import generate_dataset
from src.tasks.neural_ode.ode_model import SequentialODEFunc
from src.tasks.neural_ode.trainer import train
from src.evaluation import GradientFlowTracker
from src.utils import set_global_seed


def _tiny_run(clip_norm, tracker=None, seed=0):
    rng = set_global_seed(seed)
    t, y = generate_dataset('spiral', data_size=60)
    model = SequentialODEFunc(make_dense_model('gru', units=4, output_neurons=2))
    losses = train(model, t, y, n_iters=3, batch_size=4, batch_time=8,
                   lr=1e-3, rng=rng, gradient_tracker=tracker,
                   clip_norm=clip_norm, verbose=False)
    return losses


def test_clip_norm_none_is_default_path():
    """clip_norm=None reproduces the deterministic v1 behavior."""
    assert _tiny_run(clip_norm=None) == _tiny_run(clip_norm=None)


def test_clip_norm_changes_training():
    """A tiny clip threshold must alter the loss trajectory."""
    baseline = _tiny_run(clip_norm=None)
    clipped = _tiny_run(clip_norm=1e-6)
    assert baseline != clipped


def test_tracker_records_clip_norms():
    tracker = GradientFlowTracker(log_every=1)
    _tiny_run(clip_norm=0.5, tracker=tracker)
    d = tracker.as_dict()
    assert d['clip']['iterations'] == [1, 2, 3]
    assert len(d['clip']['pre_clip_norm']) == 3
    assert all(c == 0.5 for c in d['clip']['clip_norm'])
    assert all(n >= 0 for n in d['clip']['pre_clip_norm'])


def test_tracker_clip_block_empty_without_clipping():
    tracker = GradientFlowTracker(log_every=1)
    _tiny_run(clip_norm=None, tracker=tracker)
    d = tracker.as_dict()
    assert d['clip']['iterations'] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/tasks/test_trainer_clipping.py -v`
Expected: FAIL with `TypeError: train() got an unexpected keyword argument 'clip_norm'`

- [ ] **Step 3: Implement clipping in the trainer**

In `src/tasks/neural_ode/trainer.py`, change the `train` signature:

```python
def train(model, t, y, n_iters, batch_size=16, batch_time=16, lr=1e-3,
          loss='mse', rng=None, gradient_tracker=None, clip_norm=None,
          verbose=True):
```

Add to the docstring args:

```
        clip_norm:  optional float; if set, gradients are rescaled with
                    tf.clip_by_global_norm(grads, clip_norm) before the
                    optimizer step. None (default) = v1 behavior.
```

Replace the gradient block (currently lines 79-82) with:

```python
        grads = tape.gradient(loss_value, model.trainable_variables)
        if clip_norm:
            applied_grads, pre_clip_norm = tf.clip_by_global_norm(grads, clip_norm)
        else:
            applied_grads = grads
        if gradient_tracker is not None and gradient_tracker.should_log(itr):
            gradient_tracker.record(itr, model, grads)
            if clip_norm:
                gradient_tracker.record_clip(
                    itr, float(pre_clip_norm.numpy()), clip_norm)
        optimizer.apply_gradients(zip(applied_grads, model.trainable_variables))
```

(Per-layer norms are always recorded pre-clip so RQ4 sees the raw gradient
pathology; the clip block shows when and how hard clipping engaged.)

- [ ] **Step 4: Implement clip recording in the tracker**

In `src/evaluation/gradient_flow.py`:

In `__init__`, change:

```python
        self.history = {
            'iterations': [], 'layers': {},
            'clip': {'iterations': [], 'pre_clip_norm': [], 'clip_norm': []},
        }
```

Add method after `record`:

```python
    def record_clip(self, iteration: int, pre_clip_norm: float,
                    clip_norm: float) -> None:
        """Store the global norm before clipping and the active threshold.

        The post-clip norm is min(pre_clip_norm, clip_norm) by construction,
        so it is not stored separately.
        """
        c = self.history['clip']
        c['iterations'].append(iteration)
        c['pre_clip_norm'].append(pre_clip_norm)
        c['clip_norm'].append(clip_norm)
```

In `as_dict`, add the key:

```python
        return {
            'log_every': self.log_every,
            'iterations': self.history['iterations'],
            'layer_norms': self.history['layers'],
            'clip': self.history['clip'],
        }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/tasks/test_trainer_clipping.py tests/experiments/ -v`
Expected: all PASS (existing `test_run_one_produces_complete_result` still
passes — `gradient_flow` dict gained an additive `clip` key only)

- [ ] **Step 6: Commit**

```bash
git add src/tasks/neural_ode/trainer.py src/evaluation/gradient_flow.py tests/tasks/
git commit -m "feat(trainer): optional global-norm gradient clipping with RQ4 clip logging"
```

---

### Task 5: Runner — clip axis, v2 profile, schema 2

**Files:**
- Modify: `experiments/run_benchmark.py`
- Test: `tests/experiments/test_run_benchmark.py`

- [ ] **Step 1: Write the failing tests**

In `tests/experiments/test_run_benchmark.py`, update the import block:

```python
from experiments.run_benchmark import (
    CELLS, CELLS_V2, WIRINGS, SYSTEMS, SEEDS, V2_CLIP_NORM,
    build_specs, build_specs_v2, result_filename, run_one, save_result,
)
```

Update the two existing spec tests (specs now carry `clip_norm`):

```python
def test_spec_order_is_deterministic():
    """Index order is the SLURM array contract — must be stable."""
    a = build_specs(CELLS, WIRINGS, SYSTEMS, SEEDS)
    b = build_specs(CELLS, WIRINGS, SYSTEMS, SEEDS)
    assert a == b
    assert a[0] == {'cell': 'ltc', 'wiring': 'dense', 'system': 'spiral',
                    'seed': 0, 'clip_norm': 0.0}
```

and in `test_run_one_produces_complete_result` change the schema assertion:

```python
        assert json.load(f)['schema_version'] == 2
```

Append the new tests:

```python
def test_v2_matrix_counts():
    """v2 = 3 new cells x {clip off, on} (360) + ltc/lrc x clip on (120)."""
    assert CELLS_V2 == ['mm_ltc', 'mm_lrc', 'cfc']
    specs = build_specs_v2()
    assert len(specs) == 480
    unclipped = [s for s in specs if s['clip_norm'] == 0.0]
    clipped = [s for s in specs if s['clip_norm'] == V2_CLIP_NORM]
    assert len(unclipped) == 180
    assert len(clipped) == 300
    assert {s['cell'] for s in unclipped} == set(CELLS_V2)
    assert {s['cell'] for s in clipped} == set(CELLS_V2) | {'ltc', 'lrc'}


def test_v2_spec_order_is_deterministic():
    a = build_specs_v2()
    b = build_specs_v2()
    assert a == b
    assert a[0] == {'cell': 'mm_ltc', 'wiring': 'dense', 'system': 'spiral',
                    'seed': 0, 'clip_norm': 0.0}


def test_result_filename_with_clip():
    spec = {'cell': 'mm_ltc', 'wiring': 'ncp', 'system': 'spiral', 'seed': 3,
            'clip_norm': 1.0}
    assert result_filename(spec) == 'mm_ltc_ncp_spiral_seed3_clip1.0.json'


def test_result_filename_no_clip_suffix_when_zero():
    spec = {'cell': 'ltc', 'wiring': 'ncp', 'system': 'spiral', 'seed': 3,
            'clip_norm': 0.0}
    assert result_filename(spec) == 'ltc_ncp_spiral_seed3.json'


def test_run_one_v2_cell_with_clip(tmp_path):
    spec = {'cell': 'cfc', 'wiring': 'dense', 'system': 'spiral', 'seed': 0,
            'clip_norm': 0.5}
    result = run_one(spec, _TINY_CFG)
    assert result['schema_version'] == 2
    assert result['config']['clip_norm'] == 0.5
    assert result['gradient_flow']['clip']['iterations'] == [2]
    assert len(result['training']['loss_history']) == _TINY_CFG['n_iters']
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/experiments/test_run_benchmark.py -v`
Expected: ImportError (`CELLS_V2`), then assertion failures

- [ ] **Step 3: Implement the runner changes**

In `experiments/run_benchmark.py`:

(a) After the `SEEDS` definition add:

```python
# Benchmark v2: vanishing-gradient-fixed cells (see
# docs/superpowers/specs/2026-06-10-benchmark-v2-fixed-cells-design.md).
CELLS_V2 = ['mm_ltc', 'mm_lrc', 'cfc']
# Clip threshold for the v2 clip-axis runs. 1.0 is the common recurrent-RL
# default; the exact value is an experiment parameter, not a tuned constant.
V2_CLIP_NORM = 1.0
```

(b) Extend `CELL_KWARGS` (mm_lrc forwards the elastance choice to the inner LRC):

```python
CELL_KWARGS = {
    'lrc': dict(elastance_type='asymmetric'),
    'mm_lrc': dict(elastance_type='asymmetric'),
}
```

(c) Replace `build_specs` and add `build_specs_v2`:

```python
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
```

(d) In `run_one`, pass the clip through and stamp schema 2 — change the
`train(...)` call and the result dict:

```python
    clip_norm = float(spec.get('clip_norm', 0.0))
    losses = train(
        model, t, y,
        n_iters=cfg['n_iters'], batch_size=cfg['batch_size'],
        batch_time=cfg['batch_time'], lr=cfg['lr'], loss=cfg['loss'],
        rng=rng, gradient_tracker=tracker,
        clip_norm=clip_norm or None,
    )
```

and in the returned dict:

```python
        'schema_version': 2,
```

plus inside `'config'`:

```python
            'clip_norm': clip_norm,
```

(e) Replace `result_filename`:

```python
def result_filename(spec: dict) -> str:
    base = f"{spec['cell']}_{spec['wiring']}_{spec['system']}_seed{spec['seed']}"
    if spec.get('clip_norm'):
        base += f"_clip{spec['clip_norm']}"
    return base + '.json'
```

(f) CLI: in `parse_args`, add to the selection group:

```python
    sel.add_argument('--profile', choices=['v1', 'v2'], default='v1',
                     help='v1: thesis matrix (240 runs, results/runs). '
                          'v2: fixed cells + clip axis (480 runs, results/runs_v2)')
    sel.add_argument('--clip-norm', type=float, default=0.0,
                     help='gradient clip threshold for explicit single runs '
                          '(0 = off); profile runs take it from the spec')
```

Extend `--cell` choices so explicit v2 single runs work:

```python
    sel.add_argument('--cell', choices=CELLS + CELLS_V2, default=None)
```

Change the matrix-filter defaults to `None` (filters now *subset* the
profile's spec list instead of building it — same outcomes for v1):

```python
    mat.add_argument('--cells', default=None, help='comma-separated cell subset')
    mat.add_argument('--systems', default=None, help='comma-separated system subset')
    mat.add_argument('--wirings', default=None, help='comma-separated wiring subset')
    mat.add_argument('--seeds', default=None, help='comma-separated seeds')
```

Change `--outdir` default to `None`:

```python
    tr.add_argument('--outdir', default=None,
                    help='default: results/runs (v1) / results/runs_v2 (v2)')
```

(g) In `main`, replace the spec construction (currently the
`specs = build_specs(...)` call) with:

```python
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
```

and extend the explicit single-run branch with the clip value:

```python
    elif all(v is not None for v in (args.cell, args.wiring, args.system, args.seed)):
        todo = [{'cell': args.cell, 'wiring': args.wiring,
                 'system': args.system, 'seed': args.seed,
                 'clip_norm': args.clip_norm}]
```

(h) Extend the `--list` print line to include the clip value:

```python
        for i, s in enumerate(specs):
            print(f"{i:4d}  {s['cell']:<7} {s['wiring']:<6} {s['system']:<26} "
                  f"seed={s['seed']} clip={s['clip_norm']}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/experiments/test_run_benchmark.py -v`
Expected: all PASS

- [ ] **Step 5: Verify the CLI contract manually**

```bash
uv run python experiments/run_benchmark.py --count               # expect 240
uv run python experiments/run_benchmark.py --profile v2 --count  # expect 480
uv run python experiments/run_benchmark.py --profile v2 --list | head -3
uv run python experiments/run_benchmark.py --profile v2 --list | tail -3
```
Expected: counts 240/480; first v2 lines show `mm_ltc dense spiral seed=0
clip=0.0`, last lines show `lrc ncp nonlinear_predator_prey seed=4 clip=1.0`.

- [ ] **Step 6: Commit**

```bash
git add experiments/run_benchmark.py tests/experiments/test_run_benchmark.py
git commit -m "feat(experiments): benchmark v2 profile — fixed cells + gradient-clip axis"
```

---

### Task 6: Aggregation and plots — multi-dir loading, clip variant label

Design: `load_runs` accepts several directories; a run with `clip_norm > 0`
gets the cell label `<cell>+clip`. All existing grouping, statistics, and
plotting code then works unchanged on the combined v1+v2 data.

**Files:**
- Modify: `experiments/aggregate_results.py:30-46,144-148`
- Modify: `experiments/plot_results.py:29-36,155-163`
- Test: Create `tests/experiments/test_aggregate_results.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/experiments/test_aggregate_results.py`:

```python
# tests/experiments/test_aggregate_results.py
import json
import os

from experiments.aggregate_results import load_runs


def _write_run(dirpath, cell, clip_norm, seed=0, nrmse=0.5):
    os.makedirs(dirpath, exist_ok=True)
    name = f'{cell}_dense_spiral_seed{seed}'
    if clip_norm:
        name += f'_clip{clip_norm}'
    payload = {
        'schema_version': 2,
        'run': {'cell': cell, 'wiring': 'dense', 'system': 'spiral',
                'seed': seed, 'clip_norm': clip_norm},
        'config': {'clip_norm': clip_norm},
        'training': {'final_loss': 0.1, 'duration_s': 1.0},
        'evaluation': {'mse': 0.2, 'nrmse': nrmse},
    }
    with open(os.path.join(dirpath, name + '.json'), 'w', encoding='utf-8') as f:
        json.dump(payload, f)


def test_load_runs_multiple_dirs(tmp_path):
    d1, d2 = str(tmp_path / 'runs'), str(tmp_path / 'runs_v2')
    _write_run(d1, 'ltc', 0.0)
    _write_run(d2, 'mm_ltc', 0.0)
    df = load_runs([d1, d2])
    assert sorted(df['cell']) == ['ltc', 'mm_ltc']


def test_load_runs_clip_variant_label(tmp_path):
    d = str(tmp_path / 'runs_v2')
    _write_run(d, 'ltc', 1.0)
    _write_run(d, 'ltc', 0.0, seed=1)
    df = load_runs([d])
    assert sorted(df['cell']) == ['ltc', 'ltc+clip']
    assert sorted(df['clip_norm']) == [0.0, 1.0]


def test_load_runs_schema1_backward_compatible(tmp_path):
    """v1 result JSONs (schema 1, no clip_norm anywhere) still load."""
    d = str(tmp_path / 'runs')
    os.makedirs(d, exist_ok=True)
    payload = {
        'schema_version': 1,
        'run': {'cell': 'gru', 'wiring': 'ncp', 'system': 'duffing', 'seed': 2},
        'config': {},
        'training': {'final_loss': 0.3, 'duration_s': 2.0},
        'evaluation': {'mse': 0.4, 'nrmse': 0.6},
    }
    with open(os.path.join(d, 'gru_ncp_duffing_seed2.json'), 'w',
              encoding='utf-8') as f:
        json.dump(payload, f)
    df = load_runs([d])
    assert list(df['cell']) == ['gru']
    assert list(df['clip_norm']) == [0.0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/experiments/test_aggregate_results.py -v`
Expected: FAIL — `load_runs` receives a list but expects a str (TypeError in
`os.path.join` or empty DataFrame)

- [ ] **Step 3: Implement in aggregate_results.py**

Replace `load_runs`:

```python
def load_runs(runs_dirs) -> pd.DataFrame:
    """Load run JSONs from one or more directories into a long DataFrame.

    Runs with an active gradient clip get the cell label '<cell>+clip' so
    every downstream groupby/statistic treats them as their own variant.
    """
    if isinstance(runs_dirs, str):
        runs_dirs = [runs_dirs]
    rows = []
    for runs_dir in runs_dirs:
        for path in sorted(glob(os.path.join(runs_dir, '*.json'))):
            with open(path, encoding='utf-8') as f:
                r = json.load(f)
            clip = float(r.get('config', {}).get('clip_norm', 0.0))
            cell = r['run']['cell'] + ('+clip' if clip else '')
            rows.append({
                'cell': cell,
                'wiring': r['run']['wiring'],
                'system': r['run']['system'],
                'seed': r['run']['seed'],
                'clip_norm': clip,
                'final_loss': r['training']['final_loss'],
                'mse': r['evaluation']['mse'],
                'nrmse': r['evaluation']['nrmse'],
                'duration_s': r['training']['duration_s'],
                'file': os.path.basename(path),
            })
    return pd.DataFrame(rows)
```

In `main`, change the CLI and add a cell filter for targeted comparison
families (keeps the Bonferroni m small):

```python
    p.add_argument('--runs', nargs='+', default=['results/runs'])
    p.add_argument('--out', default='results')
    p.add_argument('--cells', default=None,
                   help='comma-separated cell-variant subset, e.g. '
                        '"ltc,mm_ltc,ltc+clip"')
    args = p.parse_args(argv)

    df = load_runs(args.runs)
    if args.cells:
        df = df[df['cell'].isin(args.cells.split(','))]
```

(The `if df.empty` error message references `args.runs` — it now prints the
list, which is fine.)

- [ ] **Step 4: Implement in plot_results.py**

Replace `load_runs` (line 29) with:

```python
def load_runs(runs_dirs):
    if isinstance(runs_dirs, str):
        runs_dirs = [runs_dirs]
    runs = []
    for runs_dir in runs_dirs:
        for path in sorted(glob(os.path.join(runs_dir, '*.json'))):
            with open(path, encoding='utf-8') as f:
                r = json.load(f)
            if float(r.get('config', {}).get('clip_norm', 0.0)):
                r['run']['cell'] += '+clip'
            runs.append(r)
    return runs
```

and in `main` change the CLI arg:

```python
    p.add_argument('--runs', nargs='+', default=['results/runs'])
```

(`_group` and all plot functions key on `r['run']['cell']` and pick colors
from the sorted cell set, so the `+clip` variants get their own color/legend
entry automatically.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/experiments/ -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add experiments/aggregate_results.py experiments/plot_results.py tests/experiments/test_aggregate_results.py
git commit -m "feat(experiments): aggregate/plot across v1+v2 result dirs with +clip variants"
```

---

### Task 7: Cluster submit wrapper + comparison guide

**Files:**
- Create: `cluster/submit_benchmark_v2.sh`
- Create: `experiments/BENCHMARK_COMPARISON.md`
- Modify: `cluster/README.md` (append a short v2 section)

- [ ] **Step 1: Create the submit wrapper**

The existing `submit_benchmark.sh` already forwards runner args to both
`--count` and the sbatch job, so v2 needs only a thin wrapper.
Create `cluster/submit_benchmark_v2.sh`:

```bash
#!/usr/bin/env bash
# Submit the benchmark v2 matrix (fixed cells + clip axis, 480 runs).
# Thin wrapper: forwards --profile v2 to submit_benchmark.sh, which
# computes the array size via `run_benchmark.py --profile v2 --count`.
#
#   ./cluster/submit_benchmark_v2.sh
#   ./cluster/submit_benchmark_v2.sh -- --iters 4000   # extra runner args
set -euo pipefail
cd "$(dirname "$0")/.."

EXTRA_ARGS=()
if [[ "${1:-}" == "--" ]]; then
    shift
    EXTRA_ARGS=("$@")
fi
exec ./cluster/submit_benchmark.sh -- --profile v2 "${EXTRA_ARGS[@]}"
```

Run: `chmod +x cluster/submit_benchmark_v2.sh`

- [ ] **Step 2: Verify the wrapper computes the right array size (dry check)**

Run: `uv run python experiments/run_benchmark.py --profile v2 --count`
Expected: `480`

(The sbatch script itself needs no change: it forwards `"$@"` to
`run_benchmark.py --index N`, and `--profile v2` makes index N resolve in
the v2 spec list; `--outdir` defaults to `results/runs_v2` inside the runner.)

- [ ] **Step 3: Write the comparison guide**

Create `experiments/BENCHMARK_COMPARISON.md`:

```markdown
# Benchmark v1 vs v2 — How to Run and Compare

v1 is the thesis matrix with the cells as published: `{ltc, lrc, gru, lstm}
x {dense, ncp}` on 6 ODE systems x 5 seeds (240 runs). LTC/LRC are expected
to show convergence/gradient problems (vanishing/exploding gradients through
the ODE, Lechner & Hasani 2020; LTC stiffness, Farsang et al. 2024).

v2 adds the *fixed* variants (480 runs):

| Group | Cells | Clip | Runs | Isolates |
|---|---|---|---|---|
| Architecture fix | `mm_ltc`, `mm_lrc`, `cfc` | off | 180 | mixed memory / closed form alone |
| Both fixes | `mm_ltc`, `mm_lrc`, `cfc` | 1.0 | 180 | interaction architecture x optimizer |
| Optimizer fix | `ltc`, `lrc` | 1.0 | 120 | clipping alone on the problem cells |

Fixed-cell background: `mm_*` = ODE-LSTM mixed memory (memory path c is
LSTM-gated, gradients bypass the ODE Jacobians); `cfc` = closed-form
continuous-time cell (no solver, sigmoid time gate instead of exponential
decay). See `docs/superpowers/specs/2026-06-10-benchmark-v2-fixed-cells-design.md`.

## 1. Run v1

Local (slow, sequential):

    uv run python experiments/run_benchmark.py --all
    # -> results/runs/*.json (240 files)

Cluster (SLURM array, throttled to 8 GPUs):

    ./cluster/submit_benchmark.sh
    ./cluster/fetch_results.sh        # after completion

## 2. Run v2

Local:

    uv run python experiments/run_benchmark.py --profile v2 --all
    # -> results/runs_v2/*.json (480 files)

Cluster:

    ./cluster/submit_benchmark_v2.sh
    ./cluster/fetch_results.sh

Subsets work the same as v1, e.g. only the architecture-fix group:

    uv run python experiments/run_benchmark.py --profile v2 \
        --cells mm_ltc,mm_lrc,cfc --all

## 3. Compare

Aggregation reads both result dirs at once; runs with active clipping appear
as their own cell variant `<cell>+clip`:

    uv run python experiments/aggregate_results.py \
        --runs results/runs results/runs_v2 --out results

Targeted comparison families (smaller Bonferroni correction, sharper tests):

    # Architecture fix: base cell vs mixed-memory vs closed-form
    uv run python experiments/aggregate_results.py \
        --runs results/runs results/runs_v2 \
        --cells ltc,mm_ltc,cfc --out results/cmp_ltc_fixes

    uv run python experiments/aggregate_results.py \
        --runs results/runs results/runs_v2 \
        --cells lrc,mm_lrc --out results/cmp_lrc_fixes

    # Optimizer fix alone: clipping on the problem cells
    uv run python experiments/aggregate_results.py \
        --runs results/runs results/runs_v2 \
        --cells ltc,ltc+clip,lrc,lrc+clip --out results/cmp_clip

Figures (loss curves, phase portraits, gradient flow) across both versions:

    uv run python experiments/plot_results.py \
        --runs results/runs results/runs_v2 --out results/figures

## 4. How to read the results

| Question | Metric | Where |
|---|---|---|
| Does the fix improve final accuracy? (RQ1) | NRMSE (full-trajectory rollout), Wilcoxon p + Cohen's d | `summary.md` test blocks |
| Does the fix repair the gradient pathology? (RQ4) | per-layer gradient norms over iterations; clip engagement (`gradient_flow.clip`) | `figures/gradflow_*.png`, run JSONs |
| Does the fix stabilize training? | loss-curve variance across seeds, convergence iteration | `figures/loss_*.png` |

Hypotheses (from the solution report, to be confirmed or falsified):

1. `mm_ltc`/`mm_lrc` converge on systems where `ltc`/`lrc` oscillate or
   diverge (architecture fix works).
2. `cfc` reaches comparable NRMSE in less wall-clock time (no solver).
3. `ltc+clip`/`lrc+clip` improve less than `mm_*` (clipping bounds the
   explosion but does not fix the vanishing direction).
4. Gradient-norm spread (max/min over layers) is smaller for `mm_*` and
   `cfc` than for `ltc`/`lrc` (RQ4 mechanism evidence).

A negative result (fixes do not help on these supervised tasks) is still a
finding: it would localize the LTC/LRC weakness in the task/data regime
rather than the gradient path — document it, do not hide it.

## Caveats

- v1 result JSONs are schema 1, v2 (and any re-run v1) are schema 2; the
  aggregation reads both.
- `clip_norm = 1.0` is an experiment parameter, not a tuned optimum. If
  clip engagement (`gradient_flow.clip.pre_clip_norm` vs threshold) shows
  clipping almost never/always active, adjust and re-run the clip group.
- Wilcoxon pairing assumes identical (system, seed) coverage between the
  compared variants — keep seeds/systems identical across versions.
```

- [ ] **Step 4: Add the v2 pointer to cluster/README.md**

Append to `cluster/README.md`:

```markdown
## Benchmark v2 (fixed cells)

`./cluster/submit_benchmark_v2.sh` submits the 480-run v2 matrix
(`--profile v2`: mm_ltc/mm_lrc/cfc + gradient-clip axis). Results land in
`results/runs_v2/`. How to compare v1 vs v2: see
`experiments/BENCHMARK_COMPARISON.md`.
```

- [ ] **Step 5: Commit**

```bash
git add cluster/submit_benchmark_v2.sh cluster/README.md experiments/BENCHMARK_COMPARISON.md
git commit -m "docs(experiments): v2 submit wrapper and v1-vs-v2 comparison guide"
```

---

### Task 8: End-to-end verification + work log

**Files:**
- Modify: `../obsidian_master_thesis/Thesis/work-documentation.md`
  (lives in the parent repo directory, outside `code/`)

- [ ] **Step 1: Full test suite**

Run: `uv run pytest tests/ -q`
Expected: all PASS (87 pre-existing + ~20 new)

- [ ] **Step 2: Mini end-to-end v2 pipeline (local, CPU, ~2 min)**

```bash
uv run python experiments/run_benchmark.py --profile v2 --all \
    --cells mm_ltc,cfc --systems spiral --seeds 0 \
    --iters 30 --data-size 120 --outdir /tmp/v2_smoke
uv run python experiments/run_benchmark.py --all \
    --cells ltc --systems spiral --seeds 0 \
    --iters 30 --data-size 120 --outdir /tmp/v1_smoke
uv run python experiments/aggregate_results.py \
    --runs /tmp/v1_smoke /tmp/v2_smoke --out /tmp/v1v2_summary
```

Expected: the v2 call writes 8 JSONs (`--cells mm_ltc,cfc` filters the
480-spec list, so each cell appears with clip 0 and clip 1.0, x 2 wirings);
the v1 call writes 2; aggregation prints a summary containing the cell
variants `cfc, cfc+clip, ltc, mm_ltc, mm_ltc+clip`.

- [ ] **Step 3: Verify nothing in the v1 contract drifted**

```bash
uv run python experiments/run_benchmark.py --count          # 240
uv run python experiments/run_benchmark.py --list | head -1
```
Expected: `240`; first line still `   0  ltc ... dense  spiral ... seed=0 clip=0.0`

- [ ] **Step 4: Update the work log**

Append a `[2026-06-10] phase3/step12` entry to
`../obsidian_master_thesis/Thesis/work-documentation.md` summarizing: new
cells (cfc, mm_ltc, mm_lrc), clip axis, v2 profile (480 runs), comparison
guide, test counts, and that cluster submission of v2 awaits user GO.

- [ ] **Step 5: Push the branch**

```bash
git status              # expect clean (work-doc lives outside code/)
git push -u origin phase3/step12-benchmark-v2-fixed-cells
```

(No merge to `main` — per repo rules the branch waits for review.)

---

## Self-Review Notes

- Spec coverage: cells (T1, T2), registry/smoke (T1, T2, T3), trainer/tracker
  (T4), runner profile + filename + schema (T5), aggregation/plots (T6),
  cluster + guide (T7), verification (T8). Spec's "new sbatch" was simplified
  to a wrapper around the existing submit script (DRY; the sbatch already
  forwards runner args) — documented in T7.
- Existing-test updates are spelled out (spec dict gains `clip_norm`,
  schema 1 -> 2, smoke NEURONS 6 -> 9).
- Type consistency: `build_specs(..., clip_norm=...)` keyword used in
  `build_specs_v2` and tests; `record_clip(iteration, pre_clip_norm,
  clip_norm)` matches the trainer call; `load_runs` accepts list-or-str in
  both aggregate and plot.
