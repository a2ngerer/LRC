# Design: CT-RNN + LSTM Cells (Phase 2 / Steps 7+8)

**Date:** 2026-03-12
**Branch:** `phase2/step7-new-cells`
**Status:** Approved

## Goal

Add two new neuron types to `src/neurons/`: `CTRNN_Cell` (Continuous-Time RNN) and `LSTM_Cell` (standard LSTM baseline). Register both in `make_model()`. Each cell must be independently testable and interchangeable with LRC via the existing factory.

## File Layout

```
src/neurons/
    ctrnn_cell.py           # new
    lstm_cell.py            # new
    __init__.py             # updated: export CTRNN_Cell, LSTM_Cell
src/models/
    rnn_model.py            # updated: 'ctrnn' and 'lstm' registry entries
tests/neurons/
    test_cells.py           # extended: +9 tests (4 CT-RNN + 5 LSTM)
```

## Component Specifications

### src/neurons/ctrnn_cell.py

```python
import tensorflow as tf
from .base_cell import BaseCell


class CTRNN_Cell(BaseCell):
    """Continuous-Time RNN cell (leaky integrator ODE).

    ODE: dh/dt = (-h + tanh(W_x·x + W_h·h + b)) / τ
    Euler step: h_new = h + (dt/τ) · (-h + tanh(W_x·x + W_h·h + b))

    Args:
        units:   number of recurrent units
        epsilon: small constant added to τ for numerical stability (default 1e-8)

    Irregular sampling convention (inherited from BaseCell):
        If inputs is a tuple (x, elapsed_time), elapsed_time is used as dt.
        Otherwise dt defaults to 1.0.
    """

    def __init__(self, units, epsilon=1e-8, **kwargs):
        super().__init__(units, **kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        if isinstance(input_shape[0], (tuple, tf.TensorShape)):
            input_dim = input_shape[0][-1]
        else:
            input_dim = input_shape[-1]

        self.W_x = self.add_weight(
            name='W_x', shape=(input_dim, self.units),
            dtype=tf.float32, initializer='glorot_uniform',
        )
        self.W_h = self.add_weight(
            name='W_h', shape=(self.units, self.units),
            dtype=tf.float32, initializer='orthogonal',
        )
        self.b = self.add_weight(
            name='b', shape=(self.units,),
            dtype=tf.float32, initializer='zeros',
        )
        self.tau = self.add_weight(
            name='tau', shape=(self.units,),
            dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(1.0),
            constraint=tf.keras.constraints.NonNeg(),
        )
        self.built = True

    def call(self, inputs, states):
        if isinstance(inputs, (tuple, list)):
            inputs, elapsed_time = inputs
        else:
            elapsed_time = 1.0
        h = states[0]
        gate = tf.nn.tanh(inputs @ self.W_x + h @ self.W_h + self.b)
        h_new = h + (elapsed_time / (self.tau + self.epsilon)) * (-h + gate)
        return h_new, [h_new]
```

### src/neurons/lstm_cell.py

```python
import tensorflow as tf
from .base_cell import BaseCell


class LSTM_Cell(BaseCell):
    """LSTM baseline cell.

    Thin wrapper around tf.keras.layers.LSTMCell to conform to BaseCell.

    State: [h, c] — two tensors of shape (batch, units) each.
    Output: h (hidden state), shape (batch, units).

    Note: state_size overrides BaseCell's default (which returns a single
    integer) because LSTM requires two state tensors. output_size inherits
    from BaseCell (returns units) — the output is h only, not [h, c].
    """

    def __init__(self, units, **kwargs):
        super().__init__(units, **kwargs)

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
        self.built = True

    def call(self, inputs, states):
        if isinstance(inputs, (tuple, list)):
            inputs, _ = inputs   # discard elapsed_time (LSTM is discrete)
        output, new_states = self._lstm(inputs, states)
        return output, new_states
```

### src/neurons/__init__.py update

Add exports:
```python
from .ctrnn_cell import CTRNN_Cell
from .lstm_cell import LSTM_Cell
```

### src/models/rnn_model.py update

Add to `_CELL_REGISTRY`:
```python
_CELL_REGISTRY = {
    'lrc':    LRC_Cell,
    'lrc_ar': LRC_AR_Cell,
    'ctrnn':  CTRNN_Cell,
    'lstm':   LSTM_Cell,
}
```

Add imports at top:
```python
from src.neurons import LRC_Cell, LRC_AR_Cell, CTRNN_Cell, LSTM_Cell
```

## Tests

Tests are appended to the existing `tests/neurons/test_cells.py` (9 new functions).
Existing imports cover `tensorflow as tf`; add `CTRNN_Cell`, `LSTM_Cell`, and `make_model` imports.

### CT-RNN tests (4) — append to test_cells.py

```python
# --- CTRNN_Cell ---

def test_ctrnn_cell_is_subclass_of_basecell():
    from src.neurons import CTRNN_Cell
    from src.neurons.base_cell import BaseCell
    assert issubclass(CTRNN_Cell, BaseCell)


def test_ctrnn_forward_pass_shape():
    from src.neurons import CTRNN_Cell
    cell = CTRNN_Cell(units=8)
    x = tf.zeros([3, 5])
    state = [tf.zeros([3, 8])]
    output, new_states = cell(x, state)
    assert output.shape == (3, 8)
    assert new_states[0].shape == (3, 8)


def test_ctrnn_irregular_sampling():
    from src.neurons import CTRNN_Cell
    cell = CTRNN_Cell(units=8)
    x = tf.zeros([3, 5])
    state = [tf.zeros([3, 8])]
    output, _ = cell((x, 0.5), state)
    assert output.shape == (3, 8)


def test_ctrnn_make_model_and_gradient_flow():
    from src.models import make_model
    model = make_model('ctrnn', 'dense', 4)
    assert isinstance(model, tf.keras.Sequential)
    x = tf.zeros([2, 5, 3])
    assert model(x).shape == (2, 5, 4)
    model(x)  # ensure weights built
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(model(x))
    grads = tape.gradient(loss, model.trainable_variables)
    assert len(model.trainable_variables) > 0
    assert any(g is not None and tf.reduce_any(g != 0).numpy() for g in grads)
```

### LSTM tests (5) — append to test_cells.py

```python
# --- LSTM_Cell ---

def test_lstm_cell_is_subclass_of_basecell():
    from src.neurons import LSTM_Cell
    from src.neurons.base_cell import BaseCell
    assert issubclass(LSTM_Cell, BaseCell)


def test_lstm_state_size():
    from src.neurons import LSTM_Cell
    cell = LSTM_Cell(units=8)
    assert cell.state_size == [8, 8]


def test_lstm_forward_pass_shape():
    from src.neurons import LSTM_Cell
    cell = LSTM_Cell(units=8)
    x = tf.zeros([3, 5])
    states = [tf.zeros([3, 8]), tf.zeros([3, 8])]
    output, new_states = cell(x, states)
    assert output.shape == (3, 8)
    assert len(new_states) == 2
    assert new_states[0].shape == (3, 8)
    assert new_states[1].shape == (3, 8)


def test_lstm_irregular_sampling_ignored():
    """LSTM discards elapsed_time (discrete cell)."""
    from src.neurons import LSTM_Cell
    cell = LSTM_Cell(units=8)
    x = tf.zeros([3, 5])
    states = [tf.zeros([3, 8]), tf.zeros([3, 8])]
    output_reg, _ = cell(x, states)
    output_irr, _ = cell((x, 0.5), states)
    assert tf.reduce_all(output_reg == output_irr).numpy()


def test_lstm_make_model_and_gradient_flow():
    from src.models import make_model
    model = make_model('lstm', 'dense', 4)
    assert isinstance(model, tf.keras.Sequential)
    x = tf.zeros([2, 5, 3])
    assert model(x).shape == (2, 5, 4)
    model(x)
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(model(x))
    grads = tape.gradient(loss, model.trainable_variables)
    assert len(model.trainable_variables) > 0
    assert any(g is not None and tf.reduce_any(g != 0).numpy() for g in grads)
```

## Success Criterion

```bash
uv run pytest tests/neurons/test_cells.py -v
```

19 tests pass (10 existing + 9 new). Full suite 48/48.

```python
from src.models import make_model
make_model('ctrnn', 'dense', 16)   # no error
make_model('lstm', 'dense', 16)    # no error
```

## Out of Scope

- NCP wiring → Step 9
- STC cell → blocked (supervisor meeting pending)
- Irregular sampling for LSTM (elapsed_time is silently discarded — LSTM has no continuous-time analog)
- Custom weight initialisation ranges for CT-RNN (can be added as `**kwargs` later)
- Multi-step ODE unfolding for CT-RNN (single Euler step is sufficient for Phase 2)
