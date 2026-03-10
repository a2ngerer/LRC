# Design: Model Factory — make_model (Phase 1 / Step 4)

**Date:** 2026-03-10
**Branch:** `phase1/step4-model-factory`
**Status:** Approved

## Goal

Implement `make_model(neuron_type, wiring_type, units, **kwargs)` in `src/models/` that returns a bare `tf.keras.Sequential` RNN model (no input/output projection layers) for any registered neuron × wiring combination.

## Scope

**In scope (Step 4):**
- `src/wirings/base_wiring.py` — abstract `BaseWiring`
- `src/wirings/dense.py` — `DenseWiring`
- `src/wirings/__init__.py`
- `src/models/rnn_model.py` — `make_model` + registries
- `src/models/__init__.py`
- `tests/models/test_make_model.py` — 6 smoke tests

**Out of scope (deferred):**
- LSTM, CT-RNN, STC cell implementations → Phase 2
- NCP wiring → Phase 2 / step3-ncp-wiring
- Input/output projection layers → added at task level in Step 5
- `return_sequences=False` mode (not needed until classification tasks)

## Architecture

### File Layout

```
src/
├── wirings/
│   ├── base_wiring.py    # abstract BaseWiring(abc.ABC)
│   ├── dense.py          # DenseWiring(BaseWiring)
│   └── __init__.py       # exports DenseWiring, BaseWiring
└── models/
    ├── rnn_model.py      # make_model + _CELL_REGISTRY + _WIRING_REGISTRY
    └── __init__.py       # exports make_model
```

### Public API

```python
def make_model(
    neuron_type: str | type,
    wiring_type: str | type,
    units: int,
    **kwargs
) -> tf.keras.Sequential:
```

- `neuron_type`: string key (`'lrc'`, `'lrc_ar'`) **or** a `BaseCell` subclass directly
- `wiring_type`: string key (`'dense'`) **or** a `BaseWiring` subclass directly
- `units`: number of RNN units, forwarded to cell constructor
- `**kwargs`: forwarded to cell constructor (e.g. `ode_unfolds`, `elastance_type`)
- Returns: `tf.keras.Sequential` wrapping the wiring's RNN layer

Unknown string key → raises `KeyError` (no silent fallback).

### Registries

```python
_CELL_REGISTRY = {
    'lrc':    LRC_Cell,
    'lrc_ar': LRC_AR_Cell,
}

_WIRING_REGISTRY = {
    'dense': DenseWiring,
}
```

Adding new cells (CT-RNN, STC, LSTM in Phase 2) = one line in `_CELL_REGISTRY`.

### Wiring Abstraction

```python
# src/wirings/base_wiring.py
class BaseWiring(abc.ABC):
    def __init__(self, cell):
        self.cell = cell

    @abc.abstractmethod
    def build_model(self) -> tf.keras.Sequential:
        pass
```

```python
# src/wirings/dense.py
class DenseWiring(BaseWiring):
    """Fully-connected wiring: standard tf.keras.layers.RNN with no connectivity mask."""

    def __init__(self, cell, return_sequences=True):
        super().__init__(cell)
        self._return_sequences = return_sequences

    def build_model(self) -> tf.keras.Sequential:
        return tf.keras.Sequential([
            tf.keras.layers.RNN(self.cell, return_sequences=self._return_sequences)
        ])
```

`NCPWiring` in Phase 2 will subclass `BaseWiring` and call `keras_ncps.wirings.AutoNCP` inside `build_model()` — no changes to `make_model` required.

## Verification

Tests in `tests/models/test_make_model.py`:

| # | Test | Expected |
|---|------|----------|
| 1 | `make_model('lrc', 'dense', 32)` | `isinstance(model, tf.keras.Sequential)` |
| 2 | Forward pass `(batch=2, timesteps=5, features=3)` through `make_model('lrc', 'dense', 4)` | output shape `(2, 5, 4)` |
| 3 | `make_model(LRC_Cell, DenseWiring, 8)` (class args) | `isinstance(model, tf.keras.Sequential)` |
| 4 | `make_model('unknown', 'dense', 4)` | raises `KeyError` |
| 5 | `make_model('lrc_ar', 'dense', 4)` forward pass `(1, 3, 4)` | output shape `(1, 3, 4)` |
| 6 | `make_model('lrc', 'dense', 64).summary()` | no exception |

All tests pass via `uv run pytest tests/models/test_make_model.py -v`.
