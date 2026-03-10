# Model Factory Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `make_model(neuron_type, wiring_type, units, **kwargs)` factory in `src/models/` and a wiring abstraction in `src/wirings/`.

**Architecture:** Registry-based factory with string-or-class dispatch. `BaseWiring` is an abstract class with `build_model()`; `DenseWiring` wraps any cell in `tf.keras.layers.RNN(return_sequences=True)`. All `**kwargs` are forwarded to the cell constructor. No projection layers — bare RNN core only.

**Tech Stack:** uv, TensorFlow 2.15, pytest

**Spec:** `docs/superpowers/specs/2026-03-10-model-factory-design.md`

---

## Chunk 1: Branch + Wiring Package

### Task 1: Create feature branch

**Files:**
- (none — branch setup only)

- [ ] **Step 1: Create feature branch from main**

```bash
git checkout main && git pull
git checkout -b phase1/step4-model-factory
```

Expected: now on branch `phase1/step4-model-factory`

---

### Task 2: Wiring package — tests + implementation

**Files:**
- Create: `src/wirings/__init__.py`
- Create: `src/wirings/base_wiring.py`
- Create: `src/wirings/dense.py`
- Create: `tests/wirings/__init__.py`
- Create: `tests/wirings/test_dense_wiring.py`

- [ ] **Step 1: Create test directory and test file**

Create `tests/wirings/__init__.py` (empty file).

Create `tests/wirings/test_dense_wiring.py` with this exact content:

```python
import pytest
import tensorflow as tf
from src.wirings import DenseWiring, BaseWiring
from src.neurons import LRC_Cell


def test_base_wiring_is_abstract():
    """BaseWiring cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseWiring(cell=None)


def test_dense_wiring_is_subclass():
    assert issubclass(DenseWiring, BaseWiring)


def test_dense_wiring_build_model_returns_sequential():
    cell = LRC_Cell(units=4)
    wiring = DenseWiring(cell)
    model = wiring.build_model()
    assert isinstance(model, tf.keras.Sequential)


def test_dense_wiring_return_sequences_true():
    """DenseWiring always produces output for every timestep."""
    cell = LRC_Cell(units=4)
    wiring = DenseWiring(cell)
    model = wiring.build_model()
    x = tf.zeros([2, 5, 3])  # (batch=2, timesteps=5, features=3)
    y = model(x)
    assert y.shape == (2, 5, 4)  # all timesteps returned, output_size = units = 4
```

- [ ] **Step 2: Run tests — verify they FAIL**

```bash
uv run pytest tests/wirings/test_dense_wiring.py -v
```

Expected: `ImportError` — `src.wirings` does not exist yet. This confirms the test setup is correct.

- [ ] **Step 3: Create `src/wirings/base_wiring.py`**

```python
import abc


class BaseWiring(abc.ABC):
    """Abstract base for wiring strategies.

    A wiring defines how a cell is wrapped into a Keras model.
    Subclass this and implement build_model().

    Phase 2 will add NCPWiring here — no changes to make_model required.
    """

    def __init__(self, cell):
        self.cell = cell

    @abc.abstractmethod
    def build_model(self) -> "tf.keras.Sequential":
        """Wrap self.cell and return a tf.keras.Sequential."""
        pass
```

- [ ] **Step 4: Create `src/wirings/dense.py`**

```python
import tensorflow as tf
from .base_wiring import BaseWiring


class DenseWiring(BaseWiring):
    """Fully-connected wiring: standard tf.keras.layers.RNN.

    All neurons see all inputs — this is the default RNN behaviour.
    return_sequences is hardcoded to True (Phase 1: sequence tasks only).
    """

    def __init__(self, cell):
        super().__init__(cell)

    def build_model(self) -> tf.keras.Sequential:
        return tf.keras.Sequential([
            tf.keras.layers.RNN(self.cell, return_sequences=True)
        ])
```

- [ ] **Step 5: Create `src/wirings/__init__.py`**

```python
from .base_wiring import BaseWiring
from .dense import DenseWiring

__all__ = ["BaseWiring", "DenseWiring"]
```

- [ ] **Step 6: Run tests — verify they PASS**

```bash
uv run pytest tests/wirings/test_dense_wiring.py -v
```

Expected: all 4 tests PASS.

Troubleshooting:
- `test_base_wiring_is_abstract` fails (did not raise `TypeError`): `BaseWiring` must both inherit from `abc.ABC` AND have `build_model` decorated with `@abc.abstractmethod`. Both are required for Python to enforce abstract instantiation.
- Shape mismatch in `test_dense_wiring_return_sequences_true`: verify `LRC_Cell(units=4)` has `output_size == 4`.

- [ ] **Step 7: Commit**

```bash
git add src/wirings/ tests/wirings/
git commit -m "feat(wirings): add BaseWiring + DenseWiring with tests"
```

---

## Chunk 2: Model Factory

### Task 3: make_model — tests + implementation

**Files:**
- Create: `src/models/__init__.py`
- Create: `src/models/rnn_model.py`
- Create: `tests/models/__init__.py`
- Create: `tests/models/test_make_model.py`

- [ ] **Step 1: Create test directory and test file**

Create `tests/models/__init__.py` (empty file).

Create `tests/models/test_make_model.py` with this exact content:

```python
import pytest
import tensorflow as tf
from src.models import make_model
from src.neurons import LRC_Cell
from src.wirings import DenseWiring


def test_make_model_returns_sequential():
    model = make_model('lrc', 'dense', 32)
    assert isinstance(model, tf.keras.Sequential)


def test_make_model_forward_pass():
    """LRC cell accepts any input feature dimension."""
    model = make_model('lrc', 'dense', 4)
    x = tf.zeros([2, 5, 3])  # (batch=2, timesteps=5, features=3)
    y = model(x)
    assert y.shape == (2, 5, 4)


def test_make_model_accepts_class_arguments():
    """make_model also accepts cell/wiring classes directly, not just string keys."""
    model = make_model(LRC_Cell, DenseWiring, 8)
    assert isinstance(model, tf.keras.Sequential)


def test_make_model_unknown_neuron_raises():
    with pytest.raises(KeyError):
        make_model('unknown_cell', 'dense', 4)


def test_make_model_lrc_ar_forward_pass():
    """LRC_AR cell: input features must equal units (input acts as autoregressive state)."""
    model = make_model('lrc_ar', 'dense', 4)
    x = tf.zeros([1, 3, 4])  # features=4 == units=4
    y = model(x)
    assert y.shape == (1, 3, 4)


def test_make_model_summary_runs():
    model = make_model('lrc', 'dense', 64)
    model.summary()  # must not raise


def test_make_model_kwargs_forwarding():
    """Cell constructor kwargs are forwarded through make_model."""
    model = make_model('lrc', 'dense', 4, ode_unfolds=5)
    assert isinstance(model, tf.keras.Sequential)
```

- [ ] **Step 2: Run tests — verify they FAIL**

```bash
uv run pytest tests/models/test_make_model.py -v
```

Expected: `ImportError` — `src.models` does not exist yet.

- [ ] **Step 3: Create `src/models/rnn_model.py`**

```python
import tensorflow as tf
from src.neurons import LRC_Cell, LRC_AR_Cell
from src.wirings import DenseWiring

_CELL_REGISTRY = {
    'lrc':    LRC_Cell,
    'lrc_ar': LRC_AR_Cell,
}

_WIRING_REGISTRY = {
    'dense': DenseWiring,
}


def make_model(neuron_type, wiring_type, units, **kwargs):
    """Build a bare RNN Sequential model for the given neuron + wiring combination.

    Args:
        neuron_type: str key ('lrc', 'lrc_ar') or a BaseCell subclass
        wiring_type: str key ('dense') or a BaseWiring subclass
        units:       number of RNN units (forwarded to cell constructor)
        **kwargs:    additional keyword args forwarded to the cell constructor
                     (e.g. ode_unfolds=5, elastance_type='symmetric')

    Returns:
        tf.keras.Sequential wrapping the cell in the specified wiring.
        Always return_sequences=True — all timesteps are returned.

    Raises:
        KeyError: if neuron_type or wiring_type is an unknown string key.
    """
    cell_cls = _CELL_REGISTRY[neuron_type] if isinstance(neuron_type, str) else neuron_type
    cell = cell_cls(units=units, **kwargs)

    wiring_cls = _WIRING_REGISTRY[wiring_type] if isinstance(wiring_type, str) else wiring_type
    wiring = wiring_cls(cell)

    return wiring.build_model()
```

- [ ] **Step 4: Create `src/models/__init__.py`**

```python
from .rnn_model import make_model

__all__ = ["make_model"]
```

- [ ] **Step 5: Run model tests — verify they PASS**

```bash
uv run pytest tests/models/test_make_model.py -v
```

Expected: all 7 tests PASS.

Troubleshooting:
- `test_make_model_lrc_ar_forward_pass` fails with shape error: the LRC_AR cell is autoregressive — input feature dim must equal `units`. The test uses `x = tf.zeros([1, 3, 4])` with `units=4`, which satisfies this. If it fails, check the cell is instantiated as `LRC_AR_Cell(units=4)`.
- `test_make_model_unknown_neuron_raises` doesn't raise `KeyError`: verify `_CELL_REGISTRY[neuron_type]` is called before the `isinstance` check (i.e., string path hits the dict directly).

- [ ] **Step 6: Run full test suite — check for regressions**

```bash
uv run pytest -v
```

Expected: all tests PASS (neurons + wirings + models).

- [ ] **Step 7: Commit**

```bash
git add src/models/ tests/models/
git commit -m "feat(models): add make_model factory with cell + wiring registries"
```

---

## Chunk 3: Merge + Work Log

### Task 4: Merge to main and update documentation

- [ ] **Step 1: Merge branch to main**

```bash
git checkout main
git merge --no-ff phase1/step4-model-factory -m "merge: phase1/step4-model-factory — make_model factory with DenseWiring"
```

- [ ] **Step 2: Push branch and main to origin**

```bash
git push origin phase1/step4-model-factory
git push origin main
```

- [ ] **Step 3: Update Obsidian work log**

Add an entry at the **top** of `Thesis/work-documentation.md` (via Obsidian MCP). Use `wholeFile/overwrite` after reading the current content and prepending:

```markdown
## 2026-03-10 — Phase 1 / Step 4: Model Factory

**Branch:** `phase1/step4-model-factory` (merged ✅)

### What was done
- Created `src/wirings/` package: `BaseWiring` (abstract, abc.ABC) + `DenseWiring`
- Created `src/models/` package: `make_model(neuron_type, wiring_type, units, **kwargs)`
- String registry: `{'lrc': LRC_Cell, 'lrc_ar': LRC_AR_Cell}` / `{'dense': DenseWiring}`
- Accepts both string keys and class references
- `return_sequences=True` hardcoded (Phase 1: sequence tasks only)
- 11 new tests (4 wiring + 7 factory)

### Files changed
- `src/wirings/base_wiring.py` (new)
- `src/wirings/dense.py` (new)
- `src/wirings/__init__.py` (new)
- `src/models/rnn_model.py` (new)
- `src/models/__init__.py` (new)
- `tests/wirings/test_dense_wiring.py` (new)
- `tests/models/test_make_model.py` (new)

### Next
`phase1/step5-port-neural-ode-tasks` — port the 6 Neural ODE data generators + training loop

---
```

- [ ] **Step 4: Update `Meta/claude-project-status.md`**

Mark `phase1/step4-model-factory` as done (`[x]`) and add `phase1/step5-port-neural-ode-tasks` as next step.
