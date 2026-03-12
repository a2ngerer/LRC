# NCP Wiring Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `NCPWiring` (3-layer sparse RNN) and replace the single `make_model` factory with `make_dense_model` + `make_ncp_model`.

**Architecture:** `SparseLinear` (trainable weights with fixed binary mask) sits between three stacked RNN layers (inter → command → motor). The binary masks come from `AutoNCP` in the `keras-ncps` library. `make_dense_model` creates stacked same-size RNN layers; `make_ncp_model` creates the 3-layer NCP structure. `make_model` is removed; all callers are migrated.

**Tech Stack:** uv, TensorFlow 2.15, keras-ncps, pytest

**Spec:** `docs/superpowers/specs/2026-03-12-ncp-wiring-design.md`

---

## Chunk 1: Branch + Setup

### Task 1: Create feature branch

**Files:** (none — branch setup only)

- [ ] **Step 1: Create branch from main**

```bash
cd /Users/angeral/Repositories/master_thesis/code
git checkout main && git pull
git checkout -b phase2/step8-ncp-wiring
git branch --show-current
```

Expected: `phase2/step8-ncp-wiring`

---

### Task 2: Install keras-ncps + update BaseWiring

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/wirings/base_wiring.py`

- [ ] **Step 1: Add keras-ncps to pyproject.toml**

In `pyproject.toml`, add `"keras-ncps"` to the `dependencies` list:

```toml
dependencies = [
    "tensorflow>=2.15,<2.16",
    "numpy",
    "matplotlib",
    "scipy",
    "pyyaml",
    "tqdm",
    "pandas",
    "keras-ncps",
]
```

- [ ] **Step 2: Install the new dependency**

```bash
uv sync
```

Expected: no error. Then verify:

```bash
uv run python -c "from ncps.wirings import AutoNCP; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Verify AutoNCP adjacency matrix convention**

Run this snippet to understand the matrix layout before implementing NCPWiring:

```bash
uv run python -c "
from ncps.wirings import AutoNCP
import numpy as np
w = AutoNCP(18, 4, seed=42)  # 8 inter + 6 command + 4 motor = 18 total
A = w.adjacency_matrix
print('shape:', A.shape)       # must be (18, 18)
print('output_dim:', w.output_dim)  # must be 4
print('sparsity: {:.0%} zeros'.format(np.mean(A == 0)))
print('first row:', A[0, :])   # check if row=dst or row=src
"
```

Note the output — you'll need it for `NCPWiring`. If `A[dst, src]` (destination = row), use `.T` when slicing. If `A[src, dst]`, no transpose needed.

- [ ] **Step 4: Make `cell` optional in BaseWiring**

Current `src/wirings/base_wiring.py`:
```python
def __init__(self, cell):
    self.cell = cell
```

Replace with:
```python
def __init__(self, cell=None):
    self.cell = cell
```

- [ ] **Step 5: Run existing tests — verify no regression**

```bash
uv run pytest -v
```

Expected: 48/48 pass.

- [ ] **Step 6: Commit setup**

```bash
git add pyproject.toml uv.lock src/wirings/base_wiring.py
git commit -m "chore: add keras-ncps dependency + make BaseWiring.cell optional"
```

---

## Chunk 2: SparseLinear + NCPWiring

### Task 3: SparseLinear — tests + implementation

**Files:**
- Create: `tests/wirings/test_ncp_wiring.py`
- Create: `src/wirings/ncp.py` (partial — SparseLinear only)

- [ ] **Step 1: Create `tests/wirings/test_ncp_wiring.py` with 2 SparseLinear tests**

```python
import numpy as np
import tensorflow as tf
from src.wirings import SparseLinear


def test_sparse_linear_output_shape():
    mask = np.ones((4, 6), dtype=np.float32)
    layer = SparseLinear(units=6, mask=mask)
    x = tf.zeros([2, 5, 4])
    assert layer(x).shape == (2, 5, 6)


def test_sparse_linear_respects_mask():
    """Weights at mask=0 positions produce zero output."""
    mask = np.zeros((4, 6), dtype=np.float32)
    mask[:, :3] = 1.0   # only first 3 outputs connected
    layer = SparseLinear(units=6, mask=mask)
    x = tf.ones([1, 1, 4])
    out = layer(x).numpy()
    assert np.all(out[..., 3:] == 0.0)
```

- [ ] **Step 2: Run tests — verify they FAIL**

```bash
uv run pytest tests/wirings/test_ncp_wiring.py -v
```

Expected: `ImportError: cannot import name 'SparseLinear' from 'src.wirings'`

- [ ] **Step 3: Create `src/wirings/ncp.py` with SparseLinear**

```python
import numpy as np
import tensorflow as tf
from ncps.wirings import AutoNCP
from .base_wiring import BaseWiring


class SparseLinear(tf.keras.layers.Layer):
    """Dense layer with a fixed binary connectivity mask.

    The mask determines which connections exist. Weights at masked-off
    positions are zeroed each forward pass (W * mask), so gradients there
    are also zero — the sparsity is permanent throughout training.

    Args:
        units:  output dimension
        mask:   numpy bool/int array of shape (input_dim, units), 1 = connected
    """

    def __init__(self, units, mask, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self._mask_np = np.array(mask, dtype=np.float32)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='W', shape=(input_shape[-1], self.units),
            dtype=tf.float32, initializer='glorot_uniform',
        )
        self.mask = tf.constant(self._mask_np, dtype=tf.float32)
        self.built = True

    def call(self, x):
        return x @ (self.W * self.mask)
```

(Leave `NCPWiring` for the next step — only `SparseLinear` for now.)

- [ ] **Step 4: Export SparseLinear from `src/wirings/__init__.py`**

Current content:
```python
from .base_wiring import BaseWiring
from .dense import DenseWiring

__all__ = ["BaseWiring", "DenseWiring"]
```

Replace with:
```python
from .base_wiring import BaseWiring
from .dense import DenseWiring
from .ncp import SparseLinear

__all__ = ["BaseWiring", "DenseWiring", "SparseLinear"]
```

- [ ] **Step 5: Run SparseLinear tests — verify they PASS**

```bash
uv run pytest tests/wirings/test_ncp_wiring.py -v
```

Expected: 2/2 pass.

- [ ] **Step 6: Run full suite — check no regressions**

```bash
uv run pytest -v
```

Expected: 50/50 pass (48 existing + 2 new).

- [ ] **Step 7: Commit**

```bash
git add src/wirings/ncp.py src/wirings/__init__.py tests/wirings/test_ncp_wiring.py
git commit -m "feat(wirings): add SparseLinear with binary connectivity mask"
```

---

### Task 4: NCPWiring — tests + implementation

**Files:**
- Modify: `tests/wirings/test_ncp_wiring.py` (append 3 NCPWiring tests)
- Modify: `src/wirings/ncp.py` (append NCPWiring class)
- Modify: `src/wirings/__init__.py` (add NCPWiring export)

- [ ] **Step 1: Append 3 NCPWiring tests to `tests/wirings/test_ncp_wiring.py`**

Add these imports at the top of the file (after existing imports):
```python
from src.neurons import LRC_Cell, LSTM_Cell, CTRNN_Cell
from src.wirings import NCPWiring
```

Append at the end of the file:

```python
def test_ncp_wiring_build_model_returns_sequential():
    wiring = NCPWiring(LRC_Cell, inter_neurons=8, command_neurons=6, motor_neurons=4)
    model = wiring.build_model()
    assert isinstance(model, tf.keras.Sequential)


def test_ncp_wiring_output_shape_lstm():
    wiring = NCPWiring(LSTM_Cell, inter_neurons=8, command_neurons=6, motor_neurons=4)
    model = wiring.build_model()
    x = tf.zeros([2, 5, 3])
    assert model(x).shape == (2, 5, 4)


def test_ncp_wiring_output_shape_ctrnn():
    wiring = NCPWiring(CTRNN_Cell, inter_neurons=8, command_neurons=6, motor_neurons=4)
    model = wiring.build_model()
    x = tf.zeros([2, 5, 3])
    assert model(x).shape == (2, 5, 4)
```

- [ ] **Step 2: Run NCPWiring tests — verify they FAIL**

```bash
uv run pytest tests/wirings/test_ncp_wiring.py -k "ncp_wiring" -v
```

Expected: `ImportError: cannot import name 'NCPWiring' from 'src.wirings'`

- [ ] **Step 3: Append NCPWiring to `src/wirings/ncp.py`**

Add after the `SparseLinear` class:

```python
class NCPWiring(BaseWiring):
    """Three stacked RNN layers (inter → command → motor) with sparse
    inter-layer connections generated by AutoNCP.

    Connections within each layer are dense (standard RNN recurrent weights).
    Connections *between* layers are sparse: a SparseLinear layer with a
    binary mask derived from the AutoNCP adjacency matrix sits between each
    pair of RNN layers.

    Args:
        cell_cls:         BaseCell subclass (e.g. LRC_Cell, LSTM_Cell)
        inter_neurons:    number of inter neurons
        command_neurons:  number of command neurons
        motor_neurons:    number of motor neurons (= model output size)
        seed:             random seed for AutoNCP wiring generation (default 42)
        **cell_kwargs:    forwarded to each cell constructor
    """

    def __init__(self, cell_cls, inter_neurons, command_neurons, motor_neurons,
                 seed=42, **cell_kwargs):
        super().__init__(cell=None)
        self.cell_cls = cell_cls
        self.inter_neurons = inter_neurons
        self.command_neurons = command_neurons
        self.motor_neurons = motor_neurons
        self.seed = seed
        self.cell_kwargs = cell_kwargs

        # Generate sparse inter-layer masks via AutoNCP.
        # NOTE: AutoNCP(n_neurons, n_outputs) — verify adjacency_matrix convention
        # from the sanity check run in Task 2 Step 3 before adjusting slices/transpose.
        total = inter_neurons + command_neurons + motor_neurons
        wiring = AutoNCP(total, motor_neurons, seed=seed)
        A = wiring.adjacency_matrix  # numpy array, shape (total, total)
        i, c = inter_neurons, command_neurons
        # Assuming A[dst, src] = 1 (destination row convention).
        # If output of Step 3 in Task 2 shows A[src, dst], remove the .T calls.
        self._inter_to_command = A[i:i + c, :i].T.astype(np.float32)     # (i, c)
        self._command_to_motor = A[i + c:, i:i + c].T.astype(np.float32) # (c, m)

    def build_model(self) -> tf.keras.Sequential:
        return tf.keras.Sequential([
            tf.keras.layers.RNN(
                self.cell_cls(units=self.inter_neurons, **self.cell_kwargs),
                return_sequences=True,
            ),
            SparseLinear(self.command_neurons, self._inter_to_command),
            tf.keras.layers.RNN(
                self.cell_cls(units=self.command_neurons, **self.cell_kwargs),
                return_sequences=True,
            ),
            SparseLinear(self.motor_neurons, self._command_to_motor),
            tf.keras.layers.RNN(
                self.cell_cls(units=self.motor_neurons, **self.cell_kwargs),
                return_sequences=True,
            ),
        ])
```

- [ ] **Step 4: Export NCPWiring from `src/wirings/__init__.py`**

```python
from .base_wiring import BaseWiring
from .dense import DenseWiring
from .ncp import SparseLinear, NCPWiring

__all__ = ["BaseWiring", "DenseWiring", "SparseLinear", "NCPWiring"]
```

- [ ] **Step 5: Run NCPWiring tests — verify they PASS**

```bash
uv run pytest tests/wirings/test_ncp_wiring.py -k "ncp_wiring" -v
```

Expected: 3/3 pass. If `test_ncp_wiring_output_shape_*` fails with a shape mismatch, re-check the adjacency matrix slicing — see the note in Step 3.

- [ ] **Step 6: Run full suite — check no regressions**

```bash
uv run pytest -v
```

Expected: 53/53 pass (50 from previous + 3 new NCPWiring tests).

- [ ] **Step 7: Commit**

```bash
git add src/wirings/ncp.py src/wirings/__init__.py tests/wirings/test_ncp_wiring.py
git commit -m "feat(wirings): add NCPWiring with 3-layer sparse RNN topology"
```

---

## Chunk 3: make_dense_model + make_ncp_model

### Task 5: make_dense_model — tests + implementation

**Files:**
- Modify: `tests/models/test_make_model.py` (append 3 new tests)
- Modify: `src/models/rnn_model.py` (add make_dense_model)
- Modify: `src/models/__init__.py` (export make_dense_model)

- [ ] **Step 1: Append 3 failing make_dense_model tests to `tests/models/test_make_model.py`**

Add after the last existing test:

```python
# --- make_dense_model ---

def test_make_dense_model_single_layer():
    from src.models import make_dense_model
    model = make_dense_model('lrc', units=4)
    assert isinstance(model, tf.keras.Sequential)
    assert model(tf.zeros([2, 5, 3])).shape == (2, 5, 4)


def test_make_dense_model_multi_layer():
    from src.models import make_dense_model
    model = make_dense_model('ctrnn', units=4, num_layers=3)
    x = tf.zeros([2, 5, 3])
    assert model(x).shape == (2, 5, 4)


def test_make_dense_model_output_neurons():
    from src.models import make_dense_model
    model = make_dense_model('lstm', units=8, num_layers=2, output_neurons=2)
    x = tf.zeros([2, 5, 3])
    assert model(x).shape == (2, 5, 2)
```

- [ ] **Step 2: Run new tests — verify they FAIL**

```bash
uv run pytest tests/models/test_make_model.py -k "make_dense" -v
```

Expected: `ImportError: cannot import name 'make_dense_model' from 'src.models'`

- [ ] **Step 3: Add `make_dense_model` to `src/models/rnn_model.py`**

Add after the `_CELL_REGISTRY` dict (keep `make_model` untouched for now):

```python
def make_dense_model(neuron_type, units, num_layers=1, output_neurons=None, **cell_kwargs):
    """Build a stacked Dense-wired RNN model.

    Args:
        neuron_type:    str key ('lrc', 'lrc_ar', 'ctrnn', 'lstm') or BaseCell subclass
        units:          neurons per RNN layer (all layers the same size)
        num_layers:     number of stacked RNN layers (default 1)
        output_neurons: if given, appends a Dense(output_neurons) projection layer
        **cell_kwargs:  forwarded to each cell constructor

    Returns:
        tf.keras.Sequential, always return_sequences=True on every RNN layer
    """
    cell_cls = _CELL_REGISTRY[neuron_type] if isinstance(neuron_type, str) else neuron_type
    layers = []
    for _ in range(num_layers):
        cell = cell_cls(units=units, **cell_kwargs)
        layers.append(tf.keras.layers.RNN(cell, return_sequences=True))
    if output_neurons is not None:
        layers.append(tf.keras.layers.Dense(output_neurons))
    return tf.keras.Sequential(layers)
```

- [ ] **Step 4: Export `make_dense_model` from `src/models/__init__.py`**

```python
from .rnn_model import make_model, make_dense_model

__all__ = ["make_model", "make_dense_model"]
```

(Keep `make_model` for now — it will be removed in Task 7.)

- [ ] **Step 5: Run new tests — verify they PASS**

```bash
uv run pytest tests/models/test_make_model.py -k "make_dense" -v
```

Expected: 3/3 pass.

- [ ] **Step 6: Run full suite — check no regressions**

```bash
uv run pytest -v
```

Expected: 56/56 pass (53 + 3 new).

- [ ] **Step 7: Commit**

```bash
git add src/models/rnn_model.py src/models/__init__.py tests/models/test_make_model.py
git commit -m "feat(models): add make_dense_model factory with num_layers support"
```

---

### Task 6: make_ncp_model — tests + implementation

**Files:**
- Modify: `tests/models/test_make_model.py` (append 3 new tests)
- Modify: `src/models/rnn_model.py` (add make_ncp_model)
- Modify: `src/models/__init__.py` (export make_ncp_model)

- [ ] **Step 1: Append 3 failing make_ncp_model tests to `tests/models/test_make_model.py`**

Add after the last `make_dense_model` test:

```python
# --- make_ncp_model ---

def test_make_ncp_model_returns_sequential():
    from src.models import make_ncp_model
    model = make_ncp_model('lrc', inter_neurons=8, command_neurons=6, motor_neurons=4)
    assert isinstance(model, tf.keras.Sequential)


def test_make_ncp_model_output_shape():
    from src.models import make_ncp_model
    model = make_ncp_model('lstm', inter_neurons=8, command_neurons=6, motor_neurons=4)
    x = tf.zeros([2, 5, 3])
    assert model(x).shape == (2, 5, 4)


def test_make_ncp_model_gradient_flow():
    from src.models import make_ncp_model
    model = make_ncp_model('ctrnn', inter_neurons=8, command_neurons=6, motor_neurons=4)
    x = tf.zeros([2, 5, 3])
    model(x)  # build weights
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(model(x))
    grads = tape.gradient(loss, model.trainable_variables)
    assert len(model.trainable_variables) > 0
    assert any(g is not None and tf.reduce_any(g != 0).numpy() for g in grads)
```

- [ ] **Step 2: Run new tests — verify they FAIL**

```bash
uv run pytest tests/models/test_make_model.py -k "make_ncp" -v
```

Expected: `ImportError: cannot import name 'make_ncp_model' from 'src.models'`

- [ ] **Step 3: Add `make_ncp_model` to `src/models/rnn_model.py`**

Add the import for `NCPWiring` at the top:
```python
from src.wirings import DenseWiring, NCPWiring
```

Then add after `make_dense_model`:

```python
def make_ncp_model(neuron_type, inter_neurons, command_neurons, motor_neurons,
                   seed=42, **cell_kwargs):
    """Build a three-layer NCP-wired RNN model.

    Layers: inter → (sparse) → command → (sparse) → motor
    Output shape: (batch, timesteps, motor_neurons)

    Args:
        neuron_type:      str key ('lrc', 'lrc_ar', 'ctrnn', 'lstm') or BaseCell subclass
        inter_neurons:    neurons in inter layer
        command_neurons:  neurons in command layer
        motor_neurons:    neurons in motor layer (= output size)
        seed:             AutoNCP seed (default 42)
        **cell_kwargs:    forwarded to each cell constructor

    Returns:
        tf.keras.Sequential
    """
    cell_cls = _CELL_REGISTRY[neuron_type] if isinstance(neuron_type, str) else neuron_type
    wiring = NCPWiring(cell_cls, inter_neurons, command_neurons, motor_neurons,
                       seed=seed, **cell_kwargs)
    return wiring.build_model()
```

- [ ] **Step 4: Export `make_ncp_model` from `src/models/__init__.py`**

```python
from .rnn_model import make_model, make_dense_model, make_ncp_model

__all__ = ["make_model", "make_dense_model", "make_ncp_model"]
```

- [ ] **Step 5: Run new tests — verify they PASS**

```bash
uv run pytest tests/models/test_make_model.py -k "make_ncp" -v
```

Expected: 3/3 pass.

- [ ] **Step 6: Run full suite — check no regressions**

```bash
uv run pytest -v
```

Expected: 59/59 pass (56 + 3 new).

- [ ] **Step 7: Commit**

```bash
git add src/models/rnn_model.py src/models/__init__.py tests/models/test_make_model.py
git commit -m "feat(models): add make_ncp_model factory"
```

---

## Chunk 4: Migration + Merge

### Task 7: Migrate make_model → make_dense_model + remove make_model

**Files:**
- Modify: `tests/models/test_make_model.py` (update 7 existing tests)
- Modify: `tests/neurons/test_cells.py` (update 2 existing tests)
- Modify: `src/models/rnn_model.py` (remove make_model + _WIRING_REGISTRY)
- Modify: `src/models/__init__.py` (remove make_model export)

- [ ] **Step 1: Update 7 existing tests in `tests/models/test_make_model.py`**

At the top of the file, change:
```python
from src.models import make_model
```
to:
```python
from src.models import make_dense_model, make_ncp_model
```

Also remove the unused import: `from src.wirings import DenseWiring`

Then update each test:

| Old call | New call |
|----------|----------|
| `make_model('lrc', 'dense', 32)` | `make_dense_model('lrc', units=32)` |
| `make_model('lrc', 'dense', 4)` | `make_dense_model('lrc', units=4)` |
| `make_model(LRC_Cell, DenseWiring, 8)` | `make_dense_model(LRC_Cell, units=8)` |
| `make_model('unknown_cell', 'dense', 4)` | `make_dense_model('unknown_cell', units=4)` |
| `make_model('lrc_ar', 'dense', 4)` | `make_dense_model('lrc_ar', units=4)` |
| `make_model('lrc', 'dense', 64)` | `make_dense_model('lrc', units=64)` |
| `make_model('lrc', 'dense', 4, ode_unfolds=5)` | `make_dense_model('lrc', units=4, ode_unfolds=5)` |

Note: `test_make_model_accepts_class_arguments` passes `DenseWiring` as second arg. With `make_dense_model`, there is no wiring argument — just pass the cell class. Update the test body:
```python
def test_make_dense_model_accepts_cell_class():
    model = make_dense_model(LRC_Cell, units=8)
    assert isinstance(model, tf.keras.Sequential)
```

Also rename `test_make_model_unknown_neuron_raises` to `test_make_dense_model_unknown_neuron_raises` for clarity.

- [ ] **Step 2: Update 2 tests in `tests/neurons/test_cells.py`**

Find `test_ctrnn_make_model_and_gradient_flow` and `test_lstm_make_model_and_gradient_flow`.

In each, change:
```python
from src.models import make_model
model = make_model('ctrnn', 'dense', 4)
```
to:
```python
from src.models import make_dense_model
model = make_dense_model('ctrnn', units=4)
```

Same for the lstm variant: `make_dense_model('lstm', units=4)`.

- [ ] **Step 3: Run full suite — verify all pass with updated imports**

```bash
uv run pytest -v
```

Expected: 59/59 pass (the 9 migrated tests still pass, 11 new ones still pass).

- [ ] **Step 4: Remove `make_model` from `src/models/rnn_model.py`**

Delete the entire `make_model` function and the `_WIRING_REGISTRY` dict. Also remove the `from src.wirings import DenseWiring` line (NCPWiring import stays).

The file should now contain only:
- imports
- `_CELL_REGISTRY`
- `make_dense_model`
- `make_ncp_model`

- [ ] **Step 5: Update `src/models/__init__.py`**

```python
from .rnn_model import make_dense_model, make_ncp_model

__all__ = ["make_dense_model", "make_ncp_model"]
```

- [ ] **Step 6: Run full suite — verify still 59/59**

```bash
uv run pytest -v
```

Expected: 59/59. If anything fails, it means a caller of `make_model` was missed.

- [ ] **Step 7: Commit**

```bash
git add src/models/rnn_model.py src/models/__init__.py tests/models/test_make_model.py tests/neurons/test_cells.py
git commit -m "refactor(models): replace make_model with make_dense_model + make_ncp_model"
```

---

### Task 8: Migrate ODEFuncModel + verify_neural_ode.py

**Files:**
- Modify: `src/tasks/neural_ode/ode_model.py`
- Modify: `experiments/verify_neural_ode.py`

- [ ] **Step 1: Update `src/tasks/neural_ode/ode_model.py`**

Current signature and import:
```python
from src.models import make_model

class ODEFuncModel(tf.keras.Model):
    def __init__(self, neuron_type, wiring_type, units, features, **cell_kwargs):
        ...
        self.rnn = make_model(neuron_type, wiring_type, units, **cell_kwargs)
```

Replace with:
```python
from src.models import make_dense_model

class ODEFuncModel(tf.keras.Model):
    def __init__(self, neuron_type, units, features, **cell_kwargs):
        ...
        self.rnn = make_dense_model(neuron_type, units=units, **cell_kwargs)
```

Update the docstring to remove `wiring_type` from the Args section.

- [ ] **Step 2: Update `experiments/verify_neural_ode.py`**

Find the `ODEFuncModel` instantiation in `run_verification()`:
```python
model = ODEFuncModel(cfg['neuron'], cfg['wiring'], cfg['units'], features=2)
```

Replace with:
```python
model = ODEFuncModel(cfg['neuron'], cfg['units'], features=2)
```

Also remove `'wiring': 'dense'` from `DEFAULT_CONFIG` (no longer used):
```python
DEFAULT_CONFIG = {
    'neuron': 'lrc_ar',
    'units': 16,
    'niters': 2000,
    'batch_size': 16,
    'batch_time': 16,
    'lr': 1e-3,
}
```

- [ ] **Step 3: Run full suite — verify 59/59**

```bash
uv run pytest -v
```

Expected: 59/59. The `tests/experiments/test_verify_script.py` tests may call `run_verification` — they should still pass because the function signature doesn't change (only internal `ODEFuncModel` call changes).

Troubleshooting: if `test_run_verification_output_shape` fails, check that `ODEFuncModel` is being called correctly and that `cfg['wiring']` removal doesn't cause a KeyError anywhere.

- [ ] **Step 4: Commit**

```bash
git add src/tasks/neural_ode/ode_model.py experiments/verify_neural_ode.py
git commit -m "refactor: migrate ODEFuncModel + verify_neural_ode to make_dense_model"
```

---

### Task 9: Merge + Obsidian docs

- [ ] **Step 1: Merge branch to main**

```bash
git checkout main
git merge --no-ff phase2/step8-ncp-wiring -m "merge: phase2/step8-ncp-wiring — NCP wiring + make_dense_model/make_ncp_model"
```

- [ ] **Step 2: Push branch and main**

```bash
git push origin phase2/step8-ncp-wiring
git push origin main
```

- [ ] **Step 3: Update `Thesis/work-documentation.md` via Obsidian MCP**

Read the current file and prepend after `## Log` heading:

```markdown
### 2026-03-12 (Phase 2 / Step 8)
**Branch**: `phase2/step8-ncp-wiring`
**Status**: Merged to main ✅

**What was done**:
- Created `src/wirings/ncp.py`: `SparseLinear` (fixed binary mask) + `NCPWiring` (3-layer inter→command→motor)
- Added `keras-ncps` dependency (used only for `AutoNCP` adjacency matrix generation)
- Replaced `make_model` with `make_dense_model` (stacked RNN, configurable num_layers) + `make_ncp_model` (NCP topology)
- Migrated `ODEFuncModel` and `verify_neural_ode.py` to `make_dense_model`
- 11 new tests — total suite: 59/59 passing

**Files changed**:
- `src/wirings/ncp.py` (new)
- `src/wirings/base_wiring.py` (cell optional)
- `src/wirings/__init__.py` (NCPWiring, SparseLinear exports)
- `src/models/rnn_model.py` (make_dense_model + make_ncp_model; make_model removed)
- `src/models/__init__.py` (updated exports)
- `src/tasks/neural_ode/ode_model.py` (migrated)
- `experiments/verify_neural_ode.py` (migrated)
- `tests/wirings/test_ncp_wiring.py` (new)
- `tests/models/test_make_model.py` (migrated + extended)
- `tests/neurons/test_cells.py` (2 tests migrated)
- `pyproject.toml` (keras-ncps added)

**Notes**:
- NCP wiring: 3 stacked RNN layers, each using the chosen cell type (LRC/LSTM/CTRNN)
- Sparse inter-layer connections from AutoNCP adjacency matrix, applied as frozen binary mask in SparseLinear
- Input→Inter connections remain dense; Inter→Command and Command→Motor are sparse
- Next: `phase2/step9-smoke-test` (all cell×wiring combinations)

---
```

- [ ] **Step 4: Update `Meta/claude-project-status.md` via Obsidian MCP**

Mark `phase2/step8-ncp-wiring` as `[x]` and add `phase2/step9-smoke-test` as next pending step.
