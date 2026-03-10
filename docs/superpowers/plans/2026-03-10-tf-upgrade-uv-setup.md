# TF Upgrade + uv Setup Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the conda/TF-2.4.1 environment with a `uv`-managed `pyproject.toml` using TF 2.15, and verify all neuron cells import and instantiate correctly.

**Architecture:** Create `pyproject.toml` at repo root as the single source of truth for dependencies. Write pytest smoke tests for `src/neurons/` that serve as the migration acceptance criterion. Keep `environment.yml` as a deprecated legacy reference.

**Tech Stack:** uv 0.9+, TensorFlow 2.15, tensorflow-metal (Mac optional extra), pytest

**Spec:** `docs/superpowers/specs/2026-03-10-tf-upgrade-uv-setup-design.md`

---

## Chunk 1: Project Setup + pyproject.toml

### Task 1: Create branch and pyproject.toml

**Files:**
- Create: `pyproject.toml`
- Modify: `environment.yml` (deprecation comment)

- [ ] **Step 1: Create the feature branch**

```bash
git checkout main
git checkout -b phase1/step3-tf-uv-setup
```

- [ ] **Step 2: Create `pyproject.toml`**

Create `/path/to/code/pyproject.toml` with this exact content:

```toml
[project]
name = "lrc-benchmark"
version = "0.1.0"
description = "Benchmark: 4 neuron types x 2 wirings on Neural ODE + control tasks"
requires-python = ">=3.11, <3.13"
dependencies = [
    "tensorflow>=2.15,<2.16",
    "numpy",
    "matplotlib",
    "scipy",
    "pyyaml",
    "tqdm",
    "pandas",
]

[project.optional-dependencies]
metal = ["tensorflow-metal"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Add deprecation comment to `environment.yml`**

Prepend this comment block at the very top of `environment.yml`:

```yaml
# DEPRECATED: This conda environment (TF 2.4.1, Python 3.9) is kept as a
# historical reference only. The active environment is managed via uv.
# Setup: uv sync --extra metal (Mac) or uv sync (Linux)
# See pyproject.toml for current dependencies.
```

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml environment.yml
git commit -m "build: add pyproject.toml with TF 2.15, deprecate environment.yml"
```

---

## Chunk 2: Tests + Dependency Installation

### Task 2: Write smoke tests for neuron cells

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/neurons/__init__.py`
- Create: `tests/neurons/test_cells.py`

- [ ] **Step 1: Create test package structure**

```bash
mkdir -p tests/neurons
touch tests/__init__.py tests/neurons/__init__.py
```

- [ ] **Step 2: Write the tests**

Create `tests/neurons/test_cells.py`:

```python
import pytest
import tensorflow as tf
from src.neurons import BaseCell, LRC_Cell, LRC_AR_Cell


# --- BaseCell ---

def test_basecell_is_abstract():
    """BaseCell cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseCell(units=4)


def test_basecell_is_abstract_rnn_cell():
    assert issubclass(BaseCell, tf.keras.layers.AbstractRNNCell)


# --- LRC_Cell ---

def test_lrc_cell_is_subclass_of_basecell():
    assert issubclass(LRC_Cell, BaseCell)


def test_lrc_cell_state_size():
    cell = LRC_Cell(units=32)
    assert cell.state_size == 32


def test_lrc_cell_output_size():
    cell = LRC_Cell(units=16)
    assert cell.output_size == 16


def test_lrc_cell_units_stored():
    cell = LRC_Cell(units=8)
    assert cell.units == 8


def test_lrc_cell_forward_pass():
    """Full forward pass through one time step."""
    units = 4
    batch = 2
    input_dim = 3
    cell = LRC_Cell(units=units)
    inputs = tf.zeros([batch, input_dim])
    state = [tf.zeros([batch, units])]
    output, new_state = cell(inputs, state)
    assert output.shape == (batch, units)
    assert new_state[0].shape == (batch, units)


# --- LRC_AR_Cell ---

def test_lrc_ar_cell_is_subclass_of_basecell():
    assert issubclass(LRC_AR_Cell, BaseCell)


def test_lrc_ar_cell_state_size():
    cell = LRC_AR_Cell(units=32)
    assert cell.state_size == 32


def test_lrc_ar_cell_forward_pass():
    units = 4
    batch = 2
    input_dim = 4  # AR cell: input_dim == units
    cell = LRC_AR_Cell(units=units, output_mapping=None, input_mapping=None)
    inputs = tf.zeros([batch, input_dim])
    state = [tf.zeros([batch, units])]
    output, new_state = cell(inputs, state)
    assert output.shape == (batch, units)
    assert new_state[0].shape == (batch, units)
```

- [ ] **Step 3: Run tests — verify they FAIL (TF not installed yet)**

```bash
uv run pytest tests/neurons/test_cells.py -v
```

Expected: `ModuleNotFoundError: No module named 'tensorflow'`
This confirms the test setup is correct and TF is not yet available.

- [ ] **Step 4: Commit tests**

```bash
git add tests/
git commit -m "test(neurons): add smoke tests for BaseCell, LRC_Cell, LRC_AR_Cell"
```

### Task 3: Install dependencies and verify

- [ ] **Step 1: Install all dependencies including Metal extra**

```bash
uv sync --extra metal
```

Expected: uv resolves and installs TF 2.15.x, tensorflow-metal, and all other deps.
A `uv.lock` file is created at the repo root.

- [ ] **Step 2: Verify TF version**

```bash
uv run python -c "import tensorflow as tf; print(tf.__version__)"
```

Expected output: `2.15.x`

- [ ] **Step 3: Run tests — verify they PASS**

```bash
uv run pytest tests/neurons/test_cells.py -v
```

Expected: all 10 tests PASS.

If any test fails, read the error carefully:
- `TypeError: Can't instantiate abstract class` on `BaseCell` → expected and correct (test_basecell_is_abstract passes)
- Shape mismatch → check `units` / `input_dim` values in test
- Import error → check `src/neurons/__init__.py` exports

- [ ] **Step 4: Commit uv.lock**

```bash
git add uv.lock
git commit -m "build: add uv.lock after initial uv sync with TF 2.15"
```

---

## Chunk 3: README + Merge

### Task 4: Update README setup instructions

**Files:**
- Modify: `README.md` (add Setup section)

- [ ] **Step 1: Add Setup section to README**

Find the `---` separator after the Architecture Matrix table (around line 38) and add
a new `## Setup` section directly after the Tasks/Evaluation sections, before any
existing setup content. The section should read:

```markdown
## Setup

**Requirements:** [uv](https://docs.astral.sh/uv/) — install once with `curl -LsSf https://astral.sh/uv/install.sh | sh`

```bash
# Mac (Apple Silicon — includes Metal GPU acceleration)
uv sync --extra metal

# Linux / CPU-only
uv sync

# Run any script
uv run python src/...
uv run pytest
```

> **Legacy:** `environment.yml` (conda, TF 2.4.1) is kept as a historical reference only.
> The active environment is `pyproject.toml`.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README with uv setup instructions"
```

### Task 5: Merge to main

- [ ] **Step 1: Merge branch to main**

```bash
git checkout main
git merge --no-ff phase1/step3-tf-uv-setup -m "merge: phase1/step3-tf-uv-setup — TF 2.15 + uv pyproject.toml"
```

- [ ] **Step 2: Push branch and main**

```bash
git push origin phase1/step3-tf-uv-setup
git push origin main
```

- [ ] **Step 3: Update Obsidian work log**

Add an entry to `Thesis/work-documentation.md` (via Obsidian MCP) documenting:
- Branch: `phase1/step3-tf-uv-setup`
- What was done: created pyproject.toml, TF 2.15, Metal extra, smoke tests, uv.lock
- Files changed: `pyproject.toml`, `uv.lock`, `environment.yml`, `tests/neurons/test_cells.py`, `README.md`

- [ ] **Step 4: Update `Meta/claude-project-status.md`**

Mark the TF upgrade step as done and set next step to `phase1/step4-model-factory`.
