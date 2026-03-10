# Design: TF Upgrade + uv Setup (Phase 1 / Step 3)

**Date:** 2026-03-10
**Branch:** `phase1/step3-tf-uv-setup`
**Status:** Approved

## Goal

Replace the existing conda environment (`environment.yml`, TF 2.4.1) with a
`uv`-managed `pyproject.toml` using TF 2.15. Verify that `BaseCell`, `LRC_Cell`,
and `LRC_AR_Cell` import and instantiate correctly under the new setup.

## TF Version Decision

**TF 2.15** — last release with native Keras 2. Avoids the Keras 3 migration
introduced in TF 2.16, which would break `tf.keras.layers.AbstractRNNCell` and
the entire cell architecture. Has native Apple Silicon (arm64) wheels on PyPI.
Compatible with `tensorflow-metal` for Mac GPU acceleration.

Deferred:
- `keras-ncps` → Phase 2 (NCP wiring step)
- `tfdiffeq` → Phase 1 Step 5 (Neural ODE porting; likely needs replacement)

## pyproject.toml Structure

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
```

## Platform Handling

| Platform         | Command              | Notes                        |
|------------------|----------------------|------------------------------|
| Mac (dev)        | `uv sync --extra metal` | Metal GPU via tensorflow-metal |
| Linux/CPU        | `uv sync`            | Standard TF, no extras       |
| Linux/CUDA (future) | `uv sync` + `[cuda]` extra | Extra added when needed   |

`environment.yml` stays in the repo as legacy reference with a deprecation comment.

## Verification Steps

1. `uv sync --extra metal` completes without errors
2. `uv run python -c "import tensorflow as tf; print(tf.__version__)"` → `2.15.x`
3. `uv run python -c "from src.neurons import BaseCell, LRC_Cell, LRC_AR_Cell; c = LRC_Cell(32); assert c.state_size == 32"` passes
4. `pytest` smoke tests for `BaseCell`, `LRC_Cell`, `LRC_AR_Cell` pass

## Out of Scope

- `tfdiffeq` compatibility (Step 5)
- `keras-ncps` integration (Phase 2)
- CUDA cluster setup (future)
- Porting `neuralODE/` scripts (Step 5)
