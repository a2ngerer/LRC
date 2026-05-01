# LRC Repository Overview

> Forked from [MoniFarsang/LRC](https://github.com/MoniFarsang/LRC) → working copy at `a2ngerer/LRC`  
> Local path: `master_thesis/code/`  
> Paper: [[farsang2024-lrc]] — NeurIPS 2024 NeuroAI Workshop ([arXiv:2403.08791](https://arxiv.org/abs/2403.08791))

## What LRC Cells Are

[[liquid-resistance-liquid-capacitance|LRC]] extends [[liquid-time-constant-network|LTC]] by replacing the fixed time-constant with an **input- and state-dependent elastance** (adaptive capacitance). The ODE update becomes:

```
v' = -v · sigmoid(f) + v_leak · tanh(g)
v_next = v + elastance(x, v) · v'
```

Where `elastance` is a small Dense layer followed by a sigmoid, yielding a per-step, data-driven integration step size. Two elastance variants:

| Variant | Elastance formula |
|---------|------------------|
| `symmetric` | `sigmoid(Wx + b_shift) - sigmoid(Wx - b_shift)` |
| `asymmetric` | `sigmoid(Wx)` |

The **forget gate** (enabled by default) replaces the fixed `tau` with a learnable recurrent gating, analogous to LSTM's forget gate but biophysically motivated.

## Repo Structure

```
code/
├── classification/
│   ├── lrc_cell.py               # LRC_Cell — standard classification
│   ├── lstm_cell.py              # LSTM baseline
│   ├── gru_cell.py               # GRU baseline
│   ├── irregular_sampled_datasets.py
│   ├── run_imdb.py               # Sentiment classification (IMDB)
│   ├── run_mnist.py              # Permuted MNIST
│   └── run_person_activity.py    # Person activity localization
├── neuralODE/
│   ├── lrc_ar_cell.py            # LRC_AR_Cell — autoregressive ODE variant
│   ├── ode_model.py              # Neural ODE wrapper
│   └── run_ode.py                # ODE system fitting
├── environment.yml               # Conda environment spec
├── download_dataset.sh           # Person Activity dataset download
└── README.md
```

## Key Files

### `classification/lrc_cell.py` — `LRC_Cell`
- Extends `tf.keras.layers.AbstractRNNCell`
- Parameters: `gleak`, `vleak`, `sigma`, `mu`, `w` (forget gate), `h` (recurrent), `sensory_*` (input synapses), `elastance_mapping` (Dense layer)
- Solver options: `explicit` (Euler), `hybrid` (implicit-explicit)
- Supports irregularly sampled sequences via `(inputs, elapsed_time)` tuple input

### `neuralODE/lrc_ar_cell.py` — `LRC_AR_Cell`
- **Autoregressive** variant: input is treated as state, outputs `v_prime` (derivative)
- No sensory weight parameters (no cross-neuron input synapses)
- Used with `tfdiffeq` ODE solver for fitting continuous dynamical systems
- Default elastance: `symmetric` (vs. `interp` default in LRC_Cell)

## Existing Benchmarks

### Classification (`classification/`)
| Task | Script | Dataset |
|------|--------|---------|
| Sentiment | `run_imdb.py` | IMDB (25k reviews) |
| Sequence | `run_mnist.py` | Permuted MNIST |
| Activity | `run_person_activity.py` | Person Activity (localization) |

Models: `lstm`, `mgu`, `gru`, `LRC_sym_elastance`, `LRC_asym_elastance`

### Neural ODE (`neuralODE/`)
| System | `--data` flag |
|--------|--------------|
| Sinusoid | `periodic_sinusodial` |
| Spiral | `spiral` |
| Duffing oscillator | `duffing` |
| Lotka-Volterra (periodic) | `periodic_predator_prey` |
| Lotka-Volterra (limited) | `limited_predator_prey` |
| Lotka-Volterra (nonlinear) | `nonlinear_predator_prey` |

## Framework

| Component | Version |
|-----------|---------|
| TensorFlow | 2.4.1 (GPU) |
| Keras | 2.4.3 |
| Python | 3.9.19 |
| tfdiffeq | 0.0.1 |
| CUDA | 10.1 |
| cuDNN | 7.6.5 |
| Package manager | conda (`environment.yml`) |

> **Migration note**: Thesis benchmarks will use TF 2.13+ and `uv` for package management. See [[benchmark-repo-roadmap]].

## Related Notes

- [[liquid-resistance-liquid-capacitance]] — concept explanation
- [[liquid-time-constant-network]] — LTC (parent model)
- [[neural-circuit-policy]] — NCP wiring (planned for thesis)
- [[farsang2024-lrc]] — paper summary
- [[benchmark-repo-roadmap]] — development plan for thesis benchmarks
