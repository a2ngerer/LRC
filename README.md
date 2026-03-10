# RNN Architecture Benchmark — Master Thesis

Systematic benchmark of **bio-inspired RNN cell types** across continuous-time dynamical system modeling and reinforcement learning control tasks.

This repository is part of a master's thesis at TU Wien evaluating novel neuron architectures against classical baselines in a controlled, reproducible setting.

---

## What We Benchmark

**4 neuron types × 2 wiring architectures = 8 model combinations**, evaluated on Neural ODE fitting tasks and robotics control environments.

### Neuron Types

| Neuron | Type | Description |
|--------|------|-------------|
| **LRC** | Bio-inspired ODE | Liquid-Resistance Liquid-Capacitance — adaptive elastance cell extending LTC ([Farsang et al., 2024](https://arxiv.org/abs/2403.08791)) |
| **STC** | Bio-inspired ODE | Saturated Liquid Time-Constant — LTC with bounded forget/update conductances ([Farsang et al., 2024](https://arxiv.org/abs/2403.08791)) |
| **LSTM** | Classical baseline | Long Short-Term Memory |
| **CT-RNN** | ODE baseline | Continuous-Time RNN — leaky integrator |

### Wiring Architectures

| Wiring | Description |
|--------|-------------|
| **Dense** | Fully connected — all neurons connect to all neurons |
| **NCP** | Neural Circuit Policy — sparse, biologically-inspired wiring ([Lechner et al., 2020](https://www.nature.com/articles/s42256-020-00237-3)) |

### Architecture Matrix

| | Dense | NCP |
|---|:---:|:---:|
| **LRC** | LRC + Dense | LRC + NCP |
| **STC** | STC + Dense | STC + NCP |
| **LSTM** | LSTM + Dense | LSTM + NCP |
| **CT-RNN** | CT-RNN + Dense | CT-RNN + NCP |

---

## Tasks

### Neural ODE — Dynamical System Fitting

Fit continuous-time trajectories of 6 dynamical systems:

| System | Type |
|--------|------|
| Spiral | 2D decaying spiral |
| Duffing oscillator | Nonlinear forced oscillator |
| Periodic sinusoid | Simple periodic signal |
| Periodic Lotka-Volterra | Classic predator-prey |
| Limited Lotka-Volterra | Carrying-capacity variant |
| Nonlinear Lotka-Volterra | Nonlinear predator-prey |

### Control — Gymnasium

| Environment | Task |
|-------------|------|
| `Pendulum-v1` | Continuous control, swing-up |
| `CartPole-v1` | Discrete control, balancing |

---

## Evaluation Metrics

- **Performance**: MSE (Neural ODE), reward/accuracy (control), convergence speed
- **Dynamics**: Lipschitz constant, phase portrait analysis
- **Efficiency**: Training time per epoch, inference speed, GPU memory

---

## Repository Structure

```
src/
├── neurons/          # Cell implementations (LRC, STC, LSTM, CT-RNN)
├── wirings/          # Dense and NCP wiring
├── models/           # make_model(neuron, wiring) factory
├── tasks/
│   ├── neural_ode/   # Data generators + training loop for 6 systems
│   └── control/      # Gymnasium environments
└── evaluation/       # Metrics, visualization, profiling

experiments/
└── configs/          # YAML configs per experiment

classification/       # Original LRC classification experiments (Farsang et al.)
neuralODE/            # Original LRC neural ODE experiments (Farsang et al.)
docs/                 # Documentation, original paper README
```

> The `classification/` and `neuralODE/` directories contain the original code from [Farsang et al. (2024)](https://arxiv.org/abs/2403.08791), kept as reference. New experiments are built under `src/` and `experiments/`.

---

## Status

| Phase | Description | Timeline | Status |
|-------|-------------|----------|--------|
| **Phase 1** | Foundation: modular structure, TF upgrade, model factory, Neural ODE port | März 2026 | In progress |
| **Phase 2** | New architectures: CT-RNN, STC, NCP wiring | April 2026 | Planned |
| **Phase 3** | Full benchmark run, evaluation pipeline, plots | Mai–Juni 2026 | Planned |
| **Phase 4** | Reproducibility, thesis export, final README | Juli–Sept 2026 | Planned |

---

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

---

## Based On

This benchmark extends the LRC implementation by Farsang, Neubauer & Grosu (2024):

```bibtex
@misc{farsang2024liquidresistanceliquidcapacitance,
      title={Liquid Resistance Liquid Capacitance Networks},
      author={Mónika Farsang and Sophie A. Neubauer and Radu Grosu},
      year={2024},
      eprint={2403.08791},
      archivePrefix={arXiv},
      primaryClass={cs.NE},
      url={https://arxiv.org/abs/2403.08791},
}
```
