# Benchmark Repo Roadmap

> Development plan for the thesis benchmark codebase.  
> Starting from: [[lrc-repo-overview]] (forked `a2ngerer/LRC`)  
> Local path: `master_thesis/code/`

## Goal

Benchmark **4 neuron types × 2 wiring architectures = 8 combinations** across Neural ODE and robotics control tasks to systematically compare novel bio-inspired RNN cells. Additionally: investigate LRC+NCP convergence failure via gradient flow analysis (RQ4) and ablate architectural features via intermediate models (RQ5).

## Architecture Matrix (4×2)

| | Dense | NCP ([[neural-circuit-policy]]) |
|---|---|---|
| **LRC** | LRC + Dense | LRC + NCP |
| **STC** | STC + Dense | STC + NCP |
| **LSTM** | LSTM + Dense | LSTM + NCP |
| **CT-RNN** | CT-RNN + Dense | CT-RNN + NCP |

- **LRC**: [[liquid-resistance-liquid-capacitance]] — adaptive elastance, bio-inspired ODE cell
- **STC**: Saturated LTC — constrains forget/update conductances with σ(f) and tanh(g) to reduce stiff ODE behavior ([[farsang2024-lrc]])
- **LSTM**: Classical gated RNN baseline
- **CT-RNN**: Continuous-Time RNN — leaky integrator ODE cell

## Extended Scope (added 2026-03-29, Monika Farsang feedback)

### Intermediate Models — RQ5
CT-RNN augmented with liquid elastance ε(w_i) as ablation instrument. Allows isolating the contribution of liquid elastance independently from chemical synapses.

| | Dense | NCP |
|---|---|---|
| **CT-RNN + ε(w_i)** | Intermediate + Dense | Intermediate + NCP |

### Gradient Flow Analysis — RQ4
LRCs converge less effectively in NCP architectures than in dense. Systematic gradient norm tracking per layer/wiring to understand this divergence.

## Planned Repo Structure (Approach A: Full Modular Restructure)

```
code/
├── src/
│   ├── neurons/
│   │   ├── lrc_cell.py        # Migrated from classification/
│   │   ├── stc_cell.py        # New: STC neuron
│   │   ├── lstm_cell.py       # Baseline
│   │   ├── ctrnn_cell.py      # CT-RNN
│   │   └── intermediate_cell.py  # New: CT-RNN + liquid elastance (RQ5)
│   ├── wirings/
│   │   ├── dense.py           # Fully connected (default)
│   │   └── ncp.py             # Neural Circuit Policy (keras-ncps)
│   ├── models/
│   │   └── rnn_model.py       # Factory: make_model(neuron, wiring)
│   ├── tasks/
│   │   ├── neural_ode/        # Duffing, Lotka-Volterra, Spiral, Sinusoid
│   │   └── control/           # Pendulum, CartPole (Gymnasium)
│   └── evaluation/
│       ├── metrics.py         # MSE, accuracy, Lipschitz, stability
│       ├── visualization.py   # Phase portraits, training curves
│       ├── gradient_flow.py   # Gradient norm tracking per layer (RQ4)
│       └── profiling.py       # Training time, inference speed, memory
├── experiments/
│   └── configs/               # YAML configs per experiment
├── results/                   # gitignored
├── notebooks/                 # Exploratory analysis
└── pyproject.toml             # uv, TF 2.13+
```

## Development Phases

### Phase 1 — Foundation (März 2026)
- [x] Migrate `lrc_cell.py` and `lrc_ar_cell.py` into `src/neurons/` ✅
- [x] Define `BaseCell` interface (abstract class all neurons implement) ✅
- [x] Upgrade TF 2.4.1 → TF 2.15, migrate from conda to `uv` ✅
- [x] Implement `make_dense_model` / `make_ncp_model` factories in `src/models/` ✅
- [x] Port existing Neural ODE tasks (6 systems) to new structure ✅
- [x] Verify LRC + Dense reproduces original paper results ✅

### Phase 2 — New Architectures (April 2026)
- [x] Implement `ctrnn_cell.py` (CT-RNN / leaky integrator ODE) ✅
- [x] Implement `lstm_cell.py` (LSTM baseline) ✅
- [ ] Implement `stc_cell.py` (STC) ← **unblockiert** ✓
- [x] Implement NCP wiring (`src/wirings/ncp.py`) — `SparseLinear` + `NCPWiring`, 3-layer (inter→command→motor) ✅
- [x] Integration smoke test: alle verfügbaren cell×wiring Kombinationen (7 PASS + 1 XFAIL) ✅
- [ ] Implement `intermediate_cell.py` (CT-RNN + liquid elastance ε(w_i)) — RQ5

### Phase 3 — Benchmarks (Mai–Juli 2026)
- [ ] Run all 8 combinations on Neural ODE tasks (6 systems)
- [ ] Add Gymnasium control tasks: Pendulum-v1, CartPole-v1
- [ ] Implement evaluation pipeline: metrics, phase portraits, training curves
- [ ] **Gradient Flow Analysis (RQ4):** Gradient-Norm Logging für LRC+NCP vs. LRC+Dense, Layer-wise Analyse
- [ ] **Intermediate Model Ablations (RQ5):** CT-RNN + ε(w_i) in Dense & NCP, Vergleich mit CT-RNN und LRC
- [ ] Stability analysis: Lipschitz constant estimation
- [ ] Generate publication-quality plots for thesis

### Phase 4 — Thesis Finalization (Juli–September 2026)
- [ ] Full reproducibility pass (fixed seeds, documented hyperparams)
- [ ] Results exported to `obsidian_master_thesis/` as tables
- [ ] Clean README with setup and run instructions
- [ ] Archive final trained models

## Evaluation Metrics

### Performance
- MSE (Neural ODE tasks), Accuracy (classification tasks)
- Convergence speed (epochs to 95% of best performance)

### Dynamics
- Lipschitz constant (stability measure)
- Phase portrait qualitative analysis
- **Gradient norms per layer** (new, RQ4)

### Efficiency
- Training time (wall clock per epoch)
- Inference speed (samples/sec)
- Memory footprint (peak GPU memory)

### Visualization
- Phase portraits (predicted vs. true trajectories)
- Architecture diagrams (neuron×wiring)
- Training curves (loss, metrics over epochs)
- **Gradient flow plots** (norm per layer over training, RQ4)

## NCP Wiring — Architekturentscheidung

**Entscheidung (2026-03-12):** Wir bauen die NCP-Verdrahtung **selbst**, anstatt die `CfC`/`LTC`-Zellen aus `keras-ncps` zu verwenden.

**Grund:** `keras-ncps` implementiert NCP-Wiring tief in seine eigenen Zelltypen (`CfC`, `LTC`). Diese sind nicht auf unsere Zellen (LRC, LSTM, CT-RNN, STC) anwendbar. Da unser Ziel die **4×2-Matrix** ist (jede Zelle mit jeder Verdrahtung), müssen wir NCP-Wiring zellagnostisch implementieren.

**Ansatz:**
1. `keras-ncps` wird **nur** genutzt, um über `ncps.wirings.AutoNCP` die Adjazenzmatrix (N×N, binär) zu generieren — diese kodiert das NCP-Verbindungsmuster
2. `NCPMaskedCell` — ein generischer Wrapper um jede `BaseCell`, der die Adjazenzmatrix als feste Maske auf die rekurrenten Verbindungen anwendet (multipliziert den Zustandsvektor mit der Maske vor dem Zellaufruf)
3. `NCPWiring(BaseWiring)` — verpackt jede Zelle in `NCPMaskedCell` und dann in `tf.keras.layers.RNN`

**Ergebnis:** `make_model('lrc', 'ncp', 16)`, `make_model('lstm', 'ncp', 16)` usw. funktionieren alle korrekt.

---

## Key Dependencies (Planned)

```toml
[project]
dependencies = [
    "tensorflow>=2.13",
    "keras-ncps",
    "gymnasium",
    "tfdiffeq",
    "numpy",
    "matplotlib",
    "scipy",
    "pyyaml",
]
```

## Control Tasks — RL Algorithm

**Planned: REPPO** ([[REPPO]] | [[voelcker2025relative]])

- On-policy algorithm from the same research group (Grosu, TU Wien) — arXiv:2507.11019
- Uses pathwise policy gradients (reparameterization trick) instead of score-function estimators → lower variance than PPO
- Distributional critic (Q-value distribution, no replay buffer)
- Adaptive KL + entropy tuning → fewer hyperparameters to tune compared to PPO
- Environments: `Pendulum-v1` (continuous), `CartPole-v1` (discrete)

## Related Notes

- [[lrc-repo-overview]] — existing repo structure and code
- [[neural-circuit-policy]] — NCP wiring architecture
- [[liquid-resistance-liquid-capacitance]] — LRC cell concept
- [[liquid-time-constant-network]] — LTC (LRC parent)
- [[Benchmarks für die Masterarbeit]] — broader benchmark context
- [[REPPO]] — planned RL algorithm for control tasks
