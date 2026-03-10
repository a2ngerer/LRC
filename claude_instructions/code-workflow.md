# Code Workflow — Phased Development Plan

> **Branch naming**: `phase{N}/step{M}-{descriptive-name}`
> **After each step**: merge to `main` via `git merge --no-ff` with a summary message
> **Progress tracking**: `obsidian_master_thesis/Thesis/work-documentation.md`

---

## Branch & Commit Protocol

### Starting a step
```bash
git checkout main && git pull
git checkout -b phase1/step1-migrate-lrc-cells
```

### Commit format
```
<type>(<scope>): <what> — <why if not obvious>

Types: feat, refactor, fix, chore, test
Examples:
  feat(neurons): add BaseCell abstract class — shared interface for all neuron types
  refactor(neurons): move lrc_cell.py to src/neurons/
  chore: add pyproject.toml with uv and TF 2.13 deps
```

### Merging a step to main
```bash
git checkout main
git merge --no-ff phase{N}/step{M}-name -m "merge: phase{N}/step{M} — <one-line summary>"
```

Keep the branch after merging — it documents the history.

---

## Phase 1 — Foundation (März 2026)

### step1: migrate-lrc-cells
**Branch**: `phase1/step1-migrate-lrc-cells`
**Goal**: Move existing LRC cells into the new modular structure.

Steps:
1. `mkdir -p src/neurons`
2. Copy `classification/lrc_cell.py` → `src/neurons/lrc_cell.py`
3. Copy `neuralODE/lrc_ar_cell.py` → `src/neurons/lrc_ar_cell.py`
4. Create `src/neurons/__init__.py` exporting `LRC_Cell`, `LRC_AR_Cell`
5. Create `src/__init__.py`

Commit after each sub-step. Success: `from src.neurons import LRC_Cell, LRC_AR_Cell` works.

---

### step2: basecell-interface
**Branch**: `phase1/step2-basecell-interface`
**Goal**: Abstract `BaseCell` that all neuron types must implement.

Steps:
1. Create `src/neurons/base_cell.py` — abstract class extending `tf.keras.layers.AbstractRNNCell`
2. Required abstract methods: `build`, `call`, `state_size`, `output_size`
3. Move shared helpers (`_sigmoid`, `_map_inputs`, `_map_outputs`) into `BaseCell`
4. Update `lrc_cell.py` → subclass `BaseCell`
5. Update `lrc_ar_cell.py` → subclass `BaseCell`
6. Update `src/neurons/__init__.py`

Success: `isinstance(LRC_Cell(64), BaseCell)` is `True`.

---

### step3: tf-upgrade-uv-migration
**Branch**: `phase1/step3-tf-upgrade-uv-migration`
**Goal**: Migrate from conda / TF 2.4.1 to uv / TF 2.13+.

Steps:
1. Create `pyproject.toml` at `code/` root:
   - `tensorflow>=2.13`, `keras-ncps`, `tfdiffeq`, `gymnasium`, `numpy`, `matplotlib`, `scipy`, `pyyaml`, `tqdm`
2. Run `uv sync` to create `.venv`
3. Add `.venv/` and `results/` to `.gitignore`
4. Fix breaking API changes (TF 2.4 → 2.13): check `tf.compat.v1` usage, `model.fit` API
5. Verify: `uv run python -c "from src.neurons import LRC_Cell; c = LRC_Cell(32); print('OK')"`

Success: import works under TF 2.13+, uv resolves deps without conflict.

---

### step4: model-factory
**Branch**: `phase1/step4-model-factory`
**Goal**: `make_model(neuron, wiring, units)` factory that returns a compiled Keras model.

Steps:
1. `mkdir -p src/models src/wirings`
2. `src/wirings/dense.py` — wraps `tf.keras.layers.RNN(cell, return_sequences=True)`
3. `src/wirings/__init__.py` — exports `DenseWiring`
4. `src/models/rnn_model.py` — `make_model(neuron_type, wiring_type, units, **kwargs)`:
   - Instantiates the cell (from neuron_type string or class)
   - Wraps in the specified wiring
   - Returns a `tf.keras.Model`
5. `src/models/__init__.py`
6. Test: `make_model('lrc', 'dense', units=64).summary()` prints without error

Success: factory returns a working Keras model for `('lrc', 'dense', 64)`.

---

### step5: port-neural-ode-tasks
**Branch**: `phase1/step5-port-neural-ode-tasks`
**Goal**: Port the 6 Neural ODE data generators and training loop to new structure.

Steps:
1. `mkdir -p src/tasks/neural_ode experiments/configs`
2. `src/tasks/neural_ode/datasets.py` — data generators for all 6 systems:
   spiral, duffing, periodic_sinusoidal, periodic_predator_prey,
   limited_predator_prey, nonlinear_predator_prey
3. `src/tasks/neural_ode/trainer.py` — training loop (takes model + dataset)
4. `experiments/configs/neural_ode_lrc_spiral.yaml` — example config:
   neuron: lrc, wiring: dense, data: spiral, units: 64, niters: 1000
5. `experiments/run_neural_ode.py` — entry point reading config, running experiment

Success: `uv run python experiments/run_neural_ode.py --config experiments/configs/neural_ode_lrc_spiral.yaml` runs 5 iterations without error.

---

### step6: verify-lrc-results
**Branch**: `phase1/step6-verify-lrc-results`
**Goal**: Confirm migrated LRC + Dense qualitatively reproduces original results.

Steps:
1. Run spiral + duffing with LRC + Dense (100 iterations) via new entry point
2. Compare loss curve qualitatively to original `neuralODE/run_ode.py` output
3. Document comparison in `obsidian_master_thesis/Thesis/work-documentation.md`
4. If results diverge significantly: investigate and fix before proceeding to Phase 2

Success: Training converges, loss behavior matches original implementation.

---

## Phase 2 — New Architectures (April 2026)

### step1: ctrnn-cell
**Branch**: `phase2/step1-ctrnn-cell`
**Goal**: Continuous-Time RNN (leaky integrator ODE cell).

Steps:
1. `src/neurons/ctrnn_cell.py` — subclasses `BaseCell`
2. Dynamics: `dh/dt = (-h + tanh(W_in * x + W_rec * h + b)) / tau`
3. Explicit Euler solver with configurable `dt`
4. Add `CTRNN_Cell` to `src/neurons/__init__.py`
5. Register `'ctrnn'` in `make_model` factory

Success: `make_model('ctrnn', 'dense', 64)` trains on spiral task.

---

### step2: stc-cell
**Branch**: `phase2/step2-stc-cell`
**Goal**: STC (Spike-Threshold Capacitance) neuron type.

**BLOCKER**: Consult supervisor for spec before implementing.
Steps: TBD after supervisor meeting.

---

### step3: ncp-wiring
**Branch**: `phase2/step3-ncp-wiring`
**Goal**: NCP wiring via `keras-ncps`.

Steps:
1. `src/wirings/ncp.py` — wraps `keras_ncps.wirings.AutoNCP` with any `BaseCell`
2. Update `src/wirings/__init__.py`
3. Register `'ncp'` in `make_model` factory
4. Test: `make_model('lrc', 'ncp', 64)` instantiates and runs a forward pass

Success: LRC + NCP model works end-to-end.

---

### step4: test-all-combinations
**Branch**: `phase2/step4-test-all-combinations`
**Goal**: Smoke-test all available neuron × wiring combinations.

Steps:
1. `experiments/test_combinations.py` — loops over all (neuron, wiring) pairs
2. Runs 3 training iterations per combination on spiral
3. Logs pass/fail with exception trace if any fails
4. Fix any combination-specific issues found

Success: All combinations complete 3 training iterations without exception.

---

## Phase 3 — Benchmarks (Mai–Juni 2026)

Full benchmark run: all 8 combinations × 6 Neural ODE tasks + Gymnasium control.
Details: `obsidian_master_thesis/Thesis/benchmark-repo-roadmap.md` (Phase 3 section).
Specific branch plan: drafted at end of Phase 2.

---

## Phase 4 — Thesis Finalization (Juli–September 2026)

Reproducibility, clean README, archived results, thesis tables.
Details: `obsidian_master_thesis/Thesis/benchmark-repo-roadmap.md` (Phase 4 section).
Specific branch plan: drafted at end of Phase 3.

---

## Work Documentation Protocol

After every session modifying `/code`:

1. Open `obsidian_master_thesis/Thesis/work-documentation.md`
2. Add an entry at the top (newest first):
   - Date, branch, status (in progress / merged)
   - Bullet list of what was done
   - Files changed
   - Any issues or decisions

Template in `work-documentation.md`.
