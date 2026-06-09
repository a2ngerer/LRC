# Phase 3a — Benchmark Matrix Completion (phase3/step11)

**Date:** 2026-06-09
**Branch:** `phase3/step11-benchmark-matrix`
**Decision:** STC is not finalized yet → benchmark matrix changed to
**LTC, LRC, GRU, LSTM × Dense, NCP** (user decision, supersedes the K8 note
"GRU wird nicht separat implementiert" in masterarbeit-projektplan.md).

## Scope

1. **New cells** — `LTC_Cell` (ncps-style fused solver) and `GRU_Cell`
   (wrapper around `tf.keras.layers.GRUCell`, analogous to `LSTM_Cell`).
   Registered in `_CELL_REGISTRY` as `'ltc'` / `'gru'`.
2. **Bug fixes**
   - `lrc_cell.py` / `lrc_ar_cell.py`: `_init_range` → `_init_ranges` typo
     (AttributeError in the error path).
   - `lrc_cell.py`: `tf.keras.layers.Concatenate()` was instantiated inside
     `_ode_solver` on every call (memory leak in eager training) → `tf.concat`.
   - **Latent step10 bug:** `benchmark_neural_ode.py` passed `Sequential`
     models to `euler_odeint`, which calls `func(t, y)` — `Sequential`'s
     signature `(inputs, training, mask)` misroutes `y` into `training`
     (IndexError). Fixed via `SequentialODEFunc` wrapper in
     `src/tasks/neural_ode/ode_model.py`; the step10 script never ran
     end-to-end before this fix.
3. **Reproducibility** — `src/utils/seeding.py` (`set_global_seed`), seeded
   batch sampling (`rng` parameter through `train`/`get_batch`).
4. **Metrics** — trainer loss switched MAE → MSE (configurable);
   `src/evaluation/metrics.py` adds MSE + NRMSE (range-normalized) for
   full-trajectory rollout evaluation.
5. **Gradient-flow tooling (RQ4)** — `src/evaluation/gradient_flow.py`
   (`GradientFlowTracker`, per-leaf-layer gradient norms; distinguishes
   inter/command/motor RNN layers in NCP models).
6. **Cluster runner** — `experiments/run_benchmark.py`: one invocation = one
   `(cell, wiring, system, seed)` run = one JSON; `--list/--count/--index`
   define the SLURM array contract; 4×2×6×5 = 240 specs.
7. **Evaluation pipeline** — `experiments/aggregate_results.py` (mean±std
   tables, Wilcoxon signed-rank α=0.05 paired over (system, seed), paired
   Cohen's d) and `experiments/plot_results.py` (loss curves, phase
   portraits, gradient-flow plots).
8. **dataLAB cluster setup** — `cluster/` (config, sync, setup_env,
   benchmark_array.sbatch, submit_benchmark, fetch_results, README) adapted
   from the proven `arc-jax-rl` pattern; `cuda` extra in pyproject
   (`tensorflow[and-cuda]`, CUDA 12).

## Verification

- 87 pytest tests pass (incl. new cell tests, runner spec/repro tests).
- Smoke test: 11 PASS + 1 XFAIL (`lrc_ar`+ncp, documented constraint) —
  all 8 thesis-matrix combinations train.
- End-to-end pipeline smoke (8 tiny runs → aggregate → plots) verified locally.
- `uv lock` resolves with `--extra cuda` (nvidia CUDA-12 wheels).

## Open / not in scope

- Actual GPU runs: **waiting for explicit user GO** (submission via
  `./cluster/submit_benchmark.sh`).
- STC cell + intermediate cell (RQ5), control tasks (PPO), Lipschitz
  metrics: unchanged, still Phase-3a/3b backlog.
