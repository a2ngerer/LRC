# Benchmark v2: Vanishing-Gradient-Fixed Cells — Design

**Date:** 2026-06-10
**Branch:** `phase3/step12-benchmark-v2-fixed-cells` (based on `phase3/step11-benchmark-matrix`)
**Status:** approved by user (design dialogue 2026-06-10)

## Goal

The thesis matrix (v1) benchmarks `{ltc, lrc, gru, lstm} x {dense, ncp}` on 6 ODE
systems. LTC/LRC are expected to show convergence weaknesses caused by the proven
vanishing/exploding gradient problem of BPTT through the ODE (Lechner & Hasani 2020),
amplified by LTC stiffness (Farsang et al. 2024). Benchmark v2 adds *fixed* cell
variants and an optimizer-level fix, so the thesis can show a controlled
before/after comparison: same tasks, same seeds, same trainer — only the fix varies.

Evidence base: `~/Downloads/Vanishing-Gradient-Loesung-LTC-LRC.md` (solution report,
2026-06-10). Fixes implemented here are the supervised-applicable subset: mixed
memory (architecture), CfC (architecture/numerics), gradient clipping (optimization).
RL-level fixes (BC warm-start, TBPTT) are out of scope until the control tasks exist.

## Decisions (from design dialogue)

1. **Fix scope:** mixed-memory cells (`mm_ltc`, `mm_lrc`) + own CfC implementation
   (`cfc`) + gradient clipping in the trainer.
2. **Clipping is its own axis** (`--clip-norm`), not bundled into the new cells —
   so architecture fix and optimizer fix are separable in the analysis.
3. **CfC is implemented from the paper** (Hasani et al. 2022) as a `BaseCell`
   subclass, consistent with the hand-implemented LTC/LRC; not a wrapper around
   `ncps.tf.CfCCell`.
4. **Structure:** one branch, one runner, additive extension (approach A).
   v1 defaults and the 240-run SLURM array contract stay byte-identical.

## New cells

All in `src/neurons/`, subclassing `BaseCell`, registered in `_CELL_REGISTRY`
(`src/models/rnn_model.py`) and exported via `src/neurons/__init__.py`.

### `CfC_Cell` (`cfc_cell.py`, key `cfc`)

Closed-form continuous-time cell after Hasani et al. 2022 (arXiv:2106.13898):

```
x(t) = sigmoid(-f(x, I) * t) * g(x, I)  +  (1 - sigmoid(-f(x, I) * t)) * h(x, I)
```

- Three small backbone heads `f`, `g`, `h` over `[input, state]` (shared backbone
  layer + per-head projection, as in the reference implementation).
- No solver, no `ode_unfolds`. `elapsed_time` from the input tuple is used as `t`
  (BaseCell irregular-sampling convention); default `t = 1.0`.
- State: single tensor `(batch, units)`.
- Verification note: gate signs/assignment to be cross-checked against the
  paper PDF and `raminmh/CfC` before thesis use (solution report, appendix).

### `MixedMemoryCell` (`mixed_memory_cell.py`, keys `mm_ltc`, `mm_lrc`)

Generic mixed-memory wrapper after the ODE-LSTM pattern (Lechner & Hasani 2020,
arXiv:2006.04418): an LSTM gating structure owns the memory path `c` (additive
update, constant error propagation); the wrapped ODE cell (LTC or LRC) provides
the continuous-time dynamics of the hidden state.

- Step: LSTM gates compute `c_new` and candidate hidden state from `(x, h)`;
  the inner ODE cell evolves the hidden state with input `x` over `dt`.
  Output: evolved `h`. Exact wiring to be verified against
  `mlech26l/ode-lstms` during implementation.
- State: `[h, c]` + inner ODE cell state (list state, NCP-compatible — the
  wiring already handles LSTM's multi-state cells).
- Concrete registry classes: `MM_LTC_Cell`, `MM_LRC_Cell`. `mm_lrc` forwards
  `elastance_type="asymmetric"` to the inner LRC (via `CELL_KWARGS`).

## Trainer extension (`src/tasks/neural_ode/trainer.py`)

- New parameter `clip_norm=None`. When set: `tf.clip_by_global_norm(grads, clip_norm)`
  before `apply_gradients`. `None` = exact v1 behavior (bit-identical path).
- `GradientFlowTracker` records pre-clip norms and (when clipping is active)
  the post-clip global norm, so RQ4 analysis can see *when* clipping engaged.

## Runner extension (`experiments/run_benchmark.py`)

- `CELLS_V2 = ['mm_ltc', 'mm_lrc', 'cfc']`; default `CELLS` list unchanged
  (SLURM array contract for the 240 v1 runs is preserved).
- New axis `--clip-norm <float>` (0 = off). Goes into the run spec, the result
  JSON (`config.clip_norm`), and the result filename (`..._clip1.0.json`) so
  both axes coexist in one directory.
- `--profile v1|v2` shorthand:
  - `v1`: today's matrix — 240 runs, `results/runs/`.
  - `v2`: `{mm_ltc, mm_lrc, cfc} x {clip 0, clip 1.0}` (360 runs)
    + `{ltc, lrc} x clip 1.0` (120 runs; isolates the optimizer fix on the
    problem cells) = 480 runs, `results/runs_v2/`.
    GRU/LSTM with clipping intentionally omitted (no convergence problem;
    saves 120 GPU runs).
- `CELL_KWARGS` gains `'mm_lrc': dict(elastance_type='asymmetric')`.
- Result `schema_version` -> 2 (additive fields only).
- Cluster: `cluster/benchmark_v2_array.sbatch` + `submit_benchmark_v2.sh`
  following the existing pattern (array throttle %8).

## Tests

Extend `tests/neurons/test_cells.py` and `tests/experiments/test_run_benchmark.py`:

- `cfc`, `mm_ltc`, `mm_lrc`: build/call shapes (dense + ncp), multi-state
  handling, `elapsed_time` tuple convention, seed determinism (same seed ->
  identical loss history).
- Trainer: `clip_norm` provably bounds the global norm; `clip_norm=None`
  bit-identical to the v1 path.
- Runner: `--profile v2` produces exactly 480 specs in deterministic order;
  `--profile v1` produces the unchanged 240 (contract regression test);
  result JSON contains `clip_norm` and `cell_kwargs`.
- `smoke_test_combinations.py`: 3 new cells x 2 wirings as mini runs.

## Aggregation, plots, comparison guide

- `aggregate_results.py`: `--runs-dirs` (multiple result dirs), `clip_norm`
  as a grouping column. Wilcoxon pairing over (system, seed) works unchanged
  because v1/v2 share systems and seeds — paired comparisons `ltc vs mm_ltc`,
  `lrc vs mm_lrc`, `ltc vs cfc`, `clip on vs off` are direct.
- `plot_results.py`: loss curves and gradient-flow plots groupable by cell
  family (base vs fixed).
- `experiments/BENCHMARK_COMPARISON.md` (English): how to run v1 (local +
  cluster), how to run v2, which aggregation calls produce which paired
  comparisons, which metric answers which question (NRMSE = RQ1 performance,
  gradient norms = RQ4 mechanism, convergence iteration = trainability),
  hypothesis table (expected outcome if the fix works, how to read a negative
  result).

## Unchanged (compatibility guarantees)

- All existing cells, `--profile v1` behavior, v1 result schema (new fields
  additive; aggregation reads schema 1 and 2), v1 cluster scripts.

## Out of scope

- RL/control-level fixes (BC warm-start, TBPTT, target networks) — these bind
  to the not-yet-existing control tasks (Phase 3 control work).
- `mm_cfc` (CfC inside mixed memory) — can be added later as a single registry
  entry if the thesis needs it; YAGNI for now.
- Van der Pol / irregular-sampling extra benchmarks (separate step).
