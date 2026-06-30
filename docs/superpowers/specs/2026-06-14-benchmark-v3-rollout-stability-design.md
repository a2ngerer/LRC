# Benchmark v3 — Forward-Rollout Stability of Stiff ODE Cells

- **Date:** 2026-06-14
- **Status:** implemented, pending cluster run
- **Branch:** `phase3/step14-benchmark-v3-rollout-stability`
- **Profile:** `--profile v3` → `results/runs_v3` (720 runs)
- **Predecessors:** v1 (240, published cells), v2 (480, vanishing-gradient fixes)

## 1. Motivation

The v1+v2 gradient-flow analysis (all 720 existing run JSONs already carry
per-layer gradient norms under `result.gradient_flow`) resolved the central RQ4
question — *what role does gradient flow play in the convergence pathology* — and
the answer reframes the next benchmark.

Two **distinct, cell-specific** pathologies, not one:

1. **LRC + NCP = localized vanishing gradient.** In the sparse NCP wiring the
   upstream layers receive almost no signal: median per-layer gradient norm
   `0_RNN = 3.95e-6`, `1_SparseLinear = 4.4e-7` vs the LRC dense `0_RNN = 3.4e-3`
   (3–4 orders of magnitude smaller). The motor/inter norm ratio is ~1601 — only
   the output-attached motor layer learns. This is a **median shift**, not a tail
   (NCP divergence rate ~3 %). The mixed-memory path already repairs it
   (`mm_lrc` motor/inter ratio 1.75, 0 % divergence), because the additive memory
   path routes gradient around the ODE Jacobians. **This pathology is already
   explained by the existing data; it needs no new runs.**

2. **LTC family = forward-rollout stiffness.** The "divergence" of `ltc`/`mm_ltc`
   (NRMSE up to 4.96, divergence rate `mm_ltc/ncp` 17 %) is **not** a training or
   gradient problem:
   - 0 / 23 divergent runs had an exploding training loss (max loss < 100× init).
   - Divergent runs reach low final training loss (some `1.5e-5`, below the
     converged median `7.7e-5`) but their full-trajectory Euler rollout blows up.
   - The 1e6 pre-clip gradient spike on `mm_ltc/ncp` is a single transient
     iteration the training recovers from — a symptom, not the cause.
   Interpretation: with `batch_time = 16` the model fits short windows but learns
   dynamics that integrate unstably over the full trajectory. A **stiff forward
   integration** problem, localized to the stiff systems (duffing,
   periodic_predator_prey — the only systems producing divergence).

**Why gradient clipping (and the documented factorial clip axis) is rejected:**
training already converges, so an optimization-level fix cannot help. Empirically
`clip_norm = 1.0` never fires (`fire_frac_med = 0` on every cell/wiring) and,
where it does anything on the unstable cells, it slightly worsens divergence
(`ltc/ncp` 7 %→10 %, `mm_ltc/ncp` 17 %→20 %). A clip sweep would mostly confirm a
null result. The Phase-3b "numeric vs closed-form LRC" factorial is also not
buildable: no closed-form LRC (`cfc_lrc`) exists in the codebase (no cell, no
derivation doc), so that axis has no second arm.

## 2. Design

A focused probe testing whether the LTC stiffness divergence is fixable at the
**solver / training-horizon level**, plus statistically solid divergence rates.

| Block | Cells | Wirings | Systems | Seeds | Intervention | Runs |
|-------|-------|---------|---------|-------|--------------|------|
| 1 Robustness | ltc, mm_ltc, lrc, mm_lrc | dense, ncp | all 6 | 5–14 (10 new) | baseline (`ode_unfolds=6`, `batch_time=16`) | 480 |
| 2 Solver fidelity | ltc, mm_ltc | dense, ncp | duffing, periodic_predator_prey | 0–14 (15) | `ode_unfolds=24` | 120 |
| 3 Training horizon | ltc, mm_ltc | dense, ncp | duffing, periodic_predator_prey | 0–14 (15) | `batch_time=64` | 120 |

**Total: 720 runs** → `results/runs_v3`.

- Block 1 + the v1/v2 seeds 0–4 give **15 seeds** of baseline for all four ODE
  cells → divergence rate becomes a first-class metric (n=5 was too few for the
  heavy tails; mean and median disagreed by 10×).
- Block 2/3 baselines for the paired comparison come from v1/v2 (seeds 0–4) plus
  Block 1 (seeds 5–14), so aggregation pools `results/runs` + `results/runs_v2` +
  `results/runs_v3`.
- `ode_unfolds=24` (vs LTC default 6) makes the cell's internal semi-implicit
  Euler 4× finer during both training and the eval rollout — the direct
  forward-stiffness lever. `batch_time=64` (vs 16) forces the model to learn
  dynamics stable over a 4× longer horizon — the orthogonal lever.

## 3. Implementation

All changes are additive; v1 (240) and v2 (480) SLURM array contracts stay
byte-identical (regression-tested).

- **`experiments/run_benchmark.py`**
  - Constants `CELLS_V3`, `CELLS_V3_STIFF`, `SYSTEMS_V3_STIFF`, `SEEDS_V3_EXTRA`,
    `SEEDS_V3_FULL`, `V3_ODE_UNFOLDS=24`, `V3_BATCH_TIME=64`.
  - `build_specs_v3()` concatenates the three blocks in fixed order (= the v3
    array contract): `[0,480)` robustness, `[480,600)` solver, `[600,720)` horizon.
    Intervention specs carry an extra key (`ode_unfolds` or `batch_time`); v1/v2
    specs never do.
  - `build_model(cell, wiring, ode_unfolds=None)` — threads `ode_unfolds` into the
    cell kwargs (reaches `LTC_Cell` directly and `MM_LTC_Cell`'s inner LTC via
    `MixedMemoryCell._inner_kwargs`).
  - `run_one` reads `spec['ode_unfolds']` and `spec.get('batch_time', cfg)`,
    applies the `batch_time` override to `train(...)`, and records both effective
    values in `result.config` (`ode_unfolds` only when set).
  - `result_filename` appends `_unfolds<n>` / `_bt<n>` so the three arms never
    collide (verified: 720 unique filenames, 0 clashes with v1/v2).
  - `--profile` gains `v3`; `main` dispatch + default outdir `results/runs_v3`.
- **`experiments/aggregate_results.py` + `plot_results.py`** — `load_runs` appends
  `+unfolds<n>` / `+bt<n>` to the cell variant label when `config.ode_unfolds≠6`
  / `batch_time≠16`. v1/v2 labels (`<cell>`, `<cell>+clip`) unchanged; schema-1
  path unchanged. `aggregate_results` also emits `ode_unfolds` + `batch_time`
  CSV columns.
- **`tests/experiments/test_run_benchmark.py`** — `test_v3_matrix_counts`,
  `test_v3_spec_order_is_deterministic`, `test_v1_v2_contracts_unchanged_by_v3`,
  `test_result_filename_v3_suffixes`, `test_run_one_v3_ode_unfolds_threaded`,
  `test_run_one_v3_batch_time_override`. Full suite green (20 passed).

## 4. Cluster execution

`./cluster/submit_benchmark_v3.sh` forwards `--profile v3` and lowers GPU packing
for the heavy cells: `CLUSTER_RUNS_PER_GPU=6`, `CLUSTER_MEM=64G`,
`CLUSTER_TIME=08:00:00` (all overridable). The wrapper exists because v3 is
dominated by `ltc`/`mm_ltc` (LTC ~2× LRC, ode_unfolds=24 a further ~4×) and
`mm_ltc/ncp` OOMs at the default 12/GPU. At 6 runs/GPU × 8 GPUs = 48 concurrent
runs; 720 runs → 120 array tasks → ~15 throttle rounds. Estimate ~4–7 h wall,
weighted by the unfolds-24 bundles.

Result fetch + re-aggregation pool all three dirs:

```
./cluster/fetch_results.sh
uv run python experiments/aggregate_results.py \
    --runs results/runs results/runs_v2 results/runs_v3 --out results
uv run python experiments/plot_results.py \
    --runs results/runs results/runs_v2 results/runs_v3 --out results/figures
```

## 5. Analysis plan

- **Divergence rate** per cell × wiring × system (fraction NRMSE > 1.0) at 15
  seeds, baseline vs each intervention.
- **Paired Wilcoxon** (α=0.05, paired on system×seed, Cohen's d) per cell ×
  wiring × stiff system: baseline vs `+unfolds24`, baseline vs `+bt64`, restricted
  to the `{ltc, mm_ltc}` comparison family (small Bonferroni m).
- **Rollout vs training loss** to confirm the interventions act on the forward
  side (training loss already low; the eval NRMSE is the target).

## 6. v3 → v4 decision logic

- If **`ode_unfolds=24` removes the divergence** → the pathology is forward
  integration fidelity. v4 scales finer integration across all cells/systems and
  tests an implicit/adaptive solver and a stiffness penalty.
- If **`batch_time=64` removes it but solver fidelity does not** → it is a
  learning-horizon problem. v4 adds scheduled sampling / multiple-shooting
  training and a horizon sweep.
- If **neither helps** → the stiffness is intrinsic to the learned LTC dynamics
  under Euler rollout. v4 escalates to the outer solver (`euler_odeint` →
  adaptive/implicit) and multiple-shooting, and the LTC family is reported as
  rollout-unstable on stiff systems regardless of the fixes tried.

A null result is itself a finding: it locates the LTC limitation in forward
integration of stiff dynamics rather than in the gradient path.
