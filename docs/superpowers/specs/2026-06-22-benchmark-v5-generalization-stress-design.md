# Benchmark v5 — Generalization Stress Test (Design)

**Date:** 2026-06-22
**Status:** implemented, cluster run pending
**Profile:** `--profile v5` → `results/runs_v5/`
**Predecessor:** v4 (cross-family generalization + classical championship, `results/runs_v4`)

## Motivation

v1–v4 train and evaluate on the **same single trajectory**: one initial condition
`y0`, one regular time grid, the full horizon, the same Euler solver for training
batches and the final rollout. benchmark-findings §8 flags this explicitly as the
**Q2 caveat**: on a clean-fit harness the continuous-time advantage is
near-tautological, because "predict the same trajectory you fit" rewards any model
that simply memorised the field along that one path.

Two independent sources name the same gap:
- **Deep-research review K15** (`benchmark-repo-roadmap.md`): the benchmark list has
  *no explicit robustness protocol*; add a **robustness rail** (Noise /
  sampling-irregularity / domain-shift) — the reports agree this is what actually
  *differentiates* the liquid models.
- **benchmark-findings §10**: the open checklist item "v4b — generalization
  stressor (observation noise + extrapolation horizon + OOD initial conditions);
  `datasets.py`-only change, no new cells."

v5 is that rail. It turns the clean-fit Q2 into a real generalization claim:
*do the repaired bio cells stay ahead when train ≠ eval?*

## Core idea: decouple the train and eval trajectories

The clean path trains on `(t, y)` and evaluates the rollout against the same
`(t, y)`. v5 keeps the **identical training config** (units, NCP 16/8/2 seed 42,
elastance_type, 2000 iters, batch_time 16, lr 1e-3, Euler) and changes **only the
data**, along three independent axes:

| Regime | Train sees | Eval rolls out against | Isolates |
|---|---|---|---|
| `noise` | targets + Gaussian noise (σ = 10 % of each dim's trajectory std) | the **clean** trajectory from `y0` | robustness to observation noise |
| `extrapolation` | first **50 %** of the trajectory | the **full** horizon from `y0` (tail = unseen time) | temporal extrapolation of the learned field |
| `ood_init` | the canonical trajectory from `y0` | the **true** trajectory from a perturbed `y0'` (±20 % of each dim's std) | state-space generalization off the training path |

Parameters live as named constants in `src/tasks/neural_ode/datasets.py`
(`STRESS_NOISE_LEVEL=0.1`, `STRESS_TRAIN_FRACTION=0.5`, `STRESS_OOD_SCALE=0.2`).
The stress realization depends only on `(system, regime, seed)` via
`np.random.default_rng(seed)` — never on the cell — so every cell sees the
**identical** stressed data for a given `(system, seed)`, keeping the paired
Wilcoxon design valid both cell-vs-cell (within v5) and v5-vs-v4 (stress-vs-clean).

The clean baseline is **v4 itself** (same config, no stress): no clean re-run is
needed, and `aggregate_results.py` pairs `ltc` (from `runs_v4`) against
`ltc+noise`/`ltc+extrapolation`/`ltc+ood_init` (from `runs_v5`) on `(system, seed)`.

## Matrix

```
per regime: CELLS_V5 (= CELLS_V4, 11) x WIRINGS (2) x SYSTEMS (6) x SEEDS (5) = 660
3 regimes (noise, extrapolation, ood_init)                                   = 1980 runs
```

Cells (11, unchanged from v4): `ltc, cfc, mm_ltc, cfc_mm_ltc, lrc, cfc_lrc,
mm_lrc, cfc_mm_lrc, gru, lstm, ctrnn`. **No new cells** — the stress lives entirely
in `datasets.py`.

SLURM array contract (`build_specs_v5`): regime-major blocks of 660, standard
`build_specs` cell-major order within each block:
```
[0, 660)     noise
[660, 1320)  extrapolation
[1320, 1980) ood_init
```
Filenames carry a `_stress-<regime>` suffix
(`cfc_mm_ltc_ncp_duffing_seed0_stress-ood_init.json`) so v5 never collides with v4.

n = 5 seeds × 6 systems = **30 paired samples** per cell × wiring × regime,
matching the v1–v4 standard. No tail-seed supplement: v5's question is robustness
ordering, not the divergence-tail rate that motivated v4's extra seeds.

## Hypotheses (to be tested, not assumed)

- **H-v5-1 (robustness ordering).** The closed-form bio cells (`cfc`, `cfc_lrc`,
  `cfc_mm_lrc`, `cfc_mm_ltc`) degrade **less** under stress than the numerical bio
  cells (`ltc`, `lrc`) and the classical baselines (`gru`, `lstm`) on ncp — i.e.
  the v4 ncp ranking survives the stressors.
- **H-v5-2 (the real Q2).** If the bio advantage over `gru`/`lstm` *grows* (or at
  least holds) under `extrapolation`/`ood_init`, the v4 Q2 claim is no longer
  clean-fit-tautological: the liquid/closed-form mechanism generalizes where the
  classical gate does not.
- **H-v5-3 (per-axis attribution).** The three regimes need not rank cells the
  same way. `ood_init` (state-space) and `extrapolation` (time) probe the learned
  *field*; `noise` probes optimization robustness. Reporting them separately is the
  point — a single combined stressor would confound the attribution.

A genuinely informative **negative** result is in scope: if the classical cells
match the bio cells under stress, the v4 advantage *was* largely clean-fit, and
that belongs in the thesis honestly (feeds RQ1/H1).

## Evaluation

`aggregate_results.py --runs results/runs_v4 results/runs_v5` (the `stress` suffix
makes every regime its own variant). Primary metric unchanged: full-trajectory
NRMSE, paired Wilcoxon over (system, seed) + paired Cohen's d, divergence rate
(share NRMSE > 1) and mean as the robustness summary (not median alone). Key
comparisons: each cell `+regime` vs its clean v4 self (degradation), and bio vs
classical within each regime on ncp (the H-v5-2 test).

## Cluster

`cluster/submit_benchmark_v5.sh` — mirrors v4's heavy-cell packing (6 runs/GPU,
64 G, 10 h) because v5 carries the same slow `ltc`/`mm_ltc` ncp cells and the
stress regimes do not change per-run cost. 1980 runs → ceil(1980/6) = 330 array
tasks, %8 GPU throttle → ≤ 48 concurrent runs. Pre-flight a single heavy canary
(`mm_ltc/ncp/duffing/ood_init`) for OOM/timing before releasing the array.

## Scope / honesty

- Noise is on the **observations**, not the dynamics; the eval target is the clean
  (or true-from-`y0'`) trajectory, so NRMSE measures recovered dynamics, not noise
  fitting.
- `extrapolation` keeps `n_iters`/`batch_time` fixed, so the *only* change vs clean
  is the shorter training grid — per-run cost is unchanged.
- Single NCP wiring seed (42), as in v1–v4: an open confounder already tracked in
  benchmark-findings §10 (multi-wiring-seed robustness is a separate item).
- Parameters (0.1 / 0.5 / 0.2) are deliberate moderate values: strong enough to
  bite, mild enough to preserve discrimination between cells. They are named
  constants, easy to sweep if the first run shows a floor/ceiling effect.
