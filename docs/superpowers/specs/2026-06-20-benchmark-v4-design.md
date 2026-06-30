# Benchmark v4 — Cross-family generalization of the architecture fixes + classical championship

**Date:** 2026-06-20
**Status:** launched (dataLAB job 488078, `--profile v4` → `results/runs_v4`)
**Predecessor context:** v1 (thesis matrix), v2 (fixed cells + clip), v3 (rollout
stability), v3.1 (closed-form LRC `cfc_lrc`), v3.2 (LRC 2x2), v3.3 (param-matched
`lrc_pm`). See the respective design docs.
**Process:** designed autonomously; the design was put through a 4-lens adversarial
panel (thesis-RQ / mechanistic / feasibility / literature) + a judge before launch.

---

## 1. Where v1–v3.3 left us (grounded in `results/`)

Full-trajectory NRMSE (Euler rollout), median dense / median ncp, divergence% =
share with NRMSE>1, all at identical config (n_iters=2000, batch_time=16, units=16
dense, NCP 16/8/2 seed 42, LRC-family `elastance_type='asymmetric'`):

| Cell | dense | ncp | ncp div% | note |
|------|-------|-----|----------|------|
| gru | 0.049 | 0.172 | 0 | classical baseline |
| lstm | 0.051 | 0.222 | 0 | classical baseline |
| ltc | 0.032 | 0.060 | 6.7 | great median, unstable tail (mean 0.30) |
| lrc | 0.068 | 0.311 | 3.3 | catastrophic on ncp (mean 0.38) |
| cfc (= closed-form LTC) | 0.030 | 0.069 | 0 | stable |
| cfc_lrc (closed-form LRC) | 0.017 | 0.070 | 0 | one dense outlier |
| mm_ltc | 0.022 | 0.065 | **16.7** | great median, **catastrophic tail** (mean 0.50) |
| mm_lrc | 0.024 | 0.090 | 0 | stable |
| cfc_mm_lrc | 0.019 | **0.051** | 0 | best + most robust LRC cell |
| cfc_pm / lrc_pm | capacity controls | | | capacity alone does **not** fix numerical LRC on ncp |

**Established narrative.** On the sparse 3-layer NCP wiring the numerical liquid
cells degrade. Two architectural changes each repair the LRC independently:
(a) closed-form (CfC's bounded sigmoid gate, +172 params) and (b) mixed-memory
(an LSTM additive memory path `c` that bypasses the ODE-solver Jacobians, +1216
params; Lechner & Hasani 2020). They are complementary — `cfc_mm_lrc` (both) is best
and most robust. Param-matched controls (`cfc_pm`, `lrc_pm`) show the gains are
**mechanistic, not capacity**.

**The gaps v4 closes.**
1. The entire 2x2 story was run **only on LRC**. The LTC family has `ltc`,
   `cfc` (= closed-form LTC), `mm_ltc` — but **`cfc_mm_ltc` never existed**. And the
   families behave *differently*: `mm_ltc` has a 16.7% ncp divergence tail while
   `mm_lrc` has 0%. Does the combined fix transfer?
2. The fixed bio cells were **never put head-to-head against classical gru/lstm in
   one clean self-contained matrix** (the thesis core RQ).
3. `ctrnn` (continuous-time RNN baseline) is in the registry but was **never run**.

## 2. Design — one clean self-contained matrix

`results/runs_v4`, identical config to v3.2, **only the cell architecture differs**.

**Cells (11).** Two cross-family 2x2s + classical/CT baselines:

| | numerical | closed-form |
|--|-----------|-------------|
| **plain** (LTC) | `ltc` | `cfc` |
| **mixed-memory** (LTC) | `mm_ltc` | **`cfc_mm_ltc`** (new protagonist) |
| **plain** (LRC) | `lrc` | `cfc_lrc` |
| **mixed-memory** (LRC) | `mm_lrc` | `cfc_mm_lrc` |

Baselines: `gru`, `lstm` (classical discrete), `ctrnn` (vanilla continuous-time —
isolates "ODE bias" from "LTC/LRC-specific bias"). `ctrnn` is an **extra baseline,
not a 2x2 member** (it has no closed-form / mixed-memory partner).

**New code.** `cfc_mm_ltc = MixedMemoryCell(CfC_Cell)` — mirrors
`cfc_mm_lrc = MixedMemoryCell(CfC_LRC_Cell)` exactly. CfC inner takes **no
`elastance_type` kwarg** (it raises `TypeError`), so `cfc_mm_ltc` and `ctrnn`
deliberately get **no `CELL_KWARGS` entry** — the one trap that would crash every
run of those cells. Everything else is boilerplate (`CELLS_V4`, `build_specs_v4`,
`--profile v4`, outdir, `--cell` choices) mirroring v3.2/v3.3.

**Seed scheme.** Seeds 0–4 on all 6 systems (n=5; pooled n=30 per cell × wiring),
PLUS seeds 5–9 on the only two divergence-producing systems (`duffing`,
`periodic_predator_prey` = `SYSTEMS_V3_STIFF`) for all cells × both wirings. A ~17%
divergence rate over n=5/system has a ~±13 pp CI — too wide to tell a real residual
tail from zero. Doubling seeds *only* where divergence actually occurs sharpens the
headline Q1 claim cheaply. Paired tests stay valid (every cell shares the seed
scheme). **Total: 660 base + 220 tail = 880 runs.**

## 3. Headline questions

- **Q1 (primary, well-powered):** Does the closed-form + mixed-memory fix that
  repairs LRC on ncp **generalize to LTC** — does `cfc_mm_ltc` eliminate `mm_ltc`'s
  ~17% ncp divergence tail, the way `cfc_mm_lrc` gives `mm_lrc` 0%?
- **Q2 (core thesis RQ):** Do the architecture-fixed bio cells match/beat classical
  `gru`/`lstm`/`ctrnn` on ncp — and, honestly, do they match-or-beat the classical
  **dense** ceiling (the real classical reference), not only classical ncp?
- **Q3 (robustness):** Reported as **divergence rate (share NRMSE>1) + mean +
  per-system breakdown**, not median alone.

## 4. Honest scoping (must hold in the thesis text)

1. **Contribution framing.** "Closed-form fixes the vanishing gradient" and
   "mixed-memory gives a constant-error memory path" are the **established** results
   of Hasani et al. 2022 (CfC) and Lechner & Hasani 2020 (ODE-LSTM). v4's novelty is
   the **cross-family transfer** to the LRC family (whose `cfc_lrc`/`mm_lrc`/
   `cfc_mm_lrc`/`cfc_mm_ltc` variants are this thesis's own constructions) and the
   **composition** (`cfc_mm_*`) plus the divergence-rate robustness lens. Do not sell
   Q2 as the scientific novelty.
2. **Mechanism language.** Distinguish **vanishing** (plain numerical cells on deep
   NCP) from **forward/exploding instability** (the `mm_ltc` NRMSE>1 tail — a >1
   blow-up is the *opposite* failure mode). Do not file both under "vanishing
   gradient".
3. **Tautology / external validity of Q2.** The harness trains and evaluates on the
   *same* single clean trajectory from one fixed `y0` on a regular grid, rolled out
   by the same Euler integrator that generated the target. On that task a
   continuous-time cell is structurally favored over gru/lstm; "bio beats classical
   on ncp" is near-tautological. v4 answers Q2 only in this weak clean-fit regime.
   The generalization regime (noise / extrapolation / OOD-IC) is **deferred to v4b**
   (see §6) and is what turns the headline RQ from a fit claim into a generalization
   claim.
4. **Single NCP wiring seed (42).** All ncp conclusions are conditional on one sparse
   graph. Accepted for v4 (isolates init/batch variance); multi-wiring deferred.
5. **`cfc_lrc` near-degeneracy.** Per the v3.1 derivation, elastance cancels from the
   equilibrium and survives only as a timescale multiplier (`eps→1` recovers CfC), so
   the closed-form-LRC axis may add little signal beyond `cfc`. A learned-`eps`
   distribution probe is a deferred diagnostic.

## 5. Cluster config (grounded)

v4 contains the slow numerical-LTC cells. Measured wall times (repo history):
`ltc/ncp` median ~2.2 h, `mm_ltc/ncp` median ~2.7 h, **max ~7.3 h** per run; these
OOM at the config.sh default 12 runs/GPU @ 32G (see `backfill_array.sbatch`,
`backfill_v3_ncp.sbatch`). v4 uses **baseline** config (no `ode_unfolds=24` /
`batch_time=64` 4x-cost arms), so the v3 main-sweep packing is the proven-safe level.
`cluster/submit_benchmark_v4.sh` sets **6 runs/GPU, 64G, 10 h** walltime (10 h for
the 7.3 h-max margin). 147 array tasks, throttle 8 → 48 concurrent runs.
Pre-flight verified: first wave (incl. heavy `ltc` cells) trained healthily, no OOM,
no errors.

## 6. Deferred (committed next, not open-ended)

- **v4b — generalization stressor** (the highest-value gap): extrapolation-horizon
  train/eval split (train on `t ∈ [0, T_train]`, evaluate the held-out tail) + a
  fixed-SNR observation-noise condition. Pure `datasets.py` + trainer change (t_span
  / y0 already parameterized), no new cells. This is the experiment that supplies the
  "why bio beats classical".
- OOD-initial-condition eval (perturb `y0` at test); bundle into v4b.
- Backbone-matched control (`cfc`/`cfc_lrc` with `backbone_layers=0`) to isolate the
  closed-form axis from the MLP-frontend.
- Matched-solver controls (LRC at `ode_unfolds=6`, LTC at `ode_unfolds=1`) to
  decouple "family" from "integrator scheme".
- Per-step gradient-norm / forward-state-norm logging (`src/evaluation/gradient_flow.py`)
  to *measure* vanishing vs exploding.
- Multiple NCP wiring seeds (≥3) for a wiring-class claim.
- **STC**: dynamics are undefined in the literature and unimplemented — resolve in
  the thesis text (define + cite, or scope as future work and adjust the title).
- LTC-specific `cfc_pm`-style capacity control: low priority — the existing `lrc_pm`
  (3294/4620 params) already exceeds `cfc_mm_ltc` (2642/3794) and already shows
  capacity does not rescue the numerical cell on ncp.

## 7. Analysis to add (zero re-run, do while cluster runs)

`aggregate_results.py` currently emits only mean±std tables and the median-diff
Wilcoxon. Add: **divergence rate (share NRMSE>1)** as the pre-registered primary
robustness metric, **mean NRMSE**, and a **per-system breakdown**. Pre-register the
named contrasts (`cfc_mm_ltc` vs `mm_ltc` on ncp divergence [Q1]; each fixed bio cell
vs `gru`/`lstm` on ncp [Q2]; the LTC 2x2 internal pairs) with multiplicity
correction. Add a post-array **completeness assertion** (every (cell,wiring,system,
seed) JSON present) — `_paired_vectors` silently drops unpaired cells, so a dropped
run would quietly shrink n.
