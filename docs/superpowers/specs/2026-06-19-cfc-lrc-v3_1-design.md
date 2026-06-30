# Benchmark v3.1 — Closed-form LRC (`cfc_lrc`) cell: design + ablation

**Date:** 2026-06-19
**Status:** design / pre-implementation
**Profile:** `v3.1` (new), output dir `results/runs_v3_1`
**Predecessor context:** v1/v2/v3 (see `2026-06-10-benchmark-v2-fixed-cells-design.md`,
`2026-06-14-benchmark-v3-rollout-stability-design.md`).

---

## 1. Motivation

The repo has a numerical `lrc` cell (Liquid Resistance Liquid Capacitance, Farsang
et al., arXiv:2403.08791) and a closed-form `cfc` cell (Hasani et al. 2022,
arXiv:2106.13898). CfC is the closed-form approximation of the **LTC** ODE. There is
**no closed-form LRC**: `cfc_lrc` does not exist. That gap blocks the planned
factorial LRC matrix (Phase 3b) and is the reason this interlude (v3.1) is run
*before* v4: implement a closed-form LRC cell and benchmark whether the closed form
helps relative to plain CfC and to the numerical LRC.

## 2. Research result: one cell, not two routes

Two derivation routes were investigated and adversarially verified (math /
implementation / thesis-validity lenses):

- **Route A — principled:** apply Hasani's closed-form recipe (freeze coefficients →
  integrating factor → replace the exponential decay by a bounded learned gate)
  directly to the LRC ODE.
- **Route B — pragmatic:** keep the CfC architecture and warp the gate's effective
  elapsed time by a liquid-elastance factor.

**Verified finding:** the two routes specify the *same neural network*. By
associativity, `sigmoid(eps*(t_a*dt)+t_b) == sigmoid(t_a*(eps*dt)+t_b)` (checked
numerically, diff < 1e-12). There is no A-vs-B implementation choice. Route A
supplies the principled justification; Route B supplies the honest scoping. We ship
**one cell** with Route A's derivation and Route B's epistemic humility.

## 3. Derivation (Route A, concise)

LRC ODE (per neuron, matching `lrc_cell.py` `forget_gate=True`):

```
dh/dt = eps(u,h) * ( -f(u,h) * h + b(u,h) )
  eps = sigmoid(W_e[u,h] + p)            liquid elastance in [0,1]   (asymmetric)
  f   = sigmoid(gleak + sum_j w*syn + sensory_w)   decay rate >= 0
  b   = vleak * tanh(gleak + sum_j h*syn + sensory_h)   drive
```

For `eps == 1` this is exactly the LTC ODE that CfC was derived from.

Freeze `(eps, f, b)` over one step `[0, dt]` (the same approximation Hasani makes for
LTC; Farsang et al. justify a single Euler unfold for LRC via the elastance damping,
Thm. A.2). The rate `a := eps*f` is then constant and the integrating factor closes
the ODE exactly:

```
h(dt) = h0 * exp(-eps*f*dt) + (b/f) * (1 - exp(-eps*f*dt))
```

**Key result (numerically verified to 2e-6 vs fine Euler):** the elastance `eps`
**cancels from the equilibrium** `h_inf = b/f` (identical to LTC) and survives **only
as the decay-rate multiplier** `lambda = eps*f`. Physically right: elastance (= 1/C)
rescales how fast the neuron relaxes toward the *same* equilibrium; it does not move
the equilibrium. This is a genuine statement about LRC that neither source paper
makes, and it is the justification for placing `eps` as a multiplier on the gate's
time-rate.

Rewrite as a two-regime interpolation (Hasani) with gate `g = 1 - exp(-lambda*dt)`:

```
h(dt) = h0*(1-g) + h_inf*g,   lambda = eps*f
```

Replace the exponential by Hasani's bounded learned sigmoid gate (peak gradient 0.25,
bounded for any `dt`), keeping `eps` as the rate multiplier, and generalise the two
regimes to free Dense heads (Hasani step 7):

```
g = sigmoid( eps * (t_a * dt) + t_b ),   h' = ff1*(1-g) + g*ff2
```

This is plain CfC plus exactly one bounded multiplicative elastance factor on the
gate time-rate. `eps -> 1` recovers plain CfC exactly.

## 4. Cell design (`CfC_LRC_Cell`)

Clone of `cfc_cell.py`, single structural addition:

```
x      = backbone(concat([inputs, state]))     # shared, lecun_tanh (as CfC)
ff1    = Dense_ff1(x); ff2 = Dense_ff2(x)
t_a    = Dense_ta(x);  t_b = Dense_tb(x)
e_pre  = elastance_mapping(concat([inputs, state]))   # raw (u,h), as numerical LRC
eps    = sigmoid(e_pre)                                 # asymmetric (default)
       | sigmoid(e_pre + k) - sigmoid(e_pre - k)        # symmetric, k = distr_shift >= 0
g      = sigmoid( eps * (t_a * elapsed_time) + t_b )
h'     = ff1*(1 - g) + g*ff2
return h', [h']
```

Decisions (each with rationale):

- **Elastance input domain = raw `concat([inputs, state])`**, *not* the backbone
  features, so the head is the identical formula the numerical `lrc` cell uses
  (`elastance_mapping`) and stays interpretable as liquid elastance for the
  comparison. The four CfC heads stay on the shared backbone.
- **`eps` multiplies `t_a*elapsed_time`, not `t_b`** — faithful to the derived
  exponent `lambda = eps*f` (`t_b` is the gate bias). `elapsed_time` is kept (not
  hard-coded to 1) so the cell stays correct under future irregular sampling.
- **Warm-start `eps` high** via `elastance_init_bias` (default 1.0 → eps≈0.73, healthy
  sigmoid gradient): training begins near plain CfC and the optimiser must actively
  introduce elastance. Makes `cfc` vs `cfc_lrc` a clean nested-model comparison.
- **No `_ode_solver`, no `ode_unfolds`** — single forward pass, like CfC.
- **Default `elastance_type='asymmetric'`** (matches `lrc`/`mm_lrc` CELL_KWARGS and
  the RQ2/RQ5 focus; monotone, `eps↓` ⇒ slower relaxation). Symmetric supported for
  completeness but has inverted monotonicity — do not mix interpretations.

Constructor: `CfC_LRC_Cell(units, backbone_units=None, backbone_layers=1, dt=1.0,
elastance_type='asymmetric', elastance_init_bias=1.0, **kwargs)`.

## 5. Honest scoping (must hold in the thesis text)

1. **Not an exact closed form of the LRC ODE.** Closed-form only under the per-step
   frozen-coefficient approximation (order `O(dt)`, the *same status* as CfC for LTC
   and Farsang's 1-unfold Euler). The true `∫ eps*f` has no closed form. Do not write
   "exact" / "faithful solution" / "the LRC trajectory".
2. **Redundancy risk.** Because `t_a` is a free learned head, `eps*t_a` is
   representable by a plain CfC's free `t_a'`; in isolation the gate function class is
   unchanged. The `eps` head adds a **bounded [0,1] inductive bias + capacity**, not a
   new function class. Whether the bias helps is **empirical** → the parameter-matched
   ablation is mandatory, not optional.
3. **Modeling ceiling.** Under freezing `eps` cancels from the equilibrium, so the
   cell expresses elastance *only* as timescale modulation, never equilibrium
   reshaping. Faithful to the ODE, but a hard expressivity limit; state it.
4. **`eps→0` is not a hard freeze.** At `eps=0` the gate is `sigmoid(t_b)` (~0.5 by
   default), not 0. A true hold needs `t_b << 0`. Do not claim a freeze regime without
   a negative `t_b` bias.

## 6. Validation gates (all must pass before any cluster run)

1. **Reduction sanity:** with `eps` forced to 1 (constant), `CfC_LRC_Cell` outputs
   match `CfC_Cell` to numerical tolerance on identical weights/inputs.
2. **Forward-pass shapes:** dense (units=16) and NCP (16/8/2) build and produce
   `(batch, T, 2)`; `eps ∈ [0,1]`.
3. **Param-count check:** `params(cfc) < params(cfc_lrc) <= params(cfc_pm)`; set
   `cfc_pm` backbone width so plain CfC's dense param count ≥ `cfc_lrc`'s.
4. **Gradient-flow check** (reuse `GradientFlowTracker`): the extra `eps` multiply does
   not reintroduce vanishing gradients vs plain CfC.
5. **Trajectory NRMSE vs numerical `lrc`** (elastance matched, `ode_unfolds=1`): report
   a number; "close to numerical LRC" must be quantified, not asserted.

## 7. v3.1 benchmark design

Focused ablation (not a broad matrix). Cells, all `x {dense, ncp} x 6 systems x
seeds 0–4`:

| Cell | Role |
|------|------|
| `cfc` | baseline closed-form (LTC-derived) |
| `cfc_lrc` | the new closed-form LRC cell (elastance head) |
| `cfc_pm` | parameter-matched plain CfC (wider backbone) — capacity control |
| `lrc` | numerical LRC reference (what `cfc_lrc` approximates) |

Run count: 4 cells × 2 wirings × 6 systems × 5 seeds = **240 runs** →
`results/runs_v3_1`. Metric: full-trajectory NRMSE (as v1–v3), paired by seed.

Ablation logic:
- `cfc_lrc` vs `cfc`: does the LRC-flavoured closed form differ in performance?
- `cfc_lrc` vs `cfc_pm`: is any gain due to the **elastance inductive bias** rather
  than the extra parameters? (`cfc_pm` is over-provisioned capacity → conservative.)
- `cfc_lrc` vs `lrc`: how close does the closed form get to the numerical LRC, at a
  single forward pass instead of an Euler step?

NCP param-matching is approximate (a single `backbone_units` cannot match all three
NCP layers exactly; dominated by the inter layer). Documented, not hidden.

## 8. Integration points

- `src/neurons/cfc_lrc_cell.py` (new).
- `src/neurons/__init__.py`: import + `__all__`.
- `src/models/rnn_model.py`: import; `_CELL_REGISTRY` gets `'cfc_lrc': CfC_LRC_Cell`
  and `'cfc_pm': CfC_Cell` (alias; param-matched via CELL_KWARGS).
- `experiments/run_benchmark.py`: `CELLS_V3_1`, `build_specs_v3_1`, `--profile v3.1`,
  outdir `results/runs_v3_1`, CELL_KWARGS (`cfc_lrc`: elastance_type; `cfc_pm`:
  backbone_units=K), `--cell` choices extended.
- Tests: `tests/neurons/test_cfc_lrc_cell.py`; extend
  `tests/experiments/test_run_benchmark.py`.

## 9. Open questions (for the thesis, not blockers)

- Does the elastance head beat parameter-matched CfC at all, or is it redundant on
  these tasks? (The headline empirical question.)
- Gate-sign / orientation cross-check of `f` against arXiv:2403.08791 before binding
  any analytic-mode baseline (deferred: analytic gate variant not in v3.1 scope).
- `eps` distribution over training (collapse / dead-unit monitoring, esp. symmetric).
