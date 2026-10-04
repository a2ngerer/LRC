# cNCP — Cortically-Informed Neural Circuit Policy: Design Specification

**Date:** 2026-07-02
**Status:** design → implementation
**Scope:** a new sparse recurrent wiring variant for the benchmark's RNN cells.

---

## 0. What this is (and is not)

The cNCP is a **sparse connectivity pattern for a recurrent neural network**, in
the exact sense that the existing `NCPWiring` (`src/wirings/ncp.py`) is. It is a
directed graph whose nodes are small standard RNN sub-cells and whose edges are
fixed binary connectivity masks applied to learnable weight matrices
(`SparseLinear`, i.e. `y = x @ (W ⊙ mask)`). Nothing here is a simulation of a
biological system: the node names (`L4`, `L23`, `L5IT`, `L5ET`, `L6CC`, `L6CT`,
`Thal`, `TRN`) are **labels for nodes in a computation graph**. The whole object
is tensor algebra — matrix multiplies, elementwise gates, and a hidden-state
carry — trained by backprop like any other Keras layer.

The design is *inspired by* the connectivity motifs of the canonical cortical
microcircuit, the same way the standard NCP is inspired by the *C. elegans*
connectome. "Inspired by" means: the fixed sparsity structure and a multiplicative
gain edge are copied as an inductive bias. The claim under test is purely an ML
claim — *does this particular fixed sparsity + gain structure improve a recurrent
policy on tasks that need top-down gain / temporal fill-in, beyond what the same
parameter count buys in a dense or randomly-rewired graph?*

### Relation to the standard NCP

| | `NCPWiring` (existing) | `cNCP` (this spec) |
|---|---|---|
| Graph source | `ncps.wirings.NCP` (worm connectome prior) | fixed laminar mask inventory (this doc) |
| Layers | 3 RNN layers: inter → command → motor | 6 RNN sub-cells + 2 linear relay nodes |
| Topology | acyclic feedforward chain | cyclic: feedback + lateral + re-entrant relay loop |
| Keras container | `tf.keras.Sequential` (no cycles possible) | one composite `AbstractRNNCell` (state carries the cycles) |
| Edge combiner | additive only (`SparseLinear`) | additive **plus** one multiplicative gain edge |
| Cell dynamics | pluggable `cell_cls` | pluggable `cell_cls` (**identical orthogonality**) |

**Orthogonality principle (unchanged from NCP):** recurrence and continuous-time
dynamics live *inside* each sub-cell (`cell_cls`, e.g. `LTC_Cell`, `LRC_Cell`,
`CfC_LRC_Cell`, `GRU_Cell`); the cNCP contributes *only* the sparse inter-node
mask inventory and the gain combiner. Swapping `cell_cls` must not touch the
wiring. This is what lets the benchmark separate a wiring effect from a cell
effect (the cell-orthogonality control, §9).

---

## 1. Why `Sequential` cannot express it

`NCPWiring.build_model()` returns a `tf.keras.Sequential` of `RNN → SparseLinear
→ RNN → …`. A `Sequential` is a DAG; it cannot express top-down feedback
(`deep → superficial`), lateral edges, or the `L6CT → Thal → L4` re-entrant loop.
The cNCP therefore lives in **one composite `AbstractRNNCell`** whose hidden state
is the concatenation of all node states. `tf.keras.layers.RNN` unrolls it over
time; cycles are broken by reading feedback edges from the **previous timestep's**
stored state (delayed-state approximation — the same trick ODE-LSTM /
`MixedMemoryCell` use to compose sub-cells, and a defensible stand-in for the
slower biological feedback loops). Feedforward edges within one timestep are
evaluated in a fixed topological order and see the current step's freshly computed
values.

---

## 2. Nodes

Eight nodes. Six are RNN sub-cells (one `cell_cls` instance each, own unit count);
two are lightweight linear relay nodes (a leaky affine map with its own state).
`L1` is **not** a node — it is realised as the multiplicative gain combiner (§5).

| Node | Kind | Role in the graph | Default units |
|------|------|-------------------|---------------|
| `L4`    | RNN sub-cell | input-entry node; receives external input + re-entrant relay | 8 |
| `L23`   | RNN sub-cell | integration/relay node (analogue of NCP *inter*) | 8 |
| `L5IT`  | RNN sub-cell | intracortical relay node | 6 |
| `L5ET`  | RNN sub-cell | **output hub** (fused command+motor); its state is the cell output | 8 |
| `L6CC`  | RNN sub-cell | intracortical feedback source | 4 |
| `L6CT`  | RNN sub-cell | relay-loop driver | 4 |
| `Thal`  | linear relay | re-entrant relay node feeding `L4` (delayed) | 4 |
| `TRN`   | linear relay | sign-negative gate on `Thal` | 2 |

Defaults are **design choices, not measured constants** (§10). They are tunable
and are chosen so the total parameter count is comparable to the standard
`inter=16/command=8/motor=2` NCP for a fair benchmark (verify empirically; add a
parameter-matched control regardless, §9).

`output_size = units[L5ET]`. The model factory appends a `Dense(output_neurons)`
projection, exactly as `make_dense_model(..., output_neurons=…)` does.

`state_size = [units[L4], units[L23], units[L5IT], units[L5ET], units[L6CC],
units[L6CT], units[Thal], units[TRN]]` (a **list** — like `MixedMemoryCell`'s
`[units, units]`; needs a matching custom `get_initial_state`).

---

## 3. Edges (the mask inventory)

Each directed edge is a `SparseLinear(out_units, mask)` with a **fixed binary
mask** of shape `(in_units, out_units)`. Masks are generated once at construction
from a seed at a given **density** ∈ (0,1] (1.0 = dense). A "dense" edge is just
density 1.0 (all-ones mask) — still a learnable `Dense`, kept in the same code
path for uniformity. All densities are hyperparameters.

### 3a. Feedforward edges — read the CURRENT step

| Mask | Edge | Default density | Notes |
|------|------|-----------------|-------|
| `M_in_L4`     | input → L4        | 1.0 (dense) | driver input |
| `M_L4_L23`    | L4 → L23          | 1.0 (dense) | strongest interlaminar edge |
| `M_L23_L5ET`  | L23 → L5ET        | 1.0 (dense) | superficial→output, biased to ET |
| `M_L23_L5IT`  | L23 → L5IT        | 0.5 (sparse)| weaker superficial→IT |
| `M_L5IT_L5ET` | L5IT → L5ET       | 1.0 (dense) | IT→ET dense |
| `M_L5_L6CC`   | [L5IT‖L5ET] → L6CC| 1.0 (dense) | deep readout |
| `M_L5_L6CT`   | [L5IT‖L5ET] → L6CT| 1.0 (dense) | deep readout |

### 3b. Feedback / lateral edges — read the PREVIOUS step (delayed state)

| Mask | Edge | Default density | Notes |
|------|------|-----------------|-------|
| `M_L5ET_L5IT` | L5ET → L5IT       | 0.25 (sparse) | ET→IT low fan-in, **not zero** |
| `M_L6CC_L23`  | L6CC → L23        | 0.5 (sparse)  | weak ascending intracortical feedback |

### 3c. Apical top-down (feeds the MULTIPLICATIVE combiner, §5) — PREVIOUS step

| Mask | Edge | Default density | Notes |
|------|------|-----------------|-------|
| `M_ap_L23`  | [L5ET‖L6CT] → L23 apical  | 0.5 | top-down gain onto L23 |
| `M_ap_L5ET` | [L6CT‖L6CC] → L5ET apical | 0.5 | top-down gain onto the output hub |

### 3d. Re-entrant relay loop — PREVIOUS step for `Thal→L4`, current step inside the loop

| Mask | Edge | Sign | Density | Notes |
|------|------|------|---------|-------|
| `M_L6CT_Thal` | L6CT → Thal | + | 1.0 | modulator drive into relay |
| `M_L6CT_TRN`  | L6CT → TRN  | + | 1.0 | drive into the gate node |
| `M_TRN_Thal`  | TRN → Thal  | **−** (sign-locked) | 1.0 | inhibitory gate (§6) |
| `M_Thal_L4`   | Thal → L4   | + | 1.0 | re-entrant driver into L4 (**delayed**) |

**There is deliberately no `L6CT → L4` excitatory edge.** The deep→entry gain
effect is expressed *only* through the indirect relay loop `L6CT → Thal → L4`
(gated by `TRN`). This is the one hard structural constraint of the design;
adding a direct `M_L6CT_L4` mask would defeat the point of the loop.

### 3e. Optional divisive inhibition (ablatable, default OFF)

`M_L6CT_div_L4`: `L6CT → L4` divisive term (previous step): instead of adding,
it scales L4's drive by `1 / (1 + softplus(SparseLinear(prev.L6CT)))`. Gated by a
constructor flag `divisive_inhibition: bool = False` so the additive baseline is
the default and the divisive variant is a clean ablation.

---

## 4. Per-timestep update

`call(inputs, states)` — `inputs` may be `(x, elapsed_time)` (irregular sampling);
unpack exactly like `CfC_LRC_Cell.call`. `prev = states` (list, previous step).
Every sub-cell is invoked as `subcell.call((basal, elapsed_time), [prev_node])`
so continuous-time cells receive `dt`. `⊕` = additive `SparseLinear`, `‖` =
concat. `combine(h, apical)` is defined in §5.

```
# 0. relay re-entry (DELAYED): Thal→L4 reads prev.Thal
thal_to_L4 = M_Thal_L4(prev.Thal)

# 1. entry node
basal_L4 = M_in_L4(x) ⊕ thal_to_L4
[ if divisive_inhibition: basal_L4 /= (1 + softplus(M_L6CT_div_L4(prev.L6CT))) ]
h_L4, _ = L4.call((basal_L4, dt), [prev.L4])

# 2. integration node (with top-down gain)
basal_L23 = M_L4_L23(h_L4) ⊕ M_L6CC_L23(prev.L6CC)
apical_L23 = M_ap_L23(prev.L5ET ‖ prev.L6CT)
h_L23_raw, _ = L23.call((basal_L23, dt), [prev.L23])
h_L23 = combine(h_L23_raw, apical_L23)

# 3. intracortical relay node
basal_L5IT = M_L23_L5IT(h_L23) ⊕ M_L5ET_L5IT(prev.L5ET)   # ET→IT delayed
h_L5IT, _ = L5IT.call((basal_L5IT, dt), [prev.L5IT])

# 4. output hub (with top-down gain)
basal_L5ET = M_L23_L5ET(h_L23) ⊕ M_L5IT_L5ET(h_L5IT)
apical_L5ET = M_ap_L5ET(prev.L6CT ‖ prev.L6CC)
h_L5ET_raw, _ = L5ET.call((basal_L5ET, dt), [prev.L5ET])
h_L5ET = combine(h_L5ET_raw, apical_L5ET)

# 5. deep readout nodes
deep_in = h_L5IT ‖ h_L5ET
h_L6CC, _ = L6CC.call((M_L5_L6CC(deep_in), dt), [prev.L6CC])
h_L6CT, _ = L6CT.call((M_L5_L6CT(deep_in), dt), [prev.L6CT])

# 6. relay loop (acyclic within the step: L6CT→TRN→Thal)
h_TRN  = relay_TRN(M_L6CT_TRN(h_L6CT),  prev.TRN)
thal_in = M_L6CT_Thal(h_L6CT) − |M_TRN_Thal|(h_TRN)   # sign-locked negative
h_Thal = relay_Thal(thal_in, prev.Thal)

output   = h_L5ET
new_state = [h_L4, h_L23, h_L5IT, h_L5ET, h_L6CC, h_L6CT, h_Thal, h_TRN]
return output, new_state
```

`relay_X(drive, prev)` is a leaky affine update: `h = (1 − α)·prev + α·tanh(drive
+ b)`, with a learnable or fixed leak `α ∈ (0,1]`. Keep it minimal; it only has to
carry and gate the loop.

---

## 5. The multiplicative gain combiner (the "L1" edge)

The single non-additive edge. For a node with an apical top-down input:

```
combine(h_basal, apical) = h_basal * (1 + g * sigmoid(apical))
```

- `sigmoid(apical) ∈ (0,1)` is the top-down modulation.
- `g` is a learnable gain (scalar or per-unit `Dense`-free weight, default per-unit
  vector initialised to a small value so training starts near identity, `g≈0`).
- With `g=0` (or apical masks zeroed) the combiner is the identity → the cell
  reduces to a purely additive laminar wiring. This gives the **combiner-ablation**
  control (§9) for free: additive-only vs. additive+multiplicative.

This is the ML analogue of an apical-gain / BAC-firing motif: top-down input
*scales* a node's response rather than *adding* to it. It is the design's main
departure from the additive-only NCP.

---

## 6. Sign constraint

`M_TRN_Thal` is a **sign-locked inhibitory** edge. Implement as a `SparseLinear`
whose effective weight is `-abs(W) ⊙ mask` (or apply `tf.nn.softplus(W)` and
subtract), so the edge can never become excitatory during training. A
`sign_constraint: bool = True` flag makes the unconstrained variant an ablation.

---

## 7. Reduction properties (sanity anchors)

The design must degrade gracefully; these double as tests:

1. **Zero all feedback + apical + relay-loop masks** (`M_L5ET_L5IT`, `M_L6CC_L23`,
   `M_ap_*`, `M_L6CT_Thal`, `M_TRN_Thal`, `M_Thal_L4`) → a **feedforward-only
   laminar stack** `input→L4→L23→{L5IT,L5ET}→{L6CC,L6CT}`, i.e. a cortical
   *feedforward* NCP analogue. This is the `cncp_ff` control wiring.
2. **`g=0`** → additive-only cNCP (no gain).
3. Any `cell_cls` plugs in unchanged (orthogonality). LTC/LRC/CfC/GRU must all
   build and train.

---

## 8. Public API / integration points

Mirror the existing wiring/model/registry discipline exactly. **Do not commit;
do not touch git.** Work under `code-benchmarks/`.

1. `src/wirings/cncp.py` — new module:
   - reuse `SparseLinear` from `src/wirings/ncp.py` (import it) and add
     `SignedSparseLinear` (for the negative edge) if needed.
   - `CorticalColumnCell(BaseCell)` — the composite cell above.
   - `class CNCPWiring(BaseWiring)` **or** a `build_cncp(...)` helper that wraps
     `RNN(CorticalColumnCell(...), return_sequences=True)` + optional
     `Dense(output_neurons)` into a `Sequential`, so it drops into
     `SequentialODEFunc` like the others.
2. `src/wirings/__init__.py` — export the new symbols.
3. `src/models/rnn_model.py` — add `make_cncp_model(neuron_type, output_neurons,
   lamina_units=None, mask_densities=None, seed=42, combiner='multiplicative',
   divisive_inhibition=False, sign_constraint=True, **cell_kwargs)`. `neuron_type`
   is a `_CELL_REGISTRY` key or `BaseCell` subclass (this is `cell_cls`).
4. `src/benchmark/registry.py` — add `'cncp'` (default `cell_cls='lrc'` with
   `_ASYM`, or make the wiring a third value of the `WIRINGS` tuple:
   `('dense','ncp','cncp')` — prefer adding `'cncp'` to `WIRINGS` so any existing
   cell can be run with cncp wiring, matching the "wiring factor: dense/ncp_std/
   ncp_cortical" design). Update `is_known_wiring`. Keep the equivalence-lock test
   green (add matching legacy entries if you extend `run_benchmark.py`).
5. `tests/wirings/test_cncp_wiring.py` and `tests/neurons/` as needed — the
   validation suite in §9.

**Verification loop:** run `uv run pytest tests/ -q` from `code-benchmarks/` and
iterate until green. Also run one smoke training step
(`make_cncp_model('lrc', output_neurons=2)` → 3 iterations on the `spiral`
dataset via the existing trainer) to prove end-to-end differentiability.

---

## 9. Test / control suite (the 7-point cell checklist + wiring controls)

Follow `tests/neurons/test_cfc_lrc_cell.py` conventions:

1. `isinstance(CorticalColumnCell(...), BaseCell)`.
2. `state_size` is the 8-element list; `get_initial_state` returns 8 zero tensors
   of the right shapes; `output_size == units[L5ET]`.
3. Single forward pass: `cell((x, dt), init_state)` → `(output, new_state)` with
   `output.shape == (batch, units[L5ET])` and each new state the right shape.
4. Irregular sampling: different `dt` ⇒ different state for continuous `cell_cls`
   (e.g. `lrc`); ignored for `gru`.
5. Finiteness over ≥50 steps (stiffness guard).
6. `make_cncp_model('lrc', output_neurons=2)` and the same with `'gru'`, `'cfc'`,
   `'ltc'` all build and produce `(batch, timesteps, 2)`.
7. **Gradient flow:** every `model.trainable_variables` gradient non-None, finite,
   non-zero — including through the sign-locked and multiplicative edges.

Wiring-level controls (these become benchmark conditions later; implement at least
as toggles + a test each):

- **combiner ablation:** `combiner='additive'` (g forced 0) vs `'multiplicative'`.
- **feedforward reduction:** `cncp_ff` (feedback/apical/loop masks zeroed) builds
  and trains (§7.1).
- **sign-constraint ablation:** `sign_constraint=False` still builds/trains.
- **divisive ablation:** `divisive_inhibition=True` still builds/trains.
- **cell-orthogonality:** identical wiring works for ≥2 `cell_cls` (already in 6).
- (documented, not necessarily coded now) parameter-matched control and
  degree-preserving random-rewired control for the eventual benchmark.

---

## 10. Honest limitations (carry into the thesis, do not overclaim)

- Mask **densities and unit allocations are design choices**, not measured
  constants; the anatomy only fixes edge *existence/absence* and the one sign.
- **Delayed-state feedback** is an approximation of true recurrent settling; an
  iterative multi-step settling variant is possible but out of scope here.
- More nodes/edges ⇒ more parameters. A cNCP win is only meaningful against the
  **parameter-matched** and **random-rewired** controls; on generic Neural-ODE
  benchmarks the structure may act only as extra capacity. A null result there
  does not falsify the hypothesis — the discriminating tasks are top-down-gain /
  occlusion / regime-shift tasks (see `benchmark-decision-brief`).
- This is a *structural* prior only; it makes no claim about a biological learning
  rule. Training is ordinary backprop-through-time.

---

## 11. References (anchors for the structure; full list in vault `cNCP 08`)

Structural prior: Douglas & Martin 2004; Thomson & Lamy 2007; Harris & Shepherd
2015. Output-node split (IT vs ET): Kiritani et al. 2012; Shepherd 2013.
Relay-loop gating: Olsen et al. 2012; Crandall et al. 2015; Bortone et al. 2014.
Multiplicative gain motif: Larkum et al. 1999/2013; Takahashi et al. 2016.
NCP lineage: Lechner et al. 2018/2020; Hasani et al. 2021 (LTC). Cell family:
Farsang et al. 2024 (LRC); Hasani/Lechner et al. 2022 (CfC).

---

## 12. Implementation status & verified findings (2026-07-02)

Implemented and verified (334 tests green; adversarial Fable5 review + reverify).
Artifacts: `src/wirings/cncp.py` (`CorticalColumnCell`, `SignedSparseLinear`,
`CNCPWiring`), `make_cncp_model` in `src/models/rnn_model.py`, `'cncp'` added to
`WIRINGS`, an explicit `cncp` branch in `run_benchmark.build_model` (a bare `else`
previously mislabelled a cncp run as a standard NCP — fixed + guard-tested),
`tests/wirings/test_cncp_wiring.py`.

**Critical benchmark caveat (verified, not a code bug).** The Neural-ODE trainer
wraps the model in `SequentialODEFunc`, which calls `net(state)` on a length-1
sequence with a **fresh zero state at every Euler step**. Under this harness the
composite RNN unrolls exactly one timestep, so every **delayed-state edge**
(feedback, apical, the `Thal→L4` re-entry, and the whole `L6/relay` block) reads a
zero previous state and receives a **None or exactly-zero gradient** (measured:
41/119 variables None-grad, 13 more zero-grad for `lrc`). On the T=1 Neural-ODE
regression tasks the cNCP therefore **degenerates to its feedforward reduction**
plus a constant gain offset; a `cncp`-vs-`ncp` comparison there is really
`cncp_ff`-with-dead-parameter-overhead vs. `ncp`. This is a pre-existing property
of the ODE-func harness (standard NCP / dense recurrent kernels are equally
zero-grad there), so it does not invalidate `cncp.py`, but it means:

> **The recurrent cNCP structure cannot be exercised on the T=1 Neural-ODE tasks.**
> It must be benchmarked on **sequence-rollout tasks** where the RNN unrolls over
> real timesteps and the delayed-state edges become trainable — e.g. the MuJoCo-RL
> recurrent-policy track, an irregular-dt sequence task (Person Activity), or a
> regime-shift / occlusion task. Alternatively, carry the ODE-func state across
> Euler steps. This sharpens §10: on generic ODE tasks the structure is not merely
> "extra capacity" — its recurrent half is literally untrained.
