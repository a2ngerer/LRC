# tbt_cNCP — a differentiable Thousand-Brains extension of cNCP (concept)

Status: concept / design (2026-07-04). Not yet implemented. Companion to
`2026-07-02-cncp-design.md`.

## 1. Motivation and the gap

cNCP already models a cortical column (laminar nodes L4, L2/3, L5IT, L5ET,
L6CC, L6CT + Thal, TRN) with feedforward, top-down feedback, a multiplicative
apical gain, and a thalamo-reticular gate. It has the *anatomy* of a column.

The Thousand-Brains Theory (TBT; Hawkins/Numenta) says that what makes a column
*intelligent* is a specific computation on top of that anatomy:

1. a **location / reference-frame signal** carried in **L6a** (grid-cell-like),
   which tells the column *where on the object* the current sensation comes
   from, and which — combined with a **movement** command — is updated by
   **path integration**;
2. **sensorimotor prediction**: the column predicts the *next* sensation given
   the current state and the *next* location (predict → sense → move → update);
3. **L2/3** pools feature+location pairs into a **stable object representation**
   that disambiguates over successive sensations;
4. **voting**: many columns, each sensing a different part of the object, reach
   consensus through **long-range lateral L2/3 connections** (Numenta 2017: a
   single column needs ~11 sensations to recognise an object, three columns ~4).

cNCP currently has (3) and (partly) top-down feedback, but **no location signal
and no sensorimotor objective**. Its L6 is generic corticothalamic feedback, not
a reference frame. tbt_cNCP closes that gap.

**Honest scope caveat.** TBT's native implementation (HTM / Numenta's *Monty*,
Cortical Messaging Protocol) is NOT gradient-trained: it uses sparse distributed
representations, explicit reference frames, and Hebbian/evidence-based learning.
tbt_cNCP is a **differentiable reinterpretation** — TBT's *architectural and
objective* ideas grafted onto the differentiable cNCP so it trains by
backprop in the existing TensorFlow pipeline. It is inspired-by, not a faithful
re-implementation of Monty. That distinction must be stated in the thesis.

## 2. Architecture: what tbt_cNCP adds to cNCP

Base: the cNCP `CorticalColumnCell` (unchanged orthogonality — dynamics live in
`cell_cls`). Four additions, each a small differentiable module.

### 2.1 Location signal in L6 (reference frame)
- Per timestep the sensor is at 2D position `p_t` on the object; the movement
  is `m_t = p_{t+1} - p_t` (the action).
- Encode location with a fixed (or learnable) **grid-cell-like code**
  `g(p) ∈ R^d`: banks of `sin/cos(2π f_k · R(θ_k) p)` at several spatial
  frequencies `f_k` and orientations `θ_k` — the standard differentiable analogue
  of multiple grid modules (orientation-diverse, periodic, "nearby locations →
  similar codes", as TBT requires).
- **Path-integration variant (preferred):** feed only `m_t` and let L6 integrate
  location recurrently (`loc_{t} = loc_{t-1} + f(m_t)`), so the reference frame
  is maintained internally rather than handed in.
- L6 receives `g(p_t)` (or the integrated `loc_t`) as an extra input.

### 2.2 Location → L4 modulation
- TBT: L6a projects to L4 basal dendrites (~45% of synapses), putting L4 into a
  *predictive state*. Implement as a **modulatory L6→L4 edge** reusing cNCP's
  existing gain machinery: `h_L4 = h_L4_sensory * (1 + g_loc * σ(W_loc · loc))`.
  Location gates sensation instead of adding to it, matching the biology and
  cNCP's multiplicative-gain design choice.

### 2.3 Sensorimotor prediction head
- A head that, from the current column state **plus the next location**
  `g(p_{t+1})`, predicts the next sensory feature `x̂_{t+1}`.
- Self-supervised auxiliary loss `L_pred = ||x̂_{t+1} − x_{t+1}||²`.
- This gives the top-down feedback a concrete job (predictive coding: top-down
  carries the prediction, feedforward the error).

### 2.4 L2/3 object readout (+ optional multi-column voting)
- Read the **object class** from **L2/3** (TBT's object/output layer), not L5ET,
  via an `L2/3 → Dense` classifier. L5ET stays the motor/action output (optional
  active-sensing policy that emits `m_t`).
- **Voting (the "thousand brains" part):** run `K` columns in parallel, each
  starting at a different location. After each step, mix their L2/3 states with a
  differentiable consensus op (mean or learned attention over the other columns'
  L2/3 = a stand-in for TBT's lateral L2/3 excitation). Consensus should cut the
  number of sensations needed.

## 3. Testable task (fits the existing pipeline)

**Active-sensing 2D object recognition.** Objects = small 2D feature fields
(synthetic shapes, Omniglot characters, or MNIST-as-objects). An episode:
- sensor starts at `p_0`, observes a local patch `x_0 = patch(object, p_0)`;
- a policy (fixed raster / random walk / learned L5ET policy) moves the sensor
  to `p_1 … p_T`; at each step the model gets `(x_t, m_t)`;
- outputs per step: (a) object-class logits from L2/3; (b) next-patch prediction.

This is a **sequence-rollout** task, so cNCP's recurrence is actually trained
(unlike the T=1 ODE harness — see `2026-07-02-cncp-design.md` §12). It reuses the
`RNN(return_sequences=True)` + `Dense` head machinery already in
`src/tasks/{person_activity,lotka_volterra}`.

## 4. Metrics, controls, hypotheses

**Metrics**
- classification accuracy **as a function of number of sensations** (the TBT
  sample-efficiency signature — the headline curve);
- next-patch prediction MSE (auxiliary self-supervision quality);
- robustness to **occlusion** (mask part of the object) and to **novel start
  positions / rotations** (reference-frame invariance).

**Controls / ablations**
- plain cNCP (no location signal, no prediction head) — isolates the
  reference-frame contribution;
- tbt_cNCP without the prediction loss — isolates the sensorimotor objective;
- single column vs `K`-column voting — tests the thousand-brains claim;
- NCP / dense wirings fed the same location input — isolates topology vs.
  ingredient;
- parameter-matched across all arms (Farsang constraint).

**Falsifiable hypotheses**
- **H1** tbt_cNCP recognises objects in fewer sensations than plain cNCP (the
  location signal disambiguates faster).
- **H2** `K`-column voting needs fewer sensations than one column (thousand
  brains).
- **H3** tbt_cNCP generalises better to novel start positions / rotations and to
  occlusion (reference frames + top-down prediction fill gaps).

If H1–H3 fail, the honest conclusion is that the cortical *anatomy* alone (plain
cNCP) already captures whatever benefit exists, and the explicit
reference-frame/voting machinery does not add measurable value in this
differentiable form — itself a publishable negative result.

### Preliminary result (implemented 2026-07-04, single seed, local)
The implementation is `src/wirings/tbt_cncp.py` (TbtCorticalColumnCell) +
`src/tasks/active_sensing/`. Two object modes were run:
- **shapes** (locally distinguishable outlines): tbt_cNCP 0.594 vs no-location
  0.600 — location does NOT help, because a single patch already leaks the
  class (curvature/orientation). This mode is an *uninformative* H1 test.
- **compositional** (shared local motif, class = spatial configuration → the
  fair H1 test): tbt_cNCP **0.454** vs no-location **0.267** (baseline 0.183) —
  a **+0.19** absolute gain from the location signal, supporting H1.

Takeaway: the reference-frame signal helps exactly when the task makes local
patches ambiguous, as TBT predicts; on locally-separable objects it is redundant.
The compositional mode is therefore the default.

### Mechanism vs topology (3 seeds, local, compositional)
Location can reach the column two ways: as a multiplicative GATE on L4 (the
TBT-faithful L6a mechanism) or CONCATENATED into the sensory input (as ncp/dense
receive it). Factoring these apart (wirings tbt_cncp / tbt_cncp_concat /
tbt_cncp_both) resolves why plain tbt_cncp trailed ncp/dense:

| wiring                    | final acc      |
|---------------------------|----------------|
| tbt_cncp (concat)         | **0.932 ±0.012** (best) |
| tbt_cncp (gate + concat)  | 0.898 ±0.027   |
| NCP                       | 0.865 ±0.020   |
| dense                     | 0.742 ±0.071   |
| tbt_cncp (gate only)      | 0.656 ±0.072   |
| tbt_cncp (no location)    | 0.343 ±0.027   |

Findings: (1) the earlier "NCP beats tbt_cncp" was a MECHANISM artefact, not a
topology deficit -- with location concatenated, the cNCP topology is the BEST
wiring (0.932 > NCP 0.865 > dense 0.742). (2) The bottleneck was the gate-only
pathway; the multiplicative gate adds nothing on top of concat (0.898 <= 0.932)
and even slightly hurts, so the TBT-faithful L6a->L4 gating is, in this simple
form, the wrong mechanism here. (3) H1 holds regardless (no-location 0.343 vs
any-location 0.66-0.93). Caveat: one synthetic task, one cell (cfc_lrc), 3 seeds.

Multi-seed cluster run pending (cluster/active_sensing.*).

### Voting / H2 (implemented 2026-07-04, 3 seeds, local)
Multi-column voting is `MultiColumnVotingCell` (src/wirings/tbt_cncp.py): K
weight-shared columns, each on its own glimpse stream, that mix their L2/3
(object) states toward the mean each step (learnable strength `vote_raw`) -- the
differentiable stand-in for lateral L2/3 excitation. Runner/plot:
`experiments/run_voting_benchmark.py` + `plot_voting.py`.

| K columns | final acc     | glimpses to 0.7 |
|-----------|---------------|-----------------|
| K=1       | 0.884 ±0.018  | 3.3             |
| K=2       | 0.953 ±0.017  | 2.0             |
| K=3       | 0.969 ±0.006  | 1.3             |

H2 confirmed: more columns -> fewer glimpses-per-column to reach threshold and
higher final accuracy, the Numenta sample-efficiency signature (1 column many
sensations, more columns fewer). Crucially the columns SHARE weights, so K=1/2/3
have ~identical parameter counts -- the gain is pure voting/consensus, not
capacity. After a single glimpse per column, K=3 is already at 0.70 vs K=1 0.48.

All three TBT ingredients are now implemented and individually validated:
reference frame (H1), sensorimotor rollout, and voting (H2).

### Cluster benchmark suite (2026-07-05, dataLAB, 3 seeds, cfc_lrc)
Four plottable NCP-vs-cNCP-vs-tbt_cNCP benchmarks run on the dataLAB cluster
(SLURM jobs 521921/2/3/4). The trajectory tasks add tbt_cNCP via a PHASE-SPACE
reference frame (a grid-cell Fourier code of the current 2D state, fed to the
L6a->L4 gate / concatenated); dense/ncp/cncp are the pure-topology arms.
Parameter-matched across ncp/cncp/tbt (~45k); dense is a lighter baseline (25k).

**(1) Lotka-Volterra predator-prey (trajectory, seq_len 64, 200 ep).** Every
wiring learns the one-step map almost perfectly (teacher-forced MSE 0.001-0.008
vs persistence 0.19), but ALL diverge in closed-loop rollout (the orbits spiral
inward -- energy not conserved), with large seed variance and no clear topology
winner (closed-loop MSE: dense 1.43, cNCP 1.64, tbt 1.79, NCP 1.84, tbt_noloc
3.28). The reference frame is redundant here: on a single smooth closed orbit the
current state already determines the flow.

**(2) Duffing oscillator (trajectory, double-well, seq_len 64, 200 ep).** Here
the reference frame HELPS: tbt_cNCP (concat) has the lowest closed-loop MSE
**1.146 +/-0.119**, halving the no-location arm (2.283) and beating dense (1.565),
NCP (1.763) and cNCP (2.288); the horizon plot shows tbt_cNCP's per-step error
growing slowest and cNCP's fastest. In the double well the local state is
ambiguous (which well?), so the phase-space reference frame disambiguates -- the
same pattern as active-sensing's compositional objects.

**(3) Active-sensing object recognition (classification, 3 seeds).** Reproduces
the earlier local finding on the cluster: tbt_cNCP (concat) **0.941 +/-0.014** >
NCP 0.910 > dense 0.848 > tbt_cNCP (gate) 0.741 > no-location 0.335. Location is
essential (H1); the concat mechanism beats the multiplicative L6a->L4 gate; the
cNCP topology + concat beats NCP. Occlusion degrades all (concat most robust,
0.498).

**(4) Multi-column voting (sample efficiency, K=1/2/3).** H2 confirmed: glimpses-
per-column to 0.7 accuracy fall 3.7 -> 2.0 -> 1.7 and final accuracy rises
0.894 -> 0.969 -> 0.969 with weight-shared columns (gain is pure consensus, not
capacity).

**Unified finding.** The reference frame helps the cortical column *exactly when
local observations are ambiguous* -- the duffing double-well and compositional
objects -- and is redundant when the current state/patch already disambiguates
(the simple predator-prey orbit, locally-separable shapes). This holds across the
trajectory and classification domains, and the concat mechanism consistently
beats the gate. Caveats: one cell (cfc_lrc), 3 seeds; on fully-observed
trajectory tasks the reference frame is a re-encoding of the state (a positional/
Fourier feature), not extra information; closed-loop rollout suffers exposure-bias
divergence for every wiring (teacher-forced is near-perfect), so predator-prey's
closed-loop metric is high-variance.

## 5. Relation to the thesis

- Sits inside the existing wiring/cell orthogonality (tbt_cNCP is cNCP + a
  location channel + a prediction head + optional voting; the `cell_cls` is
  untouched).
- Turns the "occlusion / regime-shift" open task noted after the Farsang pivot
  into a concrete, grounded benchmark.
- Provides a *narrative* linking cNCP to an established theory of cortical
  computation — with the caveat of §1 stated up front.
- Scope warning: reference frames + voting are a real scope increase over the
  current parameter-matched dense/NCP/cNCP matrix; treat tbt_cNCP as an
  exploratory extension arm, not a replacement for the core comparison.

## 6. Sources
- Hawkins et al. 2017, *A Theory of How Columns in the Neocortex Enable Learning
  the Structure of the World*, Frontiers in Neural Circuits.
- Hawkins et al. 2019, *A Framework for Intelligence and Cortical Function Based
  on Grid Cells in the Neocortex*, Frontiers.
- Hawkins 2021, *A Thousand Brains: A New Theory of Intelligence*.
- Thousand Brains Project / Monty: arXiv:2412.18354 (2024); arXiv:2507.04494
  (2025, Neural Computation) — Learning Modules, Cortical Messaging Protocol,
  evidence-based voting.
