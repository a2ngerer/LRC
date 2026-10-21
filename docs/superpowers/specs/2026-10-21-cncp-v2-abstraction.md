# cNCP v2: an abstracted cortical-column wiring with independent multi-column coupling

**Date:** 2026-10-21
**Status:** design proposal (no code changed, nothing committed, no training run)
**Supersedes nothing yet.** The current cNCP (`docs/superpowers/specs/2026-07-02-cncp-design.md`, implementation `src/wirings/cncp.py`) stays the reference arm until this design has passed the acceptance test R6 in section B.5.
**Review status:** Codex (gpt-6-astra, 2026-10-21, final section) does not clear the draft for implementation as a whole. Accepted sequencing: reproduce and fix the canary divergence on the existing eight-node cNCP first, run the main campaign on the existing arms, treat V3 as a bounded extra ablation and V4 and the hub as outlook.

---

## 0. Scope and reading guide

The cNCP is a fixed sparse connectivity pattern for a recurrent network, in the sense of `NCPWiring`. Node names (L4, L2/3, L5, L6, thalamus, TRN) label nodes of a computation graph; nothing here simulates a cell. This document does three things:

- **Part A** reviews the anatomy at the level an abstraction needs, with inter-column connectivity as the focus, and separates established findings from theory and from contested points.
- **Part B** applies the NCP recipe (Lechner et al. 2018, 2020) to the cortical column: every element of the current eight-node design is kept, merged or dropped, with the ML reason. It defines cNCP v2 (four node classes, eight edge rules), its initialisation and stability rules, the parameter and compute formulas, the multi-column extension with independent weights, the reduction table and the falsification criteria. It also diagnoses the teacher-forced divergence of the 2026-10-21 canary.
- **Part C** defines the four comparison arms V1 to V4 with formulas and costs at the two budgets (4k rollouts, 30k Person Activity).
- **Codex review** (final section) records an independent review and the response to each finding.

Measured numbers in this document come from two scratch scripts run against the current `code-benchmarks` tree (`init_stability.py`, `v2_prototype.py` in the session scratchpad); they build cells through the repository's own classes and change no repository file.

---

## Part A: what the anatomy says, and how firmly

### A.1 The NCP recipe as the template

Lechner et al. (2020) derived the NCP from the *C. elegans* wiring by four rules (their rules 1 to 4, `papers/.extracted/lechner2020-ncp.md`, "Designing and learning NCPs"): (1) four neuron classes in a fixed order, sensory, inter, command, motor; (2) between consecutive classes, every source neuron gets a fixed number of random synapses to targets; (3) targets left without a synapse are back-filled to the mean fan-in; (4) recurrent synapses only among command neurons. In Lechner's LTC the sampled polarity enters the dynamics through the reversal potential E_ij of their eq. (1) (`lechner2020-ncp.md`, line 361); in this repository's wiring the polarity is discarded by taking the absolute value of the adjacency (`ncp.py` line 134), so sign plays no structural role *here*. The two levels are kept apart in this document. Everything else about the worm, the nine neuron classes of the tap-withdrawal circuit, gap junctions, the actual connectome, was dropped. The 2018 paper (`lechner2018-ncp.md`) still used the literal tap-withdrawal circuit; the 2020 abstraction is what became useful.

Three properties of this recipe carry over as design constraints: classes are defined by *role in the graph*, not by anatomy; the edge rules are *statistical* (fan-out, fan-in, density), not a connectome; and there is *one* structural asymmetry that distinguishes the graph from a layered MLP (recurrence only in the command class).

### A.2 Canonical microcircuit and cell classes (established)

- **Laminar excitatory classes.** Harris and Shepherd (2015) group neocortical excitatory neurons into three classes with consistent projection targets across areas: intratelencephalic (IT; L2 to L6, cortex and striatum), pyramidal-tract or extratelencephalic (PT/ET; L5B, brainstem, spinal cord, thalamus) and corticothalamic (CT; L6, thalamus). The review notes that CT cells have "few local inputs, sparse local ascending axons, and [are] mostly silent in vivo". Source: [Harris & Shepherd 2015, Nat Neurosci 18:170](https://pmc.ncbi.nlm.nih.gov/articles/PMC4889215).
- **Canonical interlaminar flow.** L4 sits upstream of the local excitatory network and projects to L2/3; L2/3 IT cells send a dense descending projection to L5A/B; PT cells receive extensive IT input and give little back (IT to PT asymmetry; Kiritani et al. 2012). Sources: [Harris & Shepherd 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4889215); [Douglas & Martin 2004, Annu Rev Neurosci 27:419](https://www.jci.org/references/scholar/28292/B15); [Thomson & Lamy 2007, Front Neurosci 1:19](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/neuro.01.1.1.002.2007/full); [Kiritani et al. 2012, J Neurosci 32:4992](https://pmc.ncbi.nlm.nih.gov/articles/PMC3329752).
- **L6 to L4 is weak.** Harris and Shepherd describe the L6 to L4 projection in rodents as weak and possibly modulatory or inhibitory. This supports the current spec's "no direct excitatory L6 to L4 edge".

### A.3 Thalamocortical loop and reticular gating (established mechanism, contested weighting)

- **Drivers and modulators.** Sherman and Guillery's distinction: L5 corticothalamic axons drive higher-order thalamic nuclei; L6 corticothalamic axons modulate first-order relays. Sources: [Sherman & Guillery 1998, PNAS 95:7121](https://doi.org/10.1073/pnas.95.12.7121); [Sherman & Guillery 2002, Phil Trans R Soc B](https://pmc.ncbi.nlm.nih.gov/articles/PMC1693087).
- **L6CT acts through two routes.** Olsen et al. (2012) showed that L6 scales the gain of upper-layer visual responses without changing tuning, through intracortical and thalamic routes, with the intracortical route the larger one. Bortone et al. (2014) identified the intracortical arm: L6CT cells recruit deep fast-spiking interneurons with translaminar axons. Crandall et al. (2015) showed that the thalamic arm combines monosynaptic facilitating excitation with disynaptic depressing TRN inhibition, so the net effect switches with firing rate. Sources: [Olsen et al. 2012, Nature 483:47](https://ideas.repec.org/a/nat/nature/v483y2012i7387d10.1038_nature10835.html); [Bortone et al. 2014, Neuron 82:474](https://pmc.ncbi.nlm.nih.gov/articles/PMC4068343); [Crandall et al. 2015, Neuron 86:768](https://pmc.ncbi.nlm.nih.gov/articles/PMC4425600).
- **TRN as a state-dependent gate.** TRN is organised in subnetworks; sensory-projecting TRN cells are suppressed during attention (Halassa et al. 2014), and prefrontal control of visual TRN sets visual thalamic gain through feedforward inhibition (Wimmer et al. 2015). Sources: [Halassa et al. 2014, Cell](https://cenl.ucsd.edu/labspace/JclubLabMeetings/Stephanie-Halassa%20et%20al-architecture-thalamic-subnetworks+2014.pdf); [Wimmer et al. 2015, Nature 526:705](https://pmc.ncbi.nlm.nih.gov/articles/PMC4626291).
- **Weighting is contested.** Olsen's finding that the intracortical route dominates the gain effect means the relay loop is the smaller half of "L6 gain control" in the one quantitative study available. A design that keeps the relay loop as the carrier of deep-to-entry gain (as the current spec does) chooses the mechanistically cleaner, not the empirically larger, route.

### A.4 Apical gain (established mechanism, speculative function in a controller)

Coincidence of basal (feedforward) and apical (feedback) input in L5 pyramidal cells triggers a dendritic calcium spike and a burst (BAC firing), a multiplicative coupling of the two input streams rather than an additive one; suppressing apical calcium raises the perceptual threshold. L1, the termination zone of top-down and higher-order thalamic input onto apical tufts, has no excitatory somata and is gated by NDNF interneurons. Sources: [Larkum, Zhu & Sakmann 1999, Nature 398:338](https://doi.org/10.1038/18686); [Larkum 2013, Trends Neurosci 36:141](https://doi.org/10.1016/j.tins.2012.11.006); [Takahashi et al. 2016, Science 354:1587](https://www.science.org/doi/10.1126/science.aah6066); [Schuman et al. 2021, Annu Rev Neurosci 44:221](https://annualreviews.org/doi/pdf/10.1146/annurev-neuro-100520-012117). Whether a within-column gain source helps a sequence model is an open ML question; the Lehrreihe (`cNCP 07`) already marks it "established mechanism, speculative benefit".

### A.5 Inter-column connectivity (the part the current design lacks)

Four anatomical routes connect columns of the same area or of neighbouring areas:

1. **L2/3 long-range horizontal axons (established).** L2/3 pyramidal axon collaterals run 2 to 5 mm across the cortical surface and terminate in patches; beyond about 500 µm they preferentially link columns with similar orientation preference and are elongated along the preferred orientation axis (tree shrew V1). They shape L2/3 responses mainly through surround effects, while L4 feedforward input drives the classical receptive field. Sources: [Gilbert & Wiesel 1979, 1983, cited in Bosking et al. 1997, J Neurosci 17:2112](https://pmc.ncbi.nlm.nih.gov/articles/PMC6793759); [Bosking et al. 1997 PDF](https://redwood.berkeley.edu/wp-content/uploads/2020/08/bosking-etal.pdf); [Chisum et al. 2003](https://pmc.ncbi.nlm.nih.gov/articles/PMC6742080). In the ML reading: a sparse, feature-selective, modulatory lateral edge between the *integration* nodes of neighbouring columns.
2. **Deep-layer horizontal projections (established anatomy, function emerging).** Cell-type-specific horizontal axon patterns exist in all layers of rat barrel cortex (Narayanan et al. 2015), and horizontally projecting deep-layer corticocortical neurons sit at the thalamocortical terminal fields, collect strong thalamic input and gate PT output (Egger et al. 2020). Sources: [Narayanan et al. 2015, Cereb Cortex 25:4450](https://pmc.ncbi.nlm.nih.gov/articles/PMC4816792); [Egger et al. 2020, Neuron 105:122](https://pmc.ncbi.nlm.nih.gov/articles/PMC6953434). In the ML reading: a lateral edge between *output* nodes is anatomically defensible too, but the evidence is thinner and the function (gating of output) overlaps with what the relay loop already provides.
3. **Transthalamic route via higher-order thalamus (established route, function debated).** L5 PT collaterals drive higher-order nuclei (pulvinar, POm); these project back to cortex, mainly to L1 and upper layers, forming a cortex to thalamus to cortex path that parallels direct corticocortical connections. Information transfer between somatosensory areas survived disruption of the direct route and was blocked only by thalamic inhibition. Sources: [Sherman & Usrey 2024, J Neurosci 44:e0909242024](https://conte.thalamus.princeton.edu/publications/transthalamic-pathways-cortical-function); [Mease & Gonzalez 2021, Front Neural Circuits 15:730211](https://www.frontiersin.org/articles/10.3389/fncir.2021.730211/full); [Frontiers review on L5 cortico-thalamo-cortical networks 2021](https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2021.721186/full); [Usrey et al. 2019, J Comp Neurol](https://usreylab.ucdavis.edu/sites/g/files/dgvnsk6251/files/inline-files/Usrey_et_al-2019-Journal_of_Comparative_Neurology.pdf). In the ML reading: a *shared hub* that reads every column's output node and writes to every column's integration node, with its own gate.
4. **Inter-areal counterstreams (established at area level, not a within-area rule).** Feedforward projections originate mostly in supragranular layers, feedback mostly in infragranular layers (SLN index, Markov et al. 2014); about half of V2 to V1 feedback neurons are nevertheless supragranular. Source: [Markov et al. 2014, J Comp Neurol 522:225](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4255240/). Relevant only if columns are stacked hierarchically, which this design does not do.

### A.6 Many columns over receptive fields (established pattern, contested unit)

Hubel and Wiesel's orientation columns and the later pinwheel maps show one area tiling its input space with repeated circuits that differ in preferred feature and receptive-field position. The "hypercolumn" as a unit has drifted in meaning, and Horton and Adams (2005) argue the column may have no function at all; a later philosophy-of-science review finds no agreed general definition. Sources: [Horton & Adams 2005, Phil Trans R Soc B 360:837](https://redwood.berkeley.edu/wp-content/uploads/2018/08/horton05.pdf); [Haueis 2020 preprint](https://philsci-archive.pitt.edu/18192/1/Haueis_2020_The%20Death%20of%20the%20Column_Preprint.pdf); [Mountcastle 1997 as cited in Frontiers 2010](https://www.frontiersin.org/articles/10.3389/fnana.2010.00017/pdf). For the design this means: "a column" is a *modelling* unit, and the only anatomically grounded statements about it are its laminar circuit and its lateral connections; column size, count and borders are free parameters.

### A.7 Thousand Brains (theory)

Hawkins, Ahmad and Cui (2017) model a column as an input layer (L4), an output layer (L2/3) and a location signal (L6); columns connect through long-range excitatory connections in the *output* layer, lateral input is modulatory ("cells with more modulatory input will win"), each column "learns from its own sensor", and a single column converges after 11 sensations where three converge after four. The Thousand Brains Project (Monty) implements this with learning modules that each hold their own object model, exchange hypotheses through the Cortical Messaging Protocol (features at poses), vote all-to-all, and in the reported experiments use eight sensor patches feeding eight learning modules, which lets inference "converge in significantly fewer steps as the number of SMs (and associated LMs) scale". Sources: [Hawkins et al. 2017, Front Neural Circuits 11:81](https://www.frontiersin.org/articles/10.3389/fncir.2017.00081/full); [TBP framework paper, arXiv 2412.18354](https://arxiv.org/abs/2412.18354); [TBP evaluation paper 2025, arXiv 2507.04494](https://arxiv.org/html/2507.04494v1).

Two points matter for Part B. First, in both the 2017 model and Monty the columns are *independent learners* with their own models; the weight-shared committee of July 2026 (`src/wirings/committee.py`, `MultiColumnVotingCell`) departed from the theory it cites. Second, the lateral/voting signal in the theory is a *modulatory bias on the output layer*, which maps onto a sparse additive or multiplicative edge between integration nodes, not onto a convex mix of states.

### A.8 Predictive coding (theory, contested laminar mapping)

Rao and Ballard (1999) and Bastos et al. (2012) assign prediction errors to superficial and predictions to deep pyramidal cells. The laminar mapping is a proposal, not a measurement; the Lehrreihe (`cNCP 07`, section 3) already files it as speculative. Sources: [Rao & Ballard 1999, Nat Neurosci 2:79](https://www.nature.com/articles/nn0199_79); [Bastos et al. 2012, Neuron 76:695](https://pubmed.ncbi.nlm.nih.gov/23177956/). No v2 element depends on it.

### A.9 Summary table

| Element | Status | Load-bearing for v2? |
|---|---|---|
| L4 entry, L4 to L2/3 to L5 chain | established | yes (feedforward spine) |
| IT/PT/CT classes; IT to PT asymmetry | established | partly (merged, see B.2) |
| L6CT modulates first-order relay, TRN inhibits relay | established | yes (relay rule) |
| Intracortical L6 to FS route dominates gain | established (one study) | acknowledged, not modelled |
| Apical multiplicative coupling (BAC) | established mechanism | yes (one gain edge) |
| L2/3 long-range horizontal, feature-selective | established | yes (lateral coupling mode) |
| Deep horizontal projections gate output | established anatomy, function emerging | optional (not default) |
| Transthalamic L5 to HO thalamus to upper layers | established route, function debated | yes (hub coupling mode) |
| Supra/infra counterstreams | established at area level | no (no hierarchy of columns) |
| Column as functional unit, hypercolumn | contested | no; columns are modelling units |
| Thousand Brains voting, CMP | theory | informs coupling and independence |
| Predictive-coding laminar roles | theory, contested | no |

---

## Part B: the abstraction (cNCP v2)

### B.1 Design goal

Keep the fewest node classes and edge rules that still carry the three motifs that distinguish a cortical column from the worm circuit:

1. top-down feedback from a deep node to a superficial node (the NCP has none),
2. one multiplicative gain edge (the NCP is additive only),
3. a re-entrant relay loop with an inhibitory gate (the NCP has no loop through a non-neuronal node).

Everything that does not serve one of the three motifs, or that duplicates a motif, is merged or dropped. Stability at initialisation is a hard requirement (B.5).

### B.2 Keep / merge / drop

| Element of the real circuit (current cNCP node or edge) | Decision | Reason |
|---|---|---|
| L4 entry node (`L4`) | **KEEP** as class **G** (granular) | Input entry and target of the re-entrant relay. The only node that reads the external input (motif 3 needs a distinct re-entry target). |
| L2/3 integration node (`L23`) | **KEEP** as class **S** (superficial) | Target of top-down feedback (motif 1) and of the gain edge (motif 2); carrier of the lateral inter-column edge (A.5.1). In the `NCPLayeredCell` reading (inter → command → motor) S is the *command* analogue: the recurrent layer that receives feedback. |
| L5 IT and L5 ET (`L5IT`, `L5ET`) | **MERGE** into class **D** (deep output hub) | The IT to ET asymmetry is a within-L5 detail whose ML content (a recurrent hub with a sparse back-edge) is approximated by the sub-cell's own dense recurrence. The dense recurrence does not preserve the IT → ET directionality; the asymmetry is given up, not reproduced, and whether it carries task information is untested (B.8). Two nodes cost 2p(u,u) plus three edges. D is the *motor* analogue (the read-out layer), G the *inter* analogue. |
| L6 CC and L6 CT (`L6CC`, `L6CT`) | **MERGE** into D | Both are "deep nodes that send signals back up or into the relay". With D as the single deep node, D at step t-1 is the source of the additive feedback, the gain edge and the relay drive. An optional split (`F` = L6, one extra sub-cell fed by D, source of feedback and relay drive) is kept as an ablation; it costs p(u,u) + 2u² and is justified only if the readout node must be decoupled from the control signals. Default: merged. Cost of the merge: task output and control signals (feedback, relay drive) share one state and one timescale. Anatomical fidelity lost: CT cells are a distinct, mostly silent class (Harris & Shepherd). The NCP made the same move when it folded the worm's AVA/AVB command neurons, which are both recurrent and motor-driving, into one class. |
| Thalamic relay and TRN (`Thal`, `TRN`) | **MERGE** into class **R** (gated relay) | One leaky integrator with a multiplicative inhibitory gate computed from the same driver (B.3). The TRN's function in the loop, feedforward inhibition that depresses the relay (Crandall), is a gate on the relay target; a second stateful node with its own rate and a sign-locked edge was two mechanisms for one effect. What is lost: two states with separate time constants and a one-step delay between them, and Crandall's rate-dependent switch between facilitating excitation and depressing inhibition, of which the gate keeps only the attenuating half (A.3). |
| Direct L6 to L4 excitatory edge | **DROP** (unchanged) | Reported as weak and modulatory in rodents (A.2); this is a judgement about weight, not an anatomical absence. The loop carries the deep-to-entry effect. |
| Intracortical L6 to FS to all-layer inhibition (`divisive_inhibition`) | **DROP** | Empirically the larger half of L6 gain control (Olsen), but modelling it adds a second multiplicative mechanism and the thesis asks for one. Record as the reason a null result on gain cannot be read as "the cortical gain motif fails". |
| Second gain target on L5ET (`M_ap_L5ET`) | **DROP** | One gain edge (motif 2) is the thesis question; a second target adds a mechanism without a hypothesis. The reason is scope, not redundancy: a state-dependent gain G_t in front of the head is not absorbable by a fixed head (W_out G_t h_t has no fixed-W' equivalent), and in the current code the modulated L5ET state also feeds L6 and the next step (`cncp.py` lines 604 and 615). |
| L6CC to L2/3 additive feedback (`M_L6CC_L23`) and L5ET to L5IT back-edge (`M_L5ET_L5IT`) | **MERGE** into one additive edge D to S (previous step) | Both are delayed sparse deep-to-upper edges, with different sources (L6CC, L5ET) and targets (L2/3, L5IT). With one deep node they share the source; the merge keeps the motif and gives up the two-target structure. Functional equivalence is assumed, not shown. One edge, density 0.5. |
| Sign-locked `SignedSparseLinear` on TRN to Thal | **DROP** (replaced by the gate's form) | The gate (1 - sigmoid) can only attenuate, so the sign constraint becomes structural. The replacement keeps the attenuating effect only; a generally negative additive effect is not reproduced. Ablation: gate fixed at 0. |
| Sensory route through the relay (`sensory_route='thalamic'`) | **DROP** from the default, keep as ablation flag | Issue a2ngerer/LRC #7 shows it was never wired into a runner. Not a v2 question. With the route off, R has no sensory driver input; "first-order relay" labels its position in the loop. |
| Per-lamina timescale prior (`timescale_prior`) | **DROP** | Iteration 9 measured -0.001 over five seeds (`docs/tbt_cncp_study_summary.md`). |
| Location gate (`TbtCorticalColumnCell`) | **DROP** from the wiring | A task-input mechanism, not a column property; the concat variant won (2026-07-04 work log). |
| Partition of the input over columns, lateral coupling, hub | **ADD** (B.7) | The element the current design lacks and the user's question. A substantial extension of the experiment, not a simplification; hub deferred to the outlook. |

Result: **four node classes** (three RNN sub-cells G, S, D of the pluggable `cell_cls`, one gated relay R) and **eight edges**, down from eight nodes and fifteen edges.

### B.3 Node classes, widths, edges, update equations

**Widths.** One knob u; G, S, D have width u, R has width u_R = max(1, u // 2). Equal widths for the three sub-cells avoid a bottleneck and keep the parameter formula one-dimensional; the relay is a gate, half width suffices. (The current proportions 8:8:6:8:4:4:4:2 are design choices too, spec section 10.)

**Edges.** `E` denotes an additive `SparseLinear` (fixed Bernoulli(density) mask, no bias), `A` the apical edge feeding the gain, `fb(.)` = tanh (the bounded-feedback rule, B.5). "prev" means the stored state of step t-1.

| Edge | From | To | Reads | Density | Role |
|---|---|---|---|---|---|
| `E_in`  | x_t | G | current | 1.0 | external driver |
| `E_RG`  | fb(r_{t-1}) | G | **prev** | 1.0 | re-entrant relay (motif 3) |
| `E_GS`  | g_t | S | current | 1.0 | strongest interlaminar edge |
| `E_DS`  | fb(d_{t-1}) | S | **prev** | 0.5 | additive top-down feedback (motif 1) |
| `A_DS`  | fb(d_{t-1}) | S (apical) | **prev** | 0.5 | multiplicative gain (motif 2) |
| `E_SD`  | s̃_t | D | current | 1.0 | superficial to deep |
| `E_DR`  | d_t | R (drive) | current | 1.0 | relay driver |
| `E_DRg` | d_t | R (gate) | current | 1.0 | inhibitory gate (TRN role) |

Within a step the order G, S, D, R is acyclic; every cycle is closed by a "prev" read, exactly as in the current cell (`cncp.py` lines 14 to 21). There is no edge from G or S into R and no edge from D into G.

**Update** (style of spec section 4; `cell_X((basal, dt), [prev])` is the pluggable sub-cell; `⊕` adds):

```
# previous state: g_, s_, d_, r_   (G, S, D, R)

# 1. entry node: driver + delayed re-entrant relay
basal_G = E_in(x_t) ⊕ E_RG(tanh(r_))
g_t, _  = cell_G((basal_G, dt), [g_])

# 2. integration node: feedforward + delayed additive feedback, then one gain edge
basal_S = E_GS(g_t) ⊕ E_DS(tanh(d_))
s_t, _  = cell_S((basal_S, dt), [s_])
a_t     = A_DS(tanh(d_))
s̃_t     = s_t * (1 + γ * tanh(a_t / 2))        # γ per unit, init 0 -> identity

# 3. deep output hub
d_t, _  = cell_D((E_SD(s̃_t), dt), [d_])

# 4. gated relay (leaky integrator over dt; the gate can only attenuate)
gate_t  = sigmoid(E_DRg(d_t) + b_gate)          # b_gate init -2
target  = tanh(E_DR(d_t) + b_r) * (1 - gate_t)
α       = 1 - exp(-dt * softplus(ρ))             # ρ init 0 -> α(dt=1) = 0.5
r_t     = (1 - α) * r_ + α * target

output    = d_t
new_state = [g_t, s_t, d_t, r_t]                 # S stores s_t (raw), not s̃_t
```

Notes on the equations:

- `tanh(a/2) = 2 sigmoid(a) - 1`, so the gain factor is exactly 1 when the apical drive is zero, for every γ. The current combiner `1 + g sigmoid(a)` is the constant `1 + g/2` at zero drive (spec section 13, closing remark); v2 removes that offset.
- γ is initialised to **exactly 0**. At γ = 0 the apical weights `A_DS` receive no gradient, but γ does (∂/∂γ = s_t · tanh(a_t/2) ≠ 0 once d_{t-1} ≠ 0), so after the first update γ ≠ 0 and `A_DS` starts learning. This is the zero-init gating pattern of FiLM-style conditioning ([Perez et al. 2018](https://arxiv.org/abs/1709.07871v2)) and of the `film` location mode already in `tbt_cncp.py` lines 92 to 101. The gain factor is bounded in (1 - |γ|, 1 + |γ|).
- S stores the raw sub-cell state and passes the modulated value downstream. The current cell stores the modulated value (`cncp.py` lines 629 to 633), so the gain compounds through the sub-cell's own recurrence. Storing raw keeps the sub-cell's dynamics exactly those of the pluggable cell (orthogonality) and makes the gain a modulation of the *message*, not of the *memory*. The alternative is one flag.
- The relay target is a bounded drive times (1 - gate). This is the TRN motif in multiplicative form: Crandall's feedforward inhibition depresses the relay response and can only reduce it. The ablation "no TRN" fixes gate = 0; the ablation "free sign" replaces (1 - gate) by an unconstrained learned factor.

### B.4 Parameter count and compute cost

Treat the sub-cell as a black box with p(u_in, u) trainable parameters when built on a u_in-wide drive. Every node's basal drive has the node's own width, so p(u, u) throughout. Masked-off entries are excluded (`effective_param_count`, `ncp.py` line 323), so sparse edges count density × size. All counts below are *expected values* over the Bernoulli masks; the realised count differs by a few units per seed. Tables for the campaign are to be produced from the built model with `effective_param_count`, which must then also support stacked `(K, u_in, u_out)` weight tensors (B.7), and the formula is for planning only.

**One column (no head):**

```
P1(u) = 3 p(u,u)                                  # G, S, D
      + u²(E_GS) + u²(E_SD) + 0.5u²(E_DS) + 0.5u²(A_DS)
      + u·u_R(E_DR) + u·u_R(E_DRg) + u_R·u(E_RG)   # = 1.5u² at u_R = u/2
      + F_in·u (E_in)
      + u (γ) + 3 u_R (ρ, b_r, b_gate)
      = 3 p(u,u) + 3 u² + 3 u u_R + F_in u + u + 3 u_R   (exact, u_R = ⌊u/2⌋)
      = 3 p(u,u) + 4.5 u² + F_in u + 2.5 u          (even u only)
```

For `cfc_lrc` under `ncp_cell_kwargs` (backbone_layers = 0): four heads of (2u × u + u) plus the elastance head (2u × u + u) give **p(u,u) = 10u² + 5u**. Measured: 180 / 680 / 2640 at u = 4 / 8 / 16, which matches 10u² + 5u and not the 8u² + 4u quoted in spec section 13 and the 2026-10-04 work log; that formula is the plain `cfc` cell. So for `cfc_lrc`:

```
P1(u) = 34.5 u² + (F_in + 17.5) u
```

Verified with a scratch prototype of the v2 cell built from the repository's `CfC_LRC_Cell` and `ncp_cell_kwargs`: u = 10, F_in = 2 gives 3633 effective parameters (formula 3645; the difference is the realised count of the two Bernoulli(0.5) masks); u = 29, F_in = 7 gives 29663 (formula 29725); u = 14, F_in = 3 gives 7032 (7049); u = 7, F_in = 2 gives 1810 (1827).

**Head:** `Dense(n_out)` on D: (u + 1) n_out, as in every arm.

**Widths at the thesis budgets** (`size_for_budget` picks the integer u closest to the budget):

| Budget, task | F_in, n_out | u | P1 + head | Deviation |
|---|---|---|---|---|
| 4 000, Lotka-Volterra / Duffing | 2, 2 | 10 | 3 667 | -8.3 % (flagged, > 5 %) |
| 4 000 | 2, 2 | 11 | 4 395 | +9.9 % |
| 30 000, Person Activity | 7, 7 | 29 | 29 890 | -0.4 % |

At 4k the integer knob is coarse (one unit of u is 700 parameters). If the deviation flag matters, use u_R = u (P1 = 36u² + (F_in + 19) u, u = 10 gives 3 832 with head, -4.2 %) or let the matcher scan u_R too. For comparison the current cNCP sits at size 18 (4 082 effective, canary JSON) and size 50 (30 926 at F_in = n_out = 7, work log 2026-10-04); the NCP at inter = 43 (3 984) and 119 (29 640).

**Compute per step.** At these sizes the matrices are small and wall time is dominated by the number of sequential kernel launches; the counts below are launch estimates. `SparseLinear` runs a dense matmul on masked weights (`ncp.py` line 77), so effective parameter counts say nothing about FLOPs or memory. Rough op counts per step (one `cfc_lrc` sub-cell call ≈ concat + 5 matmuls + 8 elementwise ≈ 14 ops; one `SparseLinear` ≈ 2; one relay ≈ 6):

| Arm | Sub-cells | Edges | Relays | ≈ ops / step | Relative to NCP |
|---|---|---|---|---|---|
| NCP (`NCPLayeredCell`, 3 layers) | 3 | 0 | 0 | ≈ 50 | 1.0 |
| current cNCP | 6 | 15 | 2 | ≈ 135 | ≈ 2.7 (observed 16 min vs 7 min ≈ 2.3) |
| cNCP v2 | 3 | 8 | 1 | ≈ 70 | ≈ 1.4 |

These are estimates; the only measured number is the 16 vs 7 minutes. Wall time, peak memory and raw parameter counts are to be measured on the built arms before any compute statement enters the thesis.

**Counted after implementation (#9).** Operations in the traced graph of one `tf.function` step (`cfc_lrc`, batch 4; `Placeholder`, `ReadVariableOp`, `Const`, `Identity` and `NoOp` excluded; `tests/wirings/test_cncp2_wiring.py::_step_graph_ops`): cNCP v2 at u = 10: **100** ops (23 matmuls); its feedforward reduction: 66; current cNCP at size 18: **190** (45 matmuls); NCP at inter 43: **80** (15 matmuls). Relative to the NCP: v2 1.25 (estimate 1.4), current cNCP 2.4 (estimate 2.7). These are graph-op counts on CPU, not measured GPU kernel launches (grappler may fuse elementwise ops) and not wall time; no GPU was available for the count.

### B.5 Initialisation and stability rules, and the diagnosis of the canary divergence

**What happened.** Canary job 896122 (Lotka-Volterra, `cfc_lrc`, budget 4k, seed 0, 3 epochs; `cluster_archive/2026-10-21/thesis-person/results/_canary/`): the cNCP arm's validation loss was already 4.3e18 after the *first* epoch (`val_loss_curve` = [4.30e18, 4.19e18, 4.11e18]) and the teacher-forced test trajectories reach |x| ≈ 1.1e11 (denormalised units, `run_lotka_volterra_benchmark.py` line 182; the MSE is in normalised units); NCP (0.94) and dense3 (0.48) trained normally. The closed-loop MSE of the same cNCP model is finite (1.92) but not healthy: the runner averages over the whole 128-step horizon without clipping (lines 165 to 185), and the per-step error grows from 0.90 (first half) to 2.93 (second half) and 11.5 at the last step. Under teacher forcing the model receives the true, bounded trajectory at every step, so the 1e11 outputs are produced by one step of the trained model from bounded inputs: the trained weights map bounded states to unbounded outputs. A validation loss of 4.3e18 after epoch 1 shows the model was outside a usable regime after one epoch; it does not by itself show divergence before the first optimiser step. The initialisation measurements below are the evidence for that, and they are synthetic (N(0,1) inputs). The instrumented reproduction Codex asks for (real canary inputs and dt, identical builder, per-node state norms before training, after one update and after one epoch; backbone, gain and feedback varied one at a time; weights copied explicitly, because removing edges changes the order in which `SparseLinear` layers are built and initialised, `cncp.py` lines 416 to 424, so "same seed" does not mean "same weights") is the first item of the sequencing in the final section.

**Measured at initialisation** (scratch script, repository classes, no training, N(0,1) inputs, dt = 0.078125 (= 10/128, the canary's dt; an earlier draft of this paragraph misreported it as 1/128, the scripts used the canary value), 128 steps, state norm after step 1 / 32 / 128 and spectral radius ρ of the state-to-state Jacobian at the zero state):

| Cell at init | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| current cNCP, size 18, `cfc_lrc` | 0.86 / 2.2 / **210**, ρ 1.01 | 0.53 / 1.9 / 2.2, ρ 0.95 | 1.2 / 5.3 / **200**, ρ 0.97 |
| same, `combiner='additive'` | 1.1 / 2.6 / 3.5, ρ 0.96 | 0.74 / 2.6 / 2.5, ρ 0.95 | 1.7 / 4.2 / 4.4, ρ 0.97 |
| same, `feedforward_only=True` | 1.1 / 6.4 / **1.2e7**, ρ 1.10 | 0.55 / 1.5 / 1.4, ρ 1.00 | 2.2 / 4.9 / 4.3, ρ 1.00 |
| same, `cell_cls='gru'` | 0.43 / 1.5 / 2.3, ρ 0.96 | 0.47 / 1.3 / 1.6, ρ 0.95 | 0.49 / 1.2 / 1.8, ρ 0.95 |
| NCP, inter 43, `cfc_lrc` | 0.75 / 0.67 / 1.1, ρ 0.00 | 0.44 / 0.86 / 0.63, ρ 0.00 | 0.84 / 0.85 / 0.85, ρ 0.00 |
| dense `cfc_lrc` 16, backbone 1 (the dense/dense3 cell) | 0.92 / 2.1 / 2.2, ρ 1.06 | 0.84 / 2.1 / 1.8, ρ 0.62 | 1.1 / 1.2 / 2.0, ρ 0.75 |
| dense `cfc_lrc` 16, backbone 0 | 0.75 / 1.3 / 1.5, ρ 0.71 | 0.55 / 1.2 / 1.1, ρ 0.65 | 1.1 / 1.3 / 1.6, ρ 0.76 |

Two of three seeds of the current cNCP grow exponentially *at initialisation*, before any training; the feedforward reduction grows even faster in one seed; the GRU-based column does not grow; the NCP has no growing mode at all.

**Diagnosis.** Three facts combine; the first is the proximate cause.

1. **The sub-cell state candidates are affine and unbounded.** Since revision 13.1 (2026-10-04) the cNCP builds its CfC-family sub-cells through `ncp_cell_kwargs` with `backbone_layers = 0` (`cncp.py` line 432). In `src/neurons/cfc_lrc_cell.py` the heads `ff1`, `ff2`, `time_a`, `time_b` are `Dense(units)` **without activation** (`_head`, lines 118 to 121), and the state update `new_state = ff1 * (1 - t) + t * ff2` (line 191) mixes two affine maps of `[x, h]` (line 164) through a state-dependent gate t (sigmoid of `time_a`, `time_b`, line 185) and the elastance factor. The cell is nonlinear through its gates, but nothing bounds the candidates: without the backbone there is no squashing activation on the state path. The gate cannot rescue this: if `ff1` and `ff2` both map h to 1.1 h, `new_state = 1.1 h` for every gate value. The same holds for the ncps reference `WiredCfCCell` (ncps `keras/cfc_cell.py` lines 158 to 182: `ff1 = matmul + bias`), so this is faithful to the reference, and harmless there.
2. **The composite graph is cyclic with additive multi-edge fan-in.** Every `SparseLinear` is glorot-initialised independently of how many edges sum into a node (`cncp.py` lines 577, 587, 596, 601 add two drives each), and the delayed edges (`M_Thal_L4`, `M_L6CC_L23`, `M_L5ET_L5IT`, `M_ap_*`) close several cycles through the linear nodes. The spectral radius of the zero-state Jacobian is a local indicator for this input-dependent system, not a global stability statement; it is 0.95 to 1.10 at init across seeds and ablations, i.e. the column sits at the stability boundary by construction, and the norm trajectories show that two of three seeds leave it. Training moves the weights freely.
3. **The stored state is the gain-modulated value** (`cncp.py` lines 629 to 633), so when γ grows during training the factor compounds through the sub-cell's own recurrence. This is secondary (the additive ablation did not grow in three seeds), but it is why training can push a marginal system over the edge quickly.

Why the other arms are unaffected: the dense and dense3 arms build the cell *with* its backbone (`person_activity/model.py` line 98 and `lotka_volterra/model.py` line 106 pass `cell_kwargs` straight to `cell_cls`, default `backbone_layers = 1`, lecun-tanh, `cfc_lrc_cell.py` line 109), so the inputs of their candidate heads are bounded every step. This is also a confound of the existing comparison: cNCP and NCP run `backbone_layers = 0` via `ncp_cell_kwargs` (`ncp.py` line 15, `setdefault`, so an explicit `backbone_layers = 1` in `cell_kwargs` overrides it), dense and dense3 run 1; the same `cell` label denotes two different state maps. The NCP arm uses the same cell, but its state-to-state Jacobian at size 43 is nilpotent for most seeds: inter and motor layers have an empty recurrent mask, and the command layer receives `max(1, command // 2)` recurrent synapses *in total* (`ncp.py` line 44 and ncps `wirings.py` lines 543 to 549, which add `recurrent_command_synapses` synapses, not that many per neuron), i.e. 10 recurrent weights among 21 command neurons at size 43; measured ρ = 0.00 for seeds 0 to 3 and 5 to 7. The generator does allow command cycles and self-edges (ncps `wirings.py` lines 543 to 549): seed 4 at size 43 draws two self-edges and has ρ = 1.0. "The NCP has no growing mode" therefore holds per seed, not in general, and seeds are not to be selected on it. The cNCP is the arm in which unbounded candidates run inside a genuinely recurrent composite loop with multi-edge fan-in. The 2026-07-02 Lotka-Volterra run, with the backbone still in place, had a teacher-forced MSE of 0.0046 (work log); the explosion appeared with the backbone removal.

(The NCP observation has a side implication outside this document's scope: with `cfc_lrc` the NCP arm's inter neurons have no memory of their own state at all, because the concat mask zeroes the h-part of their heads. The project documents this as a deliberate departure from ncps' `fully_recurrent=True` (`ncp.py` lines 176 to 177). It is worth a separate note for the fairness discussion of H-c.)

**Stabilisation hypotheses for v2 (R1 to R5) and acceptance test (R6).** None of R1 to R5 bounds the within-sub-cell h → h map, so none is a stability guarantee; each removes one amplification path the current design has. Their effect at training time is untested until R6(b) runs.

- **R1, bounded feedback.** Every "prev" read passes through `tanh` before its mask (`E_RG`, `E_DS`, `A_DS`); the relay state is bounded by construction. Zero parameters. Consequence: every cross-node cycle is cut by a bounded function, so the only unbounded recurrence left is each sub-cell's own h → h map. The dense arm bounds that map's inputs through its backbone (`cfc_lrc_cell.py` line 109); v2 does not, and a bound there would need `backbone_layers = 1` or a squashed candidate, which changes the sub-cell and the NCP comparison. R1 has no measurable effect at init (tanh is the identity for small signals; see the table below); its intended value is at training time: no *cross-node* loop can carry an exponentially growing signal. It is a hypothesis about where growth would start, not a proof that it cannot.
- **R2, fan-in-aware edge init.** An edge into a node with n incoming edges is initialised with glorot variance divided by n, so the summed drive has the variance one edge would have. Zero parameters.
- **R3, gain identity at init.** γ = 0 exactly, centred combiner (B.3). The factor is exactly 1 at init and at zero apical drive.
- **R4, relay near-open and slow.** `b_gate = -2` (gate ≈ 0.12), ρ = 0 (α(1) = 0.5 as now); the relay cannot amplify (|target| < 1).
- **R5, raw state carry.** S stores s_t, not s̃_t (B.3).
- **R6, acceptance test** (before any campaign). (a) At initialisation, five seeds, task dt, 128 steps and 512 steps, with N(0,1) inputs *and* real task windows (normalised Lotka-Volterra at 4k, Person Activity at 30k), each at input scales 0.1, 1 and 10: states must stay finite and show no monotone growth over the second half of the horizon; the zero-state spectral radius is reported, not thresholded (the earlier "five times the step-1 norm" criterion depended on a randomly small step-1 value and is dropped). (b) A short training run with the canary settings (one epoch, same builder, same seed) must keep per-node states, loss and gradient norms finite, with the teacher-forced validation loss within a factor of ten of the dense3 arm's epoch-1 loss. (c) The current eight-node design is run through the same test as the control. Failing (a) or (b) rejects the design before any campaign; the current design fails (a) in two of three seeds with synthetic inputs.

**Measured on the v2 prototype** (u = 10, F_in = 2, five seeds; norm after 1 / 32 / 128 steps, ρ):

| v2 configuration | seeds 0 to 4 |
|---|---|
| default (R1 + R2 + R3 + R5) | 0.79/1.1/1.4 ρ .95 · 0.51/1.5/1.4 ρ .95 · 1.1/1.7/1.7 ρ .95 · 0.92/0.95/0.97 ρ .96 · 0.79/1.7/2.0 ρ .97 |
| without R1 | identical to default (signals too small for tanh to act) |
| without R2 (plain glorot) | 1.2/1.9/2.2 ρ .95 · 0.8/2.7/2.6 ρ .96 · 1.8/3.5/3.6 ρ .96 · 1.5/1.5/1.6 ρ .97 · 1.2/3.0/3.5 ρ .99 |
| without R1, R2; γ = 0.01; modulated state stored | 1.2/1.9/2.2 ρ .95 · 0.8/2.6/2.6 ρ .96 · 1.8/3.6/3.7 ρ .96 · 1.5/1.5/1.6 ρ .97 · 1.2/3.1/3.5 ρ .99 |
| default, u = 29, F_in = 7 (three seeds) | 1.2/1.9/2.0 ρ .95 · 1.2/1.8/1.9 ρ .95 · 1.6/2.5/2.4 ρ .96 |

Honest reading: at initialisation the four-node graph is stable in every configuration tested, because it has fewer cycles and fewer summed edges than the eight-node graph; R2 lowers the norms modestly; R1 and R5 target training time, are not visible at init and are untested until R6(b) runs. The eight-node graph is what makes the linear-cell problem bite. A smaller graph is therefore the first stability measure, R1 to R5 are the belt and braces.

### B.6 Relationship to the existing arms (kinship, not numerically exact reductions)

None of the four targets is reachable from v2 by a weight assignment with verifiable output or state equivalence; the table records architectural kinship and the settings that bring v2 closest.

| Target | v2 settings | Exact? |
|---|---|---|
| (i) current cNCP (8 nodes) | not reachable by settings: v2 has no L5IT/L5ET, L6CC/L6CT or Thal/TRN split and one gain target | **No.** The current implementation is the control arm (V2 in Part C). Mechanism-wise, "R1 off, R2 off, R5 off, γ init 0.01, uncentred combiner, TRN as a second relay with a sign-locked edge" recovers the current rules on the smaller graph. |
| (ii) NCP | `E_RG`, `E_DS`, `A_DS`, R removed (`feedforward_only`); `E_GS`, `E_SD` masks set from the NCP fan-out rules (`ncp_graph`); sub-cell recurrence masks: none for G and D, the NCP command adjacency for S | **No** (kinship). The chain input → G → S → D mirrors inter → command → motor of `NCPLayeredCell` (G = inter, S = command, D = motor, the reading used in B.2). It is not the same map: `NCPLayeredCell` passes the sensory and recurrent masks into each sub-cell's heads (`ncp.py` line 214, `base_cell.py` line 123), whereas v2 masks a `SparseLinear` in front of a densely mixing sub-cell, and the product M·W can realise input dependencies the per-head mask excludes. With dense edges and full sub-cell recurrence v2 is the `cncp_ff` control of spec 7.1 instead. |
| (iii) dense3 | `feedforward_only`, all densities 1.0, no recurrence masks | **No** (kinship). dense3 feeds layer outputs directly (no inter-layer `SparseLinear`) and builds its cells with `backbone_layers = 1` (`lotka_volterra/model.py` line 106, `person_activity/model.py` line 98), while v2 sub-cells use `ncp_cell_kwargs` (backbone 0). The earlier claim that the extra u² map is "redundant in function" is withdrawn: with a tanh backbone between input and heads there is no linear reparametrisation from one to the other. |
| (iv) current weight-shared committee (`MultiColumnVotingCell`) | K columns, `share_weights=True`, no lateral/hub edges, consensus mix on S with λ (`VotingMixin`), readout = consensus of S or of D, partial views | **No** (kinship). The same `consensus_mix` operator can be applied to S of K weight-shared v2 columns, but the existing `MultiColumnVotingCell` wraps the eight-node `TbtCorticalColumnCell`; a four-node committee is a new arm, not the existing one. |

### B.7 Multi-column extension with independent weights

**Columns.** K copies of the v2 column, each with its own parameters (no sharing). Implementation note: independent weights do not require a Python loop over K; stacking the K weight matrices of each edge and head as `(K, u_in, u_out)` tensors and using one `einsum('bki,kio->bko')` per edge keeps the number of sequential kernel launches independent of K. FLOPs, activation memory and the backward pass still scale with K, and lateral coupling with K², so wall time rises with K, less than linearly only while launch latency dominates. The current committee loops over K in Python (`committee.py` lines 198 to 200, `tbt_cncp.py` lines 200 to 205); it can be vectorised the same way by folding K into the batch axis, so a "V1 loop vs V4 einsum" timing would measure implementation, not architecture. Compute comparisons must use the same implementation style and must be measured (wall time, peak memory, raw and effective parameters).

**Input partition.** Two modes, chosen per task:

- `partition='region'`: column k receives a fixed subset of the input features (retinotopy). On Person Activity each row is one reading of one of four tags (ankle left, ankle right, belt, chest; `datasets.py` line 74, feature = one-hot(tag) ++ xyz), so K = 4 is the natural partition. Column k receives x_k = [m_k, m_k · xyz] with the explicit observation flag m_k = 1[tag = k], F_in = 4 per column; the one-hot becomes redundant. Semantics, fixed here: every column advances at every event with the shared dt (the gap to the previous row of *any* sensor, `datasets.py` lines 73 to 74), so an inactive column performs a regular `cfc_lrc` update with input [0, 0, 0, 0] over dt. The flag is what distinguishes "no reading" from "a reading at the origin"; bare zeros would not, and dt = 0 would not freeze the cell either (`cfc_lrc_cell.py` line 185). The alternative, updating only the active column with the time since its own last reading, needs a per-column dt and a masked state write; it is the cleaner model of asynchronous sensors and is an option, not the default. Control required: a partitioned dense or GRU arm with the same [m_k, m_k · xyz] layout, so that a V4 gain can be attributed to the column wiring and not to the input routing. On the 2-D rollouts a region partition is degenerate (one state dimension per column); use `same`.
- `partition='same'`: every column receives the full input (plus its own corruption in the ρ study, as the committee does now). This is the "same job, vote" case.

**Coupling.** Two modes, mapped from A.5; a third is the control.

- `coupling='lateral'` (A.5.1, L2/3 horizontal axons). Column c's integration node S receives, at step t, an additive drive from the other columns' S at t-1: `basal_S_c ⊕= Σ_{c'≠c} L_{c'c} tanh(s_{c'}(t-1))`, with `L_{c'c}` a `SparseLinear(u, u)` of density d_lat (default 0.25) and independent weights. Parameters: K(K-1) d_lat u². Reads prev, bounded (R1). Why S and not D: the best-documented inter-column excitatory path within an area runs between L2/3 populations (A.5.1) and modulates the surround, which a sparse additive edge onto the integration node expresses. What the model does not keep: the feature selectivity and spatial alignment Bosking describes. A Bernoulli mask between anonymous hidden units relates no unit of column c to any particular unit of column c'; the mask supplies sparsity, the learned weights have to supply any selectivity. The lateral drive is additive, not a convex mix: a convex mix of states (the current `consensus_mix`) forces the columns toward one representation and is the mechanism the July committee tested and lost with; an additive sparse edge lets each column keep its own code and read the others as context, which is what Hawkins et al. describe as "modulatory input" that biases the local winner.
- `coupling='hub'` (A.5.3, transthalamic route; **deferred to the outlook**, defined here so that the cost is on record). One shared gated relay H of width u_H (default ⌊u/2⌋), updated after all columns' D at step t and read by every column at t+1:

  ```
  drive_H = Σ_c W_c^{DH}  tanh(d_c(t))                    # K maps u × u_H
  gate_H  = sigmoid(Σ_c W_c^{DHg} tanh(d_c(t)) + b_H)     # K maps u × u_H, own gate projection, b_H init -2
  H_t     = (1 - α_H) H_{t-1} + α_H · tanh(drive_H + b_rH) · (1 - gate_H),  α_H = 1 - exp(-dt softplus(ρ_H))
  basal_S_c(t+1) ⊕= W_c^{HS} tanh(H_t)                    # K maps u_H × u, read as prev
  ```

  Parameters: 3 K u u_H + 3 u_H (drive, gate and return projection per column; three u_H vectors ρ_H, b_rH, b_H) = 1.5 K u² + 1.5 u at u_H = u/2. Unlike lateral, the hub returns column c's own output to itself (D_c → H → S_c), so it adds self-feedback through a compressed state; a hub-vs-lateral comparison varies self-feedback and compression as well as topology. Cost at K = 4, u = 14: lateral d_lat 0.25 = 3u² = 588, hub = 6u² + 21 = 1 197; the two are not equal at K = 4. Why it is defined at all: it scales linearly in K where lateral scales quadratically, and it is the only anatomical inter-column path with its own gate. H is "higher-order thalamus" in the same sense that R is "first-order thalamus": a label.
- `coupling='none'`: independent columns, readout only. The control that separates "more, smaller, independent learners" from "coupled learners".

Lateral is the default and the only coupling in the five-week plan; hub belongs to the outlook. The two modes are not nested; each is tested against `none`. Two comparisons must be kept apart: `none` vs `lateral` at the *same u* (isolates the mechanism, lateral has more parameters) and at *re-matched u* (same budget, lateral columns are narrower); they answer different questions and both are reported.

**Readout.** `readout='concat'`: `Dense(n_out)` on `[d_1 … d_K]`, (K u + 1) n_out parameters; the right choice under `region`, where columns carry different information. `readout='mean'`: `Dense(n_out)` on the mean of the d_c, parameter-free consensus; the choice under `same`, where columns do the same job. Mean before a shared linear head equals the mean of the projected outputs, so it is well defined over independently learned hidden spaces; the readout is symmetric in the columns, but the network as a whole is not permutation-invariant over input slots, since each f_c and L_{c'c} is slot-specific. `readout='median'` (fault-tolerance control): a coordinate-wise median over *hidden* coordinates of independently trained columns has no shared meaning, so the median is taken over task-aligned quantities: each column gets its own head `Dense(n_out)` (K (u + 1) n_out parameters, counted) and the lower median is taken over the K logit vectors. In the ρ corruption study the corruption passes through the coupled system; coupling can propagate a corrupted column's state, and independent weights do not make errors independent. Hawkins et al. motivate learned lateral support, not a fault-tolerance guarantee for this architecture.

**Parameters for K columns at total budget B:**

```
P_K(u) = K · P1(u)                                 # independent columns
       + C(K, u)                                    # coupling: 0 | K(K-1) d_lat u² [lateral] | 1.5 K u² + 1.5 u [hub, u_H = u/2]
       + head: (K u + 1) n_out [concat] | (u + 1) n_out [mean] | K (u + 1) n_out [median]
```

with P1(u) in its exact form (u_R = ⌊u/2⌋), and u is the largest integer with P_K(u) ≤ B (or the closest, as `size_for_budget` does). Expected values for `cfc_lrc`; realised counts come from the built model:

| Budget | K | partition / coupling | F_in per column | u | P_K | Note |
|---|---|---|---|---|---|---|
| 4 000 | 1 | – | 2 | 10 | 3 667 | V3 |
| 4 000 | 2 | same / lateral (d_lat 0.5), mean | 2 | 7 | 3 695 | -7.6 %; 1 815 expected per column (prototype realised 1 810) |
| 4 000 | 3 | same / lateral (d_lat 0.5), mean | 2 | 5 | 2 940 | -26 %; u = 6 already exceeds 4k. Columns of 5 units: narrower than any arm that has fitted the rollouts (dense3 uses 13), which is a reference point, not a measured minimum |
| 30 000 | 1 | – | 7 | 29 | 29 890 | V3 |
| 30 000 | 4 | region (one per tag) / lateral (d_lat 0.25), concat | 4 | 14 | 29 239 | -2.5 %; 7 063 expected per column |
| 30 000 | 4 | region / hub (u_H 7), concat (outlook) | 4 | 14 | 29 848 | -0.5 % |
| 30 000 | 30 | same / hub, mean (outlook) | 7 | 5 | 30 228 | +0.8 %; columns of 5 units |
| 30 000 | 300 | same / hub, mean | 7 | 1 | 19 517 | fits arithmetically; u = 2 gives 57 924. A one-unit column is not a model |

**Sensible K.** At 4k, K = 2 is the only multi-column setting that leaves a column of more than five units (u = 7); V4 at 4k is a feasibility check, the hypotheses are tested at 30k. At 30k, K = 4 with the sensor-tag partition is a pragmatic choice: it matches the task's structure and keeps u = 14 per column. Independent weights do not cap K arithmetically (K = 300 at u = 1 fits under 30k with a hub); they trade K against width per column: u = 14 at K = 4, 5 at K = 30, 1 at K = 300. Whether five units suffice is unknown; the width at which dense3 fits (13 at 4k, 36 at 30k) is a reference point, not a measured minimum for a v2 column. K ≈ 4 is therefore a judgement about minimum useful width, to be checked with a K ∈ {1, 2, 4, 8} sweep if time allows; the H-d regime K = 300 remains a weight-sharing regime. The "many weak columns" and the "independent learners" regimes exclude each other at a fixed budget, which the thesis should state.

**Why this coupling and not another.**

- Not a convex mix of states (the committee's `consensus_mix`): it was refuted by the dropout-trained monolith (iteration 7b), and with mean consensus the readout is independent of λ (committee.py header). It survives only as reduction (iv).
- Not coupling at G (input layer), by design choice: Narayanan et al. (2015) describe transcolumnar L2/3 axons that reach the granular layer, so a lateral edge into G is anatomically admissible. It is excluded because coupling the entry node would let the columns share input before they have formed their own representation, which defeats the region partition.
- Not coupling at D alone: deep horizontal projections exist (Egger et al. 2020), but D is also the readout, and the hub reads D already; a direct D to D lateral edge is a possible third mode (`coupling='deep_lateral'`), not a default.
- Not a learned attention over columns: parameter cost and a second mechanism; the budget is the constraint.

### B.8 Limitations and what would falsify the design

**Limitations.**

- Delayed-state feedback is a one-step approximation of recurrent settling, as before.
- The four-node graph is less faithful to the laminar anatomy than the eight-node graph: CT and PT cells are merged, and the intracortical L6 to FS gain route, the empirically larger one, is absent. A null result on gain therefore tests the thalamic-loop reading of L6 gain control only.
- Densities and width proportions remain design choices (spec section 10).
- Independent weights at a fixed budget trade K against width per column (u = 14 at K = 4, 5 at K = 30, 1 at K = 300 at 30k); the H-d sweep to K = 300 is a weight-sharing regime.
- R1 to R5 are stabilisation hypotheses; the within-sub-cell recurrence stays unbounded, and the initialisation measurements use synthetic inputs (R6 adds real sequences and a short training run).
- V2 vs V3 changes node count, width distribution, feedforward depth, feedback activation, initialisation, gain centring, state carry and relay dynamics at once: it compares two architecture packages. Before a V3 result is attributed to the four-node abstraction, a stabilised eight-node control (V2s, Part C) must exist. The existing arms also differ in backbone (cNCP and NCP 0, dense and dense3 1), so "same cell label" does not mean "same state map"; a pure wiring comparison needs the backbone aligned and budgets re-matched, or the comparison is declared one of architecture packages and read that narrowly.
- The lateral mask does not encode feature-selective topography (B.7).
- In the ρ corruption study, coupling can propagate a corrupted column's state; independent weights do not make errors independent.
- The region partition is natural only on Person Activity; on the rollouts the multi-column arm tests voting on identical inputs, which the July study already found not to help under weight sharing.
- The dense3 reduction is inexact (extra inter-layer maps); the current cNCP is not a reduction of v2 at all (B.6).
- No random-rewired control is planned (decision of 2026-10-04); a cNCP v2 win over dense3 can therefore be attributed to sparsity-plus-structure, not to this structure specifically.
- The compute estimates in B.4 are op counts, not measurements.

**Falsification on the thesis tasks.**

- *Acceptance (before any campaign):* R6 fails for v2 at the matched widths, or the teacher-forced loss on Lotka-Volterra at init is not finite over five seeds. Then the design is rejected without further runs.
Statistics as in chapter 5 (lines 84 to 88): five seeds, paired comparisons, median paired difference with its range; with five pairs the smallest exact two-sided Wilcoxon p is 0.0625, so the test is a formality and non-significance is never read as equivalence. "No advantage" below means: the median paired improvement is below a pre-registered tolerance (proposal: 5 % relative NRMSE on the rollouts, 1 macro-F1 point on Person Activity) and the five paired differences straddle zero. Each criterion falsifies the named claim in this setting, not the family of designs.

- *Lotka-Volterra / Duffing (H-c):* V3 shows no advantage in closed-loop NRMSE over dense3 at equal budget. Then no benefit of the column wiring on rollouts is demonstrated; the thesis reports the effect size and the range.
- *Person Activity (H-c):* V3 shows no advantage in macro-F1 over dense3 and over NCP at 30k. Then no benefit of the laminar prior is demonstrated at this budget.
- *Person Activity, independent columns:* V4 (K = 4, region, lateral) shows no advantage over V3 and over the partitioned dense/GRU control at the same 30k, and `coupling='none'` and `lateral` lie within the tolerance of each other. Then neither partition nor coupling has a demonstrated benefit in this setting; it does not show that coupling is useless.
- *ρ column corruption:* V4 with median readout shows no advantage over the dropout-trained dense3 monolith at ρ ∈ {0.3, 0.5}. Then independent columns *with this coupling and readout* are refuted on the same task as the weight-shared committee; other readouts and couplings are not covered.
- *Motif ablations:* the decisive test is the retrained ablation (γ fixed at 0, gate fixed at 0, `E_DS` removed), each compared with the full V3 under the same tolerance. Learned-parameter readings are hints only: |γ| < 0.05 after training does not prove the gain edge inert (a small modulation can be amplified downstream), and a gate that is open on average can still act on rare events, so the gate's distribution over time is reported, not its mean. A negative retrained ablation is reported as "no demonstrated contribution of the motif on this task".

---

## Part C: the four comparison arms

All arms: same inputs (features, dt), same `Dense(n_out)` head, same effective parameter budget via `size_for_budget`, same optimiser and seeds. The sub-cell configuration (backbone 0 for cNCP/NCP, 1 for dense/dense3) is to be documented per arm; it is a confound of the existing comparison (B.5). p(u,u) = 10u² + 5u for `cfc_lrc` throughout; "ops" are the sequential kernel-launch estimates of B.4.

### V1: current committee (status quo for H-d)

K weight-shared cNCP columns (`MultiColumnVotingCell`, eight-node `TbtCorticalColumnCell` with location off) on partial views; after each step the L2/3 states are pulled toward their mean (or lower median) with learnable λ; the output is the consensus of the L2/3 states, read by the head. **Parameters:** P_cNCP(size) + 1 (λ) + (u_L23 + 1) n_out, independent of K; size 18 at 4k, 50 at 30k. **Compute:** K × (≈135 launches) per step in the current implementation, because the shared column is called K times in a Python loop (an implementation property; folding K into the batch axis removes it); at K = 300 this is 300 column evaluations per step. **Tests:** whether consensus beats input averaging and λ = 0 at the same K (H-d). **Cannot test:** anything about independent weights or the laminar graph (every column computes the same function); with mean consensus the readout is independent of λ, so λ acts only through later steps. **Expected failure modes:** already refuted against the dropout monolith (iteration 7b); shares the initialisation instability of V2 (same cell); K × compute. **Cost:** 4k: size 18, K × 135 ops; 30k: size 50, K × 135 ops.

### V2: current cNCP, single column, scaled in width (status quo for H-c)

The eight-node `CorticalColumnCell` with `scaled_lamina_units(size)` (proportions 8:8:6:8:4:4:4:2 scaled by size/16). **Parameters:** Σ_nodes p(u_n, u_n) over the six sub-cells + Σ_edges density × shape + relay (2 × 2 u) + gains (u_L23 + u_L5ET) + head; measured 4 082 at size 18 (node widths 9/9/7/9/4/4/4/2) and 30 926 at size 50 (25/25/19/25/12/12/12/6), both `cfc_lrc`. **Compute:** ≈ 135 ops per step, ≈ 2.7 × NCP (observed 16 vs 7 min). **Tests:** whether the full laminar graph beats dense3 and NCP at equal budget (H-c). **Cannot test:** which motif matters (needs the ablation flags, which exist: `combiner`, `feedforward_only`, `sign_constraint`); nothing about multiple columns. **Expected failure modes:** teacher-forced divergence at init in a fraction of seeds (B.5: two of three seeds grow exponentially at init at size 18; the canary diverged to 4.1e18); per-node widths of 4 and 2 at 4k, so the deep nodes and relays are nearly scalar; closed-loop damping to a fixed point seen in July. **Cost:** 4k: size 18, 135 ops; 30k: size 50, 135 ops.

### V2s: stabilised eight-node cNCP (control required before V3 is read)

The eight-node `CorticalColumnCell` with the least invasive stabilisation: (a) `cell_kwargs={'backbone_layers': 1}`, which `ncp_cell_kwargs` honours as an explicit override (`ncp.py` lines 17 to 19) and which restores the bounded candidate inputs the dense arms have; (b) alternatively the current graph with R1 (tanh on the delayed reads), R3 (γ = 0, centred combiner) and R5 (raw state carry) applied, no backbone. **Parameters:** re-matched with `size_for_budget`; with backbone 1 each sub-cell gains a (2u × u_b + u_b) backbone layer, so the size drops below 18 and 50. **Compute:** ≈ 135 launches plus one per sub-cell. **Tests:** whether the canary divergence is a sub-cell property (a stabilises, b does not), a graph property (b stabilises) or both; and, as the control arm for V3, whether a V3 result survives against an eight-node column that does not diverge. **Expected failure modes:** (a) changes the sub-cell and breaks the "same sub-cell as NCP" convention of revision 13.1, so the NCP arm would need the same override for a fair H-c; (b) may not be enough, since the within-node recurrence stays unbounded. **Cost:** 4k and 30k at re-matched sizes, to be counted from the built model.

### V3: cNCP v2, single column, scaled in width

The four-node column of B.3 with rules R1 to R5. **Parameters:** P1(u) + (u + 1) n_out, with the exact P1 of B.4 (34.5 u² + (F_in + 17.5) u for even u); u = 10 (3 667) at 4k, u = 29 (29 890) at 30k; expected values. **Compute:** ≈ 70 launches per step, ≈ 1.4 × NCP by launch count, about half of V2; to be measured. **Tests:** whether the three motifs alone, in a stable column, beat dense3 and NCP at equal budget (H-c), and, through the flags (`E_DS` off, γ fixed 0, gate fixed 0, `feedforward_only`), which motif carries any effect. **Cannot test:** anything about L5/L6 or Thal/TRN sub-structure (merged away); anything about columns. **Expected failure modes:** γ stays at zero (gain unused on these tasks, as the July studies suggest for structural priors with `cfc_lrc`); the relay loop damps the output (as in July); at 4k the -8.3 % budget deviation is flagged. **Cost:** 4k: u = 10, 70 ops; 30k: u = 29, 70 ops.

### V4: cNCP v2, K independent columns, coupled

K columns of V3 with independent weights, `partition` and `coupling` per B.7, `readout='concat'` under region and `'mean'` under same. **Parameters:** P_K(u) = K P1(u) + C(K, u) + head; K = 2, u = 7, same/lateral(0.5), mean: 3 695 at 4k; K = 4, u = 14, region (F_in 4)/lateral(0.25), concat: 29 239 at 30k (hub, outlook: 29 848); expected values, realised counts from `effective_param_count`. **Compute:** with stacked-weight einsum ≈ 70 launches per step regardless of K, while FLOPs and memory scale with K and lateral with K²; with a Python loop K × 70; to be measured. **Tests:** (a) at 30k on Person Activity, whether four independent per-sensor columns with a lateral or hub coupling beat one column of the same budget (V3) and the dense3 monolith; (b) `coupling='none'` vs `lateral` vs `hub` at the same K and partition isolates the coupling; (c) on the ρ corruption task with `partition='same'` and median readout, whether independent columns resist faults where the weight-shared committee did not. **Cannot test:** K beyond about 4 at a useful width (budget); the H-d slope over K ∈ {3, 30, 300}; on the rollouts, anything beyond a feasibility check at K = 2. **Expected failure modes:** columns of 7 units (4k) or 14 units (30k) underfit individually and the coupling cannot compensate; the concat head at K = 4 grows to (4 u + 1) n_out and takes budget from the columns; a region partition starves columns whose sensor fires rarely; on `same` inputs the result repeats the July null. **Cost:** 4k: K = 2, u = 7, 70 ops (einsum) or 140 (loop); 30k: K = 4, u = 14, 70 or 280 ops.

### Summary

| Arm | Nodes | Params (4k / 30k) | ops/step | Primary question | Known risk |
|---|---|---|---|---|---|
| V1 committee | 8 × K shared | 4 082 / 30 926 (+1) | K × 135 | consensus vs averaging (H-d) | refuted vs dropout monolith; shares V2 instability |
| V2 current cNCP | 8 | 4 082 / 30 926 | 135 | laminar graph vs dense3, NCP (H-c) | init divergence (canary) |
| V2s stabilised cNCP | 8 | re-matched | ≈ 141 | is the divergence a sub-cell or a graph property; control for V3 | backbone override breaks the NCP sub-cell convention |
| V3 cNCP v2 | 4 | 3 667 / 29 890 | 70 | three motifs vs dense3, NCP, V2s (H-c); motif ablation | motifs unused (γ ≈ 0); eight-way package change vs V2 |
| V4 v2 × K independent | 4 × K | 3 695 (K 2) / 29 239 (K 4) | 70 launches (einsum), FLOPs K × | independent columns + coupling vs monolith and partitioned control | columns too narrow; outlook per review |

"ops" are launch-count estimates (B.4); parameter counts are expected values and must be regenerated from the built models.

---

## Implementation status (2026-10-21, #9)

Implemented in `src/wirings/cncp2.py` (`CorticalColumnV2Cell`, `CNCP2Wiring`, `MultiColumnV2VotingCell`), registered as wiring `cncp2` in `src/wirings/__init__.py`, `src/benchmark/registry.py` (`WIRINGS`), the Lotka-Volterra, Person Activity and committee builders, and the neural-ODE run path (`experiments/run_benchmark.py::_make_net`, because `WIRINGS` feeds campaign-config validation). Tests: `tests/wirings/test_cncp2_wiring.py`, `tests/wirings/test_cncp2_stability.py`. The update equations of B.3 are implemented as written: tanh on every delayed read (R1), centred gain with γ = 0 (R3), ρ = 0 and b_gate = -2 (R4), raw S stored and s̃ passed to D (R5), output d.

**Choices and deviations from B.3 / B.4.**

- **R2 (fan-in-aware init).** Each `SparseLinear` is drawn glorot-uniform and then scaled by 1/√n, n = number of edges summed into its target node under the active flags (variance / n). Full column: E_in and E_RG into G (n = 2), E_GS and E_DS into S (n = 2); E_SD, E_DR, E_DRg and A_DS have n = 1 (A_DS feeds the gain, not a sum). Mask density is not taken into account (E_DS at density 0.5 has half the realised fan-in). With `feedback=False` S has n = 1, with `feedforward_only` every edge has n = 1.
- **`relay_gate='free'`.** B.3 only says "an unconstrained learned factor". Implemented as the linear factor E_DRg(d) + b_gate in place of (1 - gate), with b_gate initialised to 1 - sigmoid(-2) ≈ 0.881, so 'free' and 'attenuate' coincide at d = 0 at initialisation. The relay target is then unbounded; its read into G still passes tanh (R1).
- **`relay_gate='off'`.** E_DRg and b_gate are not built (no dead parameters); the relay keeps ρ and b_r, so P1 loses u·u_R + u_R.
- **`feedforward_only=True`** overrides `gain` and `relay_gate`: E_RG, E_DS, A_DS and the relay are not built, the R slot stays zero, the state layout is unchanged (as in the eight-node cell).
- **Build order.** A_DS and γ are created last, so `gain=True` and `gain=False` columns with the same global seed draw identical weights for every common variable; the feedforward reduction draws its weights in the order of a hand-built three-layer stack. Both reduction tests therefore compare same-seed models without copying weights (and assert the weights coincide).
- **Masks.** The full eight-edge inventory is drawn once per seed in fixed order with `cncp._make_mask` (dense masks consume no RNG, an all-zero sparse draw is repaired with one connection), so all ablations of one seed share the two Bernoulli(0.5) masks.
- **Committee arm.** B.6 (iv) leaves the readout open (consensus of S or of D). `MultiColumnV2VotingCell` pulls the stored (raw) S states toward their consensus and outputs the consensus of the mixed S, the analogue of the L2/3 vote and readout of the `cncp` committee.
- **Width knob.** The builders use u = size directly (no proportional scaling); `CNCP2Wiring` defaults to u = 8, the neural-ODE legacy path to u = 16 (`DENSE_UNITS`).

**Parameter count (B.4), measured.** `effective_param_count` of the column without head equals the exact formula with the realised mask counts for u = 4, 8, 10, 16, 29 (test at 4 / 8 / 16). With F_in = 2, seed 0: 627 / 2 368 / 9 141 at u = 4 / 8 / 16 (expected-value formula 630 / 2 364 / 9 144). Budget sizing through the builders (`size_for_budget`, `cfc_lrc`, asymmetric elastance, with head): Lotka-Volterra 4k → **u = 10, 3 655** (-8.6 %, above the 5 % tolerance flag; u = 11 gives +9.9 %), Person Activity 30k → **u = 29, 29 873** (-0.4 %). The current cNCP lands at size 18 (4 082) and 50 (30 926), as in B.4.

**R6 as implemented (issue #9 form).** (a) Builders at 4k (LV, F_in = 2, dt = 10/128) and 30k (PA, F_in = 7, dt = 1), seeds 0 to 4, 128 steps of N(0, scale²) inputs at scale 0.1 / 1 / 10, criterion: max per-sample norm of the full state < 10 × mean per-step input norm. (b) One teacher-forced epoch on the Lotka-Volterra data (40 trajectories, seq 128, batch 16, RMSprop 1e-3), train and validation loss finite and below 10. The bound in (b) is added to the issue's "finite" because the eight-node control reaches finite losses of 1e18 (the constant replaces "within ten times dense3", whose epoch-1 validation loss is 0.41 to 1.03 here). Not implemented from R6: 512-step horizons, real task windows as inputs, gradient-norm checks, the monotone-growth criterion.

**R6 results.**

| | (a) cases passed | (a) state / input norm ratio | (b) epoch-1 val loss, seeds 0 to 4 |
|---|---|---|---|
| cNCP v2 | 29 / 30 | 0.77 to 4.0; one case **4.2e9** | 0.92, 0.64, 0.77, 0.70, 0.81 |
| current cNCP | 6 / 30 | scale 0.1: 11 to 93; scale 1: 2.0 to 2.3e7; scale 10: 1.1e6 to 2.8e13 | 1.4e10, 0.65, 7.7e7, 4.8e17, 0.86 |
| dense3 (reference, size 13) | not run | not run | 0.74, 0.68, 0.72, 1.03, 0.41 |

The current cNCP passes (a) only at scale 1 (LV seeds 1, 3, 4; PA seeds 1, 2, 3) and fails (b) in seeds 0, 2 and 3; it is the xfail control. Its scale-0.1 failures are partly an artefact of the ratio criterion: the sigmoid TRN relay keeps the state away from zero when the input shrinks, so the ratio grows without any growth in time.

**Open: R6(a) is not met in one case.** cNCP v2, Lotka-Volterra 4k, seed 1, input scale 10: the S and D norms grow from about 20 at step 8 to 4e10 at step 128 while G stays near 25 and R below 2. The S sub-cell alone, driven by the G chain with every cross-node edge removed (no E_DS, no gain, no relay), grows to 6e10 on the same input, so the growth is the S sub-cell's own h → h map (`cfc_lrc`, backbone 0, affine candidates under saturated gates), the case B.5 (R1) and Codex finding 2 name: R1 to R5 do not bound it. It occurs in one of 30 cases, but not as a one-off draw: ten standalone backbone-0 `cfc_lrc` cells of width 10 at scale 10 stay at ratio ≈ 1.6, while a second, independently drawn feedforward-only column (seed 1, built outside the builder) also grows (1.4e7) and a column with `backbone_layers=1` stays bounded (norms ≤ 4.3). The risk is a property of backbone-0 `cfc_lrc` sub-cells in a chain under large inputs. The test marks this case `xfail(strict=True)`. Closing it needs a sub-cell change (`backbone_layers = 1` or a squashed candidate), which also changes the NCP comparison (B.5, finding 13 pushback); that decision is outside #9.

**Runner probe for the canary gate.** `experiments/run_lotka_volterra_benchmark.py --probe-stability` (default off) adds a `stability` block to the result JSON for the model the runner actually trains: per-state-tensor L2 norm (max over batch and time, on the first `batch_size` training sequences, real dt) at init, after exactly one optimiser step and after epoch 1 (`norms_init`, `norms_step1`, `norms_epoch1`, keyed `<rnn layer>/<state>`: the eight node names for cncp, G/S/D/R for cncp2, `state0..` for ncp (one concatenated state) and per dense layer), the epoch-1 train loss `tf_loss_epoch1` and `finite`. The probe is a Keras callback running forward passes through read-only state taps; a three-epoch cncp2 run gives an identical validation-loss curve with and without the flag.

**GPU.** The single-step `tf.function` test runs on `/GPU:0` when TensorFlow sees one, otherwise on CPU; it has only run on CPU so far (macOS, no `tensorflow-metal`).

## References not in the vault list (`cNCP 08`)

Bosking WH, Zhang Y, Schofield B, Fitzpatrick D (1997) Orientation selectivity and the arrangement of horizontal connections in tree shrew striate cortex. J Neurosci 17:2112. https://pmc.ncbi.nlm.nih.gov/articles/PMC6793759
Chisum HJ, Mooser F, Fitzpatrick D (2003) Emergent properties of layer 2/3 neurons reflect the collinear arrangement of horizontal connections in tree shrew visual cortex. J Neurosci 23:2947. https://pmc.ncbi.nlm.nih.gov/articles/PMC6742080
Egger R et al. (2020) Cortical output is gated by horizontally projecting neurons in the deep layers. Neuron 105:122. https://pmc.ncbi.nlm.nih.gov/articles/PMC6953434
Halassa MM et al. (2014) State-dependent architecture of thalamic reticular subnetworks. Cell 158:808.
Harris KD, Shepherd GMG (2015) The neocortical circuit: themes and variations. Nat Neurosci 18:170. https://pmc.ncbi.nlm.nih.gov/articles/PMC4889215
Hawkins J, Ahmad S, Cui Y (2017) A theory of how columns in the neocortex enable learning the structure of the world. Front Neural Circuits 11:81. https://www.frontiersin.org/articles/10.3389/fncir.2017.00081/full
Horton JC, Adams DL (2005) The cortical column: a structure without a function. Phil Trans R Soc B 360:837. https://redwood.berkeley.edu/wp-content/uploads/2018/08/horton05.pdf
Markov NT et al. (2014) Anatomy of hierarchy: feedforward and feedback pathways in macaque visual cortex. J Comp Neurol 522:225. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4255240/
Narayanan RT et al. (2015) Beyond columnar organization: cell type- and target layer-specific principles of horizontal axon projection patterns in rat vibrissal cortex. Cereb Cortex 25:4450. https://pmc.ncbi.nlm.nih.gov/articles/PMC4816792
Perez E, Strub F, de Vries H, Dumoulin V, Courville A (2018) FiLM: visual reasoning with a general conditioning layer. AAAI. https://arxiv.org/abs/1709.07871v2
Thousand Brains Project (2024) The Thousand Brains Project: a new paradigm for sensorimotor intelligence. arXiv:2412.18354. https://arxiv.org/abs/2412.18354
Thousand Brains Project (2025) Thousand Brains Systems: sensorimotor intelligence for rapid, robust learning and inference. arXiv:2507.04494. https://arxiv.org/html/2507.04494v1
Wimmer RD et al. (2015) Thalamic control of sensory selection in divided attention. Nature 526:705. https://pmc.ncbi.nlm.nih.gov/articles/PMC4626291

---

## Codex review (gpt-6-astra, 2026-10-21)

Independent review by OpenAI Codex, model `gpt-6-astra`, reasoning effort high, read-only sandbox, run from the repository root on the draft of this document (prompt: `codex-cncp-v2-prompt.md`, output: `codex-cncp-v2-review.md`, log: `codex-cncp-v2.log`, all in the session scratchpad). Every claim Codex makes about the code was checked against the cited file and line before it was accepted; the findings are reproduced verbatim (German, as written), numbered as in the review, each followed by the response. Codex notes that it could not run TensorFlow in its read-only environment, so the scratch measurements of B.5 are not independently reproduced.

### Overall verdict (verbatim)

> **Ich würde den Entwurf noch nicht zur Implementierung freigeben.** V3 ist eine vertretbare Vereinfachung als ML-Architektur. Die behaupteten Reduktionen, Stabilitätsgarantien und Teile der Budgetrechnung tragen aber nicht. Für die verbleibenden fünf Wochen würde ich den bestehenden cNCP stabilisieren und evaluieren; V3 käme als begrenzte Zusatzablation infrage, V4 als Ausblick.
> 
> Ich habe die angeforderten Dateien und Canary-JSONs gelesen, die bestehenden Parameterzahlen aus Gewichtsformen und Masken nachgerechnet und zentrale Anatomiebelege geprüft. Dynamische TensorFlow-Reproduktionen waren wegen des schreibgeschützten Environments nicht ausführbar: Bereits der Import benötigt ein beschreibbares temporäres Verzeichnis. Die Scratch-Messungen des Entwurfs sind daher nicht unabhängig reproduziert. Keine Datei wurde geändert.
> 

### Findings and responses

#### Finding 1 (verbatim)

> **[blocking] Die Canary-Diagnose ist plausibel, aber mit falschem `dt` und einer falschen Erklärung des Closed-loop-Ergebnisses begründet.**
> 
>    [Entwurf:223](docs/superpowers/specs/2026-10-21-cncp-v2-abstraction.md:223) und Z. 225 nennen synthetische Initialisierungstests mit `dt=1/128`. Der [Canary:1](<thesis repo>/cluster_archive/2026-10-21/thesis-person/results/_canary/cfc_lrc_cncp_seed0.json:1) verwendet **`dt=0.078125=10/128`**. Die initiale Relay-Updatefraktion beträgt damit etwa **0,0527 statt 0,0054**. Das ist für die untersuchte Rückkopplung relevant.
> 
>    Weitere Korrekturen:
> 
>    - Die Trajektorienwerte bis \(1{,}06·10^{11}\) sind **denormalisiert**; nur die MSE ist normalisiert. Das steht ausdrücklich im [Runner:182](experiments/run_lotka_volterra_benchmark.py:182).
>    - Closed-loop verwendet weder Clipping noch eine Gewichtung früher Schritte. Der [Runner:174](experiments/run_lotka_volterra_benchmark.py:174) mittelt über den gesamten Horizont. Beim cNCP beträgt der mittlere Fehler der ersten Hälfte etwa **0,902**, der zweiten **2,933**, der letzte Schritt **11,457**.
>    - Hoher Verlust nach Epoche 1 beweist keine Explosion **vor** dem ersten Optimizer-Schritt.
> 
>    **Änderung:** Den tatsächlichen Canary vor Training, nach einem Update und nach einer Epoche instrumentieren: echte Inputs, echtes `dt`, identischer Builder, Zustandsnormen pro Knoten. Backbone, Gain und Rückkopplungen anschließend getrennt variieren. Gemeinsame Gewichte explizit übernehmen: Derselbe Seed reicht nicht, wenn entfernte Kanten die Reihenfolge der Zufallsinitialisierung verändern ([cncp.py:416](src/wirings/cncp.py:416)).

**Response:** **FIXED, with one PUSHBACK.** FIXED: B.5 now states dt = 0.078125 (the scripts `init_stability.py` and `v2_prototype.py` used the canary value; "1/128" was a transcription error in the text, verified against the scripts). The 1e11 trajectory values are now marked as denormalised (`run_lotka_volterra_benchmark.py` line 182). The closed-loop explanation was replaced: the runner averages the whole horizon without clipping (lines 165 to 185), and the per-step closed-loop error of the canary cNCP grows 0.90 (first half), 2.93 (second half), 11.46 (last step), recomputed from the canary JSON. The epoch-1 loss is no longer used as evidence of divergence before the first optimiser step. The instrumented reproduction (real inputs, real dt, identical builder, per-node norms before training, after one update, after one epoch; one factor at a time; weights copied explicitly because `cncp.py` lines 416 to 424 build `SparseLinear` layers only for active edges, so removing an edge changes the initialisation order) is written into B.5 and is step 1 of the sequencing. PUSHBACK on one clause: the growth *at initialisation* in two of three seeds is a measured fact at the canary dt (B.5 table), independent of the epoch-1 argument; its stated limitation is synthetic N(0,1) input and a cell built outside the task builder, which R6(a) and the reproduction close.

#### Finding 2 (verbatim)

> **[blocking] R1–R5 garantieren keine Trainingsstabilität; „lineare Subzellen“ ist mathematisch unzutreffend.**
> 
>    Die Kandidaten `ff1` und `ff2` sind affine, unbeschränkte Maps. Die gesamte Zelle ist wegen zustandsabhängiger Sigmoid- und Elastanz-Gates **nichtlinear** ([cfc_lrc_cell.py:164](src/neurons/cfc_lrc_cell.py:164)). „Keine squashing nonlinearity anywhere“ in [Entwurf:241](docs/superpowers/specs/2026-10-21-cncp-v2-abstraction.md:241) sollte heißen: *keine beschränkende Aktivierung auf den Zustandskandidaten*.
> 
>    R1 begrenzt externe Rückkopplungsnachrichten. Die interne Rekurrenz bleibt unbeschränkt. Einfaches Gegenbeispiel: `ff1=ff2=1.1*h` ergibt unabhängig vom Gate `h_next=1.1*h`. Raw-state carry ändert daran nichts. Auch die Analogie zum Dense-Arm ist falsch: Dessen Backbone begrenzt die Eingänge der Kandidatenköpfe ([cfc_lrc_cell.py:109](src/neurons/cfc_lrc_cell.py:109)).
> 
>    Ein Spektralradius unter 1 an einem Nullzustand ist keine globale Stabilitätsaussage für dieses inputabhängige System. Ein Verhältnis zur zufällig kleinen Step-1-Norm ist ebenfalls kein zuverlässiges Akzeptanzkriterium.
> 
>    **Änderung:** R1–R5 als Stabilisierungshypothesen bezeichnen. R6 um reale Sequenzen, längere Horizonte, mehrere Inputskalen sowie endliche Zustände, Verluste und Gradienten **während kurzer Trainingsläufe** ergänzen. Falls eine formale Beschränktheitsgarantie gewünscht ist, muss auch die interne Zustandsdynamik entsprechend eingeschränkt werden.

**Response:** **FIXED, with one narrow PUSHBACK.** FIXED: "the sub-cells are linear" is replaced by "the state candidates are affine and unbounded inside a gated nonlinear cell" (`cfc_lrc_cell.py` lines 118 to 121, 164, 185, 191), with the `ff1 = ff2 = 1.1 h` counterexample. The dense-arm analogy is withdrawn: the backbone bounds the inputs of the candidate heads (`cfc_lrc_cell.py` line 109), v2 has no such bound. R1 to R5 are now labelled stabilisation hypotheses with the explicit statement that none bounds the within-sub-cell h → h map; the zero-state spectral radius is called a local indicator. R6 was rewritten: real task windows and N(0,1), 128 and 512 steps, input scales 0.1 / 1 / 10, finite states, loss and gradient norms during a one-epoch training run, and the step-1-ratio criterion is dropped. PUSHBACK (wording only): R1's claim in the draft was already restricted to cross-node loops; the text now says so explicitly and no formal boundedness guarantee is sought, because bounding the internal dynamics would change the sub-cell and with it the NCP comparison.

#### Finding 3 (verbatim)

> **[blocking] V2 gegen V3 isoliert weder die Abstraktion noch die drei Motive.**
> 
>    V3 verändert gleichzeitig Knotenzahl, Breitenverteilung, Feedforward-Tiefe, Rückkopplungsaktivierung, Initialisierung, Gain-Zentrierung, State-Carry und Relay-Dynamik. Ein Sieg über einen explodierenden V2 wäre kaum Evidenz für die Vier-Knoten-Abstraktion.
> 
>    Hinzu kommt eine bestehende Konfundierung: cNCP/NCP verwenden standardmäßig `backbone_layers=0`, Dense/Dense3 standardmäßig `1` ([ncp.py:15](src/wirings/ncp.py:15), [person_activity/model.py:98](src/tasks/person_activity/model.py:98), [lotka_volterra/model.py:106](src/tasks/lotka_volterra/model.py:106)). Gleiches `cell`-Label bedeutet hier unterschiedliche Zustandsabbildungen.
> 
>    **Änderung:** Zuerst einen stabilisierten Acht-Knoten-cNCP als Kontrollarm festlegen. Für einen reinen Wiring-Vergleich die Backbone-Konfiguration angleichen und Budgets neu bestimmen. Alternativ ausdrücklich vollständige Architekturpakete vergleichen und entsprechend eng interpretieren.

**Response:** **FIXED.** Part C gains the control arm V2s: the eight-node cNCP with `cell_kwargs={'backbone_layers': 1}`, which `ncp_cell_kwargs` honours as an explicit override (`ncp.py` lines 17 to 19, `setdefault`), or alternatively with R1, R3 and R5 applied to the eight-node graph; its purpose (sub-cell vs graph property of the divergence; control for V3) and its failure modes are stated. The backbone confound of the existing arms (cNCP/NCP 0 via `ncp_cell_kwargs`, dense/dense3 1 via `lotka_volterra/model.py` line 106 and `person_activity/model.py` line 98) is now documented in B.5, B.8 and the Part C preamble, and V2 vs V3 is declared a comparison of architecture packages unless the backbone is aligned and budgets re-matched.

#### Finding 4 (verbatim)

> **[blocking] Die V3/V4-Parameterrechnung vermischt Erwartungswerte, realisierte Masken und gebrochene Relay-Breiten.**
> 
>    Die Formel \(p(u,u)=10u^2+5u\) stimmt für die asymmetrische CfC-LRC-Zelle ohne Backbone. Die **V2-Zahlen 4.082 und 30.926** stimmen ebenfalls; ich konnte sie aus dem aktuellen Code reproduzieren.
> 
>    Für V3 gilt dagegen mit \(r=\max(1,\lfloor u/2\rfloor)\), drei Relay-Vektoren `ρ`, `b_r`, `b_gate` und realisierten Maskenzahlen \(m_{DS},m_A\):
> 
>    \[
>    P_1=3p(u,u)+2u^2+m_{DS}+m_A+3ur+F_{\rm in}u+u+3r.
>    \]
> 
>    In [Entwurf:187](docs/superpowers/specs/2026-10-21-cncp-v2-abstraction.md:187) steht fälschlich `1.5 u_R` für diese drei Vektoren. Die vereinfachte Endformel ist bei geradem \(u\) trotzdem korrekt; bei ungeradem \(u\) ignoriert sie die Abrundung.
> 
>    | Konfiguration, jeweils inklusive Head | Korrekter Erwartungswert | Aus den im Entwurf genannten Prototype-Zahlen |
>    |---|---:|---:|
>    | V3, \(u=10,F=2,O=2\) | 3.667 | **3.655** |
>    | V3, \(u=11,F=2,O=2\) | **4.395** | nicht angegeben |
>    | V3, \(u=29,F=7,O=7\) | **29.890** | **29.873** |
>    | V4 lateral, \(K=2,u=7,d=.5,F=2,O=2\), mean | **3.695** | abhängig von allen realisierten Masken |
>    | V4 lateral, \(K=4,u=14,d=.25,F=3,O=7\), concat | **29.183** | abhängig von allen realisierten Masken |
> 
>    Damit sind die **3.733 beziehungsweise ≈29.500** aus [Entwurf:310](docs/superpowers/specs/2026-10-21-cncp-v2-abstraction.md:310) nicht aus den definierten Formeln ableitbar.
> 
>    **Änderung:** Erwartungswerte nur zur Planung verwenden; Tabellen aus gebautem Modell plus `effective_param_count` erzeugen ([ncp.py:323](src/wirings/ncp.py:323)). Bei unabhängigen Masken über die tatsächlichen Spaltenzahlen summieren. Für gestapelte Gewichtstensoren muss die Maskenzählung ausdrücklich unterstützt werden.

**Response:** **FIXED.** B.4 labels the three relay vectors as 3 u_R and gives the exact formula with u_R = ⌊u/2⌋ next to the even-u shorthand. Recomputed expected values: u = 11 → 4 395, u = 29 → 29 890, u_R = u variant 3 832; V4 K = 2, u = 7 → 3 695; V4 K = 4, u = 14 → 29 239 with F_in = 4 (the observation flag added under finding 10; with F_in = 3 the value is 29 183 as stated in the finding), hub 29 848, K = 30 → 30 228, K = 300 at u = 1 → 19 517. All counts are marked as expected values over the Bernoulli masks, and B.4 requires the campaign tables to come from the built model via `effective_param_count` (`ncp.py` line 323), including support for stacked `(K, u_in, u_out)` tensors. The confirmation of p(u,u) = 10u² + 5u and of the V2 counts 4 082 / 30 926 is noted.

#### Finding 5 (verbatim)

> **[blocking] Der Hub ist nicht vollständig definiert; seine Kosten und der behauptete Schnittpunkt mit lateral stimmen nicht.**
> 
>    [Entwurf:291](docs/superpowers/specs/2026-10-21-cncp-v2-abstraction.md:291) fordert die gleiche gated-relay-Regel wie R, zählt aber nur zwei Projektionsfamilien und `1.5 u_H`.
> 
>    Bei unabhängigen Projektionen für **Drive**, **Gate** und **Rückgabe** gilt:
> 
>    \[
>    C_{\rm hub}=3Kuu_H+3u_H.
>    \]
> 
>    Wird das Gate stattdessen aus dem bereits aggregierten Hub-Drive berechnet, muss seine zusätzliche Map definiert und gezählt werden. Bei gemeinsam genutzter Drive-/Gate-Projektion muss diese Gewichtsteilung ausdrücklich im Entwurf stehen.
> 
>    Schon mit der dort angegebenen Formel kosten lateral und hub bei \(K=4,d=.25\) quadratisch **\(3u^2\)** beziehungsweise **\(4u^2\)**, also nicht gleich viel.
> 
>    Außerdem enthält der Hub einen zusätzlichen Selbst-Rückweg \(D_c\to H\to S_c\); lateral schließt \(c'=c\) aus. Der Vergleich verändert somit auch Selbstfeedback und Informationskompression.
> 
>    **Änderung:** Vollständige Hub-Gleichungen, Gate-Parameter, Update-Reihenfolge und Selbstanteile festlegen. Für fünf Wochen würde ich den Hub streichen.

**Response:** **FIXED.** B.7 now defines the hub completely: drive, gate and return projections per column (three families), gate bias b_H, relay vectors ρ_H, b_rH, b_H, update order (H updated after all D at step t, read as prev at t+1), and the self-return D_c → H → S_c with its consequence for a hub-vs-lateral comparison. Cost: 3 K u u_H + 3 u_H = 1.5 K u² + 1.5 u at u_H = u/2; at K = 4, u = 14 the lateral edge (d_lat 0.25) costs 588 and the hub 1 197, the equality claim is removed. The hub is deferred to the outlook in B.2, B.7, the P_K table and Part C, as proposed.

#### Finding 6 (verbatim)

> **[should-fix] Die Reduktionstabelle behauptet Äquivalenzen, die der Code nicht erfüllt.**
> 
>    In [Entwurf:275](docs/superpowers/specs/2026-10-21-cncp-v2-abstraction.md:275):
> 
>    - **NCP:** Ein maskierter `SparseLinear` vor einer dicht mischenden Subzelle entspricht keiner NCP-Maske auf jedem einzelnen Head. Das Produkt \(MW\) kann Eingangsabhängigkeiten erzeugen, die die ursprüngliche Maske ausschließt. `NCPLayeredCell` übergibt dagegen sensory/recurrent masks direkt an die Zellen ([ncp.py:214](src/wirings/ncp.py:214), [base_cell.py:123](src/neurons/base_cell.py:123)).
>    - **Dense3:** Neben den Projektionen unterscheiden sich die Backbone-Einstellungen. Die aktuelle Dense3-Implementierung ist deshalb keine bloße Umparametrisierung von V3.
>    - **Committee:** Auf vier Knoten lässt sich derselbe Consensus-Operator anwenden. Dadurch entsteht keine Äquivalenz zum bestehenden Acht-Knoten-`MultiColumnVotingCell`.
> 
>    Zusätzlich wechseln die NCP-Analogien: B.2 nennt S „inter“ und D „command“; B.6 nennt G „inter“, S „command“, D „motor“.
> 
>    **Änderung:** „Architekturverwandtschaft“ und „numerisch exakte Reduktion“ getrennt ausweisen. Exakte Reduktionen nur mit expliziter Gewichtszuordnung und überprüfbarer Output-/State-Äquivalenz behaupten.

**Response:** **FIXED.** B.6 is retitled "Relationship to the existing arms (kinship, not numerically exact reductions)"; each row now says No and gives the reason: NCP masks enter the heads (`ncp.py` line 214, `base_cell.py` line 123) whereas v2 masks a `SparseLinear` before a dense sub-cell; dense3 has no inter-layer map and runs backbone 1; the committee wraps the eight-node cell. The claim that the inter-layer map is "redundant in function" is withdrawn. The NCP analogy is unified across B.2 and B.6 to G = inter, S = command, D = motor.

#### Finding 7 (verbatim)

> **[should-fix] Die KEEP/MERGE/DROP-Entscheidungen sind überwiegend brauchbare ML-Vereinfachungen; mehrere Begründungen behaupten jedoch unbelegte Redundanz.**
> 
>    Bewertung der vollständigen Tabelle in [Entwurf:103](docs/superpowers/specs/2026-10-21-cncp-v2-abstraction.md:103):
> 
>    | Entscheidung | Bewertung |
>    |---|---|
>    | **KEEP L4→G** | Sinnvoll als separater Eingangs- und Reentry-Knoten. Anatomische Exklusivität des externen Inputs folgt daraus nicht. |
>    | **KEEP L23→S** | Sinnvoller Integrations- und lateraler Kopplungspunkt. |
>    | **MERGE L5IT/L5ET→D** | Als Kompression vertretbar. Dichte interne Rekurrenz erhält die IT→ET-Asymmetrie jedoch nicht. „The loss cannot see“ ist unbelegt. |
>    | **MERGE L6CC/L6CT→D** | Spart Zustand und Parameter, koppelt aber Task-Output und Kontrollsignale. Eigenständige Zustände und Zeitskalen gehen verloren. |
>    | **MERGE Thal/TRN→R** | Als stateless-gate-Näherung vertretbar. Zwei dynamische Zustände sind nicht grundsätzlich redundant: unterschiedliche Zeitkonstanten und Verzögerungen verschwinden. |
>    | **DROP direkte L6→L4-Exzitation** | Vertretbare Modellentscheidung; „schwach“ belegt keine anatomische Abwesenheit. |
>    | **DROP intracortical inhibition** | Für Scope vertretbar, zumal aktuell default-off. Die Einschränkung der anatomischen Aussage muss bleiben. |
>    | **DROP Gain auf L5ET** | Ein Gain-Ziel genügt als Forschungsfrage. Die Absorbierbarkeit durch den linearen Head begründet das nicht. |
>    | **MERGE die zwei Feedback-Kanten** | Beide sind verzögerte Rückwege, aber mit unterschiedlichen Quellen und Zielen. Funktionale Gleichwertigkeit ist nicht gezeigt. |
>    | **DROP sign-locked TRN→Thal** | Für einen amplitudendämpfenden Gate-Ersatz konsistent. Der Ersatz erhält keine allgemein negative additive Wirkung. |
>    | **DROP thalamic sensory route aus Default** | Sinnvolle Scope-Entscheidung. R hat dann keine eigene sensorische Driver-Eingabe; „first-order thalamus“ bleibt eine lockere Analogie. |
>    | **DROP timescale prior** | Durch den bisherigen task-spezifischen Nullbefund ausreichend motiviert. |
>    | **DROP location gate** | Für diese Tasks und die Trennung von Eingaberepräsentation und Wiring sinnvoll. |
>    | **ADD partition/lateral/hub** | Plausible neue Hypothesen, aber eine erhebliche Erweiterung des Experiments. |
> 
>    Besonders die Head-Begründung ist falsch: Für zustandsabhängigen Gain \(G_t\) lässt sich \(W_{\rm out}G_t h_t\) nicht allgemein durch einen festen Head ersetzen. Im aktuellen Code beeinflusst der modulierte L5ET-Zustand außerdem L6 und spätere Schritte ([cncp.py:604](src/wirings/cncp.py:604), [cncp.py:615](src/wirings/cncp.py:615)).
> 
>    Crandall zeigt gerade eine dynamische Verschiebung zwischen Exzitation und Inhibition durch unterschiedliche Kurzzeitplastizität. Der vorgeschlagene Gate-Ersatz bewahrt davon nur die Dämpfungsfunktion. Das ist für reines ML zulässig, sollte aber als verlorene Dynamik dokumentiert werden. [Crandall et al. 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4425600/)

**Response:** **FIXED.** B.2 reasons rewritten per the table: the head-absorption argument is replaced by a scope argument with the note that W_out G_t h_t has no fixed-head equivalent and that the modulated L5ET state feeds L6 and the next step (`cncp.py` lines 604, 615); "the loss cannot see" is removed and the IT → ET asymmetry is marked as given up and untested; the L6 merge names its cost (shared state and timescale for output and control); the Thal/TRN merge names the lost dynamics (two time constants, one-step delay, Crandall's rate-dependent switch reduced to attenuation); L6 → L4 is a weight judgement, not an anatomical absence; the feedback-edge merge states that equivalence is assumed; the sign-lock replacement keeps the attenuating effect only; R has no sensory driver when the thalamic route is off; ADD is marked as an extension of the experiment.

#### Finding 8 (verbatim)

> **[should-fix] Einige anatomische Schlussfolgerungen sind zu stark oder direkt falsch.**
> 
>    **„Keine inter-column excitatory path into L4“** in [Entwurf:324](docs/superpowers/specs/2026-10-21-cncp-v2-abstraction.md:324) widerspricht der selbst zitierten Narayanan-Arbeit: Sie beschreibt transkolumnare L2-/L3-Axonprojektionen in die granuläre Schicht. S als Default-Kopplungspunkt ist trotzdem vertretbar; G auszuschließen ist eine Designentscheidung. [Narayanan et al. 2015](https://academic.oup.com/cercor/article/25/11/4450/2367294)
> 
>    Die zufälligen lateralen Masken bewahren außerdem **keine feature-selektive Topographie**. Bosking untersucht Verbindungen zwischen koorientierten, räumlich ausgerichteten Rezeptivfeldern; Bernoulli-Masken zwischen anonymen Hidden Units kodieren diese Beziehung nicht. [Bosking et al. 1997](https://pmc.ncbi.nlm.nih.gov/articles/PMC6793759/)
> 
>    Die Aussage zur dominierenden intracortical route ist dagegen gut gestützt: Olsen berichtet einen intracortical-Anteil von etwa 73 % unter den untersuchten Bedingungen. Ihr Weglassen ist eine bewusste Einschränkung. [Olsen et al. 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3636977/)
> 
>    Schließlich überträgt [Entwurf:26](docs/superpowers/specs/2026-10-21-cncp-v2-abstraction.md:26) die lokale Behandlung von Polarität auf Lechner 2020: Die Original-LTC-Gleichung enthält ein reversal potential, das die Synapsenpolarität bestimmt ([extrahiertes Paper:361](<thesis repo>/papers/.extracted/lechner2020-ncp.md:361)). Das lokale Wiring nimmt dagegen Absolutwerte der Adjazenz ([ncp.py:134](src/wirings/ncp.py:134)). Diese Ebenen müssen auseinandergehalten werden.

**Response:** **FIXED.** B.7 "Not coupling at G" now cites Narayanan's transcolumnar L2/3 axons into the granular layer and calls the exclusion a design choice. The lateral-edge paragraph states that a Bernoulli mask between anonymous hidden units encodes no feature-selective topography (Bosking) and that selectivity, if any, must come from the learned weights. A.1 attributes the polarity treatment correctly: Lechner's LTC carries polarity in the reversal potential E_ij of eq. (1) (`lechner2020-ncp.md` line 361), this repository discards it via the absolute value of the adjacency (`ncp.py` line 134). The Olsen statement stays as it was.

#### Finding 9 (verbatim)

> **[should-fix] Unabhängige Gewichte plus gelernte Kopplung sind sinnvoll, aber weder automatisch fair noch automatisch robuste „Votes“.**
> 
>    Eigene Parameter passen zu spezialisierten Sensorströmen. Gelernte \(L_{c'c}\)-Maps können zwischen unterschiedlichen Repräsentationen übersetzen. Das ist eine plausible Wahl.
> 
>    Drei Einschränkungen fehlen in [Entwurf:290](docs/superpowers/specs/2026-10-21-cncp-v2-abstraction.md:290):
> 
>    - `none` gegen `lateral` bei **gleichem \(u\)** verändert das Parameterbudget. Neumatching verändert dagegen \(u\) und damit die Kapazität jeder Spalte. Für Mechanismus und Budgetnutzen sind das unterschiedliche Vergleiche.
>    - Mean vor einem gemeinsamen linearen Head ist algebraisch ein Mittel der entsprechend projizierten Outputs. Das funktioniert. **Median im Hidden Space** hat diese Eigenschaft nicht; unabhängig gelernte Hidden-Koordinaten haben keine garantierte gemeinsame Bedeutung.
>    - Eine symmetrische Aggregation macht das Gesamtnetz mit slotspezifischen \(f_c\) und \(L_{c'c}\) nicht automatisch invariant gegenüber dem Vertauschen der Eingaben zwischen Slots.
> 
>    **Änderung:** Für Robustheitsvoting zunächst task-aligned Outputs aggregieren, etwa mittels eigener kleiner Heads pro Spalte, und deren Parameter mitzählen. Korruption muss zudem durch das gekoppelte System laufen: Kopplung kann Fehler verbreiten. Unabhängige Gewichte garantieren keine unabhängigen Fehler. Hawkins motiviert gelernte laterale Unterstützung, aber keine Robustheitsgarantie für diese konkrete Architektur. [Hawkins et al. 2017](https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2017.00081/full)

**Response:** **FIXED.** B.7 now separates `none` vs `lateral` at the same u (mechanism) from the re-matched comparison (budget) and requires both. The median readout is redefined over task-aligned quantities: per-column heads `Dense(n_out)` (K (u + 1) n_out parameters, counted in P_K) and a lower median over the K logit vectors; the mean before a shared head is kept with the algebraic justification. The permutation statement is corrected (symmetric readout, slot-specific network). The ρ study notes that corruption passes through the coupled system and that independent weights do not imply independent errors; Hawkins is cited as motivation, not as a guarantee.

#### Finding 10 (verbatim)

> **[should-fix] Die Person-Activity-Partition braucht eine explizite Missing-observation- und Zeitsemantik.**
> 
>     Der Datensatz enthält tatsächlich einen Tag plus xyz pro Ereignis; `dt` bezeichnet den Abstand zur **vorherigen Zeile**, nicht zur vorherigen Beobachtung desselben Sensors ([datasets.py:73](src/tasks/person_activity/datasets.py:73)).
> 
>     In [Entwurf:285](docs/superpowers/specs/2026-10-21-cncp-v2-abstraction.md:285) erhalten inaktive Spalten Nullen. Für CfC-LRC ist das ein reguläres Update mit Nullinput; es ist kein „keine Beobachtung“. Auch `dt=0` hält die Zelle nicht automatisch an ([cfc_lrc_cell.py:185](src/neurons/cfc_lrc_cell.py:185)).
> 
>     **Änderung:** Festlegen, ob alle Spalten bei jedem Ereignis fortschreiten oder nur die aktive Spalte aktualisiert wird. Eine explizite Observation-Maske verhindert die Gleichsetzung fehlender Messungen mit xyz=0. Bei übersprungenen Updates muss die seit dem letzten eigenen Update verstrichene Zeit berücksichtigt werden. Danach Inputbreiten und Budgets neu zählen.
> 
>     Als Kontrolle braucht es einen einfachen partitionierten Dense/GRU-Ansatz mit derselben Eingabedarstellung. Sonst bleibt offen, ob ein Gewinn vom Sensor-Routing stammt.

**Response:** **FIXED.** The region partition now uses x_k = [m_k, m_k · xyz] with an explicit observation flag, F_in = 4, and fixes the semantics: all columns advance at every event with the shared dt (gap to the previous row of any sensor, `datasets.py` lines 73 to 74), an inactive column performs a regular update with zero input, dt = 0 would not freeze the cell (`cfc_lrc_cell.py` line 185). The alternative (active column only, per-column elapsed time, masked state write) is recorded as an option. A partitioned dense/GRU control with the same input layout is required. Budgets are recomputed with F_in = 4 (29 239 at K = 4, u = 14).

#### Finding 11 (verbatim)

> **[should-fix] Die Compute-Aussagen und die harte Grenze \(K\approx4\) sind nicht belegt.**
> 
>     Die Aussage „wall time stays at the single-column level“ in [Entwurf:281](docs/superpowers/specs/2026-10-21-cncp-v2-abstraction.md:281) folgt nicht aus einer konstanten Anzahl von `einsum`-Aufrufen. Tensorgrößen, Speicherverkehr, Backpropagation und Kopplungsaggregation wachsen weiterhin. Lateral verursacht zusätzlich quadratischen Aufwand in \(K\).
> 
>     Weitere Punkte:
> 
>     - `SparseLinear` führt **dichte** Matrixmultiplikation mit maskierten Gewichten aus ([ncp.py:77](src/wirings/ncp.py:77)). Effektive Parameterzahl entspricht weder FLOPs noch Speicherbedarf.
>     - Auch V1 lässt sich über eine zusammengelegte Batch-/Spaltenachse vektorisieren. Ein Vergleich „V1 Python-Loop versus V4 optimiertes Einsum“ misst zusätzlich Implementierungsaufwand.
>     - „Independent weights cap K at about 4“ widerspricht schon der eigenen K=30-Zeile. Es ist eine vermutete Mindestbreite für brauchbare Modelle, keine Parametergrenze. Selbst \(K=300,u=1\) passt rechnerisch mit einfacher Kopplung unter 30k; seine Nützlichkeit ist eine andere Frage.
>     - Die erfolgreiche Breite eines Dense3-Arms ist keine experimentell bestimmte Mindestbreite für eine V3-Spalte.
> 
>     **Änderung:** Laufzeit, Peak Memory sowie rohe und effektive Parameter messen. K=4 als pragmatische Auswahl begründen.

**Response:** **FIXED.** The "wall time stays at the single-column level" claim is withdrawn; B.7 states that FLOPs, memory and the backward pass scale with K and lateral with K², that `SparseLinear` performs a dense matmul (`ncp.py` line 77), that V1 can be vectorised the same way, and that any compute comparison must be measured with the same implementation style. The K ≈ 4 statement is reworded as a judgement about minimum useful width (u = 14 at K = 4, 5 at K = 30, 1 at K = 300, which fits arithmetically at 19 517), with a K ∈ {1, 2, 4, 8} sweep as the check; the dense3 width is called a reference point, not a measured minimum. B.4 and the Part C summary mark the op counts as launch estimates.

#### Finding 12 (verbatim)

> **[should-fix] Die Falsifikationskriterien erlauben mehrere unzulässige Schlussfolgerungen.**
> 
>     [Entwurf:344](docs/superpowers/specs/2026-10-21-cncp-v2-abstraction.md:344) koppelt fünf Seeds, Wilcoxon und Aussagen wie „adds nothing“. Bei fünf Paaren ist der kleinste exakte zweiseitige p-Wert **0,0625**. Das ist in [Kapitel 5:86](<thesis repo>/Thesis/chapters/05_experimental_setup.tex:86) bereits ausdrücklich dokumentiert; Z. 88 verbietet zudem die Interpretation von Nichtsignifikanz als Äquivalenz.
> 
>     Auch die Mechanismuskriterien müssen enger werden:
> 
>     - \(|\gamma|<0,05\) beweist keine Nutzlosigkeit; kleine Modulation kann downstream verstärkt werden.
>     - Ein weitgehend offenes Gate kann seltene, relevante Ereignisse beeinflussen.
>     - `none≈lateral≈hub` zeigt höchstens keinen nachgewiesenen Kopplungsnutzen in diesem Setting.
>     - Ein einzelner negativer Median-/Korruptionstest widerlegt keine ganze Familie unabhängiger Spaltenmodelle.
> 
>     Die NCP-Stabilitätsbehauptung ist ebenfalls zu allgemein: Die Zufallsgenerierung erlaubt Command-Zyklen und Selbstkanten ([wirings.py:543](.venv/lib/python3.11/site-packages/ncps/wirings/wirings.py:543)). Bei Größe 43 waren die Command-Graphen für Seeds 0–2 azyklisch; **Seed 4 enthält bereits einen Zyklus**. Das widerlegt die pauschale Nilpotenzannahme, ohne für diesen Seed Instabilität zu beweisen.
> 
>     **Änderung:** Effektgrößen und Unsicherheit berichten; für Äquivalenz eine praktische Toleranz definieren. Motive durch gezielte Interventionen und retrainierte Ablationen beurteilen. Seeds nicht nach Stabilität aussortieren.

**Response:** **FIXED.** B.8 restates the statistics of chapter 5 (lines 84 to 88: five pairs, minimum exact two-sided p = 0.0625, non-significance never read as equivalence) and defines "no advantage" by a pre-registered tolerance (5 % relative NRMSE, 1 macro-F1 point) plus a paired-difference range straddling zero; "adds nothing" and "closed for both weight regimes" are removed, each criterion is scoped to the setting. Motif criteria now rest on retrained ablations; |γ| < 0.05 and an open gate are hints, the gate's distribution is reported. The NCP statement is corrected to a per-seed fact: at size 43 the command graph has ρ = 0.00 for seeds 0 to 3 and 5 to 7, while seed 4 draws two self-edges and has ρ = 1.0 (verified with `ncp_graph`; ncps `wirings.py` lines 543 to 549 allow self-edges), and seeds are not to be selected on stability.

#### Finding 13 (verbatim)

> **[should-fix] Für fünf Wochen ist der vollständige V1–V4-Plan zu breit.**
> 
>     V1 und V2 sind als bestehende Implementierungen hinreichend identifizierbar. V3 hat einen brauchbaren Rechenkern, benötigt aber korrigierte Budgets und Akzeptanztests. V4 enthält mehrere zusätzliche Architekturentscheidungen und testet eine andere Frage als das bestehende H-d mit \(K=300\).
> 
>     Meine konkrete Reihenfolge wäre:
> 
>     1. **Zwei bis drei Tage:** Canary-Ursache mit realem `dt` reproduzieren; bestehenden Acht-Knoten-cNCP stabilisieren. `backbone_layers=1` ist bereits als expliziter Override möglich und ein sinnvoller diagnostischer Kontrollversuch. Anschließend Budget neu matchen.
>     2. **Hauptkampagne:** Bestehenden cNCP, NCP und Dense3 unter klar dokumentierter Subzellkonfiguration evaluieren. Einen stabilen Nullbefund als Ergebnis akzeptieren.
>     3. **Optionaler, eng begrenzter V3-Versuch:** Nur Single-column, eine primäre Aufgabe, Vergleich gegen stabilisierten V2 und Dense3. Auswahl anhand von Validierungsdaten; als explorative Zusatzablation ausweisen.
>     4. **V4, Hub und große Kopplungsablationen:** In den Ausblick verschieben.
> 
>     Der stärkste verbleibende Thesis-Beitrag ist eine belastbare Aussage darüber, **wann die bestehende Wiring-Struktur hilft, wirkungslos bleibt oder mit der Subzelldynamik instabil wird**. Der aktuelle Entwurf liefert noch keine ausreichende Evidenz, dass eine neue Mehrspaltenarchitektur dieses Ergebnis rechtzeitig verbessert.

**Response:** **FIXED (sequencing accepted), with one PUSHBACK.** The header now carries the review status and the accepted order: (1) reproduce the canary with real dt and instrumented states, stabilise the eight-node cNCP (V2s), re-match budgets; (2) main campaign with the existing cNCP, NCP and dense3 under a documented sub-cell configuration, a stable null accepted as a result; (3) V3 only as a bounded single-column extra ablation on one primary task against V2s and dense3, selected on validation data; (4) V4, hub and coupling ablations in the outlook. PUSHBACK on one element: `backbone_layers = 1` is accepted as a diagnostic control, but as a *campaign* setting for the cNCP it reverses revision 13.1 (sub-cells without backbone, matching the ncps `WiredCfCCell` convention of the NCP arm) and would have to be applied to the NCP arm as well, otherwise the sub-cell-depth confound returns with the opposite sign. Whether to restore the backbone in both arms or to stabilise the graph without it is a decision for Alexander, not for this document.
