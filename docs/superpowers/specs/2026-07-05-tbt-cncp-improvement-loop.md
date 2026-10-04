# tbt_cNCP improvement loop (autonomous)

Goal (user, 2026-07-05): iterate on tbt_cNCP -- concept -> implement -> cluster
benchmark -> evaluate -> brainstorm -> repeat -- until tbt_cNCP delivers results
that are genuinely good and defensible in the master's thesis.

This log tracks each iteration so the reasoning survives context compaction.
Concept + first cluster suite live in `2026-07-04-tbt-cncp-concept.md`.

## Starting point (diagnosis, 2026-07-05)

Cluster suite (3 seeds, cfc_lrc, parameter-matched ~45-50k, dense lighter 25-31k):

| task | dense | ncp | tbt_concat | tbt_gate | tbt_noloc |
|---|---|---|---|---|---|
| active_sensing (acc, higher better) | 0.848 | 0.910 | **0.941** | 0.741 | 0.335 |
| duffing (closed-loop MSE, lower better) | 1.57 | 1.76 | **1.15** | -- | 2.28 |
| predator-prey (CL-MSE) | **1.43** | 1.84 | 1.79 | -- | 3.28 |

Where tbt underperforms and why (mechanistic, from the code):
1. `noloc` (0.335) = ablation, no location fed at all -> unfair vs dense/ncp
   which get `concat(patch, location)`. Design control, not a defect.
2. `gate` (0.741 < ncp/dense) = the TBT-faithful L6a->L4 signal is purely
   MULTIPLICATIVE: `h_l4 * (1 + g*sigmoid(W@loc))`. It can only rescale L4, never
   add a new coordinate. Where `basal_l4 ~ 0`, no gain injects information.
   Plus weak init (g=0.1) and propagation depth. `concat` (additive) fixes it.
3. predator-prey: reference frame is redundant (state fully observed); dense
   wins on the lighter param budget (less rollout overfitting). Not a tbt defect.

So the scientifically unsatisfying gap is (2): the biologically-motivated
gating path does not carry the location information well.

## Iteration 1 -- FiLM location gating (affine L6a->L4)

**Concept.** Replace the pure multiplicative gate with FiLM conditioning
(Perez et al. 2018): L6a supplies L4 both a per-unit scale AND a shift,
`h_l4 = h_l4 * (1 + gamma(loc)) + beta(loc)`. The shift `beta` is exactly the
coordinate the multiplicative gate cannot express, yet it stays TBT-faithful
(L6a modulates L4 as a predictive prior) instead of falling back to generic
concat. Zero-init both maps -> training starts at the identity, no dead start
(d/dW = loc (x) upstream, nonzero from step one).

**Hypothesis.** FiLM closes the gate->concat gap and may beat concat, because it
is affine (as expressive as concat on L4) but injected on the cNCP topology
through the deep-layer path rather than at the raw input.

**Implementation.** `TbtCorticalColumnCell.use_location` now accepts
`'none' | 'gate' | 'film'` (bool still valid: True=gate, False=none). New
`tbt_cncp_film` mode in `TBT_MODES`; `LV_WIRINGS` inherits it. Parameter-matched:
film adds only the shift block (~+350 params vs gate, <1%). 19 active_sensing +
26 lotka_volterra tests green, incl. two new FiLM tests (zero-init identity ->
location-active after nonzero; both FiLM maps get gradient).

**Cluster.** Jobs 536195 (active_sensing: gate/concat/film/ncp/dense) + 536196
(duffing: dense/ncp/cncp/gate/concat/film), 3 seeds, cfc_lrc.

**Results.** (3 seeds, cfc_lrc, jobs 536195/536196 COMPLETED)

| mode | active_sensing acc | duffing CL-MSE |
|---|---|---|
| **tbt_cncp_film** | **0.963 +/- 0.008** | **1.219 +/- 0.069** |
| tbt_cncp_concat | 0.958 +/- 0.007 | 1.445 +/- 0.044 |
| ncp | 0.924 +/- 0.012 | 2.124 +/- 0.515 |
| dense | 0.873 +/- 0.033 | 1.313 +/- 0.140 |
| tbt_cncp (gate) | 0.718 +/- 0.048 | 2.277 +/- 0.137 |
| tbt_cncp_noloc | 0.335 +/- 0.021 | 2.283 +/- 0.669 |

Hypothesis confirmed. FiLM closes the gate->concat gap entirely (active_sensing
gate 0.718 -> film 0.963; duffing gate 2.277 -> film 1.219) and matches concat on
active_sensing while BEATING it on duffing (1.219 < 1.445, gap > combined sigma).
FiLM is the most stable mode (smallest sigma on both) and the only mode that is
best on both location-relevant tasks (on duffing even below the lean dense).

**Brainstorm.** The multiplicative gate is structurally too weak (on duffing it
adds nothing over no-loc). The additive shift `beta` is the decisive piece.
FiLM > concat on trajectories despite equal affine expressiveness -> the win is
the INJECTION SITE: FiLM modulates L4 deep as a predictive prior, concat only
appends to the raw input; in closed-loop rollout the prior anchors the state in
the correct potential well -> more stable rollout. This is the TBT reading
(reference frame as predictive state on sensation, Hawkins), and it is exactly
why it helps where local observation is ambiguous (double well) yet stays
redundant where the state is already unambiguous.
Caveats: 3 seeds, closed-loop MSE varies run-to-run; film vs dense on duffing
not cleanly significant (overlapping CIs, film more stable). Self-designed tasks.
Verdict: solid, defensible step; tbt_cNCP(FiLM) is now the best mode. Biggest
remaining lever for "genuinely good" = SAMPLE EFFICIENCY (voting gave the largest
single effect so far). -> Iteration 2 brings FiLM into the voting cell.

## Iteration 2 -- FiLM voting (best single-column mechanism x multi-column consensus)

**Concept.** `MultiColumnVotingCell` currently injects location via concat
(use_location=False + location concatenated into features). Iteration 1 showed
FiLM is the strongest single-column mechanism. Swap the voting cell's location
path to FiLM and measure sample efficiency (glimpses-to-threshold) for K=1/2/3.
Hypothesis: FiLM voting reaches the accuracy threshold in fewer glimpses than
concat voting, i.e. the better per-column prior compounds with consensus.

**Implementation.** `build_voting_model(location_mode='concat'|'film'|'gate')`:
concat appends location to each column's patch (column gate off); film/gate
inject into L4 inside the column. `run_voting_benchmark.py --location-mode`
records the mode in the JSON + filename; `plot_voting.py` groups by
(K, location_mode) and overlays them (film solid, concat dashed) + grouped
glimpses-to-threshold bars. `voting.sbatch` iterates VOTE_MODES="concat film".
20 active_sensing tests green (incl. FiLM voting build/forward/maps).
Parameter-matched: film voting +384 params (~0.8%) over concat.

**Cluster.** Job 536198 (concat + film x K=1/2/3 x 3 seeds = 18 runs). Old
no-mode voting JSONs moved aside (local + cluster) so the concat group is clean.

**Results.** (3 seeds, job 536198)

| K | concat acc / glimpses | film acc / glimpses |
|---|---|---|
| 1 | 0.916 / 3.3 | 0.881 / 3.7 |
| 2 | 0.981 / 1.7 | 0.969 / 2.0 |
| 3 | 0.994 / 1.0 | 0.987 / 2.0 |

Hypothesis REFUTED: concat voting beats FiLM voting at every K (final acc AND
glimpses-to-threshold). Sign flip vs Iteration 1 (there film > concat single
column). Likely cause: FiLM's zero-init identity start needs more training signal
to build the location gain; the voting setup runs 70 epochs and NO self-
supervised prediction head, whereas Iteration 1 had 80 epochs + the pred head
that anchors FiLM's L4 predictive prior. predator-prey FiLM control (job 536199):
film 2.169 vs concat 1.785 vs dense 1.435 -> FiLM slightly HURTS with a redundant
frame, cleanly confirming the core thesis (frame helps only when local obs is
ambiguous).

**Brainstorm.** Two robust positives survive: (1) voting itself is strong and
mechanism-agnostic (K=1->3 concat: 3.3->1.0 glimpses, 0.916->0.994 acc -- near
perfect from a single glimpse at K=3); (2) the predator-prey control nails the
unified story. concat voting at K=3 is near-saturated -> little headroom to "fix"
FiLM voting. The larger untapped lever is OCCLUSION: all models are weak there
(occ acc ~0.4-0.5) and FiLM already leads (0.498 vs concat 0.443, ncp 0.475,
dense 0.401). -> Iteration 3.

## Iteration 3 -- graded occlusion robustness

**Concept.** The reference frame should compensate for occluded visual input via
spatial context, so location-carrying tbt should degrade more gracefully as the
object is progressively blanked. `load_active_sensing(occlude_frac)` blanks the
left `frac` of the object width for TEST glimpses; the runner re-evaluates the
same trained model at frac in {0, 0.25, 0.5, 0.75} (occlusion_accs field);
plot_active_sensing.py gains a 3rd panel (accuracy vs occlusion). tbt_cncp_noloc
is the key control (no location -> should collapse fastest under occlusion).

**Hypothesis.** tbt_cncp_film (and concat) degrade more gracefully than dense/ncp
and far more gracefully than noloc; the gap widens with occlusion.

**Implementation.** occlude_frac in datasets (back-compat: occlude=True == 0.5);
runner sweep + occlusion_fracs/accs fields; plot 3rd panel; graded-occlusion
test. 21 active_sensing tests green. Job 536204 (dense/ncp/concat/film/noloc x 3
seeds, occlusion sweep).

**Results.** FiLM best at EVERY occlusion level (50%: film 0.502 > ncp 0.465 >
concat 0.442 > dense 0.419; noloc collapses 0.234). "Graceful degradation" only
partly holds: the frame lifts the whole curve but does not flatten it much.

**Brainstorm.** FiLM leads under occlusion; noloc collapse confirms location is
essential. But absolute drops are similar across location modes -> the frame
raises the ceiling, not the slope. Core wins are real but need >3 seeds to be
defensible. -> Iteration 4 consolidates to 8 seeds.

## Iteration 4 -- consolidation to 8 seeds

**Results (8 seeds, jobs 536210/536211, 95% CIs).**
- active_sensing clean: film 0.949+/-0.008 == concat 0.949+/-0.015 >> ncp
  0.888 >> dense 0.849 (film/concat CIs SEPARATE from ncp/dense -> real).
- occlusion 50%: film 0.496 best, but CIs overlap ncp/concat (not clean-sig).
- duffing CL-MSE: film 1.288+/-0.12 best + most stable, beats ncp 1.75 / cncp
  2.06 / gate 1.90 clearly; edges concat 1.41 / dense 1.48 but CIs overlap.

**Verdict / honest correction.** The Iteration-1 "film > concat" was seed noise:
at 8 seeds they are EQUAL on clean active_sensing (both 0.949). What holds
robustly: (a) cortical-column topology + location >> dense/ncp (separated CIs);
(b) FiLM is the most STABLE injection (smallest sigma everywhere) and best on
duffing; (c) frame helps only under ambiguity (predator-prey counter-example).
Defensible core story, not a dramatic single-mechanism breakthrough. User chose:
continue the loop with a new architecture iteration.

## Iteration 5 -- active glimpse control (sensorimotor, L5 motor hub)

**Concept.** So far glimpses are RANDOM (offline). TBT's core is sensorimotor:
the column should STEER its own sensor. The cNCP already has an explicit motor
hub (L5ET). Let the column predict the next glimpse location from L5ET and take
the next glimpse there (differentiable bilinear sampling from the full object
image). Hypothesis: active sensing reaches target accuracy in fewer glimpses than
random, and the cortical column -- which has a natural motor hub -- is structurally
suited to it in a way dense/ncp are not. This is the distinctive tbt contribution
the loop has not yet exploited.

**Implementation.** New `src/tasks/active_sensing/active_glimpse.py`:
differentiable bilinear glimpse sampler (gradient flows through the glimpse
CENTRE -- verified), tf reference-frame code matching encode_location bit-for-bit,
ActiveGlimpseModel (unrolled T-step loop: sample glimpse -> column -> class logits
-> motor head sets next centre). tbt reads the motor from L5ET; dense/gru from the
hidden state. policy=random ignores the motor (passive baseline). A 2-step random
exploration warmup is essential: without it the zero-motor + centre start keeps
the gaze stuck in the (empty) image centre -- a real exploration failure found in
the prototype (active stuck at chance 0.305). Runner + plot + 5 tests green,
400-test suite collects clean. Cluster job 536224 (tbt/dense x active/random x
3 seeds, 8 classes).

**Results (prototype + local smoke).** compositional, 3 seeds: active
1.000+/-0.000 vs random 0.599+/-0.109 (glimpses-to-0.6: 4 vs 6.7). 5-class local
smoke: active 1.0 (BOTH tbt and dense) vs random 0.57-0.61. STRONG active >>
random. HONEST CAVEAT: dense (hidden-state motor) steers as well as tbt (L5
motor) -- the headline win is active sensing itself, not the cortical motor hub.
Cluster run (8 classes, harder) tests whether tbt separates from dense.

**Brainstorm.** Cluster (8 classes, 3 seeds, explore=0): active sensing WORKS but
tbt-active is UNSTABLE. dense active 0.89 > dense random 0.64 (all 3 seeds, clean
win). tbt active 0.44+/-0.33: seed0 0.91 but seed1/2 collapse to chance -- the
complex column + differentiable gaze is a hard optimisation; the gaze sticks in a
bad local optimum, never sees the discriminative parts. tbt random 0.87 stable.
So the L5 motor hub is NOT superior -- dense steers more reliably. But seed0 proves
stable tbt-active reaches 0.91: an optimisation, not a capability problem.
-> Iteration 6 stabilises with exploration.

## Iteration 6 -- stabilise tbt-active with epsilon-greedy exploration

**Concept.** The tbt-active collapse is a gaze-policy collapse: without ongoing
exploration the motor settles early on a look policy that misses the object.
Standard fix for differentiable active perception: train-time epsilon-greedy
exploration -- with prob epsilon take a random glimpse instead of the steered one,
so the model keeps seeing varied locations and cannot lock into a dead policy.
Eval uses the pure learned policy. explore=0.3 default.

**Implementation.** ActiveGlimpseModel gains explore + a training-aware call
(random/steered mix only when training=True). Runner --explore, recorded. 5 tests
green. sbatch/submit gain AG_EXPLORE.

**Results (FINAL, honest correction).** explore=0.3 (job 536229, 8 classes, 5
seeds): tbt active 0.68+/-0.38, only 3/5 seeds converge. explore=0.5 (job 536234,
5 seeds) DOES NOT fully fix it either -- my earlier "explore=0.5 FIXES it" was a
lucky 4-seed local check that missed the bad seed. Final cluster numbers:
- dense/active   0.952 +/-0.015  (0.92 0.97 0.95 0.96 0.95) -- 5/5 converge, tight
- dense/random   0.634 +/-0.014  passive baseline
- tbt_cncp/active 0.785 +/-0.334 (0.97 0.11 0.95 0.98 0.92) -- STILL BIMODAL, seed 1
  collapses below chance (1/8=0.125) even at explore=0.5; 4/5 converge
- tbt_cncp/random 0.842 +/-0.038  more stable AND higher-mean than tbt-active
So for dense, active is a large stable win (+0.32 over random). For tbt, the L5
motor is fragile: active (0.785, bimodal) is on average WORSE than tbt's own
random policy (0.842), and far worse/less stable than dense-active (0.952).

**Brainstorm (closes the touch/motor direction).** The honest bottom line for
Iterations 5-6: sensorimotor active sensing is a real, large effect (active >>
random, ~+0.3) but NOT wiring-specific -- a generic dense hidden-state motor does
it better (0.952 vs 0.785) and dramatically more stably (+/-0.015 vs +/-0.334)
than the cortical L5 motor hub, which stays bimodal (one full collapse per 5
seeds) even with strong exploration. Two side-facts worth keeping: tbt's PASSIVE
(random-glimpse) recogniser beats dense's (0.842 vs 0.634), so the cortical column
is a better passive object model; but its L5 motor makes it a worse, fragile
active steerer. Net: a clean NEGATIVE result for the L5-motor hypothesis -- a
genuine capability without a tbt advantage. This validates the pivot AWAY from the
touch/sensorimotor direction: the thesis win is not here. Distinct from the robust
core wins (FiLM, voting, occlusion) of Iterations 1-4. Direction CLOSED.

## Iteration 7 -- partial-view column committee as a GENERAL architecture

**User steer (2026-07-05).** Move away from the touch/object-recognition framing
(the TBT origin story, narrow as a benchmark). The transferable principle is what
matters: K weaker, weight-shared columns, each receiving a PART of the whole,
computing locally, then a VOTE assembles a good model of the whole. Test it as a
general transfer across the real task families (person_activity, Lotka-Volterra,
later IMDB/MuJoCo), not the touch toy.

**Concept.** A task-agnostic partial-view committee. Given a sequence input
X (B,T,F): a view operator produces K partial views (feature/sensor partition,
temporal windows, or noisy copies). K WEIGHT-SHARED columns each process their own
view; a lateral vote (learnable convex mix toward the mean, reusing the L2/3 voting
of MultiColumnVotingCell, generalised to arbitrary per-column cells) aggregates
their states; a task head reads the voted representation. Weight-sharing is the
lever: K shared columns have ~the same parameters as ONE column, so the comparison
is razor-sharp -- committee (K partial views + vote) vs a single monolith (full
view) at IDENTICAL param count, the only difference being partial-view+voting.

**Why this is not just "ensembles work".** The contribution is the conjunction:
a bio-inspired cortical (cNCP) column as the weight-shared weak learner + partial
view + lateral L2/3 voting, param-matched against (a) a single monolithic model of
the same size and (b) a vanilla-dense committee. If the cNCP column shows no edge
over a dense column as a committee member, that is an honest negative result.

**Hypotheses.**
- H1 (graceful degradation): under partial observability (sensor dropout / noise
  at test time) the committee degrades more slowly than a single monolith, because
  each column already trained on partial views and voting averages out corrupted
  ones. Clean full-observability may favour the monolith (it sees everything) --
  the committee's win should appear on the degradation axis.
- H2 (frontier): "weaker" is not free -- too-weak columns raise bias and a single
  adequate model wins. At a fixed param budget there is an optimal split
  K x size-per-column; the question is whether the cortical wiring shifts it. So
  sweep K in {1,2,4,...}, not a fixed K.
- H3 (wiring): a cNCP column as the per-view weak learner beats a dense cell under
  the voting regime (or, honestly, does not).

**Plan.** Generic committee module (views + generic voting cell wrapping any
per-column cell). Prototype on person_activity first (real sequence classifier,
4 body-worn sensors -> sensor partition is the most literal instance; robustness =
test-time sensor dropout). Then transfer to Lotka-Volterra (regression/dynamics;
views = temporal windows / noisy channels) to show generality across task types.
Compare param-matched: committee-cNCP vs single-cNCP vs committee-dense, K sweep,
clean vs dropout. Cluster-test, evaluate, brainstorm, iterate.

**Implementation.** src/wirings/committee.py (CommitteeVotingCell: generic
single-state analogue of MultiColumnVotingCell, votes on hidden state);
src/tasks/committee/{views.py,model.py} (partition/noisy views + drop_features;
build_committee_model with cncp=MultiColumnVotingCell, dense=CommitteeVotingCell);
experiments/run_committee_benchmark.py (+plot). 14 tests green. Weight-sharing
verified: params K-independent (cncp 25592 for K=1,2,4). Matched pair cncp size=48
(25592) ~ dense size=64 (26312), ratio 0.97.

**Results (local signal, 12 epochs, 1 seed, dropout averaged over 5 channel
choices; person-activity F=7 C=7, majority baseline 0.371).**
             clean   drop.15        drop.30        drop.45
  cncp  K=1  0.760   0.387 +/-.25   0.420 +/-.23   0.271 +/-.05
  cncp  K=4  0.651   0.488 +/-.12   0.456 +/-.11   0.327 +/-.01
  dense K=1  0.778   0.396 +/-.26   0.366 +/-.19   0.289 +/-.05
  dense K=4  0.730   0.498 +/-.16   0.426 +/-.15   0.330 +/-.08
Three findings: (1) H1 confirmed for BOTH wirings -- the K=4 committee beats the
K=1 monolith at every dropout level, trading ~0.05-0.11 clean accuracy for the
robustness. (2) Variance collapse -- the committee is far LESS sensitive to WHICH
sensor fails (the +/- over channel choice roughly halves, e.g. cncp .15: +/-.25 ->
+/-.12); voting spreads reliance across columns. This is arguably the stronger
story than the mean gain. (3) H3 negative -- cncp and dense committees are
essentially equal (cncp K=4 vs dense K=4 within noise), so the committee EFFECT is
wiring-agnostic, like the active-sensing effect. Honest caveat: at drop 0.45 every
model is below the majority baseline (0.37) -- degenerate; the meaningful range is
clean/0.15/0.30, where the committee keeps models above the trivial baseline
longer. Full cluster run (job 536345: 5 seeds, 40 epochs, K in {1,2,4}) pending for
tight CIs.

**Results (cluster, CONFIRMED headline -- dense committee, job 536568, 5 seeds,
40 epochs, batch 128, tight CIs; majority baseline 0.371).**
  dense K   drop0%        drop15%       drop30%       drop45%
  K=1       0.812+/-.007  0.390+/-.038  0.355+/-.017  0.292+/-.033
  K=2       0.809+/-.005  0.413+/-.032  0.392+/-.024  0.348+/-.014
  K=4       0.784+/-.008  0.560+/-.028  0.423+/-.013  0.368+/-.007
All at IDENTICAL 26312 params. The K sweep is MONOTONIC in robustness at every
dropout level (more columns = more robust), CIs separated. Clean cost is tiny
(K=4 -0.028 vs K=1); at 15% sensor dropout K=4 beats K=1 by +0.170 (CIs
non-overlapping). Under dropout only the committee stays near/above the majority
baseline (at 30% K=4 0.423 > 0.371 > K=1 0.355; at 45% K=4 0.368 ~ baseline, K=1
0.292 below). Per-seed variance also shrinks with K. Confirms H1 + variance
collapse with tight statistics.

**H3 RESULT -- cncp vs dense committee (job 536402, 5 seeds each, FIRST clean
cortical win of the whole loop).**
             drop15%              drop30%              drop45%
  cncp  K=1  0.370  dense 0.391   0.372 / 0.356        0.328 / 0.295   ~equal (no vote)
  cncp  K=2  0.465  dense 0.416   0.453 / 0.396        0.400 / 0.350   cncp +0.05
  cncp  K=4  0.630  dense 0.555   0.483 / 0.425        0.415 / 0.372   cncp +0.04..+0.08
Clean accuracy identical (cncp K4 0.784 = dense K4 0.785). At K=1 (no voting) the
two are equivalent; the cortical advantage appears ONLY under voting (K>=2) and
GROWS with K (delta K2 ~+0.05, K4 up to +0.075). At K4/30% and /45% the CIs are
separated (significant); at 15% the cncp CI is wide (+/-0.079, cncp committee has
higher seed variance). So the cortical column votes MORE ROBUSTLY than a dense
cell in committee form.

**CONFOUND (must flag -- Codex will).** The two arms differ in TWO ways, not one:
(1) neuron type (8-node cortical column vs plain cell) and (2) vote target -- the
cncp arm (MultiColumnVotingCell) votes on the L2/3 object SUBSPACE, the dense arm
(CommitteeVotingCell) votes on the FULL hidden state. So the cncp>dense win is a
DESIGN-level result, not cleanly attributable to the neuron. Disentangling needs a
3rd arm (Iteration 7c: dense cell + subspace vote, or cortical cell + full-state
vote). The fact that the gap is zero at K=1 and grows with K hints the cortical
L2/3 representation votes better, but 7c is needed to confirm.

Engineering: cncp is GPU-launch-bound slow (8 nodes x K Python loop, esp. K4); ran
dense-only concurrently on a 2nd node (direct sbatch, no rsync) to secure the
headline fast while the cncp job ground through K4.

**Engineering note.** The cortical committee is slow on GPU (nested Python loops
over 8 lamina nodes x K columns -> thousands of tiny kernel launches per step;
launch-bound, batch size barely helps -- only fewer steps do). Fix for 7b/LV: try
the cncp arm on CPU (tiny ops, no launch overhead / GPU contention) or cut cncp to
K in {1,4} x 3 seeds. dense arm is fast and carries the headline.

**Iteration 7b (built, submit after the cncp job frees ~/thesis-person -- must not
rsync over a running job).** The essential missing control: a single monolith TRAINED with
sensor-dropout augmentation (ChannelDropout: train-time whole-channel dropout, no
inverted-dropout rescaling, inference no-op; 0 extra params). If the committee
(no dropout training) still beats the augmented K=1 monolith's degradation curve,
the win is structural, not "ensembles get dropout robustness for free". Ready as a
one-liner: CO_KS="1" CO_TRAINDROP="0.3" over cncp+dense, 5 seeds; the runner tags
train_drop into the filename so it does not clobber the plain K=1 runs. Second
transfer (generality across task types): the Lotka-Volterra committee (regression,
noisy views, F=2 too small to partition) -- reuses build_committee_model(
task="regression") + make_views(mode="noisy").

**7b RESULT (DECISIVE -- REFUTES the committee as a robustness contribution; job
536618, 5 seeds).**
  dense                    drop0%   drop15%   drop30%   drop45%
  K=1 plain (monolith)     0.817    0.391     0.356     0.295
  K=1 +dropout-train (AUG) 0.781    0.759     0.655     0.626
  K=4 plain (COMMITTEE)    0.785    0.551     0.428     0.375
  K=4 +dropout-train       0.773    0.752     0.660     0.630
The dropout-trained monolith CRUSHES the plain committee: +0.21 at 15%, +0.25 at
45%. And committee+dropout ~ monolith+dropout (0.752 ~ 0.759): the committee adds
NOTHING once the standard technique is used. Honest conclusion: the partial-view
voting committee is a weak, implicit form of dropout regularisation; explicit
dropout augmentation is simpler and strictly better. The committee robustness
headline AND the cncp>dense committee sub-finding are both dominated by the trivial
baseline -- not a thesis-worthy robustness contribution. The control did its job.

Caveat (only surviving niche, untested): dropout-train had a "home advantage" --
it trained on the SAME corruption family it was tested on (random channel drop);
the committee never trained on any corruption. So a NARROWER claim survives:
robustness to UNANTICIPATED corruption (train aug-monolith on noise, test on
channel drop, vs committee). Speculative; would be 7d.

**Loop meta-pattern (Iterations 1-7).** Across the whole loop the tbt/cNCP-specific
machinery keeps NOT beating simpler baselines under fair (param-matched, control-
augmented) comparison: FiLM ~ concat; voting < concat; L5 active motor < dense
motor (fragile); committee < dropout augmentation. Honest thesis-level finding is
shaping up as a rigorous comparative study: bio-inspired cortical wiring does not
provide consistent robust advantages over simpler techniques. Whether to (a)
reframe the thesis around that nuanced finding, (b) pivot to a regime where the
cortical structure might genuinely win (inherent partial observability,
unanticipated shift, a property other than robustness), or (c) keep committee as a
documented negative and move on -- STRATEGIC decision for Alexander, surfaced
2026-07-05.

## Iteration 7d -- corruption-agnostic robustness (the surviving niche; RESULT: NEGATIVE)

**Concept.** The one claim 7b left open: the committee (trained clean) might
generalise its architectural robustness to an UNANTICIPATED corruption where
corruption-specific augmentation does not transfer. Test on Gaussian NOISE; the
dropout-aug monolith trained on channel dropout (the wrong corruption). Added
add_noise + NoiseAugment + --test-corruption (19 tests green). 3 dense jobs (noise
test), 5 seeds: plain / dropout-aug / committee / noise-aug (upper bound).

**Result (5 seeds, tight CIs, test on Gaussian noise).**
  model                          sig0.25      sig0.5       sig1.0
  K1 plain                       0.674+/-.021 0.467+/-.020 0.332+/-.018
  K1 +dropout-aug (wrong corr.)  0.691+/-.014 0.494+/-.010 0.311+/-.007
  K4 committee                   0.669+/-.005 0.480+/-.008 0.363+/-.010
  K1 +noise-aug (upper bound)    0.724+/-.012 0.670+/-.006 0.474+/-.004
NEGATIVE. The 1-seed go/no-go misled: at 5 seeds (1) the dropout-aug monolith DOES
partially transfer to noise (0.691/0.494 > plain 0.674/0.467 -- dropout aug is a
mild general regulariser, not corruption-specific), (2) the committee does NOT beat
it except at sig=1.0 (+0.053), where the committee's 0.363 is BELOW the majority
baseline 0.371 (a useless regime), (3) the correct augmentation (noise-monolith)
dominates everything. Even the narrowest surviving niche yields no reliable
committee advantage under proper statistics.

**Iteration 7 verdict -- CLOSED, negative.** The partial-view voting committee is
not a robustness contribution: dominated by dropout augmentation for anticipated
corruption (7b), no reliable advantage for unanticipated corruption (7d), and the
cncp>dense committee edge (7c-confounded) lives inside a dominated regime. Combined
with the whole loop's meta-pattern (FiLM~concat, voting<concat, L5 motor<dense,
committee<dropout aug), the honest thesis-level outcome is a RIGOROUS NEGATIVE /
comparative study: under fair (param-matched, control-augmented) evaluation the
tbt/cNCP cortical machinery does not beat simple baselines. That is a legitimate,
publishable finding -- but an OUTLOOK-chapter result, not the positive core the
/goal loop set out to find. Recommended next: return to the thesis core (Farsang
dense/NCP/cNCP x 3 families x Optuna) and write the tbt loop up as the honest
negative outlook. Surfaced to Alexander 2026-07-05.

## Iteration 8 -- data efficiency / inductive bias (RESULT: NEGATIVE + a methodology catch)

**Concept.** After the robustness direction closed, the user chose "a different
property where the cortical structure might genuinely win". The best-motivated
shot: DATA EFFICIENCY -- does the sparse cNCP wiring generalise better from LESS
data (a steeper learning curve) at matched params? Learning curves on
person-activity, cncp/ncp/dense, train fractions {5,10,25,50,100}%, 5 seeds
(run_data_efficiency_benchmark.py + plot + cluster scripts; job 536660, 75 runs).

**Result (5 seeds).**
  wiring params  5%          10%         25%         50%         100%
  cncp   44999  0.522+/-.068 0.612+/-.044 0.741+/-.026 0.795+/-.029 0.844+/-.006
  ncp    45927  0.481+/-.115 0.634+/-.045 0.735+/-.043 0.786+/-.025 0.842+/-.010
  dense  26311  0.478+/-.082 0.607+/-.052 0.696+/-.049 0.782+/-.011 0.811+/-.015

**METHODOLOGY CATCH (self-flagged).** params are NOT matched: dense 26k vs cncp/ncp
45k (1.7x). The person-activity builder matches cncp<->ncp but leaves dense a
smaller reference; I failed to check before running. So the small cncp>dense edge
is confounded by capacity, not structure. The FAIR comparison is already present:
cncp vs ncp (both ~45k) shows NO data-efficiency advantage (5% 0.522 vs 0.481, 10%
0.612 vs 0.634, then ~identical, mixed signs, all CIs overlapping). The only
separated CI (cncp-dense +0.033) is at 100% data -- the opposite of the low-data
hypothesis. NEGATIVE. The 1-seed go/no-go (+0.19 at 10%) was seed noise: the dense
10% seed was unlucky (0.403), 5-seed mean 0.607. Recurring lesson: 1-seed checks
mislead; the effect sizes here are within seed noise.

**LOOP CONCLUSION (Iterations 1-8, 6 consecutive fair negatives).** Across
robustness (FiLM, voting, occlusion, committee, corruption-agnostic), sensorimotor
(L5 motor), and now sample complexity (data efficiency), the tbt/cNCP cortical
machinery does NOT beat fair (parameter-matched, control-augmented) baselines, and
cncp ~ ncp throughout (the specific cortical topology adds nothing over a generic
sparse NCP). This is a robust, rigorously-established NEGATIVE result -- valuable
and honest, and exactly the kind of careful comparative study a thesis can defend,
but it is not the positive "genuinely good tbt_cNCP" the /goal loop sought. Two
1-seed go/no-gos in a row produced false positives that evaporated at 5 seeds; the
evidence that there is no positive to find via more angles is now strong.
Recommendation (2026-07-05): STOP the positive-tbt search; return to the thesis
core (Farsang dense/NCP/cNCP x 3 families x Optuna) and write the tbt loop up as
the honest negative/comparative outlook.

## Iteration 9 -- multi-timescale lamina (a real ARCHITECTURE change; RESULT: NEGATIVE)

**Concept.** Not another property test -- an actual architectural change to
differentiate cNCP from a uniform NCP: give each lamina a different timescale
(fast sensory L4/Thal, slow object/context L2-3/L6) by scaling its integration
step (opt-in timescale_prior on CorticalColumnCell; default None = thesis
behaviour byte-identical, ~0 extra params; 28 cncp tests + 33 person/committee
tests green).

**Honest prior (stated before running).** Weak: cfc_lrc already LEARNS its time
constants (t_a/t_b), so a per-lamina prior only affects init, not the converged
solution.

**Result (5 seeds, param-matched: identical model, only the prior differs).**
  cncp-plain    [0.797 0.733 0.798 0.799 0.810] -> 0.787 +/- 0.027
  cncp-cortical [0.753 0.816 0.812 0.793 0.758] -> 0.786 +/- 0.026   delta -0.001
NEGATIVE, exactly as predicted. The 3-seed check showed +0.018 (seed noise); seeds
3-4 reversed it; at 5 seeds the effect is -0.001 (CIs identical). This time I did
NOT chase the small-sample signal -- I confirmed it away. The mechanistic reason
(learned time constants) explains the null and generalises: cfc_lrc is expressive
enough that structural priors on quantities it already learns add nothing.

**FINAL loop conclusion (9 iterations, 7 fair negatives, incl. a genuine
architecture change).** The result is robust and mechanistic, not a run of bad
luck: with an expressive cell (cfc_lrc) at matched parameters, cNCP ~ NCP ~ dense
across every property tried (location conditioning, voting, active sensing,
robustness, corruption-agnostic robustness, data efficiency, multi-timescale). The
specific cortical topology and its add-ons do not beat simple baselines under fair
comparison. This is a rigorously-established NEGATIVE -- a legitimate, honest,
publishable comparative finding and a good thesis OUTLOOK chapter, but not the
positive "genuinely good tbt_cNCP" the loop sought. Continuing to try more
architectural tweaks is not warranted: the null is explained by cell
expressiveness + matched params, which no single-mechanism tweak changes.
Constructive terminal deliverable: consolidate the 9-iteration study into a
thesis-ready results summary (the "vorweisbar" artifact). Surfaced to Alexander
2026-07-06.

## Candidate concepts for later iterations (if FiLM is not enough)

- Stronger gate init sweep (loc_gain_init 0.1 -> 1.0) to separate init-weakness
  from structural weakness of the multiplicative gate.
- FiLM + concat combined (does the additive prior still add over raw concat?).
- Learnable reference frame (replace the fixed grid-cell Fourier code with a
  small learned location encoder).
- Voting on trajectories (multi-column consensus for the double-well, where
  local ambiguity is exactly the regime voting should help).
- Deeper location injection (condition L2/3 or L5, not only L4).
- Clean parameter-matching for dense (currently 25-31k vs 45-50k) so the
  predator-prey comparison is fair.
