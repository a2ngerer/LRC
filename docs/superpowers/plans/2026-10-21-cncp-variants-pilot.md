# Pilot plan: which column variant works for which use case

Status: revision 2, 2026-10-21, after the Codex review of revision 1 (gpt-6-astra,
high; verdict "release with changes", nine blocking findings; the review and the
responses are in section 8). Nothing in this plan is a final-suite result; every
run is a pilot that decides which arms enter the final suite (decision log, Q1
scope recommendation (c)) and which are dropped or kept as outlook. Spec of the
arms: `docs/superpowers/specs/2026-10-21-cncp-v2-abstraction.md` (Part C).
Protocol gaps that make these runs pilots and not results: rollout tasks select
on the test split (a2ngerer/LRC #8), Person Activity has no subject split and no
macro-F1, no HPO grid runner exists yet.

## 1. Arms and the backbone rows

| Arm | Wiring flag | Builder | Issue | Role |
|---|---|---|---|---|
| dense3 | `dense3` | three CfC-family layers, backbone 1 (its default) | merged | parameter-matched baseline |
| ncp | `ncp` | inter/command/motor, backbone 0 (default) or 1 (flag) | merged, #11 | baseline, H-c reference |
| V2 | `cncp` | eight-node column, six RNN sub-cells plus two relays, backbone 0 | merged | status quo (diverged in the canary) |
| V2s | `cncp --backbone-layers 1` | same graph, backbone 1 in the six sub-cells | #11 | stabilised control |
| V3 | `cncp2` | four-node v2 column (G, S, D, R), rules R1 to R5, backbone 0 | #9 | abstraction, single column |
| V4 | `cncp2_multi --columns K --partition --coupling --readout` | K independent v2 columns, coupling none or lateral (no hub in this pilot) | #10 | independent columns, feasibility only |
| V1 | committee runner, `--wiring cncp --n-columns K` | K weight-shared V2 columns on partial views, L2/3 consensus | merged | status quo for H-d |

Codex finding 4: the backbone depth is fixed per matrix row and never mixed
inside a comparison.

- **Row bb0 (architecture packages, backbone 0):** ncp, cncp (V2), cncp2 (V3).
  Read as package comparisons; a V3 result in this row says nothing about the
  graph alone.
- **Row bb1 (wiring comparison, backbone 1):** dense3, ncp bb1, cncp bb1 (V2s), cncp2 bb1,
  all with `--backbone-layers 1`, all re-matched to the budget. This is the
  row where "graph vs graph at equal sub-cell" is read. Known deviation: ncp
  bb1 at 4k lands 5.4 % under budget (size 23, 3 784 effective); it is
  reported with its realised count.
  Known from #9: cncp2 at 4k lands 8.6 % under budget (u = 10, 3 655 effective,
  "≈ matched"); at 30k u = 29, 29 873. Its R6(a) init check passed 29 of 30
  synthetic cases; the one failure (LV 4k, seed 1, input scale 10) is the S
  sub-cell recurrence, which backbone 1 bounds, so cncp2 bb1 is in row bb1.
- **V3 vs V2s** crosses the rows and is a package comparison; the plan says so
  wherever it is read.
- **Fixed-width pair (canary only):** cncp bb0 vs cncp bb1 at `--size 18`
  without budget matching, so a backbone statement is not confounded with the
  capacity change of re-matching.

All budget-matched arms use `size_for_budget`; the realised
`params_effective` is in every result JSON and is the x-axis of UC-D and a
column in every table ("≈ matched" when the deviation exceeds 5 %). Cell
`cfc_lrc` (`elastance_type='asymmetric'`) everywhere; `gru` ignores the
backbone flag (no backbone) and runs only as a discrete control on dense3 and
ncp. lr 1e-3, sbatch batch sizes, seeds 0 to 2.

## 2. Use cases, metrics as the runners write them, decision rules

### UC-A: smooth autonomous rollouts (Lotka-Volterra 01, Duffing 02)

128-step closed-loop rollout after teacher-forced training, budget 4k, 200
epochs. Metrics from the LV result JSON: `closed_loop_mse`,
`teacher_forced_mse`, `closed_loop_mse_per_step`, `val_loss_curve`,
`params_effective`; with `--probe-stability` (#9) the `stability` block.

Decision rules (Codex finding 7):

- An arm **trains** when, in all three seeds, the teacher-forced loss is finite,
  the final `val_loss_curve` value is below its first value, and no seed has a
  closed-loop MSE above 10 × the dense3 median of the same row. A seed that
  violates this is a failure of the arm and is counted, never dropped.
- An arm is **non-inferior** to dense3 (row bb1) when its median closed-loop MSE
  is at or below the maximum over the three dense3 seeds (one-sided). It is
  **better** when its median is below the dense3 minimum.
- V3 is read against V2s and ncp bb1 (package comparison) and against ncp bb0
  (same row). A V3 result is never read against a V2 that failed to train.
- V4 (K = 2, `same`, `lateral`, `mean`) is a feasibility check: trains or not.

### UC-B: irregularly sampled multi-sensor classification (Person Activity 03)

Budget 30k, 50 epochs. Metric (Codex finding 1): the final `test_accuracy`
only; the runner stores no predictions, so no confusion matrix or F1 is
promised. `best_test_accuracy` is reported as the optimistic bound it is
(selection on test).

Decision rule: paired per-seed differences. Arm A **beats** arm B when the
median of the three paired differences is at least 1.0 percentage points and
all three differences are positive. Comparisons read: V3 vs V2s, V3 vs ncp bb0,
cncp bb1 vs ncp bb1 vs dense3 (row bb1), V4 lateral vs V4 none (same K and
partition). V4 vs V3 or vs dense3 is a feasibility observation only, because
the partitioned-input control the spec asks for (B.7, dense/GRU on the same
`[m_k, m_k·xyz]` views) does not exist yet (Codex finding 5); no routing or
coupling attribution is drawn from this pilot. If the budget-matched widths of
`none` and `lateral` differ, a third launch repeats `lateral` at the `none`
width via `--size` (fixed-width mechanism pair).

### UC-C: feature-channel dropout (column-voting task 04)

The committee task: training on clean partial views (`--view-mode partition`),
evaluation with channel dropout fractions 0, 0.15, 0.3, 0.45 of the seven
features (realised: 0, 1, 2, 3 channels), five channel-choice seeds each.
Metrics from the result JSON: `clean_accuracy`, `drop_fracs`, `drop_accs`,
`drop_stds` (spread over the five channel seeds, not over training seeds),
`params_effective`.

Decision rules (Codex findings 6 and 7): primary metric is the **absolute
accuracy at fractions 0.3 and 0.45** (mean over the five channel seeds, then
median over the three training seeds). An arm is **eligible** only if its clean
accuracy is within 2 pp of the best clean accuracy in the use case. Area and
the drop from 0 to 0.45 are secondary. Every row is a complete configuration
(section 3); V1 vs V4 is a package comparison (weight sharing, graph, L2/3
state vote vs logit median all change at once) under identical external views
and corruption masks (same runner, same `--view-seed` and `--drop-seeds`). The
shared/independent comparison of one graph is not part of this pilot.

V1 enters the final suite only if K = 4 with learnable λ is eligible and beats
both the λ = 0 control and the dropout-trained monolith (K = 1,
`--train-drop 0.3`) at 0.3 and 0.45 by the UC-B rule. The λ = 0 control at
K = 4 is never cut for cost.

### UC-D: width scaling (Lotka-Volterra 01)

Budgets 1k, 4k (reused from UC-A), 16k; 200 epochs; rows bb0 and bb1 of
section 1. Read: closed-loop MSE (median and all seeds) over realised
`params_effective` per arm; failed seeds plotted as failures. Outcome
statement is "a crossing between {V3, V2s} and ncp was / was not observed
under budgets 1k to 16k at lr 1e-3", nothing stronger; a crossing motivates a
64k point in the final suite.

### Priority rules (replace the stop rules of revision 1)

- A loss at lr 1e-3 with three seeds decides **priority for the HPO and the
  final suite**, never the rejection of an architecture. Only failure to train
  (UC-A rule) removes an arm from the final-suite candidates.
- V3 losing to V2s says nothing about V4; V4 is kept or dropped on its own
  feasibility and cost.
- `none ≈ lateral` on UC-B is reported as "no coupling benefit observed in this
  setting", not as unused or equivalent coupling.

## 3. Launch list (one configuration per launch, one run folder each)

Every launch is one SLURM job (one A40, fits in parallel via `*_MAXPAR`) into
its own run folder (`cluster/_run.sh`); result files cannot collide across
launches, and inside a launch the file name carries cell, wiring, seed and the
`_bb<N>` suffix (Codex findings 2 and 8). `LAUNCH.md` carries the purpose,
the SHA, the full env and the expected file count. All labels below are
prefixed with the date by the launcher.

Canary (00_canary; `LV_EPOCHS=3`, `LV_SEEDS="0 1 2"`, `LV_CELLS=cfc_lrc`,
`LV_EXTRA="--probe-stability"`):

| Label | Env | Fits |
|---|---|---|
| canary-bb0 | `LV_WIRINGS="dense3 ncp cncp cncp2"` | 12 |
| canary-bb1 | `LV_WIRINGS="dense3 ncp cncp cncp2" LV_BACKBONE=1` | 12 |
| canary-v4 | `LV_WIRINGS=cncp2_multi LV_EXTRA="--probe-stability --columns 2 --partition same --coupling lateral --readout mean"` | 3 |
| canary-size18 | `LV_WIRINGS=cncp LV_PARAM_BUDGET=0 LV_EXTRA="--probe-stability --size 18"` and the same with `LV_BACKBONE=1` | 3 + 3 |
| canary-committee | `CO_WIRINGS=cncp CO_KS="4 16" CO_SEEDS=0 CO_EPOCHS=2` (cost of K = 4 and K = 16 in the parallel setting) | 2 |

UC-A (01 and 02; `LV_EPOCHS=200`, `LV_SEEDS="0 1 2"`; Duffing adds
`LV_SYSTEM=duffing`):

| Label | Env | Fits |
|---|---|---|
| pilot-uca-{lv,duffing}-bb0 | `LV_CELLS=cfc_lrc LV_WIRINGS="ncp cncp cncp2"` | 9 |
| pilot-uca-{lv,duffing}-bb1 | `LV_CELLS=cfc_lrc LV_WIRINGS="dense3 ncp cncp cncp2" LV_BACKBONE=1` | 12 |
| pilot-uca-{lv,duffing}-gru | `LV_CELLS=gru LV_WIRINGS="dense3 ncp"` | 6 |
| pilot-uca-{lv,duffing}-v4 | `LV_CELLS=cfc_lrc LV_WIRINGS=cncp2_multi LV_EXTRA="--columns 2 --partition same --coupling lateral --readout mean"` | 3 |

UC-D (01; as UC-A bb0/bb1 with `LV_PARAM_BUDGET=1000` and `16000`):

| Label | Fits |
|---|---|
| pilot-ucd-1k-bb0, pilot-ucd-1k-bb1 | 9 + 12 |
| pilot-ucd-16k-bb0, pilot-ucd-16k-bb1 | 9 + 12 |

UC-B (03; `PA_EPOCHS=50`, `PA_SEEDS="0 1 2"`, `PA_CELLS=cfc_lrc`;
`SBATCH_EXTRA="--time=03:00:00"`):

| Label | Env | Fits |
|---|---|---|
| pilot-ucb-bb0 | `PA_WIRINGS="ncp cncp cncp2"` | 9 |
| pilot-ucb-bb1 | `PA_WIRINGS="dense3 ncp cncp cncp2" PA_BACKBONE=1` | 12 |
| pilot-ucb-v4-none | `PA_WIRINGS=cncp2_multi PA_EXTRA="--columns 4 --partition region --coupling none --readout concat"` | 3 |
| pilot-ucb-v4-lateral | same with `--coupling lateral` (density 0.25) | 3 |
| pilot-ucb-v4-lateral-fixed (conditional) | as lateral with `PA_PARAM_BUDGET=0 PA_EXTRA="... --size <u of none>"` if the matched widths differ | 3 |

UC-C (04; `CO_CELL=cfc_lrc`, `CO_SEEDS="0 1 2"`, `CO_EPOCHS=40`,
`CO_VIEWMODE=partition`, budget 30k; `--view-seed` and `--drop-seeds` at
their defaults in every row):

| Label | Env | Fits |
|---|---|---|
| pilot-ucc-v1 | `CO_WIRINGS=cncp CO_KS=4` (learnable λ, mean) | 3 |
| pilot-ucc-v1-lambda0 | `CO_WIRINGS=cncp CO_KS=4 CO_EXTRA="--vote-lambda 0"` | 3 |
| pilot-ucc-v1-median | `CO_WIRINGS=cncp CO_KS=4 CO_EXTRA="--consensus median"` | 3 |
| pilot-ucc-k1 | `CO_WIRINGS=cncp CO_KS=1` | 3 |
| pilot-ucc-viewavg | `CO_WIRINGS=cncp CO_KS=4 CO_EXTRA="--view-average"` | 3 |
| pilot-ucc-traindrop | `CO_WIRINGS=cncp CO_KS=1 CO_EXTRA="--train-drop 0.3"` | 3 |
| pilot-ucc-v4 | `CO_WIRINGS=cncp2_multi CO_KS=4 CO_EXTRA="--consensus median"` (independent columns on the committee views, logit median) | 3 |
| pilot-ucc-v4-k1 | `CO_WIRINGS=cncp2_multi CO_KS=1` | 3 |

Backbone in UC-C: 0 throughout (V1 and V4 are both backbone-0 packages).

## 4. Gates

1. Unit tests green in every worktree (#9, #10, #11, #12) and on the stacked
   integration branch `int/pilot-2026-10-21`; `cluster/test_launchers.sh` green.
2. Integration branch rsynced to `~/masterarbeit/code`, `uv sync --extra cuda`
   done once, `squeue` empty of own jobs.
3. Canary gate per new arm (cncp2, cncp2_multi K = 2, cncp bb1), read from the
   `stability` block and the losses (R6 of the spec, Codex finding 3): all
   state norms finite at init, after one step and after one epoch; teacher-
   forced train loss after epoch 1 at most 10 × the dense3 bb1 value of the
   same canary; closed-loop MSE finite. The known-diverging cncp bb0 is the
   negative reference and is expected to fail; its failure blocks nothing.
   A new arm that fails the gate blocks only the comparisons that depend on
   it (V3 failing blocks V3 and V4; V4 failing blocks V4), the baselines and
   the other rows launch.
4. Committee canary gives the per-fit cost of K = 4 and K = 16; UC-C is sized
   with it (section 5).

## 5. Cost (A40, one job per GPU, up to 8 jobs; measured and assumed)

Measured: LV 200 epochs ≈ 1.3 GPU-min per fit for cfc_lrc dense/ncp/gru with
six fits sharing one GPU; cncp ≈ 2.7 × ncp (PA: 16 vs 6.8 min at 50 epochs).
Assumed until the canary measures them: cncp2 ≈ 1.4 × ncp (half the launches
of cncp, spec B.4), cncp2_multi K = 2 ≈ 2 × cncp2, K = 4 ≈ 4 × cncp2 (Python
loop over columns).

| Block | Fits | GPU-min (est.) | Jobs |
|---|---|---|---|
| Canaries LV (33 fits, 3 epochs) | 33 | 6 | 5 |
| Canary committee | 2 | 10 to 40 | 1 |
| UC-A LV: bb0 9 (ncp 1.3, cncp 3.5, cncp2 1.8 per seed), bb1 12 (1.3, 1.3, 3.5, 1.8), gru 6 (1.3), v4 3 (3.6) | 30 | 20 + 24 + 8 + 11 = 63 | 4 |
| UC-A Duffing | 30 | 63 | 4 |
| UC-D 1k (≈ 0.6 ×) and 16k (≈ 2.5 ×) of the bb0 + bb1 blocks | 42 | 26 + 110 = 136 | 4 |
| UC-B: bb0 (7 + 16 + 10 per seed), bb1 (7 + 7 + 16 + 10), v4 none and lateral (≈ 40 each per seed) | 27 (+3) | 99 + 120 + 120 + 120 = 459 (+120) | 4 (+1) |
| UC-C: 24 fits, K = 4 cncp unmeasured | 24 | 240 to 720 | 8 |
| Total | 188 (+3) | ≈ 16.5 to 25 GPU-h | 30 (+1) |

Wall-clock: the canaries finish in under 15 minutes; UC-A, UC-D and UC-B are
16 jobs of 20 to 60 minutes each and run in two waves of eight; UC-C is one
wave sized by its canary. If the committee canary puts K = 4 cncp above 25
GPU-min per fit, UC-C runs the λ = 0 control, V1 K = 4, the dropout monolith
and V4 K = 4 (12 fits) and defers the median, K = 1 and view-average rows.

## 6. Order

1. Gates 1 and 2; all five LV canaries and the committee canary at once.
2. Read gate 3. Launch UC-A LV (bb0, bb1, gru) at once; v4 only if canary-v4
   passed.
3. When UC-A LV bb1 shows the arms train (first job done), launch UC-A Duffing,
   UC-D and UC-B (16 jobs, two waves).
4. UC-C after its canary cost is known, sized per section 5.
5. Fetch every run folder (`cluster/fetch_run.sh`), write the pilot summary
   with realised `params_effective`, failures counted, and the decision per
   arm (final-suite candidate / outlook / dropped) into the work-log and the
   decision log.

## 7. What this pilot cannot decide

- Final-suite numbers: the rollout tasks select on the test split (#8) and
  Person Activity has no subject split; all accuracies are optimistic.
- The HPO question: every arm runs at lr 1e-3 and the sbatch batch size; the
  final suite tunes 8 candidates per arm (section 2, priority rules).
- Statistical claims: three seeds give ranges and paired signs, not tests.
  Final seeds are 10 for H-a/H-b and 5 for H-c.
- Any attribution to routing, coupling or weight sharing (UC-B, UC-C): the
  controls the spec requires for that are not built; the pilot reports package
  comparisons and feasibility.
- The cncp2 relay is a convex mixture only for dt ≥ 0; the Person Activity
  loader keeps negative gaps (12 in the raw file). All arms see the same dt,
  so the comparison is fair, but the boundedness statement holds for dt ≥ 0
  only (Codex code review, finding 6).
- The unit-level R6 check reads the final state only; the cluster probe
  (`--probe-stability`) takes the maximum over batch and time, which is the
  gate quantity. The known cncp2 outlier (synthetic scale 10) is an `xfail`,
  so a green suite is not an R6 acceptance; the canary is.

## 8. Codex review of revision 1 and the responses

Review file: scratchpad `codex-pilot-plan-review.md` (gpt-6-astra, high,
read-only, run from the `benchmarks` checkout). Verdict: release with changes.

| # | Finding (short) | Response in revision 2 |
|---|---|---|
| 1 | UC-B promises F1/confusion that the runner does not store | UC-B metric is the final `test_accuracy` only (section 2) |
| 2 | result files collide across backbone, budget, V4 configurations | one launch per configuration, own run folder, `_bb<N>` suffix, expected file count in LAUNCH.md (section 3; launcher commit on #12) |
| 3 | "finite" is no stability gate; no state or gradient norms stored | gate 3 tied to R6: `--probe-stability` block (init / one step / one epoch), TF loss ≤ 10 × dense3, dense3 as a simultaneous reference, cncp bb0 as the negative reference, failure blocks only dependants (section 4; flag on #9) |
| 4 | V2s removes the backbone confound only partly | backbone fixed per row (bb0 packages, bb1 wiring comparison), V3 vs V2s declared a package comparison, fixed-width canary pair at `--size 18` (section 1) |
| 5 | UC-B lacks the partitioned-input and width-matched coupling controls | V4 on UC-B is feasibility only, no routing attribution; `none` vs `lateral` at matched budget plus a fixed-width repeat if the widths differ; hub removed from the pilot and from #10 (section 2) |
| 6 | UC-C does not isolate weight sharing; rows underspecified | V1 vs V4 declared a package comparison under identical views and masks; every row spelled out with wiring, K, flags; monolith is K = 1 with `--train-drop 0.3`; λ = 0 at K = 4 protected (sections 2 and 3) |
| 7 | decision and stop rules allow wrong conclusions | numeric rules: one-sided non-inferiority (UC-A), paired differences ≥ 1 pp with consistent sign (UC-B), absolute accuracy at 0.3/0.45 with a 2 pp clean-accuracy eligibility (UC-C), failures counted, UC-D wording; stop rules replaced by priority rules (section 2) |
| 8 | launchers cannot express the matrix; A40 vs A100; PA walltime | matrix expressed as per-configuration launches with `LV_/PA_/CO_EXTRA`, `*_BACKBONE`, `*_PARAM_BUDGET=0` + `--size`, `SBATCH_EXTRA` for walltime; the cluster is A40 (the review prompt said A100 in error) (section 3) |
| 9 | cost table counted an undefined matrix | matrix fixed first, 188 fits recounted per block, committee canary at K = 4 and K = 16, wall-clock by waves (section 5) |
| 10 (minor) | budget tolerance and "sensor" wording | realised `params_effective` everywhere, "≈ matched" above 5 %; UC-C renamed feature-channel dropout with realised 0/1/2/3 of 7 channels |

## 9. Codex code review of the merged branches and the responses

Review file: scratchpad `codex-code-review.md` (gpt-6-astra, high, read-only on
`int/pilot-2026-10-21` at 501e51d). Verdict: fix first. Spec conformance of
cncp2 (B.3 equations, R2 init, parameter counting, committee S slot, probe
semantics, backbone pass-through, ssh quoting) was confirmed.

| # | Finding (short) | Response |
|---|---|---|
| 1 | dataset symlink not copied to the cluster | rsync `--copy-links` (21 MB dataset lands as a real folder) |
| 2 | `CO_PARAM_BUDGET=0` still passed `--param-budget` | same conditional `budget_flag` as LV/PA |
| 3 | failed fits exit 0 through `echo` | `run_one` returns the runner's status; each job counts result files against the matrix and exits 1 if any is missing (no offline failure simulation; the count check is the regression) |
| 4 | collision check after the code sync, not atomic | run folder reserved with a plain `mkdir` before `run_rsync` |
| 5 | PA job lists the old result path | lists `$OUTDIR`; smoke-test regex now catches the old paths |
| 6 | relay boundedness needs dt ≥ 0, PA has negative gaps | documented in section 7; a shared dt treatment for all arms is a separate issue |
| 7 | R6 unit test reads the final state only, outlier is xfail | documented in section 7; the gate is the cluster probe (max over time) |
| 8 | test gaps: GPU 128 steps, cncp2 bb1 budget, probe on/off equality, cncp2 committee | covered by the canaries (GPU, 128 steps, bb1 budget) and the #9 report (identical val-loss curve with and without the probe); cncp2 committee tests are part of #10 |
| 9 | LV plotter omits dense3 and cncp2 | added to `WIRING_ORDER` (one launch per backbone depth, so no mixing) |
