# Running the benchmark on the TU Wien dataLAB cluster

Workflow and conventions adopted from the proven `arc-jax-rl` cluster setup
(same cluster, same submission pattern), adapted from JAX to TensorFlow.

## Prerequisites (one-time)

1. **TU Wien VPN** must be active — the cluster is not reachable from the
   public internet.
2. **SSH config** (`~/.ssh/config`):

   ```sshconfig
   Host datalab
       HostName cluster.datalab.tuwien.ac.at
       User e<matrikelnummer>
       IdentityFile ~/.ssh/id_datalab
       IdentitiesOnly yes
       Compression yes
       TCPKeepAlive yes
       ServerAliveInterval 60
   ```

3. **Environment setup on the cluster** (inside a GPU session, NOT on the
   login node — compute nodes have no internet, but the interactive session
   inherits the login node's network for package downloads):

   ```bash
   ./cluster/sync_to_cluster.sh         # from your machine
   ssh datalab
   srun -p GPU-a40 --gres=gpu:a40:1 -n1 --pty bash
   cd ~/thesis-benchmark
   bash cluster/setup_env.sh            # installs uv, syncs deps, verifies GPU
   exit; exit
   ```

## Submitting the benchmark

```bash
# Full 4x2x6x5 matrix (240 runs), array throttled to 8 concurrent GPUs:
./cluster/submit_benchmark.sh

# Subset / different training config (args after -- go to run_benchmark.py):
./cluster/submit_benchmark.sh -- --cells ltc,lrc --seeds 0,1,2 --iters 2000

# Different partition (A100 is faster; max 8 concurrent GPUs per user):
CLUSTER_PARTITION=GPU-a100 CLUSTER_GPU=a100:1 ./cluster/submit_benchmark.sh
```

One array task = one `(cell, wiring, system, seed)` run = one JSON in
`results/runs/`. The index mapping is printed by
`uv run python experiments/run_benchmark.py --list`.

## Monitoring

```bash
ssh datalab squeue -u \$USER
ssh datalab tail -f thesis-benchmark/outputs/slurm/thesis-benchmark-<jobid>_<task>.out
```

GPU utilization is logged to `outputs/slurm/*.gpu.csv` (30 s polling).

## Fetching and evaluating results

```bash
./cluster/fetch_results.sh             # rsync results/ back (add --logs for SLURM logs)
uv run python experiments/aggregate_results.py   # summary.md/csv + Wilcoxon + Cohen's d
uv run python experiments/plot_results.py        # loss curves, phase portraits, gradient flow
```

## Cluster facts (dataLAB)

| Fact | Value |
|------|-------|
| Login | `cluster.datalab.tuwien.ac.at` (VPN required) |
| Partitions | GPU-v100, GPU-a40 (default), GPU-a100, GPU-a100s, GPU-l40s, GPU-h100 |
| Hard limits | max 8 concurrent GPUs/user, max walltime 168 h, `/home` quota 100 GiB |
| Code transfer | rsync only (no GitHub access on the cluster) |
| Environment | uv + `tensorflow[and-cuda]` (CUDA 12 wheels, no `module load` needed) |
| `/tmp` | not writable on virtual nodes (`i-*`) — sbatch redirects TMPDIR to `$HOME/tmp` |
| Quota hygiene | `uv cache prune` and delete old `results/runs` on the cluster regularly |

## Estimated cost

A single run (2000 iterations, eager TF, small RNNs) takes well under an
hour on an A40; one timing run locally first is recommended
(`uv run python experiments/run_benchmark.py --index 0 --iters 100`).
With the %8 throttle the full 240-run matrix completes in a few hours of
wall-clock time without ever exceeding the GPU quota.

The solver cells (`ltc`, `lrc` and its variants, `ctrnn`, `mm_ltc`, `mm_lrc`)
run `--ode-unfolds` explicit Euler sub-steps per input step (default 6 in the
Lotka-Volterra and Person Activity runners, on every wiring), so their wall
time scales with the sub-step count. Since `ctrnn` follows this rule (issue
#56) instead of its former single step, a `ctrnn` arm under `ncp` takes about
3x the wall time it used to; size the walltime accordingly.

## Benchmark v2 (fixed cells)

`./cluster/submit_benchmark_v2.sh` submits the 480-run v2 matrix
(`--profile v2`: mm_ltc/mm_lrc/cfc + gradient-clip axis). Results land in
`results/runs_v2/`. How to compare v1 vs v2: see
`experiments/BENCHMARK_COMPARISON.md`.

## Re-extracting papers with Docling (knowledge ingestion)

`cluster/extract_papers.sbatch` re-extracts every `papers/**/*.pdf` to clean
Markdown with Docling on a GPU, running the full pipeline (layout + tables +
`--enrich-formula` → LaTeX + `--enrich-code`). Output mirrors into
`papers/.extracted/`. Rationale: formula enrichment (CodeFormulaV2, an
autoregressive VLM) is too slow on CPU and has no MPS support — see
`claude_instructions/karpathy-pattern.md` → Extraktions-Schicht.

**Compute nodes have no internet**, so warm the uvx env + HF model cache once
in an interactive GPU session before submitting the batch job:

```bash
./cluster/sync_to_cluster.sh                 # NOTE: ensure papers/ is synced too
ssh datalab
srun -p GPU-a40 --gres=gpu:a40:1 -n1 --pty bash
cd ~/thesis-benchmark
# downloads Docling + ~1 GB models, one paper as canary:
uvx --from docling docling convert --enrich-formula --device cuda \
    --to md --output /tmp/_warm papers/lechner2020-ncp.pdf
exit; exit

sbatch cluster/extract_papers.sbatch         # now runs offline from the cache
```

Fetch the Markdown back to your machine:

```bash
rsync -a datalab:thesis-benchmark/papers/.extracted/ papers/.extracted/
```

`sync_to_cluster.sh` currently rsyncs the code tree; extend it to include
`papers/`, or push them once with `rsync -a papers/ datalab:thesis-benchmark/papers/`.

## Fixed-grid HPO (Chapter 5 protocol)

`./cluster/submit_grid_hpo.sh <label>` submits `experiments/run_grid_hpo.py` for
ONE (task, cell, wiring) as one job (`cluster/grid_hpo.sbatch`): grid lr
{1e-4, 1e-3, 3e-3, 1e-2} x clipnorm {off, 1.0} on tuning seed s1, the top 2 on
s2, the better mean wins; a winner at 1e-4 / 1e-2 extends the grid once by
3e-5 / 3e-2. The fits run sequentially on one GPU, each into its own folder
under `results/`, next to `results/hpo_selection.json`; a requeued job reuses
the finished fits.

```bash
GH_TASK=lv GH_CELL=cfc_lrc GH_WIRING=ncp GH_SYSTEM=duffing ./cluster/submit_grid_hpo.sh <label>
GH_TASK=pa GH_CELL=gru GH_WIRING=ncp ./cluster/submit_grid_hpo.sh --dry-run <label>
GH_TASK=co GH_CELL=cfc_lrc GH_WIRING=ncp GH_EXTRA="--grid-clip off --n-columns 3 --backbone-layers 1" \
    ./cluster/submit_grid_hpo.sh <label>
```

`GH_TASK=co` is the consensus study (committee runner, `04_column_voting`):
learning rate only with one fixed `--grid-clip`, selection on
`val_macro_f1_mean`, epoch guard in `hpo_selection.json`
(`epochs_guard_triggered`, `final_epochs`) and the K=3 -> K=30 transfer check
`run_grid_hpo.py --transfer-guard <K3 hpo_selection.json> <K30 result JSON>`
(see the driver docstring).

`GH_TUNING_SEEDS` (default `100 101`, disjoint from the final seeds 0..9;
`run_grid_hpo.py` refuses seeds in 0..9 without `--allow-seed-overlap`),
`GH_PARAM_BUDGET` (default 4000 lv / 30000 pa, co) and `GH_EXTRA` (driver flags
such as `--grid-lr`, `--grid-clip`, and runner flags such as `--epochs`,
forwarded) are optional. Clipping is on the global gradient norm
(`optimizer.global_clipnorm`). `run_lotka_volterra_hpo.py`
(Optuna) is exploratory and not part of the final suite.

## Run folders (task launchers, since 2026-10-21)

The cluster workspace is one code tree (`CLUSTER_REPO_DIR`, rsync'd with
`--delete`) and a benchmarks tree (`CLUSTER_BENCH_DIR`) with one folder per
benchmark family (`00_canary`, `01_lotka_volterra`, `02_duffing`,
`03_person_activity`, `04_column_voting`, `05_halfcheetah_ppo`). The task
launchers (`submit_lotka_volterra.sh`, `submit_person_activity.sh`,
`submit_committee.sh`) take a run label and create

```
<CLUSTER_BENCH_DIR>/<NN_benchmark>/runs/<YYYY-MM-DD>_<label>/
    LAUNCH.md   written before sbatch: purpose, git SHA, command, env, job id
    results/    runner --outdir (JSONs)
    logs/       SLURM stdout/stderr
    figures/    plots rendered at the end of the job
    wandb/      offline wandb runs (only with *_WANDB=1)
```

A run folder is never reused (an existing folder aborts the launch). `-n` is a
dry run that resolves every path and prints the sbatch command without touching
the cluster. `BENCH=00_canary` redirects a smoke test into the canary folder.
Fill the `Purpose` line of `LAUNCH.md` (`LAUNCH_PURPOSE="..."` at submit time or
by hand afterwards) and the `Outcome` line once the run is evaluated. Fetch with
`./cluster/fetch_run.sh <NN_benchmark> <run>` (mirrors to
`results/<NN_benchmark>/<run>/`); the old `fetch_*.sh` are thin wrappers.
`cluster/test_launchers.sh` checks the dry-run paths offline.

## Dataset (Person Activity)

The Person Activity CSV is not tracked in git. On the cluster it lives inside
the code tree at `<CLUSTER_REPO_DIR>/data/person/ConfLongDemo_JSI.txt` and is
not part of the rsync'd tree: `run_rsync` (`cluster/_run.sh`) and
`sync_to_cluster.sh` pass `--filter='P data/'`, which protects `data/` on the
cluster from `--delete`. A launch from a checkout without `data/` (a fresh
worktree) leaves the cluster copy untouched; a checkout that has `data/`
(e.g. the `data` symlink to the main checkout, followed by `--copy-links`)
still updates it. `submit_person_activity.sh`, `submit_committee.sh` and
`submit_grid_hpo.sh` with `GH_TASK=pa|co` check the file with `ssh test -f`
after the rsync and abort before `sbatch` when it is missing. Copy it once:

```bash
rsync -a data/person/ datalab:masterarbeit/code/data/person/
```

## Final suite (manifest-driven arrays)

`./cluster/submit_suite.sh [-n] [--wandb] <config> <label>` submits one campaign
YAML (`configs/suite/*.yaml`, schema in `experiments/suite_manifest.py`) as SLURM
arrays into one run folder. `experiments/suite_manifest.py` expands it into one
manifest line per task: stage `hpo` one `run_grid_hpo.py` call per arm into
`results/hpo/<cell>_<wiring>/`, stage `final` one fit per (arm, seed) into
`results/` with the lr and clipnorm of the arm's `hpo_selection.json` under
`hpo_dir` (the HPO run fetched with `fetch_run.sh`). Each group (`default`, and
`heavy` for the arms the YAML lists under `heavy:`) is its own array
`--array=0-(T-1)%8` with its own `runs_per_gpu`, `mem`, `walltime` and `cpus`;
`cluster/suite_array.sbatch` runs `RUNS_PER_GPU` lines per GPU (`cc_launch_bundle`)
and writes per-fit logs to `<run>/logs/run-<job>_idx<line>.log`.
Every campaign sets `ode_unfolds` (6) and `dt_scale` (`auto` = 1 / the task's
dt: 12.8 Lotka-Volterra, 5.12 Duffing, 1 / the median train dt for Person
Activity) explicitly; the manifest passes both to every runner call, and the
runners record the numeric value in `config.effective.dt_scale`.

```bash
uv run python experiments/suite_manifest.py configs/suite/canary_lv_hpo.yaml --list
BENCH=00_canary ./cluster/submit_suite.sh -n configs/suite/canary_lv_hpo.yaml suite-canary
# after a walltime kill: same config into the existing run folder; lines whose
# result JSON / hpo_selection.json exists are skipped
./cluster/submit_suite.sh --resume <config> <YYYY-MM-DD_label>
```

`--wandb` adds `--wandb` to every call; the offline runs land under `<run>/wandb/`
in project `thesis-benchmarks` with group = the campaign (the YAML stem,
`SUITE_CAMPAIGN`), job_type = `<task>/<stage>` (`SUITE_STAGE`), run name = the
result JSON stem and tag `protocol:<protocol_version>` (`src/benchmark/tracking.py`
`track_fit`). Upload a fetched run folder with
`./cluster/wandb_sync.sh results/<NN_benchmark>/<run>`. The LV and PA result
stems carry neither lr nor clip, so in stage hpo the grid points of one arm and
tuning seed share a run name: split them by `config.lr` / `config.clipnorm` or
`config.config_hash`.

Every fit stamps `protocol_version` (`PROTOCOL_VERSION` in
`src/benchmark/checkpoints.py`) and `config_hash` (sha256 of the rest of
`config.effective`) into `config.effective`. Besides the builder arguments,
`config.effective` holds the training knobs (epochs, batch size, clipnorm), the
data draw and, per task, the evaluation knobs (LV: anchors, horizon, theta) and
the corruption protocol (committee), so two fits that differ in any of them
differ in `config_hash`. Any change of the frozen protocol after the smoke
stage S0, including what `config_hash` covers, is a new `PROTOCOL_VERSION`.
Exceptions that do not bump the version: a key that records a decision
pre-registered in F1 and taken at S0 (op-determinism, F1 decision l, #76), and
a change pre-registered in F1 and decided at S1 (the epochs of the S1 epoch
gate, F1 decision m, #78, #79); both change `config_hash`.

The frozen protocol (F1-2026-10-26, #71) is one hpo and one final YAML per task:
`configs/suite/{lv,duffing,mt,pa}_{hpo,final}.yaml` and the consensus study
`co_k3_hpo.yaml`, `co_k3_final.yaml`, `co_k30_final.yaml` (K = 30 after the
transfer guard `guard_co_k30_transfer.yaml`, resumed into its run folder), plus
`mt_final_static_w.yaml` and the descriptive controls `control_*.yaml` (own run
folders: their result names equal those of the main arm). Rollout arms train
300 epochs and Person Activity 50. The epochs belong to the task: after the S1
epoch gate (G0-E, F1 decision m) every Duffing and multitimescale_lib campaign
of S2 and S3 trains 450, the width-matched `mt_final_static_w.yaml` and the
k = 1 control `control_duffing_k1.yaml` included (#78, #79; a test pins each
width and control YAML to the epochs of its task's final YAML), so their
mixed-forcing ramp (`AlphaSchedule`, `epochs // 2`) spans 225 epochs. The S0
smokes, S1 pilots and canaries keep their epochs. Before a final stage,
set `hpo_dir` to the fetched HPO run. For lv the manifest adds the multi-anchor
endpoint (`--anchors 8,16,24,32 --horizon 96`) and the system's `--ic-spread`;
arms without HPO name their reference under `config_from`; Person Activity runs
the solver arms on ncp one per GPU (`heavy.arms`). `--count` gives the planned
fits without any `hpo_selection.json` (an hpo line counts 11 fits, co 6):

```bash
uv run python experiments/suite_manifest.py --count configs/suite/duffing_*.yaml   # 519
```

### S0 op-determinism probe (#70)

`./cluster/submit_determinism_probe.sh [-n] <label>` runs
`experiments/determinism_probe.py` on one GPU into `00_canary`: cfc_lrc/ncp on
Lotka-Volterra (300 epochs, 4k) and Person Activity (50 epochs, 30k), three fits
each (`--deterministic` on, on, off), one at a time. It prints the runtime ratio
on/off and the |Delta| of the validation selection metrics of the two "on" fits
(must stay within 1e-3; exit 1 otherwise) and writes
`results/determinism_probe.json`. Extra runner flags via `DP_LV_ARGS` /
`DP_PA_ARGS` (e.g. `DP_LV_ARGS="--forcing mixed"`). Ratio at most 1.3 on both
tasks: put `--deterministic` into the suite YAML's `extra_args`; every result
records the setting as `config.effective.deterministic`.

The first S0 run measured a ratio on/off of 0.997 (Lotka-Volterra) and 1.03
(Person Activity) with |Delta| 0 on the "on" pair, so the suite campaigns run
with op-determinism on (#76): every `configs/suite/*.yaml` except the S0 smokes
carries `--deterministic` in `extra_args`, the committee runner included
(`run_grid_hpo.py` forwards the flag to the task runner). The smoke YAMLs
(`smoke_*.yaml`) stay unchanged as the record of the S0 run.

### S0 smoke (#72)

Stage S0 of the frozen protocol smokes every arm-task once before any HPO fit
is spent: `configs/suite/smoke_{lv,duffing,mt,pa}.yaml` (93 arm-tasks) and the
consensus smoke `smoke_co.yaml` (the nine K = 3 arms) run each arm of the
frozen `*_hpo.yaml` (same budget, `ode_unfolds`, `dt_scale`, forcing) for 3
epochs as one `run_grid_hpo.py` fit: tuning seed 100 only (stage 1), one lr
(1e-3), clip off. Submit order, all into `00_canary`:

```bash
uv run python experiments/suite_manifest.py --count configs/suite/smoke_{lv,duffing,mt,pa}.yaml   # 93
BENCH=00_canary ./cluster/submit_suite.sh configs/suite/smoke_lv.yaml s0-smoke-lv
BENCH=00_canary ./cluster/submit_suite.sh configs/suite/smoke_duffing.yaml s0-smoke-duffing
BENCH=00_canary ./cluster/submit_suite.sh configs/suite/smoke_mt.yaml s0-smoke-mt
BENCH=00_canary ./cluster/submit_suite.sh configs/suite/smoke_pa.yaml s0-smoke-pa
BENCH=00_canary ./cluster/submit_suite.sh configs/suite/smoke_co.yaml s0-smoke-co
DP_LV_ARGS="--forcing mixed" ./cluster/submit_determinism_probe.sh s0-determinism
```

Then fetch the six run folders, write the floor lines of each rollout smoke
(`--floors` takes the tuning-seed draws for the floor lines, the arm marking
stays on the final seeds) and run the gate:

```bash
D=<YYYY-MM-DD>
for t in lv duffing mt pa co; do ./cluster/fetch_run.sh 00_canary ${D}_s0-smoke-$t; done
./cluster/fetch_run.sh 00_canary ${D}_s0-determinism
for t in lv duffing mt; do
    uv run python experiments/aggregate_results.py --runs none \
        --out results/00_canary/s0-gate/$t \
        --rollout-runs results/00_canary/${D}_s0-smoke-$t --floors
done
uv run python experiments/suite_gate.py --stage s0 \
    --runs results/00_canary/${D}_s0-smoke-{lv,duffing,mt,pa,co} \
    --determinism results/00_canary/${D}_s0-determinism/results/determinism_probe.json \
    --floors results/00_canary/s0-gate/{lv,duffing,mt}/floors.json \
    --out results/00_canary/s0-gate/REPORT.md
```

`suite_gate.py` reads every `results/hpo/<arm>/lr*_clip*_seed*/*.json` below
the run folders and checks the S0 go criteria: every arm-task of the four
smoke YAMLs present with one fit; `config.effective` with `ode_unfolds`,
`dt_scale`, `forcing` and `anchors` (rollouts) and one `protocol_version`, and
a `weights_path` (a top-level field of the result JSON, not a
`config.effective` key);
the rollout keys (`vpt_multi_median`, `vpt_anchor_{8,16,24,32}`,
`nrmse_32_anchor_8`, outcome rates, `non_finite` false) or the finite primary
metric (Person Activity, committee); no NaN/Inf (the `inv_drift_*` fields of
`multitimescale_lib`, and a NaN `rollout_metrics_{val,test}.inv_drift_*` of a
block with `n_inv_drift_undefined > 0`, i.e. every trajectory undefined, are
listed, not failed); `budget_above_tolerance` only on
gru/dense3 and lstm/dense3; `elastance_type` asymmetric on the cells with an
elastance in the registry (lrc, mm_lrc, cfc_lrc, cfc_lrc_outer, cfc_mm_lrc;
mm_ltc and cfc_mm_ltc have none);
`ode_unfolds` 6 on the solver cells; the determinism probe (|Delta| of the "on"
pair at most 1e-3 on lv and pa; a runtime ratio on/off above 1.3 is reported
as DECISION "op-determinism off for the suite", a missing or unreadable probe
file fails as "not measured") and a floor line per system and data seed. The
report lists every check as PASS, FAIL or DECISION with the offending arms;
exit 0 only when no check fails. A failed S0 is fixed and repeated.

### S1 pilot (#75)

Stage S1 of the frozen protocol pilots the floors, the epoch counts and the
consensus protocol with a fixed learning rate before any HPO fit: six campaign
YAMLs, 61 fits. The finals use `fixed: {lr: 0.001, clipnorm: null}` instead of
`hpo_dir` (lr 1e-3, clip off, no `hpo_selection.json`) with the budget,
`ode_unfolds`, `dt_scale` and forcing of the frozen `*_hpo.yaml`:
`pilot_lv.yaml` and `pilot_duffing.yaml` (cfc_lrc/ncp, gru/ncp, gru/dense3,
cfc/ncp, lrc/ncp; seeds 0-2, 300 epochs, 15 each), `pilot_mt.yaml` (cfc/ncp,
cfc_static/ncp, 6), `pilot_pa.yaml` (cfc_lrc/ncp, lrc/ncp, 50 epochs, 6;
lrc/ncp alone on its GPU, so its `wallclock_s` is the single-fit wall time)
and the dt_scale control `pilot_lv_dt1.yaml` (cfc_lrc/ncp at `dt_scale: 1`, 3,
its own run folder). `pilot_co.yaml` is stage 1 of the K = 3 consensus HPO for
ncp_cons, ncp_lam0, avg_in and aug_mono (tuning seed 100, the default lr grid,
clip off, 16). Submit order (the longest chains first):

```bash
uv run python experiments/suite_manifest.py --count configs/suite/pilot_{lv,duffing,mt,pa,lv_dt1,co}.yaml   # 61
./cluster/submit_suite.sh --wandb configs/suite/pilot_pa.yaml s1-pilot-pa              # 03_person_activity
./cluster/submit_suite.sh --wandb configs/suite/pilot_co.yaml s1-pilot-co              # 04_column_voting
./cluster/submit_suite.sh --wandb configs/suite/pilot_lv.yaml s1-pilot-lv              # 01_lotka_volterra
./cluster/submit_suite.sh --wandb configs/suite/pilot_duffing.yaml s1-pilot-duffing    # 02_duffing
./cluster/submit_suite.sh --wandb configs/suite/pilot_mt.yaml s1-pilot-mt              # 01_lotka_volterra
./cluster/submit_suite.sh --wandb configs/suite/pilot_lv_dt1.yaml s1-pilot-lv_dt1      # 01_lotka_volterra
```

Then fetch the six run folders, write the floor lines of the three rollout
pilots and run the gate:

```bash
D=<YYYY-MM-DD>
./cluster/fetch_run.sh 03_person_activity ${D}_s1-pilot-pa
./cluster/fetch_run.sh 04_column_voting ${D}_s1-pilot-co
./cluster/fetch_run.sh 02_duffing ${D}_s1-pilot-duffing
for t in lv mt lv_dt1; do ./cluster/fetch_run.sh 01_lotka_volterra ${D}_s1-pilot-$t; done
for t in lv mt; do
    uv run python experiments/aggregate_results.py --runs none --out results/s1-gate/$t \
        --rollout-runs results/01_lotka_volterra/${D}_s1-pilot-$t --floors
done
uv run python experiments/aggregate_results.py --runs none --out results/s1-gate/duffing \
    --rollout-runs results/02_duffing/${D}_s1-pilot-duffing --floors
uv run python experiments/suite_gate.py --stage s1 \
    --runs results/01_lotka_volterra/${D}_s1-pilot-{lv,mt,lv_dt1} \
           results/02_duffing/${D}_s1-pilot-duffing \
           results/03_person_activity/${D}_s1-pilot-pa \
           results/04_column_voting/${D}_s1-pilot-co \
    --floors results/s1-gate/{lv,duffing,mt}/floors.json \
    --out results/s1-gate/REPORT.md
```

`suite_gate.py --stage s1` reads every fit JSON below the run folders (a
rollout fit at `dt_scale` 1 is the control), checks that all 61 fits and a
floor line per rollout data seed are present and that every fit ran with
op-determinism (`config.effective.deterministic` true), and evaluates the S1
go criteria with the frozen thresholds (test split for the rollouts):

| Gate | Criterion | On FAIL (source) |
|------|-----------|------------------|
| config | `config.effective.deterministic` true on every pilot fit (#76); the offending fits are listed | operational rule (#76): rerun them with `--deterministic` (the pilot YAML's `extra_args`) and rerun the gate |
| G1 lv | `vpt_multi_median` of cfc_lrc/ncp > max(persistence, AR(1)) of its data seed + 2 steps in >= 2 of 3 seeds | FAIL: stop, find the cause (endpoint, floors, dt_scale) before S2; H-c and RQ1 are affected (operational rule, F1 section 10.5) |
| G1 duffing | as G1 lv | DECISION (exit 0): Duffing stays in the plan, H-b/H-c Duffing contrasts floor-bound, H-c conjunction over LV and PA (F1 section 7); FAIL only when a cfc_lrc/ncp seed is missing, has a non-finite VPT or no floor line (operational rule, #77) |
| G1 multitimescale_lib | cfc/ncp >= 2 x persistence in >= 2 of 3 seeds, `rate_unbounded` 0 on every cfc/ncp fit | spread 0.25 or 256 steps, repeat once (F1 section 7) |
| G0-E per task | an (arm, seed) qualifies when the mean of the last tenth of `val_loss_curve` (lower is better) or `val_macro_f1_curve` (Person Activity, higher is better) beats its value at 2/3 of the epochs by > 5 % relative; `ceil(2n/3)` of the task's n pilot arms (LV, Duffing 4 of 5; MT, PA 2 of 2; the dt_scale control not counted) with >= 2 qualifying seeds give the DECISION "epochs x1.5 (300 to 450)" or "epochs x2 (50 to 100)", else "epochs unchanged"; the summary reads "k of n arms qualify, needs t" | a fit without the curve fails the row (operational rule, #75) |
| G2 pa | median test macro-F1 of cfc_lrc/ncp >= 0.60, every lrc/ncp fit finite (`training_diverged` false); `wallclock_s` of both arms listed | stop, find the cause before S2 (F1 section 7) |
| G3 lv | descriptive: median VPT of the dt_scale=1 control next to the main arm | never fails |
| GK co | per arm the fit with the best `val_macro_f1_mean`: validation macro-F1 at m=0 >= 0.60; test macro-F1 degrades over the fault grid: no step m -> m' with MF1(m') - MF1(m) above one paired bootstrap SE (10 000 resamples of the test recordings in `test_confusion_by_recording_by_fault`, fixed seed, macro-F1 of the summed confusions) and MF1(m_max) < MF1(0); without the confusions the strict check (`test_macro_f1_by_fault` non-increasing), said so in the detail line; ncp_cons `vote_lambda_final` in (0.05, 0.95) | ncp_cons below 0.60 at m=0: architecture check (width, backbone) before the consensus study (F1 section 7); another arm below 0.60: arm check (lr grid, head) before the consensus study (F1 section 10.5); no degradation: protocol check (fault mask, noise scale) before the consensus study (F1 section 10.5); vote_init check of one fit for lambda at the edges (F1 section 7) |

The GK detail line prints the raw `test_macro_f1_by_fault` list (the strict
reading) and every step `m->m' difference (SE ...)`. The committee JSON
stores no per-batch fault-count sequence or hash, so the gate says so and
does not compare the fault schedule across the four arms (the runner draws
every batch from `corruption_seed`, `CorruptedBatches`; the fault grid is
K-specific). Every row is PASS, FAIL (with the consequence, which names its
source) or DECISION; exit 0 only when no row fails, a DECISION does not fail
the gate. The readings of G0-E, G1 and GK follow protocol freeze
F1-2026-10-26, section 10.5 (#77).
