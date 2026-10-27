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
```

`GH_TUNING_SEEDS` (default `100 101`, disjoint from the final seeds 0..9;
`run_grid_hpo.py` refuses seeds in 0..9 without `--allow-seed-overlap`),
`GH_PARAM_BUDGET` (default 4000 lv / 30000 pa) and `GH_EXTRA` (driver flags
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
`submit_grid_hpo.sh` with `GH_TASK=pa` check the file with `ssh test -f`
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

`--wandb` adds `--wandb` to every call; the offline runs land under `<run>/wandb/`.
