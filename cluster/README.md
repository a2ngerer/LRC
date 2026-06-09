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
