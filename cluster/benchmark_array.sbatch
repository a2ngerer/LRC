#!/usr/bin/env bash
# SLURM array job: one task = one benchmark run (cell x wiring x system x seed).
# Submitted by cluster/submit_benchmark.sh; the array index maps onto
# `experiments/run_benchmark.py --index` (see --list for the mapping).
#
#SBATCH --job-name=thesis-benchmark
#SBATCH --partition=GPU-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm/%x-%A_%a.out
#SBATCH --error=outputs/slurm/%x-%A_%a.err
#SBATCH --open-mode=append
#SBATCH --requeue
set -euo pipefail

# Virtual nodes (hostnames i-*) have a non-writable /tmp — redirect all
# temp paths into $HOME (same workaround as arc-jax-rl train.sbatch).
export TMPDIR="$HOME/tmp"
export TEST_TMPDIR="$HOME/tmp"
export TMP="$HOME/tmp"
export TEMP="$HOME/tmp"
mkdir -p "$TMPDIR"

export PATH="$HOME/.local/bin:$PATH"

mkdir -p outputs/slurm results/runs

# GPU utilization log (CSV, 30s polling) next to the SLURM logs.
GPU_IDX="${CUDA_VISIBLE_DEVICES:-0}"
GPU_LOG="outputs/slurm/${SLURM_JOB_NAME:-thesis-benchmark}-${SLURM_JOB_ID}.gpu.csv"
nohup nvidia-smi -i "${GPU_IDX}" \
  --query-gpu=index,timestamp,utilization.gpu,utilization.memory,memory.used,memory.free,memory.total \
  --format=csv -l 30 > "${GPU_LOG}" 2>&1 &

echo "=== task ${SLURM_ARRAY_TASK_ID} on $(hostname) — $(date -u +%FT%TZ) ==="
# Any extra arguments passed after the sbatch script are forwarded to the
# runner (e.g. --iters 2000 --grad-log-every 25).
uv run python experiments/run_benchmark.py --index "${SLURM_ARRAY_TASK_ID}" "$@"
echo "=== done — $(date -u +%FT%TZ) ==="
