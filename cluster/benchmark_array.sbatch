#!/usr/bin/env bash
# SLURM array job: one task = a BUNDLE of benchmark runs sharing one GPU.
# Submitted by cluster/submit_benchmark.sh. The array index maps onto a
# contiguous block of run specs: task t handles indices
#   [t*RUNS_PER_GPU, ..., t*RUNS_PER_GPU + RUNS_PER_GPU - 1]  (capped at N_SPECS),
# launched in parallel on the single allocated GPU.
#
# Rationale: the cells are tiny RNNs (16 units, batch 16) with sequential ODE
# integration. A single run keeps the A40 at ~2% utilization — the workload is
# launch-latency/CPU bound, not compute bound. Packing several runs onto one GPU
# fills the idle capacity and multiplies throughput without changing any run's
# training dynamics. TF_FORCE_GPU_ALLOW_GROWTH stops each process from grabbing
# the whole 46 GB so they coexist.
#
#SBATCH --job-name=thesis-benchmark
#SBATCH --partition=GPU-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=32G
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

# tensorflow[and-cuda] installs CUDA libs under .venv/lib/python*/site-packages/nvidia/*/lib/.
# TF 2.15 does not auto-discover these without LD_LIBRARY_PATH on all cluster distros.
VENV_SITE="$(uv run python -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)"
if [[ -d "${VENV_SITE}/nvidia" ]]; then
    CUDA_LD="$(find "${VENV_SITE}/nvidia" -maxdepth 2 -name 'lib' -type d 2>/dev/null | tr '\n' ':')"
    export LD_LIBRARY_PATH="${CUDA_LD}${LD_LIBRARY_PATH:-}"
fi

# Let multiple runs share one GPU: each process allocates only what it needs
# (~1.5 GB) instead of pre-grabbing all of VRAM.
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=1

# How many runs to pack onto this GPU (set by submit script; default 6).
RUNS_PER_GPU="${RUNS_PER_GPU:-6}"
# Total number of run specs, so the last task does not run past the end.
N_SPECS="${N_SPECS:?N_SPECS must be exported by submit_benchmark.sh}"

# Split the allocated CPUs across the concurrent runs so the per-process TF
# thread pools do not oversubscribe the cores.
CPUS_TOTAL="${SLURM_CPUS_PER_TASK:-24}"
THREADS_PER_RUN=$(( CPUS_TOTAL / RUNS_PER_GPU ))
[[ "${THREADS_PER_RUN}" -lt 1 ]] && THREADS_PER_RUN=1
export OMP_NUM_THREADS="${THREADS_PER_RUN}"
export TF_NUM_INTEROP_THREADS=1
export TF_NUM_INTRAOP_THREADS="${THREADS_PER_RUN}"

mkdir -p outputs/slurm results/runs

# GPU utilization log (CSV, 30s polling) next to the SLURM logs.
GPU_IDX="${CUDA_VISIBLE_DEVICES:-0}"
GPU_LOG="outputs/slurm/${SLURM_JOB_NAME:-thesis-benchmark}-${SLURM_JOB_ID}.gpu.csv"
nohup nvidia-smi -i "${GPU_IDX}" \
  --query-gpu=index,timestamp,utilization.gpu,utilization.memory,memory.used,memory.free,memory.total \
  --format=csv -l 30 > "${GPU_LOG}" 2>&1 &

START=$(( SLURM_ARRAY_TASK_ID * RUNS_PER_GPU ))
echo "=== task ${SLURM_ARRAY_TASK_ID} on $(hostname) — $(date -u +%FT%TZ) ==="
echo "    indices ${START}..$(( START + RUNS_PER_GPU - 1 )) (cap ${N_SPECS}), ${THREADS_PER_RUN} threads/run"

# Launch the bundle in parallel on the shared GPU; any extra args (e.g.
# --profile v2 --iters 4000) are forwarded to every run.
pids=()
for off in $(seq 0 $(( RUNS_PER_GPU - 1 ))); do
    IDX=$(( START + off ))
    [[ "${IDX}" -ge "${N_SPECS}" ]] && break
    echo "    -> launch index ${IDX}"
    uv run python experiments/run_benchmark.py --index "${IDX}" "$@" &
    pids+=("$!")
done

# Wait for all runs; surface a non-zero exit if any of them failed.
rc=0
for pid in "${pids[@]}"; do
    wait "${pid}" || rc=1
done
echo "=== done (rc=${rc}) — $(date -u +%FT%TZ) ==="
exit "${rc}"
