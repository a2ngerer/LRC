#!/usr/bin/env bash
# One-off backfill for the OOM/incomplete runs of the v1 (results/runs) and
# v2 (results/runs_v2) sweeps. Unlike benchmark_array.sbatch this uses an
# EXPLICIT command list (not --index): the missing runs are scattered across
# two profiles (v1 default + v2) whose index spaces differ, so a contiguous
# --index block cannot target them. All missing runs are the heaviest cells
# (ltc / mm_ltc on ncp) that OOMed at 12 runs/GPU; this packs fewer per GPU.
#
# Submit with (RUNS_PER_GPU=6 => ceil(13/6)=3 array tasks):
#   ssh datalab "cd thesis-benchmark && mkdir -p outputs/slurm && \
#     sbatch --array=0-2%8 --export=ALL,RUNS_PER_GPU=6 cluster/backfill_array.sbatch"
#
#SBATCH --job-name=thesis-backfill
#SBATCH --partition=GPU-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm/%x-%A_%a.out
#SBATCH --error=outputs/slurm/%x-%A_%a.err
#SBATCH --open-mode=append
#SBATCH --requeue
set -euo pipefail

# Virtual nodes have a non-writable /tmp — redirect temp paths into $HOME.
export TMPDIR="$HOME/tmp"
export TEST_TMPDIR="$HOME/tmp"
export TMP="$HOME/tmp"
export TEMP="$HOME/tmp"
mkdir -p "$TMPDIR"

export PATH="$HOME/.local/bin:$PATH"

# TF 2.15 needs LD_LIBRARY_PATH to find the pip-installed CUDA libs.
VENV_SITE="$(uv run python -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)"
if [[ -d "${VENV_SITE}/nvidia" ]]; then
    CUDA_LD="$(find "${VENV_SITE}/nvidia" -maxdepth 2 -name 'lib' -type d 2>/dev/null | tr '\n' ':')"
    export LD_LIBRARY_PATH="${CUDA_LD}${LD_LIBRARY_PATH:-}"
fi

export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=1

# Fewer runs/GPU than the main sweep (was 12 -> OOM); 6 gives ~10 GB/run at 64 GB.
RUNS_PER_GPU="${RUNS_PER_GPU:-6}"
CPUS_TOTAL="${SLURM_CPUS_PER_TASK:-12}"
THREADS_PER_RUN=$(( CPUS_TOTAL / RUNS_PER_GPU ))
[[ "${THREADS_PER_RUN}" -lt 1 ]] && THREADS_PER_RUN=1
export OMP_NUM_THREADS="${THREADS_PER_RUN}"
export TF_NUM_INTEROP_THREADS=1
export TF_NUM_INTRAOP_THREADS="${THREADS_PER_RUN}"

mkdir -p outputs/slurm results/runs results/runs_v2

# Explicit arg-strings for every missing run. v1 entries (no --profile) write to
# results/runs; the single v2 entry (--profile v2) writes to results/runs_v2.
# clip_norm defaults to 0 => filenames carry no clip suffix (matches the gaps).
RUNS=(
  "--cell ltc --wiring ncp --system duffing --seed 3"
  "--cell ltc --wiring ncp --system duffing --seed 4"
  "--cell ltc --wiring ncp --system periodic_sinusoidal --seed 1"
  "--cell ltc --wiring ncp --system periodic_sinusoidal --seed 2"
  "--cell ltc --wiring ncp --system periodic_sinusoidal --seed 3"
  "--cell ltc --wiring ncp --system periodic_predator_prey --seed 1"
  "--cell ltc --wiring ncp --system limited_predator_prey --seed 2"
  "--cell ltc --wiring ncp --system limited_predator_prey --seed 4"
  "--cell ltc --wiring ncp --system nonlinear_predator_prey --seed 0"
  "--cell ltc --wiring ncp --system nonlinear_predator_prey --seed 1"
  "--cell ltc --wiring ncp --system nonlinear_predator_prey --seed 2"
  "--cell ltc --wiring ncp --system nonlinear_predator_prey --seed 3"
  "--profile v2 --cell mm_ltc --wiring ncp --system periodic_sinusoidal --seed 0"
)

START=$(( SLURM_ARRAY_TASK_ID * RUNS_PER_GPU ))
echo "=== backfill task ${SLURM_ARRAY_TASK_ID} on $(hostname) — $(date -u +%FT%TZ) ==="
echo "    entries ${START}..$(( START + RUNS_PER_GPU - 1 )) of ${#RUNS[@]}, ${THREADS_PER_RUN} threads/run"

pids=()
for off in $(seq 0 $(( RUNS_PER_GPU - 1 ))); do
    i=$(( START + off ))
    [[ "${i}" -ge "${#RUNS[@]}" ]] && break
    echo "    -> ${RUNS[$i]}"
    # shellcheck disable=SC2086
    uv run python experiments/run_benchmark.py ${RUNS[$i]} &
    pids+=("$!")
done

rc=0
for pid in "${pids[@]}"; do
    wait "${pid}" || rc=1
done
echo "=== done (rc=${rc}) — $(date -u +%FT%TZ) ==="
exit "${rc}"
