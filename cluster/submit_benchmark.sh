#!/usr/bin/env bash
# Sync code and submit the full benchmark matrix as a SLURM array job.
#
#   ./cluster/submit_benchmark.sh                       # full 240-run matrix
#   ./cluster/submit_benchmark.sh -- --iters 4000       # forward runner args
#   CLUSTER_PARTITION=GPU-a100 CLUSTER_GPU=a100:1 ./cluster/submit_benchmark.sh
#
# The array is throttled to %8 so it never exceeds the dataLAB hard limit of
# 8 concurrent GPUs per user.
set -euo pipefail

cd "$(dirname "$0")/.."
source cluster/config.sh

RUNNER_ARGS=()
if [[ "${1:-}" == "--" ]]; then
    shift
    RUNNER_ARGS=("$@")
fi

# Number of array tasks = number of run specs (computed locally; the spec
# list is deterministic, see experiments/run_benchmark.py build_specs).
N_SPECS="$(uv run python experiments/run_benchmark.py --count "${RUNNER_ARGS[@]}" 2>/dev/null || uv run python experiments/run_benchmark.py --count)"
echo "Benchmark matrix: ${N_SPECS} runs"

./cluster/sync_to_cluster.sh

echo "Submitting array job (0-$((N_SPECS - 1))%${CLUSTER_ARRAY_THROTTLE}) on ${CLUSTER_PARTITION} ..."
# shellcheck disable=SC2029
ssh "${CLUSTER_HOST}" "cd ${CLUSTER_REPO_DIR} && mkdir -p outputs/slurm && \
sbatch \
  --partition='${CLUSTER_PARTITION}' \
  --gres=gpu:${CLUSTER_GPU} \
  --cpus-per-task='${CLUSTER_CPUS}' \
  --mem='${CLUSTER_MEM}' \
  --time='${CLUSTER_TIME}' \
  --array=0-$((N_SPECS - 1))%${CLUSTER_ARRAY_THROTTLE} \
  cluster/benchmark_array.sbatch ${RUNNER_ARGS[*]:-}"

echo
echo "Monitor:  ssh ${CLUSTER_HOST} squeue -u \\\$USER"
echo "Logs:     ssh ${CLUSTER_HOST} tail -f ${CLUSTER_REPO_DIR}/outputs/slurm/thesis-benchmark-<jobid>_<task>.out"
echo "Fetch:    ./cluster/fetch_results.sh"
