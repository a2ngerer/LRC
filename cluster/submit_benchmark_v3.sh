#!/usr/bin/env bash
# Submit the benchmark v3 matrix (forward-rollout-stability probe, 720 runs).
# Thin wrapper: forwards --profile v3 to submit_benchmark.sh, which computes the
# array size via `run_benchmark.py --profile v3 --count`.
#
# v3 is dominated by the heavy ltc/mm_ltc cells, and the solver-fidelity arm runs
# ode_unfolds=24 (~4x the LTC ODE cost). mm_ltc on ncp also OOMs at the default
# 12 runs/GPU. So this wrapper lowers the GPU packing and raises memory/walltime
# headroom by default (matching the backfill_array.sbatch tuning that fixed the
# mm_ltc/ncp OOM); every value stays overridable via the environment.
#
#   ./cluster/submit_benchmark_v3.sh
#   ./cluster/submit_benchmark_v3.sh -- --iters 4000        # extra runner args
#   CLUSTER_RUNS_PER_GPU=8 ./cluster/submit_benchmark_v3.sh # override packing
set -euo pipefail
cd "$(dirname "$0")/.."

# Heavy-cell defaults (overridable). At 6 runs/GPU x 8 GPUs = 48 concurrent runs.
export CLUSTER_RUNS_PER_GPU="${CLUSTER_RUNS_PER_GPU:-6}"
export CLUSTER_MEM="${CLUSTER_MEM:-64G}"
export CLUSTER_TIME="${CLUSTER_TIME:-08:00:00}"

EXTRA_ARGS=()
if [[ "${1:-}" == "--" ]]; then
    shift
    EXTRA_ARGS=("$@")
fi
exec ./cluster/submit_benchmark.sh -- --profile v3 ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
