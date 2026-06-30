#!/usr/bin/env bash
# Submit the benchmark v3.1 matrix (closed-form LRC ablation, 240 runs).
# Thin wrapper: forwards --profile v3.1 to submit_benchmark.sh, which computes the
# array size via `run_benchmark.py --profile v3.1 --count` (= 240) and rsyncs +
# submits the SLURM array.
#
# Unlike v3, v3.1 is light: the cells are closed-form (cfc, cfc_lrc, cfc_pm) plus
# the numerical lrc at ode_unfolds=1 -- no heavy solver-fidelity / long-horizon
# arms and no OOM risk. So the default config (12 runs/GPU, 32G, 4h) is left as-is;
# every value stays overridable via the environment.
#
#   ./cluster/submit_benchmark_v3_1.sh
#   ./cluster/submit_benchmark_v3_1.sh -- --iters 4000      # extra runner args
#   CLUSTER_RUNS_PER_GPU=8 ./cluster/submit_benchmark_v3_1.sh
#
# Results land in results/runs_v3_1/ (set by run_benchmark.py for --profile v3.1).
set -euo pipefail
cd "$(dirname "$0")/.."

EXTRA_ARGS=()
if [[ "${1:-}" == "--" ]]; then
    shift
    EXTRA_ARGS=("$@")
fi
exec ./cluster/submit_benchmark.sh -- --profile v3.1 ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
