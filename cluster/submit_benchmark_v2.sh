#!/usr/bin/env bash
# Submit the benchmark v2 matrix (fixed cells + clip axis, 480 runs).
# Thin wrapper: forwards --profile v2 to submit_benchmark.sh, which
# computes the array size via `run_benchmark.py --profile v2 --count`.
#
#   ./cluster/submit_benchmark_v2.sh
#   ./cluster/submit_benchmark_v2.sh -- --iters 4000   # extra runner args
set -euo pipefail
cd "$(dirname "$0")/.."

EXTRA_ARGS=()
if [[ "${1:-}" == "--" ]]; then
    shift
    EXTRA_ARGS=("$@")
fi
exec ./cluster/submit_benchmark.sh -- --profile v2 ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
