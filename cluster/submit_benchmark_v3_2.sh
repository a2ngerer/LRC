#!/usr/bin/env bash
# Submit the benchmark v3.2 matrix (LRC 2x2 architecture ablation, 240 runs).
# Thin wrapper: forwards --profile v3.2 to submit_benchmark.sh, which computes the
# array size via `run_benchmark.py --profile v3.2 --count` (= 240) and rsyncs +
# submits the SLURM array.
#
# Cells {lrc, cfc_lrc, mm_lrc, cfc_mm_lrc} = {numerical, closed-form} x
# {plain, mixed-memory}, identical hyperparameters. The closed-form cells are cheap
# and the mixed-memory cells light (no heavy solver-fidelity / long-horizon arms,
# no OOM risk), so the default config (12 runs/GPU, 32G, 4h) is kept; every value
# stays overridable via the environment.
#
#   ./cluster/submit_benchmark_v3_2.sh
#   ./cluster/submit_benchmark_v3_2.sh -- --iters 4000      # extra runner args
#
# Results land in results/runs_v3_2/ (set by run_benchmark.py for --profile v3.2).
set -euo pipefail
cd "$(dirname "$0")/.."

EXTRA_ARGS=()
if [[ "${1:-}" == "--" ]]; then
    shift
    EXTRA_ARGS=("$@")
fi
exec ./cluster/submit_benchmark.sh -- --profile v3.2 ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
