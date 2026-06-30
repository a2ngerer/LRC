#!/usr/bin/env bash
# Submit the benchmark v3.3 matrix (parameter-matched LRC capacity control, 60 runs).
# Thin wrapper: forwards --profile v3.3 to submit_benchmark.sh (-> count=60, rsync +
# SLURM array). lrc_pm = a plain numerical LRC widened (CELL_UNITS/CELL_NCP) to >=
# the largest v3.2 fixed cell (cfc_mm_lrc); it is compared against the existing
# runs_v3_2 cells (identical training config), so only this control is run.
# Default config kept. Results land in results/runs_v3_3/.
#
#   ./cluster/submit_benchmark_v3_3.sh
set -euo pipefail
cd "$(dirname "$0")/.."

EXTRA_ARGS=()
if [[ "${1:-}" == "--" ]]; then
    shift
    EXTRA_ARGS=("$@")
fi
exec ./cluster/submit_benchmark.sh -- --profile v3.3 ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
