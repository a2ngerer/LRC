#!/usr/bin/env bash
# Submit the benchmark v4 matrix (cross-family generalization + classical
# championship, 880 runs). Thin wrapper: forwards --profile v4 to
# submit_benchmark.sh, which computes the array size via
# `run_benchmark.py --profile v4 --count` (= 880) and rsyncs + submits the array.
#
# Cells (11): LTC 2x2 {ltc, cfc, mm_ltc, cfc_mm_ltc} + LRC 2x2 {lrc, cfc_lrc,
# mm_lrc, cfc_mm_lrc} + classical/CT baselines {gru, lstm, ctrnn}. Identical
# config to v3.2; only the cell architecture differs. Results -> results/runs_v4/.
#
# HEAVY-CELL DEFAULTS (overridable). v4 contains the slow numerical-LTC cells
# (ltc, mm_ltc): on the ncp wiring their fused 6-unfold ODE graph runs ~2.2-2.7 h
# (max ~7.3 h) PER RUN and OOMs at the config.sh default 12 runs/GPU @ 32G (see
# cluster/backfill_array.sbatch / backfill_v3_ncp.sbatch, which fixed the v3
# mm_ltc/ncp OOM). So this wrapper mirrors submit_benchmark_v3.sh's heavy packing
# (6 runs/GPU, 64G) and raises walltime to 10 h for the 7.3 h-max safety margin.
# v4 uses BASELINE config (no ode_unfolds=24 / batch_time=64 4x-cost arms), so
# 6/GPU @ 64G is the level at which the v3 main sweep's ltc/mm_ltc ncp baselines
# completed without OOM.
#
#   ./cluster/submit_benchmark_v4.sh
#   ./cluster/submit_benchmark_v4.sh -- --iters 4000          # extra runner args
#   CLUSTER_RUNS_PER_GPU=4 CLUSTER_MEM=96G ./cluster/submit_benchmark_v4.sh  # safer
#
# Recommended pre-flight (OOM/timing canary): submit ONE heavy run first and
# confirm it neither OOMs nor exceeds walltime before releasing the full array:
#   ssh datalab "cd thesis-benchmark && uv run python experiments/run_benchmark.py \
#       --cell mm_ltc --wiring ncp --system duffing --seed 0 --profile v4"
set -euo pipefail
cd "$(dirname "$0")/.."

# Heavy-cell defaults (overridable). At 6 runs/GPU x 8 GPUs = 48 concurrent runs.
export CLUSTER_RUNS_PER_GPU="${CLUSTER_RUNS_PER_GPU:-6}"
export CLUSTER_MEM="${CLUSTER_MEM:-64G}"
export CLUSTER_TIME="${CLUSTER_TIME:-10:00:00}"

EXTRA_ARGS=()
if [[ "${1:-}" == "--" ]]; then
    shift
    EXTRA_ARGS=("$@")
fi
exec ./cluster/submit_benchmark.sh -- --profile v4 ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
