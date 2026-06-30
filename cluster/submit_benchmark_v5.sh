#!/usr/bin/env bash
# Submit the benchmark v5 matrix (generalization stress test, 1980 runs). Thin
# wrapper: forwards --profile v5 to submit_benchmark.sh, which computes the array
# size via `run_benchmark.py --profile v5 --count` (= 1980) and rsyncs + submits
# the array.
#
# What v5 is: the missing "robustness rail" the roadmap / deep-research review
# flagged (Noise / Sampling-irregularity / Domain-shift). It reruns the FULL v4
# cell set (11 cells) under three INDEPENDENT generalization stressors, decoupling
# the train and eval trajectories so any robustness gap between cells surfaces:
#   noise         -- train on observation-noised targets, eval vs the clean truth
#   extrapolation -- train on the first half, eval the rollout over the full horizon
#   ood_init      -- train from the canonical y0, eval from a perturbed y0'
# Cells (11) = LTC 2x2 {ltc, cfc, mm_ltc, cfc_mm_ltc} + LRC 2x2 {lrc, cfc_lrc,
# mm_lrc, cfc_mm_lrc} + classical/CT baselines {gru, lstm, ctrnn}. Identical
# training config to v4; the only change is the stressed data. The clean baseline
# is v4 itself (results/runs_v4), so v5-vs-v4 is a paired comparison.
# Results -> results/runs_v5/.
#
# HEAVY-CELL DEFAULTS (overridable). v5 contains the same slow numerical-LTC cells
# (ltc, mm_ltc) as v4: on the ncp wiring their fused 6-unfold ODE graph runs
# ~2.2-2.7 h (max ~7.3 h) PER RUN and OOMs at the config.sh default 12 runs/GPU
# @ 32G. The stress regimes do NOT change per-run cost (noise/ood are full-horizon
# like v4; extrapolation trains on half the grid but n_iters/batch_time are
# unchanged, so cost is the same). So this wrapper mirrors submit_benchmark_v4.sh:
# 6 runs/GPU, 64G, 10 h walltime. With 1980 runs that is ceil(1980/6)=330 array
# tasks; the %8 GPU throttle => up to 48 concurrent runs, so expect a longer
# wall-clock than v4 (more tasks, same per-task cost).
#
#   ./cluster/submit_benchmark_v5.sh
#   ./cluster/submit_benchmark_v5.sh -- --regimes noise        # one regime only (660 runs)
#   CLUSTER_RUNS_PER_GPU=4 CLUSTER_MEM=96G ./cluster/submit_benchmark_v5.sh  # safer
#
# Recommended pre-flight (OOM/timing canary): submit ONE heavy run first and
# confirm it neither OOMs nor exceeds walltime before releasing the full array:
#   ssh datalab "cd thesis-benchmark && uv run python experiments/run_benchmark.py \
#       --cell mm_ltc --wiring ncp --system duffing --seed 0 --stress ood_init --profile v5"
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
exec ./cluster/submit_benchmark.sh -- --profile v5 ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
