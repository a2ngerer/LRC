#!/usr/bin/env bash
# Submit the eps-ablation matrix (liquid-elastance over-parameterization ablation,
# see scratchpad/eps-ablation/finalSpec.md). Thin wrapper: forwards --profile eps
# to submit_benchmark.sh, which computes the array size via
# `run_benchmark.py --profile eps --count` and rsyncs + submits the array.
#
# What eps is: 8 LRC conditions (A interp / B asym / C sym / D frozen / E pm-pad /
# E_C pm-pad+extra / F asym-hybrid / G interp-hybrid) x 2 wirings on a two-tier
# task suite. The confirmatory headline equivalence (B~A, C~A) lives on
# multitimescale @ uf=1, with uf in {2,4} as per-level robustness; spiral and
# stiff_linear_k1 are flat falsification anchors; stiff_linear_k{10,100,1000} are
# exploratory. sym (C, E_C) and hybrid (F, G) arms are PRUNED off the spiral
# anchor (run only on the informative multiscale + stiff tasks) -- enforced in
# run_benchmark.build_specs_eps, so the unpruned cross-product never materializes.
# Results -> results/runs_eps/.
#
# THROTTLE / PACKING (pinned, NOT inherited). config.sh defaults
# RUNS_PER_GPU=12; submit_benchmark_v5.sh used 6 for the heavy numerical-LTC
# cells. The eps matrix is LRC-only (no ltc/mm_ltc), but LRC carries the extra
# elastance Dense per step, so this wrapper pins CLUSTER_RUNS_PER_GPU=6
# EXPLICITLY (overridable) and the wave count must be recomputed from a measured
# per-run wall, NOT reused from v4/v5 by analogy.
#
# COMPUTE GATE (do BEFORE releasing the array): measure one real run on the
# target hardware and write the resulting wave count into the spec --
#   ssh datalab "cd thesis-benchmark && uv run python experiments/run_benchmark.py \
#       --profile eps --cell lrc_asym --wiring dense \
#       --system multitimescale --seed 0 --ode-unfolds 1"
# (repeat with --wiring ncp). Per-run cost is UNVERIFIED until measured.
#
#   ./cluster/submit_benchmark_eps.sh
#   ./cluster/submit_benchmark_eps.sh -- --tasks multitimescale --ode-unfolds 1  # headline only
#   CLUSTER_RUNS_PER_GPU=4 CLUSTER_MEM=64G ./cluster/submit_benchmark_eps.sh      # safer packing
set -euo pipefail
cd "$(dirname "$0")/.."

# Pinned packing/walltime (explicit; reconciled against config.sh default 12).
export CLUSTER_RUNS_PER_GPU="${CLUSTER_RUNS_PER_GPU:-6}"
export CLUSTER_MEM="${CLUSTER_MEM:-64G}"
export CLUSTER_TIME="${CLUSTER_TIME:-10:00:00}"

# The sym/hybrid pruning is enforced in build_specs_eps (the spec's "task list"),
# so the wrapper needs no extra task filtering -- the --count the submit script
# reads already reflects the pruned matrix. Output dir for the eps profile.
EXTRA_ARGS=()
if [[ "${1:-}" == "--" ]]; then
    shift
    EXTRA_ARGS=("$@")
fi
exec ./cluster/submit_benchmark.sh -- --profile eps --outdir results/runs_eps \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
