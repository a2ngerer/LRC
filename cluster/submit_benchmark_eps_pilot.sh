#!/usr/bin/env bash
# Submit the eps §3.0 PILOT (the hard go/no-go gate before the full eps matrix).
# Thin wrapper: forwards --profile eps_pilot to submit_benchmark.sh, which
# computes the array size via `run_benchmark.py --profile eps_pilot --count`
# (= 640) and rsyncs + submits the array.
#
# What the pilot is: ALL 8 LRC conditions (A interp / B asym / C sym / D frozen /
# E pm-pad / E_C pm-pad+extra / F asym-hybrid / G interp-hybrid) x 2 wirings on
# the pilot task subset -- multitimescale @ uf in {1,2,4} + the two flat anchors
# (spiral, stiff_linear_k1) @ uf=1 -- at seeds 0..7 (8 seeds), data jitter ON,
# the REAL n_iters=2000 so the paired-diff SD reflects converged runs. Unlike the
# full eps matrix, NO sym/hybrid pruning: the SD pilot needs every condition on
# every task to estimate all the paired-diff SDs that feed Delta_min and the
# corrected Wilcoxon-TOST power precompute (spec §3.0). The exploratory
# stiff_linear_k{10,100,1000} sweep is NOT in the pilot.
# Results -> results/runs_eps_pilot/.
#
# COUNT / WAVES: 640 runs => ceil(640/6)=107 array tasks at the pinned
# CLUSTER_RUNS_PER_GPU=6, throttle %8 => up to 48 concurrent runs.
#
# THROTTLE / PACKING (pinned, NOT inherited from config.sh default 12). Pinned to
# 6 to match the eps wrapper; the per-run wall is unverified until measured, so
# recompute the wave-time from a real one-run measurement before reasoning about
# wall-clock (see the MEASURE command in the eps run plan).
#
# After the pilot finishes, run the §3.0 analysis locally:
#   uv run python -m experiments.eps_analysis --pilot --runs results/runs_eps_pilot
#
#   ./cluster/submit_benchmark_eps_pilot.sh
#   CLUSTER_RUNS_PER_GPU=4 CLUSTER_MEM=64G ./cluster/submit_benchmark_eps_pilot.sh  # safer
set -euo pipefail
cd "$(dirname "$0")/.."

# Pinned packing/walltime (explicit; reconciled against config.sh default 12).
export CLUSTER_RUNS_PER_GPU="${CLUSTER_RUNS_PER_GPU:-6}"
export CLUSTER_MEM="${CLUSTER_MEM:-64G}"
export CLUSTER_TIME="${CLUSTER_TIME:-10:00:00}"

# The pilot subset (all 8 conditions, no pruning, seeds 0..7, the pilot tasks) is
# encoded in build_specs_eps_pilot, so --count already reflects it; no extra task
# filtering is needed here. n_iters defaults to 2000 (DEFAULTS) -> converged SDs.
EXTRA_ARGS=()
if [[ "${1:-}" == "--" ]]; then
    shift
    EXTRA_ARGS=("$@")
fi
exec ./cluster/submit_benchmark.sh -- --profile eps_pilot --outdir results/runs_eps_pilot \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
