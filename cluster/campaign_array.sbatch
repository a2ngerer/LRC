#!/usr/bin/env bash
# SLURM array job: one task = a BUNDLE of campaign runs sharing one GPU.
# Generic sibling of benchmark_array.sbatch -- identical GPU-packing contract,
# but the run list comes from a declarative campaign config via run_campaign.py
# (--config <path>) instead of run_benchmark.py (--profile). All the runtime
# boilerplate (TMPDIR redirect, CUDA lib discovery, TF env, parallel-launch
# loop) is sourced from cluster/_common.sh so this and benchmark_array.sbatch
# cannot drift. Submitted by cluster/submit_campaign.sh.
#
# The array index maps onto a contiguous block of run specs: task t handles
#   [t*RUNS_PER_GPU, ..., t*RUNS_PER_GPU + RUNS_PER_GPU - 1]  (capped at N_SPECS),
# launched in parallel on the single allocated GPU -- byte-identical bundling to
# benchmark_array.sbatch (see cc_launch_bundle in _common.sh).
#
# Required exports (set by submit_campaign.sh):
#   CONFIG        repo-relative path to the campaign YAML.
#   N_SPECS       total run specs (so the last task does not run past the end).
#   RUNS_PER_GPU  bundle size K.
# Extra forwarded args (optional) are appended to every run, e.g. --regimes noise.
#
#SBATCH --job-name=thesis-campaign
#SBATCH --partition=GPU-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm/%x-%A_%a.out
#SBATCH --error=outputs/slurm/%x-%A_%a.err
#SBATCH --open-mode=append
#SBATCH --requeue
set -euo pipefail

source cluster/_common.sh

CONFIG="${CONFIG:?CONFIG must be exported by submit_campaign.sh}"
N_SPECS="${N_SPECS:?N_SPECS must be exported by submit_campaign.sh}"
RUNS_PER_GPU="${RUNS_PER_GPU:-6}"

cc_setup_tmpdir
cc_setup_cuda_ld
cc_setup_tf_env "${RUNS_PER_GPU}"

mkdir -p outputs/slurm results
cc_start_gpu_logger "${SLURM_JOB_NAME:-thesis-campaign}"

# Forward any extra runner args (training overrides / matrix filters) to every
# run in the bundle; --config + --index are injected by cc_launch_bundle.
RUN_EXTRA_ARGS=( "$@" )

rc=0
cc_launch_bundle uv run python experiments/run_campaign.py --config "${CONFIG}" || rc=$?
exit "${rc}"
