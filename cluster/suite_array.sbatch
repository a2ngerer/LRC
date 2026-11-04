#!/usr/bin/env bash
# SLURM array task of the final suite (#52): one task = a bundle of
# RUNS_PER_GPU manifest lines sharing one GPU (cc_launch_bundle), each line one
# run_grid_hpo.py arm (stage hpo) or one runner fit (stage final), see
# experiments/suite_manifest.py. A line whose result JSON / hpo_selection.json
# exists is skipped, so a requeue or a re-submit only runs what is missing.
# Submitted by cluster/submit_suite.sh, which sets --array, --mem, --time and
# --cpus-per-task per group (default / heavy).
#
# Required exports: RUN_DIR (cluster/_run.sh), SUITE_GROUP (reads
# $RUN_DIR/manifest_<group>.jsonl), N_SPECS (lines in it), RUNS_PER_GPU;
# SUITE_CAMPAIGN and SUITE_STAGE pass through to the runners (wandb metadata).
#
#SBATCH --job-name=suite
#SBATCH --partition=GPU-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
# --output/--error are passed on the sbatch command line by cluster/_run.sh
# (<run folder>/logs/).
#SBATCH --open-mode=append
#SBATCH --requeue
set -euo pipefail
# Run folder (cluster/_run.sh): never write under the code tree (rsync --delete).
RUN_DIR="${RUN_DIR:?RUN_DIR not set: submit via cluster/submit_suite.sh}"
SUITE_GROUP="${SUITE_GROUP:?SUITE_GROUP not set: submit via cluster/submit_suite.sh}"
mkdir -p "$RUN_DIR/results" "$RUN_DIR/logs" "$RUN_DIR/wandb"

source cluster/_common.sh
cc_setup_tmpdir
cc_setup_cuda_ld
cc_setup_tf_env "${RUNS_PER_GPU:?RUNS_PER_GPU not set}"
# A --wandb (submit_suite.sh --wandb or extra_args) logs offline into the run folder.
export WANDB_MODE=offline WANDB_DIR="$RUN_DIR/wandb" WANDB_SILENT=true
# wandb's service start has a 30 s default wait that a loaded node can exceed (#84).
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"
# Per-fit logs: <run folder>/logs/run-<array job>_idx<line>.log
CC_LOG_DIR="$RUN_DIR/logs"

cc_launch_bundle uv run python experiments/suite_manifest.py \
    --run "$RUN_DIR/manifest_${SUITE_GROUP}.jsonl" --results "$RUN_DIR/results"
