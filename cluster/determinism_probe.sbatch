#!/usr/bin/env bash
# S0 op-determinism probe (#70): experiments/determinism_probe.py runs
# cfc_lrc/ncp on Lotka-Volterra and Person Activity three times each (on, on,
# off), one fit at a time on one GPU so the runtime ratio is not skewed by
# sharing. Submitted by cluster/submit_determinism_probe.sh.
#
#SBATCH --job-name=s0-determinism
#SBATCH --partition=GPU-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=16G
#SBATCH --time=08:00:00
# --output/--error are passed on the sbatch command line by cluster/_run.sh
# (<run folder>/logs/); RUN_DIR is exported by the launcher.
#SBATCH --open-mode=append
#SBATCH --requeue
set -euo pipefail
# Run folder (cluster/_run.sh): never write under the code tree (rsync --delete).
RUN_DIR="${RUN_DIR:?RUN_DIR not set: submit via cluster/submit_determinism_probe.sh}"
mkdir -p "$RUN_DIR/results"

source cluster/_common.sh
cc_setup_tmpdir
cc_setup_cuda_ld
cc_setup_tf_env 1

# Finished fits are reused (a requeue only runs what is missing).
uv run python experiments/determinism_probe.py --outdir "$RUN_DIR/results" \
    --lv-args "${DP_LV_ARGS:-}" --pa-args "${DP_PA_ARGS:-}"
