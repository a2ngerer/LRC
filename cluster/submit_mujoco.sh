#!/usr/bin/env bash
# Sync code to the ISOLATED cluster repo ~/thesis-mujoco and submit the MuJoCo PPO
# benchmark as a SLURM array (one array task = one (model, seed) run).
#
# Isolated on purpose: a separate directory + its own uv venv means this never
# perturbs the ~/thesis-benchmark environment used by the running Neural-ODE jobs.
#
#   # one-time env setup (login node has internet; ~GB TF-CUDA + gymnasium):
#   ./cluster/submit_mujoco.sh setup
#   # submit the full matrix (7 models x 3 seeds = 21 runs) at 1.5M steps each:
#   ./cluster/submit_mujoco.sh -- --profile v1 --total-steps 1500000
#
# The array is throttled to %8 (dataLAB hard limit of 8 concurrent GPUs/user).
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/config.sh

REPO_DIR="thesis-mujoco"
PARTITION="${CLUSTER_PARTITION:-GPU-a40}"
GPU="${CLUSTER_GPU:-a40:1}"
THROTTLE="${CLUSTER_ARRAY_THROTTLE:-8}"
TIME="${CLUSTER_TIME:-03:00:00}"
MEM="${CLUSTER_MEM:-16G}"
CPUS="${CLUSTER_CPUS:-8}"

_rsync() {
    echo "rsync $(pwd) -> ${CLUSTER_HOST}:${REPO_DIR}/"
    rsync -avz --delete \
        --exclude='.git/' --exclude='.venv/' --exclude='results/' \
        --exclude='outputs/' --exclude='__pycache__/' --exclude='*.pyc' \
        --exclude='.DS_Store' --exclude='.pytest_cache/' --exclude='datasets/' \
        --exclude='papers/' \
        ./ "${CLUSTER_HOST}:${REPO_DIR}/"
}

if [[ "${1:-}" == "setup" ]]; then
    _rsync
    echo "Setting up isolated venv (uv sync --extra cuda --extra rl) on the login node ..."
    ssh "${CLUSTER_HOST}" "cd ${REPO_DIR} && export PATH=\$HOME/.local/bin:\$PATH && \
        export TMPDIR=\$HOME/tmp && mkdir -p \$HOME/tmp && \
        uv sync --extra cuda --extra rl && \
        uv run python -c 'import tensorflow as tf, gymnasium, mujoco; print(\"tf\", tf.__version__, \"gym\", gymnasium.__version__, \"mujoco\", mujoco.__version__)'"
    echo "Setup done."
    exit 0
fi

RUNNER_ARGS=()
if [[ "${1:-}" == "--" ]]; then
    shift
    RUNNER_ARGS=("$@")
fi

N_SPECS="$(uv run python experiments/run_mujoco_benchmark.py --count ${RUNNER_ARGS[@]+"${RUNNER_ARGS[@]}"} 2>/dev/null)"
echo "MuJoCo matrix: ${N_SPECS} runs -> array 0-$((N_SPECS - 1))%${THROTTLE} on ${PARTITION}"

_rsync

# shellcheck disable=SC2029
ssh "${CLUSTER_HOST}" "cd ${REPO_DIR} && mkdir -p outputs/slurm results/mujoco && \
export PATH=\$HOME/.local/bin:\$PATH && \
sbatch \
  --partition='${PARTITION}' \
  --gres=gpu:${GPU} \
  --cpus-per-task='${CPUS}' \
  --mem='${MEM}' \
  --time='${TIME}' \
  --array=0-$((N_SPECS - 1))%${THROTTLE} \
  cluster/mujoco_array.sbatch ${RUNNER_ARGS[*]:-}"

echo
echo "Monitor:  ssh ${CLUSTER_HOST} squeue -u \\\$USER"
echo "Logs:     ssh ${CLUSTER_HOST} tail -f ${REPO_DIR}/outputs/slurm/mujoco-rl-<jobid>_<task>.out"
echo "Fetch:    ./cluster/fetch_mujoco.sh"
