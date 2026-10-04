#!/usr/bin/env bash
# Sync code + dataset to the ISOLATED cluster repo ~/thesis-person and submit the
# cNCP-vs-NCP Person-Activity benchmark as a SINGLE SLURM job (cluster/person_activity.sbatch).
#
# Isolated on purpose: a separate directory + its own uv venv means this never
# perturbs ~/thesis-benchmark (Neural-ODE) or ~/thesis-mujoco.
#
#   # one-time env setup (login node has internet; downloads TF-CUDA):
#   ./cluster/submit_person_activity.sh setup
#   # submit the benchmark (cfc_lrc + gru x {ncp,cncp} x 3 seeds, 50 epochs):
#   ./cluster/submit_person_activity.sh
#   # smaller/faster override via env:
#   PA_CELLS="cfc_lrc" PA_SEEDS="0 1 2" ./cluster/submit_person_activity.sh
#
# Prereqs: TU Wien VPN active + ~/.ssh/config "datalab" entry (see cluster/config.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/config.sh

REPO_DIR="thesis-person"

_rsync() {
    echo "rsync $(pwd) -> ${CLUSTER_HOST}:${REPO_DIR}/  (includes data/ for the dataset)"
    rsync -avz --delete \
        --exclude='.git/' --exclude='.venv/' --exclude='results/' \
        --exclude='outputs/' --exclude='__pycache__/' --exclude='*.pyc' \
        --exclude='.DS_Store' --exclude='.pytest_cache/' --exclude='datasets/' \
        --exclude='papers/' \
        ./ "${CLUSTER_HOST}:${REPO_DIR}/"
}

if [[ "${1:-}" == "setup" ]]; then
    _rsync
    echo "Setting up isolated venv (uv sync --extra cuda --extra tracking) on the login node ..."
    ssh "${CLUSTER_HOST}" "cd ${REPO_DIR} && export PATH=\$HOME/.local/bin:\$PATH && \
        export TMPDIR=\$HOME/tmp && mkdir -p \$HOME/tmp && \
        uv sync --extra cuda --extra tracking && \
        uv run python -c 'import tensorflow as tf; print(\"tf\", tf.__version__)'"
    echo "Setup done."
    exit 0
fi

# Forward any PA_* matrix overrides through the ssh environment to the sbatch.
PA_ENV=""
for v in PA_EPOCHS PA_CELLS PA_WIRINGS PA_SEEDS PA_MAXPAR PA_WANDB \
         PA_PARAM_BUDGET; do
    if [[ -n "${!v:-}" ]]; then PA_ENV+="${v}='${!v}' "; fi
done

_rsync

# shellcheck disable=SC2029
ssh "${CLUSTER_HOST}" "cd ${REPO_DIR} && mkdir -p outputs/slurm results/person_activity && \
export PATH=\$HOME/.local/bin:\$PATH && \
${PA_ENV}sbatch --export=ALL cluster/person_activity.sbatch"

echo
echo "Monitor:  ssh ${CLUSTER_HOST} squeue -u \\\$USER"
echo "Logs:     ssh ${CLUSTER_HOST} tail -f ${REPO_DIR}/outputs/slurm/person-cncp-<jobid>.out"
echo "Fetch:    ./cluster/fetch_person_activity.sh"
