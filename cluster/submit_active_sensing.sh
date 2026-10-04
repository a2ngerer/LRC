#!/usr/bin/env bash
# Sync code to the ISOLATED cluster repo ~/thesis-person and submit the tbt_cNCP
# active-sensing benchmark as a SINGLE SLURM job (cluster/active_sensing.sbatch).
# Reuses ~/thesis-person (its uv venv). No external dataset (objects generated
# in-memory), so no data rsync / setup needed.
#
#   ./cluster/submit_active_sensing.sh
#   AS_WIRINGS="tbt_cncp tbt_cncp_noloc" AS_SEEDS="0" ./cluster/submit_active_sensing.sh
#
# Prereqs: TU Wien VPN + ~/.ssh/config "datalab" (cluster/config.sh) and a
# one-time ~/thesis-person venv (./cluster/submit_person_activity.sh setup).
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/config.sh

REPO_DIR="thesis-person"

_rsync() {
    echo "rsync $(pwd) -> ${CLUSTER_HOST}:${REPO_DIR}/"
    rsync -avz --delete \
        --exclude='.git/' --exclude='.venv/' --exclude='results/' \
        --exclude='outputs/' --exclude='__pycache__/' --exclude='*.pyc' \
        --exclude='.DS_Store' --exclude='.pytest_cache/' --exclude='datasets/' \
        --exclude='papers/' \
        ./ "${CLUSTER_HOST}:${REPO_DIR}/"
}

AS_ENV=""
for v in AS_EPOCHS AS_CELLS AS_WIRINGS AS_SEEDS AS_MAXPAR; do
    if [[ -n "${!v:-}" ]]; then AS_ENV+="${v}='${!v}' "; fi
done

_rsync

# shellcheck disable=SC2029
ssh "${CLUSTER_HOST}" "cd ${REPO_DIR} && mkdir -p outputs/slurm results/active_sensing && \
export PATH=\$HOME/.local/bin:\$PATH && \
${AS_ENV}sbatch --export=ALL cluster/active_sensing.sbatch"

echo
echo "Monitor:  ssh ${CLUSTER_HOST} squeue -u \\\$USER"
echo "Logs:     ssh ${CLUSTER_HOST} tail -f ${REPO_DIR}/outputs/slurm/tbt-as-<jobid>.out"
echo "Fetch:    ./cluster/fetch_active_sensing.sh"
