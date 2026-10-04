#!/usr/bin/env bash
# Sync code to the ISOLATED cluster repo ~/thesis-person and submit the
# active-glimpse sensorimotor benchmark as a SINGLE SLURM job
# (cluster/active_glimpse.sbatch). Reuses ~/thesis-person (its uv venv). No
# external dataset (objects generated in-memory), so no data rsync / setup.
#
#   ./cluster/submit_active_glimpse.sh
#   AG_WIRINGS="tbt_cncp" AG_SEEDS="0" ./cluster/submit_active_glimpse.sh
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

AG_ENV=""
for v in AG_EPOCHS AG_CELL AG_WIRINGS AG_POLICIES AG_SEEDS AG_NCLASSES \
         AG_SEQLEN AG_EXPLORE AG_MAXPAR; do
    if [[ -n "${!v:-}" ]]; then AG_ENV+="${v}='${!v}' "; fi
done

_rsync

# shellcheck disable=SC2029
ssh "${CLUSTER_HOST}" "cd ${REPO_DIR} && mkdir -p outputs/slurm results/active_glimpse && \
export PATH=\$HOME/.local/bin:\$PATH && \
${AG_ENV}sbatch --export=ALL cluster/active_glimpse.sbatch"

echo
echo "Monitor:  ssh ${CLUSTER_HOST} squeue -u \\\$USER"
echo "Logs:     ssh ${CLUSTER_HOST} tail -f ${REPO_DIR}/outputs/slurm/tbt-aglimpse-<jobid>.out"
echo "Fetch:    ./cluster/fetch_active_glimpse.sh"
