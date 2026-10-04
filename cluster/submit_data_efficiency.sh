#!/usr/bin/env bash
# Sync code to the ISOLATED cluster repo ~/thesis-person and submit the
# data-efficiency benchmark (Iteration 8) as a SINGLE SLURM job
# (cluster/data_efficiency.sbatch). Reuses ~/thesis-person (its uv venv). The
# person-activity CSV under data/person/ is rsynced with the code.
#
#   ./cluster/submit_data_efficiency.sh
#   DE_WIRINGS="cncp dense" DE_FRACS="0.1 1.0" DE_SEEDS="0" ./cluster/submit_data_efficiency.sh
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

DE_ENV=""
for v in DE_EPOCHS DE_CELL DE_WIRINGS DE_FRACS DE_SEEDS DE_SIZE DE_SEQLEN \
         DE_BATCH DE_MAXPAR; do
    if [[ -n "${!v:-}" ]]; then DE_ENV+="${v}='${!v}' "; fi
done

_rsync

# shellcheck disable=SC2029
ssh "${CLUSTER_HOST}" "cd ${REPO_DIR} && mkdir -p outputs/slurm results/data_efficiency && \
export PATH=\$HOME/.local/bin:\$PATH && \
${DE_ENV}sbatch --export=ALL cluster/data_efficiency.sbatch"

echo
echo "Monitor:  ssh ${CLUSTER_HOST} squeue -u \\\$USER"
echo "Logs:     ssh ${CLUSTER_HOST} tail -f ${REPO_DIR}/outputs/slurm/tbt-dataeff-<jobid>.out"
echo "Fetch:    ./cluster/fetch_data_efficiency.sh"
