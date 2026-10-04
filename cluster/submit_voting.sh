#!/usr/bin/env bash
# Sync code to the ISOLATED cluster repo ~/thesis-person and submit the tbt_cNCP
# multi-column VOTING benchmark as a SINGLE SLURM job (cluster/voting.sbatch).
# Reuses ~/thesis-person (its uv venv). No external dataset (objects generated
# in-memory), so no data rsync / setup needed.
#
#   ./cluster/submit_voting.sh
#   VOTE_KCOLS="1 2 3 4" VOTE_SEEDS="0" ./cluster/submit_voting.sh
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

VOTE_ENV=""
for v in VOTE_EPOCHS VOTE_CELL VOTE_KCOLS VOTE_SEEDS VOTE_MODES VOTE_MAXPAR \
         VOTE_OUTDIR; do
    if [[ -n "${!v:-}" ]]; then VOTE_ENV+="${v}='${!v}' "; fi
done

_rsync

# shellcheck disable=SC2029
ssh "${CLUSTER_HOST}" "cd ${REPO_DIR} && mkdir -p outputs/slurm results/active_sensing_voting && \
export PATH=\$HOME/.local/bin:\$PATH && \
${VOTE_ENV}sbatch --export=ALL cluster/voting.sbatch"

echo
echo "Monitor:  ssh ${CLUSTER_HOST} squeue -u \\\$USER"
echo "Logs:     ssh ${CLUSTER_HOST} tail -f ${REPO_DIR}/outputs/slurm/tbt-vote-<jobid>.out"
echo "Fetch:    ./cluster/fetch_voting.sh"
