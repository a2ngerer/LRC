#!/usr/bin/env bash
# Sync code to the ISOLATED cluster repo ~/thesis-person and submit the
# partial-view voting-committee benchmark (Iteration 7) as a SINGLE SLURM job
# (cluster/committee.sbatch). Reuses ~/thesis-person (its uv venv). The
# person-activity CSV under data/person/ is rsynced with the code (not excluded).
#
#   ./cluster/submit_committee.sh
#   CO_WIRINGS="cncp" CO_KS="1 4" CO_SEEDS="0" ./cluster/submit_committee.sh
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

CO_ENV=""
for v in CO_EPOCHS CO_CELL CO_WIRINGS CO_KS CO_SEEDS CO_SEQLEN CO_BATCH \
         CO_VIEWMODE CO_DROPFRACS CO_TRAINDROP CO_TRAINNOISE CO_TESTCORRUPT \
         CO_NOISESIGMAS CO_PARAM_BUDGET CO_MAXPAR; do
    if [[ -n "${!v:-}" ]]; then CO_ENV+="${v}='${!v}' "; fi
done

_rsync

# shellcheck disable=SC2029
ssh "${CLUSTER_HOST}" "cd ${REPO_DIR} && mkdir -p outputs/slurm results/committee && \
export PATH=\$HOME/.local/bin:\$PATH && \
${CO_ENV}sbatch --export=ALL cluster/committee.sbatch"

echo
echo "Monitor:  ssh ${CLUSTER_HOST} squeue -u \\\$USER"
echo "Logs:     ssh ${CLUSTER_HOST} tail -f ${REPO_DIR}/outputs/slurm/tbt-committee-<jobid>.out"
echo "Fetch:    ./cluster/fetch_committee.sh"
