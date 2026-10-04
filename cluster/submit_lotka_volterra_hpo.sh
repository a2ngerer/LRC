#!/usr/bin/env bash
# Sync code to the ISOLATED cluster repo ~/thesis-person and submit the
# parameter-matched Lotka-Volterra Optuna+wandb HPO sweep as a SINGLE SLURM job
# (cluster/lotka_volterra_hpo.sbatch).
#
# Reuses ~/thesis-person (its own uv venv). The venv must have the 'tracking'
# (wandb) and 'hpo' (optuna) extras -- run once, if missing:
#   ssh datalab 'cd thesis-person && uv sync --extra cuda --extra tracking --extra hpo'
#
#   # submit the default matrix ({cfc_lrc,ltc,gru} x {dense,ncp,cncp}, 30 trials):
#   ./cluster/submit_lotka_volterra_hpo.sh
#   # smaller/faster override via env:
#   HPO_CELLS="cfc_lrc gru" HPO_TRIALS=15 ./cluster/submit_lotka_volterra_hpo.sh
#
# Prereqs: TU Wien VPN active + ~/.ssh/config "datalab" entry (cluster/config.sh)
# and the one-time ~/thesis-person venv (./cluster/submit_person_activity.sh setup).
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

# Forward any HPO_* matrix overrides through the ssh environment to the sbatch.
HPO_ENV=""
for v in HPO_CELLS HPO_WIRINGS HPO_SEEDS HPO_TRIALS HPO_EPOCHS HPO_BUDGET \
         HPO_NTRAJ HPO_SEQLEN HPO_SYSTEM HPO_MAXPAR HPO_OUTDIR HPO_STORAGE \
         HPO_WANDB; do
    if [[ -n "${!v:-}" ]]; then HPO_ENV+="${v}='${!v}' "; fi
done

_rsync

# shellcheck disable=SC2029
ssh "${CLUSTER_HOST}" "cd ${REPO_DIR} && \
mkdir -p outputs/slurm results/lotka_volterra_hpo outputs/optuna outputs/wandb && \
export PATH=\$HOME/.local/bin:\$PATH && \
${HPO_ENV}sbatch --export=ALL cluster/lotka_volterra_hpo.sbatch"

echo
echo "Monitor:  ssh ${CLUSTER_HOST} squeue -u \\\$USER"
echo "Logs:     ssh ${CLUSTER_HOST} tail -f ${REPO_DIR}/outputs/slurm/lv-hpo-<jobid>.out"
echo "Fetch:    ./cluster/fetch_lotka_volterra_hpo.sh"
