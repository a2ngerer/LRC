#!/usr/bin/env bash
# Sync code to the ISOLATED cluster repo ~/thesis-person and submit the
# cNCP-vs-NCP-vs-dense predator-prey (Lotka-Volterra) rollout benchmark as a
# SINGLE SLURM job (cluster/lotka_volterra.sbatch).
#
# Reuses ~/thesis-person (its own uv venv, --extra cuda already set up for the
# person-activity benchmark). Lotka-Volterra needs no external dataset -- the
# trajectories are generated in-memory via scipy -- so no data rsync and no
# separate venv setup is required.
#
#   # submit (cfc_lrc + gru x {dense,ncp,cncp} x 3 seeds, 300 epochs):
#   ./cluster/submit_lotka_volterra.sh
#   # smaller/faster override via env:
#   LV_CELLS="cfc_lrc" LV_SEEDS="0" ./cluster/submit_lotka_volterra.sh
#
# Prereqs: TU Wien VPN active + ~/.ssh/config "datalab" entry (cluster/config.sh)
# and a one-time ~/thesis-person venv (./cluster/submit_person_activity.sh setup).
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

# Forward any LV_* matrix overrides through the ssh environment to the sbatch.
LV_ENV=""
for v in LV_EPOCHS LV_CELLS LV_WIRINGS LV_SEEDS LV_MAXPAR LV_PLOT_CELL \
         LV_SYSTEM LV_SEQLEN LV_OUTDIR LV_WANDB; do
    if [[ -n "${!v:-}" ]]; then LV_ENV+="${v}='${!v}' "; fi
done

_rsync

# shellcheck disable=SC2029
ssh "${CLUSTER_HOST}" "cd ${REPO_DIR} && \
mkdir -p outputs/slurm results/lotka_volterra results/lotka_volterra_duffing && \
export PATH=\$HOME/.local/bin:\$PATH && \
${LV_ENV}sbatch --export=ALL cluster/lotka_volterra.sbatch"

echo
echo "Monitor:  ssh ${CLUSTER_HOST} squeue -u \\\$USER"
echo "Logs:     ssh ${CLUSTER_HOST} tail -f ${REPO_DIR}/outputs/slurm/lv-cncp-<jobid>.out"
echo "Fetch:    ./cluster/fetch_lotka_volterra.sh"
