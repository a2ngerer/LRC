#!/usr/bin/env bash
# Fetch the Person-Activity benchmark result JSONs from the isolated cluster repo
# ~/thesis-person back into the local results/ tree, so plot_person_activity.py
# can render them. Prereqs: TU Wien VPN + "datalab" ssh entry (cluster/config.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/config.sh

REPO_DIR="thesis-person"
mkdir -p results/person_activity
echo "rsync ${CLUSTER_HOST}:${REPO_DIR}/results/person_activity/ -> results/person_activity/"
rsync -avz "${CLUSTER_HOST}:${REPO_DIR}/results/person_activity/" results/person_activity/

# Bring back any offline wandb runs (only present if submitted with PA_WANDB=1);
# upload them locally with ./cluster/wandb_sync.sh.
mkdir -p outputs/wandb
echo "rsync ${CLUSTER_HOST}:${REPO_DIR}/outputs/wandb/ -> outputs/wandb/"
rsync -avz "${CLUSTER_HOST}:${REPO_DIR}/outputs/wandb/" outputs/wandb/ || true

echo "Done. Result JSONs:"
ls -la results/person_activity/*.json 2>/dev/null | tail -20
