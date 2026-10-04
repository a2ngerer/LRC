#!/usr/bin/env bash
# Fetch the Lotka-Volterra benchmark result JSONs + rendered plots from the
# isolated cluster repo ~/thesis-person back into the local results/ tree.
# Prereqs: TU Wien VPN + "datalab" ssh entry (cluster/config.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/config.sh

REPO_DIR="thesis-person"
# predator_prey + duffing are submitted as two jobs into two result dirs.
for sub in lotka_volterra lotka_volterra_duffing; do
    mkdir -p "results/${sub}"
    echo "rsync ${CLUSTER_HOST}:${REPO_DIR}/results/${sub}/ -> results/${sub}/"
    rsync -avz "${CLUSTER_HOST}:${REPO_DIR}/results/${sub}/" "results/${sub}/" || true
    echo "-- results/${sub}:"
    ls -la "results/${sub}"/*.json 2>/dev/null | tail -20
    ls -la "results/${sub}"/*.png 2>/dev/null
done

# Bring back any offline wandb runs (only present if submitted with LV_WANDB=1);
# upload them locally with ./cluster/wandb_sync.sh.
mkdir -p outputs/wandb
echo "rsync ${CLUSTER_HOST}:${REPO_DIR}/outputs/wandb/ -> outputs/wandb/"
rsync -avz "${CLUSTER_HOST}:${REPO_DIR}/outputs/wandb/" outputs/wandb/ || true

echo "Done."
