#!/usr/bin/env bash
# Fetch the Lotka-Volterra HPO results (per-study JSONs + summary.md), the Optuna
# journal storage, and the offline wandb runs from the isolated cluster repo
# ~/thesis-person back into the local tree. Upload the wandb runs afterwards with
# ./cluster/wandb_sync.sh. Prereqs: TU Wien VPN + "datalab" ssh entry.
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/config.sh

REPO_DIR="thesis-person"

mkdir -p results/lotka_volterra_hpo outputs/optuna outputs/wandb
for pair in "results/lotka_volterra_hpo" "outputs/optuna" "outputs/wandb"; do
    echo "rsync ${CLUSTER_HOST}:${REPO_DIR}/${pair}/ -> ${pair}/"
    rsync -avz "${CLUSTER_HOST}:${REPO_DIR}/${pair}/" "${pair}/" || true
done

echo "-- HPO result JSONs:"
ls -la results/lotka_volterra_hpo/*.json 2>/dev/null | tail -20
echo "-- summary:"
cat results/lotka_volterra_hpo/summary.md 2>/dev/null || echo "(no summary.md yet)"
echo "Done. Upload offline wandb runs with ./cluster/wandb_sync.sh"
