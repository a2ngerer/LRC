#!/usr/bin/env bash
# Fetch the tbt_cNCP active-sensing result JSONs + rendered plot from the
# isolated cluster repo ~/thesis-person into the local results/ tree.
# Prereqs: TU Wien VPN + "datalab" ssh entry (cluster/config.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/config.sh

REPO_DIR="thesis-person"
mkdir -p results/active_sensing
echo "rsync ${CLUSTER_HOST}:${REPO_DIR}/results/active_sensing/ -> results/active_sensing/"
rsync -avz "${CLUSTER_HOST}:${REPO_DIR}/results/active_sensing/" results/active_sensing/
echo "Done."
ls -la results/active_sensing/*.json 2>/dev/null | tail -20
ls -la results/active_sensing/*.png 2>/dev/null
