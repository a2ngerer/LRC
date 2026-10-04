#!/usr/bin/env bash
# Pull the active-glimpse benchmark results (JSONs + plot) back from the cluster.
#   ./cluster/fetch_active_glimpse.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/config.sh

REPO_DIR="thesis-person"
mkdir -p results/active_glimpse
rsync -avz "${CLUSTER_HOST}:${REPO_DIR}/results/active_glimpse/" \
    results/active_glimpse/
ls -la results/active_glimpse/ | tail
echo "Done."
