#!/usr/bin/env bash
# Pull the data-efficiency benchmark results (JSONs + plot) back from the cluster.
#   ./cluster/fetch_data_efficiency.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/config.sh

REPO_DIR="thesis-person"
mkdir -p results/data_efficiency
rsync -avz "${CLUSTER_HOST}:${REPO_DIR}/results/data_efficiency/" \
    results/data_efficiency/
ls -la results/data_efficiency/ | tail
echo "Done."
