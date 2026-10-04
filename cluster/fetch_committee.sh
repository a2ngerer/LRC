#!/usr/bin/env bash
# Pull the committee benchmark results (JSONs + plot) back from the cluster.
#   ./cluster/fetch_committee.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/config.sh

REPO_DIR="thesis-person"
mkdir -p results/committee
rsync -avz "${CLUSTER_HOST}:${REPO_DIR}/results/committee/" \
    results/committee/
ls -la results/committee/ | tail
echo "Done."
