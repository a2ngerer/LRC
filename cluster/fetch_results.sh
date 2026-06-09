#!/usr/bin/env bash
# Fetch benchmark results (and SLURM logs) back from the cluster.
#
#   ./cluster/fetch_results.sh           # results/runs JSONs
#   ./cluster/fetch_results.sh --logs    # additionally outputs/slurm logs
set -euo pipefail

cd "$(dirname "$0")/.."
source cluster/config.sh

mkdir -p results
echo "Fetching results from ${CLUSTER_HOST}:${CLUSTER_REPO_DIR}/results/ ..."
rsync -avz "${CLUSTER_HOST}:${CLUSTER_REPO_DIR}/results/" results/

if [[ "${1:-}" == "--logs" ]]; then
    mkdir -p outputs/slurm
    rsync -avz "${CLUSTER_HOST}:${CLUSTER_REPO_DIR}/outputs/slurm/" outputs/slurm/
fi

echo "Done. Next steps:"
echo "  uv run python experiments/aggregate_results.py"
echo "  uv run python experiments/plot_results.py"
