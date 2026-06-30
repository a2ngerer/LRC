#!/usr/bin/env bash
# Fetch MuJoCo benchmark results (JSON, checkpoints, videos) back from the
# isolated cluster repo ~/thesis-mujoco into results/mujoco_cluster/.
#
#   ./cluster/fetch_mujoco.sh           # results only
#   ./cluster/fetch_mujoco.sh --logs    # also SLURM logs
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/config.sh

REPO_DIR="thesis-mujoco"
mkdir -p results/mujoco_cluster
echo "Fetching ${CLUSTER_HOST}:${REPO_DIR}/results/mujoco/ -> results/mujoco_cluster/ ..."
rsync -avz "${CLUSTER_HOST}:${REPO_DIR}/results/mujoco/" results/mujoco_cluster/ || true

if [[ "${1:-}" == "--logs" ]]; then
    mkdir -p outputs/slurm_mujoco
    rsync -avz "${CLUSTER_HOST}:${REPO_DIR}/outputs/slurm/" outputs/slurm_mujoco/ || true
fi
echo "Done. Checkpoints: results/mujoco_cluster/checkpoints  Videos: results/mujoco_cluster/videos"
