#!/usr/bin/env bash
# Sync the code repo to the dataLAB cluster via rsync.
# Compute nodes have no internet access and no GitHub access — rsync is the
# only code-transfer path (same approach as arc-jax-rl).
set -euo pipefail

cd "$(dirname "$0")/.."
source cluster/config.sh

echo "Syncing $(pwd) -> ${CLUSTER_HOST}:${CLUSTER_REPO_DIR}/"
rsync -avz --delete \
    --exclude='.git/' \
    --exclude='.venv/' \
    --exclude='results/' \
    --exclude='outputs/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    --exclude='.pytest_cache/' \
    --exclude='datasets/' \
    ./ "${CLUSTER_HOST}:${CLUSTER_REPO_DIR}/"

echo "Done."
