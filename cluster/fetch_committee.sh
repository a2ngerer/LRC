#!/usr/bin/env bash
# Thin wrapper: fetch column-voting run folders (04_column_voting).
#   ./cluster/fetch_committee.sh                 # list runs
#   ./cluster/fetch_committee.sh <run> [--logs]
set -euo pipefail
cd "$(dirname "$0")/.."
exec ./cluster/fetch_run.sh 04_column_voting "$@"
