#!/usr/bin/env bash
# Thin wrapper: fetch Person-Activity run folders (03_person_activity).
#   ./cluster/fetch_person_activity.sh                 # list runs
#   ./cluster/fetch_person_activity.sh <run> [--logs]
set -euo pipefail
cd "$(dirname "$0")/.."
exec ./cluster/fetch_run.sh 03_person_activity "$@"
