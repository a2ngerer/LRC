#!/usr/bin/env bash
# Thin wrapper: fetch Lotka-Volterra (01) or Duffing (02) run folders.
#   ./cluster/fetch_lotka_volterra.sh                 # list runs of both
#   ./cluster/fetch_lotka_volterra.sh <run> [--logs]  # 01_lotka_volterra/<run>
#   LV_SYSTEM=duffing ./cluster/fetch_lotka_volterra.sh <run>   # 02_duffing/<run>
set -euo pipefail
cd "$(dirname "$0")/.."
case "${LV_SYSTEM:-periodic_predator_prey}" in
    duffing) bench=02_duffing ;;
    *)       bench=01_lotka_volterra ;;
esac
if [[ $# -eq 0 ]]; then
    ./cluster/fetch_run.sh 01_lotka_volterra
    ./cluster/fetch_run.sh 02_duffing
    exit 0
fi
exec ./cluster/fetch_run.sh "${bench}" "$@"
