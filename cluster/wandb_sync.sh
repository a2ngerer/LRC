#!/usr/bin/env bash
# Upload offline wandb runs to Weights & Biases.
#
#   ./cluster/wandb_sync.sh results/<NN_benchmark>/<run>   # a fetched run folder
#   ./cluster/wandb_sync.sh                                # outputs/wandb/
#
# Compute nodes have no internet, so --wandb fits run OFFLINE
# (WANDB_MODE=offline) with WANDB_DIR=<run folder>/wandb (older launchers:
# outputs/wandb). Fetch the run folder first (cluster/fetch_run.sh); this script
# uploads every offline-run-* directory below the given folder. Run it LOCALLY,
# where you are logged in (`uv run wandb login` once).
set -euo pipefail
root=""
if [[ $# -gt 0 ]]; then
    [[ -d "$1" ]] || { echo "no directory $1 -- fetch the run folder first" >&2; exit 1; }
    root="$(cd "$1" && pwd)"
fi
cd "$(dirname "$0")/.."
root="${root:-outputs/wandb}"

runs=()
while IFS= read -r d; do
    runs+=("$d")
done < <(find "$root" -type d -name 'offline-run-*' -prune 2>/dev/null | sort)
if (( ${#runs[@]} == 0 )); then
    echo "no offline-run-* dirs below $root -- nothing to sync" >&2
    exit 1
fi

echo "syncing ${#runs[@]} offline run(s) below $root to wandb ..."
uv run wandb sync "${runs[@]}"
echo "done."
