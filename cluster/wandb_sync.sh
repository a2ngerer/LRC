#!/usr/bin/env bash
# Upload the offline wandb runs to Weights & Biases.
#
# Compute nodes have no internet, so benchmark runs submitted with PA_WANDB=1 /
# LV_WANDB=1 are recorded OFFLINE under outputs/wandb/. fetch_person_activity.sh
# / fetch_lotka_volterra.sh bring those run dirs back here; this script uploads
# them. Run it LOCALLY, where you are logged in (`uv run wandb login` once, with
# the TU Wien academic account).
set -euo pipefail
cd "$(dirname "$0")/.."

RUNS_DIR="outputs/wandb/wandb"
if [[ ! -d "$RUNS_DIR" ]]; then
    echo "no offline runs at $RUNS_DIR -- submit a --wandb job and fetch first" >&2
    exit 1
fi

shopt -s nullglob
runs=("$RUNS_DIR"/offline-run-*)
if (( ${#runs[@]} == 0 )); then
    echo "no offline-run-* dirs under $RUNS_DIR -- nothing to sync" >&2
    exit 1
fi

echo "syncing ${#runs[@]} offline run(s) to wandb ..."
uv run wandb sync "${runs[@]}"
echo "done."
