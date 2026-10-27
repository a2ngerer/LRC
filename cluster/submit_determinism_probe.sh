#!/usr/bin/env bash
# Sync the code tree and submit the S0 op-determinism probe (#70,
# cluster/determinism_probe.sbatch) as ONE SLURM job into its own run folder
# (default 00_canary, see cluster/_run.sh for the layout).
#
#   ./cluster/submit_determinism_probe.sh s0-determinism
#   DP_LV_ARGS="--forcing mixed" ./cluster/submit_determinism_probe.sh <label>
#   DP_LV_ARGS="--epochs 3" DP_PA_ARGS="--epochs 1" ./cluster/submit_determinism_probe.sh -n smoke
#
# Prereqs: TU Wien VPN + ~/.ssh/config "datalab" (cluster/config.sh) and the
# venv on the cluster (./cluster/submit_person_activity.sh setup).
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/config.sh
source cluster/_run.sh
run_parse_args "$@"

DP_ENV=""
for v in DP_LV_ARGS DP_PA_ARGS; do
    if [[ -n "${!v:-}" ]]; then DP_ENV+="${v}='${!v}' "; fi
done
RUN_ENV="${DP_ENV:-<defaults>}"
RUN_COMMAND="${DP_ENV}./cluster/submit_determinism_probe.sh ${LABEL}"
RUN_COUNT="6 (on, on, off on lv and pa) + determinism_probe.json"
echo "fits: ${RUN_COUNT}"

run_prepare 00_canary   # reserve the run folder first, then touch the shared code tree
run_rsync
run_require_dataset
run_sbatch cluster/determinism_probe.sbatch "${DP_ENV}"
