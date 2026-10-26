#!/usr/bin/env bash
# Sync the code tree and submit the fixed-grid HPO (cluster/grid_hpo.sbatch) for
# ONE (task, cell, wiring) as ONE SLURM job into its own run folder (layout:
# cluster/_run.sh). lv -> 01_lotka_volterra, lv + GH_SYSTEM=duffing ->
# 02_duffing, pa -> 03_person_activity.
#
#   GH_TASK=lv GH_CELL=cfc_lrc GH_WIRING=dense3 ./cluster/submit_grid_hpo.sh <label>
#   GH_TASK=pa GH_CELL=gru GH_WIRING=ncp ./cluster/submit_grid_hpo.sh -n <label>   # dry run
#   BENCH=00_canary GH_TASK=lv GH_CELL=cfc_lrc GH_WIRING=dense3 \
#       GH_EXTRA="--epochs 2 --grid-lr 1e-3 3e-3 --grid-clip off" \
#       ./cluster/submit_grid_hpo.sh grid-hpo-canary
#
# Env: GH_TASK, GH_CELL, GH_WIRING (required), GH_SYSTEM (lv), GH_PARAM_BUDGET
# (default 4000 lv / 30000 pa), GH_TUNING_SEEDS (default "0 1"), GH_EXTRA
# (driver + runner flags). Prereqs: VPN + the cluster venv (cluster/README.md).
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/config.sh
source cluster/_run.sh
run_parse_args "$@"
: "${GH_TASK:?GH_TASK=lv|pa required}" "${GH_CELL:?GH_CELL required}" \
  "${GH_WIRING:?GH_WIRING required}"

# Forward the GH_* settings through the ssh environment to the sbatch.
GH_ENV=""
for v in GH_TASK GH_CELL GH_WIRING GH_SYSTEM GH_PARAM_BUDGET GH_TUNING_SEEDS \
         GH_EXTRA; do
    if [[ -n "${!v:-}" ]]; then GH_ENV+="${v}='${!v}' "; fi
done
RUN_ENV="${GH_ENV}"
RUN_COMMAND="${GH_ENV}./cluster/submit_grid_hpo.sh ${LABEL}"
RUN_COUNT="hpo_selection.json + 10 fit JSONs (full grid; +2..4 with the edge rule)"

case "${GH_TASK}:${GH_SYSTEM:-}" in
    pa:*)       BENCH_DEFAULT="03_person_activity" ;;
    lv:duffing) BENCH_DEFAULT="02_duffing" ;;
    lv:*)       BENCH_DEFAULT="01_lotka_volterra" ;;
    *) echo "GH_TASK must be lv or pa, got '${GH_TASK}'" >&2; exit 2 ;;
esac

run_prepare "${BENCH_DEFAULT}"   # reserve the run folder first, then touch the shared code tree
run_rsync
run_sbatch cluster/grid_hpo.sbatch "${GH_ENV}"
