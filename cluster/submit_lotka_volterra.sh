#!/usr/bin/env bash
# Sync the code tree and submit the Lotka-Volterra / Duffing rollout benchmark
# (cluster/lotka_volterra.sbatch) as ONE SLURM job into its own run folder
# (see cluster/_run.sh for the layout).
#
#   ./cluster/submit_lotka_volterra.sh <label>              # periodic_predator_prey -> 01_lotka_volterra
#   LV_SYSTEM=duffing ./cluster/submit_lotka_volterra.sh <label>   # -> 02_duffing
#   LV_SYSTEM=multitimescale LV_TSPAN="0 10" ./cluster/submit_lotka_volterra.sh <label>   # t_span override
#   LV_CELLS="cfc_lrc" LV_SEEDS="0" ./cluster/submit_lotka_volterra.sh -n smoke   # dry run
#   BENCH=00_canary LV_EPOCHS=3 ./cluster/submit_lotka_volterra.sh lv-smoke
#
# Prereqs: TU Wien VPN + ~/.ssh/config "datalab" entry (cluster/config.sh) and
# the venv on the cluster (./cluster/submit_person_activity.sh setup).
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/config.sh
source cluster/_run.sh
run_parse_args "$@"

# Forward any LV_* matrix overrides through the ssh environment to the sbatch.
LV_ENV=""
for v in LV_EPOCHS LV_CELLS LV_WIRINGS LV_SEEDS LV_MAXPAR LV_PLOT_CELL \
         LV_SYSTEM LV_TSPAN LV_SEQLEN LV_WANDB LV_PARAM_BUDGET LV_LR LV_BACKBONE LV_EXTRA \
         LV_DATA_SEED_OFFSET; do
    if [[ -n "${!v:-}" ]]; then LV_ENV+="${v}='${!v}' "; fi
done
RUN_ENV="${LV_ENV:-<defaults>}"
RUN_COMMAND="${LV_ENV}./cluster/submit_lotka_volterra.sh ${LABEL}"
# ponytail: defaults mirror cluster/lotka_volterra.sbatch
run_count "${LV_CELLS:-cfc_lrc gru}" "${LV_WIRINGS:-dense dense3 ncp cncp tbt_cncp_noloc tbt_cncp_concat}" "${LV_SEEDS:-0 1 2}"

case "${LV_SYSTEM:-periodic_predator_prey}" in
    duffing) BENCH_DEFAULT="02_duffing" ;;
    *)       BENCH_DEFAULT="01_lotka_volterra" ;;
esac

run_prepare "${BENCH_DEFAULT}"   # reserve the run folder first, then touch the shared code tree
run_rsync
run_sbatch cluster/lotka_volterra.sbatch "${LV_ENV}"
