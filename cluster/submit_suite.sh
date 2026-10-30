#!/usr/bin/env bash
# Sync the code tree and submit one final-suite campaign (configs/suite/*.yaml,
# experiments/suite_manifest.py) as SLURM arrays into ONE run folder (layout:
# cluster/_run.sh): one array per resource group (default, heavy), each
# --array=0-(T-1)%<array_throttle> with T = ceil(lines / runs_per_gpu).
#
#   ./cluster/submit_suite.sh configs/suite/lv_hpo.yaml lv-hpo
#   ./cluster/submit_suite.sh -n configs/suite/lv_hpo.yaml lv-hpo        # dry run
#   ./cluster/submit_suite.sh --wandb configs/suite/lv_final.yaml lv-final
#   # after a walltime kill: the same config into the existing run folder; done
#   # lines are skipped, only the missing ones run
#   ./cluster/submit_suite.sh --resume configs/suite/lv_final.yaml 2026-10-28_lv-final
#
# The manifests are expanded locally (stage final reads the fetched HPO run,
# hpo_dir) and uploaded as <run>/manifest_<group>.jsonl. Resources per group:
# the YAML cluster / heavy.cluster keys, else cluster/config.sh. The jobs get
# SUITE_CAMPAIGN (the YAML stem) and SUITE_STAGE, the wandb group and job_type
# of every --wandb fit (src/benchmark/tracking.py track_fit).
# Prereqs: VPN + the cluster venv (cluster/README.md).
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/config.sh
source cluster/_run.sh

CONFIG=""; WANDB_FLAG=""; RUN_RESUME=0; args=()
for a in "$@"; do
    case "$a" in
        --wandb) WANDB_FLAG="--wandb" ;;
        --resume) RUN_RESUME=1 ;;
        -*) args+=("$a") ;;
        *) if [[ -z "$CONFIG" ]]; then CONFIG="$a"; else args+=("$a"); fi ;;
    esac
done
if [[ -z "$CONFIG" || ! -f "$CONFIG" ]]; then
    echo "usage: $0 [-n] [--wandb] [--resume] <config.yaml> <label | run folder with --resume>" >&2
    exit 2
fi
run_parse_args ${args[@]+"${args[@]}"}

PLAN_DIR="$(mktemp -d)"
trap 'rm -rf "$PLAN_DIR"' EXIT
plan="$(uv run python experiments/suite_manifest.py "$CONFIG" --write "$PLAN_DIR" $WANDB_FLAG)"
eval "$plan"
SUITE_CAMPAIGN="$(basename "$CONFIG" .yaml)"
RUN_ENV="config ${CONFIG}, stage ${SUITE_STAGE}, manifest_{${SUITE_GROUPS// /,}}.jsonl in the run folder"
RESUME_FLAG=""; [[ $RUN_RESUME -eq 1 ]] && RESUME_FLAG="--resume "
RUN_COMMAND="./cluster/submit_suite.sh ${WANDB_FLAG:+--wandb }${RESUME_FLAG}${CONFIG} ${LABEL}"
DONE_FILE="one result JSON"; [[ $SUITE_STAGE == hpo ]] && DONE_FILE="hpo/<arm>/hpo_selection.json"
RUN_COUNT="${SUITE_COUNT} manifest lines (${DONE_FILE} each)"
echo "fits: ${RUN_COUNT}"

run_prepare "${SUITE_BENCH}"   # reserve the run folder first, then touch the shared code tree
for g in $SUITE_GROUPS; do
    _run_ssh "cat > ${RUN_DIR}/manifest_${g}.jsonl" < "$PLAN_DIR/manifest_${g}.jsonl"
done
run_rsync
for g in $SUITE_GROUPS; do
    n="SUITE_${g}_N"; k="SUITE_${g}_RUNS_PER_GPU"; mem="SUITE_${g}_MEM"
    time="SUITE_${g}_WALLTIME"; cpus="SUITE_${g}_CPUS"; thr="SUITE_${g}_ARRAY_THROTTLE"
    N="${!n}"; K="${!k:-$CLUSTER_RUNS_PER_GPU}"
    T=$(( (N + K - 1) / K ))
    echo "group ${g}: ${N} lines -> ${T} array tasks, ${K} per GPU"
    SBATCH_EXTRA="--job-name=suite-${g} --array=0-$((T - 1))%${!thr:-$CLUSTER_ARRAY_THROTTLE} \
--partition=${CLUSTER_PARTITION} --gres=gpu:${CLUSTER_GPU} --cpus-per-task=${!cpus:-$CLUSTER_CPUS} \
--mem=${!mem:-$CLUSTER_MEM} --time=${!time:-$CLUSTER_TIME}"
    run_sbatch cluster/suite_array.sbatch "SUITE_GROUP=${g} N_SPECS=${N} RUNS_PER_GPU=${K} \
SUITE_CAMPAIGN='${SUITE_CAMPAIGN}' SUITE_STAGE='${SUITE_STAGE}' "
done
