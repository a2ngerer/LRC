#!/usr/bin/env bash
# Sync code and submit the partial-view voting-committee benchmark
# (cluster/committee.sbatch) as ONE SLURM job into its own run folder under
# 04_column_voting (see cluster/_run.sh for the layout). The person-activity
# CSV under data/person/ is rsynced with the code when present locally and
# protected on the cluster otherwise (cluster/README.md, "Dataset").
#
#   ./cluster/submit_committee.sh <label>
#   CO_WIRINGS="cncp" CO_KS="1 4" CO_SEEDS="0" ./cluster/submit_committee.sh <label>
#   ./cluster/submit_committee.sh -n <label>      # dry run
#
# Prereqs: TU Wien VPN + ~/.ssh/config "datalab" (cluster/config.sh) and the
# venv on the cluster (./cluster/submit_person_activity.sh setup).
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/config.sh
source cluster/_run.sh
run_parse_args "$@"

CO_ENV=""
for v in CO_EPOCHS CO_CELL CO_WIRINGS CO_KS CO_SEEDS CO_SEQLEN CO_BATCH \
         CO_VIEWMODE CO_DROPFRACS CO_TRAINDROP CO_TRAINNOISE CO_TESTCORRUPT \
         CO_NOISESIGMAS CO_PARAM_BUDGET CO_MAXPAR CO_EXTRA CO_BACKBONE; do
    if [[ -n "${!v:-}" ]]; then CO_ENV+="${v}='${!v}' "; fi
done
RUN_ENV="${CO_ENV:-<defaults>}"
RUN_COMMAND="${CO_ENV}./cluster/submit_committee.sh ${LABEL}"
# ponytail: defaults mirror cluster/committee.sbatch
run_count "${CO_WIRINGS:-cncp ncp dense}" "${CO_KS:-1 2 4}" "${CO_SEEDS:-0 1 2}"

run_prepare 04_column_voting   # reserve the run folder first, then touch the shared code tree
run_rsync
run_require_dataset
run_sbatch cluster/committee.sbatch "${CO_ENV}"
