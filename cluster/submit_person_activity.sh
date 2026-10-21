#!/usr/bin/env bash
# Sync code + dataset and submit the Person-Activity benchmark
# (cluster/person_activity.sbatch) as ONE SLURM job into its own run folder
# under 03_person_activity (see cluster/_run.sh for the layout).
#
#   ./cluster/submit_person_activity.sh setup        # one-time venv on the login node
#   ./cluster/submit_person_activity.sh <label>      # cfc_lrc + gru x {ncp,cncp} x 3 seeds, 50 epochs
#   PA_CELLS="cfc_lrc" PA_SEEDS="0 1 2" ./cluster/submit_person_activity.sh <label>
#   ./cluster/submit_person_activity.sh -n <label>   # dry run
#
# Prereqs: TU Wien VPN + ~/.ssh/config "datalab" entry (cluster/config.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/config.sh
source cluster/_run.sh
run_parse_args "$@"

if [[ "${SETUP}" -eq 1 ]]; then
    run_rsync
    echo "Setting up the venv (uv sync --extra cuda --extra tracking) on the login node ..."
    _run_ssh "cd ${CLUSTER_REPO_DIR} && export PATH=\$HOME/.local/bin:\$PATH && \
        export TMPDIR=\$HOME/tmp && mkdir -p \$HOME/tmp && \
        uv sync --extra cuda --extra tracking && \
        uv run python -c 'import tensorflow as tf; print(\"tf\", tf.__version__)'"
    echo "Setup done."
    exit 0
fi

# Forward any PA_* matrix overrides through the ssh environment to the sbatch.
PA_ENV=""
for v in PA_EPOCHS PA_CELLS PA_WIRINGS PA_SEEDS PA_MAXPAR PA_WANDB \
         PA_PARAM_BUDGET PA_LR PA_BACKBONE; do
    if [[ -n "${!v:-}" ]]; then PA_ENV+="${v}='${!v}' "; fi
done
RUN_ENV="${PA_ENV:-<defaults>}"
RUN_COMMAND="${PA_ENV}./cluster/submit_person_activity.sh ${LABEL}"

run_rsync
run_prepare 03_person_activity
run_sbatch cluster/person_activity.sbatch "${PA_ENV}"
