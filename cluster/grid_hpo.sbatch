#!/usr/bin/env bash
# Fixed-grid HPO (Chapter 5 protocol, experiments/run_grid_hpo.py) for ONE
# (task, cell, wiring): the candidate fits run one after another on one GPU,
# each into its own folder under <run folder>/results/, and the selection lands
# in <run folder>/results/hpo_selection.json. A requeued job resumes: fits whose
# result JSON exists are reused. Submitted by cluster/submit_grid_hpo.sh.
#
#SBATCH --job-name=grid-hpo
#SBATCH --partition=GPU-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
# --output/--error are passed on the sbatch command line by cluster/_run.sh
# (<run folder>/logs/); RUN_DIR is exported by the launcher.
#SBATCH --open-mode=append
#SBATCH --requeue
set -euo pipefail
# Run folder (cluster/_run.sh): never write under the code tree (rsync --delete).
RUN_DIR="${RUN_DIR:?RUN_DIR not set: submit via cluster/submit_*.sh}"
mkdir -p "$RUN_DIR/results" "$RUN_DIR/wandb"

source cluster/_common.sh
cc_setup_tmpdir
cc_setup_cuda_ld
cc_setup_tf_env 1                       # sequential fits: all CPUs to one run
# A forwarded --wandb (GH_EXTRA) logs offline into the run folder.
export WANDB_MODE=offline WANDB_DIR="$RUN_DIR/wandb" WANDB_SILENT=true

TASK="${GH_TASK:?GH_TASK (lv|pa) not set}"
CELL="${GH_CELL:?GH_CELL not set}"
WIRING="${GH_WIRING:?GH_WIRING not set}"
SEEDS="${GH_TUNING_SEEDS:-100 101}"   # disjoint from the final seeds 0..9
# Thesis budgets: 4k for the rollouts, 30k for Person Activity.
if [[ "$TASK" == pa ]]; then BUDGET="${GH_PARAM_BUDGET:-30000}"; else BUDGET="${GH_PARAM_BUDGET:-4000}"; fi
SYSTEM_FLAG=""
[[ -n "${GH_SYSTEM:-}" ]] && SYSTEM_FLAG="--system ${GH_SYSTEM}"
# Driver flags (--grid-lr, --grid-clip, --select-on) and runner flags (e.g.
# --epochs 2) in one string; the driver forwards what it does not know.
EXTRA="${GH_EXTRA:-}"

echo "=== grid HPO ${TASK} ${CELL}/${WIRING} ${GH_SYSTEM:-} seeds ${SEEDS} budget ${BUDGET} on $(hostname) $(date -u +%FT%TZ) ==="
# shellcheck disable=SC2086  # flag lists are split on purpose
uv run python experiments/run_grid_hpo.py --task "$TASK" --cell "$CELL" \
    --wiring "$WIRING" $SYSTEM_FLAG --param-budget "$BUDGET" \
    --tuning-seeds $SEEDS --outdir "$RUN_DIR/results" $EXTRA
echo "=== grid HPO done $(date -u +%FT%TZ) ==="
