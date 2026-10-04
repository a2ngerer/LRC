#!/usr/bin/env bash
# Run the parameter-matched Lotka-Volterra Optuna+wandb HPO sweep LOCALLY.
# Fallback for when the datalab cluster login node is unreachable: same matrix
# and env knobs as cluster/lotka_volterra_hpo.sbatch, executed on this machine
# with wandb in offline mode. Upload the runs afterwards with
# ./cluster/wandb_sync.sh (once `uv run wandb login` is done).
#
#   # default matrix ({cfc_lrc,cfc,gru} x {dense,ncp,cncp} x 3 seeds, 40 trials):
#   ./cluster/run_lotka_volterra_hpo_local.sh
#   # override, e.g. quick check:
#   HPO_SEEDS=0 HPO_TRIALS=10 HPO_EPOCHS=60 ./cluster/run_lotka_volterra_hpo_local.sh
#
# Default cells are the fast closed-form family; the iterative `ltc` cell is
# ~50x slower here (sequential ODE solver) and is deliberately NOT in the
# default matrix -- opt in with HPO_CELLS and a reduced trial/epoch budget.
set -euo pipefail
cd "$(dirname "$0")/.."

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="$PWD/outputs/wandb"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"
export PYTHONUNBUFFERED=1
mkdir -p "$WANDB_DIR"

CELLS="${HPO_CELLS:-cfc_lrc cfc gru}"
WIRINGS="${HPO_WIRINGS:-dense ncp cncp}"
SEEDS="${HPO_SEEDS:-0 1 2}"
TRIALS="${HPO_TRIALS:-40}"
EPOCHS="${HPO_EPOCHS:-150}"
BUDGET="${HPO_BUDGET:-4000}"
NTRAJ="${HPO_NTRAJ:-80}"
SEQLEN="${HPO_SEQLEN:-128}"
SYSTEM="${HPO_SYSTEM:-periodic_predator_prey}"
MAXPAR="${HPO_MAXPAR:-6}"
OUTDIR="${HPO_OUTDIR:-results/lotka_volterra_hpo}"
STORAGE="${HPO_STORAGE:-outputs/optuna}"
mkdir -p "$OUTDIR" "$STORAGE"

run_one() {
    local cell="$1" wiring="$2" seed="$3"
    echo "[start] hpo ${cell}_${wiring}_seed${seed} $(date -u +%FT%TZ)"
    uv run python experiments/run_lotka_volterra_hpo.py \
        --cell "$cell" --wiring "$wiring" --seed "$seed" \
        --n-trials "$TRIALS" --epochs "$EPOCHS" --param-budget "$BUDGET" \
        --n-trajectories "$NTRAJ" --seq-len "$SEQLEN" --system "$SYSTEM" \
        --outdir "$OUTDIR" --storage-dir "$STORAGE" --wandb
    echo "[done ] hpo ${cell}_${wiring}_seed${seed} (exit $?)"
}
export -f run_one
export TRIALS EPOCHS BUDGET NTRAJ SEQLEN SYSTEM OUTDIR STORAGE

JOBS=""
for cell in $CELLS; do
    for wiring in $WIRINGS; do
        for seed in $SEEDS; do
            JOBS+="${cell} ${wiring} ${seed}"$'\n'
        done
    done
done

echo "=== LV-HPO LOCAL ${SYSTEM}: $(printf '%s' "$JOBS" | grep -c .) studies, par ${MAXPAR}, ${TRIALS} trials x ${EPOCHS} epochs, budget ${BUDGET} ==="
# `|| true`: a single failing study must not abort the sweep (set -e + pipefail)
# before the surviving studies are aggregated.
printf '%s' "$JOBS" | grep . | xargs -P "$MAXPAR" -n 3 bash -c 'run_one "$1" "$2" "$3"' _ || true
echo "=== all studies done $(date -u +%FT%TZ) ==="

echo "=== aggregating summary table ==="
uv run python experiments/aggregate_hpo.py --indir "$OUTDIR"
echo "=== HPO sweep complete $(date -u +%FT%TZ) ==="
