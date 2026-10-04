#!/usr/bin/env bash
# Optuna HPO sweep for the parameter-matched Lotka-Volterra rollout benchmark.
# Runs one Optuna study per (cell, wiring) -- the thesis-core matrix
# {cfc_lrc, ltc, gru} x {dense, ncp, cncp} -- with bounded parallelism on one
# GPU. Each study tunes the soft hyper-parameters (lr, batch size) with a TPE
# sampler + MedianPruner while the width is fixed per arm to a shared parameter
# budget (parameter-matched, per Farsang 2026-07-02). Every Optuna trial is
# logged to wandb (offline here; sync from the login node). Submitted by
# cluster/submit_lotka_volterra_hpo.sh from the isolated ~/thesis-person repo
# (its own uv venv). Lotka-Volterra needs no external dataset (scipy in-memory).
#
#SBATCH --job-name=lv-hpo
#SBATCH --partition=GPU-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=24G
#SBATCH --time=14:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --requeue
set -euo pipefail

# Virtual nodes have a non-writable /tmp -> redirect temp into $HOME.
export TMPDIR="$HOME/tmp"; export TMP="$TMPDIR"; export TEMP="$TMPDIR"
mkdir -p "$TMPDIR"
export PATH="$HOME/.local/bin:$PATH"

# tensorflow[and-cuda] ships CUDA libs under site-packages/nvidia/*/lib; TF 2.15
# needs them on LD_LIBRARY_PATH (same workaround as lotka_volterra.sbatch).
VENV_SITE="$(uv run python -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)"
if [[ -d "${VENV_SITE}/nvidia" ]]; then
    CUDA_LD="$(find "${VENV_SITE}/nvidia" -maxdepth 2 -name 'lib' -type d 2>/dev/null | tr '\n' ':')"
    export LD_LIBRARY_PATH="${CUDA_LD}${LD_LIBRARY_PATH:-}"
fi
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=1
export PYTHONUNBUFFERED=1

# wandb offline on the compute node (no worker internet). Runs land under
# outputs/wandb/, are fetched by fetch_lotka_volterra_hpo.sh, uploaded locally
# via cluster/wandb_sync.sh.
export WANDB_MODE=offline
export WANDB_DIR="$PWD/outputs/wandb"
export WANDB_SILENT=true
mkdir -p "$WANDB_DIR"
# Per-trial wandb on/off. The wandb service (wandb-core + wandb-xpu system-stats)
# can DEADLOCK under high parallelism when WANDB_DIR is on NFS -- observed on
# datalab, job 572364 wedged on a straggler. Set HPO_WANDB=0 to get clean result
# JSONs without wandb (wandb is demonstrated separately by the local run); keep 1
# only at low HPO_MAXPAR.
USE_WANDB="${HPO_WANDB:-1}"

# Matrix (override via env at submit time, e.g. HPO_CELLS="cfc_lrc gru"):
# Fast closed-form family by default (LRC / CfC / classical GRU). The iterative
# `ltc` cell is ~50x slower (sequential ODE solver, CPU-bound even on GPU) and
# would blow the walltime at 30 trials x 150 epochs -- opt in via HPO_CELLS with
# a reduced HPO_TRIALS/HPO_EPOCHS budget only.
CELLS="${HPO_CELLS:-cfc_lrc cfc gru}"
WIRINGS="${HPO_WIRINGS:-dense ncp cncp}"
SEEDS="${HPO_SEEDS:-0}"
TRIALS="${HPO_TRIALS:-30}"
EPOCHS="${HPO_EPOCHS:-150}"
BUDGET="${HPO_BUDGET:-4000}"
NTRAJ="${HPO_NTRAJ:-80}"
SEQLEN="${HPO_SEQLEN:-128}"
SYSTEM="${HPO_SYSTEM:-periodic_predator_prey}"
MAXPAR="${HPO_MAXPAR:-6}"
OUTDIR="${HPO_OUTDIR:-results/lotka_volterra_hpo}"
STORAGE="${HPO_STORAGE:-outputs/optuna}"

mkdir -p outputs/slurm "$OUTDIR" "$STORAGE"

run_one() {
    local cell="$1" wiring="$2" seed="$3"
    local wandb_flag=""
    [[ "${USE_WANDB:-1}" == "1" ]] && wandb_flag="--wandb"
    echo "[start] hpo ${cell}_${wiring}_seed${seed} $(date -u +%FT%TZ)"
    uv run python experiments/run_lotka_volterra_hpo.py \
        --cell "$cell" --wiring "$wiring" --seed "$seed" \
        --n-trials "$TRIALS" --epochs "$EPOCHS" --param-budget "$BUDGET" \
        --n-trajectories "$NTRAJ" --seq-len "$SEQLEN" --system "$SYSTEM" \
        --outdir "$OUTDIR" --storage-dir "$STORAGE" $wandb_flag
    echo "[done ] hpo ${cell}_${wiring}_seed${seed} (exit $?)"
}
export -f run_one
export TRIALS EPOCHS BUDGET NTRAJ SEQLEN SYSTEM OUTDIR STORAGE USE_WANDB

JOBS=""
for cell in $CELLS; do
    for wiring in $WIRINGS; do
        for seed in $SEEDS; do
            JOBS+="${cell} ${wiring} ${seed}"$'\n'
        done
    done
done

echo "=== LV-HPO ${SYSTEM}: $(printf '%s' "$JOBS" | grep -c .) studies, par ${MAXPAR}, ${TRIALS} trials x ${EPOCHS} epochs, budget ${BUDGET}, on $(hostname) ==="
# `|| true`: a single failing study must not abort the sweep (set -e + pipefail)
# before the surviving studies are aggregated.
printf '%s' "$JOBS" | grep . | xargs -P "$MAXPAR" -n 3 bash -c 'run_one "$1" "$2" "$3"' _ || true
echo "=== all studies done $(date -u +%FT%TZ) ==="
ls -la "$OUTDIR"/*.json

echo "=== aggregating summary table ==="
uv run python experiments/aggregate_hpo.py --indir "$OUTDIR"
echo "=== HPO sweep complete $(date -u +%FT%TZ) ==="
