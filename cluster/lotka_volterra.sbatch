#!/usr/bin/env bash
# Single SLURM job for the cNCP-vs-NCP-vs-dense predator-prey (Lotka-Volterra)
# sequence-rollout benchmark, then renders the trajectory-overlay plots.
# Runs the whole small matrix (cells x {dense,ncp,cncp} x seeds) in ONE job with
# bounded parallelism on one GPU -- the models are tiny. Submitted by
# cluster/submit_lotka_volterra.sh from the ISOLATED ~/thesis-person repo (its
# own uv venv, --extra cuda), reused because Lotka-Volterra needs no external
# dataset (trajectories are generated in-memory via scipy).
#
#SBATCH --job-name=lv-cncp
#SBATCH --partition=GPU-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=16G
#SBATCH --time=06:00:00
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
# needs them on LD_LIBRARY_PATH (same workaround as person_activity.sbatch).
VENV_SITE="$(uv run python -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)"
if [[ -d "${VENV_SITE}/nvidia" ]]; then
    CUDA_LD="$(find "${VENV_SITE}/nvidia" -maxdepth 2 -name 'lib' -type d 2>/dev/null | tr '\n' ':')"
    export LD_LIBRARY_PATH="${CUDA_LD}${LD_LIBRARY_PATH:-}"
fi
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=1
export PYTHONUNBUFFERED=1

# wandb: offline on the compute node (no worker internet). Runs land under
# outputs/wandb/, are fetched back by fetch_lotka_volterra.sh, and uploaded
# locally via cluster/wandb_sync.sh. Opt in with LV_WANDB=1 at submit time;
# default off keeps existing runs untouched.
export WANDB_MODE=offline
export WANDB_DIR="$PWD/outputs/wandb"
export WANDB_SILENT=true
mkdir -p "$WANDB_DIR"
USE_WANDB="${LV_WANDB:-0}"

# Matrix (override via env at submit time, e.g. LV_CELLS="cfc_lrc"):
EPOCHS="${LV_EPOCHS:-300}"
CELLS="${LV_CELLS:-cfc_lrc gru}"
# dense/ncp/cncp = pure topology (no reference frame); tbt_cncp_noloc isolates the
# location benefit; tbt_cncp_concat is the strongest reference-frame arm.
WIRINGS="${LV_WIRINGS:-dense ncp cncp tbt_cncp_noloc tbt_cncp_concat}"
SEEDS="${LV_SEEDS:-0 1 2}"
MAXPAR="${LV_MAXPAR:-6}"
PLOT_CELL="${LV_PLOT_CELL:-cfc_lrc}"
# System / resolution / output dir (predator_prey -> results/lotka_volterra;
# duffing -> results/lotka_volterra_duffing, submitted as a second job).
SYSTEM="${LV_SYSTEM:-periodic_predator_prey}"
SEQLEN="${LV_SEQLEN:-128}"
OUTDIR="${LV_OUTDIR:-results/lotka_volterra}"

mkdir -p outputs/slurm "$OUTDIR"

run_one() {
    local cell="$1" wiring="$2" seed="$3"
    local wandb_flag=""
    [[ "${USE_WANDB:-0}" == "1" ]] && wandb_flag="--wandb"
    echo "[start] ${cell}_${wiring}_seed${seed} $(date -u +%FT%TZ)"
    uv run python experiments/run_lotka_volterra_benchmark.py \
        --wiring "$wiring" --cell "$cell" --seed "$seed" --epochs "$EPOCHS" \
        --system "$SYSTEM" --seq-len "$SEQLEN" --outdir "$OUTDIR" $wandb_flag
    echo "[done ] ${cell}_${wiring}_seed${seed} (exit $?)"
}
export -f run_one
export EPOCHS SYSTEM SEQLEN OUTDIR USE_WANDB

JOBS=""
for cell in $CELLS; do
    for wiring in $WIRINGS; do
        for seed in $SEEDS; do
            JOBS+="${cell} ${wiring} ${seed}"$'\n'
        done
    done
done

echo "=== ${SYSTEM} topology-vs-tbt: $(printf '%s' "$JOBS" | grep -c .) runs, par ${MAXPAR}, ${EPOCHS} epochs, seq_len ${SEQLEN}, -> ${OUTDIR}, on $(hostname) ==="
printf '%s' "$JOBS" | grep . | xargs -P "$MAXPAR" -n 3 bash -c 'run_one "$1" "$2" "$3"' _
echo "=== all runs done $(date -u +%FT%TZ) ==="
ls -la "$OUTDIR"/*.json

echo "=== rendering plots (cell ${PLOT_CELL}, ${SYSTEM}) ==="
uv run python experiments/plot_lotka_volterra.py \
    --indir "$OUTDIR" --cell "$PLOT_CELL" --seed 0 --n-examples 4
ls -la "$OUTDIR"/*.png
echo "=== plots done $(date -u +%FT%TZ) ==="
