#!/usr/bin/env bash
# Single SLURM job for the cNCP-vs-NCP Person-Activity benchmark.
# Runs the whole small matrix (cells x {ncp,cncp} x seeds) in ONE job with
# bounded parallelism on one GPU -- the models are tiny, so an array would be
# pure overhead. Submitted by cluster/submit_person_activity.sh from the
# ISOLATED ~/thesis-person repo (its own uv venv, --extra cuda), so it never
# perturbs the ~/thesis-benchmark or ~/thesis-mujoco environments.
#
#SBATCH --job-name=person-cncp
#SBATCH --partition=GPU-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=16G
#SBATCH --time=01:30:00
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
# needs them on LD_LIBRARY_PATH (same workaround as mujoco_array.sbatch).
VENV_SITE="$(uv run python -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)"
if [[ -d "${VENV_SITE}/nvidia" ]]; then
    CUDA_LD="$(find "${VENV_SITE}/nvidia" -maxdepth 2 -name 'lib' -type d 2>/dev/null | tr '\n' ':')"
    export LD_LIBRARY_PATH="${CUDA_LD}${LD_LIBRARY_PATH:-}"
fi
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=1
export PYTHONUNBUFFERED=1

# wandb: offline on the compute node (no worker internet). Runs land under
# outputs/wandb/, are fetched back by fetch_person_activity.sh, and uploaded
# locally via cluster/wandb_sync.sh. Opt in with PA_WANDB=1 at submit time;
# default off keeps existing runs untouched.
export WANDB_MODE=offline
export WANDB_DIR="$PWD/outputs/wandb"
export WANDB_SILENT=true
mkdir -p "$WANDB_DIR"
USE_WANDB="${PA_WANDB:-0}"

# Matrix (override via env at submit time, e.g. PA_CELLS="cfc_lrc"):
EPOCHS="${PA_EPOCHS:-50}"
CELLS="${PA_CELLS:-cfc_lrc gru}"
SEEDS="${PA_SEEDS:-0 1 2}"
MAXPAR="${PA_MAXPAR:-3}"
DATA_PATH="$PWD/data/person/ConfLongDemo_JSI.txt"

mkdir -p outputs/slurm results/person_activity

if [[ ! -f "$DATA_PATH" ]]; then
    echo "FATAL: dataset not found at $DATA_PATH (rsync must include data/)." >&2
    exit 1
fi

run_one() {
    local cell="$1" wiring="$2" seed="$3"
    local wandb_flag=""
    [[ "${USE_WANDB:-0}" == "1" ]] && wandb_flag="--wandb"
    echo "[start] ${cell}_${wiring}_seed${seed} $(date -u +%FT%TZ)"
    uv run python experiments/run_person_activity_benchmark.py \
        --wiring "$wiring" --cell "$cell" --seed "$seed" --epochs "$EPOCHS" \
        --data-path "$DATA_PATH" $wandb_flag
    echo "[done ] ${cell}_${wiring}_seed${seed} (exit $?)"
}
export -f run_one
export EPOCHS DATA_PATH USE_WANDB

JOBS=""
for cell in $CELLS; do
    for wiring in ncp cncp; do
        for seed in $SEEDS; do
            JOBS+="${cell} ${wiring} ${seed}"$'\n'
        done
    done
done

echo "=== person-activity cNCP-vs-NCP: $(printf '%s' "$JOBS" | grep -c .) runs, par ${MAXPAR}, ${EPOCHS} epochs, on $(hostname) ==="
printf '%s' "$JOBS" | grep . | xargs -P "$MAXPAR" -n 3 bash -c 'run_one "$1" "$2" "$3"' _
echo "=== all runs done $(date -u +%FT%TZ) ==="
ls -la results/person_activity/*.json
