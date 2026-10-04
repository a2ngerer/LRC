#!/usr/bin/env bash
# Single SLURM job: train cfc_lrc (NCP + cNCP) on person-activity and render the
# prediction plots (confusion matrices + predicted-vs-true activity overlays).
# Runs in the isolated ~/thesis-person repo (own uv venv, --extra cuda).
#
#SBATCH --job-name=person-pred
#SBATCH --partition=GPU-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --requeue
set -euo pipefail

export TMPDIR="$HOME/tmp"; export TMP="$TMPDIR"; export TEMP="$TMPDIR"
mkdir -p "$TMPDIR"
export PATH="$HOME/.local/bin:$PATH"

VENV_SITE="$(uv run python -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)"
if [[ -d "${VENV_SITE}/nvidia" ]]; then
    CUDA_LD="$(find "${VENV_SITE}/nvidia" -maxdepth 2 -name 'lib' -type d 2>/dev/null | tr '\n' ':')"
    export LD_LIBRARY_PATH="${CUDA_LD}${LD_LIBRARY_PATH:-}"
fi
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=1
export PYTHONUNBUFFERED=1

EPOCHS="${PA_EPOCHS:-30}"
SEED="${PA_SEED:-0}"
mkdir -p outputs/slurm results/person_activity_cluster

echo "=== person-activity predictions: cfc_lrc ncp+cncp, ${EPOCHS} epochs, on $(hostname) $(date -u +%FT%TZ) ==="
uv run python experiments/plot_person_activity_predictions.py --epochs "$EPOCHS" --seed "$SEED"
echo "=== done $(date -u +%FT%TZ) ==="
ls -la results/person_activity_cluster/*.png
