#!/usr/bin/env bash
# SLURM array job for the MuJoCo PPO control benchmark.
# One array task = one (model, seed) run, selected by --index $SLURM_ARRAY_TASK_ID.
# Submitted by cluster/submit_mujoco.sh from the ISOLATED ~/thesis-mujoco repo
# (its own uv venv with the --extra cuda --extra rl deps), so it never touches the
# ~/thesis-benchmark environment that the Neural-ODE benchmark jobs run from.
#
#SBATCH --job-name=mujoco-rl
#SBATCH --partition=GPU-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/slurm/%x-%A_%a.out
#SBATCH --error=outputs/slurm/%x-%A_%a.err
#SBATCH --open-mode=append
#SBATCH --requeue
set -euo pipefail

# Virtual nodes (i-*) have a non-writable /tmp -> redirect temp into $HOME.
export TMPDIR="$HOME/tmp"; export TMP="$TMPDIR"; export TEMP="$TMPDIR"
mkdir -p "$TMPDIR"
export PATH="$HOME/.local/bin:$PATH"

# tensorflow[and-cuda] ships CUDA libs under .venv/.../site-packages/nvidia/*/lib;
# TF 2.15 needs them on LD_LIBRARY_PATH (same workaround as benchmark_array.sbatch).
VENV_SITE="$(uv run python -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)"
if [[ -d "${VENV_SITE}/nvidia" ]]; then
    CUDA_LD="$(find "${VENV_SITE}/nvidia" -maxdepth 2 -name 'lib' -type d 2>/dev/null | tr '\n' ':')"
    export LD_LIBRARY_PATH="${CUDA_LD}${LD_LIBRARY_PATH:-}"
fi
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=1
export PYTHONUNBUFFERED=1

# Headless MuJoCo offscreen rendering via the NVIDIA EGL backend. Best-effort:
# run_mujoco_benchmark wraps rendering in try/except and always writes the policy
# checkpoint, so videos can be re-rendered locally if EGL is unavailable here.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p outputs/slurm results/mujoco

echo "=== mujoco task ${SLURM_ARRAY_TASK_ID} on $(hostname) $(date -u +%FT%TZ) ==="
echo "    runner args: $*"
uv run python experiments/run_mujoco_benchmark.py --index "${SLURM_ARRAY_TASK_ID}" "$@"
echo "=== done $(date -u +%FT%TZ) ==="
