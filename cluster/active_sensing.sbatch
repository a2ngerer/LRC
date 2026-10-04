#!/usr/bin/env bash
# Single SLURM job for the tbt_cNCP active-sensing benchmark, then renders the
# accuracy-vs-glimpses plot. Whole matrix (wirings x cells x seeds) in ONE job
# on one GPU (tiny models). Submitted by cluster/submit_active_sensing.sh from
# the ISOLATED ~/thesis-person repo (own uv venv). No external dataset -- the
# objects are generated in-memory (numpy), so nothing to rsync.
#
#SBATCH --job-name=tbt-as
#SBATCH --partition=GPU-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=16G
#SBATCH --time=02:30:00
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

EPOCHS="${AS_EPOCHS:-80}"
CELLS="${AS_CELLS:-cfc_lrc}"
WIRINGS="${AS_WIRINGS:-tbt_cncp tbt_cncp_noloc ncp dense}"
SEEDS="${AS_SEEDS:-0 1 2}"
MAXPAR="${AS_MAXPAR:-6}"

mkdir -p outputs/slurm results/active_sensing

run_one() {
    local cell="$1" wiring="$2" seed="$3"
    echo "[start] ${cell}_${wiring}_seed${seed} $(date -u +%FT%TZ)"
    uv run python experiments/run_active_sensing_benchmark.py \
        --wiring "$wiring" --cell "$cell" --seed "$seed" --epochs "$EPOCHS"
    echo "[done ] ${cell}_${wiring}_seed${seed} (exit $?)"
}
export -f run_one
export EPOCHS

JOBS=""
for cell in $CELLS; do
    for wiring in $WIRINGS; do
        for seed in $SEEDS; do
            JOBS+="${cell} ${wiring} ${seed}"$'\n'
        done
    done
done

echo "=== tbt_cNCP active-sensing: $(printf '%s' "$JOBS" | grep -c .) runs, par ${MAXPAR}, ${EPOCHS} epochs, on $(hostname) ==="
printf '%s' "$JOBS" | grep . | xargs -P "$MAXPAR" -n 3 bash -c 'run_one "$1" "$2" "$3"' _
echo "=== all runs done $(date -u +%FT%TZ) ==="
ls -la results/active_sensing/*.json

echo "=== rendering plot ==="
uv run python experiments/plot_active_sensing.py --indir results/active_sensing
ls -la results/active_sensing/*.png
echo "=== done $(date -u +%FT%TZ) ==="
