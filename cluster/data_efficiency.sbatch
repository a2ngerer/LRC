#!/usr/bin/env bash
# Single SLURM job for the DATA-EFFICIENCY benchmark (Iteration 8): does the sparse
# bio-inspired cNCP wiring generalise better from LESS data than dense/NCP at
# MATCHED parameters (a steeper learning curve)? Runs the matrix
# (wirings x train-fractions x seeds) in ONE job on one GPU, then renders the
# learning-curve plot. Submitted by cluster/submit_data_efficiency.sh from the
# ISOLATED ~/thesis-person repo. Reads the person-activity CSV under data/person/.
#
#SBATCH --job-name=tbt-dataeff
#SBATCH --partition=GPU-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=16G
#SBATCH --time=03:00:00
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

EPOCHS="${DE_EPOCHS:-60}"
CELL="${DE_CELL:-cfc_lrc}"
WIRINGS="${DE_WIRINGS:-cncp ncp dense}"
FRACS="${DE_FRACS:-0.05 0.1 0.25 0.5 1.0}"
SEEDS="${DE_SEEDS:-0 1 2 3 4}"
SIZE="${DE_SIZE:-64}"
SEQLEN="${DE_SEQLEN:-32}"
BATCH="${DE_BATCH:-128}"
MAXPAR="${DE_MAXPAR:-6}"

mkdir -p outputs/slurm results/data_efficiency

run_one() {
    local cell="$1" wiring="$2" frac="$3" seed="$4"
    echo "[start] ${cell}_${wiring}_f${frac}_seed${seed} $(date -u +%FT%TZ)"
    uv run python experiments/run_data_efficiency_benchmark.py \
        --wiring "$wiring" --frac "$frac" --cell "$cell" --seed "$seed" \
        --size "$SIZE" --epochs "$EPOCHS" --seq-len "$SEQLEN" --batch-size "$BATCH"
    echo "[done ] ${cell}_${wiring}_f${frac}_seed${seed} (exit $?)"
}
export -f run_one
export EPOCHS SIZE SEQLEN BATCH

JOBS=""
for cell in $CELL; do
    for wiring in $WIRINGS; do
        for frac in $FRACS; do
            for seed in $SEEDS; do
                JOBS+="${cell} ${wiring} ${frac} ${seed}"$'\n'
            done
        done
    done
done

echo "=== data-efficiency: $(printf '%s' "$JOBS" | grep -c .) runs, par ${MAXPAR}, ${EPOCHS} epochs, on $(hostname) ==="
printf '%s' "$JOBS" | grep . | xargs -P "$MAXPAR" -n 4 bash -c 'run_one "$1" "$2" "$3" "$4"' _
echo "=== all runs done $(date -u +%FT%TZ) ==="
ls -la results/data_efficiency/*.json | tail

echo "=== rendering plot ==="
uv run python experiments/plot_data_efficiency.py --indir results/data_efficiency || echo "plot step failed (non-fatal)"
ls -la results/data_efficiency/*.png 2>/dev/null || true
echo "=== done $(date -u +%FT%TZ) ==="
