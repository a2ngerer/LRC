#!/usr/bin/env bash
# Single SLURM job for the tbt_cNCP multi-column VOTING benchmark (TBT ingredient
# 3 / hypothesis H2: do K weight-shared columns that vote reach a given accuracy
# in fewer glimpses-per-column than one column?). Runs K = 1, 2, 3 x seeds in ONE
# job with bounded parallelism on one GPU (tiny models), then renders the
# sample-efficiency plot. Submitted by cluster/submit_voting.sh from the ISOLATED
# ~/thesis-person repo (its own uv venv). No external dataset -- objects are
# generated in-memory (numpy), so nothing to rsync.
#
#SBATCH --job-name=tbt-vote
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

EPOCHS="${VOTE_EPOCHS:-70}"
CELL="${VOTE_CELL:-cfc_lrc}"
KCOLS="${VOTE_KCOLS:-1 2 3}"
SEEDS="${VOTE_SEEDS:-0 1 2}"
# Location mechanism per column: concat (baseline) vs film (Iteration 1 winner).
MODES="${VOTE_MODES:-concat film}"
MAXPAR="${VOTE_MAXPAR:-4}"
OUTDIR="${VOTE_OUTDIR:-results/active_sensing_voting}"

mkdir -p outputs/slurm "$OUTDIR"

run_one() {
    local kcol="$1" seed="$2" mode="$3"
    echo "[start] k${kcol}_${mode}_seed${seed} $(date -u +%FT%TZ)"
    uv run python experiments/run_voting_benchmark.py \
        --n-columns "$kcol" --cell "$CELL" --seed "$seed" --epochs "$EPOCHS" \
        --location-mode "$mode" --outdir "$OUTDIR"
    echo "[done ] k${kcol}_${mode}_seed${seed} (exit $?)"
}
export -f run_one
export EPOCHS CELL OUTDIR

JOBS=""
for kcol in $KCOLS; do
    for mode in $MODES; do
        for seed in $SEEDS; do
            JOBS+="${kcol} ${seed} ${mode}"$'\n'
        done
    done
done

echo "=== tbt_cNCP voting: $(printf '%s' "$JOBS" | grep -c .) runs, par ${MAXPAR}, ${EPOCHS} epochs, modes '${MODES}', on $(hostname) ==="
printf '%s' "$JOBS" | grep . | xargs -P "$MAXPAR" -n 3 bash -c 'run_one "$1" "$2" "$3"' _
echo "=== all runs done $(date -u +%FT%TZ) ==="
ls -la "$OUTDIR"/*.json

echo "=== rendering plot ==="
uv run python experiments/plot_voting.py --indir "$OUTDIR"
ls -la "$OUTDIR"/*.png
echo "=== done $(date -u +%FT%TZ) ==="
