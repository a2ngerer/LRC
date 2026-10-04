#!/usr/bin/env bash
# Single SLURM job for the partial-view VOTING COMMITTEE benchmark (Iteration 7):
# do K weight-shared columns, each seeing a PART of the input, degrade more
# gracefully under sensor dropout than a single monolith at matched parameters --
# and does a cortical (cNCP) column beat a plain dense cell as the weak learner?
# Runs the matrix (wirings x K x seeds) in ONE job on one GPU, then renders the
# plot. Submitted by cluster/submit_committee.sh from the ISOLATED ~/thesis-person
# repo (own uv venv). Reads the person-activity CSV rsynced under data/person/.
#
#SBATCH --job-name=tbt-committee
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

EPOCHS="${CO_EPOCHS:-40}"
CELL="${CO_CELL:-cfc_lrc}"
WIRINGS="${CO_WIRINGS:-cncp ncp dense}"
KS="${CO_KS:-1 2 4}"
SEEDS="${CO_SEEDS:-0 1 2}"
SEQLEN="${CO_SEQLEN:-32}"
BATCH="${CO_BATCH:-128}"     # larger batch -> fewer, bigger steps (cell is launch-bound)
VIEWMODE="${CO_VIEWMODE:-partition}"
DROPFRACS="${CO_DROPFRACS:-0.0,0.15,0.3,0.45}"
TRAINDROP="${CO_TRAINDROP:-0.0}"     # >0 -> train-time sensor-dropout baseline
TRAINNOISE="${CO_TRAINNOISE:-0.0}"   # >0 -> train-time Gaussian-noise baseline (7d)
TESTCORRUPT="${CO_TESTCORRUPT:-dropout}"   # dropout | noise (7d test corruption)
NOISESIGMAS="${CO_NOISESIGMAS:-0.0,0.25,0.5,1.0}"
# Every arm is sized on its own width knob to this EFFECTIVE parameter budget
# (thesis protocol: 30k for the consensus study); weight sharing keeps the count
# independent of K.
PARAM_BUDGET="${CO_PARAM_BUDGET:-30000}"
MAXPAR="${CO_MAXPAR:-4}"

mkdir -p outputs/slurm results/committee

run_one() {
    local cell="$1" wiring="$2" K="$3" seed="$4"
    echo "[start] ${cell}_${wiring}_K${K}_pb${PARAM_BUDGET}_seed${seed} $(date -u +%FT%TZ)"
    uv run python experiments/run_committee_benchmark.py \
        --wiring "$wiring" --n-columns "$K" --param-budget "$PARAM_BUDGET" \
        --cell "$cell" --seed "$seed" --epochs "$EPOCHS" --seq-len "$SEQLEN" \
        --batch-size "$BATCH" \
        --view-mode "$VIEWMODE" --drop-fracs "$DROPFRACS" \
        --train-drop "$TRAINDROP" --train-noise "$TRAINNOISE" \
        --test-corruption "$TESTCORRUPT" --noise-sigmas "$NOISESIGMAS"
    echo "[done ] ${cell}_${wiring}_K${K}_pb${PARAM_BUDGET}_seed${seed} (exit $?)"
}
export -f run_one
export EPOCHS SEQLEN BATCH VIEWMODE DROPFRACS PARAM_BUDGET TRAINDROP \
       TRAINNOISE TESTCORRUPT NOISESIGMAS

JOBS=""
for cell in $CELL; do
    for wiring in $WIRINGS; do
        for K in $KS; do
            for seed in $SEEDS; do
                JOBS+="${cell} ${wiring} ${K} ${seed}"$'\n'
            done
        done
    done
done

echo "=== committee: $(printf '%s' "$JOBS" | grep -c .) runs, par ${MAXPAR}, ${EPOCHS} epochs, mode ${VIEWMODE}, on $(hostname) ==="
printf '%s' "$JOBS" | grep . | xargs -P "$MAXPAR" -n 4 bash -c 'run_one "$1" "$2" "$3" "$4"' _
echo "=== all runs done $(date -u +%FT%TZ) ==="
ls -la results/committee/*.json

echo "=== rendering plot ==="
uv run python experiments/plot_committee.py --indir results/committee || echo "plot step failed (non-fatal)"
ls -la results/committee/*.png 2>/dev/null || true
echo "=== done $(date -u +%FT%TZ) ==="
