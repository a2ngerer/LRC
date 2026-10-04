#!/usr/bin/env bash
# Single SLURM job for the ACTIVE-glimpse sensorimotor benchmark (Iteration 5):
# does a column that STEERS its own glimpses (L5 motor hub) recognise objects in
# fewer glimpses than random? Runs the matrix (wirings x policies x seeds) in ONE
# job on one GPU (tiny models), then renders the plot. Submitted by
# cluster/submit_active_glimpse.sh from the ISOLATED ~/thesis-person repo (own uv
# venv). No external dataset -- objects generated in-memory (numpy).
#
#SBATCH --job-name=tbt-aglimpse
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

EPOCHS="${AG_EPOCHS:-40}"
CELL="${AG_CELL:-cfc_lrc}"
WIRINGS="${AG_WIRINGS:-tbt_cncp dense}"
POLICIES="${AG_POLICIES:-active random}"
SEEDS="${AG_SEEDS:-0 1 2}"
NCLASSES="${AG_NCLASSES:-8}"
SEQLEN="${AG_SEQLEN:-10}"
EXPLORE="${AG_EXPLORE:-0.5}"
MAXPAR="${AG_MAXPAR:-4}"

mkdir -p outputs/slurm results/active_glimpse

run_one() {
    local cell="$1" wiring="$2" policy="$3" seed="$4"
    echo "[start] ${cell}_${wiring}_${policy}_seed${seed} $(date -u +%FT%TZ)"
    uv run python experiments/run_active_glimpse_benchmark.py \
        --wiring "$wiring" --policy "$policy" --cell "$cell" --seed "$seed" \
        --epochs "$EPOCHS" --n-classes "$NCLASSES" --seq-len "$SEQLEN" \
        --explore "$EXPLORE"
    echo "[done ] ${cell}_${wiring}_${policy}_seed${seed} (exit $?)"
}
export -f run_one
export EPOCHS NCLASSES SEQLEN EXPLORE

JOBS=""
for cell in $CELL; do
    for wiring in $WIRINGS; do
        for policy in $POLICIES; do
            for seed in $SEEDS; do
                JOBS+="${cell} ${wiring} ${policy} ${seed}"$'\n'
            done
        done
    done
done

echo "=== active-glimpse: $(printf '%s' "$JOBS" | grep -c .) runs, par ${MAXPAR}, ${EPOCHS} epochs, ${NCLASSES} classes, on $(hostname) ==="
printf '%s' "$JOBS" | grep . | xargs -P "$MAXPAR" -n 4 bash -c 'run_one "$1" "$2" "$3" "$4"' _
echo "=== all runs done $(date -u +%FT%TZ) ==="
ls -la results/active_glimpse/*.json

echo "=== rendering plot ==="
uv run python experiments/plot_active_glimpse.py --indir results/active_glimpse
ls -la results/active_glimpse/*.png
echo "=== done $(date -u +%FT%TZ) ==="
