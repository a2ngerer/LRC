#!/usr/bin/env bash
# Single SLURM job for the VOTING COMMITTEE benchmark: the consensus study of
# Chapter 5 (identical views, per-column faults and coordinate noise, grid
# validation; experiments/run_committee_benchmark.py, issue #55) by default;
# CO_VIEWMODE=partition|noisy and CO_DROPFRACS / CO_TRAINDROP / CO_TRAINNOISE /
# CO_TESTCORRUPT drive the exploratory July 2026 path (not part of the suite).
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
# --output/--error are passed on the sbatch command line by cluster/_run.sh
# (<run folder>/logs/); RUN_DIR is exported by the launcher.
#SBATCH --open-mode=append
#SBATCH --requeue
set -euo pipefail
# Run folder (cluster/_run.sh): results, logs, figures and wandb live here,
# never under the code tree (which is rsync --delete-d).
RUN_DIR="${RUN_DIR:?RUN_DIR not set: submit via cluster/submit_*.sh}"
mkdir -p "$RUN_DIR/results" "$RUN_DIR/figures" "$RUN_DIR/wandb"

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
VIEWMODE="${CO_VIEWMODE:-identical}"
DROPFRACS="${CO_DROPFRACS:-}"        # exploratory sweep, e.g. 0.0,0.15,0.3,0.45; empty = off
TRAINDROP="${CO_TRAINDROP:-0.0}"     # >0 -> train-time sensor-dropout baseline
TRAINNOISE="${CO_TRAINNOISE:-0.0}"   # >0 -> train-time Gaussian-noise baseline (7d)
TESTCORRUPT="${CO_TESTCORRUPT:-dropout}"   # dropout | noise (7d test corruption)
NOISESIGMAS="${CO_NOISESIGMAS:-0.0,0.25,0.5,1.0}"
# Every arm is sized on its own width knob to this EFFECTIVE parameter budget
# (thesis protocol: 30k for the consensus study); weight sharing keeps the count
# independent of K.
PARAM_BUDGET="${CO_PARAM_BUDGET:-30000}"   # 0 = no --param-budget (use --size via CO_EXTRA)
MAXPAR="${CO_MAXPAR:-4}"

OUTDIR="$RUN_DIR/results"
DATA_PATH="$PWD/data/person/ConfLongDemo_JSI.txt"
EXTRA="${CO_EXTRA:-}"   # extra runner flags, e.g. "--vote-lambda 0 --lr 3e-3 --clipnorm 1"
# CfC-family backbone depth for every arm: 1 (suite protocol) until the
# backbone default of issue #21 is merged (CO_BACKBONE=0: no backbone); results
# get a _bb<N> filename suffix.
BACKBONE="${CO_BACKBONE:-1}"
[[ -n "${BACKBONE}" ]] && EXTRA="--backbone-layers ${BACKBONE} ${EXTRA}"
# wandb: offline on the compute node (no worker internet), into the run folder;
# opt in with CO_WANDB=1 at submit time, upload the fetched run folder with
# cluster/wandb_sync.sh.
export WANDB_MODE=offline WANDB_DIR="$RUN_DIR/wandb" WANDB_SILENT=true
[[ "${CO_WANDB:-0}" == "1" ]] && EXTRA="${EXTRA} --wandb"

run_one() {
    local cell="$1" wiring="$2" K="$3" seed="$4" budget_flag="" rc
    [[ "${PARAM_BUDGET}" != "0" ]] && budget_flag="--param-budget ${PARAM_BUDGET}"
    echo "[start] ${cell}_${wiring}_K${K}_pb${PARAM_BUDGET}_seed${seed} $(date -u +%FT%TZ)"
    # shellcheck disable=SC2086  # flag lists are split on purpose
    uv run python experiments/run_committee_benchmark.py \
        --wiring "$wiring" --n-columns "$K" $budget_flag \
        --cell "$cell" --seed "$seed" --epochs "$EPOCHS" --seq-len "$SEQLEN" \
        --batch-size "$BATCH" --data-path "$DATA_PATH" \
        --view-mode "$VIEWMODE" --drop-fracs "$DROPFRACS" \
        --train-drop "$TRAINDROP" --train-noise "$TRAINNOISE" \
        --test-corruption "$TESTCORRUPT" --noise-sigmas "$NOISESIGMAS" \
        --outdir "$OUTDIR" $EXTRA
    rc=$?
    echo "[done ] ${cell}_${wiring}_K${K}_pb${PARAM_BUDGET}_seed${seed} (exit $rc)"
    return $rc
}
export -f run_one
export EPOCHS SEQLEN BATCH VIEWMODE DROPFRACS PARAM_BUDGET TRAINDROP \
       TRAINNOISE TESTCORRUPT NOISESIGMAS OUTDIR EXTRA DATA_PATH

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
N_FITS=$(printf '%s' "$JOBS" | grep -c .)
printf '%s' "$JOBS" | grep . | xargs -P "$MAXPAR" -n 4 bash -c 'run_one "$1" "$2" "$3" "$4"' _ || echo "WARNING: at least one fit exited non-zero (xargs exit $?)"
echo "=== all runs done $(date -u +%FT%TZ) ==="
N_JSON=$(find "$OUTDIR" -maxdepth 1 -name '*.json' | wc -l)
echo "result files: ${N_JSON} of ${N_FITS} expected"
ls -la "$OUTDIR"/*.json

echo "=== rendering plot ==="
uv run python experiments/plot_committee.py --indir "$OUTDIR" || echo "plot step failed (non-fatal)"
mv "$OUTDIR"/*.png "$RUN_DIR/figures/" 2>/dev/null || true
ls -la "$RUN_DIR/figures" 2>/dev/null || true
echo "=== done $(date -u +%FT%TZ) ==="
[[ "$N_JSON" -ge "$N_FITS" ]] || { echo "FATAL: ${N_FITS} fits expected, ${N_JSON} result files written" >&2; exit 1; }
