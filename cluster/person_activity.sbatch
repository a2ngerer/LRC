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
# --output/--error are passed on the sbatch command line by cluster/_run.sh
# (<run folder>/logs/); RUN_DIR is exported by the launcher.
#SBATCH --open-mode=append
#SBATCH --requeue
set -euo pipefail
# Run folder (cluster/_run.sh): results, logs, figures and wandb live here,
# never under the code tree (which is rsync --delete-d).
RUN_DIR="${RUN_DIR:?RUN_DIR not set: submit via cluster/submit_*.sh}"
mkdir -p "$RUN_DIR/results" "$RUN_DIR/figures" "$RUN_DIR/wandb"

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
# $RUN_DIR/wandb/, come back with the fetched run folder and are uploaded
# locally via cluster/wandb_sync.sh <run folder>. Opt in with PA_WANDB=1 at submit time;
# default off keeps existing runs untouched.
export WANDB_MODE=offline
export WANDB_DIR="$RUN_DIR/wandb"
export WANDB_SILENT=true
mkdir -p "$WANDB_DIR"
USE_WANDB="${PA_WANDB:-0}"

# Matrix (override via env at submit time, e.g. PA_CELLS="cfc_lrc"):
EPOCHS="${PA_EPOCHS:-50}"
CELLS="${PA_CELLS:-cfc_lrc gru}"
WIRINGS="${PA_WIRINGS:-ncp cncp}"          # also: dense dense3
SEEDS="${PA_SEEDS:-0 1 2}"
MAXPAR="${PA_MAXPAR:-3}"
# Every arm is sized on its own width knob to this EFFECTIVE parameter budget
# (thesis protocol: 30k for Person Activity).
PARAM_BUDGET="${PA_PARAM_BUDGET:-30000}"   # 0 = no --param-budget (use --size via PA_EXTRA)
DATA_PATH="$PWD/data/person/ConfLongDemo_JSI.txt"

OUTDIR="$RUN_DIR/results"
LR="${PA_LR:-1e-3}"
# CfC-family backbone depth for every arm (V2s control: PA_BACKBONE=1); empty = flag
# not passed, per-wiring defaults unchanged; results get a _bb<N> filename suffix.
BACKBONE="${PA_BACKBONE:-}"
# Grouped recording split (default) or the literature-anchor window split;
# PA_SPLIT_SEED fixes the grouped recording assignment for the whole suite.
SPLIT="${PA_SPLIT:-grouped}"               # also: random_window
SPLIT_SEED="${PA_SPLIT_SEED:-0}"
EXTRA="${PA_EXTRA:-}"   # extra runner flags, e.g. "--columns 4 --partition region --coupling lateral"

if [[ ! -f "$DATA_PATH" ]]; then
    echo "FATAL: dataset not found at $DATA_PATH (rsync must include data/)." >&2
    exit 1
fi

# shellcheck disable=SC2086  # flag lists are split on purpose
run_one() {
    local cell="$1" wiring="$2" seed="$3"
    local wandb_flag="" bb_flag="" budget_flag=""
    [[ "${PARAM_BUDGET}" != "0" ]] && budget_flag="--param-budget ${PARAM_BUDGET}"
    [[ -n "${BACKBONE:-}" ]] && bb_flag="--backbone-layers ${BACKBONE}"
    [[ "${USE_WANDB:-0}" == "1" ]] && wandb_flag="--wandb"
    echo "[start] ${cell}_${wiring}_seed${seed} $(date -u +%FT%TZ)"
    uv run python experiments/run_person_activity_benchmark.py \
        --wiring "$wiring" --cell "$cell" --seed "$seed" --epochs "$EPOCHS" \
        $budget_flag --data-path "$DATA_PATH" \
        --split "$SPLIT" --split-seed "$SPLIT_SEED" \
        --outdir "$OUTDIR" --lr "$LR" $bb_flag $EXTRA $wandb_flag
    local rc=$?
    echo "[done ] ${cell}_${wiring}_seed${seed} (exit $rc)"
    return $rc
}
export -f run_one
export EPOCHS DATA_PATH USE_WANDB PARAM_BUDGET OUTDIR LR BACKBONE SPLIT SPLIT_SEED EXTRA

JOBS=""
for cell in $CELLS; do
    for wiring in $WIRINGS; do
        for seed in $SEEDS; do
            JOBS+="${cell} ${wiring} ${seed}"$'\n'
        done
    done
done

echo "=== person-activity cNCP-vs-NCP: $(printf '%s' "$JOBS" | grep -c .) runs, par ${MAXPAR}, ${EPOCHS} epochs, on $(hostname) ==="
N_FITS=$(printf '%s' "$JOBS" | grep -c .)
printf '%s' "$JOBS" | grep . | xargs -P "$MAXPAR" -n 3 bash -c 'run_one "$1" "$2" "$3"' _ || echo "WARNING: at least one fit exited non-zero (xargs exit $?)"
echo "=== all runs done $(date -u +%FT%TZ) ==="
N_JSON=$(find "$OUTDIR" -maxdepth 1 -name '*.json' | wc -l)
echo "result files: ${N_JSON} of ${N_FITS} expected"
ls -la "$OUTDIR"/*.json
[[ "$N_JSON" -ge "$N_FITS" ]] || { echo "FATAL: ${N_FITS} fits expected, ${N_JSON} result files written" >&2; exit 1; }
