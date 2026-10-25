#!/usr/bin/env bash
# Single SLURM job for the cNCP-vs-NCP-vs-dense predator-prey (Lotka-Volterra)
# sequence-rollout benchmark, then renders the trajectory-overlay plots.
# Runs the whole small matrix (cells x {dense,ncp,cncp} x seeds) in ONE job with
# bounded parallelism on one GPU -- the models are tiny. Submitted by
# cluster/submit_lotka_volterra.sh from the ISOLATED ~/thesis-person repo (its
# own uv venv, --extra cuda), reused because Lotka-Volterra needs no external
# dataset (trajectories are generated in-memory via scipy).
#
#SBATCH --job-name=lv-cncp
#SBATCH --partition=GPU-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=16G
#SBATCH --time=06:00:00
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
# needs them on LD_LIBRARY_PATH (same workaround as person_activity.sbatch).
VENV_SITE="$(uv run python -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)"
if [[ -d "${VENV_SITE}/nvidia" ]]; then
    CUDA_LD="$(find "${VENV_SITE}/nvidia" -maxdepth 2 -name 'lib' -type d 2>/dev/null | tr '\n' ':')"
    export LD_LIBRARY_PATH="${CUDA_LD}${LD_LIBRARY_PATH:-}"
fi
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=1
export PYTHONUNBUFFERED=1

# wandb: offline on the compute node (no worker internet). Runs land under
# outputs/wandb/, are fetched back by fetch_lotka_volterra.sh, and uploaded
# locally via cluster/wandb_sync.sh. Opt in with LV_WANDB=1 at submit time;
# default off keeps existing runs untouched.
export WANDB_MODE=offline
export WANDB_DIR="$RUN_DIR/wandb"
export WANDB_SILENT=true
mkdir -p "$WANDB_DIR"
USE_WANDB="${LV_WANDB:-0}"

# Matrix (override via env at submit time, e.g. LV_CELLS="cfc_lrc"):
EPOCHS="${LV_EPOCHS:-300}"
CELLS="${LV_CELLS:-cfc_lrc gru}"
# dense/dense3/ncp/cncp = pure topology (no reference frame); tbt_cncp_noloc
# isolates the location benefit; tbt_cncp_concat is the strongest reference-frame
# arm.
WIRINGS="${LV_WIRINGS:-dense dense3 ncp cncp tbt_cncp_noloc tbt_cncp_concat}"
SEEDS="${LV_SEEDS:-0 1 2}"
MAXPAR="${LV_MAXPAR:-6}"
# Every arm is sized on its own width knob to this EFFECTIVE parameter budget
# (thesis protocol: 4k for the rollouts).
PARAM_BUDGET="${LV_PARAM_BUDGET:-4000}"   # 0 = no --param-budget (use --size via LV_EXTRA)
PLOT_CELL="${LV_PLOT_CELL:-cfc_lrc}"
# System / resolution; output dir is the run folder (duffing is a second launch
# into 02_duffing).
SYSTEM="${LV_SYSTEM:-periodic_predator_prey}"
TSPAN="${LV_TSPAN:-}"   # "T0 T1" time-span override; empty = the system's own
SEQLEN="${LV_SEQLEN:-128}"
OUTDIR="$RUN_DIR/results"
LR="${LV_LR:-1e-3}"
# CfC-family backbone depth for every arm (V2s control: LV_BACKBONE=1); empty = flag
# not passed, per-wiring defaults unchanged; results get a _bb<N> filename suffix.
BACKBONE="${LV_BACKBONE:-}"
EXTRA="${LV_EXTRA:-}"   # extra runner flags, e.g. "--columns 2 --partition same" or "--size 18"
# Data-seed offset (tuning runs: 1000, disjoint from the final replicates' draws);
# empty = flag not passed, data seed DATA_SEED_BASE + seed.
DATA_SEED_OFFSET="${LV_DATA_SEED_OFFSET:-}"


# shellcheck disable=SC2086  # flag lists are split on purpose
run_one() {
    local cell="$1" wiring="$2" seed="$3"
    local wandb_flag="" bb_flag="" budget_flag="" ds_flag="" tspan_flag=""
    [[ "${PARAM_BUDGET}" != "0" ]] && budget_flag="--param-budget ${PARAM_BUDGET}"
    [[ -n "${BACKBONE:-}" ]] && bb_flag="--backbone-layers ${BACKBONE}"
    [[ -n "${DATA_SEED_OFFSET:-}" ]] && ds_flag="--data-seed-offset ${DATA_SEED_OFFSET}"
    [[ -n "${TSPAN:-}" ]] && tspan_flag="--t-span ${TSPAN}"
    [[ "${USE_WANDB:-0}" == "1" ]] && wandb_flag="--wandb"
    echo "[start] ${cell}_${wiring}_seed${seed} $(date -u +%FT%TZ)"
    uv run python experiments/run_lotka_volterra_benchmark.py \
        --wiring "$wiring" --cell "$cell" --seed "$seed" --epochs "$EPOCHS" \
        --system "$SYSTEM" $tspan_flag --seq-len "$SEQLEN" --outdir "$OUTDIR" \
        $budget_flag --lr "$LR" $bb_flag $ds_flag $EXTRA $wandb_flag
    local rc=$?
    echo "[done ] ${cell}_${wiring}_seed${seed} (exit $rc)"
    return $rc
}
export -f run_one
export EPOCHS SYSTEM TSPAN SEQLEN OUTDIR USE_WANDB PARAM_BUDGET LR BACKBONE EXTRA DATA_SEED_OFFSET

JOBS=""
for cell in $CELLS; do
    for wiring in $WIRINGS; do
        for seed in $SEEDS; do
            JOBS+="${cell} ${wiring} ${seed}"$'\n'
        done
    done
done

echo "=== ${SYSTEM} topology-vs-tbt: $(printf '%s' "$JOBS" | grep -c .) runs, par ${MAXPAR}, ${EPOCHS} epochs, seq_len ${SEQLEN}, -> ${OUTDIR}, on $(hostname) ==="
N_FITS=$(printf '%s' "$JOBS" | grep -c .)
printf '%s' "$JOBS" | grep . | xargs -P "$MAXPAR" -n 3 bash -c 'run_one "$1" "$2" "$3"' _ || echo "WARNING: at least one fit exited non-zero (xargs exit $?)"
echo "=== all runs done $(date -u +%FT%TZ) ==="
N_JSON=$(find "$OUTDIR" -maxdepth 1 -name '*.json' | wc -l)
echo "result files: ${N_JSON} of ${N_FITS} expected"
ls -la "$OUTDIR"/*.json

echo "=== rendering plots (cell ${PLOT_CELL}, ${SYSTEM}) ==="
uv run python experiments/plot_lotka_volterra.py \
    --indir "$OUTDIR" --cell "$PLOT_CELL" --seed 0 --n-examples 4
mv "$OUTDIR"/*.png "$RUN_DIR/figures/" 2>/dev/null || true
ls -la "$RUN_DIR/figures"
echo "=== plots done $(date -u +%FT%TZ) ==="
[[ "$N_JSON" -ge "$N_FITS" ]] || { echo "FATAL: ${N_FITS} fits expected, ${N_JSON} result files written" >&2; exit 1; }
