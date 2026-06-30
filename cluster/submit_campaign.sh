#!/usr/bin/env bash
# Sync code and submit ONE config-driven benchmark CAMPAIGN as a SLURM array job.
# Generic replacement for the 10 per-profile submit_benchmark_v*.sh wrappers: the
# matrix, hyperparameters AND the cluster resources all come from one campaign
# YAML, so a new campaign needs a YAML, not a new wrapper.
#
#   ./cluster/submit_campaign.sh configs/campaigns/v5.yaml
#   ./cluster/submit_campaign.sh v5                          # profile name also works
#   ./cluster/submit_campaign.sh configs/campaigns/v5.yaml -- --systems duffing
#   CLUSTER_MEM=96G ./cluster/submit_campaign.sh v5          # env override still wins
#   ./cluster/submit_campaign.sh -n v5                       # dry run (no ssh/rsync)
#
# Resource precedence (per key): explicit environment override > the YAML
# `cluster:` value > cluster/config.sh default. The array is throttled to %8 so
# it never exceeds the dataLAB hard limit of 8 concurrent GPUs per user.
set -euo pipefail

# --- explicit env overrides, captured BEFORE config.sh fills in defaults ------
# (config.sh sets CLUSTER_* to env-or-hardcoded-default, which would otherwise
#  mask the YAML value; capturing here keeps env > YAML > default ordering.)
ENV_PARTITION="${CLUSTER_PARTITION:-}"
ENV_GPU="${CLUSTER_GPU:-}"
ENV_CPUS="${CLUSTER_CPUS:-}"
ENV_MEM="${CLUSTER_MEM:-}"
ENV_TIME="${CLUSTER_TIME:-}"
ENV_K="${CLUSTER_RUNS_PER_GPU:-}"
ENV_THROTTLE="${CLUSTER_ARRAY_THROTTLE:-}"

cd "$(dirname "$0")/.."
source cluster/config.sh

# --- args -------------------------------------------------------------------
DRY_RUN=0
if [[ "${1:-}" == "-n" || "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift
fi

CONFIG_ARG="${1:?usage: submit_campaign.sh [-n] <config.yaml|profile> [-- runner args]}"
shift

RUNNER_ARGS=()
if [[ "${1:-}" == "--" ]]; then
    shift
    RUNNER_ARGS=("$@")
fi

# Resolve the YAML: a literal path, or a bare profile name under configs/campaigns/.
if [[ -f "${CONFIG_ARG}" ]]; then
    YAML="${CONFIG_ARG}"
elif [[ -f "configs/campaigns/${CONFIG_ARG}.yaml" ]]; then
    YAML="configs/campaigns/${CONFIG_ARG}.yaml"
else
    echo "submit_campaign: config not found: ${CONFIG_ARG}" >&2
    echo "  (looked for a file and for configs/campaigns/${CONFIG_ARG}.yaml)" >&2
    exit 1
fi
# Repo-relative path handed to the cluster (sbatch runs with CWD = repo root).
CONFIG="${YAML#"$PWD/"}"
CONFIG_NAME="$(basename "${YAML}" .yaml)"

# --- read the `cluster:` section from the YAML (tiny python yaml read) --------
# Emits `CL_*='value'` lines only for keys that are present; absent keys stay
# unset so the precedence fallback below uses env-or-config.sh default.
eval "$(uv run python -c '
import sys, yaml
d = yaml.safe_load(open(sys.argv[1])) or {}
c = d.get("cluster") or {}
def emit(name, val):
    if val is not None:
        print(f"{name}=\x27{val}\x27")
emit("CL_PARTITION", c.get("partition"))
emit("CL_GPU", c.get("gpu"))
emit("CL_CPUS", c.get("cpus"))
emit("CL_MEM", c.get("mem"))
emit("CL_WALLTIME", c.get("walltime"))
emit("CL_RUNS_PER_GPU", c.get("runs_per_gpu"))
emit("CL_THROTTLE", c.get("array_throttle"))
rq = c.get("requeue")
if rq is not None:
    emit("CL_REQUEUE", "true" if rq else "false")
' "${YAML}")"

# Resolve each resource: env override > YAML > config.sh default.
PARTITION="${ENV_PARTITION:-${CL_PARTITION:-${CLUSTER_PARTITION}}}"
GPU="${ENV_GPU:-${CL_GPU:-${CLUSTER_GPU}}}"
CPUS="${ENV_CPUS:-${CL_CPUS:-${CLUSTER_CPUS}}}"
MEM="${ENV_MEM:-${CL_MEM:-${CLUSTER_MEM}}}"
WALLTIME="${ENV_TIME:-${CL_WALLTIME:-${CLUSTER_TIME}}}"
K="${ENV_K:-${CL_RUNS_PER_GPU:-${CLUSTER_RUNS_PER_GPU}}}"
THROTTLE="${ENV_THROTTLE:-${CL_THROTTLE:-${CLUSTER_ARRAY_THROTTLE}}}"

# requeue defaults on (campaign_array.sbatch bakes #SBATCH --requeue); only an
# explicit `requeue: false` in the YAML flips it off via --no-requeue.
REQUEUE_FLAG=""
if [[ "${CL_REQUEUE:-true}" == "false" ]]; then
    REQUEUE_FLAG="--no-requeue"
fi

# --- matrix size ------------------------------------------------------------
# The spec list is deterministic (src/benchmark expand); RUNNER_ARGS (matrix
# filters) are forwarded so a filtered submit sizes its array correctly.
N_SPECS="$(uv run python experiments/run_campaign.py --config "${YAML}" --count \
    ${RUNNER_ARGS[@]+"${RUNNER_ARGS[@]}"})"
if ! [[ "${N_SPECS}" =~ ^[0-9]+$ ]]; then
    echo "submit_campaign: could not determine N_SPECS (got: '${N_SPECS}')" >&2
    exit 1
fi
if [[ "${N_SPECS}" -eq 0 ]]; then
    echo "submit_campaign: campaign '${CONFIG_NAME}' expands to 0 runs -- nothing to submit." >&2
    exit 1
fi

# Each array task processes K run specs in parallel on one shared GPU, so the
# number of array tasks is ceil(N_SPECS / K) -- identical to submit_benchmark.sh.
N_TASKS=$(( (N_SPECS + K - 1) / K ))

echo "Campaign '${CONFIG_NAME}' (${CONFIG}):"
echo "  ${N_SPECS} runs -> ${N_TASKS} array tasks (${K} runs/GPU, throttle ${THROTTLE} => up to $((K * THROTTLE)) concurrent runs)"
echo "  resources: partition=${PARTITION} gpu=${GPU} cpus=${CPUS} mem=${MEM} time=${WALLTIME} requeue=${CL_REQUEUE:-true}"

# --- build the remote sbatch command ----------------------------------------
# CONFIG + N_SPECS + RUNS_PER_GPU are exported so campaign_array.sbatch knows the
# campaign and the block bounds; --export=ALL keeps the login env intact. The
# sbatch reads CONFIG from the env and injects --config/--index itself; RUNNER_ARGS
# are forwarded positionally as per-run overrides.
SBATCH_CMD="cd ${CLUSTER_REPO_DIR} && mkdir -p outputs/slurm && \
sbatch \
  --partition='${PARTITION}' \
  --gres=gpu:${GPU} \
  --cpus-per-task='${CPUS}' \
  --mem='${MEM}' \
  --time='${WALLTIME}' \
  --array=0-$((N_TASKS - 1))%${THROTTLE} ${REQUEUE_FLAG} \
  --export=ALL,CONFIG=${CONFIG},N_SPECS=${N_SPECS},RUNS_PER_GPU=${K} \
  cluster/campaign_array.sbatch ${RUNNER_ARGS[*]:-}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo
    echo "[dry-run] would sync, then:"
    echo "  ssh ${CLUSTER_HOST} \"${SBATCH_CMD}\""
    exit 0
fi

./cluster/sync_to_cluster.sh

echo "Submitting array job (0-$((N_TASKS - 1))%${THROTTLE}) on ${PARTITION} ..."
# shellcheck disable=SC2029
SUBMIT_OUT="$(ssh "${CLUSTER_HOST}" "${SBATCH_CMD}")"
echo "${SUBMIT_OUT}"

# Best-effort work-log entry (no-op when the Obsidian vault log is absent here).
JOB_ID="$(printf '%s\n' "${SUBMIT_OUT}" | grep -oE '[0-9]+' | tail -1 || true)"
if [[ -x cluster/log_campaign.sh ]]; then
    cluster/log_campaign.sh - "${CONFIG_NAME}" "${JOB_ID:--}" "${N_SPECS}" || true
fi

echo
echo "Monitor:  ssh ${CLUSTER_HOST} squeue -u \\\$USER"
echo "Logs:     ssh ${CLUSTER_HOST} tail -f ${CLUSTER_REPO_DIR}/outputs/slurm/thesis-campaign-<jobid>_<task>.out"
echo "Fetch:    ./cluster/fetch_results.sh"
