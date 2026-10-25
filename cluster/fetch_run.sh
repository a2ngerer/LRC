#!/usr/bin/env bash
# Mirror one run folder from the cluster benchmarks tree into the local
# results tree:
#   <CLUSTER_BENCH_DIR>/<NN_benchmark>/runs/<run>/  ->  results/<NN_benchmark>/<run>/
# (LAUNCH.md, results/, figures/, wandb/; logs/ only with --logs).
#
#   ./cluster/fetch_run.sh <NN_benchmark>                 # list the run folders
#   ./cluster/fetch_run.sh <NN_benchmark> <run> [--logs]  # mirror one run
#
# Offline wandb runs land under results/<NN>/<run>/wandb/; upload them with
# ./cluster/wandb_sync.sh <that directory>.
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/config.sh

bench="${1:?usage: fetch_run.sh <NN_benchmark> [<run>] [--logs]}"
run="${2:-}"
with_logs=0
for a in "$@"; do [[ "$a" == "--logs" ]] && with_logs=1; done
[[ "${run}" == "--logs" ]] && run=""

remote="${CLUSTER_HOST}:${CLUSTER_BENCH_DIR}/${bench}/runs"
if [[ -z "${run}" ]]; then
    echo "run folders under ${remote}/:"
    # shellcheck disable=SC2029
    ssh "${CLUSTER_HOST}" "ls -1 ${CLUSTER_BENCH_DIR}/${bench}/runs/ 2>/dev/null || echo '(none)'"
    exit 0
fi

local_dir="results/${bench}/${run}"
mkdir -p "${local_dir}"
excl=(--exclude='logs/')
[[ ${with_logs} -eq 1 ]] && excl=()
echo "rsync ${remote}/${run}/ -> ${local_dir}/"
rsync -avz "${excl[@]}" "${remote}/${run}/" "${local_dir}/"
echo "-- ${local_dir}:"
ls -la "${local_dir}/results"/*.json 2>/dev/null | tail -20 || true
ls -la "${local_dir}/figures" 2>/dev/null || true
echo "Done."
