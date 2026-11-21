#!/usr/bin/env bash
# Sync the code tree and submit one eval-only cNCP job of F3 (#156,
# cluster/eval_x.sbatch) as ONE SLURM job into its own run folder (default
# 00_canary, BENCH overrides; layout: cluster/_run.sh). Mode and record globs
# come from the environment; one call submits one job, X1 and X2 are two calls.
#
#   EX_MODE=ablation EX_RESULTS='<glob> <glob> <glob>' ./cluster/submit_eval_x.sh f3-x1-ablation
#   EX_MODE=gains EX_RESULTS='<glob> ...' ./cluster/submit_eval_x.sh f3-x2-gains
#   EX_MODE=ablation EX_RESULTS='<glob>' EX_DATA_PATH=<file> ./cluster/submit_eval_x.sh -n <label>
#
# The globs are relative to the code tree on the cluster
# (../benchmarks/<bench>/runs/<run>/results/<cell>_<wiring>_seed*.json) and
# reach the script unexpanded. Prereqs: VPN + the cluster venv (cluster/README.md).
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/config.sh
source cluster/_run.sh
run_parse_args "$@"

case "${EX_MODE:-}" in
    ablation) EX_OUT="results/cncp_ablation.csv (--out)" ;;
    gains) EX_OUT="results/cncp_gains.csv (--out) and results/cncp_paths.csv (--paths)" ;;
    *) echo "EX_MODE must be ablation or gains, got '${EX_MODE:-}'" >&2; exit 2 ;;
esac
read -r -a ex_globs <<<"${EX_RESULTS:-}"   # as eval_x.sbatch splits them
if [[ ${#ex_globs[@]} -eq 0 ]]; then
    echo "EX_RESULTS is empty: give one or more record globs" >&2; exit 2
fi
if [[ -n "${EX_DATA_PATH:-}" && "$EX_MODE" != ablation ]]; then
    echo "EX_DATA_PATH applies to EX_MODE=ablation only" >&2; exit 2
fi

EX_ENV=""
for v in EX_MODE EX_RESULTS EX_DATA_PATH; do
    if [[ -n "${!v:-}" ]]; then EX_ENV+="${v}='${!v}' "; fi
done
RUN_ENV="${EX_ENV}output in the run folder: ${EX_OUT}"
RUN_COMMAND="${BENCH:+BENCH=${BENCH} }${EX_ENV}./cluster/submit_eval_x.sh ${LABEL}"
RUN_COUNT="1 job, eval-only (${EX_OUT})"
echo "fits: ${RUN_COUNT}"

run_prepare "${BENCH:-00_canary}"   # reserve the run folder first, then touch the shared code tree
run_rsync
run_require_dataset
run_sbatch cluster/eval_x.sbatch "${EX_ENV}"
