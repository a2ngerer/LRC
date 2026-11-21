#!/usr/bin/env bash
# Eval-only cNCP job of F3 (#156): one call of experiments/eval_cncp_ablation.py
# (X1, path ablation) or experiments/eval_cncp_gains.py (X2, per-lamina gain
# chains) on stored result records, outside the suite manifest. Submitted by
# cluster/submit_eval_x.sh.
#
#   EX_MODE       ablation | gains
#   EX_RESULTS    one or more record globs, space separated, relative to the
#                 code tree; they reach the script unexpanded (it globs them)
#   EX_DATA_PATH  optional, ablation only: --data-path of the script
#
#SBATCH --job-name=f3-eval-x
#SBATCH --partition=GPU-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=16G
# The scripts write their CSV only at the end: a timeout loses the whole output.
#SBATCH --time=08:00:00
# --output/--error are passed on the sbatch command line by cluster/_run.sh
# (<run folder>/logs/); RUN_DIR is exported by the launcher.
#SBATCH --open-mode=append
#SBATCH --requeue
set -euo pipefail
# Run folder (cluster/_run.sh): never write under the code tree (rsync --delete).
RUN_DIR="${RUN_DIR:?RUN_DIR not set: submit via cluster/submit_eval_x.sh}"

# read -a splits on whitespace without pathname expansion: the globs stay quoted.
read -r -a globs <<<"${EX_RESULTS:-}"
if [[ ${#globs[@]} -eq 0 ]]; then
    echo "FATAL: EX_RESULTS is empty" >&2; exit 2
fi
case "${EX_MODE:-}" in
    ablation)
        cmd=(uv run python experiments/eval_cncp_ablation.py --results "${globs[@]}"
             --out "$RUN_DIR/results/cncp_ablation.csv")
        if [[ -n "${EX_DATA_PATH:-}" ]]; then cmd+=(--data-path "$EX_DATA_PATH"); fi ;;
    gains)
        cmd=(uv run python experiments/eval_cncp_gains.py --results "${globs[@]}"
             --out "$RUN_DIR/results/cncp_gains.csv"
             --paths "$RUN_DIR/results/cncp_paths.csv") ;;
    *) echo "FATAL: EX_MODE must be ablation or gains, got '${EX_MODE:-}'" >&2; exit 2 ;;
esac
mkdir -p "$RUN_DIR/results"

source cluster/_common.sh
cc_setup_tmpdir
cc_setup_cuda_ld
cc_setup_tf_env 1

echo "+ ${cmd[*]}"
"${cmd[@]}"
