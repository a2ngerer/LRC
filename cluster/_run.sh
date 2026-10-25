# shellcheck shell=bash
# Shared launcher plumbing for the task launchers (submit_lotka_volterra.sh,
# submit_person_activity.sh, submit_committee.sh). Sourced, not executed.
#
# Every launch gets its own run folder on the cluster, outside the code tree:
#   <CLUSTER_BENCH_DIR>/<NN_benchmark>/runs/<YYYY-MM-DD>_<label>/
#       LAUNCH.md   written BEFORE sbatch (purpose, git SHA, command, env, job id)
#       results/    runner --outdir
#       logs/       SLURM --output / --error
#       figures/    plots rendered at the end of the job
#       wandb/      offline wandb runs (only with *_WANDB=1)
# A run folder is never reused: an existing folder aborts the launch.
#
# Contract for a launcher that sources this file:
#   run_parse_args "$@"        -> sets LABEL, DRY_RUN (-n), BENCH (env override)
#   run_prepare <NN_benchmark> -> sets RUN_DIR (cluster path), creates the folder
#                                 and LAUNCH.md (skipped on DRY_RUN)
#   run_rsync                  -> rsync the code tree (skipped on DRY_RUN)
#   run_sbatch <sbatch file> "<ENV assignments>" -> submits, records the job id
#   The sbatch file reads RUN_DIR from the environment (--export=ALL).
#
# Dry run (-n): print the resolved paths, env and sbatch command, touch nothing.

set -euo pipefail

_run_usage() {
    cat >&2 <<EOF
usage: $0 [-n] <label> [setup]
  <label>   run label; the run folder is <date>_<label> (letters, digits, - _ .)
  -n        dry run: resolve paths and print the sbatch command, do nothing
  setup     one-time venv setup on the login node (person_activity launcher)
env:  BENCH=<NN_benchmark>  override the target benchmark folder (e.g. 00_canary)
      CLUSTER_HOST, CLUSTER_REPO_DIR, CLUSTER_BENCH_DIR  (cluster/config.sh)
EOF
    exit 2
}

run_parse_args() {
    DRY_RUN=0; LABEL=""; SETUP=0
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -n) DRY_RUN=1 ;;
            setup) SETUP=1 ;;
            -h|--help) _run_usage ;;
            -*) echo "unknown flag $1" >&2; _run_usage ;;
            *) if [[ -n "$LABEL" ]]; then echo "one label only" >&2; _run_usage; fi
               LABEL="$1" ;;
        esac
        shift
    done
    if [[ $SETUP -eq 0 && -z "$LABEL" ]]; then _run_usage; fi
    if [[ -n "$LABEL" && ! "$LABEL" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
        echo "label must match [A-Za-z0-9][A-Za-z0-9._-]*, got '$LABEL'" >&2
        exit 2
    fi
}

# ssh wrapper: a dry run never reaches the cluster.
_run_ssh() {
    if [[ "${DRY_RUN:-0}" -eq 1 ]]; then
        echo "[dry-run] ssh ${CLUSTER_HOST} $*"
        return 0
    fi
    # shellcheck disable=SC2029
    ssh "${CLUSTER_HOST}" "$@"
}

run_rsync() {
    if [[ "${DRY_RUN:-0}" -eq 1 ]]; then
        echo "[dry-run] rsync $(pwd)/ -> ${CLUSTER_HOST}:${CLUSTER_REPO_DIR}/"
        return 0
    fi
    echo "rsync $(pwd) -> ${CLUSTER_HOST}:${CLUSTER_REPO_DIR}/"
    rsync -avz --delete --copy-links \
        --exclude='.git/' --exclude='.venv/' --exclude='results/' \
        --exclude='outputs/' --exclude='__pycache__/' --exclude='*.pyc' \
        --exclude='.DS_Store' --exclude='.pytest_cache/' --exclude='datasets/' \
        --exclude='papers/' \
        ./ "${CLUSTER_HOST}:${CLUSTER_REPO_DIR}/"
}

# run_count <list> <list> ...: product of the word counts; sets RUN_COUNT (fits)
# so LAUNCH.md and the work-log carry the expected number of result JSONs.
run_count() {
    local n=1 l
    for l in "$@"; do n=$(( n * $(wc -w <<<"$l") )); done
    RUN_COUNT="$n"
    echo "fits: ${RUN_COUNT}"
}

# Local provenance for LAUNCH.md.
_run_git_sha() {
    local sha dirty=""
    sha="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then dirty=" (dirty)"; fi
    echo "${sha}${dirty}"
}

# run_prepare <NN_benchmark>: resolve RUN_DIR, create it, write LAUNCH.md.
# Globals read: LABEL, DRY_RUN, BENCH (optional override), LAUNCH_PURPOSE
# (optional), RUN_COMMAND (the exact local command, set by the launcher).
run_prepare() {
    local bench="${BENCH:-$1}"
    local date_utc; date_utc="$(date -u +%F)"
    RUN_NAME="${date_utc}_${LABEL}"
    RUN_REL="${CLUSTER_BENCH_DIR}/${bench}/runs/${RUN_NAME}"
    # $HOME-relative on the cluster unless absolute.
    if [[ "${RUN_REL}" == /* ]]; then RUN_DIR="${RUN_REL}"; else RUN_DIR="\$HOME/${RUN_REL}"; fi
    RUN_BENCH="${bench}"
    echo "run folder: ${CLUSTER_HOST}:${RUN_REL}"
    if [[ "${DRY_RUN:-0}" -eq 1 ]]; then return 0; fi
    local launch_md
    launch_md="$(_run_launch_md "${RUN_NAME}")"
    # Reserve the folder atomically: a plain mkdir fails if it exists (run
    # folders are never reused), so two launches cannot share one folder.
    # shellcheck disable=SC2029
    if ! ssh "${CLUSTER_HOST}" "mkdir -p $(dirname "${RUN_DIR}") && mkdir ${RUN_DIR} && mkdir ${RUN_DIR}/results ${RUN_DIR}/logs ${RUN_DIR}/figures ${RUN_DIR}/wandb && cat > ${RUN_DIR}/LAUNCH.md" <<<"${launch_md}"; then
        echo "FATAL: could not reserve run folder ${RUN_REL} (exists already? run folders are never reused; pick a new label)" >&2
        exit 1
    fi
    echo "LAUNCH.md written (job id is appended after sbatch)."
}

_run_launch_md() {
    local name="$1"
    cat <<EOF
# ${name}

- Purpose: ${LAUNCH_PURPOSE:-<fill in: what question this launch answers>}
- Code: git SHA $(_run_git_sha) (local \`$(basename "$(pwd)")/\`, branch $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)), synced $(date -u +%FT%TZ)
- Command: \`${RUN_COMMAND:-$0 $LABEL}\`
- Env: ${RUN_ENV:-<none>}
- SLURM job IDs: <pending>
- Arms / seeds / epochs / budget: ${RUN_MATRIX:-<see Env>}
- Expected result files: ${RUN_COUNT:-<unknown>}
- Started / finished: $(date -u +%FT%TZ) / <pending>
- Outcome: <pending>
EOF
}

# run_sbatch <sbatch file> "<ENV assignments>": submit with RUN_DIR exported,
# SLURM logs under the run folder, then record the job id in LAUNCH.md and
# the benchmarks index.
run_sbatch() {
    local sbatch_file="$1" env_assign="${2:-}"
    local cmd="cd ${CLUSTER_REPO_DIR} && export PATH=\$HOME/.local/bin:\$PATH && \
${env_assign}RUN_DIR=${RUN_DIR} sbatch --export=ALL \
${SBATCH_EXTRA:+${SBATCH_EXTRA} }--output=${RUN_DIR}/logs/%x-%j.out --error=${RUN_DIR}/logs/%x-%j.err ${sbatch_file}"
    if [[ "${DRY_RUN:-0}" -eq 1 ]]; then
        echo "[dry-run] ssh ${CLUSTER_HOST} \"${cmd}\""
        return 0
    fi
    local out job_id
    # shellcheck disable=SC2029
    out="$(ssh "${CLUSTER_HOST}" "${cmd}")"
    echo "${out}"
    job_id="$(sed -n 's/^Submitted batch job \([0-9]*\).*/\1/p' <<<"${out}")"
    if [[ -z "${job_id}" ]]; then
        echo "WARNING: could not parse the job id; add it to ${RUN_REL}/LAUNCH.md by hand." >&2
        return 0
    fi
    _run_record_job "${job_id}"
    echo
    echo "Monitor:  ssh ${CLUSTER_HOST} squeue -u \\\$USER"
    echo "Logs:     ssh ${CLUSTER_HOST} tail -f ${RUN_REL}/logs/*-${job_id}.out"
    echo "Fetch:    ./cluster/fetch_run.sh ${RUN_BENCH} ${RUN_NAME}"
    if [[ -x cluster/log_campaign.sh ]]; then
        cluster/log_campaign.sh - "${RUN_BENCH}/${RUN_NAME}" "${job_id}" "${RUN_COUNT:--}" || true
    fi
}

_run_record_job() {
    local job_id="$1"
    # LAUNCH.md: fill the job id. README index: set the "Last run folder" cell
    # of this benchmark's row (the last cell before the closing pipe); print
    # the row when the table does not match.
    # shellcheck disable=SC2029
    ssh "${CLUSTER_HOST}" "sed -i 's#^- SLURM job IDs: <pending>#- SLURM job IDs: ${job_id}#' ${RUN_DIR}/LAUNCH.md && \
readme=\$HOME/${CLUSTER_BENCH_DIR}/README.md && \
if grep -q '^| ${RUN_BENCH} |' \"\$readme\"; then \
  awk -v b='${RUN_BENCH}' -v r='${RUN_NAME}' 'BEGIN{FS=OFS=\"|\"} \$2==\" \" b \" \" {\$(NF-1)=\" \" r \" \"} {print}' \"\$readme\" > \"\$readme.tmp\" && mv \"\$readme.tmp\" \"\$readme\"; \
else echo \"README index: no row for ${RUN_BENCH}; add: | ${RUN_BENCH} | ... | ${RUN_NAME} |\"; fi"
}
