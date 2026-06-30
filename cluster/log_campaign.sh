#!/usr/bin/env bash
# Append a one-line "campaign submitted" entry to the thesis work-documentation
# log in the Obsidian vault. Safe to call unconditionally: when the vault log is
# not present on this machine (e.g. on the cluster, in CI, or a code-only clone)
# the script is a no-op and exits 0, so submit_campaign.sh can always call it.
#
# Usage:
#   cluster/log_campaign.sh <date|-> <campaign> <slurm_job_id> <run_count>
#
#   <date|->        ISO date for the entry; pass '-' (or empty) to use `date -u`.
#   <campaign>      campaign/profile name (e.g. v5).
#   <slurm_job_id>  SLURM array job id returned by sbatch (or '-').
#   <run_count>     number of run specs in the campaign (or '-').
#
# Override the target path with CAMPAIGN_WORKLOG=/some/other/file.md.
#
# Emitted line (appended verbatim, one per call):
#   - 2026-06-30  campaign `v5` submitted — SLURM job 123456, 1980 runs
set -euo pipefail

WORKLOG="${CAMPAIGN_WORKLOG:-/Users/angeral/Repositories/master_thesis_v2/obsidian_master_thesis/Thesis/work-documentation.md}"

log_date="${1:--}"
campaign="${2:?usage: log_campaign.sh <date|-> <campaign> <slurm_job_id> <run_count>}"
job_id="${3:--}"
run_count="${4:--}"

# Empty / placeholder date -> today (UTC).
if [[ -z "${log_date}" || "${log_date}" == "-" ]]; then
    log_date="$(date -u +%F)"
fi

# No-op when the vault log is absent on this host.
if [[ ! -f "${WORKLOG}" ]]; then
    echo "log_campaign: ${WORKLOG} absent — skipping work-log entry"
    exit 0
fi

printf -- '- %s  campaign `%s` submitted — SLURM job %s, %s runs\n' \
    "${log_date}" "${campaign}" "${job_id}" "${run_count}" >> "${WORKLOG}"
echo "log_campaign: appended entry to ${WORKLOG}"
