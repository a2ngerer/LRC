# shellcheck shell=bash
# Cluster configuration. Edit here or override via env (env wins).
# The ssh alias (set up in ~/.ssh/config) and the cluster paths. The cluster
# workspace (since 2026-10-21) is one code directory plus a benchmarks tree:
#
#   <CLUSTER_REPO_DIR>/          the rsync'd code tree (rsync --delete: never
#                                write results here)
#   <CLUSTER_BENCH_DIR>/
#     README.md                  index table, one row per benchmark folder
#     LAUNCH.template.md         template for every launch note
#     <NN_benchmark>/runs/<YYYY-MM-DD>_<label>/{LAUNCH.md,results,logs,figures,wandb}
#
# Paths are relative to $HOME on the cluster unless absolute.
CLUSTER_HOST="${CLUSTER_HOST:-datalab}"
CLUSTER_REPO_DIR="${CLUSTER_REPO_DIR:-masterarbeit/code}"
CLUSTER_BENCH_DIR="${CLUSTER_BENCH_DIR:-masterarbeit/benchmarks}"

# SLURM resources for the campaign array (submit_campaign.sh); the task
# launchers (submit_lotka_volterra.sh etc.) carry their own #SBATCH headers.
CLUSTER_PARTITION="${CLUSTER_PARTITION:-GPU-a40}"
CLUSTER_GPU="${CLUSTER_GPU:-a40:1}"
CLUSTER_CPUS="${CLUSTER_CPUS:-12}"
CLUSTER_MEM="${CLUSTER_MEM:-32G}"
CLUSTER_TIME="${CLUSTER_TIME:-04:00:00}"
CLUSTER_ARRAY_THROTTLE="${CLUSTER_ARRAY_THROTTLE:-8}"
CLUSTER_RUNS_PER_GPU="${CLUSTER_RUNS_PER_GPU:-12}"
