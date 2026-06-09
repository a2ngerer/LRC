# cluster/config.sh — shared defaults for the TU Wien dataLAB cluster.
# Sourced by sync_to_cluster.sh / submit_benchmark.sh / fetch_results.sh.
# Pattern adopted from arc-jax-rl (same cluster, proven setup).
#
# Prerequisites (one-time, see cluster/README.md):
#   - TU Wien VPN active (cluster is not reachable from the public internet)
#   - ~/.ssh/config entry "datalab" -> cluster.datalab.tuwien.ac.at

CLUSTER_HOST="${CLUSTER_HOST:-datalab}"
CLUSTER_REPO_DIR="${CLUSTER_REPO_DIR:-thesis-benchmark}"

# SLURM resource defaults — override via environment, e.g.
#   CLUSTER_PARTITION=GPU-a100 CLUSTER_GPU=a100:1 ./cluster/submit_benchmark.sh
# Available partitions: GPU-v100, GPU-a40 (default), GPU-a100, GPU-a100s,
# GPU-l40s, GPU-h100. Hard limits: max 8 concurrent GPUs per user,
# max walltime 168h, /home quota 100 GiB.
CLUSTER_PARTITION="${CLUSTER_PARTITION:-GPU-a40}"
CLUSTER_GPU="${CLUSTER_GPU:-a40:1}"
CLUSTER_CPUS="${CLUSTER_CPUS:-4}"
CLUSTER_MEM="${CLUSTER_MEM:-16G}"
CLUSTER_TIME="${CLUSTER_TIME:-04:00:00}"

# Array throttle: never occupy more than 8 GPUs at once (dataLAB hard rule).
CLUSTER_ARRAY_THROTTLE="${CLUSTER_ARRAY_THROTTLE:-8}"
