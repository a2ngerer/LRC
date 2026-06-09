#!/usr/bin/env bash
# One-time environment setup ON THE CLUSTER.
#
# IMPORTANT: run this inside an interactive GPU session, not on the login node:
#   ssh datalab
#   srun -p GPU-a40 --gres=gpu:a40:1 -n1 --pty bash
#   cd ~/thesis-benchmark && bash cluster/setup_env.sh
set -euo pipefail

cd "$(dirname "$0")/.."

# Some virtual nodes (hostnames i-*) have a non-writable /tmp.
export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Syncing dependencies (with CUDA extra) ..."
uv sync --extra cuda

echo "Verifying TensorFlow sees the GPU ..."
uv run python - <<'PY'
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print('tensorflow', tf.__version__, 'GPUs:', gpus)
assert gpus, ("no GPU visible to TensorFlow -- make sure you are inside an "
              "srun --gres=gpu session, not on the login node")
PY

echo "Environment ready."
