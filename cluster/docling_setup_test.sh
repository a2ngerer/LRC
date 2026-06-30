#!/usr/bin/env bash
# One-time Docling GPU setup + verification, run inside a GPU srun session:
#   srun -p GPU-a40 --gres=gpu:a40:1 -n1 -t 0:20:00 bash cluster/docling_setup_test.sh
#
# The cluster's CUDA 13.2 driver + the default cu130 torch wheel hit
# CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH in conv2d. Fix: pin a stable CUDA 12.x
# torch (forward-compatible with the newer driver) and clear LD_LIBRARY_PATH so
# torch loads only its own bundled cuDNN. Converts one paper as a canary.
# Override the backend with: BACKEND=cu126 bash cluster/docling_setup_test.sh
set -uo pipefail
VENV="$HOME/docling-venv"
ART="$HOME/.cache/docling/models"
TEST_PDF="papers/proposal_papers/cho2014-gru-encoder-decoder.pdf"
OUT="$HOME/_dtest"
BACKEND="${BACKEND:-cu124}"

echo "=== CUDA driver ==="
nvidia-smi | grep -i "CUDA Version" || true
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<empty>}"

echo "=== build venv with torch backend=$BACKEND ==="
uv venv "$VENV" --python 3.12 --clear 2>&1 | tail -2
UV_TORCH_BACKEND="$BACKEND" uv pip install --python "$VENV" docling 2>&1 | tail -4

echo "=== torch sanity ==="
"$VENV/bin/python" -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| avail', torch.cuda.is_available())"

echo "=== canary convert (cleared LD_LIBRARY_PATH, GPU, formula enrichment) ==="
rm -rf "$OUT"
env -u LD_LIBRARY_PATH "$VENV/bin/docling" convert --artifacts-path "$ART" \
  --to md --image-export-mode placeholder --enrich-formula --device cuda \
  --output "$OUT" "$TEST_PDF" 2>&1 | tail -12

echo "=== result ==="
ls -la "$OUT/" 2>/dev/null
echo "LaTeX blocks (\$\$): $(grep -c '\$\$' "$OUT"/*.md 2>/dev/null || echo 0)"
