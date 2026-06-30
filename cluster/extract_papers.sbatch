#!/usr/bin/env bash
# SLURM job: re-extract every paper PDF to clean Markdown with Docling, on a GPU.
#
# Why the cluster: Docling's formula enrichment uses CodeFormulaV2, an
# autoregressive vision-language model that emits LaTeX token-by-token per
# formula. That is CPU-bound and painfully slow on a laptop (~30+ min/paper,
# and CodeFormulaV2 has no MPS support). On the dataLAB CUDA GPUs it is fast,
# so here we run the FULL pipeline: layout + tables + --enrich-formula (LaTeX)
# + --enrich-code. Output mirrors papers/ into papers/.extracted/.
#
# One-time setup (see cluster/README.md):
#   - sync the papers/ tree to the cluster (extend sync_to_cluster.sh, or
#     `rsync -a papers/ datalab:thesis-benchmark/papers/`)
#   - no persistent install needed: uvx provisions Docling on first run; the
#     torch wheel ships CUDA 12 on Linux, so --device cuda works out of the box.
#     Model weights are cached under $HF_HOME across runs.
#
# Submit (login node, repo root):
#   sbatch cluster/extract_papers.sbatch
# Fetch results back:
#   rsync -a datalab:thesis-benchmark/papers/.extracted/ papers/.extracted/
#
#SBATCH --job-name=docling-extract
#SBATCH --partition=GPU-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH --open-mode=append
set -euo pipefail
# sbatch copies this script to a spool dir, so $0 is NOT the repo path. SLURM
# sets SLURM_SUBMIT_DIR to the directory `sbatch` was called from (the repo
# root, thesis-benchmark/). Submit with: cd thesis-benchmark && sbatch cluster/extract_papers.sbatch
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Virtual nodes (hostnames i-*) have a non-writable /tmp — redirect temp paths.
export TMPDIR="$HOME/tmp"; mkdir -p "$TMPDIR"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export PATH="$HOME/.local/bin:$PATH"
# Persist HF model weights (layout, tableformer, CodeFormulaV2) across runs.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p outputs/slurm

# Use the dedicated venv built by cluster/docling_setup_test.sh. The default
# cu130 torch hits CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH on the CUDA 13.2
# driver; pinned cu124 torch + cleared LD_LIBRARY_PATH (torch uses its own
# bundled cuDNN) works. Build the venv once before submitting:
#   srun -p GPU-a40 --gres=gpu:a40:1 -n1 -t 0:20:00 bash cluster/docling_setup_test.sh
VENV="$HOME/docling-venv"
if [[ ! -x "$VENV/bin/docling" ]]; then
    echo "ERROR: $VENV/bin/docling missing — run cluster/docling_setup_test.sh first." >&2
    exit 1
fi

PAPERS_DIR="papers"
OUT_ROOT="papers/.extracted"
mkdir -p "$OUT_ROOT"

# Pre-downloaded models (via `docling-tools models download ...` on the login
# node) live here; pass them explicitly so the offline compute node does not hit
# the HF Hub. Includes CodeFormulaV2 for --enrich-formula/--enrich-code.
ARTIFACTS="$HOME/.cache/docling/models"

echo "=== docling extract — $(date -u +%FT%TZ) on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

rc=0
while IFS= read -r -d '' pdf; do
    rel="${pdf#"$PAPERS_DIR"/}"          # e.g. Monikas_recommendations/Foo.pdf
    sub="$(dirname "$rel")"              # mirror sub-directory structure
    out="$OUT_ROOT/$sub"
    mkdir -p "$out"
    echo "--- $rel"
    # docling exits 0 even when a doc fails to convert, so verify the .md exists.
    # env -u LD_LIBRARY_PATH: torch must use its bundled cuDNN, not a system one.
    base="$(basename "${pdf%.pdf}")"
    env -u LD_LIBRARY_PATH "$VENV/bin/docling" convert \
            --artifacts-path "$ARTIFACTS" \
            --to md --image-export-mode placeholder \
            --enrich-formula --enrich-code --device cuda \
            --output "$out" "$pdf" || true
    if [ -s "$out/$base.md" ]; then
        echo "   OK -> $out/$base.md ($(wc -l < "$out/$base.md") lines)"
    else
        echo "!! FAILED (no .md produced): $rel"
        rc=1
    fi
done < <(find "$PAPERS_DIR" -name '*.pdf' -not -path "*/.extracted/*" -print0)

echo "=== done (rc=$rc) — $(date -u +%FT%TZ) ==="
exit "$rc"
