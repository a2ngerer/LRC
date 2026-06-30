#!/usr/bin/env bash
# Local overnight MuJoCo control benchmark: train each model sequentially, render
# a video per model, and copy each video + metrics into the iCloud folder as soon
# as it is produced (so finished videos appear incrementally through the night).
#
# This is the dependency-free guaranteed path: no cluster, no VPN. Launch under
# `caffeinate -dimsu` so the Mac does not sleep mid-run, e.g.
#   caffeinate -dimsu bash experiments/run_local_overnight.sh
#
# Env overrides: STEPS, SEED, OUT, MODELS, ICLOUD.
set -uo pipefail
cd "$(dirname "$0")/.."

STEPS="${STEPS:-1000000}"
SEED="${SEED:-0}"
OUT="${OUT:-results/mujoco_local}"
MODELS="${MODELS:-mlp lrc gru ltc cfc lstm ctrnn}"
ICLOUD="${ICLOUD:-$HOME/Library/Mobile Documents/com~apple~CloudDocs/Masterarbeit/mujoco-benchmark-videos}"
DATE="$(date +%Y-%m-%d)"
DEST="$ICLOUD/$DATE-local"
mkdir -p "$OUT" "$DEST"
LOG="$OUT/overnight.log"

{
  echo "==================================================================="
  echo "=== local overnight start $(date) ==="
  echo "    steps=$STEPS seed=$SEED models: $MODELS"
  echo "    out=$OUT  icloud=$DEST"
  echo "==================================================================="
} | tee -a "$LOG"

cat > "$DEST/README.txt" <<EOF
MuJoCo control benchmark — HalfCheetah-v5
Thesis neuron types (LTC/LRC/CfC/CT-RNN) vs classical (LSTM/GRU/MLP) as PPO policies.
Each mp4 shows a trained policy driving the cheetah for one episode; the overlay
shows model name, step and cumulative reward. Higher reward = faster, more stable
running. Source: local training pass, $STEPS steps/model, seed $SEED.
Generated $(date).
EOF
cp -f "$DEST/README.txt" "$ICLOUD/README.txt" 2>/dev/null || true

for M in $MODELS; do
  echo "------- $M  start $(date) -------" | tee -a "$LOG"
  if uv run python experiments/run_mujoco_benchmark.py \
        --model "$M" --seed "$SEED" --total-steps "$STEPS" \
        --out-dir "$OUT" >> "$LOG" 2>&1; then
    VID="$OUT/videos/${M}_s${SEED}.mp4"
    JSON="$OUT/runs/${M}_s${SEED}.json"
    if [[ -f "$VID" ]]; then
      cp -f "$VID" "$DEST/${M}_s${SEED}.mp4"
      cp -f "$JSON" "$DEST/${M}_s${SEED}.json" 2>/dev/null || true
      echo "    copied $M -> $DEST" | tee -a "$LOG"
    else
      echo "    WARNING: no video for $M" | tee -a "$LOG"
    fi
  else
    echo "    ERROR: training failed for $M (see log); continuing" | tee -a "$LOG"
  fi
done

echo "=== local overnight DONE $(date) ===" | tee -a "$LOG"
