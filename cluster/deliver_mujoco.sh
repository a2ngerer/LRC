#!/usr/bin/env bash
# One-shot cluster delivery: fetch MuJoCo results from ~/thesis-mujoco, render any
# videos that the compute node could not (EGL) from their checkpoints locally,
# build a comparison grid, and copy everything into the iCloud folder.
#
#   bash cluster/deliver_mujoco.sh
set -uo pipefail
cd "$(dirname "$0")/.."

ICLOUD="${ICLOUD:-$HOME/Library/Mobile Documents/com~apple~CloudDocs/Masterarbeit/mujoco-benchmark-videos}"
DEST="$ICLOUD/$(date +%Y-%m-%d)-cluster"
mkdir -p "$DEST" results/mujoco_cluster

echo "[deliver] fetching cluster results ..."
bash cluster/fetch_mujoco.sh || true

if [[ -d results/mujoco_cluster/checkpoints ]]; then
  echo "[deliver] rendering any missing videos from checkpoints ..."
  uv run python experiments/render_from_checkpoints.py \
      --ckpt-dir results/mujoco_cluster/checkpoints \
      --out-dir results/mujoco_cluster/videos 2>&1 | grep -vE "oneDNN|cpu_feature|TF-TRT|tensorflow/core|WARNING" || true
fi

if [[ -d results/mujoco_cluster/videos ]]; then
  echo "[deliver] building comparison grid (seed 0) ..."
  uv run python experiments/render_montage.py \
      --video-dir results/mujoco_cluster/videos \
      --out results/mujoco_cluster/_comparison_grid.mp4 --seed-tag s0 2>/dev/null || true
  cp -f results/mujoco_cluster/videos/*.mp4 "$DEST/" 2>/dev/null || true
  cp -f results/mujoco_cluster/_comparison_grid.mp4 "$DEST/" 2>/dev/null || true
fi
cp -f results/mujoco_cluster/runs/*.json "$DEST/" 2>/dev/null || true

# Self-documenting multi-seed results table.
DEST="$DEST" uv run python - <<'PY' 2>/dev/null || true
import json, glob, os, statistics as st
from collections import defaultdict
dest=os.environ["DEST"]
ev=defaultdict(list); fi=defaultdict(list)
for f in glob.glob("results/mujoco_cluster/runs/*.json"):
    d=json.load(open(f))
    if d.get("eval_return") is not None: ev[d["model"]].append(d["eval_return"])
    if d.get("final_return") is not None: fi[d["model"]].append(d["final_return"])
fam={"mlp":"feedforward","lstm":"gated","gru":"gated","ctrnn":"continuous-time",
     "ltc":"liquid (LTC)","lrc":"liquid LRC (Thesis)","cfc":"closed-form liquid (CfC)"}
rows=[]
for m,e in ev.items():
    rows.append((st.mean(e), m, len(e), st.pstdev(e) if len(e)>1 else 0.0, st.mean(fi[m])))
rows.sort(reverse=True)
L=["# MuJoCo Control Benchmark — HalfCheetah-v5 (Cluster, Multi-Seed)","",
   "PPO-Steuer-Policies, 1.5M Steps, 3 Seeds/Modell. `_comparison_grid.mp4` = alle Modelle (Seed 0) nebeneinander.","",
   "| Rang | Modell | Familie | eval mean | ± sd | final mean | Seeds |","|---|---|---|---|---|---|---|"]
for i,(mean,m,n,sd,fm) in enumerate(rows,1):
    L.append(f"| {i} | `{m}` | {fam.get(m,'')} | {mean:.0f} | {sd:.0f} | {fm:.0f} | {n} |")
L+= ["","eval = deterministischer 1000-Schritt-Rollout (Mean-Action) je Run.",
     "Niedrige sd = stabil über Seeds."]
open(os.path.join(dest,"RESULTS.md"),"w").write("\n".join(L)+"\n")
print("wrote RESULTS.md")
PY

echo "[deliver] done -> $DEST"
ls -la "$DEST" 2>/dev/null | grep -cE "mp4" | xargs echo "mp4 files in iCloud cluster folder:"
