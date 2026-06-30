"""Render mp4 videos locally from saved policy checkpoints.

Used when training happened where offscreen rendering was unavailable (e.g. a
compute node without a working EGL backend): the run always writes a ``.npz``
checkpoint, and this script reconstructs each policy and rolls out a deterministic
episode into an mp4. Rendering is known to work locally, so this decouples
"where we trained" from "where we render".

    uv run python experiments/render_from_checkpoints.py \
        --ckpt-dir results/mujoco_cluster/checkpoints \
        --out-dir results/mujoco_cluster/videos
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    p = argparse.ArgumentParser(description="Render videos from policy checkpoints")
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--task", default="HalfCheetah-v5")
    p.add_argument("--eval-seed", type=int, default=12345)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--overwrite", action="store_true", help="re-render even if mp4 exists")
    args = p.parse_args()

    from src.tasks.mujoco_rl import record as rec

    os.makedirs(args.out_dir, exist_ok=True)
    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, "*.npz")))
    if not ckpts:
        raise SystemExit(f"no checkpoints in {args.ckpt_dir}")
    print(f"rendering {len(ckpts)} checkpoints from {args.ckpt_dir}")
    for cp in ckpts:
        tag = os.path.splitext(os.path.basename(cp))[0]
        out = os.path.join(args.out_dir, f"{tag}.mp4")
        if os.path.exists(out) and not args.overwrite:
            print(f"  skip {tag} (exists)")
            continue
        try:
            policy, mean, var, meta = rec.load_policy(cp)
            r = rec.record_episode(policy, args.task, mean, var, out,
                                   seed=args.eval_seed, max_steps=args.max_steps,
                                   label=tag)
            print(f"  {tag}: return={r['return']:.1f} len={r['length']} -> {out}")
        except Exception as e:
            print(f"  {tag}: FAILED ({e!r})")


if __name__ == "__main__":
    main()
