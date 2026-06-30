"""MuJoCo control-benchmark runner: one run = (model, seed) on one task.

Mirrors the conventions of ``run_benchmark.py`` (the Neural-ODE runner):
``--list`` / ``--count`` / ``--index`` for SLURM array submission, plus matrix
filters (``--models`` / ``--seeds``) and a ``--profile`` of sensible defaults.

A run trains a PPO policy (see ``src.tasks.mujoco_rl.ppo``), saves a JSON metrics
record, a ``.npz`` policy checkpoint, and -- unless ``--no-video`` -- a rendered
mp4 of a deterministic rollout. Rendering is wrapped so a render failure (e.g.
no GL backend on a compute node) never fails the training run: the checkpoint is
always written, so the video can be re-rendered later from the checkpoint.

Examples
--------
    uv run python experiments/run_mujoco_benchmark.py --list
    uv run python experiments/run_mujoco_benchmark.py --index 0 --profile v1
    uv run python experiments/run_mujoco_benchmark.py --model lrc --seed 0 \
        --total-steps 300000 --task HalfCheetah-v5
"""
import argparse
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone

# Allow running as a plain script (``src`` is the installed package).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_TASK = "HalfCheetah-v5"
MODELS_FULL = ["mlp", "lstm", "gru", "ctrnn", "ltc", "lrc", "cfc"]

PROFILES = {
    # full thesis matrix: 7 models x 3 seeds = 21 runs
    "v1": {"models": MODELS_FULL, "seeds": [0, 1, 2], "total_steps": 1_000_000},
    # one seed per model -- a quick full sweep
    "fast": {"models": MODELS_FULL, "seeds": [0], "total_steps": 600_000},
    # tiny end-to-end check
    "smoke": {"models": ["mlp", "lrc"], "seeds": [0], "total_steps": 16_384},
}


def build_specs(models, seeds):
    return [{"model": m, "seed": s} for m in models for s in seeds]


def _resolve_matrix(args):
    prof = PROFILES[args.profile]
    models = args.models.split(",") if args.models else list(prof["models"])
    seeds = ([int(x) for x in args.seeds.split(",")] if args.seeds else list(prof["seeds"]))
    return build_specs(models, seeds)


def _make_cfg(args):
    from src.tasks.mujoco_rl.ppo import PPOConfig
    return PPOConfig(
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        total_steps=args.total_steps,
        update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches,
        lr=args.lr,
        units=args.units,
        max_seconds=args.max_minutes * 60.0 if args.max_minutes else 0.0,
    )


def run_one(spec, args):
    from src.tasks.mujoco_rl.ppo import train
    from src.tasks.mujoco_rl import record as rec

    model, seed = spec["model"], spec["seed"]
    cfg = _make_cfg(args)
    tag = f"{model}_s{seed}"
    runs_dir = os.path.join(args.out_dir, "runs")
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    vid_dir = os.path.join(args.out_dir, "videos")
    for d in (runs_dir, ckpt_dir, vid_dir):
        os.makedirs(d, exist_ok=True)

    def log_fn(update, num_updates, global_step, mean_return, mean_len, sps):
        if update == 1 or update % max(1, num_updates // 50) == 0 or update == num_updates:
            print(f"[{tag}] upd {update}/{num_updates} step {global_step} "
                  f"return {mean_return:8.1f} len {mean_len:6.0f} {sps} sps", flush=True)

    print(f"=== TRAIN {tag} task={args.task} total_steps={cfg.total_steps} "
          f"units={cfg.units} envs={cfg.num_envs}x{cfg.num_steps} ===", flush=True)
    t0 = time.time()
    out = train(model, args.task, seed, cfg, log_fn=log_fn)
    wall = time.time() - t0

    ckpt_path = os.path.join(ckpt_dir, f"{tag}.npz")
    rec.save_policy(ckpt_path, out["policy"], out["obs_mean"], out["obs_var"],
                    extra={"task": args.task, "seed": seed,
                           "final_return": out["final_return"]})

    video_path, eval_return = None, None
    if not args.no_video:
        try:
            vp = os.path.join(vid_dir, f"{tag}.mp4")
            r = rec.record_episode(out["policy"], args.task, out["obs_mean"],
                                   out["obs_var"], vp, seed=args.eval_seed,
                                   label=f"{model} (seed {seed})")
            video_path, eval_return = r["video"], r["return"]
            print(f"[{tag}] video -> {vp}  eval_return={eval_return:.1f}", flush=True)
        except Exception as e:  # rendering must never fail the run
            print(f"[{tag}] WARNING: video rendering failed ({e!r}); "
                  f"checkpoint saved for later rendering", flush=True)

    # Downsample history so the JSON stays small.
    hist = out["history"]
    step = max(1, len(hist) // 200)
    hist_ds = [{"update": h["update"], "global_step": h["global_step"],
                "mean_return": h["mean_return"]} for h in hist[::step]]

    result = {
        "model": model, "seed": seed, "task": args.task,
        "total_steps": cfg.total_steps, "global_step": out["global_step"],
        "units": cfg.units, "num_envs": cfg.num_envs, "num_steps": cfg.num_steps,
        "update_epochs": cfg.update_epochs, "num_minibatches": cfg.num_minibatches,
        "lr": cfg.lr, "final_return": out["final_return"], "best_return": out["best_return"],
        "eval_return": eval_return, "wallclock_s": round(wall, 1),
        "video": video_path, "checkpoint": ckpt_path, "history": hist_ds,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
    }
    out_path = os.path.join(runs_dir, f"{tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[{tag}] done in {wall:.0f}s  final={out['final_return']:.1f}  "
          f"best={out['best_return']:.1f}  -> {out_path}", flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description="MuJoCo PPO control benchmark (model x seed)")
    p.add_argument("--list", action="store_true", help="print all run specs and exit")
    p.add_argument("--count", action="store_true", help="print number of specs and exit")
    p.add_argument("--index", type=int, default=None, help="run spec by index (SLURM array)")
    p.add_argument("--all", action="store_true", help="run all specs sequentially")
    p.add_argument("--profile", choices=list(PROFILES), default="v1")
    p.add_argument("--model", default=None, help="single model (overrides matrix)")
    p.add_argument("--seed", type=int, default=None, help="single seed (overrides matrix)")
    p.add_argument("--models", default=None, help="comma-separated model subset")
    p.add_argument("--seeds", default=None, help="comma-separated seed subset")
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--total-steps", type=int, default=None)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=128)
    p.add_argument("--units", type=int, default=64)
    p.add_argument("--update-epochs", type=int, default=10)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-minutes", type=float, default=0.0, help="wall-clock cap per run")
    p.add_argument("--eval-seed", type=int, default=12345)
    p.add_argument("--out-dir", default="results/mujoco")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--cpu", action="store_true", help="hide GPUs (force CPU)")
    args = p.parse_args()

    # default total_steps from the profile unless overridden
    if args.total_steps is None:
        args.total_steps = PROFILES[args.profile]["total_steps"]

    # Single (model, seed) selection short-circuits the matrix.
    if args.model is not None and args.seed is not None:
        specs = [{"model": args.model, "seed": args.seed}]
    else:
        specs = _resolve_matrix(args)

    if args.list:
        for i, s in enumerate(specs):
            print(f"{i:3d}  {s['model']:6s}  seed={s['seed']}")
        return
    if args.count:
        print(len(specs))
        return

    if args.cpu:
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")

    if args.index is not None:
        if not 0 <= args.index < len(specs):
            raise SystemExit(f"index {args.index} out of range [0,{len(specs)})")
        run_one(specs[args.index], args)
    elif args.all:
        for s in specs:
            run_one(s, args)
    elif args.model is not None and args.seed is not None:
        run_one(specs[0], args)
    else:
        raise SystemExit("nothing to do: pass --index, --all, --list, --count, "
                         "or --model with --seed")


if __name__ == "__main__":
    main()
