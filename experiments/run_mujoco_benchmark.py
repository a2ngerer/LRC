"""MuJoCo control-benchmark runner: one run = (model, seed) on one task.

Mirrors the conventions of ``run_benchmark.py`` (the Neural-ODE runner):
``--list`` / ``--count`` / ``--index`` for SLURM array submission, plus matrix
filters (``--models`` / ``--seeds``) and a ``--profile`` of sensible defaults.

A run trains a PPO policy (see ``src.tasks.mujoco_rl.ppo``), evaluates the final
policy on 20 fixed episodes (``eval_return_mean`` / ``eval_return_std``, kept
apart from the training returns ``final_return`` / ``best_return``), saves a
JSON metrics record, a ``.npz`` policy checkpoint, and -- unless ``--no-video``
-- a rendered mp4 of a deterministic rollout. Rendering is wrapped so a render
failure (e.g. no GL backend on a compute node) never fails the training run:
the checkpoint is always written, so the video can be re-rendered later from
the checkpoint.

The reduced RQ1 track (issue #61) is ``--profile final`` / ``final_numeric``:
cells take their kwargs from ``src.benchmark.registry``, and ``--param-budget``
sizes each torso so actor + critic land closest to the budget.

Examples
--------
    uv run python experiments/run_mujoco_benchmark.py --list
    uv run python experiments/run_mujoco_benchmark.py --index 0 --profile v1
    uv run python experiments/run_mujoco_benchmark.py --index 0 --profile final
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
BUDGET_TOLERANCE = 0.05      # as in the other task runners
MODELS_FULL = ["mlp", "lstm", "gru", "ctrnn", "ltc", "lrc", "cfc"]

# Reduced PPO track (issue #61): lr 3e-4 (the --lr default) is fixed, no tuning;
# actor + critic are sized to the 12k budget. Seeds per model: the tested
# contrast (cfc_lrc vs gru) gets ten, the descriptive arms five.
FINAL_BUDGET = 12_000
_FINAL_SEEDS = {"cfc_lrc": list(range(10)), "gru": list(range(10)),
                "mlp": list(range(5)), "cfc": list(range(5))}

PROFILES = {
    # full thesis matrix: 7 models x 3 seeds = 21 runs
    "v1": {"models": MODELS_FULL, "seeds": [0, 1, 2], "total_steps": 1_000_000},
    # one seed per model -- a quick full sweep
    "fast": {"models": MODELS_FULL, "seeds": [0], "total_steps": 600_000},
    # tiny end-to-end check
    "smoke": {"models": ["mlp", "lrc"], "seeds": [0], "total_steps": 16_384},
    # reduced track, four closed-form/discrete cells: 30 runs
    "final": {"models": ["mlp", "gru", "cfc", "cfc_lrc"], "seeds": _FINAL_SEEDS,
              "total_steps": 1_000_000, "param_budget": FINAL_BUDGET, "no_video": True},
    # final plus the numerical cells, one ODE unfold each: 40 runs
    "final_numeric": {"models": ["mlp", "gru", "cfc", "cfc_lrc", "lrc", "ltc"],
                      "seeds": {**_FINAL_SEEDS, "lrc": list(range(5)), "ltc": list(range(5))},
                      "total_steps": 1_000_000, "param_budget": FINAL_BUDGET,
                      "ode_unfolds": 1, "no_video": True},
}


def build_specs(models, seeds):
    """``seeds``: one list for every model, or a ``{model: list}`` mapping."""
    return [{"model": m, "seed": s} for m in models
            for s in (seeds[m] if isinstance(seeds, dict) else seeds)]


def _resolve_matrix(args):
    prof = PROFILES[args.profile]
    models = args.models.split(",") if args.models else list(prof["models"])
    seeds = ([int(x) for x in args.seeds.split(",")] if args.seeds else prof["seeds"])
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
        param_budget=args.param_budget,
        max_seconds=args.max_minutes * 60.0 if args.max_minutes else 0.0,
    )


def run_one(spec, args):
    from src.benchmark.registry import cell_kwargs as registry_cell_kwargs
    from src.tasks.mujoco_rl.evaluation import evaluate_policy
    from src.tasks.mujoco_rl.ppo import PPOTrainer
    from src.tasks.mujoco_rl import record as rec
    from src.tasks.person_activity.model import ode_unfolds_override

    model, seed = spec["model"], spec["seed"]
    cfg = _make_cfg(args)
    cell_kwargs = registry_cell_kwargs(model)
    if args.ode_unfolds is not None:     # only the numerical cells take the kwarg
        cell_kwargs.update(ode_unfolds_override(model, args.ode_unfolds))
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

    trainer = PPOTrainer(model, args.task, seed, cfg, log_fn=log_fn, cell_kwargs=cell_kwargs)
    budget_dev = trainer.budget_deviation
    above_tol = budget_dev is not None and budget_dev > BUDGET_TOLERANCE
    if cfg.param_budget:
        print(f"[{tag}] budget {cfg.param_budget}: units={trainer.policy.units} "
              f"params_effective={trainer.params_effective} (dev {budget_dev:.1%})"
              f"{'  ABOVE TOLERANCE' if above_tol else ''}", flush=True)
    print(f"=== TRAIN {tag} task={args.task} total_steps={cfg.total_steps} "
          f"units={trainer.policy.units} params={trainer.params_effective} "
          f"envs={cfg.num_envs}x{cfg.num_steps} cell_kwargs={cell_kwargs} ===", flush=True)
    t0 = time.time()
    out = trainer.train()
    wall = time.time() - t0

    ckpt_path = os.path.join(ckpt_dir, f"{tag}.npz")
    rec.save_policy(ckpt_path, out["policy"], out["obs_mean"], out["obs_var"],
                    extra={"task": args.task, "seed": seed,
                           "final_return": out["final_return"]})

    # Final policy on 20 fixed episodes, apart from the training returns.
    ev = evaluate_policy(out["policy"], args.task, out["obs_mean"], out["obs_var"],
                         seed0=args.eval_seed)
    print(f"[{tag}] eval {len(ev['returns'])} episodes: "
          f"{ev['mean']:.1f} +- {ev['std']:.1f}", flush=True)

    video_path, video_return = None, None
    if not args.no_video:
        try:
            vp = os.path.join(vid_dir, f"{tag}.mp4")
            r = rec.record_episode(out["policy"], args.task, out["obs_mean"],
                                   out["obs_var"], vp, seed=args.eval_seed,
                                   label=f"{model} (seed {seed})")
            video_path, video_return = r["video"], r["return"]
            print(f"[{tag}] video -> {vp}  video_return={video_return:.1f}", flush=True)
        except Exception as e:  # rendering must never fail the run
            print(f"[{tag}] WARNING: video rendering failed ({e!r}); "
                  f"checkpoint saved for later rendering", flush=True)

    # Downsample history so the JSON stays small.
    hist = out["history"]
    step = max(1, len(hist) // 200)
    hist_ds = [{"update": h["update"], "global_step": h["global_step"],
                "mean_return": h["mean_return"]} for h in hist[::step]]

    result = {
        "model": model, "seed": seed, "task": args.task, "profile": args.profile,
        "total_steps": cfg.total_steps, "global_step": out["global_step"],
        "units": out["units"], "param_budget": cfg.param_budget,
        "params_effective": out["params_effective"], "params_raw": out["params_raw"],
        "budget_deviation": budget_dev, "budget_above_tolerance": above_tol,
        "cell_kwargs": cell_kwargs,
        "num_envs": cfg.num_envs, "num_steps": cfg.num_steps,
        "update_epochs": cfg.update_epochs, "num_minibatches": cfg.num_minibatches,
        "lr": cfg.lr,
        # training-episode statistics (noisy exploration policy, last 100 episodes)
        "final_return": out["final_return"], "best_return": out["best_return"],
        # final policy, deterministic, env seeds eval_seed + i
        "eval_return_mean": ev["mean"], "eval_return_std": ev["std"],
        "eval_returns": ev["returns"], "eval_lengths": ev["lengths"],
        "eval_seed": args.eval_seed,
        "video_return": video_return, "wallclock_s": round(wall, 1),
        "video": video_path, "checkpoint": ckpt_path, "history": hist_ds,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
    }
    out_path = os.path.join(runs_dir, f"{tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[{tag}] done in {wall:.0f}s  final={out['final_return']:.1f}  "
          f"best={out['best_return']:.1f}  eval={ev['mean']:.1f}  -> {out_path}", flush=True)
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
    p.add_argument("--units", type=int, default=64,
                   help="torso width; replaced by the sized width under --param-budget")
    p.add_argument("--param-budget", type=int, default=None,
                   help="effective parameters of actor + critic; sizes each torso "
                        "(default: the profile's, if it has one)")
    p.add_argument("--ode-unfolds", type=int, default=None,
                   help="ODE unfolds of the numerical cells (ltc, lrc, ctrnn); "
                        "default: the profile's, else the cell default")
    p.add_argument("--update-epochs", type=int, default=10)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-minutes", type=float, default=0.0, help="wall-clock cap per run")
    p.add_argument("--eval-seed", type=int, default=12345,
                   help="env seed of evaluation episode 0 (episode i uses seed + i)")
    p.add_argument("--out-dir", default="results/mujoco")
    p.add_argument("--no-video", action="store_true",
                   help="skip the video episode (the final profiles default to this)")
    p.add_argument("--cpu", action="store_true", help="hide GPUs (force CPU)")
    args = p.parse_args()

    # profile defaults unless overridden
    prof = PROFILES[args.profile]
    if args.total_steps is None:
        args.total_steps = prof["total_steps"]
    if args.param_budget is None:
        args.param_budget = prof.get("param_budget")
    if args.ode_unfolds is None:
        args.ode_unfolds = prof.get("ode_unfolds")
    args.no_video = args.no_video or prof.get("no_video", False)

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
