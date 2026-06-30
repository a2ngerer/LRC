"""Render a trained policy to an mp4 and (de)serialize policy checkpoints.

The video rollout is deterministic (mean action, no exploration noise) and
applies the *frozen* observation normalization from training, so the cheetah we
see is the policy acting exactly as trained. A small text overlay (model name,
step, cumulative raw reward) is burned in best-effort via Pillow.

Checkpoints are plain ``.npz``: the policy weight arrays ``w0..wK`` plus the
observation statistics and the architecture metadata needed to rebuild the
policy. This decouples training from rendering -- the cluster can train and save
a checkpoint, and rendering can happen anywhere (e.g. locally, where MuJoCo
offscreen rendering is known to work).
"""
import os

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import tensorflow as tf

from .envs import normalize_obs
from .policies import GaussianPolicy


def save_policy(path, policy, obs_mean, obs_var, extra=None):
    """Serialize a trained policy + obs stats + architecture metadata to ``path`` (.npz)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    weights = policy.get_weights()
    payload = {f"w{i}": w for i, w in enumerate(weights)}
    payload["n_weights"] = np.array(len(weights))
    payload["obs_mean"] = np.asarray(obs_mean, np.float32)
    payload["obs_var"] = np.asarray(obs_var, np.float32)
    payload["model_type"] = np.array(policy.model_type)
    payload["units"] = np.array(policy.units)
    payload["obs_dim"] = np.array(policy.obs_dim)
    payload["act_dim"] = np.array(policy.act_dim)
    for k, v in (extra or {}).items():
        payload[k] = np.array(v)
    np.savez(path, **payload)
    return path


def load_policy(path):
    """Rebuild a GaussianPolicy from an .npz checkpoint. Returns (policy, obs_mean, obs_var, meta)."""
    d = np.load(path, allow_pickle=False)
    model_type = str(d["model_type"])
    units = int(d["units"])
    obs_dim = int(d["obs_dim"])
    act_dim = int(d["act_dim"])
    policy = GaussianPolicy(model_type, obs_dim, act_dim, units=units, seed=0)
    n = int(d["n_weights"])
    policy.set_weights([d[f"w{i}"] for i in range(n)])
    return policy, d["obs_mean"], d["obs_var"], {"model_type": model_type, "units": units}


def _overlay(frame, lines):
    """Burn text lines into the top-left of an RGB uint8 frame (best-effort)."""
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return frame
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    y = 4
    for ln in lines:
        draw.text((6, y), ln, fill=(255, 255, 0))
        draw.text((5, y), ln, fill=(20, 20, 20))  # faux outline
        y += 14
    return np.asarray(img)


def record_episode(policy, task, obs_mean, obs_var, video_path, seed=0,
                   max_steps=1000, fps=30, label=None, overlay=True):
    """Roll out the deterministic policy for one episode and write an mp4.

    Returns a dict: {"return": raw_return, "length": n_steps, "video": video_path}.
    """
    os.makedirs(os.path.dirname(os.path.abspath(video_path)), exist_ok=True)
    env = gym.make(task, render_mode="rgb_array")
    env = gym.wrappers.ClipAction(env)
    obs, _ = env.reset(seed=seed)

    astate, _ = policy.initial_state(1)
    frames, total = [], 0.0
    label = label or policy.model_type

    for t in range(max_steps):
        o = normalize_obs(np.asarray(obs, np.float32), obs_mean, obs_var)[None]
        mean, astate = policy.actor_forward(tf.constant(o, tf.float32), astate)
        action = np.clip(np.asarray(mean[0]), -1.0, 1.0)
        obs, r, term, trunc, _ = env.step(action)
        total += float(r)
        frame = env.render()
        if overlay:
            frame = _overlay(frame, [f"{label}", f"step {t+1}", f"return {total:7.1f}"])
        frames.append(frame)
        if term or trunc:
            break
    env.close()

    imageio.mimwrite(video_path, frames, fps=fps, codec="libx264",
                     quality=8, macro_block_size=1)
    return {"return": float(total), "length": len(frames), "video": video_path}
