"""Vectorized + normalized MuJoCo envs for the PPO control benchmark.

Reproduces the standard "PPO that works on MuJoCo" wrapper stack (the CleanRL /
Stable-Baselines defaults): per-env action clipping, episode-statistics logging
on the *raw* reward, then vector-level observation and reward normalization with
running statistics. Observation clipping is applied by the trainer after each
step (kept out of the wrapper stack so the eval/video rollout can apply the
identical transform with frozen statistics).

The observation running mean/var are exposed via ``get_obs_stats`` so the video
rollout normalizes inputs the same way training did -- without it a trained
policy sees out-of-distribution observations and walks like it never learned.
"""
import gymnasium as gym
import gymnasium.wrappers.vector as vw
import numpy as np

OBS_CLIP = 10.0
REW_CLIP = 10.0
NORM_EPS = 1e-8


def make_train_envs(task, num_envs, gamma, seed):
    """Build a synchronous vector of ``num_envs`` normalized training envs.

    Wrapper order (inner -> outer):
        ClipAction              clamp actions to the env's action bounds
        RecordEpisodeStatistics log raw episodic return/length (before reward norm)
        NormalizeObservation    running-mean/var normalize observations
        NormalizeReward         normalize the discounted return scale
        ClipReward              clamp normalized reward to +/- REW_CLIP

    Returns:
        (envs, norm_obs) where ``norm_obs`` is the NormalizeObservation wrapper
        instance, kept so the trainer can read ``norm_obs.obs_rms`` after training.
    """
    envs = gym.make_vec(task, num_envs=num_envs, vectorization_mode="sync")
    envs = vw.ClipAction(envs)
    envs = vw.RecordEpisodeStatistics(envs)
    norm_obs = vw.NormalizeObservation(envs, epsilon=NORM_EPS)
    envs = vw.NormalizeReward(norm_obs, gamma=gamma, epsilon=NORM_EPS)
    envs = vw.ClipReward(envs, min_reward=-REW_CLIP, max_reward=REW_CLIP)
    envs.action_space.seed(seed)
    return envs, norm_obs


def get_obs_stats(norm_obs):
    """Extract the frozen observation normalization statistics as float32 arrays.

    Returns:
        (mean, var) each of shape (obs_dim,). Apply at eval time as
        ``clip((obs - mean) / sqrt(var + NORM_EPS), -OBS_CLIP, OBS_CLIP)``.
    """
    rms = norm_obs.obs_rms
    return (np.asarray(rms.mean, dtype=np.float32),
            np.asarray(rms.var, dtype=np.float32))


def normalize_obs(obs, mean, var):
    """Apply frozen observation normalization + clipping (eval/record path)."""
    return np.clip((obs - mean) / np.sqrt(var + NORM_EPS), -OBS_CLIP, OBS_CLIP)


def clip_obs(obs):
    """Clip an already-normalized observation batch to +/- OBS_CLIP (train path)."""
    return np.clip(obs, -OBS_CLIP, OBS_CLIP)
