"""Fixed-seed evaluation of a trained policy, separate from the training returns.

``final_return`` of a PPO run is the mean over the last training episodes: noisy
exploration actions, drifting policy, whatever episodes happened to finish. The
number reported for a comparison is instead the return of the *final* policy on
``EVAL_EPISODES`` fixed episodes (env seeds ``EVAL_SEED0 + i``), acting
deterministically (mean action) under the given observation statistics -- the
frozen ones of training for the final policy, as in ``record.record_episode``;
the runner passes the initial identity statistics for the untrained policy.
"""
import gymnasium as gym
import numpy as np
import tensorflow as tf

from .envs import normalize_obs

EVAL_EPISODES = 20
EVAL_SEED0 = 12345


def evaluate_policy(policy, task, obs_mean, obs_var, n_episodes=EVAL_EPISODES,
                    seed0=EVAL_SEED0, env_fn=None):
    """Run ``n_episodes`` deterministic episodes with env seeds ``seed0 + i``.

    The episodes advance in lock-step so the policy is evaluated once per step on
    the batch of live episodes; every episode keeps its own env and hidden state.
    Batch mates do not interact, but the batch size changes float rounding, which
    chaotic dynamics (HalfCheetah) amplify: returns are reproducible for a fixed
    ``n_episodes`` (the protocol's 20), not across different batch sizes. Returns
    are the raw env reward (no reward normalization).

    Args:
        env_fn: zero-argument factory of one env; defaults to ``gym.make(task)``
                with ``ClipAction``. Exposed so tests can inject a stub.

    Returns:
        dict with ``returns`` / ``lengths`` (per episode, in seed order),
        ``mean``, ``std`` (sample std, ddof=1; 0.0 for one episode) and ``seeds``.
    """
    make = env_fn or (lambda: gym.wrappers.ClipAction(gym.make(task)))
    envs = [make() for _ in range(n_episodes)]
    seeds = [seed0 + i for i in range(n_episodes)]
    obs = np.stack([np.asarray(e.reset(seed=s)[0], np.float32)
                    for e, s in zip(envs, seeds)])
    astate, _ = policy.initial_state(n_episodes)
    alive = np.ones(n_episodes, bool)
    returns = np.zeros(n_episodes, np.float64)
    lengths = np.zeros(n_episodes, np.int64)
    while alive.any():
        o = normalize_obs(obs, obs_mean, obs_var).astype(np.float32)
        mean, astate = policy.actor_forward(tf.constant(o), astate)
        mean = np.clip(mean.numpy(), -1.0, 1.0)
        for i in np.flatnonzero(alive):
            obs_i, r, term, trunc, _ = envs[i].step(mean[i])
            obs[i] = obs_i
            returns[i] += float(r)
            lengths[i] += 1
            if term or trunc:
                alive[i] = False
    for e in envs:
        e.close()
    return {
        "returns": returns.tolist(),
        "lengths": lengths.tolist(),
        "mean": float(returns.mean()),
        "std": float(returns.std(ddof=1)) if n_episodes > 1 else 0.0,
        "seeds": seeds,
    }
