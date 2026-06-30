"""PPO trainer for the MuJoCo control benchmark.

A compact, single-file PPO that follows the well-validated CleanRL
``ppo_continuous_action`` recipe (GAE, clipped surrogate objective, clipped value
loss, advantage normalization, observation/reward normalization, orthogonal init,
linear LR anneal) so that even modest training budgets produce visibly competent
locomotion. The only addition over the feedforward recipe is recurrent support:
for the thesis cells the actor and critic each carry a hidden state across the
episode, reset on episode boundaries, and the update replays each rollout chunk
from its stored initial state (truncated BPTT over ``num_steps``).

State / done bookkeeping mirrors CleanRL exactly so collection and the update
replay are bit-for-bit consistent:
  * ``dones[t]`` is the done flag *entering* step t (the result of step t-1);
  * the hidden state is reset using ``dones[t]`` *before* the forward at step t;
  * the bootstrap value resets on ``next_done`` before evaluating ``next_obs``.
"""
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from .envs import make_train_envs, get_obs_stats, clip_obs
from .policies import GaussianPolicy


@dataclass
class PPOConfig:
    num_envs: int = 8
    num_steps: int = 128
    total_steps: int = 1_000_000
    update_epochs: int = 10
    num_minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    anneal_lr: bool = True
    norm_adv: bool = True
    clip_vloss: bool = True
    units: int = 64
    max_seconds: float = 0.0  # optional wall-clock cap; 0 = train all updates


def _compute_gae(rewards, values, dones, next_value, next_done, gamma, lam):
    """Vectorized (over envs) GAE. All arrays shaped (T, N); next_* shaped (N,)."""
    T, N = rewards.shape
    adv = np.zeros((T, N), np.float32)
    lastgae = np.zeros(N, np.float32)
    for t in reversed(range(T)):
        if t == T - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        lastgae = delta + gamma * lam * nextnonterminal * lastgae
        adv[t] = lastgae
    returns = adv + values
    return adv, returns


class PPOTrainer:
    def __init__(self, model_type, task, seed, cfg: PPOConfig, log_fn=None):
        self.model_type = model_type
        self.task = task
        self.seed = int(seed)
        self.cfg = cfg
        self.log_fn = log_fn or (lambda **kw: None)

        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        self.envs, self.norm_obs = make_train_envs(task, cfg.num_envs, cfg.gamma, self.seed)
        self.obs_dim = int(np.prod(self.envs.single_observation_space.shape))
        self.act_dim = int(np.prod(self.envs.single_action_space.shape))
        self.policy = GaussianPolicy(model_type, self.obs_dim, self.act_dim,
                                     units=cfg.units, seed=self.seed)
        self.opt = tf.keras.optimizers.Adam(learning_rate=cfg.lr, epsilon=1e-5)
        # Force-build optimizer slots so apply_gradients inside tf.function is stable.
        self.opt.build(self.policy.trainable_variables)

    # ----------------------------------------------------------- acting (graph)
    @tf.function(reduce_retracing=True)
    def _act(self, obs, a_state, c_state):
        mean, a2 = self.policy.actor_forward(obs, a_state)
        value, c2 = self.policy.value_forward(obs, c_state)
        action = self.policy.sample_action(mean)
        logp = self.policy.logprob(mean, action)
        return action, logp, value, a2, c2

    @tf.function(reduce_retracing=True)
    def _value(self, obs, c_state):
        value, _ = self.policy.value_forward(obs, c_state)
        return value

    # ----------------------------------------------------- update steps (graph)
    @tf.function(reduce_retracing=True)
    def _update_mlp(self, obs, act, oldlogp, adv, ret, oldval):
        with tf.GradientTape() as tape:
            mean, _ = self.policy.actor_forward(obs, None)
            value, _ = self.policy.value_forward(obs, None)
            newlogp = self.policy.logprob(mean, act)
            loss = self._ppo_loss(newlogp, value, oldlogp, adv, ret, oldval)
        self._apply(tape, loss)
        return loss

    @tf.function(reduce_retracing=True)
    def _update_recurrent(self, obs_seq, act_seq, done_seq, oldlogp, adv, ret, oldval,
                          init_a, init_c):
        # Static unroll over the (statically known) rollout horizon T: this avoids
        # AutoGraph while_loop carry issues with Python-list hidden states (e.g.
        # LSTM's [h, c]). T <= num_steps, so the traced graph is bounded.
        T = int(obs_seq.shape[0])
        with tf.GradientTape() as tape:
            ha = list(init_a)
            hc = list(init_c)
            logps, vals = [], []
            for t in range(T):
                dm = tf.reshape(done_seq[t], [-1, 1])
                keep = 1.0 - dm
                ha = [h * keep for h in ha]
                hc = [h * keep for h in hc]
                mean, ha = self.policy.actor_forward(obs_seq[t], ha)
                value, hc = self.policy.value_forward(obs_seq[t], hc)
                logps.append(self.policy.logprob(mean, act_seq[t]))
                vals.append(value)
            newlogp = tf.stack(logps)
            value = tf.stack(vals)
            loss = self._ppo_loss(tf.reshape(newlogp, [-1]), tf.reshape(value, [-1]),
                                  tf.reshape(oldlogp, [-1]), tf.reshape(adv, [-1]),
                                  tf.reshape(ret, [-1]), tf.reshape(oldval, [-1]))
        self._apply(tape, loss)
        return loss

    def _ppo_loss(self, newlogp, value, oldlogp, adv, ret, oldval):
        cfg = self.cfg
        if cfg.norm_adv:
            adv = (adv - tf.reduce_mean(adv)) / (tf.math.reduce_std(adv) + 1e-8)
        ratio = tf.exp(newlogp - oldlogp)
        pg1 = -adv * ratio
        pg2 = -adv * tf.clip_by_value(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
        pg_loss = tf.reduce_mean(tf.maximum(pg1, pg2))
        if cfg.clip_vloss:
            v_unclipped = tf.square(value - ret)
            v_clipped = oldval + tf.clip_by_value(value - oldval, -cfg.clip_coef, cfg.clip_coef)
            v_loss = 0.5 * tf.reduce_mean(tf.maximum(v_unclipped, tf.square(v_clipped - ret)))
        else:
            v_loss = 0.5 * tf.reduce_mean(tf.square(value - ret))
        ent = self.policy.entropy()
        return pg_loss - cfg.ent_coef * ent + cfg.vf_coef * v_loss

    def _apply(self, tape, loss):
        var = self.policy.trainable_variables
        grads = tape.gradient(loss, var)
        grads, _ = tf.clip_by_global_norm(grads, self.cfg.max_grad_norm)
        self.opt.apply_gradients(zip(grads, var))

    # ------------------------------------------------------------------- train
    def train(self):
        cfg = self.cfg
        N, T = cfg.num_envs, cfg.num_steps
        batch = N * T
        num_updates = max(1, cfg.total_steps // batch)
        recurrent = self.policy.recurrent

        obs, _ = self.envs.reset(seed=self.seed)
        obs = clip_obs(np.asarray(obs, np.float32))
        a_state, c_state = self.policy.initial_state(N)
        next_done = np.zeros(N, np.float32)

        ep_returns = deque(maxlen=100)
        ep_lengths = deque(maxlen=100)
        history = []
        best_return = -1e18
        global_step = 0
        start = time.time()

        for update in range(1, num_updates + 1):
            if cfg.anneal_lr:
                self.opt.learning_rate.assign(cfg.lr * (1.0 - (update - 1) / num_updates))

            b_obs = np.zeros((T, N, self.obs_dim), np.float32)
            b_act = np.zeros((T, N, self.act_dim), np.float32)
            b_logp = np.zeros((T, N), np.float32)
            b_val = np.zeros((T, N), np.float32)
            b_rew = np.zeros((T, N), np.float32)
            b_done = np.zeros((T, N), np.float32)

            init_a = [h.numpy() for h in a_state] if recurrent else None
            init_c = [h.numpy() for h in c_state] if recurrent else None

            for t in range(T):
                b_obs[t] = obs
                b_done[t] = next_done
                if recurrent:
                    dm = tf.constant(next_done[:, None], tf.float32)
                    a_state = self.policy.reset_state(a_state, dm)
                    c_state = self.policy.reset_state(c_state, dm)
                action, logp, value, a_state, c_state = self._act(
                    tf.constant(obs), a_state, c_state)
                action = action.numpy()
                # Store/score the UNCLIPPED Gaussian sample (CleanRL convention):
                # the env's ClipAction wrapper makes clipping part of the env, so
                # the behaviour log-prob is the Gaussian log-prob of the sample.
                b_act[t] = action
                b_logp[t] = logp.numpy()
                b_val[t] = value.numpy()

                clipped = np.clip(action, -1.0, 1.0)
                obs_next, rew, term, trunc, info = self.envs.step(clipped)
                # done = term OR trunc (CleanRL convention). HalfCheetah only ever
                # truncates (no early termination), so episode-end value is not
                # bootstrapped -- a shared, architecture-independent bias on absolute
                # returns that leaves the relative model ranking intact. Proper
                # truncation bootstrapping (via info["final_observation"]) is a
                # deferred refinement, not needed for the model comparison.
                next_done = np.logical_or(term, trunc).astype(np.float32)
                b_rew[t] = np.asarray(rew, np.float32)
                obs = clip_obs(np.asarray(obs_next, np.float32))
                global_step += N

                ep = info.get("episode")
                if ep is not None:
                    r = np.asarray(ep["r"], np.float32).reshape(-1)
                    mask = info.get("_episode")
                    mask = np.ones_like(r, bool) if mask is None else np.asarray(mask, bool).reshape(-1)
                    lns = np.asarray(ep["l"], np.float32).reshape(-1)
                    for rr, ll in zip(r[mask], lns[mask]):
                        ep_returns.append(float(rr))
                        ep_lengths.append(float(ll))

            # bootstrap value at next_obs (state reset on next_done, as in collection)
            if recurrent:
                dm = tf.constant(next_done[:, None], tf.float32)
                c_boot = self.policy.reset_state(c_state, dm)
            else:
                c_boot = None
            next_value = self._value(tf.constant(obs), c_boot).numpy()

            adv, ret = _compute_gae(b_rew, b_val, b_done, next_value, next_done,
                                    cfg.gamma, cfg.gae_lambda)

            self._optimize(b_obs, b_act, b_logp, b_val, b_done, adv, ret,
                           init_a, init_c, recurrent)

            mean_ret = float(np.mean(ep_returns)) if ep_returns else float("nan")
            mean_len = float(np.mean(ep_lengths)) if ep_lengths else float("nan")
            if ep_returns and mean_ret > best_return:
                best_return = mean_ret
            sps = int(global_step / (time.time() - start + 1e-9))
            history.append({"update": update, "global_step": global_step,
                            "mean_return": mean_ret, "mean_len": mean_len, "sps": sps})
            self.log_fn(update=update, num_updates=num_updates, global_step=global_step,
                        mean_return=mean_ret, mean_len=mean_len, sps=sps)

            if cfg.max_seconds and (time.time() - start) > cfg.max_seconds:
                break

        final_return = float(np.mean(ep_returns)) if ep_returns else float("nan")
        obs_mean, obs_var = get_obs_stats(self.norm_obs)
        self.envs.close()
        return {
            "policy": self.policy,
            "obs_mean": obs_mean,
            "obs_var": obs_var,
            "history": history,
            "final_return": final_return,
            "best_return": float(best_return) if best_return > -1e17 else float("nan"),
            "global_step": global_step,
        }

    def _optimize(self, b_obs, b_act, b_logp, b_val, b_done, adv, ret,
                  init_a, init_c, recurrent):
        cfg = self.cfg
        N, T = cfg.num_envs, cfg.num_steps
        if recurrent:
            assert N % cfg.num_minibatches == 0, "num_envs must be divisible by num_minibatches"
            envs_per_mb = N // cfg.num_minibatches
            for _ in range(cfg.update_epochs):
                perm = np.random.permutation(N)
                for s in range(0, N, envs_per_mb):
                    idx = perm[s:s + envs_per_mb]
                    self._update_recurrent(
                        tf.constant(b_obs[:, idx]), tf.constant(b_act[:, idx]),
                        tf.constant(b_done[:, idx]), tf.constant(b_logp[:, idx]),
                        tf.constant(adv[:, idx]), tf.constant(ret[:, idx]),
                        tf.constant(b_val[:, idx]),
                        [tf.constant(h[idx]) for h in init_a],
                        [tf.constant(h[idx]) for h in init_c])
        else:
            batch = N * T
            mb = batch // cfg.num_minibatches
            f_obs = b_obs.reshape(batch, self.obs_dim)
            f_act = b_act.reshape(batch, self.act_dim)
            f_logp = b_logp.reshape(batch)
            f_val = b_val.reshape(batch)
            f_adv = adv.reshape(batch)
            f_ret = ret.reshape(batch)
            for _ in range(cfg.update_epochs):
                perm = np.random.permutation(batch)
                for s in range(0, batch, mb):
                    idx = perm[s:s + mb]
                    self._update_mlp(
                        tf.constant(f_obs[idx]), tf.constant(f_act[idx]),
                        tf.constant(f_logp[idx]), tf.constant(f_adv[idx]),
                        tf.constant(f_ret[idx]), tf.constant(f_val[idx]))


def train(model_type, task, seed, cfg, log_fn=None):
    return PPOTrainer(model_type, task, seed, cfg, log_fn=log_fn).train()
