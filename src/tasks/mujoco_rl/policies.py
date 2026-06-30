"""Gaussian actor-critic policies for PPO, feedforward or single-cell recurrent.

A single class, ``GaussianPolicy``, serves both the feedforward MLP baseline and
every recurrent cell type from the thesis registry. The recurrent variants use
one cell as the actor torso and a *separate* cell as the critic torso, each
carrying its own hidden state across the episode. The cell registry is reused
verbatim from ``src.models.rnn_model`` so the RL policies are the exact same
neuron implementations benchmarked on the Neural-ODE tasks.

Hidden state is always represented as a Python list of tensors (LSTM uses
``[h, c]``; the continuous-time cells use ``[h]``), so the reset / carry logic in
the trainer is uniform across cell types.

Action distribution: diagonal Gaussian with a state-independent, learnable
log-std (the standard MuJoCo-PPO parameterization). Actions are sampled
unsquashed and clipped to the action bounds at env-step time; log-probs are
computed on the unclipped sample (CleanRL convention).
"""
import numpy as np
import tensorflow as tf

from src.models.rnn_model import _CELL_REGISTRY

LOG_STD_INIT = -0.5
# Clamp the learnable log-std before every use: prevents a drifting log_std from
# driving exp(2*log_std) to 0 (gradient death) or underflowing to a NaN that
# would permanently corrupt the optimizer state across an overnight run.
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0
_LOG_2PI = float(np.log(2.0 * np.pi))
_LOG_2PIE = float(np.log(2.0 * np.pi * np.e))

# Cell types that are valid recurrent torsos here (single-cell, dense-wired).
RECURRENT_CELLS = ("lstm", "gru", "ctrnn", "ltc", "lrc", "cfc")
MODEL_TYPES = ("mlp",) + RECURRENT_CELLS


def _ortho(scale):
    return tf.keras.initializers.Orthogonal(gain=scale)


def _mlp_torso(seed):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="tanh", kernel_initializer=_ortho(np.sqrt(2))),
        tf.keras.layers.Dense(64, activation="tanh", kernel_initializer=_ortho(np.sqrt(2))),
    ])


class GaussianPolicy(tf.keras.Model):
    """Actor-critic policy with a uniform stepping/evaluation interface.

    Args:
        model_type: one of MODEL_TYPES.
        obs_dim:    observation dimension.
        act_dim:    action dimension.
        units:      hidden size of each recurrent torso (ignored for mlp).
        cell_kwargs: extra kwargs forwarded to the cell constructor.
        seed:       initialization seed.
    """

    def __init__(self, model_type, obs_dim, act_dim, units=64, cell_kwargs=None, seed=0):
        super().__init__()
        if model_type not in MODEL_TYPES:
            raise ValueError(f"unknown model_type {model_type!r}; expected {MODEL_TYPES}")
        self.model_type = model_type
        self.recurrent = model_type != "mlp"
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.units = int(units)
        cell_kwargs = dict(cell_kwargs or {})

        tf.random.set_seed(seed)
        self.log_std = tf.Variable(
            tf.ones([act_dim], tf.float32) * LOG_STD_INIT, name="log_std", trainable=True)

        if self.recurrent:
            cell_cls = _CELL_REGISTRY[model_type]
            self.actor_cell = cell_cls(units=units, **cell_kwargs)
            self.critic_cell = cell_cls(units=units, **cell_kwargs)
            self.actor_head = tf.keras.layers.Dense(act_dim, kernel_initializer=_ortho(0.01))
            self.critic_head = tf.keras.layers.Dense(1, kernel_initializer=_ortho(1.0))
        else:
            self.actor_torso = _mlp_torso(seed)
            self.actor_head = tf.keras.layers.Dense(act_dim, kernel_initializer=_ortho(0.01))
            self.critic_torso = _mlp_torso(seed)
            self.critic_head = tf.keras.layers.Dense(1, kernel_initializer=_ortho(1.0))

        self._build_once(obs_dim)

    # ------------------------------------------------------------------ build
    def _build_once(self, obs_dim):
        dummy = tf.zeros([1, obs_dim], tf.float32)
        if self.recurrent:
            a, c = self.initial_state(1)
            self.actor_forward(dummy, a)
            self.value_forward(dummy, c)
        else:
            self.actor_head(self.actor_torso(dummy))
            self.value_forward(dummy, None)

    # ------------------------------------------------------- state management
    @staticmethod
    def _cell_init(cell, n):
        s = cell.get_initial_state(batch_size=int(n), dtype=tf.float32)
        return list(s) if isinstance(s, (list, tuple)) else [s]

    def initial_state(self, n):
        """Return (actor_state, critic_state). Each is a list of tensors, or None for mlp."""
        if not self.recurrent:
            return None, None
        return self._cell_init(self.actor_cell, n), self._cell_init(self.critic_cell, n)

    @staticmethod
    def reset_state(state, done_mask):
        """Zero the hidden state of envs that just finished an episode.

        Args:
            state: list of (N, ...) tensors, or None.
            done_mask: (N, 1) float tensor, 1.0 where the env is done.
        """
        if state is None:
            return None
        keep = 1.0 - done_mask
        return [h * keep for h in state]

    # ------------------------------------------------------------- forward ops
    def actor_forward(self, obs, astate):
        """obs: (N, obs_dim). Returns (mean: (N, act_dim), new_astate)."""
        if self.recurrent:
            out, a2 = self.actor_cell(obs, astate)
            return self.actor_head(out), list(a2)
        return self.actor_head(self.actor_torso(obs)), None

    def value_forward(self, obs, cstate):
        """obs: (N, obs_dim). Returns (value: (N,), new_cstate)."""
        if self.recurrent:
            out, c2 = self.critic_cell(obs, cstate)
            return tf.squeeze(self.critic_head(out), axis=-1), list(c2)
        return tf.squeeze(self.critic_head(self.critic_torso(obs)), axis=-1), None

    # ---------------------------------------------------------- distribution
    def _log_std(self):
        return tf.clip_by_value(self.log_std, LOG_STD_MIN, LOG_STD_MAX)

    def logprob(self, mean, action):
        """Diagonal-Gaussian log-prob, summed over action dims. Shapes broadcast on (..., act_dim)."""
        log_std = self._log_std()
        var = tf.exp(2.0 * log_std)
        logp = -0.5 * (tf.square(action - mean) / var + 2.0 * log_std + _LOG_2PI)
        return tf.reduce_sum(logp, axis=-1)

    def entropy(self):
        """Scalar diagonal-Gaussian entropy (state-independent)."""
        return tf.reduce_sum(self._log_std() + 0.5 * _LOG_2PIE)

    def sample_action(self, mean):
        std = tf.exp(self._log_std())
        return mean + std * tf.random.normal(tf.shape(mean))
