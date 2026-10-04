"""Active glimpse control (Iteration 5): the cortical column STEERS its sensor.

The passive active-sensing task (model.py) feeds pre-generated RANDOM glimpses.
The Thousand-Brains core is sensorimotor: a column should choose WHERE to look
next. The cNCP already has an explicit motor hub (L5ET). Here the column predicts
the next glimpse centre from L5ET, and the next glimpse is sampled from the full
object image at that continuous position by a DIFFERENTIABLE bilinear sampler --
so the classification loss trains the look policy end to end (no RL).

Compared control: dense/gru read the motor head from their generic hidden state
(no dedicated motor hub). policy='random' ignores the motor (glimpse centres are
random) -- the passive baseline. The headline test is active vs random, and tbt
(L5 motor) vs dense (hidden-state motor).

Everything is ordinary tensor algebra trained by backprop; the "motor hub" is a
node label in the cNCP graph, nothing simulates biology.
"""
import numpy as np
import tensorflow as tf

from src.wirings.tbt_cncp import TbtCorticalColumnCell
from src.wirings.cncp import NODE_ORDER
from src.tasks.active_sensing.datasets import (compositional_templates,
                                               _render_compositional)
from src.tasks.person_activity.model import (scaled_lamina_units,
                                             _resolve_cell_cls,
                                             _reject_multi_state)

_L5ET = NODE_ORDER.index("L5ET")
# Match datasets.encode_location: for each f, [sin(pi f x), cos, sin(pi f y), cos].
_FREQS = (1.0, 2.0, 4.0)
RF_DIM = 12
ACTIVE_GLIMPSE_WIRINGS = ("tbt_cncp", "dense", "gru")


def rf_code_tf(pos):
    """Differentiable reference-frame code, (B,2) in [0,1] -> (B,12).

    Bit-for-bit the tf analogue of datasets.encode_location (pi*f grid code),
    so the online location signal matches the offline one.
    """
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    feats = []
    for f in _FREQS:
        feats += [tf.sin(np.pi * f * x), tf.cos(np.pi * f * x),
                  tf.sin(np.pi * f * y), tf.cos(np.pi * f * y)]
    return tf.concat(feats, axis=-1)


def differentiable_glimpse(image, center, k):
    """Sample a k x k patch from image at a CONTINUOUS centre (bilinear).

    image:  (B, G, G) float; center: (B, 2) in [0,1] = (row, col) fraction.
    Returns (B, k*k). Differentiable w.r.t. center (the L5 motor can be trained
    to steer the next glimpse by backprop).
    """
    Gf = tf.cast(tf.shape(image)[1], tf.float32)
    cr = center[:, 0] * (Gf - 1.0)
    cc = center[:, 1] * (Gf - 1.0)
    off = tf.range(k, dtype=tf.float32) - (k - 1) / 2.0
    rows = tf.clip_by_value(cr[:, None] + off[None, :], 0.0, Gf - 1.0)
    cols = tf.clip_by_value(cc[:, None] + off[None, :], 0.0, Gf - 1.0)
    r0 = tf.floor(rows); r1 = tf.minimum(r0 + 1.0, Gf - 1.0)
    c0 = tf.floor(cols); c1 = tf.minimum(c0 + 1.0, Gf - 1.0)
    wr = rows - r0; wc = cols - c0
    r0g = tf.tile(r0[:, :, None], [1, 1, k]); r1g = tf.tile(r1[:, :, None], [1, 1, k])
    c0g = tf.tile(c0[:, None, :], [1, k, 1]); c1g = tf.tile(c1[:, None, :], [1, k, 1])
    wrg = tf.tile(wr[:, :, None], [1, 1, k]); wcg = tf.tile(wc[:, None, :], [1, k, 1])

    def gather(rr, cc_):
        idx = tf.cast(tf.stack([rr, cc_], axis=-1), tf.int32)
        return tf.gather_nd(image, idx, batch_dims=1)

    Ia = gather(r0g, c0g); Ib = gather(r0g, c1g)
    Ic = gather(r1g, c0g); Id = gather(r1g, c1g)
    top = Ia * (1.0 - wcg) + Ib * wcg
    bot = Ic * (1.0 - wcg) + Id * wcg
    patch = top * (1.0 - wrg) + bot * wrg
    return tf.reshape(patch, [tf.shape(image)[0], k * k])


def load_active_glimpse(n_per_class=200, n_classes=6, grid=24, seq_len=10,
                        n_dots=5, k=5, test_fraction=0.2, seed=42):
    """Full-image compositional dataset for the active-glimpse loop.

    Unlike load_active_sensing (which pre-extracts random glimpses), this returns
    the FULL object images so the model can sample its own glimpses online.

    Returns dict with train/test images (N,G,G), labels (N,), random glimpse
    centres (N,seq_len,2) in [0,1] (the random-policy path + exploration warmup),
    and metadata (n_classes, grid, k, seq_len).
    """
    templates = compositional_templates(n_classes, grid, n_dots=n_dots, seed=0)
    rng = np.random.default_rng(seed)
    N = n_per_class * n_classes
    y = np.repeat(np.arange(n_classes), n_per_class).astype(np.int32)
    imgs = np.stack([_render_compositional(templates[c], grid, rng)
                     for c in y]).astype(np.float32)
    rand_pos = rng.random((N, seq_len, 2)).astype(np.float32)
    perm = np.random.default_rng(seed + 1).permutation(N)
    n_test = int(round(test_fraction * N))
    te, tr = perm[:n_test], perm[n_test:]
    return {
        "train_img": imgs[tr], "train_y": y[tr], "train_pos": rand_pos[tr],
        "test_img": imgs[te], "test_y": y[te], "test_pos": rand_pos[te],
        "n_classes": n_classes, "grid": grid, "k": k, "seq_len": seq_len,
    }


class ActiveGlimpseModel(tf.keras.Model):
    """Sensorimotor object recognition: one wiring x {active,random} policy.

    Inputs [image (B,G,G), rand_pos (B,T,2)]. For T steps: sample a k x k glimpse
    at the current centre, feed (glimpse, time, location) to the wiring, read the
    class logits, and -- if active and past the warmup -- set the next centre from
    the motor head. Output per-step logits (B,T,n_classes).
    """

    def __init__(self, wiring="tbt_cncp", policy="active", cell="cfc_lrc",
                 size=64, grid=24, k=5, seq_len=10, n_classes=6, warm=2,
                 explore=0.3, seed=42, **cell_kwargs):
        super().__init__()
        if wiring not in ACTIVE_GLIMPSE_WIRINGS:
            raise ValueError(f"wiring must be one of {ACTIVE_GLIMPSE_WIRINGS}")
        if policy not in ("active", "random"):
            raise ValueError("policy must be 'active' or 'random'")
        self.wiring = wiring
        self.policy = policy
        self.grid = grid; self.k = k; self.T = seq_len; self.warm = warm
        self.explore = explore
        cell_cls = _resolve_cell_cls(cell)
        _reject_multi_state(cell_cls, cell_kwargs)
        tf.keras.utils.set_random_seed(seed)
        self._is_tbt = wiring == "tbt_cncp"
        if self._is_tbt:
            self.col = TbtCorticalColumnCell(
                cell_cls=cell_cls, lamina_units=scaled_lamina_units(size),
                seed=seed, use_location="film", **cell_kwargs)
            self.col.build((tf.TensorShape([None, k * k]),
                            tf.TensorShape([None, 1]),
                            tf.TensorShape([None, RF_DIM])))
        else:
            # dense/gru: a plain recurrent cell fed concat(glimpse, location);
            # the motor head reads its generic hidden state (no motor hub).
            self.cell = cell_cls(units=size, **cell_kwargs)
            self.cell.build((None, k * k + RF_DIM))
        self.clf = tf.keras.layers.Dense(n_classes, name="logits")
        self.motor = tf.keras.layers.Dense(2, name="motor")

    def call(self, inputs, training=None):
        image, rand_pos = inputs
        B = tf.shape(image)[0]
        t1 = tf.ones([B, 1])
        pos = rand_pos[:, 0, :]                     # first glimpse random
        if self._is_tbt:
            state = self.col.get_initial_state(batch_size=B, dtype=tf.float32)
        else:
            state = [tf.zeros([B, self.cell.units], tf.float32)]
        logits = []
        for t in range(self.T):
            patch = differentiable_glimpse(image, pos, self.k)
            loc = rf_code_tf(pos)
            if self._is_tbt:
                out, state = self.col((patch, t1, loc), state)
                motor_in = state[_L5ET]
            else:
                out, state = self.cell((tf.concat([patch, loc], -1), t1), state)
                motor_in = out
            logits.append(self.clf(out))
            nxt = min(t + 1, self.T - 1)
            if self.policy == "active" and t >= self.warm:
                steered = tf.sigmoid(self.motor(motor_in))   # steered centre
                if training and self.explore > 0.0:
                    # epsilon-greedy exploration: mix in a random glimpse so the
                    # motor cannot collapse into a bad look policy that never
                    # sees the discriminative parts. Train-only; eval uses the
                    # pure learned policy (fixes the tbt-active seed collapse).
                    take_rand = tf.cast(
                        tf.random.uniform([B, 1]) < self.explore, tf.float32)
                    pos = (take_rand * rand_pos[:, nxt, :]
                           + (1.0 - take_rand) * steered)
                else:
                    pos = steered
            else:
                pos = rand_pos[:, nxt, :]                 # random / explore
        return tf.stack(logits, axis=1)


def build_active_glimpse_model(wiring, policy, cell="cfc_lrc", size=64, grid=24,
                               k=5, seq_len=10, n_classes=6, lr=1e-3, explore=0.3,
                               seed=42, **cell_kwargs):
    """Build + compile an ActiveGlimpseModel (SparseCategoricalCrossentropy)."""
    m = ActiveGlimpseModel(wiring=wiring, policy=policy, cell=cell, size=size,
                           grid=grid, k=k, seq_len=seq_len, n_classes=n_classes,
                           explore=explore, seed=seed, **cell_kwargs)
    m.compile(optimizer=tf.keras.optimizers.Adam(lr),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")])
    return m
