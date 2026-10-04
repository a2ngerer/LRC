"""Committee model builder (Iteration 7): one compiled Keras model per
(wiring, K) for the partial-view voting committee.

Fairness invariant: every wiring receives the SAME inputs -- per-column features
(B, T, K, F) and a shared time (B, T, 1) -- and ends in the SAME task head on the
voted representation. Only the per-column computer + voting cell differ:

  wiring 'cncp'  -> MultiColumnVotingCell: K weight-shared cortical (tbt_cNCP)
                    columns voting on their L2/3 object layer. Location is off
                    (a zeros placeholder), so this isolates partial-view+voting
                    on the cortical topology, no reference frame.
  wiring 'dense' -> CommitteeVotingCell: K weight-shared plain cells voting on
                    their hidden state -- the SAME voting mechanism, a generic
                    per-column computer. cncp-vs-dense at matched params then
                    asks whether the cortical column is a better weak learner.

K=1 is a single monolithic column (consensus == itself, no voting), so a K sweep
from 1 upward isolates the committee effect at a fixed parameter budget (weight
sharing keeps params ~constant in K).
"""
import tensorflow as tf

from src.wirings.tbt_cncp import MultiColumnVotingCell
from src.wirings.committee import CommitteeVotingCell
from src.tasks.person_activity.model import (
    scaled_lamina_units, _resolve_cell_cls, _reject_multi_state)

COMMITTEE_WIRINGS = ("cncp", "dense")


class ChannelDropout(tf.keras.layers.Layer):
    """Train-time whole-channel dropout on (B,T,K,F).

    Zeros the SAME random subset of the F channels across all timesteps and all K
    columns of an example (a failed sensor fails for everyone, for the whole
    sequence), with drop probability `rate`; inference is a no-op and there is NO
    inverted-dropout rescaling (a missing sensor is simply absent, matching the
    test-time drop_features corruption). A K=1 monolith trained through this is
    the standard robustness baseline the committee must beat -- without it the
    committee's dropout robustness could be dismissed as ordinary ensemble
    dropout-robustness.
    """

    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.rate = float(rate)

    def call(self, x, training=None):
        if not training or self.rate <= 0.0:
            return x
        shp = tf.shape(x)
        keep = tf.cast(
            tf.random.uniform([shp[0], 1, 1, shp[-1]]) >= self.rate, x.dtype)
        return x * keep                                 # broadcast over T, K

    def get_config(self):
        return {**super().get_config(), "rate": self.rate}


class NoiseAugment(tf.keras.layers.Layer):
    """Train-time additive Gaussian noise on (B,T,K,F); inference no-op.

    The train-corruption counterpart of ChannelDropout for a DIFFERENT corruption
    family (additive noise vs channel dropout). Used in Iteration 7d to build the
    noise-augmented monolith (the home-advantage upper bound) and to test whether
    a dropout-augmented monolith -- which never saw noise -- generalises to it.
    """

    def __init__(self, sigma, **kwargs):
        super().__init__(**kwargs)
        self.sigma = float(sigma)

    def call(self, x, training=None):
        if not training or self.sigma <= 0.0:
            return x
        return x + tf.random.normal(tf.shape(x), stddev=self.sigma, dtype=x.dtype)

    def get_config(self):
        return {**super().get_config(), "sigma": self.sigma}


def build_committee_model(wiring, n_columns, cell="cfc_lrc", size=64, seed=42,
                          feature_size=7, seq_len=32, task="classification",
                          num_classes=7, lr=1e-3, vote_init=0.0, train_drop=0.0,
                          train_noise=0.0, **cell_kwargs):
    """Build + compile a partial-view voting committee.

    Args:
        wiring:       'cncp' (cortical columns, L2/3 vote) or 'dense' (plain
                      cells, hidden-state vote).
        n_columns:    committee size K (K=1 -> single model, no voting).
        cell:         _CELL_REGISTRY key or BaseCell subclass; single-state only.
        size:         width knob. cncp: lamina widths scaled by size/16; dense:
                      per-column units. (Set per arm to parameter-match.)
        task:         'classification' (Dense(num_classes)+SCCE, per-step acc) or
                      'regression' (Dense(feature_size)+MSE).
        vote_init:    initial voting strength (pre-sigmoid); 0.0 -> lambda 0.5.

    Inputs [features (B, seq_len, K, F), time (B, seq_len, 1)]; output per step
    either (B, T, num_classes) logits or (B, T, F) predictions.
    """
    if wiring not in COMMITTEE_WIRINGS:
        raise ValueError(f"wiring must be one of {COMMITTEE_WIRINGS}, "
                         f"got {wiring!r}")
    if task not in ("classification", "regression"):
        raise ValueError("task must be 'classification' or 'regression'")
    cell_cls = _resolve_cell_cls(cell)
    _reject_multi_state(cell_cls, cell_kwargs)
    tf.keras.utils.set_random_seed(seed)

    features = tf.keras.Input(shape=(seq_len, n_columns, feature_size),
                              name="features")
    time = tf.keras.Input(shape=(seq_len, 1), name="time")

    # Optional train-time corruption augmentation (the robustness baselines):
    # channel dropout and/or additive noise. Both are inference no-ops.
    feat_in = features
    if train_drop > 0.0:
        feat_in = ChannelDropout(train_drop, name="channel_dropout")(feat_in)
    if train_noise > 0.0:
        feat_in = NoiseAugment(train_noise, name="noise_augment")(feat_in)

    if wiring == "cncp":
        # Cortical committee: location off -> a zeros (B,T,K,1) placeholder built
        # from the features so the tbt column's call signature is satisfied.
        loc = tf.keras.layers.Lambda(
            lambda x: tf.zeros_like(x[..., :1]), name="loc_zeros")(feat_in)
        cellobj = MultiColumnVotingCell(
            n_columns=n_columns, cell_cls=cell_cls,
            lamina_units=scaled_lamina_units(size), seed=seed,
            use_location=False, vote_init=vote_init, **cell_kwargs)
        h = tf.keras.layers.RNN(cellobj, return_sequences=True)(
            (feat_in, time, loc))
    else:  # dense
        cellobj = CommitteeVotingCell(
            n_columns=n_columns, cell_cls=cell_cls, units=size, seed=seed,
            vote_init=vote_init, **cell_kwargs)
        h = tf.keras.layers.RNN(cellobj, return_sequences=True)((feat_in, time))

    if task == "classification":
        out = tf.keras.layers.Dense(num_classes, name="logits")(h)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="acc")]
    else:
        out = tf.keras.layers.Dense(feature_size, name="pred")(h)
        loss = tf.keras.losses.MeanSquaredError()
        metrics = [tf.keras.metrics.MeanSquaredError(name="mse")]

    model = tf.keras.Model(inputs=[features, time], outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=loss,
                  metrics=metrics)
    return model
