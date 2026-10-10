"""Committee model builder (Iteration 7, controls added 2026-10-04): one compiled
Keras model per (wiring, K) for the partial-view voting committee.

Fairness invariant: every wiring receives the SAME inputs -- per-column features
(B, T, K, F) and a shared time (B, T, 1) -- and ends in the SAME task head on the
voted representation. Only the per-column computer differs; the consensus step
is shared (src/wirings/committee.py::VotingMixin):

  wiring 'cncp'  -> MultiColumnVotingCell: K weight-shared cortical (tbt_cNCP)
                    columns voting on their L2/3 object layer. Location is off
                    (a zeros placeholder), so this isolates partial-view+voting
                    on the cortical topology, no reference frame.
  wiring 'ncp'   -> CommitteeVotingCell around one NCP cell (NCPWiring.make_cell):
                    the columns vote on the COMMAND slice of the hidden state and
                    read out the MOTOR slice. Coupling the motor slice would share
                    no representation: motor neurons have no outgoing synapses in
                    the NCP graph, so a mixed motor state only feeds the motor
                    neurons' own leak term and never reaches command or inter.
  wiring 'dense' -> CommitteeVotingCell: K weight-shared plain cells voting on
                    their hidden state (or, with dense_vote_units, on a leading
                    subspace of it, which is read out as well -- the control that
                    gives the dense column the same subspace-voting mechanism as
                    the cNCP column).

Controls on the consensus itself (same K, same views): vote_lambda=0.0 switches
the communication off (columns independent, readout averages); consensus='median'
replaces the mean by the coordinate-wise lower median. K=1 is a single monolithic
column (consensus == itself); weight sharing keeps the parameter count constant
in K.
"""
import tensorflow as tf

from src.wirings import NCPWiring
from src.wirings.tbt_cncp import MultiColumnVotingCell
from src.wirings.committee import CommitteeVotingCell
from src.tasks.person_activity.model import (
    scaled_lamina_units, ncp_layer_sizes, _resolve_cell_cls,
    _reject_multi_state)

COMMITTEE_WIRINGS = ("cncp", "ncp", "dense")


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
                          num_classes=7, lr=1e-3, vote_init=0.0, vote_lambda=None,
                          consensus="mean", dense_vote_units=None, train_drop=0.0,
                          train_noise=0.0, **cell_kwargs):
    """Build + compile a partial-view voting committee.

    Args:
        wiring:       'cncp' (cortical columns, L2/3 vote), 'ncp' (NCP cell,
                      command-slice vote, motor readout) or 'dense' (plain
                      cells, hidden-state vote).
        n_columns:    committee size K (K=1 -> single model, no voting).
        cell:         _CELL_REGISTRY key or BaseCell subclass; single-state only.
        size:         width knob. cncp: lamina widths scaled by size/16; ncp:
                      inter=size, command=motor=size//2; dense: per-column
                      units. Sized per arm to the same effective parameter
                      budget by the runner (--param-budget).
        task:         'classification' (Dense(num_classes)+SCCE, per-step acc) or
                      'regression' (Dense(feature_size)+MSE).
        vote_init:    initial voting strength (pre-sigmoid); 0.0 -> lambda 0.5.
        vote_lambda:  None (learnable) or a fixed strength in [0, 1]; 0.0 is the
                      "columns independent" control at the same K and views.
        consensus:    'mean' or 'median' (lower median for even K).
        dense_vote_units: dense only -- vote on and read out the leading n units
                      of the hidden state instead of the whole state.

    Inputs [features (B, seq_len, K, F), time (B, seq_len, 1)]; output per step
    either (B, T, num_classes) logits or (B, T, F) predictions.
    """
    if wiring not in COMMITTEE_WIRINGS:
        raise ValueError(f"wiring must be one of {COMMITTEE_WIRINGS}, "
                         f"got {wiring!r}")
    if task not in ("classification", "regression"):
        raise ValueError("task must be 'classification' or 'regression'")
    if dense_vote_units is not None and wiring != "dense":
        raise ValueError("dense_vote_units applies to the dense wiring only")
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

    vote = dict(vote_init=vote_init, vote_lambda=vote_lambda,
                consensus=consensus)
    if wiring == "cncp":
        # Cortical committee: location off -> a zeros (B,T,K,1) placeholder built
        # from the features so the tbt column's call signature is satisfied.
        loc = tf.keras.layers.Lambda(
            lambda x: tf.zeros_like(x[..., :1]), name="loc_zeros")(feat_in)
        cellobj = MultiColumnVotingCell(
            n_columns=n_columns, cell_cls=cell_cls,
            lamina_units=scaled_lamina_units(size), seed=seed,
            use_location=False, **vote, **cell_kwargs)
        h = tf.keras.layers.RNN(cellobj, return_sequences=True)(
            (feat_in, time, loc))
    elif wiring == "ncp":
        # NCP committee: one masked NCP cell shared by the K columns. ncps orders
        # the state [motor | command | inter]; the recurrent command slice is
        # coupled, the motor slice is read out.
        inter, command, motor = ncp_layer_sizes(size)
        ncp = NCPWiring(cell_cls, inter, command, motor, seed=seed,
                        **cell_kwargs)
        cellobj = CommitteeVotingCell(
            n_columns=n_columns, cell=ncp.make_cell(),
            vote_slice=slice(motor, motor + command),
            readout_slice=slice(0, motor), **vote)
        h = tf.keras.layers.RNN(cellobj, return_sequences=True)((feat_in, time))
    else:  # dense
        sub = (None if dense_vote_units is None
               else slice(0, int(dense_vote_units)))
        cellobj = CommitteeVotingCell(
            n_columns=n_columns, cell_cls=cell_cls, units=size, seed=seed,
            vote_slice=sub, readout_slice=sub, **vote, **cell_kwargs)
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
