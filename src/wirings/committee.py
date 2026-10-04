# Generic partial-view voting committee (Iteration 7).
#
# The user steer (2026-07-05): move past the touch/object-recognition framing and
# test the transferable Thousand-Brains principle as a GENERAL architecture -- K
# weaker, weight-shared columns, each seeing only a PART of the input, computing
# locally, then a VOTE assembles a model of the whole.
#
# CommitteeVotingCell is the task-agnostic, single-state analogue of
# MultiColumnVotingCell (src/wirings/tbt_cncp.py): it wraps ONE shared plain
# recurrent cell (cfc_lrc, gru, ...) and runs it over K partial views, reaching
# consensus each step by pulling every column's hidden state toward the mean
# (learnable convex mix), and emitting the voted mean. MultiColumnVotingCell does
# the identical thing with a cortical column voting on its L2/3 object layer, so a
# cNCP-column committee and a plain-cell committee share ONE voting mechanism and
# differ only in the per-column computer -- the fair cNCP-vs-dense contrast.
#
# Weight sharing is the lever: K shared columns carry ~the same parameters as ONE
# column (only a scalar vote gate is added), so a K-column committee is
# parameter-matched to a single monolithic column of the same width. The K sweep
# from K=1 (a single model, no consensus) upward then isolates partial-view+voting
# at a fixed parameter budget.

import tensorflow as tf

from .cncp import _resolve_cell_cls


class CommitteeVotingCell(tf.keras.layers.AbstractRNNCell):
    """K weight-shared single-state cells that VOTE each timestep.

    Every column runs the identical shared cell on its OWN partial view. After
    each step the columns reach consensus: a learnable convex mix pulls every
    column's hidden state toward the mean hidden state across columns (the
    differentiable stand-in for lateral voting), and the cell output is the voted
    mean, read out by a task head.

    Per-step input is a 2-tuple (features (B,K,F), time (B,1)) -- one feature
    vector per column, a shared elapsed time. State = K hidden states (one per
    column), flat in column order.

    Args:
        n_columns:   number of columns K.
        cell_cls:    _CELL_REGISTRY key (e.g. 'cfc_lrc', 'gru') or BaseCell
                     subclass; single-state cells only (state_size == units).
        units:       width of the shared per-column cell.
        vote_init:   initial voting strength (pre-sigmoid); 0.0 -> lambda 0.5.
        (rest)       forwarded to the shared cell constructor.
    """

    def __init__(self, n_columns=3, cell_cls="cfc_lrc", units=64, seed=42,
                 vote_init=0.0, **cell_kwargs):
        super().__init__()
        self.K = int(n_columns)
        self.units = int(units)
        self._vote_init = vote_init
        cls = _resolve_cell_cls(cell_cls)
        probe = cls(units=4, **cell_kwargs)
        if isinstance(probe.state_size, (list, tuple)):
            raise ValueError(
                "CommitteeVotingCell supports single-state cells only "
                f"(state_size == units); {cls.__name__} has state_size "
                f"{probe.state_size}.")
        self.col = cls(units=self.units, **cell_kwargs)

    @property
    def state_size(self):
        return [self.units] * self.K

    @property
    def output_size(self):
        return self.units

    def build(self, input_shape):
        feat_dim = int(input_shape[0][-1])
        # Match the (features, time) tuple convention every BaseCell.build reads.
        self.col.build((tf.TensorShape([None, feat_dim]),
                        tf.TensorShape([None, 1])))
        self.vote_raw = self.add_weight(
            name="vote_raw", shape=(), dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(self._vote_init))
        self.built = True

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        dtype = dtype or tf.float32
        return [tf.zeros([batch_size, self.units], dtype=dtype)
                for _ in range(self.K)]

    def call(self, inputs, states):
        features, elapsed_time = inputs             # (B,K,F), (B,1)
        outs = []
        for k in range(self.K):
            _, ns_k = self.col((features[:, k, :], elapsed_time), [states[k]])
            outs.append(ns_k[0])
        consensus = tf.add_n(outs) / float(self.K)
        lam = tf.nn.sigmoid(self.vote_raw)          # voting strength in (0,1)
        new_states = [(1.0 - lam) * outs[k] + lam * consensus
                      for k in range(self.K)]
        output = tf.add_n(new_states) / float(self.K)   # voted representation
        return output, new_states
