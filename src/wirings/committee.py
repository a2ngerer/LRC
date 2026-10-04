# Generic partial-view voting committee (Iteration 7, controls added 2026-10-04).
#
# The user steer (2026-07-05): move past the touch/object-recognition framing and
# test the transferable Thousand-Brains principle as a GENERAL architecture -- K
# weaker, weight-shared columns, each seeing only a PART of the input, computing
# locally, then a VOTE assembles a model of the whole.
#
# CommitteeVotingCell is the task-agnostic, single-state analogue of
# MultiColumnVotingCell (src/wirings/tbt_cncp.py): it wraps ONE shared recurrent
# cell (a plain cfc_lrc/gru cell, or a prebuilt NCP cell) and runs it over K
# partial views, reaching consensus each step by pulling a chosen slice of every
# column's hidden state toward the column consensus (learnable or fixed convex
# mix), and emitting the consensus of a chosen readout slice. MultiColumnVotingCell
# does the identical thing with a cortical column voting on its L2/3 object layer,
# so a cNCP-column committee, an NCP committee and a plain-cell committee share
# ONE voting mechanism (VotingMixin) and differ only in the per-column computer.
#
# Weight sharing is the lever: K shared columns carry ~the same parameters as ONE
# column (only a scalar vote gate is added), so a K-column committee is
# parameter-matched to a single monolithic column of the same width.
#
# Consensus algebra (the reason the controls exist): with the mean consensus the
# readout of the current step does not depend on the voting strength lambda,
#     mean_k((1 - lambda) h_k + lambda hbar) == hbar,
# so lambda acts only through the stored states of later steps. The protocol
# therefore compares the learned lambda against lambda = 0 at the SAME K and the
# SAME views (columns independent, readout averages) and against a coordinate-wise
# median consensus (resistance to a minority of corrupted columns).

import tensorflow as tf

from .cncp import _resolve_cell_cls

CONSENSUS_MODES = ("mean", "median")


def consensus_target(cols, how="mean"):
    """Consensus of a list of K tensors (B, U): the column mean, or the
    coordinate-wise LOWER median (index (K - 1) // 2 of the sorted columns, so
    an even K takes the lower of the two middle values)."""
    if how == "mean":
        return tf.add_n(cols) / float(len(cols))
    if how == "median":
        stacked = tf.sort(tf.stack(cols, axis=0), axis=0)      # (K, B, U)
        return stacked[(len(cols) - 1) // 2]
    raise ValueError(f"consensus must be one of {CONSENSUS_MODES}, got {how!r}")


def consensus_mix(cols, lam, how="mean"):
    """Pull every column toward the consensus; return (mixed columns, readout).

    mixed_k = (1 - lam) * h_k + lam * target, readout = consensus of the mixed
    columns. For how='mean' the readout equals the pre-mix mean for every lam
    (see the module header); for how='median' target and readout are both the
    lower median.
    """
    target = consensus_target(cols, how)
    mixed = [(1.0 - lam) * h + lam * target for h in cols]
    return mixed, consensus_target(mixed, how)


class VotingMixin:
    """The consensus step shared by CommitteeVotingCell and MultiColumnVotingCell.

    vote_lambda None -> learnable strength sigmoid(vote_raw), initialised from
    vote_init (pre-sigmoid; 0.0 -> 0.5). A float in [0, 1] -> fixed strength,
    no variable is created (lambda = 0 is the "columns independent" control).
    """

    def _init_vote(self, vote_lambda, vote_init, consensus):
        if consensus not in CONSENSUS_MODES:
            raise ValueError(
                f"consensus must be one of {CONSENSUS_MODES}, got {consensus!r}")
        if vote_lambda is not None and not 0.0 <= float(vote_lambda) <= 1.0:
            raise ValueError(
                f"vote_lambda must be None or in [0, 1], got {vote_lambda!r}")
        self._vote_lambda = None if vote_lambda is None else float(vote_lambda)
        self._vote_init = float(vote_init)
        self._consensus_mode = consensus

    def _build_vote(self):
        if self._vote_lambda is None:
            self.vote_raw = self.add_weight(
                name="vote_raw", shape=(), dtype=tf.float32,
                initializer=tf.keras.initializers.Constant(self._vote_init))

    @property
    def vote_strength(self):
        """Current voting strength lambda in (0, 1) (learned) or [0, 1] (fixed)."""
        if self._vote_lambda is None:
            return tf.nn.sigmoid(self.vote_raw)
        return tf.constant(self._vote_lambda, dtype=tf.float32)

    def _consensus(self, cols):
        """(mixed columns, readout) for a list of K per-column tensors."""
        return consensus_mix(cols, self.vote_strength, self._consensus_mode)


def _normalise_slice(sl, width, name):
    """A slice over the state axis -> (start, stop) with step 1, or the whole
    state when sl is None."""
    if sl is None:
        return 0, width
    start, stop, step = sl.indices(width)
    if step != 1 or stop <= start:
        raise ValueError(f"{name} must be a non-empty forward slice, got {sl}")
    return start, stop


class CommitteeVotingCell(VotingMixin, tf.keras.layers.AbstractRNNCell):
    """K weight-shared single-state cells that VOTE each timestep.

    Every column runs the identical shared cell on its OWN partial view. After
    each step the columns reach consensus: a convex mix (learnable or fixed
    strength) pulls the vote slice of every column's hidden state toward the
    consensus (mean or lower median) across columns, and the cell output is the
    consensus of the readout slice of the mixed states, read out by a task head.

    Per-step input is a 2-tuple (features (B,K,F), time (B,1)) -- one feature
    vector per column, a shared elapsed time. State = K hidden states (one per
    column), flat in column order.

    Args:
        n_columns:     number of columns K.
        cell_cls:      _CELL_REGISTRY key (e.g. 'cfc_lrc', 'gru') or BaseCell
                       subclass; single-state cells only (state_size == units).
                       Ignored when ``cell`` is given.
        units:         width of the shared per-column cell (ignored with ``cell``).
        vote_init:     initial voting strength (pre-sigmoid); 0.0 -> lambda 0.5.
        vote_lambda:   None (learnable) or a fixed strength in [0, 1].
        consensus:     'mean' (default) or 'median' (lower median for even K,
                       applied to the consensus target and the readout).
        vote_slice:    slice of the state axis that is pulled toward the
                       consensus (default: the whole state). For an NCP cell
                       this is the command slice: motor neurons have no outgoing
                       synapses, so a mixed motor state would only feed the
                       motor neurons' own leak term and never reach the rest of
                       the circuit.
        readout_slice: slice of the (mixed) state that forms the output
                       (default: the whole state); e.g. the NCP motor slice.
        cell:          a prebuilt single-state cell to share across the columns,
                       e.g. ``NCPWiring(...).make_cell()``; its state_size sets
                       ``units``.
        (rest)         forwarded to the shared cell constructor (cell_cls path).
    """

    def __init__(self, n_columns=3, cell_cls="cfc_lrc", units=64, seed=42,
                 vote_init=0.0, vote_lambda=None, consensus="mean",
                 vote_slice=None, readout_slice=None, cell=None, **cell_kwargs):
        super().__init__()
        self.K = int(n_columns)
        if cell is not None:
            if isinstance(cell.state_size, (list, tuple)):
                raise ValueError(
                    "CommitteeVotingCell supports single-state cells only "
                    f"(state_size == units); the given {type(cell).__name__} "
                    f"has state_size {cell.state_size}.")
            self.units = int(cell.state_size)
            self.col = cell
        else:
            cls = _resolve_cell_cls(cell_cls)
            probe = cls(units=4, **cell_kwargs)
            if isinstance(probe.state_size, (list, tuple)):
                raise ValueError(
                    "CommitteeVotingCell supports single-state cells only "
                    f"(state_size == units); {cls.__name__} has state_size "
                    f"{probe.state_size}.")
            self.units = int(units)
            self.col = cls(units=self.units, **cell_kwargs)
        self._vote = _normalise_slice(vote_slice, self.units, "vote_slice")
        self._read = _normalise_slice(readout_slice, self.units, "readout_slice")
        self._init_vote(vote_lambda, vote_init, consensus)

    @property
    def state_size(self):
        return [self.units] * self.K

    @property
    def output_size(self):
        return self._read[1] - self._read[0]

    def build(self, input_shape):
        feat_dim = int(input_shape[0][-1])
        # Match the (features, time) tuple convention every BaseCell.build reads.
        self.col.build((tf.TensorShape([None, feat_dim]),
                        tf.TensorShape([None, 1])))
        self._build_vote()
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
        v0, v1 = self._vote
        mixed, _ = self._consensus([h[:, v0:v1] for h in outs])
        new_states = [tf.concat([h[:, :v0], m, h[:, v1:]], axis=-1)
                      for h, m in zip(outs, mixed)]
        r0, r1 = self._read
        output = consensus_target([h[:, r0:r1] for h in new_states],
                                  self._consensus_mode)
        return output, new_states
