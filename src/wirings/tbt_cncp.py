# tbt_cNCP: a differentiable Thousand-Brains extension of the cNCP cortical
# column (concept: docs/superpowers/specs/2026-07-04-tbt-cncp-concept.md).
#
# TbtCorticalColumnCell subclasses CorticalColumnCell and adds two things on top
# of the unchanged cNCP graph:
#   1. a LOCATION signal (TBT's L6a reference frame) that MULTIPLICATIVELY
#      modulates L4 -- location gates sensation, putting L4 into a "predictive
#      state", exactly the L6a->L4 mechanism of Hawkins et al. 2017 and the same
#      multiplicative form cNCP already uses for its apical gain;
#   2. an output = concat([L2/3, L5ET]) so a task can read the object
#      representation from L2/3 (TBT's object/output layer) in addition to the
#      L5ET motor hub.
#
# The per-step input is a 3-tuple (features, elapsed_time, location). use_location
# False ignores the location channel (the ablation control) while keeping the
# same graph, so tbt_cNCP and its no-location control share one implementation.
#
# This is an RNN wiring extension, not a neuroscience claim; it is a
# differentiable reinterpretation of the Thousand-Brains Theory, not a
# re-implementation of Numenta's HTM/Monty (see the concept spec sec 1).

import tensorflow as tf

from .cncp import CorticalColumnCell, NODE_ORDER, DEFAULT_LAMINA_UNITS

_L23_IDX = NODE_ORDER.index("L23")   # object layer, voted across columns


class TbtCorticalColumnCell(CorticalColumnCell):
    """cNCP column + Thousand-Brains location signal.

    Adds a location-gated L4 (multiplicative) and emits concat([L2/3, L5ET]).

    Args (beyond CorticalColumnCell):
        loc_gain_init: initial per-unit location gain on L4 (default 0.1).
        use_location:  if False, the location input is ignored (ablation),
                       so the graph reduces to plain cNCP with a
                       concat([L2/3, L5ET]) readout.
    """

    def __init__(self, *args, loc_gain_init=0.1, use_location=True, **kwargs):
        super().__init__(*args, **kwargs)
        self._loc_gain_init = loc_gain_init
        # use_location accepts a bool (back-compat) or a mode string:
        #   False / 'none' -> ablation, location ignored (no reference frame);
        #   True  / 'gate' -> multiplicative L6a->L4 gain (TBT-faithful, but can
        #                     only rescale L4, never add a new dimension);
        #   'film'         -> affine FiLM conditioning h_l4*(1+gamma)+beta, so
        #                     L6a supplies a gain AND a location-prior shift
        #                     (Perez et al. 2018); the shift is exactly the
        #                     dimension the pure multiplicative gate is missing.
        self._loc_mode = self._resolve_loc_mode(use_location)
        self._use_location = self._loc_mode != "none"  # back-compat attribute

    @staticmethod
    def _resolve_loc_mode(use_location):
        if use_location is False or use_location == "none":
            return "none"
        if use_location is True or use_location == "gate":
            return "gate"
        if use_location == "film":
            return "film"
        raise ValueError(
            "use_location must be a bool or one of {'none','gate','film'}, "
            f"got {use_location!r}")

    @property
    def output_size(self):
        # object layer (L2/3) ++ motor hub (L5ET)
        return self._units['L23'] + self._units['L5ET']

    def build(self, input_shape):
        # input_shape is a 3-tuple of shapes (features, time, location); the
        # parent reads the feature dim from input_shape[0][-1].
        super().build(input_shape)
        loc_dim = int(input_shape[2][-1])
        self._loc_dim = loc_dim
        n_l4 = self._units["L4"]
        if self._loc_mode == "gate":
            self.W_loc_L4 = self.add_weight(
                name="W_loc_L4", shape=(loc_dim, n_l4),
                initializer="glorot_uniform", dtype=tf.float32)
            self.gain_loc_L4 = self.add_weight(
                name="gain_loc_L4", shape=(n_l4,), dtype=tf.float32,
                initializer=tf.keras.initializers.Constant(self._loc_gain_init))
        elif self._loc_mode == "film":
            # Zero-init both maps: training starts at the identity (gamma=beta=0
            # -> h_l4 unchanged) yet both still receive gradient from step one
            # (d/dW = location (x) upstream, nonzero for any nonzero location),
            # so there is no dead start.
            self.W_loc_scale = self.add_weight(
                name="W_loc_scale", shape=(loc_dim, n_l4),
                initializer="zeros", dtype=tf.float32)
            self.W_loc_shift = self.add_weight(
                name="W_loc_shift", shape=(loc_dim, n_l4),
                initializer="zeros", dtype=tf.float32)
        self.built = True

    def call(self, inputs, states):
        x, elapsed_time, location = inputs
        p_l4, p_l23, p_l5it, p_l5et, p_l6cc, p_l6ct, p_thal, p_trn = states
        e = self.edges
        ff = self._feedforward_only

        # 1. entry node L4 (identical to cNCP) ...
        basal_l4 = e["M_in_L4"](x)
        if not ff:
            basal_l4 = basal_l4 + e["M_Thal_L4"](p_thal)
        if self._divisive_inhibition:
            basal_l4 = basal_l4 / (
                1.0 + tf.nn.softplus(e["M_L6CT_div_L4"](p_l6ct)))
        h_l4, _ = self.subcells["L4"]((basal_l4, elapsed_time), [p_l4])

        # NEW (TBT): the location signal modulates L4 -> predictive state.
        #   gate: per-unit multiplicative gain (can only rescale L4);
        #   film: affine scale+shift (adds the location prior the gate lacks).
        if self._loc_mode == "gate":
            loc_drive = tf.matmul(location, self.W_loc_L4)
            h_l4 = h_l4 * (1.0 + self.gain_loc_L4 * tf.nn.sigmoid(loc_drive))
        elif self._loc_mode == "film":
            gamma = tf.matmul(location, self.W_loc_scale)
            beta = tf.matmul(location, self.W_loc_shift)
            h_l4 = h_l4 * (1.0 + gamma) + beta

        # 2. integration node L2/3, with optional top-down gain (cNCP).
        basal_l23 = e["M_L4_L23"](h_l4)
        if not ff:
            basal_l23 = basal_l23 + e["M_L6CC_L23"](p_l6cc)
        h_l23, _ = self.subcells["L23"]((basal_l23, elapsed_time), [p_l23])
        if self._use_gain:
            apical_l23 = e["M_ap_L23"](tf.concat([p_l5et, p_l6ct], axis=-1))
            h_l23 = h_l23 * (1.0 + self.gain_L23 * tf.nn.sigmoid(apical_l23))

        # 3. intracortical relay L5IT.
        basal_l5it = e["M_L23_L5IT"](h_l23)
        if not ff:
            basal_l5it = basal_l5it + e["M_L5ET_L5IT"](p_l5et)
        h_l5it, _ = self.subcells["L5IT"]((basal_l5it, elapsed_time), [p_l5it])

        # 4. output hub L5ET, with optional top-down gain.
        basal_l5et = e["M_L23_L5ET"](h_l23) + e["M_L5IT_L5ET"](h_l5it)
        h_l5et, _ = self.subcells["L5ET"]((basal_l5et, elapsed_time), [p_l5et])
        if self._use_gain:
            apical_l5et = e["M_ap_L5ET"](tf.concat([p_l6ct, p_l6cc], axis=-1))
            h_l5et = h_l5et * (
                1.0 + self.gain_L5ET * tf.nn.sigmoid(apical_l5et))

        if ff:
            h_l6cc, h_l6ct, h_thal, h_trn = p_l6cc, p_l6ct, p_thal, p_trn
        else:
            # 5. deep readout nodes.
            deep_in = tf.concat([h_l5it, h_l5et], axis=-1)
            h_l6cc, _ = self.subcells["L6CC"](
                (e["M_L5_L6CC"](deep_in), elapsed_time), [p_l6cc])
            h_l6ct, _ = self.subcells["L6CT"](
                (e["M_L5_L6CT"](deep_in), elapsed_time), [p_l6ct])
            # 6. re-entrant relay loop (L6CT -> TRN -> Thal).
            h_trn = self._relay(e["M_L6CT_TRN"](h_l6ct), p_trn,
                                self.trn_alpha_raw, self.trn_bias)
            thal_in = e["M_L6CT_Thal"](h_l6ct) + e["M_TRN_Thal"](h_trn)
            h_thal = self._relay(thal_in, p_thal,
                                 self.thal_alpha_raw, self.thal_bias)

        new_states = [h_l4, h_l23, h_l5it, h_l5et, h_l6cc, h_l6ct,
                      h_thal, h_trn]
        # NEW: expose the object layer (L2/3) alongside the motor hub (L5ET).
        output = tf.concat([h_l23, h_l5et], axis=-1)
        return output, new_states


class MultiColumnVotingCell(tf.keras.layers.AbstractRNNCell):
    """K weight-shared tbt_cNCP columns that VOTE each timestep (TBT ing. 3).

    Every column runs the identical algorithm (shared weights, as in TBT) on its
    OWN glimpse stream. After each step the columns reach consensus through their
    L2/3 (object) states: a learnable convex mix pulls each column's L2/3 toward
    the mean L2/3 across columns -- the differentiable stand-in for the long-range
    lateral L2/3 excitation that implements voting in Numenta 2017. The cell
    output is the voted (mean) L2/3, read out by a Dense object head.

    Per-step input is a 3-tuple (features (B,K,F), time (B,1), location (B,K,L)),
    i.e. one feature/location vector per column. State = K copies of the 8-node
    composite state, flat in column-major order.

    Args:
        n_columns:   number of columns K.
        use_location: forwarded to the shared TbtCorticalColumnCell gate (set
                      False when location is concatenated into features).
        vote_init:   initial voting strength (pre-sigmoid); 0.0 -> lambda 0.5.
        (rest)       forwarded to the shared TbtCorticalColumnCell.
    """

    def __init__(self, n_columns=3, cell_cls="cfc_lrc", lamina_units=None,
                 seed=42, use_location=False, vote_init=0.0, **cell_kwargs):
        super().__init__()
        self.K = int(n_columns)
        units = dict(DEFAULT_LAMINA_UNITS)
        if lamina_units:
            units.update(lamina_units)
        self._l23 = units["L23"]
        self._n_node = len(NODE_ORDER)
        self._vote_init = vote_init
        self.col = TbtCorticalColumnCell(
            cell_cls=cell_cls, lamina_units=lamina_units, seed=seed,
            use_location=use_location, **cell_kwargs)

    @property
    def state_size(self):
        return list(self.col.state_size) * self.K

    @property
    def output_size(self):
        return self._l23

    def build(self, input_shape):
        feat_dim = int(input_shape[0][-1])
        loc_dim = int(input_shape[2][-1])
        self.col.build((tf.TensorShape([None, feat_dim]),
                        tf.TensorShape([None, 1]),
                        tf.TensorShape([None, loc_dim])))
        self.vote_raw = self.add_weight(
            name="vote_raw", shape=(), dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(self._vote_init))
        self.built = True

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        dtype = dtype or tf.float32
        return [tf.zeros([batch_size, self.col._units[node]], dtype=dtype)
                for _ in range(self.K) for node in NODE_ORDER]

    def call(self, inputs, states):
        features, elapsed_time, locs = inputs      # (B,K,F), (B,1), (B,K,L)
        n = self._n_node
        l23_list, per_col = [], []
        for k in range(self.K):
            st_k = list(states[k * n:(k + 1) * n])
            _, ns_k = self.col(
                (features[:, k, :], elapsed_time, locs[:, k, :]), st_k)
            l23_list.append(ns_k[_L23_IDX])
            per_col.append(ns_k)

        consensus = tf.add_n(l23_list) / float(self.K)
        lam = tf.nn.sigmoid(self.vote_raw)             # voting strength in (0,1)
        new_states = []
        voted_l23 = []
        for k in range(self.K):
            ns = list(per_col[k])
            ns[_L23_IDX] = (1.0 - lam) * ns[_L23_IDX] + lam * consensus
            voted_l23.append(ns[_L23_IDX])
            new_states.extend(ns)
        output = tf.add_n(voted_l23) / float(self.K)   # voted object rep
        return output, new_states
