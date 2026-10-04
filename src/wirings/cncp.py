# cNCP: cortically-informed Neural Circuit Policy wiring.
# Design contract: docs/superpowers/specs/2026-07-02-cncp-design.md
#
# The cNCP is a sparse connectivity pattern for a recurrent neural network, in
# the exact sense that NCPWiring (src/wirings/ncp.py) is: a fixed directed
# graph whose nodes are small standard RNN sub-cells and whose edges are
# learnable linear maps under fixed binary masks (SparseLinear,
# y = x @ (W * mask)). The node names (L4, L23, L5IT, L5ET, L6CC, L6CT, Thal,
# TRN) are labels for nodes in the computation graph. Everything here is
# ordinary tensor algebra trained by backprop; nothing simulates a biological
# process.
#
# Unlike the acyclic NCP Sequential stack, the cNCP graph is cyclic (top-down
# feedback, a lateral edge and a re-entrant relay loop), so it lives in ONE
# composite RNN cell (CorticalColumnCell) whose hidden state is the list of
# all eight node states. Cycles are broken by reading feedback edges from the
# PREVIOUS timestep's stored state (delayed-state approximation -- the same
# trick MixedMemoryCell uses to compose sub-cells); feedforward edges within a
# timestep are evaluated in a fixed topological order and see the current
# step's freshly computed values.

import numpy as np
import tensorflow as tf

from src.neurons.base_cell import BaseCell
from .base_wiring import BaseWiring
from .ncp import SparseLinear

# Positional order of the composite state list (= spec section 2). The state
# is indexed positionally everywhere; this tuple is the single source for it.
NODE_ORDER = ('L4', 'L23', 'L5IT', 'L5ET', 'L6CC', 'L6CT', 'Thal', 'TRN')

# Default per-node unit counts (design choices, not measured constants; spec
# section 2). Chosen so the total parameter count is comparable to the
# standard inter=16/command=8/motor=2 NCP.
DEFAULT_LAMINA_UNITS = {
    'L4': 8, 'L23': 8, 'L5IT': 6, 'L5ET': 8,
    'L6CC': 4, 'L6CT': 4, 'Thal': 4, 'TRN': 2,
}

# Default edge densities (spec section 3). density = 1.0 means an all-ones
# mask (still a learnable SparseLinear, kept in the same code path); sparse
# masks are Bernoulli(density) samples drawn once from the wiring seed.
DEFAULT_MASK_DENSITIES = {
    # 3a. feedforward edges -- read the CURRENT step
    'M_in_L4': 1.0,
    'M_L4_L23': 1.0,
    'M_L23_L5ET': 1.0,
    'M_L23_L5IT': 0.5,
    'M_L5IT_L5ET': 1.0,
    'M_L5_L6CC': 1.0,
    'M_L5_L6CT': 1.0,
    # 3b. feedback / lateral edges -- read the PREVIOUS step
    'M_L5ET_L5IT': 0.25,
    'M_L6CC_L23': 0.5,
    # 3c. apical top-down (feeds the multiplicative combiner) -- PREVIOUS step
    'M_ap_L23': 0.5,
    'M_ap_L5ET': 0.5,
    # 3d. re-entrant relay loop (Thal->L4 is delayed; L6CT->TRN->Thal is
    # acyclic within the step)
    'M_L6CT_Thal': 1.0,
    'M_L6CT_TRN': 1.0,
    'M_TRN_Thal': 1.0,
    'M_Thal_L4': 1.0,
    # 3e. optional divisive term (only instantiated when
    # divisive_inhibition=True)
    'M_L6CT_div_L4': 1.0,
}

# Edges that exist only in the full (recurrent) wiring. The feedforward
# reduction (spec 7.1, the cncp_ff control) removes every feedback, apical and
# relay-loop edge, leaving the laminar chain input->L4->L23->{L5IT,L5ET}.
_RECURRENT_ONLY_EDGES = (
    'M_L5ET_L5IT', 'M_L6CC_L23', 'M_ap_L23', 'M_ap_L5ET',
    'M_L5_L6CC', 'M_L5_L6CT',
    'M_L6CT_Thal', 'M_L6CT_TRN', 'M_TRN_Thal', 'M_Thal_L4',
)


def _resolve_cell_cls(cell_cls):
    """Resolve a sub-cell class from a registry key or a BaseCell subclass.

    The import is deferred to call time to avoid the circular import
    src.wirings -> src.models.rnn_model -> src.wirings.
    """
    if isinstance(cell_cls, str):
        from src.models.rnn_model import _CELL_REGISTRY
        if cell_cls not in _CELL_REGISTRY:
            raise KeyError(
                f"Unknown cell key {cell_cls!r}. Known keys: "
                f"{sorted(_CELL_REGISTRY)}")
        return _CELL_REGISTRY[cell_cls]
    if isinstance(cell_cls, type) and issubclass(cell_cls, BaseCell):
        return cell_cls
    raise TypeError(
        "cell_cls must be a _CELL_REGISTRY key or a BaseCell subclass, "
        f"got {cell_cls!r}")


def _make_mask(rng, in_units, out_units, density):
    """Fixed binary mask of shape (in_units, out_units) at the given density.

    density >= 1.0 returns an all-ones mask without consuming the RNG, so the
    random stream stays aligned across ablations that only differ in which
    edges are instantiated. A sparse draw that comes out all-zero (possible
    for tiny masks at low density) is repaired by forcing one connection, so
    no edge is ever silently disconnected.
    """
    if density >= 1.0:
        return np.ones((in_units, out_units), dtype=np.float32)
    mask = (rng.random((in_units, out_units)) < density).astype(np.float32)
    if not mask.any():
        mask[rng.integers(in_units), rng.integers(out_units)] = 1.0
    return mask


class SignedSparseLinear(tf.keras.layers.Layer):
    """SparseLinear whose effective weight is sign-locked negative.

    The effective weight is -softplus(W) * mask (spec section 6): every
    existing connection is strictly negative for any value of the underlying
    parameter W, so the edge can never turn excitatory during training.
    softplus is used instead of -abs(W) because its derivative sigmoid(W) is
    strictly positive everywhere -- the edge stays differentiable with no
    dead point at W = 0.

    Args:
        units:  output dimension
        mask:   numpy bool/int array of shape (input_dim, units), 1 = connected
    """

    def __init__(self, units, mask, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self._mask_np = np.array(mask, dtype=np.float32)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='W', shape=(input_shape[-1], self.units),
            dtype=tf.float32, initializer='glorot_uniform',
        )
        self.mask = tf.constant(self._mask_np, dtype=tf.float32)
        self.built = True

    def effective_weight(self):
        """The sign-locked weight actually used in the forward pass (<= 0)."""
        return -tf.nn.softplus(self.W) * self.mask

    def call(self, x):
        return x @ self.effective_weight()


class CorticalColumnCell(BaseCell):
    """Composite RNN cell implementing the cNCP graph (spec section 4).

    Design note (2026-09-17 NCP fix): the eight graph nodes here are COMPOSITE
    nodes -- each is a whole sub-cell with its own state, not a single neuron --
    and the sparse masks sit on the edges between them. That is intentional
    (cNCP models cortical laminae, not individual NCP neurons) and is therefore
    unaffected by the NCPWiring rewrite; see docs/ncp-wiring-fix-2026-09-17.md.

    Eight graph nodes: six RNN sub-cells (one cell_cls instance each, own unit
    count) plus two lightweight linear relay nodes (leaky affine maps with
    their own state). The hidden state is the LIST of the eight node states in
    NODE_ORDER; the cell output is the L5ET node state, so
    output_size == lamina_units['L5ET'].

    Orthogonality (identical to NCPWiring): recurrence and continuous-time
    dynamics live inside each sub-cell (cell_cls); this cell contributes only
    the sparse inter-node mask inventory and the gain combiner. Every sub-cell
    is invoked as subcell((basal, elapsed_time), [prev_node]) so
    continuous-time cells receive dt; discrete cells (gru) ignore it.

    The single non-additive edge is the multiplicative gain combiner
    (spec section 5):  h = h_basal * (1 + g * sigmoid(apical)),
    with a learnable per-unit gain g initialised to a small value so training
    starts near the identity but g (and the apical masks behind it) still
    receive gradient. combiner='additive' drops the apical edges and gains
    entirely (no dead parameters), which is the combiner-ablation control.

    Args:
        cell_cls:            _CELL_REGISTRY key (e.g. 'lrc') or BaseCell
                             subclass; must have a single state tensor
                             (state_size == units), so multi-state cells like
                             LSTM/MixedMemory are rejected.
        lamina_units:        optional dict overriding DEFAULT_LAMINA_UNITS
                             entries per node.
        mask_densities:      optional dict overriding DEFAULT_MASK_DENSITIES
                             entries per edge; each density in (0, 1].
        seed:                mask-generation seed (default 42).
        combiner:            'multiplicative' (default) or 'additive'
                             (gain ablation, spec 7.2).
        divisive_inhibition: if True, adds the optional divisive term on the
                             L4 drive (spec 3e); default False.
        sign_constraint:     if True (default), the TRN->Thal edge is
                             sign-locked negative (SignedSparseLinear); if
                             False it is an ordinary free-sign SparseLinear
                             (ablation, spec 6).
        feedforward_only:    if True, builds the feedforward reduction
                             (cncp_ff control, spec 7.1): all feedback, apical
                             and relay-loop edges are removed. The deep
                             readout (L6CC/L6CT) and relay (Thal/TRN) nodes
                             are not instantiated because their outputs would
                             be unobservable sinks (dead parameters); their
                             state slots stay inert at zero so the composite
                             state layout is unchanged.
        gain_init:           initial value of the per-unit multiplicative
                             gain g (default 0.01: near-identity start, spec
                             section 5).
        dt:                  default elapsed time for regularly sampled mode.
        **cell_kwargs:       forwarded to every sub-cell constructor.
    """

    def __init__(self, cell_cls='lrc', lamina_units=None, mask_densities=None,
                 seed=42, combiner='multiplicative', divisive_inhibition=False,
                 sign_constraint=True, feedforward_only=False, gain_init=0.01,
                 dt=1.0, timescale_prior=None, **cell_kwargs):
        if combiner not in ('multiplicative', 'additive'):
            raise ValueError(
                "combiner must be 'multiplicative' or 'additive', "
                f"got {combiner!r}")
        if feedforward_only and divisive_inhibition:
            raise ValueError(
                "divisive_inhibition requires the recurrent wiring: in the "
                "feedforward reduction the L6CT node is inert, so the "
                "divisive edge would be dead. Disable one of the two flags.")
        if 'units' in cell_kwargs:
            raise ValueError(
                "Sub-cell units are set per node via lamina_units; "
                "do not pass 'units' in cell_kwargs.")

        units = dict(DEFAULT_LAMINA_UNITS)
        if lamina_units:
            unknown = set(lamina_units) - set(units)
            if unknown:
                raise ValueError(
                    f"Unknown lamina_units keys {sorted(unknown)}; "
                    f"valid nodes: {list(NODE_ORDER)}")
            units.update(lamina_units)
        densities = dict(DEFAULT_MASK_DENSITIES)
        if mask_densities:
            unknown = set(mask_densities) - set(densities)
            if unknown:
                raise ValueError(
                    f"Unknown mask_densities keys {sorted(unknown)}; "
                    f"valid edges: {list(DEFAULT_MASK_DENSITIES)}")
            densities.update(mask_densities)
        for name, d in densities.items():
            if not 0.0 < d <= 1.0:
                raise ValueError(
                    f"mask density {name} must be in (0, 1], got {d}")

        # BaseCell.units == the output hub width, so the inherited
        # output_size property returns units['L5ET'] (spec section 2). The
        # full per-node dict lives in self._units.
        super().__init__(units=units['L5ET'])
        self._units = units
        self._cell_cls = _resolve_cell_cls(cell_cls)
        self._cell_kwargs = cell_kwargs
        self._densities = densities
        self._seed = seed
        self._combiner = combiner
        self._divisive_inhibition = divisive_inhibition
        self._sign_constraint = sign_constraint
        self._feedforward_only = feedforward_only
        # Iteration 9: optional per-lamina timescale prior. None -> unchanged
        # (every node integrates the same elapsed_time; thesis default). A dict
        # or 'cortical' scales each node's elapsed_time so laminae run at
        # different speeds (fast sensory L4/Thal, slow object/context L2-3/L6),
        # a genuine multi-timescale hierarchy an NCP lacks -- ~0 extra params.
        self._ts = self._resolve_timescale(timescale_prior)
        self._gain_init = gain_init
        self._dt = dt
        # The gain combiner only exists in the full multiplicative wiring:
        # the feedforward reduction zeroes the apical edges (spec 7.1), which
        # makes the combiner the identity, so it is not built there.
        self._use_gain = (combiner == 'multiplicative'
                          and not feedforward_only)

    @property
    def state_size(self):
        return [self._units[node] for node in NODE_ORDER]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        dtype = dtype or tf.float32
        return [tf.zeros([batch_size, self._units[node]], dtype=dtype)
                for node in NODE_ORDER]

    def _edge_shapes(self, input_dim):
        """(in_dim, out_dim) per edge; out_dim == target node units."""
        u = self._units
        return {
            'M_in_L4': (input_dim, u['L4']),
            'M_L4_L23': (u['L4'], u['L23']),
            'M_L23_L5ET': (u['L23'], u['L5ET']),
            'M_L23_L5IT': (u['L23'], u['L5IT']),
            'M_L5IT_L5ET': (u['L5IT'], u['L5ET']),
            'M_L5_L6CC': (u['L5IT'] + u['L5ET'], u['L6CC']),
            'M_L5_L6CT': (u['L5IT'] + u['L5ET'], u['L6CT']),
            'M_L5ET_L5IT': (u['L5ET'], u['L5IT']),
            'M_L6CC_L23': (u['L6CC'], u['L23']),
            'M_ap_L23': (u['L5ET'] + u['L6CT'], u['L23']),
            'M_ap_L5ET': (u['L6CT'] + u['L6CC'], u['L5ET']),
            'M_L6CT_Thal': (u['L6CT'], u['Thal']),
            'M_L6CT_TRN': (u['L6CT'], u['TRN']),
            'M_TRN_Thal': (u['TRN'], u['Thal']),
            'M_Thal_L4': (u['Thal'], u['L4']),
            'M_L6CT_div_L4': (u['L6CT'], u['L4']),
        }

    def _active_edges(self):
        """Edge names instantiated under the current flags."""
        active = ['M_in_L4', 'M_L4_L23', 'M_L23_L5ET', 'M_L23_L5IT',
                  'M_L5IT_L5ET']
        if not self._feedforward_only:
            active += ['M_L5_L6CC', 'M_L5_L6CT', 'M_L5ET_L5IT', 'M_L6CC_L23',
                       'M_L6CT_Thal', 'M_L6CT_TRN', 'M_TRN_Thal', 'M_Thal_L4']
        if self._use_gain:
            active += ['M_ap_L23', 'M_ap_L5ET']
        if self._divisive_inhibition:
            active += ['M_L6CT_div_L4']
        return active

    def build(self, input_shape):
        # Nested tuple -> first item is the feature tensor shape (same
        # convention as CfC_LRC_Cell.build / LTC_Cell.build).
        if isinstance(input_shape[0], (tuple, tf.TensorShape)):
            input_dim = input_shape[0][-1]
        else:
            input_dim = input_shape[-1]
        self.input_dim = input_dim

        shapes = self._edge_shapes(input_dim)

        # Masks are generated once, in the fixed inventory order and for the
        # FULL inventory (also edges not instantiated under the current
        # flags), so ablations sharing a seed see identical masks on their
        # common edges. Dense (density 1.0) masks consume no RNG.
        rng = np.random.default_rng(self._seed)
        self._masks = {
            name: _make_mask(rng, d_in, d_out, self._densities[name])
            for name, (d_in, d_out) in shapes.items()
        }

        self.edges = {}
        for name in self._active_edges():
            d_in, d_out = shapes[name]
            if name == 'M_TRN_Thal' and self._sign_constraint:
                layer = SignedSparseLinear(d_out, self._masks[name], name=name)
            else:
                layer = SparseLinear(d_out, self._masks[name], name=name)
            layer.build((None, d_in))
            self.edges[name] = layer

        # Six RNN sub-cells (four in the feedforward reduction: the deep
        # readout nodes would be unobservable sinks there, see class doc).
        subcell_nodes = ['L4', 'L23', 'L5IT', 'L5ET']
        if not self._feedforward_only:
            subcell_nodes += ['L6CC', 'L6CT']
        self.subcells = {}
        for node in subcell_nodes:
            cell = self._cell_cls(units=self._units[node],
                                  **self._cell_kwargs)
            if isinstance(cell.state_size, (list, tuple)):
                raise ValueError(
                    "CorticalColumnCell requires sub-cells with a single "
                    "state tensor (state_size == units); "
                    f"{self._cell_cls.__name__} has state_size "
                    f"{cell.state_size}. Multi-state cells (lstm, mm_*) do "
                    "not fit the 8-slot composite state.")
            # The node's basal drive has the node's own width (every incoming
            # SparseLinear maps into units[node] and contributions add).
            cell.build((None, self._units[node]))
            self.subcells[node] = cell

        # Linear relay nodes (leaky affine, spec section 4):
        #   h = (1 - alpha) * prev + alpha * tanh(drive + b),
        # alpha = sigmoid(alpha_raw) in (0, 1), learnable, init 0.5.
        if not self._feedforward_only:
            def _relay_weights(prefix, n):
                alpha_raw = self.add_weight(
                    name=f'{prefix}_alpha_raw', shape=(n,), dtype=tf.float32,
                    initializer='zeros')
                bias = self.add_weight(
                    name=f'{prefix}_bias', shape=(n,), dtype=tf.float32,
                    initializer='zeros')
                return alpha_raw, bias

            self.thal_alpha_raw, self.thal_bias = _relay_weights(
                'thal', self._units['Thal'])
            self.trn_alpha_raw, self.trn_bias = _relay_weights(
                'trn', self._units['TRN'])

        # Per-unit multiplicative gains g (spec section 5). Small non-zero
        # init: the combiner starts near the identity, but g and the apical
        # masks behind it still receive non-zero gradient from step one.
        if self._use_gain:
            self.gain_L23 = self.add_weight(
                name='gain_L23', shape=(self._units['L23'],),
                dtype=tf.float32,
                initializer=tf.keras.initializers.Constant(self._gain_init))
            self.gain_L5ET = self.add_weight(
                name='gain_L5ET', shape=(self._units['L5ET'],),
                dtype=tf.float32,
                initializer=tf.keras.initializers.Constant(self._gain_init))

        self.built = True

    @staticmethod
    def _relay(drive, prev, alpha_raw, bias):
        """Leaky affine relay update (spec section 4)."""
        alpha = tf.nn.sigmoid(alpha_raw)
        return (1.0 - alpha) * prev + alpha * tf.nn.tanh(drive + bias)

    @staticmethod
    def _resolve_timescale(spec):
        """None -> off; 'cortical' -> a fast-sensory/slow-context prior; or a
        dict node->factor. Returns a per-node factor dict (or None)."""
        if spec is None:
            return None
        if spec == "cortical":
            # >1 faster (larger effective dt), <1 slower. Fast sensory L4/Thal,
            # slow object/context L2-3/L6, mid L5.
            spec = {"L4": 2.0, "Thal": 2.0, "L23": 0.5, "L6CC": 0.5,
                    "L6CT": 0.5, "L5IT": 1.0, "L5ET": 1.0, "TRN": 1.0}
        if isinstance(spec, dict):
            return {n: float(spec.get(n, 1.0)) for n in NODE_ORDER}
        raise ValueError("timescale_prior must be None, 'cortical', or a dict")

    def call(self, inputs, states):
        if isinstance(inputs, (tuple, list)):
            # Irregularly sampled mode
            x, elapsed_time = inputs
        else:
            # Regularly sampled mode
            x = inputs
            elapsed_time = self._dt

        # Iteration 9: optional per-lamina timescale. et(node) scales the node's
        # integration step; identity when no prior is set (thesis default).
        def et(node):
            return elapsed_time * self._ts[node] if self._ts else elapsed_time

        # Positional unpack of the previous step's state list (NODE_ORDER).
        p_l4, p_l23, p_l5it, p_l5et, p_l6cc, p_l6ct, p_thal, p_trn = states
        e = self.edges
        ff = self._feedforward_only

        # 1. entry node: driver input plus the DELAYED re-entrant relay
        # drive (Thal->L4 reads the previous step's Thal state).
        basal_l4 = e['M_in_L4'](x)
        if not ff:
            basal_l4 = basal_l4 + e['M_Thal_L4'](p_thal)
        if self._divisive_inhibition:
            basal_l4 = basal_l4 / (
                1.0 + tf.nn.softplus(e['M_L6CT_div_L4'](p_l6ct)))
        h_l4, _ = self.subcells['L4']((basal_l4, et('L4')), [p_l4])

        # 2. integration node, with optional top-down multiplicative gain.
        basal_l23 = e['M_L4_L23'](h_l4)
        if not ff:
            basal_l23 = basal_l23 + e['M_L6CC_L23'](p_l6cc)
        h_l23, _ = self.subcells['L23']((basal_l23, et('L23')), [p_l23])
        if self._use_gain:
            apical_l23 = e['M_ap_L23'](tf.concat([p_l5et, p_l6ct], axis=-1))
            h_l23 = h_l23 * (1.0 + self.gain_L23 * tf.nn.sigmoid(apical_l23))

        # 3. intracortical relay node (ET->IT feedback is delayed).
        basal_l5it = e['M_L23_L5IT'](h_l23)
        if not ff:
            basal_l5it = basal_l5it + e['M_L5ET_L5IT'](p_l5et)
        h_l5it, _ = self.subcells['L5IT']((basal_l5it, et('L5IT')),
                                          [p_l5it])

        # 4. output hub, with optional top-down multiplicative gain.
        basal_l5et = e['M_L23_L5ET'](h_l23) + e['M_L5IT_L5ET'](h_l5it)
        h_l5et, _ = self.subcells['L5ET']((basal_l5et, et('L5ET')),
                                          [p_l5et])
        if self._use_gain:
            apical_l5et = e['M_ap_L5ET'](tf.concat([p_l6ct, p_l6cc], axis=-1))
            h_l5et = h_l5et * (
                1.0 + self.gain_L5ET * tf.nn.sigmoid(apical_l5et))

        if ff:
            # Feedforward reduction: deep readout and relay nodes are not
            # built; their state slots stay inert (zeros from init).
            h_l6cc, h_l6ct, h_thal, h_trn = p_l6cc, p_l6ct, p_thal, p_trn
        else:
            # 5. deep readout nodes.
            deep_in = tf.concat([h_l5it, h_l5et], axis=-1)
            h_l6cc, _ = self.subcells['L6CC'](
                (e['M_L5_L6CC'](deep_in), et('L6CC')), [p_l6cc])
            h_l6ct, _ = self.subcells['L6CT'](
                (e['M_L5_L6CT'](deep_in), et('L6CT')), [p_l6ct])

            # 6. re-entrant relay loop (acyclic within the step:
            # L6CT -> TRN -> Thal; the Thal->L4 read-back happens next step).
            # With sign_constraint the TRN->Thal edge emits a strictly
            # negative contribution (SignedSparseLinear folds the minus sign
            # of spec section 4 into the layer); without the constraint it is
            # a free-sign SparseLinear, so the sum realises the ablation.
            h_trn = self._relay(e['M_L6CT_TRN'](h_l6ct), p_trn,
                                self.trn_alpha_raw, self.trn_bias)
            thal_in = e['M_L6CT_Thal'](h_l6ct) + e['M_TRN_Thal'](h_trn)
            h_thal = self._relay(thal_in, p_thal,
                                 self.thal_alpha_raw, self.thal_bias)

        # Node outputs are carried as the node states (spec section 4): the
        # sub-cells' own returned states are discarded, so the gain-modulated
        # value is both the value passed downstream and the stored state.
        new_states = [h_l4, h_l23, h_l5it, h_l5et, h_l6cc, h_l6ct,
                      h_thal, h_trn]
        return h_l5et, new_states


class CNCPWiring(BaseWiring):
    """cNCP wiring: one composite CorticalColumnCell wrapped in an RNN layer.

    build_model() returns a tf.keras.Sequential
    [RNN(CorticalColumnCell, return_sequences=True), Dense(output_neurons)?]
    so the model drops into SequentialODEFunc exactly like DenseWiring /
    NCPWiring models do. The optional Dense projection mirrors
    make_dense_model(..., output_neurons=...).

    Args:
        cell_cls:        _CELL_REGISTRY key or BaseCell subclass (sub-cell).
        output_neurons:  if given, appends a Dense(output_neurons) projection.
        (remaining args) forwarded to CorticalColumnCell, see there.
    """

    def __init__(self, cell_cls, output_neurons=None, lamina_units=None,
                 mask_densities=None, seed=42, combiner='multiplicative',
                 divisive_inhibition=False, sign_constraint=True,
                 feedforward_only=False, **cell_kwargs):
        super().__init__(cell=None)
        self.cell_cls = cell_cls
        self.output_neurons = output_neurons
        self.lamina_units = lamina_units
        self.mask_densities = mask_densities
        self.seed = seed
        self.combiner = combiner
        self.divisive_inhibition = divisive_inhibition
        self.sign_constraint = sign_constraint
        self.feedforward_only = feedforward_only
        self.cell_kwargs = cell_kwargs

    def build_model(self) -> tf.keras.Sequential:
        cell = CorticalColumnCell(
            cell_cls=self.cell_cls,
            lamina_units=self.lamina_units,
            mask_densities=self.mask_densities,
            seed=self.seed,
            combiner=self.combiner,
            divisive_inhibition=self.divisive_inhibition,
            sign_constraint=self.sign_constraint,
            feedforward_only=self.feedforward_only,
            **self.cell_kwargs,
        )
        layers = [tf.keras.layers.RNN(cell, return_sequences=True)]
        if self.output_neurons is not None:
            layers.append(tf.keras.layers.Dense(self.output_neurons))
        return tf.keras.Sequential(layers)
