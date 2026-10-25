# cNCP v2: the four-node cortical-column wiring (registry key 'cncp2').
# Design contract: docs/superpowers/specs/2026-10-21-cncp-v2-abstraction.md,
# Part B (B.3 nodes/edges/update, B.4 parameter count, B.5 stability rules
# R1 to R5); the implementation record is its section "Implementation status".
#
# Same construction principle as the eight-node cNCP (src/wirings/cncp.py):
# a fixed directed graph whose nodes are pluggable RNN sub-cells and whose
# edges are learnable SparseLinear maps under fixed binary masks. The node
# labels G (granular entry), S (superficial integration), D (deep output hub)
# and R (gated relay) name nodes of a computation graph; this is ordinary
# tensor algebra trained by backprop.
#
# Within one step the order G, S, D, R is acyclic; every cycle is closed by a
# read of the PREVIOUS step's stored state (R -> G, D -> S additive, D -> S
# apical), exactly as in the eight-node cell.

import numpy as np
import tensorflow as tf

from src.neurons.base_cell import BaseCell
from .base_wiring import BaseWiring
from .cncp import _make_mask, _resolve_cell_cls
from .committee import VotingMixin
from .ncp import SparseLinear, ncp_cell_kwargs

# Positional order of the composite state list (spec B.3, "new_state").
NODE_ORDER_V2 = ('G', 'S', 'D', 'R')

# Edge inventory in mask-generation order (spec B.3 edge table). Only E_DS and
# A_DS are sparse; dense masks consume no RNG (cncp._make_mask), so every
# ablation that shares a seed sees the same two Bernoulli(0.5) masks.
EDGE_DENSITIES_V2 = {
    'E_in': 1.0, 'E_RG': 1.0, 'E_GS': 1.0, 'E_DS': 0.5,
    'A_DS': 0.5, 'E_SD': 1.0, 'E_DR': 1.0, 'E_DRg': 1.0,
}

RELAY_GATES = ('attenuate', 'off', 'free')

# Gate bias init (R4): sigmoid(-2) ~ 0.12, the relay starts nearly open.
_GATE_BIAS_INIT = -2.0


class CorticalColumnV2Cell(BaseCell):
    """Composite RNN cell of the four-node cNCP v2 column (spec B.3).

    Nodes: G, S, D are sub-cells of ``cell_cls`` (width u each, built through
    ``ncp_cell_kwargs`` like the cNCP and NCP layers, so CfC-family sub-cells
    run without a backbone); R is a gated leaky relay of width
    u_R = max(1, u // 2). The state is the list [g, s, d, r] (NODE_ORDER_V2),
    the output is d (output_size == u).

    Step (spec B.3; ``_`` marks the previous step's stored state):
        g   = cell_G(E_in(x) + E_RG(tanh(r_)))
        s   = cell_S(E_GS(g) + E_DS(tanh(d_)))
        s~  = s * (1 + gamma * tanh(A_DS(tanh(d_)) / 2))
        d   = cell_D(E_SD(s~))
        r   = (1 - alpha) * r_ + alpha * tanh(E_DR(d) + b_r) * (1 - gate)
        gate  = sigmoid(E_DRg(d) + b_gate),  alpha = 1 - exp(-dt softplus(rho))
    S stores the raw s, D receives the modulated s~ (R5).

    Args:
        cell_cls:          _CELL_REGISTRY key or single-state BaseCell subclass.
        units:             the width knob u.
        seed:              mask-generation seed.
        feedback:          False drops the additive top-down edge E_DS.
        gain:              False drops the gain (gamma fixed 0, A_DS not built).
        relay_gate:        'attenuate' (default, the gate can only reduce the
                           relay target), 'off' (gate fixed 0, E_DRg/b_gate not
                           built) or 'free' (the factor (1 - gate) is replaced by
                           the unconstrained linear factor E_DRg(d) + b_gate).
        feedforward_only:  True keeps only G -> S -> D (no E_RG, E_DS, A_DS, no
                           relay); the R slot stays inert at zero so the state
                           layout is unchanged. Overrides gain and relay_gate.
        dt:                elapsed time in regularly sampled mode.
        extra_s_edges:     number of external edges summed into S (the lateral
                           coupling of cncp2_multi counts as one); only enters
                           the R2 fan-in of E_GS / E_DS. The drive itself is
                           passed as a third input element, (x, dt, s_extra),
                           and added to basal_S.
        **cell_kwargs:     forwarded to the sub-cells (after ncp_cell_kwargs).
    """

    def __init__(self, cell_cls='cfc_lrc', units=8, seed=42, feedback=True,
                 gain=True, relay_gate='attenuate', feedforward_only=False,
                 dt=1.0, extra_s_edges=0, **cell_kwargs):
        if relay_gate not in RELAY_GATES:
            raise ValueError(
                f"relay_gate must be one of {RELAY_GATES}, got {relay_gate!r}")
        if 'units' in cell_kwargs:
            raise ValueError("Sub-cell width is the column's units; do not "
                             "pass 'units' in cell_kwargs.")
        if units < 1:
            raise ValueError(f"units must be >= 1, got {units}")
        super().__init__(units=int(units))
        self._u_r = max(1, int(units) // 2)
        self._cell_cls = _resolve_cell_cls(cell_cls)
        self._cell_kwargs = cell_kwargs
        self._seed = seed
        self._ff = bool(feedforward_only)
        self._feedback = bool(feedback) and not self._ff
        self._gain = bool(gain) and not self._ff
        self._relay = not self._ff
        self._gate = relay_gate if self._relay else 'off'
        self._dt = dt
        self._extra_s = int(extra_s_edges)

    @property
    def state_size(self):
        u = self.units
        return [u, u, u, self._u_r]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        dtype = dtype or tf.float32
        return [tf.zeros([batch_size, n], dtype=dtype) for n in self.state_size]

    def _active_edges(self):
        """Edges instantiated under the current flags. Build order matters
        for weight init under a fixed global seed: the optional gain edge is
        built last, so gain=True and gain=False draw identical weights for
        every common variable (the gamma = 0 reduction test relies on it)."""
        edges = ['E_in']
        if self._relay:
            edges.append('E_RG')
        edges.append('E_GS')
        if self._feedback:
            edges.append('E_DS')
        edges.append('E_SD')
        if self._relay:
            edges.append('E_DR')
            if self._gate != 'off':
                edges.append('E_DRg')
        return edges

    def _edge_shapes(self, input_dim):
        u, ur = self.units, self._u_r
        return {'E_in': (input_dim, u), 'E_RG': (ur, u), 'E_GS': (u, u),
                'E_DS': (u, u), 'A_DS': (u, u), 'E_SD': (u, u),
                'E_DR': (u, ur), 'E_DRg': (u, ur)}

    def build(self, input_shape):
        if isinstance(input_shape[0], (tuple, list, tf.TensorShape)):
            input_shape = input_shape[0]
        input_dim = int(input_shape[-1])
        self.input_dim = input_dim
        shapes = self._edge_shapes(input_dim)

        # Masks for the FULL inventory in fixed order (see EDGE_DENSITIES_V2).
        rng = np.random.default_rng(self._seed)
        self._masks = {name: _make_mask(rng, *shapes[name], density)
                       for name, density in EDGE_DENSITIES_V2.items()}

        # R2, fan-in-aware init: an edge into a node with n summed incoming
        # edges starts with glorot variance / n (weights scaled by 1/sqrt(n)),
        # so the summed drive has the variance of one edge. G sums E_in + E_RG,
        # S sums E_GS + E_DS; D and both relay projections have one edge each,
        # and A_DS feeds the gain, not a sum. n follows the active flags.
        active = self._active_edges()
        fan_in = {'E_in': 1 + ('E_RG' in active),
                  'E_GS': 1 + ('E_DS' in active) + self._extra_s}
        fan_in['E_RG'] = fan_in['E_in']
        fan_in['E_DS'] = fan_in['E_GS']

        self.edges = {}

        def _edge(name):
            d_in, d_out = shapes[name]
            layer = SparseLinear(d_out, self._masks[name], name=name)
            layer.build((None, d_in))
            n = fan_in.get(name, 1)
            if n > 1:
                layer.W.assign(layer.W / np.sqrt(n))
            self.edges[name] = layer

        for name in active:
            _edge(name)

        kw = ncp_cell_kwargs(self._cell_cls, self._cell_kwargs)
        self.subcells = {}
        for node in ('G', 'S', 'D'):
            cell = self._cell_cls(units=self.units, **kw)
            if isinstance(cell.state_size, (list, tuple)):
                raise ValueError(
                    "CorticalColumnV2Cell requires single-state sub-cells "
                    f"(state_size == units); {self._cell_cls.__name__} has "
                    f"state_size {cell.state_size}.")
            cell.build((None, self.units))      # basal drive is u wide
            self.subcells[node] = cell

        ur = self._u_r
        if self._relay:
            # rho = 0 -> alpha(dt = 1) = 0.5 (R4); b_r = 0.
            self.rho = self.add_weight(name='rho', shape=(ur,),
                                       initializer='zeros')
            self.b_r = self.add_weight(name='b_r', shape=(ur,),
                                       initializer='zeros')
            if self._gate == 'attenuate':
                self.b_gate = self.add_weight(
                    name='b_gate', shape=(ur,),
                    initializer=tf.keras.initializers.Constant(_GATE_BIAS_INIT))
            elif self._gate == 'free':
                # Free factor starts at the attenuating factor's zero-drive
                # value 1 - sigmoid(-2), so 'free' and 'attenuate' agree at
                # init for d = 0 and differ only in what training may reach.
                self.b_gate = self.add_weight(
                    name='b_gate', shape=(ur,),
                    initializer=tf.keras.initializers.Constant(
                        1.0 - 1.0 / (1.0 + np.exp(-_GATE_BIAS_INIT))))

        if self._gain:
            _edge('A_DS')
            # R3: gamma = 0 exactly, the gain factor is the identity at init.
            self.gamma = self.add_weight(name='gamma', shape=(self.units,),
                                         initializer='zeros')
        self.built = True

    def call(self, inputs, states):
        s_extra = None
        if isinstance(inputs, (tuple, list)):
            x, dt, *rest = inputs
            s_extra = rest[0] if rest else None
        else:
            x, dt = inputs, self._dt
        g_, s_, d_, r_ = states
        e, c = self.edges, self.subcells

        # R1: every delayed cross-node read passes through tanh.
        fb_d = tf.tanh(d_) if (self._feedback or self._gain) else None

        basal_g = e['E_in'](x)
        if self._relay:
            basal_g = basal_g + e['E_RG'](tf.tanh(r_))
        g, _ = c['G']((basal_g, dt), [g_])

        basal_s = e['E_GS'](g)
        if self._feedback:
            basal_s = basal_s + e['E_DS'](fb_d)
        if s_extra is not None:
            basal_s = basal_s + s_extra
        s, _ = c['S']((basal_s, dt), [s_])
        s_mod = s
        if self._gain:
            # Centred gain: factor exactly 1 at zero apical drive (spec B.3).
            s_mod = s * (1.0 + self.gamma * tf.tanh(e['A_DS'](fb_d) / 2.0))

        d, _ = c['D']((e['E_SD'](s_mod), dt), [d_])

        if self._relay:
            target = tf.tanh(e['E_DR'](d) + self.b_r)
            if self._gate == 'attenuate':
                target = target * (1.0 - tf.sigmoid(e['E_DRg'](d) + self.b_gate))
            elif self._gate == 'free':
                target = target * (e['E_DRg'](d) + self.b_gate)
            alpha = 1.0 - tf.exp(-dt * tf.nn.softplus(self.rho))
            r = (1.0 - alpha) * r_ + alpha * target
        else:
            r = r_                                  # inert slot (feedforward)

        # R5: S stores the raw sub-cell state, not the modulated message.
        return d, [g, s, d, r]


class CNCP2Wiring(BaseWiring):
    """cNCP v2 wiring: RNN(CorticalColumnV2Cell) plus an optional Dense head,
    the same model shape as CNCPWiring.build_model.

    Args:
        cell_cls:        _CELL_REGISTRY key or BaseCell subclass (sub-cell).
        output_neurons:  if given, appends Dense(output_neurons).
        units:           column width u.
        (rest)           forwarded to CorticalColumnV2Cell.
    """

    def __init__(self, cell_cls, output_neurons=None, units=8, seed=42,
                 feedback=True, gain=True, relay_gate='attenuate',
                 feedforward_only=False, **cell_kwargs):
        super().__init__(cell=None)
        self.cell_cls = cell_cls
        self.output_neurons = output_neurons
        self.cell_args = dict(units=units, seed=seed, feedback=feedback,
                              gain=gain, relay_gate=relay_gate,
                              feedforward_only=feedforward_only, **cell_kwargs)

    def make_cell(self):
        return CorticalColumnV2Cell(cell_cls=self.cell_cls, **self.cell_args)

    def build_model(self) -> tf.keras.Sequential:
        layers = [tf.keras.layers.RNN(self.make_cell(), return_sequences=True)]
        if self.output_neurons is not None:
            layers.append(tf.keras.layers.Dense(self.output_neurons))
        return tf.keras.Sequential(layers)


_S_IDX = NODE_ORDER_V2.index('S')


class MultiColumnV2VotingCell(VotingMixin, tf.keras.layers.AbstractRNNCell):
    """K weight-shared cNCP v2 columns voting on S (spec B.6 (iv)).

    The committee arm for 'cncp2': the same consensus step as
    MultiColumnVotingCell (VotingMixin), with the four-node column in place of
    the eight-node one. The stored S states are pulled toward their consensus
    and the output is the consensus of the mixed S states (the analogue of the
    L2/3 vote and readout of the 'cncp' committee). Per-step input
    (features (B, K, F), time (B, 1)); state = K copies of [g, s, d, r], flat
    in column-major order.
    """

    def __init__(self, n_columns=3, cell_cls='cfc_lrc', units=8, seed=42,
                 vote_init=0.0, vote_lambda=None, consensus='mean',
                 **cell_kwargs):
        super().__init__()
        self.K = int(n_columns)
        self._init_vote(vote_lambda, vote_init, consensus)
        self.col = CorticalColumnV2Cell(cell_cls=cell_cls, units=units,
                                        seed=seed, **cell_kwargs)

    @property
    def state_size(self):
        return list(self.col.state_size) * self.K

    @property
    def output_size(self):
        return self.col.units

    def build(self, input_shape):
        self.col.build((tf.TensorShape([None, int(input_shape[0][-1])]),
                        tf.TensorShape([None, 1])))
        self._build_vote()
        self.built = True

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return self.col.get_initial_state(batch_size=batch_size,
                                          dtype=dtype) * self.K

    def call(self, inputs, states):
        features, dt = inputs
        n = len(NODE_ORDER_V2)
        per_col = []
        for k in range(self.K):
            _, ns = self.col((features[:, k, :], dt),
                             list(states[k * n:(k + 1) * n]))
            per_col.append(ns)
        mixed, output = self._consensus([ns[_S_IDX] for ns in per_col])
        new_states = []
        for ns, m in zip(per_col, mixed):
            ns = list(ns)
            ns[_S_IDX] = m
            new_states.extend(ns)
        return output, new_states
