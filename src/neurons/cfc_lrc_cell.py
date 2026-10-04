# Closed-form Liquid-Resistance Liquid-Capacitance (cfc_lrc) cell.
#
# This is the closed-form analogue of the LRC cell, in the same way that CfC
# (Hasani, Lechner et al. 2022, arXiv:2106.13898) is the closed-form analogue of
# LTC. The LRC ODE (Farsang, Neubauer, Grosu, arXiv:2403.08791) generalises LTC by
# a state/input-dependent liquid elastance eps in [0,1] that multiplies the whole
# right-hand side:
#
#     dh/dt = eps(u,h) * ( -f(u,h) * h + b(u,h) )
#
# Freezing (eps, f, b) over one step (the same per-step approximation Hasani uses
# for LTC -> CfC, and Farsang justify for a single LRC Euler unfold) lets the
# integrating factor close the ODE exactly:
#
#     h(dt) = h0 * exp(-eps*f*dt) + (b/f) * (1 - exp(-eps*f*dt))
#
# Two facts drive the cell design (verified numerically to ~2e-6 against fine Euler):
#   - eps CANCELS from the equilibrium h_inf = b/f (identical to LTC); it survives
#     ONLY as the decay-rate multiplier lambda = eps*f. Elastance rescales how fast
#     the neuron relaxes toward the same equilibrium, it does not move it.
#   - Replacing exp(-lambda*dt) by Hasani's bounded learned sigmoid gate and keeping
#     eps as the rate multiplier yields plain CfC plus one multiplicative elastance
#     factor on the gate time-rate:
#
#         g = sigmoid( eps * (t_a * dt) + t_b ),   h' = ff1*(1-g) + g*ff2
#
# So cfc_lrc == cfc with one extra bounded elastance head; eps -> 1 recovers CfC.
#
# Scope/honesty (see docs/superpowers/specs/2026-06-19-cfc-lrc-v3_1-design.md):
# closed-form only under the per-step frozen-coefficient approximation (O(dt), same
# status as CfC/LTC), NOT an exact solution of the full nonlinear LRC ODE. Because
# eps cancels from the equilibrium, this cell expresses elastance only as timescale
# modulation, never equilibrium reshaping. Assumes the forget_gate=True LRC variant.

import tensorflow as tf

from .base_cell import BaseCell
from .cfc_cell import _lecun_tanh


class CfC_LRC_Cell(BaseCell):
    def __init__(self, units, backbone_units=None, backbone_layers=1, dt=1.0,
                 elastance_type='asymmetric', elastance_init_bias=1.0,
                 elastance_gate='inner', **kwargs):
        """Closed-form LRC cell: a CfC whose time-gate is modulated by a liquid
        elastance head.

        Args:
            units:               hidden state size
            backbone_units:      width of the shared backbone (default: units)
            backbone_layers:     number of backbone Dense layers (default 1)
            dt:                  default elapsed time for regularly sampled mode
            elastance_type:      'asymmetric' (eps = sigmoid(W[u,h])) or 'symmetric'
                                 (eps = sigmoid(.+k) - sigmoid(.-k), k >= 0). Matches
                                 the LRC_Cell elastance formulation.
            elastance_init_bias: bias initialiser for the elastance head. A positive
                                 value warm-starts eps near 1 (asymmetric), so the
                                 cell begins close to plain CfC and the optimiser must
                                 actively introduce elastance.
            elastance_gate:      where eps enters the time-gate (see call()).
                                 'inner' (default): g = sigmoid(eps*(t_a*dt) + t_b) --
                                 the derived closed form, eps modulates the gate's
                                 time-rate. 'outer': g = eps*sigmoid(t_a*dt + t_b) --
                                 M. Farsang's 2026-07-08 review variant, eps caps the
                                 gate amplitude. Both recover CfC as eps -> 1; the two
                                 are benchmarked head-to-head (identical parameter
                                 count -- eps only changes position, not head size).
        """
        super().__init__(units, **kwargs)
        if elastance_type not in ('asymmetric', 'symmetric'):
            raise ValueError(
                "elastance_type must be 'asymmetric' or 'symmetric', got "
                f"{elastance_type!r}. (A constant elastance would collapse cfc_lrc "
                "back to plain cfc; use the 'cfc' cell for that.)"
            )
        if elastance_gate not in ('inner', 'outer'):
            raise ValueError(
                "elastance_gate must be 'inner' or 'outer', got "
                f"{elastance_gate!r}."
            )
        self._backbone_units = backbone_units or units
        self._backbone_layers = backbone_layers
        self._dt = dt
        self._elastance_type = elastance_type
        self._elastance_init_bias = elastance_init_bias
        self._elastance_gate = elastance_gate

    def build(self, input_shape):
        # Nested tuple -> first item is the feature tensor shape (same convention as
        # CfC_Cell.build / LTC_Cell.build).
        if isinstance(input_shape[0], (tuple, tf.TensorShape)):
            input_dim = input_shape[0][-1]
        else:
            input_dim = input_shape[-1]
        self.input_dim = input_dim
        self._build_masks(input_dim)
        if self.is_masked and self._backbone_units != self.units:
            raise ValueError(
                "Under a sparse (NCP) wiring the first backbone layer is an "
                "[x, h] -> units map that carries the wiring mask, so its width "
                f"must equal units ({self.units}); got backbone_units="
                f"{self._backbone_units}.")

        raw_width = input_dim + self.units

        # Shared backbone (identical to CfC).
        width = raw_width
        self._backbone = []
        for i in range(self._backbone_layers):
            layer = tf.keras.layers.Dense(
                self._backbone_units, activation=_lecun_tanh,
                name=f'backbone_{i}',
            )
            layer.build((None, width))
            width = self._backbone_units
            self._backbone.append(layer)

        def _head(name):
            head = tf.keras.layers.Dense(self.units, name=name)
            head.build((None, width))
            return head

        self._ff1 = _head('ff1')
        self._ff2 = _head('ff2')
        self._time_a = _head('time_a')
        self._time_b = _head('time_b')

        # Liquid-elastance head -- the single addition over CfC. Computed from the
        # raw [inputs, state] (same input the numerical LRC_Cell.elastance_mapping
        # uses), not from the backbone features, so it stays interpretable as the
        # LRC liquid elastance. Bias warm-starts eps high (asymmetric -> near 1).
        self._elastance = tf.keras.layers.Dense(
            self.units, name='elastance_mapping',
            bias_initializer=tf.keras.initializers.Constant(
                self._elastance_init_bias),
        )
        self._elastance.build((None, raw_width))

        if self._elastance_type == 'symmetric':
            self._distr_shift = self.add_weight(
                name='distr_shift', shape=(self.units,), dtype=tf.float32,
                constraint=tf.keras.constraints.NonNeg(),
                initializer=tf.keras.initializers.Constant(1),
            )

        self.built = True

    def _elastance_value(self, raw):
        """Liquid elastance eps in [0,1] from the raw [inputs, state] tensor."""
        e_pre = self._masked_dense(self._elastance, raw)
        if self._elastance_type == 'symmetric':
            return (tf.nn.sigmoid(e_pre + self._distr_shift)
                    - tf.nn.sigmoid(e_pre - self._distr_shift))
        return tf.nn.sigmoid(e_pre)

    def call(self, inputs, states):
        if isinstance(inputs, (tuple, list)):
            # Irregularly sampled mode
            inputs, elapsed_time = inputs
        else:
            # Regularly sampled mode
            elapsed_time = self._dt

        raw = tf.concat([inputs, states[0]], axis=-1)

        x = raw
        # NCP wiring: backbone_0 and the elastance head are the [x, h] -> units
        # maps, so both carry the concat mask.
        for i, layer in enumerate(self._backbone):
            x = self._masked_dense(layer, x) if i == 0 else layer(x)

        # backbone_layers=0 (ncps WiredCfCCell): the heads ARE the [x, h] ->
        # units maps, so each carries the concat mask (no lateral mixing).
        head = ((lambda l: self._masked_dense(l, x)) if not self._backbone
                else (lambda l: l(x)))
        ff1 = head(self._ff1)
        ff2 = head(self._ff2)
        t_a = head(self._time_a)
        t_b = head(self._time_b)

        eps = self._elastance_value(raw)
        if self._elastance_gate == 'inner':
            # Derived closed form: eps modulates the gate's time-rate (lambda =
            # eps*f in the exponent); t_b is the gate bias and is NOT scaled by eps.
            t_interp = tf.nn.sigmoid(eps * (t_a * elapsed_time) + t_b)
        else:  # 'outer' -- M. Farsang review variant (2026-07-08). eps scales the
            # gate OUTPUT, capping the interpolation weight at eps in (0,1) instead
            # of sitting in the decay exponent. NOT the integrating-factor closed
            # form; a retention-ceiling heuristic that also recovers CfC as eps->1.
            t_interp = eps * tf.nn.sigmoid(t_a * elapsed_time + t_b)
        new_state = ff1 * (1.0 - t_interp) + t_interp * ff2
        return new_state, [new_state]
