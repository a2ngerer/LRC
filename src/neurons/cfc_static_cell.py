# CfC with input-independent time constants: the H-a control (issue #31).
#
# H-a contrasts input-dependent time constants against learned, input-independent
# ones. ctrnn differs from the liquid cells also in solver and (under NCP) in the
# propagation semantics; this cell differs from cfc ONLY in the gate: the time
# heads t_a(x, h), t_b(x, h) become per-unit trainable vectors, so
#
#     g = sigmoid(t_a * dt + t_b)        (same for every input and state)
#     h' = ff1 * (1 - g) + g * ff2       (ff1, ff2, backbone as in cfc)
#
# With backbone_layers=0 and linear ff heads the gate is the only nonlinearity of
# a CfC, so this cell is then affine in [x, h]; it is a nonlinear control only
# with a backbone (or a head activation).

import tensorflow as tf

from .cfc_cell import CfC_Cell


class CfCStaticCell(CfC_Cell):
    """CfC whose gate time-rate and bias are per-unit vectors (no time heads)."""

    def build(self, input_shape):
        super().build(input_shape)
        # ponytail: the parent builds the time_a/time_b heads and they are dropped
        # here (Keras untracks a reassigned sub-layer). That keeps the RNG stream
        # of cfc, so at the same seed every shared weight (backbone, ff1, ff2,
        # readout) starts identical to cfc's.
        self._time_a = self._time_b = None
        # Zeros, like the Dense-head biases they replace.
        self.t_a = self.add_weight(name='t_a', shape=(self.units,),
                                   initializer='zeros')
        self.t_b = self.add_weight(name='t_b', shape=(self.units,),
                                   initializer='zeros')

    def _time_constants(self, head):
        return self.t_a, self.t_b

    def get_config(self):
        # The wiring masks are not part of the config; the wiring passes them.
        return {**super().get_config(), 'units': self.units,
                'backbone_units': self._backbone_units,
                'backbone_layers': self._backbone_layers, 'dt': self._dt}
