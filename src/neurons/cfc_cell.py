# Follows the CfC formulation by Hasani, Lechner et al. (2022),
# "Closed-form continuous-time neural networks", Nature Machine Intelligence
# 4(11), arXiv:2106.13898, and the reference implementation
# https://github.com/mlech26l/ncps/blob/master/ncps/tf/cfc_cell.py
# (default/gated mode). Verification note (thesis): cross-check gate signs
# against the paper PDF before citing equations.

import tensorflow as tf
from .base_cell import BaseCell


def _lecun_tanh(x):
    """Activation used by the CfC reference backbone."""
    return 1.7159 * tf.math.tanh(0.666 * x)


class CfC_Cell(BaseCell):
    def __init__(self, units, backbone_units=None, backbone_layers=1,
                 dt=1.0, **kwargs):
        """
        Closed-form Continuous-time (CfC) <https://arxiv.org/abs/2106.13898> cell.

        Approximates the LTC ODE solution in closed form -- no ODE solver,
        no unfolding. The hidden state is a sigmoid time-gated interpolation
        between two learned regimes:

            h(t) = sigma(t_a * t + t_b) interpolating ff1 <-> ff2

        which replaces the exponential decay of the exact solution and
        thereby avoids its vanishing-gradient factor (paper, Sec. 3).

        Args:
            units:           hidden state size
            backbone_units:  width of the shared backbone (default: units)
            backbone_layers: number of backbone Dense layers (default 1)
            dt:              default elapsed time for regularly sampled mode
        """
        super().__init__(units, **kwargs)
        self._backbone_units = backbone_units or units
        self._backbone_layers = backbone_layers
        self._dt = dt

    def build(self, input_shape):
        # Nested tuple -> first item is the feature tensor shape (same
        # convention as LTC_Cell.build).
        if isinstance(input_shape[0], (tuple, tf.TensorShape)):
            input_dim = input_shape[0][-1]
        else:
            input_dim = input_shape[-1]
        self.input_dim = input_dim

        width = input_dim + self.units
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
        self.built = True

    def call(self, inputs, states):
        if isinstance(inputs, (tuple, list)):
            # Irregularly sampled mode
            inputs, elapsed_time = inputs
        else:
            # Regularly sampled mode
            elapsed_time = self._dt

        x = tf.concat([inputs, states[0]], axis=-1)
        for layer in self._backbone:
            x = layer(x)

        ff1 = self._ff1(x)
        ff2 = self._ff2(x)
        t_a = self._time_a(x)
        t_b = self._time_b(x)
        t_interp = tf.nn.sigmoid(t_a * elapsed_time + t_b)
        new_state = ff1 * (1.0 - t_interp) + t_interp * ff2
        return new_state, [new_state]
