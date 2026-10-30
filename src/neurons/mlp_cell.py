# Stateless MLP control for the rollout tasks (issue #60, protocol freeze
# F1-2026-10-26 decision b): a memoryless flow map
#
#     h_t = f2(f1([x_t, dt_t])),   f_i = Dense(units, lecun_tanh)
#
# followed by the task's linear readout. It is a registry cell so it runs
# through the same runner, ForcedRolloutCell wrapper, HPO path and RQ4
# diagnostics as every recurrent arm, on the 'dense' wiring only.

import tensorflow as tf

from .base_cell import BaseCell
from .cfc_cell import _lecun_tanh


class MLP_Cell(BaseCell):
    """Two Dense(units, lecun_tanh) layers on (x_t, dt_t); no memory.

    The state slot exists because tf.keras.layers.RNN needs one. It holds
    the last output h_t and is never read, so the output at step t depends on
    the step-t input alone, and the one-step state Jacobian is zero (the RQ4
    retention diagnostics then report no memory, as they should). dt is a
    regular input feature (1.0 when the cell gets no elapsed time).
    """

    def __init__(self, units, **kwargs):
        super().__init__(units, **kwargs)
        if self.is_masked:
            raise ValueError(
                "MLP_Cell is the stateless control of the 'dense' wiring; it "
                "has no recurrent synapses for a sparse (NCP) wiring mask.")

    def build(self, input_shape):
        if isinstance(input_shape[0], (tuple, list, tf.TensorShape)):
            input_shape = input_shape[0]
        width = int(input_shape[-1]) + 1                 # x_t ++ dt_t
        # backbone_* names put both layers in the RQ4 'backbone' group.
        self._layers = []
        for i in range(2):
            layer = tf.keras.layers.Dense(self.units, activation=_lecun_tanh,
                                          name=f'backbone_{i}')
            layer.build((None, width))
            width = self.units
            self._layers.append(layer)
        self.built = True

    def call(self, inputs, states):
        if isinstance(inputs, (tuple, list)):
            x, dt = inputs
            dt = tf.broadcast_to(tf.cast(dt, x.dtype), tf.shape(x[:, :1]))
        else:
            x, dt = inputs, tf.ones_like(inputs[:, :1])
        h = tf.concat([x, dt], axis=-1)
        for layer in self._layers:
            h = layer(h)
        return h, [h]
