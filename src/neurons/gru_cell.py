import tensorflow as tf
from .base_cell import BaseCell


class GRU_Cell(BaseCell):
    """GRU baseline cell.

    Thin wrapper around tf.keras.layers.GRUCell to conform to BaseCell,
    mirroring the LSTM_Cell wrapper.

    State: [h] — one tensor of shape (batch, units).
    Output: h (hidden state), shape (batch, units).
    """

    def __init__(self, units, **kwargs):
        super().__init__(units, **kwargs)

    def build(self, input_shape):
        self._gru = tf.keras.layers.GRUCell(self.units)
        self._gru.build(input_shape)
        self.built = True

    def call(self, inputs, states):
        if isinstance(inputs, (tuple, list)):
            inputs, _ = inputs   # discard elapsed_time (GRU is discrete)
        output, new_states = self._gru(inputs, states)
        if not isinstance(new_states, (tuple, list)):
            new_states = [new_states]
        return output, new_states
