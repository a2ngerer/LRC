import tensorflow as tf
from .base_cell import BaseCell


class LSTM_Cell(BaseCell):
    """LSTM baseline cell.

    Thin wrapper around tf.keras.layers.LSTMCell to conform to BaseCell.

    State: [h, c] — two tensors of shape (batch, units) each.
    Output: h (hidden state), shape (batch, units).
    output_size: units (h dimension only, not [h, c]).

    Note: state_size and get_initial_state override BaseCell defaults
    because LSTM requires two state tensors instead of one.
    """

    def __init__(self, units, **kwargs):
        super().__init__(units, **kwargs)

    @property
    def state_size(self):
        return [self.units, self.units]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        dtype = dtype or tf.float32
        return [
            tf.zeros([batch_size, self.units], dtype=dtype),  # h
            tf.zeros([batch_size, self.units], dtype=dtype),  # c
        ]

    def build(self, input_shape):
        self._lstm = tf.keras.layers.LSTMCell(self.units)
        self._lstm.build(input_shape)
        self.built = True

    def call(self, inputs, states):
        if isinstance(inputs, (tuple, list)):
            inputs, _ = inputs   # discard elapsed_time (LSTM is discrete)
        output, new_states = self._lstm(inputs, states)
        return output, new_states
