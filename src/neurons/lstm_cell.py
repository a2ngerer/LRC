import tensorflow as tf
from .base_cell import BaseCell, MaskedLSTMCell


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
        if isinstance(input_shape[0], (tuple, list, tf.TensorShape)):
            input_shape = input_shape[0]
        self._build_masks(input_shape[-1])
        # NCP wiring: mask the LSTM's input and recurrent kernels (4 gate blocks).
        self._lstm = (MaskedLSTMCell(self.units) if self.is_masked
                      else tf.keras.layers.LSTMCell(self.units))
        self._lstm.build(input_shape)
        if self.is_masked:
            self._lstm.set_wiring_masks(self.sensory_mask, self.sparsity_mask, 4)
            self._record_mask(self._lstm._kernel_var, self._lstm._k_mask)
            self._record_mask(self._lstm._recurrent_kernel_var,
                              self._lstm._rk_mask)
        self.built = True

    def call(self, inputs, states):
        if isinstance(inputs, (tuple, list)):
            inputs, _ = inputs   # discard elapsed_time (LSTM is discrete)
        output, new_states = self._lstm(inputs, states)
        return output, new_states
