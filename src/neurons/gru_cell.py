import tensorflow as tf
from .base_cell import BaseCell, MaskedGRUCell


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
        # Nested tuple -> first item is the feature tensor shape (same
        # convention as CfC_LRC_Cell.build / LTC_Cell.build). Happens when the
        # cell is fed (x, elapsed_time) for irregularly sampled data; the
        # elapsed_time itself is discarded in call() since GRU is discrete.
        if isinstance(input_shape[0], (tuple, list, tf.TensorShape)):
            input_shape = input_shape[0]
        self._build_masks(input_shape[-1])
        # NCP wiring: mask the GRU's input and recurrent kernels (3 gate blocks).
        self._gru = (MaskedGRUCell(self.units) if self.is_masked
                     else tf.keras.layers.GRUCell(self.units))
        self._gru.build(input_shape)
        if self.is_masked:
            self._gru.set_wiring_masks(self.sensory_mask, self.sparsity_mask, 3)
            self._record_mask(self._gru._kernel_var, self._gru._k_mask)
            self._record_mask(self._gru._recurrent_kernel_var,
                              self._gru._rk_mask)
        self.built = True

    def call(self, inputs, states):
        if isinstance(inputs, (tuple, list)):
            inputs, _ = inputs   # discard elapsed_time (GRU is discrete)
        output, new_states = self._gru(inputs, states)
        if not isinstance(new_states, (tuple, list)):
            new_states = [new_states]
        return output, new_states
