# Follows the ODE-LSTM (mixed memory) pattern by Lechner & Hasani (2020),
# "Learning Long-Term Dependencies in Irregularly-Sampled Time Series",
# arXiv:2006.04418, reference implementation
# https://github.com/mlech26l/ode-lstms (ODELSTMCell): an LSTM owns the
# memory path c (additive update -> constant error propagation), while a
# continuous-time cell evolves the hidden state h. Here the inner ODE cell
# is pluggable (LTC or LRC) instead of a fixed CT-RNN.

import tensorflow as tf
from .base_cell import BaseCell
from .ltc_cell import LTC_Cell
from .lrc_cell import LRC_Cell


class MixedMemoryCell(BaseCell):
    """Mixed-memory wrapper: LSTM gating + inner continuous-time cell.

    Step (ODE-LSTM pattern):
      1. LSTM gates compute candidate hidden state h_cand and new memory c
         from (x, [h, c]).
      2. The inner ODE cell evolves h_cand with input x over elapsed_time;
         its state slot is h_cand, so the ODE state IS the hidden state.
      3. Output is the inner cell's (mapped) output; recurrent state is
         [evolved h, new c].

    Gradients along the memory path c never pass through the ODE solver's
    Jacobians -- this is the architectural fix for the vanishing/exploding
    gradient of BPTT through the ODE (Lechner & Hasani 2020, Theorem 1/2).

    State: [h, c], both (batch, units).
    """

    def __init__(self, units, inner_cell_cls, **inner_kwargs):
        super().__init__(units)
        self._inner_cell_cls = inner_cell_cls
        self._inner_kwargs = inner_kwargs

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
        # Nested tuple (irregular sampling) -> first item is the feature
        # tensor shape; the LSTM only ever sees the feature part.
        if isinstance(input_shape[0], (tuple, tf.TensorShape)):
            feature_shape = input_shape[0]
        else:
            feature_shape = input_shape
        self._lstm = tf.keras.layers.LSTMCell(self.units)
        self._lstm.build(feature_shape)
        self._inner = self._inner_cell_cls(units=self.units, **self._inner_kwargs)
        self._inner.build(input_shape)
        self.built = True

    def call(self, inputs, states):
        if isinstance(inputs, (tuple, list)):
            x, elapsed_time = inputs
            inner_inputs = (x, elapsed_time)
        else:
            x = inputs
            inner_inputs = inputs
        h, c = states

        _, lstm_states = self._lstm(x, [h, c])
        h_cand, new_c = lstm_states

        output, inner_states = self._inner(inner_inputs, [h_cand])
        new_h = inner_states[0]
        return output, [new_h, new_c]


class MM_LTC_Cell(MixedMemoryCell):
    """Mixed-memory LTC: LSTM memory path + LTC continuous-time dynamics."""

    def __init__(self, units, **kwargs):
        super().__init__(units, LTC_Cell, **kwargs)


class MM_LRC_Cell(MixedMemoryCell):
    """Mixed-memory LRC: LSTM memory path + LRC continuous-time dynamics."""

    def __init__(self, units, **kwargs):
        super().__init__(units, LRC_Cell, **kwargs)
