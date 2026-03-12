import tensorflow as tf
from .base_cell import BaseCell


class CTRNN_Cell(BaseCell):
    """Continuous-Time RNN cell (leaky integrator ODE).

    ODE: dh/dt = (-h + tanh(W_x·x + W_h·h + b)) / τ
    Euler step: h_new = h + (dt/τ) · (-h + tanh(W_x·x + W_h·h + b))

    Args:
        units:   number of recurrent units
        epsilon: small constant added to τ for numerical stability (default 1e-8)

    Irregular sampling convention (inherited from BaseCell):
        If inputs is a tuple (x, elapsed_time), elapsed_time is used as dt.
        Otherwise dt defaults to 1.0.
    """

    def __init__(self, units, epsilon=1e-8, **kwargs):
        super().__init__(units, **kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        if isinstance(input_shape[0], (tuple, tf.TensorShape)):
            input_dim = input_shape[0][-1]
        else:
            input_dim = input_shape[-1]

        self.W_x = self.add_weight(
            name='W_x', shape=(input_dim, self.units),
            dtype=tf.float32, initializer='glorot_uniform',
        )
        self.W_h = self.add_weight(
            name='W_h', shape=(self.units, self.units),
            dtype=tf.float32, initializer='orthogonal',
        )
        self.b = self.add_weight(
            name='b', shape=(self.units,),
            dtype=tf.float32, initializer='zeros',
        )
        self.tau = self.add_weight(
            name='tau', shape=(self.units,),
            dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(1.0),
            constraint=tf.keras.constraints.NonNeg(),
        )
        self.built = True

    def call(self, inputs, states):
        if isinstance(inputs, (tuple, list)):
            inputs, elapsed_time = inputs
        else:
            elapsed_time = 1.0
        h = states[0]
        gate = tf.nn.tanh(inputs @ self.W_x + h @ self.W_h + self.b)
        h_new = h + (elapsed_time / (self.tau + self.epsilon)) * (-h + gate)
        return h_new, [h_new]
