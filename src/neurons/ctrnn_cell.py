import tensorflow as tf
from .base_cell import BaseCell


class CTRNN_Cell(BaseCell):
    """Continuous-Time RNN cell (leaky integrator ODE).

    ODE: dh/dt = (-h + tanh(W_x·x + W_h·h + b)) / τ
    Euler step: h_new = h + (dt/τ) · (-h + tanh(W_x·x + W_h·h + b))

    Args:
        units:   number of recurrent units
        epsilon: small constant added to τ for numerical stability (default 1e-8)
        ode_unfolds: Euler sub-steps per input step, each over dt/ode_unfolds
                 (default 1 = the original single step). Under NCP wiring a
                 signal moves one synapse per sub-step, so >= 3 are needed for
                 the input to reach the motor neurons within one input step.

    Irregular sampling convention (inherited from BaseCell):
        If inputs is a tuple (x, elapsed_time), elapsed_time is used as dt.
        Otherwise dt defaults to 1.0.
    """

    def __init__(self, units, epsilon=1e-8, ode_unfolds=1, **kwargs):
        super().__init__(units, **kwargs)
        self.epsilon = epsilon
        self.ode_unfolds = ode_unfolds

    def build(self, input_shape):
        if isinstance(input_shape[0], (tuple, tf.TensorShape)):
            input_dim = input_shape[0][-1]
        else:
            input_dim = input_shape[-1]

        self._build_masks(input_dim)

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
        # NCP wiring: W_x is the input->state map, W_h the state->state map.
        x_in = inputs @ self._mask(self.W_x, self.sensory_mask)
        W_h = self._mask(self.W_h, self.sparsity_mask)
        rate = (elapsed_time / self.ode_unfolds) / (self.tau + self.epsilon)
        for _ in range(self.ode_unfolds):
            h = h + rate * (-h + tf.nn.tanh(x_in + h @ W_h + self.b))
        return h, [h]
