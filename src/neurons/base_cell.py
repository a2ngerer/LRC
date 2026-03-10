import abc
import tensorflow as tf


class BaseCell(tf.keras.layers.AbstractRNNCell):
    """Abstract base class for all RNN cells in this benchmark.

    All neuron types (LRC, STC, LSTM, CT-RNN) must subclass this.

    Subclasses must implement:
        - build(input_shape)
        - call(inputs, states) -> (output, [new_state])

    Irregular sampling convention:
        If inputs is a tuple (x, elapsed_time), the cell should use
        elapsed_time as the integration step dt. Otherwise dt defaults
        to 1 second (regular sampling).
    """

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    @property
    def state_size(self):
        return self.units

    @property
    def output_size(self):
        return self.units

    @abc.abstractmethod
    def build(self, input_shape):
        pass

    @abc.abstractmethod
    def call(self, inputs, states):
        """Process one time step.

        Args:
            inputs: Tensor of shape (batch, input_dim), or tuple
                    (tensor, elapsed_time) for irregularly sampled data.
            states: List of state tensors from the previous step.

        Returns:
            Tuple (output, [new_state]).
        """
        pass

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return tf.zeros([batch_size, self.state_size], dtype=dtype or tf.float32)
