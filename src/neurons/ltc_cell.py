# Follows the LTC implementation by Mathias Lechner and Ramin Hasani (2022) https://github.com/mlech26l/ncps/blob/master/ncps/tf/ltc_cell.py

import tensorflow as tf
from .base_cell import BaseCell


def _erev_initializer(shape, dtype=None):
    """Random reversal potentials in {-1, +1} (excitatory/inhibitory synapses)."""
    return tf.sign(tf.random.uniform(shape, -1.0, 1.0, dtype=dtype or tf.float32))


class LTC_Cell(BaseCell):
    def __init__(
        self,
        units,
        input_mapping="affine",
        output_mapping="affine",
        ode_unfolds=6,
        dt=1,
        epsilon=1e-8,
        initialization_ranges=None,
        **kwargs
    ):
        """
            Liquid Time-Constant (LTC) <https://arxiv.org/abs/2006.04439> cell.

            Conductance-based ODE neuron with sigmoidal chemical synapses and
            input-dependent time constants. Integrated with the fused
            (semi-implicit) Euler scheme from the original implementation.

            This is an AbstractRNNCell that processes single time-steps.
            To get a full RNN that can process sequences, it needs to be wrapped
            by a tf.keras.layers.RNN <https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN>.

             >>> cell = LTC_Cell(units)
             >>> rnn = tf.keras.layers.RNN(cell)

        """

        super().__init__(units, **kwargs)
        self._init_ranges = {
            "gleak": (0.001, 1.0),
            "vleak": (-0.2, 0.2),
            "cm": (0.4, 0.6),
            "w": (0.001, 1.0),
            "sigma": (3, 8),
            "mu": (0.3, 0.8),
            "sensory_w": (0.001, 1.0),
            "sensory_sigma": (3, 8),
            "sensory_mu": (0.3, 0.8),
        }
        if initialization_ranges is not None:
            for k, v in initialization_ranges.items():
                if k not in self._init_ranges.keys():
                    raise ValueError(
                        "Unknown parameter '{}' in initialization range dictionary! (Expected only {})".format(
                            k, str(list(self._init_ranges.keys()))
                        )
                    )
                if k in ["gleak", "cm", "w", "sensory_w"] and v[0] < 0:
                    raise ValueError(
                        "Initialization range of parameter '{}' must be non-negative!".format(
                            k
                        )
                    )
                if v[0] > v[1]:
                    raise ValueError(
                        "Initialization range of parameter '{}' is not a valid range".format(
                            k
                        )
                    )
                self._init_ranges[k] = v

        self._input_mapping = input_mapping
        self._output_mapping = output_mapping
        self._ode_unfolds = ode_unfolds
        self._dt = dt
        self._epsilon = epsilon

    @property
    def sensory_size(self):
        return self.input_dim

    def _get_initializer(self, param_name):
        minval, maxval = self._init_ranges[param_name]
        if minval == maxval:
            return tf.keras.initializers.Constant(minval)
        else:
            return tf.keras.initializers.RandomUniform(minval, maxval)

    def build(self, input_shape):
        # Check if input_shape is nested tuple/list
        if isinstance(input_shape[0], tuple) or isinstance(
            input_shape[0], tf.TensorShape
        ):
            # Nested tuple -> First item represent feature dimension
            input_dim = input_shape[0][-1]
        else:
            input_dim = input_shape[-1]

        self.input_dim = input_dim

        self._params = {}
        self._params["gleak"] = self.add_weight(
            name="gleak",
            shape=(self.state_size,),
            dtype=tf.float32,
            constraint=tf.keras.constraints.NonNeg(),
            initializer=self._get_initializer("gleak"),
        )
        self._params["vleak"] = self.add_weight(
            name="vleak",
            shape=(self.state_size,),
            dtype=tf.float32,
            initializer=self._get_initializer("vleak"),
        )
        self._params["cm"] = self.add_weight(
            name="cm",
            shape=(self.state_size,),
            dtype=tf.float32,
            constraint=tf.keras.constraints.NonNeg(),
            initializer=self._get_initializer("cm"),
        )
        self._params["sigma"] = self.add_weight(
            name="sigma",
            shape=(self.state_size, self.state_size),
            dtype=tf.float32,
            initializer=self._get_initializer("sigma"),
        )
        self._params["mu"] = self.add_weight(
            name="mu",
            shape=(self.state_size, self.state_size),
            dtype=tf.float32,
            initializer=self._get_initializer("mu"),
        )
        self._params["w"] = self.add_weight(
            name="w",
            shape=(self.state_size, self.state_size),
            dtype=tf.float32,
            constraint=tf.keras.constraints.NonNeg(),
            initializer=self._get_initializer("w"),
        )
        self._params["erev"] = self.add_weight(
            name="erev",
            shape=(self.state_size, self.state_size),
            dtype=tf.float32,
            initializer=_erev_initializer,
        )
        self._params["sensory_sigma"] = self.add_weight(
            name="sensory_sigma",
            shape=(self.sensory_size, self.state_size),
            dtype=tf.float32,
            initializer=self._get_initializer("sensory_sigma"),
        )
        self._params["sensory_mu"] = self.add_weight(
            name="sensory_mu",
            shape=(self.sensory_size, self.state_size),
            dtype=tf.float32,
            initializer=self._get_initializer("sensory_mu"),
        )
        self._params["sensory_w"] = self.add_weight(
            name="sensory_w",
            shape=(self.sensory_size, self.state_size),
            dtype=tf.float32,
            constraint=tf.keras.constraints.NonNeg(),
            initializer=self._get_initializer("sensory_w"),
        )
        self._params["sensory_erev"] = self.add_weight(
            name="sensory_erev",
            shape=(self.sensory_size, self.state_size),
            dtype=tf.float32,
            initializer=_erev_initializer,
        )

        if self._input_mapping in ["affine", "linear"]:
            self._params["input_w"] = self.add_weight(
                name="input_w",
                shape=(self.sensory_size,),
                dtype=tf.float32,
                initializer=tf.keras.initializers.Constant(1),
            )
        if self._input_mapping == "affine":
            self._params["input_b"] = self.add_weight(
                name="input_b",
                shape=(self.sensory_size,),
                dtype=tf.float32,
                initializer=tf.keras.initializers.Constant(0),
            )

        if self._output_mapping in ["affine", "linear"]:
            self._params["output_w"] = self.add_weight(
                name="output_w",
                shape=(self.state_size,),
                dtype=tf.float32,
                initializer=tf.keras.initializers.Constant(1),
            )
        if self._output_mapping == "affine":
            self._params["output_b"] = self.add_weight(
                name="output_b",
                shape=(self.state_size,),
                dtype=tf.float32,
                initializer=tf.keras.initializers.Constant(0),
            )

        self.built = True

    def _sigmoid(self, v_pre, mu, sigma):
        v_pre = tf.expand_dims(v_pre, axis=-1)  # For broadcasting
        mues = v_pre - mu
        x = sigma * mues
        return tf.nn.sigmoid(x)

    def _ode_solver(self, inputs, state, elapsed_time):
        v_pre = state

        # Pre-compute the effects of the sensory neurons (constant per step)
        sensory_w_activation = self._params["sensory_w"] * self._sigmoid(
            inputs, self._params["sensory_mu"], self._params["sensory_sigma"]
        )
        sensory_rev_activation = sensory_w_activation * self._params["sensory_erev"]

        # Reduce over dimension 1 (=source sensory neurons)
        w_numerator_sensory = tf.reduce_sum(sensory_rev_activation, axis=1)
        w_denominator_sensory = tf.reduce_sum(sensory_w_activation, axis=1)

        # cm/t is loop invariant, can be precomputed
        cm_t = self._params["cm"] / (elapsed_time / self._ode_unfolds)

        # Unfold the multiply ODE multiple times into one RNN step
        for t in range(self._ode_unfolds):
            w_activation = self._params["w"] * self._sigmoid(
                v_pre, self._params["mu"], self._params["sigma"]
            )
            rev_activation = w_activation * self._params["erev"]

            # Reduce over dimension 1 (=source neurons)
            w_numerator = tf.reduce_sum(rev_activation, axis=1) + w_numerator_sensory
            w_denominator = tf.reduce_sum(w_activation, axis=1) + w_denominator_sensory

            numerator = cm_t * v_pre + self._params["gleak"] * self._params["vleak"] + w_numerator
            denominator = cm_t + self._params["gleak"] + w_denominator

            # Avoid dividing by 0
            v_pre = numerator / (denominator + self._epsilon)

        return v_pre

    def _map_inputs(self, inputs):
        if self._input_mapping in ["affine", "linear"]:
            inputs = inputs * self._params["input_w"]
        if self._input_mapping == "affine":
            inputs = inputs + self._params["input_b"]
        return inputs

    def _map_outputs(self, state):
        output = state
        if self._output_mapping in ["affine", "linear"]:
            output = output * self._params["output_w"]
        if self._output_mapping == "affine":
            output = output + self._params["output_b"]
        return output

    def call(self, inputs, states):
        if isinstance(inputs, (tuple, list)):
            # Irregularly sampled mode
            inputs, elapsed_time = inputs
        else:
            # Regularly sampled mode (elapsed time = 1 second)
            elapsed_time = self._dt
        inputs = self._map_inputs(inputs)

        next_state = self._ode_solver(inputs, states[0], elapsed_time)

        outputs = self._map_outputs(next_state)

        return outputs, [next_state]
