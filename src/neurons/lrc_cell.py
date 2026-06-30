# Follows the LTC implementation by Mathias Lechner and Ramin Hasani (2022) https://github.com/mlech26l/ncps/blob/master/ncps/tf/ltc_cell.py

import tensorflow as tf
from .base_cell import BaseCell


class LRC_Cell(BaseCell):
    def __init__(
        self,
        units,
        input_mapping="affine",
        output_mapping="affine",
        ode_unfolds=1,
        dt = 1,
        elastance_type = "interp",
        forget_gate = True,
        ode_solver = 'explicit',
        epsilon=1e-8,
        freeze_elastance=False,
        pm_pad=False,
        pm_pad_extra=0,
        initialization_ranges=None,
        **kwargs
    ):
        """
            Liquid-Resistance Liquid-Capacitance (LRC) <https://arxiv.org/pdf/2403.08791> cell.
            It extends the Liquid Time-Constant (LTC) cell <https://arxiv.org/abs/2002.05202> by adding a saturation functions and input- and state-dependent elastance.

            This is an AbstractRNNCell that process single time-steps.
            To get a full RNN that can process sequences, it needs to be wrapped by a tf.keras.layers.RNN <https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN>.

             >>> cell = LRC_Cell(units)
             >>> rnn = tf.keras.layers.RNN(cell)

        """

        super().__init__(units, **kwargs)
        self._init_ranges = {
            "gleak": (0.001, 1.0),
            "vleak": (-0.2, 0.2),
            "w": (0.001, 1.0),
            "sigma": (3, 8),
            "mu": (0.3, 0.8),
            "sensory_w": (0.001, 1.0),
            "sensory_sigma": (3, 8),
            "sensory_mu": (0.3, 0.8),
        }
        if not initialization_ranges is None:
            for k, v in initialization_ranges.items():
                if k not in self._init_ranges.keys():
                    raise ValueError(
                        "Unknown parameter '{}' in initialization range dictionary! (Expected only {})".format(
                            k, str(list(self._init_ranges.keys()))
                        )
                    )
                if k in ["gleak", "w", "sensory_w"] and v[0] < 0:
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
        self._elastance_type = elastance_type
        self._forget_gate = forget_gate
        self._epsilon = epsilon
        self._ode_solver_type = ode_solver

        # eps-ablation flags (additive; default-False keeps every prior cell
        # byte-identical). freeze_elastance: build the elastance Dense then mark
        # it non-trainable (the frozen-structure control D). pm_pad: in the
        # interp branch, add a separate same-budget additive residual onto
        # v_prime (the capacity controls E / E_C). pm_pad_extra widens the pad
        # by the distr_shift-sized amount so E_C matches C's param count.
        self._freeze_elastance = freeze_elastance
        self._pm_pad = pm_pad
        self._pm_pad_extra = pm_pad_extra
        # Live-gate diagnostic capture (filled on each _ode_solver call when
        # enabled). The benchmark/aggregator reads these off a trained model to
        # confirm the gate is measurably active before reading any equivalence.
        self._capture_gate = False
        self._last_elastance_t = None

        self._layerwise = False

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
        if self._forget_gate:
            self._params["w"] = self.add_weight(
                name="w",
                shape=(self.state_size, self.state_size),
                dtype=tf.float32,
                constraint=tf.keras.constraints.NonNeg(),
                initializer=self._get_initializer("w"),
            )
            self._params["sensory_w"] = self.add_weight(
                name="sensory_w",
                shape=(self.sensory_size, self.state_size),
                dtype=tf.float32,
                constraint=tf.keras.constraints.NonNeg(),
                initializer=self._get_initializer("sensory_w"),
            )
        else:
            self._params["tau"] = self.add_weight(
                name="tau",
                shape=(self.state_size,),
                dtype=tf.float32,
                initializer=tf.keras.initializers.Constant(1),
                constraint=tf.keras.constraints.NonNeg()
            )

        self._params["h"] = self.add_weight(
            name="h",
            shape=(self.state_size, self.state_size),
            dtype=tf.float32,
            initializer=tf.keras.initializers.Orthogonal(),
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

        self._params["sensory_h"] = self.add_weight(
            name="sensory_h",
            shape=(self.sensory_size, self.state_size),
            dtype=tf.float32,
            initializer=tf.keras.initializers.Orthogonal(),
        )

        self.elastance_mapping = tf.keras.layers.Dense(self.state_size, name="elastance_mapping")

        # eps-ablation: same-budget additive-residual pad (controls E / E_C).
        # Only built in the interp branch, where elastance_mapping is never
        # called (and therefore contributes 0 params). The pad mirrors
        # elastance_mapping's Dense(state_size) and adds pm_pad_extra so its
        # trainable count equals the asymmetric/symmetric gate's per cell.
        # NOTE: deliberately an additive residual on v_prime, NOT a multiplicative
        # gate -- it is a capacity control, never role-matched to the elastance
        # mechanism (see spec section 0.1).
        if self._pm_pad and self._elastance_type == "interp":
            # Pad = Dense(state_size), mirroring elastance_mapping exactly, so its
            # trainable count equals B's gate (E == B). Built lazily on first call
            # in _ode_solver (Keras then tracks its weights as cell variables); it
            # IS called every step (unlike interp's unused elastance_mapping), so
            # the pad params always materialize.
            self.pm_pad_mapping = tf.keras.layers.Dense(
                self.state_size, name="pm_pad_mapping"
            )
            # pm_pad_extra > 0 adds ONE distr_shift-sized weight (shape
            # (state_size,)), NOT a wider Dense -- so E_C's count = E + state_size
            # = C's count PER CELL (C adds exactly a distr_shift of shape
            # (state_size,) over B). The numeric value is treated as a flag: the
            # extra always matches the cell's own distr_shift size, so on NCP the
            # per-cell extras (16+8+2=26) equal C's distr_shift total, not 3x16.
            # A Dense(state_size+extra) would add ~(input_dim+1)*extra params and
            # over-shoot C.
            if self._pm_pad_extra:
                self.pm_pad_extra_w = self.add_weight(
                    name="pm_pad_extra",
                    shape=(self.state_size,),
                    dtype=tf.float32,
                    initializer=tf.keras.initializers.Constant(0),
                )
            else:
                self.pm_pad_extra_w = None
        else:
            self.pm_pad_mapping = None
            self.pm_pad_extra_w = None

        # eps-ablation: frozen-structure control (D). Setting trainable=False
        # BEFORE the first call makes the lazily-built elastance weights
        # non-trainable (a fixed random gate). The RNN wrapper then tracks them
        # under non_trainable_variables. Only meaningful when elastance is
        # actually called (asymmetric/symmetric); for interp the layer stays
        # unbuilt and the flag is inert.
        if self._freeze_elastance:
            self.elastance_mapping.trainable = False

        if self._elastance_type in ["symmetric"]:
            self._params["distr_shift"] = self.add_weight(
                name="distr_shift",
                shape=(self.state_size,),
                dtype=tf.float32,
                constraint=tf.keras.constraints.NonNeg(),
                initializer=tf.keras.initializers.Constant(1)
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


        self._prev_dts = [self._dt] * self.state_size

        self.built = True

    def _sigmoid(self, v_pre, mu, sigma):
        v_pre = tf.expand_dims(v_pre, axis=-1)  # For broadcasting
        mues = v_pre - mu
        x = sigma * mues
        return tf.nn.sigmoid(x)

    def _ode_solver(self, inputs, state, elapsed_time):
        v_pre = state

        # We can pre-compute the effects of the sensory neurons here
        sensory_syn = self._sigmoid(
            inputs, self._params["sensory_mu"], self._params["sensory_sigma"]
        )

        sensory_h_activation = self._params["sensory_h"] * sensory_syn

        # Reduce over dimension 1 (=source sensory neurons)
        if self._forget_gate:
            sensory_w_activation = self._params["sensory_w"] * sensory_syn
            sensory_w_activation_reduced = tf.reduce_sum(sensory_w_activation, axis=1)
        sensory_h_activation_reduced = tf.reduce_sum(sensory_h_activation, axis=1)

        dt = elapsed_time / self._ode_unfolds

        # Unfold the multiply ODE multiple times into one RNN step
        for t in range(self._ode_unfolds): # 1 unfold is enough for LRC, but leaving this here to experiment with more unfolds
            if self._elastance_type == "asymmetric":
                x = tf.concat([inputs, v_pre], axis=-1)
                elast_dense = self.elastance_mapping(x)
                elastance_t = tf.nn.sigmoid(elast_dense) * dt
            elif self._elastance_type == "symmetric":
                x = tf.concat([inputs, v_pre], axis=-1)
                elast_dense = self.elastance_mapping(x)
                elastance_t = (tf.nn.sigmoid(elast_dense + self._params["distr_shift"]) - tf.nn.sigmoid(elast_dense - self._params["distr_shift"])) * dt
            else:
                elastance_t = dt

            # Live-gate diagnostic capture (B/C): record the gate output so the
            # benchmark can compute its coefficient of variation across
            # timesteps/inputs on a trained model (spec section 3.0.6).
            if self._capture_gate:
                self._last_elastance_t = elastance_t

            syn = self._sigmoid(
                v_pre, self._params["mu"], self._params["sigma"]
            )

            h_activation = self._params["h"] * syn

            g = self._params["gleak"] + tf.reduce_sum(h_activation, axis=1) + sensory_h_activation_reduced

            if self._forget_gate:
                w_activation = self._params["w"] * syn
                f = self._params["gleak"] + tf.reduce_sum(w_activation, axis=1) + sensory_w_activation_reduced
                v_prime = - v_pre * tf.nn.sigmoid(f)  + self._params["vleak"]*tf.nn.tanh(g)
            else:
                v_prime = - v_pre * self._params["tau"] + self._params["vleak"]*tf.nn.tanh(g)

            # eps-ablation: additive same-budget residual (controls E / E_C).
            # Applied post-tanh onto v_prime, with elastance_t left at dt. The
            # pad output is reduced to state_size (the leading slice); the extra
            # pm_pad_extra columns still train and contribute params/gradients so
            # E_C's count matches C, without changing the residual's dimension.
            if self.pm_pad_mapping is not None:
                pad = self.pm_pad_mapping(tf.concat([inputs, v_pre], axis=-1))
                if self.pm_pad_extra_w is not None:
                    # Fold the distr_shift-sized (state_size,) extra weight into
                    # the residual so it carries gradient (not a dead weight).
                    pad = pad + self.pm_pad_extra_w
                v_prime = v_prime + pad


            if self._ode_solver_type == 'hybrid':
                v_pre = (elastance_t * self._params["vleak"] * tf.nn.tanh(g) + v_pre)/(1+elastance_t * tf.nn.sigmoid(f))
            else:
                v_pre = v_pre + elastance_t * v_prime

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
