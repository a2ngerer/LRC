import abc
import numpy as np
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

    Sparse (NCP) wiring:
        ``sparsity_mask`` (units, units) and ``sensory_mask`` (input_dim,
        units) are optional fixed binary masks, following the ncps reference
        implementation (``ncps/keras/ltc_cell.py``): the recurrent synapse
        weights are multiplied by ``sparsity_mask`` and the input synapse
        weights by ``sensory_mask`` on every forward pass, so masked-off
        weights never influence the output and their gradients are exactly
        zero. Both default to ``None`` == dense, in which case every cell
        behaves byte-identically to before the masks existed.

        ``sensory_mask`` may be a callable ``input_dim -> array``, because the
        input dimension is only known at build time; subclasses materialize
        the constants by calling ``self._build_masks(input_dim)`` from
        ``build()``.
    """

    def __init__(self, units, sparsity_mask=None, sensory_mask=None, **kwargs):
        if type(self) is BaseCell:
            raise TypeError(
                "BaseCell is abstract and cannot be instantiated directly. "
                "Subclass it and implement build() and call()."
            )
        super().__init__(**kwargs)
        self.units = units
        # Mask *sources* (numpy / callable); the tf constants are created in
        # _build_masks() once the input dimension is known.
        self._sparsity_mask_src = sparsity_mask
        self._sensory_mask_src = sensory_mask
        self.sparsity_mask = None      # (units, units) tf constant or None
        self.sensory_mask = None       # (input_dim, units) tf constant or None
        self.concat_mask = None        # (input_dim + units, units) or None
        # variable -> mask, filled while masks are applied; read by
        # src.wirings.ncp.effective_param_count.
        self._masked_vars = {}

    @property
    def is_masked(self):
        """True if this cell runs under a sparse (NCP) wiring."""
        return self._sparsity_mask_src is not None

    @property
    def state_size(self):
        return self.units

    @property
    def output_size(self):
        return self.units

    # --- sparse-wiring helpers ------------------------------------------- #
    def _build_masks(self, input_dim):
        """Materialize the wiring masks as constants. No-op when unmasked."""
        if not self.is_masked:
            return
        adj = self._resolve(self._sparsity_mask_src, input_dim)
        if adj.shape != (self.units, self.units):
            raise ValueError(
                f"sparsity_mask must have shape ({self.units}, {self.units}), "
                f"got {adj.shape}")
        sens = self._resolve(self._sensory_mask_src, input_dim)
        if sens.shape != (input_dim, self.units):
            raise ValueError(
                f"sensory_mask must have shape ({input_dim}, {self.units}), "
                f"got {sens.shape}")
        self.sparsity_mask = tf.constant(adj)
        self.sensory_mask = tf.constant(sens)
        self.concat_mask = tf.constant(np.concatenate([sens, adj], axis=0))

    @staticmethod
    def _resolve(mask_src, input_dim):
        """Mask source (array or callable(input_dim)) -> float32 array."""
        return np.asarray(
            mask_src(input_dim) if callable(mask_src) else mask_src,
            dtype=np.float32)

    def _record_mask(self, weight, mask):
        """Remember that ``weight`` is masked (for effective_param_count)."""
        if isinstance(weight, tf.Variable):
            self._masked_vars[weight.ref()] = mask

    def masked_off_count(self):
        """Number of trainable weight entries switched off by the wiring.

        Requires that the cell has run at least one forward pass, because some
        masks are applied inside call(). Returns 0 for unmasked cells.
        """
        if not self.is_masked:
            return 0
        if not self._masked_vars:
            raise ValueError(
                f"{type(self).__name__} is masked but no masked weights were "
                "recorded yet -- run one forward pass before counting params.")
        return int(sum(
            np.count_nonzero(np.asarray(mask) == 0.0) *
            (1 if ref.deref().trainable else 0)
            for ref, mask in self._masked_vars.items()))

    def _mask(self, weight, mask):
        """weight * mask, or weight unchanged when mask is None."""
        if mask is None:
            return weight
        self._record_mask(weight, mask)
        return weight * mask

    def _masked_dense(self, layer, x):
        """Apply a Dense layer whose input is concat([inputs, state]).

        With a wiring mask in place the kernel is multiplied by
        concat([sensory_mask; sparsity_mask]) first, so the map obeys the NCP
        adjacency. Without a mask this is exactly ``layer(x)``.
        """
        if self.concat_mask is None:
            return layer(x)
        self._track(layer, x)
        self._record_mask(layer.kernel, self.concat_mask)
        y = tf.matmul(x, layer.kernel * self.concat_mask)
        if layer.use_bias:
            y = tf.nn.bias_add(y, layer.bias)
        return layer.activation(y) if layer.activation is not None else y

    @staticmethod
    def _track(layer, x):
        """Register a sub-layer's weights with this cell (once).

        Keras only picks up a sub-layer created in ``build()`` when it is
        __call__-ed from the parent's ``call()``; the masked path does a plain
        matmul instead, so its weights would otherwise be neither tracked nor
        trained. The throwaway call's output is discarded, so it contributes
        no gradient.
        """
        if not getattr(layer, '_ncp_tracked', False):
            layer(x)
            layer._ncp_tracked = True

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


class _MaskedKernelsMixin:
    """Mixin applying fixed wiring masks to a Keras GRU/LSTM cell.

    ``kernel`` / ``recurrent_kernel`` become read-through properties that
    return the masked tensor, while the underlying trainable Variables stay
    tracked (the setter is what ``build()`` writes to). The gate blocks are
    laid out as ``units`` columns per gate, so the (input_dim, units) /
    (units, units) masks are tiled ``n_gates`` times along the column axis.
    """

    _k_mask = None
    _rk_mask = None

    def set_wiring_masks(self, sensory_mask, sparsity_mask, n_gates):
        self._k_mask = tf.constant(np.tile(
            np.asarray(sensory_mask, np.float32), (1, n_gates)))
        self._rk_mask = tf.constant(np.tile(
            np.asarray(sparsity_mask, np.float32), (1, n_gates)))

    @property
    def kernel(self):
        k = self._kernel_var
        return k if self._k_mask is None else k * self._k_mask

    @kernel.setter
    def kernel(self, value):
        self._kernel_var = value

    @property
    def recurrent_kernel(self):
        k = self._recurrent_kernel_var
        return k if self._rk_mask is None else k * self._rk_mask

    @recurrent_kernel.setter
    def recurrent_kernel(self, value):
        self._recurrent_kernel_var = value


class MaskedGRUCell(_MaskedKernelsMixin, tf.keras.layers.GRUCell):
    """GRUCell with NCP wiring masks on its input/recurrent kernels."""


class MaskedLSTMCell(_MaskedKernelsMixin, tf.keras.layers.LSTMCell):
    """LSTMCell with NCP wiring masks on its input/recurrent kernels."""
