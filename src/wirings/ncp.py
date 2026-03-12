import numpy as np
import tensorflow as tf
from .base_wiring import BaseWiring


class SparseLinear(tf.keras.layers.Layer):
    """Dense layer with a fixed binary connectivity mask.

    The mask determines which connections exist. Weights at masked-off
    positions are zeroed each forward pass (W * mask), so gradients there
    are also zero — the sparsity is permanent throughout training.

    Args:
        units:  output dimension
        mask:   numpy bool/int array of shape (input_dim, units), 1 = connected
    """

    def __init__(self, units, mask, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self._mask_np = np.array(mask, dtype=np.float32)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='W', shape=(input_shape[-1], self.units),
            dtype=tf.float32, initializer='glorot_uniform',
        )
        self.mask = tf.constant(self._mask_np, dtype=tf.float32)
        self.built = True

    def call(self, x):
        return x @ (self.W * self.mask)
