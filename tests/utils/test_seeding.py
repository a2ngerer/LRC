import numpy as np
import tensorflow as tf

from src.utils import set_global_seed


def _dense_kernel(seed):
    set_global_seed(seed)
    layer = tf.keras.layers.Dense(3)
    layer.build((None, 4))
    return layer.kernel.numpy()


def test_set_global_seed_overrides_earlier_keras_seed_state():
    """tf.keras.utils.set_random_seed leaves a thread-local Keras generator
    behind that feeds every later initializer; set_global_seed must discard it,
    otherwise two runs with one seed differ once any earlier code seeded Keras."""
    tf.keras.utils.set_random_seed(123)
    np.testing.assert_array_equal(_dense_kernel(7), _dense_kernel(7))
