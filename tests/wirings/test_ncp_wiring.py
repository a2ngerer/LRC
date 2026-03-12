import numpy as np
import tensorflow as tf
from src.wirings import SparseLinear


def test_sparse_linear_output_shape():
    mask = np.ones((4, 6), dtype=np.float32)
    layer = SparseLinear(units=6, mask=mask)
    x = tf.zeros([2, 5, 4])
    assert layer(x).shape == (2, 5, 6)


def test_sparse_linear_respects_mask():
    """Weights at mask=0 positions produce zero output."""
    mask = np.zeros((4, 6), dtype=np.float32)
    mask[:, :3] = 1.0   # only first 3 outputs connected
    layer = SparseLinear(units=6, mask=mask)
    x = tf.ones([1, 1, 4])
    out = layer(x).numpy()
    assert np.all(out[..., 3:] == 0.0)
