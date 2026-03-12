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
from src.neurons import LRC_Cell, LSTM_Cell, CTRNN_Cell
from src.wirings import NCPWiring


def test_ncp_wiring_build_model_returns_sequential():
    wiring = NCPWiring(LRC_Cell, inter_neurons=8, command_neurons=6, motor_neurons=4)
    model = wiring.build_model()
    assert isinstance(model, tf.keras.Sequential)


def test_ncp_wiring_output_shape_lstm():
    wiring = NCPWiring(LSTM_Cell, inter_neurons=8, command_neurons=6, motor_neurons=4)
    model = wiring.build_model()
    x = tf.zeros([2, 5, 3])
    assert model(x).shape == (2, 5, 4)


def test_ncp_wiring_output_shape_ctrnn():
    wiring = NCPWiring(CTRNN_Cell, inter_neurons=8, command_neurons=6, motor_neurons=4)
    model = wiring.build_model()
    x = tf.zeros([2, 5, 3])
    assert model(x).shape == (2, 5, 4)
