import pytest
import tensorflow as tf
from src.wirings import DenseWiring, BaseWiring
from src.neurons import LRC_Cell


def test_base_wiring_is_abstract():
    """BaseWiring cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseWiring(cell=None)


def test_dense_wiring_is_subclass():
    assert issubclass(DenseWiring, BaseWiring)


def test_dense_wiring_build_model_returns_sequential():
    cell = LRC_Cell(units=4)
    wiring = DenseWiring(cell)
    model = wiring.build_model()
    assert isinstance(model, tf.keras.Sequential)


def test_dense_wiring_return_sequences_true():
    """DenseWiring always produces output for every timestep."""
    cell = LRC_Cell(units=4)
    wiring = DenseWiring(cell)
    model = wiring.build_model()
    x = tf.zeros([2, 5, 3])  # (batch=2, timesteps=5, features=3)
    y = model(x)
    assert y.shape == (2, 5, 4)  # all timesteps returned, output_size = units = 4
