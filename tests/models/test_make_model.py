import pytest
import tensorflow as tf
from src.models import make_model
from src.neurons import LRC_Cell
from src.wirings import DenseWiring


def test_make_model_returns_sequential():
    model = make_model('lrc', 'dense', 32)
    assert isinstance(model, tf.keras.Sequential)


def test_make_model_forward_pass():
    """LRC cell accepts any input feature dimension."""
    model = make_model('lrc', 'dense', 4)
    x = tf.zeros([2, 5, 3])  # (batch=2, timesteps=5, features=3)
    y = model(x)
    assert y.shape == (2, 5, 4)


def test_make_model_accepts_class_arguments():
    """make_model also accepts cell/wiring classes directly, not just string keys."""
    model = make_model(LRC_Cell, DenseWiring, 8)
    assert isinstance(model, tf.keras.Sequential)


def test_make_model_unknown_neuron_raises():
    with pytest.raises(KeyError):
        make_model('unknown_cell', 'dense', 4)


def test_make_model_lrc_ar_forward_pass():
    """LRC_AR cell: input features must equal units (input acts as autoregressive state)."""
    model = make_model('lrc_ar', 'dense', 4)
    x = tf.zeros([1, 3, 4])  # features=4 == units=4
    y = model(x)
    assert y.shape == (1, 3, 4)


def test_make_model_summary_runs():
    model = make_model('lrc', 'dense', 64)
    # Build the model first by calling it on dummy data
    model(tf.zeros([1, 1, 10]))
    model.summary()  # must not raise


def test_make_model_kwargs_forwarding():
    """Cell constructor kwargs are forwarded through make_model."""
    model = make_model('lrc', 'dense', 4, ode_unfolds=5)
    assert isinstance(model, tf.keras.Sequential)
