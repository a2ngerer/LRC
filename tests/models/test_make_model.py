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


# --- make_dense_model ---

def test_make_dense_model_single_layer():
    from src.models import make_dense_model
    model = make_dense_model('lrc', units=4)
    assert isinstance(model, tf.keras.Sequential)
    assert model(tf.zeros([2, 5, 3])).shape == (2, 5, 4)


def test_make_dense_model_multi_layer():
    from src.models import make_dense_model
    model = make_dense_model('ctrnn', units=4, num_layers=3)
    x = tf.zeros([2, 5, 3])
    assert model(x).shape == (2, 5, 4)


def test_make_dense_model_output_neurons():
    from src.models import make_dense_model
    model = make_dense_model('lstm', units=8, num_layers=2, output_neurons=2)
    x = tf.zeros([2, 5, 3])
    assert model(x).shape == (2, 5, 2)


# --- make_ncp_model ---

def test_make_ncp_model_returns_sequential():
    from src.models import make_ncp_model
    model = make_ncp_model('lrc', inter_neurons=8, command_neurons=6, motor_neurons=4)
    assert isinstance(model, tf.keras.Sequential)


def test_make_ncp_model_output_shape():
    from src.models import make_ncp_model
    model = make_ncp_model('lstm', inter_neurons=8, command_neurons=6, motor_neurons=4)
    x = tf.zeros([2, 5, 3])
    assert model(x).shape == (2, 5, 4)


def test_make_ncp_model_gradient_flow():
    from src.models import make_ncp_model
    model = make_ncp_model('ctrnn', inter_neurons=8, command_neurons=6, motor_neurons=4)
    x = tf.zeros([2, 5, 3])
    model(x)  # build weights
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(model(x))
    grads = tape.gradient(loss, model.trainable_variables)
    assert len(model.trainable_variables) > 0
    assert any(g is not None and tf.reduce_any(g != 0).numpy() for g in grads)
