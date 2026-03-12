import tensorflow as tf
from src.tasks.neural_ode.ode_model import ODEFuncModel


def test_ode_model_forward_pass_shape():
    """ODEFuncModel maps (batch, 1, features) → (batch, 1, features)."""
    model = ODEFuncModel('lrc_ar', 'dense', 4, features=2)
    t = tf.constant(0.0)
    state = tf.zeros([1, 1, 2])
    output = model(t, state)
    assert output.shape == (1, 1, 2)


def test_ode_model_batch_size_independence():
    """Output shape scales correctly with batch size."""
    model = ODEFuncModel('lrc_ar', 'dense', 4, features=2)
    t = tf.constant(0.0)
    for batch in [1, 3]:
        state = tf.zeros([batch, 1, 2])
        output = model(t, state)
        assert output.shape == (batch, 1, 2), f"batch={batch}: expected ({batch}, 1, 2), got {output.shape}"
