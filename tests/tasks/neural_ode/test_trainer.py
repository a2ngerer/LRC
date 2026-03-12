import numpy as np
import tensorflow as tf
from src.tasks.neural_ode.datasets import generate_dataset
from src.tasks.neural_ode.ode_model import ODEFuncModel
from src.tasks.neural_ode.solver import euler_odeint
from src.tasks.neural_ode.trainer import get_batch, train


def test_get_batch_shapes():
    """get_batch returns correct shapes for all three outputs."""
    t, y = generate_dataset('spiral', data_size=50)
    y0, t_batch, y_batch = get_batch(t, y, batch_size=4, batch_time=10)
    assert y0.shape == (4, 1, 2), f"Expected (4, 1, 2), got {y0.shape}"
    assert t_batch.shape == (10,), f"Expected (10,), got {t_batch.shape}"
    assert y_batch.shape == (10, 4, 1, 2), f"Expected (10, 4, 1, 2), got {y_batch.shape}"


def test_train_completes_without_error():
    """train runs 5 iterations without raising."""
    t, y = generate_dataset('spiral', data_size=50)
    model = ODEFuncModel('lrc_ar', 4, features=2)
    train(model, t, y, n_iters=5, batch_size=4, batch_time=10)


def test_train_returns_loss_list():
    """train returns a list of floats with one entry per iteration."""
    t, y = generate_dataset('spiral', data_size=50)
    model = ODEFuncModel('lrc_ar', 4, features=2)
    losses = train(model, t, y, n_iters=5, batch_size=4, batch_time=10)
    assert isinstance(losses, list), f"Expected list, got {type(losses)}"
    assert len(losses) == 5, f"Expected 5 losses, got {len(losses)}"
    assert all(isinstance(l, float) for l in losses), "All losses must be float"


def test_train_gradient_flow():
    """GradientTape captures gradients through euler_odeint."""
    t, y = generate_dataset('spiral', data_size=50)
    model = ODEFuncModel('lrc_ar', 4, features=2)

    y0_batch, t_batch, y_batch = get_batch(t, y, batch_size=2, batch_time=5)
    t_tf = tf.constant(t_batch, dtype=tf.float32)
    y0_tf = tf.constant(y0_batch, dtype=tf.float32)
    y_true = tf.constant(y_batch, dtype=tf.float32)

    # Build model before tape so variables exist
    model(tf.constant(0.0), y0_tf)

    with tf.GradientTape() as tape:
        pred = euler_odeint(model, y0_tf, t_tf)
        loss = tf.reduce_mean(tf.abs(pred - y_true))

    grads = tape.gradient(loss, model.trainable_variables)
    assert len(model.trainable_variables) > 0, "Model should have trainable variables"
    assert any(
        g is not None and tf.reduce_any(g != 0.0).numpy()
        for g in grads
    ), "At least one gradient should be non-zero"
