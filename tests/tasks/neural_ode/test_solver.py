import numpy as np
import tensorflow as tf
from src.tasks.neural_ode.solver import euler_odeint


def test_euler_odeint_zero_ode():
    """dy/dt = 0 → state is constant at all time steps."""
    func = lambda t, y: tf.zeros_like(y)
    y0 = tf.constant([[[1.0, 2.0]]])  # (1, 1, 2)
    t = tf.constant([0.0, 1.0, 2.0])  # T=3

    result = euler_odeint(func, y0, t)

    assert result.shape == (3, 1, 1, 2), f"Expected (3, 1, 1, 2), got {result.shape}"
    for step in range(3):
        np.testing.assert_allclose(
            result[step].numpy(), y0.numpy(),
            err_msg=f"Step {step}: state changed but dy/dt=0"
        )


def test_euler_odeint_linear_ode():
    """dy/dt = y → exponential growth, values should increase monotonically."""
    func = lambda t, y: y
    y0 = tf.constant([[[1.0, 1.0]]])  # (1, 1, 2), positive initial state
    t = tf.constant([0.0, 0.1, 0.2, 0.3])  # T=4

    result = euler_odeint(func, y0, t)

    assert result.shape == (4, 1, 1, 2), f"Expected (4, 1, 1, 2), got {result.shape}"
    vals = result[:, 0, 0, 0].numpy()
    assert all(
        vals[i] <= vals[i + 1] for i in range(len(vals) - 1)
    ), f"Expected monotonically increasing values, got {vals}"
