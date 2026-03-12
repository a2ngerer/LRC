import tensorflow as tf


def euler_odeint(func, y0, t):
    """Euler ODE integrator.

    Args:
        func: callable(t, y) -> dy where t is a 0-D tensor and dy has the same shape as y
        y0:   initial state, shape (batch, 1, features)
        t:    time points, shape (T,) — numpy array or TF tensor with static shape

    Returns:
        Tensor of shape (T, batch, 1, features)

    Note: Uses Python list + tf.stack (eager mode only; no @tf.function).
    """
    ys = [y0]
    for i in range(t.shape[0] - 1):
        dt = t[i + 1] - t[i]
        dy = func(t[i], ys[-1])
        ys.append(ys[-1] + tf.cast(dt, ys[-1].dtype) * dy)
    return tf.stack(ys, axis=0)
