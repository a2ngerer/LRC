"""Largest Lyapunov exponent of a learned free-run map (Benettin, #66).

For a map z_{t+1} = F(z_t) one tangent vector v is pushed through the
Jacobians, v <- J_t v / ||J_t v||, and the log growth factors are averaged:

    lambda_max ~ (1 / H) * sum_{t < H} log ||J_t v_t||,

the largest exponent of the Jacobian product J_{H-1}...J_0 in units of
1 / step (Benettin et al. 1980, one vector, renormalised every step). J_t v is
a forward-mode JVP, so no Jacobian matrix is formed.
"""
import numpy as np
import tensorflow as tf


def largest_lyapunov(step, z0, horizon, seed=0):
    """Per-row Benettin estimate (1 / step) of ``step`` over ``horizon`` steps
    from the (B, D) start states ``z0``; returns a (B,) numpy array. A row
    whose tangent vector collapses to zero gives -inf."""
    @tf.function
    def tangent_step(z, v):
        with tf.autodiff.ForwardAccumulator(z, v) as acc:
            z1 = step(z)
        return z1, acc.jvp(z1, unconnected_gradients=tf.UnconnectedGradients.ZERO)

    z = tf.convert_to_tensor(z0, tf.float32)
    v = tf.random.stateless_normal(z.shape, seed=(seed, 0))
    v /= tf.norm(v, axis=-1, keepdims=True)
    total = np.zeros(z.shape[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        for _ in range(horizon):
            z, v = tangent_step(z, v)
            r = tf.norm(v, axis=-1, keepdims=True)
            total += np.log(r.numpy()[:, 0])
            v = tf.math.divide_no_nan(v, r)
    return total / horizon
