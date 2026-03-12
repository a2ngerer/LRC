import numpy as np
import tensorflow as tf
from .solver import euler_odeint


def get_batch(t, y, batch_size, batch_time):
    """Sample a random batch of sub-sequences.

    Args:
        t:          time array, shape (data_size,)
        y:          trajectory array, shape (data_size, 2)
        batch_size: number of random start points
        batch_time: length of each sub-sequence

    Returns:
        y0:       shape (batch_size, 1, 2)          — initial states
        t_batch:  shape (batch_time,)               — fixed time window t[:batch_time]
        y_batch:  shape (batch_time, batch_size, 1, 2) — trajectory for each batch element
    """
    data_size = len(t)
    s = np.random.choice(data_size - batch_time, batch_size, replace=False)

    y0 = y[s][:, np.newaxis, :]                           # (batch_size, 1, 2)
    t_batch = t[:batch_time]                               # (batch_time,)
    y_batch = np.stack(
        [y[s_i:s_i + batch_time, np.newaxis, :] for s_i in s], axis=1
    )  # (batch_time, batch_size, 1, 2)

    return y0, t_batch, y_batch


def train(model, t, y, n_iters, batch_size=16, batch_time=16, lr=1e-3):
    """Training loop.

    Args:
        model:      ODEFuncModel instance
        t:          time array from generate_dataset, shape (data_size,)
        y:          trajectory array from generate_dataset, shape (data_size, 2)
        n_iters:    number of training iterations
        batch_size: random start points per batch
        batch_time: time steps per batch
        lr:         Adam learning rate

    Returns:
        list of float losses, one per iteration
    """
    optimizer = tf.keras.optimizers.Adam(lr)
    losses = []

    for itr in range(1, n_iters + 1):
        y0_batch, t_batch, y_batch = get_batch(t, y, batch_size, batch_time)
        t_tf = tf.constant(t_batch, dtype=tf.float32)
        y0_tf = tf.constant(y0_batch, dtype=tf.float32)
        y_true = tf.constant(y_batch, dtype=tf.float32)

        with tf.GradientTape() as tape:
            # pred: (batch_time, batch_size, 1, 2)
            # true: (batch_time, batch_size, 1, 2)
            pred = euler_odeint(model, y0_tf, t_tf)
            loss = tf.reduce_mean(tf.abs(pred - y_true))

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        loss_val = float(loss.numpy())
        losses.append(loss_val)

        if itr % 10 == 0:
            print(f'Iter {itr:04d} | Loss {loss_val:.6f}')

    return losses
