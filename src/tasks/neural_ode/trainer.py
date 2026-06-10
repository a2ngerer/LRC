import numpy as np
import tensorflow as tf
from .solver import euler_odeint

_LOSS_FNS = {
    'mse': lambda pred, true: tf.reduce_mean(tf.square(pred - true)),
    'mae': lambda pred, true: tf.reduce_mean(tf.abs(pred - true)),
}


def get_batch(t, y, batch_size, batch_time, rng=None):
    """Sample a random batch of sub-sequences.

    Args:
        t:          time array, shape (data_size,)
        y:          trajectory array, shape (data_size, 2)
        batch_size: number of random start points
        batch_time: length of each sub-sequence
        rng:        optional np.random.Generator for reproducible sampling;
                    falls back to numpy's global RNG

    Returns:
        y0:       shape (batch_size, 1, 2)          — initial states
        t_batch:  shape (batch_time,)               — fixed time window t[:batch_time]
        y_batch:  shape (batch_time, batch_size, 1, 2) — trajectory for each batch element
    """
    data_size = len(t)
    if rng is not None:
        s = rng.choice(data_size - batch_time, batch_size, replace=False)
    else:
        s = np.random.choice(data_size - batch_time, batch_size, replace=False)

    y0 = y[s][:, np.newaxis, :]                           # (batch_size, 1, 2)
    t_batch = t[:batch_time]                               # (batch_time,)
    y_batch = np.stack(
        [y[s_i:s_i + batch_time, np.newaxis, :] for s_i in s], axis=1
    )  # (batch_time, batch_size, 1, 2)

    return y0, t_batch, y_batch


def train(model, t, y, n_iters, batch_size=16, batch_time=16, lr=1e-3,
          loss='mse', rng=None, gradient_tracker=None, clip_norm=None,
          verbose=True):
    """Training loop.

    Args:
        model:      ODEFuncModel instance
        t:          time array from generate_dataset, shape (data_size,)
        y:          trajectory array from generate_dataset, shape (data_size, 2)
        n_iters:    number of training iterations
        batch_size: random start points per batch
        batch_time: time steps per batch
        lr:         Adam learning rate
        loss:       'mse' (default, thesis metric) or 'mae' (legacy)
        rng:        optional np.random.Generator for reproducible batch sampling
        gradient_tracker: optional GradientFlowTracker; record() is called on
                    iterations where should_log() is True (RQ4 tooling)
        clip_norm:  optional float; if set, gradients are rescaled with
                    tf.clip_by_global_norm(grads, clip_norm) before the
                    optimizer step. None (default) = v1 behavior.
        verbose:    print loss every 10 iterations

    Returns:
        list of float losses, one per iteration
    """
    loss_fn = _LOSS_FNS[loss]
    optimizer = tf.keras.optimizers.Adam(lr)
    losses = []

    for itr in range(1, n_iters + 1):
        y0_batch, t_batch, y_batch = get_batch(t, y, batch_size, batch_time, rng=rng)
        t_tf = tf.constant(t_batch, dtype=tf.float32)
        y0_tf = tf.constant(y0_batch, dtype=tf.float32)
        y_true = tf.constant(y_batch, dtype=tf.float32)

        with tf.GradientTape() as tape:
            # pred: (batch_time, batch_size, 1, 2)
            # true: (batch_time, batch_size, 1, 2)
            pred = euler_odeint(model, y0_tf, t_tf)
            loss_value = loss_fn(pred, y_true)

        grads = tape.gradient(loss_value, model.trainable_variables)
        if clip_norm:
            applied_grads, pre_clip_norm = tf.clip_by_global_norm(grads, clip_norm)
        else:
            applied_grads = grads
        # Per-layer norms are recorded pre-clip so RQ4 sees the raw gradient
        # pathology; the clip block shows when and how hard clipping engaged.
        if gradient_tracker is not None and gradient_tracker.should_log(itr):
            gradient_tracker.record(itr, model, grads)
            if clip_norm:
                gradient_tracker.record_clip(
                    itr, float(pre_clip_norm.numpy()), clip_norm)
        optimizer.apply_gradients(zip(applied_grads, model.trainable_variables))

        loss_val = float(loss_value.numpy())
        losses.append(loss_val)

        if verbose and itr % 10 == 0:
            print(f'Iter {itr:04d} | Loss {loss_val:.6f}')

    return losses
