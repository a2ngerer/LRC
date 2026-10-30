"""Input-level forcing for the closed-loop rollout tasks (issue #25).

The rollout tasks train teacher-forced (true state y_t as input at every step)
and evaluate free-running. ForcedRolloutCell feeds its core cell a mix of the
true state and the model's own previous prediction,

    x~_0 = y_0,    x~_t = alpha * y_t + (1 - alpha) * yhat_{t-1}   (t >= 1),

with one non-trainable alpha for the whole batch: alpha = 1 is teacher forcing,
alpha = 0 free running (the closed-loop rollout in one O(T) pass), values in
between mixed forcing. A third input f_t (B, 1) replaces alpha by a per-step
weight, x~_t = f_t * y_t + (1 - f_t) * yhat_{t-1}: f = 1 up to an anchor and
0 after it is the teacher-forced warm-up + free run of the multi-anchor
endpoint (#62), one O(T) pass per anchor. The readout runs inside the cell
(yhat_t = readout(h_t)), so the wrapper adds no trainable parameter; it shares
the weights of the core cell and readout it is given.

StackedCell turns the dense3 stack (three RNN layers, see
person_activity.stacked_dense) into one cell so it can be wrapped.
AlphaSchedule is the pre-registered alpha schedule of the mixed protocol.
"""
import tensorflow as tf


def _as_list(states):
    return list(states) if isinstance(states, (list, tuple)) else [states]


class StackedCell(tf.keras.layers.AbstractRNNCell):
    """Sub-cells applied in sequence within one step, each fed (h, dt).

    Every sub-cell receives the elapsed time, as every layer of stacked_dense
    does, so a StackedCell over the cells of a stacked_dense model computes the
    same function (one cell instead of one RNN layer per sub-cell). The state
    is the concatenated list of the sub-cells' states.
    """

    def __init__(self, cells, **kwargs):
        super().__init__(**kwargs)
        self.cells = list(cells)

    @property
    def state_size(self):
        return [s for c in self.cells for s in _as_list(c.state_size)]

    @property
    def output_size(self):
        return self.cells[-1].output_size

    def build(self, input_shape):
        x_shape, t_shape = input_shape
        for cell in self.cells:
            if not cell.built:
                cell.build((x_shape, t_shape))
            x_shape = tf.TensorShape(x_shape)[:-1].concatenate(
                [cell.output_size])
        self.built = True

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return [s for c in self.cells for s in _as_list(c.get_initial_state(
            inputs=None, batch_size=batch_size, dtype=dtype))]

    def call(self, inputs, states):
        h, dt = inputs
        states, new = list(states), []
        for cell in self.cells:
            n = len(_as_list(cell.state_size))
            h, ns = cell((h, dt), states[:n])
            states = states[n:]
            new += _as_list(ns)
        return h, new


class ForcedRolloutCell(tf.keras.layers.AbstractRNNCell):
    """Mixed-forcing wrapper around a core cell and its readout.

    Input (y_t, dt) or (y_t, dt, f_t); state = core states + [yhat_{t-1},
    started]; output yhat_t = readout(h_t). ``started`` (B, 1) is 0 before the
    first step, so x~_0 = y_0 whatever alpha is; a given f_t is used as the
    forcing weight instead (alpha is then unused, f_0 should be 1).
    ``readout`` is a layer or a list of layers applied in order (ncp: motor
    slice, then the Dense head). ``alpha`` is a
    non-trainable tf.Variable read at every step, so a callback can move it
    between epochs.
    """

    def __init__(self, core_cell, readout, alpha, **kwargs):
        super().__init__(**kwargs)
        self.core = core_cell
        self.readout = _as_list(readout)
        # Deliberately not tracked as a Keras weight: it would otherwise count
        # in count_params() (params_raw) and land in the saved weights.
        object.__setattr__(self, "alpha", alpha)
        self._dim = None

    @property
    def state_size(self):
        return [*_as_list(self.core.state_size), self._dim, 1]

    @property
    def output_size(self):
        return self._dim

    def build(self, input_shape):
        y_shape = input_shape[0]
        # yhat is mixed into the input, so it has the input's width.
        self._dim = int(y_shape[-1])
        if not self.core.built:
            self.core.build(tuple(input_shape[:2]))
        self.built = True

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        dtype = dtype or tf.float32
        core = _as_list(self.core.get_initial_state(
            inputs=None, batch_size=batch_size, dtype=dtype))
        return core + [tf.zeros([batch_size, self._dim], dtype),
                       tf.zeros([batch_size, 1], dtype)]

    def call(self, inputs, states):
        y, dt, *force = inputs
        *core_states, y_prev, started = _as_list(states)
        # 1 at t = 0, alpha after; or the given per-step weight
        a = force[0] if force else 1.0 - started * (1.0 - self.alpha)
        h, new_core = self.core((a * y + (1.0 - a) * y_prev, dt), core_states)
        y_hat = h
        for layer in self.readout:
            y_hat = layer(y_hat)
        return y_hat, [*_as_list(new_core), y_hat, tf.ones_like(started)]


class AlphaSchedule(tf.keras.callbacks.Callback):
    """alpha linear from 1.0 to alpha_min over the first half of the epochs
    (alpha_min from epoch epochs // 2 on), then constant. The value used in
    each epoch is appended to ``curve`` (the result JSON's alpha_curve)."""

    def __init__(self, alpha, epochs, alpha_min=0.2):
        super().__init__()
        self.alpha, self.alpha_min = alpha, alpha_min
        self.half = max(1, epochs // 2)
        self.curve = []

    def value(self, epoch):
        return max(self.alpha_min,
                   1.0 - (1.0 - self.alpha_min) * epoch / self.half)

    def on_epoch_begin(self, epoch, logs=None):
        self.alpha.assign(self.value(epoch))
        self.curve.append(self.value(epoch))
