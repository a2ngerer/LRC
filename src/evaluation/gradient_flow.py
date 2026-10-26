import numpy as np
import tensorflow as tf

from src.evaluation.param_groups import group_parameters
from src.neurons import CTRNN_Cell


def _leaf_layers(layer):
    """Recursively flatten nested Keras layers/models into leaf layers."""
    sub = getattr(layer, 'layers', None)
    if not sub:
        return [layer]
    leaves = []
    for child in sub:
        leaves.extend(_leaf_layers(child))
    return leaves


class GradientFlowTracker:
    """Records per-layer gradient norms during training (RQ4).

    Usage:
        tracker = GradientFlowTracker(log_every=25)
        train(model, t, y, ..., gradient_tracker=tracker)
        tracker.history  # {'iterations': [...], 'layers': {name: [norms]}}

    Layer names are '<index>_<class_name>' (e.g. '0_RNN', '1_SparseLinear'),
    indexed in model order, so inter/command/motor RNN layers of an NCP
    model are distinguishable.
    """

    def __init__(self, log_every: int = 25):
        self.log_every = log_every
        self.history = {
            'iterations': [], 'layers': {},
            'clip': {'iterations': [], 'pre_clip_norm': [], 'clip_norm': []},
        }

    def should_log(self, iteration: int) -> bool:
        return self.log_every > 0 and iteration % self.log_every == 0

    def record(self, iteration: int, model, grads) -> None:
        """Store the global gradient norm of each leaf layer.

        Args:
            iteration: current training iteration (1-based)
            model:     the Keras model that produced the gradients
            grads:     gradients aligned with model.trainable_variables
        """
        grad_by_var = {id(v): g for v, g in zip(model.trainable_variables, grads)}
        self.history['iterations'].append(iteration)
        for idx, layer in enumerate(_leaf_layers(model)):
            layer_grads = [
                grad_by_var[id(v)] for v in layer.trainable_variables
                if id(v) in grad_by_var and grad_by_var[id(v)] is not None
            ]
            if not layer_grads:
                continue
            norm = float(tf.linalg.global_norm(layer_grads).numpy())
            name = f'{idx}_{type(layer).__name__}'
            self.history['layers'].setdefault(name, []).append(norm)

    def record_clip(self, iteration: int, pre_clip_norm: float,
                    clip_norm: float) -> None:
        """Store the global norm before clipping and the active threshold.

        The post-clip norm is min(pre_clip_norm, clip_norm) by construction,
        so it is not stored separately.
        """
        c = self.history['clip']
        c['iterations'].append(iteration)
        c['pre_clip_norm'].append(pre_clip_norm)
        c['clip_norm'].append(clip_norm)

    def as_dict(self) -> dict:
        return {
            'log_every': self.log_every,
            'iterations': self.history['iterations'],
            'layer_norms': self.history['layers'],
            'clip': self.history['clip'],
        }


class StateTap(tf.keras.layers.AbstractRNNCell):
    """Read-only wrapper: runs a built cell and emits all its new states.

    Wrapped in a fresh RNN layer it returns the per-step states of the original
    (shared-weight) cell, all state slots concatenated; it creates no variables
    and changes no state of the trained model. (Same as the LV runner's
    StabilityProbe tap.)
    """

    def __init__(self, cell):
        super().__init__()
        self.cell = cell

    @property
    def state_size(self):
        return self.cell.state_size

    @property
    def output_size(self):
        return int(np.sum(np.atleast_1d(self.cell.state_size)))

    def build(self, input_shape):
        self.built = True

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return self.cell.get_initial_state(inputs=inputs, batch_size=batch_size,
                                           dtype=dtype)

    def call(self, inputs, states):
        _, new = self.cell(inputs, states)
        new = list(new) if isinstance(new, (list, tuple)) else [new]
        return tf.concat(new, axis=-1), new


def _cells(model, cls):
    return [c for layer in model.layers if isinstance(layer, tf.keras.layers.RNN)
            for c in layer.cell._flatten_layers(include_self=True)
            if isinstance(c, cls)]


class RQ4Diagnostics(tf.keras.callbacks.Callback):
    """--diagnostics (issue #30): measured on one fixed training batch before
    the first step (epoch 0) and after every epoch, result keys:

      per_group[g]  n_params; grad_norm (L2 over the group, see
                    src/evaluation/param_groups.py); slot_norm (L2 of the
                    RMSprop velocity v, 0 before the first step); eff_lr_mean
                    (mean of the Keras RMSprop step size lr / sqrt(v + eps) over
                    the entries with v > 0: masked-off NCP weights never get a
                    gradient, so their v stays 0; None before the first step)
      state_norm[l] mean / max over batch and time of ||h_t||_2 of RNN layer l
                    (all state slots concatenated), teacher forced
      ctrnn_rate    models with CTRNN cells only: max_rate = max_i dt /
                    (ode_unfolds * tau_i) over all CTRNN units and n_above_1, the
                    number of units with rate > 1; dt = largest elapsed time in
                    the batch (input 1 of the LV and PA models)

    One extra forward/backward plus one state pass per epoch. Weights,
    optimizer state and RNG are not touched, so the training run is unchanged.
    """

    def __init__(self, model, batch_inputs, batch_y):
        super().__init__()
        self._net = model
        self._x = tuple(tf.constant(a) for a in batch_inputs)
        self._y = tf.constant(batch_y)
        self._groups = group_parameters(model)
        self._vars = [v for vs in self._groups.values() for v in vs]
        self._taps = [(layer.name, tf.keras.Model(model.inputs, layer.input),
                       tf.keras.layers.RNN(StateTap(layer.cell),
                                           return_sequences=True))
                      for layer in model.layers
                      if isinstance(layer, tf.keras.layers.RNN)]
        self._ctrnn = _cells(model, CTRNN_Cell)
        self._dt = float(np.max(batch_inputs[1]))
        self.result = {
            "epochs": [],
            "per_group": {g: {"n_params": int(sum(np.prod(v.shape) for v in vs)),
                              "grad_norm": [], "slot_norm": [],
                              "eff_lr_mean": []}
                          for g, vs in self._groups.items()},
            "state_norm": {name: {"mean": [], "max": []}
                           for name, _, _ in self._taps}}
        if self._ctrnn:
            self.result["ctrnn_rate"] = {"max_rate": [], "n_above_1": []}

    @tf.function
    def _measure(self, x, y):
        with tf.GradientTape() as tape:
            loss = self._net.loss(y, self._net(x, training=False))
        grads = tape.gradient(loss, self._vars,
                              unconnected_gradients=tf.UnconnectedGradients.ZERO)
        norms = [tf.norm(tap(feed(x, training=False), training=False), axis=-1)
                 for _, feed, tap in self._taps]
        return grads, norms

    def _slots(self, variables):
        opt = self._net.optimizer
        if not hasattr(opt, "_velocities"):       # not built: before step 1
            return None
        return np.concatenate([opt._velocities[opt._index_dict[
            opt._var_key(v)]].numpy().ravel() for v in variables])

    def _record(self, epoch):
        # numpy after the one graph call: eager TF ops per group cost more
        # than the measurement itself on these small models.
        grads, norms = self._measure(self._x, self._y)
        grads = [g.numpy() for g in grads]
        opt = self._net.optimizer
        lr = float(tf.keras.backend.get_value(opt.learning_rate))
        self.result["epochs"].append(epoch)
        i = 0
        for g, vs in self._groups.items():
            out = self.result["per_group"][g]
            out["grad_norm"].append(float(np.sqrt(sum(
                np.sum(np.square(a, dtype=np.float64))
                for a in grads[i:i + len(vs)]))))
            i += len(vs)
            v = self._slots(vs)
            live = None if v is None else v[v > 0]
            out["slot_norm"].append(0.0 if v is None else float(np.linalg.norm(v)))
            out["eff_lr_mean"].append(
                float(np.mean(lr / np.sqrt(live + opt.epsilon)))
                if live is not None and live.size else None)
        for (name, _, _), n in zip(self._taps, norms):
            n = n.numpy()
            self.result["state_norm"][name]["mean"].append(float(n.mean()))
            self.result["state_norm"][name]["max"].append(float(n.max()))
        if self._ctrnn:
            rate = np.concatenate([self._dt / (c.ode_unfolds * (
                c.tau.numpy() + c.epsilon)) for c in self._ctrnn])
            self.result["ctrnn_rate"]["max_rate"].append(float(rate.max()))
            self.result["ctrnn_rate"]["n_above_1"].append(int(np.sum(rate > 1.0)))

    def on_train_begin(self, logs=None):
        self._record(0)

    def on_epoch_end(self, epoch, logs=None):
        self._record(epoch + 1)
