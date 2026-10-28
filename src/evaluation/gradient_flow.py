import numpy as np
import tensorflow as tf

from src.evaluation.param_groups import group_parameters
from src.neurons import CTRNN_Cell, LRC_Cell


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


def ncp_state_slices(inter, command, motor):
    """Neuron slices of an NCP state in ncps order [motor | command | inter]."""
    return {"motor": (0, motor), "command": (motor, motor + command),
            "inter": (motor + command, motor + command + inter)}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# Per-step size of each optimizer (issue #82), keyed by the class it applies to.
_EFF_LR = {
    "RMSprop": "lr / sqrt(v + eps)",
    "legacy.RMSprop": "lr / (sqrt(rms) + eps)",
    "Adam": "lr_t / (sqrt(v) + eps), lr_t = lr sqrt(1 - beta_2^t) / "
            "(1 - beta_1^t), t = iterations",
}


def _optimizer_kind(opt):
    if isinstance(opt, tf.keras.optimizers.legacy.RMSprop):
        return "legacy.RMSprop"
    if isinstance(opt, tf.keras.optimizers.Adam):
        return "Adam"
    return "RMSprop"


class RQ4Diagnostics(tf.keras.callbacks.Callback):
    """--diagnostics (issue #30): measured on one fixed training batch before
    the first step (epoch 0) and after every epoch, result keys:

      optimizer     class ('RMSprop', 'legacy.RMSprop' or 'Adam'), epsilon and
                    the eff_lr formula used below (issue #82)
      per_group[g]  n_params; grad_norm (L2 over the group, see
                    src/evaluation/param_groups.py); slot_norm (L2 of the
                    second-moment slot: RMSprop velocity v, legacy RMSprop
                    rms, Adam v; 0 before the first step); eff_lr_mean (mean
                    step size over the entries with a nonzero slot: masked-off
                    NCP weights never get a gradient, so their slot stays 0;
                    None before the first step). RMSprop lr / sqrt(v + eps),
                    legacy RMSprop lr / (sqrt(rms) + eps), Adam lr_t /
                    (sqrt(v) + eps) with Keras' bias-corrected lr_t = lr
                    sqrt(1 - beta_2^t) / (1 - beta_1^t): the factor Keras
                    applies to the first moment m
      state_norm[l] mean / max over batch and time of ||h_t||_2 of RNN layer l
                    (all state slots concatenated), teacher forced
      ctrnn_rate    models with CTRNN cells only: max_rate = max_i dt /
                    (ode_unfolds * tau_i) over all CTRNN units and n_above_1, the
                    number of units with rate > 1; dt = largest elapsed time in
                    the batch (input 1 of the LV and PA models)
      lrc_working_point[l][s]  RNN layers whose cell is an LRC_Cell only
                    (issue #82): the operating point of the LAST input step
                    of the batch, recomputed in numpy from the state entering
                    that step and its mapped input x * input_w + input_b:
                    sig_f = mean sigmoid(f), tanh_g = mean |tanh g| (f, g as
                    in LRC_Cell._ode_solver, masks applied), w_mean = mean of
                    the unmasked w entries (None for a slice without any: the
                    ncp inter neurons), gleak_mean, vleak_abs = mean
                    |vleak|, per neuron slice s: the cell's init_slices, else
                    ``slices`` (the runner passes ncp_state_slices for 'ncp'),
                    else 'all'. Composite cells (cncp, mixed memory) holding
                    LRC cells are not covered.

    One extra forward/backward plus one state pass per epoch. Weights,
    optimizer state and RNG are not touched, so the training run is unchanged.
    """

    def __init__(self, model, batch_inputs, batch_y, slices=None):
        super().__init__()
        self._net = model
        self._x = tuple(tf.constant(a) for a in batch_inputs)
        self._y = tf.constant(batch_y)
        self._groups = group_parameters(model)
        self._vars = [v for vs in self._groups.values() for v in vs]
        rnns = [layer for layer in model.layers
                if isinstance(layer, tf.keras.layers.RNN)]
        self._taps = [(layer.name, tf.keras.Model(model.inputs, layer.input),
                       tf.keras.layers.RNN(StateTap(layer.cell),
                                           return_sequences=True))
                      for layer in rnns]
        self._ctrnn = _cells(model, CTRNN_Cell)
        self._lrc = [(i, layer.name, layer.cell) for i, layer in enumerate(rnns)
                     if isinstance(layer.cell, LRC_Cell)]
        self._slices = slices
        opt = model.optimizer
        self._kind = _optimizer_kind(opt)
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
        self.result["optimizer"] = {
            "class": self._kind,
            "epsilon": float(tf.keras.backend.get_value(opt.epsilon)),
            "eff_lr": _EFF_LR[self._kind]}
        if self._lrc:
            self._cell_slices = {name: self._slices_of(cell)
                                 for _, name, cell in self._lrc}
            self.result["lrc_working_point"] = {
                name: {s: {k: [] for k in ("sig_f", "tanh_g", "w_mean",
                                           "gleak_mean", "vleak_abs")}
                       for s in sl}
                for name, sl in self._cell_slices.items()}

    def _slices_of(self, cell):
        n = cell.state_size
        for sl in (getattr(cell, "_init_slices", None), self._slices):
            if sl and max(b for _, b in sl.values()) == n:
                return {k: tuple(v) for k, v in sl.items()}
        return {"all": (0, n)}

    @tf.function
    def _measure(self, x, y):
        with tf.GradientTape() as tape:
            loss = self._net.loss(y, self._net(x, training=False))
        grads = tape.gradient(loss, self._vars,
                              unconnected_gradients=tf.UnconnectedGradients.ZERO)
        feeds = [feed(x, training=False) for _, feed, _ in self._taps]
        states = [tap(f, training=False)
                  for f, (_, _, tap) in zip(feeds, self._taps)]
        norms = [tf.norm(s, axis=-1) for s in states]
        wp = []
        for i, _, _ in self._lrc:
            inp = feeds[i][0] if isinstance(feeds[i], (list, tuple)) else feeds[i]
            # the state entering the last step (zeros before the first one)
            prev = tf.concat([tf.zeros_like(states[i][:, :1]), states[i]], 1)
            wp.append((inp[:, -1], prev[:, -2]))
        return grads, norms, wp

    def _slots(self, variables):
        """Second-moment slot entries of ``variables`` (None before step 1)."""
        opt = self._net.optimizer
        if self._kind == "legacy.RMSprop":
            try:
                return np.concatenate([opt.get_slot(v, "rms").numpy().ravel()
                                       for v in variables])
            except KeyError:                      # slots not created yet
                return None
        if not hasattr(opt, "_velocities"):       # not built: before step 1
            return None
        return np.concatenate([opt._velocities[opt._index_dict[
            opt._var_key(v)]].numpy().ravel() for v in variables])

    def _eff_lr(self, live, lr):
        opt = self._net.optimizer
        eps = float(tf.keras.backend.get_value(opt.epsilon))
        if self._kind == "legacy.RMSprop":
            return lr / (np.sqrt(live) + eps)
        if self._kind == "Adam":
            t = float(opt.iterations.numpy())
            b1, b2 = (float(tf.keras.backend.get_value(b))
                      for b in (opt.beta_1, opt.beta_2))
            return lr * np.sqrt(1 - b2 ** t) / (1 - b1 ** t) / (np.sqrt(live)
                                                                 + eps)
        return lr / np.sqrt(live + eps)

    @staticmethod
    def _working_point(cell, x, v, slices):
        """numpy recomputation of f and g of LRC_Cell._ode_solver at state v
        (B, N) and raw cell input x (B, in); one dict per slice."""
        p = {k: w.numpy() for k, w in cell._params.items()}
        n = v.shape[-1]
        sm = (np.ones((n, n), np.float32) if cell.sparsity_mask is None
              else cell.sparsity_mask.numpy())
        se = (np.ones((x.shape[-1], n), np.float32) if cell.sensory_mask is None
              else cell.sensory_mask.numpy())
        if cell._input_mapping in ("affine", "linear"):
            x = x * p["input_w"]
        if cell._input_mapping == "affine":
            x = x + p["input_b"]
        syn = _sigmoid(p["sigma"] * (v[:, :, None] - p["mu"]))
        ssyn = _sigmoid(p["sensory_sigma"] * (x[:, :, None] - p["sensory_mu"]))
        f = (p["gleak"] + np.sum(sm * p["w"] * syn, 1)
             + np.sum(se * p["sensory_w"] * ssyn, 1))
        g = (p["gleak"] + np.sum(sm * p["h"] * syn, 1)
             + np.sum(se * p["sensory_h"] * ssyn, 1))
        out = {}
        for s, (a, b) in slices.items():
            w = p["w"][:, a:b][sm[:, a:b] != 0]
            out[s] = {"sig_f": float(np.mean(_sigmoid(f[:, a:b]))),
                      "tanh_g": float(np.mean(np.abs(np.tanh(g[:, a:b])))),
                      "w_mean": float(w.mean()) if w.size else None,
                      "gleak_mean": float(np.mean(p["gleak"][a:b])),
                      "vleak_abs": float(np.mean(np.abs(p["vleak"][a:b])))}
        return out

    def _record(self, epoch):
        # numpy after the one graph call: eager TF ops per group cost more
        # than the measurement itself on these small models.
        grads, norms, wp = self._measure(self._x, self._y)
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
                float(np.mean(self._eff_lr(live, lr)))
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
        for (_, name, cell), (x, v) in zip(self._lrc, wp):
            point = self._working_point(cell, x.numpy(), v.numpy(),
                                        self._cell_slices[name])
            for s, vals in point.items():
                for k, val in vals.items():
                    self.result["lrc_working_point"][name][s][k].append(val)

    def on_train_begin(self, logs=None):
        self._record(0)

    def on_epoch_end(self, epoch, logs=None):
        self._record(epoch + 1)
