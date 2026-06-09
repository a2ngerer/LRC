import numpy as np
import tensorflow as tf


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
        self.history = {'iterations': [], 'layers': {}}

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

    def as_dict(self) -> dict:
        return {
            'log_every': self.log_every,
            'iterations': self.history['iterations'],
            'layer_norms': self.history['layers'],
        }
