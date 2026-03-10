import tensorflow as tf
from .base_wiring import BaseWiring


class DenseWiring(BaseWiring):
    """Fully-connected wiring: standard tf.keras.layers.RNN.

    All neurons see all inputs — this is the default RNN behaviour.
    return_sequences is hardcoded to True (Phase 1: sequence tasks only).
    """

    def __init__(self, cell):
        super().__init__(cell)

    def build_model(self) -> tf.keras.Sequential:
        return tf.keras.Sequential([
            tf.keras.layers.RNN(self.cell, return_sequences=True)
        ])
