import os
import random

import numpy as np
import tensorflow as tf
from keras.src import backend as keras_backend  # tensorflow<2.16 pins keras 2.15


def set_global_seed(seed: int, deterministic_ops: bool = False) -> np.random.Generator:
    """Seed all RNGs relevant to a benchmark run.

    Covers Python's hash seed, random, numpy's legacy global RNG and
    TensorFlow's global RNG. Weight initializers and tf.random both
    derive from the TF global seed.

    Args:
        seed:              the seed for all RNGs
        deterministic_ops: if True, additionally enable TF op determinism
                           (bit-exact reruns on GPU at the cost of speed)

    Returns:
        A seeded np.random.Generator for explicit, local random streams
        (e.g. batch sampling in the trainer).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # tf.keras.utils.set_random_seed (called by the task model builders) leaves
    # a thread-local generator that feeds every later initializer seed. Drop
    # it so the seeding above decides again; a fresh process has none, so
    # behaviour there is unchanged.
    keras_backend._SEED_GENERATOR.generator = None
    if deterministic_ops:
        tf.config.experimental.enable_op_determinism()
    return np.random.default_rng(seed)
