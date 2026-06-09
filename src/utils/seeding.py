import os
import random

import numpy as np
import tensorflow as tf


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
    if deterministic_ops:
        tf.config.experimental.enable_op_determinism()
    return np.random.default_rng(seed)
