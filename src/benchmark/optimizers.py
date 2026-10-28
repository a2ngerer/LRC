"""Optimizer choice of the LV and PA builders (issue #82).

'rmsprop' is the protocol default, tf.keras.optimizers.RMSprop(lr) exactly as
the builders compiled it before this option existed; 'rmsprop_legacy'
(tf.keras.optimizers.legacy.RMSprop, step lr * g / (sqrt(rms) + eps)) and
'adam' (tf.keras.optimizers.Adam) exist for the pre-registered init
diagnostic. ``epsilon=None`` keeps the class default.
"""
import tensorflow as tf

OPTIMIZERS = ("rmsprop", "rmsprop_legacy", "adam")
DEFAULT_OPTIMIZER = "rmsprop"


def make_optimizer(name, lr, epsilon=None):
    classes = {"rmsprop": tf.keras.optimizers.RMSprop,
               "rmsprop_legacy": tf.keras.optimizers.legacy.RMSprop,
               "adam": tf.keras.optimizers.Adam}
    if name not in classes:
        raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {name!r}")
    kw = {} if epsilon is None else {"epsilon": epsilon}
    return classes[name](lr, **kw)


def add_optimizer_args(parser):
    """--optimizer / --opt-epsilon of the LV and PA runners."""
    parser.add_argument("--optimizer", default=DEFAULT_OPTIMIZER,
                        choices=OPTIMIZERS,
                        help="training optimizer (issue #82); default rmsprop "
                             "= tf.keras.optimizers.RMSprop, the protocol "
                             "optimizer")
    parser.add_argument("--opt-epsilon", type=float, default=None,
                        help="optimizer epsilon; default: the class default")


def optimizer_build_kwargs(args):
    """The builder kwargs of --optimizer / --opt-epsilon, empty at the
    defaults so config.effective['build'] (and config_hash) of a default run
    stays as before issue #82."""
    if args.optimizer == DEFAULT_OPTIMIZER and args.opt_epsilon is None:
        return {}
    return {"optimizer": args.optimizer, "opt_epsilon": args.opt_epsilon}
