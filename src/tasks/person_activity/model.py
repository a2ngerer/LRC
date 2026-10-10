# Person-activity benchmark model builder: one compiled Keras model per
# (wiring, cell) combination, so cNCP vs NCP (vs the dense references) can be
# compared apples-to-apples on the same supervised sequence-classification
# task.
#
# Fairness invariant: every wiring receives the SAME two inputs
# (features (B, T, F), time (B, T, 1)), feeds the (features, time) tuple to
# its recurrent part (so continuous-time cells integrate the true elapsed
# time, in every layer), and ends in the SAME Dense(num_classes) logits head
# with output (B, T, num_classes). Only the wiring in between differs.
#
# Capacity: the arms are compared at the same EFFECTIVE parameter budget
# (masked-off weights excluded, src/wirings/ncp.py::effective_param_count).
# ``size`` is each arm's own width knob and is derived per arm from the budget
# by the runners (``--param-budget`` -> ``size_for_budget``); a shared ``size``
# does NOT give comparable capacity (at size 64 with cfc_lrc the four arms
# differ by more than a factor of four in effective parameters).

import tensorflow as tf

from src.neurons.base_cell import BaseCell
from src.wirings import CorticalColumnCell, NCPWiring
from src.wirings.cncp import DEFAULT_LAMINA_UNITS

# dense:  one recurrent layer (reference)
# dense3: three stacked recurrent layers, elapsed time fed to every layer -- the
#         depth control that reproduces the NCP's three-hop path without sparsity
# ncp:    one masked NCP cell (inter, command, motor), output = motor slice
# cncp:   one composite cortical-column cell, output = L5ET
WIRINGS = ("cncp", "ncp", "dense", "dense3")

# cNCP lamina widths are scaled from DEFAULT_LAMINA_UNITS (44 units at the
# reference size), so the proportions between the nodes are fixed and only the
# overall width moves with the size knob.
_LAMINA_REFERENCE_SIZE = 16
DENSE3_LAYERS = 3


def scaled_lamina_units(size):
    """Per-node cNCP widths scaled by size / 16 (see comment above)."""
    return {node: max(1, round(units * size / _LAMINA_REFERENCE_SIZE))
            for node, units in DEFAULT_LAMINA_UNITS.items()}


def ncp_layer_sizes(size):
    """NCP layer widths derived from the shared size knob.

    inter=size, command=size//2, motor=size//2. The motor layer is NOT the
    output layer: a shared Dense(num_classes) head sits on top of every
    wiring, so the NCP's readout capacity is not artificially pinned to
    num_classes.
    """
    return size, max(1, size // 2), max(1, size // 2)


def _resolve_cell_cls(cell):
    """Registry key or BaseCell subclass -> BaseCell subclass."""
    if isinstance(cell, str):
        from src.models.rnn_model import _CELL_REGISTRY
        if cell not in _CELL_REGISTRY:
            raise KeyError(f"Unknown cell key {cell!r}. "
                           f"Known keys: {sorted(_CELL_REGISTRY)}")
        return _CELL_REGISTRY[cell]
    if isinstance(cell, type) and issubclass(cell, BaseCell):
        return cell
    raise TypeError("cell must be a _CELL_REGISTRY key or a BaseCell "
                    f"subclass, got {cell!r}")


def _reject_multi_state(cell_cls, cell_kwargs):
    """Fail fast on multi-state cells (lstm, mm_*) for a composite wiring.

    CorticalColumnCell (cncp, tbt_cncp*) and the voting committees keep one
    state slot per graph node or column and therefore cannot host a cell
    with a list state. The dense, dense3 and ncp arms can (tf.keras.layers.RNN
    and NCPLayeredCell carry list states), so the builders call this only for
    the composite wirings and the thesis matrix runs lstm / mm_* on dense and
    NCP.
    """
    probe = cell_cls(units=4, **cell_kwargs)
    if isinstance(probe.state_size, (list, tuple)):
        raise ValueError(
            "This wiring supports single-state cells only "
            f"(state_size == units); {cell_cls.__name__} has state_size "
            f"{probe.state_size}. Multi-state cells (lstm, mm_*) do not fit "
            "the composite state; run them on dense, dense3 or ncp.")


def stacked_dense(cell_cls, size, features, time, n_layers=1, **cell_kwargs):
    """n_layers stacked recurrent layers of cell_cls(units=size).

    Every layer receives the (sequence, time) tuple, so continuous-time cells
    integrate the true elapsed time in every layer (the single-layer 'dense'
    arm is n_layers=1, the depth control 'dense3' is n_layers=3).
    """
    h = features
    for _ in range(n_layers):
        h = tf.keras.layers.RNN(cell_cls(units=size, **cell_kwargs),
                                return_sequences=True)((h, time))
    return h


def build_person_activity_model(wiring, cell, size=64, seed=42,
                                num_classes=7, feature_size=7, seq_len=32,
                                lr=1e-3, timescale_prior=None, **cell_kwargs):
    """Build and compile a person-activity model for one (wiring, cell).

    Args:
        wiring:       'cncp', 'ncp', 'dense' or 'dense3'.
        cell:         _CELL_REGISTRY key (e.g. 'cfc_lrc', 'gru', 'lstm') or
                      BaseCell subclass; cncp takes single-state cells only.
        size:         the arm's width knob. dense/dense3: RNN units per layer;
                      ncp: inter=size, command=motor=size//2; cncp: lamina
                      widths scaled by size/16 (see scaled_lamina_units).
                      Derive it per arm from the parameter budget
                      (src.wirings.size_for_budget).
        seed:         seeds tf/numpy/python RNGs (weight init) and the
                      wiring mask generation.
        num_classes:  output classes (logits head width).
        feature_size: input feature dimension.
        seq_len:      timesteps per sequence.
        lr:           RMSprop learning rate.
        timescale_prior: optional per-lamina timescale prior (cncp only).
        **cell_kwargs: forwarded to every cell constructor.

    Returns:
        Compiled tf.keras.Model with inputs [features (B, seq_len,
        feature_size), time (B, seq_len, 1)] and output (B, seq_len,
        num_classes) logits; loss SparseCategoricalCrossentropy(from_logits),
        metric SparseCategoricalAccuracy.
    """
    if wiring not in WIRINGS:
        raise ValueError(f"wiring must be one of {WIRINGS}, got {wiring!r}")
    cell_cls = _resolve_cell_cls(cell)
    if wiring == "cncp":
        _reject_multi_state(cell_cls, cell_kwargs)

    # Reproducibility: seeds python/numpy/tf in one call.
    tf.keras.utils.set_random_seed(seed)

    features = tf.keras.Input(shape=(seq_len, feature_size), name="features")
    time = tf.keras.Input(shape=(seq_len, 1), name="time")

    if wiring == "cncp":
        # One composite RNN cell holding the whole sparse cNCP graph; it
        # consumes (features, time) directly, so every sub-cell integrates
        # the true elapsed time at every step.
        cncp_cell = CorticalColumnCell(
            cell_cls=cell_cls, lamina_units=scaled_lamina_units(size),
            seed=seed, timescale_prior=timescale_prior, **cell_kwargs)
        h = tf.keras.layers.RNN(cncp_cell, return_sequences=True)(
            (features, time))
    elif wiring == "ncp":
        # One NCP cell: the sparse circuit lives inside a single recurrent
        # cell, so it consumes (features, time) directly and every neuron
        # integrates the true elapsed time (docs/ncp-wiring-fix-2026-09-17.md).
        inter, command, motor = ncp_layer_sizes(size)
        ncp = NCPWiring(cell_cls, inter, command, motor, seed=seed,
                        **cell_kwargs)
        h = tf.keras.layers.RNN(ncp.make_cell(), return_sequences=True)(
            (features, time))
        h = ncp.motor_slice()(h)
    else:  # dense / dense3
        h = stacked_dense(cell_cls, size, features, time,
                          n_layers=DENSE3_LAYERS if wiring == "dense3" else 1,
                          **cell_kwargs)

    # Shared per-timestep logits head. A plain Dense on a (B, T, units)
    # tensor applies to the last axis, i.e. it is equivalent to
    # TimeDistributed(Dense(num_classes)) for a single layer.
    logits = tf.keras.layers.Dense(num_classes, name="logits")(h)

    model = tf.keras.Model(inputs=[features, time], outputs=logits)
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    return model
