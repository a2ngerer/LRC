# Person-activity benchmark model builder: one compiled Keras model per
# (wiring, cell) combination, so cNCP vs NCP (vs a dense reference) can be
# compared apples-to-apples on the same supervised sequence-classification
# task.
#
# Fairness invariant: every wiring receives the SAME two inputs
# (features (B, T, F), time (B, T, 1)), feeds the (features, time) tuple to
# its recurrent part (so continuous-time cells integrate the true elapsed
# time), and ends in the SAME Dense(num_classes) logits head with output
# (B, T, num_classes). Only the wiring in between differs.

import tensorflow as tf

from src.neurons.base_cell import BaseCell
from src.wirings import CorticalColumnCell, NCPWiring
from src.wirings.cncp import DEFAULT_LAMINA_UNITS

WIRINGS = ("cncp", "ncp", "dense")

# DEFAULT_LAMINA_UNITS (44 units total) was designed to parameter-match the
# inter=16/command=8/motor=2 NCP (see src/wirings/cncp.py). The NCP here is
# sized inter=size, command=size//2, motor=size//2, so all lamina widths are
# scaled by size / 16. Both wirings' parameter counts are dominated by
# O(units^2) terms, so the same linear width scale keeps them in the same
# ballpark: at size=64, cfc_lrc gives cncp 44,999 vs ncp 45,927 parameters
# (ratio 0.98) and gru gives 36,679 vs 29,991 (ratio 1.22); the ratio is
# asserted to stay within [1/1.5, 1.5] in tests/tasks/person_activity.
_LAMINA_REFERENCE_SIZE = 16


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
    """Fail fast on multi-state cells (lstm, mm_*) for EVERY wiring.

    CorticalColumnCell cannot host them (its composite state has one slot
    per graph node), and allowing them in the ncp/dense wirings would break
    the apples-to-apples comparison, so the whole benchmark is restricted
    to single-state cells.
    """
    probe = cell_cls(units=4, **cell_kwargs)
    if isinstance(probe.state_size, (list, tuple)):
        raise ValueError(
            "The person-activity benchmark supports single-state cells only "
            f"(state_size == units); {cell_cls.__name__} has state_size "
            f"{probe.state_size}. Multi-state cells (lstm, mm_*) do not fit "
            "the cncp composite state and are rejected for all wirings.")


def build_person_activity_model(wiring, cell, size=64, seed=42,
                                num_classes=7, feature_size=7, seq_len=32,
                                lr=1e-3, timescale_prior=None, **cell_kwargs):
    """Build and compile a person-activity model for one (wiring, cell).

    Args:
        wiring:       'cncp', 'ncp' or 'dense'.
        cell:         _CELL_REGISTRY key (e.g. 'cfc_lrc', 'gru') or BaseCell
                      subclass; single-state cells only.
        size:         shared width knob. dense: RNN units; ncp: inter=size,
                      command=motor=size//2; cncp: lamina widths scaled by
                      size/16 (see scaled_lamina_units).
        seed:         seeds tf/numpy/python RNGs (weight init) and the
                      wiring mask generation.
        num_classes:  output classes (logits head width).
        feature_size: input feature dimension.
        seq_len:      timesteps per sequence.
        lr:           RMSprop learning rate.
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
    else:  # dense
        # Single fully connected RNN layer: the capacity/reference baseline.
        h = tf.keras.layers.RNN(
            cell_cls(units=size, **cell_kwargs), return_sequences=True)(
            (features, time))

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
