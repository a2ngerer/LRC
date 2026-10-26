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
# Discrete cells (DISCRETE_CELLS) discard the time input, so by default they
# get the same elapsed-time tensor concatenated to their feature input instead
# (dt_feature, issue #27). The builders multiply the time input by dt_scale
# before it reaches any cell or the dt feature (task-level time unit, issue #28).
#
# Capacity: the arms are compared at the same EFFECTIVE parameter budget
# (masked-off weights excluded, src/wirings/ncp.py::effective_param_count).
# ``size`` is each arm's own width knob and is derived per arm from the budget
# by the runners (``--param-budget`` -> ``size_for_budget``); a shared ``size``
# does NOT give comparable capacity (at size 64 with cfc_lrc the four arms
# differ by more than a factor of four in effective parameters).

import warnings

import tensorflow as tf

from src.neurons import LRC_Cell, LTC_Cell, MM_LRC_Cell, MM_LTC_Cell
from src.neurons.base_cell import BaseCell
from src.wirings import (CorticalColumnCell, CorticalColumnV2Cell, NCPWiring,
                         PARTITIONS, COUPLINGS, READOUTS, multi_column_head)
from src.wirings.cncp import DEFAULT_LAMINA_UNITS
from src.wirings.ncp import CFC_NCP_CELLS, ncp_cell_kwargs

# dense:  one recurrent layer (reference)
# dense3: three stacked recurrent layers, elapsed time fed to every layer -- the
#         depth control that reproduces the NCP's three-hop path without sparsity
# ncp:    one masked NCP cell (inter, command, motor), output = motor slice
# cncp:   one composite cortical-column cell, output = L5ET
# cncp2:  one four-node cNCP v2 column (width u = size), output = D
# cncp2_multi: K independent cNCP v2 columns (width u = size each), optional
#         lateral coupling, readout over the column outputs (spec B.7, V4)
WIRINGS = ("cncp", "ncp", "dense", "dense3", "cncp2", "cncp2_multi")
# Wirings whose composite state holds one slot per graph node.
COMPOSITE_WIRINGS = ("cncp", "cncp2", "cncp2_multi")

# Person Activity rows are one-hot(sensor tag, 4) ++ xyz (datasets.py), the
# sensor structure of partition='region' (spec B.7).
PA_SENSOR_TAGS = 4

# cncp2_multi runner flags (--columns, --partition, --coupling, --readout):
# unset flags take these defaults; readout follows the partition (concat under
# region, mean under same, spec B.7).
MULTI_COLUMN_DEFAULTS = dict(columns=2, partition="same", coupling="lateral")

# cNCP lamina widths are scaled from DEFAULT_LAMINA_UNITS (44 units at the
# reference size), so the proportions between the nodes are fixed and only the
# overall width moves with the size knob.
_LAMINA_REFERENCE_SIZE = 16
DENSE3_LAYERS = 3

# Discrete-time cells: they ignore the elapsed time of the (x, dt) input tuple.
DISCRETE_CELLS = ("lstm", "gru")

# Numerical ODE cells: explicit Euler sub-steps per input step (ode_unfolds).
# ltc, lrc (and its LRC_Cell variants such as lrc_pm), mm_ltc, mm_lrc; the
# cfc_mm_* inner cells are closed-form and take no sub-steps (issue #28).
ODE_UNFOLD_CELLS = (LTC_Cell, LRC_Cell, MM_LTC_Cell, MM_LRC_Cell)


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


def dt_feature_applies(cell, dt_feature=True):
    """True when the builder concatenates dt to the feature input: the flag is
    on and the cell is discrete (continuous-time cells already integrate dt)."""
    discrete = tuple(_resolve_cell_cls(c) for c in DISCRETE_CELLS)
    return bool(dt_feature) and issubclass(_resolve_cell_cls(cell), discrete)


def backbone_override(cell, backbone_layers):
    """cell_kwargs entry for the runners' --backbone-layers flag.

    ``{}`` when the flag is unset (every default unchanged) or when the cell has
    no backbone (gru, lstm, ltc, lrc, ... would reject the kwarg). Otherwise
    ``{'backbone_layers': N}``: ncp/cncp honour it through ``ncp_cell_kwargs``
    (explicit value wins over the family default 1), dense/dense3/committee
    cells override the CfC constructor default of 1.
    """
    if backbone_layers is None:
        return {}
    if backbone_layers < 0:
        raise ValueError(f"backbone_layers must be >= 0, got {backbone_layers}")
    if not issubclass(_resolve_cell_cls(cell), CFC_NCP_CELLS):
        return {}
    return {"backbone_layers": int(backbone_layers)}


def effective_backbone_layers(wiring, cell, cell_kwargs):
    """Backbone depth each sub-cell of this arm actually gets (None: no backbone).

    dense/dense3 take the cell constructor default (1) unless overridden; every
    other wiring goes through ncp_cell_kwargs (family default 1, explicit wins).
    """
    cell_cls = _resolve_cell_cls(cell)
    if not issubclass(cell_cls, CFC_NCP_CELLS):
        return None
    if wiring in ("dense", "dense3"):
        return cell_kwargs.get("backbone_layers", 1)
    return ncp_cell_kwargs(cell_cls, cell_kwargs)["backbone_layers"]


def effective_self_recurrent(wiring, cell, self_recurrent=True):
    """Diagonal NCP self-recurrence this arm actually gets (None: not
    applicable). Only the closed-form sub-cells under 'ncp' have it (issue #23,
    NCPLayeredCell); the runners' --no-self-recurrence passes False."""
    if wiring != "ncp" or not issubclass(_resolve_cell_cls(cell), CFC_NCP_CELLS):
        return None
    return bool(self_recurrent)


def backbone_suffix(backbone_layers):
    """Result-filename suffix: '' when unset, '_bb{N}' otherwise."""
    return "" if backbone_layers is None else f"_bb{backbone_layers}"


def ode_unfolds_override(cell, ode_unfolds):
    """cell_kwargs entry for the runners' --ode-unfolds flag.

    ``{'ode_unfolds': N}`` for the numerical ODE cells (ODE_UNFOLD_CELLS),
    ``{}`` for closed-form and discrete cells, which would reject the kwarg.
    """
    if ode_unfolds < 1:
        raise ValueError(f"ode_unfolds must be >= 1, got {ode_unfolds}")
    if not issubclass(_resolve_cell_cls(cell), ODE_UNFOLD_CELLS):
        return {}
    return {"ode_unfolds": int(ode_unfolds)}


def scaled_time(time, dt_scale=1.0):
    """The elapsed-time tensor the cells receive: time * dt_scale. At the
    default 1.0 the input itself, so the graph stays unchanged."""
    if dt_scale <= 0:
        raise ValueError(f"dt_scale must be > 0, got {dt_scale}")
    if dt_scale == 1.0:
        return time
    return tf.keras.layers.Rescaling(dt_scale, name="dt_scale")(time)


def add_multi_column_args(parser):
    """The cncp2_multi flags of the LV and PA runners (default None = unset)."""
    parser.add_argument("--columns", type=int, default=None,
                        help="cncp2_multi: number of independent columns K "
                             f"(default {MULTI_COLUMN_DEFAULTS['columns']})")
    parser.add_argument("--partition", default=None, choices=PARTITIONS,
                        help="cncp2_multi: input routing (default same)")
    parser.add_argument("--coupling", default=None, choices=COUPLINGS,
                        help="cncp2_multi: inter-column coupling "
                             "(default lateral, density 0.25)")
    parser.add_argument("--readout", default=None, choices=READOUTS,
                        help="cncp2_multi: readout over the column outputs "
                             "(default concat under region, mean under same)")


def multi_column_config(args):
    """Resolved cncp2_multi config for the builder and the result JSON, or {}
    for any other wiring (set cncp2_multi flags are then ignored with a
    warning)."""
    names = ("columns", "partition", "coupling", "readout")
    if args.wiring != "cncp2_multi":
        given = [n for n in names if getattr(args, n) is not None]
        if given:
            warnings.warn(f"--{', --'.join(given)} only apply to cncp2_multi; "
                          f"ignored for wiring {args.wiring!r}", stacklevel=2)
        return {}
    cfg = {n: getattr(args, n) for n in names}
    for n, v in MULTI_COLUMN_DEFAULTS.items():
        if cfg[n] is None:
            cfg[n] = v
    if cfg["readout"] is None:
        cfg["readout"] = default_readout(cfg["partition"])
    return cfg


def default_readout(partition):
    """Readout when none is given: concat under 'region' (columns carry
    different information), mean otherwise. Shared by the runner flags and
    the builders so direct calls resolve like the CLI."""
    return "concat" if partition == "region" else "mean"


def multi_column_suffix(cfg):
    """Result-filename suffix: '' for other wirings,
    '_K{K}_{partition}_{coupling}_{readout}' for cncp2_multi."""
    if not cfg:
        return ""
    return (f"_K{cfg['columns']}_{cfg['partition']}_{cfg['coupling']}_"
            f"{cfg['readout']}")


def stacked_dense(cell_cls, size, features, time, n_layers=1, dt_feature=False,
                  **cell_kwargs):
    """n_layers stacked recurrent layers of cell_cls(units=size).

    Every layer receives the (sequence, time) tuple, so continuous-time cells
    integrate the true elapsed time in every layer (the single-layer 'dense'
    arm is n_layers=1, the depth control 'dense3' is n_layers=3). With
    dt_feature, time is also concatenated to every layer's sequence input.
    """
    h = features
    for _ in range(n_layers):
        if dt_feature:
            h = tf.keras.layers.Concatenate()([h, time])
        h = tf.keras.layers.RNN(cell_cls(units=size, **cell_kwargs),
                                return_sequences=True)((h, time))
    return h


def build_person_activity_model(wiring, cell, size=64, seed=42,
                                num_classes=7, feature_size=7, seq_len=32,
                                lr=1e-3, timescale_prior=None, columns=MULTI_COLUMN_DEFAULTS["columns"],
                                partition=MULTI_COLUMN_DEFAULTS["partition"],
                                coupling=MULTI_COLUMN_DEFAULTS["coupling"],
                                readout=None, self_recurrent=True,
                                dt_feature=True, dt_scale=1.0,
                                **cell_kwargs):
    """Build and compile a person-activity model for one (wiring, cell).

    Args:
        wiring:       'cncp', 'ncp', 'dense', 'dense3', 'cncp2' or 'cncp2_multi'.
        cell:         _CELL_REGISTRY key (e.g. 'cfc_lrc', 'gru', 'lstm') or
                      BaseCell subclass; cncp takes single-state cells only.
        size:         the arm's width knob. dense/dense3: RNN units per layer;
                      ncp: inter=size, command=motor=size//2; cncp: lamina
                      widths scaled by size/16 (see scaled_lamina_units);
                      cncp2 / cncp2_multi: column width u = size.
                      Derive it per arm from the parameter budget
                      (src.wirings.size_for_budget).
        seed:         seeds tf/numpy/python RNGs (weight init) and the
                      wiring mask generation.
        num_classes:  output classes (logits head width).
        feature_size: input feature dimension.
        seq_len:      timesteps per sequence.
        lr:           RMSprop learning rate.
        timescale_prior: optional per-lamina timescale prior (cncp only).
        columns, partition, coupling, readout: cncp2_multi only (spec B.7);
                      partition='region' uses the four sensor tags.
        self_recurrent: ncp only, see NCPWiring (issue #23).
        dt_feature:   discrete cells (DISCRETE_CELLS) only: concatenate the
                      time input (the tensor the continuous-time cells
                      integrate, after dt_scale) to the features of every
                      layer that receives them -- each dense3 layer, the
                      input of every other wiring. No-op for continuous-time
                      cells.
        dt_scale:     multiplies the time input before it reaches any cell
                      and the dt feature (task-level time unit, issue #28);
                      1.0 adds no layer.
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
    if wiring in COMPOSITE_WIRINGS:
        _reject_multi_state(cell_cls, cell_kwargs)

    # Reproducibility: seeds python/numpy/tf in one call.
    tf.keras.utils.set_random_seed(seed)

    features = tf.keras.Input(shape=(seq_len, feature_size), name="features")
    time_in = tf.keras.Input(shape=(seq_len, 1), name="time")
    time = scaled_time(time_in, dt_scale)
    dt_feature = dt_feature_applies(cell_cls, dt_feature)
    # dense / dense3 concatenate per layer (stacked_dense).
    x = (tf.keras.layers.Concatenate()([features, time])
         if dt_feature and wiring not in ("dense", "dense3") else features)

    logits = None
    if wiring == "cncp2_multi":
        logits = multi_column_head(
            (x, time), num_classes, "logits", columns=columns,
            cell_cls=cell_cls, units=size, seed=seed, partition=partition,
            coupling=coupling, readout=readout or default_readout(partition),
            region_tags=PA_SENSOR_TAGS,
            **cell_kwargs)
    elif wiring == "cncp":
        # One composite RNN cell holding the whole sparse cNCP graph; it
        # consumes (features, time) directly, so every sub-cell integrates
        # the true elapsed time at every step.
        cncp_cell = CorticalColumnCell(
            cell_cls=cell_cls, lamina_units=scaled_lamina_units(size),
            seed=seed, timescale_prior=timescale_prior, **cell_kwargs)
        h = tf.keras.layers.RNN(cncp_cell, return_sequences=True)((x, time))
    elif wiring == "cncp2":
        col = CorticalColumnV2Cell(cell_cls=cell_cls, units=size, seed=seed,
                                   **cell_kwargs)
        h = tf.keras.layers.RNN(col, return_sequences=True)((x, time))
    elif wiring == "ncp":
        # One NCP cell: the sparse circuit lives inside a single recurrent
        # cell, so it consumes (features, time) directly and every neuron
        # integrates the true elapsed time (docs/ncp-wiring-fix-2026-09-17.md).
        inter, command, motor = ncp_layer_sizes(size)
        ncp = NCPWiring(cell_cls, inter, command, motor, seed=seed,
                        self_recurrent=self_recurrent, **cell_kwargs)
        h = tf.keras.layers.RNN(ncp.make_cell(), return_sequences=True)(
            (x, time))
        h = ncp.motor_slice()(h)
    else:  # dense / dense3
        h = stacked_dense(cell_cls, size, features, time,
                          n_layers=DENSE3_LAYERS if wiring == "dense3" else 1,
                          dt_feature=dt_feature, **cell_kwargs)

    # Shared per-timestep logits head. A plain Dense on a (B, T, units)
    # tensor applies to the last axis, i.e. it is equivalent to
    # TimeDistributed(Dense(num_classes)) for a single layer.
    if logits is None:
        logits = tf.keras.layers.Dense(num_classes, name="logits")(h)

    model = tf.keras.Model(inputs=[features, time_in], outputs=logits)
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    return model
