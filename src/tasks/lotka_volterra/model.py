# Lotka-Volterra next-step-prediction model builder: one compiled Keras model
# per (wiring, cell), so cNCP vs NCP vs dense (vs the dense depth control) can
# be compared on the SAME predator-prey sequence-rollout regression.
#
# Mirrors src/tasks/person_activity/model.py (same wiring dispatch, reused
# helpers) with three task-specific changes:
#   - variable sequence length (Input shape (None, ...)) so the trained model can
#     be driven closed-loop on a growing history at evaluation time;
#   - a linear Dense(2) regression head instead of the softmax logits head;
#   - MSE loss / MAE metric instead of cross-entropy.
#
# Fairness invariant (identical to person_activity): every wiring gets the same
# two inputs (state (B, T, 2), time (B, T, 1)), feeds the (state, time) tuple to
# its recurrent part (every layer), and ends in the SAME Dense(2) head. Only the
# wiring differs. Capacity is matched per arm on EFFECTIVE parameters by the
# runners (--param-budget), not by a shared size.

import tensorflow as tf

from src.wirings import (CorticalColumnCell, CorticalColumnV2Cell, NCPWiring,
                         multi_column_head)
from src.wirings.tbt_cncp import TbtCorticalColumnCell
from src.tasks.person_activity.model import (
    WIRINGS, COMPOSITE_WIRINGS, DENSE3_LAYERS, MULTI_COLUMN_DEFAULTS,
    default_readout,
    ncp_layer_sizes,
    scaled_lamina_units, scaled_time, stacked_dense, _resolve_cell_cls,
    _reject_multi_state)
from src.tasks.active_sensing.model import TBT_MODES

# The Lotka-Volterra / duffing trajectory task compares the thesis-core topology
# arms (dense, dense3, ncp, cncp -- no reference frame, pure wiring) against the
# tbt_cNCP arms, which additionally receive the phase-space reference-frame code.
# Reusing active_sensing's TBT_MODES keeps ONE source of truth for how the
# location signal reaches the column (gate vs concat), so a finding there
# transfers here.
#   tbt_cncp        gate-only     tbt_cncp_concat  concat-only (strongest arm)
#   tbt_cncp_noloc  no location   tbt_cncp_both    gate + concat
LV_WIRINGS = WIRINGS + tuple(TBT_MODES)


def build_lotka_volterra_model(wiring, cell, size=64, seed=42, feature_size=2,
                               loc_dim=12, lr=1e-3, columns=MULTI_COLUMN_DEFAULTS["columns"],
                               partition=MULTI_COLUMN_DEFAULTS["partition"],
                               coupling=MULTI_COLUMN_DEFAULTS["coupling"],
                               readout=None, dt_scale=1.0, **cell_kwargs):
    """Build and compile a next-step trajectory model for one (wiring, cell).

    Args:
        wiring:       one of LV_WIRINGS: 'dense', 'dense3', 'ncp', 'cncp',
                      'cncp2', 'cncp2_multi'
                      (2-input, no reference frame) or a tbt_cncp* mode
                      (3-input, with the phase-space location signal).
        cell:         _CELL_REGISTRY key (e.g. 'cfc_lrc', 'gru', 'lstm') or
                      BaseCell subclass; the composite wirings (cncp, tbt_cncp*)
                      take single-state cells only.
        size:         the arm's width knob. dense/dense3: RNN units per layer;
                      ncp: inter=size, command=motor=size//2; cncp/tbt: lamina
                      widths scaled by size/16 (person_activity.
                      scaled_lamina_units); cncp2 / cncp2_multi: column
                      width u = size.
                      Derive it per arm from the budget
                      (src.wirings.size_for_budget).
        seed:         seeds tf/numpy/python RNGs (weight init) and the wiring
                      mask generation.
        feature_size: state dimension (2 for the 2D systems).
        loc_dim:      reference-frame code width (datasets.RF_DIM == 12); only
                      used by the tbt_cncp* wirings.
        lr:           RMSprop learning rate.
        columns, partition, coupling, readout: cncp2_multi only (spec B.7);
                      the 2-D state has no sensor structure, so
                      partition='region' splits the state axis (warning).
        dt_scale:     multiplies the time input before it reaches any cell
                      (task-level time unit, issue #28); 1.0 adds no layer.
        **cell_kwargs: forwarded to every cell constructor.

    Returns:
        Compiled tf.keras.Model. Inputs are [state (B,None,feature_size),
        time (B,None,1)] for dense/dense3/ncp/cncp, or [state, time, location
        (B,None,loc_dim)] for the tbt_cncp* wirings; output (B,None,feature_size)
        = predicted next state; loss MSE, metric MAE. Variable time dimension so
        the same graph serves teacher-forced training and closed-loop rollout.
    """
    if wiring not in LV_WIRINGS:
        raise ValueError(f"wiring must be one of {LV_WIRINGS}, got {wiring!r}")
    cell_cls = _resolve_cell_cls(cell)
    if wiring in COMPOSITE_WIRINGS or wiring in TBT_MODES:
        _reject_multi_state(cell_cls, cell_kwargs)

    tf.keras.utils.set_random_seed(seed)

    state = tf.keras.Input(shape=(None, feature_size), name="state")
    time_in = tf.keras.Input(shape=(None, 1), name="time")
    time = scaled_time(time_in, dt_scale)

    if wiring in TBT_MODES:
        concat_loc, use_gate = TBT_MODES[wiring]
        location = tf.keras.Input(shape=(None, loc_dim), name="location")
        feats = (tf.keras.layers.Concatenate()([state, location])
                 if concat_loc else state)
        tbt_cell = TbtCorticalColumnCell(
            cell_cls=cell_cls, lamina_units=scaled_lamina_units(size),
            seed=seed, use_location=use_gate, **cell_kwargs)
        h = tf.keras.layers.RNN(tbt_cell, return_sequences=True)(
            (feats, time, location))
        inputs = [state, time_in, location]
    elif wiring == "cncp":
        cncp_cell = CorticalColumnCell(
            cell_cls=cell_cls, lamina_units=scaled_lamina_units(size),
            seed=seed, **cell_kwargs)
        h = tf.keras.layers.RNN(cncp_cell, return_sequences=True)(
            (state, time))
        inputs = [state, time_in]
    elif wiring == "cncp2":
        col = CorticalColumnV2Cell(cell_cls=cell_cls, units=size, seed=seed,
                                   **cell_kwargs)
        h = tf.keras.layers.RNN(col, return_sequences=True)((state, time))
        inputs = [state, time_in]
    elif wiring == "cncp2_multi":
        h = None                                    # head built below
        next_state = multi_column_head(
            (state, time), feature_size, "next_state", columns=columns,
            cell_cls=cell_cls, units=size, seed=seed, partition=partition,
            coupling=coupling, readout=readout or default_readout(partition),
            **cell_kwargs)
        inputs = [state, time_in]
    elif wiring == "ncp":
        inter, command, motor = ncp_layer_sizes(size)
        ncp = NCPWiring(cell_cls, inter, command, motor, seed=seed,
                        **cell_kwargs)
        h = tf.keras.layers.RNN(ncp.make_cell(), return_sequences=True)(
            (state, time))
        h = ncp.motor_slice()(h)
        inputs = [state, time_in]
    else:  # dense / dense3
        h = stacked_dense(cell_cls, size, state, time,
                          n_layers=DENSE3_LAYERS if wiring == "dense3" else 1,
                          **cell_kwargs)
        inputs = [state, time_in]

    # Linear regression head: predict the next (prey, predator) state. A plain
    # Dense on (B, T, units) applies to the last axis == TimeDistributed here.
    if h is not None:
        next_state = tf.keras.layers.Dense(feature_size, name="next_state")(h)

    model = tf.keras.Model(inputs=inputs, outputs=next_state)
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(lr),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()],
    )
    return model
