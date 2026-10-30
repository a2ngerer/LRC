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

from src.benchmark.optimizers import make_optimizer
from src.wirings import (CorticalColumnCell, CorticalColumnV2Cell, NCPWiring,
                         multi_column_head)
from src.wirings.tbt_cncp import TbtCorticalColumnCell
from src.neurons.forced_rollout_cell import ForcedRolloutCell, StackedCell
from src.tasks.person_activity.model import (
    WIRINGS, COMPOSITE_WIRINGS, DENSE3_LAYERS, MULTI_COLUMN_DEFAULTS,
    NCP_WIRINGS, CNCP_WIRINGS, default_readout,
    ncp_layer_sizes,
    scaled_lamina_units, scaled_time, stacked_dense, _resolve_cell_cls,
    _reject_multi_state, _reject_stateless)
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
# Wirings whose model is (state, time) -> RNN layer(s) -> [motor slice] -> Dense
# head and can therefore be wrapped in a ForcedRolloutCell (issue #25). The
# tbt_cncp* arms recompute the location code from the state and cncp2_multi has
# its own multi-column head; both keep closed_loop_rollout.
FORCEABLE_WIRINGS = ("dense", "dense3", "ncp", "cncp", "cncp2",
                     "ncp_random", "cncp_random")


def build_lotka_volterra_model(wiring, cell, size=64, seed=42, feature_size=2,
                               loc_dim=12, lr=1e-3, columns=MULTI_COLUMN_DEFAULTS["columns"],
                               partition=MULTI_COLUMN_DEFAULTS["partition"],
                               coupling=MULTI_COLUMN_DEFAULTS["coupling"],
                               readout=None, self_recurrent=True, dt_scale=1.0,
                               optimizer='rmsprop', opt_epsilon=None,
                               **cell_kwargs):
    """Build and compile a next-step trajectory model for one (wiring, cell).

    Args:
        wiring:       one of LV_WIRINGS: 'dense', 'dense3', 'ncp', 'cncp',
                      'cncp2', 'cncp2_multi', or the random-rewiring controls
                      'ncp_random' / 'cncp_random' (issue #17)
                      (2-input, no reference frame) or a tbt_cncp* mode
                      (3-input, with the phase-space location signal).
        cell:         _CELL_REGISTRY key (e.g. 'cfc_lrc', 'gru', 'lstm') or
                      BaseCell subclass; the composite wirings (cncp, tbt_cncp*)
                      take single-state cells only, the stateless control
                      'mlp' runs on 'dense' only (issue #60).
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
        lr:           learning rate of the optimizer.
        optimizer, opt_epsilon: src.benchmark.optimizers.make_optimizer
                      ('rmsprop' default = tf.keras.optimizers.RMSprop(lr);
                      'rmsprop_legacy', 'adam'; epsilon None = class
                      default; issue #82).
        columns, partition, coupling, readout: cncp2_multi only (spec B.7);
                      the 2-D state has no sensor structure, so
                      partition='region' splits the state axis (warning).
        self_recurrent: ncp only, see NCPWiring (issue #23).
        dt_scale:     multiplies the time input before it reaches any cell
                      (task-level time unit, issue #28); 1.0 adds no layer.
        **cell_kwargs: forwarded to every cell constructor. ode_unfolds
                      (solver cells, ODE_UNFOLD_CELLS: ltc, lrc, ctrnn, mm_*)
                      is the number of Euler sub-steps per input step; the
                      LV and PA runners pass --ode-unfolds (default
                      DEFAULT_ODE_UNFOLDS = 6) on every wiring (dense,
                      dense3, ncp, cncp, ...).
                      Under 'ncp' fewer than 3 never let the motor slice see
                      the current input. Unset, the constructor default
                      applies (ltc 6; lrc, ctrnn 1).

    Returns:
        Compiled tf.keras.Model. Inputs are [state (B,None,feature_size),
        time (B,None,1)] for dense/dense3/ncp/cncp, or [state, time, location
        (B,None,loc_dim)] for the tbt_cncp* wirings; output (B,None,feature_size)
        = predicted next state; loss MSE, metric MAE. Variable time dimension so
        the same graph serves teacher-forced training and closed-loop rollout.
        ``model.rewire_overlap`` is the share of the original edges the random
        wirings kept (None otherwise).
    """
    if wiring not in LV_WIRINGS:
        raise ValueError(f"wiring must be one of {LV_WIRINGS}, got {wiring!r}")
    cell_cls = _resolve_cell_cls(cell)
    _reject_stateless(cell_cls, wiring)
    if wiring in COMPOSITE_WIRINGS or wiring in TBT_MODES:
        _reject_multi_state(cell_cls, cell_kwargs)

    tf.keras.utils.set_random_seed(seed)

    state = tf.keras.Input(shape=(None, feature_size), name="state")
    time_in = tf.keras.Input(shape=(None, 1), name="time")
    time = scaled_time(time_in, dt_scale)
    rewire_overlap = None

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
    elif wiring in CNCP_WIRINGS:
        cncp_cell = CorticalColumnCell(
            cell_cls=cell_cls, lamina_units=scaled_lamina_units(size),
            seed=seed, rewire=wiring == "cncp_random", **cell_kwargs)
        h = tf.keras.layers.RNN(cncp_cell, return_sequences=True)(
            (state, time))
        rewire_overlap = cncp_cell.rewire_overlap
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
    elif wiring in NCP_WIRINGS:
        inter, command, motor = ncp_layer_sizes(size)
        ncp = NCPWiring(cell_cls, inter, command, motor, seed=seed,
                        self_recurrent=self_recurrent,
                        rewire=wiring == "ncp_random", **cell_kwargs)
        h = tf.keras.layers.RNN(ncp.make_cell(), return_sequences=True)(
            (state, time))
        h = ncp.motor_slice()(h)
        rewire_overlap = ncp.rewire_overlap(feature_size)
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
        optimizer=make_optimizer(optimizer, lr, opt_epsilon),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()],
    )
    model.rewire_overlap = rewire_overlap      # None unless a *_random wiring
    return model


def forced_rollout_model(model, alpha=None):
    """Mixed-forcing view of a plain model from build_lotka_volterra_model.

    The view runs the model's own recurrent core (dense3: its three RNN
    layers as one StackedCell) and readout (ncp: motor slice + head) inside a
    ForcedRolloutCell, so it shares every weight with ``model`` and adds none.
    Same inputs [state, time] and output as the plain model; ``alpha`` (a
    non-trainable tf.Variable) = 1 reproduces its teacher-forced predictions,
    0 the closed-loop rollout. Compiled with the plain model's optimizer, loss
    and metric, so ``--forcing mixed`` trains the view directly.

    ``alpha=None`` gives the anchored view instead: a third input (B, T, 1)
    holds the per-step forcing weight (ForcedRolloutCell), 1 for the
    teacher-forced warm-up up to the anchor and 0 for the free run (#62).
    """
    rnns = [l for l in model.layers if isinstance(l, tf.keras.layers.RNN)]
    readout = [l for l in model.layers
               if l.name in ("motor_neurons", "next_state")]
    scale = [l for l in model.layers if l.name == "dt_scale"]  # dt_scale != 1
    if (len(model.inputs) != 2 or not rnns
            or len(model.layers) != 2 + len(scale) + len(rnns) + len(readout)):
        raise ValueError("forced_rollout_model needs a (state, time) -> RNN "
                         f"-> head model, i.e. one of {FORCEABLE_WIRINGS}")
    core = rnns[0].cell if len(rnns) == 1 else StackedCell(
        [l.cell for l in rnns])
    inputs = [tf.keras.Input(shape=t.shape[1:]) for t in model.inputs]
    state, time = inputs
    if scale:                       # the plain model's own time rescaling
        time = scale[0](time)
    rnn_in = (state, time)
    if alpha is None:
        inputs.append(tf.keras.Input(shape=(None, 1), name="force"))
        rnn_in += (inputs[-1],)
    out = tf.keras.layers.RNN(ForcedRolloutCell(core, readout, alpha),
                              return_sequences=True)(rnn_in)
    view = tf.keras.Model(inputs=inputs, outputs=out)
    view.compile(optimizer=model.optimizer, loss=model.loss,
                 metrics=[tf.keras.metrics.MeanAbsoluteError()])
    return view
