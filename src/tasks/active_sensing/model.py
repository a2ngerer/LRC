# Active-sensing model builder: one compiled Keras model per (wiring, cell) for
# the tbt_cNCP object-recognition task (concept spec 2026-07-04-tbt-cncp).
#
# Every wiring receives the same three inputs -- patch (B,T,P), time (B,T,1),
# location (B,T,L) -- and ends in the SAME per-step Dense(num_classes) object
# head, with an optional Dense(P) next-patch prediction head (self-supervised).
# Only the wiring in between differs.
#
# Wirings:
#   tbt_cncp        TbtCorticalColumnCell, location gates L4 (the model);
#   tbt_cncp_noloc  same cell, use_location=False (ablation: no reference frame);
#   ncp / dense     baselines fed concat(patch, location) as the sensory input.

import tensorflow as tf

from src.wirings import NCPWiring
from src.wirings.tbt_cncp import TbtCorticalColumnCell, MultiColumnVotingCell
from src.tasks.person_activity.model import (
    scaled_lamina_units, ncp_layer_sizes, _resolve_cell_cls,
    _reject_multi_state)

# tbt wirings -> (concat_location_into_input, location_mode). This factorises
# HOW the location signal reaches the column, so topology and mechanism can be
# separated. location_mode is passed to TbtCorticalColumnCell.use_location:
# False = off, True/'gate' = multiplicative L6a->L4 gain, 'film' = affine
# scale+shift conditioning.
#   tbt_cncp         gate only   (TBT-faithful L6a->L4 multiplicative gating)
#   tbt_cncp_noloc   neither     (ablation: no reference frame at all)
#   tbt_cncp_concat  concat only (location as additive input, same MECHANISM as
#                                 ncp/dense, on the cNCP TOPOLOGY)
#   tbt_cncp_both    concat + gate
#   tbt_cncp_film    FiLM gate    (affine L6a->L4: adds the location-prior shift
#                                  the pure multiplicative gate cannot express)
# ncp/dense receive concat(patch, location) too, so ncp/dense vs tbt_cncp_concat
# isolates topology and tbt_cncp vs tbt_cncp_concat isolates mechanism; tbt_cncp
# vs tbt_cncp_film isolates gate expressiveness (multiplicative vs affine).
TBT_MODES = {
    "tbt_cncp": (False, True),
    "tbt_cncp_noloc": (False, False),
    "tbt_cncp_concat": (True, False),
    "tbt_cncp_both": (True, True),
    "tbt_cncp_film": (False, "film"),
}
ACTIVE_WIRINGS = tuple(TBT_MODES) + ("ncp", "dense")


def build_active_sensing_model(wiring, cell, size=64, seed=42, num_classes=6,
                               patch_dim=25, loc_dim=12, lr=1e-3,
                               use_prediction=True, pred_weight=0.3,
                               **cell_kwargs):
    """Build + compile an active-sensing model for one (wiring, cell).

    Returns a tf.keras.Model with inputs [patch (B,None,P), time (B,None,1),
    location (B,None,L)] and output per-step object logits (B,None,num_classes)
    (named "logits"); if use_prediction, also a next-patch head (B,None,P) named
    "pred". Loss: SparseCategoricalCrossentropy(from_logits) [+ pred_weight*MSE].
    """
    if wiring not in ACTIVE_WIRINGS:
        raise ValueError(f"wiring must be one of {ACTIVE_WIRINGS}, got {wiring!r}")
    cell_cls = _resolve_cell_cls(cell)
    _reject_multi_state(cell_cls, cell_kwargs)
    tf.keras.utils.set_random_seed(seed)

    patch = tf.keras.Input(shape=(None, patch_dim), name="patch")
    time = tf.keras.Input(shape=(None, 1), name="time")
    location = tf.keras.Input(shape=(None, loc_dim), name="location")

    if wiring in TBT_MODES:
        concat_loc, use_gate = TBT_MODES[wiring]
        feats = (tf.keras.layers.Concatenate()([patch, location])
                 if concat_loc else patch)
        cellobj = TbtCorticalColumnCell(
            cell_cls=cell_cls, lamina_units=scaled_lamina_units(size),
            seed=seed, use_location=use_gate, **cell_kwargs)
        h = tf.keras.layers.RNN(cellobj, return_sequences=True)(
            (feats, time, location))
    elif wiring == "ncp":
        feats = tf.keras.layers.Concatenate()([patch, location])
        inter, command, motor = ncp_layer_sizes(size)
        ncp = NCPWiring(cell_cls, inter, command, motor, seed=seed,
                        **cell_kwargs)
        h = tf.keras.layers.RNN(ncp.make_cell(), return_sequences=True)(
            (feats, time))
        h = ncp.motor_slice()(h)
    else:  # dense
        feats = tf.keras.layers.Concatenate()([patch, location])
        h = tf.keras.layers.RNN(
            cell_cls(units=size, **cell_kwargs), return_sequences=True)(
            (feats, time))

    logits = tf.keras.layers.Dense(num_classes, name="logits")(h)
    outputs = [logits]
    losses = {"logits": tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)}
    loss_weights = {"logits": 1.0}
    metrics = {"logits": tf.keras.metrics.SparseCategoricalAccuracy(name="acc")}
    if use_prediction:
        pred = tf.keras.layers.Dense(patch_dim, name="pred")(h)
        outputs.append(pred)
        losses["pred"] = tf.keras.losses.MeanSquaredError()
        loss_weights["pred"] = pred_weight

    model = tf.keras.Model(inputs=[patch, time, location], outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=losses,
                  loss_weights=loss_weights, metrics=metrics)
    return model


def build_voting_model(n_columns, cell, size=64, seed=42, num_classes=6,
                       patch_dim=25, loc_dim=12, lr=1e-3,
                       location_mode="concat", **cell_kwargs):
    """Build + compile a K-column voting tbt_cNCP model (TBT ingredient 3).

    K weight-shared columns each read their own glimpse stream and vote through
    L2/3 every step (MultiColumnVotingCell). This isolates the effect of VOTING
    on top of a chosen single-column location mechanism.

    Args:
        location_mode: how the reference frame reaches each column --
            'concat' appends location to each column's patch (column gate off);
            'film'/'gate' inject location into L4 inside the column (features
            stay raw). Iteration 1 found FiLM the strongest single-column path.

    Inputs [patch (B,None,K,P), time (B,None,1), location (B,None,K,L)]; output
    per-step object logits (B,None,num_classes). Classification only.
    """
    cell_cls = _resolve_cell_cls(cell)
    _reject_multi_state(cell_cls, cell_kwargs)
    tf.keras.utils.set_random_seed(seed)

    patch = tf.keras.Input(shape=(None, n_columns, patch_dim), name="patch")
    time = tf.keras.Input(shape=(None, 1), name="time")
    location = tf.keras.Input(shape=(None, n_columns, loc_dim), name="location")

    if location_mode == "concat":
        feats = tf.keras.layers.Concatenate(axis=-1)([patch, location])
        col_use_loc = False
    else:  # 'film' or 'gate' -> location injected into L4, features stay raw
        feats = patch
        col_use_loc = location_mode
    cellobj = MultiColumnVotingCell(
        n_columns=n_columns, cell_cls=cell_cls,
        lamina_units=scaled_lamina_units(size), seed=seed,
        use_location=col_use_loc, **cell_kwargs)
    h = tf.keras.layers.RNN(cellobj, return_sequences=True)(
        (feats, time, location))
    logits = tf.keras.layers.Dense(num_classes, name="logits")(h)

    model = tf.keras.Model(inputs=[patch, time, location], outputs=logits)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")])
    return model
