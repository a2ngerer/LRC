# tests/tasks/lotka_volterra/test_lotka_volterra.py
# Validation of the predator-prey (Lotka-Volterra) sequence-rollout benchmark:
#   - the dataset loader returns the documented shapes and is deterministic;
#   - build_lotka_volterra_model builds + forward-passes for every wiring with
#     the (state, time) two-input contract and a linear Dense(2) head;
#   - EVERY cncp trainable variable receives non-zero gradient on the rollout
#     task -- the property the neural_ode T=1 harness destroys (delayed-state
#     edges dead there, alive here), which is the whole reason this task exists;
#   - cncp/ncp parameter counts stay comparable (fairness invariant);
#   - multi-state cells are rejected;
#   - closed-loop rollout produces the documented shape.
import numpy as np
import pytest
import tensorflow as tf

from src.wirings import effective_param_count, match_param_budget
from src.tasks.lotka_volterra import (build_lotka_volterra_model,
                                      load_lotka_volterra, LotkaVolterraData,
                                      reference_frame, RF_DIM, LV_WIRINGS)
from src.tasks.active_sensing.model import TBT_MODES
from experiments.run_lotka_volterra_benchmark import closed_loop_rollout

FEATURES = 2


def _dummy_batch(batch=2, seq_len=16, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(batch, seq_len, FEATURES)).astype(np.float32)
    t = rng.uniform(0.05, 0.2, size=(batch, seq_len, 1)).astype(np.float32)
    y = rng.normal(size=(batch, seq_len, FEATURES)).astype(np.float32)
    return x, t, y


def _dummy_loc(batch=2, seq_len=16, loc_dim=RF_DIM, seed=1):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(batch, seq_len, loc_dim)).astype(np.float32)


# --- dataset ---

def test_dataset_shapes_and_split():
    d = load_lotka_volterra(n_trajectories=20, seq_len=32)
    assert isinstance(d, LotkaVolterraData)
    n_train, n_test = d.train_x.shape[0], d.test_x.shape[0]
    assert n_train + n_test == 20
    assert n_test == 4                          # 20% of 20
    assert d.train_x.shape == (n_train, 32, 2)
    assert d.train_y.shape == (n_train, 32, 2)
    assert d.train_t.shape == (n_train, 32, 1)
    assert d.train_traj.shape == (n_train, 33, 2)   # seq_len + 1 raw points
    assert d.feature_size == 2 and d.seq_len == 32
    assert d.dt > 0
    # next-step target is the raw trajectory shifted by one step
    assert np.allclose(d.denormalise(d.train_x)[:, 1:, :],
                       d.denormalise(d.train_y)[:, :-1, :], atol=1e-4)


def test_dataset_deterministic():
    a = load_lotka_volterra(n_trajectories=16, seq_len=24)
    b = load_lotka_volterra(n_trajectories=16, seq_len=24)
    assert np.array_equal(a.train_x, b.train_x)
    assert np.array_equal(a.test_traj, b.test_traj)


# --- model build + forward pass ---

@pytest.mark.parametrize("wiring", ["cncp", "ncp", "dense"])
@pytest.mark.parametrize("cell", ["cfc_lrc", "gru"])
def test_build_and_forward_shapes(wiring, cell):
    model = build_lotka_volterra_model(wiring, cell, size=16, seed=0)
    x, t, _ = _dummy_batch()
    out = model([x, t])
    assert out.shape == (2, 16, FEATURES)       # linear next-state head


def test_variable_length_input():
    # Variable seq length is required for closed-loop rollout on a growing
    # history: the same graph must accept lengths it was not trained on.
    model = build_lotka_volterra_model("cncp", "cfc_lrc", size=16, seed=0)
    for L in (4, 9, 20):
        x, t, _ = _dummy_batch(seq_len=L)
        assert model([x, t]).shape == (2, L, FEATURES)


# --- the scientific core: recurrence trains here ---

@pytest.mark.parametrize("wiring", ["cncp", "ncp", "dense"])
def test_all_variables_receive_gradient(wiring):
    model = build_lotka_volterra_model(wiring, "cfc_lrc", size=16, seed=0)
    x, t, y = _dummy_batch(seq_len=16)
    with tf.GradientTape() as tape:
        pred = model((tf.constant(x), tf.constant(t)), training=True)
        loss = tf.reduce_mean((pred - y) ** 2)
    grads = tape.gradient(loss, model.trainable_variables)
    dead = [v.name for v, g in zip(model.trainable_variables, grads)
            if g is None or float(tf.reduce_sum(tf.abs(g))) == 0.0]
    # On the full-sequence rollout every parameter -- including the cncp
    # delayed-state edges -- must get gradient (contrast: the T=1 ODE harness
    # leaves the delayed-state edges dead; see the design spec sec 12).
    assert dead == [], f"{wiring}: dead variables {dead}"


# --- fairness + guards ---

def test_cncp_ncp_parameter_fairness():
    """Each arm budget-matched on its own knob (EFFECTIVE parameters, masked-off
    weights excluded) -- see tests/tasks/person_activity and
    docs/ncp-wiring-fix-2026-09-17.md."""
    def count(wiring):
        return lambda size: effective_param_count(
            build_lotka_volterra_model(wiring, "cfc_lrc", size=size, seed=0))
    n_cncp = match_param_budget(count("cncp"), 4000, range(2, 129))[1]
    n_ncp = match_param_budget(count("ncp"), 4000, range(2, 129))[1]
    ratio = n_cncp / n_ncp
    assert 1 / 1.5 <= ratio <= 1.5, f"param ratio {ratio:.3f} out of band"


def test_multi_state_cells_rejected():
    with pytest.raises(ValueError):
        build_lotka_volterra_model("cncp", "lstm", size=16, seed=0)


def test_invalid_wiring_rejected():
    with pytest.raises(ValueError):
        build_lotka_volterra_model("bogus", "cfc_lrc", size=16, seed=0)


# --- closed-loop rollout ---

def test_closed_loop_rollout_shape():
    d = load_lotka_volterra(n_trajectories=12, seq_len=16)
    model = build_lotka_volterra_model("cncp", "cfc_lrc", size=16, seed=0)
    gen = closed_loop_rollout(model, d)
    assert gen.shape == (d.test_x.shape[0], d.seq_len + 1, 2)
    # column 0 is the given (normalised) initial state, verbatim
    assert np.allclose(gen[:, 0, :], d.test_x[:, 0, :], atol=1e-5)


# --- reference frame (TBT location signal) + tbt_cncp wirings ---

def test_reference_frame_shape_and_locality():
    s = np.array([[0.2, 0.2], [0.25, 0.25], [2.0, -1.5]], dtype=np.float32)
    code = reference_frame(s)
    assert code.shape == (3, RF_DIM)
    # grid-cell property: nearby states map to more similar codes
    assert (np.linalg.norm(code[0] - code[1])
            < np.linalg.norm(code[0] - code[2]))


def test_loader_emits_reference_frame():
    d = load_lotka_volterra(n_trajectories=16, seq_len=24)
    assert d.loc_dim == RF_DIM
    assert d.train_loc.shape == (d.train_x.shape[0], 24, RF_DIM)
    assert d.test_loc.shape == (d.test_x.shape[0], 24, RF_DIM)
    # the location code is a deterministic function of the normalised state
    assert np.allclose(d.train_loc, reference_frame(d.train_x), atol=1e-6)


def test_duffing_system_loads():
    d = load_lotka_volterra(n_trajectories=16, seq_len=24, system="duffing")
    assert d.system == "duffing"
    assert d.train_x.shape[0] + d.test_x.shape[0] == 16
    assert np.isfinite(d.train_traj).all()


def test_lv_wirings_membership():
    assert {"dense", "ncp", "cncp"}.issubset(LV_WIRINGS)
    assert set(TBT_MODES).issubset(LV_WIRINGS)


@pytest.mark.parametrize("wiring", ["tbt_cncp", "tbt_cncp_concat",
                                    "tbt_cncp_noloc", "tbt_cncp_both"])
def test_tbt_build_and_forward(wiring):
    model = build_lotka_volterra_model(wiring, "cfc_lrc", size=16, seed=0)
    x, t, _ = _dummy_batch()
    loc = _dummy_loc()
    out = model([x, t, loc])                    # 3-input contract for tbt
    assert out.shape == (2, 16, FEATURES)


def test_tbt_gate_location_weight_gets_gradient():
    # Gate mode (use_location=True) exercises W_loc_L4 -> it, and every other
    # trainable variable, must receive gradient on the rollout task.
    model = build_lotka_volterra_model("tbt_cncp", "cfc_lrc", size=16, seed=0)
    x, t, y = _dummy_batch(seq_len=16)
    loc = _dummy_loc(seq_len=16)
    with tf.GradientTape() as tape:
        pred = model((tf.constant(x), tf.constant(t), tf.constant(loc)),
                     training=True)
        loss = tf.reduce_mean((pred - y) ** 2)
    grads = tape.gradient(loss, model.trainable_variables)
    names = [v.name for v in model.trainable_variables]
    dead = [n for n, g in zip(names, grads)
            if g is None or float(tf.reduce_sum(tf.abs(g))) == 0.0]
    assert dead == [], f"dead variables: {dead}"
    assert any("W_loc_L4" in n for n in names)


def test_tbt_closed_loop_rollout_shape():
    d = load_lotka_volterra(n_trajectories=12, seq_len=16)
    model = build_lotka_volterra_model("tbt_cncp_concat", "cfc_lrc", size=16,
                                       seed=0)
    gen = closed_loop_rollout(model, d, needs_loc=True)
    assert gen.shape == (d.test_x.shape[0], d.seq_len + 1, 2)
    assert np.allclose(gen[:, 0, :], d.test_x[:, 0, :], atol=1e-5)
