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
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from src.neurons import CfC_Cell, CfC_LRC_Cell
from src.wirings import size_for_budget
from src.tasks.lotka_volterra import (build_lotka_volterra_model,
                                      load_lotka_volterra, LotkaVolterraData,
                                      reference_frame, RF_DIM, LV_WIRINGS)
from src.tasks.active_sensing.model import TBT_MODES
from src.tasks.person_activity.model import (backbone_override,
                                             effective_backbone_layers)
from experiments.run_lotka_volterra_benchmark import (closed_loop_rollout,
                                                      result_filename)
from experiments import run_lotka_volterra_benchmark as lv_runner

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

@pytest.mark.parametrize("wiring", ["cncp", "ncp", "dense", "dense3"])
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

@pytest.mark.parametrize("wiring", ["cncp", "ncp", "dense", "dense3"])
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

def test_arms_budget_matched_on_effective_params():
    """Each arm budget-matched on its own knob (EFFECTIVE parameters, masked-off
    weights excluded) -- see tests/tasks/person_activity and
    docs/ncp-wiring-fix-2026-09-17.md."""
    def build(wiring):
        return lambda size: build_lotka_volterra_model(
            wiring, "cfc_lrc", size=size, seed=0)
    counts = {w: size_for_budget(build(w), 4000, range(2, 129))[1]
              for w in ("dense", "dense3", "ncp", "cncp")}
    lo, hi = min(counts.values()), max(counts.values())
    assert hi / lo < 1.5, f"budget-matched effective counts {counts}"


def test_multi_state_cells_rejected_for_the_composite_wirings():
    for wiring in ("cncp", "tbt_cncp_concat"):
        with pytest.raises(ValueError, match="single-state"):
            build_lotka_volterra_model(wiring, "lstm", size=16, seed=0)


@pytest.mark.parametrize("wiring", ["dense", "dense3", "ncp"])
def test_multi_state_cells_run_on_the_non_composite_wirings(wiring):
    model = build_lotka_volterra_model(wiring, "lstm", size=8, seed=0)
    x, t, _ = _dummy_batch()
    assert model([x, t]).shape == (2, 16, FEATURES)


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


# --- --backbone-layers override (V2s control) ---

def _cfc_cells(model):
    """Every CfC-family cell of a built model (dense layers, NCP layers, cNCP nodes)."""
    return [l for l in model._flatten_layers()
            if isinstance(l, (CfC_Cell, CfC_LRC_Cell))]


def _depths(wiring, backbone_layers, cell="cfc_lrc"):
    kw = backbone_override(cell, backbone_layers)
    model = build_lotka_volterra_model(wiring, cell, size=16, seed=0, **kw)
    return model, [len(c._backbone) for c in _cfc_cells(model)]


def test_cncp_backbone_override_reaches_every_subcell():
    """cncp defaults to backbone-free sub-cells; backbone_layers=1 gives every
    RNN sub-cell (L4, L23, L5IT, L5ET, L6CC, L6CT; Thal/TRN are relays) a
    backbone layer and changes the parameter count. The direction is NOT 'more':
    a cNCP sub-cell sees [x, h] of width 2*units, so the shared backbone layer
    plus four units-wide heads (6u^2) is smaller than four 2u-wide heads (8u^2);
    hence the budget matcher must be given the override."""
    m0, d0 = _depths("cncp", 0)
    m1, d1 = _depths("cncp", 1)
    assert d0 == [0] * 6
    assert d1 == [1] * 6
    assert m1.count_params() != m0.count_params()


def test_dense_backbone_zero_has_none_and_default_is_unchanged():
    assert _depths("dense", 0)[1] == [0]
    assert _depths("dense3", 0)[1] == [0, 0, 0]
    assert _depths("dense", None)[1] == [1]          # builder default untouched
    assert _depths("ncp", None)[1] == [0, 0, 0]      # family default untouched


@pytest.mark.parametrize("backbone_layers", [None, 0, 2])
@pytest.mark.parametrize("wiring", ["dense", "dense3", "ncp", "cncp",
                                    "tbt_cncp_concat"])
def test_effective_backbone_matches_the_built_cells(wiring, backbone_layers):
    """The value recorded in the result JSON is what the cells really got."""
    kw = backbone_override("cfc_lrc", backbone_layers)
    effective = effective_backbone_layers(wiring, "cfc_lrc", kw)
    assert set(_depths(wiring, backbone_layers)[1]) == {effective}


def test_backbone_override_skips_cells_without_backbone():
    assert backbone_override("gru", 1) == {}          # gru rejects the kwarg
    assert effective_backbone_layers("dense", "gru", {}) is None
    assert backbone_override("cfc_lrc", None) == {}
    with pytest.raises(ValueError):
        backbone_override("cfc_lrc", -1)


@pytest.mark.parametrize("wiring", ["dense", "cncp"])
def test_budget_matcher_sees_the_backbone_override(wiring):
    def size_at(b):
        kw = backbone_override("cfc_lrc", b)
        return size_for_budget(lambda s: build_lotka_volterra_model(
            wiring, "cfc_lrc", size=s, seed=0, **kw), 4000, range(2, 129))[0]
    assert size_at(0) != size_at(1)      # the matched width follows the backbone


def test_result_filename_backbone_suffix():
    a = SimpleNamespace(cell="cfc_lrc", wiring="cncp", seed=1,
                        backbone_layers=None)
    assert result_filename(a) == "cfc_lrc_cncp_seed1.json"
    a.backbone_layers = 1
    assert result_filename(a) == "cfc_lrc_cncp_seed1_bb1.json"
    a.backbone_layers = 0
    assert result_filename(a) == "cfc_lrc_cncp_seed1_bb0.json"


def test_runner_records_backbone_end_to_end(tmp_path, monkeypatch):
    """Tiny real run: flag -> budget matcher -> JSON (requested + effective)."""
    argv = ["run_lotka_volterra_benchmark.py", "--wiring", "cncp", "--cell",
            "cfc_lrc", "--epochs", "1", "--n-trajectories", "6", "--seq-len",
            "6", "--param-budget", "800", "--backbone-layers", "1",
            "--outdir", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", argv)
    lv_runner.main()
    rec = json.loads((tmp_path / "cfc_lrc_cncp_seed0_bb1.json").read_text())
    assert rec["backbone_layers"] == 1
    assert rec["backbone_layers_effective"] == 1
