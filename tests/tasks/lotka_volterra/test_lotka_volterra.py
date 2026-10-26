# tests/tasks/lotka_volterra/test_lotka_volterra.py
# Validation of the predator-prey (Lotka-Volterra) sequence-rollout benchmark:
#   - the dataset loader returns the documented shapes, splits train/val/test
#     by whole trajectories and is deterministic per data seed (fresh draw per
#     seed);
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
                                      reference_frame, RF_DIM, LV_WIRINGS,
                                      DATA_SEED_BASE)
from src.tasks.active_sensing.model import TBT_MODES
from src.tasks.person_activity.model import (backbone_override,
                                             budget_sizes,
                                             effective_backbone_layers,
                                             effective_self_recurrent,
                                             lamina_sizes,
                                             scaled_lamina_units)
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
    n_train = d.train_x.shape[0]
    assert (n_train, d.val_x.shape[0], d.test_x.shape[0]) == (12, 4, 4)  # 60/20/20
    assert d.val_traj.shape == (4, 33, 2)
    assert d.val_loc.shape == (4, 32, RF_DIM)
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


def _all_traj(d):
    return np.concatenate([d.train_traj, d.val_traj, d.test_traj])


def test_data_seed_draws_fresh_trajectories():
    kw = dict(n_trajectories=10, seq_len=8, split=(6, 2, 2))
    a, b = (load_lotka_volterra(seed=DATA_SEED_BASE, **kw) for _ in range(2))
    for split in ("train", "val", "test"):
        for k in ("x", "y", "traj"):
            assert np.array_equal(getattr(a, f"{split}_{k}"),
                                  getattr(b, f"{split}_{k}"))
    c = load_lotka_volterra(seed=DATA_SEED_BASE + 1, **kw)
    # a fresh draw: no initial condition of seed+1 occurs in seed's draw
    assert not np.isin(_all_traj(c)[:, 0], _all_traj(a)[:, 0]).any()


def test_split_is_disjoint_by_whole_trajectory():
    d = load_lotka_volterra(n_trajectories=80, seq_len=8, split=(48, 16, 16))
    sizes = (len(d.train_traj), len(d.val_traj), len(d.test_traj))
    assert sizes == (48, 16, 16)
    # 80 distinct initial conditions across the three splits: no trajectory
    # is in two splits
    assert len(np.unique(_all_traj(d)[:, 0], axis=0)) == 80
    # z-score from train only
    raw_train = d.denormalise(d.train_x).reshape(-1, 2)
    assert np.allclose(d.mean, raw_train.mean(axis=0), atol=1e-4)
    assert not np.allclose(d.mean, d.denormalise(d.test_x).reshape(-1, 2)
                           .mean(axis=0), atol=1e-4)


@pytest.mark.parametrize("split", [(48, 16, 15), (64, 16, 0)])
def test_split_must_sum_to_n_trajectories(split):
    with pytest.raises(ValueError, match="split"):
        load_lotka_volterra(n_trajectories=80, seq_len=8, split=split)


# --- model build + forward pass ---

@pytest.mark.parametrize("wiring", ["cncp", "ncp", "dense", "dense3", "cncp2"])
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
              for w in ("dense", "dense3", "ncp", "cncp", "cncp2")}
    lo, hi = min(counts.values()), max(counts.values())
    assert hi / lo < 1.5, f"budget-matched effective counts {counts}"


def test_multi_state_cells_rejected_for_the_composite_wirings():
    for wiring in ("cncp", "cncp2", "tbt_cncp_concat"):
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

@pytest.mark.parametrize("split", ["test", "val"])
def test_closed_loop_rollout_shape(split):
    d = load_lotka_volterra(n_trajectories=12, seq_len=16, split=(6, 2, 4))
    model = build_lotka_volterra_model("cncp", "cfc_lrc", size=16, seed=0)
    gen = closed_loop_rollout(model, d, split=split)
    x = getattr(d, f"{split}_x")
    assert gen.shape == (x.shape[0], d.seq_len + 1, 2)
    # column 0 is the given (normalised) initial state, verbatim
    assert np.allclose(gen[:, 0, :], x[:, 0, :], atol=1e-5)


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
    assert len(d.train_x) + len(d.val_x) + len(d.test_x) == 16
    assert np.isfinite(d.train_traj).all()


def test_lv_wirings_membership():
    assert {"dense", "ncp", "cncp", "cncp2"}.issubset(LV_WIRINGS)
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
    """backbone_layers=0 gives backbone-free cncp sub-cells; 1 (the default
    since issue #21) gives every RNN sub-cell (L4, L23, L5IT, L5ET, L6CC,
    L6CT; Thal/TRN are relays) a backbone layer and changes the parameter
    count. The direction is NOT 'more': a cNCP sub-cell sees [x, h] of width
    2*units, so the shared backbone layer plus four units-wide heads (6u^2) is
    smaller than four 2u-wide heads (8u^2); hence the budget matcher must be
    given the override."""
    m0, d0 = _depths("cncp", 0)
    m1, d1 = _depths("cncp", 1)
    assert d0 == [0] * 6
    assert d1 == [1] * 6
    assert m1.count_params() != m0.count_params()


def test_dense_backbone_zero_has_none_and_default_is_unchanged():
    assert _depths("dense", 0)[1] == [0]
    assert _depths("dense3", 0)[1] == [0, 0, 0]
    assert _depths("dense", None)[1] == [1]          # builder default untouched
    assert _depths("ncp", None)[1] == [1, 1, 1]      # family default (#21)


@pytest.mark.parametrize("backbone_layers", [None, 0, 2])
@pytest.mark.parametrize("wiring", ["dense", "dense3", "ncp", "cncp",
                                    "tbt_cncp_concat"])
def test_effective_backbone_matches_the_built_cells(wiring, backbone_layers):
    """The value recorded in the result JSON is what the cells really got."""
    kw = backbone_override("cfc_lrc", backbone_layers)
    effective = effective_backbone_layers(wiring, "cfc_lrc", kw)
    assert set(_depths(wiring, backbone_layers)[1]) == {effective}


@pytest.mark.parametrize("cell", ["cfc", "cfc_lrc", "cfc_lrc_outer"])
@pytest.mark.parametrize("wiring", ["ncp", "cncp", "cncp2", "cncp2_multi"])
def test_closed_form_subcells_default_to_one_backbone_layer(wiring, cell):
    """Issue #21: without --backbone-layers every closed-form sub-cell gets one
    backbone layer (pilot decision 2026-10-21); 0 stays reachable."""
    assert effective_backbone_layers(wiring, cell, {}) == 1
    assert set(_depths(wiring, None, cell)[1]) == {1}
    assert effective_backbone_layers(wiring, cell,
                                     backbone_override(cell, 0)) == 0
    assert set(_depths(wiring, 0, cell)[1]) == {0}


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
    # budget flag next to the deviation (#59)
    assert rec["budget_above_tolerance"] is (rec["budget_deviation"] > 0.05)
    assert rec["config"]["effective"]["build"]["size"] == rec["size"]


def test_runner_head_activation_reaches_every_subcell(tmp_path, monkeypatch):
    """Tiny real run of the #24 pilot arm (cncp, bb0, tanh): the flag reaches
    ff1/ff2 of every sub-cell (gate heads stay linear) and the JSON."""
    built = []

    def spy(*args, **kwargs):
        built.append(build_lotka_volterra_model(*args, **kwargs))
        return built[-1]

    monkeypatch.setattr(lv_runner, "build_lotka_volterra_model", spy)
    monkeypatch.setattr(sys, "argv", [
        "run_lotka_volterra_benchmark.py", "--wiring", "cncp", "--cell",
        "cfc_lrc", "--epochs", "1", "--n-trajectories", "6", "--seq-len", "6",
        "--size", "8", "--backbone-layers", "0", "--head-activation", "tanh",
        "--outdir", str(tmp_path)])
    lv_runner.main()
    cells = _cfc_cells(built[-1])
    assert cells
    assert {c._ff1.activation.__name__ for c in cells} == {"tanh"}
    assert {c._ff2.activation.__name__ for c in cells} == {"tanh"}
    assert {c._time_a.activation.__name__ for c in cells} == {"linear"}
    rec = json.loads((tmp_path / "cfc_lrc_cncp_seed0_bb0.json").read_text())
    assert rec["head_activation"] == "tanh"


# --- --no-self-recurrence (issue #23 stability pilot) ---

def test_effective_self_recurrent_only_for_closed_form_ncp():
    assert effective_self_recurrent("ncp", "cfc_lrc") is True
    assert effective_self_recurrent("ncp", "cfc_lrc", False) is False
    assert effective_self_recurrent("ncp", "gru") is None
    assert effective_self_recurrent("ncp", "ltc") is None
    assert effective_self_recurrent("dense", "cfc_lrc") is None


def test_runner_records_self_recurrence_end_to_end(tmp_path, monkeypatch):
    """--no-self-recurrence reaches the NCP sub-cells (fewer effective params
    at the same size) and the JSON."""
    def run(*flags):
        monkeypatch.setattr(sys, "argv", [
            "run_lotka_volterra_benchmark.py", "--wiring", "ncp", "--cell",
            "cfc_lrc", "--epochs", "1", "--n-trajectories", "6", "--seq-len",
            "6", "--size", "8", "--outdir", str(tmp_path), *flags])
        lv_runner.main()
        return json.loads((tmp_path / "cfc_lrc_ncp_seed0.json").read_text())
    on, off = run(), run("--no-self-recurrence")
    assert on["self_recurrent"] is True and off["self_recurrent"] is False
    assert on["params_effective"] > off["params_effective"]


# --- val/test reporting and the per-replicate data seed (issue #19) ---

def _run(tmp_path, monkeypatch, wiring, seed, *extra):
    monkeypatch.setattr(sys, "argv", [
        "run_lotka_volterra_benchmark.py", "--wiring", wiring, "--cell", "gru",
        "--size", "4", "--seed", str(seed), "--epochs", "2",
        "--n-trajectories", "10", "--seq-len", "16", "--batch-size", "4",
        "--outdir", str(tmp_path), *extra])
    lv_runner.main()
    return json.loads((tmp_path / f"gru_{wiring}_seed{seed}.json").read_text())


def test_runner_reports_val_and_test_on_paired_draws(tmp_path, monkeypatch):
    a = _run(tmp_path / "a", monkeypatch, "dense", 3)
    b = _run(tmp_path / "b", monkeypatch, "ncp", 3)
    c = _run(tmp_path / "c", monkeypatch, "dense", 4)
    assert a["split"] == [6, 2, 2]
    assert a["data_seed"] == b["data_seed"] == DATA_SEED_BASE + 3   # paired arms
    assert c["data_seed"] == DATA_SEED_BASE + 4                      # fresh draw
    assert a["ground_truth_traj"] == b["ground_truth_traj"]
    assert a["ground_truth_traj"] != c["ground_truth_traj"]
    for key in ("teacher_forced_mse", "teacher_forced_nrmse",
                "closed_loop_mse", "closed_loop_nrmse", "persistence_mse"):
        assert np.isfinite(a[f"val_{key}"]) and np.isfinite(a[f"test_{key}"])
        assert a[key] == a[f"test_{key}"]           # unprefixed = test (legacy)
    assert a["val_closed_loop_nrmse"] != a["test_closed_loop_nrmse"]
    assert np.isclose(a["test_closed_loop_nrmse"] ** 2, a["test_closed_loop_mse"])
    # best = test at the best-VAL epoch (1-based)
    assert a["best_val_epoch"] in (1, 2)
    assert np.isclose(a["best_val_teacher_forced_mse"],
                      min(a["val_loss_curve"]), atol=1e-6)
    assert np.isfinite(a["best_test_teacher_forced_mse"])
    eff = a["config"]["effective"]
    assert eff["data_seed"] == a["data_seed"] and eff["split"] == a["split"]


def test_runner_data_seed_offset_and_override(tmp_path, monkeypatch):
    tuning = _run(tmp_path / "t", monkeypatch, "dense", 0, "--data-seed-offset")
    assert tuning["data_seed"] == DATA_SEED_BASE + 1000
    fixed = _run(tmp_path / "f", monkeypatch, "dense", 0, "--data-seed", "7")
    assert fixed["data_seed"] == 7


# --- finer cNCP budget knob (#59) ---

def test_lamina_knob_keeps_integer_sizes_and_grows_one_node_per_step():
    assert list(scaled_lamina_units(18).values()) == [9, 9, 7, 9, 4, 4, 4, 2]
    assert list(scaled_lamina_units(50).values()) == [25, 25, 19, 25, 12, 12,
                                                      12, 6]
    sizes = lamina_sizes(range(18, 19))
    assert sizes[0] == 18 and all(18 < s < 19 for s in sizes[1:])
    totals = [sum(scaled_lamina_units(s).values()) for s in sizes + [19]]
    assert totals == list(range(48, 55))      # one node +1 per scan step
    assert 18.5 in budget_sizes("cncp") and 18.5 in budget_sizes("tbt_cncp")
    assert budget_sizes("dense") == range(2, 513)


@pytest.mark.parametrize("cell", ["cfc_lrc", "cfc", "gru"])
def test_fractional_cncp_scan_reaches_4k_within_tolerance(cell):
    """Integer sizes overshoot 4k by 6.7 to 8.1 % on cncp (bb1); the
    fractional knob lands within 5 %."""
    kw = backbone_override(cell, 1)
    if cell == "cfc_lrc":
        kw["elastance_type"] = "asymmetric"
    build = lambda s: build_lotka_volterra_model("cncp", cell, size=s, seed=0,
                                                 **kw)
    size, n = size_for_budget(build, 4000, lamina_sizes(range(14, 30)))
    assert abs(n - 4000) / 4000 <= 0.05, (size, n)
    assert size != int(size)                  # only reachable off the integers
