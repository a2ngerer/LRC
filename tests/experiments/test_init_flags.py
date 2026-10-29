# tests/experiments/test_init_flags.py
# Init-diagnostic flags of the LV and PA runners (issue #82): --init-ranges
# reaches the LRC cell and config.effective, the default argv keeps the
# protocol optimizer and adds no config key, --optimizer rmsprop_legacy / adam
# train with --clipnorm and --diagnostics (slot norms, eff_lr, optimizer
# record), and lrc_working_point covers dense ('all') and ncp (three slices).
# The init arms lrc_init / lrc_cal through both runners: cell_kwargs in the
# result, the lrc parameter count, the init vleak range and the rebuild.
# One epoch on the synthetic stubs of tests/benchmark/test_checkpoints.py.
import json
import sys

import numpy as np
import pytest
import tensorflow as tf

import src.tasks.person_activity as pa_pkg
from experiments import run_lotka_volterra_benchmark as lv_runner
from experiments import run_person_activity_benchmark as pa_runner
from src.benchmark.checkpoints import BUILDERS, rebuild_from_result
from src.benchmark.registry import _LRC_INIT_RANGES
from src.neurons import LRC_Cell
from src.wirings import param_counts
from tests.benchmark.test_checkpoints import _capture_builds, _pa_stub

NEW_KEYS = {"optimizer", "opt_epsilon", "initialization_ranges"}


def _run(runner, wiring, tmp_path, monkeypatch, *extra, cell="lrc"):
    """One-epoch run of ``cell`` (default lrc); returns (result record, its JSON path, model)."""
    if runner == "lv":
        built = _capture_builds(monkeypatch, lv_runner,
                                "build_lotka_volterra_model")
        argv = ["--n-trajectories", "4", "--anchors", "2,4", "--horizon", "4",
                "--seq-len", "8", "--batch-size", "4"]
        main = lv_runner.main
    else:
        built = _capture_builds(monkeypatch, pa_pkg,
                                "build_person_activity_model")
        monkeypatch.setattr(pa_pkg, "load_person_activity", _pa_stub)
        argv = []
        main = pa_runner.main
    size = "8" if wiring == "ncp" else "4"
    monkeypatch.setattr(sys, "argv", [
        "runner.py", "--wiring", wiring, "--cell", cell, "--size", size,
        "--epochs", "1", "--outdir", str(tmp_path), *argv, *extra])
    main()
    (path,) = tmp_path.glob("*.json")
    return json.loads(path.read_text()), path, built[-1][0]


def _finite_point(name, vals, epochs=2):
    """Working-point lists of one slice: finite, one value per recorded epoch.
    w_mean is None on the ncp inter slice: inter neurons have no recurrent
    synapses (ncps wiring), so there is no unmasked w entry to average."""
    for k, v in vals.items():
        assert len(v) == epochs
        if k == "w_mean" and name == "inter" and None in v:
            assert v == [None] * epochs
            continue
        assert np.all(np.isfinite(v)), (name, k, v)


def _lrc_cells(model):
    return [l for l in model._flatten_layers(include_self=False, recursive=True)
            if isinstance(l, LRC_Cell)]


@pytest.mark.parametrize("runner", ["pa", "lv"])
def test_init_ranges_reach_the_cell(runner, tmp_path, monkeypatch):
    record, _, model = _run(runner, "dense", tmp_path, monkeypatch,
                            "--init-ranges", '{"vleak": [-1, 1]}')
    eff = record["config"]["effective"]
    assert eff["cell_kwargs"]["initialization_ranges"] == {"vleak": [-1, 1]}
    assert eff["cell_kwargs"]["elastance_type"] == "asymmetric"   # registry
    (cell,) = _lrc_cells(model)
    assert cell._init_ranges["vleak"] == [-1, 1]
    assert np.abs(cell._params["vleak"].numpy()).max() > 0.2   # default 0.2


def test_default_argv_keeps_optimizer_and_config(tmp_path, monkeypatch):
    record, _, model = _run("pa", "ncp", tmp_path, monkeypatch,
                            "--diagnostics")
    assert type(model.optimizer) is tf.keras.optimizers.RMSprop
    eff = record["config"]["effective"]
    assert not NEW_KEYS & (set(eff) | set(eff["build"]) | set(eff["cell_kwargs"]))
    diag = record["diagnostics"]
    assert diag["optimizer"]["class"] == "RMSprop"
    assert diag["optimizer"]["epsilon"] == pytest.approx(1e-7)
    (wp,) = diag["lrc_working_point"].values()
    assert set(wp) == {"motor", "command", "inter"}       # ncps state order
    for s, vals in wp.items():
        _finite_point(s, vals)
        assert all(0.0 < v < 1.0 for v in vals["sig_f"])
        assert all(0.0 <= v <= 1.0 for v in vals["tanh_g"])
    (cell,) = _lrc_cells(model)
    vleak = np.abs(cell._params["vleak"].numpy())
    assert wp["motor"]["vleak_abs"][-1] == pytest.approx(vleak[:4].mean())
    assert wp["inter"]["vleak_abs"][-1] == pytest.approx(vleak[8:].mean())


@pytest.mark.parametrize("runner,wiring,optimizer,cls,name", [
    ("pa", "dense", "rmsprop_legacy", tf.keras.optimizers.legacy.RMSprop,
     "legacy.RMSprop"),
    ("lv", "ncp", "adam", tf.keras.optimizers.Adam, "Adam")])
def test_optimizer_choice_trains_with_clipnorm_and_diagnostics(
        runner, wiring, optimizer, cls, name, tmp_path, monkeypatch):
    record, path, model = _run(runner, wiring, tmp_path, monkeypatch,
                               "--optimizer", optimizer, "--clipnorm", "1.0",
                               "--diagnostics")
    assert type(model.optimizer) is cls
    assert model.optimizer.global_clipnorm == 1.0
    assert record["training_diverged"] is False
    build = record["config"]["effective"]["build"]
    assert build["optimizer"] == optimizer and build["opt_epsilon"] is None
    assert type(rebuild_from_result(str(path)).optimizer) is cls
    diag = record["diagnostics"]
    assert diag["optimizer"]["class"] == name
    for g in diag["per_group"].values():
        assert g["slot_norm"][0] == 0.0 and g["eff_lr_mean"][0] is None
        assert np.isfinite(g["slot_norm"][1]) and g["slot_norm"][1] > 0
        assert np.isfinite(g["eff_lr_mean"][1]) and g["eff_lr_mean"][1] > 0
    (wp,) = diag["lrc_working_point"].values()
    assert set(wp) == ({"all"} if wiring == "dense"
                       else {"motor", "command", "inter"})
    for s, vals in wp.items():
        _finite_point(s, vals)


def test_opt_epsilon_reaches_the_optimizer(tmp_path, monkeypatch):
    record, _, model = _run("pa", "dense", tmp_path, monkeypatch,
                            "--opt-epsilon", "1e-4", "--diagnostics")
    assert type(model.optimizer) is tf.keras.optimizers.RMSprop
    assert model.optimizer.epsilon == pytest.approx(1e-4)
    assert record["config"]["effective"]["build"]["opt_epsilon"] == 1e-4
    assert record["diagnostics"]["optimizer"]["epsilon"] == pytest.approx(1e-4)


SLICE_SCALE = {"lrc_cal": {"inter": 7.2, "command": 6.6, "motor": 6.6},
               "lrc_cal_lv": {"inter": 5.3, "command": 4.3, "motor": 3.8}}


@pytest.mark.parametrize("cell", ["lrc_init", "lrc_cal", "lrc_cal_lv"])
@pytest.mark.parametrize("wiring", ["dense", "ncp"])
@pytest.mark.parametrize("runner", ["pa", "lv"])
def test_init_arms_through_the_runners(runner, wiring, cell, tmp_path,
                                       monkeypatch):
    record, path, model = _run(runner, wiring, tmp_path, monkeypatch,
                               "--diagnostics", cell=cell)
    eff = record["config"]["effective"]
    kw = eff["cell_kwargs"]
    assert "init_slices" not in kw                  # set by the wiring only
    if cell == "lrc_init":
        assert kw["init_fanin_scale"] == "units"
        assert kw["initialization_ranges"] == {
            k: list(v) for k, v in _LRC_INIT_RANGES.items()}
    else:
        assert kw["init_slice_scale"] == SLICE_SCALE[cell]
    # same parameter count as lrc at the same size and wiring
    # (the run's cell kwargs without the init options: lrc's plus ode_unfolds)
    lrc = BUILDERS[eff["task"]](**{**eff["build"], "cell": "lrc"},
                                **{k: v for k, v in kw.items()
                                   if not k.startswith("init")})
    assert record["params_effective"] == param_counts(lrc)["params_effective"]
    assert record["params_raw"] == param_counts(lrc)["params_raw"]

    (c,) = _lrc_cells(model)
    vleak = np.abs(c._params["vleak"].numpy())
    assert vleak.max() > 0.2                         # lrc draws from +-0.2
    wp = record["diagnostics"]["lrc_working_point"]
    (slices,) = wp.values()
    if wiring == "ncp":
        assert set(slices) == {"motor", "command", "inter"}
        if cell in SLICE_SCALE:
            assert c._init_slices == {"motor": (0, 4), "command": (4, 8),
                                      "inter": (8, 16)}
            for a, b in c._init_slices.values():
                assert vleak[a:b].max() > 0.2
    else:
        assert set(slices) == {"all"}
    for s, vals in slices.items():
        _finite_point(s, vals)

    (rc,) = _lrc_cells(rebuild_from_result(str(path)))
    np.testing.assert_array_equal(rc._params["vleak"].numpy(),
                                  c._params["vleak"].numpy())


# --- formula checks without a fit ------------------------------------------

def _diag(opt, kind):
    """The optimizer-facing state of RQ4Diagnostics, without a model."""
    from types import SimpleNamespace
    return SimpleNamespace(_net=SimpleNamespace(optimizer=opt), _kind=kind)


@pytest.mark.parametrize("opt_name,kind", [
    ("rmsprop", "RMSprop"), ("rmsprop_legacy", "legacy.RMSprop"),
    ("adam", "Adam")])
def test_eff_lr_reproduces_the_keras_step(opt_name, kind):
    """eff_lr * (gradient, or Adam's first moment) is the step Keras took."""
    from src.benchmark.optimizers import make_optimizer
    from src.evaluation.gradient_flow import RQ4Diagnostics
    lr = 1e-2
    # a large epsilon separates sqrt(v + eps) from sqrt(v) + eps
    opt = make_optimizer(opt_name, lr, epsilon=0.1)
    # float64: before - after carries no float32 rounding of the variable
    var = tf.Variable(np.linspace(-1, 1, 6))
    rng = np.random.default_rng(0)
    for _ in range(3):                              # t = 3: bias terms != 1
        g = rng.normal(size=6)
        before = var.numpy()
        opt.apply_gradients([(tf.constant(g), var)])
    step = before - var.numpy()
    d = _diag(opt, kind)
    eff = RQ4Diagnostics._eff_lr(d, RQ4Diagnostics._slots(d, [var]), lr)
    num = (opt._momentums[opt._index_dict[opt._var_key(var)]].numpy()
           if kind == "Adam" else g)
    np.testing.assert_allclose(step, eff * num, rtol=1e-5)


def test_legacy_rmsprop_applies_global_clipnorm():
    """The runner sets global_clipnorm after construction; the legacy
    optimizer must clip with it (rms slot sees the clipped gradient)."""
    from src.benchmark.optimizers import make_optimizer
    opt = make_optimizer("rmsprop_legacy", 1e-3)
    opt.global_clipnorm = 1.0
    var = tf.Variable(np.zeros(4, np.float32))
    g = np.full(4, 1000.0, np.float32)
    opt.apply_gradients([(tf.constant(g), var)])
    clipped = g / np.linalg.norm(g)
    np.testing.assert_allclose(opt.get_slot(var, "rms").numpy(),
                               (1 - opt.rho) * clipped ** 2, rtol=1e-5)


def test_opt_epsilon_must_be_positive():
    import argparse
    from src.benchmark.optimizers import add_optimizer_args, make_optimizer
    with pytest.raises(ValueError):
        make_optimizer("rmsprop_legacy", 1e-3, epsilon=0.0)
    p = argparse.ArgumentParser()
    add_optimizer_args(p)
    with pytest.raises(SystemExit):
        p.parse_args(["--opt-epsilon", "0"])
    assert p.parse_args(["--opt-epsilon", "1e-4"]).opt_epsilon == 1e-4


def test_working_point_matches_the_cell_solver():
    """With elastance 'interp', one unfold, dt 1 and vleak 1 the explicit and
    hybrid solver steps from the same state determine sigmoid(f) and tanh(g)
    per neuron: s = (ve - vh) / (vh - v), t = vh (1 + s) - v."""
    from src.evaluation.gradient_flow import RQ4Diagnostics
    from src.wirings import NCPWiring
    tf.keras.utils.set_random_seed(0)
    inter, command, motor, n_in = 4, 3, 2, 3
    n = inter + command + motor
    w = NCPWiring(LRC_Cell, inter, command, motor)
    cell = LRC_Cell(units=n, elastance_type="interp", ode_unfolds=1, dt=1,
                    sparsity_mask=w.adjacency_mask, sensory_mask=w.sensory_mask)
    cell.build((None, n_in))
    rng = np.random.default_rng(0)
    cell._params["vleak"].assign(np.ones(n, np.float32))
    for k in ("input_w", "input_b"):
        cell._params[k].assign(rng.normal(size=n_in).astype(np.float32))
    x = rng.normal(size=(64, n_in)).astype(np.float32)
    v = rng.uniform(-1, 1, size=(64, n)).astype(np.float32)
    out = {}
    for solver in ("explicit", "hybrid"):
        cell._ode_solver_type = solver
        out[solver] = cell(tf.constant(x), [tf.constant(v)])[1][0].numpy()
    ve, vh = (out[k].astype(np.float64) for k in ("explicit", "hybrid"))
    s = (ve - vh) / (vh - v)
    t = vh * (1 + s) - v
    slices = {"motor": (0, motor), "command": (motor, motor + command),
              "inter": (motor + command, n)}
    point = RQ4Diagnostics._working_point(cell, x, v, slices)
    for name, (a, b) in slices.items():
        assert point[name]["sig_f"] == pytest.approx(s[:, a:b].mean(), abs=1e-4)
        assert point[name]["tanh_g"] == pytest.approx(
            np.abs(t[:, a:b]).mean(), abs=1e-4)
