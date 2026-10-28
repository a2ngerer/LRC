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


@pytest.mark.parametrize("cell", ["lrc_init", "lrc_cal"])
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
        assert kw["init_slice_scale"] == {"inter": 7.2, "command": 6.6,
                                          "motor": 6.6}
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
        if cell == "lrc_cal":
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
