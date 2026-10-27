# tests/tasks/test_ode_unfolds_dt_scale.py
# Explicit ODE sub-steps and the task-level dt scale of the LV / PA runners
# (issues #28, #56):
#   - --ode-unfolds reaches the numerical ODE cells (ltc, lrc, lrc_pm, ctrnn,
#     mm_ltc, mm_lrc) and no closed-form or discrete cell; ctrnn gets the
#     runner default of 6 sub-steps under ncp and dense like lrc;
#   - dt_scale=2.0 doubles the one-step LRC increment (halves the effective
#     time constant) and equals feeding 2 * dt, for every cell and the dt
#     feature; the default 1.0 adds no layer;
#   - both runners record ode_unfolds and dt_scale in the result JSON, and
#     every runner (LV, PA, committee) reports config.effective.ode_unfolds;
#   - --dt-scale auto resolves to 1 / the task's dt (LV 12.8, Duffing 5.12,
#     PA 1 / median train dt) and lands in config.effective.dt_scale (#68).
import json
import sys

import numpy as np
import pytest
import tensorflow as tf

from experiments import run_committee_benchmark as co_runner
from experiments import run_lotka_volterra_benchmark as lv_runner
from experiments import run_person_activity_benchmark as pa_runner
from src.tasks.lotka_volterra.model import build_lotka_volterra_model
from src.tasks.person_activity import load_person_activity
from src.tasks.person_activity.model import (DEFAULT_ODE_UNFOLDS,
                                             build_person_activity_model,
                                             effective_ode_unfolds,
                                             ode_unfolds_override,
                                             resolve_dt_scale)
from tests.benchmark.test_checkpoints import _pa_stub
from tests.tasks.person_activity.test_person_activity import _write_stub

ASYM = dict(elastance_type="asymmetric")


def _rnn_cells(model):
    return [l.cell for l in model.layers if isinstance(l, tf.keras.layers.RNN)]


def _unfolds(cell):
    cell = getattr(cell, "_inner", cell)
    return getattr(cell, "_ode_unfolds", getattr(cell, "ode_unfolds", None))


@pytest.mark.parametrize("wiring", ["dense", "ncp"])
@pytest.mark.parametrize("cell", ["ltc", "lrc", "lrc_pm", "ctrnn", "mm_ltc",
                                  "mm_lrc"])
def test_ode_unfolds_reaches_the_numerical_cells(cell, wiring):
    kw = ode_unfolds_override(cell, 6)
    assert kw == {"ode_unfolds": 6}
    model = build_person_activity_model(wiring, cell, size=8, seed=0, **kw)
    model([np.zeros((1, 32, 7), np.float32), np.ones((1, 32, 1), np.float32)])
    assert [_unfolds(c) for c in _rnn_cells(model)] == [6]


@pytest.mark.parametrize("cell", ["cfc", "cfc_lrc", "cfc_mm_lrc", "cfc_mm_ltc",
                                  "gru", "lstm"])
def test_ode_unfolds_skips_closed_form_and_discrete_cells(cell):
    assert ode_unfolds_override(cell, 6) == {}


@pytest.mark.parametrize("build,features", [(build_person_activity_model, 7),
                                            (build_lotka_volterra_model, 2)])
@pytest.mark.parametrize("wiring", ["ncp", "dense"])
def test_ctrnn_runs_the_runner_default_substeps(build, features, wiring):
    """ctrnn follows the lrc path (issue #56): the runner default of 6 Euler
    sub-steps under ncp (the constructor default 1 would keep the motor slice
    from seeing the current input) and the same documented default on
    dense."""
    assert DEFAULT_ODE_UNFOLDS == 6
    kw = ode_unfolds_override("ctrnn", DEFAULT_ODE_UNFOLDS)
    assert effective_ode_unfolds("ctrnn", kw) == 6
    model = build(wiring, "ctrnn", size=8, seed=0, **kw)
    model([np.zeros((1, 32, features), np.float32),
           np.ones((1, 32, 1), np.float32)])
    assert [_unfolds(c) for c in _rnn_cells(model)] == [6]


@pytest.mark.parametrize("cell,expected", [
    ("ctrnn", 1), ("lrc", 1), ("ltc", 6), ("mm_ltc", 6), ("mm_lrc", 1),
    ("cfc_lrc", None), ("gru", None)])
def test_effective_ode_unfolds_falls_back_to_the_constructor_default(
        cell, expected):
    assert effective_ode_unfolds(cell, {}) == expected
    if expected is not None:
        assert effective_ode_unfolds(cell, {"ode_unfolds": 3}) == 3


def test_ode_unfolds_rejects_zero():
    with pytest.raises(ValueError):
        ode_unfolds_override("lrc", 0)


def test_dt_scale_doubles_the_one_step_lrc_increment():
    """One explicit Euler step from h=0: h1 = sigmoid(gate) * dt * v'(0), so
    dt_scale=2 doubles the increment, i.e. halves the effective time
    constant (output mapping starts at identity)."""
    def first_step(dt_scale):
        model = build_person_activity_model(
            "dense", "lrc", size=8, seed=0, seq_len=1, dt_scale=dt_scale,
            ode_unfolds=1, **ASYM)
        rnn = next(l for l in model.layers
                   if isinstance(l, tf.keras.layers.RNN))
        x = np.random.default_rng(0).normal(size=(4, 1, 7)).astype(np.float32)
        t = np.full((4, 1, 1), 0.3, np.float32)
        return tf.keras.Model(model.inputs, rnn.output)([x, t]).numpy()
    h1, h2 = first_step(1.0), first_step(2.0)
    assert np.abs(h1).max() > 0
    np.testing.assert_allclose(h2, 2.0 * h1, rtol=1e-5, atol=1e-7)


def _pa(wiring, cell, dt_scale, **kw):
    return build_person_activity_model(wiring, cell, size=8, seed=0,
                                       dt_scale=dt_scale, **kw)


def _lv(wiring, cell, dt_scale, **kw):
    return build_lotka_volterra_model(wiring, cell, size=8, seed=0,
                                      dt_scale=dt_scale, **kw)


@pytest.mark.parametrize("build,wiring,cell,kw,features", [
    (_pa, "dense3", "gru", {}, 7),                   # dt feature, every layer
    (_pa, "ncp", "cfc_lrc", ASYM, 7),
    (_pa, "dense", "lrc", dict(ASYM, ode_unfolds=6), 7),
    (_lv, "cncp", "cfc_lrc", ASYM, 2),
    (_lv, "dense", "ltc", dict(ode_unfolds=6), 2),
])
def test_dt_scale_equals_scaling_the_time_input(build, wiring, cell, kw,
                                                features):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 32, features)).astype(np.float32)
    t = rng.uniform(0.05, 0.5, size=(2, 32, 1)).astype(np.float32)
    plain, scaled = build(wiring, cell, 1.0, **kw), build(wiring, cell, 2.5, **kw)
    assert not any(isinstance(l, tf.keras.layers.Rescaling)
                   for l in plain.layers)                    # default: no layer
    assert scaled.count_params() == plain.count_params()
    np.testing.assert_allclose(scaled([x, t]), plain([x, 2.5 * t]),
                               rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("flags,cell,unfolds,dt_scale", [
    ([], "lrc", 6, 1.0),                                      # defaults
    ([], "ctrnn", 6, 1.0),                                    # issue #56
    (["--ode-unfolds", "2", "--dt-scale", "0.5"], "lrc", 2, 0.5),
    (["--ode-unfolds", "2"], "cfc_lrc", None, 1.0),           # closed form
])
def test_pa_runner_records_ode_unfolds_and_dt_scale(flags, cell, unfolds,
                                                    dt_scale, tmp_path,
                                                    monkeypatch):
    stub = _write_stub(tmp_path / "ConfLongDemo_JSI.txt")
    monkeypatch.setattr(sys, "argv", [
        "run_person_activity_benchmark.py", "--wiring", "dense", "--cell",
        cell, "--size", "4", "--epochs", "1", "--batch-size", "32",
        "--data-path", str(stub), "--outdir", str(tmp_path), *flags])
    pa_runner.main()
    rec = json.loads((tmp_path / f"{cell}_dense_seed0.json").read_text())
    assert rec["ode_unfolds"] == unfolds
    assert rec["cell_kwargs"].get("ode_unfolds") == unfolds
    assert rec["config"]["effective"]["ode_unfolds"] == unfolds
    assert rec["dt_scale"] == dt_scale


def test_lv_runner_records_ode_unfolds_and_dt_scale(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "run_lotka_volterra_benchmark.py", "--wiring", "dense", "--cell", "lrc",
        "--epochs", "1", "--n-trajectories", "6", "--seq-len", "6", "--size",
        "4", "--ode-unfolds", "2", "--dt-scale", "0.5", "--outdir",
        str(tmp_path)])
    lv_runner.main()
    rec = json.loads((tmp_path / "lrc_dense_seed0.json").read_text())
    assert rec["ode_unfolds"] == 2 and rec["dt_scale"] == 0.5


def test_lv_runner_reports_effective_ode_unfolds_for_ctrnn(tmp_path,
                                                          monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "run_lotka_volterra_benchmark.py", "--wiring", "ncp", "--cell",
        "ctrnn", "--epochs", "1", "--n-trajectories", "6", "--seq-len", "6",
        "--size", "4", "--outdir", str(tmp_path)])
    lv_runner.main()
    rec = json.loads((tmp_path / "ctrnn_ncp_seed0.json").read_text())
    assert rec["ode_unfolds"] == rec["config"]["effective"]["ode_unfolds"] == 6


def test_committee_runner_reports_effective_ode_unfolds(tmp_path, monkeypatch):
    """The committee runner has no --ode-unfolds: it reports the constructor
    default the ctrnn cells actually run."""
    monkeypatch.setattr(co_runner, "load_person_activity", _pa_stub)
    monkeypatch.setattr(sys, "argv", [
        "run_committee_benchmark.py", "--cell", "ctrnn", "--n-columns", "2",
        "--size", "8", "--epochs", "1", "--seq-len", "32", "--drop-fracs",
        "0.0", "--drop-seeds", "0", "--no-save-weights", "--outdir",
        str(tmp_path)])
    co_runner.main()
    (path,) = tmp_path.glob("*.json")
    rec = json.loads(path.read_text())
    assert rec["config"]["effective"]["ode_unfolds"] == 1


# --dt-scale auto (issue #68, protocol freeze F1-2026-10-26 h): 1 / the task's
# own dt, resolved by the runner and recorded as a number.

def test_resolve_dt_scale():
    assert resolve_dt_scale("auto", 0.078125) == 12.8
    assert resolve_dt_scale(2.5, 0.1) == 2.5
    with pytest.raises(ValueError):
        resolve_dt_scale("auto", 0.0)


def _effective_dt_scale(rec):
    eff = rec["config"]["effective"]
    values = {rec["dt_scale"], eff["dt_scale"], eff["build"]["dt_scale"]}
    assert len(values) == 1
    return values.pop()


@pytest.mark.parametrize("flags,expected", [
    ([], 12.8),                                     # Lotka-Volterra, t in (0, 10)
    (["--system", "duffing"], 5.12),                # t in (0, 25)
    # the multitimescale_lib time span (0, 10)
    (["--system", "multitimescale", "--t-span", "0", "10"], 12.8),
])
def test_lv_runner_resolves_dt_scale_auto(flags, expected, tmp_path,
                                          monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "run_lotka_volterra_benchmark.py", "--wiring", "dense", "--cell",
        "cfc_lrc", "--epochs", "1", "--n-trajectories", "6", "--size", "4",
        "--dt-scale", "auto", "--no-save-weights", "--outdir", str(tmp_path),
        *flags])
    lv_runner.main()
    rec = json.loads((tmp_path / "cfc_lrc_dense_seed0.json").read_text())
    assert rec["seq_len"] == 128
    assert _effective_dt_scale(rec) == pytest.approx(expected, rel=1e-12)


def _stub_200ms(path):
    """The PA stub with 200 ms instead of 100 ms between rows (dt 2)."""
    rows = [line.split(",") for line in
            _write_stub(path).read_text().splitlines()]
    t0 = int(rows[0][2])
    for row in rows:
        row[2] = str(t0 + 2 * (int(row[2]) - t0))
    path.write_text("".join(",".join(row) + "\n" for row in rows))
    return path


def test_pa_runner_resolves_dt_scale_auto(tmp_path, monkeypatch):
    stub = _stub_200ms(tmp_path / "ConfLongDemo_JSI.txt")
    train_t = load_person_activity(data_path=str(stub)).train_t
    assert np.median(train_t) == pytest.approx(2.0)
    monkeypatch.setattr(sys, "argv", [
        "run_person_activity_benchmark.py", "--wiring", "dense", "--cell",
        "cfc_lrc", "--size", "4", "--epochs", "1", "--batch-size", "32",
        "--dt-scale", "auto", "--no-save-weights", "--data-path", str(stub),
        "--outdir", str(tmp_path)])
    pa_runner.main()
    rec = json.loads((tmp_path / "cfc_lrc_dense_seed0.json").read_text())
    assert _effective_dt_scale(rec) == pytest.approx(0.5)
