# tests/tasks/test_ode_unfolds_dt_scale.py
# Explicit ODE sub-steps and the task-level dt scale of the LV / PA runners
# (issue #28):
#   - --ode-unfolds reaches the numerical ODE cells (ltc, lrc, lrc_pm, mm_ltc,
#     mm_lrc) and no closed-form or discrete cell;
#   - dt_scale=2.0 doubles the one-step LRC increment (halves the effective
#     time constant) and equals feeding 2 * dt, for every cell and the dt
#     feature; the default 1.0 adds no layer;
#   - both runners record ode_unfolds and dt_scale in the result JSON.
import json
import sys

import numpy as np
import pytest
import tensorflow as tf

from experiments import run_lotka_volterra_benchmark as lv_runner
from experiments import run_person_activity_benchmark as pa_runner
from src.tasks.lotka_volterra.model import build_lotka_volterra_model
from src.tasks.person_activity.model import (build_person_activity_model,
                                             ode_unfolds_override)
from tests.tasks.person_activity.test_person_activity import _write_stub

ASYM = dict(elastance_type="asymmetric")


def _rnn_cells(model):
    return [l.cell for l in model.layers if isinstance(l, tf.keras.layers.RNN)]


def _unfolds(cell):
    return getattr(getattr(cell, "_inner", cell), "_ode_unfolds", None)


@pytest.mark.parametrize("wiring", ["dense", "ncp"])
@pytest.mark.parametrize("cell", ["ltc", "lrc", "lrc_pm", "mm_ltc", "mm_lrc"])
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
