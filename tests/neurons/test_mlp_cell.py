# tests/neurons/test_mlp_cell.py
# mlp (issue #60): the stateless control cell of the rollout tasks. Registered
# in both registries, 'dense' wiring only, 4k budget within 5 %, output at step
# t a function of the step-t input alone, and it runs through the LV runner,
# the ForcedRolloutCell view and the RQ4 diagnostics like any recurrent arm.
import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from experiments import rq4_diagnostics as rq4
from experiments import run_lotka_volterra_benchmark as lv_runner
from src.benchmark import registry
from src.models.rnn_model import _CELL_REGISTRY
from src.neurons import MLP_Cell
from src.tasks.lotka_volterra.model import (LV_WIRINGS,
                                            build_lotka_volterra_model,
                                            forced_rollout_model)
from src.tasks.person_activity.model import (WIRINGS as PA_WIRINGS,
                                             budget_sizes,
                                             build_person_activity_model)
from src.wirings import NCPWiring, param_counts, size_for_budget


def _inputs(n=3, T=12, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(n, T, 2)).astype(np.float32),
            np.full((n, T, 1), 0.1, np.float32))


def test_registered_with_empty_kwargs():
    assert _CELL_REGISTRY["mlp"] is MLP_Cell
    assert registry.is_known_cell("mlp")
    assert registry.cell_kwargs("mlp") == {}
    assert registry.cell_kwargs("mlp", "symmetric") == {}


def test_hits_the_4k_budget_on_dense():
    size, n = size_for_budget(
        lambda s: build_lotka_volterra_model("dense", "mlp", size=s),
        4000, budget_sizes("dense"))
    assert abs(n - 4000) / 4000 <= 0.05
    # two Dense(h) on (y, dt) + Dense(2) readout: 3h+h + h*h+h + 2h+2
    assert n == size * size + 7 * size + 2


def test_output_at_t_depends_on_the_step_t_input_only():
    model = build_lotka_volterra_model("dense", "mlp", size=8, seed=0)
    x, t = _inputs()
    out = model((x, t)).numpy()
    k = 7
    x2, t2 = x.copy(), t.copy()
    x2[:, :k] = 5.0                     # every earlier input changed
    t2[:, :k] = 0.7
    out2 = model((x2, t2)).numpy()
    np.testing.assert_array_equal(out2[:, k:], out[:, k:])
    assert not np.allclose(out2[:, k - 1], out[:, k - 1])
    t2[:, k] = 0.7                      # dt is an input feature
    assert not np.allclose(model((x2, t2)).numpy()[:, k], out[:, k])


@pytest.mark.parametrize("wiring", [w for w in LV_WIRINGS if w != "dense"])
def test_other_lv_wirings_raise(wiring):
    with pytest.raises(ValueError, match="stateless control of the 'dense'"):
        build_lotka_volterra_model(wiring, "mlp", size=8)


@pytest.mark.parametrize("wiring", [w for w in PA_WIRINGS if w != "dense"])
def test_other_pa_wirings_raise(wiring):
    with pytest.raises(ValueError, match="stateless control of the 'dense'"):
        build_person_activity_model(wiring, "mlp", size=8)


def test_ncp_mask_is_rejected_by_the_cell():
    with pytest.raises(ValueError, match="dense"):
        NCPWiring(MLP_Cell, 8, 4, 2).build_model()


def test_budget_scan_reports_the_wiring_error():
    with pytest.raises(ValueError, match="stateless control"):
        size_for_budget(
            lambda s: build_lotka_volterra_model("ncp", "mlp", size=s),
            4000, range(2, 5))


def test_forced_view_matches_teacher_forcing_and_closed_loop():
    model = build_lotka_volterra_model("dense", "mlp", size=8, seed=0)
    x, t = _inputs()
    alpha = tf.Variable(1.0, trainable=False)
    view = forced_rollout_model(model, alpha)
    assert param_counts(view) == param_counts(model)
    np.testing.assert_allclose(view((x, t)).numpy(), model((x, t)).numpy(),
                               atol=1e-5)
    alpha.assign(0.0)
    ref = lv_runner.closed_loop_rollout(
        model, SimpleNamespace(seq_len=x.shape[1], test_x=x, test_t=t))
    np.testing.assert_allclose(view((x, t)).numpy(), ref[:, 1:], atol=1e-5)


def test_runner_forced_diagnostics_and_posthoc_rq4(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "run_lotka_volterra_benchmark.py", "--wiring", "dense", "--cell",
        "mlp", "--param-budget", "4000", "--epochs", "2", "--forcing",
        "mixed", "--diagnostics", "--n-trajectories", "10", "--seq-len",
        "20", "--batch-size", "4", "--outdir", str(tmp_path)])
    lv_runner.main()
    rec = json.loads((tmp_path / "mlp_dense_seed0_mixed.json").read_text())
    assert rec["cell"] == "mlp" and rec["size"] == 60
    assert not rec["budget_above_tolerance"]
    assert rec["config"]["effective"]["cell_kwargs"] == {}
    assert rec["ode_unfolds"] is None and rec["alpha_curve"] == [1.0, 0.2]
    assert np.isfinite(rec["closed_loop_mse"])
    assert (tmp_path / rec["weights_path"]).exists()
    assert set(rec["diagnostics"]["per_group"]) == {"backbone", "readout"}
    out = tmp_path / "rq4.csv"
    rq4.main(["--results", str(tmp_path / "*.json"), "--out", str(out)])
    row = pd.read_csv(out).iloc[0]
    # the state slot is never read: no retention at any lag
    for p in ("tf", "cl"):
        for d in rq4.DELTAS:
            assert row[f"{p}_jac{d}_max"] == 0.0
