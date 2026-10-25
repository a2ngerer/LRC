# tests/tasks/lotka_volterra/test_t_span.py
# Loader-level t_span override (issue #31) and the sampling of the fast
# multitimescale component it is meant to control.
import json
import sys

import numpy as np
import pytest

from experiments import run_lotka_volterra_benchmark as lv_runner
from src.tasks.lotka_volterra import load_lotka_volterra
from src.tasks.neural_ode.datasets import _SYSTEMS

FAST_RATE = 8.0   # multitimescale: dy0 = -0.05 y0 + sin(8 y1)


def test_t_span_override_sets_the_grid_and_leaves_systems_unchanged():
    before = dict(_SYSTEMS["multitimescale"])
    d = load_lotka_volterra(n_trajectories=5, seq_len=128,
                            system="multitimescale", t_span=(0, 10))
    assert d.t_span == (0.0, 10.0) and d.dt == pytest.approx(10 / 128)
    assert _SYSTEMS["multitimescale"] == before
    default = load_lotka_volterra(n_trajectories=5, seq_len=128,
                                  system="multitimescale")
    assert default.t_span == (0.0, 20.0) and default.dt == pytest.approx(20 / 128)
    with pytest.raises(ValueError, match="t1 > t0"):
        load_lotka_volterra(n_trajectories=5, seq_len=8, t_span=(1, 1))


def _samples_per_fast_period(d):
    """Worst case over every trajectory and step of the dataset.

    The fast component is sin(8 y1): its phase 8 y1 advances by 8 |dy1| per
    step, so one period spans 2 pi / (8 |dy1|) samples. y1 itself is smooth on
    the grid, so this count does not alias, unlike an FFT peak or zero-crossing
    count of an undersampled y0.
    """
    traj = np.concatenate([d.train_traj, d.val_traj, d.test_traj])
    step = FAST_RATE * np.abs(np.diff(traj[..., 1], axis=1))
    return 2 * np.pi / step.max()


@pytest.mark.parametrize("t_span", [
    (0, 0.8),
    pytest.param((0, 10), marks=pytest.mark.xfail(strict=True, reason=(
        "issue #31 default: the fast phase rate is 8 dy1/dt ~ 64 y0 with y0 in "
        "[0.6, 1.4], not 8, so (0, 10) at 128 steps undersamples it; open "
        "t_span decision"))),
])
def test_multitimescale_fast_component_sampled_10x_per_period(t_span):
    d = load_lotka_volterra(seq_len=128, system="multitimescale", t_span=t_span)
    assert _samples_per_fast_period(d) >= 10


def test_runner_records_the_t_span(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "run_lotka_volterra_benchmark.py", "--wiring", "dense", "--cell",
        "cfc_static", "--system", "multitimescale", "--t-span", "0", "10",
        "--epochs", "1", "--n-trajectories", "5", "--seq-len", "8",
        "--size", "4", "--no-save-weights", "--outdir", str(tmp_path)])
    lv_runner.main()
    record = json.loads((tmp_path / "cfc_static_dense_seed0.json").read_text())
    assert record["system"] == "multitimescale"
    assert record["t_span"] == record["config"]["effective"]["t_span"] == [0.0, 10.0]
    assert record["dt"] == pytest.approx(10 / 8)
