# tests/tasks/lotka_volterra/test_multitimescale_lib.py
# multitimescale_lib libration box, per-system spread and --ic-spread (#64).
import inspect
import json
import sys

import numpy as np
import pytest

from experiments import run_lotka_volterra_benchmark as lv_runner
from src.tasks.lotka_volterra import load_lotka_volterra, sample_ood_trajectories
from src.tasks.lotka_volterra.datasets import (DATA_SEED_BASE, LIBRATION_H_MAX,
                                               _sample_initial_conditions,
                                               libration_energy)
from src.tasks.neural_ode.datasets import _ODE_FUNCS, _SYSTEMS, generate_dataset


def _all_traj(d):
    return np.concatenate([d.train_traj, d.val_traj, d.test_traj])


def _fast_component(traj):
    """64 y0 - 0.4 y1 (= phi'' up to the slow offset): zero at the equilibrium,
    unlike y0, which sits ~0.0025 off zero there."""
    return 64.0 * traj[..., 0] - 0.4 * traj[..., 1]


def _samples_per_period(y):
    """Mean samples per oscillation period of y (N, T): spacing of the
    linearly interpolated upward zero crossings."""
    out = []
    for row in y:
        k = np.where((row[:-1] < 0) & (row[1:] >= 0))[0]
        t = k + row[k] / (row[k] - row[k + 1])
        out.append(np.diff(t).mean())
    return np.array(out)


# Final runs use data seed DATA_SEED_BASE + replicate seed.
@pytest.mark.parametrize("replicate", range(10))
def test_libration_box_80_initial_conditions(replicate):
    d = load_lotka_volterra(n_trajectories=80, seq_len=128,
                            system="multitimescale_lib",
                            seed=DATA_SEED_BASE + replicate)
    assert d.spread == 0.3 and d.t_span == (0.0, 10.0)
    traj = _all_traj(d)
    assert len(traj) == 80
    assert np.all(libration_energy(traj[:, 0]) < LIBRATION_H_MAX)
    # 100 % libration: phi = 8 y1 never leaves the well around pi.
    assert np.all(np.abs(8.0 * traj[..., 1] - np.pi) < np.pi)
    # >= 10 samples per fast period on every trajectory.
    assert _samples_per_period(_fast_component(traj)).min() >= 10


def test_multitimescale_lib_matches_tight_reference():
    """The shared rtol=1e-3 phase-flipped fast oscillations (#64 review); the
    per-system rtol must track a tight-tolerance reference."""
    from scipy.integrate import solve_ivp
    d = load_lotka_volterra(n_trajectories=80, seq_len=128,
                            system="multitimescale_lib", seed=DATA_SEED_BASE + 1)
    t0, t1 = d.t_span
    t_eval = np.linspace(t0, t1, 128)
    for y0 in _all_traj(d)[:, 0].astype(float):
        _, y = generate_dataset("multitimescale_lib", data_size=128,
                                y0=y0.tolist())
        ref = solve_ivp(_ODE_FUNCS["multitimescale_lib"], (t0, t1), y0,
                        method="DOP853", t_eval=t_eval, rtol=1e-12,
                        atol=1e-14).y.T
        # Error relative to each component's range (~5e-7 at rtol 1e-8).
        assert np.all(np.abs(y - ref).max(0) / np.ptp(ref, 0) < 1e-5)


def test_loader_fails_loudly_outside_the_libration_box():
    with pytest.raises(ValueError, match="libration"):
        load_lotka_volterra(n_trajectories=20, seq_len=8,
                            system="multitimescale_lib", spread=1.0)
    with pytest.raises(ValueError, match="libration"):
        sample_ood_trajectories(20, seed=0, spread=1.0, seq_len=8,
                                system="multitimescale_lib")


# Golden values from int/finalsuite-2026-10-26 (before #64): the two smallest
# initial states (sorted by y0) and the float64 sum of all sorted samples.
_GOLDEN = {
    "periodic_predator_prey": ([[0.6292121410369873, 0.8020075559616089],
                                [0.6317931413650513, 0.8736312985420227]],
                               613.8310504332185),
    "duffing": ([[-1.3707878589630127, 0.8020075559616089],
                 [-1.3682068586349487, 0.8736312985420227]],
                -6.733761151728686),
}


@pytest.mark.parametrize("system", ["periodic_predator_prey", "duffing"])
def test_lv_and_duffing_sample_unchanged_at_0_4(system):
    assert inspect.signature(
        _sample_initial_conditions).parameters["spread"].default == 0.4
    assert "spread" not in _SYSTEMS[system]
    d = load_lotka_volterra(n_trajectories=16, seq_len=8, system=system)
    assert d.spread == 0.4
    want = _sample_initial_conditions(
        16, np.random.default_rng(DATA_SEED_BASE),
        base=np.asarray(_SYSTEMS[system]["y0"], dtype=float), spread=0.4)
    got = _all_traj(d)[:, 0]
    np.testing.assert_allclose(np.sort(got, axis=0),
                               np.sort(want, axis=0).astype(np.float32))
    ics, total = _GOLDEN[system]
    np.testing.assert_array_equal(got[np.argsort(got[:, 0])][:2],
                                  np.asarray(ics, dtype=np.float32))
    all_traj = _all_traj(d)
    assert float(np.sort(all_traj.reshape(-1)).sum(dtype=np.float64)) == total


def test_ood_set_shares_the_loader_code_path():
    ood = sample_ood_trajectories(16, seed=5, spread=0.8, seq_len=16,
                                  system="duffing")
    d = load_lotka_volterra(n_trajectories=16, seq_len=16, seed=5,
                            system="duffing", spread=0.8)
    by_y0 = lambda a: a[np.argsort(a[:, 0, 0])]
    np.testing.assert_array_equal(by_y0(ood), by_y0(_all_traj(d)))
    with pytest.raises(ValueError, match="spread"):
        load_lotka_volterra(n_trajectories=5, seq_len=8, spread=0.0)


def test_runner_ic_spread_overrides_and_is_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "run_lotka_volterra_benchmark.py", "--wiring", "dense", "--cell",
        "cfc", "--system", "multitimescale_lib", "--ic-spread", "0.25",
        "--epochs", "1", "--n-trajectories", "5", "--seq-len", "8",
        "--size", "4", "--no-save-weights", "--outdir", str(tmp_path)])
    lv_runner.main()
    record = json.loads((tmp_path / "cfc_dense_seed0.json").read_text())
    eff = record["config"]["effective"]
    assert record["ic_spread"] == eff["ic_spread"] == 0.25
    assert record["t_span"] == eff["t_span"] == [0.0, 10.0]
