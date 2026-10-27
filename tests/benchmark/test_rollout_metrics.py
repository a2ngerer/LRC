"""Rollout endpoint metrics (#20): the pre-registered unit checks."""
import json
import subprocess
import sys

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from src.evaluation.rollout_metrics import (INVARIANTS, anchor_metrics,
                                            invariant_drift, make_norm_stats,
                                            rollout_metrics)
from src.tasks.lotka_volterra.datasets import load_lotka_volterra
from src.tasks.neural_ode.datasets import _ODE_FUNCS, _SYSTEMS

T = 128


def _circle(shift=0):
    """Four concentric orbits, exactly three periods over T steps; ``shift``
    moves the phase by whole steps (the shifted points stay on the sampled
    orbit)."""
    t = (np.arange(T + 1) + shift) * 6 * np.pi / T
    r = np.linspace(1.0, 1.5, 4)[:, None, None]
    return r * np.stack([np.cos(t), np.sin(t)], axis=-1)        # (4, T+1, 2)


TRUTH = _circle()
STATS = make_norm_stats(TRUTH)


def _metrics(pred, truth=TRUTH, stats=STATS, system="toy"):
    m = rollout_metrics(truth, pred, stats, system)
    assert m["rate_ok"] + m["rate_unbounded"] + m["rate_bounded_wrong"] == pytest.approx(1)
    assert m["rate_collapse"] + m["rate_drift"] == pytest.approx(m["rate_bounded_wrong"])
    return m


def test_vpt_of_perfect_prediction_is_T():
    m = _metrics(TRUTH)
    assert m["vpt_median"] == m["vpt_mean"] == T
    assert m["vpt_frac_median"] == 1.0
    assert m["nrmse_128_median"] == m["orbit_error_median"] == 0.0
    assert m["rate_ok"] == 1.0


def test_vpt_of_mean_prediction_is_1():
    mean = np.asarray(STATS["mean"])
    m = _metrics(np.broadcast_to(mean, TRUTH.shape))
    assert m["vpt_median"] == m["vpt_mean"] == 1


def test_phase_shift_has_zero_orbit_error_but_large_nrmse():
    m = _metrics(_circle(shift=21))                 # ~half a period
    assert m["orbit_error_median"] < 1e-9
    assert m["nrmse_128_median"] > 1.0
    assert m["rate_drift"] == 1.0                   # bounded, right amplitude


@pytest.mark.parametrize("system,fixed_point", [
    ("periodic_predator_prey", (3.0, 1.5)), ("duffing", (1.0, 0.0))])
def test_invariant_is_conserved_on_ground_truth(system, fixed_point):
    assert INVARIANTS[system][1] == fixed_point
    assert np.allclose(_ODE_FUNCS[system](0.0, fixed_point), 0.0)  # E_min at an equilibrium
    data = load_lotka_volterra(n_trajectories=20, seed=0, system=system)
    traj = np.concatenate([data.train_traj, data.val_traj, data.test_traj])
    # The code's ODE from the generated initial states, integrated tightly:
    # pins E and E_min against _ODE_FUNCS.
    t = np.linspace(*_SYSTEMS[system]["t_span"], T + 1)
    tight = np.stack([solve_ivp(_ODE_FUNCS[system], _SYSTEMS[system]["t_span"],
                                y0, method="DOP853", t_eval=t, rtol=1e-10,
                                atol=1e-12).y.T for y0 in traj[:, 0]])
    drift, violations = invariant_drift(tight, tight, system)
    assert np.max(np.abs(drift)) < 1e-2 and violations.sum() == 0
    # The generator itself (rtol 1e-3) drifts by a few 1e-3 typically; single
    # trajectories exceed 1e-2, hence the median here.
    stats = make_norm_stats(data.train_traj, data.mean, data.std)
    m = _metrics(traj, traj, stats, system)
    assert m["inv_drift_max_median"] < 1e-2 and m["n_inv_drift_undefined"] == 0


def test_lv_positivity_violation_is_bounded_wrong():
    data = load_lotka_volterra(n_trajectories=10, seed=0)
    stats = make_norm_stats(data.train_traj, data.mean, data.std)
    pred = data.test_traj.copy()
    pred[0, 5, 1] = -0.01
    m = _metrics(pred, data.test_traj, stats, "periodic_predator_prey")
    assert m["positivity_violations"] == 1 and m["n_inv_drift_undefined"] == 1
    assert m["rate_bounded_wrong"] == 1 / len(pred)


def test_exploding_sequence_is_unbounded():
    growth = np.exp(np.linspace(0, 8, T + 1))[None, :, None]
    m = _metrics(TRUTH * growth)
    assert m["rate_unbounded"] == 1.0
    nonfinite = TRUTH.copy()
    nonfinite[:, 60:] = np.nan
    m = _metrics(nonfinite)
    assert m["rate_unbounded"] == 1.0 and m["nrmse_128_median"] == np.inf
    assert m["vpt_median"] == 60                     # first non-finite step


def test_constant_sequence_is_collapse():
    m = _metrics(np.repeat(TRUTH[:, :1], T + 1, axis=1))
    assert m["rate_collapse"] == 1.0


def test_cli_recomputes_from_result_jsons(tmp_path):
    from experiments.rollout_metrics import main
    pred = _circle(shift=21)
    old = {"system": "toy", "ground_truth_traj": TRUTH.tolist(),
           "closed_loop_traj": pred.tolist()}
    (tmp_path / "old.json").write_text(json.dumps(old))
    (tmp_path / "new.json").write_text(json.dumps({**old, "norm_stats": STATS}))
    (tmp_path / "hpo_selection.json").write_text("{}")
    out = tmp_path / "out.csv"
    main([str(tmp_path / "*.json"), "--out", str(out)])
    lines = out.read_text().splitlines()
    assert len(lines) == 3 and "norm_stats_source" in lines[0]
    assert {"stored", "test_proxy"} <= set(",".join(lines[1:]).split(","))


def test_runner_stores_rollout_metrics(tmp_path, monkeypatch):
    """Tiny real LV fit: both dicts in the JSON, and the post-hoc recompute
    from the stored trajectories + norm_stats reproduces the test dict."""
    from experiments import run_lotka_volterra_benchmark as lv_runner
    monkeypatch.setattr(sys, "argv", [
        "run_lotka_volterra_benchmark.py", "--wiring", "dense", "--cell", "gru",
        "--size", "4", "--epochs", "1", "--n-trajectories", "10",
        "--seq-len", "16", "--anchors", "4,8", "--horizon", "8",
        "--batch-size", "4", "--no-save-weights",
        "--outdir", str(tmp_path)])
    lv_runner.main()
    rec = json.loads((tmp_path / "gru_dense_seed0.json").read_text())
    keys = set(rec["rollout_metrics_test"])
    assert set(rec["rollout_metrics_val"]) == keys and "vpt_median" in keys
    assert {"closed_loop_mse", "closed_loop_mse_per_step"} <= set(rec)
    for split in ("val", "test"):
        m = rec[f"rollout_metrics_{split}"]
        assert m["horizon"] == 16 and m["n_trajectories"] == 2
        assert m["rate_ok"] + m["rate_unbounded"] + m["rate_bounded_wrong"] \
            == pytest.approx(1)
    again = rollout_metrics(rec["ground_truth_traj"], rec["closed_loop_traj"],
                            rec["norm_stats"], rec["system"])
    for k, v in again.items():
        assert rec["rollout_metrics_test"][k] == pytest.approx(
            v, rel=1e-3, abs=1e-4, nan_ok=True), k
    # multi-anchor endpoint (#62) on the runner's --anchors 4,8 --horizon 8
    for split in ("val", "test"):
        m = rec[f"rollout_metrics_{split}"]
        assert (m["anchors"], m["anchor_horizon"], m["anchor_theta"]) == (
            [4, 8], 8, 0.5)
        for key in ("vpt_multi_median", "vpt_anchor_4", "vpt_anchor_8",
                    "nrmse_32_anchor_4", "nrmse_32_anchor_8"):
            assert np.isfinite(m[key]), key
        assert 1 <= m["vpt_multi_median"] <= 8
        assert set(m["theta_sensitivity"]) == {"0.3", "1.0"}


def test_module_imports_without_tensorflow():
    code = ("import sys, src.evaluation.rollout_metrics; "
            "assert 'tensorflow' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)


# --- multi-anchor endpoint (#62) ---------------------------------------------

def _free_runs(pred, anchors, H):
    """{a: pred columns a+1..a+H}: `pred` as the rollout from every anchor."""
    return {a: pred[:, a + 1:a + 1 + H] for a in anchors}


def test_anchor_metrics_of_perfect_prediction():
    m = anchor_metrics(TRUTH, _free_runs(TRUTH, (8, 16, 24, 32), 96), STATS)
    assert m["vpt_multi_median"] == 96
    assert all(m[f"vpt_anchor_{a}"] == 96 and m[f"nrmse_32_anchor_{a}"] == 0
               for a in (8, 16, 24, 32))
    assert m["theta_sensitivity"] == {"0.3": 96, "1.0": 96}
    assert (m["anchors"], m["anchor_horizon"], m["anchor_theta"]) == (
        [8, 16, 24, 32], 96, 0.5)


def test_anchor_vpt_counts_free_run_steps_and_pools_trajectories_x_anchors():
    # e_t = 0.8 (z units) from free-run step v on: VPT(0.5) = VPT(0.3) = v,
    # VPT(1.0) = H = 10; three anchors x three trajectories, v = 1..9
    stats = {"mean": [0.0, 0.0], "std": [1.0, 1.0], "train_abs_max": 1.0}
    truth = np.zeros((3, 21, 2))
    vpt = {2: [1, 2, 3], 4: [4, 5, 6], 6: [7, 8, 9]}
    preds = {}
    for a, vs in vpt.items():
        p = np.zeros((3, 10, 2))
        for i, v in enumerate(vs):
            p[i, v - 1:] = 0.8          # e_t = 0.8 > 0.5 from step v
        preds[a] = p
    m = anchor_metrics(truth, preds, stats)
    assert [m[f"vpt_anchor_{a}"] for a in vpt] == [2, 5, 8]
    assert m["vpt_multi_median"] == 5                    # median of 1..9
    assert m["theta_sensitivity"] == {"0.3": 5, "1.0": 10}
    assert m["nrmse_32_anchor_2"] == pytest.approx(0.8 * np.sqrt(9 / 10))
    nonfinite = {a: np.full((3, 10, 2), np.nan) for a in vpt}
    assert anchor_metrics(truth, nonfinite, stats)["vpt_multi_median"] == 1
    with pytest.raises(ValueError, match="exceeds"):
        anchor_metrics(truth, {11: np.zeros((3, 10, 2))}, stats)
    with pytest.raises(ValueError, match="one free-run horizon"):
        anchor_metrics(truth, {2: np.zeros((3, 10, 2)),
                               4: np.zeros((3, 8, 2))}, stats)
