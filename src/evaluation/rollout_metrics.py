"""Rollout endpoint metrics for the closed-loop trajectory tasks (issue #20).

Pre-registered definitions. Inputs are de-normalised trajectories (N, T+1, D)
as the LV runner stores them: column 0 is the given initial state, columns
1..T the rollout (step t = column t). Errors, VPT, NRMSE, orbit error and the
divergence classes are computed in train-z-score units, the invariant drift in
physical units. Every per-trajectory value is reduced to its median and mean
over trajectories (``<name>_median`` / ``<name>_mean``; NaN = undefined and
skipped, inf = non-finite rollout).

  * e_t = sqrt(mean_d (yhat_t,d - y_t,d)^2)
  * VPT(theta): first step t with e_t > theta, else T (also as fraction of T);
    theta = 0.5 is the primary endpoint, 0.3 and 1.0 are descriptive.
  * NRMSE@H, H in {16, 32, 128}: sqrt(mean_{t<=H,d} (yhat - y)^2).
  * orbit error: mean over predicted points of the squared distance to the
    nearest ground-truth point of the same trajectory (phase-free).
  * invariant drift D_t = (E(yhat_t) - E(y_0)) / (E(y_0) - E_min): D_32, D_128,
    max_t |D_t| (keys inv_drift_*). A trajectory with an LV positivity
    violation (x <= 0 or y <= 0) or a non-finite D has D undefined.
  * classes: unbounded (non-finite or max |yhat| > 10 * max |y_train|), else
    bounded_wrong (NRMSE@128 > 1 or positivity violation) split into collapse
    (amplitude ratio over the second half < 0.1) and drift, else ok.
    rate_ok + rate_unbounded + rate_bounded_wrong = 1 and
    rate_collapse + rate_drift = rate_bounded_wrong.

Pure numpy: src/evaluation/__init__ loads the TF-based GradientFlowTracker
lazily, so this module imports without TensorFlow (CPU-only aggregation).
"""
from __future__ import annotations

import numpy as np

THETA = 0.5
THETA_SENSITIVITY = (0.3, 1.0)
HORIZONS = (16, 32, 128)
UNBOUNDED_FACTOR = 10.0
COLLAPSE_RATIO = 0.1


def _lv_energy(s):
    # periodic_predator_prey: dx = 1.5 x - x y, dy = -3 y + x y
    x, y = s[..., 0], s[..., 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        return x - 3.0 * np.log(x) + y - 1.5 * np.log(y)


def _duffing_energy(s):
    # duffing: dx = v, dv = x - x^3
    x, v = s[..., 0], s[..., 1]
    return 0.5 * v ** 2 - 0.5 * x ** 2 + 0.25 * x ** 4


# system -> (invariant E, a fixed point where E is minimal); E_min = E(fixed
# point). Systems not listed get NaN drift.
INVARIANTS = {
    "periodic_predator_prey": (_lv_energy, (3.0, 1.5)),
    "duffing": (_duffing_energy, (1.0, 0.0)),
}


def make_norm_stats(train_traj, mean=None, std=None) -> dict:
    """norm_stats for rollout_metrics from raw (N, T+1, D) training trajectories:
    the z-score (mean, std) and the largest |z| of the training data. mean/std
    default to a fit on the inputs (all but the last step), as in
    load_lotka_volterra."""
    traj = np.asarray(train_traj, dtype=np.float64)
    if mean is None:
        flat = traj[:, :-1].reshape(-1, traj.shape[-1])
        mean, std = flat.mean(axis=0), flat.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    return {"mean": mean.tolist(), "std": std.tolist(),
            "train_abs_max": float(np.max(np.abs((traj - mean) / std)))}


def invariant_drift(y_true, y_pred, system):
    """Signed relative drift D (N, T) in physical units and the per-trajectory
    count of LV positivity violations (N,). D is None for systems without an
    invariant; rows with a violation or a non-finite value are NaN."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    violations = np.zeros(len(y_pred), dtype=int)
    if system not in INVARIANTS:
        return None, violations
    energy, fixed_point = INVARIANTS[system]
    e0 = energy(y_true[:, 0])[:, None]
    with np.errstate(all="ignore"):
        d =(energy(y_pred[:, 1:]) - e0) / (e0 - energy(np.array(fixed_point)))
    if system == "periodic_predator_prey":
        violations = (y_pred[:, 1:] <= 0).any(axis=-1).sum(axis=1)
    d[(violations > 0) | ~np.isfinite(d).all(axis=1)] = np.nan
    return d, violations


def _summarise(out, name, values):
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    out[f"{name}_median"] = float(np.median(v)) if v.size else float("nan")
    out[f"{name}_mean"] = float(np.mean(v)) if v.size else float("nan")


def rollout_metrics(y_true, y_pred, norm_stats, system) -> dict:
    """All #20 rollout metrics of one split (see module docstring).

    Args:
        y_true, y_pred: (N, T+1, D) de-normalised ground truth and closed-loop
            rollout; column 0 is the initial state and is not scored.
        norm_stats: {"mean", "std", "train_abs_max"} (see make_norm_stats).
        system: neural_ode system name (selects the invariant).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mean = np.asarray(norm_stats["mean"], dtype=np.float64)
    std = np.asarray(norm_stats["std"], dtype=np.float64)
    z_true = (y_true[:, 1:] - mean) / std                       # (N, T, D)
    z_pred = (y_pred[:, 1:] - mean) / std
    n, T = z_true.shape[:2]
    finite = np.isfinite(z_pred)
    with np.errstate(all="ignore"):
        sq = np.where(finite, (z_pred - z_true) ** 2, np.inf)   # non-finite -> inf
        err = np.sqrt(sq.mean(axis=2))                          # e_t, (N, T)
        nrmse = {h: np.sqrt(sq[:, :h].mean(axis=(1, 2))) for h in HORIZONS}
        dist = ((z_pred[:, :, None] - z_true[:, None]) ** 2).sum(axis=-1)
        orbit = np.where(finite.all(axis=(1, 2)),
                         dist.min(axis=2).mean(axis=1), np.inf)
        z_abs_max = np.abs(np.where(finite, z_pred, np.inf)).max(axis=(1, 2))
        half = T // 2
        amp_ratio = (np.sqrt(z_pred[:, half:].var(axis=1).mean(axis=-1))
                     / np.sqrt(z_true[:, half:].var(axis=1).mean(axis=-1)))

    out = {"n_trajectories": n, "horizon": T}
    for theta in (THETA, *THETA_SENSITIVITY):
        exceed = err > theta
        vpt = np.where(exceed.any(axis=1), exceed.argmax(axis=1) + 1, T)
        key = "vpt" if theta == THETA else f"vpt_{theta}"
        _summarise(out, key, vpt)
        if theta == THETA:
            _summarise(out, "vpt_frac", vpt / T)
    for h in HORIZONS:
        _summarise(out, f"nrmse_{h}", nrmse[h])
    _summarise(out, "orbit_error", orbit)

    drift, violations = invariant_drift(y_true, y_pred, system)
    if drift is None:
        drift = np.full((n, T), np.nan)
    for h in (32, 128):
        _summarise(out, f"inv_drift_{h}", drift[:, min(h, T) - 1])
    _summarise(out, "inv_drift_max", np.abs(drift).max(axis=1))
    out["n_inv_drift_undefined"] = int(np.isnan(drift[:, 0]).sum())
    out["positivity_violations"] = int(violations.sum())

    unbounded = z_abs_max > UNBOUNDED_FACTOR * norm_stats["train_abs_max"]
    wrong = ~unbounded & ((nrmse[128] > 1.0) | (violations > 0))
    collapse = wrong & (amp_ratio < COLLAPSE_RATIO)
    for name, mask in (("ok", ~unbounded & ~wrong), ("unbounded", unbounded),
                       ("bounded_wrong", wrong), ("collapse", collapse),
                       ("drift", wrong & ~collapse)):
        out[f"rate_{name}"] = float(mask.mean())
    return out
