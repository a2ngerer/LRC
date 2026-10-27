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
  * outcome classes (protocol freeze F1-2026-10-26, decision i), three
    disjoint labels per trajectory: unbounded (max |yhat| > 10 * max |y_train|,
    a non-finite rollout included), else bounded_wrong (collapse: amplitude
    ratio over the second half < 0.1, or orbit error > tau_orb, or an LV
    positivity violation), else ok (on the orbit, possibly phase-shifted).
    rate_ok + rate_unbounded + rate_bounded_wrong = 1 at tau_orb (default
    0.15); the same rates at TAU_ORB_SENSITIVITY (0.30) are stored as
    outcome_classes_tau_0.30. non_finite (NaN/Inf anywhere in the rollout) is a
    fit-level flag, not a trajectory label.

Multi-anchor endpoint (#62, protocol freeze F1-2026-10-26, decision c;
anchor_metrics): from each anchor a in ANCHORS the model is fed the ground
truth y_0..y_a (teacher-forced warm-up), then runs free for ANCHOR_HORIZON
steps (predictions of steps a+1..a+H). Per (trajectory, anchor) VPT(theta) on
the H free-run steps and NRMSE@32; keys:
  * vpt_multi_median: median over all (trajectory x anchor) VPTs, the primary
    rollout endpoint and the HPO selection key;
  * vpt_anchor_<a>, nrmse_32_anchor_<a>: median over trajectories per anchor
    (vpt_anchor_8 is the single-anchor secondary endpoint);
  * theta_sensitivity: {"0.3": ..., "1.0": ...}, vpt_multi_median at the
    descriptive thresholds;
  * anchors, anchor_horizon, anchor_theta: the evaluation settings.

Pure numpy: src/evaluation/__init__ loads the TF-based GradientFlowTracker
lazily, so this module imports without TensorFlow (CPU-only aggregation).
"""
from __future__ import annotations

import numpy as np

THETA = 0.5
THETA_SENSITIVITY = (0.3, 1.0)
HORIZONS = (16, 32, 128)
ANCHORS = (8, 16, 24, 32)
ANCHOR_HORIZON = 96
UNBOUNDED_FACTOR = 10.0
COLLAPSE_RATIO = 0.1
TAU_ORB = 0.15
TAU_ORB_SENSITIVITY = 0.30
SENSITIVITY_KEY = f"outcome_classes_tau_{TAU_ORB_SENSITIVITY:.2f}"


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


def _errors(y_true, y_pred, norm_stats):
    """z-scored truth and prediction (columns as given), the squared error
    (inf where the prediction is non-finite) and e_t = sqrt(mean_d sq)."""
    mean = np.asarray(norm_stats["mean"], dtype=np.float64)
    std = np.asarray(norm_stats["std"], dtype=np.float64)
    z_true = (np.asarray(y_true, dtype=np.float64) - mean) / std
    z_pred = (np.asarray(y_pred, dtype=np.float64) - mean) / std
    with np.errstate(all="ignore"):
        sq = np.where(np.isfinite(z_pred), (z_pred - z_true) ** 2, np.inf)
        err = np.sqrt(sq.mean(axis=2))
    return z_true, z_pred, sq, err


def _vpt(err, theta):
    """Per-row first step t (1-based) with e_t > theta, else the row length."""
    exceed = err > theta
    return np.where(exceed.any(axis=1), exceed.argmax(axis=1) + 1,
                    err.shape[1])


def _summarise(out, name, values):
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    out[f"{name}_median"] = float(np.median(v)) if v.size else float("nan")
    out[f"{name}_mean"] = float(np.mean(v)) if v.size else float("nan")


def _class_rates(unbounded, wrong) -> dict:
    return {"rate_ok": float((~unbounded & ~wrong).mean()),
            "rate_unbounded": float(unbounded.mean()),
            "rate_bounded_wrong": float((~unbounded & wrong).mean())}


def rollout_metrics(y_true, y_pred, norm_stats, system,
                    tau_orb=TAU_ORB) -> dict:
    """All #20 rollout metrics of one split (see module docstring).

    Args:
        y_true, y_pred: (N, T+1, D) de-normalised ground truth and closed-loop
            rollout; column 0 is the initial state and is not scored.
        norm_stats: {"mean", "std", "train_abs_max"} (see make_norm_stats).
        system: neural_ode system name (selects the invariant).
        tau_orb: orbit-error threshold of the bounded_wrong class.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    # (N, T, D); non-finite -> inf in sq, e_t (N, T)
    z_true, z_pred, sq, err = _errors(y_true[:, 1:], y_pred[:, 1:], norm_stats)
    n, T = z_true.shape[:2]
    finite = np.isfinite(z_pred)
    with np.errstate(all="ignore"):
        nrmse = {h: np.sqrt(sq[:, :h].mean(axis=(1, 2))) for h in HORIZONS}
        dist = ((z_pred[:, :, None] - z_true[:, None]) ** 2).sum(axis=-1)
        orbit = np.where(finite.all(axis=(1, 2)),
                         dist.min(axis=2).mean(axis=1), np.inf)
        z_abs_max = np.abs(np.where(finite, z_pred, np.inf)).max(axis=(1, 2))
        half = T // 2
        amp_ratio = (np.sqrt(z_pred[:, half:].var(axis=1).mean(axis=-1))
                     / np.sqrt(z_true[:, half:].var(axis=1).mean(axis=-1)))

    out = {"n_trajectories": n, "horizon": T,
           "non_finite": bool(not finite.all())}
    for theta in (THETA, *THETA_SENSITIVITY):
        vpt = _vpt(err, theta)
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
    collapse_or_sign = (amp_ratio < COLLAPSE_RATIO) | (violations > 0)
    out["tau_orb"] = tau_orb
    out.update(_class_rates(unbounded, collapse_or_sign | (orbit > tau_orb)))
    out[SENSITIVITY_KEY] = _class_rates(
        unbounded, collapse_or_sign | (orbit > TAU_ORB_SENSITIVITY))
    return out


def anchor_metrics(y_true, preds, norm_stats, theta=THETA) -> dict:
    """Multi-anchor endpoint of one split (#62, see module docstring).

    Args:
        y_true: (N, T+1, D) de-normalised ground truth.
        preds: {anchor a: (N, H, D) de-normalised free-run predictions of
            steps a+1..a+H after the teacher-forced warm-up y_0..y_a}; the
            same H for every anchor, a + H <= T.
        norm_stats: as for rollout_metrics.
        theta: VPT threshold of the primary keys.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    horizon = {np.shape(p)[1] for p in preds.values()}
    if len(horizon) != 1:
        raise ValueError(
            f"one free-run horizon for every anchor, got {horizon}")
    (H,) = horizon
    out = {"anchors": list(preds), "anchor_horizon": H, "anchor_theta": theta}
    vpts = {th: [] for th in (theta, *THETA_SENSITIVITY)}
    for a, pred in preds.items():
        if a + H >= y_true.shape[1]:
            raise ValueError(f"anchor {a} + horizon {H} exceeds the "
                             f"{y_true.shape[1] - 1} ground-truth steps")
        _, _, sq, err = _errors(y_true[:, a + 1:a + 1 + H], pred, norm_stats)
        for th in vpts:
            vpts[th].append(_vpt(err, th))
        out[f"vpt_anchor_{a}"] = float(np.median(vpts[theta][-1]))
        out[f"nrmse_32_anchor_{a}"] = float(np.median(
            np.sqrt(sq[:, :32].mean(axis=(1, 2)))))
    out["vpt_multi_median"] = float(np.median(vpts[theta]))
    out["theta_sensitivity"] = {str(th): float(np.median(vpts[th]))
                                for th in THETA_SENSITIVITY}
    return out
