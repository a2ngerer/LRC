import numpy as np
from scipy.integrate import solve_ivp

_SYSTEMS = {
    'spiral': {'t_span': (0, 25), 'y0': [0.5, 0.01]},
    'duffing': {'t_span': (0, 25), 'y0': [-1, 1]},
    'periodic_sinusoidal': {'t_span': (0, 10), 'y0': [1, 1]},
    'periodic_predator_prey': {'t_span': (0, 10), 'y0': [1, 1]},
    'limited_predator_prey': {'t_span': (0, 20), 'y0': [1, 1]},
    'nonlinear_predator_prey': {'t_span': (0, 20), 'y0': [2, 1]},
    # eps-ablation: confirmatory headline host. Coexisting slow (0.05) and fast
    # (8) timescales give the inner liquid-elastance gate a timescale-separation
    # structure to exploit (spec section 2.2).
    'multitimescale': {'t_span': (0, 20), 'y0': [1.0, 0.0]},
}

_A_SPIRAL = np.array([[-0.1, 3.0], [-3.0, -0.1]])
_A_NONLINEAR = 0.33

# eps-ablation: stiff_linear eigenvalue-spread knob kappa. A(kappa) =
# R(theta) diag(-1, -kappa) R(theta)^-1 with theta = pi/4 fixed; kappa stresses
# the OUTER Euler step, not the inner gate (spec section 2.1). Registered as
# stiff_linear_k1 / _k10 / _k100 / _k1000 below via closures over kappa.
_STIFF_THETA = np.pi / 4.0
_STIFF_KAPPAS = (1, 10, 100, 1000)


def _stiff_linear_matrix(kappa: float) -> np.ndarray:
    c, s = np.cos(_STIFF_THETA), np.sin(_STIFF_THETA)
    R = np.array([[c, -s], [s, c]])
    return R @ np.diag([-1.0, -float(kappa)]) @ R.T   # R^-1 == R.T (rotation)


def _make_stiff_linear(kappa: float):
    A = _stiff_linear_matrix(kappa)

    def _f(t, y):
        return A @ y

    return _f


def _spiral(t, y):
    return y @ _A_SPIRAL


def _duffing(t, y):
    return [y[1], y[0] - y[0] ** 3]


def _periodic_sinusoidal(t, y):
    r = np.sqrt(y[0] ** 2 + y[1] ** 2)
    return [y[0] * (1 - r) - y[1], y[0] + y[1] * (1 - r)]


def _periodic_predator_prey(t, y):
    return [1.5 * y[0] - 1.0 * y[0] * y[1], -3.0 * y[1] + 1.0 * y[0] * y[1]]


def _limited_predator_prey(t, y):
    return [y[0] * (1 - y[0]) - y[0] * y[1], -y[1] + 2.0 * y[0] * y[1]]


def _nonlinear_predator_prey(t, y):
    return [
        y[0] * (1 - y[0]) + _A_NONLINEAR * y[0] * y[1],
        y[1] * (1 - y[1]) + y[0] * y[1],
    ]


def _multitimescale(t, y):
    # dy0 = -0.05 y0 + sin(8 y1) ; dy1 = 8 y0 - 0.05 y1  (spec section 2.2)
    return [-0.05 * y[0] + np.sin(8.0 * y[1]), 8.0 * y[0] - 0.05 * y[1]]


_ODE_FUNCS = {
    'spiral': _spiral,
    'duffing': _duffing,
    'periodic_sinusoidal': _periodic_sinusoidal,
    'periodic_predator_prey': _periodic_predator_prey,
    'limited_predator_prey': _limited_predator_prey,
    'nonlinear_predator_prey': _nonlinear_predator_prey,
    'multitimescale': _multitimescale,
}

# eps-ablation: register the four stiff_linear systems (closures over kappa).
for _k in _STIFF_KAPPAS:
    _SYSTEMS[f'stiff_linear_k{_k}'] = {'t_span': (0, 8), 'y0': [1.0, 1.0]}
    _ODE_FUNCS[f'stiff_linear_k{_k}'] = _make_stiff_linear(_k)


def _stiff_kappa(name: str):
    """Return kappa for a stiff_linear_k<kappa> system name, else None."""
    if name.startswith('stiff_linear_k'):
        return int(name[len('stiff_linear_k'):])
    return None


# eps-ablation: seeded data-level jitter magnitudes (frozen before the pilot,
# spec section 3). y0 Gaussian jitter at sigma_y0 = 0.02 * ||y0||; for
# stiff_linear a parametric perturbation sigma_kappa = 0.02 * kappa. The 2% is
# deliberately small so it does not perturb the eigenstructure (equal real parts
# of spiral, the anchors' timescale separation). jitter is OFF unless a seed is
# passed AND jitter=True, so every v1-v6 caller stays byte-identical.
EPS_JITTER_Y0_FRAC = 0.02
EPS_JITTER_KAPPA_FRAC = 0.02


def generate_dataset(name: str, data_size: int = 1000,
                     y0=None, seed=None, jitter: bool = False
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Generate a trajectory for the named ODE system.

    Args:
        name: one of 'spiral', 'duffing', 'periodic_sinusoidal',
              'periodic_predator_prey', 'limited_predator_prey',
              'nonlinear_predator_prey', 'multitimescale',
              'stiff_linear_k1', 'stiff_linear_k10', 'stiff_linear_k100',
              'stiff_linear_k1000'
        data_size: number of time points
        y0: optional initial state override (length-2 sequence). Defaults to the
            system's canonical y0. Used by the v5 OOD-init stressor to roll the
            *same* dynamics from a perturbed starting point.
        seed: seed for the eps-ablation data-level jitter (only used when
              jitter=True). Keyed deterministically via np.random.default_rng so
              every condition sharing (name, seed) sees the identical realization.
        jitter: eps-ablation jitter-ON switch. When True (and seed is not None),
              applies the frozen-magnitude data variation: y0 Gaussian jitter
              (sigma_y0 = 0.02*||y0||) and, for stiff_linear, a parametric
              kappa perturbation (sigma_kappa = 0.02*kappa). Default False keeps
              the deterministic single-trajectory path (jitter-OFF mode, also
              used for the optimizer-scatter decomposition, spec section 3.0.4).

    Returns:
        t: np.ndarray, shape (data_size,)
        y: np.ndarray, shape (data_size, 2)

    Raises:
        ValueError: if name is not recognized
    """
    if name not in _SYSTEMS:
        raise ValueError(f"Unknown system '{name}'. Valid: {list(_SYSTEMS.keys())}")

    params = _SYSTEMS[name]
    y0 = list(params['y0']) if y0 is None else list(y0)
    ode_func = _ODE_FUNCS[name]

    if jitter and seed is not None:
        rng = np.random.default_rng(int(seed))
        y0_arr = np.asarray(y0, dtype=float)
        y0 = (y0_arr + rng.normal(0.0, 1.0, size=y0_arr.shape)
              * (EPS_JITTER_Y0_FRAC * np.linalg.norm(y0_arr))).tolist()
        kappa = _stiff_kappa(name)
        if kappa is not None:
            # Perturb kappa parametrically, then rebuild the closure so the
            # matrix reflects the jittered eigenvalue spread.
            kappa_j = kappa + rng.normal(0.0, 1.0) * (EPS_JITTER_KAPPA_FRAC * kappa)
            ode_func = _make_stiff_linear(kappa_j)

    t_eval = np.linspace(params['t_span'][0], params['t_span'][1], data_size)

    sol = solve_ivp(
        ode_func,
        params['t_span'],
        y0,
        method='DOP853',
        t_eval=t_eval,
        rtol=1e-3,
        atol=1e-6,
    )

    t = sol.t        # (data_size,)
    y = sol.y.T      # (data_size, 2) — solve_ivp returns (2, data_size), so transpose

    return t, y


# ---------------------------------------------------------------------------
# v5 generalization stress test (see docs/.../2026-06-22-benchmark-v5-*-design.md)
#
# v1-v4 train and evaluate on the SAME single trajectory (one y0, one regular
# grid, full horizon) -> a clean-fit regime that is near-tautological for the
# continuous-time cells (benchmark-findings v4 Q2 caveat). v5 decouples the
# train and eval trajectories along three independent generalization axes, so
# any robustness gap between cells is forced to surface. The clean baseline is
# v4 itself (identical config, no stress), so v5-vs-v4 is a paired comparison.
# ---------------------------------------------------------------------------
STRESS_REGIMES = ('noise', 'extrapolation', 'ood_init')

# Observation noise: additive Gaussian on the TRAINING targets, std = this
# fraction of each state dimension's trajectory std. Eval is against the clean
# trajectory -> tests whether a cell recovers the dynamics from noisy data.
STRESS_NOISE_LEVEL = 0.1
# Temporal extrapolation: train only on the first fraction of the trajectory,
# evaluate the rollout over the FULL horizon -> the tail is true extrapolation
# (a region of time the field was never trained on).
STRESS_TRAIN_FRACTION = 0.5
# OOD initial condition: eval rolls out from a perturbed y0' (this fraction of
# each dim's trajectory std as the max per-dim displacement), against the true
# trajectory from y0'. Tests whether the vector field generalizes off the single
# training trajectory into the surrounding state space.
STRESS_OOD_SCALE = 0.2


def generate_stress_dataset(name: str, regime: str, data_size: int = 1000,
                            seed: int = 0, noise_level=None,
                            train_fraction=None, ood_scale=None) -> dict:
    """Generate decoupled (train, eval) trajectories for a v5 stress regime.

    The realization depends only on (name, regime, seed) -- never on the cell --
    so every cell sees the identical stressed data for a given (system, seed),
    keeping the paired Wilcoxon design valid (cell-vs-cell and v5-vs-v4).

    Args:
        name:      ODE system (see generate_dataset).
        regime:    one of STRESS_REGIMES.
        data_size: number of time points on the full grid.
        seed:      controls the stochastic part of the stressor (noise draw /
                   OOD displacement); deterministic via np.random.default_rng.
        noise_level / train_fraction / ood_scale: per-run magnitude overrides
                   for the v6b dose-response sweep. None -> the module default
                   (STRESS_NOISE_LEVEL / STRESS_TRAIN_FRACTION / STRESS_OOD_SCALE),
                   so v5 stays byte-identical. CRITICAL: the level is a pure
                   scalar multiplier applied on top of the FIXED draw from
                   np.random.default_rng(seed) -- it is NOT folded into the seed.
                   So level 0.2 == 2x the 0.1 realization for the same
                   (system, seed): a clean dose-response, not a different draw.

    Returns:
        dict with keys:
          't_train', 'y_train' — what train() sees,
          't_eval',  'y_eval'  — what evaluate_full_trajectory() rolls out against,
          'y0_eval'            — the eval rollout's initial state (list of 2),
          'regime'             — echoed back for the result record.

    Raises:
        ValueError: if regime is not recognized.
    """
    if regime not in STRESS_REGIMES:
        raise ValueError(f"Unknown stress regime '{regime}'. Valid: {list(STRESS_REGIMES)}")

    noise_level = STRESS_NOISE_LEVEL if noise_level is None else noise_level
    train_fraction = STRESS_TRAIN_FRACTION if train_fraction is None else train_fraction
    ood_scale = STRESS_OOD_SCALE if ood_scale is None else ood_scale

    rng = np.random.default_rng(seed)
    t, y = generate_dataset(name, data_size=data_size)        # clean, full, canonical y0
    traj_std = y.std(axis=0)                                  # per-dim scale (shape (2,))

    if regime == 'noise':
        sigma = noise_level * traj_std
        y_train = y + rng.normal(0.0, 1.0, size=y.shape) * sigma
        return {
            't_train': t, 'y_train': y_train,
            't_eval': t, 'y_eval': y,                         # eval vs the CLEAN truth
            'y0_eval': y[0].astype(float).tolist(),
            'regime': regime,
        }

    if regime == 'extrapolation':
        n_train = max(2, int(round(data_size * train_fraction)))
        return {
            't_train': t[:n_train], 'y_train': y[:n_train],   # first fraction only
            't_eval': t, 'y_eval': y,                         # full horizon (tail = extrapolation)
            'y0_eval': y[0].astype(float).tolist(),
            'regime': regime,
        }

    # regime == 'ood_init'
    y0 = np.asarray(_SYSTEMS[name]['y0'], dtype=float)
    displacement = ood_scale * traj_std * rng.uniform(-1.0, 1.0, size=y0.shape)
    y0_eval = y0 + displacement
    t_eval, y_eval = generate_dataset(name, data_size=data_size, y0=y0_eval)
    return {
        't_train': t, 'y_train': y,                           # canonical training trajectory
        't_eval': t_eval, 'y_eval': y_eval,                   # true trajectory from perturbed y0'
        'y0_eval': y0_eval.astype(float).tolist(),
        'regime': regime,
    }
