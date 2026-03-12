import numpy as np
from scipy.integrate import solve_ivp

_SYSTEMS = {
    'spiral': {'t_span': (0, 25), 'y0': [0.5, 0.01]},
    'duffing': {'t_span': (0, 25), 'y0': [-1, 1]},
    'periodic_sinusoidal': {'t_span': (0, 10), 'y0': [1, 1]},
    'periodic_predator_prey': {'t_span': (0, 10), 'y0': [1, 1]},
    'limited_predator_prey': {'t_span': (0, 20), 'y0': [1, 1]},
    'nonlinear_predator_prey': {'t_span': (0, 20), 'y0': [2, 1]},
}

_A_SPIRAL = np.array([[-0.1, 3.0], [-3.0, -0.1]])
_A_NONLINEAR = 0.33


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


_ODE_FUNCS = {
    'spiral': _spiral,
    'duffing': _duffing,
    'periodic_sinusoidal': _periodic_sinusoidal,
    'periodic_predator_prey': _periodic_predator_prey,
    'limited_predator_prey': _limited_predator_prey,
    'nonlinear_predator_prey': _nonlinear_predator_prey,
}


def generate_dataset(name: str, data_size: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Generate a trajectory for the named ODE system.

    Args:
        name: one of 'spiral', 'duffing', 'periodic_sinusoidal',
              'periodic_predator_prey', 'limited_predator_prey',
              'nonlinear_predator_prey'
        data_size: number of time points

    Returns:
        t: np.ndarray, shape (data_size,)
        y: np.ndarray, shape (data_size, 2)

    Raises:
        ValueError: if name is not recognized
    """
    if name not in _SYSTEMS:
        raise ValueError(f"Unknown system '{name}'. Valid: {list(_SYSTEMS.keys())}")

    params = _SYSTEMS[name]
    t_eval = np.linspace(params['t_span'][0], params['t_span'][1], data_size)

    sol = solve_ivp(
        _ODE_FUNCS[name],
        params['t_span'],
        params['y0'],
        method='DOP853',
        t_eval=t_eval,
        rtol=1e-3,
        atol=1e-6,
    )

    t = sol.t        # (data_size,)
    y = sol.y.T      # (data_size, 2) — solve_ivp returns (2, data_size), so transpose

    return t, y
