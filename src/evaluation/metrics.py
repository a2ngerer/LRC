import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error over all elements."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean((y_true - y_pred) ** 2))


def nrmse(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-12) -> float:
    """Normalized root mean squared error.

    RMSE normalized by the value range of y_true (max - min), so results
    are comparable across ODE systems with different state magnitudes.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    value_range = float(np.max(y_true) - np.min(y_true))
    return float(rmse / (value_range + epsilon))
