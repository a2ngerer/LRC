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


def confusion_matrix(y_true, y_pred, num_classes: int) -> np.ndarray:
    """(num_classes, num_classes) counts; rows = true, columns = predicted."""
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    return np.bincount(num_classes * y_true + y_pred,
                       minlength=num_classes ** 2).reshape(num_classes,
                                                           num_classes)


def macro_f1(confusion) -> float:
    """Unweighted mean of the per-class F1 scores of a confusion matrix.

    Classes absent from both labels and predictions are left out of the mean,
    as in sklearn's f1_score(average='macro'). Summing per-group matrices
    first gives the pooled score (e.g. over resampled test recordings).
    """
    cm = np.asarray(confusion, dtype=np.float64)
    tp = np.diag(cm)
    denom = cm.sum(axis=0) + cm.sum(axis=1)  # 2 TP + FP + FN per class
    present = denom > 0
    return float(np.mean(2 * tp[present] / denom[present]))


# Chance guard (protocol freeze F1-2026-10-26, issue #80): a macro-F1 at most
# this far above the one-class value of its split counts as at chance.
CHANCE_MARGIN = 0.01


def one_class_macro_f1(counts) -> float:
    """Macro-F1 of always predicting the majority class of a split with these
    per-class label counts: (2p / (1 + p)) / C for majority share p and C
    classes present (the plateau of a classifier that outputs the prior)."""
    counts = np.asarray(counts, dtype=np.float64)
    cm = np.zeros((counts.size, counts.size))
    cm[:, counts.argmax()] = counts
    return macro_f1(cm)


def at_chance(f1, one_class) -> bool:
    """f1 <= one_class + CHANCE_MARGIN; a non-finite f1 counts as at chance."""
    return not f1 > one_class + CHANCE_MARGIN
