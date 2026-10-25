"""Paired effect sizes and ranking agreement for the benchmark aggregation.

  - p_a_gt_b:          P(A>B) over seed-matched pairs, divergence-aware, with an
                       exact Clopper-Pearson interval (effect size, not a decision
                       rule; the decision stays the Bonferroni-adjusted Wilcoxon)
  - clopper_pearson:   exact binomial interval
  - kendall_tau_b:     rank agreement (protocol pilot, budget-sensitivity tables)
  - bootstrap_mean_ci: percentile bootstrap of an arm mean over seeds, optionally
                       also over test recordings (crossed seeds x recordings)
"""
import numpy as np
from scipy.stats import beta, kendalltau


def clopper_pearson(k, n, alpha=0.05):
    """Exact (Clopper-Pearson) 1 - alpha interval for k successes out of n.

    k may be fractional (ties scored 0.5); the beta quantiles extend to it
    continuously.
    """
    lo = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def p_a_gt_b(x, y, diverged_x=False, diverged_y=False, alpha=0.05):
    """Paired win fraction of A over B with its exact interval: (p, lo, hi).

    Higher is better (negate error metrics). A pair scores 1 if x > y, 0.5 on
    a tie, 0 otherwise. A diverged run (flag set or non-finite value, e.g. the
    rollout class 'unbounded') loses against a finished one; two diverged
    runs tie.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    dx = np.asarray(diverged_x, bool) | ~np.isfinite(x)
    dy = np.asarray(diverged_y, bool) | ~np.isfinite(y)
    with np.errstate(invalid='ignore'):
        score = np.sign(x - y) / 2 + 0.5
    score = np.where(dx | dy, (1.0 + dy - dx) / 2, score)
    k, n = float(score.sum()), len(score)
    return (k / n, *clopper_pearson(k, n, alpha))


def kendall_tau_b(rank_a, rank_b):
    """Kendall tau-b between two rankings (or scores) of the same items."""
    return float(kendalltau(rank_a, rank_b, variant='b').statistic)


def bootstrap_mean_ci(values, resample_cols=False, n_boot=10_000, alpha=0.05,
                      seed=0):
    """Mean and percentile bootstrap interval of a (seeds, recordings) matrix.

    Rows (seeds) are always resampled. resample_cols=True also resamples the
    columns (test recordings), independently of the rows, since every seed is
    evaluated on the same recordings. A 1-D input holds one value per seed.
    Returns (mean, lo, hi).
    """
    m = np.asarray(values, float)
    m = m.reshape(len(m), -1)
    s, r = m.shape
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, s, (n_boot, s))
    cols = (rng.integers(0, r, (n_boot, r)) if resample_cols
            else np.broadcast_to(np.arange(r), (n_boot, r)))
    means = m[rows[:, :, None], cols[:, None, :]].mean(axis=(1, 2))
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(m.mean()), float(lo), float(hi)
