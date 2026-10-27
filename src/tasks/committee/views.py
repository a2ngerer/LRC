"""Partial-view operators for the committee (Iteration 7).

Turn a full multivariate sequence X (N, T, F) into K PARTIAL views (N, T, K, F),
one per column, so each weight-shared column sees only a slice of the whole and
the columns must vote to reconstruct it. Two view modes, plus a test-time
corruption for the robustness axis:

  - partition (masked): each column owns a disjoint (optionally overlapping)
    subset of the F features; the rest are zeroed. This is the literal "each
    column gets a part of the whole" -- no single column is sufficient.
  - noisy: K independently noised copies of the FULL input. For low-dimensional
    tasks (e.g. Lotka-Volterra, F=2) where a feature partition is too coarse;
    consensus then averages out per-column observation noise.

  - drop_features: zero a fraction of whole channels at TEST time (sensor
    failure). The committee's hypothesised win is graceful degradation here.

  - average_views: collapse the K views to one (mean, lower median or the
    fault-aware mean), the input of the input-averaging monolith controls.

Everything is deterministic given a seed, so train and test share the SAME masks
(a column consistently owns the same features).

Consensus-study protocol (Chapter 5, protocol freeze F1-2026-10-26, issue #55):
every column receives the SAME window (identical views); the only source of
diversity is the corruption of corrupt_views -- Gaussian noise on the
coordinates (noise_sigma) and exactly m faulty columns per window whose
coordinates are zeroed for the whole window, m from the K-specific grid
fault_counts. channel_fault is the correlated-failure sensitivity check. The
partition / noisy modes and drop_features / add_noise above belong to the
exploratory July 2026 path and are not part of the suite.
"""
from __future__ import annotations

import numpy as np

# Person Activity feature layout: sensor one-hot (4) ++ x, y, z.
COORDS = slice(4, 7)


def partition_masks(feature_size, n_columns, overlap=0, seed=42):
    """K boolean masks over F features: a balanced (optionally overlapping) split.

    Each feature is assigned to exactly one column (disjoint partition); with
    overlap > 0 every column additionally borrows `overlap` random features from
    outside its group, so views can share context. Returns (K, F) bool with every
    row having at least one True and every feature covered by >= 1 column.
    """
    F, K = int(feature_size), int(n_columns)
    if K < 1 or F < 1:
        raise ValueError("feature_size and n_columns must be >= 1")
    rng = np.random.default_rng(seed)
    order = rng.permutation(F)
    groups = np.array_split(order, K)               # balanced disjoint groups
    masks = np.zeros((K, F), dtype=bool)
    for k, g in enumerate(groups):
        masks[k, g] = True
        if overlap > 0:
            outside = np.setdiff1d(np.arange(F), g, assume_unique=False)
            if outside.size:
                extra = rng.choice(outside, size=min(overlap, outside.size),
                                   replace=False)
                masks[k, extra] = True
    empty = ~masks.any(axis=1)                      # guard tiny F, large K
    if empty.any():
        for k in np.where(empty)[0]:
            masks[k, rng.integers(F)] = True
    return masks


def masked_views(X, masks):
    """(N,T,F) x (K,F) bool -> (N,T,K,F): column k sees X with non-owned dims 0."""
    X = np.asarray(X, dtype=np.float32)
    m = masks.astype(np.float32)                    # (K,F)
    return X[:, :, None, :] * m[None, None, :, :]   # broadcast to (N,T,K,F)


def noisy_views(X, n_columns, noise=0.1, seed=42):
    """(N,T,F) -> (N,T,K,F): K Gaussian-noised copies of the full input."""
    X = np.asarray(X, dtype=np.float32)
    rng = np.random.default_rng(seed)
    K = int(n_columns)
    reps = np.repeat(X[:, :, None, :], K, axis=2)   # (N,T,K,F)
    if noise > 0.0:
        reps = reps + rng.normal(0.0, noise, size=reps.shape).astype(np.float32)
    return reps


def drop_features(X, frac, seed=0):
    """Zero a fraction of whole channels (sensor failure) for the WHOLE set.

    Deterministic given seed, so a robustness sweep over frac is reproducible.
    frac <= 0 returns X unchanged. Returns a corrupted copy of X (N,T,F).
    """
    X = np.asarray(X, dtype=np.float32)
    if frac <= 0.0:
        return X.copy()
    F = X.shape[-1]
    n_drop = int(round(frac * F))
    if n_drop <= 0:
        return X.copy()
    rng = np.random.default_rng(seed)
    drop = rng.choice(F, size=min(n_drop, F), replace=False)
    out = X.copy()
    out[:, :, drop] = 0.0
    return out


def add_noise(X, sigma, seed=0):
    """Additive Gaussian noise corruption (test-time), deterministic per seed.

    The UNANTICIPATED-corruption axis for Iteration 7d: a monolith augmented with
    channel dropout at train time has never seen additive noise, so testing on
    noise asks whether the committee's architectural robustness generalises across
    corruption types where misspecified dropout training does not. sigma <= 0
    returns X unchanged. Returns a corrupted copy of X (N,T,F).
    """
    X = np.asarray(X, dtype=np.float32)
    if sigma <= 0.0:
        return X.copy()
    rng = np.random.default_rng(seed)
    return (X + rng.normal(0.0, sigma, size=X.shape)).astype(np.float32)


def average_views(views, how="mean"):
    """(N,T,K,F) -> (N,T,1,F): the input of the input-averaging monolith.

    how='mean' is the coordinate-wise mean of the K views; how='median' the
    coordinate-wise LOWER median (index (K - 1) // 2 of the sorted views, as
    consensus='median'). With a fraction rho of the views zeroed (column
    failure) the mean shrinks the signal by (1 - rho), the median returns the
    intact value while rho < 0.5. On partition views a feature owned by fewer
    than half of the columns has median 0, so the median monolith needs
    mode='noisy' (or enough overlap). how='fault_mean' is fault_aware_mean.
    """
    if how == "mean":
        return views.mean(axis=2, keepdims=True)
    if how == "median":
        k = (views.shape[2] - 1) // 2
        return np.partition(views, k, axis=2)[:, :, k:k + 1]
    if how == "fault_mean":
        return fault_aware_mean(views)
    raise ValueError(
        f"how must be 'mean', 'median' or 'fault_mean', got {how!r}")


def make_views(X, n_columns, mode="partition", overlap=0, noise=0.1, seed=42):
    """Dispatch to a view mode. Returns (views (N,T,K,F), masks (K,F) or None)."""
    if mode == "identical":
        return noisy_views(X, n_columns, noise=0.0), None
    if mode == "partition":
        masks = partition_masks(X.shape[-1], n_columns, overlap=overlap, seed=seed)
        return masked_views(X, masks), masks
    if mode == "noisy":
        return noisy_views(X, n_columns, noise=noise, seed=seed), None
    raise ValueError(
        f"mode must be 'identical', 'partition' or 'noisy', got {mode!r}")


# --- consensus-study corruption protocol (issue #55) ---

def fault_counts(n_columns, rhos=(0, 0.1, 0.3, 0.5, 0.67)):
    """The K-specific fault grid: sorted, de-duplicated round(rho * K).

    K=3 -> [0, 1, 2] (five rhos collapse to three counts), K=30 -> [0, 3, 9,
    15, 20]. K=1 has no column count to round: the grid is the rhos
    themselves, a per-window failure probability (see corrupt_views).
    """
    K = int(n_columns)
    if K == 1:
        return sorted({float(r) for r in rhos})
    return sorted({int(round(r * K)) for r in rhos})


def noise_sigma(train_x, frac=0.3, coords=COORDS):
    """(F,) noise scale: frac * the training-split std of each coordinate
    feature (the loader does not normalise), 0 on every other feature."""
    X = np.asarray(train_x, dtype=np.float32)
    sigma = np.zeros(X.shape[-1], np.float32)
    sigma[coords] = frac * X[..., coords].reshape(-1, X[..., coords].shape[-1]
                                                  ).std(axis=0)
    return sigma


def corrupt_views(X, n_columns, n_fail, sigma, rng, coords=COORDS):
    """(N,T,F) -> (N,T,K,F): K identical copies of each window, Gaussian noise
    of scale sigma[f] on feature f of every copy, then exactly n_fail columns
    per window (drawn without replacement) with their coordinates zeroed for
    the whole window; the one-hot features are never touched.

    For K=1, n_fail is a probability p: the single view fails with
    Bernoulli(p) per window. All draws come from rng (np.random.Generator):
    noise first, then the faulty columns, so equal seeds give equal views.
    """
    X = np.asarray(X, dtype=np.float32)
    N, K = X.shape[0], int(n_columns)
    sigma = np.asarray(sigma, dtype=np.float32)
    v = np.repeat(X[:, :, None, :], K, axis=2)
    noisy = np.flatnonzero(sigma)
    if noisy.size:
        v[..., noisy] += (rng.standard_normal(v[..., noisy].shape)
                          * sigma[noisy]).astype(np.float32)
    if K == 1:
        if not 0.0 <= n_fail <= 1.0:
            raise ValueError(f"K=1: n_fail is a probability, got {n_fail}")
        fail = rng.random((N, 1)) < n_fail
    else:
        if n_fail != int(n_fail) or not 0 <= n_fail <= K:
            raise ValueError(f"n_fail must be an integer in [0, {K}], "
                             f"got {n_fail}")
        fail = np.zeros((N, K), bool)
        if n_fail:
            cols = np.stack([rng.choice(K, int(n_fail), replace=False)
                             for _ in range(N)])
            np.put_along_axis(fail, cols, True, axis=1)
    v[..., coords] = np.where(fail[:, None, :, None], 0.0, v[..., coords])
    return v


def fault_aware_mean(views, coords=COORDS):
    """(N,T,K,F) -> (N,T,1,F): the coordinates averaged over the columns whose
    coordinates are not all zero over the window (0 if every column failed),
    so a column fault neither shrinks nor biases the signal; the remaining
    features (the one-hot, identical in every column) by the ordinary mean.
    """
    out = views.mean(axis=2, keepdims=True)
    c = views[..., coords]                                    # (N,T,K,C)
    alive = (c != 0).any(axis=(1, 3))[:, None, :, None]       # (N,1,K,1)
    n = alive.sum(axis=2, keepdims=True)
    out[..., coords] = np.where(
        n > 0, (c * alive).sum(axis=2, keepdims=True) / np.maximum(n, 1), 0.0)
    return out


def channel_fault(views, channel):
    """Correlated failure: feature `channel` (a coordinate index, e.g. 4 = x)
    zeroed in every column for the whole window. Returns a copy."""
    out = np.array(views, dtype=np.float32, copy=True)
    out[..., channel] = 0.0
    return out
