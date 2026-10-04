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

Everything is deterministic given a seed, so train and test share the SAME masks
(a column consistently owns the same features).
"""
from __future__ import annotations

import numpy as np


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


def make_views(X, n_columns, mode="partition", overlap=0, noise=0.1, seed=42):
    """Dispatch to a view mode. Returns (views (N,T,K,F), masks (K,F) or None)."""
    if mode == "partition":
        masks = partition_masks(X.shape[-1], n_columns, overlap=overlap, seed=seed)
        return masked_views(X, masks), masks
    if mode == "noisy":
        return noisy_views(X, n_columns, noise=noise, seed=seed), None
    raise ValueError(f"mode must be 'partition' or 'noisy', got {mode!r}")
