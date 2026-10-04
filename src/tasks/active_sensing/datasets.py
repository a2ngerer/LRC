# Active-sensing 2D object recognition -- the testable task for tbt_cNCP
# (concept: docs/superpowers/specs/2026-07-04-tbt-cncp-concept.md).
#
# Objects are synthetic 2D shapes rendered on a grid (fully offline and
# deterministic -- no dataset download, so it runs unchanged on cluster compute
# nodes). An episode is a SEQUENCE of glimpses: at each step a sensor sits at a
# location on the object, observes a small local patch (the "sensation"), and
# the model is told WHERE the glimpse came from (the location / reference-frame
# signal). The model classifies the object; accuracy is tracked as a function of
# the number of glimpses (the TBT sample-efficiency signature).
#
# Why a location signal should matter: a single local patch is ambiguous (a
# corner of a square looks like a corner of a triangle). Only by binding each
# patch to its location can the sequence of glimpses reconstruct the global
# shape. tbt_cNCP receives that binding; the no-location control does not.

from dataclasses import dataclass

import numpy as np

SHAPE_CLASSES = ["square", "circle", "cross", "diagonal_x", "triangle",
                 "diamond"]

# Fixed data-generation seed: every wiring/cell/model-seed sees the SAME objects
# and the SAME glimpse sequences and split (fair comparison, as in the other
# tasks).
DATA_SEED = 20260704


@dataclass
class ActiveSensingData:
    """Glimpse-sequence tensors + metadata for the active-sensing task.

    N = episodes, T = glimpses per episode, P = patch_dim (K*K),
    L = location code dim.
        *_patch: (N, T, P)   float32  local patch (sensation) at each glimpse
        *_loc:   (N, T, L)   float32  grid-cell-like location code
        *_time:  (N, T, 1)   float32  elapsed time (constant 1.0)
        *_y:     (N,)        int32    object class
        *_next:  (N, T, P)   float32  next-glimpse patch (self-supervised target)
    """
    train_patch: np.ndarray
    train_loc: np.ndarray
    train_time: np.ndarray
    train_y: np.ndarray
    train_next: np.ndarray
    test_patch: np.ndarray
    test_loc: np.ndarray
    test_time: np.ndarray
    test_y: np.ndarray
    test_next: np.ndarray
    num_classes: int
    patch_dim: int
    loc_dim: int
    seq_len: int
    grid: int
    k: int
    n_columns: int = 1   # >1 adds a column axis: *_patch (N,T,K,P), *_loc (N,T,K,L)


def _render(name: str, grid: int, rng, jitter: float = 2.0,
            noise: float = 0.05) -> np.ndarray:
    """Render one shape as a (grid, grid) float array in [0, 1]."""
    g = grid
    img = np.zeros((g, g), dtype=np.float32)
    c = (g - 1) / 2.0
    dr, dc = rng.uniform(-jitter, jitter, size=2)
    ii, jj = np.mgrid[0:g, 0:g].astype(np.float32)
    ri, rj = ii - c - dr, jj - c - dc
    r = 0.30 * g
    if name == "square":
        m = (np.maximum(np.abs(ri), np.abs(rj)) > r - 0.9) & \
            (np.maximum(np.abs(ri), np.abs(rj)) < r + 0.9)
    elif name == "circle":
        rad = np.sqrt(ri ** 2 + rj ** 2)
        m = np.abs(rad - r) < 1.0
    elif name == "cross":
        m = (np.abs(ri) < 1.0) | (np.abs(rj) < 1.0)
        m &= (np.maximum(np.abs(ri), np.abs(rj)) < r + 1.0)
    elif name == "diagonal_x":
        m = (np.abs(ri - rj) < 1.2) | (np.abs(ri + rj) < 1.2)
        m &= (np.maximum(np.abs(ri), np.abs(rj)) < r + 1.0)
    elif name == "triangle":
        base = np.abs(ri - r) < 1.0
        left = np.abs(rj + (ri + r) * 0.5) < 1.0
        right = np.abs(rj - (ri + r) * 0.5) < 1.0
        m = (base | left | right) & (ri < r + 1.0) & (ri > -r - 1.0)
    elif name == "diamond":
        d = np.abs(ri) + np.abs(rj)
        m = np.abs(d - r) < 1.0
    else:
        raise ValueError(f"unknown shape {name!r}")
    img[m] = 1.0
    img = img + rng.normal(0.0, noise, size=img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


# grid-cell-like location code: multi-frequency sin/cos over normalised (r, c).
_FREQS = np.array([1.0, 2.0, 4.0], dtype=np.float32)


def encode_location(centers_norm: np.ndarray) -> np.ndarray:
    """(..., 2) normalised centres in [0,1] -> (..., 12) grid-cell-like code.

    For each of the two axes and each frequency f: [sin(pi f x), cos(pi f x)].
    Multi-frequency, orientation-separable, "nearby locations -> similar code"
    -- the differentiable analogue of several grid modules (concept spec 2.1).
    """
    x = centers_norm[..., 0:1]
    y = centers_norm[..., 1:2]
    feats = []
    for f in _FREQS:
        feats += [np.sin(np.pi * f * x), np.cos(np.pi * f * x),
                  np.sin(np.pi * f * y), np.cos(np.pi * f * y)]
    return np.concatenate(feats, axis=-1).astype(np.float32)


def compositional_templates(n_classes: int, grid: int, n_dots: int = 5,
                            seed: int = DATA_SEED):
    """One fixed dot-configuration per class (same #dots, different positions).

    All classes share the identical local motif (a blob), so a single patch is
    uninformative about the class -- only the SPATIAL CONFIGURATION separates
    the classes. This makes the location signal necessary (the fair H1 test),
    unlike the locally-distinguishable shape set.
    """
    rng = np.random.default_rng(seed + 777)
    lo, hi = int(0.15 * grid), int(0.85 * grid)
    return [rng.integers(lo, hi, size=(n_dots, 2)) for _ in range(n_classes)]


def _render_compositional(dots: np.ndarray, grid: int, rng, jitter: float = 1.0,
                          noise: float = 0.05) -> np.ndarray:
    """Render blobs at the (jittered) dot centres of one class template."""
    g = grid
    img = np.zeros((g, g), dtype=np.float32)
    ii, jj = np.mgrid[0:g, 0:g].astype(np.float32)
    for (di, dj) in dots:
        ci = di + rng.uniform(-jitter, jitter)
        cj = dj + rng.uniform(-jitter, jitter)
        img = np.maximum(img, np.exp(-((ii - ci) ** 2 + (jj - cj) ** 2) / 2.0))
    img = img + rng.normal(0.0, noise, size=img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def _glimpse(img: np.ndarray, ci: int, cj: int, k: int) -> np.ndarray:
    """Extract a (k, k) patch centred at (ci, cj), zero-padded at borders."""
    g = img.shape[0]
    h = k // 2
    pad = np.zeros((g + 2 * h, g + 2 * h), dtype=np.float32)
    pad[h:h + g, h:h + g] = img
    return pad[ci:ci + k, cj:cj + k]


def load_active_sensing(n_per_class: int = 360, seq_len: int = 12, grid: int = 20,
                        k: int = 5, test_fraction: float = 0.2,
                        occlude: bool = False, occlude_frac: float = 0.0,
                        mode: str = "compositional",
                        n_classes: int = 6, n_dots: int = 5, n_columns: int = 1,
                        seed: int = DATA_SEED) -> ActiveSensingData:
    """Generate the active-sensing glimpse dataset.

    Args:
        n_per_class: episodes per class.
        seq_len:     glimpses per episode (T).
        grid:        object grid size.
        k:           glimpse patch size (k x k).
        test_fraction: held-out share (by episode).
        occlude:     if True, blanks the left half of every TEST object
                     (H3 occlusion stressor); train stays clean. Shorthand for
                     occlude_frac=0.5.
        occlude_frac: fraction of the object width (from the left) whose TEST
                     glimpses are blanked; 0.0 = none, 0.5 = left half, 0.75 =
                     left three quarters. Enables the graded occlusion-robustness
                     sweep. Overrides occlude when > 0.
        mode:        'compositional' (default; shared local motif, class =
                     spatial configuration -> location necessary, the fair H1
                     test) or 'shapes' (locally-distinguishable outlines, where
                     location is largely redundant).
        n_classes:   number of classes (compositional mode).
        n_dots:      dots per compositional object.
        seed:        data-generation seed (fixed default).

    Returns:
        ActiveSensingData.
    """
    if mode not in ("compositional", "shapes"):
        raise ValueError(f"mode must be 'compositional' or 'shapes', got {mode!r}")
    rng = np.random.default_rng(seed)
    if mode == "shapes":
        n_classes = len(SHAPE_CLASSES)
    templates = (compositional_templates(n_classes, grid, n_dots, seed)
                 if mode == "compositional" else None)
    N = n_per_class * n_classes
    patch_dim = k * k
    K = int(n_columns)

    patches = np.zeros((N, seq_len, K, patch_dim), dtype=np.float32)
    centers = np.zeros((N, seq_len, K, 2), dtype=np.float32)
    labels = np.zeros((N,), dtype=np.int32)

    idx = 0
    for cls in range(n_classes):
        for _ in range(n_per_class):
            if mode == "compositional":
                img = _render_compositional(templates[cls], grid, rng)
            else:
                img = _render(SHAPE_CLASSES[cls], grid, rng)
            # K independent glimpse streams over the SAME object (one per column).
            gi = rng.integers(0, grid, size=(seq_len, K))
            gj = rng.integers(0, grid, size=(seq_len, K))
            for t in range(seq_len):
                for kk in range(K):
                    patches[idx, t, kk] = _glimpse(
                        img, int(gi[t, kk]), int(gj[t, kk]), k).ravel()
                    centers[idx, t, kk] = [gi[t, kk] / grid, gj[t, kk] / grid]
            labels[idx] = cls
            idx += 1

    time = np.ones((N, seq_len, 1), dtype=np.float32)
    loc = encode_location(centers)                       # (N,T,K,L)
    nxt = np.concatenate([patches[:, 1:], patches[:, -1:]], axis=1)

    perm = np.random.default_rng(seed + 1).permutation(N)
    n_test = int(round(test_fraction * N))
    te, tr = perm[:n_test], perm[n_test:]

    # occlude_frac in (0,1] blanks every TEST glimpse whose center-x falls in the
    # left `frac` of the object (graded H3 stressor); occlude=True is the
    # frac=0.5 shorthand. Train stays clean either way.
    frac = occlude_frac if occlude_frac > 0.0 else (0.5 if occlude else 0.0)
    if frac > 0.0:
        left = centers[te, :, :, 1] < frac               # (n_test, T, K)
        patches_te = patches[te].copy()
        patches_te[left] = 0.0
    else:
        patches_te = patches[te]

    if K == 1:
        # Squeeze the column axis for the single-column contract
        # (*_patch (N,T,P), *_loc (N,T,L)); byte-identical to the pre-K version.
        patches, loc, nxt = patches[:, :, 0], loc[:, :, 0], nxt[:, :, 0]
        patches_te = patches_te[:, :, 0]

    return ActiveSensingData(
        train_patch=patches[tr], train_loc=loc[tr], train_time=time[tr],
        train_y=labels[tr], train_next=nxt[tr],
        test_patch=patches_te, test_loc=loc[te], test_time=time[te],
        test_y=labels[te], test_next=nxt[te],
        num_classes=n_classes, patch_dim=patch_dim, loc_dim=loc.shape[-1],
        seq_len=seq_len, grid=grid, k=k, n_columns=K)
