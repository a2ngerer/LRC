# Lotka-Volterra (predator-prey) SEQUENCE-ROLLOUT dataset.
#
# Why a separate loader from src/tasks/neural_ode: the neural_ode harness feeds
# each ODE state through the RNN as a length-1 sequence with a fresh zero state
# per Euler step (src/tasks/neural_ode/solver.py). That leaves every cNCP
# delayed-state edge reading zeros -> no gradient -> the recurrent part is never
# trained (see docs/superpowers/specs/2026-07-02-cncp-design.md sec 12). Here the
# same predator-prey dynamics are reframed as next-step prediction and the RNN is
# unrolled over the FULL trajectory (tf.keras.layers.RNN carries the state), so
# the recurrence is actually trained -- the regime cNCP is designed for.
#
# Task: given the state (prey, predator) at step k plus the elapsed time dt,
# predict the state at step k+1. Trained teacher-forced over whole trajectories;
# evaluated both teacher-forced and closed-loop (feed predictions back).
#
# Generalisation: many trajectories from jittered initial conditions, split
# train/test by trajectory, so the test set is HELD-OUT initial conditions -- a
# real dynamics-generalisation test, not the single-trajectory near-tautology the
# neural_ode v5 comments warn about.

from dataclasses import dataclass

import numpy as np

from src.tasks.neural_ode.datasets import generate_dataset, _SYSTEMS

# Default system: periodic_predator_prey (classic Lotka-Volterra, closed orbits):
#   dy0 = 1.5*prey - prey*pred ; dy1 = -3*pred + prey*pred, t in (0, 10).
# Closed limit cycles make the phase-space overlay interpretable: a model that
# learned the dynamics traces the orbit, one that did not spirals in/out. The
# loader is system-agnostic (any 2D system from neural_ode/datasets._SYSTEMS);
# 'duffing' (nonlinear conservative double-well oscillator) is the second
# trajectory task, a structurally different regime than the predator-prey cycle.
SYSTEM = "periodic_predator_prey"
BASE_Y0 = np.array([1.0, 1.0])

# Fixed data-generation seed: every wiring/cell/model-seed must see the SAME
# trajectories and the SAME train/test split, or the comparison is not fair
# (mirrors the fixed permutation seed in person_activity/datasets.py).
DATA_SEED = 20260702

# Reference-frame (TBT) encoding: a grid-cell-like Fourier code of the current
# 2D state, the differentiable analogue of "where on the object am I sensing"
# generalised to a continuous phase space (Hawkins 2019 extends grid cells to
# abstract spaces). Banks of sin/cos at several spatial frequencies give the
# grid-cell locality property (nearby states -> similar codes). Fed to the
# tbt_cNCP L6a->L4 gate (and, for the concat mechanism, into the input) so the
# column knows its position in state space. On this FULLY-OBSERVED task the code
# is a deterministic re-encoding of the input state (a representational feature,
# like a positional encoding), NOT extra information -- unlike active-sensing
# where the location is a genuinely separate sensor channel. Stated honestly in
# the write-up.
_RF_FREQS = (1.0, 2.0, 3.0)   # spatial frequencies (cycles over the z-scored range)
RF_DIM = 2 * len(_RF_FREQS) * 2   # (sin, cos) x freqs x 2 state dims = 12


def reference_frame(state_norm: np.ndarray) -> np.ndarray:
    """Grid-cell-like Fourier code of a z-scored 2D state.

    Args:
        state_norm: (..., 2) normalised (z-scored) state.
    Returns:
        (..., RF_DIM) float32 code, concatenation of sin(pi f s) and cos(pi f s)
        over the frequencies in _RF_FREQS for both state dimensions. Continuous
        and Lipschitz, so nearby states map to nearby codes.
    """
    s = np.asarray(state_norm, dtype=np.float32)
    parts = []
    for f in _RF_FREQS:
        parts.append(np.sin(np.pi * f * s))
        parts.append(np.cos(np.pi * f * s))
    return np.concatenate(parts, axis=-1).astype(np.float32)


@dataclass
class LotkaVolterraData:
    """Next-step train/test tensors plus raw trajectories and norm stats.

    Supervised tensors (N = number of trajectories, T = seq_len):
        *_x:      (N, T, 2)   float32  normalised state at step k (input)
        *_t:      (N, T, 1)   float32  elapsed time dt (constant grid)
        *_y:      (N, T, 2)   float32  normalised state at step k+1 (target)
        *_loc:    (N, T, L)   float32  reference-frame code of the input state
                                       (TBT location signal, L = loc_dim = RF_DIM)
    Raw (de-normalised) trajectories for plotting:
        *_traj:   (N, T+1, 2) float32  the full (prey, predator) trajectory
    Normalisation (per feature dim, z-score fit on TRAIN inputs):
        mean, std: (2,) float32   x_norm = (x_raw - mean) / std
    """
    train_x: np.ndarray
    train_t: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_t: np.ndarray
    test_y: np.ndarray
    train_loc: np.ndarray
    test_loc: np.ndarray
    train_traj: np.ndarray
    test_traj: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    dt: float
    feature_size: int
    seq_len: int
    loc_dim: int
    system: str

    def denormalise(self, x_norm: np.ndarray) -> np.ndarray:
        """Invert the z-score: (..., 2) normalised -> raw (prey, predator)."""
        return x_norm * self.std + self.mean


def _sample_initial_conditions(n: int, rng, base=BASE_Y0, spread=0.4) -> np.ndarray:
    """n initial states around ``base``, each dim uniform in base +/- scale.

    scale = spread * max(|base_dim|, 0.5). Additive (not multiplicative) jitter so
    the sampling is well-defined for negative or near-zero base components (e.g.
    duffing base [-1, 1], spiral [0.5, 0.01]); for the positive predator-prey base
    [1, 1] with spread 0.4 this reproduces the old [0.6, 1.4] range exactly.
    Different amplitudes give a family of nested orbits, so the held-out test
    conditions probe interpolation across the orbit family, not a single curve.
    """
    base = np.asarray(base, dtype=float)
    scale = spread * np.maximum(np.abs(base), 0.5)
    return rng.uniform(base - scale, base + scale, size=(n, 2))


def load_lotka_volterra(n_trajectories: int = 80, seq_len: int = 128,
                        test_fraction: float = 0.2, spread: float = 0.4,
                        seed: int = DATA_SEED,
                        system: str = SYSTEM) -> LotkaVolterraData:
    """Generate the next-step-prediction dataset for a 2D dynamical system.

    Args:
        n_trajectories: total trajectories (train + test).
        seq_len:        supervised timesteps per trajectory. The underlying
                        trajectory has seq_len + 1 points; the last input is the
                        second-to-last state, its target the last state.
        test_fraction:  share of trajectories held out for the test split.
        spread:         initial-condition jitter fraction (see
                        _sample_initial_conditions).
        seed:           data-generation seed (fixed default DATA_SEED so all
                        runs share the identical data and split).
        system:         neural_ode system name (default periodic_predator_prey;
                        'duffing' is the second trajectory benchmark). The initial
                        conditions jitter around the system's canonical y0.

    Returns:
        LotkaVolterraData.
    """
    if system not in _SYSTEMS:
        raise ValueError(f"unknown system {system!r}, valid: {list(_SYSTEMS)}")
    base = np.asarray(_SYSTEMS[system]["y0"], dtype=float)

    rng = np.random.default_rng(seed)
    y0s = _sample_initial_conditions(n_trajectories, rng, base=base,
                                     spread=spread)

    data_size = seq_len + 1
    trajs = np.empty((n_trajectories, data_size, 2), dtype=np.float32)
    t_grid = None
    for i, y0 in enumerate(y0s):
        t, y = generate_dataset(system, data_size=data_size, y0=y0.tolist())
        trajs[i] = y.astype(np.float32)
        t_grid = t
    dt = float(t_grid[1] - t_grid[0])

    # Next-step pairs over the whole trajectory.
    x = trajs[:, :-1, :]          # (N, T, 2) state at step k
    y_target = trajs[:, 1:, :]    # (N, T, 2) state at step k+1
    t_col = np.full((n_trajectories, seq_len, 1), dt, dtype=np.float32)

    # Deterministic split (permutation under the same fixed seed).
    perm = np.random.default_rng(seed + 1).permutation(n_trajectories)
    n_test = max(1, int(round(test_fraction * n_trajectories)))
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    # z-score fit on TRAIN inputs only (avoid test leakage), per feature dim.
    mean = x[train_idx].reshape(-1, 2).mean(axis=0).astype(np.float32)
    std = x[train_idx].reshape(-1, 2).std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    def norm(a):
        return ((a - mean) / std).astype(np.float32)

    train_x_n, test_x_n = norm(x[train_idx]), norm(x[test_idx])

    return LotkaVolterraData(
        train_x=train_x_n, train_t=t_col[train_idx],
        train_y=norm(y_target[train_idx]),
        test_x=test_x_n, test_t=t_col[test_idx],
        test_y=norm(y_target[test_idx]),
        train_loc=reference_frame(train_x_n),
        test_loc=reference_frame(test_x_n),
        train_traj=trajs[train_idx], test_traj=trajs[test_idx],
        mean=mean, std=std, dt=dt, feature_size=2, seq_len=seq_len,
        loc_dim=RF_DIM, system=system,
    )
