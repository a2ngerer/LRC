import pytest
import numpy as np
from src.tasks.neural_ode.datasets import (
    generate_dataset, generate_stress_dataset,
    STRESS_REGIMES, STRESS_TRAIN_FRACTION, STRESS_NOISE_LEVEL, STRESS_OOD_SCALE,
)


@pytest.mark.parametrize("name", [
    'spiral',
    'duffing',
    'periodic_sinusoidal',
    'periodic_predator_prey',
    'limited_predator_prey',
    'nonlinear_predator_prey',
])
def test_generate_dataset_shape_and_no_nan(name):
    t, y = generate_dataset(name)
    assert t.shape == (1000,), f"Expected t.shape (1000,), got {t.shape}"
    assert y.shape == (1000, 2), f"Expected y.shape (1000, 2), got {y.shape}"
    assert not np.any(np.isnan(t)), "t contains NaN"
    assert not np.any(np.isnan(y)), "y contains NaN"


def test_generate_dataset_invalid_name():
    with pytest.raises(ValueError):
        generate_dataset('invalid_name')


# --- eps-ablation systems + jitter ---

@pytest.mark.parametrize("name", [
    'multitimescale',
    'stiff_linear_k1', 'stiff_linear_k10', 'stiff_linear_k100', 'stiff_linear_k1000',
])
def test_eps_system_shape_and_no_nan(name):
    t, y = generate_dataset(name)
    assert t.shape == (1000,)
    assert y.shape == (1000, 2)
    assert not np.any(np.isnan(y))


def test_jitter_off_is_deterministic():
    """Default (jitter=False) is byte-identical regardless of seed."""
    t0, y0 = generate_dataset('multitimescale')
    t1, y1 = generate_dataset('multitimescale', seed=7, jitter=False)
    assert np.array_equal(y0, y1)


def test_jitter_is_seed_deterministic():
    """jitter=True is reproducible per (name, seed) -- the pairing invariant."""
    _, ya = generate_dataset('multitimescale', seed=3, jitter=True)
    _, yb = generate_dataset('multitimescale', seed=3, jitter=True)
    assert np.array_equal(ya, yb)


def test_jitter_changes_trajectory():
    """jitter=True perturbs y0 -> a different (but bounded) trajectory."""
    _, y_clean = generate_dataset('multitimescale')
    _, y_jit = generate_dataset('multitimescale', seed=1, jitter=True)
    assert not np.array_equal(y_clean, y_jit)


def test_stiff_linear_kappa_jitter_distinct_per_seed():
    _, ya = generate_dataset('stiff_linear_k100', seed=1, jitter=True)
    _, yb = generate_dataset('stiff_linear_k100', seed=2, jitter=True)
    assert not np.array_equal(ya, yb)


def test_stiff_linear_matrix_theta_fixed():
    """A(kappa)=R(pi/4) diag(-1,-kappa) R^-1; eigenvalues are exactly -1,-kappa."""
    from src.tasks.neural_ode.datasets import _stiff_linear_matrix
    A = _stiff_linear_matrix(100)
    eig = sorted(np.linalg.eigvals(A).real)
    assert np.allclose(eig, [-100.0, -1.0])


# --- v5 generalization stress test ---

def test_generate_dataset_y0_override():
    """y0 override starts the trajectory from the given point (OOD-init backbone)."""
    _, y_default = generate_dataset('spiral', data_size=100)
    _, y_shifted = generate_dataset('spiral', data_size=100, y0=[0.9, -0.2])
    assert np.allclose(y_shifted[0], [0.9, -0.2], atol=1e-6)
    assert not np.allclose(y_default[0], y_shifted[0])


def test_stress_regimes_are_the_three_axes():
    assert STRESS_REGIMES == ('noise', 'extrapolation', 'ood_init')


@pytest.mark.parametrize("regime", ['noise', 'extrapolation', 'ood_init'])
def test_generate_stress_dataset_keys_and_eval_shape(regime):
    d = generate_stress_dataset('spiral', regime, data_size=100, seed=0)
    assert set(d) == {'t_train', 'y_train', 't_eval', 'y_eval', 'y0_eval', 'regime'}
    assert d['regime'] == regime
    # eval is always the full clean grid (100 points, 2 dims)
    assert d['t_eval'].shape == (100,)
    assert d['y_eval'].shape == (100, 2)
    assert len(d['y0_eval']) == 2
    assert not np.any(np.isnan(d['y_train']))
    assert not np.any(np.isnan(d['y_eval']))


def test_stress_noise_trains_on_noised_evals_on_clean():
    """noise: train targets are perturbed, eval target equals the clean trajectory."""
    _, y_clean = generate_dataset('duffing', data_size=200)
    d = generate_stress_dataset('duffing', 'noise', data_size=200, seed=0)
    assert d['t_train'].shape == (200,)              # full horizon, both train and eval
    assert np.allclose(d['y_eval'], y_clean)         # eval vs clean truth
    assert not np.allclose(d['y_train'], y_clean)    # train is noised
    # noise is moderate, not destruction: stays within a few std of the signal
    assert np.std(d['y_train'] - y_clean) < np.std(y_clean)


def test_stress_extrapolation_train_is_clean_prefix():
    """extrapolation: train = clean first fraction, eval = clean full horizon."""
    d = generate_stress_dataset('periodic_predator_prey', 'extrapolation',
                                data_size=200, seed=0)
    n_train = int(round(200 * STRESS_TRAIN_FRACTION))
    assert d['t_train'].shape == (n_train,)
    assert d['t_eval'].shape == (200,)
    # the training data is exactly the clean prefix of the eval trajectory
    assert np.allclose(d['y_train'], d['y_eval'][:n_train])


def test_stress_ood_init_perturbs_eval_start():
    """ood_init: eval rolls out from a perturbed y0' != the training y0."""
    d = generate_stress_dataset('spiral', 'ood_init', data_size=200, seed=1)
    # train uses the canonical trajectory; eval starts elsewhere
    assert not np.allclose(d['y_train'][0], d['y_eval'][0])
    assert np.allclose(d['y_eval'][0], d['y0_eval'], atol=1e-6)


def test_stress_dataset_is_seed_deterministic_and_seed_sensitive():
    """Same (system, regime, seed) -> identical; different seed -> different draw.
    This underpins the paired Wilcoxon design (cell-independent realization)."""
    a = generate_stress_dataset('spiral', 'noise', data_size=120, seed=3)
    b = generate_stress_dataset('spiral', 'noise', data_size=120, seed=3)
    c = generate_stress_dataset('spiral', 'noise', data_size=120, seed=4)
    assert np.array_equal(a['y_train'], b['y_train'])
    assert not np.array_equal(a['y_train'], c['y_train'])


def test_generate_stress_dataset_invalid_regime():
    with pytest.raises(ValueError):
        generate_stress_dataset('spiral', 'not_a_regime', data_size=50, seed=0)


# --- v6b stress-level dose-response (per-run level overrides) ---

def test_stress_level_override_defaults_match_v5():
    """Omitting a level override reproduces the v5 baseline byte-for-byte."""
    base = generate_stress_dataset('duffing', 'noise', data_size=200, seed=1)
    expl = generate_stress_dataset('duffing', 'noise', data_size=200, seed=1,
                                   noise_level=STRESS_NOISE_LEVEL)
    assert np.array_equal(base['y_train'], expl['y_train'])


def test_noise_level_is_pure_scalar_multiplier():
    """CRITICAL dose-response invariant: noise at level 0.2 is EXACTLY 2x the 0.1
    realization for the same (system, seed) -- a clean dose-response, not a new
    RNG draw. (If the level were folded into the seed this would fail.)"""
    d1 = generate_stress_dataset('spiral', 'noise', data_size=200, seed=3, noise_level=0.1)
    d2 = generate_stress_dataset('spiral', 'noise', data_size=200, seed=3, noise_level=0.2)
    resid1 = d1['y_train'] - d1['y_eval']     # the injected noise (eval = clean)
    resid2 = d2['y_train'] - d2['y_eval']
    assert np.allclose(resid2, 2.0 * resid1)


def test_ood_scale_is_pure_scalar_multiplier():
    """Same dose-response invariant for the OOD-init displacement."""
    y0 = np.array([0.5, 0.01])                # spiral canonical y0
    o1 = generate_stress_dataset('spiral', 'ood_init', data_size=200, seed=3, ood_scale=0.1)
    o2 = generate_stress_dataset('spiral', 'ood_init', data_size=200, seed=3, ood_scale=0.2)
    disp1 = np.array(o1['y0_eval']) - y0
    disp2 = np.array(o2['y0_eval']) - y0
    assert np.allclose(disp2, 2.0 * disp1)


def test_train_fraction_override_changes_prefix_length():
    """A smaller train_fraction shortens the training trajectory (harder extrapolation)."""
    d30 = generate_stress_dataset('spiral', 'extrapolation', data_size=200, seed=0, train_fraction=0.30)
    d70 = generate_stress_dataset('spiral', 'extrapolation', data_size=200, seed=0, train_fraction=0.70)
    assert d30['t_train'].shape[0] == 60
    assert d70['t_train'].shape[0] == 140
    assert d30['t_eval'].shape[0] == 200 and d70['t_eval'].shape[0] == 200  # eval always full
