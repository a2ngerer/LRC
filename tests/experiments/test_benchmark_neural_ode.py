# tests/experiments/test_benchmark_neural_ode.py
from experiments.benchmark_neural_ode import (
    COMBINATIONS, SYSTEMS, DENSE_UNITS, NCP_CONFIG, TRAIN_CONFIG,
)


def test_combinations_count():
    """Exactly 7 valid combinations (lrc_ar+ncp excluded)."""
    assert len(COMBINATIONS) == 7


def test_lrc_ar_ncp_excluded():
    """lrc_ar + NCP must not appear (architectural incompatibility)."""
    assert ('lrc_ar', 'ncp') not in COMBINATIONS


def test_systems_count():
    assert len(SYSTEMS) == 6


def test_systems_names():
    """System names match datasets available in generate_dataset."""
    expected = {
        'spiral', 'duffing', 'periodic_sinusoidal',
        'periodic_predator_prey', 'limited_predator_prey',
        'nonlinear_predator_prey',
    }
    assert set(SYSTEMS) == expected


def test_dense_units_covers_all_neurons():
    neurons_in_combinations = {n for n, _ in COMBINATIONS}
    for n in neurons_in_combinations:
        assert n in DENSE_UNITS


def test_ncp_motor_neurons_equals_ode_features():
    """motor_neurons must equal ODE output dimension (2)."""
    assert NCP_CONFIG['motor_neurons'] == 2
