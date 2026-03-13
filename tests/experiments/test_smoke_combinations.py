# tests/experiments/test_smoke_combinations.py
from experiments.smoke_test_combinations import (
    NEURONS, WIRINGS, EXPECTED_FAIL, DENSE_UNITS, NCP_CONFIG,
)


def test_combination_matrix_size():
    """Matrix covers all planned neurons × wirings."""
    assert len(NEURONS) == 4
    assert len(WIRINGS) == 2


def test_expected_fail_is_subset_of_combinations():
    all_combos = {(n, w) for n in NEURONS for w in WIRINGS}
    assert EXPECTED_FAIL.issubset(all_combos)


def test_dense_units_covers_all_neurons():
    for n in NEURONS:
        assert n in DENSE_UNITS


def test_lrc_ar_dense_units_equals_spiral_features():
    """lrc_ar must have units == input features (2 for spiral)."""
    assert DENSE_UNITS['lrc_ar'] == 2


def test_ncp_motor_neurons_equals_spiral_features():
    """motor_neurons == output features == 2 for spiral."""
    assert NCP_CONFIG['motor_neurons'] == 2
