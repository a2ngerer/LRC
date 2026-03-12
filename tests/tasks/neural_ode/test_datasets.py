import pytest
import numpy as np
from src.tasks.neural_ode.datasets import generate_dataset


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
