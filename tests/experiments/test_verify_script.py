import pytest
from experiments.verify_neural_ode import run_verification, check_convergence


def test_run_verification_output_shape():
    """run_verification returns dict with correct structure."""
    result = run_verification(systems=['spiral'], n_iters=5)

    assert set(result.keys()) == {'timestamp', 'config', 'systems'}
    assert 'spiral' in result['systems']

    sys_data = result['systems']['spiral']
    assert set(sys_data.keys()) == {'initial_loss', 'final_loss', 'loss_history'}
    assert len(sys_data['loss_history']) == 5
    assert sys_data['initial_loss'] > 0
    assert sys_data['final_loss'] > 0
    assert all(isinstance(v, float) for v in sys_data['loss_history'])


def test_check_convergence_passes():
    """check_convergence returns True when all final losses are < 0.5 * initial."""
    results = {
        'systems': {
            'spiral': {'initial_loss': 1.0, 'final_loss': 0.3},
            'duffing': {'initial_loss': 0.8, 'final_loss': 0.2},
        }
    }
    assert check_convergence(results) is True


def test_check_convergence_fails():
    """check_convergence returns False when any final loss >= 0.5 * initial."""
    results = {
        'systems': {
            'spiral': {'initial_loss': 1.0, 'final_loss': 0.3},
            'duffing': {'initial_loss': 0.8, 'final_loss': 0.6},
        }
    }
    assert check_convergence(results) is False
