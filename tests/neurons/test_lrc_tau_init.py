"""Log-uniform time-scale init of the elastance bias in LRC_Cell (issue #87,
protocol F2 arm C7): bias -log(T - 1) with T log-uniform on [tau_min, tau_max],
so the initial gate sigmoid(bias) is 1 / T."""
import numpy as np
import pytest
import tensorflow as tf
from scipy import stats

from src.benchmark.registry import cell_kwargs
from src.neurons import LRC_Cell
from src.neurons.lrc_cell import LogUniformTauBias
from src.wirings import NCPWiring

IN = 3


def _built(units=6, seed=0, masked=False, **kw):
    tf.keras.utils.set_random_seed(seed)
    kw.setdefault("elastance_type", "asymmetric")
    if masked:
        w = NCPWiring(LRC_Cell, 3, 2, 1)
        cell = LRC_Cell(units=6, sparsity_mask=w.adjacency_mask,
                        sensory_mask=w.sensory_mask, **kw)
    else:
        cell = LRC_Cell(units, **kw)
    cell.build((None, IN))
    # one step: the gate Dense builds lazily on the first call
    cell(np.zeros((1, IN), np.float32), [tf.zeros((1, cell.state_size))])
    return cell


def _weights(cell):
    """Weights by name without the layer counter Keras appends."""
    return {"/".join(v.name.split("/")[-2:]) if "elastance" in v.name
            else v.name.split("/")[-1]: v.numpy() for v in cell.weights}


def _implied_tau(bias):
    return 1.0 + np.exp(-np.asarray(bias, np.float64))


def test_draw_is_log_uniform_within_the_bounds():
    tf.keras.utils.set_random_seed(0)
    bias = LogUniformTauBias(1.01, 32.0)((20000,), tf.float32).numpy()
    tau = _implied_tau(bias)
    assert tau.min() >= 1.01 * (1 - 1e-5) and tau.max() <= 32.0 * (1 + 1e-5)
    u = (np.log(tau) - np.log(1.01)) / (np.log(32.0) - np.log(1.01))
    assert stats.kstest(u, "uniform").pvalue > 0.01
    # a uniform draw of T on the same interval fails the same check
    t_lin = np.random.default_rng(0).uniform(1.01, 32.0, 20000)
    u_lin = (np.log(t_lin) - np.log(1.01)) / (np.log(32.0) - np.log(1.01))
    assert stats.kstest(u_lin, "uniform").pvalue < 1e-6


@pytest.mark.parametrize("masked", [False, True])
def test_cell_gate_starts_at_dt_over_tau(masked):
    """At zero input and state the gate of a fresh cell is dt / T, T the
    implied time constant inside the bounds."""
    cell = _built(units=64, masked=masked, tau_init="log_uniform",
                  tau_min=2.0, tau_max=8.0)
    tau = _implied_tau(cell.elastance_mapping.bias.numpy())
    assert tau.min() >= 2.0 * (1 - 1e-5) and tau.max() <= 8.0 * (1 + 1e-5)
    cell._capture_gate = True
    cell([np.zeros((1, IN), np.float32), np.full((1, 1), 0.5, np.float32)],
         [tf.zeros((1, cell.state_size))])
    np.testing.assert_allclose(cell._last_elastance_t.numpy()[0], 0.5 / tau,
                               rtol=1e-5)


def test_draw_follows_the_seed():
    def bias(seed):
        return _built(seed=seed, tau_init="log_uniform").elastance_mapping.bias.numpy()
    np.testing.assert_array_equal(bias(1), bias(1))
    assert not np.allclose(bias(1), bias(2))


def test_defaults_are_the_protocol_bounds():
    cell = LRC_Cell(4, elastance_type="asymmetric")
    assert (cell._tau_init, cell._tau_min, cell._tau_max) == (None, 1.01, 32.0)


@pytest.mark.parametrize("masked", [False, True])
def test_option_changes_only_the_elastance_bias(masked):
    """Same seed: every cell weight but the gate bias is drawn as without the
    option (the bias is the last draw of the cell build). In a full model,
    layers built after the cell (the readout) draw differently."""
    off = _weights(_built(masked=masked))
    on = _weights(_built(masked=masked, tau_init="log_uniform"))
    assert off.keys() == on.keys()
    bias = "elastance_mapping/bias:0"
    assert bias in off
    for k in off:
        if k == bias:
            np.testing.assert_array_equal(off[k], 0.0)
            assert (on[k] != 0.0).all()
        else:
            np.testing.assert_array_equal(off[k], on[k], err_msg=k)


def test_option_off_is_the_cell_without_the_kwargs():
    a = _weights(_built())
    b = _weights(_built(tau_init=None, tau_min=1.01, tau_max=32.0))
    for k in a:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)


@pytest.mark.parametrize("kw", [
    dict(tau_init="uniform"),
    dict(tau_init="log_uniform", elastance_type="symmetric"),
    dict(tau_init="log_uniform", elastance_type="interp"),
    dict(tau_init="log_uniform", tau_min=1.0),
    dict(tau_init="log_uniform", tau_min=4.0, tau_max=4.0),
    dict(tau_init="log_uniform", tau_max=np.inf),
])
def test_invalid_settings_raise(kw):
    with pytest.raises(ValueError):
        LRC_Cell(4, **{"elastance_type": "asymmetric", **kw})


@pytest.mark.parametrize("elastance_type", ["symmetric", "interp"])
def test_elastance_override_on_a_tau_arm_raises(elastance_type):
    """The runners' --elastance-type override cannot silently drop C7."""
    with pytest.raises(ValueError):
        LRC_Cell(4, **cell_kwargs("lrc_tau", elastance_type))


@pytest.mark.parametrize("cell,base", [("lrc_tau", "lrc"),
                                       ("lrc_init_tau", "lrc_init")])
def test_registry_keys_are_the_base_arm_plus_c7(cell, base):
    assert cell_kwargs(cell) == {**cell_kwargs(base), "tau_init": "log_uniform",
                                 "tau_min": 1.01, "tau_max": 32.0}
