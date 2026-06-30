# tests/neurons/test_cfc_lrc_cell.py
# Validation gates for the closed-form LRC cell (see
# docs/superpowers/specs/2026-06-19-cfc-lrc-v3_1-design.md).
import pytest
import tensorflow as tf

from src.neurons import BaseCell, CfC_Cell, CfC_LRC_Cell
from src.models import make_dense_model, make_ncp_model

_NCP = dict(inter_neurons=16, command_neurons=8, motor_neurons=2)


def test_is_base_cell():
    assert isinstance(CfC_LRC_Cell(16), BaseCell)


def test_is_closed_form_no_solver():
    """Closed-form cell: no ODE solver / unfold loop (single forward pass)."""
    cell = CfC_LRC_Cell(16)
    assert not hasattr(cell, '_ode_solver')
    assert not hasattr(cell, '_ode_unfolds')


def test_build_and_forward_dense():
    model = make_dense_model('cfc_lrc', units=16, output_neurons=2,
                             elastance_type='asymmetric')
    out = model(tf.zeros((4, 10, 2)))
    assert tuple(out.shape) == (4, 10, 2)


def test_build_and_forward_ncp():
    model = make_ncp_model('cfc_lrc', seed=42, elastance_type='asymmetric', **_NCP)
    out = model(tf.zeros((4, 10, 2)))
    assert tuple(out.shape) == (4, 10, 2)


def test_eps_in_unit_interval():
    cell = CfC_LRC_Cell(16)
    cell.build((None, 2))
    raw = tf.concat([tf.random.normal((128, 2)), tf.random.normal((128, 16))], -1)
    eps = cell._elastance_value(raw)
    assert float(tf.reduce_min(eps)) >= 0.0
    assert float(tf.reduce_max(eps)) <= 1.0


def test_warmstart_bias_pushes_eps_high():
    """Default elastance_init_bias warm-starts eps above 0.5 (near plain CfC)."""
    cell = CfC_LRC_Cell(16, elastance_init_bias=4.0)
    cell.build((None, 2))
    raw = tf.concat([tf.zeros((32, 2)), tf.zeros((32, 16))], -1)
    eps = cell._elastance_value(raw)
    # zero input/state -> eps = sigmoid(bias) ~ 0.98
    assert float(tf.reduce_mean(eps)) > 0.9


def test_reduces_to_cfc_when_eps_one():
    """eps forced to 1 + shared backbone/heads -> identical to plain CfC."""
    cfc = CfC_Cell(16)
    cfc.build((None, 2))
    clrc = CfC_LRC_Cell(16)
    clrc.build((None, 2))

    clrc._backbone[0].set_weights(cfc._backbone[0].get_weights())
    for a, b in [(clrc._ff1, cfc._ff1), (clrc._ff2, cfc._ff2),
                 (clrc._time_a, cfc._time_a), (clrc._time_b, cfc._time_b)]:
        a.set_weights(b.get_weights())
    # zero kernel, large positive bias -> eps = sigmoid(30) ~ 1.0
    k, bvec = clrc._elastance.get_weights()
    clrc._elastance.set_weights([k * 0.0, bvec * 0.0 + 30.0])

    inp = tf.random.normal((7, 2))
    st = [tf.random.normal((7, 16))]
    o_cfc, _ = cfc.call(inp, st)
    o_clrc, _ = clrc.call(inp, st)
    assert float(tf.reduce_max(tf.abs(o_cfc - o_clrc))) < 1e-6


def test_invalid_elastance_type_raises():
    with pytest.raises(ValueError):
        CfC_LRC_Cell(16, elastance_type='interp')


def test_symmetric_builds_and_bounded():
    cell = CfC_LRC_Cell(16, elastance_type='symmetric')
    cell.build((None, 2))
    raw = tf.concat([tf.random.normal((64, 2)), tf.random.normal((64, 16))], -1)
    eps = cell._elastance_value(raw)
    assert float(tf.reduce_min(eps)) >= 0.0
    assert float(tf.reduce_max(eps)) <= 1.0
    # forward pass works end-to-end in a model
    model = make_dense_model('cfc_lrc', units=8, output_neurons=2,
                             elastance_type='symmetric')
    assert tuple(model(tf.zeros((3, 5, 2))).shape) == (3, 5, 2)


def test_gradient_flow_no_vanishing():
    """The extra eps multiply must not zero out or NaN any gradient."""
    model = make_dense_model('cfc_lrc', units=16, output_neurons=2,
                             elastance_type='asymmetric')
    x = tf.random.normal((8, 20, 2))
    y = tf.random.normal((8, 20, 2))
    model(x)
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean((model(x) - y) ** 2)
    grads = tape.gradient(loss, model.trainable_variables)
    assert all(g is not None for g in grads)
    assert all(bool(tf.math.is_finite(tf.norm(g))) for g in grads)
    assert min(float(tf.norm(g)) for g in grads) > 0.0
