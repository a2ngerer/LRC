# tests/neurons/test_cfc_lrc_cell.py
# Validation gates for the closed-form LRC cell (see
# docs/superpowers/specs/2026-06-19-cfc-lrc-v3_1-design.md).
import numpy as np
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


def test_outer_gate_reduces_to_cfc_when_eps_one():
    """eps -> 1 recovers plain CfC for the 'outer' gate too (Farsang variant):
    eps*sigmoid(.) -> 1*sigmoid(.). This is the property Farsang noted."""
    cfc = CfC_Cell(16)
    cfc.build((None, 2))
    clrc = CfC_LRC_Cell(16, elastance_gate='outer')
    clrc.build((None, 2))

    clrc._backbone[0].set_weights(cfc._backbone[0].get_weights())
    for a, b in [(clrc._ff1, cfc._ff1), (clrc._ff2, cfc._ff2),
                 (clrc._time_a, cfc._time_a), (clrc._time_b, cfc._time_b)]:
        a.set_weights(b.get_weights())
    k, bvec = clrc._elastance.get_weights()
    clrc._elastance.set_weights([k * 0.0, bvec * 0.0 + 30.0])

    inp = tf.random.normal((7, 2))
    st = [tf.random.normal((7, 16))]
    o_cfc, _ = cfc.call(inp, st)
    o_clrc, _ = clrc.call(inp, st)
    assert float(tf.reduce_max(tf.abs(o_cfc - o_clrc))) < 1e-6


def test_inner_and_outer_differ_at_mid_eps():
    """With eps in (0,1) the two gate placements produce different states --
    they are genuinely distinct mechanisms, not a relabelling. Shared weights
    + eps==0.5 isolate the gate position as the only difference."""
    tf.random.set_seed(0)
    inner = CfC_LRC_Cell(16, elastance_gate='inner')
    inner.build((None, 2))
    outer = CfC_LRC_Cell(16, elastance_gate='outer')
    outer.build((None, 2))

    outer._backbone[0].set_weights(inner._backbone[0].get_weights())
    for a, b in [(outer._ff1, inner._ff1), (outer._ff2, inner._ff2),
                 (outer._time_a, inner._time_a), (outer._time_b, inner._time_b)]:
        a.set_weights(b.get_weights())
    # zero kernel + zero bias -> eps = sigmoid(0) = 0.5 for both cells
    k, bvec = inner._elastance.get_weights()
    for c in (inner, outer):
        c._elastance.set_weights([k * 0.0, bvec * 0.0])

    inp = tf.random.normal((7, 2))
    st = [tf.random.normal((7, 16))]
    o_in, _ = inner.call(inp, st)
    o_out, _ = outer.call(inp, st)
    assert float(tf.reduce_max(tf.abs(o_in - o_out))) > 1e-4


def test_invalid_elastance_gate_raises():
    with pytest.raises(ValueError):
        CfC_LRC_Cell(16, elastance_gate='sideways')


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


# --- head_activation (issue #24): optional tanh on ff1/ff2 ---

_SENS = np.array([[1, 0, 1], [0, 1, 1]], np.float32)            # (input 2, units 3)
_ADJ = np.array([[1, 1, 0], [0, 0, 1], [1, 0, 0]], np.float32)  # (3, 3)
_NCP_BB0 = dict(backbone_layers=0, sparsity_mask=_ADJ, sensory_mask=_SENS)
_HEAD_ARMS = {
    'cfc_bb1': (CfC_Cell, {}),
    'cfc_bb0_ncp': (CfC_Cell, _NCP_BB0),
    'cfc_lrc_bb1': (CfC_LRC_Cell, {}),
    'cfc_lrc_bb0_ncp': (CfC_LRC_Cell, _NCP_BB0),
}
# One-step outputs of _seeded_step, recorded on the cells BEFORE the
# head_activation kwarg existed (benchmarks 2b4f68f): the default must stay
# the linear-head cell.
_DEFAULT_FIXTURE = {
    'cfc_bb1': [[-3.2951531410217285, -1.2200011014938354, 0.03846590965986252], [-0.35560038685798645, -0.929089367389679, 1.619911789894104]],
    'cfc_bb0_ncp': [[-1.5770059823989868, 0.366857647895813, 0.21311990916728973], [3.580240488052368, 1.3405214548110962, -0.26034873723983765]],
    'cfc_lrc_bb1': [[0.058397531509399414, -1.2332098484039307, 0.3973008692264557], [0.8396316766738892, -0.4443165957927704, 1.6694730520248413]],
    'cfc_lrc_bb0_ncp': [[-0.6562386751174927, -1.1248618364334106, 1.1318702697753906], [1.7081272602081299, 0.4391779601573944, -0.7801792621612549]],
}


def _seeded_step(cls, **kwargs):
    """Build a 3-unit cell on 2 inputs, set every weight from a fixed numpy
    seed (no TF RNG, so the values survive Keras init changes) and take one
    step. Returns (cell, new_state)."""
    rng = np.random.default_rng(0)
    cell = cls(3, **kwargs)
    cell.build((None, 2))
    for layer in [*cell._backbone, cell._ff1, cell._ff2, cell._time_a,
                  cell._time_b, *filter(None, [getattr(cell, '_elastance', None)])]:
        layer.set_weights([rng.normal(size=w.shape).astype(np.float32)
                           for w in layer.get_weights()])
    x = tf.constant(rng.normal(size=(2, 2)), tf.float32)
    h = tf.constant(rng.normal(size=(2, 3)), tf.float32)
    out, _ = cell.call(x, [h])
    return cell, out.numpy()


@pytest.mark.parametrize('arm', sorted(_HEAD_ARMS))
def test_default_head_activation_matches_linear_fixture(arm):
    cls, kw = _HEAD_ARMS[arm]
    for extra in ({}, {'head_activation': None}):
        _, out = _seeded_step(cls, **kw, **extra)
        np.testing.assert_allclose(out, _DEFAULT_FIXTURE[arm], rtol=0, atol=1e-6)


@pytest.mark.parametrize('arm', sorted(_HEAD_ARMS))
def test_tanh_bounds_ff_heads_and_leaves_gate_heads(arm):
    """Same weights, large head inputs: tanh ff1/ff2 stay in [-1, 1] where the
    linear heads do not; time_a/time_b are identical in both cells."""
    cls, kw = _HEAD_ARMS[arm]
    lin, _ = _seeded_step(cls, **kw)
    tanh, _ = _seeded_step(cls, head_activation='tanh', **kw)
    feats = tf.constant(100 * np.random.default_rng(1).normal(
        size=(64, tanh._ff1.kernel.shape[0])), tf.float32)
    for name in ('_ff1', '_ff2'):
        assert np.abs(tanh._masked_dense(getattr(tanh, name), feats)).max() <= 1.0
        assert np.abs(lin._masked_dense(getattr(lin, name), feats)).max() > 1.0
    for name in ('_time_a', '_time_b'):
        np.testing.assert_array_equal(
            tanh._masked_dense(getattr(tanh, name), feats),
            lin._masked_dense(getattr(lin, name), feats))


@pytest.mark.parametrize('arm', sorted(_HEAD_ARMS))
def test_head_activation_keeps_parameter_count(arm):
    cls, kw = _HEAD_ARMS[arm]
    lin, _ = _seeded_step(cls, **kw)
    tanh, _ = _seeded_step(cls, head_activation='tanh', **kw)
    assert tanh.count_params() == lin.count_params()
    assert tanh.masked_off_count() == lin.masked_off_count()


@pytest.mark.parametrize('cls', [CfC_Cell, CfC_LRC_Cell])
def test_head_activation_config_round_trip(cls):
    assert cls.from_config(
        cls(4, head_activation='tanh').get_config())._head_activation == 'tanh'
    assert cls.from_config(cls(4).get_config())._head_activation is None


@pytest.mark.parametrize('cls', [CfC_Cell, CfC_LRC_Cell])
def test_invalid_head_activation_raises(cls):
    with pytest.raises(ValueError):
        cls(4, head_activation='relu')
