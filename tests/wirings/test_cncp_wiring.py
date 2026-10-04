# tests/wirings/test_cncp_wiring.py
# Validation gates for the cNCP wiring (see
# docs/superpowers/specs/2026-07-02-cncp-design.md, sections 7 and 9):
# the 7-point cell checklist plus the wiring-level ablation controls.
import numpy as np
import pytest
import tensorflow as tf

from src.models import make_cncp_model
from src.neurons import BaseCell
from src.wirings import (BaseWiring, CNCPWiring, CorticalColumnCell,
                         SignedSparseLinear, SparseLinear, NODE_ORDER,
                         DEFAULT_LAMINA_UNITS, DEFAULT_MASK_DENSITIES)

_ASYM = dict(elastance_type='asymmetric')


def _train_step(model, x, y, lr=1e-3):
    """One Adam step on MSE; returns the (pre-step) loss value."""
    opt = tf.keras.optimizers.Adam(lr)
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean((model(x) - y) ** 2)
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return float(loss)


# --- 1. BaseCell contract ---

def test_is_base_cell():
    assert isinstance(CorticalColumnCell('lrc', **_ASYM), BaseCell)


# --- 2. composite state list / initial state / output size ---

def test_state_size_is_eight_node_list():
    cell = CorticalColumnCell('gru')
    assert cell.state_size == [DEFAULT_LAMINA_UNITS[n] for n in NODE_ORDER]
    assert len(cell.state_size) == 8
    assert cell.output_size == DEFAULT_LAMINA_UNITS['L5ET']


def test_get_initial_state_returns_eight_zero_tensors():
    cell = CorticalColumnCell('gru')
    states = cell.get_initial_state(batch_size=3)
    assert len(states) == 8
    for s, node in zip(states, NODE_ORDER):
        assert tuple(s.shape) == (3, DEFAULT_LAMINA_UNITS[node])
        assert float(tf.reduce_max(tf.abs(s))) == 0.0


def test_lamina_units_override_changes_output_size():
    cell = CorticalColumnCell('gru', lamina_units={'L5ET': 12})
    assert cell.output_size == 12
    assert cell.state_size[NODE_ORDER.index('L5ET')] == 12


# --- 3. single forward pass ---

def test_forward_pass_shapes():
    tf.random.set_seed(0)
    cell = CorticalColumnCell('lrc', **_ASYM)
    x = tf.random.normal((3, 2))
    states = cell.get_initial_state(batch_size=3)
    out, new_states = cell((x, 1.0), states)
    assert tuple(out.shape) == (3, DEFAULT_LAMINA_UNITS['L5ET'])
    assert len(new_states) == 8
    for s, node in zip(new_states, NODE_ORDER):
        assert tuple(s.shape) == (3, DEFAULT_LAMINA_UNITS[node])


# --- 4. irregular sampling ---

def test_irregular_dt_changes_state_for_continuous_cell():
    """Continuous sub-cells (lrc) integrate dt: different dt, different state."""
    tf.random.set_seed(0)
    cell = CorticalColumnCell('lrc', **_ASYM)
    x = tf.random.normal((2, 3))
    states = cell.get_initial_state(batch_size=2)
    out1, _ = cell((x, 1.0), states)
    out2, _ = cell((x, 0.1), states)
    assert not bool(tf.reduce_all(tf.abs(out1 - out2) < 1e-7))


def test_irregular_dt_ignored_for_gru():
    """Discrete sub-cells (gru) discard dt; edges and relays are dt-free."""
    tf.random.set_seed(0)
    cell = CorticalColumnCell('gru')
    x = tf.random.normal((2, 3))
    states = cell.get_initial_state(batch_size=2)
    out1, s1 = cell((x, 1.0), states)
    out2, s2 = cell((x, 0.1), states)
    assert bool(tf.reduce_all(tf.abs(out1 - out2) < 1e-7))
    for a, b in zip(s1, s2):
        assert bool(tf.reduce_all(tf.abs(a - b) < 1e-7))


# --- 5. finiteness over many steps (stiffness guard) ---

def test_state_stays_finite_over_50_steps():
    tf.random.set_seed(0)
    cell = CorticalColumnCell('lrc', **_ASYM)
    x = tf.random.normal((2, 3))
    states = cell.get_initial_state(batch_size=2)
    for _ in range(60):
        out, states = cell((x, 1.0), states)
    assert bool(tf.reduce_all(tf.math.is_finite(out)))
    for s in states:
        assert bool(tf.reduce_all(tf.math.is_finite(s)))


# --- 6. model factory / cell orthogonality ---

@pytest.mark.parametrize('neuron_type,kwargs', [
    ('lrc', _ASYM),
    ('gru', {}),
    ('cfc', {}),
    ('ltc', {}),
])
def test_make_cncp_model_builds_and_runs(neuron_type, kwargs):
    model = make_cncp_model(neuron_type, output_neurons=2, **kwargs)
    out = model(tf.zeros((4, 10, 2)))
    assert tuple(out.shape) == (4, 10, 2)


def test_make_cncp_model_without_projection_exposes_l5et():
    model = make_cncp_model('gru', output_neurons=None)
    out = model(tf.zeros((2, 5, 3)))
    assert tuple(out.shape) == (2, 5, DEFAULT_LAMINA_UNITS['L5ET'])


def test_cncp_wiring_is_base_wiring_and_builds_sequential():
    wiring = CNCPWiring('gru', output_neurons=2)
    assert isinstance(wiring, BaseWiring)
    model = wiring.build_model()
    assert isinstance(model, tf.keras.Sequential)
    assert tuple(model(tf.zeros((2, 5, 3))).shape) == (2, 5, 2)


def test_multi_state_subcell_rejected():
    """Sub-cells must have a single state tensor; lstm ([h, c]) is rejected."""
    model = make_cncp_model('lstm', output_neurons=2)
    with pytest.raises(ValueError):
        model(tf.zeros((2, 5, 2)))


# --- 7. gradient flow through EVERY trainable variable ---

def test_gradient_flow_every_variable():
    """Non-None, finite, non-zero gradient for every trainable variable --
    explicitly including the sign-locked (M_TRN_Thal) and multiplicative
    (gains + apical masks) edges."""
    tf.random.set_seed(0)
    model = make_cncp_model('lrc', output_neurons=2, **_ASYM)
    x = tf.random.normal((4, 20, 2))
    y = tf.random.normal((4, 20, 2))
    model(x)
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean((model(x) - y) ** 2)
    tvars = model.trainable_variables
    grads = tape.gradient(loss, tvars)
    assert len(tvars) > 0

    missing = [v.name for v, g in zip(tvars, grads) if g is None]
    assert not missing, f'no gradient for: {missing}'
    assert all(bool(tf.math.is_finite(tf.norm(g))) for g in grads)
    dead = [v.name for v, g in zip(tvars, grads) if float(tf.norm(g)) == 0.0]
    assert not dead, f'zero gradient for: {dead}'

    # Targeted checks by variable handle (robust to name scoping).
    cell = model.layers[0].cell
    idx = {id(v): i for i, v in enumerate(tvars)}
    targets = {
        'sign_locked_W': cell.edges['M_TRN_Thal'].W,
        'gain_L23': cell.gain_L23,
        'gain_L5ET': cell.gain_L5ET,
        'apical_L23_W': cell.edges['M_ap_L23'].W,
        'apical_L5ET_W': cell.edges['M_ap_L5ET'].W,
    }
    for label, var in targets.items():
        g = grads[idx[id(var)]]
        assert float(tf.norm(g)) > 0.0, f'zero gradient through {label}'


# --- constructor validation ---

def test_invalid_combiner_raises():
    with pytest.raises(ValueError):
        CorticalColumnCell('gru', combiner='divisive')


def test_unknown_lamina_key_raises():
    with pytest.raises(ValueError):
        CorticalColumnCell('gru', lamina_units={'L7': 4})


def test_unknown_mask_density_key_raises():
    with pytest.raises(ValueError):
        CorticalColumnCell('gru', mask_densities={'M_L4_L4': 0.5})


def test_feedforward_plus_divisive_raises():
    with pytest.raises(ValueError):
        CorticalColumnCell('gru', feedforward_only=True,
                           divisive_inhibition=True)


# --- sign-locked edge (spec section 6) ---

def test_signed_sparse_linear_is_strictly_negative():
    mask = np.ones((3, 4), dtype=np.float32)
    layer = SignedSparseLinear(4, mask)
    y = layer(tf.ones((2, 3)))
    # positive input through strictly negative effective weights
    assert (y.numpy() < 0.0).all()
    assert (layer.effective_weight().numpy() < 0.0).all()


def test_sign_locked_edge_stays_negative_after_training():
    tf.random.set_seed(0)
    model = make_cncp_model('gru', output_neurons=2)
    x = tf.random.normal((4, 10, 2))
    y = tf.random.normal((4, 10, 2))
    model(x)
    edge = model.layers[0].cell.edges['M_TRN_Thal']
    assert isinstance(edge, SignedSparseLinear)
    for _ in range(3):
        assert np.isfinite(_train_step(model, x, y))
    eff = edge.effective_weight().numpy()
    assert (eff <= 0.0).all()
    assert (eff[edge._mask_np > 0] < 0.0).all()


# --- wiring-level ablation controls (spec section 9) ---

def test_additive_combiner_ablation():
    """combiner='additive': no apical edges, no gains, still trains."""
    tf.random.set_seed(0)
    model = make_cncp_model('lrc', output_neurons=2, combiner='additive',
                            **_ASYM)
    x = tf.random.normal((2, 10, 2))
    y = tf.random.normal((2, 10, 2))
    assert tuple(model(x).shape) == (2, 10, 2)
    cell = model.layers[0].cell
    assert 'M_ap_L23' not in cell.edges
    assert 'M_ap_L5ET' not in cell.edges
    assert not hasattr(cell, 'gain_L23')
    assert not hasattr(cell, 'gain_L5ET')
    assert np.isfinite(_train_step(model, x, y))


def test_feedforward_reduction_control():
    """cncp_ff: feedback/apical/relay-loop edges removed; the remaining
    laminar chain input->L4->L23->{L5IT,L5ET} builds and trains."""
    tf.random.set_seed(0)
    model = make_cncp_model('lrc', output_neurons=2, feedforward_only=True,
                            **_ASYM)
    x = tf.random.normal((2, 10, 2))
    y = tf.random.normal((2, 10, 2))
    assert tuple(model(x).shape) == (2, 10, 2)

    cell = model.layers[0].cell
    for name in ('M_L5ET_L5IT', 'M_L6CC_L23', 'M_ap_L23', 'M_ap_L5ET',
                 'M_L5_L6CC', 'M_L5_L6CT', 'M_L6CT_Thal', 'M_L6CT_TRN',
                 'M_TRN_Thal', 'M_Thal_L4'):
        assert name not in cell.edges, name
    assert set(cell.subcells) == {'L4', 'L23', 'L5IT', 'L5ET'}
    assert not hasattr(cell, 'gain_L23')

    # every remaining variable sits on the output path -> gradients exist
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean((model(x) - y) ** 2)
    grads = tape.gradient(loss, model.trainable_variables)
    assert all(g is not None for g in grads)
    assert np.isfinite(float(loss))
    assert np.isfinite(_train_step(model, x, y))

    # strictly fewer parameters than the full recurrent wiring
    full = make_cncp_model('lrc', output_neurons=2, **_ASYM)
    full(x)
    assert model.count_params() < full.count_params()


def test_sign_constraint_ablation_builds_and_trains():
    tf.random.set_seed(0)
    model = make_cncp_model('gru', output_neurons=2, sign_constraint=False)
    x = tf.random.normal((2, 10, 2))
    y = tf.random.normal((2, 10, 2))
    model(x)
    edge = model.layers[0].cell.edges['M_TRN_Thal']
    assert isinstance(edge, SparseLinear)
    assert not isinstance(edge, SignedSparseLinear)
    assert np.isfinite(_train_step(model, x, y))


def test_divisive_inhibition_builds_and_trains():
    tf.random.set_seed(0)
    model = make_cncp_model('gru', output_neurons=2, divisive_inhibition=True)
    x = tf.random.normal((2, 10, 2))
    y = tf.random.normal((2, 10, 2))
    model(x)
    cell = model.layers[0].cell
    assert 'M_L6CT_div_L4' in cell.edges

    with tf.GradientTape() as tape:
        loss = tf.reduce_mean((model(x) - y) ** 2)
    tvars = model.trainable_variables
    grads = tape.gradient(loss, tvars)
    idx = {id(v): i for i, v in enumerate(tvars)}
    g_div = grads[idx[id(cell.edges['M_L6CT_div_L4'].W)]]
    assert g_div is not None
    assert bool(tf.math.is_finite(tf.norm(g_div)))
    assert float(tf.norm(g_div)) > 0.0
    assert np.isfinite(_train_step(model, x, y))


# --- mask determinism ---

def _built_cell(seed):
    cell = CorticalColumnCell('gru', seed=seed)
    cell.build((None, 2))
    return cell


def test_masks_deterministic_in_seed():
    a, b, c = _built_cell(7), _built_cell(7), _built_cell(8)
    for name in a._masks:
        assert np.array_equal(a._masks[name], b._masks[name]), name
    sparse = [n for n, d in DEFAULT_MASK_DENSITIES.items() if d < 1.0]
    assert any(not np.array_equal(a._masks[n], c._masks[n]) for n in sparse)


def test_dense_edges_have_all_ones_masks():
    cell = _built_cell(42)
    for name, density in DEFAULT_MASK_DENSITIES.items():
        if density >= 1.0:
            assert cell._masks[name].all(), name
        else:
            # sparse masks are neither empty nor full (statistically certain
            # at the default sizes and densities)
            m = cell._masks[name]
            assert m.any(), name
            assert not m.all(), name
