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


def test_dt_reaches_relays_but_not_gru_nodes():
    """Discrete sub-cells (gru) discard dt, the relay nodes integrate it: at
    the first step the six gru node states and the L5ET output are dt-free
    (Thal->L4 is delayed), while the Thal/TRN states depend on dt."""
    tf.random.set_seed(0)
    cell = CorticalColumnCell('gru')
    x = tf.random.normal((2, 3))
    states = cell.get_initial_state(batch_size=2)
    out1, s1 = cell((x, 1.0), states)
    out2, s2 = cell((x, 0.1), states)
    assert bool(tf.reduce_all(tf.abs(out1 - out2) < 1e-7))
    for node, a, b in zip(NODE_ORDER, s1, s2):
        same = bool(tf.reduce_all(tf.abs(a - b) < 1e-7))
        assert same == (node not in ('Thal', 'TRN')), node


def test_relay_alpha_is_half_at_init_for_unit_dt():
    """alpha = 1 - exp(-dt * softplus(rate_raw)); rate_raw starts at 0, so
    alpha(dt = 1) = 1 - exp(-ln 2) = 0.5, the pre-revision leak value."""
    cell = CorticalColumnCell('gru')
    cell.build((None, 3))
    for raw in (cell.thal_rate_raw, cell.trn_rate_raw):
        alpha = 1.0 - tf.exp(-1.0 * tf.nn.softplus(raw))
        assert np.allclose(alpha.numpy(), 0.5, atol=1e-6)


def test_relays_carry_state_unchanged_for_vanishing_dt():
    """dt -> 0 leaves the relay states at their previous value (alpha -> 0);
    dt = 1 moves them (the TRN drive is non-zero after one gru step)."""
    tf.random.set_seed(0)
    cell = CorticalColumnCell('gru')
    x = tf.random.normal((2, 3))
    states = cell.get_initial_state(batch_size=2)
    thal, trn = NODE_ORDER.index('Thal'), NODE_ORDER.index('TRN')
    _, s_tiny = cell((x, 1e-6), states)
    _, s_unit = cell((x, 1.0), states)
    assert float(tf.reduce_max(tf.abs(s_tiny[thal]))) < 1e-4
    assert float(tf.reduce_max(tf.abs(s_tiny[trn]))) < 1e-4
    assert float(tf.reduce_max(tf.abs(s_unit[trn]))) > 1e-2


def test_timescale_prior_reaches_the_relays():
    """A per-lamina factor on Thal scales the relay's dt (previously the
    relays ignored elapsed time, so the prior had no effect on them)."""
    def thal_state(prior):
        tf.keras.utils.set_random_seed(0)
        cell = CorticalColumnCell('gru', timescale_prior=prior)
        x = tf.ones((2, 3))
        _, s = cell((x, 1.0), cell.get_initial_state(batch_size=2))
        return s[NODE_ORDER.index('Thal')].numpy(), s[NODE_ORDER.index('TRN')].numpy()
    thal_a, trn_a = thal_state(None)
    thal_b, trn_b = thal_state({'Thal': 4.0})
    assert not np.allclose(thal_a, thal_b)
    assert np.allclose(trn_a, trn_b)          # TRN factor defaults to 1.0


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

@pytest.mark.parametrize('route', ['direct', 'thalamic'])
def test_gradient_flow_every_variable(route):
    """Non-None, finite, non-zero gradient for every trainable variable --
    explicitly including the sign-locked (M_TRN_Thal) and multiplicative
    (gains + apical masks) edges -- for both sensory routes."""
    tf.random.set_seed(0)
    model = make_cncp_model('lrc', output_neurons=2, sensory_route=route,
                            **_ASYM)
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


# --- revision 2026-10-04: sub-cell defaults shared with the NCP layers ---

@pytest.mark.parametrize('neuron_type,kwargs', [('cfc_lrc', _ASYM), ('cfc', {})])
def test_cfc_subcells_have_no_backbone_by_default(neuron_type, kwargs):
    """CfC-family sub-cells follow the NCP rule (ncp_cell_kwargs): no backbone,
    the heads act on [x, h] directly. An explicit backbone_layers wins."""
    cell = CorticalColumnCell(neuron_type, **kwargs)
    cell.build((None, 3))
    assert all(sub._backbone == [] for sub in cell.subcells.values())
    explicit = CorticalColumnCell(neuron_type, backbone_layers=1, **kwargs)
    explicit.build((None, 3))
    assert all(len(sub._backbone) == 1 for sub in explicit.subcells.values())


def test_non_cfc_subcells_unaffected_by_the_backbone_rule():
    cell = CorticalColumnCell('gru')
    cell.build((None, 3))
    assert set(cell.subcells) == {'L4', 'L23', 'L5IT', 'L5ET', 'L6CC', 'L6CT'}


# --- revision 2026-10-04: the TRN -> Thal edge inhibits ---

def test_trn_activity_nonnegative_and_contribution_to_thal_nonpositive():
    """TRN is a sigmoid relay (state >= 0); through the sign-locked edge its
    contribution to the Thal drive is <= 0 for every input, also after a few
    training steps on random data (the previous tanh relay could flip the
    sign of the 'inhibitory' edge)."""
    tf.random.set_seed(0)
    model = make_cncp_model('gru', output_neurons=2)
    x = tf.random.normal((4, 12, 3))
    y = tf.random.normal((4, 12, 2))
    model(x)
    for _ in range(3):
        assert np.isfinite(_train_step(model, x, y, lr=1e-2))
    cell = model.layers[0].cell
    states = cell.get_initial_state(batch_size=4)
    trn_idx = NODE_ORDER.index('TRN')
    for t in range(12):
        _, states = cell((x[:, t], 1.0), states)
        h_trn = states[trn_idx]
        assert bool(tf.reduce_all(h_trn >= 0.0))
        assert bool(tf.reduce_all(cell.edges['M_TRN_Thal'](h_trn) <= 0.0))


# --- revision 2026-10-04: tbt reuses the parent step through hooks ---

def _build_pair(seed, **kwargs):
    from src.wirings.tbt_cncp import TbtCorticalColumnCell
    tf.keras.utils.set_random_seed(seed)
    base = CorticalColumnCell('gru', seed=1, **kwargs)
    base.build((tf.TensorShape([None, 3]), tf.TensorShape([None, 1])))
    tf.keras.utils.set_random_seed(seed)
    tbt = TbtCorticalColumnCell('gru', seed=1, use_location='none', **kwargs)
    tbt.build((tf.TensorShape([None, 3]), tf.TensorShape([None, 1]),
               tf.TensorShape([None, 2])))
    return base, tbt


def test_tbt_without_location_matches_cncp_step_for_step():
    """With the location ignored, the tbt column must compute exactly the cNCP
    step (same weights by seed): the L5ET half of its output and every node
    state equal the parent's over several steps."""
    base, tbt = _build_pair(0)
    x = tf.random.normal((2, 5, 3))
    loc = tf.random.normal((2, 2))
    s_b = base.get_initial_state(batch_size=2)
    s_t = tbt.get_initial_state(batch_size=2)
    n_l23 = base._units['L23']
    for t in range(5):
        out_b, s_b = base((x[:, t], 0.7), s_b)
        out_t, s_t = tbt((x[:, t], 0.7, loc), s_t)
        assert np.allclose(out_t[:, n_l23:].numpy(), out_b.numpy(), atol=1e-6)
        assert np.allclose(out_t[:, :n_l23].numpy(),
                           s_b[NODE_ORDER.index('L23')].numpy(), atol=1e-6)
        for a, b in zip(s_t, s_b):
            assert np.allclose(a.numpy(), b.numpy(), atol=1e-6)


def test_tbt_honours_timescale_prior():
    """The copied call() used to drop the per-lamina factor; via the shared
    step the prior now reaches the tbt column too."""
    _, plain = _build_pair(0)
    _, prior = _build_pair(0, timescale_prior={'Thal': 4.0})
    x = tf.ones((2, 3))
    loc = tf.zeros((2, 2))
    _, s_plain = plain((x, 1.0, loc), plain.get_initial_state(batch_size=2))
    _, s_prior = prior((x, 1.0, loc), prior.get_initial_state(batch_size=2))
    thal = NODE_ORDER.index('Thal')
    assert not np.allclose(s_plain[thal].numpy(), s_prior[thal].numpy())


# --- revision 2026-10-04: optional sensory route through the relay ---

def test_thalamic_route_replaces_the_direct_input_edge():
    tf.random.set_seed(0)
    model = make_cncp_model('gru', output_neurons=2, sensory_route='thalamic')
    x = tf.random.normal((2, 6, 3))
    y = tf.random.normal((2, 6, 2))
    assert tuple(model(x).shape) == (2, 6, 2)
    cell = model.layers[0].cell
    assert 'M_in_Thal' in cell.edges
    assert 'M_in_L4' not in cell.edges
    assert np.isfinite(_train_step(model, x, y))


def test_thalamic_route_input_reaches_the_output_within_one_step():
    """Thal is evaluated before L4 in 'thalamic' mode, so the input of step t
    reaches L5ET at step t (no extra delay from the relay)."""
    tf.random.set_seed(0)
    cell = CorticalColumnCell('gru', sensory_route='thalamic')
    states = cell.get_initial_state(batch_size=2)
    x1 = tf.random.normal((2, 3))
    x2 = tf.random.normal((2, 3))
    out1, _ = cell((x1, 1.0), states)
    out2, _ = cell((x2, 1.0), states)
    assert not np.allclose(out1.numpy(), out2.numpy())


def test_thalamic_route_keeps_the_default_masks():
    """M_in_Thal is dense and appended last, so a given seed yields the same
    sparse masks with and without the thalamic route."""
    a = CorticalColumnCell('gru', seed=7)
    b = CorticalColumnCell('gru', seed=7, sensory_route='thalamic')
    a.build((None, 2))
    b.build((None, 2))
    for name in a._masks:
        assert np.array_equal(a._masks[name], b._masks[name]), name


def test_thalamic_route_plus_feedforward_raises():
    with pytest.raises(ValueError):
        CorticalColumnCell('gru', feedforward_only=True,
                           sensory_route='thalamic')


def test_invalid_sensory_route_raises():
    with pytest.raises(ValueError):
        CorticalColumnCell('gru', sensory_route='cortical')
