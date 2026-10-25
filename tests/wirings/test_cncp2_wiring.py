# tests/wirings/test_cncp2_wiring.py
# Cell-level gates for the four-node cNCP v2 column (issue a2ngerer/LRC#9,
# spec docs/superpowers/specs/2026-10-21-cncp-v2-abstraction.md, B.3 to B.5):
# state layout, the B.4 parameter formula, the init rules R3/R4, the ablation
# flags, the two reductions and a single compiled step. The stability
# acceptance test (R6) lives in test_cncp2_stability.py.
import numpy as np
import pytest
import tensorflow as tf

from src.neurons import CfC_LRC_Cell
from src.wirings import (CNCP2Wiring, CorticalColumnCell, CorticalColumnV2Cell,
                         SparseLinear, effective_param_count, ncp_cell_kwargs)

_ASYM = dict(elastance_type='asymmetric')
F_IN = 2


def _cell(seed=0, units=6, **kw):
    """A built column; the global seed fixes the weight draw (build order)."""
    tf.keras.utils.set_random_seed(seed)
    cell = CorticalColumnV2Cell('cfc_lrc', units=units, seed=seed,
                                **_ASYM, **kw)
    cell.build((tf.TensorShape([None, F_IN]), tf.TensorShape([None, 1])))
    return cell


def _rollout(cell, steps=12, batch=4, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((steps, batch, F_IN)).astype('float32')
    dt = tf.fill([batch, 1], 0.1)
    states = cell.get_initial_state(batch_size=batch)
    outs = []
    for k in range(steps):
        out, states = cell((x[k], dt), states)
        outs.append(out)
    return tf.stack(outs).numpy(), states


# --- layout ---

def test_state_layout_and_output_size():
    cell = CorticalColumnV2Cell('cfc_lrc', units=7)
    assert cell.state_size == [7, 7, 7, 3]          # u_R = max(1, u // 2)
    assert cell.output_size == 7
    assert CorticalColumnV2Cell('gru', units=1).state_size == [1, 1, 1, 1]
    zeros = cell.get_initial_state(batch_size=2)
    assert [tuple(s.shape) for s in zeros] == [(2, 7), (2, 7), (2, 7), (2, 3)]


def test_rejects_bad_arguments():
    with pytest.raises(ValueError, match='relay_gate'):
        CorticalColumnV2Cell('gru', relay_gate='closed')
    lstm_column = CorticalColumnV2Cell('lstm', units=4)
    with pytest.raises(ValueError, match='single-state'):
        lstm_column.build((None, 2))


# --- B.4 parameter formula ---

@pytest.mark.parametrize('u', [4, 8, 16])
def test_parameter_formula_cfc_lrc(u):
    """effective_param_count of the column (no head) equals
    3 p(u,u) + 3u^2 + 3u floor(u/2) + F_in u + u + 3 floor(u/2),
    p(u,u) = 10u^2 + 5u, up to the realised Bernoulli(0.5) masks of E_DS and
    A_DS, which enter with their actual number of ones."""
    model = CNCP2Wiring('cfc_lrc', units=u, seed=0, **_ASYM).build_model()
    model((tf.zeros([1, 3, F_IN]), tf.ones([1, 3, 1])))
    cell = model.layers[0].cell
    n = effective_param_count(model)

    sub = CfC_LRC_Cell(units=u, **ncp_cell_kwargs(CfC_LRC_Cell, _ASYM))
    sub.build((None, u))
    p = 10 * u * u + 5 * u
    assert sub.count_params() == p

    ur = u // 2
    formula = 3 * p + 3 * u * u + 3 * u * ur + F_IN * u + u + 3 * ur
    m_ds, m_a = (int(cell._masks[k].sum()) for k in ('E_DS', 'A_DS'))
    realised = formula - u * u + m_ds + m_a          # 0.5u^2 each -> realised
    assert n == realised
    assert abs(n - formula) <= abs(m_ds - u * u / 2) + abs(m_a - u * u / 2)


# --- init rules ---

def test_init_rules_r3_r4():
    cell = _cell(units=6)
    assert np.all(cell.gamma.numpy() == 0.0)                         # R3
    assert np.allclose(cell.b_gate.numpy(), -2.0)                    # R4
    alpha = 1.0 - np.exp(-1.0 * tf.nn.softplus(cell.rho).numpy())
    assert np.allclose(alpha, 0.5, atol=1e-6)


def test_relay_state_bounded_by_construction():
    cell = _cell(units=6)
    x = 100.0 * tf.random.normal([4, F_IN], seed=1)
    states = cell.get_initial_state(batch_size=4)
    for _ in range(20):
        _, states = cell((x, tf.fill([4, 1], 1.0)), states)
    assert float(tf.reduce_max(tf.abs(states[3]))) < 1.0


# --- ablation flags ---

def _edges_and_vars(cell):
    return set(cell.edges), {v.name.split('/')[-1].split(':')[0]
                             for v in cell.trainable_weights}


def test_flags_build_only_the_live_parts():
    edges, names = _edges_and_vars(_cell())
    assert edges == {'E_in', 'E_RG', 'E_GS', 'E_DS', 'A_DS', 'E_SD', 'E_DR',
                     'E_DRg'}
    assert {'gamma', 'rho', 'b_r', 'b_gate'} <= names

    assert 'E_DS' not in _cell(feedback=False).edges
    edges, names = _edges_and_vars(_cell(gain=False))
    assert 'A_DS' not in edges and 'gamma' not in names
    edges, names = _edges_and_vars(_cell(relay_gate='off'))
    assert 'E_DRg' not in edges and 'b_gate' not in names
    edges, names = _edges_and_vars(_cell(relay_gate='free'))
    assert 'E_DRg' in edges
    assert np.allclose(_cell(relay_gate='free').b_gate.numpy(),
                       1.0 - 1.0 / (1.0 + np.exp(2.0)))

    ff = _cell(feedforward_only=True)
    assert set(ff.edges) == {'E_in', 'E_GS', 'E_SD'}
    _, states = _rollout(ff)
    assert float(tf.reduce_max(tf.abs(states[3]))) == 0.0          # inert R


def test_masks_shared_across_ablations():
    full, nofb = _cell(seed=3), _cell(seed=3, feedback=False)
    for k in full._masks:
        assert np.array_equal(full._masks[k], nofb._masks[k])


def test_fan_in_aware_init_r2():
    """G and S sum two edges in the full column, so their glorot draws are
    scaled by 1/sqrt(2); in the feedforward reduction they are not."""
    full, ff = _cell(seed=0), _cell(seed=0, feedforward_only=True)
    lim = lambda e: np.sqrt(6.0 / sum(e.W.shape))
    for name in ('E_in', 'E_RG', 'E_GS', 'E_DS'):
        e = full.edges[name]
        assert np.max(np.abs(e.W.numpy())) <= lim(e) / np.sqrt(2) + 1e-6
    e = ff.edges['E_in']
    assert np.max(np.abs(e.W.numpy())) > lim(e) / np.sqrt(2)


# --- reductions ---

def test_gamma_zero_equals_gain_off():
    """At gamma = 0 the gain factor is exactly 1, so the column equals the
    gain=False column built from the same seed (A_DS and gamma are built
    last, so every common weight is drawn identically)."""
    on, off = _cell(seed=5), _cell(seed=5, gain=False)
    for k in off.edges:
        assert np.array_equal(on.edges[k].W.numpy(), off.edges[k].W.numpy())
    for k in ('G', 'S', 'D'):
        for a, b in zip(on.subcells[k].get_weights(),
                        off.subcells[k].get_weights()):
            assert np.array_equal(a, b), k
    out_on, _ = _rollout(on)
    out_off, _ = _rollout(off)
    assert np.array_equal(out_on, out_off)
    on.gamma.assign(tf.fill(on.gamma.shape, 0.5))   # the gain does act
    assert not np.allclose(_rollout(on)[0], out_off)


def test_feedforward_reduction_equals_masked_three_layer_stack():
    """feedforward_only (gain off, gate off) is the chain
    x -E_in-> G -E_GS-> S -E_SD-> D of three ncp_cell_kwargs sub-cells with
    masked SparseLinear edges. The reference is built from the same seed in
    the same order, so the weights coincide without copying."""
    u, seed = 6, 2
    col = _cell(seed=seed, units=u, feedforward_only=True, gain=False,
                relay_gate='off')

    tf.keras.utils.set_random_seed(seed)
    edges = []
    for name, d_in in (('E_in', F_IN), ('E_GS', u), ('E_SD', u)):
        e = SparseLinear(u, col._masks[name])
        e.build((None, d_in))
        edges.append(e)
    kw = ncp_cell_kwargs(CfC_LRC_Cell, _ASYM)
    subs = []
    for _ in range(3):
        c = CfC_LRC_Cell(units=u, **kw)
        c.build((None, u))
        subs.append(c)
    for a, b in zip(col.edges.values(), edges):
        assert np.array_equal(a.W.numpy(), b.W.numpy())

    rng = np.random.default_rng(0)
    x = rng.standard_normal((12, 4, F_IN)).astype('float32')
    dt = tf.fill([4, 1], 0.1)
    h = [tf.zeros([4, u])] * 3
    ref = []
    for k in range(12):
        v = x[k]
        for i in range(3):
            h[i], _ = subs[i]((edges[i](v), dt), [h[i]])
            v = h[i]
        ref.append(v)
    out, _ = _rollout(col)
    assert np.array_equal(out, tf.stack(ref).numpy())


# --- gradient at gamma = 0 (spec B.3, notes) ---

def test_gamma_gets_gradient_while_apical_edge_starts_dead():
    """At gamma = 0, A_DS receives exactly zero gradient but gamma does not,
    so the gain edge switches on after the first update (FiLM-style)."""
    cell = _cell(units=6)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, 4, F_IN)).astype('float32')
    with tf.GradientTape() as tape:
        states = cell.get_initial_state(batch_size=4)
        for k in range(8):
            out, states = cell((x[k], tf.fill([4, 1], 0.1)), states)
        loss = tf.reduce_sum(out ** 2)
    g_gamma, g_ap = tape.gradient(loss, [cell.gamma, cell.edges['A_DS'].W])
    assert float(tf.reduce_sum(tf.abs(g_gamma))) > 0.0
    assert float(tf.reduce_sum(tf.abs(g_ap))) == 0.0


# --- one compiled step ---

def _step_graph_ops(cell, batch=4, input_dim=F_IN):
    states = cell.get_initial_state(batch_size=batch)
    x = tf.zeros([batch, input_dim])
    dt = tf.fill([batch, 1], 0.1)
    step = tf.function(lambda x, dt, s: cell((x, dt), s))
    out, _ = step(x, dt, states)
    out2, _ = step(x + 1.0, dt, states)            # same signature: no retrace
    assert step.experimental_get_tracing_count() == 1
    graph = step.get_concrete_function(x, dt, states).graph
    skip = {'Placeholder', 'ReadVariableOp', 'Const', 'Identity', 'NoOp'}
    return out, sum(op.type not in skip for op in graph.get_operations())


def test_single_step_under_tf_function_on_available_device():
    device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
    with tf.device(device):
        v2 = _cell(units=8)
        out, n_v2 = _step_graph_ops(v2)
        assert out.shape == (4, 8)
        assert bool(tf.reduce_all(tf.math.is_finite(out)))
        v1 = CorticalColumnCell('cfc_lrc', **_ASYM)
        v1.build((tf.TensorShape([None, F_IN]), tf.TensorShape([None, 1])))
        _, n_v1 = _step_graph_ops(v1)
    # Four nodes / eight edges against eight nodes / fifteen edges: the v2
    # step has clearly fewer compute ops (B.4 estimates ~70 vs ~135).
    assert n_v2 < 0.75 * n_v1, (n_v2, n_v1)
