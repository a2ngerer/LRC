# tests/wirings/test_cncp2_multi.py
# K independent cNCP v2 columns (issue a2ngerer/LRC#10, spec
# docs/superpowers/specs/2026-10-21-cncp-v2-abstraction.md B.7 / V4): shapes
# over K, partition, coupling and readout; weight independence; the P_K
# parameter formula against effective_param_count; budget sizing.
import numpy as np
import pytest
import tensorflow as tf

from src.neurons import CfC_LRC_Cell
from src.wirings import (COUPLINGS, READOUTS, ColumnReadout, MultiColumnV2Cell,
                         effective_param_count, ncp_cell_kwargs,
                         size_for_budget)
from src.tasks.lotka_volterra import build_lotka_volterra_model
from src.tasks.person_activity import build_person_activity_model
from src.tasks.committee.model import build_committee_model

_ASYM = dict(elastance_type='asymmetric')
F_IN = 8          # divisible by 1, 2 and 4 for the region fallback split


def _model(K, partition='same', coupling='none', readout='concat', u=4,
           F=F_IN, n_out=3, seed=0, **kw):
    tf.keras.utils.set_random_seed(seed)
    x = tf.keras.Input((None, F))
    t = tf.keras.Input((None, 1))
    cell = MultiColumnV2Cell(K, 'cfc_lrc', units=u, seed=seed,
                             partition=partition, coupling=coupling,
                             **_ASYM, **kw)
    h = tf.keras.layers.RNN(cell, return_sequences=True)((x, t))
    out = ColumnReadout(n_out, K, readout)(h)
    return tf.keras.Model([x, t], [h, out]), cell


def _inputs(F=F_IN, batch=3, steps=5, seed=0):
    rng = np.random.default_rng(seed)
    return (tf.constant(rng.standard_normal((batch, steps, F)), tf.float32),
            tf.fill([batch, steps, 1], 0.5))


@pytest.mark.filterwarnings('ignore:partition=')
@pytest.mark.parametrize('readout', READOUTS)
@pytest.mark.parametrize('coupling', COUPLINGS)
@pytest.mark.parametrize('partition', ['same', 'region'])
@pytest.mark.parametrize('K', [1, 2, 4])
def test_shapes_and_finite(K, partition, coupling, readout):
    model, cell = _model(K, partition, coupling, readout)
    h, out = model(_inputs())
    assert h.shape == (3, 5, K * 4)
    assert out.shape == (3, 5, 3)
    assert bool(tf.reduce_all(tf.math.is_finite(out)))
    assert cell.state_size == [4, 4, 4, 2] * K
    want = [F_IN] * K if partition == 'same' else [F_IN // K] * K
    assert list(cell.column_input_dims) == want
    assert (cell.lateral is not None) == (coupling == 'lateral' and K > 1)


def test_region_with_sensor_tags_routes_flag_and_masked_values():
    """Person Activity layout: x = [one-hot(tag, 4), xyz]; column k reads
    [m_k, m_k * xyz] and therefore sees zeros at rows of other tags."""
    cell = MultiColumnV2Cell(4, 'gru', units=3, partition='region',
                             region_tags=4)
    cell.build((tf.TensorShape([None, 7]), tf.TensorShape([None, 1])))
    assert list(cell.column_input_dims) == [4] * 4
    x = tf.constant([[0, 0, 1, 0, 2.0, -1.0, 3.0]])
    parts = [p.numpy()[0] for p in cell._split(x)]
    np.testing.assert_allclose(parts[2], [1, 2, -1, 3])
    for k in (0, 1, 3):
        np.testing.assert_allclose(parts[k], 0.0)
    with pytest.raises(ValueError, match='sensor tags'):
        bad = MultiColumnV2Cell(2, 'gru', units=3, partition='region',
                                region_tags=4)
        bad.build((tf.TensorShape([None, 7]), tf.TensorShape([None, 1])))


def test_region_without_structure_warns():
    with pytest.warns(UserWarning, match='no sensor structure'):
        _model(2, partition='region', F=7)


def test_rejects_bad_options():
    for kw in (dict(partition='hub'), dict(coupling='hub'), dict(columns=0)):
        with pytest.raises(ValueError):
            MultiColumnV2Cell(**{'columns': 2, **kw})
    with pytest.raises(ValueError, match='readout'):
        ColumnReadout(2, 2, 'vote')


@pytest.mark.parametrize('coupling', COUPLINGS)
def test_columns_have_independent_weights(coupling):
    """Perturbing column 0 moves column 1's output only through the coupling."""
    model, cell = _model(2, coupling=coupling)
    x = _inputs()
    h0 = model(x)[0].numpy()
    assert not np.allclose(cell.cols[0].edges['E_in'].W.numpy(),
                           cell.cols[1].edges['E_in'].W.numpy())
    assert not np.array_equal(cell.cols[0]._masks['E_DS'],
                              cell.cols[1]._masks['E_DS'])
    w = cell.cols[0].edges['E_in'].W
    w.assign(w + 0.5)
    h1 = model(x)[0].numpy()
    u = cell.units
    assert not np.allclose(h0[..., :u], h1[..., :u])            # column 0
    if coupling == 'none':
        np.testing.assert_array_equal(h0[..., u:], h1[..., u:])
    else:
        assert not np.allclose(h0[..., u:], h1[..., u:])


def test_median_readout_is_lower_median_of_per_column_heads():
    model, _ = _model(4, readout='median')
    h, out = model(_inputs())
    ro = model.layers[-1]
    hk = h.numpy().reshape(3, 5, 4, -1)
    z = np.einsum('btku,kuo->btko', hk, ro.kernel.numpy()) + ro.bias.numpy()
    np.testing.assert_allclose(out.numpy(), np.sort(z, axis=2)[:, :, 1],
                               rtol=1e-5, atol=1e-6)


def _p1_realised(col, u, f_in):
    """Exact B.4 column count with the realised E_DS / A_DS masks."""
    p = 10 * u * u + 5 * u
    ur = u // 2
    formula = 3 * p + 3 * u * u + 3 * u * ur + f_in * u + u + 3 * ur
    return formula - u * u + int(col._masks['E_DS'].sum()) + int(
        col._masks['A_DS'].sum())


@pytest.mark.parametrize('K,partition,coupling,readout,f_in', [
    (2, 'same', 'lateral', 'mean', 8),
    (4, 'region', 'lateral', 'concat', 2),
    (3, 'same', 'none', 'median', 8),
])
def test_effective_params_match_P_K(K, partition, coupling, readout, f_in):
    """effective_param_count == sum over columns + lateral mask + head, which
    is P_K (B.7) evaluated with the realised masks and within a few percent of
    its expected value."""
    u, n_out = 6, 3
    model, cell = _model(K, partition, coupling, readout, u=u, n_out=n_out)
    model(_inputs())
    n = effective_param_count(model)

    sub = CfC_LRC_Cell(units=u, **ncp_cell_kwargs(CfC_LRC_Cell, _ASYM))
    sub.build((None, u))
    assert sub.count_params() == 10 * u * u + 5 * u

    cols = [effective_param_count(c) for c in cell.cols]
    assert cols == [_p1_realised(c, u, f_in) for c in cell.cols]
    lat = 0 if cell.lateral is None else int(cell.lateral._mask_np.sum())
    head = {'concat': (K * u + 1) * n_out, 'mean': (u + 1) * n_out,
            'median': K * (u + 1) * n_out}[readout]
    assert n == sum(cols) + lat + head

    ur = u // 2
    p1 = 3 * (10 * u * u + 5 * u) + 3 * u * u + 3 * u * ur + f_in * u + u + 3 * ur
    c = K * (K - 1) * 0.25 * u * u if coupling == 'lateral' else 0
    expected = K * p1 + c + head
    assert abs(n - expected) / expected < 0.05, (n, expected)


def test_size_for_budget_4k_K2_and_30k_K4():
    def lv(size):
        return build_lotka_volterra_model(
            'cncp2_multi', 'cfc_lrc', size=size, seed=0, columns=2,
            partition='same', coupling='lateral', readout='mean', **_ASYM)
    u, n = size_for_budget(lv, 4000, range(2, 33))
    assert 4 <= u <= 10 and abs(n - 4000) / 4000 < 0.25, (u, n)

    def pa(size):
        return build_person_activity_model(
            'cncp2_multi', 'cfc_lrc', size=size, seed=0, columns=4,
            partition='region', coupling='lateral', readout='concat', **_ASYM)
    u, n = size_for_budget(pa, 30000, range(2, 33))
    assert 10 <= u <= 18 and abs(n - 30000) / 30000 < 0.1, (u, n)


@pytest.mark.parametrize('K', [1, 2])
@pytest.mark.parametrize('consensus', ['mean', 'median'])
def test_committee_builder_independent_columns(K, consensus):
    model = build_committee_model('cncp2_multi', K, size=4, seed=0,
                                  feature_size=7, seq_len=6, num_classes=5,
                                  consensus=consensus, **_ASYM)
    x = tf.random.normal([2, 6, K, 7])
    out = model((x, tf.ones([2, 6, 1])))
    assert out.shape == (2, 6, 5)
    cell = next(l.cell for l in model.layers
                if isinstance(l, tf.keras.layers.RNN))
    assert isinstance(cell, MultiColumnV2Cell) and cell.lateral is None
    assert list(cell.column_input_dims) == [7] * K


def test_single_column_path_unchanged_by_extra_s_edges():
    """extra_s_edges=0 and the two-element input are the #9 column: same
    variables, same fan-in, same outputs as the plain constructor."""
    from src.wirings import CorticalColumnV2Cell
    x, t = _inputs(F=F_IN, batch=2, steps=4)
    outs = []
    for kw in ({}, dict(extra_s_edges=0)):
        tf.keras.utils.set_random_seed(3)
        col = CorticalColumnV2Cell('cfc_lrc', units=4, seed=3, **_ASYM, **kw)
        h = tf.keras.layers.RNN(col, return_sequences=True)((x, t))
        outs.append((h.numpy(), [v.name for v in col.trainable_variables]))
    np.testing.assert_array_equal(outs[0][0], outs[1][0])
    assert outs[0][1] == outs[1][1]
