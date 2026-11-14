# tests/experiments/test_eval_cncp_gains.py
# Issue a2ngerer/LRC#136: per-lamina blocks, hop gains and path lengths
# (experiments/eval_cncp_gains.py). The laminar blocks tile the cNCP state
# and drop the inert cncp_ff slots; the Jacobian block of a linear same-step
# chain is gain x source retention; path lengths match a hand count; the
# script runs on 2-epoch LV fits without touching the sources.
import hashlib

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from experiments import eval_cncp_gains as eg
from experiments.rq4_diagnostics import OneStep, active_state
from src.tasks.lotka_volterra.model import build_lotka_volterra_model
from src.wirings.cncp import NODE_ORDER
from tests.experiments.test_rq4_diagnostics import _lv


def _cncp_model(wiring):
    return build_lotka_volterra_model(wiring, 'cfc_lrc', size=8, seed=0)


@pytest.mark.parametrize('wiring', ['cncp', 'cncp_random', 'cncp_ff'])
def test_laminar_blocks_tile_the_state(wiring):
    """One block per node, cut at the cumulative offsets of the cell's state
    sizes: the blocks tile S (cncp, cncp_random) or its active part
    (cncp_ff: the four inert slots L6CC, L6CT, Thal, TRN are left out). On
    the one-step Jacobian, L2/3_t reaches L4_{t+1} through no edge, L4_t
    reaches L2/3_{t+1} through the same-step edge."""
    model = _cncp_model(wiring)
    one = OneStep(model)
    record = {'wiring': wiring}
    blk = eg.laminar_blocks(record, one)
    cell = one.cell.core
    idx = np.concatenate(list(blk.values()))
    if wiring == 'cncp_ff':
        assert list(blk) == list(NODE_ORDER[:4])
        assert np.array_equal(idx, active_state(record, one))
        assert one.dim - len(idx) == sum(cell._units[n]
                                         for n in NODE_ORDER[4:])
    else:
        assert list(blk) == list(NODE_ORDER)
        assert np.array_equal(idx, np.arange(one.dim))
    assert [len(blk[n]) for n in blk] == [cell._units[n] for n in blk]
    rng = np.random.default_rng(0)
    _, _, jacs = one.trajectory(rng.normal(size=(2, 5, 2)).astype(np.float32),
                                np.full((2, 5, 1), 0.1, np.float32), 1.0)
    assert eg.block_norms(jacs, blk['L4'], blk['L23']).max() == 0
    assert eg.block_norms(jacs, blk['L23'], blk['L4']).min() > 0


def test_linear_same_step_chain_gives_gain_times_retention():
    """a_{t+1} = r a_t + x_t, b_{t+1} = g a_{t+1} (same step): definition (b)
    reads J[b, a] = g r (rows = target), J[a, b] = 0; definition (a) reads
    the std ratio g."""
    g, r = 1.7, 0.6
    rng = np.random.default_rng(0)
    z = tf.constant(rng.normal(size=(5, 2)), tf.float32)
    x = tf.constant(rng.normal(size=(5, 1)), tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(z)
        a = r * z[:, :1] + x
        z1 = tf.concat([a, g * a], axis=-1)
    jac = tape.batch_jacobian(z1, z)
    np.testing.assert_allclose(eg.block_norms(jac, [1], [0]), g * r,
                               rtol=1e-6)
    np.testing.assert_allclose(eg.block_norms(jac, [0], [0]), r, rtol=1e-6)
    assert eg.block_norms(jac, [0], [1]).max() == 0
    a_traj = rng.normal(size=(8, 3, 1))
    states = np.concatenate([a_traj, g * a_traj], axis=-1)
    assert eg.std_gain(states, [1], [0]) == pytest.approx(g)


def test_path_lengths_match_a_hand_count():
    """input -> readout over same-step edges (shortest, longest) and over all
    edges: cNCP in->L4->L23->L5ET (3), in->L4->L23->L5IT->L5ET (4), no
    delayed shortcut (3); NCP in->inter->command->motor and dense3 three
    layers (3, 3, 3); a delayed shortcut counts in sp_all only."""
    hand = [('in', 'a', 'S'), ('a', 'b', 'S'), ('in', 'b', 'D'),
            ('b', 'a', 'D')]
    assert eg.path_lengths(hand, 'in', 'b') == (2, 2, 1)
    assert eg.path_lengths(hand, 'in', 'c') == (None, None, None)
    for wiring in ('cncp', 'cncp_random', 'cncp_ff'):
        model = _cncp_model(wiring)
        edges, readout = eg.model_edges({'wiring': wiring}, model)
        assert readout == 'L5ET'
        assert eg.path_lengths(edges, 'in', 'L5ET') == (3, 4, 3), wiring
    ff, _ = eg.model_edges({'wiring': 'cncp_ff'}, _cncp_model('cncp_ff'))
    assert all(d == 'S' for *_, d in ff) and len(ff) == 5
    for cell in ('cfc_lrc', 'ltc'):          # layered cell / one ODE cell
        model = build_lotka_volterra_model('ncp', cell, size=8, seed=0)
        edges, readout = eg.model_edges({'wiring': 'ncp', 'size': 8}, model)
        assert {e for e in edges if e[2] == 'S'} == {
            ('in', 'inter', 'S'), ('inter', 'command', 'S'),
            ('command', 'motor', 'S')}, cell
        assert eg.path_lengths(edges, 'in', readout) == (3, 3, 3)
    assert eg.path_lengths(eg.dense3_edges(), 'in', 'layer2') == (3, 3, 3)


def test_script_on_two_epoch_lv_fits(tmp_path, monkeypatch):
    """Two rows per fit (trained, untrained) with the hop columns of the
    wiring and the end-to-end block, one path row per fit; the sources stay
    byte-identical and --out may not lie in a source folder."""
    d = tmp_path / 'lv'
    for wiring in ('cncp', 'cncp_ff', 'ncp', 'dense3'):
        _lv(d, monkeypatch, 'cfc_lrc', wiring, 4)
    sources = sorted(d.iterdir())
    sha = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    with pytest.raises(SystemExit, match='source folder'):
        eg.main(['--results', str(d / '*.json'), '--out', str(d / 'g.csv')])
    out, paths = tmp_path / 'gains.csv', tmp_path / 'paths.csv'
    eg.main(['--results', str(d / '*.json'), '--out', str(out),
             '--paths', str(paths)])
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.iterdir())} == sha
    df = pd.read_csv(out)
    assert len(df) == 8
    for wiring, fit in df.groupby('wiring'):
        assert list(fit['weights']) == ['trained', 'untrained']
        first, readout = eg.CHAIN[wiring]
        cols = [f'gain_{t}_from_{s}' for t, s in eg.HOPS[wiring]] + [
            f'jac1_{t}_from_{s}' for t, s in eg.HOPS[wiring]] + [
            f'jac1_{readout}_from_{first}', 'jac1_median', 'hnorm_mean']
        assert np.isfinite(fit[cols].to_numpy(float)).all(), wiring
        assert (fit[[c for c in cols if c.startswith('jac1_')]] > 0
                ).to_numpy().all(), wiring
    ff = df[df['wiring'] == 'cncp_ff']
    assert ff['hnorm_L6CC_mean'].isna().all()
    assert (ff['state_dim'] < df[df['wiring'] == 'cncp']['state_dim'].iloc[0]
            ).all()
    p = pd.read_csv(paths).set_index('wiring')
    assert p.loc['cncp', ['sp_same', 'lp_same', 'sp_all']].tolist() == [3, 4, 3]
    assert p.loc['ncp', ['sp_same', 'lp_same', 'sp_all']].tolist() == [3, 3, 3]
    assert p.loc['dense3', 'readout'] == 'layer2'
