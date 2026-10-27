# tests/experiments/test_rq4_diagnostics.py
# RQ4 diagnostics (issue a2ngerer/LRC#30): the runners' --diagnostics callback
# writes its keys without changing the training run, the Jacobian products
# match a direct unroll, and experiments/rq4_diagnostics.py runs on 2-epoch
# local LV fits. Tiny synthetic sizes, CPU only.
import json
import sys

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

import src.tasks.person_activity as pa_pkg
from experiments import rq4_diagnostics as rq4
from experiments import run_lotka_volterra_benchmark as lv_runner
from experiments import run_person_activity_benchmark as pa_runner
from src.tasks.lotka_volterra.model import build_lotka_volterra_model
from tests.benchmark.test_checkpoints import _pa_stub


def _lv(outdir, monkeypatch, cell, wiring, size, *extra):
    monkeypatch.setattr(sys, 'argv', [
        'run_lotka_volterra_benchmark.py', '--wiring', wiring, '--cell', cell,
        '--size', str(size), '--epochs', '2', '--n-trajectories', '10',
        '--anchors', '4,8', '--horizon', '12',
        '--seq-len', '20', '--batch-size', '4', '--outdir', str(outdir),
        *extra])
    lv_runner.main()
    return json.loads((outdir / f'{cell}_{wiring}_seed0.json').read_text())


def test_lv_diagnostics_keys_and_training_unchanged(tmp_path, monkeypatch):
    off = _lv(tmp_path / 'off', monkeypatch, 'ctrnn', 'dense', 4)
    on = _lv(tmp_path / 'on', monkeypatch, 'ctrnn', 'dense', 4,
             '--diagnostics')
    assert 'diagnostics' not in off
    assert on['val_loss_curve'] == off['val_loss_curve']
    assert on['closed_loop_mse'] == off['closed_loop_mse']
    diag = on['diagnostics']
    assert diag['epochs'] == [0, 1, 2]
    assert set(diag['per_group']) == {'timescale', 'recurrent', 'input',
                                      'readout'}
    for g in diag['per_group'].values():
        assert g['slot_norm'][0] == 0.0 and g['eff_lr_mean'][0] is None
        assert np.all(np.isfinite(g['grad_norm'])) and g['grad_norm'][0] > 0
        assert g['eff_lr_mean'][-1] > 0
    (layer,) = diag['state_norm'].values()
    assert len(layer['mean']) == 3 and layer['max'][0] >= layer['mean'][0] > 0
    # tau starts at 1 and the runner passes --ode-unfolds to ctrnn (#56):
    # rate = dt / ode_unfolds at step 0
    assert on['ode_unfolds'] == 6
    assert diag['ctrnn_rate']['max_rate'][0] == pytest.approx(
        on['dt'] / on['ode_unfolds'])
    assert diag['ctrnn_rate']['n_above_1'] == [0, 0, 0]


def test_pa_diagnostics_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(pa_pkg, 'load_person_activity', _pa_stub)
    monkeypatch.setattr(sys, 'argv', [
        'run_person_activity_benchmark.py', '--wiring', 'ncp', '--cell',
        'cfc_lrc', '--size', '8', '--epochs', '2', '--diagnostics',
        '--outdir', str(tmp_path)])
    pa_runner.main()
    diag = json.loads((tmp_path / 'cfc_lrc_ncp_seed0.json').read_text()
                      )['diagnostics']
    assert diag['epochs'] == [0, 1, 2] and 'ctrnn_rate' not in diag
    assert {'timescale', 'recurrent', 'readout'} <= set(diag['per_group'])


@pytest.mark.parametrize('closed', (False, True))
def test_jacobian_products_match_direct_unroll(closed):
    one = rq4.OneStep(build_lotka_volterra_model('dense3', 'gru', size=3))
    rng = np.random.default_rng(0)
    xs = rng.normal(size=(2, 9, 2)).astype(np.float32)
    dts = np.full((2, 9, 1), 0.1, np.float32)
    _, states, jacs = one.trajectory(xs, dts, closed)
    prod = rq4.jacobian_products(jacs)[4][2]            # dS_6 / dS_2
    s = tf.constant(states[:, 1])                       # states[:, t] = S_{t+1}
    with tf.GradientTape() as tape:
        tape.watch(s)
        h, x = s, tf.constant(xs[:, 2])
        for t in range(2, 6):
            y, h = one.step(x, tf.constant(dts[:, t]), h)
            nxt = tf.constant(xs[:, t + 1])
            # closed loop: the input value stays xs, its derivative is dy
            x = y + tf.stop_gradient(nxt - y) if closed else nxt
    direct = tape.batch_jacobian(h, s).numpy()
    np.testing.assert_allclose(prod[:, :one.dim, :one.dim], direct,
                               rtol=1e-4, atol=1e-6)


def test_posthoc_script_on_2_epoch_fits(tmp_path, monkeypatch):
    _lv(tmp_path, monkeypatch, 'gru', 'dense3', 4)
    _lv(tmp_path, monkeypatch, 'cfc_lrc', 'ncp', 8)
    out = tmp_path / 'rq4.csv'
    rq4.main(['--results', str(tmp_path / '*.json'), '--out', str(out),
              '--hessian-top'])
    df = pd.read_csv(out).set_index('wiring')
    assert sorted(df.index) == ['dense3', 'ncp']
    for p in ('tf', 'cl'):
        assert np.isfinite(df[f'{p}_hnorm_max']).all()
        for d in rq4.DELTAS:
            assert np.isfinite(df[f'{p}_jac{d}_median']).all()
    assert np.isfinite(df.loc['dense3', 'cl_jac16_layer2_median'])
    assert np.isfinite(df.loc['ncp', 'tf_jac4_command_median'])
    assert np.isfinite(df['hessian_top']).all()
