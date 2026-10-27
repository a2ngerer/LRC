# tests/experiments/test_rq4_diagnostics.py
# RQ4 diagnostics (issue a2ngerer/LRC#30): the runners' --diagnostics callback
# writes its keys without changing the training run, the Jacobian products
# match a direct unroll, and experiments/rq4_diagnostics.py runs on 2-epoch
# local LV fits. Issue #65: the chain runs on the ForcedRolloutCell wrapper
# (alpha 0) through the dt_scale layer, its Jacobians match finite
# differences, every registry cell is covered and the result JSON gets
# rq4_diagnostics with the config.effective hash. Tiny synthetic sizes, CPU
# only.
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
from src.benchmark.checkpoints import config_hash, rebuild_from_result
from src.benchmark.registry import KNOWN_CELLS, cell_kwargs
from src.tasks.lotka_volterra.model import build_lotka_volterra_model
from src.tasks.person_activity.model import ode_unfolds_override
from tests.benchmark.test_checkpoints import _pa_stub


def _lv(outdir, monkeypatch, cell, wiring, size, *extra):
    monkeypatch.setattr(sys, 'argv', [
        'run_lotka_volterra_benchmark.py', '--wiring', wiring, '--cell', cell,
        '--size', str(size), '--epochs', '2', '--n-trajectories', '10',
        '--anchors', '4,8', '--horizon', '12',
        '--seq-len', '20', '--batch-size', '4', '--outdir', str(outdir),
        *extra])
    lv_runner.main()
    (path,) = outdir.glob(f'{cell}_{wiring}_seed0*.json')    # _mixed suffix
    return json.loads(path.read_text())


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


def _model(wiring, cell, size, dt_scale=5.12):
    """LV model as the runner builds it: registry kwargs, k = 6 substeps."""
    return build_lotka_volterra_model(
        wiring, cell, size=size, seed=0, dt_scale=dt_scale,
        **cell_kwargs(cell), **ode_unfolds_override(cell, 6))


def _inputs(T=9):
    rng = np.random.default_rng(0)
    return (rng.normal(size=(2, T, 2)).astype(np.float32),
            np.full((2, T, 1), 0.078, np.float32))


@pytest.mark.parametrize('alpha', (1.0, 0.0))
def test_jacobian_products_match_direct_unroll(alpha):
    one = rq4.OneStep(build_lotka_volterra_model('dense3', 'gru', size=3,
                                                 dt_scale=5.0))
    xs, dts = _inputs()
    _, _, jacs = one.trajectory(xs, dts, alpha)
    prod = rq4.jacobian_products(jacs)[4][2]            # dZ_6 / dZ_2
    z = tf.zeros([2, jacs.shape[-1]])
    for t in range(2):                                  # Z_2
        _, z = one.step(xs[:, t], dts[:, t], z, tf.fill([2, 1], float(t)))
    with tf.GradientTape() as tape:
        tape.watch(z)
        h = z
        for t in range(2, 6):
            _, h = one.step(xs[:, t], dts[:, t], h, tf.ones([2, 1]))
    np.testing.assert_allclose(prod, tape.batch_jacobian(h, z).numpy(),
                               rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize('cell', ('lrc', 'ltc', 'cfc_lrc', 'mlp'))
def test_jacobian_matches_finite_differences_through_dt_scale(cell):
    """Two-unit cell, dt_scale 5.12, alpha 0: the autodiff Jacobians over the
    wrapper state Z and the raw time input match central differences
    (Richardson-extrapolated, float32) within 1e-4."""
    one = rq4.OneStep(_model('dense', cell, size=2))
    one.alpha.assign(0.0)
    rng = np.random.default_rng(1)
    dim = one.dim + 2
    y = tf.constant(rng.normal(size=(3, 2)), tf.float32)
    dt = tf.fill([3, 1], 0.2)
    z = tf.constant(rng.normal(size=(3, dim)) * 0.5, tf.float32)
    started = tf.ones([3, 1])

    def fd(f, x, h=1e-2):
        def cd(h):
            e = h * np.eye(x.shape[-1], dtype=np.float32)
            return np.stack([(f(x + e[i]) - f(x - e[i])).numpy() / (2 * h)
                             for i in range(len(e))], -1)
        return (4 * cd(h / 2) - cd(h)) / 3

    _, _, jac = one._jac(y, dt, z, started)
    np.testing.assert_allclose(
        jac.numpy(), fd(lambda v: one.step(y, dt, v, started)[1], z),
        atol=1e-4)
    with tf.GradientTape() as tape:
        tape.watch(dt)
        z1 = one.step(y, dt, z, started)[1]
    np.testing.assert_allclose(
        tape.batch_jacobian(z1, dt).numpy(),
        fd(lambda v: one.step(y, v, z, started)[1], dt), atol=1e-4)


@pytest.mark.parametrize('wiring,cell', [
    *[('dense', c) for c in KNOWN_CELLS],
    *[(w, 'lrc') for w in ('dense3', 'ncp', 'cncp', 'cncp2')]])
def test_one_step_map_reproduces_every_registry_cell(wiring, cell):
    # lrc_ar feeds its input as the state: dense width = state width
    size = 2 if cell == 'lrc_ar' else (8 if wiring == 'ncp' else 4)
    model = _model(wiring, cell, size)
    one = rq4.OneStep(model)
    xs, dts = _inputs(T=6)
    preds, states, jacs = one.trajectory(xs, dts, 1.0)
    np.testing.assert_allclose(preds, model.predict((xs, dts), verbose=0),
                               rtol=1e-4, atol=1e-5)
    preds, states, jacs = one.trajectory(xs, dts, 0.0)
    np.testing.assert_allclose(preds, one.view.predict((xs, dts), verbose=0),
                               rtol=1e-4, atol=1e-5)
    assert np.isfinite(jacs).all() and np.isfinite(states).all()
    if cell == 'mlp':           # the stateless control never reads S
        assert not np.any(jacs[:, :, :one.dim, :one.dim])


def test_posthoc_script_on_2_epoch_fits(tmp_path, monkeypatch):
    _lv(tmp_path, monkeypatch, 'gru', 'dense3', 4)
    _lv(tmp_path, monkeypatch, 'lrc', 'ncp', 8, '--dt-scale', 'auto',
        '--forcing', 'mixed')
    _lv(tmp_path, monkeypatch, 'mlp', 'dense', 4, '--dt-scale', 'auto')
    _lv(tmp_path, monkeypatch, 'gru', 'tbt_cncp', 4)      # skipped: no view
    out, outdir = tmp_path / 'rq4.csv', tmp_path / 'rq4'
    rq4.main(['--results', str(tmp_path / '*.json'), '--out', str(out),
              '--outdir', str(outdir), '--hessian-top'])
    df = pd.read_csv(out).set_index('wiring')
    assert sorted(df.index) == ['dense', 'dense3', 'ncp']
    for p in ('tf', 'cl'):
        assert np.isfinite(df[f'{p}_hnorm_max']).all()
        for d in rq4.DELTAS:
            assert np.isfinite(df[f'{p}_jac{d}_median']).all()
    assert np.isfinite(df.loc['dense3', 'cl_jac16_layer2_median'])
    assert np.isfinite(df.loc['ncp', 'tf_jac4_command_median'])
    assert df.loc['dense', 'cl_jac1_max'] == 0.0           # mlp: no memory
    assert np.isfinite(df['hessian_top']).all()
    assert not (outdir / 'gru_tbt_cncp_seed0.json').exists()
    for name in ('lrc_ncp_seed0_mixed.json', 'mlp_dense_seed0.json'):
        record = json.loads((outdir / name).read_text())
        eff = record['config']['effective']
        assert eff['dt_scale'] == pytest.approx(1 / record['dt'])  # auto
        diag = record['rq4_diagnostics']
        assert diag['config_effective_sha256'] == config_hash(eff)
        assert diag['cl_jac16_median'] == pytest.approx(
            df.loc[name.split('_')[1], 'cl_jac16_median'])
        rebuild_from_result(str(outdir / name))    # weights_path still valid
