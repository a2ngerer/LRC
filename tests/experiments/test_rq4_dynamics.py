# tests/experiments/test_rq4_dynamics.py
# Issue a2ngerer/LRC#66: --eval-only --rq4-dynamics adds the largest Lyapunov
# exponent of the free run (Benettin) and the solver sweep k in {1, 2, 6, 12}
# to a fit's result JSON, from weights.h5 and config.effective only.
import json
import sys

import numpy as np
import pytest
import tensorflow as tf

from experiments import run_lotka_volterra_benchmark as runner
from src.benchmark.checkpoints import rebuild_from_result
from src.evaluation.lyapunov import largest_lyapunov
from src.tasks.lotka_volterra.datasets import load_lotka_volterra


def _train(tmp_path, monkeypatch, wiring, cell, *extra):
    monkeypatch.setattr(sys, 'argv', [
        'run_lotka_volterra_benchmark.py', '--wiring', wiring, '--cell', cell,
        '--size', '4', '--epochs', '2', '--n-trajectories', '6',
        '--seq-len', '8', '--anchors', '2,4', '--horizon', '4',
        '--batch-size', '4', '--outdir', str(tmp_path), *extra])
    runner.main()
    (path,) = tmp_path.glob('*.json')
    return path


def _dynamics(monkeypatch, path):
    monkeypatch.setattr(sys, 'argv', [
        'run_lotka_volterra_benchmark.py', '--eval-only', str(path),
        '--anchors', '2,4', '--horizon', '4', '--rq4-dynamics'])
    runner.main()
    return json.loads(path.read_text())


def _data(rec):
    eff = rec['config']['effective']
    return load_lotka_volterra(n_trajectories=eff['n_trajectories'],
                               seq_len=eff['seq_len'], split=eff['split'],
                               seed=eff['data_seed'], system=eff['system'],
                               t_span=eff['t_span'], spread=eff['ic_spread'])


def test_benettin_recovers_the_largest_exponent_of_a_linear_map():
    a = tf.constant([[2.0, 1.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, -1.5]])
    lam = largest_lyapunov(lambda z: z @ tf.transpose(a),
                           np.ones((5, 3), np.float32), 96)
    assert lam.shape == (5,)
    np.testing.assert_allclose(lam, np.log(2.0), atol=0.05)
    zero = largest_lyapunov(lambda z: 0.0 * z, np.ones((2, 3), np.float32), 4)
    assert np.all(zero == -np.inf)


def test_free_run_map_reproduces_the_anchored_rollout(tmp_path, monkeypatch):
    """Iterating the map from the anchor state yields the forced view's
    multi-anchor free-run predictions (dense3: stacked core, dt_scale)."""
    path = _train(tmp_path, monkeypatch, 'dense3', 'lrc', '--ode-unfolds', '2',
                  '--dt-scale', 'auto')
    rec = json.loads(path.read_text())
    model, data = rebuild_from_result(path), _data(rec)
    ref = runner.anchored_rollouts(model, data, 'test', False, (2, 4), 4,
                                   runner.forced_rollout_model(model))
    for a in (2, 4):
        step, z = runner.free_run_map(model, data, 'test', a)
        preds = []
        for _ in range(4):
            preds.append(z[:, -3:-1].numpy())   # last prediction, width 2
            z = step(z)
        np.testing.assert_allclose(np.stack(preds, 1), ref[a], rtol=1e-4,
                                   atol=1e-5)


def test_eval_only_rq4_dynamics_on_an_lrc_fit(tmp_path, monkeypatch):
    path = _train(tmp_path, monkeypatch, 'ncp', 'lrc', '--ode-unfolds', '2')
    rec = json.loads(path.read_text())
    again = _dynamics(monkeypatch, path)
    assert again['eval_only']['max_rel_dev'] <= 1e-3
    dyn = again['rq4_dynamics']
    lyap = dyn['lyapunov']
    assert lyap['split'] == 'test' and lyap['horizon'] == 4
    for key in ('lambda_max_median', 'lambda_max_anchor_2',
                'lambda_max_anchor_4'):
        assert np.isfinite(lyap[key])
    assert list(dyn['solver']) == ['1', '2', '6', '12']
    for metrics in dyn['solver'].values():
        assert {'vpt_multi_median', 'vpt_anchor_2', 'nrmse_32_anchor_4'} <= set(
            metrics)
    # k = the trained substeps reproduces the stored test endpoint
    trained, stored = dyn['solver']['2'], rec['rollout_metrics_test']
    for key in ('vpt_multi_median', 'nrmse_32_anchor_2', 'nrmse_32_anchor_4'):
        assert trained[key] == pytest.approx(stored[key], rel=1e-3, abs=1e-6)
    assert dyn['wallclock_s'] >= 0


def test_solver_rebuild_overrides_the_substeps(tmp_path, monkeypatch):
    path = _train(tmp_path, monkeypatch, 'dense', 'lrc', '--ode-unfolds', '2')
    for k in (1, 12):
        model = rebuild_from_result(path, ode_unfolds=k)
        (rnn,) = [l for l in model.layers if isinstance(l, tf.keras.layers.RNN)]
        assert rnn.cell._ode_unfolds == k


@pytest.mark.parametrize('wiring,cell,missing', [
    ('ncp', 'cfc_lrc', 'solver'),               # closed form: no substeps
    ('tbt_cncp_concat', 'lrc', 'lyapunov'),     # no forced wrapper
])
def test_inapplicable_parts_are_none(wiring, cell, missing, tmp_path,
                                     monkeypatch):
    path = _train(tmp_path, monkeypatch, wiring, cell)
    dyn = _dynamics(monkeypatch, path)['rq4_dynamics']
    assert dyn[missing] is None
    assert dyn[{'solver': 'lyapunov', 'lyapunov': 'solver'}[missing]]


def test_rq4_dynamics_needs_eval_only(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['run_lotka_volterra_benchmark.py',
                                      '--wiring', 'dense', '--rq4-dynamics'])
    with pytest.raises(SystemExit):
        runner.main()
