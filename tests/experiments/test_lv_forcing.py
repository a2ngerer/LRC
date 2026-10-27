# tests/experiments/test_lv_forcing.py
# Issue #25: --forcing / --alpha-min of the LV runner and the pilot gate
# (experiments/forcing_gate.py).
import json
import sys
import types

import numpy as np
import pytest
from scipy.stats import kendalltau

from experiments import forcing_gate as gate_mod
from experiments import run_lotka_volterra_benchmark as runner

TINY = ['--seed', '0', '--epochs', '2', '--size', '8', '--n-trajectories', '10',
        '--anchors', '4,8', '--horizon', '8',
        '--seq-len', '16', '--batch-size', '4', '--no-save-weights']

# Pre-change runner (base commit 3657ac5, CPU), same TINY arguments: per-epoch
# val loss, final test teacher-forced MSE, test closed-loop MSE
# (closed_loop_rollout then, the alpha = 0 wrapper now).
GOLDEN = {
    ('cfc_lrc', 'dense3'): ([1.460373, 1.363666], 0.7659856081008911,
                            0.7995055913925171),
    ('gru', 'ncp'): ([1.364178, 1.361511], 0.7714312672615051,
                     0.7728983163833618),
}


def _run(tmp_path, monkeypatch, cell, wiring, *extra):
    monkeypatch.setattr(sys, 'argv', [
        'run_lotka_volterra_benchmark.py', '--wiring', wiring, '--cell', cell,
        *TINY, '--outdir', str(tmp_path), *extra])
    runner.main()


@pytest.mark.parametrize('cell,wiring', list(GOLDEN))
def test_teacher_default_matches_pre_change_runner(cell, wiring, tmp_path,
                                                   monkeypatch):
    _run(tmp_path, monkeypatch, cell, wiring)        # no flag = teacher
    r = json.loads((tmp_path / f'{cell}_{wiring}_seed0.json').read_text())
    val_curve, tf_mse, cl_mse = GOLDEN[(cell, wiring)]
    np.testing.assert_allclose(r['val_loss_curve'], val_curve, atol=1e-5)
    assert r['test_teacher_forced_mse'] == pytest.approx(tf_mse, abs=1e-5)
    assert r['test_closed_loop_mse'] == pytest.approx(cl_mse, abs=1e-5)
    assert (r['forcing'], r['alpha_min'], r['alpha_curve']) == (
        'teacher', None, None)
    assert r['config']['effective']['forcing'] == 'teacher'


def test_mixed_forcing_run(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch, 'gru', 'ncp', '--forcing', 'mixed')
    r = json.loads((tmp_path / 'gru_ncp_seed0_mixed.json').read_text())
    assert r['forcing'] == 'mixed' and r['alpha_min'] == 0.2
    assert r['alpha_curve'] == [1.0, 0.2]
    assert r['config']['effective']['alpha_min'] == 0.2
    # epoch 1 runs at alpha = 1 from the teacher arm's initial weights
    val0, _, _ = GOLDEN[('gru', 'ncp')]
    assert r['val_loss_curve'][0] == pytest.approx(val0[0], abs=1e-5)
    assert r['val_loss_curve'][1] != pytest.approx(val0[1], abs=1e-5)
    assert np.isfinite(r['rollout_metrics_test']['vpt_median'])
    assert len(r['closed_loop_traj'][0]) == 17


@pytest.mark.parametrize('extra', [
    ['--wiring', 'tbt_cncp_concat', '--forcing', 'mixed'],
    ['--wiring', 'dense', '--alpha-min', '1.5']])
def test_invalid_forcing_args(extra, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', [
        'run_lotka_volterra_benchmark.py', *extra, '--outdir', str(tmp_path)])
    with pytest.raises(SystemExit):
        runner.main()


# --- pilot gate --------------------------------------------------------------
T = 8
STATS = {'mean': [0.0, 0.0], 'std': [1.0, 1.0], 'train_abs_max': 1.0}


def _rec(system, forcing, cell, wiring, seed, vpts, diverged=False):
    """Result JSON whose test rollouts have exactly the given VPTs (1..T)."""
    gt = np.zeros((len(vpts), T + 1, 2))
    cl = gt.copy()
    for i, v in enumerate(vpts):
        cl[i, v:] = 1.0                              # e_t = 1 > 0.5 from step v
    return {'system': system, 'forcing': forcing, 'cell': cell,
            'wiring': wiring, 'seed': seed, 'norm_stats': STATS,
            'ground_truth_traj': gt.tolist(), 'closed_loop_traj': cl.tolist(),
            'training_diverged': diverged}


def _pilot(score, systems=('periodic_predator_prey',)):
    """score(forcing, cell, wiring) -> VPT of every trajectory (2 seeds x 3)."""
    return [_rec(s, f, c, w, k, [score(f, c, w)] * 3)
            for s in systems for f in gate_mod.FORCINGS
            for c, w in gate_mod.ARMS for k in (0, 1)]


@pytest.fixture
def with_tau(monkeypatch):
    stats = types.ModuleType('src.evaluation.stats')
    stats.kendall_tau_b = lambda a, b: float(kendalltau(a, b,
                                                        variant='b').statistic)
    monkeypatch.setitem(sys.modules, 'src.evaluation.stats', stats)


def _rank(f, c, w):
    return 1 + gate_mod.ARMS.index((c, w))           # distinct, same order


def test_gate_same_ranking_keeps_teacher(with_tau):
    report, decision = gate_mod.gate(_pilot(_rank))
    r = report['periodic_predator_prey']
    assert r['tau_b'] == pytest.approx(1.0)
    assert r['median_vpt']['teacher'] == [1, 2, 3, 4, 5, 6]
    assert r['pair_sign'] == {'teacher': -1, 'mixed': -1}
    assert not r['sign_flip'] and decision == 'teacher'


def test_gate_low_tau_switches_to_mixed(with_tau):
    # mixed reverses the order of the four ncp/cfc arms, keeps the dense3 pair
    order = {('gru', 'ncp'): 6, ('cfc', 'dense3'): 5, ('cfc', 'ncp'): 4,
             ('cfc_lrc', 'ncp'): 3}
    score = lambda f, c, w: (_rank(f, c, w) if f == 'teacher'
                             else order.get((c, w), _rank(f, c, w)))
    report, decision = gate_mod.gate(_pilot(score))
    tau = report['periodic_predator_prey']['tau_b']
    assert tau < gate_mod.TAU_MIN and decision == 'mixed'


def test_gate_sign_flip_decides_without_tau(monkeypatch):
    monkeypatch.setitem(sys.modules, 'src.evaluation.stats', None)  # no #29
    flip = lambda f, c, w: (7 - _rank(f, c, w)) if (f == 'mixed'
                                                     and w == 'dense3') \
        else _rank(f, c, w)
    report, decision = gate_mod.gate(_pilot(flip))
    r = report['periodic_predator_prey']
    assert r['tau_b'] is None and r['sign_flip'] and decision == 'mixed'
    _, decision = gate_mod.gate(_pilot(_rank))
    assert decision.startswith('undecided')


def test_gate_failed_fit_ranks_worst_and_missing_fit_errors(with_tau):
    recs = _pilot(_rank)
    for r in recs:
        if (r['forcing'], r['cell'], r['wiring']) == ('mixed', 'cfc_lrc',
                                                      'ncp'):
            r['training_diverged'] = True
    report, _ = gate_mod.gate(recs)
    assert report['periodic_predator_prey']['median_vpt']['mixed'][-1] \
        == -np.inf
    with pytest.raises(ValueError, match='1 pilot fits missing'):
        gate_mod.gate(recs[1:])
