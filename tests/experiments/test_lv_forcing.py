# tests/experiments/test_lv_forcing.py
# Issue #25: --forcing / --alpha-min of the LV runner; issue #69: the per-fit
# rank report of the forcing pilot (experiments/forcing_gate.py).
import json
import os
import subprocess
import sys
from glob import glob

import numpy as np
import pytest

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
    assert r['config']['effective']['deterministic'] is False   # #70


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


# --- pilot rank report (#69) ------------------------------------------------
T = 8
STATS = {'mean': [0.0, 0.0], 'std': [1.0, 1.0], 'train_abs_max': 1.0}
SYS = 'periodic_predator_prey'
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def _rec(system, forcing, cell, wiring, seed, vpts, diverged=False,
         multi=None):
    """Result JSON whose test rollouts have exactly the given VPTs (1..T)."""
    gt = np.zeros((len(vpts), T + 1, 2))
    cl = gt.copy()
    for i, v in enumerate(vpts):
        cl[i, v:] = 1.0                              # e_t = 1 > 0.5 from step v
    r = {'system': system, 'forcing': forcing, 'cell': cell,
         'wiring': wiring, 'seed': seed, 'norm_stats': STATS,
         'ground_truth_traj': gt.tolist(), 'closed_loop_traj': cl.tolist(),
         'training_diverged': diverged}
    if multi is not None:
        r['rollout_metrics_test'] = {'vpt_multi_median': multi}
    return r


def _pilot(vpts, multi=None):
    """vpts(forcing, cell, wiring, seed) -> VPT of the fit's 3 trajectories."""
    return [_rec(SYS, f, c, w, k, vpts(f, c, w, k),
                 multi=None if multi is None else multi(f, c, w, k))
            for f in gate_mod.FORCINGS for c, w in gate_mod.ARMS
            for k in (0, 1, 2)]


def _rank(f, c, w):
    return 1 + gate_mod.ARMS.index((c, w))           # distinct, same order


def test_per_fit_and_pooled_aggregation_differ():
    # per-fit medians 8, 8, 2 -> 8; pooled 1, 1, 2, 2, 2, 8, 8, 8, 8 -> 2
    fit = {0: [1, 8, 8], 1: [1, 8, 8], 2: [2, 2, 2]}
    recs = [_rec(SYS, 'teacher', 'gru', 'dense3', k, v)
            for k, v in fit.items()]
    assert gate_mod.arm_score(recs) == 8.0
    assert gate_mod.arm_score(recs, pooled=True) == 2.0


def test_report_is_descriptive_with_tau_se():
    rep = gate_mod.report(_pilot(lambda f, c, w, k: [_rank(f, c, w)] * 3))
    r = rep[SYS]
    assert r['median_vpt'] == {'teacher@k0': [1, 2, 3, 4, 5, 6],
                               'mixed@k0': [1, 2, 3, 4, 5, 6]}
    assert r['tau_b'] == {'teacher@k0 vs mixed@k0': pytest.approx(1.0)}
    assert r['tau_b_se'] == pytest.approx(np.sqrt(34 / 270))   # n = 6
    assert r['aggregation'] == 'per_fit' and r['n_seeds'] == 3
    assert not {'decision', 'sign_flip'} & set(r)


def test_report_compares_k0_and_multi_anchor_rankings():
    # multi-anchor reverses the mixed ranking, keeps the teacher one
    multi = lambda f, c, w, k: (_rank(f, c, w) if f == 'teacher'
                                else 7 - _rank(f, c, w)) + 0.1 * k
    rep = gate_mod.report(_pilot(lambda f, c, w, k: [_rank(f, c, w)] * 3,
                                 multi))
    tau = rep[SYS]['tau_b']
    assert tau == pytest.approx({
        'teacher@k0 vs mixed@k0': 1.0,
        'teacher@multi_anchor vs mixed@multi_anchor': -1.0,
        'teacher@k0 vs teacher@multi_anchor': 1.0,
        'mixed@k0 vs mixed@multi_anchor': -1.0})
    assert rep[SYS]['median_vpt']['teacher@multi_anchor'][0] \
        == pytest.approx(1.1)                           # median over seeds


def test_failed_fit_ranks_worst_and_missing_fit_errors():
    recs = _pilot(lambda f, c, w, k: [_rank(f, c, w)] * 3)
    for r in recs:
        if (r['forcing'], r['cell'], r['wiring']) == ('mixed', 'cfc_lrc',
                                                      'ncp'):
            r['training_diverged'] = True
    rep = gate_mod.report(recs)
    assert rep[SYS]['median_vpt']['mixed@k0'][-1] == -np.inf
    with pytest.raises(ValueError, match='1 pilot fits missing'):
        gate_mod.report(recs[1:])


def test_stats_importable_from_a_plain_runner_script(tmp_path):
    # feat-25 printed tau_b "n/a" because src.evaluation.stats was missing;
    # the script now imports it at module level, from any working directory.
    out = subprocess.run(
        [sys.executable, os.path.join(ROOT, 'experiments', 'forcing_gate.py'),
         '--help'], cwd=tmp_path, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


# Forcing-gate pilot (#25, 200 epochs, 3 seeds): tau_b teacher vs mixed on the
# k0 endpoint. Pooled = the #25 gate (Duffing -0.645; quoted as -0.65 and as
# -0.64 elsewhere, both inside the tolerance), per-fit = this report.
PILOT_DIRS = [os.path.join(ROOT, 'results', d, '2026-10-25_forcing25-pilot',
                           'results')
              for d in ('01_lotka_volterra', '02_duffing')]
PILOT_TAU = {'pooled': {'periodic_predator_prey': 0.07, 'duffing': -0.65},
             'per_fit': {'periodic_predator_prey': -0.07, 'duffing': -0.47}}


@pytest.mark.skipif(not all(os.path.isdir(d) for d in PILOT_DIRS),
                    reason='forcing pilot JSONs not under results/')
def test_reproduces_pilot_tau_b():
    recs = []
    for d in PILOT_DIRS:
        for p in sorted(glob(os.path.join(d, '*.json'))):
            with open(p) as f:
                recs.append(json.load(f))
    assert len(recs) == 72
    for agg, expected in PILOT_TAU.items():
        rep = gate_mod.report(recs, pooled=agg == 'pooled')
        got = {s: r['tau_b']['teacher@k0 vs mixed@k0'] for s, r in rep.items()}
        assert got == pytest.approx(expected, abs=0.01), agg
