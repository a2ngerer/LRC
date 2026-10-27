# tests/experiments/test_eval_robustness.py
# Issue a2ngerer/LRC#34: eval-only robustness endpoint of H-c. Coordinate
# noise on Person Activity, out-of-distribution initial conditions and a
# longer horizon on the rollouts, from weights.h5 + config.effective only;
# one CSV row per fit keyed like aggregate_results.load_sequence_runs.
import json
import sys

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

import src.tasks.person_activity as pa_pkg
from experiments import eval_robustness as er
from experiments import run_lotka_volterra_benchmark as lv_runner
from experiments import run_person_activity_benchmark as pa_runner
from experiments.aggregate_results import load_sequence_runs
from src.benchmark.checkpoints import rebuild_from_result
from src.tasks.lotka_volterra.datasets import (load_lotka_volterra,
                                               sample_ood_trajectories)
from tests.benchmark.test_checkpoints import _pa_stub


def _lv_fit(d, monkeypatch, system, wiring, seed=0):
    monkeypatch.setattr(sys, 'argv', [
        'run_lotka_volterra_benchmark.py', '--system', system, '--wiring',
        wiring, '--cell', 'cfc_lrc', '--size', '4', '--seed', str(seed),
        '--epochs', '2', '--n-trajectories', '6', '--seq-len', '8',
        '--anchors', '2,4', '--horizon', '4', '--batch-size', '4',
        '--forcing', 'mixed', '--dt-scale', 'auto', '--outdir', str(d)])
    lv_runner.main()


def _pa_fit(d, monkeypatch, epochs, *extra):
    monkeypatch.setattr(pa_pkg, 'load_person_activity', _pa_stub)
    monkeypatch.setattr(er, 'load_person_activity', _pa_stub)
    monkeypatch.setattr(sys, 'argv', [
        'run_person_activity_benchmark.py', '--wiring', 'ncp', '--cell',
        'cfc_lrc', '--size', '8', '--epochs', str(epochs), '--outdir',
        str(d), *extra])
    pa_runner.main()
    return d / 'cfc_lrc_ncp_seed0.json'


def test_ood_initial_conditions_leave_the_training_hull():
    """Spread 0.8 puts initial conditions outside the convex hull of the
    training draw at 0.4 (LV and Duffing)."""
    for system in er.SYSTEMS:
        train = load_lotka_volterra(n_trajectories=48, seq_len=2,
                                    system=system).train_traj[:, 0]
        ood = sample_ood_trajectories(er.N_ROLLOUT, 0, er.OOD_SPREAD,
                                      seq_len=2, system=system)[:, 0]
        assert (Delaunay(train).find_simplex(ood) < 0).any(), system


def test_sigma_zero_reproduces_the_clean_metric(tmp_path, monkeypatch):
    """The runner saves the best-val epoch's weights (here epoch 2 of 3,
    val curve scripted), so sigma 0 on the rebuilt fit is the runner's test
    macro-F1; the noisy draws are fixed."""
    real_f1, real_cms, val_weights = (pa_runner.macro_f1,
                                      pa_runner.split_confusions, [])

    def spy_cms(model, data, part, batch_size):     # weights of each epoch
        if part == 'val':
            val_weights.append(model.get_weights())
        return real_cms(model, data, part, batch_size)
    curve = iter((0.1, 0.9, 0.2))      # the first three calls: val curve
    monkeypatch.setattr(pa_runner, 'split_confusions', spy_cms)
    monkeypatch.setattr(pa_runner, 'macro_f1',
                        lambda cm: next(curve, None) or real_f1(cm))
    path = _pa_fit(tmp_path, monkeypatch, 3, '--lr', '0.05')
    record = json.loads(path.read_text())
    assert (record['best_val_epoch'], record['epochs']) == (2, 3)
    saved = rebuild_from_result(str(path)).get_weights()

    def is_saved(weights):
        return (len(weights) == len(saved)
                and all(map(np.array_equal, weights, saved)))
    assert is_saved(val_weights[1]) and not is_saved(val_weights[2])
    row = er.evaluate(str(path))
    assert row['f1_sigma_0'] == record['test_macro_f1']
    assert row['metric_robust'] == row['f1_sigma_0.3']
    assert (row['task'], row['system'], row['arm']) == (
        'pa', 'pa', 'cfc_lrc/ncp [dt_scale=1, fixed width]')
    assert er.evaluate(str(path))['f1_sigma_0.3'] == row['f1_sigma_0.3']


def test_script_runs_on_two_epoch_fits_of_each_task(tmp_path, monkeypatch):
    """LV (two arms of one replicate), Duffing and PA: one row per fit with
    the loader's arm key; both arms of the replicate get the same fresh
    draws (OOD at spread 0.8 over seq_len, the long set at the fit's spread
    over the last anchor + 256 steps); other JSONs are skipped."""
    for wiring in ('ncp', 'dense3'):
        _lv_fit(tmp_path / 'lv', monkeypatch, 'periodic_predator_prey',
                wiring)
    _lv_fit(tmp_path / 'duffing', monkeypatch, 'duffing', 'cncp', seed=1)
    _pa_fit(tmp_path / 'pa', monkeypatch, epochs=2)
    (tmp_path / 'lv' / 'hpo_selection.json').write_text('{"task": "lv"}')
    draws = []
    real = er.sample_ood_trajectories

    def spy(n, seed, spread, seq_len, system, t_span):
        draws.append((n, seed, spread, seq_len, system, t_span))
        return real(n, seed, spread, seq_len, system, t_span)
    monkeypatch.setattr(er, 'sample_ood_trajectories', spy)
    out = tmp_path / 'rob.csv'
    er.main(['--results', str(tmp_path / '**' / '*.json'), '--out', str(out)])

    rows = pd.read_csv(out)
    assert len(rows) == 4
    seq = pd.concat([load_sequence_runs(str(tmp_path / d), t) for d, t in (
        ('lv', 'lv'), ('duffing', 'lv'), ('pa', 'pa'))])
    key = ['system', 'arm', 'seed']
    assert sorted(map(tuple, rows[key].values.tolist())) == sorted(
        map(tuple, seq[key].values.tolist()))
    lv = rows[rows['task'] == 'lv']
    assert (lv['metric_robust'] == lv['ood_vpt_multi_median']).all()
    assert (lv['ood_horizon'] == 8).all()
    assert (lv['ood_anchor_horizon'] == 4).all()
    assert (lv['long_anchor_horizon'] == er.LONG_HORIZON).all()
    assert (lv['long_horizon'] == 4 + er.LONG_HORIZON).all()
    for col in ('ood_nrmse_128_median', 'ood_orbit_error_median',
                'ood_inv_drift_max_median', 'ood_rate_ok',
                'ood_rate_unbounded', 'ood_rate_bounded_wrong',
                'long_vpt_multi_median'):
        assert lv[col].notna().all(), col
    pa = rows[rows['task'] == 'pa'].iloc[0]
    assert pa['metric_robust'] == pa['f1_sigma_0.3']
    assert {f'f1_sigma_{s:g}' for s in er.SIGMAS} <= set(rows)

    ppp = [d for d in draws if d[4] == 'periodic_predator_prey']
    assert ppp[:2] == ppp[2:]                 # ncp and dense3 share the draws
    ood, long = ppp[:2]
    assert ood[:4] == (er.N_ROLLOUT, 20260702 + er.SEED_OFFSET, 0.8, 8)
    assert long[2:4] == (0.4, 4 + er.LONG_HORIZON)
    assert ood[5] == (0.0, 10.0)              # the fit's dt: 10 / 8 per step
    assert long[5] == (0.0, 10.0 * (4 + er.LONG_HORIZON) / 8)
