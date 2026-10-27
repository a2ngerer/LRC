# tests/experiments/test_suite_gate.py
# S0 smoke gate (issue a2ngerer/LRC#72): a synthetic run of the 93 smoke
# arm-tasks (plus one committee fit outside the list) passes with exit 0; one
# failure per check class fails exactly that check with exit 1.
import copy
import json
import math

import pytest

from experiments.suite_gate import SMOKE, expected_arms, main

# Written out here rather than taken from the gate: the passing fixture checks
# the gate's registry-derived cell sets.
SOLVER = {'lrc', 'mm_lrc', 'ctrnn', 'ltc', 'mm_ltc'}
ELASTANCE = {'lrc', 'mm_lrc', 'cfc_lrc', 'cfc_lrc_outer', 'cfc_mm_lrc'}
OVER_BUDGET = {('gru', 'dense3'), ('lstm', 'dense3')}
ROLLOUT = ('periodic_predator_prey', 'duffing', 'multitimescale_lib')
FOLDER = 'lr0.001_clipoff_seed100'
DATA_SEED = 20261802
CHECKS = ('arms', 'config', 'protocol_version', 'metrics', 'nan', 'budget',
          'elastance', 'ode_unfolds', 'determinism', 'floors')


def _rollout_metrics():
    return {'vpt_multi_median': 4.0, 'nrmse_32_anchor_8': 0.9,
            'nrmse_32_median': 0.9, 'non_finite': False, 'rate_ok': 0.5,
            'rate_bounded_wrong': 0.5, 'rate_unbounded': 0.0,
            **{f'vpt_anchor_{a}': 4.0 for a in (8, 16, 24, 32)}}


def _fit(key, cell, wiring):
    """A fit JSON of the real schema's gate-relevant fields."""
    task = {'pa': 'person_activity', 'co': 'committee'}.get(key,
                                                             'lotka_volterra')
    eff = {'task': task, 'ode_unfolds': 6 if cell in SOLVER else None,
           'elastance_type': 'asymmetric' if cell in ELASTANCE else None,
           'protocol_version': 'F1-2026-10-26'}
    rec = {'cell': cell, 'wiring': wiring, 'seed': 100,
           'budget_above_tolerance': (cell, wiring) in OVER_BUDGET,
           'budget_deviation': 0.01, 'weights_path': 'weights.h5',
           'config': {'effective': eff}}
    if task == 'lotka_volterra':
        eff.update(dt_scale=12.8, forcing='mixed', data_seed=DATA_SEED)
        rec.update(system=key, data_seed=DATA_SEED,
                   rollout_metrics_val=_rollout_metrics(),
                   rollout_metrics_test=_rollout_metrics())
        if key == 'multitimescale_lib':          # exempt, listed
            rec['rollout_metrics_val']['inv_drift_32_median'] = math.nan
    elif task == 'person_activity':
        eff['dt_scale'] = 3.7
        rec.update(val_macro_f1=0.5, test_macro_f1=0.5)
    else:
        rec.update(val_macro_f1_mean=0.5, test_macro_f1_mean_grid=0.5)
    return rec


@pytest.fixture()
def suite():
    """fits {(task key, arm, fit folder): record}, probe, floors."""
    fits = {(*arm, FOLDER): _fit(arm[0], *cw)
            for arm, cw in expected_arms(SMOKE).items()}
    fits[('co', 'ncp_cons', FOLDER)] = _fit('co', 'cfc_lrc', 'ncp')
    probe = {'tasks': {t: {'ratio_on_off': 1.1, 'max_abs_delta_on': 0.0}
                       for t in ('lv', 'pa')}}
    floors = {'floors': [{'system': s, 'data_seed': DATA_SEED,
                          'val': {'floor': 3.0}, 'test': {'floor': 4.0}}
                         for s in ROLLOUT]}
    return fits, probe, floors


def _gate(tmp_path, fits, probe, floors):
    """Write one run folder per task key, run the gate; (exit, report)."""
    runs = set()
    for (key, arm, folder), rec in fits.items():
        d = tmp_path / f'run_{key}' / 'results' / 'hpo' / arm / folder
        d.mkdir(parents=True)
        (d / f'{arm}_seed100.json').write_text(json.dumps(rec))
        runs.add(str(tmp_path / f'run_{key}'))
    (tmp_path / 'probe.json').write_text(json.dumps(probe))
    (tmp_path / 'floors.json').write_text(json.dumps(floors))
    out = tmp_path / 'REPORT.md'
    code = main(['--stage', 's0', '--runs', *sorted(runs),
                 '--determinism', str(tmp_path / 'probe.json'),
                 '--floors', str(tmp_path / 'floors.json'),
                 '--out', str(out)])
    return code, out.read_text()


def _failed(report):
    return [c for c in CHECKS if f'| {c} | FAIL |' in report]


def test_passing_fixture(tmp_path, suite):
    code, report = _gate(tmp_path, *suite)
    assert code == 0, report
    assert report.startswith('# S0 smoke gate: PASS')
    assert all(f'| {c} | PASS |' in report for c in CHECKS)
    assert '93/93 arm-tasks present' in report
    assert '(F1-2026-10-26)' in report and '(3 draws)' in report
    assert ('- multitimescale_lib/cfc_ncp: '
            'rollout_metrics_val.inv_drift_32_median') in report
    assert '- co/ncp_cons' in report.split('outside the expected arm list')[1]


def _arm(fits, key, arm):
    return fits[(key, arm, FOLDER)]


def _eff(fits, key, arm):
    return _arm(fits, key, arm)['config']['effective']


LV, DUF, MT = ROLLOUT
FAILURES = {
    'missing arm': ('arms', lambda f, p, fl: f.pop(('pa', 'gru_ncp', FOLDER))),
    'two fits of one arm': ('arms', lambda f, p, fl: f.__setitem__(
        ('duffing', 'gru_ncp', 'lr0.003_clipoff_seed100'),
        copy.deepcopy(_arm(f, DUF, 'gru_ncp')))),
    'no dt_scale': ('config', lambda f, p, fl: _eff(f, LV, 'cfc_ncp').pop(
        'dt_scale')),
    'no weights saved': ('config', lambda f, p, fl: _arm(
        f, 'pa', 'cfc_dense').__setitem__('weights_path', None)),
    'no forcing': ('config', lambda f, p, fl: _eff(f, MT, 'gru_ncp').pop(
        'forcing')),
    'no protocol_version': ('protocol_version', lambda f, p, fl: _eff(
        f, DUF, 'lrc_ncp').pop('protocol_version')),
    'two protocol versions': ('protocol_version', lambda f, p, fl: _eff(
        f, LV, 'lrc_ncp').__setitem__('protocol_version', 'F1-2026-10-27')),
    'no anchor key': ('metrics', lambda f, p, fl: _arm(
        f, DUF, 'cfc_lrc_ncp')['rollout_metrics_test'].pop('vpt_anchor_24')),
    'non-finite rollout': ('metrics', lambda f, p, fl: _arm(
        f, LV, 'mlp_dense')['rollout_metrics_val'].__setitem__(
            'non_finite', True)),
    'no PA primary metric': ('metrics', lambda f, p, fl: _arm(
        f, 'pa', 'lrc_ncp').pop('val_macro_f1')),
    'NaN metric': ('nan', lambda f, p, fl: _arm(
        f, DUF, 'gru_dense')['rollout_metrics_test'].__setitem__(
            'nrmse_32_median', math.inf)),
    'inv_drift NaN outside multitimescale_lib': ('nan', lambda f, p, fl: _arm(
        f, LV, 'gru_dense')['rollout_metrics_val'].__setitem__(
            'inv_drift_32_median', math.nan)),
    'over budget': ('budget', lambda f, p, fl: _arm(
        f, 'pa', 'cfc_ncp').__setitem__('budget_above_tolerance', True)),
    'interp elastance': ('elastance', lambda f, p, fl: _eff(
        f, LV, 'lrc_dense').__setitem__('elastance_type', 'interp')),
    'cfc_mm_lrc without elastance': ('elastance', lambda f, p, fl: _eff(
        f, DUF, 'cfc_mm_lrc_ncp').__setitem__('elastance_type', None)),
    'ctrnn one sub-step': ('ode_unfolds', lambda f, p, fl: _eff(
        f, 'pa', 'ctrnn_dense3').__setitem__('ode_unfolds', 1)),
    'ltc without ode_unfolds': ('ode_unfolds', lambda f, p, fl: _eff(
        f, DUF, 'ltc_ncp').__setitem__('ode_unfolds', None)),
    'slow determinism': ('determinism', lambda f, p, fl: p['tasks']['pa']
                         .__setitem__('ratio_on_off', 1.4)),
    'irreproducible on pair': ('determinism', lambda f, p, fl: p['tasks']['lv']
                               .__setitem__('max_abs_delta_on', 2e-3)),
    'probe without pa': ('determinism', lambda f, p, fl: p['tasks'].pop('pa')),
    'no duffing floor line': ('floors', lambda f, p, fl: fl['floors'].pop(1)),
    'NaN test floor': ('floors', lambda f, p, fl: fl['floors'][2][
        'test'].__setitem__('floor', math.nan)),
}


@pytest.mark.parametrize('case', FAILURES)
def test_one_failure_per_check_class(case, tmp_path, suite):
    check, mutate = FAILURES[case]
    mutate(*suite)
    code, report = _gate(tmp_path, *suite)
    assert code == 1
    assert report.startswith('# S0 smoke gate: FAIL')
    assert _failed(report) == [check], report
    assert f'## FAIL: {check}' in report
