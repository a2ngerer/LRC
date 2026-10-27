# tests/experiments/test_suite_gate.py
# S0 smoke gate (issue a2ngerer/LRC#72): a synthetic run of the 93 smoke
# arm-tasks (plus one committee fit outside the list) passes with exit 0; one
# failure per check class fails exactly that check with exit 1. Corrections
# after the first S0 run (#74): all-undefined inv_drift_* summaries are listed,
# a slow determinism probe is a DECISION, a missing probe fails. S1 pilot gate
# (#75): a synthetic set of the 61 pilot fits passes; one fixture per gate
# branch (PASS, FAIL and the G0-E epoch DECISION) built from synthetic records.
import copy
import json
import math
import re

import pytest

from experiments.suite_gate import (PILOT, SMOKE, expected_arms, main,
                                    pilot_fits)

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
            'inv_drift_32_median': 0.1, 'n_inv_drift_undefined': 0,
            **{f'vpt_anchor_{a}': 4.0 for a in (8, 16, 24, 32)}}


def _undefined(m, **extra):
    """Every trajectory of a rollout metrics block violates positivity (the
    ctrnn/dense LV fit of the first S0 run): the inv_drift_* summaries are
    NaN and n_inv_drift_undefined equals the trajectory count."""
    m.update({**{f'inv_drift_{h}_{s}': math.nan for h in (32, 128, 'max')
                 for s in ('median', 'mean')},
              'n_inv_drift_undefined': 16, **extra})


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
        eff.update(dt_scale=12.8, forcing='mixed', data_seed=DATA_SEED,
                   anchors=[8, 16, 24, 32])
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
    """Write one run folder per task key, run the gate; (exit, report).
    probe: a dict or raw text (written to probe.json), a path (passed as
    is) or None (no --determinism)."""
    runs = set()
    for (key, arm, folder), rec in fits.items():
        d = tmp_path / f'run_{key}' / 'results' / 'hpo' / arm / folder
        d.mkdir(parents=True)
        (d / f'{arm}_seed100.json').write_text(json.dumps(rec))
        runs.add(str(tmp_path / f'run_{key}'))
    det = []
    if isinstance(probe, (dict, str)):
        (tmp_path / 'probe.json').write_text(
            probe if isinstance(probe, str) else json.dumps(probe))
        probe = tmp_path / 'probe.json'
    if probe is not None:
        det = ['--determinism', str(probe)]
    (tmp_path / 'floors.json').write_text(json.dumps(floors))
    out = tmp_path / 'REPORT.md'
    code = main(['--stage', 's0', '--runs', *sorted(runs), *det,
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
    # the pipes of |Delta| are escaped: the row keeps its four columns
    row = next(r for r in report.splitlines()
               if r.startswith('| determinism |'))
    assert '\\|Delta on\\|' in row
    assert len(re.split(r'(?<!\\)\|', row)) == 6, row


def _arm(fits, key, arm):
    return fits[(key, arm, FOLDER)]


def _eff(fits, key, arm):
    return _arm(fits, key, arm)['config']['effective']


def _to_top(fits, key, arm, field):
    """Move a config.effective field to the top level of the record (only
    weights_path is read there)."""
    _arm(fits, key, arm)[field] = _eff(fits, key, arm).pop(field)


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
    'no anchors': ('config', lambda f, p, fl: _eff(f, DUF, 'lstm_ncp').pop(
        'anchors')),
    'dt_scale only at the top level': ('config', lambda f, p, fl: _to_top(
        f, 'pa', 'gru_ncp', 'dt_scale')),
    'no protocol_version': ('protocol_version', lambda f, p, fl: _eff(
        f, DUF, 'lrc_ncp').pop('protocol_version')),
    'protocol_version only at the top level': (
        'protocol_version', lambda f, p, fl: _to_top(
            f, LV, 'gru_ncp', 'protocol_version')),
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
    'NaN beside undefined inv_drift': ('nan', lambda f, p, fl: _undefined(
        _arm(f, LV, 'ctrnn_dense')['rollout_metrics_val'],
        nrmse_32_median=math.nan)),
    'Inf inv_drift of an undefined block': ('nan', lambda f, p, fl: _undefined(
        _arm(f, DUF, 'gru_dense')['rollout_metrics_test'],
        inv_drift_32_mean=math.inf)),
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
    'irreproducible on pair': ('determinism', lambda f, p, fl: p['tasks']['lv']
                               .__setitem__('max_abs_delta_on', 2e-3)),
    'probe without pa': ('determinism', lambda f, p, fl: p['tasks'].pop('pa')),
    'no duffing floor line': ('floors', lambda f, p, fl: fl['floors'].pop(1)),
    'NaN test floor': ('floors', lambda f, p, fl: fl['floors'][2][
        'test'].__setitem__('floor', math.nan)),
}


def test_all_undefined_inv_drift_is_listed(tmp_path, suite):
    fit = _arm(suite[0], LV, 'ctrnn_dense')
    for split in ('val', 'test'):
        _undefined(fit[f'rollout_metrics_{split}'])
    code, report = _gate(tmp_path, *suite)
    assert code == 0, report
    assert '| nan | PASS |' in report
    listed = report.split('## Listed')[1]
    for split in ('val', 'test'):
        assert (f'- periodic_predator_prey/ctrnn_dense: '
                f'rollout_metrics_{split} (n_inv_drift_undefined 16): '
                f'inv_drift_128_mean, inv_drift_128_median, '
                f'inv_drift_32_mean, ') in listed


def test_slow_determinism_is_a_decision(tmp_path, suite):
    suite[1]['tasks']['pa']['ratio_on_off'] = 1.5      # |Delta| stays 0
    code, report = _gate(tmp_path, *suite)
    assert code == 0, report
    assert report.startswith('# S0 smoke gate: PASS')
    assert '| determinism | DECISION |' in report and not _failed(report)
    assert ('## DECISION: determinism\n\n- op-determinism off for the suite '
            '(F1 decision l): runtime ratio on/off above 1.3 on pa 1.5') \
        in report


@pytest.mark.parametrize('probe', [None, '{"tasks": ', 'missing.json'])
def test_determinism_not_measured(probe, tmp_path, suite):
    fits, _, floors = suite
    if probe == 'missing.json':
        probe = tmp_path / probe
    code, report = _gate(tmp_path, fits, probe, floors)
    assert code == 1
    assert _failed(report) == ['determinism'], report
    assert ('## FAIL: determinism\n\n- not measured (no readable '
            '--determinism probe file)') in report


@pytest.mark.parametrize('case', FAILURES)
def test_one_failure_per_check_class(case, tmp_path, suite):
    check, mutate = FAILURES[case]
    mutate(*suite)
    code, report = _gate(tmp_path, *suite)
    assert code == 1
    assert report.startswith('# S0 smoke gate: FAIL')
    assert _failed(report) == [check], report
    assert f'## FAIL: {check}' in report


# --- S1 pilot gate (#75) ------------------------------------------------------

DS0 = 20261700                     # data seed of pilot seed 0 (any value)
FLOOR, PERSISTENCE = 10.0, 5.0     # test floor lines of every pilot draw
LV_ARMS = ('cfc_lrc_ncp', 'gru_ncp', 'gru_dense3', 'cfc_ncp', 'lrc_ncp')
GATES = (('fits', 'all'), ('floors', 'rollouts'), ('G1', 'lv'),
         ('G1', 'duffing'), ('G1', 'multitimescale_lib'), ('G0-E', 'lv'),
         ('G0-E', 'duffing'), ('G0-E', 'multitimescale_lib'), ('G0-E', 'pa'),
         ('G2', 'pa'), ('G3', 'lv'), ('GK', 'co'))


def _pilot_fit(key, arm, seed, tag):
    """A pilot fit JSON with the gate-relevant fields of the real schema;
    every gate passes, G0-E with flat curves (epochs unchanged). The
    consensus pilot's best lr (1e-3) is the only one that passes GK."""
    cell, wiring = arm.rsplit('_', 1) if key != 'co' else ('cfc_lrc', 'ncp')
    rec = {'cell': cell, 'wiring': wiring, 'seed': seed, 'lr': 1e-3,
           'training_diverged': False, 'wallclock_s': 100.0 + seed}
    if key == 'co':
        lr = float(tag.split()[1])
        best = lr == 1e-3
        rec.update(lr=lr, val_macro_f1_mean=0.6 if best else 0.5,
                   best_val_epoch=40, fault_counts=[0, 1, 2],
                   val_macro_f1_grid_curve=[[0.65 if best else 0.4, 0.6, 0.5]]
                   * 40,
                   test_macro_f1_by_fault=[0.64, 0.6, 0.52] if best
                   else [0.3, 0.4, 0.5],
                   vote_lambda_final=0.4 if best else 0.99, corruption_seed=0,
                   config={'effective': {'task': 'committee'}})
    elif key == 'pa':
        rec.update(val_macro_f1=0.6, test_macro_f1=0.63,
                   val_macro_f1_curve=[0.6] * 50,
                   config={'effective': {'task': 'person_activity'}})
    else:
        rec.update(system=key, data_seed=DS0 + seed,
                   rollout_metrics_test={'vpt_multi_median': 20.0,
                                         'rate_unbounded': 0.0},
                   val_loss_curve=[1.0] * 300,
                   config={'effective': {
                       'task': 'lotka_volterra', 'data_seed': DS0 + seed,
                       'dt_scale': 1 if tag else 12.8}})
    return rec


@pytest.fixture()
def pilot():
    """fits {(task key, arm, seed, tag): record} of the six pilot YAMLs and
    the floors.json content of the rollout draws."""
    fits = {fid: _pilot_fit(*fid) for fid in pilot_fits(PILOT)}
    floors = {'floors': [{'system': s, 'data_seed': DS0 + i,
                          'val': {'floor': FLOOR},
                          'test': {'floor': FLOOR, 'persistence': {
                              'vpt_multi_median': PERSISTENCE}}}
                         for s in ROLLOUT for i in range(3)]}
    return fits, floors


def _s1(tmp_path, fits, floors):
    """One run folder per pilot YAML (finals flat in results/, the consensus
    pilot under results/hpo/<arm>/lr*_clipoff_seed100/), run the S1 gate;
    (exit, report, {(gate, task): result})."""
    runs = set()
    for (key, arm, seed, tag), rec in fits.items():
        run = tmp_path / f"pilot_{key}{'_dt1' if tag and key != 'co' else ''}"
        d = run / 'results'
        if key == 'co':
            d = d / 'hpo' / arm / f"lr{rec['lr']:g}_clipoff_seed{seed}"
        d.mkdir(parents=True, exist_ok=True)
        (d / f'{arm}_seed{seed}.json').write_text(json.dumps(rec))
        runs.add(str(run))
    (tmp_path / 'floors.json').write_text(json.dumps(floors))
    out = tmp_path / 'S1.md'
    code = main(['--stage', 's1', '--runs', *sorted(runs), '--floors',
                 str(tmp_path / 'floors.json'), '--out', str(out)])
    report = out.read_text()
    rows = {tuple(c.strip() for c in row.split('|')[1:3]):
            row.split('|')[3].strip()
            for row in report.splitlines()
            if row.startswith('| ') and not row.startswith('| Gate')}
    return code, report, rows


def _only(rows, status, *gates):
    """Every row PASS except gates, which have status."""
    return rows == {g: status if g in gates else 'PASS' for g in GATES}


def test_s1_passing_fixture(tmp_path, pilot):
    assert len(pilot[0]) == 61
    code, report, rows = _s1(tmp_path, *pilot)
    assert code == 0, report
    assert report.startswith('# S1 pilot gate: PASS')
    assert _only(rows, 'PASS'), rows
    assert '61/61 present' in report
    assert 'epochs unchanged (0 of 5 arms qualify)' in report
    assert '- lrc_ncp seed 2: test macro-F1 0.630, wallclock_s 102.0' in report
    assert ('- ncp_cons: best lr 0.001 (val_macro_f1_mean 0.600), val '
            'macro-F1 at m=0 0.650') in report
    assert ('fault schedule: the committee JSON stores no per-batch '
            'fault-count sequence or hash') in report


def _set(fits, key, arm, seeds, path, value):
    """Set a (dotted) field on the fits of one arm and the given seeds."""
    for (k, a, s, _), rec in fits.items():
        if (k, a) == (key, arm) and s in seeds:
            *head, last = path.split('.')
            d = rec
            for h in head:
                d = d[h]
            d[last] = value


def _curve(fits, key, arms, seeds, curve):
    for arm in arms:
        _set(fits, key, arm, seeds,
             'val_macro_f1_curve' if key == 'pa' else 'val_loss_curve', curve)


LVK = 'periodic_predator_prey'
S1_FAILURES = {
    # (mutation, failing row, consequence text in the report)
    'missing fit': (lambda f, fl: f.pop((DUF, 'gru_dense3', 1, '')),
                    ('fits', 'all'), 'complete the pilot runs'),
    'no floor line': (lambda f, fl: fl['floors'].pop(4),
                      ('floors', 'rollouts'), 'write the floor lines'),
    'G1 lv at floor + 2 (strict)': (
        lambda f, fl: _set(f, LVK, 'cfc_lrc_ncp', (0, 1),
                           'rollout_metrics_test.vpt_multi_median',
                           FLOOR + 2), ('G1', 'lv'),
        'floor-bound: H-b/H-c contrasts on this task are floor-bound, the '
        'H-c conjunction runs over the tasks that pass'),
    'G1 duffing non-finite VPT': (
        lambda f, fl: _set(f, DUF, 'cfc_lrc_ncp', (1, 2),
                           'rollout_metrics_test.vpt_multi_median', math.nan),
        ('G1', 'duffing'), 'floor-bound'),
    'G1 mt below 2 x persistence': (
        lambda f, fl: _set(f, MT, 'cfc_ncp', (0, 2),
                           'rollout_metrics_test.vpt_multi_median', 9.0),
        ('G1', 'multitimescale_lib'), 'spread 0.25 or 256 steps, repeat once'),
    'G1 mt one unbounded fit': (
        lambda f, fl: _set(f, MT, 'cfc_ncp', (1,),
                           'rollout_metrics_test.rate_unbounded', 0.0625),
        ('G1', 'multitimescale_lib'), 'spread 0.25 or 256 steps, repeat once'),
    'G0-E no curve': (
        lambda f, fl: _set(f, 'pa', 'lrc_ncp', (0,), 'val_macro_f1_curve',
                           None), ('G0-E', 'pa'), 'no validation curve'),
    'G2 median below 0.60': (
        lambda f, fl: _set(f, 'pa', 'cfc_lrc_ncp', (0, 1), 'test_macro_f1',
                           0.59), ('G2', 'pa'), 'stop, find the cause before S2'),
    'G2 lrc/ncp diverged': (
        lambda f, fl: _set(f, 'pa', 'lrc_ncp', (1,), 'training_diverged',
                           True), ('G2', 'pa'), 'stop, find the cause'),
    'G2 lrc/ncp NaN macro-F1': (
        lambda f, fl: _set(f, 'pa', 'lrc_ncp', (2,), 'test_macro_f1',
                           math.nan), ('G2', 'pa'), 'stop, find the cause'),
    'GK below 0.60 at m=0': (
        lambda f, fl: _set(f, 'co', 'ncp_lam0', (100,),
                           'val_macro_f1_grid_curve', [[0.55, 0.5, 0.4]] * 40),
        ('GK', 'co'), 'architecture check (width, backbone) before the '
                      'consensus continues'),
    'GK not monotone over the fault grid': (
        lambda f, fl: _set(f, 'co', 'avg_in', (100,),
                           'test_macro_f1_by_fault', [0.6, 0.62, 0.5]),
        ('GK', 'co'), 'architecture check'),
    'GK lambda at the edge': (
        lambda f, fl: _set(f, 'co', 'ncp_cons', (100,), 'vote_lambda_final',
                           0.96), ('GK', 'co'), 'vote_init check (one fit)'),
}


@pytest.mark.parametrize('case', S1_FAILURES)
def test_s1_one_failure_per_gate_branch(case, tmp_path, pilot):
    mutate, row, consequence = S1_FAILURES[case]
    mutate(*pilot)
    code, report, rows = _s1(tmp_path, *pilot)
    assert code == 1
    assert report.startswith('# S1 pilot gate: FAIL')
    assert _only(rows, 'FAIL', row), rows
    section = report.split(f'## FAIL: {row[0]} {row[1]}\n')[1].split('## ')[0]
    assert f'- Consequence (F1 section 7): ' in section
    assert consequence in section


def test_s1_one_seed_below_the_floor_passes(tmp_path, pilot):
    """>= 2 of 3 seeds: one seed at the floor leaves G1 at PASS."""
    _set(pilot[0], LVK, 'cfc_lrc_ncp', (2,),
         'rollout_metrics_test.vpt_multi_median', 3.0)
    code, report, rows = _s1(tmp_path, *pilot)
    assert code == 0 and _only(rows, 'PASS'), report
    assert ('- seed 2 (data seed 20261702): VPT 3 vs 12 (floor 10 + 2) -> '
            'no') in report


@pytest.mark.parametrize('key, arms, curve, decision', [
    (LVK, LV_ARMS[:2], [1.0] * 270 + [0.9] * 30, 'epochs x1.5 (300 to 450)'),
    (DUF, LV_ARMS[3:], [1.0] * 199 + [2.0] + [1.0] * 70 + [1.875] * 30,
     'epochs x1.5 (300 to 450)'),
    ('pa', ('cfc_lrc_ncp', 'lrc_ncp'), [0.5] * 45 + [0.6] * 5,
     'epochs x2 (50 to 100)')])
def test_s1_epoch_gate_decision(key, arms, curve, decision, tmp_path, pilot):
    """Two pilot arms with two qualifying seeds each change the task's epochs
    (a DECISION, exit 0); one arm with three does not."""
    _curve(pilot[0], key, arms, (0, 1), curve)
    code, report, rows = _s1(tmp_path, *pilot)
    task = 'lv' if key == LVK else key
    assert code == 0, report
    assert _only(rows, 'DECISION', ('G0-E', task)), rows
    assert f'- DECISION: {decision}: {", ".join(arms)} qualify' in report

    _curve(pilot[0], key, arms[1:], (0, 1), [1.0] * len(curve))
    _curve(pilot[0], key, arms[:1], (2,), curve)
    code, report, rows = _s1(tmp_path / 'one_arm', *pilot)
    assert code == 0 and _only(rows, 'PASS'), rows
    assert 'epochs unchanged (1 of' in report


def test_s1_dt_scale_control_is_descriptive(tmp_path, pilot):
    """G3 never fails: the control's median VPT next to the main arm. The
    control fits stay out of G1 (they would lift it to PASS) and G0-E (they
    would make cfc_lrc/ncp the second qualifying arm)."""
    fits = pilot[0]
    _set(fits, LVK, 'cfc_lrc_ncp', (0, 1),
         'rollout_metrics_test.vpt_multi_median', 3.0)
    _curve(fits, LVK, ('gru_ncp',), (0, 1), [1.0] * 270 + [0.9] * 30)
    for (k, a, s, tag), rec in fits.items():
        if tag == 'dt_scale 1':
            rec['rollout_metrics_test']['vpt_multi_median'] = 30.0 + s
            rec['val_loss_curve'] = [1.0] * 270 + [0.5] * 30
    code, report, rows = _s1(tmp_path, fits, pilot[1])
    assert code == 1 and _only(rows, 'FAIL', ('G1', 'lv')), rows
    assert '| G1 | lv | FAIL |' in report and '1/3 seeds pass' in report
    assert 'epochs unchanged (1 of 5 arms qualify)' in report
    assert ('median VPT 31 (dt_scale 1, 3 fits) vs 3 (main, 3 fits)'
            in report)
