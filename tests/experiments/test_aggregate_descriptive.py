# tests/experiments/test_aggregate_descriptive.py
"""The descriptive F2 blocks of experiments/aggregate_results.py (#109), on
synthetic records with invented values only."""
import json
import os
import re

import numpy as np
import pandas as pd
import pytest
import yaml

from experiments.aggregate_results import (B5_EXIT_THRESHOLD, C7_CAVEAT,
                                           DESC_COLUMNS, DESCRIPTIVE_BLOCKS,
                                           DIRECTION_TEXT, _block_labels,
                                           b5_seed_reading, b5_verdict,
                                           descriptive_blocks, direction5,
                                           fit_floor, load_descriptive_spec,
                                           load_sequence_runs, main)

F1, F2 = 'F1-2026-10-26', 'F2-2026-11-05'
DS = 4242
EPOCHS = {'periodic_predator_prey': 300, 'duffing': 450, 'pa': 50}
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))))
TEMPLATE = os.path.join(REPO, 'configs/aggregate/descriptive_blocks.yaml')
F1_FIXTURE = os.path.join(REPO, 'tests/experiments/data/f1_aggregate')
GOOD = {f'r{i}': [[6, 1], [1, 5]] for i in range(3)}   # one-class 0.35


def _dump(path, rec):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(rec, f)


def _lv_rec(system, cell, wiring, seed, vpt, pv, epochs, bve=1,
            curve=(1.0, 0.4), vpt8=None, unbounded=False, anchors=True):
    m = {'vpt_multi_median': vpt, 'vpt_anchor_8': vpt if vpt8 is None
         else vpt8, 'rate_unbounded': 1.0 if unbounded else 0.0,
         'non_finite': False, 'anchor_horizon': 16, 'anchor_theta': 0.5}
    if anchors:
        m['anchors'] = [8]
    eff = {'task': 'lotka_volterra', 'system': system, 't_span': [0.0, 10.0],
           'ic_spread': 0.4, 'n_trajectories': 10, 'seq_len': 32,
           'data_seed': DS, 'split': [6, 2, 2], 'param_budget': 4000,
           'ode_unfolds': 6, 'dt_scale': 12.8, 'forcing': 'mixed',
           'epochs': epochs, 'protocol_version': pv}
    return {'system': system, 'cell': cell, 'wiring': wiring, 'seed': seed,
            'param_budget': 4000, 'budget_deviation': 0.01,
            'budget_above_tolerance': False, 'epochs': epochs, 'lr': 1e-3,
            'clipnorm': None, 'params_effective': 3960, 'dt_scale': 12.8,
            'forcing': 'mixed', 'best_val_epoch': bve,
            'val_loss_curve': list(curve), 'training_diverged': False,
            'rollout_metrics_test': m, 'config': {'effective': eff}}


def _pa_rec(cell, wiring, seed, f1, pv, epochs, bve=1, curve=None,
            diverged=False, conf=GOOD):
    eff = {'task': 'person_activity', 'param_budget': 30000,
           'epochs': epochs, 'split_seed': 0, 'ode_unfolds': 6,
           'protocol_version': pv}
    return {'cell': cell, 'wiring': wiring, 'seed': seed, 'epochs': epochs,
            'param_budget': 30000, 'budget_deviation': 0.02,
            'budget_above_tolerance': False, 'lr': 3e-3, 'clipnorm': 1.0,
            'params_effective': 29500, 'dt_scale': 3.7, 'best_val_epoch': bve,
            'test_macro_f1': f1, 'test_confusion_by_recording': conf,
            'val_macro_f1_curve': list(curve if curve is not None
                                       else [0.2] * 12),
            'training_diverged': diverged, 'config': {'effective': eff}}


def _write(folder, task, label, seed, value, pv, **kw):
    cell, rest = label.split('/')
    wiring = rest.split(' ')[0]
    ep = re.search(r'epochs=(\d+)', label)
    epochs = int(ep.group(1)) if ep else EPOCHS[task]
    if task == 'pa':
        rec = _pa_rec(cell, wiring, seed, value, pv, epochs, **kw)
    else:
        rec = _lv_rec(task, cell, wiring, seed, value, pv, epochs, **kw)
    _dump(os.path.join(folder, f'{cell}_{wiring}_seed{seed}.json'), rec)
    return rec


_FLOORS = {}


def floor_of(system):
    """The floor of the synthetic data draw of a rollout system."""
    if system not in _FLOORS:
        _FLOORS[system] = fit_floor(_lv_rec(system, 'lrc', 'ncp', 0, 1.0, F2,
                                            300), {})[1]
    return _FLOORS[system]


def _world(tmp_path, values=None, opts=None, empty=()):
    """A spec (template protocol versions, one folder per block and arm)
    where every arm holds the block's seeds at 20 VPT (0.6 macro-F1); B2
    lrc/ncp splits into F1 seeds 0-4 and F2 seeds 5-9. values: {(block,
    label): per-seed values}; opts: {(block, label): {seed: writer kw}};
    empty: (block, label) arms written with paths []."""
    values, opts = values or {}, opts or {}
    with open(TEMPLATE, encoding='utf-8') as f:
        spec = yaml.safe_load(f)
    for block, arms in spec.items():
        task = DESCRIPTIVE_BLOCKS[block]['task']
        seeds = list(DESCRIPTIVE_BLOCKS[block]['seeds'])
        for i, arm in enumerate(arms):
            label = arm['label']
            if (block, label) in empty:
                arm['paths'] = []
                continue
            pvs = arm['protocol_version']
            pvs = pvs if isinstance(pvs, list) else [pvs]
            vals = values.get((block, label),
                              [0.6 if task == 'pa' else 20.0] * len(seeds))
            arm['paths'] = []
            for j, pv in enumerate(pvs):
                folder = str(tmp_path / block / str(i) / pv)
                arm['paths'].append(folder)
                os.makedirs(folder, exist_ok=True)
                for s, v in zip(seeds, vals):
                    if len(pvs) == 2 and (s < 5) != (j == 0):
                        continue        # F1 seeds 0-4, F2 seeds 5-9
                    _write(folder, task, label, s, v, pv,
                           **opts.get((block, label), {}).get(s, {}))
    path = str(tmp_path / 'desc.yaml')
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(spec, f)
    return path


def _rows(spec_path):
    return descriptive_blocks(load_descriptive_spec(spec_path))


def _row(df, block, a, b='*'):
    g = df[(df['block'] == block) & (df['a'] == a)]
    if b != '*':
        g = g[g['b'] == b] if b is not None else g[g['b'].isna()]
    assert len(g) == 1, (block, a, b)
    return g.iloc[0]


# --- table and template ------------------------------------------------------

def test_block_table_matches_protocol_f2_section_5():
    b = DESCRIPTIVE_BLOCKS
    assert {k: (v['task'], list(v['seeds'])) for k, v in b.items()} == {
        'B1': ('duffing', [0, 1, 2]),
        'B2': ('periodic_predator_prey', list(range(10))),
        'B3': ('periodic_predator_prey', [0, 1, 2]),
        'B4': ('periodic_predator_prey', [0, 1, 2]),
        'B5': ('pa', [0, 1, 2]), 'BA-2': ('pa', list(range(5))),
        'C-2': ('duffing', list(range(5))),
        'budget_2x_lv': ('periodic_predator_prey', list(range(10))),
        'budget_2x_duffing': ('duffing', list(range(10)))}
    # delta = treatment - reference; BA-2 cncp - cncp_random; C-2 signs
    assert [(p['a'], p['b']) for p in b['BA-2']['pairs']] == [
        ('cfc_lrc/cncp', 'cfc_lrc/cncp_random')]
    assert [(p['a'], p['b']) for p in b['C-2']['pairs']] == [
        ('cfc_lrc/ncp', 'cfc_lrc_eps1/ncp'), ('cfc_lrc_eps1/ncp', 'cfc/ncp')]
    assert [(p['a'], p['b']) for p in b['B3']['pairs']] == [
        (f'{a} [epochs=900]', a) for a in ('lrc/ncp', 'lrc/dense',
                                           'lrc_init/ncp', 'lrc_cal_lv/ncp')]
    assert [(p['a'], p['b'], p['a_1x'], p['b_1x'])
            for p in b['budget_2x_duffing']['pairs']] == [
        ('cfc_lrc/ncp [epochs=900]', 'gru/ncp [epochs=900]', 'cfc_lrc/ncp',
         'gru/ncp'),
        ('cfc_lrc/ncp [epochs=900]', 'cfc_lrc/dense3 [epochs=900]',
         'cfc_lrc/ncp', 'cfc_lrc/dense3'),
        ('cfc_lrc/cncp [epochs=900]', 'cfc_lrc/dense3 [epochs=900]',
         'cfc_lrc/cncp', 'cfc_lrc/dense3')]
    assert B5_EXIT_THRESHOLD == pytest.approx(0.0851)


def test_template_lists_every_block_arm_with_placeholders():
    with open(TEMPLATE, encoding='utf-8') as f:
        t = yaml.safe_load(f)
    assert set(t) == set(DESCRIPTIVE_BLOCKS)
    for block, arms in t.items():
        assert [a['label'] for a in arms] == _block_labels(block)
        assert all(p.startswith('<') for a in arms for p in a['paths'])
    assert [a for a in t['B2'] if a['label'] == 'lrc/ncp'][0][
        'protocol_version'] == [F1, F2]
    with pytest.raises(SystemExit, match='no such path'):
        load_descriptive_spec(TEMPLATE)


def test_spec_rejects_missing_blocks_and_wrong_arms(tmp_path):
    with open(_world(tmp_path), encoding='utf-8') as f:
        good = yaml.safe_load(f)

    def check(mutate, match):
        s = json.loads(json.dumps(good))
        mutate(s)
        path = str(tmp_path / 'bad.yaml')
        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(s, f)
        with pytest.raises(SystemExit, match=match):
            load_descriptive_spec(path)

    check(lambda s: s.pop('B5'), r"missing \['B5'\]")
    check(lambda s: s['B1'].pop(), 'B1 needs exactly the arms')
    check(lambda s: s['B1'][0].__setitem__('label', 'lrc/dense'),
          'B1 needs exactly the arms')
    check(lambda s: s['B1'][0].__setitem__('seeds', [0]), 'keys')
    check(lambda s: s['B1'][0]['paths'].append(str(tmp_path / 'x' / '*')),
          'pattern')


# --- three-seed rule ---------------------------------------------------------

@pytest.mark.parametrize('offsets, k, summary', [
    ((3.0, 2.5, 4.0), 3, 'stable'),
    ((3.0, 2.0, 4.0), 2, 'mixed'),           # seed 1 exactly at floor + 2
    ((3.0, 1.0, 2.0), 1, 'not supported')])
def test_b1_floor_plus_two_is_strict(tmp_path, offsets, k, summary):
    fl = floor_of('duffing')
    df = _rows(_world(tmp_path, values={
        ('B1', 'lrc_init/ncp'): [fl + o for o in offsets],
        ('B1', 'lrc/ncp'): [fl] * 3}, opts={
        ('B1', 'lrc_init/ncp'): {0: {'bve': 440}, 1: {'bve': 427},
                                 2: {'bve': 428}}}))
    r = _row(df, 'B1', 'lrc_init/ncp')
    assert (r['n_satisfied'], r['summary']) == (k, summary)
    assert r['b'] == 'lrc/ncp' and pd.isna(r['status'])
    assert r['median_diff'] == pytest.approx(sorted(offsets)[1])
    assert r['n'] == 3 and r['n_positive'] == 3
    assert r['n_flagged_a'] == 2 and r['n_flagged_b'] == 0   # >= 427.5
    assert r['best_val_epoch_a'] == '0: 440; 1: 427; 2: 428'


def test_b3_and_b4_compare_strictly_against_the_same_seed(tmp_path):
    df = _rows(_world(tmp_path, values={
        ('B3', 'lrc/ncp [epochs=900]'): [21.0, 20.0, 25.0],
        ('B3', 'lrc/ncp'): [20.0, 20.0, 30.0],
        ('B4', 'lrc_init_tau/ncp'): [21.0, 22.0, 23.0]}, opts={
        ('B3', 'lrc/ncp [epochs=900]'): {0: {'vpt8': 30.0}}}))
    r = _row(df, 'B3', 'lrc/ncp [epochs=900]')
    assert r['per_seed'] == '0: yes; 1: no; 2: no'     # tie and lower fail
    assert (r['n_satisfied'], r['summary']) == (1, 'not supported')
    assert r['vpt8_n_positive'] == 1 and r['vpt8_median_diff'] == 0.0
    assert r['note'].startswith('Epochs and learning-rate schedule change '
                                'together')
    assert r['origin'] == (f'lrc/ncp [epochs=900]: {F2} seeds 0-2; '
                           f'lrc/ncp: {F1} seeds 0-2')
    r = _row(df, 'B4', 'lrc_init_tau/ncp')
    assert (r['summary'], r['b']) == ('stable', 'lrc_init/ncp')
    assert pd.isna(r['caveat'])
    assert _row(df, 'B4', 'lrc_tau/ncp')['caveat'] == (
        'The initial plateau of lrc/ncp can mask a retention effect, and the '
        'effective gain of the base arm is shifted relative to the init arm; '
        'a result on lrc/ncp is therefore not attributed to the time-scale '
        'initialisation alone.') == C7_CAVEAT


# --- B2 ----------------------------------------------------------------------

def test_b2_interval_origin_floor_flag_and_escape_epoch(tmp_path):
    up = [21.0 + s for s in range(10)]
    df = _rows(_world(tmp_path, values={
        ('B2', 'lrc_init/ncp'): up,
        ('B2', 'lrc_cal_lv/ncp'): [20.0 + (-1) ** s for s in range(10)]},
        opts={('B2', 'lrc_init/ncp'): {
            0: {'curve': [0.9, 0.6, 0.45, 0.3], 'bve': 285},
            1: {'curve': [0.9, 0.8, 0.7], 'bve': 284}}}))
    r = _row(df, 'B2', 'lrc_init/ncp', 'lrc/ncp')
    assert (r['status'], r['n'], r['n_positive']) == ('descriptive', 10, 10)
    assert r['interval_excludes_0'] is True or r['interval_excludes_0'] == 1
    assert r['origin'] == (f'lrc_init/ncp: {F2} seeds 0-9; lrc/ncp: {F1} '
                           f'seeds 0-4, {F2} seeds 5-9')
    assert r['ci_units'] == 'seed'
    assert r['escape_epoch_a'].startswith('0: 3; 1: none; 2: 2')
    assert r['n_flagged_a'] == 1                       # 285 = 0.95 x 300
    assert r['seeds_below_floor_a'] == 'none' and not r['not_learned_a']
    assert r['vpt8_median_a'] == pytest.approx(np.median(up))
    r = _row(df, 'B2', 'lrc_cal_lv/ncp', 'cfc_lrc/ncp')
    assert r['n_positive'] == 5 and not r['interval_excludes_0']
    assert r['median_a_lo'] <= r['median_a'] <= r['median_a_hi']
    assert not any('equival' in c or 'inferior' in c for c in df.columns)


def test_b2_floor_marking_by_the_seed_rule(tmp_path):
    vals = [20.0] * 10
    vals[2] = vals[7] = vals[8] = -1.0
    df = _rows(_world(tmp_path, values={('B2', 'lrc_init/ncp'): vals}))
    r = _row(df, 'B2', 'lrc_init/ncp', 'lrc/ncp')
    assert r['seeds_below_floor_a'] == '2, 7-8' and r['not_learned_a']


# --- five-seed direction rule ------------------------------------------------

@pytest.mark.parametrize('d, expect', [
    ([1, 2, 3, 4, 5], 'delta > 0'), ([-1, -2, -3, -4, -5], 'delta < 0'),
    ([1, 2, 3, 4, -5], 'no direction'), ([1, 2, 0, 4, 5], 'no direction'),
    ([1, 2, 3, 4], 'no direction')])
def test_direction_rule(d, expect):
    assert direction5(d, 5) == expect


def test_ba2_and_c2_rows(tmp_path):
    df = _rows(_world(tmp_path, values={
        ('BA-2', 'cfc_lrc/cncp'): [0.61, 0.62, 0.63, 0.64, 0.65],
        ('C-2', 'cfc_lrc/ncp'): [21.0, 22.0, 23.0, 24.0, 19.0],
        ('C-2', 'cfc/ncp'): [21.0, 22.0, 23.0, 24.0, 25.0]}))
    r = _row(df, 'BA-2', 'cfc_lrc/cncp')
    assert (r['status'], r['direction'], r['text']) == (
        'undecidable', 'delta > 0', DIRECTION_TEXT)
    assert r['ci_units'] == 'seed x recording' and r['n_positive'] == 5
    r1 = _row(df, 'C-2', 'cfc_lrc/ncp')
    assert (r1['b'], r1['n_positive'], r1['direction']) == (
        'cfc_lrc_eps1/ncp', 4, 'no direction')
    r2 = _row(df, 'C-2', 'cfc_lrc_eps1/ncp')
    assert (r2['b'], r2['direction'], r2['status']) == (
        'cfc/ncp', 'delta < 0', 'undecidable')
    assert 'cfc_lrc/ncp config' in r1['config_note']
    assert 'own tuned config' in r2['config_note']


# --- budget pairs ------------------------------------------------------------

def test_budget_direction_and_comparator_flag(tmp_path):
    up = [21.0 + s for s in range(10)]
    down = [19.0 - s for s in range(10)]
    df = _rows(_world(tmp_path, values={
        ('budget_2x_lv', 'cfc_lrc/ncp [epochs=600]'): up,
        ('budget_2x_lv', 'cfc_lrc/ncp'): up,
        ('budget_2x_lv', 'cfc_lrc/cncp [epochs=600]'): down,
        ('budget_2x_lv', 'cfc_lrc/cncp'): up}, opts={
        ('budget_2x_lv', 'gru/ncp [epochs=600]'): {3: {'bve': 570}}}))
    r = _row(df, 'budget_2x_lv', 'cfc_lrc/ncp [epochs=600]',
             'gru/ncp [epochs=600]')
    assert (r['direction'], r['median_sign'], r['majority']) == (
        'unchanged', 1.0, 'delta > 0')
    assert (r['n_positive_1x'], r['n_1x'], r['median_sign_1x']) == (10, 10,
                                                                     1.0)
    assert r['n_flagged_b'] == 1 and r['comparator_flagged']
    assert r['status'] == 'descriptive'
    r = _row(df, 'budget_2x_lv', 'cfc_lrc/cncp [epochs=600]')
    assert (r['direction'], r['majority'], r['majority_1x']) == (
        'changed', 'delta < 0', 'delta > 0')
    assert not r['comparator_flagged']
    assert r['origin'].startswith(f'cfc_lrc/cncp [epochs=600]: {F2} seeds '
                                  '0-9; cfc_lrc/dense3 [epochs=600]')


# --- B5 ----------------------------------------------------------------------

T = B5_EXIT_THRESHOLD


def _curve(top, bad=False):
    c = [0.07] * 12
    c[4] = top
    if bad:
        c[2] = float('nan')
    return c


def test_b5_seed_reading_relative_to_the_threshold():
    assert b5_seed_reading(_curve(T + 0.001), False) == 'satisfied'
    assert b5_seed_reading(_curve(T), False) == 'not satisfied'
    assert b5_seed_reading(_curve(T - 0.001), False) == 'not satisfied'
    assert b5_seed_reading(_curve(T + 0.001, bad=True), False) == \
        'not readable'
    assert b5_seed_reading(_curve(T + 0.001), True) == 'not readable'
    late = [0.07] * 10 + [0.5]          # exits after epoch 10 only
    assert b5_seed_reading(late, False) == 'not satisfied'


@pytest.mark.parametrize('readings, verdict', [
    ('SSS', 'drive parametrisation discriminates'),
    ('SSU', 'drive parametrisation discriminates (readable seeds 0-1 only)'),
    ('NNN', 'init gain not sufficient'),
    ('UNN', 'init gain not sufficient (readable seeds 1-2 only)'),
    ('SNS', 'mixed, no statement'),
    ('SUU', 'not readable'), ('UUU', 'not readable')])
def test_b5_verdict(readings, verdict):
    word = {'S': 'satisfied', 'N': 'not satisfied', 'U': 'not readable'}
    assert b5_verdict({s: word[c] for s, c in enumerate(readings)}) == verdict


def test_b5_row_reads_the_curves_and_the_diverged_flag(tmp_path):
    df = _rows(_world(tmp_path, opts={('B5', 'ltc_minel/ncp'): {
        0: {'curve': _curve(T + 0.001)}, 1: {'curve': _curve(T + 0.001)},
        2: {'curve': _curve(T + 0.001), 'diverged': True}}}))
    r = _row(df, 'B5', 'ltc_minel/ncp', None)
    assert r['per_seed'] == '0: satisfied; 1: satisfied; 2: not readable'
    assert r['summary'] == ('drive parametrisation discriminates (readable '
                            'seeds 0-1 only)')
    assert pd.isna(r['status']) and pd.isna(r['median_diff'])


# --- not run, errors, columns ------------------------------------------------

def test_arm_without_records_is_not_run_and_leaves_the_others(tmp_path):
    fl = floor_of('duffing')
    df = _rows(_world(tmp_path, values={
        ('B1', 'lrc_init/ncp'): [fl + 3] * 3}, empty={
        ('B1', 'lrc_cal_duffing/ncp'), ('BA-2', 'cfc_lrc/cncp_random')}))
    r = _row(df, 'B1', 'lrc_cal_duffing/ncp')
    assert (r['status'], r['not_run']) == ('not run', 'lrc_cal_duffing/ncp')
    assert pd.isna(r['summary'])
    assert _row(df, 'B1', 'lrc_init/ncp')['summary'] == 'stable'
    r = _row(df, 'BA-2', 'cfc_lrc/cncp')
    assert (r['status'], r['not_run']) == ('not run', 'cfc_lrc/cncp_random')


def test_missing_floor_of_an_existing_fit_is_an_error(tmp_path):
    with pytest.raises(SystemExit, match=r'B1\.0 \(lrc_init/ncp\).*no floor'):
        _rows(_world(tmp_path, opts={('B1', 'lrc_init/ncp'): {
            1: {'anchors': False}}}))


def test_f2_record_in_an_f1_path_list_is_rejected(tmp_path):
    s = load_descriptive_spec(_world(tmp_path))
    f1_dir = [a for a in s['B1'] if a['label'] == 'lrc/ncp'][0]['paths'][0]
    _write(os.path.join(f1_dir, 'rerun'), 'duffing', 'lrc/ncp', 0, 20.0, F2)
    with pytest.raises(SystemExit, match='rerun.*protocol_version'):
        descriptive_blocks(s)


def test_block_csv_has_no_test_columns(tmp_path):
    df = _rows(_world(tmp_path))
    assert list(df.columns) == DESC_COLUMNS
    for c in ('p_raw', 'p_adj', 'significant', 'method', 'verdict',
              'p_a_gt_b', 'p_a_gt_b_lo', 'p_a_gt_b_hi'):
        assert c not in df.columns
    assert len(df) == sum(len(b['pairs']) for b in DESCRIPTIVE_BLOCKS.values())


def test_main_writes_the_block_csv_and_summary_section(tmp_path):
    spec = _world(tmp_path)
    out = str(tmp_path / 'out')
    assert main(['--runs', str(tmp_path / 'none'), '--out', out,
                 '--rollout-runs', os.path.join(F1_FIXTURE, 'rollouts'),
                 '--contrasts', '--descriptive', spec]) == 0
    df = pd.read_csv(os.path.join(out, 'descriptive_blocks.csv'))
    assert list(df.columns) == DESC_COLUMNS
    seq = pd.read_csv(os.path.join(out, 'sequence_runs.csv'))
    assert list(seq.columns[-2:]) == ['val_loss_curve', 'training_diverged']
    with open(os.path.join(out, 'summary.md'), encoding='utf-8') as f:
        text = f.read()
    assert '## Descriptive F2 blocks' in text
    for block in DESCRIPTIVE_BLOCKS:
        assert f'### {block}\n' in text
    assert C7_CAVEAT in text


# --- default behaviour unchanged ---------------------------------------------


@pytest.mark.parametrize('sub, task, key', [('rollouts', 'lv',
                                             'val_loss_curve'),
                                            ('pa', 'pa',
                                             'val_macro_f1_curve')])
def test_curves_only_append_columns(sub, task, key):
    base = load_sequence_runs(os.path.join(F1_FIXTURE, sub), task)
    assert key not in base and 'training_diverged' not in base
    full = load_sequence_runs(os.path.join(F1_FIXTURE, sub), task,
                              curves=True)
    assert list(full.columns) == list(base.columns) + [key,
                                                       'training_diverged']
    pd.testing.assert_frame_equal(
        full[base.columns].drop(columns=['confusion'], errors='ignore'),
        base.drop(columns=['confusion'], errors='ignore'))
