# tests/experiments/test_aggregate_f3_descriptive.py
"""The F3 descriptive path of experiments/aggregate_results.py (#152): block
A arms, budget arm and script arm, on synthetic records only."""
import json
import os

import pandas as pd
import pytest
import yaml

from experiments.aggregate_results import (B5_EXIT_THRESHOLD,
                                           F3_DESC_ARM_COLUMNS,
                                           F3_DESC_FIT_COLUMNS,
                                           F3_DESC_READING_COLUMNS,
                                           F3_DESCRIPTIVE_ARMS, F3_TESTS,
                                           _desc_arm_row, _f3_arm, _f3_test,
                                           arm_label, f3_descriptive,
                                           load_f3_descriptive_spec, main)
from tests.experiments.test_aggregate_f3 import (F3, GOOD, SUMMARY, _cell,
                                                 _dump, _run_folder)
from tests.experiments.test_aggregate_f3 import _world as _f3_world

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))))
BLOCK_A = ['a_lrc', 'a_lrc_legacy', 'a_cal', 'a_cal_legacy', 'a_init',
           'a_init_legacy']
SUFFIX20 = ' (exploratory, 20 seeds, one split)'
SUFFIX5 = ' (exploratory, 5 seeds, one split)'


# --- records and worlds ------------------------------------------------------

def _curve(exit_at, n, low=0.0751, high=0.2):
    """A validation curve of n epochs whose first value above the exit
    threshold is at epoch exit_at (1-based; None: never)."""
    if exit_at is None:
        return [low] * n
    return [low] * (exit_at - 1) + [high] * (n - exit_at + 1)


def _rec(path, aid, seed, last=0.5, best=None, exit_at=5, diverged=False,
         test_curve=None, val_curve=None, pv=F3, name=None, extra=None):
    """One Person Activity record of the arm, the registry config."""
    reg = F3_DESCRIPTIVE_ARMS[aid]
    cell, wiring, build = _cell(reg['label'], reg['lr'])
    e = reg['epochs']
    eff = {'task': 'person_activity', 'param_budget': 30000, 'epochs': e,
           'split_seed': 0, 'ode_unfolds': 6, 'protocol_version': pv,
           'build': build}
    rec = {'cell': cell, 'wiring': wiring, 'seed': seed, 'epochs': e,
           'param_budget': 30000, 'budget_deviation': 0.02,
           'budget_above_tolerance': False, 'lr': reg['lr'],
           'clipnorm': None, 'params_effective': 29500, 'dt_scale': 3.7,
           'best_val_epoch': 10, 'test_macro_f1': last if best is None
           else best, 'test_macro_f1_last': last,
           'test_macro_f1_curve': test_curve or [last] * e,
           'val_macro_f1_curve': val_curve or _curve(exit_at, e),
           'training_diverged': diverged, 'split_fold': None,
           'test_confusion_by_recording': GOOD, 'config': {'effective': eff}}
    rec.update(extra or {})
    _dump(os.path.join(path, name or f'{cell}_{wiring}_seed{seed}.json'), rec)


def _folders(aid):
    """The run folders of an arm: block A by optimizer and seed half (the
    three cells share them), budget and script on their own."""
    if aid not in BLOCK_A:
        return {s: aid for s in F3_DESCRIPTIVE_ARMS[aid]['seeds']}
    opt = 'legacy' if aid.endswith('_legacy') else 'rms'
    return {s: f'{opt}_s{10 if s < 20 else 20}' for s in range(10, 30)}


def _world(tmp_path, opts=None, skip=None, empty=()):
    """Spec path of a world with every seed of every arm (learning success
    0.5, exit at epoch 5). opts: {arm: writer kwargs or {seed: kwargs}};
    skip: {arm: seeds without a record}; empty: arms whose paths are one
    empty folder."""
    opts, skip = opts or {}, skip or {}
    spec = {}
    for aid, reg in F3_DESCRIPTIVE_ARMS.items():
        if aid in empty:
            paths = [str(tmp_path / f'empty_{aid}')]
            os.makedirs(paths[0])
        else:
            folders = _folders(aid)
            o = opts.get(aid, {})
            per_seed = bool(o) and all(isinstance(k, int) for k in o)
            for s in reg['seeds']:
                path = str(tmp_path / folders[s])
                os.makedirs(path, exist_ok=True)
                if s not in skip.get(aid, ()):
                    _rec(path, aid, s, **(o.get(s, {}) if per_seed else o))
            paths = [str(tmp_path / f) for f in sorted(set(folders.values()))]
        spec[aid] = {'label': reg['label'], 'protocol_version': F3,
                     'paths': paths}
    path = str(tmp_path / 'desc.yaml')
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(spec, f)
    return path


def _run(tmp_path, **kw):
    fits, arms, readings = f3_descriptive(load_f3_descriptive_spec(
        _world(tmp_path, **kw)))
    return fits, arms.set_index('arm'), readings.set_index('reading')


# --- registry, spec, template ------------------------------------------------

def test_registry_holds_block_a_budget_and_script_arms():
    assert list(F3_DESCRIPTIVE_ARMS) == [*BLOCK_A, 'budget', 'script']
    for aid in BLOCK_A:
        r = F3_DESCRIPTIVE_ARMS[aid]
        base = {'a_lrc': 'lrc/ncp', 'a_cal': 'lrc_cal/ncp',
                'a_init': 'lrc_init/ncp'}[aid.replace('_legacy', '')]
        assert r == {'label': base + (' [optimizer=rmsprop_legacy]' if
                                      aid.endswith('_legacy') else ''),
                     'lr': 1e-3, 'clipnorm': None, 'epochs': 50,
                     'seeds': range(10, 30), 'folds': None}
    assert F3_DESCRIPTIVE_ARMS['budget'] == {
        'label': 'lrc/ncp [epochs=200]', 'lr': 1e-2, 'clipnorm': None,
        'epochs': 200, 'seeds': range(10, 15), 'folds': None}
    assert F3_DESCRIPTIVE_ARMS['script'] == {
        'label': 'lrc/ncp [epochs=200, optimizer=rmsprop_legacy]',
        'lr': 1e-3, 'clipnorm': None, 'epochs': 200, 'seeds': range(10, 15),
        'folds': None}


def test_registry_labels_are_the_arm_labels_of_their_records(tmp_path):
    for aid, reg in F3_DESCRIPTIVE_ARMS.items():
        _rec(str(tmp_path), aid, 10, name=f'{aid}.json')
        with open(tmp_path / f'{aid}.json', encoding='utf-8') as f:
            assert arm_label(json.load(f)) == reg['label']


def test_template_lists_every_registered_arm_with_placeholders():
    path = os.path.join(REPO, 'configs/aggregate/f3_descriptive.yaml')
    with open(path, encoding='utf-8') as f:
        t = yaml.safe_load(f)
    assert set(t) == set(F3_DESCRIPTIVE_ARMS)
    for aid, reg in F3_DESCRIPTIVE_ARMS.items():
        assert t[aid]['label'] == reg['label']
        assert t[aid]['protocol_version'] == 'F3-<FREEZE_DATE>'
        paths = t[aid]['paths']
        assert len(paths) == (2 if aid in BLOCK_A else 1), aid
        assert all(p.startswith('<') and p.endswith('_RUN_DIR>/results')
                   for p in paths), aid
        if aid in BLOCK_A:
            assert ['S10' in paths[0], 'S20' in paths[1]] == [True, True]
    with pytest.raises(SystemExit, match='no such path'):
        load_f3_descriptive_spec(path)


def test_spec_rules(tmp_path):
    with open(_world(tmp_path), encoding='utf-8') as f:
        good = yaml.safe_load(f)

    def check(mutate, match):
        s = json.loads(json.dumps(good))
        mutate(s)
        path = str(tmp_path / 'bad.yaml')
        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(s, f)
        with pytest.raises(SystemExit, match=match):
            load_f3_descriptive_spec(path)

    check(lambda s: s.pop('script'), r"missing \['script'\], unknown \[\]")
    check(lambda s: s.__setitem__('a_zz', s['a_lrc']), r"unknown \['a_zz'\]")
    check(lambda s: s['budget'].__setitem__('label', 'lrc/ncp'), 'label')
    check(lambda s: s['a_cal'].__setitem__('protocol_version',
                                           'F2-2026-11-05'),
          'not an F3 stamp')
    check(lambda s: s['a_cal'].__setitem__('seeds', [10]), 'keys')
    check(lambda s: s['a_cal']['paths'].__setitem__(0, str(tmp_path / '*')),
          'pattern')
    check(lambda s: s['a_cal'].__setitem__('paths', []), 'non-empty list')


# --- main --------------------------------------------------------------------

def test_main_writes_the_three_tables_and_the_summary(tmp_path):
    out = str(tmp_path / 'out')
    assert main(['--runs', str(tmp_path / 'missing'), '--out', out,
                 '--f3-descriptive', _world(tmp_path)]) == 0
    tables = {}
    for name, cols in (('fits', F3_DESC_FIT_COLUMNS),
                       ('arms', F3_DESC_ARM_COLUMNS),
                       ('readings', F3_DESC_READING_COLUMNS)):
        tables[name] = pd.read_csv(os.path.join(
            out, f'f3_descriptive_{name}.csv'), dtype=str)
        assert list(tables[name].columns) == cols
        # no p-value, nothing of the test path (m, significant)
        assert not [c for c in cols if c.startswith(('p_', 'm', 'signif'))]
    assert len(tables['fits']) == 6 * 20 + 2 * 5
    arms = tables['arms'].set_index('arm')
    assert arms.loc['a_cal', 'success_rate'] == '20/20 [0.83, 1.00]'
    assert arms.loc['budget', 'n_exit_within_50'] == '5'          # Int64
    assert pd.isna(arms.loc['a_cal', 'n_exit_within_50'])
    assert tables['fits']['exit_epoch'].eq('5').all()             # Int64
    assert list(tables['readings']['reading']) == [
        'manipulation_check', 'lrc_init', 'count_t_a1', 'budget', 'script',
        'script_retained']
    with open(os.path.join(out, 'summary.md'), encoding='utf-8') as f:
        text = f.read()
    check = text.index('## F3 block A manipulation check')
    expl = text.index('## F3 descriptive readings (exploratory)')
    assert check < expl
    assert 'passed: 20 of 20' in text[check:expl]
    assert 'manipulation check' not in text[expl:].split('\n- ')[0]
    for s in tables['readings']['sentence'].dropna():
        assert s in text
    assert not [w for w in ('p adj', 'p perm', 'significant', 'Wilcoxon',
                            'McNemar') if w in text[check:]]


def test_descriptive_path_changes_nothing_on_the_test_path(tmp_path):
    spec = _f3_world(tmp_path / 'f3')
    outs = [str(tmp_path / o) for o in ('alone', 'both')]
    assert main(['--runs', 'none', '--out', outs[0],
                 '--f3-families', spec]) == 0
    assert main(['--runs', 'none', '--out', outs[1], '--f3-families', spec,
                 '--f3-descriptive', _world(tmp_path / 'd')]) == 0
    for name in ('f3_families.csv', 'f3_pa_fits.csv'):
        with open(os.path.join(outs[0], name), 'rb') as a, \
                open(os.path.join(outs[1], name), 'rb') as b:
            assert a.read() == b.read(), name


# --- loading -----------------------------------------------------------------

def test_wandb_summaries_under_an_arm_path_change_nothing(tmp_path):
    spec = load_f3_descriptive_spec(_world(tmp_path))
    want = f3_descriptive(spec)
    run = str(tmp_path / 'run')
    os.makedirs(run)
    for aid in spec:
        spec[aid]['paths'] = [*spec[aid]['paths'], run]
    _run_folder(run)
    with open(os.path.join(run, 'wandb', 'wandb', 'latest-run', 'files',
                           'wandb-summary.json'), encoding='utf-8') as f:
        assert 'test_macro_f1' in json.load(f)
    for got, exp in zip(f3_descriptive(spec), want):
        pd.testing.assert_frame_equal(got, exp)


@pytest.mark.parametrize('half, missing', [
    ({'cell': 'lrc'}, r"\['wiring'\]"),
    ({'config': {'effective': {}}}, r"\['cell', 'wiring'\]")])
def test_half_record_without_seed_names_the_file(tmp_path, half, missing):
    spec = load_f3_descriptive_spec(_world(tmp_path))
    _dump(os.path.join(spec['budget']['paths'][0], 'half.json'),
          {**SUMMARY, **half})
    with pytest.raises(SystemExit, match=r'half\.json.*' + missing):
        f3_descriptive(spec)


def test_budget_and_script_arms_keep_apart_by_label(tmp_path):
    """Their result names equal those of lrc/ncp: listing every run folder
    under every arm changes no count."""
    spec = load_f3_descriptive_spec(_world(tmp_path, opts={
        'budget': {'last': 0.1}, 'a_lrc': {'exit_at': None}}))
    want = f3_descriptive(spec)
    every = sorted({p for a in spec.values() for p in a['paths']})
    for aid in spec:
        spec[aid]['paths'] = every
    got = f3_descriptive(spec)
    pd.testing.assert_frame_equal(got[1], want[1])
    arms = got[1].set_index('arm')
    assert arms.loc['budget', 'k_success'] == 0
    assert arms.loc['a_lrc', 'n_censored'] == 20
    assert arms.loc['script', 'k_success'] == 5


@pytest.mark.parametrize('field, value', [('lr', 0.002), ('clipnorm', 1.0),
                                          ('epochs', 150)])
def test_config_off_the_registry_names_the_file(tmp_path, field, value):
    """epochs: the top-level field (the label reads config.effective)."""
    spec = _world(tmp_path, opts={'budget': {12: {'extra': {field: value}}}})
    with pytest.raises(SystemExit, match=rf'lrc_ncp_seed12\.json: budget .*'
                                         rf'{field} {value!r}'):
        f3_descriptive(load_f3_descriptive_spec(spec))


def test_records_outside_the_seed_list_are_counted_named_not_checked(
        tmp_path):
    spec = load_f3_descriptive_spec(_world(tmp_path))
    for seed in (0, 3, 15):        # tuning seeds, another lr
        _rec(spec['budget']['paths'][0], 'budget', seed,
             extra={'lr': 0.05, 'clipnorm': 1.0}, last=0.0)
    fits, arms, _ = f3_descriptive(spec)
    b = arms.set_index('arm').loc['budget']
    assert (b['n_outside_list'], b['outside_list']) == (3, '0, 3, 15')
    assert (b['n_records'], b['k_success']) == (5, 5)
    assert sorted(fits[fits['arm'] == 'budget']['seed']) == list(range(10, 15))
    out = str(tmp_path / 'out')
    main(['--runs', 'none', '--out', out, '--f3-descriptive',
          str(tmp_path / 'desc.yaml')])
    with open(os.path.join(out, 'summary.md'), encoding='utf-8') as f:
        assert 'not used): budget seeds 0, 3, 15 (3).' in f.read()


@pytest.mark.parametrize('fold', [0, 2])
def test_fold_record_is_an_error(tmp_path, fold):
    spec = _world(tmp_path, opts={'a_cal': {25: {'extra': {
        'split_fold': fold}}}})
    with pytest.raises(SystemExit, match=rf'lrc_cal_ncp_seed25\.json: '
                                         rf'split_fold {fold} in a_cal'):
        f3_descriptive(load_f3_descriptive_spec(spec))


def test_two_records_of_one_arm_and_seed_are_an_error(tmp_path):
    spec = load_f3_descriptive_spec(_world(tmp_path))
    _rec(spec['script']['paths'][0], 'script', 11, name='again.json')
    with pytest.raises(SystemExit, match=r'two records of script .*again'):
        f3_descriptive(spec)


def test_empty_arm_folder_counts_zero_and_reads_not_readable(tmp_path):
    fits, arms, rd = _run(tmp_path, empty=('a_cal', 'a_lrc_legacy',
                                           'a_init', 'budget', 'script'))
    for aid in ('a_cal', 'a_lrc_legacy', 'a_init', 'budget', 'script'):
        a = arms.loc[aid]
        assert [a[c] for c in ('n_records', 'k_success', 'n_diverged',
                               'n_censored', 'n_outside_list')] == [0] * 5
        assert not (fits['arm'] == aid).any()
    assert arms.loc['budget', 'n_exit_after_50'] == 0
    assert rd.loc['manipulation_check', 'classification'] == 'not readable'
    assert rd.loc['manipulation_check', 'counts'].startswith(
        '0 of 20 records')
    assert rd.loc['lrc_init', 'classification'] == \
        'not readable (20 of 40 records)'
    assert rd.loc['count_t_a1', 'classification'] == \
        'not readable (0 of 20 records)'
    for r in ('budget', 'script', 'script_retained'):
        assert rd.loc[r, 'classification'] == 'not readable (0 of 5 records)'
        assert not rd.loc[r, 'readable'] and pd.isna(rd.loc[r, 'sentence'])


# --- per-fit and per-arm columns ---------------------------------------------

def test_fit_columns_success_divergence_and_censoring(tmp_path):
    t = B5_EXIT_THRESHOLD
    fits, arms, _ = _run(tmp_path, opts={'a_lrc': {
        10: {'last': 0.30},                              # inclusive
        11: {'last': 0.29, 'best': 0.40},                # fields differ
        12: {'last': float('nan'), 'exit_at': None},     # diverged, censored
        13: {'diverged': True, 'last': 0.6, 'exit_at': None},
        14: {'diverged': True, 'last': 0.6, 'exit_at': 7},
        15: {'val_curve': [t] * 49 + [t + 1e-9]},        # strict: epoch 50
        16: {'val_curve': [float('nan'), None, 0.5] + [0.5] * 47}}})
    f = fits[fits['arm'] == 'a_lrc'].set_index('seed')
    assert f.loc[10, 'success'] and not f.loc[11, 'success']
    assert f.loc[11, 'success_fields_differ'] is True
    assert f.loc[10, 'success_fields_differ'] is False
    assert f.loc[12, 'success_fields_differ'] is None
    assert f.loc[12, ['diverged', 'success', 'censored']].tolist() == [
        True, False, True]
    assert pd.isna(f.loc[12, 'exit_epoch'])
    assert f.loc[13, ['diverged', 'success', 'censored']].tolist() == [
        True, False, True]
    assert (f.loc[14, 'diverged'], f.loc[14, 'exit_epoch'],
            f.loc[14, 'censored']) == (True, 7, False)
    assert f.loc[15, 'exit_epoch'] == 50 and f.loc[16, 'exit_epoch'] == 3
    a = arms.loc['a_lrc']
    assert (a['n_records'], a['k_success'], a['n_diverged'],
            a['n_censored']) == (20, 16, 3, 2)
    for c in ('test_macro_f1_e50', 'success_e50', 'first_50_equal'):
        assert f[c].isna().all(), c
    out = str(tmp_path / 'out')
    main(['--runs', 'none', '--out', out, '--f3-descriptive',
          str(tmp_path / 'desc.yaml')])
    with open(os.path.join(out, 'summary.md'), encoding='utf-8') as fh:
        assert 'different sides of 0.3: a_lrc seed 11 (0.400 / 0.290).' \
            in fh.read()


def test_200_epoch_columns(tmp_path):
    curve = [0.1] * 49 + [0.35] + [0.2] * 50 + [0.31] * 99 + [0.8, 0.29]
    curve = curve[:200]
    fits, _, _ = _run(tmp_path, opts={'script': {10: {
        'test_curve': curve, 'last': curve[-1]}}})
    f = fits[(fits['arm'] == 'script') & (fits['seed'] == 10)].iloc[0]
    assert [f['test_macro_f1_e50'], f['test_macro_f1_e100'],
            f['test_macro_f1_e200'], f['test_macro_f1_max']] == [
        0.35, 0.2, 0.8, 0.8]
    assert [f['success_e50'], f['success_e100'], f['success_e200']] == [
        True, False, f['success']]
    assert f['success']


def test_n_diverged_equals_n_diverged_a_of_t_a1(tmp_path):
    opts = {'a_lrc_legacy': {11: {'diverged': True, 'last': 0.6},
                             12: {'last': float('nan')}, 13: {'last': 0.1}}}
    spec = load_f3_descriptive_spec(_world(tmp_path, opts=opts))
    _, arms, _ = f3_descriptive(spec)
    seeds = list(range(10, 30))
    entry = F3_TESTS['T-A1']
    a, _, _ = _f3_arm('T-A1', 'a', entry, {**spec['a_lrc_legacy']}, seeds, {})
    b, _, _ = _f3_arm('T-A1', 'b', entry, {**spec['a_lrc']}, seeds, {})
    row = _f3_test('T-A1', entry, 2, 8, a, b, seeds, {'a': [], 'b': []})
    assert arms.set_index('arm').loc['a_lrc_legacy', 'n_diverged'] == \
        row['n_diverged_a'] == 2


def test_zero_of_five_successes_rate():
    fits = [{'success': False, 'censored': True, 'exit_epoch': None,
             'diverged': False, 'test_macro_f1': 0.1}] * 5
    assert _desc_arm_row('budget', fits, [])['success_rate'] == \
        '0/5 [0.00, 0.52]'


def _fits(exits):
    return [{'success': True, 'censored': e is None, 'exit_epoch': e,
             'diverged': False, 'test_macro_f1': 0.5} for e in exits]


@pytest.mark.parametrize('aid, exits, want', [
    ('a_lrc', list(range(1, 11)) + [None] * 10, '> 50'),
    ('a_lrc', list(range(1, 12)) + [None] * 9, '10.5'),
    ('budget', [3, None, 9, None, 1], '9'),
    ('budget', [3, None, 9, None, None], '> 200')])
def test_exit_epoch_median(aid, exits, want):
    row = _desc_arm_row(aid, _fits(exits), [])
    assert row['exit_epoch_median'] == want
    assert row['n_censored'] == exits.count(None)


def test_decision_table_counts_with_one_missing_record_per_arm(tmp_path):
    """M, J, R, C as k_success / n_records of a_cal, a_lrc_legacy, a_lrc,
    a_cal_legacy; E = n_records - n_censored of a_lrc_legacy."""
    fail = {'last': 0.1}
    _, arms, _ = _run(tmp_path, skip={a: (29,) for a in BLOCK_A}, opts={
        'a_cal': {10: fail}, 'a_lrc_legacy': {10: fail, 11: fail, 12: {
            'exit_at': None}, 13: {'exit_at': None}, 14: {'exit_at': None}},
        'a_lrc': {s: fail for s in range(10, 25)},
        'a_cal_legacy': {}})
    got = {a: (arms.loc[a, 'k_success'], arms.loc[a, 'n_records'])
           for a in ('a_cal', 'a_lrc_legacy', 'a_lrc', 'a_cal_legacy')}
    assert got == {'a_cal': (18, 19), 'a_lrc_legacy': (17, 19),
                   'a_lrc': (4, 19), 'a_cal_legacy': (19, 19)}
    e = arms.loc['a_lrc_legacy', 'n_records'] - \
        arms.loc['a_lrc_legacy', 'n_censored']
    assert e == 16


# --- readings ----------------------------------------------------------------

@pytest.mark.parametrize('n_fail, skip, cls, counts', [
    (4, (), 'passed', '16 of 20'),
    (5, (), 'failed', '15 of 20'),
    (3, (29,), 'passed', '16 of 19 records'),
    (4, (29,), 'not readable', '19 of 20 records; 15 with learning success')])
def test_manipulation_check(tmp_path, n_fail, skip, cls, counts):
    opts = {'a_cal': {s: {'last': 0.1} for s in range(10, 10 + n_fail)}}
    _, arms, rd = _run(tmp_path, opts=opts, skip={'a_cal': skip})
    r = rd.loc['manipulation_check']
    assert (r['classification'], r['counts']) == (cls, counts)
    assert r['readable'] == (cls != 'not readable')
    n = 20 - len(skip)
    assert (arms.loc['a_cal', 'k_success'],
            arms.loc['a_cal', 'n_records']) == (n - n_fail, n)
    # the other rows do not depend on the check
    _, _, base = _run(tmp_path / 'base')
    pd.testing.assert_frame_equal(rd.drop('manipulation_check'),
                                  base.drop('manipulation_check'))
    out = str(tmp_path / 'out')
    main(['--runs', 'none', '--out', out, '--f3-descriptive',
          str(tmp_path / 'desc.yaml')])
    with open(os.path.join(out, 'summary.md'), encoding='utf-8') as f:
        text = f.read()
    expl = text.split('## F3 descriptive readings (exploratory)\n\n')[1]
    first = expl.split('\n')[0]
    assert first == {
        'passed': '- lrc_init: ' + base.loc['lrc_init', 'sentence'],
        'failed': 'manipulation check failed: block A not readable',
        'not readable': 'manipulation check not readable (19 of 20 '
                        'records): block A not readable'}[cls]


def test_manipulation_check_on_an_empty_folder(tmp_path):
    _, _, rd = _run(tmp_path, empty=('a_cal',))
    r = rd.loc['manipulation_check']
    assert (r['classification'], r['counts']) == (
        'not readable', '0 of 20 records; 0 with learning success')
    out = str(tmp_path / 'out')
    main(['--runs', 'none', '--out', out, '--f3-descriptive',
          str(tmp_path / 'desc.yaml')])
    with open(os.path.join(out, 'summary.md'), encoding='utf-8') as f:
        assert 'manipulation check not readable (0 of 20 records): block A ' \
            'not readable' in f.read()


def test_lrc_init_reading(tmp_path):
    opts = {'a_init': {s: {'last': 0.1} for s in range(10, 17)},
            'a_init_legacy': {10: {'diverged': True}}}
    _, _, rd = _run(tmp_path, opts=opts)
    assert rd.loc['lrc_init', 'sentence'] == (
        'the data-free initialisation reaches learning success in 13 of 20 '
        'seeds under the suite optimiser and in 19 of 20 under the legacy '
        'placement' + SUFFIX20)
    assert not [s for s in rd['sentence'].dropna() if 'lrc_cal' in s]
    _, _, rd = _run(tmp_path / 'gap', opts=opts, skip={'a_init_legacy': (
        17,)})
    assert rd.loc['lrc_init', 'classification'] == \
        'not readable (39 of 40 records)'
    assert pd.isna(rd.loc['lrc_init', 'sentence'])


def test_count_next_to_t_a1(tmp_path):
    opts = {'a_lrc_legacy': {s: {'last': 0.1} for s in range(10, 18)}}
    _, _, rd = _run(tmp_path, opts=opts)
    assert rd.loc['count_t_a1', 'sentence'] == (
        'at the end of 50 epochs, `lrc/ncp` reaches learning success in 12 '
        'of 20 seeds under the legacy placement' + SUFFIX20)
    _, _, rd = _run(tmp_path / 'gap', opts=opts, skip={'a_lrc_legacy': (29,)})
    r = rd.loc['count_t_a1']
    assert (r['readable'], r['counts']) == (True, '11 of 19 records')
    assert r['sentence'] == (
        'at the end of 50 epochs, `lrc/ncp` reaches learning success in 11 '
        'of 19 records under the legacy placement (exploratory, 19 seeds, '
        'one split)')


@pytest.mark.parametrize('exits, cls, sentence', [
    ([None] * 5, 'no seed escapes within four times the budget at learning '
     'rate 10⁻² (0 of 5; upper limit of the two-sided 95 % Clopper–Pearson '
     'interval 0.52)', True),
    ([None, 120, None, None, None], 'exit epoch 120 (seed 11)', False),
    ([50, 51, None, 3, 200], 'delayed escape (after epoch 50) in 2 of 5 '
     'seeds; 2 of 5 seeds escape within 50 epochs', True)])
def test_budget_reading(tmp_path, exits, cls, sentence):
    opts = {'budget': {s: {'exit_at': e} for s, e in zip(range(10, 15),
                                                         exits)}}
    _, arms, rd = _run(tmp_path, opts=opts)
    r = rd.loc['budget']
    assert r['classification'] == cls
    assert r['sentence'] == (cls + SUFFIX5 if sentence else None) or \
        (not sentence and pd.isna(r['sentence']))
    late = sum(e is not None and e > 50 for e in exits)
    early = sum(e is not None and e <= 50 for e in exits)
    assert (arms.loc['budget', 'n_exit_after_50'],
            arms.loc['budget', 'n_exit_within_50']) == (late, early)
    _, _, rd = _run(tmp_path / 'gap', opts=opts, skip={'budget': (14,)})
    assert rd.loc['budget', 'classification'] == \
        'not readable (4 of 5 records)'


def _script(success_at):
    """A 200-epoch test curve with learning success exactly at the epochs
    in success_at (of 50, 100, 200)."""
    c = [0.1] * 200
    for e in success_at:
        c[e - 1] = 0.5
    return {'test_curve': c, 'last': c[-1]}


@pytest.mark.parametrize('per_seed, counts, cls, retained', [
    ([(), (), (), (), ()], '0 of 5 at epoch 200; 0 of 5 at epoch 100; 0 of 5 '
     'at epoch 50', 'no learning success with four times the suite budget',
     '0 of 0'),
    ([(200,), (200,), (100, 200), (50, 200), (50,)], '4 of 5 at epoch 200; 1 '
     'of 5 at epoch 100; 2 of 5 at epoch 50', None, '1 of 2'),
    ([(200,), (200,), (100, 200), (50, 200), ()], '4 of 5 at epoch 200; 1 of '
     '5 at epoch 100; 1 of 5 at epoch 50', 'learning success under the '
     'legacy placement needs more than the suite budget in these seeds',
     '1 of 1')])
def test_script_reading(tmp_path, per_seed, counts, cls, retained):
    opts = {'script': {s: _script(e) for s, e in zip(range(10, 15),
                                                     per_seed)}}
    _, _, rd = _run(tmp_path, opts=opts)
    r = rd.loc['script']
    assert r['counts'] == counts
    assert r['classification'] == cls or (cls is None and pd.isna(
        r['classification']))
    k, m, j = (int(counts.split(';')[i].split()[0]) for i in range(3))
    assert r['sentence'] == (
        f'`lrc/ncp` reaches learning success at the end of 200 epochs in {k} '
        'of 5 seeds under the optimiser settings and the epoch budget of the '
        f'LRC reference script ({m} of 5 at epoch 100, {j} of 5 at epoch 50'
        + ('; upper limit of the 95 % Clopper–Pearson interval 0.52'
           if k == 0 else '') + ')' + SUFFIX5)
    rr = int(retained.split()[0])
    assert rd.loc['script_retained', 'counts'] == retained
    assert rd.loc['script_retained', 'sentence'] == (
        f'of the {j} seeds with learning success at epoch 50, {rr} retain it '
        'at the end of 200 epochs' + SUFFIX5)
    _, _, rd = _run(tmp_path / 'gap', opts=opts, skip={'script': (10,)})
    for name in ('script', 'script_retained'):
        assert rd.loc[name, 'classification'] == \
            'not readable (4 of 5 records)'


def test_script_pairing_with_the_50_epoch_fit(tmp_path):
    same = _curve(5, 50) + [0.3] * 150
    other = list(same)
    other[20] = 0.25
    opts = {'script': {10: {'val_curve': same}, 11: {'val_curve': other}},
            'a_lrc_legacy': {10: {'val_curve': _curve(5, 50)},
                             11: {'val_curve': _curve(5, 50)}}}
    fits, _, _ = _run(tmp_path, opts=opts, skip={'a_lrc_legacy': (12,)})
    f = fits[fits['arm'] == 'script'].set_index('seed')
    assert f.loc[10, 'first_50_equal'] is True
    assert f.loc[10, 'first_50_max_abs_diff'] == 0.0
    assert f.loc[11, 'first_50_equal'] is False
    assert f.loc[11, 'first_50_max_abs_diff'] == pytest.approx(0.05)
    assert f.loc[12, 'first_50_equal'] is None
    assert pd.isna(f.loc[12, 'first_50_max_abs_diff'])
    assert f.loc[11, 'success']                  # reported as usual
    out = str(tmp_path / 'out')
    main(['--runs', 'none', '--out', out, '--f3-descriptive',
          str(tmp_path / 'desc.yaml')])
    with open(os.path.join(out, 'summary.md'), encoding='utf-8') as fh:
        text = ' '.join(fh.read().split())
    assert 'fit of the seed (their 200-epoch values are reported as usual): ' \
        '11.' in text
