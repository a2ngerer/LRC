# tests/experiments/test_aggregate_f3.py
"""The F3 family engine of experiments/aggregate_results.py (#131), on
synthetic records and frames only."""
import json
import os
import time
import tracemalloc

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.stats import wilcoxon

from experiments.aggregate_results import (EXCLUDED, F3_FAMILIES,
                                           F3_FIT_COLUMNS, F3_TESTS,
                                           _f3_arm, _f3_test, _sequence_row,
                                           _signed_ranks, arm_label,
                                           f3_family_blocks, f3_report,
                                           learning_success, load_f3_spec,
                                           main, mcnemar_exact,
                                           permutation_p)

F1, F2, F3 = 'F1-2026-10-26', 'F2-2026-11-05', 'F3-2026-11-18'
DS = 4242            # one data draw per system (floor 5.5 LV, 10.5 Duffing)
LV_EPOCHS = {'periodic_predator_prey': 300, 'duffing': 450}
LEGACY = ' [optimizer=rmsprop_legacy]'
REQUIRED = [t for t, e in F3_TESTS.items() if e['status'] == 'required']
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))))


# --- record writers ---------------------------------------------------------

def _dump(path, rec):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(rec, f)


def _cell(label, lr):
    """(cell, wiring, config.effective.build) of a registry label."""
    cw, _, tag = label.partition(' [')
    cell, wiring = cw.split('/')
    build = {'wiring': wiring, 'cell': cell, 'lr': lr}
    if 'optimizer=rmsprop_legacy' in tag:   # optimizer_build_kwargs
        build.update(optimizer='rmsprop_legacy', opt_epsilon=None)
    return cell, wiring, build


def _lv(path, system, label, seed, vpt, pv=F3, epochs=None, budget=4000,
        above=False, lr=1e-3, name=None, unbounded=False, extra=None):
    cell, wiring, build = _cell(label, lr)
    epochs = epochs or LV_EPOCHS[system]
    m = {'vpt_multi_median': vpt, 'rate_unbounded': 1.0 if unbounded else 0.0,
         'non_finite': False,
         'anchors': [8], 'anchor_horizon': 16, 'anchor_theta': 0.5}
    eff = {'task': 'lotka_volterra', 'system': system, 't_span': [0.0, 10.0],
           'ic_spread': 0.4, 'n_trajectories': 10, 'seq_len': 32,
           'data_seed': DS, 'split': [6, 2, 2], 'param_budget': budget,
           'ode_unfolds': 6, 'dt_scale': 12.8, 'forcing': 'mixed',
           'epochs': epochs, 'protocol_version': pv, 'build': build}
    rec = {'system': system, 'cell': cell, 'wiring': wiring, 'seed': seed,
           'param_budget': budget, 'budget_deviation': 0.01,
           'budget_above_tolerance': above, 'epochs': epochs, 'lr': lr,
           'clipnorm': None, 'params_effective': 3960, 'dt_scale': 12.8,
           'forcing': 'mixed', 'best_val_epoch': 1,
           'rollout_metrics_test': m, 'config': {'effective': eff}}
    rec.update(extra or {})
    _dump(os.path.join(path, name or f'{cell}_{wiring}_seed{seed}.json'), rec)


GOOD = {f'r{i}': [[6, 1], [1, 5]] for i in range(3)}   # one-class 0.35


def _pa(path, label, seed, f1, pv=F3, last=None, lr=3e-3, budget=30000,
        above=False, name=None, drop=(), split_seed=0, extra=None):
    cell, wiring, build = _cell(label, lr)
    eff = {'task': 'person_activity', 'param_budget': budget, 'epochs': 50,
           'split_seed': split_seed, 'ode_unfolds': 6, 'protocol_version': pv,
           'build': build}
    rec = {'cell': cell, 'wiring': wiring, 'seed': seed, 'epochs': 50,
           'param_budget': budget, 'budget_deviation': 0.02,
           'budget_above_tolerance': above, 'lr': lr, 'clipnorm': 1.0,
           'params_effective': 29500, 'dt_scale': 3.7, 'best_val_epoch': 40,
           'test_macro_f1': f1,
           'test_macro_f1_last': f1 if last is None else last,
           'test_confusion_by_recording': GOOD, 'config': {'effective': eff}}
    for k in drop:
        del rec[k]
    rec.update(extra or {})
    _dump(os.path.join(path, name or f'{cell}_{wiring}_seed{seed}.json'), rec)


def _write(test, side, path, seed, value, **kw):
    e = F3_TESTS[test]
    if e['task'] == 'pa':
        _pa(path, e[side], seed, value, **kw)
    else:
        _lv(path, e['task'], e[side], seed, value, **kw)


# --- a synthetic F3 world -----------------------------------------------------

def _seeds(test):
    return list(range(10, 30 if F3_TESTS[test]['family'] in 'ABCD' else 20))


def _world(tmp_path, tests=REQUIRED, values=None, opts=None, skip=None):
    """Spec path of a world where every test has A = B + step * (i + 1) on
    its seed list (B = 20 VPT or 0.6 macro-F1, step 1 or 0.01), each arm in
    its own folder. values: {test: (a values, b values)} over the seed
    list; opts: {(test, side): writer kwargs, or {seed: kwargs}}; skip:
    {(test, side): seeds without a record}."""
    values, opts, skip = values or {}, opts or {}, skip or {}
    spec = {}
    for test in tests:
        seeds = _seeds(test)
        base, step = (0.6, 0.01) if F3_TESTS[test]['task'] == 'pa' \
            else (20.0, 1.0)
        va, vb = values.get(test, ([base + step * (i + 1)
                                    for i in range(len(seeds))],
                                   [base] * len(seeds)))
        spec[test] = {'seeds': seeds}
        for side, vals in (('a', va), ('b', vb)):
            path = str(tmp_path / test / side)
            os.makedirs(path, exist_ok=True)
            o = opts.get((test, side), {})
            per_seed = bool(o) and all(isinstance(k, int) for k in o)
            for s, v in zip(seeds, vals):
                if s not in skip.get((test, side), ()):
                    _write(test, side, path, s, v,
                           **(o.get(s, {}) if per_seed else o))
            spec[test][side] = {'label': F3_TESTS[test][side],
                                'protocol_version': F3, 'paths': [path]}
    path = str(tmp_path / 'spec.yaml')
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(spec, f)
    return path


def _blocks(spec_path):
    return f3_family_blocks(load_f3_spec(spec_path))[0].set_index('test')


def _row(tmp_path, test, tests=REQUIRED, **kw):
    return _blocks(_world(tmp_path, tests, **kw)).loc[test]


# --- registry -------------------------------------------------------------------

def test_registry_defines_every_test_once_with_its_rules():
    assert {f: (v['status'], list(v['tests'])) for f, v in
            F3_FAMILIES.items()} == {
        'A': ('required', ['T-A1', 'T-A2']),
        'C': ('required', ['T-C1', 'T-C2']),
        'D': ('required', ['T-D1', 'T-D2']),
        'F': ('required', ['T-F1', 'T-F2']),
        'G': ('conditional', ['T-G1', 'T-G2']),
        'H': ('optional', ['T-H1', 'T-H2'])}
    for test, e in F3_TESTS.items():
        assert e['kind'] in ('wilcoxon', 'mcnemar')
        assert e['side'] == ('greater' if e['family'] == 'H' else 'two-sided')
        # F-9: the floor rule binds every contrast except the learning
        # success tests (E8 exception)
        assert e['floor'] == ('informational' if 'success' in e
                              else 'exclusive')
        assert e['endpoint'] in (('test_macro_f1', 'test_macro_f1_last')
                                 if e['task'] == 'pa'
                                 else ('vpt_multi_median',))
        if e['kind'] == 'mcnemar':
            assert 'success' in e and e['side'] == 'two-sided'
    assert {t: e['endpoint'] for t, e in F3_TESTS.items()
            if e['task'] == 'pa'} == {
        'T-A1': 'test_macro_f1_last', 'T-A2': 'test_macro_f1',     # E14
        'T-D1': 'test_macro_f1', 'T-D2': 'test_macro_f1',
        'T-G2': 'test_macro_f1', 'T-H2': 'test_macro_f1'}
    assert F3_TESTS['T-A1']['a'] == 'lrc/ncp' + LEGACY
    assert F3_TESTS['T-A1']['success'] == ('threshold', 0.30)
    assert F3_TESTS['T-A1']['match'] == ('lr', 'clipnorm', 'epochs',
                                         'param_budget')
    assert F3_TESTS['T-F1']['success'] == ('floor+', 2.0)


# --- exact statistics -----------------------------------------------------------

def _brute(x, y, alternative='two-sided'):
    """permutation_p as on main before #131 (2^k pattern matrix), with the
    one-sided share added."""
    sign, ranks = _signed_ranks(x, y)
    if not ranks.size:
        return 1.0
    k = ranks.size
    flips = (np.arange(2 ** k, dtype=np.uint32)[:, None]
             >> np.arange(k, dtype=np.uint32)) & 1
    w, obs = flips @ ranks, ranks[sign > 0].sum()
    if alternative == 'greater':
        return float(np.mean(w >= obs - 1e-9))
    mu = ranks.sum() / 2
    return float(np.mean(np.abs(w - mu) >= abs(obs - mu) - 1e-9))


F2_FIXTURE_DELTAS = ([2, 3, 3, -2, 1, 4, 2, 4, -1, 4],
                     [-1, 2, -3, 4, 5, -6, 7, 8, 9, 10], list(range(1, 11)),
                     [-1, -2, *range(3, 11)], [0, 0, *range(1, 9)])


def test_permutation_p_equals_the_pattern_matrix_up_to_k20():
    rng = np.random.default_rng(0)
    cases = [np.asarray(d, float) for d in F2_FIXTURE_DELTAS]
    for k in (5, 9, 13, 16):
        cases.append(rng.integers(-4, 5, k).astype(float))   # ties, zeros
        cases.append(rng.permutation(np.arange(1, k + 1))
                     * rng.choice([-1, 1], k))
    cases.append(rng.integers(-6, 7, 20).astype(float))
    for d in cases:
        y = np.zeros(len(d))
        assert permutation_p(d, y) == _brute(d, y)
        assert permutation_p(d, y, 'greater') == _brute(d, y, 'greater')
    assert permutation_p(F2_FIXTURE_DELTAS[0], np.zeros(10)) == 26 / 1024
    assert permutation_p([1, 1], [1, 1], 'greater') == 1.0


def test_permutation_p_at_k25_is_exact_fast_and_small():
    rng = np.random.default_rng(1)
    d = rng.permutation(np.arange(1, 26)) * rng.choice([-1, 1], 25)
    y = np.zeros(25)
    tracemalloc.start()
    t0 = time.perf_counter()
    p2, pg = permutation_p(d, y), permutation_p(d, y, 'greater')
    elapsed = time.perf_counter() - t0
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert elapsed < 60 and peak < 2 ** 30
    assert p2 == pytest.approx(wilcoxon(d, method='exact').pvalue)
    assert pg == pytest.approx(wilcoxon(d, alternative='greater',
                                        method='exact').pvalue)
    assert permutation_p(np.arange(1, 26), y) == 2 / 2 ** 25


def test_mcnemar_exact_cases():
    assert mcnemar_exact([False] * 10, [True] * 10) == (
        pytest.approx(2 / 1024), 0, 10)
    p, n10, n01 = mcnemar_exact([True] * 3 + [False] * 7, [False] * 10)
    assert (p, n10, n01) == (pytest.approx(0.25), 3, 0)
    assert mcnemar_exact([True, False] * 5, [True, False] * 5) == (1.0, 0, 0)


def test_learning_success_is_inclusive_at_the_threshold():
    assert learning_success(0.30, False, 0.30)
    assert not learning_success(0.2999, False, 0.30)
    assert not learning_success(0.9, True, 0.30)          # diverged
    assert not learning_success(float('nan'), False, 0.30)


# --- end to end ---------------------------------------------------------------

def test_world_through_main_writes_csv_and_summary(tmp_path):
    spec = _world(tmp_path)
    out = str(tmp_path / 'out')
    assert main(['--runs', str(tmp_path / 'none'), '--out', out,
                 '--f3-families', spec]) == 0
    df = pd.read_csv(os.path.join(out, 'f3_families.csv')).set_index('test')
    assert list(df.index) == REQUIRED
    assert (df['m'] == 2).all() and (df['m_f3'] == 8).all()
    assert (df['n_pairs'] == df['n_expected']).all()
    assert (df['n_outside_list'] == 0).all()
    r = df.loc['T-C1']
    assert (r['reading'], r['method'], r['direction']) == ('PASS', 'exact',
                                                           'A>B')
    assert r['p_raw'] == pytest.approx(2 / 2 ** 20)
    assert r['p_adj'] == pytest.approx(4 / 2 ** 20)
    assert r['p_adj_f3'] == pytest.approx(16 / 2 ** 20)
    assert r['p_perm'] == pytest.approx(r['p_raw'])
    assert df.loc['T-D1', 'ci_units'] == 'seed x recording'
    assert df.loc['T-A2', 'ci_units'] == 'seed x recording'
    assert df.loc['T-A1', 'method'] == 'mcnemar-exact'
    assert df.loc['T-A1', 'reading'] == 'FAIL'     # 20/20 against 20/20
    assert df.loc['T-F1', 'rate_a'].startswith('10/10')
    assert str(df.loc['T-C1', 'lr_a']) == '0.001'
    fits = pd.read_csv(os.path.join(out, 'f3_pa_fits.csv'))
    assert list(fits.columns) == F3_FIT_COLUMNS
    assert len(fits) == 8 * 20                 # A1, A2, D1, D2, two arms
    with open(os.path.join(out, 'summary.md'), encoding='utf-8') as f:
        text = f.read()
    assert '## F3 test families' in text and 'F3-wide m=8' in text
    assert '| C | 2 | T-C1 | periodic_predator_prey | vpt_multi_median ' \
        '| two-sided | cfc_lrc/ncp | gru/ncp | 20/20 | tested | PASS ' \
        '| exact |' in text


def test_mcnemar_row_on_the_learning_success(tmp_path):
    """T-A1: the legacy arm learns on every seed, the reference on none
    (below 0.30 and at chance); the floor marking is informational."""
    n = 20
    r = _row(tmp_path, 'T-A1', values={'T-A1': ([0.6] * n, [0.1] * n)})
    assert (r['method'], r['n10'], r['n01']) == ('mcnemar-exact', 20, 0)
    assert r['p_raw'] == pytest.approx(2 / 2 ** 20)
    assert r['rate_a'].startswith('20/20') and r['rate_b'].startswith('0/20')
    assert r['rate_diff'] == 1.0
    assert (r['floor_b'], r['floor_bound'], r['floor_bound_arms']) == (
        '20/20', True, 'B')
    assert (r['reading'], r['direction']) == ('PASS', 'A>B')
    for k in ('p_perm', 'p_perm_adj', 'verdict_flip',
              'w_plus_minus_w_minus'):
        assert pd.isna(r[k])


def test_endpoint_field_of_the_registry_is_used(tmp_path):
    """T-A1 reads test_macro_f1_last: at the best-validation epoch both arms
    learn, at the end of the budget only arm A."""
    s = _seeds('T-A1')
    r = _row(tmp_path, 'T-A1', opts={('T-A1', 'b'): {'last': 0.1}},
             values={'T-A1': ([0.6] * len(s), [0.6] * len(s))})
    assert (r['n10'], r['n01'], r['reading']) == (20, 0, 'PASS')


def test_missing_endpoint_field_names_the_file(tmp_path):
    with pytest.raises(SystemExit, match=r'seed13\.json: no '
                                         r'test_macro_f1_last'):
        _blocks(_world(tmp_path, opts={
            ('T-A1', 'a'): {13: {'drop': ('test_macro_f1_last',)}}}))


def test_t_a2_reads_the_best_val_field_with_the_crossed_interval(tmp_path):
    """E14: T-A2 compares test_macro_f1, the epoch the confusions belong to,
    so its interval is crossed; a last-epoch value that would reverse the
    direction is not read. lrc/ncp legacy at the one-class value in six of
    twenty seeds makes T-A2 floor-bound (E13, F-9)."""
    r = _row(tmp_path, 'T-A2', opts={('T-A2', 'a'): {'last': 0.1}})
    assert r['endpoint'] == 'test_macro_f1'
    assert (r['reading'], r['direction'], r['ci_units']) == (
        'PASS', 'A>B', 'seed x recording')
    vb = [0.2] * 6 + [0.6] * 14
    r = _row(tmp_path / 'floor', 'T-A2', values={
        'T-A2': ([0.7 + 0.01 * i for i in range(20)], vb)})
    assert (r['floor_b'], r['floor_bound'], r['reading']) == (
        '6/20', True, 'floor-bound')


def test_pa_fits_carry_both_endpoint_fields(tmp_path):
    """E14: both fields per Person Activity fit, a fit two tests use once; a
    record without test_macro_f1_last (written before #138) leaves the
    column empty with no fallback and does not stop a best-val test; a fit
    whose fields lie on different sides of 0.30 is named in the summary."""
    spec = load_f3_spec(_world(tmp_path, opts={
        ('T-A1', 'a'): {12: {'last': 0.25}},
        ('T-D1', 'b'): {14: {'drop': ('test_macro_f1_last',)}}}))
    spec['T-A2']['b']['paths'] = spec['T-A1']['a']['paths']
    blocks, fits = f3_family_blocks(spec)
    assert list(fits.columns) == F3_FIT_COLUMNS and len(fits) == 7 * 20
    shared = fits[fits['tests'] == 'T-A1.a; T-A2.b'].set_index('seed')
    assert len(shared) == 20
    assert shared.loc[12, ['test_macro_f1', 'test_macro_f1_last']].tolist() \
        == [pytest.approx(0.63), 0.25]
    assert shared.loc[12, 'success_fields_differ'] is True
    old = fits[fits['tests'] == 'T-D1.b'].set_index('seed').loc[14]
    assert old['test_macro_f1'] == 0.6 and pd.isna(old['test_macro_f1_last'])
    assert old['success_fields_differ'] is None
    assert fits['success_fields_differ'].eq(True).sum() == 1
    assert blocks.set_index('test').loc['T-D1', 'reading'] == 'PASS'
    text = f3_report(blocks, fits)
    assert ('T-A1.a; T-A2.b lrc/ncp [optimizer=rmsprop_legacy] seed 12 '
            '(0.630 / 0.250)') in text
    assert 'without a test_macro_f1_last value (missing or NaN): 1.' in text


def test_sequence_row_fields_are_opt_in():
    r = {'cell': 'gru', 'wiring': 'ncp', 'seed': 0, 'test_macro_f1': 0.5,
         'test_macro_f1_last': 0.4, 'config': {'effective': {}}}
    plain = _sequence_row('x.json', r, 'pa')
    extra = _sequence_row('x.json', r, 'pa', fields=('test_macro_f1_last',))
    assert list(extra) == [*plain, 'test_macro_f1_last']
    assert extra['test_macro_f1_last'] == 0.4
    with pytest.raises(SystemExit, match='x.json: no nothing'):
        _sequence_row('x.json', r, 'pa', fields=('nothing',))


def test_floor_rule_exclusive_against_informational(tmp_path):
    """Five of twenty fits of arm B below the floor make the arm not
    learned: the exclusive C test reads floor-bound, the informational F
    test (ten seeds, three below) keeps PASS and reports the marking."""
    s20, s10 = _seeds('T-C1'), _seeds('T-F1')
    vc = [-1.0 if i < 5 else 20.0 for i in range(20)]
    vf = [-1.0 if i < 3 else 20.0 for i in range(10)]
    b = _blocks(_world(tmp_path, values={
        'T-C1': ([21.0 + i for i in range(len(s20))], vc),
        'T-F1': ([21.0 + i for i in range(len(s10))], vf)}))
    c, f = b.loc['T-C1'], b.loc['T-F1']
    assert (c['floor_b'], c['floor_bound'], c['reading']) == (
        '5/20', True, 'floor-bound')
    assert c['p_adj'] < 0.05                       # still reported
    assert (f['floor_b'], f['floor_bound'], f['reading']) == (
        '3/10', True, 'PASS')
    assert f['rate_b'].startswith('7/10') and f['n10'] == 3


def test_duffing_learning_success_at_floor_plus_two(tmp_path):
    """floor 10.5 on the synthetic Duffing draw: 12.5 succeeds, 12.4 not."""
    spec = load_f3_spec(_world(tmp_path))
    path = str(tmp_path / 'edge')
    _lv(path, 'duffing', 'lrc/ncp', 10, 12.5)
    _lv(path, 'duffing', 'lrc/ncp', 11, 12.4)
    arm = {**spec['T-F1']['b'], 'paths': [path]}
    df, _ = _f3_arm('T-F1', 'b', F3_TESTS['T-F1'], arm, [10, 11], {})
    assert df['floor'].tolist() == [10.5, 10.5]
    assert df['success'].tolist() == [True, False]


def test_incomplete_test_reports_its_statistics(tmp_path):
    r = _row(tmp_path, 'T-C1', skip={('T-C1', 'b'): [12]})
    assert (r['n_expected'], r['n_pairs'], r['reading']) == (20, 19,
                                                             'incomplete')
    assert r['p_raw'] == pytest.approx(2 / 2 ** 19)
    assert not r['significant'] and not r['significant_f3']
    r = _row(tmp_path / 'empty', 'T-C1', skip={('T-C1', 'b'): _seeds('T-C1')})
    assert (r['n_pairs'], r['reading']) == (0, 'incomplete')


def test_incomplete_reads_before_floor_bound(tmp_path):
    vb = [-1.0 if i < 5 else 20.0 for i in range(20)]
    r = _row(tmp_path, 'T-C1', skip={('T-C1', 'b'): [29]},
             values={'T-C1': ([21.0 + i for i in range(20)], vb)})
    assert (r['floor_b'], r['floor_bound'], r['reading']) == (
        '5/19', True, 'incomplete')


def test_records_outside_the_seed_list_are_counted_not_used(tmp_path):
    spec = load_f3_spec(_world(tmp_path))
    folder = spec['T-C1']['a']['paths'][0]
    for seed in range(10):                  # e.g. fits of another wave
        _lv(os.path.join(folder, 'other'), 'periodic_predator_prey',
            'cfc_lrc/ncp', seed, -50.0)
    _lv(os.path.join(folder, 'other'), 'periodic_predator_prey',
        'cfc_lrc/ncp', 100, -50.0, name='tuning.json')
    r = f3_family_blocks(spec)[0].set_index('test').loc['T-C1']
    assert (r['n_outside_list'], r['outside_list']) == (11,
                                                        'a: seeds 0-9, 100')
    assert r['n_pairs'] == 20 and r['p_raw'] == pytest.approx(2 / 2 ** 20)
    out = str(tmp_path / 'out')
    with open(tmp_path / 'spec2.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(spec, f)
    assert main(['--runs', str(tmp_path / 'none'), '--out', out,
                 '--f3-families', str(tmp_path / 'spec2.yaml')]) == 0
    with open(os.path.join(out, 'summary.md'), encoding='utf-8') as f:
        assert 'T-C1 a: seeds 0-9, 100 (11)' in f.read()


def test_duplicate_arm_and_seed_is_an_error(tmp_path):
    spec = load_f3_spec(_world(tmp_path))
    _lv(os.path.join(spec['T-C1']['a']['paths'][0], 'again'),
        'periodic_predator_prey', 'cfc_lrc/ncp', 12, 25.0)
    with pytest.raises(SystemExit, match=r'two records of T-C1\.a'):
        f3_family_blocks(spec)


@pytest.mark.parametrize('pv', [F1, F2, None])
def test_non_f3_or_unstamped_record_is_rejected_whatever_its_seed(
        tmp_path, pv):
    spec = load_f3_spec(_world(tmp_path))
    _lv(os.path.join(spec['T-C1']['b']['paths'][0], 'old'),
        'periodic_predator_prey', 'gru/ncp', 3, 20.0, pv=pv)
    with pytest.raises(SystemExit, match=r'old.*protocol_version.*T-C1\.b'):
        f3_family_blocks(spec)


@pytest.mark.parametrize('field, value', [
    ('lr', 1e-2), ('clipnorm', 0.5), ('epochs', 60), ('param_budget', 31000)])
def test_t_a1_match_mismatch_names_the_seed(tmp_path, field, value):
    with pytest.raises(SystemExit,
                       match=rf'T-A1: .*differ in {field}.*\[15\]'):
        _blocks(_world(tmp_path, opts={
            ('T-A1', 'b'): {15: {'extra': {field: value}}}}))


def test_per_arm_lr_is_reported_without_a_match_entry(tmp_path):
    r = _row(tmp_path / 'd', 'T-D2', opts={('T-D2', 'b'): {'lr': 1e-2}})
    assert (r['lr_a'], r['lr_b'], r['reading']) == ('0.003', '0.01', 'PASS')


def test_budget_exclusion(tmp_path):
    r = _row(tmp_path, 'T-D1', opts={('T-D1', 'b'): {12: {'above': True}}})
    assert r['status'] == EXCLUDED and pd.isna(r['reading'])
    assert pd.isna(r['p_raw']) and r['n_pairs'] == 20   # descriptive only


def test_optimizer_variants_of_one_cell_pair_by_seed(tmp_path):
    """Both variants of lrc/ncp below one folder: the optimizer tag keeps
    the arms apart, the pairs form by seed."""
    spec = load_f3_spec(_world(tmp_path))
    shared = str(tmp_path / 'shared')
    for seed in _seeds('T-A1'):     # A learns on seeds = 0 mod 3, B on odd
        if seed != 10:
            _pa(shared, 'lrc/ncp' + LEGACY, seed,
                0.6 if seed % 3 == 0 else 0.1, name=f'legacy_{seed}.json')
        if seed != 13:
            _pa(shared, 'lrc/ncp', seed, 0.6 if seed % 2 else 0.1,
                name=f'f1_{seed}.json')
    spec['T-A1']['a']['paths'] = spec['T-A1']['b']['paths'] = [shared]
    r = f3_family_blocks(spec)[0].set_index('test').loc['T-A1']
    # seed pairs 11, 12, 14-29: only A on 12, 18, 24; only B on 11, 17, 19,
    # 23, 25, 29
    assert (r['n_pairs'], r['n10'], r['n01'], r['reading']) == (
        18, 3, 6, 'incomplete')
    assert r['rate_a'].startswith('6/18') and r['rate_b'].startswith('9/18')


def test_diverged_and_non_finite_fits_fail_and_enter_at_the_worst_value(
        tmp_path):
    spec = load_f3_spec(_world(tmp_path, opts={
        ('T-F1', 'b'): {10: {'unbounded': True},
                        11: {'extra': {'training_diverged': True}}},
        ('T-A1', 'b'): {10: {'extra': {'test_macro_f1_last': None}}}}))
    entry = F3_TESTS['T-F1']
    df, _ = _f3_arm('T-F1', 'b', entry, spec['T-F1']['b'], _seeds('T-F1'),
                    {})
    df = df.set_index('seed')
    assert df.loc[10, 'diverged'] and not df.loc[11, 'diverged']
    assert not df.loc[10, 'success'] and not df.loc[11, 'success']
    b = f3_family_blocks(spec)[0].set_index('test')
    assert (b.loc['T-F1', 'n_nonfinite'], b.loc['T-F1', 'rate_b'][:4]) == (
        1, '8/10')
    # the unbounded fit enters at VPT 0: seed 10 has A - B = 21
    assert b.loc['T-F1', 'n_positive'] == 10
    assert (b.loc['T-A1', 'n_nonfinite'], b.loc['T-A1', 'rate_b'][:5]) == (
        1, '19/20')
    with pytest.raises(SystemExit, match=r'T-D1: .*data key.*\[14\]'):
        _blocks(_world(tmp_path / 'dk', opts={
            ('T-D1', 'b'): {14: {'split_seed': 1}}}))


# --- one test row from frames ------------------------------------------------

def _frame(vals, above=False, width=False):
    v = np.asarray(vals, float)
    return pd.DataFrame({
        'seed': range(10, 10 + len(v)), 'metric': v, 'task': 'lv',
        'diverged': ~np.isfinite(v), 'below': False,
        'budget_above_tolerance': above, 'width_matched': width,
        'data_key': 1, 'lr': 1e-3, 'clipnorm': None, 'epochs': 300,
        'param_budget': 4000})


def _frow(delta, m=2, m_f3=8, a=None, b=None):
    """_f3_test of T-C1 with B = 0 and A = delta on seeds 10, 11, ..."""
    return _f3_test('T-C1', F3_TESTS['T-C1'], m, m_f3,
                    _frame(delta, **(a or {})),
                    _frame([0.0] * len(delta), **(b or {})),
                    list(range(10, 10 + len(delta))), {'a': [], 'b': []})


def test_family_cut_permutation_flip_direction_and_non_finite():
    r = _frow([-1, -2, *range(3, 11)])
    assert r['p_raw'] == pytest.approx(10 / 1024)
    assert r['significant'] and not r['significant_f3']
    assert r['only_under_family_cut'] and r['reading'] == 'PASS'
    r = _frow([2, 3, 3, -2, 1, 4, 2, 4, -1, 4])
    assert r['method'] == 'approx'
    assert r['p_perm_adj'] == pytest.approx(52 / 1024)
    assert r['verdict_flip'] and r['reading'] == 'PASS'
    r = _frow([-d for d in range(1, 11)])
    assert (r['reading'], r['direction']) == ('PASS', 'A<B')
    assert r['w_plus_minus_w_minus'] == -55
    r = _frow([float('nan'), *range(2, 11)])
    assert r['n_nonfinite'] == 1 and r['n_positive'] == 9


def test_budget_exclusion_spares_only_the_width_matched_arm():
    r = _frow(range(1, 11), b={'above': True, 'width': True})
    assert (r['status'], r['reading']) == ('tested', 'PASS')
    r = _frow(range(1, 11), b={'above': True})
    assert (r['status'], r['reading']) == (EXCLUDED, None)


def test_one_sided_h_test(tmp_path):
    tests = REQUIRED + ['T-H1', 'T-H2']
    b = _blocks(_world(tmp_path, tests))
    r = b.loc['T-H1']
    assert (r['side'], r['m'], r['m_f3']) == ('greater', 2, 10)
    assert r['p_raw'] == pytest.approx(1 / 1024)
    assert r['p_perm'] == pytest.approx(1 / 1024)
    assert r['p_adj_f3'] == pytest.approx(10 / 1024)
    assert (r['reading'], r['direction']) == ('PASS', 'A>B')
    rev = [20.0 - 0.1 * (i + 1) for i in range(10)]
    r = _row(tmp_path / 'rev', 'T-H1', tests,
             values={'T-H1': (rev, [20.0] * 10)})
    assert r['p_raw'] == pytest.approx(1.0) and r['p_perm'] == 1.0
    assert r['reading'] == 'FAIL'


# --- spec ---------------------------------------------------------------------

def test_spec_rules(tmp_path):
    spec = _world(tmp_path, REQUIRED + ['T-G1', 'T-G2'])
    with open(spec, encoding='utf-8') as f:
        good = yaml.safe_load(f)

    def check(mutate, match):
        s = json.loads(json.dumps(good))
        mutate(s)
        path = str(tmp_path / 'bad.yaml')
        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(s, f)
        if match is None:
            return load_f3_spec(path)
        with pytest.raises(SystemExit, match=match):
            load_f3_spec(path)

    assert set(check(lambda s: [s.pop(t) for t in ('T-G1', 'T-G2')],
                     None)) == set(REQUIRED)
    check(lambda s: s.pop('T-G2'), 'left out only together')
    check(lambda s: s.pop('T-D1'), r"missing required tests \['T-D1'\]")
    check(lambda s: s.__setitem__('T-Z9', s['T-D1']), r"unknown.*T-Z9")
    check(lambda s: s['T-C1'].pop('seeds'), 'keys')
    check(lambda s: s['T-C1'].__setitem__('seeds', [10, 10]), 'distinct')
    check(lambda s: s['T-C1'].__setitem__('seeds', '10-29'), 'distinct')
    check(lambda s: s['T-C1']['b'].__setitem__('label', 'gru/dense'),
          'label')
    check(lambda s: s['T-C1']['b'].__setitem__('protocol_version', F1),
          'not an F3 stamp')
    check(lambda s: s['T-C1']['b']['paths'].__setitem__(
        0, str(tmp_path / 'T-C1' / '*')), 'pattern')
    check(lambda s: s['T-C1']['b'].__setitem__('seeds', [10]), 'keys')


def test_template_lists_every_registered_test_with_placeholders():
    path = os.path.join(REPO, 'configs/aggregate/f3_families.yaml')
    with open(path, encoding='utf-8') as f:
        t = yaml.safe_load(f)
    assert list(t) == list(F3_TESTS)
    for test, e in F3_TESTS.items():
        assert [t[test][s]['label'] for s in 'ab'] == [e['a'], e['b']]
        assert t[test]['seeds'] == _seeds(test)
        assert all(p.startswith('<') for s in 'ab'
                   for p in t[test][s]['paths'])
    with pytest.raises(SystemExit, match='no such path'):
        load_f3_spec(path)


# --- labels -------------------------------------------------------------------

def _rec(pv, system='periodic_predator_prey', epochs=300, build=None, **top):
    eff = {'protocol_version': pv, 'epochs': epochs, 'ode_unfolds': 6}
    if build is not None:
        eff['build'] = build
    if system == 'pa':
        eff['task'] = 'person_activity'
        rec = {'cell': 'lrc', 'wiring': 'ncp', 'seed': 0,
               'param_budget': 30000, 'test_macro_f1': 0.5}
    else:
        eff.update(task='lotka_volterra', system=system)
        rec = {'cell': 'lrc', 'wiring': 'ncp', 'seed': 0, 'system': system,
               'param_budget': 4000, 'forcing': 'mixed', 'dt_scale': 12.8,
               'rollout_metrics_test': {}}
    rec.update(top)
    rec['config'] = {'effective': {**eff, **top.pop('eff', {})}}
    return rec


def test_optimizer_tags_and_f3_epochs_gate():
    default = {'optimizer': 'rmsprop', 'opt_epsilon': None}
    legacy = {'optimizer': 'rmsprop_legacy', 'opt_epsilon': None}
    for build in (None, {}, default, {'cell': 'lrc', 'lr': 0.01}):
        assert arm_label(_rec(F1, build=build)) == 'lrc/ncp'
        assert arm_label(_rec(F1, build=build, param_budget=None,
                              dt_scale=1, forcing='free',
                              eff={'ode_unfolds': 1})) == \
            'lrc/ncp [k=1, dt_scale=1, fixed width, forcing=free]'
        assert arm_label(_rec(F2, epochs=600, build=build)) == \
            'lrc/ncp [epochs=600]'
    assert arm_label(_rec(F3, 'pa', 50, build=legacy)) == \
        'lrc/ncp [optimizer=rmsprop_legacy]'
    assert arm_label(_rec(F3, build={'optimizer': 'rmsprop',
                                     'opt_epsilon': 1e-8})) == \
        'lrc/ncp [opt_epsilon=1e-08]'
    assert arm_label(_rec(F3, epochs=600, forcing='teacher',
                          build={'optimizer': 'adam',
                                 'opt_epsilon': 1e-8})) == \
        'lrc/ncp [forcing=teacher, epochs=600, optimizer=adam, ' \
        'opt_epsilon=1e-08]'
    assert arm_label(_rec(F3, 'duffing', 900)) == 'lrc/ncp [epochs=900]'
    assert arm_label(_rec(F3, 'pa', 100)) == 'lrc/ncp [epochs=100]'
    for system, ep in (('periodic_predator_prey', 300), ('duffing', 450),
                       ('pa', 50)):
        assert arm_label(_rec(F3, system, ep)) == 'lrc/ncp'
        assert arm_label(_rec(F1, system, 2 * ep)) == 'lrc/ncp'
