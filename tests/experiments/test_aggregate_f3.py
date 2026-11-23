# tests/experiments/test_aggregate_f3.py
"""The F3 family engine of experiments/aggregate_results.py (#131), on
synthetic records and frames only."""
import copy
import io
import json
import os
import shutil
import time
import tracemalloc

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.stats import wilcoxon

from experiments.aggregate_results import (CONVERGED, EXCLUDED, F3_B_COLUMNS,
                                           F3_COLUMNS, F3_E_COLUMNS,
                                           F3_E_L17_RECORDINGS,
                                           F3_E_NESTED_LABEL, F3_FAMILIES,
                                           F3_FIT_COLUMNS, F3_FOLD_SEEDS,
                                           F3_SEEDS, F3_TESTS, N_BOOT,
                                           _const_phase_flagged,
                                           _e_one_class, _f2_arm, _f3_arm,
                                           _f3_test, _median_diff_ci,
                                           _sequence_row, _signed_ranks,
                                           arm_label, f3_family_b,
                                           f3_family_blocks, f3_family_e,
                                           f3_report, floor_share_bound,
                                           learning_success, load_f3_spec,
                                           load_sequence_runs, main,
                                           mcnemar_exact, not_learned,
                                           permutation_p, wilcoxon_test)
from src.evaluation.metrics import one_class_macro_f1
from src.evaluation.stats import (bootstrap_median_ci, crossed_median_diff_ci,
                                  nested_median_ci, stratified_median_ci)
from tests.experiments.test_aggregate_results import _write_co

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
        above=False, name=None, drop=(), split_seed=0, extra=None,
        eff_extra=None):
    cell, wiring, build = _cell(label, lr)
    eff = {'task': 'person_activity', 'param_budget': budget, 'epochs': 50,
           'split_seed': split_seed, 'ode_unfolds': 6, 'protocol_version': pv,
           'build': build, **(eff_extra or {})}
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


def _fold_recs(fold):
    """The test recordings of fold k (#134): index k of every person."""
    return [f'{p}{fold + 1:02d}' for p in 'ABCDE']


def _pe(path, label, seed, f1, fold=None, per_rec=None, cms=None,
        dt_scale=3.7, eff_fold=None, extra=None, **kw):
    """A family E fold record (#133) through _pa: split_fold = the fold of
    the seed (F3_FOLD_SEEDS) unless given, every test recording of the fold
    at f1 (or per_rec) with the confusions GOOD['r0'] (or cms), the resolved
    dt_scale at the top level and in config.effective; eff_fold sets
    config.effective.split_fold alone."""
    fold = (seed - 10) // 5 if fold is None else fold
    recs = _fold_recs(fold)
    _pa(path, label, seed, f1, extra={
        'split_fold': fold, 'dt_scale': dt_scale,
        'test_macro_f1_per_recording': per_rec or dict.fromkeys(recs, f1),
        'test_confusion_by_recording': cms or dict.fromkeys(recs,
                                                            GOOD['r0']),
        **(extra or {})},
        eff_extra={'split_fold': fold if eff_fold is None else eff_fold,
                   'dt_scale': dt_scale}, **kw)


CO_VIEW = {'cons': 'off', 'avg_in': 'mean', 'fault_mean': 'fault_mean'}


def _co(path, label, seed, f1, pv=F3, epochs=60, by_fault=None,
        corruption_seed=None, above=False, best=40, wiring='ncp',
        extra=None):
    """A family B committee record (#132) through _write_co: test macro-F1
    f1 at m = 15 (or the whole by_fault list), split_seed 0, val_seed 0,
    corruption_seed = seed unless given, the confusions GOOD at every
    fault level."""
    os.makedirs(path, exist_ok=True)
    _write_co(path, wiring, seed, f1, 0.8, 0.6, view_average=CO_VIEW[label],
              above=above, epochs=epochs, protocol_version=pv,
              corruption_seed=seed if corruption_seed is None
              else corruption_seed, test_macro_f1_by_fault=by_fault,
              **{'split_seed': 0, 'val_seed': 0, 'lr': 3e-3, 'clipnorm': 1.0,
                 'param_budget': 30000, 'best_val_epoch': best,
                 'training_diverged': False,
                 'test_confusion_by_recording_by_fault': [GOOD] * 5,
                 **(extra or {})})


def _write(test, side, path, seed, value, **kw):
    e = F3_TESTS[test]
    if e['family'] == 'E':
        _pe(path, e[side], seed, value, **kw)
    elif e['task'] == 'pa':
        _pa(path, e[side], seed, value, **kw)
    elif e['task'] == 'co':
        _co(path, e[side], seed, value, **kw)
    else:
        _lv(path, e['task'], e[side], seed, value, **kw)


# --- a synthetic F3 world -----------------------------------------------------

def _seeds(test):
    return list(F3_SEEDS.get(F3_TESTS[test]['family'], range(10, 20)))


def _world(tmp_path, tests=REQUIRED, values=None, opts=None, skip=None):
    """Spec path of a world where every test has A = B + step * (i + 1) on
    its seed list (B = 20 VPT or 0.6 macro-F1, family B at m = 15, step 1
    or 0.01), each arm in
    its own folder. values: {test: (a values, b values)} over the seed
    list; opts: {(test, side): writer kwargs, or {seed: kwargs}}; skip:
    {(test, side): seeds without a record}."""
    values, opts, skip = values or {}, opts or {}, skip or {}
    spec = {}
    for test in tests:
        seeds = _seeds(test)
        base, step = (0.6, 0.01) if F3_TESTS[test]['task'] in ('pa', 'co') \
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


def _report_cells(text, test):
    """{header: cell} of the row of test in the main f3_report table."""
    lines = text.splitlines()
    head = next(ln for ln in lines if ln.startswith('| Family |'))
    row = next(ln for ln in lines if f'| {test} |' in ln)
    return dict(zip(*([c.strip() for c in ln.split('|')[1:-1]]
                      for ln in (head, row))))


# --- registry -------------------------------------------------------------------

def test_registry_defines_every_test_once_with_its_rules():
    assert {f: (v['status'], list(v['tests'])) for f, v in
            F3_FAMILIES.items()} == {
        'A': ('required', ['T-A1', 'T-A2']),
        'B': ('required', ['T-B1', 'T-B2']),
        'C': ('required', ['T-C1', 'T-C2']),
        'D': ('required', ['T-D1', 'T-D2']),
        'E': ('required', ['T-E1', 'T-E2', 'T-E3']),
        'F': ('required', ['T-F1', 'T-F2']),
        'G': ('required', ['T-G1', 'T-G2']),
        'H': ('optional', ['T-H1', 'T-H2'])}
    for test, e in F3_TESTS.items():
        assert e['kind'] in ('wilcoxon', 'mcnemar', 'did')
        assert e['side'] == ('greater' if e['family'] in 'BH'
                             else 'two-sided')
        # F-9: the floor rule binds every contrast except the learning
        # success tests (E8 exception) and family B (no floor rule)
        assert e['floor'] == ('none' if e['family'] == 'B' else
                              'informational' if 'success' in e
                              else 'exclusive')
        assert e['endpoint'] in (('test_macro_f1_per_recording',)
                                 if e['family'] == 'E'
                                 else ('test_macro_f1', 'test_macro_f1_last')
                                 if e['task'] == 'pa'
                                 else ('test_macro_f1_by_fault',)
                                 if e['task'] == 'co'
                                 else ('vpt_multi_median',))
        assert ('faults' in e) == (e['family'] == 'B')
        if e['kind'] == 'mcnemar':
            assert 'success' in e and e['side'] == 'two-sided'
    assert {t: e['endpoint'] for t, e in F3_TESTS.items()
            if e['task'] == 'pa'} == {
        'T-A1': 'test_macro_f1_last', 'T-A2': 'test_macro_f1',     # E14
        'T-D1': 'test_macro_f1', 'T-D2': 'test_macro_f1',
        **dict.fromkeys(('T-E1', 'T-E2', 'T-E3'),
                        'test_macro_f1_per_recording'),
        'T-G2': 'test_macro_f1', 'T-H2': 'test_macro_f1'}
    assert F3_TESTS['T-A1']['a'] == 'lrc/ncp' + LEGACY
    assert F3_TESTS['T-A1']['success'] == ('threshold', 0.30)
    assert {t: e.get('match') for t, e in F3_TESTS.items()} == {
        **{t: ('lr', 'clipnorm', 'epochs', 'param_budget')
           for t in ('T-A1', 'T-A2')},
        **{t: ('split_seed', 'corruption_seed', 'val_seed')
           for t in ('T-B1', 'T-B2')},
        **{t: ('epochs', 'param_budget') for t in (
            'T-C1', 'T-C2', 'T-D1', 'T-D2', 'T-E1', 'T-E2', 'T-E3', 'T-F1',
            'T-F2', 'T-G1', 'T-G2', 'T-H1', 'T-H2')}}
    assert F3_TESTS['T-F1']['success'] == ('floor+', 2.0)
    # family B (#132): the committee arms by name, the registry's record
    # fields, the fault levels each test reads
    b = F3_FAMILIES['B']
    assert {k: b[k] for k in ('cell', 'wiring', 'n_columns', 'epochs')} == {
        'cell': 'cfc_lrc', 'wiring': 'ncp', 'n_columns': 30, 'epochs': 60}
    assert {t: (e['task'], e['a'], e['b'], e['kind'], e['side'], e['floor'],
                e['faults']) for t, e in F3_TESTS.items()
            if e['family'] == 'B'} == {
        'T-B1': ('co', 'cons', 'fault_mean', 'wilcoxon', 'greater', 'none',
                 (15,)),
        'T-B2': ('co', 'cons', 'avg_in', 'did', 'greater', 'none', (0, 15))}
    assert list(F3_TESTS).index('T-B1') == list(F3_TESTS).index('T-A2') + 1


def test_floor_share_bound_cases():
    """F-9: an arm is floor-bound from 30 % of its fits below the floor;
    an empty arm is not; the F1/F2 rule not_learned keeps a fifth."""
    for k, n in ((2, 5), (3, 10), (6, 20)):
        assert floor_share_bound(k, n)
    for k, n in ((1, 5), (2, 10), (5, 20), (0, 0)):
        assert not floor_share_bound(k, n)
    assert not_learned(5, 20) and not not_learned(1, 5)


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
    assert (df['m'] == 2).all() and (df['m_f3'] == 14).all()
    assert (df['n_pairs'] == df['n_expected']).all()
    assert (df['n_outside_list'] == 0).all()
    raw = pd.read_csv(os.path.join(out, 'f3_families.csv'),
                      dtype=str).set_index('test')
    assert raw.loc['T-C1', 'n_positive'] == '20'          # Int64, not 20.0
    assert pd.isna(raw.loc['T-A1', 'n_positive'])
    assert F3_COLUMNS.index('p_perm_adj_f3') == F3_COLUMNS.index(
        'p_perm_adj') + 1
    r = df.loc['T-C1']
    assert (r['reading'], r['method'], r['direction']) == ('PASS', 'exact',
                                                           'A>B')
    assert r['p_raw'] == pytest.approx(2 / 2 ** 20)
    assert r['p_adj'] == pytest.approx(4 / 2 ** 20)
    assert r['p_adj_f3'] == pytest.approx(28 / 2 ** 20)
    assert r['p_perm'] == pytest.approx(r['p_raw'])
    assert df.loc['T-A2', 'ci_units'] == 'seed x recording'
    assert df.loc['T-A1', 'method'] == 'mcnemar-exact'
    assert df.loc['T-A1', 'reading'] == 'FAIL'     # 20/20 against 20/20
    assert df.loc['T-F1', 'rate_a'].startswith('10/10')
    assert str(df.loc['T-C1', 'lr_a']) == '0.001'
    fits = pd.read_csv(os.path.join(out, 'f3_pa_fits.csv'))
    assert list(fits.columns) == F3_FIT_COLUMNS
    # A1, A2, D1, D2 and G2 (ten seeds), two arms; E1, E2: 25 runs
    assert len(fits) == 8 * 20 + 2 * 10 + 4 * 25
    with open(os.path.join(out, 'summary.md'), encoding='utf-8') as f:
        text = f.read()
    assert '## F3 test families' in text and 'F3-wide m=14' in text
    for legend in ('The deciding p of a Wilcoxon row is the exact sign-flip '
                   'permutation p of the Wilcoxon statistic',
                   'reading and significant read p perm adj, significant_f3 '
                   'p perm adj F3',
                   'The deciding p of the McNemar row (T-A1) is the exact '
                   'McNemar p',
                   'flag "p of wilcoxon_test would decide otherwise"',
                   'an arm at the floor in at least 30 % of the seeds'):
        assert legend in ' '.join(text.split()), legend
    assert _report_cells(text, 'T-C1')['p perm adj F3'] == format(
        28 / 2 ** 20, '.4g')
    assert '| C | 2 | T-C1 | periodic_predator_prey | vpt_multi_median ' \
        '| two-sided | cfc_lrc/ncp | gru/ncp | 20/20 | tested | PASS ' \
        '| exact |' in text
    # #151: the descriptive columns, appended in this order
    with open(os.path.join(out, 'f3_families.csv'), encoding='utf-8') as f:
        assert f.readline().rstrip('\n').split(',') == F3_COLUMNS
    tail = NEW_COLUMNS + B_COLUMNS
    assert F3_COLUMNS[-len(tail):] == tail
    assert F3_COLUMNS[-len(tail) - 1] == 'param_budget_b'
    for t in REQUIRED:
        fam = F3_TESTS[t]['family']
        for c in ('n_diverged_a', 'n_diverged_b', 'n_other_label_a',
                  'n_other_label_b'):
            assert raw.loc[t, c] == '0', (t, c)            # Int64
        for c in NEW_COLUMNS[-4:]:                          # flags: C and F
            assert (raw.loc[t, c] == '0') if fam in 'CF' \
                else pd.isna(raw.loc[t, c]), (t, c)
        for c in NEW_COLUMNS[:5]:                           # intervals: D
            assert pd.isna(raw.loc[t, c]) == (fam != 'D'), (t, c)
        for c in ('crossed_ci_note', 'interval_note', 'other_labels_a',
                  'other_labels_b'):
            assert pd.isna(raw.loc[t, c]), (t, c)
        for c in B_COLUMNS:                                 # family B only
            want = ('0' if fam == 'B' and (t == 'T-B2' or 'evalonly' in c)
                    else None)
            assert (raw.loc[t, c] == want) if want else pd.isna(
                raw.loc[t, c]), (t, c)
    for c in ('median_diff_lo', 'median_diff_hi', 'ci_units'):
        assert pd.isna(df.loc['T-D1', c]), c
    norm = ' '.join(text.split())
    assert norm.count(D_LEGEND) == 1
    assert 'Family D interval notes (E17): none.' in text
    assert '- T-C1 a: none\n- T-C1 b: none' in text
    assert '| T-C1 | 0 | 0 | 0 | 0 | 0 | 0 |' in text
    assert '| T-D1 | 0 | 0 | — | — | — | — |' in text
    assert 'The deciding p of the McNemar row (T-A1) is the exact McNemar p ' \
        'on the learning success (p, p adj, p adj F3; reading and ' \
        'significant read p adj, significant_f3 p adj F3)' in norm


NEW_COLUMNS = ['median_diff_seed_lo', 'median_diff_seed_hi',
               'median_diff_crossed_lo', 'median_diff_crossed_hi',
               'seed_ci_includes_zero', 'crossed_ci_note', 'interval_note',
               'n_diverged_a', 'n_diverged_b', 'n_other_label_a',
               'n_other_label_b', 'other_labels_a', 'other_labels_b',
               'n_flag_whole_a', 'n_flag_whole_b', 'n_flag_const_a',
               'n_flag_const_b']
B_COLUMNS = ['n_nonfinite_cons_m0', 'n_nonfinite_cons_m15',
             'n_nonfinite_avg_in_m0', 'n_nonfinite_avg_in_m15', 'n_penalised',
             'n_evalonly_a', 'n_evalonly_b']
D_LEGEND = ('Seed interval: percentile bootstrap of the median paired '
            'difference over the 20 seeds, 10 000 draws, generator seed 0, '
            '95 %, not adjusted for the two contrasts, test recordings held '
            'fixed. Crossed interval: the same procedure, additionally '
            'resampling the five test recordings. Both intervals describe; '
            'the registered test decides at raw p < 0.025.')
WILCOXON_ONLY = ('median_diff', 'median_diff_lo', 'median_diff_hi',
                 'ci_units', 'd_z', 'p_a_gt_b', 'p_a_gt_b_lo', 'p_a_gt_b_hi',
                 'n_positive', 'ci_includes_zero')


def test_mcnemar_row_on_the_learning_success(tmp_path):
    """T-A1: the legacy arm learns on every seed, the reference on none
    (below 0.30 and at chance); the floor marking is informational. The
    exact McNemar p decides (also F3-wide); the row carries counts, rates
    and floor columns, no Wilcoxon-type column."""
    n = 20
    blocks, fits = f3_family_blocks(load_f3_spec(_world(
        tmp_path, values={'T-A1': ([0.6] * n, [0.1] * n)})))
    r = blocks.set_index('test').loc['T-A1']
    assert (r['method'], r['n10'], r['n01']) == ('mcnemar-exact', 20, 0)
    assert r['p_raw'] == pytest.approx(2 / 2 ** 20)
    assert r['p_adj_f3'] == pytest.approx(28 / 2 ** 20)
    assert r['significant'] and r['significant_f3']
    assert r['rate_a'].startswith('20/20') and r['rate_b'].startswith('0/20')
    assert r['rate_diff'] == 1.0
    assert (r['floor_b'], r['floor_bound'], r['floor_bound_arms']) == (
        '20/20', True, 'B')
    assert r['n_nonfinite'] == 0
    assert (r['status'], r['reading'], r['direction']) == ('tested', 'PASS',
                                                           'A>B')
    for k in ('p_perm', 'p_perm_adj', 'p_perm_adj_f3', 'verdict_flip',
              'w_plus_minus_w_minus', *WILCOXON_ONLY):
        assert pd.isna(r[k]), k
    cells = _report_cells(f3_report(blocks, fits), 'T-A1')
    for col in ('p perm', 'p perm adj', 'p perm adj F3',
                'median delta [95% CI]', 'd_z', 'P(A>B) [95% CI]',
                'seeds delta>0'):
        assert cells[col] == '—', col
    assert (cells['flags'], cells['floor_bound']) == ('A>B', 'YES (B)')
    # eight discordant pairs, all n10: exact p 2/256 passes the family cut
    # (m = 2) but not the F3-wide one (m_f3 = 14)
    r = _row(tmp_path / 'cut', 'T-A1',
             values={'T-A1': ([0.6] * n, [0.1] * 8 + [0.6] * 12)})
    assert (r['n10'], r['n01']) == (8, 0)
    assert r['p_adj'] == pytest.approx(4 / 256)
    assert r['p_adj_f3'] == pytest.approx(28 / 256)
    assert r['significant'] and not r['significant_f3']
    assert r['only_under_family_cut'] and r['reading'] == 'PASS'


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
    # one A/D fit shared by two tests; G2 ten seeds, E1 and E2 25 runs
    assert list(fits.columns) == F3_FIT_COLUMNS and len(fits) == (
        7 * 20 + 2 * 10 + 4 * 25)
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
    """F-9: five of twenty fits of arm B below the floor leave the
    exclusive C test readable, six make the arm floor-bound and the test
    reads floor-bound; the informational F test (ten seeds, three below) is
    bound, keeps PASS and reports the marking."""
    s20, s10 = _seeds('T-C1'), _seeds('T-F1')
    va = [21.0 + i for i in range(len(s20))]

    def vc(k):
        return [-1.0 if i < k else 20.0 for i in range(20)]

    vf = [-1.0 if i < 3 else 20.0 for i in range(10)]
    b = _blocks(_world(tmp_path, values={
        'T-C1': (va, vc(5)),
        'T-F1': ([21.0 + i for i in range(len(s10))], vf)}))
    c, f = b.loc['T-C1'], b.loc['T-F1']
    assert (c['floor_b'], c['floor_bound'], c['reading']) == (
        '5/20', False, 'PASS')
    assert (f['floor_b'], f['floor_bound'], f['reading']) == (
        '3/10', True, 'PASS')
    assert f['rate_b'].startswith('7/10') and f['n10'] == 3
    c = _row(tmp_path / 'six', 'T-C1', values={'T-C1': (va, vc(6))})
    assert (c['floor_b'], c['floor_bound'], c['floor_bound_arms'],
            c['reading']) == ('6/20', True, 'B', 'floor-bound')
    assert c['p_perm_adj'] < 0.05                  # still reported


def test_duffing_learning_success_at_floor_plus_two(tmp_path):
    """floor 10.5 on the synthetic Duffing draw: 12.5 succeeds, 12.4 not."""
    spec = load_f3_spec(_world(tmp_path))
    path = str(tmp_path / 'edge')
    _lv(path, 'duffing', 'lrc/ncp', 10, 12.5)
    _lv(path, 'duffing', 'lrc/ncp', 11, 12.4)
    arm = {**spec['T-F1']['b'], 'paths': [path]}
    df, _, _ = _f3_arm('T-F1', 'b', F3_TESTS['T-F1'], arm, [10, 11], {})
    assert df['floor'].tolist() == [10.5, 10.5]
    assert df['success'].tolist() == [True, False]


def test_incomplete_test_reports_its_statistics(tmp_path):
    r = _row(tmp_path, 'T-C1', skip={('T-C1', 'b'): [12]})
    assert (r['n_expected'], r['n_pairs'], r['status'], r['reading']) == (
        20, 19, 'incomplete', 'incomplete')
    assert r['p_raw'] == pytest.approx(2 / 2 ** 19)
    assert r['p_perm'] == pytest.approx(2 / 2 ** 19)
    assert not r['significant'] and not r['significant_f3']
    r = _row(tmp_path / 'empty', 'T-C1', skip={('T-C1', 'b'): _seeds('T-C1')})
    assert (r['n_pairs'], r['status'], r['reading']) == (0, 'incomplete',
                                                         'incomplete')
    assert (r['floor_b'], r['floor_bound']) == ('0/0', False)  # empty arm


def test_incomplete_reads_before_floor_bound(tmp_path):
    vb = [-1.0 if i < 6 else 20.0 for i in range(20)]
    r = _row(tmp_path, 'T-C1', skip={('T-C1', 'b'): [29]},
             values={'T-C1': ([21.0 + i for i in range(20)], vb)})
    assert (r['floor_b'], r['floor_bound'], r['status'], r['reading']) == (
        '6/19', True, 'incomplete', 'incomplete')


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


def test_per_arm_lr_is_reported_not_matched(tmp_path):
    """T-D2 matches epochs and param_budget; lr is no match field."""
    r = _row(tmp_path / 'd', 'T-D2', opts={('T-D2', 'b'): {'lr': 1e-2}})
    assert (r['lr_a'], r['lr_b'], r['reading']) == ('0.003', '0.01', 'PASS')


@pytest.mark.parametrize('test, field, value', [
    ('T-D1', 'epochs', 60), ('T-G1', 'param_budget', 4100),
    ('T-H2', 'param_budget', 31000)])
def test_d_g_h_match_epochs_and_budget(tmp_path, test, field, value):
    with pytest.raises(SystemExit,
                       match=rf'{test}: .*differ in {field}.*\[13\]'):
        _blocks(_world(tmp_path, REQUIRED + ['T-H1', 'T-H2'],
                       opts={(test, 'b'): {13: {'extra': {field: value}}}}))


def test_budget_exclusion(tmp_path):
    """An excluded row reports p_perm (Wilcoxon rows) and the family D
    intervals, but no other p, no significance, reading or interval note;
    the McNemar row T-A1 has no p_perm, excluded or not."""
    b = _blocks(_world(tmp_path, opts={
        (t, 'b'): {12: {'above': True}} for t in ('T-A1', 'T-C1', 'T-D1')}))
    r = b.loc['T-D1']
    assert r['status'] == EXCLUDED and pd.isna(r['reading'])
    assert pd.isna(r['p_raw']) and r['n_pairs'] == 20   # descriptive only
    assert r['p_perm'] == pytest.approx(2 / 2 ** 20)
    for k in ('p_adj', 'p_adj_f3', 'p_perm_adj', 'p_perm_adj_f3', 'method',
              'significant', 'significant_f3', 'interval_note',
              'crossed_ci_note', 'median_diff_lo', 'ci_units'):
        assert pd.isna(r[k]), k
    assert r['median_diff_seed_lo'] > 0
    assert (r['seed_ci_includes_zero'], r['ci_includes_zero']) == (False,
                                                                   False)
    assert r['median_diff_crossed_lo'] == r['median_diff_crossed_hi'] == 0
    c, a1 = b.loc['T-C1'], b.loc['T-A1']
    assert c['status'] == EXCLUDED and pd.isna(c['p_raw'])
    assert c['p_perm'] == pytest.approx(2 / 2 ** 20)
    assert pd.isna(c['median_diff_seed_lo'])                # D only
    assert a1['status'] == EXCLUDED and pd.isna(a1['p_perm'])
    assert pd.isna(a1['p_raw']) and a1['n10'] == 0


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
    # each arm counts the other's records as another label (#151)
    assert (r['n_other_label_a'], r['other_labels_a']) == (19,
                                                           'lrc/ncp (19)')
    assert (r['n_other_label_b'], r['other_labels_b']) == (
        19, 'lrc/ncp [optimizer=rmsprop_legacy] (19)')


def test_diverged_and_non_finite_fits_fail_and_enter_at_the_worst_value(
        tmp_path):
    spec = load_f3_spec(_world(tmp_path, opts={
        ('T-F1', 'b'): {10: {'unbounded': True},
                        11: {'extra': {'training_diverged': True}}},
        ('T-A1', 'b'): {10: {'extra': {'test_macro_f1_last': None}}}}))
    entry = F3_TESTS['T-F1']
    df, _, _ = _f3_arm('T-F1', 'b', entry, spec['T-F1']['b'],
                       _seeds('T-F1'), {})
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
    # diverged counts: the unbounded and the training_diverged fit (#151)
    assert (b.loc['T-F1', 'n_diverged_a'], b.loc['T-F1', 'n_diverged_b'],
            b.loc['T-A1', 'n_diverged_b']) == (0, 2, 1)
    with pytest.raises(SystemExit, match=r'T-D1: .*data key.*\[14\]'):
        _blocks(_world(tmp_path / 'dk', opts={
            ('T-D1', 'b'): {14: {'split_seed': 1}}}))


# --- one test row from frames ------------------------------------------------

def _frame(vals, above=False, width=False, task='lv', tdiv=False, best=1,
           curve=None, below=False):
    v = np.asarray(vals, float)
    return pd.DataFrame({
        'seed': range(10, 10 + len(v)), 'metric': v, 'task': task,
        'diverged': ~np.isfinite(v), 'training_diverged': tdiv,
        'below': below, 'budget_above_tolerance': above,
        'width_matched': width, 'data_key': 1, 'lr': 1e-3, 'clipnorm': None,
        'epochs': 300, 'param_budget': 4000, 'best_val_epoch': best,
        'val_loss_curve': [curve] * len(v)})


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
    assert r['verdict_flip'] and r['reading'] == 'FAIL'
    r = _frow([-d for d in range(1, 11)])
    assert (r['reading'], r['direction']) == ('PASS', 'A<B')
    assert r['w_plus_minus_w_minus'] == -55
    r = _frow([float('nan'), *range(2, 11)])
    assert r['n_nonfinite'] == 1 and r['n_positive'] == 9


def test_reading_follows_the_exact_permutation_p():
    """E15: with ties the p of wilcoxon_test (normal approximation) is
    reported and the exact permutation p decides, within the family and
    F3-wide (m_f3 = m = 2 here, so both cuts read the same p)."""
    r = _frow([2, 3, 3, -2, 1, 4, 2, 4, -1, 4], m_f3=2)
    assert r['method'] == 'approx'
    assert r['p_adj'] < 0.05 and r['p_adj_f3'] < 0.05
    assert r['p_perm_adj'] == r['p_perm_adj_f3'] == pytest.approx(52 / 1024)
    assert (r['reading'], r['significant'], r['significant_f3'],
            r['verdict_flip']) == ('FAIL', False, False, True)
    tied = [-2, -1, 1, 1, 3, 3, 3, 3, 3, 4]
    r = _frow(tied, m_f3=2)
    assert r['method'] == 'approx'
    assert r['p_adj'] >= 0.05 and r['p_adj_f3'] >= 0.05
    assert r['p_perm_adj'] == r['p_perm_adj_f3'] == pytest.approx(48 / 1024)
    assert (r['reading'], r['direction'], r['significant'],
            r['significant_f3'], r['verdict_flip']) == (
        'PASS', 'A>B', True, True, True)
    cells = _report_cells(f3_report(
        pd.DataFrame([r]).reindex(columns=F3_COLUMNS),
        pd.DataFrame(columns=F3_FIT_COLUMNS)), 'T-C1')
    assert cells['flags'] == 'A>B; p of wilcoxon_test would decide otherwise'
    assert cells['p perm adj F3'] == format(48 / 1024, '.4g')
    r = _frow(tied)                                   # m_f3 = 8
    assert r['p_perm_adj_f3'] == pytest.approx(192 / 1024)
    assert not r['significant_f3'] and r['only_under_family_cut']


def test_budget_exclusion_spares_only_the_width_matched_arm():
    r = _frow(range(1, 11), b={'above': True, 'width': True})
    assert (r['status'], r['reading']) == ('tested', 'PASS')
    r = _frow(range(1, 11), b={'above': True})
    assert (r['status'], r['reading']) == (EXCLUDED, None)


def test_one_sided_h_test(tmp_path):
    tests = REQUIRED + ['T-H1', 'T-H2']
    b = _blocks(_world(tmp_path, tests))
    r = b.loc['T-H1']
    assert (r['side'], r['m'], r['m_f3']) == ('greater', 2, 16)
    assert r['p_raw'] == pytest.approx(1 / 1024)
    assert r['p_perm'] == pytest.approx(1 / 1024)
    assert r['p_adj_f3'] == pytest.approx(16 / 1024)
    assert (r['reading'], r['direction']) == ('PASS', 'A>B')
    rev = [20.0 - 0.1 * (i + 1) for i in range(10)]
    r = _row(tmp_path / 'rev', 'T-H1', tests,
             values={'T-H1': (rev, [20.0] * 10)})
    assert r['p_raw'] == pytest.approx(1.0) and r['p_perm'] == 1.0
    assert r['reading'] == 'FAIL'


# --- descriptive columns (#151) ---------------------------------------------

CM = np.array(list(GOOD.values()), float)      # (recordings, C, C) of _pa


def _dframe(vals, cms=None, **kw):
    f = _frame(vals, task='pa', **kw)
    f['confusion'] = pd.Series(cms or [CM] * len(f), index=f.index,
                               dtype=object)
    return f


def _drow(delta, a=None, b=None, n_listed=20):
    """_f3_test of T-D1 (m = 2) with B = 0.6 and A = 0.6 + delta on seeds
    10, 11, ..., every fit with the confusions CM unless cms is given."""
    return _f3_test('T-D1', F3_TESTS['T-D1'], 2, 8,
                    _dframe([0.6 + d for d in delta], **(a or {})),
                    _dframe([0.6] * len(delta), **(b or {})),
                    list(range(10, 10 + n_listed)), {'a': [], 'b': []})


# one delta vector per interval_note branch (reading, seed interval holds 0)
NOTE_DELTAS = {
    ('FAIL', True): [0.01 * (i + 1) * (-1) ** i for i in range(20)],
    ('FAIL', False): [0.001 * k for k in range(1, 16)]
    + [-(0.1 + 0.01 * k) for k in range(5)],
    ('PASS', True): [-0.0001 * k for k in range(1, 8)]
    + [0.1 + 0.01 * k for k in range(13)],
    ('PASS', False): [0.01 * (i + 1) for i in range(20)]}


def test_family_d_null_flag_reads_the_seed_interval(tmp_path):
    """E17: T-D1 with A = B + 0.01 (i + 1): the seed interval of the paired
    differences excludes 0, the crossed interval of the equal confusion
    matrices is [0, 0] and contains it. The null flag, ci_includes_zero and
    the report flag read the seed interval; the old interval columns stay
    empty; the PASS row with the seed interval off 0 has no note."""
    blocks, fits = f3_family_blocks(load_f3_spec(_world(tmp_path)))
    r = blocks.set_index('test').loc['T-D1']
    va = np.array([0.6 + 0.01 * (i + 1) for i in range(20)])
    lo, hi = bootstrap_median_ci(va - 0.6, N_BOOT)
    cms = np.stack([CM] * 20)
    assert lo > 0
    assert (r['median_diff_seed_lo'], r['median_diff_seed_hi']) == (lo, hi)
    assert (r['median_diff_crossed_lo'], r['median_diff_crossed_hi']) == \
        crossed_median_diff_ci(cms, cms, N_BOOT) == (0.0, 0.0)
    assert r['median_diff'] == pytest.approx(np.median(va - 0.6))
    assert (r['seed_ci_includes_zero'], r['ci_includes_zero']) == (False,
                                                                   False)
    assert r['reading'] == 'PASS'
    for k in ('median_diff_lo', 'median_diff_hi', 'ci_units',
              'interval_note', 'crossed_ci_note'):
        assert pd.isna(r[k]), k
    cells = _report_cells(f3_report(blocks, fits), 'T-D1')
    assert cells['median delta [95% CI]'] == (
        f"{r['median_diff']:+.4f} (seed [{lo:+.4f}, {hi:+.4f}]; crossed "
        '[+0.0000, +0.0000])')
    assert cells['flags'] == 'A>B'                  # no "interval includes 0"


@pytest.mark.parametrize('reading, zero', list(NOTE_DELTAS))
def test_interval_note_branches(reading, zero):
    r = _drow(NOTE_DELTAS[(reading, zero)])
    assert (r['reading'], r['seed_ci_includes_zero'],
            r['ci_includes_zero']) == (reading, zero, zero)
    lo, hi = r['median_diff_seed_lo'], r['median_diff_seed_hi']
    assert (lo <= 0 <= hi) == zero
    note = r.get('interval_note')
    assert note == {
        ('FAIL', True): 'not shown, no equivalence; median paired '
        f'differences outside [{lo:+.4f}, {hi:+.4f}] macro-F1 (unadjusted '
        '95 % seed interval) are not compatible with these seeds on this '
        'split',
        ('FAIL', False): 'not shown, no equivalence; The seed interval '
        '(unadjusted 95 %) excludes 0, but the registered test is not '
        'significant at its threshold (exact p ≥ 0.025). The test decides.',
        ('PASS', True): 'The seed interval (unadjusted 95 %) contains 0, '
        'while the registered test is significant at its threshold (exact '
        'p < 0.025). The test decides.',
        ('PASS', False): None}[(reading, zero)]
    text = f3_report(pd.DataFrame([r]).reindex(columns=F3_COLUMNS),
                     pd.DataFrame(columns=F3_FIT_COLUMNS))
    assert ('interval includes 0' in _report_cells(text, 'T-D1')['flags']) \
        == zero
    assert [ln for ln in text.splitlines() if ln.startswith('- T-D1:')] == (
        [f'- T-D1: {note}'] if note else [])


def test_null_flag_reads_the_unrounded_seed_bounds():
    """Deltas of order 1e-6 (15 above 0, five larger ones below): the seed
    interval lies within 5e-5 above 0, so bounds rounded to four decimals
    would hold 0; the unrounded flag is False and the FAIL row takes the
    'excludes 0' branch."""
    r = _drow([1e-6 * k for k in range(1, 16)]
              + [-(1e-4 + 1e-5 * k) for k in range(5)])
    lo, hi = r['median_diff_seed_lo'], r['median_diff_seed_hi']
    assert 0 < lo < hi < 5e-5 and round(lo, 4) == round(hi, 4) == 0
    assert (r['reading'], r['seed_ci_includes_zero'],
            r['ci_includes_zero']) == ('FAIL', False, False)
    assert r['interval_note'].startswith(
        'not shown, no equivalence; The seed interval (unadjusted 95 %) '
        'excludes 0')


def test_interval_note_is_empty_without_pass_or_fail():
    """floor-bound, incomplete and excluded D rows keep the median, both
    intervals and p_perm, without a note."""
    d = NOTE_DELTAS[('FAIL', True)]
    for r, reading in (
            (_drow(d, b={'below': [True] * 6 + [False] * 14}), 'floor-bound'),
            (_drow(d[:19]), 'incomplete'),
            (_drow(d, b={'above': True}), None)):
        assert r['reading'] == reading and pd.isna(r.get('interval_note'))
        assert r['p_perm'] == pytest.approx(permutation_p(
            np.array([0.6 + v for v in d[:r['n_pairs']]]),
            np.full(r['n_pairs'], 0.6)))
        assert np.isfinite([r['median_diff'], r['median_diff_seed_lo'],
                            r['median_diff_crossed_lo']]).all()


def test_crossed_interval_not_available(tmp_path):
    """One diverged fit (non-finite test_macro_f1) in a pair: no crossed
    interval and no fallback onto the seed interval; the note names the
    seed; the fit enters the seed interval at the worst value. Confusion
    matrices of different shapes are the other reason."""
    blocks, fits = f3_family_blocks(load_f3_spec(_world(tmp_path, opts={
        ('T-D1', 'b'): {12: {'extra': {'test_macro_f1': None}}}})))
    r = blocks.set_index('test').loc['T-D1']
    note = 'crossed interval not available: diverged fits: seed 12 (B)'
    assert r['crossed_ci_note'] == note
    assert pd.isna(r['median_diff_crossed_lo']) and pd.isna(
        r['median_diff_crossed_hi'])
    va = np.array([0.6 + 0.01 * (i + 1) for i in range(20)])
    vb = np.full(20, 0.6)
    vb[2] = 0.0                                     # seed 12 at the worst
    lo, hi = bootstrap_median_ci(va - vb, N_BOOT)
    assert (r['median_diff_seed_lo'], r['median_diff_seed_hi']) == (lo, hi)
    assert (r['n_nonfinite'], r['n_diverged_b']) == (1, 1)
    text = f3_report(blocks, fits)
    assert f'- T-D1: {note}' in text.splitlines()
    assert _report_cells(text, 'T-D1')['median delta [95% CI]'] == (
        f"{r['median_diff']:+.4f} (seed [{lo:+.4f}, {hi:+.4f}]; crossed —)")
    three = np.stack([np.eye(3)] * 3)
    d = NOTE_DELTAS[('PASS', False)]
    r = _drow(d, b={'cms': [CM] * 19 + [three]})
    assert r['crossed_ci_note'] == ('crossed interval not available: '
                                    'confusion matrices of different shapes')
    r = _drow([float('nan'), *d[1:]], b={'cms': [CM] * 19 + [three]})
    assert r['crossed_ci_note'] == (
        'crossed interval not available: diverged fits: seed 10 (A); '
        'confusion matrices of different shapes')
    assert np.isnan([r['median_diff_crossed_lo'],
                     r['median_diff_crossed_hi']]).all()


def test_training_diverged_fit_keeps_its_value_and_counts_as_diverged(
        tmp_path):
    """A training_diverged fit with a finite endpoint enters at its
    best-epoch value (as in F1): it counts in n_diverged_b, not in
    n_nonfinite. n_diverged counts every fit of the arm on the seed list,
    paired or not; a record without training_diverged reads False."""
    s = _seeds('T-C1')
    va = [21.0 + i for i in range(20)]
    va[2] = 10.0                                    # seed 12: A < B
    spec = load_f3_spec(_world(
        tmp_path, values={'T-C1': (va, [20.0] * 20)},
        opts={('T-C1', 'b'): {12: {'extra': {'training_diverged': True}},
                              13: {'unbounded': True}}},
        skip={('T-C1', 'a'): [13]}))
    df, _, _ = _f3_arm('T-C1', 'b', F3_TESTS['T-C1'], spec['T-C1']['b'], s,
                       {})
    df = df.set_index('seed')
    assert df['training_diverged'].dtype == bool
    assert df['training_diverged'].tolist() == [x == 12 for x in df.index]
    assert (df.loc[12, 'diverged'], df.loc[12, 'metric']) == (False, 20.0)
    assert df.loc[13, 'diverged'] and 'val_loss_curve' in df
    pa, _, _ = _f3_arm('T-D1', 'b', F3_TESTS['T-D1'], spec['T-D1']['b'],
                       _seeds('T-D1'), {})
    assert 'val_macro_f1_curve' in pa and not pa['training_diverged'].any()
    r = f3_family_blocks(spec)[0].set_index('test').loc['T-C1']
    assert (r['n_diverged_a'], r['n_diverged_b'], r['n_nonfinite']) == (0, 2,
                                                                        0)
    # 19 pairs (seed 13 has no A); seed 12 pairs at A - B = -10
    assert (r['n_pairs'], r['n_positive']) == (19, 18)


def test_descriptive_counts_enter_no_reading():
    """Diverged and flag counts on every fit leave reading, floor_bound,
    significance and the statistics as they are (a PASS and a FAIL row)."""
    down, seen = list(np.linspace(1.0, 0.1, 300)), set()
    for delta in (list(range(1, 11)), [2, 3, 3, -2, 1, 4, 2, 4, -1, 4]):
        base = _frow(delta)
        seen.add(base['reading'])
        busy = _frow(delta, a={'tdiv': True}, b={'tdiv': True, 'best': 300,
                                                 'curve': down})
        assert (base['n_diverged_b'], base['n_flag_whole_b'],
                base['n_flag_const_b']) == (0, 0, 0)
        assert (busy['n_diverged_a'], busy['n_diverged_b'],
                busy['n_flag_whole_b'], busy['n_flag_const_b']) == (
            10, 10, 10, 10)
        for k in ('reading', 'floor_bound', 'significant', 'significant_f3',
                  'p_perm_adj', 'p_adj', 'n_nonfinite', 'n_positive',
                  'median_diff'):
            assert busy[k] == base[k], k
    assert seen == {'PASS', 'FAIL'}


def test_other_labels_are_counted_named_and_not_used(tmp_path):
    """Records under an arm's paths of the test's task and system with an
    F3 stamp and another arm label are counted and named; records of
    another system or task, and non-F3 records, are not counted."""
    spec = load_f3_spec(_world(tmp_path))
    folder = os.path.join(spec['T-C1']['a']['paths'][0], 'mixed')
    for seed in (10, 11, 12):
        _lv(folder, 'periodic_predator_prey', 'gru/ncp', seed, -50.0)
    _lv(folder, 'periodic_predator_prey', 'gru/ncp', 100, -50.0,
        name='tuning.json')                         # off the seed list
    _lv(folder, 'periodic_predator_prey', 'cfc_lrc/ncp', 10, -50.0,
        epochs=600, name='long.json')
    _lv(folder, 'periodic_predator_prey', 'gru/ncp', 13, -50.0, pv=F1,
        name='f1.json')
    _lv(folder, 'duffing', 'gru/ncp', 10, -50.0, name='duffing.json')
    _pa(folder, 'gru/ncp', 10, 0.1, name='pa.json')
    r = f3_family_blocks(spec)[0].set_index('test').loc['T-C1']
    assert (r['n_other_label_a'], r['other_labels_a']) == (
        5, 'cfc_lrc/ncp [epochs=600] (1); gru/ncp (4)')
    assert r['n_other_label_b'] == 0 and pd.isna(r['other_labels_b'])
    assert r['n_pairs'] == 20 and r['p_raw'] == pytest.approx(2 / 2 ** 20)
    assert r['n_outside_list'] == 0         # outside = the arm's own label
    with open(tmp_path / 'spec2.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(spec, f)
    out = str(tmp_path / 'out')
    assert main(['--runs', str(tmp_path / 'none'), '--out', out,
                 '--f3-families', str(tmp_path / 'spec2.yaml')]) == 0
    with open(os.path.join(out, 'summary.md'), encoding='utf-8') as f:
        lines = f.read().splitlines()
    assert '- T-C1 a: cfc_lrc/ncp [epochs=600] (1); gru/ncp (4)' in lines
    assert '- T-C1 b: none' in lines
    raw = pd.read_csv(os.path.join(out, 'f3_families.csv'),
                      dtype=str).set_index('test')
    assert raw.loc['T-C1', 'n_other_label_a'] == '5'


def _curve(at, n=300, low=0.1):
    """val_loss_curve of n epochs, V-shaped with its minimum at the 1-based
    epoch at."""
    return [low + abs(e - at) / n for e in range(1, n + 1)]


def test_convergence_flags_on_c_and_f(tmp_path):
    """F-5 on T-C1 (300 epochs: flagged from epoch 285, constant phase
    epochs 151-300). Arm A is a clean fixture (best_val_epoch = the 1-based
    argmin of the whole curve): n_flag_whole equals the count from that
    argmin and n_flag_const agrees. Arm B has one fit whose whole-curve
    minimum lies in the ramp phase and whose constant-phase minimum is
    flagged: it counts in n_flag_const only."""
    holes = _curve(300)
    holes[295:] = [None] * 5                        # ignored: argmin 295
    shapes = [_curve(300), _curve(100), _curve(285), _curve(284), holes]
    curves_a = [shapes[i % 5] for i in range(20)]
    best_a = [int(np.nanargmin(np.array(c, float))) + 1 for c in curves_a]
    ramp = [1.0] * 150 + list(np.linspace(0.5, 0.1, 150))
    ramp[49] = 0.01                                 # whole minimum: epoch 50
    curves_b = [ramp] + [_curve(100)] * 19
    best_b = [50] + [100] * 19
    opts = {('T-C1', s): {seed: {'extra': {'val_loss_curve': c,
                                           'best_val_epoch': e}}
                          for seed, c, e in zip(_seeds('T-C1'), cs, bs)}
            for s, cs, bs in (('a', curves_a, best_a),
                              ('b', curves_b, best_b))}
    blocks, fits = f3_family_blocks(load_f3_spec(_world(tmp_path,
                                                        opts=opts)))
    b = blocks.set_index('test')
    r = b.loc['T-C1']
    whole = sum(e >= CONVERGED * 300 for e in best_a)
    assert whole == 12                              # 300, 285, 295 x 4
    assert (r['n_flag_whole_a'], r['n_flag_const_a']) == (whole, whole)
    assert (r['n_flag_whole_b'], r['n_flag_const_b']) == (0, 1)
    assert r['reading'] == 'PASS'
    assert (b.loc['T-F1', 'n_flag_whole_a'], b.loc['T-F1', 'n_flag_const_a']) \
        == (0, 0)
    for t in ('T-A1', 'T-A2', 'T-D1', 'T-D2'):
        assert pd.isna(b.loc[t, 'n_flag_whole_a']), t
    assert '| T-C1 | 0 | 0 | 12 | 0 | 12 | 1 |' in f3_report(blocks, fits)


def test_flags_count_unpaired_fits_and_stay_empty_outside_c_and_f(tmp_path):
    """Arm B of T-C1 has a flagged fit on seed 29, where arm A has no
    record: it counts in both flags although it is not paired. The LV rows
    of families G and H leave the flag columns empty."""
    b = _blocks(_world(
        tmp_path, REQUIRED + ['T-H1', 'T-H2'],
        opts={('T-C1', 'b'): {29: {'extra': {'val_loss_curve': _curve(300),
                                             'best_val_epoch': 300}}}},
        skip={('T-C1', 'a'): [29]}))
    r = b.loc['T-C1']
    assert (r['n_pairs'], r['n_flag_whole_a'], r['n_flag_const_a'],
            r['n_flag_whole_b'], r['n_flag_const_b']) == (19, 0, 0, 1, 1)
    for t in ('T-G1', 'T-G2', 'T-H1', 'T-H2'):
        for c in NEW_COLUMNS[-4:]:
            assert pd.isna(b.loc[t, c]), (t, c)


def test_const_phase_window_edges():
    """Window epochs 11-20 of 20 (flagged from 19); 21 epochs: 11-21
    (flagged from 19.95, i.e. 20). Non-finite values are ignored; a window
    without a finite value, or none at all, is not flagged."""
    def at(k, n=20, fill=1.0):
        c = [fill] * n
        c[k - 1] = 0.1
        return c

    assert _const_phase_flagged(at(19), 20)
    assert not _const_phase_flagged(at(18), 20)
    assert not _const_phase_flagged(at(5), 20)      # ramp minimum ignored
    late = at(20, fill=1.0)
    late[19] = 0.2
    late[10] = 0.1              # epoch 11 opens the window: its argmin
    assert not _const_phase_flagged(late, 20)
    late[10], late[9] = 1.0, 0.1                    # epoch 10 lies outside
    assert _const_phase_flagged(late, 20)
    assert _const_phase_flagged(at(20, 21), 21)
    assert not _const_phase_flagged(at(19, 21), 21)
    assert _const_phase_flagged(at(20, fill=float('inf')), 20)
    assert _const_phase_flagged([1.0] * 18 + [0.5, float('nan')], 20)
    assert _const_phase_flagged([1.0] * 18 + [0.5, None], 20)
    assert not _const_phase_flagged([1.0] * 10 + [None] * 10, 20)
    assert not _const_phase_flagged([1.0] * 10, 20)
    assert not _const_phase_flagged(None, 20)
    assert _const_phase_flagged(np.array(at(19)), 20)    # array curve
    assert not _const_phase_flagged(np.array([]), 20)
    for epochs in (None, float('nan')):                 # as _flagged
        assert not _const_phase_flagged(at(19), epochs)


# --- spec ---------------------------------------------------------------------

def test_spec_rules(tmp_path):
    spec = _world(tmp_path, REQUIRED + ['T-H1', 'T-H2'])
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

    assert set(check(lambda s: [s.pop(t) for t in ('T-H1', 'T-H2')],
                     None)) == set(REQUIRED)
    check(lambda s: s.pop('T-H2'), 'left out only together')
    check(lambda s: [s.pop(t) for t in ('T-G1', 'T-G2')],
          r"missing required tests \['T-G1', 'T-G2'\]")
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


def test_spec_seed_list_must_equal_the_family_list(tmp_path):
    """F-1: a test of a family with a registry list names exactly those
    seeds (any order); H has none."""
    assert {f: list(r) for f, r in F3_SEEDS.items()} == {
        **{f: list(range(10, 30)) for f in 'ABCD'},
        'E': list(range(10, 35)), 'F': list(range(10, 20)),
        'G': list(range(10, 20))}
    with open(_world(tmp_path, REQUIRED + ['T-H1', 'T-H2']),
              encoding='utf-8') as f:
        good = yaml.safe_load(f)

    def load(mutate):
        s = json.loads(json.dumps(good))
        mutate(s)
        path = str(tmp_path / 'seeds.yaml')
        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(s, f)
        return load_f3_spec(path)

    for mutate, match in (
            (lambda s: s['T-C1']['seeds'].pop(),
             r'T-C1\.seeds .*family C.*missing \[29\], extra \[\]'),
            (lambda s: s['T-F2']['seeds'].extend([20, 9]),
             r'T-F2\.seeds .*family F.*missing \[\], extra \[9, 20\]'),
            (lambda s: s['T-A1'].__setitem__('seeds', [0, 1, *range(12, 30)]),
             r'T-A1\.seeds .*missing \[10, 11\], extra \[0, 1\]')):
        with pytest.raises(SystemExit, match=match):
            load(mutate)
    spec = load(lambda s: (s['T-D1']['seeds'].reverse(),
                           s['T-H1'].__setitem__('seeds', [0, 1, 2])))
    assert spec['T-D1']['seeds'][0] == 29 and spec['T-H1']['seeds'] == [0, 1,
                                                                         2]


def test_template_lists_every_registered_test_with_placeholders():
    """The template lists the registered tests whose status (test or
    family) is not optional, every path a <..._RUN_DIR>/results placeholder;
    family A one path per seed half, family B one run folder, family E one
    run folder per fold (T-E3 optional, not listed)."""
    path = os.path.join(REPO, 'configs/aggregate/f3_families.yaml')
    with open(path, encoding='utf-8') as f:
        t = yaml.safe_load(f)
    listed = [test for test, e in F3_TESTS.items()
              if 'optional' not in (e['status'],
                                    F3_FAMILIES[e['family']]['status'])]
    assert list(t) == listed and 'T-H1' not in t and 'T-B1' in t
    assert 'T-E3' not in t and {'T-E1', 'T-G1'} <= set(t)
    for test in listed:
        e = F3_TESTS[test]
        assert [t[test][s]['label'] for s in 'ab'] == [e['a'], e['b']]
        assert t[test]['seeds'] == _seeds(test)
        assert all(p.startswith('<') and p.endswith('/results')
                   for s in 'ab' for p in t[test][s]['paths'])
        if e['family'] == 'A':
            assert all(len(t[test][s]['paths']) == 2 for s in 'ab')
        if e['family'] == 'E':
            assert all(t[test][s]['paths'] == [
                f'<F3_E_PA_FOLD{k}_RUN_DIR>/results' for k in range(5)]
                for s in 'ab')
    assert {p for test in ('T-B1', 'T-B2') for s in 'ab'
            for p in t[test][s]['paths']} == {'<F3_B_CO_K30_RUN_DIR>/results'}
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


# --- run folders and folds (#159) ---------------------------------------------

# the wandb summary of an offline Person Activity run: test_macro_f1, no
# cell, wiring, seed or config
SUMMARY = {'_step': 49, 'best_val_epoch': 12, 'val_macro_f1': 0.5,
           'test_macro_f1': 0.5, 'test_accuracy': 0.6}
OFFLINE = 'offline-run-20261122_000000-abcd1234'


def _run_folder(run):
    """Next to <run>/results: wandb/wandb/<offline run>/files/
    wandb-summary.json and the latest-run link (found twice by a recursive
    glob, as on the cluster)."""
    w = os.path.join(run, 'wandb', 'wandb')
    _dump(os.path.join(w, OFFLINE, 'files', 'wandb-summary.json'), SUMMARY)
    os.symlink(OFFLINE, os.path.join(w, 'latest-run'))
    return run


def test_f3_run_folders_equal_their_results_subfolders(tmp_path):
    """Every Person Activity arm listed as its run folder (records under
    results/, wandb summaries next to them) gives the frames of the same
    spec on the results/ subfolders."""
    runs = load_f3_spec(_world(tmp_path))
    on_results = copy.deepcopy(runs)
    for test in (t for t in REQUIRED if F3_TESTS[t]['task'] == 'pa'):
        for side in 'ab':
            p = runs[test][side]['paths'][0]
            os.rename(p, p + '.tmp')
            os.makedirs(p)
            os.rename(p + '.tmp', os.path.join(p, 'results'))
            _run_folder(p)
            on_results[test][side]['paths'] = [os.path.join(p, 'results')]
    for got, want in zip(f3_family_blocks(runs),
                         f3_family_blocks(on_results)):
        pd.testing.assert_frame_equal(got, want)


def test_f1_and_f2_loaders_skip_wandb_summaries(tmp_path):
    """load_sequence_runs and _f2_arm read a run folder as its results/;
    _f2_arm no longer rests on the seed check running first."""
    run = str(tmp_path / 'run')
    res = os.path.join(run, 'results')
    for seed in (0, 1, 10):
        _pa(res, 'lrc/ncp', seed, 0.5, pv=F2)
    _run_folder(run)
    pd.testing.assert_frame_equal(load_sequence_runs(run, 'pa'),
                                  load_sequence_runs(res, 'pa'))
    arm = {'label': 'lrc/ncp', 'protocol_version': [F2]}
    (got, _), (want, _) = (_f2_arm('t', 'b', 'pa', {**arm, 'paths': [p]}, {})
                           for p in (run, res))
    assert len(got) == 2
    pd.testing.assert_frame_equal(got, want)


@pytest.mark.parametrize('half, missing', [
    ({'cell': 'lrc'}, r"\['wiring'\]"),
    ({'config': {'effective': {}}}, r"\['cell', 'wiring'\]")])
def test_pa_record_without_wiring_names_the_file(tmp_path, half, missing):
    """Also in _f2_arm, where the record has no seed: the helper runs
    before the seed filter."""
    run = str(tmp_path / 'run')
    _pa(os.path.join(run, 'results'), 'lrc/ncp', 10, 0.5)
    _dump(os.path.join(run, 'half.json'), {**SUMMARY, **half})
    arm = {'label': 'lrc/ncp', 'protocol_version': [F3], 'paths': [run]}
    with pytest.raises(SystemExit, match=r'half\.json.*' + missing):
        _f3_arm('T-A1', 'b', F3_TESTS['T-A1'], arm, _seeds('T-A1'), {})
    with pytest.raises(SystemExit, match=r'half\.json.*' + missing):
        load_sequence_runs(run, 'pa')
    with pytest.raises(SystemExit, match=r'half\.json.*' + missing):
        _f2_arm('t', 'b', 'pa', arm, {})


def _fold_arm(tmp_path, folds):
    """A T-A2.b arm with one record per (seed, split_fold); split_fold
    'none' leaves the field out, ('eff', f) sets it in config.effective
    only."""
    path = str(tmp_path / 'b')
    label = F3_TESTS['T-A2']['b']
    for i, (seed, fold) in enumerate(folds):
        _pa(path, label, seed, 0.6, name=f'r{i}.json', extra=(
            {} if fold == 'none' or isinstance(fold, tuple)
            else {'split_fold': fold}))
        if isinstance(fold, tuple):
            with open(os.path.join(path, f'r{i}.json'), encoding='utf-8') as f:
                rec = json.load(f)
            rec['config']['effective']['split_fold'] = fold[1]
            _dump(os.path.join(path, f'r{i}.json'), rec)
    return {'label': label, 'protocol_version': [F3], 'paths': [path]}


@pytest.mark.parametrize('seed, fold', [(11, 2), (11, 0), (3, 2),
                                        (11, ('eff', 1))])
def test_fold_record_in_a_test_without_folds_names_the_file(tmp_path, seed,
                                                            fold):
    arm = _fold_arm(tmp_path, [(10, None), (seed, fold)])
    f = fold[1] if isinstance(fold, tuple) else fold
    with pytest.raises(SystemExit, match=rf'r1\.json: split_fold {f} in '
                                         r'T-A2\.b'):
        _f3_arm('T-A2', 'b', F3_TESTS['T-A2'], arm, _seeds('T-A2'), {})


def test_null_or_missing_fold_loads_and_fold_keys_accept_folds(tmp_path):
    seeds = _seeds('T-A2')
    df, _, _ = _f3_arm('T-A2', 'b', F3_TESTS['T-A2'],
                       _fold_arm(tmp_path, [(10, None), (11, 'none')]),
                       seeds, {})
    assert df['seed'].tolist() == [10, 11]
    assert df['split_fold'].tolist() == [None, None]
    df, _, _ = _f3_arm('T-A2', 'b', F3_TESTS['T-A2'],
                       _fold_arm(tmp_path / 'e', [(10, 2), (10, 3), (11, 0)]),
                       seeds, {}, key=('split_fold', 'seed'))
    assert df[['split_fold', 'seed']].values.tolist() == [[2, 10], [3, 10],
                                                         [0, 11]]


# --- family B (#132) ----------------------------------------------------------

B_SEEDS = list(range(10, 30))


def _bf(m0=0.7, m15=0.6):
    """test_macro_f1_by_fault over the grid 0, 3, 9, 15, 20."""
    return [m0, 0.6, 0.5, m15, 0.3]


def _b_opts(test, side, by_seed, **kw):
    return {(test, side): {s: {'by_fault': v, **kw}
                           for s, v in by_seed.items()}}


def test_family_b_rows_of_the_default_world(tmp_path):
    """A = B + 0.01 (i + 1) at m = 15 on every test: one-sided PASS with
    the exact p 1 / 2^20, no floor columns; T-B1 carries the crossed
    interval of the equal confusions, T-B2 the seed interval of delta."""
    b = _blocks(_world(tmp_path))
    delta = np.array([(0.6 + 0.01 * (i + 1) - 0.6) - (0.7 - 0.7)
                      for i in range(20)])
    for t in ('T-B1', 'T-B2'):
        r = b.loc[t]
        assert (r['family'], r['task'], r['m'], r['m_f3'], r['side'],
                r['floor_rule']) == ('B', 'co', 2, 14, 'greater', 'none')
        assert (r['n_pairs'], r['n_expected'], r['status'], r['reading'],
                r['direction']) == (20, 20, 'tested', 'PASS', 'A>B')
        assert r['p_perm'] == pytest.approx(1 / 2 ** 20)
        assert r['p_perm_adj'] == pytest.approx(2 / 2 ** 20)
        assert r['p_perm_adj_f3'] == pytest.approx(14 / 2 ** 20)
        assert (r['p_raw'], r['method']) == (pytest.approx(1 / 2 ** 20),
                                             'exact')
        assert r['n_positive'] == 20 and not r['median_sign_differs']
        for c in ('floor_a', 'floor_b', 'floor_bound', 'floor_bound_arms'):
            assert pd.isna(r[c]), (t, c)
        assert (r['n_diverged_a'], r['n_diverged_b']) == (0, 0)
    r = b.loc['T-B1']
    assert r['ci_units'] == 'seed x recording'
    assert (r['median_diff_lo'], r['median_diff_hi']) == (0.0, 0.0)
    assert r['ci_includes_zero']
    r = b.loc['T-B2']
    assert r['median_diff'] == pytest.approx(np.median(delta))
    assert (r['median_diff_lo'], r['median_diff_hi']) == \
        bootstrap_median_ci(delta, N_BOOT)
    assert r['ci_units'] == 'seed (d15 - d0, non-finite as 0)'
    assert not r['ci_includes_zero'] and r['n_penalised'] == 0
    assert 'floor-bound' not in set(b['reading'])


def test_t_b1_exact_one_sided_p_at_twenty_seeds(tmp_path):
    """cons - fault_mean = 0.01 k at m = 15, k = 1..20, negative for k = 1
    and 2: W+ = 207, W- = 3, p = P(W+ >= 207) = 5 / 2^20 (the subsets of
    1..20 with a sum <= 3)."""
    k = np.arange(1, 21)
    d = np.where(k <= 2, -1, 1) * 0.01 * k
    r = _row(tmp_path, 'T-B1', values={'T-B1': (list(0.5 + d), [0.5] * 20)})
    assert r['p_perm'] == pytest.approx(5 / 2 ** 20)
    assert (r['p_raw'], r['method']) == (pytest.approx(5 / 2 ** 20), 'exact')
    assert r['p_perm_adj'] == pytest.approx(10 / 2 ** 20)
    assert r['p_perm_adj_f3'] == pytest.approx(70 / 2 ** 20)
    assert r['w_plus_minus_w_minus'] == 204
    assert (r['reading'], r['direction'], r['n_positive']) == ('PASS', 'A>B',
                                                               18)


def test_t_b2_non_finite_cons_enters_as_the_penalty(tmp_path):
    """Seed 10: cons non-finite at m = 0. Under the 0 rule its delta is
    (0.61 - 0.6) - (0 - 0.7) = 0.71, the largest of all; the tested vector
    holds -3 there instead, the highest rank with a negative sign."""
    r = _row(tmp_path, 'T-B2',
             opts=_b_opts('T-B2', 'a', {10: _bf(None, 0.61)}))
    zero_rule = np.array([0.71] + [0.01 * (i + 1) for i in range(1, 20)])
    tested = np.array([-3.0, *zero_rule[1:]])
    assert (r['n_penalised'], r['n_nonfinite_cons_m0'],
            r['n_nonfinite_cons_m15'], r['n_nonfinite_avg_in_m0'],
            r['n_nonfinite_avg_in_m15']) == (1, 1, 0, 0, 0)
    assert r['median_diff'] == pytest.approx(np.median(zero_rule))
    p = permutation_p(tested, np.zeros(20), 'greater')
    assert r['p_perm'] == pytest.approx(p)
    assert p != permutation_p(zero_rule, np.zeros(20), 'greater')
    assert r['p_raw'] == pytest.approx(wilcoxon_test(
        tested, np.zeros(20), 'greater')[0])
    assert r['w_plus_minus_w_minus'] == 190 - 20
    assert (r['n_diverged_a'], r['n_diverged_b'], r['n_nonfinite'],
            r['n_positive']) == (1, 0, 1, 20)


def test_t_b2_non_finite_avg_in_enters_as_the_penalty(tmp_path):
    r = _row(tmp_path, 'T-B2',
             opts=_b_opts('T-B2', 'b', {13: _bf(0.7, float('nan'))}))
    assert (r['n_penalised'], r['n_nonfinite_cons_m0'],
            r['n_nonfinite_cons_m15'], r['n_nonfinite_avg_in_m0'],
            r['n_nonfinite_avg_in_m15']) == (1, 0, 0, 0, 1)
    tested = np.array([0.01 * (i + 1) for i in range(20)])
    tested[3] = -3.0
    assert r['p_perm'] == pytest.approx(permutation_p(tested, np.zeros(20),
                                                      'greater'))
    assert (r['n_diverged_a'], r['n_diverged_b']) == (0, 1)


def test_t_b2_without_a_valid_seed_reads_fail(tmp_path):
    """Every cons value at m = 15 non-finite: all twenty seeds at -3, W+ =
    0, p = 1, FAIL; the median reads the 0 rule (0 - 0.6)."""
    r = _row(tmp_path, 'T-B2', opts=_b_opts(
        'T-B2', 'a', {s: _bf(0.7, None) for s in B_SEEDS}))
    assert (r['n_penalised'], r['n_nonfinite_cons_m15']) == (20, 20)
    assert (r['p_perm'], r['p_perm_adj'], r['p_perm_adj_f3']) == (1.0, 1.0,
                                                                  1.0)
    assert (r['status'], r['reading'], r['w_plus_minus_w_minus']) == (
        'tested', 'FAIL', -210)
    assert r['median_diff'] == pytest.approx(-0.6) and r['n_positive'] == 0
    assert pd.isna(r['direction']) and pd.isna(r['median_sign_differs'])


def test_t_b2_pass_with_median_zero(tmp_path):
    """delta 8 x -1, 3 x 0, 2 x +2, 7 x +3 (x 0.01): median 0, zeros
    dropped, W+ = 2 x 9.5 + 7 x 14 = 117, W- = 8 x 4.5 = 36, one-sided p
    0.0202; PASS with the sign of the median (0) off the sign of W+ - W-."""
    units = [-1] * 8 + [0] * 3 + [2] * 2 + [3] * 7
    r = _row(tmp_path, 'T-B2', opts={
        **_b_opts('T-B2', 'a', {s: _bf(0.5, 0.5 + 0.01 * u)
                                for s, u in zip(B_SEEDS, units)}),
        **_b_opts('T-B2', 'b', {s: _bf(0.5, 0.5) for s in B_SEEDS})})
    assert r['p_perm'] == pytest.approx(0.0202, abs=5e-5)
    assert r['median_diff'] == 0 and r['w_plus_minus_w_minus'] == 81
    assert (r['reading'], r['direction'], r['median_sign_differs']) == (
        'PASS', 'A>B', True)
    assert r['n_positive'] == 9 and r['n_penalised'] == 0


@pytest.mark.parametrize('opts, match', [
    ({('T-B1', 'a'): {13: {'epochs': 50}}},
     r'seed13\.json: T-B1\.a \(cons\): epochs 50 \(registry 60\)'),
    ({('T-B1', 'b'): {13: {'wiring': 'cncp'}}}, r'seed13\.json: .*wiring'),
    ({('T-B1', 'b'): {13: {'extra': {'cell': 'lrc'}}}},
     r'seed13\.json: .*cell'),
    ({('T-B2', 'b'): {13: {'extra': {'n_columns': 3}}}},
     r'K3_seed13\.json: .*n_columns 3'),
    ({('T-B1', 'a'): {13: {'by_fault': [0.7, 0.6, 0.5, 0.3],
                           'extra': {'fault_counts': [0, 3, 9, 20]}}}},
     r'seed13\.json: no fault level m=15'),
    # a corruption_seed mismatch within a pair: the record rule fires first
    ({('T-B1', 'b'): {13: {'corruption_seed': 14}}},
     r'seed13\.json: corruption_seed 14 differs from seed 13'),
    ({('T-B2', 'a'): {15: {'corruption_seed': 0}}},
     r'seed15\.json: corruption_seed 0 differs from seed 15'),
    ({('T-B1', 'b'): {13: {'extra': {'val_seed': 1}}}},
     r'T-B1: arms A and B differ in val_seed on seed\(s\) \[13\]'),
    ({('T-B2', 'b'): {17: {'extra': {'split_seed': 1}}}},
     r'T-B2: arms A and B differ in the data key.*\[17\]')])
def test_family_b_record_rules(tmp_path, opts, match):
    with pytest.raises(SystemExit, match=match):
        _blocks(_world(tmp_path, opts=opts))


def test_family_b_duplicate_and_f1_record_raise(tmp_path):
    spec = load_f3_spec(_world(tmp_path))
    path = spec['T-B1']['a']['paths'][0]
    _co(os.path.join(path, 'again'), 'cons', 12, 0.65)
    with pytest.raises(SystemExit, match=r'two records of T-B1\.a \(cons\) '
                                         r'and one seed'):
        f3_family_blocks(spec)
    spec = load_f3_spec(_world(tmp_path / 'f1'))
    _co(os.path.join(spec['T-B1']['a']['paths'][0], 'old'), 'cons', 3, 0.6,
        pv=F1)
    with pytest.raises(SystemExit, match=r'old.*protocol_version.*T-B1\.a'):
        f3_family_blocks(spec)


def test_family_b_arms_in_one_run_folder(tmp_path):
    """All three arms in one run folder, as the campaign writes them: the
    test statistics equal those of separate folders; each arm counts the
    other two names (F3 stamp) and its own records off the seed list."""
    spec = load_f3_spec(_world(tmp_path))
    want = f3_family_blocks(spec)[0].set_index('test')
    shared = str(tmp_path / 'shared')
    os.makedirs(shared)
    for t in ('T-B1', 'T-B2'):
        for s in 'ab':
            src = spec[t][s]['paths'][0]
            for name in os.listdir(src):
                shutil.copy(os.path.join(src, name), shared)
            spec[t][s]['paths'] = [shared]
    for seed in (5, 100):
        _co(shared, 'cons', seed, 0.1)
    got = f3_family_blocks(spec)[0].set_index('test')
    for t in ('T-B1', 'T-B2'):
        for c in ('n_pairs', 'p_perm', 'p_raw', 'median_diff',
                  'median_diff_lo', 'median_diff_hi', 'reading'):
            assert got.loc[t, c] == want.loc[t, c], (t, c)
        assert (got.loc[t, 'n_outside_list'], got.loc[t, 'outside_list']) \
            == (2, 'a: seeds 5, 100')
    r = got.loc['T-B1']
    assert (r['n_other_label_a'], r['other_labels_a']) == (
        40, 'avg_in (20); fault_mean (20)')
    assert (r['n_other_label_b'], r['other_labels_b']) == (
        42, 'avg_in (20); cons (22)')
    assert got.loc['T-B2', 'other_labels_b'] == 'cons (22); fault_mean (20)'


def test_family_b_skips_eval_only_files_and_wandb_summaries(tmp_path):
    """A wandb run folder (the Person Activity summary and one carrying
    n_columns) and an eval-only output with cell but no wiring below an arm
    path change no output; the eval-only file is counted."""
    spec = load_f3_spec(_world(tmp_path))
    want, want_b = f3_family_blocks(spec)[0], f3_family_b(spec)
    run = spec['T-B1']['a']['paths'][0]
    _run_folder(run)
    _dump(os.path.join(run, 'wandb', 'co', 'files', 'wandb-summary.json'),
          {**SUMMARY, 'n_columns': 30, 'test_macro_f1_m15': 0.5})
    _dump(os.path.join(run, 'cons_seed12.lambda_grid.evalonly.json'),
          {'cell': 'cfc_lrc', 'n_columns': 30, 'seed': 12})
    got = f3_family_blocks(spec)[0]
    assert got.set_index('test').loc['T-B1', 'n_evalonly_a'] == 1
    assert want.set_index('test').loc['T-B1', 'n_evalonly_a'] == 0
    pd.testing.assert_frame_equal(got.drop(columns='n_evalonly_a'),
                                  want.drop(columns='n_evalonly_a'))
    pd.testing.assert_frame_equal(f3_family_b(spec), want_b)
    text = f3_report(got, pd.DataFrame(columns=F3_FIT_COLUMNS))
    assert '| T-B1 | — | — | — | — | — | 1 | 0 |' in text
    assert '| T-B2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |' in text


def test_family_b_record_without_wiring_names_the_file(tmp_path):
    """A half committee record (n_columns and cell, no wiring, no seed)
    stops the loader before any label or seed is read."""
    spec = load_f3_spec(_world(tmp_path))
    _dump(os.path.join(spec['T-B1']['a']['paths'][0], 'half.json'),
          {'n_columns': 30, 'cell': 'cfc_lrc'})
    with pytest.raises(SystemExit, match=r"half\.json.*\['wiring'\]"):
        f3_family_blocks(spec)
    with pytest.raises(SystemExit, match=r"half\.json.*\['wiring'\]"):
        f3_family_b(spec)


def test_family_b_record_without_seed_names_the_file(tmp_path):
    """A cons record with every field the loader checks before the seed
    but without seed and corruption_seed stops it naming the file."""
    spec = load_f3_spec(_world(tmp_path))
    _dump(os.path.join(spec['T-B1']['a']['paths'][0], 'noseed.json'), {
        'cell': 'cfc_lrc', 'wiring': 'ncp', 'n_columns': 30, 'epochs': 60,
        'view_mode': 'identical', 'train_corruption': 'on',
        'view_average': 'off', 'vote_lambda': None, 'consensus': 'mean',
        'config': {'effective': {'protocol_version': F3}}})
    with pytest.raises(SystemExit, match=r'noseed\.json: .*without seed'):
        f3_family_blocks(spec)


def test_family_b_empty_arm_folder_is_incomplete(tmp_path):
    spec = _world(tmp_path, skip={('T-B1', 'b'): B_SEEDS})
    r = _blocks(spec).loc['T-B1']
    assert (r['n_pairs'], r['status'], r['reading']) == (0, 'incomplete',
                                                         'incomplete')
    assert pd.isna(r['floor_b']) and pd.isna(r['p_perm'])
    out = str(tmp_path / 'out')
    assert main(['--runs', str(tmp_path / 'none'), '--out', out,
                 '--f3-families', spec]) == 0
    fb = pd.read_csv(os.path.join(out, 'f3_family_b.csv'))
    fm = fb[fb['arm'] == 'fault_mean']
    assert len(fm) == 6 and (fm['n'] == 0).all()
    assert fb[fb['quantity'] == 'cons - fault_mean']['n'].tolist() == [0, 0]


def test_family_b_arm_above_budget_is_excluded(tmp_path):
    r = _row(tmp_path, 'T-B1', opts={('T-B1', 'b'): {12: {'above': True}}})
    assert r['status'] == EXCLUDED and pd.isna(r['reading'])
    assert r['p_perm'] == pytest.approx(1 / 2 ** 20)
    for k in ('p_raw', 'p_perm_adj', 'p_perm_adj_f3', 'significant',
              'significant_f3'):
        assert pd.isna(r[k]), k
    assert (r['median_diff_lo'], r['median_diff_hi']) == (0.0, 0.0)


def test_family_b_descriptive_rows_and_section(tmp_path):
    """f3_family_b with cons non-finite at m = 0 on seed 10 (entering as 0)
    and fits at or past best_val_epoch 57; through main: the csv, the
    summary section, no p column."""
    spec = _world(tmp_path, opts={
        ('T-B1', 'a'): {10: {'by_fault': _bf(None, 0.61), 'best': 58},
                        11: {'best': 57}},
        ('T-B1', 'b'): {12: {'best': 56}}})
    fb = f3_family_b(load_f3_spec(spec))
    assert list(fb.columns) == F3_B_COLUMNS
    assert not [c for c in fb.columns if c == 'p' or c.startswith('p_')]
    rows = {(q, a if isinstance(a, str) else None,
             None if pd.isna(m) else int(m)): r for q, a, m, r in zip(
        fb['quantity'], fb['arm'], fb['m'], fb.to_dict('records'))}
    assert len(rows) == len(fb) == 3 * 5 + 6 + 3
    cons15 = 0.6 + 0.01 * np.arange(1, 21)
    r = rows['test_macro_f1', 'cons', 15]
    assert (r['n'], r['median'], r['n_nonfinite']) == (
        20, pytest.approx(np.median(cons15)), 0)
    assert (r['lo'], r['hi']) == bootstrap_median_ci(cons15, N_BOOT)
    r = rows['test_macro_f1', 'cons', 0]
    assert (r['median'], r['n_nonfinite']) == (0.7, 1)
    assert rows['test_macro_f1', 'avg_in', 20]['median'] == 0.3
    d15 = cons15 - 0.6
    r = rows['d15', None, 15]
    assert (r['median'], r['n_positive']) == (pytest.approx(np.median(d15)),
                                              20)
    assert (r['median_crossed_lo'], r['median_crossed_hi']) == (0.0, 0.0)
    r = rows['d0', None, 0]
    assert (r['median'], r['n_nonfinite']) == (0.0, 1)
    assert pd.isna(r['median_crossed_lo']) and pd.isna(r['n_positive'])
    did = d15.copy()
    did[0] += 0.7
    r = rows['d15 - d0', None, None]
    assert r['median'] == pytest.approx(np.median(did))
    assert (r['lo'], r['hi']) == bootstrap_median_ci(did, N_BOOT)
    assert rows['cons - fault_mean', None, 15]['median'] == \
        pytest.approx(np.median(d15))
    assert rows['cons - fault_mean (15) - (0)', None, None]['n'] == 20
    conv = {a: rows['best_val_epoch >= 57', a, None]['n_converged']
            for a in ('cons', 'avg_in', 'fault_mean')}
    assert conv == {'cons': 2, 'avg_in': 0, 'fault_mean': 0}
    out = str(tmp_path / 'out')
    assert main(['--runs', str(tmp_path / 'none'), '--out', out,
                 '--f3-families', spec]) == 0
    pd.testing.assert_frame_equal(
        pd.read_csv(os.path.join(out, 'f3_family_b.csv')),
        pd.read_csv(io.StringIO(fb.to_csv(index=False))))
    with open(os.path.join(out, 'summary.md'), encoding='utf-8') as f:
        text = f.read()
    section = text[text.index('## F3 family B, descriptive'):]
    assert '| d15 | — | 15 | 20 |' in section
    assert not [w for w in ('p adj', 'p perm', 'significant', 'PASS', 'FAIL')
                if w in section]


# --- family E (#133) ----------------------------------------------------------

E_SEEDS = list(range(10, 35))
E_FOLDS = [(s - 10) // 5 for s in E_SEEDS]


def test_family_e_registry_seeds_and_l17():
    assert {k: list(r) for k, r in F3_FOLD_SEEDS.items()} == {
        k: list(range(10 + 5 * k, 15 + 5 * k)) for k in range(5)}
    assert sorted(s for r in F3_FOLD_SEEDS.values() for s in r) == E_SEEDS
    l17 = [r for k, rs in F3_E_L17_RECORDINGS.items() for r in rs]
    assert len(l17) == len(set(l17)) == 15
    assert all(r in _fold_recs(k) for k, rs in F3_E_L17_RECORDINGS.items()
               for r in rs)
    assert F3_E_L17_RECORDINGS[2] == tuple(_fold_recs(2))
    e = F3_TESTS['T-E3']
    assert (e['status'], e['a'], e['b']) == ('optional', 'cfc/dense',
                                             'cfc_static/dense')
    assert 'T-E3' not in REQUIRED and {'T-E1', 'T-E2', 'T-G1'} <= set(
        REQUIRED)
    assert list(F3_TESTS).index('T-E1') == list(F3_TESTS).index('T-D2') + 1


def test_family_e_rows_of_the_default_world(tmp_path):
    """A = B + 0.01 (i + 1) on the 25 runs: exact p 2 / 2^25 (also the p of
    wilcoxon_test), m = 2, n_expected 25, the fold-stratified interval;
    T-E3 is not in the spec; with it the E rows and m_f3 stay the same."""
    b = _blocks(_world(tmp_path))
    d = np.array([0.6 + 0.01 * (i + 1) for i in range(25)]) - 0.6
    for t in ('T-E1', 'T-E2'):
        r = b.loc[t]
        assert (r['family'], r['m'], r['m_f3'], r['n_expected'], r['n_pairs'],
                r['status'], r['reading'], r['direction']) == (
            'E', 2, 14, 25, 25, 'tested', 'PASS', 'A>B')
        assert r['endpoint'] == 'test_macro_f1_per_recording'
        assert r['p_perm'] == pytest.approx(2 / 2 ** 25)
        assert (r['p_raw'], r['method']) == (pytest.approx(2 / 2 ** 25),
                                             'exact')
        assert r['p_perm_adj_f3'] == pytest.approx(28 / 2 ** 25)
        assert not r['verdict_flip'] and not r['median_sign_differs']
        assert r['ci_units'] == 'fold-stratified'
        assert (r['median_diff_lo'], r['median_diff_hi']) == \
            stratified_median_ci(d, E_FOLDS, N_BOOT)
        assert r['median_diff'] == pytest.approx(0.13)
        assert not r['ci_includes_zero'] and r['n_positive'] == 25
        assert (r['floor_a'], r['floor_b'], r['floor_bound']) == (
            '0/25', '0/25', False)
        assert r['w_plus_minus_w_minus'] == 325
    assert 'T-E3' not in b.index
    with_e3 = _blocks(_world(tmp_path / 'e3', REQUIRED + ['T-E3']))
    assert (with_e3['m_f3'] == 14).all()
    assert with_e3.loc[['T-E1', 'T-E2'], 'm'].tolist() == [2, 2]


def test_family_e_pass_with_median_zero(tmp_path):
    """Run differences 11 x -1, 2 x 0, 2 x +2, 10 x +3 (x 0.01): exact
    two-sided p 0.0194, median 0, PASS with median_sign_differs."""
    units = [-1] * 11 + [0] * 2 + [2] * 2 + [3] * 10
    r = _row(tmp_path, 'T-E1', values={
        'T-E1': ([0.6 + 0.01 * u for u in units], [0.6] * 25)})
    assert r['p_perm'] == pytest.approx(0.0194, abs=5e-5)
    assert r['median_diff'] == 0 and r['w_plus_minus_w_minus'] == 144
    assert (r['reading'], r['direction'], r['median_sign_differs']) == (
        'PASS', 'A>B', True)


def test_family_e_run_value_is_the_recording_mean(tmp_path):
    """The run value averages the per-recording values of the fold (not the
    pooled test_macro_f1); a non-finite one makes the run diverged, entering
    at 0 and at the floor; training_diverged counts in n_diverged; both
    pooled fields go to f3_pa_fits."""
    per = dict(zip(_fold_recs(0), (0.5, 0.6, 0.7, 0.8, 0.9)))
    spec = load_f3_spec(_world(tmp_path, opts={('T-E1', 'b'): {
        10: {'per_rec': per},
        11: {'per_rec': {**per, 'C01': None}},
        12: {'extra': {'training_diverged': True}}}}))
    df, _, _ = _f3_arm('T-E1', 'b', F3_TESTS['T-E1'], spec['T-E1']['b'],
                       E_SEEDS, {}, ('split_fold', 'seed'))
    df = df.set_index('seed')
    assert df.loc[10, 'metric'] == pytest.approx(0.7)
    assert df.loc[10, 'test_macro_f1'] == 0.6          # pooled, secondary
    assert df.loc[11, 'diverged'] and df.loc[11, 'below']
    assert df.loc[12, 'training_diverged'] and not df.loc[12, 'diverged']
    blocks, fits = f3_family_blocks(spec)
    r = blocks.set_index('test').loc['T-E1']
    assert (r['n_diverged_a'], r['n_diverged_b'], r['n_nonfinite'],
            r['floor_b']) == (0, 2, 1, '1/25')
    e1 = fits[fits['tests'] == 'T-E1.b'].set_index('seed')
    assert len(e1) == 25 and e1.loc[10, 'test_macro_f1'] == 0.6
    assert list(fits.columns) == F3_FIT_COLUMNS


# two recordings of unequal length with different class sets (rows = true):
# pooled labels 90 / 13 / 7, majority class 0; one-class values 0.4737 (A01)
# and 0 (B01, no class-0 window): recording mean 0.2368, pooled 0.30
FLOOR_CMS = {'A01': [[85, 5, 0], [5, 5, 0], [0, 0, 0]],
             'B01': [[0, 0, 0], [0, 2, 1], [0, 3, 4]]}


def test_family_e_floor_on_the_recording_mean(tmp_path):
    cms = np.array([FLOOR_CMS[k] for k in sorted(FLOOR_CMS)], float)
    assert _e_one_class(cms) == pytest.approx((180 / 190) / 2 / 2)
    assert one_class_macro_f1(cms.sum(axis=(0, 2))) == pytest.approx(0.3)
    path = str(tmp_path / 'b')
    for seed, (a01, b01) in ((10, (0.30, 0.24)), (11, (0.28, 0.20))):
        _pe(path, 'gru/ncp', seed, 0.27, cms=FLOOR_CMS,
            per_rec={'A01': a01, 'B01': b01})
    arm = {'label': 'gru/ncp', 'protocol_version': [F3], 'paths': [path]}
    df, _, _ = _f3_arm('T-E1', 'b', F3_TESTS['T-E1'], arm, E_SEEDS, {},
                       ('split_fold', 'seed'))
    # run 10 at 0.27: above 0.2368 + 0.01, at the pooled one-class 0.30;
    # run 11 at 0.24: below the margin of the recording mean
    assert df['metric'].tolist() == pytest.approx([0.27, 0.24])
    assert df['below'].tolist() == [False, True]


@pytest.mark.parametrize('k, bound', [(8, True), (7, False)])
def test_family_e_floor_share(tmp_path, k, bound):
    vb = [0.2] * k + [0.6] * (25 - k)
    r = _row(tmp_path, 'T-E1', values={
        'T-E1': ([0.7 + 0.01 * i for i in range(25)], vb)})
    assert (r['floor_b'], r['floor_bound'], r['reading']) == (
        f'{k}/25', bound, 'floor-bound' if bound else 'PASS')


def test_family_e_records_off_their_fold_are_outside(tmp_path):
    spec = load_f3_spec(_world(tmp_path))
    folder = os.path.join(spec['T-E1']['a']['paths'][0], 'other')
    _pe(folder, 'cfc_lrc/ncp', 12, 0.1, fold=3, name='off_fold.json')
    _pe(folder, 'cfc_lrc/ncp', 40, 0.1, fold=0, name='off_list.json')
    r = f3_family_blocks(spec)[0].set_index('test').loc['T-E1']
    assert (r['n_outside_list'], r['outside_list']) == (
        2, 'a: fold 0 seed 40, fold 3 seed 12')
    assert r['n_pairs'] == 25 and r['p_perm'] == pytest.approx(2 / 2 ** 25)


@pytest.mark.parametrize('opts, match', [
    ({17: {'eff_fold': 2}},
     r'T-E1: arms A and B differ in the data key \(split_seed, split_fold\)'
     r': .*seed17\.json / .*seed17\.json'),
    ({17: {'split_seed': 1}}, r'T-E1: .*data key.*seed17\.json'),
    ({17: {'dt_scale': 4.2}},
     r'T-E1: arms A and B differ in config\.effective\.dt_scale: '
     r'.*seed17\.json'),
    ({17: {'extra': {'split_fold': None}}},
     r'seed17\.json: T-E1\.b \(gru/ncp\): record without split_fold'),
    ({17: {'extra': {'epochs': 60}}}, r'T-E1: .*differ in epochs.*\[17\]')])
def test_family_e_record_and_pair_rules(tmp_path, opts, match):
    with pytest.raises(SystemExit, match=match):
        _blocks(_world(tmp_path, opts={('T-E1', 'b'): opts}))


def test_family_e_duplicate_run_raises(tmp_path):
    spec = load_f3_spec(_world(tmp_path))
    _pe(os.path.join(spec['T-E1']['b']['paths'][0], 'again'), 'gru/ncp', 22,
        0.6)
    with pytest.raises(SystemExit, match=r'two records of T-E1\.b \(gru/ncp\) '
                                         r'and one split_fold/seed'):
        f3_family_blocks(spec)


def test_family_e_record_without_seed_names_the_file(tmp_path):
    spec = load_f3_spec(_world(tmp_path))
    _dump(os.path.join(spec['T-E1']['a']['paths'][0], 'noseed.json'), {
        'cell': 'cfc_lrc', 'wiring': 'ncp', 'param_budget': 30000,
        'test_macro_f1': 0.5, 'split_fold': 0,
        'config': {'effective': {'protocol_version': F3, 'epochs': 50,
                                 'task': 'person_activity', 'split_fold': 0}}})
    with pytest.raises(SystemExit, match=r'noseed\.json: T-E1\.a '
                                         r'\(cfc_lrc/ncp\): record without '
                                         r'seed'):
        f3_family_blocks(spec)
    with pytest.raises(SystemExit, match=r'noseed\.json: .*without seed'):
        f3_family_e(spec)


def test_family_e_skips_wandb_summaries(tmp_path):
    spec = load_f3_spec(_world(tmp_path))
    want = f3_family_blocks(spec), f3_family_e(spec)
    for t in ('T-E1', 'T-E2'):
        for s in 'ab':
            _run_folder(spec[t][s]['paths'][0])
    got = f3_family_blocks(spec), f3_family_e(spec)
    for g, w in zip((*got[0], got[1]), (*want[0], want[1])):
        pd.testing.assert_frame_equal(g, w)


def test_family_e_empty_fold_folders_are_incomplete(tmp_path):
    spec = load_f3_spec(_world(tmp_path))
    for s in 'ab':
        spec['T-E1'][s]['paths'] = [str(tmp_path / 'empty' / s / f'fold{k}')
                                    for k in range(5)]
        for p in spec['T-E1'][s]['paths']:
            os.makedirs(p)
    r = f3_family_blocks(spec)[0].set_index('test').loc['T-E1']
    assert (r['n_pairs'], r['status'], r['reading']) == (0, 'incomplete',
                                                         'incomplete')
    with open(tmp_path / 'spec2.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(spec, f)
    out = str(tmp_path / 'out')
    assert main(['--runs', str(tmp_path / 'none'), '--out', out,
                 '--f3-families', str(tmp_path / 'spec2.yaml')]) == 0
    fe = pd.read_csv(os.path.join(out, 'f3_family_e.csv'))
    assert set(fe['test']) == {'T-E2'}


def test_family_e_incomplete_below_25_pairs(tmp_path):
    r = _row(tmp_path, 'T-E1', skip={('T-E1', 'b'): [33]})
    assert (r['n_expected'], r['n_pairs'], r['status'], r['reading']) == (
        25, 24, 'incomplete', 'incomplete')
    assert r['p_perm'] == pytest.approx(2 / 2 ** 24)
    assert r['ci_units'] == 'fold-stratified'


def test_family_e_arms_in_shared_fold_folders(tmp_path):
    """The three arms of a fold in one run folder per fold, as the campaign
    writes them: the statistics equal those of separate folders; each arm
    counts the other two labels."""
    spec = load_f3_spec(_world(tmp_path))
    want = f3_family_blocks(spec)[0].set_index('test')
    folds = [str(tmp_path / f'fold{k}' / 'results') for k in range(5)]
    for p in folds:
        os.makedirs(p)
    for t in ('T-E1', 'T-E2'):
        for s in 'ab':
            src = spec[t][s]['paths'][0]
            for name in os.listdir(src):
                seed = int(name.split('seed')[1].split('.')[0])
                shutil.copy(os.path.join(src, name), folds[(seed - 10) // 5])
            spec[t][s]['paths'] = folds
    got = f3_family_blocks(spec)[0].set_index('test')
    for t in ('T-E1', 'T-E2'):
        for c in ('n_pairs', 'p_perm', 'p_raw', 'median_diff',
                  'median_diff_lo', 'median_diff_hi', 'reading'):
            assert got.loc[t, c] == want.loc[t, c], (t, c)
    r = got.loc['T-E1']
    assert (r['n_other_label_a'], r['other_labels_a']) == (
        50, 'cfc_lrc/dense3 (25); gru/ncp (25)')
    assert r['other_labels_b'] == 'cfc_lrc/dense3 (25); cfc_lrc/ncp (25)'


def test_family_e_never_calls_the_crossed_interval(tmp_path, monkeypatch):
    spec = load_f3_spec(_world(tmp_path))
    e = F3_TESTS['T-E1']
    a, b = (_f3_arm('T-E1', s, e, spec['T-E1'][s], E_SEEDS, {},
                    ('split_fold', 'seed'))[0] for s in 'ab')

    def boom(*args, **kw):
        raise AssertionError('crossed path called')

    monkeypatch.setattr('experiments.aggregate_results.crossed_median_diff_ci',
                        boom)
    r = _f3_test('T-E1', e, 2, 14, a, b, E_SEEDS, {'a': [], 'b': []},
                 ('split_fold', 'seed'))
    assert (r['reading'], r['ci_units']) == ('PASS', 'fold-stratified')


def test_median_diff_ci_refuses_records_on_other_recordings():
    """Fold records stack into confusions of one shape; the guard on the
    recording ids stops the crossed path. A frame without the column (every
    caller before #133) keeps the crossed interval."""
    a, b = _dframe([0.7] * 10), _dframe([0.6] * 10)
    p = a.merge(b, on='seed', suffixes=('_a', '_b')).assign(
        v_a=0.7, v_b=0.6)
    want = _median_diff_ci(p, 'pa')
    assert want[1] == 'seed x recording'
    folds = [dict.fromkeys(_fold_recs(i % 2), 0.6) for i in range(10)]
    for s in 'ab':
        p[f'test_macro_f1_per_recording_{s}'] = folds
    with pytest.raises(ValueError, match='arm A differ in their test '
                                         'recordings'):
        _median_diff_ci(p, 'pa')
    for s in 'ab':
        p[f'test_macro_f1_per_recording_{s}'] = [folds[0]] * 10
    assert _median_diff_ci(p, 'pa') == want


def test_nested_and_stratified_intervals():
    rng = np.random.default_rng(3)
    cells = rng.normal(size=(5, 5))
    assert nested_median_ci(cells - cells) == (0.0, 0.0)
    assert nested_median_ci(np.full((5, 5), 0.25)) == (0.25, 0.25)
    lo, hi = nested_median_ci(cells)
    assert lo < np.median(cells) < hi
    assert stratified_median_ci(np.full(25, -0.1), E_FOLDS) == (-0.1, -0.1)
    # one value per fold: every resample keeps it
    v = np.repeat(np.arange(5.0), 5)
    assert stratified_median_ci(v, E_FOLDS) == (2.0, 2.0)


def _e_section(text):
    return text[text.index('## F3 family E, descriptive'):]


def test_family_e_descriptive_rows_and_section(tmp_path):
    """The default world: run difference 0.01 (s - 9) on seed s, every
    recording at the run value. Fold k: median and mean 0.01 (5k + 3); all
    five folds have the sign of W+ - W-; cells = the fold medians, person
    medians 0.13; L17 = the run differences; no run below 0.30."""
    spec = _world(tmp_path)
    fe = f3_family_e(load_f3_spec(spec))
    assert list(fe.columns) == F3_E_COLUMNS
    assert not [c for c in fe.columns if c == 'p' or c.startswith('p_')]
    e1 = fe[fe['test'] == 'T-E1']
    q = {k: g for k, g in e1.groupby('quantity')}
    med = q['median run difference'].set_index('fold')['value']
    assert med.tolist() == pytest.approx([0.01 * (5 * k + 3)
                                          for k in range(5)])
    va = q['median run value'].set_index(['fold', 'arm'])['value']
    assert va[(0, 'cfc_lrc/ncp')] == pytest.approx(0.63)
    assert va[(0, 'gru/ncp')] == pytest.approx(0.6)
    r = q['folds with the sign of W+ - W-'].iloc[0]
    assert (r['n'], r['value']) == (5, 5)
    r = q['mean of fold means'].iloc[0]
    means = np.array([0.01 * (5 * k + 3) for k in range(5)])
    half = 2.7764451051977987 * means.std(ddof=1) / np.sqrt(5)
    assert (r['value'], r['lo'], r['hi']) == (
        pytest.approx(0.13), pytest.approx(0.13 - half),
        pytest.approx(0.13 + half))
    assert r['interval'] == 't interval (4 df)'
    cells = q['cell'].set_index('recording')
    assert len(cells) == 25 and (cells['n'] == 5).all()
    assert cells.loc['C04', 'value'] == pytest.approx(0.18)
    assert cells.loc['C04', 'person'] == 'C'
    pm = q['person median']
    assert len(pm) == 5 and (pm['n'] == 5).all()
    assert pm['value'].tolist() == pytest.approx([0.13] * 5)
    r = q['median over cells'].iloc[0]
    grid = np.array([[0.01 * (5 * k + 3) for k in range(5)]] * 5)
    assert r['value'] == pytest.approx(0.13)
    assert (r['lo'], r['hi']) == pytest.approx(nested_median_ci(grid, N_BOOT))
    assert r['interval'] == F3_E_NESTED_LABEL
    r = q['L17 run difference'].iloc[0]
    d = np.array([0.6 + 0.01 * (i + 1) for i in range(25)]) - 0.6
    assert (r['n'], r['value']) == (25, pytest.approx(0.13))
    assert (r['lo'], r['hi']) == pytest.approx(stratified_median_ci(
        d, E_FOLDS, N_BOOT))
    below = q['runs below 0.3']
    assert len(below) == 10 and (below['value'] == 0).all()
    assert (q['dt_scale']['dt_scale'] == '3.7').all()
    assert set(fe['test']) == {'T-E1', 'T-E2'}
    out = str(tmp_path / 'out')
    assert main(['--runs', str(tmp_path / 'none'), '--out', out,
                 '--f3-families', spec]) == 0
    pd.testing.assert_frame_equal(
        pd.read_csv(os.path.join(out, 'f3_family_e.csv')),
        pd.read_csv(io.StringIO(fe.to_csv(index=False))))
    with open(os.path.join(out, 'summary.md'), encoding='utf-8') as f:
        section = _e_section(f.read())
    assert F3_E_NESTED_LABEL in section
    assert 'config.effective.dt_scale across folds: one value per test.' \
        in section
    assert not [w for w in ('p adj', 'p perm', 'significant', 'PASS', 'FAIL')
                if w in section]


def test_family_e_l17_cells_and_dt_scale_flag(tmp_path):
    """Fold 0 of T-E1: arm A higher by 0.1 on the L17 recordings C01 and D01
    only. The L17 difference of the fold-0 runs is 0.1, their run
    difference 0.04; the cells C01 and D01 read 0.1, A01 0. A dt_scale of
    4.2 on fold 4 (both arms) is flagged."""
    per = dict.fromkeys(_fold_recs(0), 0.6)
    hi = {**per, 'C01': 0.7, 'D01': 0.7}
    opts = {('T-E1', 'a'): {s: {'per_rec': hi} for s in F3_FOLD_SEEDS[0]},
            ('T-E1', 'b'): {s: {'per_rec': per} for s in F3_FOLD_SEEDS[0]}}
    for side in 'ab':
        opts[('T-E1', side)].update({s: {'dt_scale': 4.2}
                                     for s in F3_FOLD_SEEDS[4]})
    spec = _world(tmp_path, values={'T-E1': ([0.6] * 25, [0.6] * 25)},
                  opts=opts)
    fe = f3_family_e(load_f3_spec(spec))
    e1 = fe[fe['test'] == 'T-E1']
    cells = e1[e1['quantity'] == 'cell'].set_index('recording')['value']
    assert (cells['C01'], cells['D01'], cells['A01'], cells['A02']) == (
        pytest.approx(0.1), pytest.approx(0.1), 0.0, 0.0)
    med = e1[e1['quantity'] == 'median run difference'].set_index('fold')
    assert med.loc[0, 'value'] == pytest.approx(0.04)
    r = e1[e1['quantity'] == 'L17 run difference'].iloc[0]
    assert r['value'] == 0.0 and r['n'] == 25     # 5 of 25 runs at 0.1
    dt = e1[e1['quantity'] == 'dt_scale'].set_index('fold')['dt_scale']
    assert (dt[0], dt[4]) == ('3.7', '4.2')
    out = str(tmp_path / 'out')
    assert main(['--runs', str(tmp_path / 'none'), '--out', out,
                 '--f3-families', spec]) == 0
    with open(os.path.join(out, 'summary.md'), encoding='utf-8') as f:
        assert 'config.effective.dt_scale across folds: differs in T-E1.' \
            in _e_section(f.read())
