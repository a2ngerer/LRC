# tests/experiments/test_aggregate_f2.py
"""The F2 test families of experiments/aggregate_results.py (#107), on
synthetic records and frames only."""
import json
import os

import numpy as np
import pandas as pd
import pytest
import yaml

from experiments.aggregate_results import (EXCLUDED, F2_FAMILIES, F2_M,
                                           F2_TESTS, _f2_arm, _f2_test,
                                           arm_label, f2_family_blocks,
                                           load_f2_spec, load_ppo_runs, main,
                                           not_learned, permutation_p)

F1, F2 = 'F1-2026-10-26', 'F2-2026-11-05'
DS = 4242            # one data draw for every rollout fit (floors once)
LV_EPOCHS = {'periodic_predator_prey': 300, 'duffing': 450}
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))))


# --- record writers ---------------------------------------------------------

def _lv(path, system, cell, wiring, seed, vpt, pv, epochs=None, budget=4000,
        above=False, data_seed=DS, unbounded=False, anchors=True,
        forcing='mixed'):
    m = {'vpt_multi_median': vpt, 'rate_unbounded': 1.0 if unbounded else 0.0,
         'non_finite': False, 'anchor_horizon': 16, 'anchor_theta': 0.5}
    if anchors:
        m['anchors'] = [8]
    epochs = epochs or LV_EPOCHS[system]
    eff = {'task': 'lotka_volterra', 'system': system, 't_span': [0.0, 10.0],
           'ic_spread': 0.4, 'n_trajectories': 10, 'seq_len': 32,
           'data_seed': data_seed, 'split': [6, 2, 2], 'param_budget': budget,
           'ode_unfolds': 6, 'dt_scale': 12.8, 'forcing': forcing,
           'epochs': epochs, 'protocol_version': pv}
    rec = {'system': system, 'cell': cell, 'wiring': wiring, 'seed': seed,
           'param_budget': budget, 'budget_deviation': None if budget is None
           else 0.01, 'budget_above_tolerance': above, 'epochs': epochs,
           'lr': 1e-3, 'clipnorm': None, 'params_effective': 3960,
           'dt_scale': 12.8, 'forcing': forcing, 'best_val_epoch': 1,
           'rollout_metrics_test': m, 'config': {'effective': eff}}
    _dump(os.path.join(path, f'{cell}_{wiring}_seed{seed}_mixed.json'), rec)


GOOD = {f'r{i}': [[6, 1], [1, 5]] for i in range(3)}   # one-class 0.35


def _pa(path, cell, wiring, seed, f1, pv, budget=30000, above=False,
        conf=GOOD, split_seed=0, epochs=50):
    eff = {'task': 'person_activity', 'param_budget': budget,
           'epochs': epochs, 'split_seed': split_seed, 'ode_unfolds': 6,
           'protocol_version': pv}
    rec = {'cell': cell, 'wiring': wiring, 'seed': seed, 'epochs': epochs,
           'param_budget': budget, 'budget_deviation': 0.02,
           'budget_above_tolerance': above, 'lr': 3e-3, 'clipnorm': 1.0,
           'params_effective': 29500, 'dt_scale': 3.7, 'best_val_epoch': 40,
           'test_macro_f1': f1, 'config': {'effective': eff}}
    if conf is not None:
        rec['test_confusion_by_recording'] = conf
    _dump(os.path.join(path, f'{cell}_{wiring}_seed{seed}.json'), rec)


def _ppo(path, model, seed, ret, um=-100.0, us=1.0, profile='final',
         task='HalfCheetah-v5', above=False, drop=()):
    rec = {'model': model, 'seed': seed, 'task': task, 'profile': profile,
           'eval_return_mean': ret, 'eval_return_std': 5.0,
           'untrained_return_mean': um, 'untrained_return_std': us,
           'params_effective': 12000, 'budget_deviation': 0.01,
           'budget_above_tolerance': above}
    for k in drop:
        del rec[k]
    _dump(os.path.join(path, f'{model}_seed{seed}_{profile}.json'), rec)


def _dump(path, rec):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(rec, f)


# --- a complete synthetic F2 world ------------------------------------------

def _arm_writer(test, side):
    """(writer(path, seed, value, **kw), protocol_version) of one arm."""
    family, task, a, b = F2_TESTS[test]
    label = a if side == 'a' else b
    f1_side = {'Rewiring': 'a', 'Width': 'a', 'Budget': 'b'}.get(family)
    pv = None if family == 'PPO' else (F1 if side == f1_side else F2)
    cell, rest = label.split('/') if '/' in label else (label, '')
    wiring = rest.split(' ')[0]
    kw = {}
    if label.endswith('[fixed width]'):
        kw['budget'] = None
    if '[epochs=' in label:
        kw['epochs'] = int(label.split('=')[1].rstrip(']'))
    if task == 'pa':
        return (lambda p, s, v, **o: _pa(p, cell, wiring, s, v, pv,
                                         **{**kw, **o})), pv
    if family == 'PPO':
        return (lambda p, s, v, **o: _ppo(p, label, s, v, **o)), pv
    return (lambda p, s, v, **o: _lv(p, task, cell, wiring, s, v, pv,
                                     **{**kw, **o})), pv


BASE = {'pa': 0.6, 'HalfCheetah-v5': 100.0}   # rollouts: 20 (above floor)


def _world(tmp_path, values=None, opts=None, seeds=None):
    """Spec path of a world where every test has A = B + 1..10 (B = 20 VPT,
    0.6 macro-F1, 100 return) over seeds 0-9, each arm in its own folder.
    values: {test: (a values, b values)}; opts: {(test, side): writer
    kwargs, or {seed: kwargs}}; seeds: {(test, side): seeds}."""
    values, opts, seeds = values or {}, opts or {}, seeds or {}
    spec = {}
    for test, (family, task, a, b) in F2_TESTS.items():
        base = BASE.get(task, 20.0)
        step = 0.01 if task == 'pa' else 1.0
        va, vb = values.get(test, ([base + step * (s + 1) for s in range(10)],
                                   [base] * 10))
        spec[test] = {}
        for side, label, vals in (('a', a, va), ('b', b, vb)):
            write, pv = _arm_writer(test, side)
            path = str(tmp_path / test / side)
            o = opts.get((test, side), {})
            for s in seeds.get((test, side), range(len(vals))):
                write(path, s, vals[s], **(o.get(s, {}) if all(
                    isinstance(k, int) for k in o) else o))
            spec[test][side] = {'label': label, 'protocol_version': pv,
                                'paths': [path]}
    path = str(tmp_path / 'spec.yaml')
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(spec, f)
    return path


def _blocks(spec_path):
    return f2_family_blocks(load_f2_spec(spec_path)).set_index('test')


# --- families and the exact statistics ---------------------------------------

def test_f2_families_hold_the_seven_tests():
    assert {f: len(t) for f, t in F2_FAMILIES.items()} == {
        'Rewiring': 2, 'Width': 2, 'Budget': 2, 'PPO': 1}
    assert F2_M == 7
    assert F2_TESTS['width_lv'][3] == 'cfc_lrc/dense3 [fixed width]'
    assert F2_TESTS['budget_duffing'][2] == 'cfc_lrc/ncp [epochs=900]'


def _frame(vals, label='A', kind='lv', below=0, seeds=None, above=False,
           width=False):
    seeds = list(seeds if seeds is not None else range(len(vals)))
    v = np.asarray(vals, float)[:len(seeds)]
    return pd.DataFrame({
        'arm': label, 'seed': seeds, 'metric': v, 'task': kind,
        'diverged': ~np.isfinite(v),
        'below': [i < below for i in range(len(v))],
        'budget_above_tolerance': above, 'width_matched': width,
        'data_key': 1})


def _row(delta, m=2, **kw):
    """_f2_test with B = 0 and A = delta (task periodic_predator_prey)."""
    a = _frame(delta, 'A', **kw.pop('a', {}))
    b = _frame([0.0] * 10, 'B', **kw.pop('b', {}))
    return _f2_test('Rewiring', 'rewiring_lv', 'periodic_predator_prey', m,
                    a, b)


def test_ten_positive_differences_pass_with_the_exact_p():
    r = _row(range(1, 11))
    assert (r['reading'], r['method'], r['direction']) == ('PASS', 'exact',
                                                           'A>B')
    assert r['p_raw'] == pytest.approx(2 / 1024)
    assert r['p_adj'] == pytest.approx(4 / 1024)
    assert r['p_adj_f2'] == pytest.approx(14 / 1024)
    assert round(r['p_adj_f2'], 4) == 0.0137
    assert r['p_perm'] == pytest.approx(r['p_raw'])
    assert not r['only_under_family_cut'] and not r['verdict_flip']
    assert (r['n_pairs'], r['n_positive'], r['floor_bound']) == (10, 10,
                                                                 False)


def test_two_small_negative_differences_pass_only_under_the_family_cut():
    r = _row([-1, -2, *range(3, 11)])
    assert r['reading'] == 'PASS'
    assert r['p_raw'] == pytest.approx(10 / 1024)
    assert round(r['p_adj'], 4) == 0.0195
    assert round(r['p_adj_f2'], 4) == 0.0684
    assert r['significant'] and not r['significant_f2']
    assert r['only_under_family_cut']


def test_mixed_signs_fail():
    r = _row([-1, 2, -3, 4, 5, -6, 7, 8, 9, 10])
    assert r['reading'] == 'FAIL'
    assert r['p_raw'] == pytest.approx(86 / 1024)
    assert round(r['p_adj'], 3) == 0.168
    assert r['p_perm'] == pytest.approx(86 / 1024)


def test_ties_use_the_approximation_and_flag_the_permutation_flip():
    r = _row([2, 3, 3, -2, 1, 4, 2, 4, -1, 4])
    assert r['method'] == 'approx'
    assert r['p_raw'] == pytest.approx(0.0240, abs=1e-3)
    assert r['p_adj'] < 0.05
    assert r['p_adj'] == pytest.approx(0.0480, abs=1e-3)
    assert r['p_perm'] == pytest.approx(26 / 1024)
    assert r['p_perm_adj'] == pytest.approx(52 / 1024)
    assert round(r['p_perm_adj'], 4) == 0.0508
    assert r['verdict_flip'] and r['reading'] == 'PASS'


def test_permutation_p_equals_the_exact_wilcoxon_without_ties():
    from scipy.stats import wilcoxon
    rng = np.random.default_rng(0)
    for _ in range(5):
        d = rng.permutation(np.arange(1, 11)) * rng.choice([-1, 1], 10)
        assert permutation_p(d, np.zeros(10)) == pytest.approx(
            wilcoxon(d, method='exact').pvalue)
    assert permutation_p([1, 1], [1, 1]) == 1.0


def test_seven_pairs_are_undecidable_and_zero_differences_fail():
    r = _row(range(1, 11), b={'seeds': range(7)})
    assert r['n_pairs'] == 7 and r['reading'] == 'undecidable'
    assert r['p_raw'] == pytest.approx(2 * 1 / 2 ** 7)   # still reported
    r = _row([0.0] * 10)
    assert (r['n_pairs'], r['p_raw'], r['p_adj'], r['reading']) == (
        10, 1.0, 1.0, 'FAIL')
    assert r['p_perm'] == 1.0
    r = _row([0.0] * 10, b={'seeds': range(7)})
    assert (r['n_pairs'], r['reading']) == (7, 'undecidable')


def test_zero_differences_count_as_pairs_but_enter_no_test():
    r = _row([0, 0, *range(1, 9)])
    assert r['n_pairs'] == 10 and r['method'] == 'approx'
    assert r['n_positive'] == 8


@pytest.mark.parametrize('k, bound', [(2, False), (3, True)])
def test_floor_bound_at_three_of_ten_seeds(k, bound):
    r = _row(range(1, 11), b={'below': k})
    assert r['floor_bound'] is bound and r['floor_b'] == f'{k}/10'
    assert r['floor_bound_arms'] == ('B' if bound else None)
    assert r['reading'] == ('floor-bound' if bound else 'PASS')
    assert r['p_raw'] == pytest.approx(2 / 1024)


def test_seed_rule_counts():
    assert not not_learned(2, 10) and not not_learned(1, 5)
    assert not_learned(3, 10) and not_learned(2, 5)


# --- budget-tolerance exclusion ----------------------------------------------

def test_budget_exclusion_spares_only_the_width_matched_arm():
    r = _row(range(1, 11), b={'above': True, 'width': True})
    assert r['status'] == 'tested' and r['reading'] == 'PASS'
    r = _row(range(1, 11), b={'above': True})
    assert r['status'] == EXCLUDED and r['reading'] is None
    assert 'p_raw' not in r and r['median_diff'] == 5.5   # descriptive


# --- end to end over the spec ------------------------------------------------

def test_world_through_main_writes_csv_and_summary(tmp_path):
    spec = _world(tmp_path)
    out = str(tmp_path / 'out')
    assert main(['--runs', str(tmp_path / 'none'), '--out', out,
                 '--f2-families', spec]) == 0
    df = pd.read_csv(os.path.join(out, 'f2_families.csv')).set_index('test')
    assert list(df.index) == list(F2_TESTS)
    assert set(df['reading']) == {'PASS'}
    assert (df['n_pairs'] == 10).all() and (df['floor_a'] == '0/10').all()
    assert df.loc['rewiring_pa', 'ci_units'] == 'seed x recording'
    assert df.loc['rewiring_lv', 'p_adj'] == pytest.approx(4 / 1024)
    assert df.loc['ppo_halfcheetah', 'p_adj'] == pytest.approx(2 / 1024)
    assert df.loc['ppo_halfcheetah', 'm'] == 1
    with open(os.path.join(out, 'summary.md'), encoding='utf-8') as f:
        text = f.read()
    assert '## F2 test families' in text
    assert '| Width | 2 | width_lv | periodic_predator_prey | cfc_lrc/ncp ' \
        '| cfc_lrc/dense3 [fixed width] | 10 | tested | PASS | exact |' in text


def test_rollout_floor_marking_reads_each_arm_from_its_own_folder(tmp_path):
    """Three fits of the F2 arm below max(persistence, AR(1)) of its draw
    make the test floor-bound (p still reported), two do not; a non-finite
    VPT counts as below."""
    vb = [20.0] * 10
    vb[0], vb[1], vb[2] = -1.0, float('nan'), -1.0
    b = _blocks(_world(tmp_path / 'three', values={
        'rewiring_lv': ([21.0 + s for s in range(10)], vb)}))
    r = b.loc['rewiring_lv']
    assert (r['floor_b'], r['floor_bound'], r['floor_bound_arms'],
            r['reading']) == ('3/10', True, 'B', 'floor-bound')
    assert r['p_adj'] > 0 and r['n_nonfinite'] == 1
    vb[2] = 20.0
    b = _blocks(_world(tmp_path / 'two', values={
        'rewiring_lv': ([21.0 + s for s in range(10)], vb)}))
    assert b.loc['rewiring_lv', 'floor_b'] == '2/10'
    assert not b.loc['rewiring_lv', 'floor_bound']


def test_pa_chance_marking_and_interval_flag(tmp_path):
    vb = [0.6] * 10
    vb[:3] = [0.355, 0.30, float('nan')]    # one-class 0.35 + 0.01
    b = _blocks(_world(tmp_path, values={
        'width_pa': ([0.61 + s / 100 for s in range(10)], vb)}))
    r = b.loc['width_pa']
    assert (r['floor_b'], r['reading']) == ('3/10', 'floor-bound')
    assert r['ci_includes_zero'] in (True, False)
    assert pd.isna(b.loc['width_lv', 'ci_includes_zero'])


def test_width_matched_arm_above_tolerance_is_tested_other_arm_excluded(
        tmp_path):
    b = _blocks(_world(tmp_path, opts={
        ('width_lv', 'b'): {'above': True},
        ('rewiring_lv', 'b'): {'above': True},
        ('ppo_halfcheetah', 'a'): {'above': True}}))
    assert b.loc['width_lv', 'status'] == 'tested'
    for test in ('rewiring_lv', 'ppo_halfcheetah'):
        assert b.loc[test, 'status'] == EXCLUDED
        assert pd.isna(b.loc[test, 'reading'])
        assert b.loc[test, 'n_pairs'] == 10   # reported descriptively


def test_budget_arms_get_their_epochs_label_next_to_the_f1_twin(tmp_path):
    """The budget arm and its F1 twin are both cfc_lrc/ncp; the F2 label
    keeps them apart even below one listed folder."""
    s = load_f2_spec(_world(tmp_path))
    shared = tmp_path / 'shared'
    for seed in range(10):
        _lv(str(shared), 'periodic_predator_prey', 'cfc_lrc', 'ncp', seed,
            20.0, F1)
        _lv(str(shared / 'f2'), 'periodic_predator_prey', 'cfc_lrc', 'ncp',
            seed, 21.0 + seed, F2, epochs=600)
    s['budget_lv']['a']['paths'] = s['budget_lv']['b']['paths'] = [
        str(shared)]
    b = f2_family_blocks(s).set_index('test')
    assert b.loc['budget_lv', 'reading'] == 'PASS'
    assert b.loc['budget_lv', 'a'] == 'cfc_lrc/ncp [epochs=600]'
    assert b.loc['budget_lv', 'p_raw'] == pytest.approx(2 / 1024)


# --- PPO ---------------------------------------------------------------------

def test_load_ppo_runs_keeps_final_records_and_counts_the_rest(tmp_path):
    d = str(tmp_path / 'ppo')
    for s in range(3):
        _ppo(d, 'gru', s, 10.0)
    _ppo(d, 'gru', 0, 10.0, profile='smoke')
    _ppo(d, 'gru', 0, 10.0, profile='final_numeric')
    _ppo(d, 'gru', 5, 10.0, task='Hopper-v5')
    df, skipped = load_ppo_runs([d])
    assert len(df) == 3 and skipped == 3
    assert set(df['arm']) == {'gru'} and df['metric'].tolist() == [10.0] * 3
    _ppo(str(tmp_path / 'ppo' / 'again'), 'gru', 0, 11.0)
    with pytest.raises(SystemExit, match='two PPO records'):
        load_ppo_runs([d])


@pytest.mark.parametrize('n, k, expect', [(10, 2, False), (10, 3, True),
                                          (5, 1, False), (5, 2, True)])
def test_ppo_floor_rule_with_the_equality_case(tmp_path, n, k, expect):
    """eval_return_mean <= untrained mean + std counts as below (the last
    below seed sits exactly on the threshold); seeds 0-9 need 3, the
    descriptive arms with seeds 0-4 need 2."""
    d = str(tmp_path / 'ppo')
    for s in range(n):
        ret = -99.0 if s == k - 1 else (-150.0 if s < k else 50.0)
        _ppo(d, 'mlp', s, ret)
    df, _ = _f2_arm('ppo_halfcheetah', 'a', 'HalfCheetah-v5',
                    {'label': 'mlp', 'protocol_version': None,
                     'paths': [d]}, {})
    assert int(df['below'].sum()) == k
    assert not_learned(int(df['below'].sum()), len(df)) is expect


def test_ppo_non_finite_return_enters_at_the_smallest_finite_value(tmp_path):
    va = [100.0 + s for s in range(1, 11)]
    va[0] = float('nan')
    b = _blocks(_world(tmp_path, values={
        'ppo_halfcheetah': (va, [100.0] * 10)}))
    r = b.loc['ppo_halfcheetah']
    assert r['n_nonfinite'] == 1
    # seed 0 enters at 100 (the smallest finite return): a zero difference
    assert r['n_positive'] == 9 and r['method'] == 'approx'
    assert r['p_a_gt_b'] == pytest.approx(0.9)   # the non-finite fit loses


# --- labels ------------------------------------------------------------------

def _rec(pv, system='periodic_predator_prey', epochs=300, **top):
    eff = {'protocol_version': pv, 'epochs': epochs, 'ode_unfolds': 6}
    if system == 'pa':
        eff['task'] = 'person_activity'
        rec = {'cell': 'cfc_lrc', 'wiring': 'ncp', 'seed': 0,
               'param_budget': 30000, 'test_macro_f1': 0.5}
    else:
        eff.update(task='lotka_volterra', system=system)
        rec = {'cell': 'cfc_lrc', 'wiring': 'ncp', 'seed': 0,
               'system': system, 'param_budget': 4000, 'forcing': 'mixed',
               'dt_scale': 12.8, 'rollout_metrics_test': {}}
    rec.update(top)
    rec['config'] = {'effective': {**eff, **top.pop('eff', {})}}
    return rec


def test_f1_labels_unchanged_and_f2_epochs_tags():
    assert arm_label(_rec(F1)) == 'cfc_lrc/ncp'
    assert arm_label(_rec(F1, eff={'ode_unfolds': 1})) == 'cfc_lrc/ncp [k=1]'
    assert arm_label(_rec(F1, dt_scale=1)) == 'cfc_lrc/ncp [dt_scale=1]'
    assert arm_label(_rec(F1, param_budget=None)) == \
        'cfc_lrc/ncp [fixed width]'
    assert arm_label(_rec(F1, forcing='teacher')) == \
        'cfc_lrc/ncp [forcing=teacher]'
    assert arm_label(_rec(F1, param_budget=None, dt_scale=1, forcing='free',
                          eff={'ode_unfolds': 1})) == \
        'cfc_lrc/ncp [k=1, dt_scale=1, fixed width, forcing=free]'
    for system, ep in (('periodic_predator_prey', 300), ('duffing', 450),
                       ('multitimescale_lib', 450), ('pa', 50)):
        assert arm_label(_rec(F1, system, ep)) == 'cfc_lrc/ncp'
        assert arm_label(_rec(F2, system, ep)) == 'cfc_lrc/ncp'
    assert arm_label(_rec(F1, epochs=120)) == 'cfc_lrc/ncp'      # a pilot
    assert arm_label(_rec(F1, 'duffing', 900)) == 'cfc_lrc/ncp'
    assert arm_label(_rec(F2, epochs=600)) == 'cfc_lrc/ncp [epochs=600]'
    assert arm_label(_rec(F2, 'duffing', 900)) == \
        'cfc_lrc/ncp [epochs=900]'
    assert arm_label(_rec(F2, 'pa', 100)) == 'cfc_lrc/ncp [epochs=100]'
    assert arm_label(_rec(F2, epochs=600, forcing='teacher')) == \
        'cfc_lrc/ncp [forcing=teacher, epochs=600]'
    assert arm_label(_rec(F2, param_budget=None, wiring='dense3')) == \
        'cfc_lrc/dense3 [fixed width]'


# --- errors ------------------------------------------------------------------

def test_f2_record_in_an_f1_path_list_and_the_reverse_are_rejected(tmp_path):
    s = load_f2_spec(_world(tmp_path))
    # a guard re-run: the F1 arm on the F2 commit, same label
    _lv(os.path.join(s['rewiring_lv']['a']['paths'][0], 'guard'),
        'periodic_predator_prey', 'cfc_lrc', 'ncp', 0, 21.0, F2)
    with pytest.raises(SystemExit, match='guard.*protocol_version'):
        f2_family_blocks(s)
    s = load_f2_spec(_world(tmp_path / 'rev'))
    _lv(os.path.join(s['rewiring_lv']['b']['paths'][0], 'f1'),
        'periodic_predator_prey', 'cfc_lrc', 'ncp_random', 0, 21.0, F1)
    with pytest.raises(SystemExit, match='f1.*protocol_version'):
        f2_family_blocks(s)


def test_spec_rejects_patterns_missing_paths_and_wrong_keys(tmp_path):
    spec = _world(tmp_path)
    with open(spec, encoding='utf-8') as f:
        good = yaml.safe_load(f)

    def check(mutate, match):
        s = json.loads(json.dumps(good))
        mutate(s)
        path = str(tmp_path / 'bad.yaml')
        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(s, f)
        with pytest.raises(SystemExit, match=match):
            load_f2_spec(path)

    check(lambda s: s['width_pa']['b']['paths'].__setitem__(
        0, str(tmp_path / 'width_pa' / '*')), 'pattern')
    check(lambda s: s['width_pa']['b']['paths'].__setitem__(
        0, str(tmp_path / 'nowhere')), 'no such path')
    check(lambda s: s.pop('budget_lv'), r"missing \['budget_lv'\]")
    check(lambda s: s.__setitem__('extra', s['width_pa']),
          r"unknown \['extra'\]")
    check(lambda s: s['width_pa']['b'].__setitem__('seeds', [0]), 'keys')
    check(lambda s: s['width_pa'].__setitem__('c', {}), 'a, b')
    check(lambda s: s['width_pa']['b'].__setitem__('label', 'cfc_lrc/dense3'),
          'label')
    check(lambda s: s['ppo_halfcheetah']['a'].__setitem__(
        'protocol_version', F2), 'protocol_version')
    check(lambda s: s['width_pa']['a'].__setitem__('protocol_version', None),
          'protocol_version')
    assert load_f2_spec(spec)['width_pa']['a']['protocol_version'] == [F1]


def test_suffix_version_list_is_accepted(tmp_path):
    spec = _world(tmp_path, opts={})
    s = load_f2_spec(spec)
    s['rewiring_lv']['b']['protocol_version'] = [F2, F2 + '-b']
    assert f2_family_blocks(s).set_index('test').loc[
        'rewiring_lv', 'reading'] == 'PASS'


def test_missing_floor_marking_is_an_error_naming_the_arm(tmp_path):
    with pytest.raises(SystemExit, match=r'rewiring_lv\.b.*no floor marking'):
        _blocks(_world(tmp_path / 'lv', opts={
            ('rewiring_lv', 'b'): {3: {'anchors': False}}}))
    with pytest.raises(SystemExit, match=r'width_pa\.b.*confusion'):
        _blocks(_world(tmp_path / 'pa', opts={
            ('width_pa', 'b'): {0: {'conf': None}}}))
    with pytest.raises(SystemExit, match='arm gru without'):
        _blocks(_world(tmp_path / 'ppo', opts={
            ('ppo_halfcheetah', 'b'): {
                0: {'drop': ('untrained_return_std',)}}}))


def test_data_key_mismatch_names_the_seed(tmp_path):
    with pytest.raises(SystemExit, match=r'rewiring_lv.*data key.*\[4\]'):
        _blocks(_world(tmp_path / 'lv', opts={
            ('rewiring_lv', 'b'): {4: {'data_seed': DS + 1}}}))
    with pytest.raises(SystemExit, match=r'width_pa.*data key.*\[2\]'):
        _blocks(_world(tmp_path / 'pa', opts={
            ('width_pa', 'b'): {2: {'split_seed': 1}}}))


def test_template_lists_the_seven_tests_with_placeholders():
    with open(os.path.join(REPO, 'configs/aggregate/f2_families.yaml'),
              encoding='utf-8') as f:
        t = yaml.safe_load(f)
    assert set(t) == set(F2_TESTS)
    for test, (family, task, a, b) in F2_TESTS.items():
        assert [t[test][s]['label'] for s in 'ab'] == [a, b]
        assert all(p.startswith('<') for s in 'ab'
                   for p in t[test][s]['paths'])
        if family == 'PPO':
            assert t[test]['a']['protocol_version'] is None
    with pytest.raises(SystemExit, match='no such path'):
        load_f2_spec(os.path.join(REPO, 'configs/aggregate/f2_families.yaml'))
