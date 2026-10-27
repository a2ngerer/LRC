# tests/experiments/test_aggregate_results.py
import json
import os

import numpy as np
import pandas as pd
import pytest

from experiments.aggregate_results import (BONFERRONI_FAMILIES, _test_block,
                                           consensus_family, load_runs, main,
                                           not_learned, pa_arm_means)
from src.evaluation.stats import kendall_tau_b, p_a_gt_b


def _write_run(dirpath, cell, clip_norm, seed=0, nrmse=0.5, **top):
    os.makedirs(dirpath, exist_ok=True)
    name = f'{cell}_dense_spiral_seed{seed}'
    if clip_norm:
        name += f'_clip{clip_norm}'
    payload = {
        'schema_version': 2,
        'run': {'cell': cell, 'wiring': 'dense', 'system': 'spiral',
                'seed': seed, 'clip_norm': clip_norm},
        'config': {'clip_norm': clip_norm},
        'training': {'final_loss': 0.1, 'duration_s': 1.0},
        'evaluation': {'mse': 0.2, 'nrmse': nrmse},
        **top,
    }
    with open(os.path.join(dirpath, name + '.json'), 'w', encoding='utf-8') as f:
        json.dump(payload, f)


def test_load_runs_multiple_dirs(tmp_path):
    d1, d2 = str(tmp_path / 'runs'), str(tmp_path / 'runs_v2')
    _write_run(d1, 'ltc', 0.0)
    _write_run(d2, 'mm_ltc', 0.0)
    df = load_runs([d1, d2])
    assert sorted(df['cell']) == ['ltc', 'mm_ltc']


def test_load_runs_clip_variant_label(tmp_path):
    d = str(tmp_path / 'runs_v2')
    _write_run(d, 'ltc', 1.0)
    _write_run(d, 'ltc', 0.0, seed=1)
    df = load_runs([d])
    assert sorted(df['cell']) == ['ltc', 'ltc+clip']
    assert sorted(df['clip_norm']) == [0.0, 1.0]


def test_load_runs_schema1_backward_compatible(tmp_path):
    """v1 result JSONs (schema 1, no clip_norm anywhere) still load."""
    d = str(tmp_path / 'runs')
    os.makedirs(d, exist_ok=True)
    payload = {
        'schema_version': 1,
        'run': {'cell': 'gru', 'wiring': 'ncp', 'system': 'duffing', 'seed': 2},
        'config': {},
        'training': {'final_loss': 0.3, 'duration_s': 2.0},
        'evaluation': {'mse': 0.4, 'nrmse': 0.6},
    }
    with open(os.path.join(d, 'gru_ncp_duffing_seed2.json'), 'w',
              encoding='utf-8') as f:
        json.dump(payload, f)
    df = load_runs([d])
    assert list(df['cell']) == ['gru']
    assert list(df['clip_norm']) == [0.0]


def test_load_runs_v5_stress_variant_label(tmp_path):
    """v5 stress runs get a '+<regime>' cell label and a 'stress' column, so a
    clean v4 run and its stressed v5 counterpart are distinct paired variants."""
    d = str(tmp_path / 'runs_v5')
    os.makedirs(d, exist_ok=True)
    for regime in ('noise', 'extrapolation', 'ood_init'):
        payload = {
            'schema_version': 2,
            'run': {'cell': 'cfc_lrc', 'wiring': 'ncp', 'system': 'spiral',
                    'seed': 0, 'clip_norm': 0.0, 'stress': regime},
            'config': {'clip_norm': 0.0, 'stress': regime},
            'training': {'final_loss': 0.1, 'duration_s': 1.0},
            'evaluation': {'mse': 0.2, 'nrmse': 0.5},
        }
        fname = f'cfc_lrc_ncp_spiral_seed0_stress-{regime}.json'
        with open(os.path.join(d, fname), 'w', encoding='utf-8') as f:
            json.dump(payload, f)
    df = load_runs([d])
    assert sorted(df['cell']) == ['cfc_lrc+extrapolation', 'cfc_lrc+noise',
                                  'cfc_lrc+ood_init']
    assert sorted(df['stress']) == ['extrapolation', 'noise', 'ood_init']


def test_load_runs_v6_wiring_and_level_variant_labels(tmp_path):
    """v6a wiring graphs get a +w<seed> label, v6b levels get a +lvl<v> label, so
    each verification condition is a distinct paired variant and seed 42 (the v5
    anchor) carries NO +w token (it merges with the v5 baseline by design)."""
    d = str(tmp_path / 'runs_v6')
    os.makedirs(d, exist_ok=True)

    def write(name, run, cfg):
        payload = {'schema_version': 2, 'run': run, 'config': cfg,
                   'training': {'final_loss': 0.1, 'duration_s': 1.0},
                   'evaluation': {'mse': 0.2, 'nrmse': 0.5}}
        with open(os.path.join(d, name), 'w', encoding='utf-8') as f:
            json.dump(payload, f)

    # v6a: new wiring graph 7 under noise
    write('cfc_ncp_spiral_seed0_stress-noise_wseed7.json',
          {'cell': 'cfc', 'wiring': 'ncp', 'system': 'spiral', 'seed': 0,
           'clip_norm': 0.0, 'stress': 'noise', 'ncp_wiring_seed': 7},
          {'clip_norm': 0.0, 'stress': 'noise', 'ncp_wiring_seed': 7})
    # v5 anchor graph 42: NO +w token
    write('cfc_ncp_spiral_seed0_stress-noise.json',
          {'cell': 'cfc', 'wiring': 'ncp', 'system': 'spiral', 'seed': 0,
           'clip_norm': 0.0, 'stress': 'noise', 'ncp_wiring_seed': 42},
          {'clip_norm': 0.0, 'stress': 'noise', 'ncp_wiring_seed': 42})
    # v6b: noise level 0.2
    write('cfc_ncp_spiral_seed0_stress-noise_lvl0.2.json',
          {'cell': 'cfc', 'wiring': 'ncp', 'system': 'spiral', 'seed': 0,
           'clip_norm': 0.0, 'stress': 'noise', 'stress_noise_level': 0.2},
          {'clip_norm': 0.0, 'stress': 'noise', 'stress_level': 0.2})

    df = load_runs([d])
    assert sorted(df['cell']) == ['cfc+noise', 'cfc+noise+lvl0.2', 'cfc+noise+w7']
    assert set(df['ncp_wiring_seed'].dropna()) == {7, 42}
    assert 0.2 in set(df['stress_level'].dropna())


def test_p_a_gt_b_exact_intervals():
    """5/5 and 9/10 paired wins give the exact Clopper-Pearson 95 % bounds."""
    p, lo, hi = p_a_gt_b(np.ones(5), np.zeros(5))
    assert (p, round(lo, 3), hi) == (1.0, 0.478, 1.0)
    p, lo, hi = p_a_gt_b(np.r_[np.ones(9), 0.0], np.r_[np.zeros(9), 1.0])
    assert (p, round(lo, 3), round(hi, 3)) == (0.9, 0.555, 0.997)


def test_p_a_gt_b_divergence_and_ties():
    """A diverged run loses (flag or non-finite value), two diverged runs tie,
    equal values tie."""
    x = np.array([np.nan, np.inf, 1.0, 5.0, 2.0])
    y = np.array([0.0, np.nan, 3.0, 0.0, 2.0])
    div_y = np.array([False, False, False, True, False])
    # pair scores: A non-finite 0, both diverged 0.5, A loses 0,
    # B flagged 1 (despite 5 > 0 anyway), tie 0.5
    p, _, _ = p_a_gt_b(x, y, diverged_y=div_y)
    assert p == pytest.approx(2.0 / 5)
    # a flagged A loses even when its finite value is the larger one
    assert p_a_gt_b([9.0], [0.0], diverged_x=[True])[0] == 0.0


def test_kendall_tau_b_identical_and_reversed():
    assert kendall_tau_b([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert kendall_tau_b([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_test_block_finite_pairs_and_family_alpha():
    """Wilcoxon/d_z exclude the non-finite pair (count printed), P(A>B) keeps
    it (A wins against the diverged B run), alpha is adjusted by the family."""
    seeds = range(7)
    df = pd.DataFrame(
        [{'cell': 'a', 'wiring': 'dense', 'system': 'spiral', 'seed': s,
          'nrmse': 0.1 + 0.01 * s} for s in seeds]
        + [{'cell': 'b', 'wiring': 'dense', 'system': 'spiral', 'seed': s,
            'nrmse': np.nan if s == 0 else 0.5 + 0.02 * s} for s in seeds])
    text = _test_block(df, [('a vs b', {'cell': 'a', 'wiring': 'dense'},
                             {'cell': 'b', 'wiring': 'dense'})], 'T',
                       family='H-c')
    assert 'Bonferroni m=12' in text
    row = next(line for line in text.splitlines() if line.startswith('| a vs b'))
    cols = [c.strip() for c in row.strip('|').split('|')]
    assert cols[1:3] == ['7', '1']                       # n pairs, excluded
    assert cols[6] == f'{0.05 / 12:.4g}'                 # alpha adj
    assert cols[-1] == '1.00 [0.590, 1.000]'             # 7/7 wins, lower nrmse


def test_budget_flag_excludes_the_contrast_from_a_confirmatory_family(
        tmp_path):
    """budget_above_tolerance (#59) is read (False when absent) and drops a
    contrast with a flagged arm from a pre-registered family only."""
    d = str(tmp_path / 'runs')
    for s in range(6):
        _write_run(d, 'a', 0.0, seed=s, nrmse=0.1 + 0.01 * s)
        _write_run(d, 'b', 0.0, seed=s, nrmse=0.5 + 0.02 * s,
                   budget_above_tolerance=s == 0)
    df = load_runs([d])
    assert df.loc[df['cell'] == 'a', 'budget_above_tolerance'].eq(False).all()
    assert df['budget_above_tolerance'].sum() == 1
    pairs = [('a vs b', {'cell': 'a', 'wiring': 'dense'},
              {'cell': 'b', 'wiring': 'dense'})]
    confirmatory = _test_block(df, pairs, 'T', family='H-c')
    assert 'excluded: arm above budget tolerance' in confirmatory
    assert 'excluded: arm above budget tolerance' not in _test_block(
        df, pairs, 'T')


def _write_hpo(root, system, cell, wiring, scores, edge=False):
    d = os.path.join(root, f'{cell}_{wiring}_{system}')
    os.makedirs(d, exist_ok=True)
    payload = {'task': 'lv', 'system': system, 'cell': cell, 'wiring': wiring,
               'lr_at_grid_edge': edge, 'grid_extended': edge,
               'ranking': [{'lr': lr, 'clipnorm': None, 'score': sc}
                           for lr, sc in zip((1e-3, 3e-3), scores)]}
    with open(os.path.join(d, 'hpo_selection.json'), 'w', encoding='utf-8') as f:
        json.dump(payload, f)


def test_hpo_fragility_flags_and_sign_stability(tmp_path):
    """--hpo-dir alone (no run JSONs) reports grid-edge flags per arm and flags
    a core contrast whose sign flips under the second-best configurations."""
    hpo = str(tmp_path / 'hpo')
    _write_hpo(hpo, 'duffing', 'cfc_lrc', 'ncp', (0.9, 0.5), edge=True)
    _write_hpo(hpo, 'duffing', 'lrc', 'ncp', (0.8, 0.6))
    out = str(tmp_path / 'out')
    assert main(['--runs', str(tmp_path / 'none'), '--out', out,
                 '--hpo-dir', hpo]) == 0
    with open(os.path.join(out, 'summary.md'), encoding='utf-8') as f:
        text = f.read()
    assert '| duffing | cfc_lrc/ncp | 0.001 | True | True |' in text
    assert '| duffing | lrc/ncp | 0.001 | False | False |' in text
    assert '| H-b | duffing | cfc_lrc/ncp | lrc/ncp | +1 | -1 | NO |' in text


def test_pa_arm_means_hierarchical_bootstrap(tmp_path):
    """Seeds that agree give a zero-width seed bootstrap; resampling the test
    recordings as well widens the interval around the same mean."""
    d = str(tmp_path / 'pa')
    os.makedirs(d)
    per_rec = {f'rec{i}': v for i, v in enumerate((0.2, 0.4, 0.6, 0.8, 1.0))}
    for seed in range(5):
        with open(os.path.join(d, f'cfc_lrc_dense3_seed{seed}.json'), 'w',
                  encoding='utf-8') as f:
            json.dump({'cell': 'cfc_lrc', 'wiring': 'dense3', 'seed': seed,
                       'test_macro_f1': 0.6,
                       'test_macro_f1_per_recording': per_rec}, f)
    assert '| cfc_lrc_dense3 | 5 | 0.6000 | [0.6000, 0.6000] |' in pa_arm_means(d)
    row = pa_arm_means(d, 'seed,recording').splitlines()[-1]
    lo, hi = (float(v) for v in row.split('[')[1].rstrip('] |').split(','))
    assert '| cfc_lrc_dense3 | 5 | 0.6000 |' in row and lo < 0.6 < hi


def _write_rollout(d, tag, seed, rates, sens, non_finite, n=16):
    m = {'n_trajectories': n, 'tau_orb': 0.15, 'non_finite': non_finite,
         **{f'rate_{c}': v for c, v in zip(('ok', 'bounded_wrong', 'unbounded'), rates)},
         'outcome_classes_tau_0.30': {
             f'rate_{c}': v for c, v in zip(('ok', 'bounded_wrong', 'unbounded'), sens)}}
    with open(os.path.join(d, f'{tag}_seed{seed}.json'), 'w', encoding='utf-8') as f:
        json.dump({'system': 'periodic_predator_prey', 'seed': seed,
                   'rollout_metrics_test': m}, f)


def test_rollout_outcome_classes_pool_trajectories_with_cp_and_flag_non_finite(
        tmp_path):
    """Class shares pool the test trajectories of all fits of an arm (exact
    Clopper-Pearson), once at tau_orb and once at 0.30; a non-finite fit gives
    the arm one label; files without outcome classes are skipped. A control
    run folder with the same file names is its own arm, and HPO tuning fits
    (run_grid_hpo layout, seed 100) are never pooled into the final arm."""
    from src.evaluation.stats import clopper_pearson
    d = tmp_path / 'lv'
    hpo = d / 'hpo' / 'cfc_ncp' / 'lr0.01_clip1_seed100'
    os.makedirs(d / 'ctrl_k1')
    os.makedirs(hpo)
    _write_rollout(d, 'cfc_ncp_mixed', 0, (1, 0, 0), (1, 0, 0), False)
    _write_rollout(d, 'cfc_ncp_mixed', 1, (0.5, 0.25, 0.25),
                   (0.75, 0, 0.25), True)
    _write_rollout(d / 'ctrl_k1', 'cfc_ncp_mixed', 0, (0, 1, 0), (0, 1, 0), False)
    _write_rollout(hpo, 'cfc_ncp_mixed', 100, (0, 0, 1), (0, 0, 1), True)
    _write_rollout(d, 'gru_dense_mixed', 0, (0.5, 0.5, 0), (1, 0, 0), False)
    (d / 'hpo_selection.json').write_text('{"task": "lv"}')
    (d / 'old_seed0.json').write_text(
        '{"system": "duffing", "seed": 0, "rollout_metrics_test": {"rate_drift": 1}}')
    (d / 'noseed.json').write_text(json.dumps(
        {'rollout_metrics_test': {'outcome_classes_tau_0.30': {}}}))
    out = str(tmp_path / 'out')
    assert main(['--runs', str(tmp_path / 'none'), '--out', out,
                 '--rollout-runs', str(d)]) == 0
    with open(os.path.join(out, 'summary.md'), encoding='utf-8') as f:
        text = f.read()

    def share(k, n):
        lo, hi = clopper_pearson(k, n)
        return f'{k / n:.2f} [{lo:.2f}, {hi:.2f}]'

    lo, hi = clopper_pearson(1, 2)
    label = f'at least 1 of 2 fits non-finite (1/2 [{lo:.2f}, {hi:.2f}])'
    assert (f'| periodic_predator_prey | cfc_ncp_mixed | 2 | 32 | 0.15 | '
            f'{share(24, 32)} | {share(4, 32)} | {share(4, 32)} | {label} |') in text
    assert (f'| periodic_predator_prey | cfc_ncp_mixed | 2 | 32 | 0.3 | '
            f'{share(28, 32)} | {share(0, 32)} | {share(4, 32)} | {label} |') in text
    assert (f'| periodic_predator_prey | gru_dense_mixed | 1 | 16 | 0.15 | '
            f'{share(8, 16)} | {share(8, 16)} | {share(0, 16)} | — |') in text
    assert (f'| periodic_predator_prey | ctrl_k1/cfc_ncp_mixed | 1 | 16 | 0.15 | '
            f'{share(0, 16)} | {share(16, 16)} | {share(0, 16)} | — |') in text
    assert 'lr0.01' not in text
    assert 'Skipped 3 file(s) without outcome classes.' in text
    assert 'Left out 1 tuning-seed fit(s) (seed outside 0..9).' in text


FLOOR_DRAW = {'system': 'periodic_predator_prey', 'n_trajectories': 10,
              'seq_len': 32, 'split': [6, 2, 2], 't_span': [0.0, 10.0],
              'ic_spread': 0.4}


def _write_floor_fit(d, cell, wiring, seed, vpt, data_seed):
    """A rollout fit with the #62 multi-anchor keys (anchor 8, horizon 16)."""
    m = {'vpt_multi_median': vpt, 'anchors': [8], 'anchor_horizon': 16,
         'anchor_theta': 0.5}
    eff = {**FLOOR_DRAW, 'data_seed': data_seed}
    with open(os.path.join(d, f'{cell}_{wiring}_seed{seed}_mixed.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'system': eff['system'], 'cell': cell, 'wiring': wiring,
                   'seed': seed, 'rollout_metrics_test': m,
                   'config': {'effective': eff}}, f)


def test_seed_rule_more_than_a_fifth_of_the_seeds():
    assert [not_learned(k, 5) for k in (0, 1, 2)] == [False, False, True]
    assert [not_learned(k, 10) for k in (2, 3)] == [False, True]


def test_floors_json_marks_arms_by_the_seed_rule_and_flags_contrasts(
        tmp_path):
    """floors.json: persistence/AR(1) per data seed (val and test) recomputed
    from config.effective; each fit is compared with the test floor of its own
    data draw; one of five seeds below keeps an arm above floor, two make it
    not learned and every core contrast with it floor_bound. Tuning seeds and
    files without the multi-anchor keys are left out."""
    from src.evaluation.rollout_metrics import floor_lines
    from src.tasks.lotka_volterra.datasets import (DATA_SEED_BASE,
                                                   load_lotka_volterra)
    d = tmp_path / 'lv'
    os.makedirs(d / 'hpo')
    floors = {}
    for seed in range(5):
        data = load_lotka_volterra(
            n_trajectories=10, seq_len=32, split=(6, 2, 2),
            seed=DATA_SEED_BASE + seed, system=FLOOR_DRAW['system'])
        floors[seed] = floor_lines(data, (8,), 16)
    for seed in range(5):
        fl = floors[seed]['test']['floor']
        for wiring, n_below in (('ncp', 0), ('dense3', 1), ('cncp', 2)):
            _write_floor_fit(d, 'cfc_lrc', wiring, seed,
                             fl - 1 if seed < n_below else fl,
                             DATA_SEED_BASE + seed)
    _write_floor_fit(d / 'hpo', 'cfc_lrc', 'ncp', 100, 0, DATA_SEED_BASE + 999)
    (d / 'old_seed0.json').write_text(json.dumps(
        {'system': 'duffing', 'seed': 0, 'rollout_metrics_test': {'vpt': 3}}))
    out = str(tmp_path / 'out')
    assert main(['--runs', str(tmp_path / 'none'), '--out', out,
                 '--rollout-runs', str(d), '--floors']) == 0
    with open(os.path.join(out, 'floors.json'), encoding='utf-8') as f:
        got = json.load(f)
    assert sorted(x['data_seed'] for x in got['floors']) == [
        DATA_SEED_BASE + s for s in range(5)]
    for x in got['floors']:
        ref = floors[x['data_seed'] - DATA_SEED_BASE]
        for split in ('val', 'test'):
            for line in ('persistence', 'ar1'):
                assert x[split][line]['vpt_multi_median'] == \
                    ref[split][line]['vpt_multi_median']
            assert x[split]['floor'] == ref[split]['floor']
    arms = {a['arm']: a for a in got['arms']}
    assert set(arms) == {f'cfc_lrc_{w}_mixed' for w in ('ncp', 'dense3', 'cncp')}
    assert arms['cfc_lrc_dense3_mixed']['seeds_below'] == [0]
    assert [arms[f'cfc_lrc_{w}_mixed']['above_floor']
            for w in ('ncp', 'dense3', 'cncp')] == [True, True, False]
    hc = {c['a'][1]: c['floor_bound'] for c in got['contrasts']
          if c['family'] == 'H-c'}
    assert hc == {'ncp': False, 'cncp': True}
    assert all(c['task'] == 'periodic_predator_prey' for c in got['contrasts'])
    with open(os.path.join(out, 'summary.md'), encoding='utf-8') as f:
        text = f.read()
    assert '| periodic_predator_prey | cfc_lrc_cncp_mixed | 5 | [0, 1] ' \
        '| NO (not learned) |' in text


def test_floors_needs_rollout_runs(tmp_path):
    with pytest.raises(SystemExit):
        main(['--out', str(tmp_path), '--floors'])


def _write_co(dirpath, wiring, seed, f1_m15, auc_m0, auc_m15, view_average='off',
              vote_lambda=None, n_columns=30, above=False):
    """A committee runner JSON with the fault-grid endpoints (#55)."""
    grid = [0, 3, 9, 15, 20]
    tag = f'{wiring}_{view_average}_{vote_lambda}_K{n_columns}_seed{seed}'
    with open(os.path.join(dirpath, f'{tag}.json'), 'w') as f:
        json.dump({'cell': 'cfc_lrc', 'wiring': wiring, 'seed': seed,
                   'n_columns': n_columns, 'view_mode': 'identical',
                   'train_corruption': 'on', 'view_average': view_average,
                   'vote_lambda': vote_lambda, 'consensus': 'mean',
                   'budget_above_tolerance': above, 'fault_counts': grid,
                   'test_macro_f1_by_fault': [0.7, 0.6, 0.5, f1_m15, 0.3],
                   'per_position_macro_f1_auc': [auc_m0, 0, 0, auc_m15, 0]}, f)


def test_consensus_family_one_sided_bonferroni_and_attribution(tmp_path):
    d = str(tmp_path)
    for s in range(10):
        e = 0.01 * s                    # distinct differences, no ties
        for w in ('ncp', 'cncp'):
            # learned consensus beats input averaging (part i) and lambda=0
            # on the AUC (part ii); at m=15 it beats lambda=0 only for ncp
            _write_co(d, w, s, 0.6 + e, 0.8 + e, 0.6 + e)
            _write_co(d, w, s, (0.5 if w == 'ncp' else 0.7) + e / 2,
                      0.7, 0.5, vote_lambda=0.0)
            _write_co(d, w, s, 0.6 + e, 0.8 + e, 0.6 + e, n_columns=3)
        _write_co(d, 'ncp', s, 0.4 + e / 3, 0.9, 0.2, view_average='mean')
    assert BONFERRONI_FAMILIES['H-d'] == 4
    report = consensus_family(d)
    rows = [r for r in report.splitlines() if r.startswith('| cfc_lrc')]
    assert len(rows) == 4 and all('| 10 |' in r and 'YES' in r for r in rows)
    # exact one-sided p of 10 positive differences is 1/1024, x4 Bonferroni
    assert all('0.0009766 | 0.003906' in r for r in rows)
    assert 'cfc_lrc/ncp: learned lambda vs lambda=0 at m=15' in report
    assert '-> attribution: **consensus**' in report
    assert 'cfc_lrc/cncp' in report and '**late fusion**' in report


def test_consensus_family_excludes_arm_above_budget(tmp_path):
    d = str(tmp_path)
    for s in range(6):
        _write_co(d, 'ncp', s, 0.6 + s / 100, 0.8, 0.6, above=(s == 0))
        _write_co(d, 'ncp', s, 0.4, 0.7, 0.5, vote_lambda=0.0)
    report = consensus_family(d)
    assert 'missing arm or arm above budget tolerance' in report
    assert main(['--runs', str(tmp_path / 'none'), '--co-runs', d,
                 '--out', str(tmp_path / 'out')]) == 0
