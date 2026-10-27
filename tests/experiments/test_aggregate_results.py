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
    assert 'Bonferroni m=2' in text
    row = next(line for line in text.splitlines() if line.startswith('| a vs b'))
    cols = [c.strip() for c in row.strip('|').split('|')]
    assert cols[1:3] == ['7', '1']                       # n pairs, excluded
    assert cols[6] == f'{0.05 / 2:.4g}'                  # alpha adj
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


# --- sequence-runner suite (#57) -------------------------------------------

def _write_lv_fit(d, system, cell, wiring, seed, vpt, rate_unbounded=0.0,
                  non_finite=False, ode_unfolds=6, dt_scale=12.8,
                  param_budget=4000, above=False, forcing='mixed'):
    """A run_lotka_volterra_benchmark.py result JSON (flat, #62 keys)."""
    os.makedirs(d, exist_ok=True)
    m = {'vpt_multi_median': vpt, 'vpt_anchor_8': vpt + 1,
         'nrmse_32_anchor_8': 0.5, 'theta_sensitivity': {'0.3': vpt - 1,
                                                         '1.0': vpt + 2},
         'rate_unbounded': rate_unbounded, 'non_finite': non_finite}
    rec = {'wiring': wiring, 'cell': cell, 'seed': seed, 'system': system,
           'param_budget': param_budget, 'budget_deviation': 0.01,
           'budget_above_tolerance': above, 'epochs': 300, 'lr': 1e-3,
           'clipnorm': None, 'params_effective': 3960, 'dt_scale': dt_scale,
           'forcing': forcing, 'best_val_epoch': 250,
           'rollout_metrics_test': m,
           'config': {'effective': {'ode_unfolds': ode_unfolds}}}
    with open(os.path.join(d, f'{cell}_{wiring}_seed{seed}_mixed.json'), 'w',
              encoding='utf-8') as f:
        json.dump(rec, f)


def _write_pa_fit(d, cell, wiring, seed, f1, conf=None, above=False):
    """A run_person_activity_benchmark.py result JSON (indented, no
    system); with conf, test_macro_f1 is its pooled macro-F1."""
    from src.evaluation.metrics import macro_f1
    os.makedirs(d, exist_ok=True)
    if conf is not None:
        f1 = macro_f1(np.sum([np.array(c) for c in conf.values()], axis=0))
    rec = {'wiring': wiring, 'cell': cell, 'seed': seed, 'epochs': 50,
           'param_budget': 30000, 'budget_deviation': 0.02,
           'budget_above_tolerance': above, 'lr': 3e-3, 'clipnorm': 1.0,
           'params_effective': 29500, 'dt_scale': 3.7, 'best_val_epoch': 40,
           'test_macro_f1': f1, 'config': {'effective': {'ode_unfolds': 6}}}
    if conf is not None:
        rec['test_confusion_by_recording'] = conf
    with open(os.path.join(d, f'{cell}_{wiring}_seed{seed}.json'), 'w',
              encoding='utf-8') as f:
        json.dump(rec, f, indent=2)


def test_load_sequence_runs_both_runner_layouts(tmp_path):
    """One row per fit with the #57 columns; tuning seeds are stage hpo,
    control arms carry their tag, other runners' JSONs and non-fit files are
    skipped; a fit with every test rollout unbounded is diverged."""
    from experiments.aggregate_results import (HIGHER_IS_BETTER,
                                               load_sequence_runs)
    lv, pa = tmp_path / 'lv', tmp_path / 'pa'
    _write_lv_fit(lv, 'duffing', 'cfc_lrc', 'ncp', 0, 12.0)
    _write_lv_fit(lv, 'duffing', 'lrc', 'ncp', 1, 3.0, rate_unbounded=1.0)
    _write_lv_fit(lv / 'k1', 'duffing', 'lrc', 'ncp', 0, 5.0, ode_unfolds=1)
    _write_lv_fit(lv / 'dt1', 'duffing', 'cfc_lrc', 'ncp', 0, 5.0,
                  dt_scale=1.0)
    _write_lv_fit(lv / 'w', 'multitimescale_lib', 'cfc_static', 'ncp', 0,
                  5.0, param_budget=None)
    _write_lv_fit(lv / 'hpo' / 'lr0.001_clipoff_seed100', 'duffing',
                  'cfc_lrc', 'ncp', 100, 9.0)
    (lv / 'hpo' / 'hpo_selection.json').write_text('{"task": "lv"}')
    _write_pa_fit(pa, 'cfc_lrc', 'ncp', 0, 0.6,
                  {'r2': [[1, 0], [0, 1]], 'r1': [[2, 0], [0, 2]]})
    (pa / 'co_K30_seed0.json').write_text(json.dumps(
        {'cell': 'cfc_lrc', 'wiring': 'ncp', 'seed': 0, 'n_columns': 30,
         'test_macro_f1': 0.5}))
    df = load_sequence_runs([str(lv)], 'lv')
    assert len(df) == 6
    for col in ('cell', 'wiring', 'task', 'system', 'seed', 'stage', 'metric',
                'diverged', 'best_val_epoch', 'epochs', 'params_effective',
                'budget_deviation', 'lr', 'clipnorm', 'vpt_anchor_8',
                'nrmse_32_anchor_8', 'vpt_theta_0.3', 'rate_unbounded'):
        assert col in df
    row = df[(df['arm'] == 'cfc_lrc/ncp') & (df['stage'] == 'final')].iloc[0]
    assert (row['task'], row['system'], row['metric'], row['diverged'],
            row['vpt_theta_1.0'], row['epochs']) == (
        'lv', 'duffing', 12.0, False, 14.0, 300)
    assert df.loc[df['seed'] == 100, 'stage'].item() == 'hpo'
    assert df.loc[(df['cell'] == 'lrc') & (df['seed'] == 1),
                  'diverged'].item()
    assert set(df.loc[df['control'], 'arm']) == {
        'lrc/ncp [k=1]', 'cfc_lrc/ncp [dt_scale=1]',
        'cfc_static/ncp [fixed width]'}
    dp = load_sequence_runs(str(pa), 'pa')
    assert len(dp) == 1 and dp['system'].item() == 'pa'
    assert dp['metric'].item() == 1.0 and dp['clipnorm'].item() == 1.0
    # recordings sorted: r1 first
    assert dp['confusion'].item()[0].tolist() == [[2, 0], [0, 2]]
    assert HIGHER_IS_BETTER['lv'] and HIGHER_IS_BETTER['pa']


def test_load_sequence_runs_rejects_old_harness_and_duplicates(tmp_path):
    from experiments.aggregate_results import load_sequence_runs
    _write_run(str(tmp_path / 'old'), 'ltc', 0.0)
    with pytest.raises(SystemExit, match='old single-step harness'):
        load_sequence_runs(str(tmp_path / 'old'), 'lv')
    (tmp_path / 'pre62').mkdir()
    (tmp_path / 'pre62' / 'x_seed0.json').write_text(json.dumps(
        {'cell': 'gru', 'wiring': 'ncp', 'seed': 0, 'system': 'duffing',
         'rollout_metrics_test': {'vpt_median': 3}}))
    with pytest.raises(SystemExit, match='pre-#62'):
        load_sequence_runs(str(tmp_path / 'pre62'), 'lv')
    for sub in ('a', 'b'):
        _write_lv_fit(tmp_path / 'dup' / sub, 'duffing', 'gru', 'ncp', 0, 4.0)
    with pytest.raises(SystemExit, match='two final fits'):
        load_sequence_runs(str(tmp_path / 'dup'), 'lv')


def _suite(tmp_path, n=10):
    """n seeds: duffing cfc_lrc/ncp beats lrc/ncp (H-b) with distinct
    differences, seed 0 of lrc/ncp unbounded; PA cfc_lrc/ncp beats lrc/ncp
    with confusion matrices; cfc/ncp vs cfc_static/ncp on duffing tie in
    half the pairs."""
    lv, pa = tmp_path / 'lv', tmp_path / 'pa'
    for s in range(n):
        _write_lv_fit(lv, 'duffing', 'cfc_lrc', 'ncp', s, 20.0 + s)
        _write_lv_fit(lv, 'duffing', 'lrc', 'ncp', s, 10.0 + s / 2,
                      rate_unbounded=1.0 if s == 0 else 0.0)
        _write_lv_fit(lv, 'duffing', 'cfc', 'ncp', s, 10.0 + (s % 2))
        _write_lv_fit(lv, 'duffing', 'cfc_static', 'ncp', s, 10.0)
        good = {f'r{i}': [[5, 1], [1, 5 + i + s]] for i in range(4)}
        bad = {f'r{i}': [[3, 3], [2, 4 + i]] for i in range(4)}
        _write_pa_fit(pa, 'cfc_lrc', 'ncp', s, 0.8 + s / 100, good)
        _write_pa_fit(pa, 'lrc', 'ncp', s, 0.5, bad)
    _write_lv_fit(lv / 'k1', 'duffing', 'lrc', 'ncp', 0, 7.0, ode_unfolds=1)
    return lv, pa


def test_core_contrast_blocks_exact_wilcoxon_family_and_effects(tmp_path):
    """H-b on duffing and PA: ten positive differences give the exact
    two-sided p 2/1024, x m=2; the unbounded fit enters at VPT 0 and is
    counted; P(A>B) 10/10 with its Clopper-Pearson bound; the PA interval is
    the crossed seeds x recordings bootstrap; ties switch to the normal
    approximation; missing arms are reported as missing."""
    from experiments.aggregate_results import (CORE_CONTRASTS,
                                               core_contrast_blocks,
                                               load_sequence_runs)
    lv, pa = _suite(tmp_path)
    df = pd.concat([load_sequence_runs(str(lv), 'lv'),
                    load_sequence_runs(str(pa), 'pa')], ignore_index=True)
    blocks = core_contrast_blocks(df)
    assert len(blocks) == sum(len(e) for e in CORE_CONTRASTS.values()) == 14
    hb = blocks[blocks['family'] == 'H-b'].set_index('task')
    for task in ('duffing', 'pa'):
        b = hb.loc[task]
        assert (b['n_pairs'], b['status'], b['method'], b['m']) == (
            10, 'tested', 'exact', 2)
        assert b['p_raw'] == pytest.approx(2 / 1024)
        assert b['p_adj'] == pytest.approx(4 / 1024) and b['significant']
        assert b['p_a_gt_b'] == 1.0 and round(b['p_a_gt_b_lo'], 3) == 0.692
        assert b['median_diff_lo'] <= b['median_diff'] <= b['median_diff_hi']
    assert hb.loc['duffing', 'diverged_b'].startswith('1/10 ')
    assert hb.loc['duffing', 'diverged_a'].startswith('0/10 ')
    assert hb.loc['duffing', 'ci_units'] == 'seed'
    assert hb.loc['pa', 'ci_units'] == 'seed x recording'
    ha = blocks[blocks['family'] == 'H-a'].set_index('task')
    assert ha.loc['duffing', 'method'] == 'approx'      # ties and zeros
    assert ha.loc['multitimescale_lib', 'status'] == 'missing'
    assert not blocks.loc[blocks['status'] == 'missing', 'significant'].any()
    assert set(blocks['m']) == {2, 3}


def test_core_contrast_blocks_undecidable_budget_floor_and_sign(tmp_path):
    """Below eight pairs the block is undecidable (never significant); an
    arm above the budget tolerance excludes the block; floor_bound comes
    from floors.json; sign stability compares the two tuning seeds."""
    from experiments.aggregate_results import (core_contrast_blocks,
                                               hc_conjunction,
                                               load_sequence_runs)
    lv, pa = _suite(tmp_path, n=7)
    for s in range(7):
        _write_lv_fit(lv / 'ppp', 'periodic_predator_prey', 'cfc_lrc', 'ncp',
                      s, 30)
        _write_lv_fit(lv / 'ppp', 'periodic_predator_prey', 'cfc_lrc',
                      'dense3', s, 10, above=s == 3)
    df = pd.concat([load_sequence_runs(str(lv), 'lv'),
                    load_sequence_runs(str(pa), 'pa')], ignore_index=True)
    floors = {'contrasts': [{'family': 'H-b', 'task': 'duffing',
                             'a': ['cfc_lrc', 'ncp'], 'b': ['lrc', 'ncp'],
                             'floor_bound': True}]}
    hpo = {('duffing', 'cfc_lrc', 'ncp'): {'ranking': [
               {'scores': {'100': 5.0, '101': 6.0}}]},
           ('duffing', 'lrc', 'ncp'): {'ranking': [
               {'scores': {'100': 4.0, '101': 7.0}}]},
           ('pa', 'cfc_lrc', 'ncp'): {'ranking': [
               {'scores': {'100': 0.6, '101': 0.7}}]},
           ('pa', 'lrc', 'ncp'): {'ranking': [
               {'scores': {'100': 0.5, '101': 0.6}}]}}
    blocks = core_contrast_blocks(df, floors, hpo)
    hb = blocks[blocks['family'] == 'H-b'].set_index('task')
    assert (hb.loc['duffing', 'status'], hb.loc['duffing', 'n_pairs']) == (
        'undecidable', 7)
    assert hb.loc['duffing', 'p_raw'] == pytest.approx(2 / 128)
    assert not hb['significant'].any()
    assert hb.loc['duffing', 'floor_bound'] is True
    assert hb.loc['pa', 'floor_bound'] is None
    assert hb.loc['duffing', 'sign_stable_tuning'] is False
    assert hb.loc['pa', 'sign_stable_tuning'] is True
    hc = blocks[(blocks['family'] == 'H-c')
                & (blocks['task'] == 'periodic_predator_prey')]
    assert (hc['status'] == 'excluded: arm above budget tolerance').sum() == 1
    assert hc_conjunction(blocks) == {'cfc_lrc/cncp': False,
                                      'cfc_lrc/ncp': False}


def test_hc_conjunction_skips_floor_bound_tasks():
    from experiments.aggregate_results import hc_conjunction
    blocks = pd.DataFrame([
        {'family': 'H-c', 'task': t, 'a': 'cfc_lrc/ncp', 'significant': sig,
         'floor_bound': fb}
        for t, sig, fb in (('periodic_predator_prey', True, False),
                           ('duffing', False, True), ('pa', True, None))])
    assert hc_conjunction(blocks) == {'cfc_lrc/ncp': True}
    blocks.loc[1, 'floor_bound'] = False
    assert hc_conjunction(blocks) == {'cfc_lrc/ncp': False}


def test_robustness_family_runs_on_metric_robust(tmp_path):
    from experiments.aggregate_results import (ROBUSTNESS_FAMILY,
                                               core_contrast_blocks,
                                               load_sequence_runs)
    lv, _ = _suite(tmp_path)
    df = load_sequence_runs(str(lv), 'lv')
    assert ROBUSTNESS_FAMILY not in set(core_contrast_blocks(df)['family'])
    rob = core_contrast_blocks(df.assign(metric_robust=df['metric'],
                                         diverged_robust=df['diverged']))
    rob = rob[rob['family'] == ROBUSTNESS_FAMILY]
    assert len(rob) == 6 and set(rob['m']) == {2}


def test_robustness_csv_runs_the_second_family(tmp_path):
    """--robustness (#34) merges metric_robust and diverged_robust on
    (system, arm, seed) of the final fits: the H-c contrasts rerun as the
    second family (m = 2) with a seed interval (the clean confusions do not
    apply) and the robustness set's divergence flag: a fit diverged only on
    the OOD set enters at VPT 0, one diverged only on the clean test set
    keeps its OOD value. A fit without a row is left out, a tuning-seed row
    is ignored; a duplicate row, a CSV without diverged_robust and a row of
    another config raise. The report has the per-arm robustness table."""
    from experiments.aggregate_results import ROBUSTNESS_FAMILY
    from src.benchmark.checkpoints import config_hash
    lv, pa = _suite(tmp_path)
    rows = []
    for s in range(10):
        _write_lv_fit(lv, 'duffing', 'cfc_lrc', 'dense3', s, 5.0,
                      rate_unbounded=1.0 if s == 5 else 0.0)  # clean only
        _write_pa_fit(pa, 'cfc_lrc', 'dense3', s, 0.5,
                      {f'r{i}': [[3, 3], [2, 4 + i]] for i in range(4)})
        rows += [('duffing', 'cfc_lrc/ncp', s, 8.0 + s, s == 9),  # OOD only
                 ('duffing', 'cfc_lrc/dense3', s, 4.0, False),
                 ('pa', 'cfc_lrc/ncp', s, 0.7, False),
                 ('pa', 'cfc_lrc/dense3', s, 0.4 + s / 100, False)]
    csv = tmp_path / 'rob.csv'
    cols = ['system', 'arm', 'seed', 'metric_robust', 'diverged_robust']

    def write(rows, sha=config_hash({'ode_unfolds': 6})):
        pd.DataFrame(rows, columns=cols).assign(
            config_effective_sha256=sha, ood_nrmse_128_median=0.25).to_csv(
            csv, index=False)
    write(rows[1:] + [('duffing', 'cfc_lrc/ncp', 100, 1.0, False)])
    out = tmp_path / 'out'
    argv = ['--runs', str(tmp_path / 'none'), '--out', str(out),
            '--rollout-runs', str(lv), '--pa-runs', str(pa), '--contrasts',
            '--robustness', str(csv)]
    assert main(argv) == 0
    blocks = pd.read_csv(out / 'contrasts.csv')
    rob = blocks[blocks['family'] == ROBUSTNESS_FAMILY].set_index(['task',
                                                                   'a'])
    assert len(rob) == 6 and set(rob['m']) == {2}
    assert set(rob['endpoint']) == {'metric_robust'}
    d = rob.loc[('duffing', 'cfc_lrc/ncp')]
    assert (d['n_pairs'], d['status'], d['ci_units']) == (9, 'tested', 'seed')
    # seeds 1-8: 8+s-4; seed 9 diverged on the OOD set: 0-4
    assert d['median_diff'] == 8.0
    assert d['diverged_a'].startswith('1/9') and d['diverged_b'].startswith(
        '0/9')
    p = rob.loc[('pa', 'cfc_lrc/ncp')]
    assert (p['n_pairs'], p['ci_units']) == (10, 'seed')
    assert rob.loc[('pa', 'cfc_lrc/cncp'), 'status'] == 'missing'
    hc = blocks[(blocks['family'] == 'H-c') & (blocks['task'] == 'pa')
                & (blocks['a'] == 'cfc_lrc/ncp')]
    assert hc['ci_units'].item() == 'seed x recording'
    summary = (out / 'summary.md').read_text()
    assert '| duffing | cfc_lrc/dense3 | 10 | 0 | 4 | 0.25 |' in summary
    assert '| duffing | cfc_lrc/ncp | 9 | 1 |' in summary
    write(rows + rows[:1])
    with pytest.raises(SystemExit, match='two robustness rows'):
        main(argv)
    write(rows, sha='0' * 64)
    with pytest.raises(SystemExit, match='config_effective_sha256'):
        main(argv)
    pd.DataFrame(rows, columns=cols).drop(columns='diverged_robust').to_csv(
        csv, index=False)
    with pytest.raises(SystemExit, match='without diverged_robust'):
        main(argv)


def test_main_contrasts_writes_csv_and_descriptive_pairs(tmp_path):
    """--contrasts: contrasts.csv with one row per core contrast, the
    descriptive table holds the other pairs (the k=1 control included) and
    no core pair."""
    lv, pa = _suite(tmp_path)
    out = tmp_path / 'out'
    assert main(['--runs', str(tmp_path / 'none'), '--out', str(out),
                 '--rollout-runs', str(lv), '--pa-runs', str(pa),
                 '--contrasts']) == 0
    blocks = pd.read_csv(out / 'contrasts.csv')
    assert len(blocks) == 14
    runs = pd.read_csv(out / 'sequence_runs.csv')
    assert 'confusion' not in runs and len(runs) == 61
    text = (out / 'summary.md').read_text()
    desc = text.split('## Descriptive pairs')[1]
    assert '| duffing | lrc/ncp | lrc/ncp [k=1] | 1 |' in desc
    assert '| duffing | cfc_lrc/ncp | lrc/ncp |' not in desc
    assert '| H-b | 2 | duffing | cfc_lrc/ncp | lrc/ncp | 10 | tested | exact' \
        in text
    assert '- cfc_lrc/ncp vs cfc_lrc/dense3: **not supported**' in text


def test_contrasts_needs_sequence_runs(tmp_path):
    with pytest.raises(SystemExit):
        main(['--out', str(tmp_path), '--contrasts'])


def test_median_bootstrap_intervals():
    from src.evaluation.stats import (bootstrap_median_ci,
                                      crossed_median_diff_ci)
    assert bootstrap_median_ci([2.0] * 5) == (2.0, 2.0)
    lo, hi = bootstrap_median_ci(np.arange(10.0))
    assert lo < 4.5 < hi
    # identical fits on every seed: only the recording draw varies
    a = np.array([[[[9, 1], [1, 9]], [[5, 5], [5, 5]]]] * 4, float)
    b = np.array([[[[5, 5], [5, 5]], [[5, 5], [5, 5]]]] * 4, float)
    lo, hi = crossed_median_diff_ci(a, b)
    assert 0.0 <= lo < hi <= 0.4
    assert crossed_median_diff_ci(a, a) == (0.0, 0.0)
