# tests/experiments/test_aggregate_results.py
import json
import os

import numpy as np
import pandas as pd
import pytest

from experiments.aggregate_results import (_test_block, load_runs, main,
                                           pa_arm_means)
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
