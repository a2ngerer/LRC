# tests/experiments/test_pa_crossed_robustness.py
# Issue a2ngerer/LRC#96: crossed bootstrap (seeds x test recordings) for the
# Person Activity robustness family. eval_robustness.py writes per-recording,
# per-draw confusions on request and reproduces the stored scalars bit for
# bit; the aggregator reports the crossed interval next to the seed interval.
import json
import os

import numpy as np
import pandas as pd
import pytest

from experiments import eval_robustness as er
from experiments.aggregate_results import (ROBUSTNESS_FAMILY,
                                           core_contrast_blocks,
                                           load_sequence_runs,
                                           merge_robustness)
from experiments.suite_manifest import expand, load
from src.benchmark.checkpoints import config_hash
from src.evaluation.metrics import macro_f1
from src.evaluation.stats import crossed_median_diff_ci
from tests.experiments.test_aggregate_results import _write_pa_fit
from tests.experiments.test_eval_robustness import _pa_fit

COL = f'confusions_sigma_{er.SIGMA_DECISION:g}'


def test_crossed_ci_accepts_noise_draws():
    """A draw axis of identical matrices equals the plain (seeds, recordings,
    C, C) input; the macro-F1 is averaged over distinct draws."""
    rng = np.random.default_rng(0)
    a = rng.integers(1, 20, (6, 4, 3, 3)).astype(float)
    b = rng.integers(1, 20, (6, 4, 3, 3)).astype(float)
    plain = crossed_median_diff_ci(a, b, n_boot=2000)
    same = crossed_median_diff_ci(np.repeat(a[:, :, None], 3, 2),
                                  np.repeat(b[:, :, None], 3, 2), n_boot=2000)
    np.testing.assert_allclose(same, plain, rtol=1e-12)
    mixed = crossed_median_diff_ci(np.stack([a, b], 2), np.stack([b, a], 2),
                                   n_boot=2000)
    assert mixed == (0.0, 0.0)      # draw 1 of A = draw 2 of B and vice versa


def _two_recordings(monkeypatch):
    """Person Activity stub whose two test windows are two recordings."""
    stub = er.load_person_activity

    def load(*a, **k):
        data = stub(*a, **k)
        data.test_rec = np.array(['A02', 'A03'])
        return data
    monkeypatch.setattr(er, 'load_person_activity', load)


def test_confusion_flag_is_off_by_default_and_keeps_the_scalars(
        tmp_path, monkeypatch):
    """Without the flag no confusions column; with it, per sigma a JSON
    {recording: [draws x C x C]} whose sum reproduces the scalar, and every
    other column equal bit for bit (the control of #96)."""
    path = _pa_fit(tmp_path, monkeypatch, 2)
    _two_recordings(monkeypatch)
    plain, flagged = tmp_path / 'plain.csv', tmp_path / 'flagged.csv'
    er.main(['--results', str(path), '--out', str(plain)])
    er.main(['--results', str(path), '--out', str(flagged),
             '--per-recording-confusions', '--verify-against', str(plain)])
    a, b = (pd.read_csv(f, float_precision='round_trip')
            for f in (plain, flagged))
    assert not [c for c in a if c.startswith('confusions_')]
    assert {f'confusions_sigma_{s:g}' for s in er.SIGMAS} <= set(b)
    extra = [c for c in b if c.startswith('confusions_')]
    pd.testing.assert_frame_equal(b.drop(columns=extra), a, check_exact=True)
    conf = json.loads(b[COL].iloc[0])
    assert sorted(conf) == ['A02', 'A03']
    assert np.array(conf['A02']).shape[0] == er.N_DRAWS
    assert len(json.loads(b['confusions_sigma_0'].iloc[0])['A02']) == 1
    pooled = np.sum([np.array(v) for v in conf.values()], axis=0)
    assert np.mean([macro_f1(m) for m in pooled]) == b['f1_sigma_0.3'].iloc[0]


def test_verify_against_rejects_a_changed_or_missing_scalar(
        tmp_path, monkeypatch):
    path = _pa_fit(tmp_path, monkeypatch, 2)
    stored = tmp_path / 'stored.csv'
    er.main(['--results', str(path), '--out', str(stored)])
    rows = pd.read_csv(stored, float_precision='round_trip')
    run = ['--results', str(path), '--out', str(tmp_path / 'new.csv'),
           '--verify-against']
    er.main([*run, str(tmp_path / 'stored*.csv')])      # globs are expanded
    rows['f1_sigma_0.3'] = np.nextafter(rows['f1_sigma_0.3'], 2.0)
    rows.to_csv(stored, index=False)
    with pytest.raises(SystemExit, match=r'f1_sigma_0\.3 .* != stored'):
        er.main([*run, str(stored)])
    rows.assign(seed=rows['seed'] + 1).to_csv(stored, index=False)
    with pytest.raises(SystemExit, match='0 stored rows'):
        er.main([*run, str(stored)])
    with pytest.raises(SystemExit, match='no CSV matches'):
        er.main([*run, str(tmp_path / 'none*.csv')])


# --- aggregator ----------------------------------------------------------------

def _cm(acc, n=100):
    """Two balanced classes at accuracy ``acc``: macro-F1 = acc."""
    k = int(round(n * acc))
    return [[k, n - k], [n - k, k]]


def _suite(tmp_path, effects_a, effects_b, draws=2):
    """Ten seeds of cfc_lrc/ncp vs cfc_lrc/dense3 with one noisy confusion per
    recording and draw: accuracy 0.6 / 0.5 + the recording effect + a small
    seed shift; returns the loaded + merged frame."""
    pa = tmp_path / 'pa'
    rows = []
    for s in range(10):
        for wiring, base, eff in (('ncp', 0.6, effects_a),
                                  ('dense3', 0.5, effects_b)):
            acc = [base + e + 0.01 * (s % 3) for e in eff]
            _write_pa_fit(str(pa), 'cfc_lrc', wiring, s, 0.5, {
                f'R{i}': _cm(a) for i, a in enumerate(acc)})
            conf = {f'R{i}': [_cm(a)] * draws for i, a in enumerate(acc)}
            f1 = float(np.mean([macro_f1(np.sum([m[d] for m in conf.values()],
                                                axis=0))
                                for d in range(draws)]))
            rows.append({'system': 'pa', 'arm': f'cfc_lrc/{wiring}',
                         'seed': s, 'metric_robust': f1,
                         'diverged_robust': False, COL: json.dumps(conf),
                         'config_effective_sha256': config_hash(
                             {'ode_unfolds': 6})})
    csv = tmp_path / 'rob.csv'
    pd.DataFrame(rows).to_csv(csv, index=False)
    return merge_robustness(load_sequence_runs(str(pa), 'pa'), [str(csv)])


def _block(df):
    blocks = core_contrast_blocks(df)
    b = blocks[(blocks['family'] == ROBUSTNESS_FAMILY)
               & (blocks['task'] == 'pa') & (blocks['a'] == 'cfc_lrc/ncp')]
    return b[b['b'] == 'cfc_lrc/dense3'].iloc[0]


def test_crossed_interval_widens_with_recording_variance(tmp_path):
    """Same seed noise in both fixtures. Equal recordings: A - B is the same
    in every recording and the crossed interval is not wider than the seed
    one. Recordings in which A beats B by 0.3 and loses by 0.1 alternately:
    resampling the recordings widens it far beyond the seed interval."""
    flat = np.zeros(6)
    het = np.array([0.2, -0.2, 0.2, -0.2, 0.2, -0.2])
    homog = _block(_suite(tmp_path / 'h', flat, flat))
    mixed = _block(_suite(tmp_path / 'm', het, -het / 2))
    for b in (homog, mixed):
        assert b['ci_units'] == 'seed x recording'
        assert b['endpoint'] == 'metric_robust' and b['n_pairs'] == 10
    width = lambda b, kind: b[f'median_diff{kind}_hi'] - b[f'median_diff{kind}_lo']  # noqa: E731
    assert width(homog, '') <= width(homog, '_seed') + 1e-12
    assert width(mixed, '') > 3 * width(mixed, '_seed')
    assert width(mixed, '') > 3 * width(homog, '')
    assert homog['median_diff'] == pytest.approx(0.1)


def test_main_reports_both_intervals(tmp_path):
    """--contrasts --robustness: the PA robustness rows carry the crossed
    interval and the seed interval next to it in contrasts.csv and in the
    report; the confusion arrays stay out of sequence_runs.csv."""
    from experiments.aggregate_results import main
    het = np.array([0.2, -0.2, 0.2, -0.2, 0.2, -0.2])
    _suite(tmp_path, het, -het / 2)
    out = tmp_path / 'out'
    assert main(['--runs', str(tmp_path / 'none'), '--out', str(out),
                 '--pa-runs', str(tmp_path / 'pa'), '--contrasts',
                 '--robustness', str(tmp_path / 'rob.csv')]) == 0
    blocks = pd.read_csv(out / 'contrasts.csv')
    rob = blocks[(blocks['family'] == ROBUSTNESS_FAMILY)
                 & (blocks['task'] == 'pa') & (blocks['a'] == 'cfc_lrc/ncp')]
    assert rob['ci_units'].item() == 'seed x recording'
    assert rob['median_diff_seed_lo'].notna().all()
    assert 'confusion_robust' not in pd.read_csv(out / 'sequence_runs.csv')
    assert '(seed x recording); seed [' in (out / 'summary.md').read_text()


def test_robustness_rows_without_confusions_stay_seed_only(tmp_path):
    df = _suite(tmp_path, np.zeros(6), np.zeros(6)).assign(
        confusion_robust=None)
    b = _block(df)
    assert b['ci_units'] == 'seed'
    assert (b['median_diff_seed_lo'], b['median_diff_seed_hi']) == (
        b['median_diff_lo'], b['median_diff_hi'])


def test_merge_rejects_fits_with_other_test_recordings(tmp_path):
    pa = tmp_path / 'pa'
    for s in range(2):
        _write_pa_fit(str(pa), 'cfc_lrc', 'ncp', s, 0.5)
    sha = config_hash({'ode_unfolds': 6})
    rows = [{'system': 'pa', 'arm': 'cfc_lrc/ncp', 'seed': s,
             'metric_robust': 0.5, 'diverged_robust': False,
             'config_effective_sha256': sha,
             COL: json.dumps({f'R{s}': [_cm(0.5)]})} for s in range(2)]
    pd.DataFrame(rows).to_csv(tmp_path / 'r.csv', index=False)
    with pytest.raises(SystemExit, match='different test|differ in their test'):
        merge_robustness(load_sequence_runs(str(pa), 'pa'),
                         [str(tmp_path / 'r.csv')])


# --- manifest ------------------------------------------------------------------

YAML = 'configs/suite/eval/pa_crossed_robustness.yaml'


def test_eval_manifest_of_the_pa_final_fits():
    """30 eval-only lines (3 H-c arms x seeds 0-9), no training runner; each
    adds the confusions and verifies against the stored CSVs, and all name
    the old rollout seed streams (#101: this rerun stays on v1)."""
    lines = expand(load(YAML))
    assert len(lines) == 30
    for line in lines:
        argv = line['argv']
        assert argv[3] == 'experiments/eval_robustness.py'
        assert '--per-recording-confusions' in argv
        assert '--verify-against' in argv
        assert argv[argv.index('--seed-streams') + 1] == 'v1'
        assert '--outdir' not in argv
        assert line['done'].endswith('.csv')
    assert {ln['id'].split('/')[1].rsplit('_seed', 1)[0] for ln in lines} == {
        'cfc_lrc_ncp', 'cfc_lrc_dense3', 'cfc_lrc_cncp'}
    assert os.path.isfile(YAML)


def test_eval_extra_args_are_pa_only(tmp_path):
    cfg = {'task': 'lv', 'system': 'duffing', 'stage': 'eval',
           'extra_args': ['--x'], 'sources': [
               {'config': 'configs/suite/duffing_final.yaml',
                'records_dir': 'r'}]}
    p = tmp_path / 'ev.yaml'
    import yaml
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(SystemExit, match='extra_args applies to task pa'):
        load(str(p))
