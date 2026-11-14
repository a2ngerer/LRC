# tests/experiments/test_eval_cncp_ablation.py
# Issue a2ngerer/LRC#136: experiments/eval_cncp_ablation.py on 2-epoch
# cfc_lrc/cncp fits of each task. One row per (fit, ablation); ablation
# 'none' reproduces the stored endpoint (exactly on Person Activity, within
# the rebuild tolerance on the rollouts); the sources stay byte-identical.
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import pytest

import src.tasks.person_activity as pa_pkg
from experiments import eval_cncp_ablation as ab
from experiments import run_person_activity_benchmark as pa_runner
from experiments.eval_robustness import REBUILD_TOL
from src.benchmark.checkpoints import config_hash
from tests.benchmark.test_checkpoints import _pa_stub
from tests.experiments.test_eval_robustness import _lv_fit


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_script_on_two_epoch_fits_of_each_task(tmp_path, monkeypatch):
    _lv_fit(tmp_path / 'lv', monkeypatch, 'periodic_predator_prey', 'cncp')
    _lv_fit(tmp_path / 'duffing', monkeypatch, 'duffing', 'cncp', seed=1)
    monkeypatch.setattr(pa_pkg, 'load_person_activity', _pa_stub)
    monkeypatch.setattr(ab, 'load_person_activity', _pa_stub)
    monkeypatch.setattr(sys, 'argv', [
        'run_person_activity_benchmark.py', '--wiring', 'cncp', '--cell',
        'cfc_lrc', '--size', '8', '--epochs', '2', '--outdir',
        str(tmp_path / 'pa')])
    pa_runner.main()
    (tmp_path / 'lv' / 'hpo_selection.json').write_text('{"task": "lv"}')
    sources = sorted(p for p in tmp_path.glob('*/*')
                     if p.suffix in ('.json', '.h5'))
    before = {p: _sha(p) for p in sources}

    with pytest.raises(SystemExit, match='source folder'):
        ab.main(['--results', str(tmp_path / 'lv' / '*.json'),
                 '--out', str(tmp_path / 'lv' / 'ablation.csv')])
    out = tmp_path / 'out' / 'ablation.csv'
    out.parent.mkdir()
    ab.main(['--results', str(tmp_path / 'lv' / '*.json'),
             str(tmp_path / 'duffing' / '*.json'),
             str(tmp_path / 'pa' / '*.json'), '--out', str(out)])

    assert {p: _sha(p) for p in sources} == before
    assert not (tmp_path / 'lv' / 'ablation.csv').exists()
    df = pd.read_csv(out, float_precision='round_trip')
    assert len(df) == 3 * len(ab.ABLATIONS)
    for file, fit in df.groupby('file'):
        record = json.loads(open(file).read())
        eff = record['config']['effective']
        assert list(fit['ablation']) == list(ab.ABLATIONS)
        assert set(fit['config_effective_sha256']) == {config_hash(eff)}
        assert set(fit['protocol_version']) == {eff['protocol_version']}
        none = fit[fit['ablation'] == 'none'].iloc[0]
        assert fit['rebuild_rel_dev'].notna().sum() == 1
        if none['task'] == 'pa':
            assert none['endpoint'] == 'test_macro_f1'
            assert none['metric'] == record['test_macro_f1']
            assert fit['hnorm_mean'].isna().all()
        else:
            assert none['endpoint'] == 'test_vpt_multi_median'
            assert none['metric_stored'] == record['rollout_metrics_test'][
                'vpt_multi_median']
            assert none['rebuild_rel_dev'] <= REBUILD_TOL
            assert (fit['hnorm_max'] >= fit['hnorm_mean']).all()
            assert (fit['hnorm_mean'] > 0).all()
            # every ablation moves the free-run states at fixed weights
            assert np.all(fit['hnorm_mean'].iloc[1:].to_numpy()
                          != none['hnorm_mean'])
