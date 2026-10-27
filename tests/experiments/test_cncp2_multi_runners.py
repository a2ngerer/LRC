# tests/experiments/test_cncp2_multi_runners.py
# cncp2_multi through the runners (issue a2ngerer/LRC#10): the LV and PA flag
# handling, a one-epoch Lotka-Volterra run writing the new JSON fields and the
# filename suffix, and a one-epoch committee run on the partial views.
import json
import sys
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import run_committee_benchmark as co_runner
from experiments import run_lotka_volterra_benchmark as lv_runner
from experiments.run_person_activity_benchmark import result_tag
from src.tasks.person_activity.model import multi_column_config

_FLAGS = dict(columns=None, partition=None, coupling=None, readout=None,
              backbone_layers=None)


def test_multi_column_config_defaults_and_ignore_warning():
    args = SimpleNamespace(wiring='cncp2_multi', cell='cfc_lrc', seed=1,
                           **{**_FLAGS, 'columns': 4, 'partition': 'region'})
    cfg = multi_column_config(args)
    assert cfg == dict(columns=4, partition='region', coupling='lateral',
                       readout='concat')
    assert result_tag(args, cfg) == \
        'cfc_lrc_cncp2_multi_seed1_K4_region_lateral_concat'
    assert multi_column_config(SimpleNamespace(
        wiring='cncp2_multi', **_FLAGS))['readout'] == 'mean'

    other = SimpleNamespace(wiring='ncp', cell='cfc_lrc', seed=0,
                            **{**_FLAGS, 'columns': 2})
    with pytest.warns(UserWarning, match='only apply to cncp2_multi'):
        assert multi_column_config(other) == {}
    assert result_tag(other, {}) == 'cfc_lrc_ncp_seed0'
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        multi_column_config(SimpleNamespace(wiring='ncp', **_FLAGS))


def test_lv_runner_one_epoch_writes_multi_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', [
        'run_lotka_volterra_benchmark.py', '--wiring', 'cncp2_multi',
        '--columns', '2', '--cell', 'cfc_lrc', '--epochs', '1', '--size', '4',
        '--n-trajectories', '10', '--seq-len', '8', '--batch-size', '4',
        '--outdir', str(tmp_path), '--probe-stability'])
    lv_runner.main()
    path = tmp_path / 'cfc_lrc_cncp2_multi_seed0_K2_same_lateral_mean.json'
    record = json.loads(path.read_text())
    assert (record['columns'], record['partition'], record['coupling'],
            record['readout']) == (2, 'same', 'lateral', 'mean')
    assert record['params_effective'] > 0
    assert np.isfinite(record['teacher_forced_mse'])
    names = [k.split('/')[-1] for k in record['stability']['norms_init']]
    assert names == ['G0', 'S0', 'D0', 'R0', 'G1', 'S1', 'D1', 'R1']


def test_committee_runner_one_epoch_writes_drop_accs(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', [
        'run_committee_benchmark.py', '--wiring', 'cncp2_multi',
        '--n-columns', '2', '--size', '4', '--epochs', '1',
        '--consensus', 'median', '--vote-lambda', '0.5',
        '--drop-fracs', '0.0,0.3', '--drop-seeds', '0',
        '--batch-size', '256', '--outdir', str(tmp_path)])
    with pytest.warns(UserWarning, match='vote-lambda'):
        co_runner.main()
    (path,) = tmp_path.glob('*.json')
    assert path.name == \
        'cfc_lrc_cncp2_multi_K2_s4_identical_median_lr0.001_clipoff_cs0_seed0.json'
    record = json.loads(path.read_text())
    assert len(record['drop_accs']) == 2
    assert all(0.0 <= a <= 1.0 for a in record['drop_accs'])
    assert record['readout'] == 'median' and record['coupling'] == 'none'
    assert record['vote_lambda'] is None
