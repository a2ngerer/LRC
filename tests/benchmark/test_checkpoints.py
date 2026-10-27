# tests/benchmark/test_checkpoints.py
# Per-fit weight checkpoints (issue a2ngerer/LRC#18): each sequence runner
# writes <stem>.weights.h5 and config.effective, and rebuild_from_result
# reproduces the trained model's outputs (atol 1e-6). Synthetic data only: 4 LV
# trajectories and a 2-window Person Activity stub, no download.
import json
import sys
from types import SimpleNamespace

import numpy as np

import src.tasks.person_activity as pa_pkg
from experiments import run_committee_benchmark as co_runner
from experiments import run_lotka_volterra_benchmark as lv_runner
from experiments import run_person_activity_benchmark as pa_runner
from src.benchmark.checkpoints import rebuild_from_result
from src.tasks.lotka_volterra.datasets import load_lotka_volterra

T, F, C = 32, 7, 7


def _capture_builds(monkeypatch, owner, name):
    """Patch owner.name to keep every (model, fresh-rebuild) pair it returns."""
    real, built = getattr(owner, name), []

    def build(*a, **k):
        built.append((real(*a, **k), lambda: real(*a, **k)))
        return built[-1][0]
    monkeypatch.setattr(owner, name, build)
    return built


def _pa_stub(*_, **__):
    """load_person_activity stand-in: two random windows per split, one
    recording per split (the PersonActivityData fields of issue #26)."""
    rng = np.random.default_rng(0)
    arrays, recordings = {}, {}
    for i, split in enumerate(('train', 'test', 'val')):
        arrays[f'{split}_x'] = rng.normal(size=(2, T, F)).astype(np.float32)
        arrays[f'{split}_t'] = np.ones((2, T, 1), np.float32)
        arrays[f'{split}_y'] = rng.integers(0, C, size=(2, T)).astype(np.int32)
        recordings[split] = [f'A0{i + 1}']
        arrays[f'{split}_rec'] = np.array(recordings[split] * 2)
    return SimpleNamespace(**arrays, feature_size=F, num_classes=C, seq_len=T,
                           split='grouped', split_seed=0,
                           recordings=recordings, n_recordings=3,
                           n_negative_dt=0)


def _assert_round_trip(json_path, built, inputs):
    model, fresh = built[-1]
    rebuilt = rebuild_from_result(json_path)
    expected = model.predict(inputs, verbose=0)
    np.testing.assert_allclose(rebuilt.predict(inputs, verbose=0), expected,
                               atol=1e-6)
    # the weights were loaded, not re-initialised from the same seed
    assert not np.allclose(fresh().predict(inputs, verbose=0), expected,
                           atol=1e-6)
    record = json.loads(json_path.read_text())
    assert record['weights_path'] == json_path.name[:-5] + '.weights.h5'
    assert (json_path.parent / record['weights_path']).is_file()
    assert record['training_diverged'] is False
    return record, rebuilt


def _lv_argv(tmp_path, *extra):
    return ['run_lotka_volterra_benchmark.py', '--epochs', '2',
            '--anchors', '2,4', '--horizon', '4',
            '--n-trajectories', '4', '--seq-len', '8', '--batch-size', '4',
            '--outdir', str(tmp_path), *extra]


def test_lv_round_trip_reproduces_closed_loop_mse(tmp_path, monkeypatch):
    built = _capture_builds(monkeypatch, lv_runner,
                            'build_lotka_volterra_model')
    monkeypatch.setattr(sys, 'argv', _lv_argv(
        tmp_path, '--wiring', 'ncp', '--cell', 'cfc_lrc', '--size', '8',
        '--seed', '2'))
    lv_runner.main()
    x = np.random.default_rng(1).normal(size=(3, 8, 2)).astype(np.float32)
    record, rebuilt = _assert_round_trip(
        tmp_path / 'cfc_lrc_ncp_seed2.json', built,
        (x, np.full((3, 8, 1), 0.1, np.float32)))
    eff = record['config']['effective']
    assert eff['task'] == 'lotka_volterra'
    assert eff['backbone_layers'] == record['backbone_layers_effective']
    assert eff['cell_kwargs'] == {'elastance_type': 'asymmetric'}
    # the cluster pass rule: closed_loop_mse recomputed from the JSON alone
    data = load_lotka_volterra(n_trajectories=eff['n_trajectories'],
                               seq_len=eff['seq_len'], system=eff['system'],
                               split=eff['split'], seed=eff['data_seed'],
                               t_span=eff['t_span'])
    gen = lv_runner.closed_loop_rollout(rebuilt, data)
    truth = (data.test_traj - data.mean) / data.std
    assert abs(np.mean((gen[:, 1:] - truth[:, 1:]) ** 2)
               - record['closed_loop_mse']) < 1e-6


def test_pa_round_trip(tmp_path, monkeypatch):
    built = _capture_builds(monkeypatch, pa_pkg, 'build_person_activity_model')
    monkeypatch.setattr(pa_pkg, 'load_person_activity', _pa_stub)
    monkeypatch.setattr(sys, 'argv', [
        'run_person_activity_benchmark.py', '--wiring', 'dense3', '--cell',
        'gru', '--size', '4', '--epochs', '2', '--outdir', str(tmp_path)])
    pa_runner.main()
    stub = _pa_stub()
    record, _ = _assert_round_trip(tmp_path / 'gru_dense3_seed0.json', built,
                                   (stub.test_x, stub.test_t))
    assert record['config']['effective']['build']['wiring'] == 'dense3'


def test_committee_round_trip(tmp_path, monkeypatch):
    built = _capture_builds(monkeypatch, co_runner, 'build_committee_model')
    monkeypatch.setattr(co_runner, 'load_person_activity', _pa_stub)
    monkeypatch.setattr(sys, 'argv', [
        'run_committee_benchmark.py', '--n-columns', '2', '--size', '8',
        '--epochs', '2', '--seq-len', str(T), '--drop-fracs', '0.0',
        '--drop-seeds', '0', '--outdir', str(tmp_path)])
    co_runner.main()
    stub = _pa_stub()
    views, _ = co_runner.make_views(stub.test_x, 2, mode='identical')
    (path,) = tmp_path.glob('*.json')
    record, _ = _assert_round_trip(path, built, (views, stub.test_t))
    eff = record['config']['effective']
    assert eff['task'] == 'committee' and eff['build']['n_columns'] == 2
    assert eff['views']['mode'] == 'identical'


def test_diverged_fit_still_writes_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', _lv_argv(
        tmp_path, '--wiring', 'dense', '--cell', 'gru', '--size', '4',
        '--lr', '1e30'))
    lv_runner.main()
    record = json.loads((tmp_path / 'gru_dense_seed0.json').read_text())
    assert record['training_diverged'] is True
    assert (tmp_path / 'gru_dense_seed0.weights.h5').is_file()


def test_no_save_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', _lv_argv(
        tmp_path, '--wiring', 'dense', '--cell', 'gru', '--size', '4',
        '--no-save-weights'))
    lv_runner.main()
    record = json.loads((tmp_path / 'gru_dense_seed0.json').read_text())
    assert record['weights_path'] is None
    assert not list(tmp_path.glob('*.h5'))


def test_pre_issue23_record_rebuilds_without_self_recurrence(tmp_path,
                                                            monkeypatch):
    """A record written before #23 has no build.self_recurrent: its NCP had
    no self-recurrence and its CfC depth is config.effective.backbone_layers
    (here 0, while today's NCP default is 1)."""
    built = _capture_builds(monkeypatch, lv_runner,
                            'build_lotka_volterra_model')
    monkeypatch.setattr(sys, 'argv', _lv_argv(
        tmp_path, '--wiring', 'ncp', '--cell', 'cfc', '--size', '8',
        '--backbone-layers', '0', '--no-self-recurrence'))
    lv_runner.main()
    path = tmp_path / 'cfc_ncp_seed0_bb0.json'
    record = json.loads(path.read_text())
    eff = record['config']['effective']
    del eff['build']['self_recurrent']
    eff['cell_kwargs'].pop('backbone_layers')
    path.write_text(json.dumps(record))
    x = np.random.default_rng(1).normal(size=(3, 8, 2)).astype(np.float32)
    inputs = (x, np.full((3, 8, 1), 0.1, np.float32))
    np.testing.assert_allclose(
        rebuild_from_result(path).predict(inputs, verbose=0),
        built[-1][0].predict(inputs, verbose=0), atol=1e-6)
