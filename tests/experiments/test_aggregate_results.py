# tests/experiments/test_aggregate_results.py
import json
import os

from experiments.aggregate_results import load_runs


def _write_run(dirpath, cell, clip_norm, seed=0, nrmse=0.5):
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
