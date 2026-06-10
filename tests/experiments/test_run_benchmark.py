# tests/experiments/test_run_benchmark.py
import json

from experiments.run_benchmark import (
    CELLS, CELLS_V2, WIRINGS, SYSTEMS, SEEDS, V2_CLIP_NORM,
    build_specs, build_specs_v2, result_filename, run_one, save_result,
)

_TINY_CFG = dict(n_iters=3, batch_size=4, batch_time=8, lr=1e-3, loss='mse',
                 grad_log_every=2, data_size=60, deterministic=False)


def test_matrix_is_thesis_matrix():
    """4 cells x 2 wirings x 6 systems x 5 seeds = 240 run specs."""
    assert CELLS == ['ltc', 'lrc', 'gru', 'lstm']
    assert WIRINGS == ['dense', 'ncp']
    assert len(SYSTEMS) == 6
    assert len(SEEDS) == 5
    assert len(build_specs(CELLS, WIRINGS, SYSTEMS, SEEDS)) == 240


def test_spec_order_is_deterministic():
    """Index order is the SLURM array contract — must be stable."""
    a = build_specs(CELLS, WIRINGS, SYSTEMS, SEEDS)
    b = build_specs(CELLS, WIRINGS, SYSTEMS, SEEDS)
    assert a == b
    assert a[0] == {'cell': 'ltc', 'wiring': 'dense', 'system': 'spiral',
                    'seed': 0, 'clip_norm': 0.0}


def test_result_filename():
    spec = {'cell': 'ltc', 'wiring': 'ncp', 'system': 'spiral', 'seed': 3}
    assert result_filename(spec) == 'ltc_ncp_spiral_seed3.json'


def test_run_one_produces_complete_result(tmp_path):
    spec = {'cell': 'gru', 'wiring': 'dense', 'system': 'spiral', 'seed': 0}
    result = run_one(spec, _TINY_CFG)

    assert result['run'] == spec
    assert len(result['training']['loss_history']) == _TINY_CFG['n_iters']
    assert result['evaluation']['mse'] >= 0
    assert result['evaluation']['nrmse'] >= 0
    assert len(result['evaluation']['trajectory_pred']) == _TINY_CFG['data_size']
    assert result['gradient_flow']['iterations'] == [2]

    path = save_result(result, str(tmp_path))
    with open(path, encoding='utf-8') as f:
        assert json.load(f)['schema_version'] == 2


def test_run_one_is_seed_reproducible():
    """Same seed twice -> identical loss history (M3a seed-handling requirement)."""
    spec = {'cell': 'gru', 'wiring': 'dense', 'system': 'spiral', 'seed': 7}
    cfg = dict(_TINY_CFG, grad_log_every=0)
    r1 = run_one(spec, cfg)
    r2 = run_one(spec, cfg)
    assert r1['training']['loss_history'] == r2['training']['loss_history']
    assert r1['evaluation']['mse'] == r2['evaluation']['mse']


def test_ncp_run_smoke(tmp_path):
    """One tiny NCP run end-to-end (covers SparseLinear + gradient tracker)."""
    spec = {'cell': 'ltc', 'wiring': 'ncp', 'system': 'duffing', 'seed': 1}
    result = run_one(spec, _TINY_CFG)
    assert len(result['training']['loss_history']) == _TINY_CFG['n_iters']
    # NCP model has 5 leaf layers (RNN, SparseLinear, RNN, SparseLinear, RNN)
    assert len(result['gradient_flow']['layer_norms']) == 5


def test_v2_matrix_counts():
    """v2 = 3 new cells x {clip off, on} (360) + ltc/lrc x clip on (120)."""
    assert CELLS_V2 == ['mm_ltc', 'mm_lrc', 'cfc']
    specs = build_specs_v2()
    assert len(specs) == 480
    unclipped = [s for s in specs if s['clip_norm'] == 0.0]
    clipped = [s for s in specs if s['clip_norm'] == V2_CLIP_NORM]
    assert len(unclipped) == 180
    assert len(clipped) == 300
    assert {s['cell'] for s in unclipped} == set(CELLS_V2)
    assert {s['cell'] for s in clipped} == set(CELLS_V2) | {'ltc', 'lrc'}


def test_v2_spec_order_is_deterministic():
    a = build_specs_v2()
    b = build_specs_v2()
    assert a == b
    assert a[0] == {'cell': 'mm_ltc', 'wiring': 'dense', 'system': 'spiral',
                    'seed': 0, 'clip_norm': 0.0}


def test_result_filename_with_clip():
    spec = {'cell': 'mm_ltc', 'wiring': 'ncp', 'system': 'spiral', 'seed': 3,
            'clip_norm': 1.0}
    assert result_filename(spec) == 'mm_ltc_ncp_spiral_seed3_clip1.0.json'


def test_result_filename_no_clip_suffix_when_zero():
    spec = {'cell': 'ltc', 'wiring': 'ncp', 'system': 'spiral', 'seed': 3,
            'clip_norm': 0.0}
    assert result_filename(spec) == 'ltc_ncp_spiral_seed3.json'


def test_run_one_v2_cell_with_clip(tmp_path):
    spec = {'cell': 'cfc', 'wiring': 'dense', 'system': 'spiral', 'seed': 0,
            'clip_norm': 0.5}
    result = run_one(spec, _TINY_CFG)
    assert result['schema_version'] == 2
    assert result['config']['clip_norm'] == 0.5
    assert result['gradient_flow']['clip']['iterations'] == [2]
    assert len(result['training']['loss_history']) == _TINY_CFG['n_iters']
