# tests/experiments/test_run_benchmark.py
import json

import pytest

from experiments.run_benchmark import (
    CELLS, CELLS_V2, CELLS_V3_1, CELLS_V3_2, CELLS_V3_3, CELLS_V4, CELLS_V5, CELLS_V6, WIRINGS,
    SYSTEMS, SEEDS, SEEDS_V4_BASE, SEEDS_V4_TAIL, SYSTEMS_V4_TAIL,
    SEEDS_V5, STRESS_REGIMES_V5, WIRING_SEEDS_V6A, STRESS_LEVELS_V6B, NCP_WIRING_SEED,
    V2_CLIP_NORM, CELL_UNITS, CELL_KWARGS,
    build_specs, build_specs_v2, build_specs_v3, build_specs_v3_1, build_specs_v3_2,
    build_specs_v3_3, build_specs_v4, build_specs_v5, build_specs_v6a, build_specs_v6b,
    result_filename, run_one, save_result,
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


def test_v3_matrix_counts():
    """v3 = robustness (480) + solver-fidelity (120) + training-horizon (120)."""
    from experiments.run_benchmark import (
        CELLS_V3, CELLS_V3_STIFF, SYSTEMS_V3_STIFF,
        SEEDS_V3_EXTRA, SEEDS_V3_FULL, V3_ODE_UNFOLDS, V3_BATCH_TIME,
    )
    specs = build_specs_v3()
    assert len(specs) == 720
    robustness, solver, horizon = specs[:480], specs[480:600], specs[600:]

    # robustness: 4 ODE cells x 2 wirings x 6 systems x 10 extra seeds, baseline
    assert len(robustness) == 480
    assert {s['cell'] for s in robustness} == set(CELLS_V3)
    assert {s['seed'] for s in robustness} == set(SEEDS_V3_EXTRA)
    assert all('ode_unfolds' not in s and 'batch_time' not in s for s in robustness)

    # solver-fidelity: 2 stiff cells x 2 wirings x 2 stiff systems x 15 seeds
    assert len(solver) == 120
    assert {s['cell'] for s in solver} == set(CELLS_V3_STIFF)
    assert {s['system'] for s in solver} == set(SYSTEMS_V3_STIFF)
    assert {s['seed'] for s in solver} == set(SEEDS_V3_FULL)
    assert all(s['ode_unfolds'] == V3_ODE_UNFOLDS for s in solver)
    assert all('batch_time' not in s for s in solver)

    # training-horizon: same matrix, batch_time override
    assert len(horizon) == 120
    assert all(s['batch_time'] == V3_BATCH_TIME for s in horizon)
    assert all('ode_unfolds' not in s for s in horizon)


def test_v3_spec_order_is_deterministic():
    a = build_specs_v3()
    b = build_specs_v3()
    assert a == b
    assert a[0] == {'cell': 'ltc', 'wiring': 'dense', 'system': 'spiral',
                    'seed': 5, 'clip_norm': 0.0}
    assert a[480]['ode_unfolds'] == 24
    assert a[600]['batch_time'] == 64


def test_v1_v2_contracts_unchanged_by_v3():
    """v3 must not perturb the frozen v1/v2 SLURM array contracts."""
    v1 = build_specs(CELLS, WIRINGS, SYSTEMS, SEEDS)
    assert len(v1) == 240
    assert v1[0] == {'cell': 'ltc', 'wiring': 'dense', 'system': 'spiral',
                     'seed': 0, 'clip_norm': 0.0}
    assert len(build_specs_v2()) == 480


def test_result_filename_v3_suffixes():
    assert result_filename({'cell': 'ltc', 'wiring': 'ncp', 'system': 'duffing',
                            'seed': 5}) == 'ltc_ncp_duffing_seed5.json'
    assert result_filename({'cell': 'ltc', 'wiring': 'ncp', 'system': 'duffing',
                            'seed': 0, 'ode_unfolds': 24}) == \
        'ltc_ncp_duffing_seed0_unfolds24.json'
    assert result_filename({'cell': 'mm_ltc', 'wiring': 'dense', 'system': 'duffing',
                            'seed': 0, 'batch_time': 64}) == \
        'mm_ltc_dense_duffing_seed0_bt64.json'


def test_run_one_v3_ode_unfolds_threaded():
    """ode_unfolds reaches the LTC cell and is recorded in the result config."""
    spec = {'cell': 'ltc', 'wiring': 'dense', 'system': 'spiral', 'seed': 0,
            'clip_norm': 0.0, 'ode_unfolds': 24}
    result = run_one(spec, _TINY_CFG)
    assert result['config']['ode_unfolds'] == 24
    assert len(result['training']['loss_history']) == _TINY_CFG['n_iters']


def test_run_one_v3_batch_time_override():
    """Per-spec batch_time overrides cfg and is recorded in the result config."""
    spec = {'cell': 'mm_ltc', 'wiring': 'dense', 'system': 'spiral', 'seed': 0,
            'clip_norm': 0.0, 'batch_time': 12}
    cfg = dict(_TINY_CFG, batch_time=8)
    result = run_one(spec, cfg)
    assert result['config']['batch_time'] == 12


def test_v3_ode_unfolds_changes_behavior():
    """Same seed, different ode_unfolds -> different loss trajectory. Proves the
    value reaches the LTC solver, not just the config record (behavioral check)."""
    base = {'cell': 'ltc', 'wiring': 'dense', 'system': 'spiral', 'seed': 0,
            'clip_norm': 0.0}
    r_low = run_one({**base, 'ode_unfolds': 1}, _TINY_CFG)
    r_high = run_one({**base, 'ode_unfolds': 24}, _TINY_CFG)
    assert r_low['training']['loss_history'] != r_high['training']['loss_history']


def test_v3_1_matrix_counts():
    """v3.1 = closed-form LRC ablation: 4 cells x 2 wirings x 6 systems x 5 seeds."""
    assert CELLS_V3_1 == ['cfc', 'cfc_lrc', 'cfc_pm', 'lrc']
    specs = build_specs_v3_1()
    assert len(specs) == 240
    assert {s['cell'] for s in specs} == set(CELLS_V3_1)
    assert {s['wiring'] for s in specs} == set(WIRINGS)
    assert {s['seed'] for s in specs} == set(SEEDS)
    assert all(s['clip_norm'] == 0.0 for s in specs)
    assert all('ode_unfolds' not in s and 'batch_time' not in s for s in specs)


def test_v3_1_spec_order_is_deterministic():
    a = build_specs_v3_1()
    b = build_specs_v3_1()
    assert a == b
    assert a[0] == {'cell': 'cfc', 'wiring': 'dense', 'system': 'spiral',
                    'seed': 0, 'clip_norm': 0.0}


def test_v3_1_filenames_have_no_variant_suffix():
    """v3.1 sets no clip/unfolds/bt, so filenames are bare cell_wiring_system_seed."""
    assert result_filename({'cell': 'cfc_lrc', 'wiring': 'dense',
                            'system': 'spiral', 'seed': 0}) == \
        'cfc_lrc_dense_spiral_seed0.json'
    assert result_filename({'cell': 'cfc_pm', 'wiring': 'ncp',
                            'system': 'duffing', 'seed': 4}) == \
        'cfc_pm_ncp_duffing_seed4.json'


def test_run_one_cfc_lrc(tmp_path):
    """End-to-end tiny run for the new closed-form LRC cell; elastance kwarg forwarded."""
    spec = {'cell': 'cfc_lrc', 'wiring': 'dense', 'system': 'spiral', 'seed': 0,
            'clip_norm': 0.0}
    result = run_one(spec, _TINY_CFG)
    assert result['schema_version'] == 2
    assert result['config']['cell_kwargs'] == {'elastance_type': 'asymmetric'}
    assert len(result['training']['loss_history']) == _TINY_CFG['n_iters']
    assert result['evaluation']['nrmse'] >= 0


def test_run_one_cfc_pm_forwards_backbone_units():
    """cfc_pm is plain CfC with a wider backbone (parameter-matched capacity control)."""
    spec = {'cell': 'cfc_pm', 'wiring': 'dense', 'system': 'spiral', 'seed': 0,
            'clip_norm': 0.0}
    result = run_one(spec, _TINY_CFG)
    assert result['config']['cell_kwargs'] == {'backbone_units': 20}
    assert len(result['training']['loss_history']) == _TINY_CFG['n_iters']


def test_v3_2_matrix_counts():
    """v3.2 = LRC 2x2 architecture ablation: 4 cells x 2 wirings x 6 systems x 5 seeds."""
    assert CELLS_V3_2 == ['lrc', 'cfc_lrc', 'mm_lrc', 'cfc_mm_lrc']
    specs = build_specs_v3_2()
    assert len(specs) == 240
    assert {s['cell'] for s in specs} == set(CELLS_V3_2)
    assert {s['wiring'] for s in specs} == set(WIRINGS)
    assert {s['seed'] for s in specs} == set(SEEDS)
    assert all(s['clip_norm'] == 0.0 for s in specs)
    assert all('ode_unfolds' not in s and 'batch_time' not in s for s in specs)


def test_v3_2_spec_order_is_deterministic():
    a = build_specs_v3_2(); b = build_specs_v3_2()
    assert a == b
    assert a[0] == {'cell': 'lrc', 'wiring': 'dense', 'system': 'spiral',
                    'seed': 0, 'clip_norm': 0.0}


def test_run_one_cfc_mm_lrc(tmp_path):
    """End-to-end tiny run for the mixed-memory closed-form LRC cell (dense)."""
    spec = {'cell': 'cfc_mm_lrc', 'wiring': 'dense', 'system': 'spiral', 'seed': 0,
            'clip_norm': 0.0}
    result = run_one(spec, _TINY_CFG)
    assert result['schema_version'] == 2
    assert result['config']['cell_kwargs'] == {'elastance_type': 'asymmetric'}
    assert len(result['training']['loss_history']) == _TINY_CFG['n_iters']
    assert result['evaluation']['nrmse'] >= 0


def test_run_one_cfc_mm_lrc_ncp():
    """cfc_mm_lrc in NCP wiring (mixed-memory state [h, c] across 3 stacked layers)."""
    spec = {'cell': 'cfc_mm_lrc', 'wiring': 'ncp', 'system': 'duffing', 'seed': 1,
            'clip_norm': 0.0}
    result = run_one(spec, _TINY_CFG)
    assert len(result['training']['loss_history']) == _TINY_CFG['n_iters']


def test_v3_3_matrix_counts():
    """v3.3 = param-matched capacity control: lrc_pm x 2 wirings x 6 systems x 5 seeds."""
    assert CELLS_V3_3 == ['lrc_pm']
    specs = build_specs_v3_3()
    assert len(specs) == 60
    assert {s['cell'] for s in specs} == {'lrc_pm'}
    assert {s['wiring'] for s in specs} == set(WIRINGS)


def test_lrc_pm_capacity_override_recorded():
    """lrc_pm is a widened plain LRC: run_one records the per-cell units override
    (24, not the default 16), while a normal cell is unchanged (backward compat)."""
    assert CELL_UNITS['lrc_pm'] == 24
    r_pm = run_one({'cell': 'lrc_pm', 'wiring': 'dense', 'system': 'spiral',
                    'seed': 0, 'clip_norm': 0.0}, _TINY_CFG)
    assert r_pm['config']['dense_units'] == 24
    assert r_pm['config']['cell_kwargs'] == {'elastance_type': 'asymmetric'}
    r_lrc = run_one({'cell': 'lrc', 'wiring': 'dense', 'system': 'spiral',
                     'seed': 0, 'clip_norm': 0.0}, _TINY_CFG)
    assert r_lrc['config']['dense_units'] == 16


# --- v4: cross-family generalization + classical championship ---

def test_v4_cell_set():
    """v4 = LTC 2x2 + LRC 2x2 + classical/CT baselines, 11 cells."""
    assert CELLS_V4 == ['ltc', 'cfc', 'mm_ltc', 'cfc_mm_ltc',
                        'lrc', 'cfc_lrc', 'mm_lrc', 'cfc_mm_lrc',
                        'gru', 'lstm', 'ctrnn']
    # the cross-family 2x2 closed-form+mixed-memory partners are both present
    assert 'cfc_mm_ltc' in CELLS_V4 and 'cfc_mm_lrc' in CELLS_V4


def test_v4_matrix_counts():
    """base 11x2x6x5=660 + tail 11x2x2x5=220 (extra seeds on stiff systems) = 880."""
    specs = build_specs_v4()
    assert len(specs) == 880
    assert {s['cell'] for s in specs} == set(CELLS_V4)
    assert {s['wiring'] for s in specs} == set(WIRINGS)
    assert all(s['clip_norm'] == 0.0 for s in specs)
    # no v3 solver/horizon overrides leak into v4 specs
    assert all('ode_unfolds' not in s and 'batch_time' not in s for s in specs)
    # every cell appears the same number of times (balanced design)
    counts = {c: sum(s['cell'] == c for s in specs) for c in CELLS_V4}
    assert set(counts.values()) == {80}


def test_v4_tail_supplement_seeds_and_systems():
    """The tail adds seeds 5-9 ONLY on the two divergence-prone systems."""
    assert SEEDS_V4_BASE == [0, 1, 2, 3, 4]
    assert SEEDS_V4_TAIL == [5, 6, 7, 8, 9]
    assert set(SYSTEMS_V4_TAIL) == {'duffing', 'periodic_predator_prey'}
    specs = build_specs_v4()
    # high seeds occur only on the tail systems; never on the other four
    for s in specs:
        if s['seed'] >= 5:
            assert s['system'] in SYSTEMS_V4_TAIL
    # each non-tail system has exactly seeds 0-4
    non_tail = [sy for sy in SYSTEMS if sy not in SYSTEMS_V4_TAIL]
    for sy in non_tail:
        seeds = {s['seed'] for s in specs if s['system'] == sy}
        assert seeds == set(SEEDS_V4_BASE)
    # each tail system has seeds 0-9
    for sy in SYSTEMS_V4_TAIL:
        seeds = {s['seed'] for s in specs if s['system'] == sy}
        assert seeds == set(SEEDS_V4_BASE) | set(SEEDS_V4_TAIL)


def test_v4_spec_order_is_deterministic():
    a = build_specs_v4(); b = build_specs_v4()
    assert a == b
    assert a[0] == {'cell': 'ltc', 'wiring': 'dense', 'system': 'spiral',
                    'seed': 0, 'clip_norm': 0.0}


def test_v4_new_cells_take_no_kwargs():
    """The CfC kwarg trap: cfc_mm_ltc / ctrnn must NOT be in CELL_KWARGS (CfC and
    CTRNN inner cells reject elastance_type and would crash every run)."""
    assert 'cfc_mm_ltc' not in CELL_KWARGS
    assert 'ctrnn' not in CELL_KWARGS


def test_v4_filenames_have_no_variant_suffix():
    """v4 is a baseline-config matrix: plain {cell}_{wiring}_{system}_seed{n}.json."""
    for s in build_specs_v4()[:5]:
        assert result_filename(s) == (
            f"{s['cell']}_{s['wiring']}_{s['system']}_seed{s['seed']}.json")


def test_run_one_cfc_mm_ltc_dense(tmp_path):
    """End-to-end tiny run for the v4 protagonist (closed-form mixed-memory LTC)."""
    spec = {'cell': 'cfc_mm_ltc', 'wiring': 'dense', 'system': 'spiral', 'seed': 0,
            'clip_norm': 0.0}
    result = run_one(spec, _TINY_CFG)
    assert result['schema_version'] == 2
    assert result['config']['cell_kwargs'] == {}  # no elastance_type forwarded
    assert len(result['training']['loss_history']) == _TINY_CFG['n_iters']
    assert result['evaluation']['nrmse'] >= 0


def test_run_one_cfc_mm_ltc_ncp():
    """cfc_mm_ltc in NCP wiring (the regime Q1 is about)."""
    result = run_one({'cell': 'cfc_mm_ltc', 'wiring': 'ncp', 'system': 'duffing',
                      'seed': 1, 'clip_norm': 0.0}, _TINY_CFG)
    assert len(result['training']['loss_history']) == _TINY_CFG['n_iters']


def test_run_one_ctrnn(tmp_path):
    """ctrnn baseline (never benchmarked before v4) runs end-to-end."""
    result = run_one({'cell': 'ctrnn', 'wiring': 'ncp', 'system': 'spiral',
                      'seed': 0, 'clip_norm': 0.0}, _TINY_CFG)
    assert result['config']['cell_kwargs'] == {}
    assert len(result['training']['loss_history']) == _TINY_CFG['n_iters']


def test_v1_v3_contracts_unchanged_by_v4():
    """Adding v4 must not perturb earlier profiles' spec lists."""
    assert len(build_specs(CELLS, WIRINGS, SYSTEMS, SEEDS)) == 240
    assert len(build_specs_v3_2()) == 240
    assert len(build_specs_v3_3()) == 60


# --- v5: generalization stress test (noise / extrapolation / ood_init) ---

def test_v5_cell_set_is_full_v4_set():
    """v5 reruns the entire v4 cell set under stress (no new cells)."""
    assert CELLS_V5 == CELLS_V4
    assert len(CELLS_V5) == 11


def test_v5_regimes():
    assert STRESS_REGIMES_V5 == ['noise', 'extrapolation', 'ood_init']
    assert SEEDS_V5 == [0, 1, 2, 3, 4]


def test_v5_matrix_counts():
    """3 regimes x 11 cells x 2 wirings x 6 systems x 5 seeds = 1980."""
    specs = build_specs_v5()
    assert len(specs) == 1980
    assert {s['cell'] for s in specs} == set(CELLS_V5)
    assert {s['wiring'] for s in specs} == set(WIRINGS)
    assert {s['stress'] for s in specs} == set(STRESS_REGIMES_V5)
    assert all(s['clip_norm'] == 0.0 for s in specs)
    # no v3 solver/horizon overrides leak into v5 specs
    assert all('ode_unfolds' not in s and 'batch_time' not in s for s in specs)
    # balanced: every cell appears the same number of times (3 regimes x 2 x 6 x 5)
    counts = {c: sum(s['cell'] == c for s in specs) for c in CELLS_V5}
    assert set(counts.values()) == {180}
    # 660 specs per regime
    for regime in STRESS_REGIMES_V5:
        assert sum(s['stress'] == regime for s in specs) == 660


def test_v5_regime_major_order():
    """Array contract: regime-major blocks of 660, cell-major within each."""
    specs = build_specs_v5()
    assert all(s['stress'] == 'noise' for s in specs[:660])
    assert all(s['stress'] == 'extrapolation' for s in specs[660:1320])
    assert all(s['stress'] == 'ood_init' for s in specs[1320:])
    assert specs[0] == {'cell': 'ltc', 'wiring': 'dense', 'system': 'spiral',
                        'seed': 0, 'clip_norm': 0.0, 'stress': 'noise'}


def test_v5_spec_order_is_deterministic():
    a = build_specs_v5(); b = build_specs_v5()
    assert a == b


def test_v5_filenames_have_stress_suffix():
    """v5 files carry a _stress-<regime> suffix so they never collide with v4."""
    assert result_filename({'cell': 'cfc', 'wiring': 'ncp', 'system': 'duffing',
                            'seed': 2, 'stress': 'noise'}) == \
        'cfc_ncp_duffing_seed2_stress-noise.json'
    assert result_filename({'cell': 'cfc_mm_ltc', 'wiring': 'dense',
                            'system': 'spiral', 'seed': 0,
                            'stress': 'extrapolation'}) == \
        'cfc_mm_ltc_dense_spiral_seed0_stress-extrapolation.json'


def test_v5_does_not_perturb_earlier_contracts():
    """Adding v5 must not change the frozen v1/v4 SLURM array contracts."""
    assert len(build_specs(CELLS, WIRINGS, SYSTEMS, SEEDS)) == 240
    assert len(build_specs_v4()) == 880
    # a clean (no-stress) v4 spec still produces a bare filename
    assert result_filename({'cell': 'ltc', 'wiring': 'ncp', 'system': 'spiral',
                            'seed': 0}) == 'ltc_ncp_spiral_seed0.json'


@pytest.mark.parametrize("regime", ['noise', 'extrapolation', 'ood_init'])
def test_run_one_v5_regime_end_to_end(regime):
    """Each stress regime runs end-to-end and records its regime in the config."""
    spec = {'cell': 'gru', 'wiring': 'dense', 'system': 'spiral', 'seed': 0,
            'clip_norm': 0.0, 'stress': regime}
    result = run_one(spec, _TINY_CFG)
    assert result['schema_version'] == 2
    assert result['config']['stress'] == regime
    assert result['run']['stress'] == regime
    assert len(result['training']['loss_history']) == _TINY_CFG['n_iters']
    assert result['evaluation']['nrmse'] >= 0


def test_run_one_v5_ncp_bio_cell():
    """A bio cell under stress on the NCP wiring (the regime v5 is about)."""
    spec = {'cell': 'cfc_mm_lrc', 'wiring': 'ncp', 'system': 'duffing', 'seed': 1,
            'clip_norm': 0.0, 'stress': 'ood_init'}
    result = run_one(spec, _TINY_CFG)
    assert result['config']['stress'] == 'ood_init'
    assert result['config']['cell_kwargs'] == {'elastance_type': 'asymmetric'}
    assert len(result['training']['loss_history']) == _TINY_CFG['n_iters']


def test_run_one_clean_path_has_no_stress_key():
    """A spec without 'stress' keeps the v1-v4 clean behavior (no stress in config)."""
    result = run_one({'cell': 'gru', 'wiring': 'dense', 'system': 'spiral',
                      'seed': 0, 'clip_norm': 0.0}, _TINY_CFG)
    assert 'stress' not in result['config']


# --- v6a: multi-wiring-seed robustness ---

def test_v6_cell_subset():
    """v6 uses the 8-cell representative subset (4 closed-form bio + 3 classical + lrc)."""
    assert CELLS_V6 == ['cfc', 'cfc_lrc', 'cfc_mm_lrc', 'cfc_mm_ltc',
                        'gru', 'lstm', 'ctrnn', 'lrc']
    # all v6 cells are valid benchmark cells (subset of v4)
    assert set(CELLS_V6) <= set(CELLS_V4)


def test_v6a_matrix_counts():
    """v6a = 4 new wiring graphs x 8 cells x ncp x 6 systems x 5 seeds = 960."""
    assert WIRING_SEEDS_V6A == [7, 13, 21, 99]
    specs = build_specs_v6a()
    assert len(specs) == 960
    assert {s['cell'] for s in specs} == set(CELLS_V6)
    assert {s['wiring'] for s in specs} == {'ncp'}          # ncp-only (the hard wiring)
    assert {s['stress'] for s in specs} == {'noise'}        # the most fragile regime
    assert {s['ncp_wiring_seed'] for s in specs} == set(WIRING_SEEDS_V6A)
    assert NCP_WIRING_SEED not in {s['ncp_wiring_seed'] for s in specs}  # seed 42 reused from v5
    # balanced: every (cell, graph) appears 6 systems x 5 seeds = 30 times
    counts = {(s['cell'], s['ncp_wiring_seed']) for s in specs}
    assert len(counts) == 8 * 4


def test_v6a_wiring_seed_major_order():
    """Array contract: wiring-seed-major blocks of 240, cell-major within."""
    specs = build_specs_v6a()
    assert all(s['ncp_wiring_seed'] == 7 for s in specs[:240])
    assert all(s['ncp_wiring_seed'] == 99 for s in specs[720:])
    assert specs[0] == {'cell': 'cfc', 'wiring': 'ncp', 'system': 'spiral',
                        'seed': 0, 'clip_norm': 0.0, 'stress': 'noise', 'ncp_wiring_seed': 7}


def test_v6a_filename_has_wseed_token():
    """v6a files carry _wseed<n> so seed-7 never overwrites the seed-42 v5 file."""
    assert result_filename({'cell': 'cfc', 'wiring': 'ncp', 'system': 'spiral',
                            'seed': 0, 'stress': 'noise', 'ncp_wiring_seed': 7}) == \
        'cfc_ncp_spiral_seed0_stress-noise_wseed7.json'
    # seed 42 (the v5 baseline graph) gets NO wseed token -> identical to the v5 file
    assert result_filename({'cell': 'cfc', 'wiring': 'ncp', 'system': 'spiral',
                            'seed': 0, 'stress': 'noise', 'ncp_wiring_seed': 42}) == \
        'cfc_ncp_spiral_seed0_stress-noise.json'


def test_run_one_v6a_threads_wiring_seed():
    """The per-spec wiring seed reaches the model and is recorded in the config."""
    spec = {'cell': 'gru', 'wiring': 'ncp', 'system': 'spiral', 'seed': 0,
            'clip_norm': 0.0, 'stress': 'noise', 'ncp_wiring_seed': 13}
    result = run_one(spec, _TINY_CFG)
    assert result['config']['ncp_wiring_seed'] == 13
    assert result['config']['stress'] == 'noise'
    assert len(result['training']['loss_history']) == _TINY_CFG['n_iters']


def test_v6a_wiring_seed_changes_behavior():
    """Same everything, different wiring graph -> different loss (proves the seed
    reaches the NCP wiring, not just the config record)."""
    base = {'cell': 'gru', 'wiring': 'ncp', 'system': 'spiral', 'seed': 0,
            'clip_norm': 0.0, 'stress': 'noise'}
    cfg = dict(_TINY_CFG, grad_log_every=0)
    r7 = run_one({**base, 'ncp_wiring_seed': 7}, cfg)
    r99 = run_one({**base, 'ncp_wiring_seed': 99}, cfg)
    assert r7['training']['loss_history'] != r99['training']['loss_history']


# --- v6b: stress-level dose-response sweep ---

def test_v6b_matrix_counts():
    """v6b = 3 regimes x 2 new levels x 8 cells x ncp x 6 systems x 5 seeds = 1440."""
    specs = build_specs_v6b()
    assert len(specs) == 1440
    assert {s['cell'] for s in specs} == set(CELLS_V6)
    assert {s['wiring'] for s in specs} == {'ncp'}
    assert {s['stress'] for s in specs} == {'noise', 'extrapolation', 'ood_init'}
    # every spec carries exactly one level-override key matching its regime
    for s in specs:
        keys = [k for k in s if k.startswith('stress_') and k != 'stress']
        assert len(keys) == 1


def test_v6b_levels_bracket_the_v5_baseline():
    """The two new levels per regime bracket (not equal) the v5 baseline magnitude."""
    assert STRESS_LEVELS_V6B['noise'] == [('stress_noise_level', 0.05), ('stress_noise_level', 0.20)]
    assert STRESS_LEVELS_V6B['extrapolation'] == [('stress_train_fraction', 0.70), ('stress_train_fraction', 0.30)]
    assert STRESS_LEVELS_V6B['ood_init'] == [('stress_ood_scale', 0.10), ('stress_ood_scale', 0.40)]


def test_v6b_filename_has_lvl_token():
    """v6b files carry _lvl<v> so the two new levels never overwrite each other / v5."""
    assert result_filename({'cell': 'cfc', 'wiring': 'ncp', 'system': 'spiral',
                            'seed': 0, 'stress': 'noise', 'stress_noise_level': 0.2}) == \
        'cfc_ncp_spiral_seed0_stress-noise_lvl0.2.json'
    assert result_filename({'cell': 'lrc', 'wiring': 'ncp', 'system': 'duffing',
                            'seed': 3, 'stress': 'ood_init', 'stress_ood_scale': 0.4}) == \
        'lrc_ncp_duffing_seed3_stress-ood_init_lvl0.4.json'


def test_run_one_v6b_threads_level_and_records_it():
    """The per-spec stress level reaches generate_stress_dataset and the config."""
    spec = {'cell': 'cfc', 'wiring': 'ncp', 'system': 'spiral', 'seed': 0,
            'clip_norm': 0.0, 'stress': 'noise', 'stress_noise_level': 0.2}
    result = run_one(spec, _TINY_CFG)
    assert result['config']['stress'] == 'noise'
    assert result['config']['stress_level'] == 0.2
    assert len(result['training']['loss_history']) == _TINY_CFG['n_iters']


def test_v6_does_not_perturb_earlier_contracts():
    """Adding v6 must not change the frozen v1/v4/v5 SLURM array contracts."""
    assert len(build_specs(CELLS, WIRINGS, SYSTEMS, SEEDS)) == 240
    assert len(build_specs_v4()) == 880
    assert len(build_specs_v5()) == 1980
    # a clean v4 spec and a v5 stress spec still produce their original filenames
    assert result_filename({'cell': 'ltc', 'wiring': 'ncp', 'system': 'spiral',
                            'seed': 0}) == 'ltc_ncp_spiral_seed0.json'
    assert result_filename({'cell': 'cfc', 'wiring': 'ncp', 'system': 'spiral',
                            'seed': 0, 'stress': 'noise'}) == \
        'cfc_ncp_spiral_seed0_stress-noise.json'
