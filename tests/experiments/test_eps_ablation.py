# tests/experiments/test_eps_ablation.py
"""Unit tests for the eps-ablation benchmark (liquid-elastance over-parameterization).

All param counts are asserted on BUILT models (D=2 ODE state), never analytically
(spec section 4 / 6). Covers: 304/320/450 elastance counts, E==B and E_C==C param
match, freeze => elastance non-trainable, all 8 condition labels build with their
intended kwargs, _uf<n>/_k<kappa> result-filename routing, and the hybrid solver
running with forget_gate=True.
"""
import numpy as np
import pytest
import tensorflow as tf

from experiments.run_benchmark import (
    CELLS_EPS, build_specs_eps, build_specs_eps_pilot, SEEDS_EPS_PILOT,
    result_filename,
    _build_eps_model, _elastance_param_count, _pad_param_count, _iter_lrc_cells,
)
from src.neurons import LRC_Cell


# --- param counts on BUILT models -------------------------------------------

@pytest.mark.parametrize('wiring,cond,expected', [
    ('dense', 'lrc_interp', 0),
    ('dense', 'lrc_asym', 304),
    ('dense', 'lrc_sym', 320),
    ('ncp', 'lrc_interp', 0),
    ('ncp', 'lrc_asym', 450),
    ('ncp', 'lrc_sym', 476),
])
def test_elastance_param_counts_built(wiring, cond, expected):
    m = _build_eps_model(cond, wiring)
    assert _elastance_param_count(m) == expected


def test_ncp_elastance_is_per_cell_450():
    """NCP has three cells with different elastance shapes summing to 450."""
    m = _build_eps_model('lrc_asym', 'ncp')
    per_cell = sorted(
        int(np.prod(v.shape)) for v in m.trainable_variables
        if 'elastance_mapping' in v.name and 'kernel' in v.name
    )
    # inter (18x16=288), command (16x8=128), motor (4x2=8)
    assert per_cell == [8, 128, 288]
    assert _elastance_param_count(m) == 450


@pytest.mark.parametrize('wiring', ['dense', 'ncp'])
def test_E_matches_B(wiring):
    """E (pm_pad) trainable params equal B (asym) elastance params per wiring."""
    b = _elastance_param_count(_build_eps_model('lrc_asym', wiring))
    e = _pad_param_count(_build_eps_model('lrc_pmctrl', wiring))
    assert e == b


@pytest.mark.parametrize('wiring', ['dense', 'ncp'])
def test_E_C_matches_C(wiring):
    """E_C (pm_pad + extra) trainable params equal C (sym) params per wiring."""
    c = _elastance_param_count(_build_eps_model('lrc_sym', wiring))
    ec = _pad_param_count(_build_eps_model('lrc_pmctrl_c', wiring))
    assert ec == c


@pytest.mark.parametrize('wiring', ['dense', 'ncp'])
def test_C_carries_distr_shift(wiring):
    m = _build_eps_model('lrc_sym', wiring)
    ds = sum(int(np.prod(v.shape)) for v in m.trainable_variables
             if 'distr_shift' in v.name)
    assert ds == (16 if wiring == 'dense' else 26)   # 16 / 16+8+2


@pytest.mark.parametrize('wiring', ['dense', 'ncp'])
def test_A_elastance_dense_unbuilt(wiring):
    """interp's elastance Dense is never called -> 0 params (Keras lazy build)."""
    m = _build_eps_model('lrc_interp', wiring)
    assert _elastance_param_count(m) == 0
    for c in _iter_lrc_cells(m):
        assert c.elastance_mapping.built is False


# --- freeze => elastance non-trainable --------------------------------------

@pytest.mark.parametrize('wiring', ['dense', 'ncp'])
def test_freeze_makes_elastance_non_trainable(wiring):
    m = _build_eps_model('lrc_frozen', wiring)
    cells = _iter_lrc_cells(m)
    assert cells, 'no LRC cell found'
    for c in cells:
        assert c.elastance_mapping.trainable is False
        # frozen weights still EXIST (built), they are just not optimized
        assert c.elastance_mapping.built is True
    # no elastance params show up among the trainable variables
    assert _elastance_param_count(m) == 0


def test_freeze_weights_exist_in_non_trainable():
    m = _build_eps_model('lrc_frozen', 'dense')
    nt = [v for v in m.non_trainable_variables if 'elastance_mapping' in v.name]
    assert nt, 'frozen elastance weights must exist as non-trainable variables'


# --- all 8 labels build with intended kwargs --------------------------------

_EXPECTED_KWARGS = {
    'lrc_interp': dict(elastance_type='interp', ode_solver='explicit',
                       freeze=False, pm_pad=False, extra=0),
    'lrc_asym': dict(elastance_type='asymmetric', ode_solver='explicit',
                     freeze=False, pm_pad=False, extra=0),
    'lrc_sym': dict(elastance_type='symmetric', ode_solver='explicit',
                    freeze=False, pm_pad=False, extra=0),
    'lrc_frozen': dict(elastance_type='asymmetric', ode_solver='explicit',
                       freeze=True, pm_pad=False, extra=0),
    'lrc_pmctrl': dict(elastance_type='interp', ode_solver='explicit',
                       freeze=False, pm_pad=True, extra=0),
    'lrc_pmctrl_c': dict(elastance_type='interp', ode_solver='explicit',
                         freeze=False, pm_pad=True, extra=16),
    'lrc_asym_hybrid': dict(elastance_type='asymmetric', ode_solver='hybrid',
                            freeze=False, pm_pad=False, extra=0),
    'lrc_interp_hybrid': dict(elastance_type='interp', ode_solver='hybrid',
                              freeze=False, pm_pad=False, extra=0),
}


def test_all_8_eps_labels_registered():
    assert CELLS_EPS == list(_EXPECTED_KWARGS.keys())


@pytest.mark.parametrize('label', list(_EXPECTED_KWARGS.keys()))
def test_label_builds_with_intended_kwargs(label):
    exp = _EXPECTED_KWARGS[label]
    m = _build_eps_model(label, 'dense')
    cells = _iter_lrc_cells(m)
    assert cells
    for c in cells:
        assert c._elastance_type == exp['elastance_type']
        assert c._ode_solver_type == exp['ode_solver']
        assert c._forget_gate is True               # required everywhere
        assert c._freeze_elastance is exp['freeze']
        assert c._pm_pad is exp['pm_pad']
        assert c._pm_pad_extra == exp['extra']
        if exp['pm_pad']:
            assert c.pm_pad_mapping is not None
        else:
            assert c.pm_pad_mapping is None


# --- result-filename routing (_uf<n> and _k<kappa>) -------------------------

def test_result_filename_uf_token():
    spec = {'cell': 'lrc_asym', 'wiring': 'dense', 'system': 'multitimescale',
            'seed': 0, 'ode_unfolds': 2, 'eps_jitter': True}
    assert result_filename(spec) == 'lrc_asym_dense_multitimescale_seed0_uf2.json'


def test_result_filename_uf1_is_tagged():
    """uf=1 must still get a token so the per-level files never collide."""
    spec = {'cell': 'lrc_asym', 'wiring': 'dense', 'system': 'multitimescale',
            'seed': 3, 'ode_unfolds': 1, 'eps_jitter': True}
    assert result_filename(spec).endswith('_uf1.json')


def test_result_filename_kappa_in_system_name():
    spec = {'cell': 'lrc_interp', 'wiring': 'ncp', 'system': 'stiff_linear_k100',
            'seed': 5, 'ode_unfolds': 1, 'eps_jitter': True}
    assert result_filename(spec) == \
        'lrc_interp_ncp_stiff_linear_k100_seed5_uf1.json'


# --- build_specs_eps structure + sym/hybrid pruning -------------------------

def test_eps_specs_prune_sym_hybrid_off_spiral():
    specs = build_specs_eps(seeds=[0], wirings=['dense'])
    spiral_cells = {s['cell'] for s in specs if s['system'] == 'spiral'}
    # only the always-run interp/asym arms (A, B, D, E) on the flat spiral anchor
    assert spiral_cells == {'lrc_interp', 'lrc_asym', 'lrc_frozen', 'lrc_pmctrl'}
    # sym + hybrid DO appear on the informative multiscale task
    multi_cells = {s['cell'] for s in specs if s['system'] == 'multitimescale'}
    assert {'lrc_sym', 'lrc_pmctrl_c', 'lrc_asym_hybrid',
            'lrc_interp_hybrid'} <= multi_cells


def test_eps_specs_headline_has_three_unfold_levels():
    specs = build_specs_eps(seeds=[0], wirings=['dense'])
    ufs = {s['ode_unfolds'] for s in specs if s['system'] == 'multitimescale'}
    assert ufs == {1, 2, 4}
    # anchors / exploratory run only at uf=1
    for sys_name in ('spiral', 'stiff_linear_k1', 'stiff_linear_k1000'):
        anchor_ufs = {s['ode_unfolds'] for s in specs if s['system'] == sys_name}
        assert anchor_ufs == {1}


def test_eps_specs_are_jittered():
    specs = build_specs_eps(seeds=[0], wirings=['dense'])
    assert all(s.get('eps_jitter') for s in specs)


# --- §3.0 pilot subset (build_specs_eps_pilot) ------------------------------

def test_pilot_default_seeds_are_zero_to_seven():
    assert SEEDS_EPS_PILOT == list(range(8))


def test_pilot_count_is_640():
    """8 conditions x 2 wirings x 8 seeds x 5 task-levels (multi@{1,2,4}+2 anchors)."""
    assert len(build_specs_eps_pilot()) == 640


def test_pilot_has_ALL_8_conditions_on_anchors():
    """Unlike the full eps matrix, the pilot does NOT prune sym/hybrid off the
    flat anchors -- every SD must be estimable."""
    specs = build_specs_eps_pilot(seeds=[0], wirings=['dense'])
    for anchor in ('spiral', 'stiff_linear_k1'):
        cells = {s['cell'] for s in specs if s['system'] == anchor}
        assert cells == set(CELLS_EPS), f'{anchor} missing conditions: {cells}'


def test_pilot_tasks_and_levels():
    specs = build_specs_eps_pilot(seeds=[0], wirings=['dense'])
    # multitimescale at three unfold levels; anchors only at uf=1; NO exploratory.
    multi_ufs = {s['ode_unfolds'] for s in specs if s['system'] == 'multitimescale'}
    assert multi_ufs == {1, 2, 4}
    assert {s['system'] for s in specs} == {
        'multitimescale', 'spiral', 'stiff_linear_k1'}
    for anchor in ('spiral', 'stiff_linear_k1'):
        assert {s['ode_unfolds'] for s in specs if s['system'] == anchor} == {1}


def test_pilot_specs_are_jittered():
    assert all(s.get('eps_jitter') for s in build_specs_eps_pilot(seeds=[0]))


# --- hybrid runs with forget_gate=True --------------------------------------

@pytest.mark.parametrize('label', ['lrc_asym_hybrid', 'lrc_interp_hybrid'])
def test_hybrid_forward_pass(label):
    m = _build_eps_model(label, 'dense')
    out = m(tf.constant(0.0), tf.zeros([1, 1, 2]))
    assert out.shape == (1, 1, 2)
    for c in _iter_lrc_cells(m):
        assert c._ode_solver_type == 'hybrid'
        assert c._forget_gate is True


def test_hybrid_requires_forget_gate():
    """The hybrid solver references f, defined only under forget_gate=True."""
    cell = LRC_Cell(units=4, elastance_type='asymmetric', ode_solver='hybrid',
                    forget_gate=True)
    out, _ = cell(tf.zeros([2, 3]), [tf.zeros([2, 4])])
    assert out.shape == (2, 4)
