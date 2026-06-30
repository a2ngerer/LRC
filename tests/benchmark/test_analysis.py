# tests/benchmark/test_analysis.py
"""Unit tests for the consolidated variant-suffix parsing (src/benchmark/variants).

These exercise the ONE suffix implementation that replaces the three divergent
copies (aggregate_results.load_runs, the drifted plot_results.load_runs, and
compare_v1_v2.mean_nrmse_by_variant). They run on SYNTHETIC result_filename
strings and synthetic run dicts only -- no real result JSONs are read -- so they
stay green on a fresh checkout with an empty results/ tree.

The drift that motivated the consolidation: the old plot_results.load_runs
emitted only +clip/+unfolds/+bt and silently merged the v5 stress, v6a wiring
and v6b dose-level variants. The cases below pin every token, including those
three, and the cell/wiring/system underscore-collision edge cases.
"""
import pytest

from src.benchmark.variants import (
    build_suffix, cell_base, cell_label, filename_suffix, run_suffix,
)


# A synthetic filename is exactly what run_benchmark.result_filename emits:
#   <cell>_<wiring>_<system>_seed<seed>[tokens...].json
# The cases below deliberately use cells/systems whose own names contain
# underscores (cfc_mm_ltc, periodic_predator_prey, stiff_linear_k1, ood_init)
# to prove the parser keys off the distinctive token prefixes, not on splitting.
FILENAME_CASES = [
    # plain v1 -- no variant tokens at all
    ('ltc_dense_spiral_seed0.json', ''),
    ('cfc_mm_ltc_ncp_periodic_predator_prey_seed3.json', ''),
    # v2 gradient clip
    ('ltc_ncp_duffing_seed4_clip1.0.json', '+clip'),
    # v3 solver-fidelity (non-default ode_unfolds) and training-horizon (batch_time)
    ('ltc_dense_duffing_seed0_unfolds24.json', '+unfolds24'),
    ('mm_ltc_ncp_periodic_predator_prey_seed7_bt64.json', '+bt64'),
    # v5 stress regimes (the ones the drifted plot_results.load_runs dropped)
    ('cfc_ncp_spiral_seed1_stress-noise.json', '+noise'),
    ('gru_dense_duffing_seed2_stress-extrapolation.json', '+extrapolation'),
    # ood_init: the regime name itself contains an underscore
    ('lrc_ncp_spiral_seed0_stress-ood_init.json', '+ood_init'),
    # v6a non-default wiring graph (seed 42 default carries no token)
    ('cfc_ncp_spiral_seed0_stress-noise_wseed7.json', '+noise+w7'),
    # v6b dose-response level, stacked on the regime
    ('cfc_ncp_spiral_seed0_stress-ood_init_lvl0.10.json', '+ood_init+lvl0.10'),
    # eps _uf token maps to ode_unfolds; uf=2 != default 6 -> +unfolds2
    ('lrc_asym_dense_stiff_linear_k1_seed5_uf2.json', '+unfolds2'),
    # full stack, in canonical token order
    ('cfc_ncp_duffing_seed0_stress-noise_wseed13_lvl0.20_clip1.0_bt64.json',
     '+noise+w13+lvl0.20+clip+bt64'),
]


@pytest.mark.parametrize('filename,expected', FILENAME_CASES)
def test_filename_suffix(filename, expected):
    assert filename_suffix(filename) == expected


def test_filename_suffix_handles_missing_extension():
    # The .json suffix is optional for the parser.
    assert filename_suffix('ltc_ncp_duffing_seed4_clip1.0') == '+clip'


def test_eps_default_unfolds_uf1_still_tagged():
    # eps headline uf=1 differs from the LTC default (6), so it is a variant.
    assert filename_suffix('lrc_interp_dense_multitimescale_seed0_uf1.json') \
        == '+unfolds1'


# --- run-dict path: the loader that feeds the plot functions ----------------
def _run(cell, wiring='ncp', system='spiral', seed=0, config=None, run_extra=None):
    """A minimal synthetic run JSON dict (only the fields the suffix reads)."""
    meta = {'cell': cell, 'wiring': wiring, 'system': system, 'seed': seed}
    meta.update(run_extra or {})
    return {'run': meta, 'config': config or {}}


def test_run_suffix_clip_from_config():
    r = _run('ltc', config={'clip_norm': 1.0})
    assert run_suffix(r) == '+clip'
    assert cell_label(r) == 'ltc+clip'


def test_run_suffix_stress_from_run_block():
    r = _run('cfc', run_extra={'stress': 'noise'})
    assert run_suffix(r) == '+noise'


def test_run_suffix_v6a_wiring_graph():
    r = _run('cfc', config={'stress': 'noise', 'ncp_wiring_seed': 7})
    assert run_suffix(r) == '+noise+w7'
    # the default graph 42 carries NO suffix token
    r42 = _run('cfc', config={'stress': 'noise', 'ncp_wiring_seed': 42})
    assert run_suffix(r42) == '+noise'


def test_run_suffix_v6b_dose_level_keys():
    # the level lives under one of three regime-specific keys in the run block
    r = _run('cfc', run_extra={'stress': 'ood_init', 'stress_ood_scale': 0.10})
    assert run_suffix(r) == '+ood_init+lvl0.1'


def test_run_suffix_defaults_carry_no_token():
    # ode_unfolds=6 and batch_time=16 are the defaults -> no suffix.
    r = _run('ltc', config={'ode_unfolds': 6, 'batch_time': 16, 'clip_norm': 0.0})
    assert run_suffix(r) == ''


def test_full_stack_run_matches_filename():
    # The two entry points must agree on the canonical suffix.
    r = _run('cfc', wiring='ncp', system='duffing',
             config={'stress': 'noise', 'ncp_wiring_seed': 13,
                     'clip_norm': 1.0, 'batch_time': 64},
             run_extra={'stress_noise_level': 0.20})
    assert run_suffix(r) == '+noise+w13+lvl0.2+clip+bt64'


# --- cell_base: strip ALL suffixes (the analyze_v6 contract) -----------------
@pytest.mark.parametrize('label,base', [
    ('cfc', 'cfc'),
    ('cfc+noise', 'cfc'),
    ('cfc+noise+w7', 'cfc'),
    ('cfc_mm_ltc+noise+lvl0.2', 'cfc_mm_ltc'),
    ('ltc+clip', 'ltc'),
])
def test_cell_base_strips_every_suffix(label, base):
    assert cell_base(label) == base


# --- build_suffix: the single core builder both paths delegate to -----------
def test_build_suffix_token_order_is_canonical():
    # stress, wiring graph, dose level, clip, unfolds, horizon -- fixed order.
    s = build_suffix(stress='noise', ncp_wiring_seed=7, stress_level='0.05',
                     clip_norm=1.0, ode_unfolds=24, batch_time=64)
    assert s == '+noise+w7+lvl0.05+clip+unfolds24+bt64'


def test_build_suffix_empty_when_all_default():
    assert build_suffix() == ''
    assert build_suffix(clip_norm=0.0, ode_unfolds=6, batch_time=16,
                        ncp_wiring_seed=42) == ''
