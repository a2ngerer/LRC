# tests/benchmark/test_init_variant_keys.py
"""Registry test of the five F3 init-variant keys (issue #144).

lrc_dose050 / lrc_dose075 / lrc_dose0875 (the lrc_cal factors raised to
alpha = 0.5 / 0.75 / 0.875), lrc_fanin (only the fan-in rule of lrc_init) and
lrc_ranges (only its init ranges): plain LRC_Cell with elastance 'asymmetric'.
Per key: the kwargs, the key in the model registry, the width the Person
Activity NCP budget search lands on (equal to lrc), and a PA final line in the
suite manifest. Nothing trains and no data file is read.
"""
import pytest

import experiments.run_benchmark as legacy
from experiments import suite_manifest
from src.benchmark.registry import (_LRC_INIT_RANGES, cell_kwargs, cell_ncp,
                                    cell_units, is_known_cell)
from src.models.rnn_model import _CELL_REGISTRY
from src.neurons import LRC_Cell
from src.tasks.person_activity import build_person_activity_model
from src.tasks.person_activity.model import budget_sizes
from src.wirings import size_for_budget, width_cell_kwargs
from src.wirings.ncp import effective_param_count

ASYM = {'elastance_type': 'asymmetric'}
SEQ_LEN, NUM_CLASSES, FEATURES = 32, 7, 7


def dose(inter, command, motor):
    """The dose-level builder of the init probe (probe_e16b.py)."""
    return dict(ASYM, init_slice_scale={'inter': inter, 'command': command,
                                        'motor': motor})


# key -> (alpha or None, expected cell kwargs)
EXPECTED = {
    'lrc_dose050': (0.5, dose(2.68, 2.57, 2.57)),
    'lrc_dose075': (0.75, dose(4.40, 4.12, 4.12)),
    'lrc_dose0875': (0.875, dose(5.63, 5.21, 5.21)),
    'lrc_fanin': (None, dict(ASYM, init_fanin_scale='units')),
    'lrc_ranges': (None, dict(ASYM, initialization_ranges=_LRC_INIT_RANGES)),
}
KEYS = list(EXPECTED)


@pytest.mark.parametrize('key', KEYS)
def test_kwargs_match_the_spec(key):
    _, expected = EXPECTED[key]
    assert cell_kwargs(key) == expected
    # the mirror in the legacy runner module (test_equivalence checks all keys)
    assert legacy.CELL_KWARGS[key] == expected
    # plain lrc width: no override of units or NCP layout
    assert cell_units(key) == cell_units('lrc')
    assert cell_ncp(key) == cell_ncp('lrc')


@pytest.mark.parametrize('key', [k for k in KEYS if EXPECTED[k][0]])
def test_dose_factors_are_lrc_cal_to_the_alpha(key):
    """alpha 0 is lrc, alpha 1 is lrc_cal: the factors are the lrc_cal factors
    raised to alpha, two decimals."""
    alpha, _ = EXPECTED[key]
    cal = cell_kwargs('lrc_cal')['init_slice_scale']
    got = cell_kwargs(key)['init_slice_scale']
    assert got == {k: round(v ** alpha, 2) for k, v in cal.items()}


@pytest.mark.parametrize('key', KEYS)
def test_key_is_registered_for_the_model_builder(key):
    assert is_known_cell(key)
    assert _CELL_REGISTRY[key] is LRC_Cell


def _pa_model(cell, size):
    kwargs = width_cell_kwargs(cell_kwargs(cell))
    return build_person_activity_model(
        'ncp', cell, size=size, seed=0, num_classes=NUM_CLASSES,
        feature_size=FEATURES, seq_len=SEQ_LEN, **kwargs)


@pytest.fixture(scope='module')
def lrc_pick():
    """Width and effective count the runner's --param-budget 30000 scan picks
    for lrc on PA/ncp."""
    return size_for_budget(lambda s: _pa_model('lrc', s), 30000,
                           budget_sizes('ncp'))


@pytest.mark.parametrize('key', KEYS)
def test_pa_ncp_budget_gives_the_lrc_size(key, lrc_pick):
    """The init options change no weight shape: at budget 30000 the key gets
    the width of lrc and the same effective parameter count (the full scan is
    run for lrc once; a window around its pick is scanned for the key)."""
    size, n = lrc_pick
    window = range(size - 1, size + 2)
    assert size_for_budget(lambda s: _pa_model(key, s), 30000,
                           window) == (size, n)
    for s in window:
        assert (effective_param_count(_pa_model(key, s))
                == effective_param_count(_pa_model('lrc', s))), s


@pytest.mark.parametrize('key', KEYS)
def test_manifest_plans_a_pa_final_entry(key, tmp_path, capsys):
    """Read-only: a PA final YAML (written to tmp_path, not a campaign) with
    the key on ncp is listed as one run line; nothing is submitted."""
    yaml_path = tmp_path / 'f3_probe.yaml'
    yaml_path.write_text(
        'task: pa\nstage: final\n'
        f'cells:\n  {key}: [ncp]\n'
        'param_budget: 30000\nseeds: [10]\nepochs: 50\n'
        'ode_unfolds: 6\ndt_scale: auto\nextra_args: [--deterministic]\n'
        'fixed: {lr: 0.001, clipnorm: null}\n')
    suite_manifest.main([str(yaml_path), '--list'])
    (line,) = capsys.readouterr().out.strip().splitlines()
    assert f'{key}_ncp_seed10' in line
    assert f'--cell {key} --wiring ncp ' in line
    assert 'run_person_activity_benchmark.py' in line
    assert '--param-budget 30000' in line
