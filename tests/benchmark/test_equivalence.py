# tests/benchmark/test_equivalence.py
"""GOLDEN equivalence test for the config-driven benchmark engine.

Parametrized over every campaign config in ``configs/campaigns/``. For each:
load -> expand -> compare against the legacy ``build_specs_<name>()`` it
replaces, element-by-element. The spec list (order + content) is the SLURM array
contract, so this is the equivalence proof that the migration is a no-op for
already-completed result dirs.

Three structural checks beyond plain ``==``:
  * a TYPE-strict comparison (``_typed``), because Python dict ``==`` treats
    ``0 == 0.0`` / ``1 == True`` as equal -- a seed emitted as ``0.0`` instead of
    ``0`` would pass ``==`` yet break ``result_filename`` (``seed0`` vs
    ``seed0.0``) and downstream model construction.
  * a KEY-ORDER comparison (``list(spec.keys())``), because dict ``==`` ignores
    insertion order -- the serialized spec dict (e.g. the manifest snapshot) must
    match legacy byte-for-byte, so ``..., clip_norm, ode_unfolds, eps_jitter``
    may not silently reorder to ``..., clip_norm, eps_jitter, ode_unfolds``.
  * ``result_filename`` identity per element, the on-disk addressing contract.
"""
from pathlib import Path

import pytest

import experiments.run_benchmark as legacy
from src.benchmark import registry
from src.benchmark.config import load_config
from src.benchmark.expand import expand

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / 'configs' / 'campaigns'
CONFIG_FILES = sorted(CONFIG_DIR.glob('*.yaml'))


# Map a campaign `name` to the legacy spec list it must reproduce. v1 has no
# build_specs_v1 -- it is the bare generic build_specs() call in main().
LEGACY_SPECS = {
    'v1': lambda: legacy.build_specs(legacy.CELLS, legacy.WIRINGS,
                                     legacy.SYSTEMS, legacy.SEEDS),
    'v2': legacy.build_specs_v2,
    'v3': legacy.build_specs_v3,
    'v3.1': legacy.build_specs_v3_1,
    'v3.2': legacy.build_specs_v3_2,
    'v3.3': legacy.build_specs_v3_3,
    'v4': legacy.build_specs_v4,
    'v5': legacy.build_specs_v5,
    'v6a': legacy.build_specs_v6a,
    'v6b': legacy.build_specs_v6b,
    'eps': legacy.build_specs_eps,
    'eps_pilot': legacy.build_specs_eps_pilot,
}


def _typed(spec: dict) -> dict:
    """Spec with each value tagged by its type name (strict 0 vs 0.0, int vs bool)."""
    return {k: (type(v).__name__, v) for k, v in spec.items()}


def test_configs_present():
    assert CONFIG_FILES, f'no campaign configs found in {CONFIG_DIR}'


def test_required_target_profiles_present():
    names = {load_config(p).name for p in CONFIG_FILES}
    assert {'v1', 'v5', 'v6a', 'eps'} <= names, f'missing target profiles: {names}'


@pytest.mark.parametrize('config_path', CONFIG_FILES, ids=[p.stem for p in CONFIG_FILES])
def test_expand_matches_legacy(config_path):
    cfg = load_config(config_path)
    assert cfg.name in LEGACY_SPECS, f'no legacy mapping for profile {cfg.name!r}'

    expanded = expand(cfg)
    golden = LEGACY_SPECS[cfg.name]()

    # 1. count
    assert len(expanded) == len(golden), (
        f'{cfg.name}: count {len(expanded)} != legacy {len(golden)}')

    # 2. element-by-element: value-equal, key-order-equal, type-strict, same file
    for i, (got, want) in enumerate(zip(expanded, golden)):
        assert got == want, f'{cfg.name}: spec[{i}] differs\n got={got}\nwant={want}'
        # dict == ignores insertion order; the serialized spec must match legacy
        # key order exactly, so compare list(keys()) explicitly.
        assert list(got.keys()) == list(want.keys()), (
            f'{cfg.name}: spec[{i}] key ORDER differs\n'
            f' got={list(got.keys())}\nwant={list(want.keys())}')
        assert _typed(got) == _typed(want), (
            f'{cfg.name}: spec[{i}] type/value differs\n'
            f' got={_typed(got)}\nwant={_typed(want)}')
        assert legacy.result_filename(got) == legacy.result_filename(want), (
            f'{cfg.name}: result_filename[{i}] differs '
            f'({legacy.result_filename(got)} != {legacy.result_filename(want)})')

    # 3. whole-list key-order proof (closes the dict-== blind spot in aggregate)
    assert [list(s.keys()) for s in expanded] == [list(s.keys()) for s in golden], (
        f'{cfg.name}: spec key order differs from legacy')


def test_registry_matches_legacy():
    """The structured registry must reproduce the legacy scattered globals."""
    assert registry.DENSE_UNITS == legacy.DENSE_UNITS
    assert registry.NCP_CONFIG == legacy.NCP_CONFIG
    assert registry.NCP_WIRING_SEED == legacy.NCP_WIRING_SEED

    for cell in registry.KNOWN_CELLS:
        assert registry.cell_kwargs(cell) == legacy.CELL_KWARGS.get(cell, {}), cell
        assert registry.cell_units(cell) == legacy.CELL_UNITS.get(
            cell, legacy.DENSE_UNITS), cell
        assert registry.cell_ncp(cell) == legacy.CELL_NCP.get(
            cell, legacy.NCP_CONFIG), cell

    # every cell the legacy module references is registered (no validation gap)
    legacy_cells = (
        set(legacy.CELL_KWARGS) | set(legacy.CELL_UNITS) | set(legacy.CELL_NCP)
        | set(legacy.CELLS) | set(legacy.CELLS_V2) | set(legacy.CELLS_V4)
        | set(legacy.CELLS_V6) | set(legacy.CELLS_EPS)
    )
    missing = legacy_cells - set(registry.KNOWN_CELLS)
    assert not missing, f'registry missing cells referenced by legacy: {missing}'
