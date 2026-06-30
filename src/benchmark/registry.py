"""Cell / wiring / system registry for the config-driven benchmark engine.

This module is the single declarative source for:

  * which cells / wirings / systems a campaign config may reference
    (used by ``config.validate`` to reject typos at load time), and
  * the per-cell construction config (constructor kwargs, dense units, NCP
    config) that the legacy scattered globals ``CELL_KWARGS`` / ``CELL_UNITS`` /
    ``CELL_NCP`` encode.

It deliberately reproduces the legacy values byte-for-byte; the spec list is
the migration contract, but the per-cell config is locked too via
``tests/benchmark/test_equivalence.py::test_registry_matches_legacy`` so the new
structured source can never silently drift from ``experiments/run_benchmark.py``.

The actual training run path still imports ``build_model`` / ``run_one`` from the
legacy module (minimal-invasion rule) -- this registry feeds config validation
and documents the cell configs in one place for the engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# --- defaults (legacy: run_benchmark.py lines 174-180) ----------------------
DENSE_UNITS = 16
NCP_CONFIG = {'inter_neurons': 16, 'command_neurons': 8, 'motor_neurons': 2}
NCP_WIRING_SEED = 42

WIRINGS = ('dense', 'ncp')

# Base ODE systems (legacy SYSTEMS) plus the eps-ablation task suite
# (multitimescale + the stiff_linear kappa sweep). The kappa is encoded in the
# system name, so each variant is a distinct first-class system here.
_BASE_SYSTEMS = (
    'spiral',
    'duffing',
    'periodic_sinusoidal',
    'periodic_predator_prey',
    'limited_predator_prey',
    'nonlinear_predator_prey',
)
_EPS_SYSTEMS = (
    'multitimescale',
    'stiff_linear_k1',
    'stiff_linear_k10',
    'stiff_linear_k100',
    'stiff_linear_k1000',
)
KNOWN_SYSTEMS = _BASE_SYSTEMS + _EPS_SYSTEMS


@dataclass(frozen=True)
class CellConfig:
    """Per-cell construction config (mirrors the legacy per-cell globals)."""
    kwargs: dict = field(default_factory=dict)
    units: int = DENSE_UNITS
    ncp: dict = field(default_factory=lambda: dict(NCP_CONFIG))


# Per-cell overrides. Cells with no special construction use ``CellConfig()``
# (empty kwargs, units=16, default NCP). Values verbatim from the legacy module:
#   CELL_KWARGS  (run_benchmark.py:197-224)
#   CELL_UNITS   (run_benchmark.py:186)  -> only lrc_pm overrides
#   CELL_NCP     (run_benchmark.py:187)  -> only lrc_pm overrides
_ASYM = {'elastance_type': 'asymmetric'}

CELL_REGISTRY: dict[str, CellConfig] = {
    # --- v1 thesis cells + classical baselines (empty kwargs) ---------------
    'ltc': CellConfig(),
    'lrc': CellConfig(kwargs=dict(_ASYM)),
    'gru': CellConfig(),
    'lstm': CellConfig(),
    # --- v2 vanishing-gradient-fixed cells ----------------------------------
    'mm_ltc': CellConfig(),
    'mm_lrc': CellConfig(kwargs=dict(_ASYM)),
    'cfc': CellConfig(),
    # --- v3.1 / v3.2 / v4 closed-form + mixed-memory family -----------------
    'cfc_lrc': CellConfig(kwargs=dict(_ASYM)),
    'cfc_pm': CellConfig(kwargs={'backbone_units': 20}),
    'cfc_mm_lrc': CellConfig(kwargs=dict(_ASYM)),
    'cfc_mm_ltc': CellConfig(),          # CfC inner cell rejects elastance_type
    'ctrnn': CellConfig(),               # classical CT baseline, no extra kwargs
    # --- v3.3 param-matched capacity control (the only width override) ------
    'lrc_pm': CellConfig(
        kwargs=dict(_ASYM),
        units=24,
        ncp={'inter_neurons': 22, 'command_neurons': 12, 'motor_neurons': 2},
    ),
    # --- eps liquid-elastance ablation (8 plain-LRC conditions) -------------
    'lrc_interp': CellConfig(kwargs={'elastance_type': 'interp'}),
    'lrc_asym': CellConfig(kwargs={'elastance_type': 'asymmetric'}),
    'lrc_sym': CellConfig(kwargs={'elastance_type': 'symmetric'}),
    'lrc_frozen': CellConfig(
        kwargs={'elastance_type': 'asymmetric', 'freeze_elastance': True}),
    'lrc_pmctrl': CellConfig(
        kwargs={'elastance_type': 'interp', 'pm_pad': True}),
    'lrc_pmctrl_c': CellConfig(
        kwargs={'elastance_type': 'interp', 'pm_pad': True, 'pm_pad_extra': 16}),
    'lrc_asym_hybrid': CellConfig(
        kwargs={'elastance_type': 'asymmetric', 'ode_solver': 'hybrid'}),
    'lrc_interp_hybrid': CellConfig(
        kwargs={'elastance_type': 'interp', 'ode_solver': 'hybrid'}),
}

KNOWN_CELLS = tuple(CELL_REGISTRY)


# --- accessors (return copies so callers cannot mutate the registry) --------
def cell_kwargs(cell: str) -> dict:
    """Constructor kwargs for ``cell`` (legacy ``CELL_KWARGS.get(cell, {})``)."""
    return dict(CELL_REGISTRY[cell].kwargs) if cell in CELL_REGISTRY else {}


def cell_units(cell: str) -> int:
    """Dense units for ``cell`` (legacy ``CELL_UNITS.get(cell, DENSE_UNITS)``)."""
    return CELL_REGISTRY[cell].units if cell in CELL_REGISTRY else DENSE_UNITS


def cell_ncp(cell: str) -> dict:
    """NCP config for ``cell`` (legacy ``CELL_NCP.get(cell, NCP_CONFIG)``)."""
    return dict(CELL_REGISTRY[cell].ncp) if cell in CELL_REGISTRY else dict(NCP_CONFIG)


def is_known_cell(cell: str) -> bool:
    return cell in CELL_REGISTRY


def is_known_wiring(wiring: str) -> bool:
    return wiring in WIRINGS


def is_known_system(system: str) -> bool:
    return system in KNOWN_SYSTEMS
