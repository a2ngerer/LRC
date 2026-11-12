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

Init diagnostic keys (issue #82), plain LRC_Cell with elastance 'asymmetric':
``lrc_init`` (re-centred init ranges + fan-in scaling of h / sensory_h) and
``lrc_cal`` (per-layer scaling of vleak / h / sensory_h, NCP slices from the
wiring), ``lrc_cal_lv`` / ``lrc_cal_duffing`` (the same with the factors of the
LV / Duffing graph, issue #86); all have the parameter count of ``lrc``. The
factors come from ``experiments/calibrate_lrc_init.py``.

F3 init-variant keys (issue #144), also plain LRC_Cell with the parameter
count of ``lrc``: ``lrc_dose050`` / ``lrc_dose075`` / ``lrc_dose0875`` (the
``lrc_cal`` factors raised to alpha = 0.5 / 0.75 / 0.875), ``lrc_fanin`` (only
the fan-in rule of ``lrc_init``) and ``lrc_ranges`` (only its init ranges).

``cfc_lrc_eps1`` (issue #90): ``cfc_lrc`` with ``elastance_one`` (eps = 1, head
built and frozen); its params_effective leaves the frozen head out.

C7 tau-init keys (issue #87): ``lrc_tau`` and ``lrc_init_tau`` are ``lrc`` and
``lrc_init`` with the elastance bias drawn for a log-uniform time scale
(``tau_init`` / ``tau_min`` / ``tau_max``); same parameter count.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# --- defaults (legacy: run_benchmark.py lines 174-180) ----------------------
DENSE_UNITS = 16
NCP_CONFIG = {'inter_neurons': 16, 'command_neurons': 8, 'motor_neurons': 2}
NCP_WIRING_SEED = 42

# 'cncp' is the cortically-informed NCP wiring (src/wirings/cncp.py, spec
# docs/superpowers/specs/2026-07-02-cncp-design.md). It is a wiring factor
# like dense/ncp, so any registered cell can run under it: the shared run
# path (run_benchmark.build_model, used by run_campaign via run_one)
# dispatches it to make_cncp_model and raises on unknown wirings instead of
# falling back to ncp. No frozen legacy profile enumerates it, so the
# equivalence lock is untouched.
# 'ncp_stacked' is the DEPRECATED pre-2026-09-17 three-layer approximation
# that 'ncp' used to mean (see docs/ncp-wiring-fix-2026-09-17.md). It is kept
# as an explicit key so old runs can be reproduced; no frozen legacy profile
# enumerates it, so the equivalence lock is untouched.
# 'cncp2' is the four-node cNCP v2 column (src/wirings/cncp2.py, spec
# docs/superpowers/specs/2026-10-21-cncp-v2-abstraction.md Part B); the run
# path builds it with width u = the arm's size knob.
WIRINGS = ('dense', 'ncp', 'ncp_stacked', 'cncp', 'cncp2')

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
# Init ranges of the init diagnostic arm lrc_init (issue #82), against the
# LRC_Cell defaults: vleak wider, mu / sensory_mu centred near zero, w /
# sensory_w / gleak narrower.
_LRC_INIT_RANGES = {'vleak': (-1.0, 1.0), 'mu': (-0.2, 0.2),
                    'sensory_mu': (-0.2, 0.2), 'w': (0.001, 0.05),
                    'sensory_w': (0.001, 0.5), 'gleak': (0.001, 0.1)}
# Time-scale init of the C7 arms (issue #87, protocol F2 B4).
_TAU_INIT = {'tau_init': 'log_uniform', 'tau_min': 1.01, 'tau_max': 32.0}

CELL_REGISTRY: dict[str, CellConfig] = {
    # --- v1 thesis cells + classical baselines (empty kwargs) ---------------
    'ltc': CellConfig(),
    # ltc_minel: LTC_Cell with the elapsed-time floor 0.05 (issue #89, protocol
    # F2 P2b): on Person Activity a zero time gap gives a NaN state in the fused
    # LTC step. The value enters config.effective.cell_kwargs.min_elapsed.
    'ltc_minel': CellConfig(kwargs={'min_elapsed': 0.05}),
    'lrc': CellConfig(kwargs=dict(_ASYM)),
    # lrc_ar: autoregressive LRC (class default 'symmetric'), same elastance
    # as lrc for the sequence runners.
    'lrc_ar': CellConfig(kwargs=dict(_ASYM)),
    'gru': CellConfig(),
    'lstm': CellConfig(),
    # --- v2 vanishing-gradient-fixed cells ----------------------------------
    'mm_ltc': CellConfig(),
    'mm_lrc': CellConfig(kwargs=dict(_ASYM)),
    'cfc': CellConfig(),
    # --- v3.1 / v3.2 / v4 closed-form + mixed-memory family -----------------
    'cfc_lrc': CellConfig(kwargs=dict(_ASYM)),
    # cfc_lrc_eps1: cfc_lrc with the elastance held at exactly 1 (issue #90,
    # protocol F2 C-2, label [eps=1]). The head is built (same init draws as
    # cfc_lrc) and frozen, so it leaves params_effective; the width comes from
    # the cfc_lrc arm. The flag enters config.effective.cell_kwargs.
    'cfc_lrc_eps1': CellConfig(kwargs={'elastance_type': 'asymmetric',
                                       'elastance_one': True}),
    # cfc_lrc_outer: eps scales the gate output, not its time-rate (M. Farsang
    # review variant). Same cell/param count as cfc_lrc; head-to-head benchmark.
    'cfc_lrc_outer': CellConfig(kwargs={'elastance_type': 'asymmetric',
                                        'elastance_gate': 'outer'}),
    'cfc_pm': CellConfig(kwargs={'backbone_units': 20}),
    'cfc_mm_lrc': CellConfig(kwargs=dict(_ASYM)),
    'cfc_mm_ltc': CellConfig(),          # CfC inner cell rejects elastance_type
    'ctrnn': CellConfig(),               # classical CT baseline, no extra kwargs
    # stateless control of the rollout tasks, 'dense' wiring only (#60)
    'mlp': CellConfig(),
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
    # --- init diagnostic arms (issue #82), plain LRC_Cell -------------------
    # lrc_init: re-centred init ranges + fan-in scaling of h / sensory_h. The
    # sensory reference ('units' vs 'input_dim') is set by the protocol
    # addendum; both are implemented.
    'lrc_init': CellConfig(kwargs={'elastance_type': 'asymmetric',
                                   'initialization_ranges': _LRC_INIT_RANGES,
                                   'init_fanin_scale': 'units'}),
    # lrc_cal: per-layer scaling of vleak / h / sensory_h (NCP slices from
    # the wiring; the dense cell uses the 'inter' factor). Factors from
    #   uv run python experiments/calibrate_lrc_init.py --task pa --seed 0
    'lrc_cal': CellConfig(kwargs={'elastance_type': 'asymmetric',
                                  'init_slice_scale': {'inter': 7.2,
                                                       'command': 6.6,
                                                       'motor': 6.6}}),
    # lrc_cal_lv: lrc_cal with the factors of the LV graph (square roots of
    # its layer-wise unit-variance calibration, inter 28.0 / command 18.4 /
    # motor 14.4); the PA factors overshoot there. LV stage of the init
    # diagnostic only. Factors from
    #   uv run python experiments/calibrate_lrc_init.py --task lv --seed 0
    'lrc_cal_lv': CellConfig(kwargs={'elastance_type': 'asymmetric',
                                     'init_slice_scale': {'inter': 5.3,
                                                          'command': 4.3,
                                                          'motor': 3.8}}),
    # lrc_cal_duffing: lrc_cal with the factors of the Duffing graph (square
    # roots of its layer-wise unit-variance calibration, inter 39.75 /
    # command 17.40 / motor 15.54). Factors from
    #   uv run python experiments/calibrate_lrc_init.py --task lv --system duffing --seed 0
    # (ncp, size 43 at budget 4000, seed 0, ode_unfolds 6, dt_scale auto).
    # Duffing stage of the init diagnostic only.
    'lrc_cal_duffing': CellConfig(kwargs={'elastance_type': 'asymmetric',
                                          'init_slice_scale': {'inter': 6.3,
                                                               'command': 4.2,
                                                               'motor': 3.9}}),
    # --- F3 init-variant arms (issue #144), plain LRC_Cell ------------------
    # Dose arms: the lrc_cal factors (inter 7.2 / command 6.6 / motor 6.6)
    # raised to alpha and rounded to two decimals; alpha 0 is lrc, alpha 1 is
    # lrc_cal. Descriptive Person Activity arms of protocol F3.
    # lrc_dose050: alpha = 0.5
    'lrc_dose050': CellConfig(kwargs={'elastance_type': 'asymmetric',
                                      'init_slice_scale': {'inter': 2.68,
                                                           'command': 2.57,
                                                           'motor': 2.57}}),
    # lrc_dose075: alpha = 0.75
    'lrc_dose075': CellConfig(kwargs={'elastance_type': 'asymmetric',
                                      'init_slice_scale': {'inter': 4.40,
                                                           'command': 4.12,
                                                           'motor': 4.12}}),
    # lrc_dose0875: alpha = 0.875
    'lrc_dose0875': CellConfig(kwargs={'elastance_type': 'asymmetric',
                                       'init_slice_scale': {'inter': 5.63,
                                                            'command': 5.21,
                                                            'motor': 5.21}}),
    # Component arms of lrc_init: lrc_fanin = only the fan-in rule (reference
    # init ranges), lrc_ranges = only the re-centred ranges (no fan-in rule).
    'lrc_fanin': CellConfig(kwargs={'elastance_type': 'asymmetric',
                                    'init_fanin_scale': 'units'}),
    'lrc_ranges': CellConfig(kwargs={'elastance_type': 'asymmetric',
                                     'initialization_ranges': _LRC_INIT_RANGES}),
    # --- C7 tau-init arms (issue #87, protocol F2 B4) -----------------------
    # The two base arms lrc / lrc_init plus a log-uniform time-scale init of
    # the elastance bias, -log(T - 1) with T on [1.01, 32].
    'lrc_tau': CellConfig(kwargs={'elastance_type': 'asymmetric', **_TAU_INIT}),
    'lrc_init_tau': CellConfig(kwargs={'elastance_type': 'asymmetric',
                                       'initialization_ranges': _LRC_INIT_RANGES,
                                       'init_fanin_scale': 'units',
                                       **_TAU_INIT}),
}

KNOWN_CELLS = tuple(CELL_REGISTRY)


# --- accessors (return copies so callers cannot mutate the registry) --------
def cell_kwargs(cell: str, elastance_type: str | None = None) -> dict:
    """Constructor kwargs for ``cell`` (legacy ``CELL_KWARGS.get(cell, {})``).

    ``elastance_type`` (the runners' --elastance-type) overrides the registry
    value for cells that carry one; other cells ignore it.
    """
    kwargs = dict(CELL_REGISTRY[cell].kwargs) if cell in CELL_REGISTRY else {}
    if elastance_type is not None and 'elastance_type' in kwargs:
        kwargs['elastance_type'] = elastance_type
    return kwargs


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
