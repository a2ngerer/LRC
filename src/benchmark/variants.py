"""Single source of truth for benchmark run variant-suffix parsing.

A benchmark run's cell label gets a *variant suffix* so every experimental
condition is treated as its own variant in aggregation, statistics and plots:

  ``+clip``        active gradient clip                 (v2)
  ``+unfolds<n>``  non-default ODE solver substeps      (v3 solver-fidelity arm)
  ``+bt<n>``       non-default training horizon         (v3 training-horizon arm)
  ``+<regime>``    v5 generalization stressor           (noise/extrapolation/ood_init)
  ``+w<seed>``     v6a non-default NCP wiring graph
  ``+lvl<val>``    v6b stress-level dose-response point

``experiments/aggregate_results.load_runs`` carries the original (correct) copy
of this logic; ``experiments/plot_results.load_runs`` historically carried a
DRIFTED copy that only handled ``+clip``/``+unfolds``/``+bt`` and silently merged
the ``+stress``/``+w``/``+lvl`` variants into one label. This module is the
consolidated implementation so the drift is not repeated in new code: both the
run-dict path (used when loading JSONs for plotting) and the filename path (used
for offline inspection / tests) feed ONE core builder, ``build_suffix``.

Entry points:
  * ``run_suffix(run)``           -- suffix from a loaded run JSON dict.
  * ``filename_suffix(filename)`` -- suffix from a ``result_filename`` token string.
  * ``cell_label(run)``           -- ``run['run']['cell'] + run_suffix(run)``.
  * ``cell_base(label)``          -- strip ALL suffixes (``label.split('+')[0]``).
"""
from __future__ import annotations

import re

# Defaults that suppress a suffix token (a value equal to the default is the
# "plain" condition and carries no suffix). Mirror run_benchmark.py / the legacy
# aggregate_results.load_runs thresholds exactly.
NCP_DEFAULT_SEED = 42
DEFAULT_UNFOLDS = 6
DEFAULT_BATCH_TIME = 16

# Token prefixes emitted by run_benchmark.result_filename, in their fixed order.
# Used as the lookahead set that bounds the (underscore-bearing) stress regime.
_NEXT_TOKEN = r'(?=_wseed|_lvl|_clip|_uf|_unfolds|_bt|$)'


def build_suffix(*, stress=None, ncp_wiring_seed=None, stress_level=None,
                 clip_norm=0.0, ode_unfolds=None, batch_time=None) -> str:
    """Build the canonical variant suffix from condition fields.

    The token order (stress, wiring graph, dose level, clip, unfolds, horizon)
    is the contract: it must match ``aggregate_results.load_runs`` so plot
    labels and table labels are the same grouping key. ``stress_level`` is
    emitted verbatim (str or number) -- callers pass the same form the source
    used so ``+lvl0.05`` is reproduced identically.
    """
    suffix = ''
    if stress:
        suffix += f'+{stress}'
    if ncp_wiring_seed and int(ncp_wiring_seed) != NCP_DEFAULT_SEED:
        suffix += f'+w{int(ncp_wiring_seed)}'
    if stress_level is not None:
        suffix += f'+lvl{stress_level}'
    if clip_norm and float(clip_norm):
        suffix += '+clip'
    if ode_unfolds and int(ode_unfolds) != DEFAULT_UNFOLDS:
        suffix += f'+unfolds{int(ode_unfolds)}'
    if batch_time and int(batch_time) != DEFAULT_BATCH_TIME:
        suffix += f'+bt{int(batch_time)}'
    return suffix


def run_suffix(run: dict) -> str:
    """Variant suffix for a loaded run JSON dict (``config`` + ``run`` blocks).

    Fields are read from ``config`` first, then the ``run`` block (the stress
    level lives under one of three regime-specific keys), matching the legacy
    ``aggregate_results.load_runs`` extraction.
    """
    cfg = run.get('config', {}) or {}
    meta = run.get('run', {}) or {}
    stress = cfg.get('stress') or meta.get('stress')
    wseed = cfg.get('ncp_wiring_seed') or meta.get('ncp_wiring_seed')
    stress_level = next(
        (v for v in (cfg.get('stress_level'),
                     meta.get('stress_noise_level'),
                     meta.get('stress_train_fraction'),
                     meta.get('stress_ood_scale')) if v is not None),
        None)
    return build_suffix(
        stress=stress,
        ncp_wiring_seed=wseed,
        stress_level=stress_level,
        clip_norm=cfg.get('clip_norm', 0.0),
        ode_unfolds=cfg.get('ode_unfolds'),
        batch_time=cfg.get('batch_time', DEFAULT_BATCH_TIME),
    )


def filename_suffix(filename: str) -> str:
    """Variant suffix parsed from a ``result_filename`` token string.

    Cell / wiring / system names themselves contain underscores
    (``cfc_mm_ltc``, ``periodic_predator_prey``, ``stiff_linear_k1``), so the
    tokens are extracted by their distinctive ``_<key>`` prefixes rather than by
    naive ``_`` splitting. The eps ``_uf<n>`` token and the legacy
    ``_unfolds<n>`` token both map to ``ode_unfolds``.
    """
    name = filename[:-5] if filename.endswith('.json') else filename

    def grab(pattern):
        m = re.search(pattern, name)
        return m.group(1) if m else None

    stress = grab(r'_stress-(.+?)' + _NEXT_TOKEN)
    wseed = grab(r'_wseed(\d+)')
    lvl = grab(r'_lvl([0-9.]+)')
    clip = grab(r'_clip([0-9.]+)')
    unfolds = grab(r'_uf(\d+)') or grab(r'_unfolds(\d+)')
    bt = grab(r'_bt(\d+)')

    return build_suffix(
        stress=stress,
        ncp_wiring_seed=int(wseed) if wseed is not None else None,
        # Keep the level as the parsed string so f-string formatting reproduces
        # the on-disk token (e.g. '0.05') without float-repr drift.
        stress_level=lvl,
        clip_norm=float(clip) if clip is not None else 0.0,
        ode_unfolds=int(unfolds) if unfolds is not None else None,
        batch_time=int(bt) if bt is not None else None,
    )


def cell_label(run: dict) -> str:
    """The suffixed cell label used as the grouping key for a run dict."""
    return run['run']['cell'] + run_suffix(run)


def cell_base(label: str) -> str:
    """Strip ALL variant suffixes from a cell label (``'cfc+noise+w7'`` -> ``'cfc'``)."""
    return label.split('+')[0]
