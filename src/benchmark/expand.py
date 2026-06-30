"""Expand a campaign config into the ordered spec list.

The output must be BYTE-IDENTICAL (order and content) to the legacy
``build_specs_<profile>()`` it replaces, because the index order is the SLURM
array contract and completed result dirs are addressed by ``result_filename``.

Two modes (see ``config`` for the model):

  axes:     OUTER loop over ``extra_axes`` (first axis outermost), INNER cross
            product ``product(cells, wirings, systems, seeds)`` -- the exact
            cell -> wiring -> system -> seed nesting of legacy ``build_specs``.
            Each spec gets ``clip_norm``, the constant keys, and one value per
            extra axis. Matches v1 (no extras), v5 (stress outer), v6a
            (ncp_wiring_seed outer + constant stress).

  explicit: ordered ``blocks``; within each block the nesting is
            ode_unfolds -> cell -> wiring -> seed (system fixed per block).
            Matches eps / eps_pilot.

``filters`` (drop rules) run last and only remove specs, never reorder -- which
is why pruning (e.g. eps sym/hybrid off the spiral anchor) stays byte-identical
to the legacy per-task cell selection that preserves canonical cell order.
"""
from __future__ import annotations

from itertools import product

from .config import CampaignConfig


def expand(config: CampaignConfig) -> list[dict]:
    """Return the ordered list of spec dicts for ``config``."""
    if config.mode == 'axes':
        specs = _expand_axes(config)
    elif config.mode == 'explicit':
        specs = _expand_explicit(config)
    else:                                       # pragma: no cover - validated
        raise ValueError(f'unknown mode {config.mode!r}')
    return _apply_filters(specs, config.filters)


def _expand_axes(config: CampaignConfig) -> list[dict]:
    specs: list[dict] = []
    extra_keys = [ax.key for ax in config.extra_axes]
    extra_value_lists = [ax.values for ax in config.extra_axes]

    # product() over no axes yields a single empty tuple, so the v1 (no extra
    # axes) case falls out of the same loop with one outer iteration.
    for combo in product(*extra_value_lists):
        extra = dict(zip(extra_keys, combo))
        for cell, wiring, system, seed in product(
                config.cells, config.wirings, config.systems, config.seeds):
            spec = {'cell': cell, 'wiring': wiring, 'system': system,
                    'seed': seed, 'clip_norm': config.clip_norm}
            spec.update(config.constants)
            spec.update(extra)
            specs.append(_apply_per_cell(spec, config))
    return specs


def _expand_explicit(config: CampaignConfig) -> list[dict]:
    specs: list[dict] = []
    for block in config.blocks:
        cells = block.cells if block.cells is not None else config.cells
        # ode_unfolds is the within-block outer loop; None -> a single pass with
        # no ode_unfolds key (keeps explicit mode usable beyond eps).
        unfolds = block.ode_unfolds if block.ode_unfolds is not None else [None]
        for uf in unfolds:
            for cell in cells:
                for wiring in config.wirings:
                    for seed in config.seeds:
                        spec = {'cell': cell, 'wiring': wiring,
                                'system': block.system, 'seed': seed,
                                'clip_norm': config.clip_norm}
                        spec.update(config.constants)
                        if uf is not None:
                            spec['ode_unfolds'] = uf
                        specs.append(_apply_per_cell(spec, config))
    return specs


def _apply_per_cell(spec: dict, config: CampaignConfig) -> dict:
    """Merge any per-cell spec overrides for this spec's cell.

    ``per_cell`` maps a cell name to a dict of extra spec keys. It is an escape
    hatch for cells that must carry a non-uniform key directly in the spec
    (none of the v1/v5/v6a/eps profiles use it; legacy per-cell *construction*
    config lives in the registry and is applied at model-build time, not here).
    """
    overrides = config.per_cell.get(spec['cell'])
    if overrides:
        spec.update(overrides)
    return spec


def _apply_filters(specs: list[dict], filters) -> list[dict]:
    if not filters:
        return specs
    return [s for s in specs if not any(_dropped(s, rule) for rule in filters)]


def _dropped(spec: dict, rule) -> bool:
    """True iff ``spec`` matches every condition of ``rule`` (AND over fields)."""
    return all(spec.get(field) in values
               for field, values in rule.conditions.items())
