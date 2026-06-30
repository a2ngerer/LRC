"""Expand a campaign config into the ordered spec list.

The output must be BYTE-IDENTICAL (order and content) to the legacy
``build_specs_<profile>()`` it replaces, because the index order is the SLURM
array contract and completed result dirs are addressed by ``result_filename``.

Three modes (see ``config`` for the model):

  axes:     OUTER loop over ``extra_axes`` (first axis outermost), INNER cross
            product ``product(cells, wirings, systems, seeds)`` -- the exact
            cell -> wiring -> system -> seed nesting of legacy ``build_specs``.
            Each spec gets ``clip_norm``, the constant keys, and one value per
            extra axis. Matches v1 (no extras), v5 (stress outer), v6a
            (ncp_wiring_seed outer + constant stress).

  concat:   ordered ``concat_blocks`` concatenated in list order; each block is
            one ``axes`` expansion (``_expand_block``) over the block's own
            cells / wirings / systems / seeds / clip_norm / constants /
            extra_axes, inheriting the campaign-level value where the block omits
            one. Matches v2 (3 clip/cell blocks over all systems), v3
            (robustness seeds 5-14 then two intervention blocks that append a
            constant ode_unfolds=24 / batch_time=64 over a cell/system subset at
            seeds 0-14), v4 (base seeds 0-4 + tail seeds 5-9 on the stiff
            systems) and v6b (6 blocks, each appending the regime's stress key +
            a regime-specific level key).

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
    elif config.mode == 'concat':
        specs = _expand_concat(config)
    elif config.mode == 'explicit':
        specs = _expand_explicit(config)
    else:                                       # pragma: no cover - validated
        raise ValueError(f'unknown mode {config.mode!r}')
    return _apply_filters(specs, config.filters)


def _expand_block(config: CampaignConfig, *, cells, wirings, systems, seeds,
                  clip_norm, constants, extra_axes) -> list[dict]:
    """One axes cross-product: ``extra_axes`` outer, then cell->wiring->system->seed.

    Shared by ``axes`` (the whole campaign) and ``concat`` (one block). The spec
    key order is fixed: cell, wiring, system, seed, clip_norm, then ``constants``
    in their mapping order, then one key per ``extra_axes`` entry -- exactly the
    legacy ``build_specs`` dict followed by the ``{**s, ...}`` overlays.
    """
    specs: list[dict] = []
    extra_keys = [ax.key for ax in extra_axes]
    extra_value_lists = [ax.values for ax in extra_axes]

    # product() over no axes yields a single empty tuple, so the v1 (no extra
    # axes) case falls out of the same loop with one outer iteration.
    for combo in product(*extra_value_lists):
        extra = dict(zip(extra_keys, combo))
        for cell, wiring, system, seed in product(cells, wirings, systems, seeds):
            spec = {'cell': cell, 'wiring': wiring, 'system': system,
                    'seed': seed, 'clip_norm': clip_norm}
            spec.update(constants)
            spec.update(extra)
            specs.append(_apply_per_cell(spec, config))
    return specs


def _expand_axes(config: CampaignConfig) -> list[dict]:
    return _expand_block(
        config, cells=config.cells, wirings=config.wirings,
        systems=config.systems, seeds=config.seeds, clip_norm=config.clip_norm,
        constants=config.constants, extra_axes=config.extra_axes)


def _expand_concat(config: CampaignConfig) -> list[dict]:
    specs: list[dict] = []
    for block in config.concat_blocks:
        # Campaign constants come first (stable position), then block constants
        # override / extend -- so the appended-key order matches legacy.
        constants = dict(config.constants)
        if block.constants:
            constants.update(block.constants)
        extra_axes = (block.extra_axes if block.extra_axes is not None
                      else config.extra_axes)
        specs.extend(_expand_block(
            config,
            cells=block.cells if block.cells is not None else config.cells,
            wirings=block.wirings if block.wirings is not None else config.wirings,
            systems=block.systems if block.systems is not None else config.systems,
            seeds=block.seeds if block.seeds is not None else config.seeds,
            clip_norm=(block.clip_norm if block.clip_norm is not None
                       else config.clip_norm),
            constants=constants,
            extra_axes=extra_axes,
        ))
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
                        # ode_unfolds BEFORE constants: legacy eps emits the dict
                        # literal {..., clip_norm, ode_unfolds, eps_jitter}, so
                        # the per-block ode_unfolds key precedes constant keys.
                        if uf is not None:
                            spec['ode_unfolds'] = uf
                        spec.update(config.constants)
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
