"""Campaign config model + YAML loader + load-time validation.

A campaign config describes ONE benchmark matrix declaratively. It replaces a
legacy ``build_specs_<profile>()`` function; ``expand.expand`` turns it into the
byte-identical spec list. Two expansion modes:

  * ``axes``     -- cross product of cells x wirings x systems x seeds, with
                    optional OUTER ``extra_axes`` (e.g. v5 stress, v6a wiring
                    seed) and constant keys.
  * ``explicit`` -- an ordered list of per-task ``blocks`` (e.g. eps), each with
                    its own systems / ode_unfolds and (optionally) cell subset,
                    plus ``drop`` filters for pruning.

No pydantic: plain dataclasses + a small hand-written validator. The validator
is intentionally lightweight -- it catches typos (unknown cell/wiring/system),
empty axes and a bad ``mode`` at load time, and otherwise trusts the schema.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import registry


class ConfigError(ValueError):
    """Raised for a malformed or invalid campaign config."""


@dataclass
class ExtraAxis:
    """An OUTER axis nested outside the cell x wiring x system x seed product.

    ``key`` is the spec dict key it writes (e.g. ``stress``,
    ``ncp_wiring_seed``); each value in ``values`` becomes one outer iteration.
    Axes are nested in list order -- the first ExtraAxis is the outermost loop.
    """
    key: str
    values: list


@dataclass
class Block:
    """One explicit-mode task block (e.g. an eps task at a set of unfolds)."""
    system: str
    ode_unfolds: list | None = None   # within-block outer loop; None -> no key
    cells: list | None = None         # per-block cell subset; None -> global cells


@dataclass
class DropRule:
    """Post-expansion filter. A spec is dropped iff, for EVERY field in
    ``conditions``, ``spec.get(field)`` is in that field's value list (AND over
    fields). Multiple rules are OR-ed (dropped if any rule matches)."""
    conditions: dict


@dataclass
class CampaignConfig:
    name: str
    mode: str
    outdir: str
    description: str = ''
    # axes-mode base axes
    cells: list = field(default_factory=list)
    wirings: list = field(default_factory=list)
    systems: list = field(default_factory=list)
    seeds: list = field(default_factory=list)
    clip_norm: float = 0.0
    # shared
    constants: dict = field(default_factory=dict)
    extra_axes: list = field(default_factory=list)   # list[ExtraAxis]
    blocks: list = field(default_factory=list)        # list[Block]
    filters: list = field(default_factory=list)       # list[DropRule]
    per_cell: dict = field(default_factory=dict)      # informational overrides
    raw: dict = field(default_factory=dict)           # original YAML (manifest snapshot)


# --- range helper -----------------------------------------------------------
def _as_list(value, ctx: str):
    """Resolve a list or a ``{range: [start, stop[, step]]}`` mapping to a list.

    ``range`` keeps long integer axes (e.g. eps seeds 0..29) readable in YAML
    while still producing native ints.
    """
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and set(value) == {'range'}:
        args = value['range']
        if not isinstance(args, list) or not (2 <= len(args) <= 3):
            raise ConfigError(f'{ctx}: range must be [start, stop] or '
                              f'[start, stop, step], got {args!r}')
        return list(range(*args))
    raise ConfigError(f'{ctx}: expected a list or a {{range: [...]}} mapping, '
                      f'got {type(value).__name__}')


# --- loader -----------------------------------------------------------------
def load_config(path) -> CampaignConfig:
    """Load, parse, validate and return the campaign config at ``path``."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding='utf-8'))
    except FileNotFoundError as exc:
        raise ConfigError(f'config not found: {path}') from exc
    if not isinstance(raw, dict):
        raise ConfigError(f'{path}: top level must be a mapping, '
                          f'got {type(raw).__name__}')

    cfg = _build(raw, path)
    validate(cfg)
    return cfg


def _build(raw: dict, path: Path) -> CampaignConfig:
    name = raw.get('name')
    where = f'{path.name} ({name})'

    extra_axes = []
    for i, ax in enumerate(raw.get('extra_axes') or []):
        if not isinstance(ax, dict) or 'key' not in ax or 'values' not in ax:
            raise ConfigError(f'{where}: extra_axes[{i}] needs a key and values')
        extra_axes.append(ExtraAxis(
            key=ax['key'],
            values=_as_list(ax['values'], f'{where} extra_axes[{i}].values'),
        ))

    blocks = []
    for i, blk in enumerate(raw.get('blocks') or []):
        if not isinstance(blk, dict) or 'system' not in blk:
            raise ConfigError(f'{where}: blocks[{i}] needs a system')
        blocks.append(Block(
            system=blk['system'],
            ode_unfolds=_as_list(blk.get('ode_unfolds'),
                                 f'{where} blocks[{i}].ode_unfolds'),
            cells=_as_list(blk.get('cells'), f'{where} blocks[{i}].cells'),
        ))

    filters = []
    for i, flt in enumerate(raw.get('filters') or []):
        if not isinstance(flt, dict) or 'drop' not in flt:
            raise ConfigError(f'{where}: filters[{i}] needs a `drop` mapping')
        drop = flt['drop']
        if not isinstance(drop, dict) or not drop:
            raise ConfigError(f'{where}: filters[{i}].drop must be a non-empty '
                              'mapping of field -> [values]')
        conditions = {}
        for fld, vals in drop.items():
            if not isinstance(vals, list) or not vals:
                raise ConfigError(f'{where}: filters[{i}].drop.{fld} must be a '
                                  'non-empty list')
            conditions[fld] = vals
        filters.append(DropRule(conditions=conditions))

    return CampaignConfig(
        name=name,
        mode=raw.get('mode'),
        outdir=raw.get('outdir'),
        description=raw.get('description', ''),
        cells=_as_list(raw.get('cells'), f'{where} cells') or [],
        wirings=_as_list(raw.get('wirings'), f'{where} wirings') or [],
        systems=_as_list(raw.get('systems'), f'{where} systems') or [],
        seeds=_as_list(raw.get('seeds'), f'{where} seeds') or [],
        clip_norm=raw.get('clip_norm', 0.0),
        constants=dict(raw.get('constants') or {}),
        extra_axes=extra_axes,
        blocks=blocks,
        filters=filters,
        per_cell=dict(raw.get('per_cell') or {}),
        raw=raw,
    )


# --- validation -------------------------------------------------------------
def _truncate(values, n: int = 8) -> str:
    values = list(values)
    head = ', '.join(map(str, values[:n]))
    return head + (', ...' if len(values) > n else '')


def _check_members(values, known, kind: str, where: str):
    for v in values:
        if v not in known:
            raise ConfigError(
                f'{where}: unknown {kind} {v!r}. Known {kind}s: {_truncate(known)}')


def validate(cfg: CampaignConfig) -> None:
    where = f'config {cfg.name!r}'
    if not cfg.name:
        raise ConfigError('config is missing a `name`')
    if cfg.mode not in ('axes', 'explicit'):
        raise ConfigError(
            f'{where}: mode must be one of {{axes, explicit}}, got {cfg.mode!r}')
    if not cfg.outdir:
        raise ConfigError(f'{where}: missing `outdir`')
    if not isinstance(cfg.clip_norm, (int, float)):
        raise ConfigError(f'{where}: clip_norm must be numeric, '
                          f'got {type(cfg.clip_norm).__name__}')

    if cfg.mode == 'axes':
        _validate_axes(cfg, where)
    else:
        _validate_explicit(cfg, where)

    # extra_axes apply to both modes (rare in explicit, allowed for generality)
    for ax in cfg.extra_axes:
        if not ax.key:
            raise ConfigError(f'{where}: an extra_axis is missing its key')
        if not ax.values:
            raise ConfigError(f'{where}: extra_axis {ax.key!r} has no values')


def _validate_axes(cfg: CampaignConfig, where: str) -> None:
    for axis_name, values in (('cells', cfg.cells), ('wirings', cfg.wirings),
                              ('systems', cfg.systems), ('seeds', cfg.seeds)):
        if not values:
            raise ConfigError(f'{where}: axes mode requires a non-empty '
                              f'`{axis_name}` axis')
    _check_members(cfg.cells, registry.KNOWN_CELLS, 'cell', where)
    _check_members(cfg.wirings, registry.WIRINGS, 'wiring', where)
    _check_members(cfg.systems, registry.KNOWN_SYSTEMS, 'system', where)


def _validate_explicit(cfg: CampaignConfig, where: str) -> None:
    if not cfg.blocks:
        raise ConfigError(f'{where}: explicit mode requires at least one block')
    if not cfg.wirings:
        raise ConfigError(f'{where}: explicit mode requires a non-empty '
                          '`wirings` axis')
    if not cfg.seeds:
        raise ConfigError(f'{where}: explicit mode requires a non-empty '
                          '`seeds` axis')
    _check_members(cfg.wirings, registry.WIRINGS, 'wiring', where)
    for i, blk in enumerate(cfg.blocks):
        if not registry.is_known_system(blk.system):
            raise ConfigError(
                f'{where}: blocks[{i}] unknown system {blk.system!r}. '
                f'Known systems: {_truncate(registry.KNOWN_SYSTEMS)}')
        cells = blk.cells if blk.cells is not None else cfg.cells
        if not cells:
            raise ConfigError(f'{where}: blocks[{i}] resolves to an empty cell '
                              'set (no block cells and no global cells)')
        _check_members(cells, registry.KNOWN_CELLS, 'cell', f'{where} blocks[{i}]')
        if blk.ode_unfolds is not None and not blk.ode_unfolds:
            raise ConfigError(f'{where}: blocks[{i}] has an empty ode_unfolds list')
