"""Campaign config model + YAML loader + load-time validation.

A campaign config describes ONE benchmark matrix declaratively. It replaces a
legacy ``build_specs_<profile>()`` function; ``expand.expand`` turns it into the
byte-identical spec list. Three expansion modes:

  * ``axes``     -- cross product of cells x wirings x systems x seeds, with
                    optional OUTER ``extra_axes`` (e.g. v5 stress, v6a wiring
                    seed) and constant keys.
  * ``concat``   -- an ORDERED list of ``blocks`` that concatenate; each block is
                    a full axes cross-product (cell x wiring x system x seed,
                    extra_axes outer) with its OWN cells / wirings / systems /
                    seeds / clip_norm / constants / extra_axes, inheriting the
                    campaign-level value for anything it omits. ``axes`` is the
                    one-block special case. Matches v2 (per-block clip_norm),
                    v3 (per-block seeds + per-block constant ode_unfolds /
                    batch_time over a cell/system subset), v4 (base vs tail
                    seeds) and v6b (per-block heterogeneous stress-level key).
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
class ConcatBlock:
    """One concat-mode block: a full axes cross-product with its own axes.

    Every field defaults to ``None`` meaning "inherit the campaign-level value".
    A block thus overrides only what differs from the campaign default, so the
    YAML reads as a short diff per block (e.g. v2 block 2 sets only
    ``clip_norm``; v6b blocks set only ``constants``). Nesting within a block is
    identical to ``axes`` mode: ``extra_axes`` outermost, then the
    cell -> wiring -> system -> seed product.
    """
    cells: list | None = None
    wirings: list | None = None
    systems: list | None = None
    seeds: list | None = None
    clip_norm: float | None = None
    constants: dict | None = None
    extra_axes: list | None = None    # list[ExtraAxis]; None -> inherit campaign


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
    blocks: list = field(default_factory=list)        # list[Block] (explicit mode)
    concat_blocks: list = field(default_factory=list)  # list[ConcatBlock] (concat mode)
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


def _parse_extra_axes(raw_list, where: str, ctx: str) -> list:
    """Parse an ``extra_axes`` list (campaign-level or per concat block)."""
    axes = []
    for i, ax in enumerate(raw_list or []):
        if not isinstance(ax, dict) or 'key' not in ax or 'values' not in ax:
            raise ConfigError(f'{where}: {ctx}[{i}] needs a key and values')
        axes.append(ExtraAxis(
            key=ax['key'],
            values=_as_list(ax['values'], f'{where} {ctx}[{i}].values'),
        ))
    return axes


def _build_concat_block(blk, where: str, i: int) -> ConcatBlock:
    """Parse one concat-mode block. Every axis is optional (None -> inherit)."""
    if not isinstance(blk, dict):
        raise ConfigError(f'{where}: blocks[{i}] must be a mapping')
    raw_extra = blk.get('extra_axes')
    return ConcatBlock(
        cells=_as_list(blk.get('cells'), f'{where} blocks[{i}].cells'),
        wirings=_as_list(blk.get('wirings'), f'{where} blocks[{i}].wirings'),
        systems=_as_list(blk.get('systems'), f'{where} blocks[{i}].systems'),
        seeds=_as_list(blk.get('seeds'), f'{where} blocks[{i}].seeds'),
        clip_norm=blk.get('clip_norm', None),
        constants=dict(blk['constants']) if blk.get('constants') is not None else None,
        extra_axes=(_parse_extra_axes(raw_extra, where, f'blocks[{i}].extra_axes')
                    if raw_extra is not None else None),
    )


def _build(raw: dict, path: Path) -> CampaignConfig:
    name = raw.get('name')
    mode = raw.get('mode')
    where = f'{path.name} ({name})'

    extra_axes = _parse_extra_axes(raw.get('extra_axes'), where, 'extra_axes')

    # ``blocks`` means different things per mode: explicit blocks fix ONE system
    # per block, concat blocks are full axes cross-products. Parse accordingly so
    # a concat block (no ``system`` scalar) is not rejected by the explicit guard.
    blocks: list = []
    concat_blocks: list = []
    if mode == 'concat':
        for i, blk in enumerate(raw.get('blocks') or []):
            concat_blocks.append(_build_concat_block(blk, where, i))
    else:
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
        mode=mode,
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
        concat_blocks=concat_blocks,
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
    if cfg.mode not in ('axes', 'explicit', 'concat'):
        raise ConfigError(
            f'{where}: mode must be one of {{axes, concat, explicit}}, '
            f'got {cfg.mode!r}')
    if not cfg.outdir:
        raise ConfigError(f'{where}: missing `outdir`')
    if not isinstance(cfg.clip_norm, (int, float)):
        raise ConfigError(f'{where}: clip_norm must be numeric, '
                          f'got {type(cfg.clip_norm).__name__}')

    if cfg.mode == 'axes':
        _validate_axes(cfg, where)
    elif cfg.mode == 'concat':
        _validate_concat(cfg, where)
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


def _validate_concat(cfg: CampaignConfig, where: str) -> None:
    if not cfg.concat_blocks:
        raise ConfigError(f'{where}: concat mode requires at least one block')
    for i, blk in enumerate(cfg.concat_blocks):
        bwhere = f'{where} blocks[{i}]'
        # Resolve each axis against the campaign-level default (block wins).
        cells = blk.cells if blk.cells is not None else cfg.cells
        wirings = blk.wirings if blk.wirings is not None else cfg.wirings
        systems = blk.systems if blk.systems is not None else cfg.systems
        seeds = blk.seeds if blk.seeds is not None else cfg.seeds
        for axis_name, values in (('cells', cells), ('wirings', wirings),
                                  ('systems', systems), ('seeds', seeds)):
            if not values:
                raise ConfigError(
                    f'{bwhere}: resolves to an empty `{axis_name}` axis '
                    '(no block value and no campaign-level default)')
        _check_members(cells, registry.KNOWN_CELLS, 'cell', bwhere)
        _check_members(wirings, registry.WIRINGS, 'wiring', bwhere)
        _check_members(systems, registry.KNOWN_SYSTEMS, 'system', bwhere)
        clip = blk.clip_norm if blk.clip_norm is not None else cfg.clip_norm
        if not isinstance(clip, (int, float)):
            raise ConfigError(f'{bwhere}: clip_norm must be numeric, '
                              f'got {type(clip).__name__}')
        for ax in (blk.extra_axes or []):
            if not ax.key:
                raise ConfigError(f'{bwhere}: an extra_axis is missing its key')
            if not ax.values:
                raise ConfigError(
                    f'{bwhere}: extra_axis {ax.key!r} has no values')


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
