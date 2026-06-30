# Campaign config schema

A campaign config is a YAML file describing **one** benchmark matrix. The engine
(`src/benchmark/`) loads it (`load_config`), validates it, and expands it
(`expand`) into the ordered list of spec dicts that drives the SLURM array. Each
config replaces one legacy `build_specs_<profile>()`; the spec list it produces
is **byte-identical** (order + content) to the function it replaces — this is the
migration contract, locked by `tests/benchmark/test_equivalence.py`.

The training run path (`run_one` / `build_model` / `train` / `result_filename`)
is still imported from `experiments/run_benchmark.py`. Only spec *generation*
lives here. Per-cell *construction* config (constructor kwargs, dense units, NCP
config) is the registry's (`src/benchmark/registry.py`) and is applied at
model-build time — it does **not** appear in the spec dict.

Run a campaign with `experiments/run_campaign.py --config <name>`.

---

## Top-level keys

| key           | required | type            | meaning |
|---------------|----------|-----------------|---------|
| `name`        | yes      | str             | Profile id. The golden test maps it to the legacy `build_specs_<name>()`. |
| `mode`        | yes      | `axes`/`explicit` | Expansion strategy (see below). |
| `outdir`      | yes      | str             | Result directory (e.g. `results/runs_v5`). |
| `description` | no       | str             | Free-text note (ignored by the engine). |
| `cells`       | axes / explicit\* | list[str] | Cells (must be registered in the registry). |
| `wirings`     | yes      | list[str]       | Subset of `{dense, ncp}`. |
| `systems`     | axes     | list[str]       | ODE systems (registered). |
| `seeds`       | yes      | list[int] or `{range: [...]}` | Training seeds. |
| `clip_norm`   | no       | float (default `0.0`) | Gradient clip; written into every spec. |
| `constants`   | no       | mapping         | Constant key/value pairs stamped onto every spec (e.g. `stress: noise`). |
| `extra_axes`  | no       | list            | OUTER axes nested outside the base product (see below). |
| `blocks`      | explicit | list            | Ordered per-task blocks (see below). |
| `filters`     | no       | list            | `drop` rules applied last (pruning). |
| `per_cell`    | no       | mapping         | Per-cell extra spec keys (escape hatch; unused by v1/v5/v6a/eps). |

\* `cells` is required in `axes` mode; in `explicit` mode it is the default cell
list for blocks that do not override it.

### Spec dict shape

Every emitted spec always has `cell`, `wiring`, `system`, `seed`, `clip_norm`.
Optional keys are added by `constants`, `extra_axes`, block `ode_unfolds`, or
`per_cell`: e.g. `stress`, `ncp_wiring_seed`, `ode_unfolds`, `eps_jitter`,
`batch_time`. Value **types** matter (`seed: 0` is int, `clip_norm: 0.0` is
float) — the golden test compares types, not just values.

### `range` shorthand

`seeds`, each `extra_axes[*].values`, and block `ode_unfolds` accept either a
plain list or `{range: [start, stop]}` / `{range: [start, stop, step]}`, expanded
with Python `range` semantics (stop exclusive). Used for the eps 30-seed axis:

```yaml
seeds: {range: [0, 30]}   # -> [0, 1, ..., 29]
```

---

## `mode: axes` — cross product

Iterates `itertools.product(cells, wirings, systems, seeds)` with the exact
legacy nesting **cell → wiring → system → seed** (seed varies fastest, cell
slowest). `extra_axes` wrap this product as OUTER loops.

```yaml
extra_axes:
  - key: stress              # spec dict key written
    values: [noise, extrapolation, ood_init]
```

- Each `extra_axes` entry adds one key to every spec, taking one value per outer
  iteration.
- Multiple entries nest in list order: the **first** entry is the **outermost**
  loop. The base `cell→wiring→system→seed` product is always innermost.
- `constants` (e.g. `stress: noise` in v6a) add a fixed key to every spec; they
  do not create iterations.

Order for one extra axis `A` with values `[a0, a1]`:
`A=a0 × (full product)`, then `A=a1 × (full product)` — i.e. regime-major /
graph-major, matching `build_specs_v5` / `build_specs_v6a`.

| config | shape |
|--------|-------|
| `v1`   | pure product, no extras. |
| `v5`   | one outer axis `stress` (3 regimes), regime-major. |
| `v6a`  | constant `stress: noise` + one outer axis `ncp_wiring_seed` (4 graphs), `wirings: [ncp]` only. |

---

## `mode: explicit` — ordered task blocks

For matrices that are **not** a single cross product (per-task cell pruning,
per-task `ode_unfolds` lists). The matrix is a concatenation of `blocks` in list
order. Within each block the nesting is **ode_unfolds → cell → wiring → seed**
(the block's `system` is fixed).

```yaml
cells: [lrc_interp, lrc_asym, ...]      # default cells for blocks
wirings: [dense, ncp]
seeds: {range: [0, 30]}
constants: {eps_jitter: true}
blocks:
  - {system: multitimescale, ode_unfolds: [1, 2, 4]}   # 3 unfold levels
  - {system: spiral,         ode_unfolds: [1]}
```

Per block:

| field         | required | meaning |
|---------------|----------|---------|
| `system`      | yes      | The (fixed) system for this block. |
| `ode_unfolds` | no       | List looped as the within-block outer axis; each value is written to `ode_unfolds`. Omit for no `ode_unfolds` key. |
| `cells`       | no       | Per-block cell subset; defaults to the top-level `cells`. |

---

## `filters` — drop rules (pruning)

Applied **after** expansion, in both modes. A rule lists field → allowed-values.
A spec is **dropped** iff it matches **every** field in the rule (AND within a
rule); multiple rules are OR-ed (dropped if any rule matches). Filters only
*remove* specs, never reorder — so pruning stays order-identical to legacy.

```yaml
filters:
  - drop:
      system: [spiral]
      cell: [lrc_sym, lrc_pmctrl_c, lrc_asym_hybrid, lrc_interp_hybrid]
```

This is how `eps` prunes the sym/hybrid arms off the flat spiral anchor: legacy
`_eps_cells_for_task` builds spiral with only the 4 always-cells in canonical
order; building all 8 in canonical order and dropping the 4 yields the identical
surviving order.

---

## Validation (load time)

`load_config` raises `ConfigError` for: unknown `mode`; missing `name`/`outdir`;
an empty required axis; an unknown cell / wiring / system (typo guard, checked
against the registry); a malformed `range`; an empty block cell set; a malformed
`drop`. It is intentionally lightweight — it catches typos and structural
mistakes, not semantic ones.
