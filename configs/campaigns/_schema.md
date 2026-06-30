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
| `mode`        | yes      | `axes`/`concat`/`explicit` | Expansion strategy (see below). |
| `outdir`      | yes      | str             | Result directory (e.g. `results/runs_v5`). |
| `description` | no       | str             | Free-text note (ignored by the engine). |
| `cells`       | axes / explicit\* / concat\* | list[str] | Cells (must be registered in the registry). |
| `wirings`     | yes\*\*  | list[str]       | Subset of `{dense, ncp}`. |
| `systems`     | axes / concat\* | list[str] | ODE systems (registered). |
| `seeds`       | yes\*\*  | list[int] or `{range: [...]}` | Training seeds. |
| `clip_norm`   | no       | float (default `0.0`) | Gradient clip; written into every spec (concat blocks may override it). |
| `constants`   | no       | mapping         | Constant key/value pairs stamped onto every spec (e.g. `stress: noise`). |
| `extra_axes`  | no       | list            | OUTER axes nested outside the base product (see below). |
| `blocks`      | concat / explicit | list   | Ordered blocks; the per-block shape depends on `mode` (see below). |
| `filters`     | no       | list            | `drop` rules applied last (pruning). |
| `per_cell`    | no       | mapping         | Per-cell extra spec keys (escape hatch; unused by v1/v5/v6a/eps). |
| `cluster`     | no       | mapping         | SLURM resource hints, read by `cluster/submit_campaign.sh` (NOT the engine). |
| `analysis`    | no       | mapping         | Aggregation/plot hints, read by `experiments/analyze_campaign.py` (NOT the engine). |

\* `cells`/`systems` are required in `axes` mode; in `explicit`/`concat` mode
they are the campaign-level defaults that a block inherits when it omits the key.

\*\* `wirings`/`seeds` are required at the campaign level in `axes`/`explicit`
mode; in `concat` mode a block may instead supply its own (each block must
*resolve* to a non-empty `cells`/`wirings`/`systems`/`seeds`).

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

## `cluster` and `analysis` — tooling sections (engine-ignored)

The engine (`load_config`/`expand`) reads only the matrix keys above; `cluster`
and `analysis` are passed through untouched in `cfg.raw` and consumed by the
shell/analysis tooling. Both are optional — omit them and the cluster defaults
(`cluster/config.sh`) and analysis defaults apply.

`cluster` — SLURM resource hints, read by `cluster/submit_campaign.sh`. Each key
maps onto the corresponding `CLUSTER_*` default; an environment variable still
overrides the YAML (`CLUSTER_MEM=64G ./cluster/submit_campaign.sh ...`).

| key             | maps to                  | meaning |
|-----------------|--------------------------|---------|
| `partition`     | `CLUSTER_PARTITION`      | dataLAB partition (e.g. `GPU-a40`). |
| `gpu`           | `CLUSTER_GPU`            | `--gres` spec (e.g. `a40:1`). |
| `cpus`          | `CLUSTER_CPUS`           | `--cpus-per-task`. |
| `mem`           | `CLUSTER_MEM`            | `--mem` (e.g. `64G`). |
| `walltime`      | `CLUSTER_TIME`           | `--time` (e.g. `10:00:00`). |
| `runs_per_gpu`  | `CLUSTER_RUNS_PER_GPU`   | runs packed onto one shared GPU (array-task block size). |
| `array_throttle`| `CLUSTER_ARRAY_THROTTLE` | max concurrent array tasks (dataLAB hard limit 8 GPUs). |

```yaml
cluster: { partition: GPU-a40, mem: 64G, runs_per_gpu: 6, walltime: "10:00:00" }
```

`analysis` — aggregation/plot hints, read by `experiments/analyze_campaign.py`.
Free-form; typical keys:

| key       | meaning |
|-----------|---------|
| `metric`  | headline metric (e.g. `nrmse`). |
| `tests`   | statistical tests to run (e.g. `[wilcoxon, cohen_d]`). |
| `plots`   | plot kinds to render (e.g. `[loss, phase, gradflow]`). |

```yaml
analysis: { metric: nrmse, tests: [wilcoxon, cohen_d], plots: [loss, gradflow] }
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

## `mode: concat` — ordered axes blocks

For a matrix that is the **concatenation** of several axes cross-products with
**different per-block** cells / clip_norm / seeds / systems / constants (v2, v3,
v4, v6b). The campaign is an ordered list of `blocks`; each block is exactly one
`axes` expansion, and the blocks concatenate in list order. `axes` mode is the
single-block special case.

A block sets only what differs from the campaign-level defaults; every omitted
field inherits the campaign value, so the YAML reads as a short diff per block.

```yaml
mode: concat
cells: [mm_ltc, mm_lrc, cfc]     # campaign defaults, inherited unless overridden
wirings: [dense, ncp]
systems: [spiral, duffing, ...]
seeds: [0, 1, 2, 3, 4]
clip_norm: 0.0
blocks:
  - {clip_norm: 0.0}                      # inherits cells/wirings/systems/seeds
  - {clip_norm: 1.0}
  - {cells: [ltc, lrc], clip_norm: 1.0}   # overrides cells, inherits the rest
```

Per block (every field optional — omitted → inherit the campaign-level value):

| field        | meaning |
|--------------|---------|
| `cells`      | per-block cell subset |
| `wirings`    | per-block wirings |
| `systems`    | per-block systems |
| `seeds`      | per-block seeds (`range` shorthand allowed) |
| `clip_norm`  | per-block gradient clip (the v2 lever) |
| `constants`  | extra constant keys for this block, merged **after** campaign-level `constants` in mapping order (v3 `{ode_unfolds: 24}` / `{batch_time: 64}`; v6b's heterogeneous `stress_*` level key) |
| `extra_axes` | per-block OUTER axes, same shape as the top-level `extra_axes` |

Within a block the nesting is identical to `axes`: `extra_axes` outermost, then
`cell → wiring → system → seed`. The emitted spec **key order** is `cell, wiring,
system, seed, clip_norm`, then campaign `constants`, then block `constants`, then
extra-axis keys — matching the legacy `build_specs(...)` dict followed by the
per-block `{**s, ...}` overlays. (The golden test asserts this key order, not
just value equality.)

| config | shape |
|--------|-------|
| `v2`   | 3 blocks over all systems; `clip_norm` 0/1/1, block 3 restricts cells to `[ltc, lrc]`. |
| `v3`   | robustness (4 cells, all systems, seeds 5-14) + solver (2 cells, 2 systems, seeds 0-14, `constants: {ode_unfolds: 24}`) + horizon (same matrix, `constants: {batch_time: 64}`). |
| `v4`   | base (all systems, seeds 0-4) + tail (2 stiff systems, seeds 5-9). |
| `v6b`  | 6 blocks (regime-major), each only `constants: {stress: <regime>, <regime-specific level key>: <value>}` — a different spec key per regime. |

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
`drop`. In `concat` mode it additionally rejects a campaign with no blocks, a
block that resolves to an empty `cells`/`wirings`/`systems`/`seeds` axis (no
block value and no campaign default), and a non-numeric block `clip_norm`. It is
intentionally lightweight — it catches typos and structural mistakes, not
semantic ones.

---

## Worked examples (one per mode)

Minimal, self-contained configs. Each adds the optional `cluster`/`analysis`
sections to show the full shape; drop them to fall back to the defaults.

### `axes` — a plain cross product

```yaml
name: demo_axes
mode: axes
outdir: results/runs_demo_axes
cells:   [ltc, lrc, gru, lstm]
wirings: [dense, ncp]
systems: [spiral, duffing]
seeds:   [0, 1, 2]
clip_norm: 0.0
# 4 x 2 x 2 x 3 = 48 specs, nested cell -> wiring -> system -> seed.
cluster:  { mem: 32G, runs_per_gpu: 12 }
analysis: { metric: nrmse, tests: [wilcoxon], plots: [loss] }
```

Add an OUTER axis to make it regime-major (v5 shape):

```yaml
extra_axes:
  - key: stress
    values: [noise, extrapolation, ood_init]   # -> 3 x 48 = 144 specs
```

### `concat` — ordered blocks with per-block diffs

```yaml
name: demo_concat
mode: concat
outdir: results/runs_demo_concat
cells:   [ltc, lrc]      # campaign defaults, inherited unless a block overrides
wirings: [dense, ncp]
systems: [spiral, duffing]
seeds:   [0, 1, 2, 3, 4]
clip_norm: 0.0
blocks:
  - {clip_norm: 0.0}                     # block 1: inherit all, no clip
  - {clip_norm: 1.0}                     # block 2: same matrix, clip on
  - {cells: [lrc], seeds: {range: [5, 10]}, constants: {ode_unfolds: 24}}
# Each block is one axes cross-product; the blocks concatenate in list order.
cluster:  { mem: 48G, runs_per_gpu: 8 }
analysis: { metric: nrmse, tests: [wilcoxon, cohen_d] }
```

### `explicit` — ordered per-task blocks (per-task pruning)

```yaml
name: demo_explicit
mode: explicit
outdir: results/runs_demo_explicit
cells:   [lrc_interp, lrc_asym, lrc_sym, lrc_frozen]   # default cells per block
wirings: [dense, ncp]
seeds:   {range: [0, 30]}
constants: {eps_jitter: true}
blocks:
  - {system: multitimescale, ode_unfolds: [1, 2, 4]}   # within-block: uf -> cell -> wiring -> seed
  - {system: spiral,         ode_unfolds: [1]}
filters:
  - drop: {system: [spiral], cell: [lrc_sym]}          # drop the sym arm off the spiral anchor
cluster:  { partition: GPU-a40, mem: 64G, walltime: "10:00:00" }
analysis: { metric: nrmse, plots: [loss, gradflow] }
```
