# Config-Driven Benchmark Pipeline — Implementation Plan

> Branch: `benchmarks` (long-lived, general benchmark home; not on the `phase/step` scheme).
> Worktree: `master_thesis_v2/code-benchmarks/`. Forked from `phase3/step22-eps-ablation`
> so all 11 profiles (incl. eps) are present as an equivalence reference.

## Goal

Move the benchmark matrix out of hard-coded Python into **declarative campaign configs**.
One YAML per run defines matrix + hyperparameters + cluster resources + analysis. The
pipeline expands the config into the run list, submits the SLURM array, and aggregates —
so a new benchmark means *editing a config*, not editing five code locations. The pipeline
must be runnable both agent-driven (the default workflow) and by hand.

## Guiding constraint — the spec order is a frozen contract

The order of the spec list **is** the SLURM array contract. `tests/experiments/test_run_benchmark.py`
asserts exact `specs[0]`/`specs[480]` and per-profile counts; already-completed result dirs
(240/480/720/.../1440 JSONs) are addressed by `result_filename(spec)`. The new pipeline must
reproduce the **byte-identical** spec order and filenames for every existing profile, or the
completed cluster results become inconsistent with re-runs.

## Migration safety — golden equivalence test

The central safety net: for every profile, `expand(configs/campaigns/<profile>.yaml)` must
produce exactly the same spec list (order **and** `result_filename`) as the legacy
`build_specs_<profile>()`. While that test is green, everything already run stays reproducible.
The legacy `experiments/run_benchmark.py` stays in place as the equivalence reference and is
kept (deprecated) per user decision — nothing legacy is deleted.

## Current state (from infrastructure survey)

The matrix lives as ~30 constants + 11 `build_specs_v*()` functions in a 1120-line
`run_benchmark.py`. Pain points to resolve:

1. Matrix = growing code, not config (a new experiment edits `build_specs_*` + `CELLS_*` +
   `--profile` choices + outdir map + tests).
2. Profile->outdir map duplicated (runner + every `analyze_v*.py`).
3. `load_runs` reimplemented & divergent in 3+ places (variant-suffix convention drifts;
   `plot_results.py` misses `+stress`/`+w`/`+lvl`).
4. sbatch boilerplate (TMPDIR/CUDA/TF env + parallel-launch loop) duplicated across 4 files.
5. 10 submit wrappers ~90% identical (only `--profile` + 3 env lines differ).
6. Version-specific analysis scripts proliferate (`analyze_v5/v6`, `plot_v5/v6`, `compare_*`, ...).
7. Dead in-file SBATCH directives overridden by CLI (cpus 24 vs 12).
8. `run_benchmark.py` monolith; per-spec overrides as loose dict keys, no schema.
9. Doc drift: `cluster/README.md` stops at v2; v3-v6/eps undocumented (argparse help only).
10. Cell hyperparameters as global dicts with comment-encoded constraints.

### Profile reference table

| Profile | Cells | Wirings | Systems | Seeds | Extra axis | Runs | Outdir |
|---|---|---|---|---|---|---|---|
| v1 | ltc,lrc,gru,lstm | dense,ncp | 6 ODE | 5 | - | 240 | `runs/` |
| v2 | mm_ltc,mm_lrc,cfc (+ltc,lrc) | dense,ncp | 6 | - | clip on/off | 480 | `runs_v2/` |
| v3 | CELLS_V3 / _STIFF | dense,ncp | 6 / stiff | - | ode_unfolds=24, batch_time=64 | 720 | `runs_v3/` |
| v3.1 | cfc,cfc_lrc,cfc_pm,lrc | dense,ncp | 6 | 5 | - | 240 | `runs_v3_1/` |
| v3.2 | lrc,cfc_lrc,mm_lrc,cfc_mm_lrc | dense,ncp | 6 | 5 | - | 240 | `runs_v3_2/` |
| v3.3 | lrc_pm | dense,ncp | 6 | 5 | - | 60 | `runs_v3_3/` |
| v4 | 11 (LTC/LRC 2x2 + gru,lstm,ctrnn) | dense,ncp | 6 | 5 | - | 880 | `runs_v4/` |
| v5 | = v4 set | dense,ncp | 6 | 5 | stress noise/extrap/ood (x3) | 1980 | `runs_v5/` |
| v6a | CELLS_V6 (8) | ncp | 6 | 5 | wiring_seed (x4 graphs), noise | 960 | `runs_v6a/` |
| v6b | CELLS_V6 (8) | dense,ncp | 6 | 5 | stress (3 regime x 2 level) | 1440 | `runs_v6b/` |
| eps | CELLS_EPS (8 cond A-G) | dense,ncp | multitimescale + anchors | - | ode_unfolds, per-task pruning | var | `runs_eps/` |
| eps_pilot | CELLS_EPS | dense,ncp | as eps | 8 | no pruning | 640 | `runs_eps_pilot/` |

Exact `build_specs_*` semantics are extracted in the Recon phase; this table is orientation.

## Target structure (in this worktree)

```
src/benchmark/
  config.py     # YAML data model + loader + lightweight load-time checks
  expand.py     # config -> deterministic spec list (cross product + filters + conditions)
  registry.py   # cell/wiring/system registry (replaces CELLS_*/CELL_KWARGS dicts)
  manifest.py   # run manifest: git SHA + SLURM job id + env + config snapshot
configs/campaigns/
  v1.yaml ... v6b.yaml, eps.yaml, eps_pilot.yaml
  _schema.md    # schema documentation
experiments/
  run_campaign.py      # config-driven entry (supersedes run_benchmark.py --profile)
  analyze_campaign.py  # config-driven analysis (supersedes analyze_v*.py)
  run_benchmark.py     # KEPT - equivalence reference (deprecated header)
cluster/
  _common.sh           # de-duplicated boilerplate (TMPDIR/CUDA/TF/parallel-launch)
  campaign_array.sbatch# one generic array sbatch (sources _common.sh)
  submit_campaign.sh   # one generic submit (--config X.yaml); supersedes 10 wrappers
tests/benchmark/
  test_equivalence.py  # GOLDEN: expand(config) == build_specs_<profile>() for all profiles
  test_config.py       # loading / validation / failure cases
```

## Config schema (two modes; covers the special cases)

```yaml
name: v6a
mode: axes                 # or: explicit  (list of spec dicts, for eps-style pruning)
base:
  cells:   [ltc, lrc, gru, lstm]
  wirings: [ncp]           # ncp-only = a plain axis restriction
  systems: [spiral, ...]
  seeds:   [0, 1, 2, 3, 4]
extra_axes:                # extra dimensions extend the cross product
  wiring_seed: [1, 2, 3, 4]
per_cell:                  # cell-specific kwargs/units (replaces CELL_KWARGS)
  lrc: { elastance_type: asymmetric }
filters:                   # include/exclude rules for special cases (eps per-task pruning)
  - drop: { cell: lrc_sym, system: spiral }
training: { iters: 2000, lr: 1.0e-3, grad_log_every: 25 }
cluster:  { partition: GPU-a40, mem: 64G, runs_per_gpu: 6, walltime: "10:00:00", requeue: true }
outdir:   results/runs_v6a
analysis: { metric: nrmse, tests: [wilcoxon, cohen_d], plots: [loss, phase, gradflow] }
```

Cross-product order is fixed (cells -> wirings -> systems -> seeds -> extra_axes, nested in
that sequence) and must match the legacy iteration order per profile. Where a profile cannot
be expressed cleanly via axes (likely eps), the config uses `mode: explicit` with a literal
spec list — still config-driven, just not generated.

## Build workflow (dynamic agent workflow, with verify gates)

1. **Recon (parallel, read-only):** agents extract the exact spec definitions of all 11
   profiles from the legacy code AND the existing `docs/superpowers/specs/*-design.md` as
   structured data.
2. **Core (sequential, 1 agent):** `config/expand/registry/manifest` + golden-test harness.
   Gate: pytest green.
3. **Profile configs (parallel, separate files):** one agent per profile writes
   `configs/campaigns/<p>.yaml` and verifies equivalence inline. Gate: golden test green for all 11.
4. **Periphery (parallel, separate domains):** cluster consolidation ‖ `analyze_campaign` +
   run-manifest ‖ docs (README + auto work-log entry).
5. **Verify (parallel):** full pytest ‖ local smoke (`--list`/`--count` diff old vs new) ‖
   adversarial review ("does the migration endanger existing results?").

## Verification & scope

- **All verification is local** (`uv run pytest`, `--list` diff). **No cluster submit** in this
  task; the pipeline is built and proven, then run only on explicit GO with a real campaign.
- **Out of scope (user decision):** Control/PPO (only a clean docking point is prepared, not
  implemented), no dedicated RUNBOOK artifact, no Pydantic schema. Instead: concise README +
  lightweight load-time checks (unknown cell / empty axis -> error before submit). MuJoCo track
  untouched.
- **Docs (user decision):** run manifest per run + auto entry in `work-documentation.md` /
  `timeline.md`.

## Git model

`benchmarks` is the long-lived home for the pipeline and all campaign configs. It is **not**
auto-merged into `main`; it pulls code updates from `main` via merge when needed. Legacy
wrappers and `analyze_v*.py` are kept (deprecated), not deleted.
