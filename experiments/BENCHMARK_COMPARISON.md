# Benchmark v1 vs v2 — How to Run and Compare

v1 is the thesis matrix with the cells as published: `{ltc, lrc, gru, lstm}
x {dense, ncp}` on 6 ODE systems x 5 seeds (240 runs). LTC/LRC are expected
to show convergence/gradient problems (vanishing/exploding gradients through
the ODE, Lechner & Hasani 2020; LTC stiffness, Farsang et al. 2024).

v2 adds the *fixed* variants (480 runs):

| Group | Cells | Clip | Runs | Isolates |
|---|---|---|---|---|
| Architecture fix | `mm_ltc`, `mm_lrc`, `cfc` | off | 180 | mixed memory / closed form alone |
| Both fixes | `mm_ltc`, `mm_lrc`, `cfc` | 1.0 | 180 | interaction architecture x optimizer |
| Optimizer fix | `ltc`, `lrc` | 1.0 | 120 | clipping alone on the problem cells |

Fixed-cell background: `mm_*` = ODE-LSTM mixed memory (memory path c is
LSTM-gated, gradients bypass the ODE Jacobians); `cfc` = closed-form
continuous-time cell (no solver, sigmoid time gate instead of exponential
decay). See `docs/superpowers/specs/2026-06-10-benchmark-v2-fixed-cells-design.md`.

## 1. Run v1

Local (slow, sequential):

    uv run python experiments/run_benchmark.py --all
    # -> results/runs/*.json (240 files)

Cluster (SLURM array, throttled to 8 GPUs):

    ./cluster/submit_benchmark.sh
    ./cluster/fetch_results.sh        # after completion

## 2. Run v2

Local:

    uv run python experiments/run_benchmark.py --profile v2 --all
    # -> results/runs_v2/*.json (480 files)

Cluster:

    ./cluster/submit_benchmark_v2.sh
    ./cluster/fetch_results.sh

Subsets work the same as v1, e.g. only the architecture-fix group:

    uv run python experiments/run_benchmark.py --profile v2 \
        --cells mm_ltc,mm_lrc,cfc --all

## 3. Compare

Aggregation reads both result dirs at once; runs with active clipping appear
as their own cell variant `<cell>+clip`:

    uv run python experiments/aggregate_results.py \
        --runs results/runs results/runs_v2 --out results

Targeted comparison families (smaller Bonferroni correction, sharper tests):

    # Architecture fix: base cell vs mixed-memory vs closed-form
    uv run python experiments/aggregate_results.py \
        --runs results/runs results/runs_v2 \
        --cells ltc,mm_ltc,cfc --out results/cmp_ltc_fixes

    uv run python experiments/aggregate_results.py \
        --runs results/runs results/runs_v2 \
        --cells lrc,mm_lrc --out results/cmp_lrc_fixes

    # Optimizer fix alone: clipping on the problem cells
    uv run python experiments/aggregate_results.py \
        --runs results/runs results/runs_v2 \
        --cells ltc,ltc+clip,lrc,lrc+clip --out results/cmp_clip

Figures (loss curves, phase portraits, gradient flow) across both versions:

    uv run python experiments/plot_results.py \
        --runs results/runs results/runs_v2 --out results/figures

## 4. How to read the results

| Question | Metric | Where |
|---|---|---|
| Does the fix improve final accuracy? (RQ1) | NRMSE (full-trajectory rollout), Wilcoxon p + Cohen's d | `summary.md` test blocks |
| Does the fix repair the gradient pathology? (RQ4) | per-layer gradient norms over iterations; clip engagement (`gradient_flow.clip`) | `figures/gradflow_*.png`, run JSONs |
| Does the fix stabilize training? | loss-curve variance across seeds, convergence iteration | `figures/loss_*.png` |

Hypotheses (from the solution report, to be confirmed or falsified):

1. `mm_ltc`/`mm_lrc` converge on systems where `ltc`/`lrc` oscillate or
   diverge (architecture fix works).
2. `cfc` reaches comparable NRMSE in less wall-clock time (no solver).
3. `ltc+clip`/`lrc+clip` improve less than `mm_*` (clipping bounds the
   explosion but does not fix the vanishing direction).
4. Gradient-norm spread (max/min over layers) is smaller for `mm_*` and
   `cfc` than for `ltc`/`lrc` (RQ4 mechanism evidence).

A negative result (fixes do not help on these supervised tasks) is still a
finding: it would localize the LTC/LRC weakness in the task/data regime
rather than the gradient path — document it, do not hide it.

## Caveats

- v1 result JSONs are schema 1, v2 (and any re-run v1) are schema 2; the
  aggregation reads both.
- `clip_norm = 1.0` is an experiment parameter, not a tuned optimum. If
  clip engagement (`gradient_flow.clip.pre_clip_norm` vs threshold) shows
  clipping almost never/always active, adjust and re-run the clip group.
- Wilcoxon pairing assumes identical (system, seed) coverage between the
  compared variants — keep seeds/systems identical across versions.
