# experiments/run_benchmark.py
"""Cluster-ready benchmark runner for the thesis matrix.

Thesis matrix: {ltc, lrc, gru, lstm} x {dense, ncp} x 6 ODE systems x N seeds.

One invocation = one training run = one JSON result file. This maps directly
onto a SLURM array job (see cluster/benchmark_array.sbatch):

    # list all run specs with their indices
    uv run python experiments/run_benchmark.py --list

    # number of specs (for --array=0-$((N-1)))
    uv run python experiments/run_benchmark.py --count

    # run spec i (SLURM array task)
    uv run python experiments/run_benchmark.py --index $SLURM_ARRAY_TASK_ID

    # explicit single run
    uv run python experiments/run_benchmark.py --cell ltc --wiring ncp \
        --system spiral --seed 0

    # full matrix sequentially (local, slow)
    uv run python experiments/run_benchmark.py --all
"""
import argparse
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from itertools import product

import numpy as np
import tensorflow as tf

from src.models import make_dense_model, make_ncp_model
from src.tasks.neural_ode.datasets import (
    generate_dataset, generate_stress_dataset, STRESS_REGIMES,
)
from src.tasks.neural_ode.ode_model import SequentialODEFunc
from src.tasks.neural_ode.solver import euler_odeint
from src.tasks.neural_ode.trainer import train
from src.evaluation import GradientFlowTracker, mse, nrmse
from src.utils import set_global_seed

CELLS = ['ltc', 'lrc', 'gru', 'lstm']
WIRINGS = ['dense', 'ncp']
SYSTEMS = [
    'spiral',
    'duffing',
    'periodic_sinusoidal',
    'periodic_predator_prey',
    'limited_predator_prey',
    'nonlinear_predator_prey',
]
SEEDS = [0, 1, 2, 3, 4]

# Benchmark v2: vanishing-gradient-fixed cells (see
# docs/superpowers/specs/2026-06-10-benchmark-v2-fixed-cells-design.md).
CELLS_V2 = ['mm_ltc', 'mm_lrc', 'cfc']
# Clip threshold for the v2 clip-axis runs. 1.0 is the common recurrent-RL
# default; the exact value is an experiment parameter, not a tuned constant.
V2_CLIP_NORM = 1.0

# Benchmark v3: forward-rollout stability of stiff ODE cells (see
# docs/superpowers/specs/2026-06-14-benchmark-v3-rollout-stability-design.md).
# v1/v2 gradient-flow analysis showed the ltc/mm_ltc "divergence" is a
# forward-rollout stiffness problem (training converges to low loss, but the
# full-trajectory Euler rollout blows up), NOT a backward gradient/optimization
# problem -- so clipping is inert. v3 probes two forward-stability levers on the
# divergent cells: finer ODE integration (ode_unfolds) and a longer training
# horizon (batch_time). LRC's separate NCP-vanishing pathology is already
# explained by the existing gradient data, so v3 carries lrc/mm_lrc only in the
# robustness block, and intervenes on the stiff ltc-family on the trigger systems.
CELLS_V3 = ['ltc', 'mm_ltc', 'lrc', 'mm_lrc']        # robustness block (all ODE cells)
CELLS_V3_STIFF = ['ltc', 'mm_ltc']                   # intervention protagonists (the divergers)
SYSTEMS_V3_STIFF = ['duffing', 'periodic_predator_prey']  # the only systems that produce divergence
SEEDS_V3_EXTRA = list(range(5, 15))                  # 10 new seeds; extends v1/v2 (0-4) -> 15 total
SEEDS_V3_FULL = list(range(0, 15))                   # 15 seeds for the intervention arms
V3_ODE_UNFOLDS = 24                                  # solver-fidelity arm (vs LTC default 6)
V3_BATCH_TIME = 64                                   # training-horizon arm (vs default 16)

# Benchmark v3.1: closed-form LRC ablation (see
# docs/superpowers/specs/2026-06-19-cfc-lrc-v3_1-design.md). Implements cfc_lrc
# (CfC + a bounded liquid-elastance gate) and benchmarks it against plain cfc, a
# parameter-matched cfc (cfc_pm, wider backbone -> capacity control), and the
# numerical lrc reference, to test whether the closed-form elastance helps.
CELLS_V3_1 = ['cfc', 'cfc_lrc', 'cfc_pm', 'lrc']

# Benchmark v3.2: 2x2 architectural ablation of the LRC cell --
# {numerical, closed-form} x {plain, mixed-memory}. All four share identical
# hyperparameters (units, NCP config, training, elastance_type='asymmetric') and
# differ ONLY in cell architecture, so any pair isolates one design choice.
CELLS_V3_2 = ['lrc', 'cfc_lrc', 'mm_lrc', 'cfc_mm_lrc']

# Benchmark v3.3: parameter-matched capacity control for the v3.2 2x2. lrc_pm is a
# plain numerical LRC widened (units / NCP via CELL_UNITS / CELL_NCP) to >= the
# largest fixed cell (cfc_mm_lrc). If this over-capacity plain LRC still loses on
# ncp, the v3.2 gains are mechanistic, not capacity. Compared against the existing
# runs_v3_2 cells (identical training config), so only this control is run.
CELLS_V3_3 = ['lrc_pm']

# Benchmark v4: cross-family generalization of the architecture fixes + classical
# championship. One clean self-contained matrix (results/runs_v4), identical config
# to v3.2 (units=16, NCP 16/8/2 seed 42, LRC-family elastance_type='asymmetric',
# default training); the only difference between cells is architecture. Two LTC/LRC
# 2x2s {numerical, closed-form} x {plain, mixed-memory} plus classical baselines:
#   LTC 2x2: ltc, cfc (= closed-form LTC), mm_ltc, cfc_mm_ltc (the v4 protagonist)
#   LRC 2x2: lrc, cfc_lrc,                 mm_lrc, cfc_mm_lrc
#   classical / CT baselines: gru, lstm, ctrnn (ctrnn never benchmarked before)
# Q1 (primary): does the closed-form + mixed-memory fix that repairs LRC on the
# sparse NCP wiring GENERALIZE to LTC -- does cfc_mm_ltc eliminate mm_ltc's ~17%
# ncp divergence tail, the way cfc_mm_lrc gives mm_lrc 0%? Q2: do the fixed bio
# cells match/beat classical gru/lstm/ctrnn on ncp (and the classical dense
# ceiling)? Q3: robustness = divergence rate (share NRMSE>1) + mean, not median.
# cfc_mm_ltc and ctrnn take NO extra kwargs (CfC inner / CTRNN reject elastance_type),
# so they deliberately get no CELL_KWARGS entry. See the v4 design spec.
CELLS_V4 = ['ltc', 'cfc', 'mm_ltc', 'cfc_mm_ltc',
            'lrc', 'cfc_lrc', 'mm_lrc', 'cfc_mm_lrc',
            'gru', 'lstm', 'ctrnn']
# Base = all systems, seeds 0-4 (n=5, as v1-v3.3 -> pooled n=30 per cell x wiring).
SEEDS_V4_BASE = [0, 1, 2, 3, 4]
# Tail-power supplement: the only two systems that actually produce divergence
# (= SYSTEMS_V3_STIFF) get 5 extra seeds, so the headline Q1 tail claim
# ("cfc_mm_ltc kills mm_ltc's ncp divergence tail") rests on n=10 not n=5 there --
# a ~17% rate over n=5/system has a ~+/-13pp CI, too wide to call a residual tail
# from zero. Paired tests stay valid (every cell shares the identical seed scheme).
SEEDS_V4_TAIL = [5, 6, 7, 8, 9]
SYSTEMS_V4_TAIL = SYSTEMS_V3_STIFF  # ['duffing', 'periodic_predator_prey']

# Benchmark v5: generalization stress test (see
# docs/superpowers/specs/2026-06-22-benchmark-v5-generalization-stress-design.md).
# v1-v4 train and evaluate on the SAME single trajectory (one y0, regular grid,
# full horizon) -> a clean-fit regime that is near-tautological for the
# continuous-time cells (the explicit v4 Q2 caveat in benchmark-findings). This
# is the missing "robustness rail" the roadmap/deep-research review flagged
# (Noise / Sampling-irregularity / Domain-shift). v5 reruns the FULL v4 cell set
# under three INDEPENDENT generalization stressors -- decoupling the train and
# eval trajectories so a robustness gap between cells is forced to surface:
#   noise         -- train on observation-noised targets, eval vs the clean truth
#   extrapolation -- train on the first half, eval the rollout over the full horizon
#   ood_init      -- train from the canonical y0, eval from a perturbed y0'
# The clean baseline is v4 itself (identical config, no stress), so v5-vs-v4 is a
# paired comparison. No new cells (stress lives entirely in datasets.py).
CELLS_V5 = list(CELLS_V4)                 # all 11: LTC 2x2 + LRC 2x2 + gru/lstm/ctrnn
STRESS_REGIMES_V5 = list(STRESS_REGIMES)  # ('noise', 'extrapolation', 'ood_init')
SEEDS_V5 = [0, 1, 2, 3, 4]                # n=5 x 6 systems = 30 paired per cell x wiring x regime

# Benchmark v6: two verification runs probing the robustness of the v5/v4 ncp
# findings (benchmark-findings section 10). Both use a representative 8-cell
# subset -- the 4 closed-form bio champions + the 3 classical baselines + lrc as
# the single stress-invariant bottom anchor -- on ncp only (the hard wiring; dense
# never collapses). The heavy numerical ltc/mm_ltc/mm_lrc are dropped: they only
# re-confirm the bottom band lrc already anchors and would double walltime.
CELLS_V6 = ['cfc', 'cfc_lrc', 'cfc_mm_lrc', 'cfc_mm_ltc', 'gru', 'lstm', 'ctrnn', 'lrc']
# v6a (multi-wiring-seed robustness): does the ncp bio>classical ordering survive
# OTHER wiring graphs, or was seed 42 lucky? 4 NEW graphs + the existing seed-42
# data (reused from v5) = 5 graphs, run under the NOISE regime -- the most fragile
# regime on graph 42 (the only one where bio loses to classical), the hardest test.
# 5 graphs let a unanimous sign test reach p = 2^-5 = 0.031 < 0.05.
WIRING_SEEDS_V6A = [7, 13, 21, 99]
# v6b (stress-level dose-response): is the v5 ordering a monotone dose-response or
# a floor/ceiling artifact at the single chosen magnitude? Each stressor gets 2 NEW
# levels bracketing the v5 point; v4 (clean, level 0) and v5 (mid) are the reused
# anchors. The level is a pure scalar multiplier on the fixed RNG draw (datasets.py),
# so the 4 ordered points form a clean dose-response (Page/Jonckheere trend test).
STRESS_LEVELS_V6B = {
    'noise': [('stress_noise_level', 0.05), ('stress_noise_level', 0.20)],
    'extrapolation': [('stress_train_fraction', 0.70), ('stress_train_fraction', 0.30)],
    'ood_init': [('stress_ood_scale', 0.10), ('stress_ood_scale', 0.40)],
}

DENSE_UNITS = 16
# inter=16, command=8 chosen to be comparable to Dense units=16.
# motor_neurons=2 matches the 2-dimensional ODE output.
NCP_CONFIG = dict(inter_neurons=16, command_neurons=8, motor_neurons=2)
# Fixed wiring seed: all training seeds share the same NCP graph, so seed
# variance measures init/batch randomness, not wiring randomness.
NCP_WIRING_SEED = 42

# Per-cell capacity overrides (default = DENSE_UNITS / NCP_CONFIG for every cell,
# so v1/v2/v3/v3.1/v3.2 are unchanged). lrc_pm is the v3.3 param-matched control:
# a plain LRC widened to >= the largest v3.2 fixed cell (cfc_mm_lrc: 2946 dense /
# 4244 ncp). units=24 -> 3294 dense; (inter=22,command=12,motor=2) -> 4620 ncp.
CELL_UNITS = {'lrc_pm': 24}
CELL_NCP = {'lrc_pm': dict(inter_neurons=22, command_neurons=12, motor_neurons=2)}

DEFAULTS = dict(n_iters=2000, batch_size=16, batch_time=16, lr=1e-3,
                loss='mse', grad_log_every=25, data_size=1000)

# Per-cell constructor overrides forwarded to the cell (and every NCP layer).
# LRC defaults to elastance_type="interp", which falls through to a constant
# elastance (no liquid capacitance) -- effectively a saturated LTC. "asymmetric"
# activates the input/state-dependent liquid elastance that defines the LRC model
# and that RQ2/RQ5 isolate. Other cells take no extra kwargs.
CELL_KWARGS = {
    'lrc': dict(elastance_type='asymmetric'),
    'mm_lrc': dict(elastance_type='asymmetric'),
    # cfc_lrc: liquid-elastance gate, asymmetric to match lrc/mm_lrc.
    'cfc_lrc': dict(elastance_type='asymmetric'),
    # cfc_mm_lrc: inner CfC_LRC gets the same asymmetric elastance (v3.2 2x2).
    'cfc_mm_lrc': dict(elastance_type='asymmetric'),
    # cfc_pm: parameter-matched plain CfC. backbone_units=20 makes its dense
    # param count slightly exceed cfc_lrc's (cfc/cfc_lrc default backbone=units=16);
    # for NCP it is over-provisioned (conservative capacity control). See the v3.1
    # design doc for the exact counts.
    'cfc_pm': dict(backbone_units=20),
    # lrc_pm: plain LRC capacity control (v3.3), same elastance as lrc; widened
    # via CELL_UNITS / CELL_NCP, not via kwargs.
    'lrc_pm': dict(elastance_type='asymmetric'),
    # eps-ablation (8 conditions, all plain LRC_Cell). See the eps spec table:
    # A interp / B asym / C sym / D frozen / E pm-pad / E_C pm-pad+extra /
    # F asym hybrid / G interp hybrid. forget_gate=True is the LRC_Cell default
    # (required by the hybrid solver), so it is not re-stated here.
    'lrc_interp': dict(elastance_type='interp'),                       # A
    'lrc_asym': dict(elastance_type='asymmetric'),                     # B
    'lrc_sym': dict(elastance_type='symmetric'),                      # C
    'lrc_frozen': dict(elastance_type='asymmetric', freeze_elastance=True),  # D
    'lrc_pmctrl': dict(elastance_type='interp', pm_pad=True),          # E
    'lrc_pmctrl_c': dict(elastance_type='interp', pm_pad=True, pm_pad_extra=16),  # E_C
    'lrc_asym_hybrid': dict(elastance_type='asymmetric', ode_solver='hybrid'),    # F
    'lrc_interp_hybrid': dict(elastance_type='interp', ode_solver='hybrid'),      # G
}

# Benchmark eps: liquid-elastance over-parameterization ablation (see
# scratchpad/eps-ablation/finalSpec.md). 8 LRC conditions x 2 wirings on a
# two-tier task suite. The headline equivalence test (B~A) lives on
# multitimescale @ uf=1; spiral / stiff_linear_k1 are flat falsification anchors;
# stiff_linear_k{10,100,1000} are exploratory. n=30 seeds for the confirmatory
# power scheme (Delta_min=1.0*SD, alpha=0.05/15).
CELLS_EPS = ['lrc_interp', 'lrc_asym', 'lrc_sym', 'lrc_frozen', 'lrc_pmctrl',
             'lrc_pmctrl_c', 'lrc_asym_hybrid', 'lrc_interp_hybrid']
# sym (C, E_C) and hybrid (F, G) arms are pruned to where they are informative;
# the always-run interp/asym arms (A, B, D, E) cover every task.
CELLS_EPS_SYM = ['lrc_sym', 'lrc_pmctrl_c']
CELLS_EPS_HYBRID = ['lrc_asym_hybrid', 'lrc_interp_hybrid']
CELLS_EPS_ALWAYS = ['lrc_interp', 'lrc_asym', 'lrc_frozen', 'lrc_pmctrl']
# Confirmatory headline host + its unfold robustness sweep, flat anchors, and the
# exploratory outer-stiffness sweep (spec section 2 / 5).
EPS_HEADLINE_UNFOLDS = [1, 2, 4]
EPS_ANCHORS = ['spiral', 'stiff_linear_k1']
EPS_EXPLORATORY = ['stiff_linear_k10', 'stiff_linear_k100', 'stiff_linear_k1000']
SEEDS_EPS = list(range(30))   # n=30 confirmatory paired seeds


def build_specs(cells, wirings, systems, seeds, clip_norm=0.0):
    """Deterministic run-spec list; index order is the SLURM array contract."""
    return [
        {'cell': c, 'wiring': w, 'system': sy, 'seed': se, 'clip_norm': clip_norm}
        for c, w, sy, se in product(cells, wirings, systems, seeds)
    ]


def build_specs_v2(systems=SYSTEMS, seeds=SEEDS):
    """v2 matrix: fixed cells x {no clip, clip} + problem cells x clip.

    Order (= SLURM array contract for v2):
      [0,180):   CELLS_V2, clip off
      [180,360): CELLS_V2, clip V2_CLIP_NORM
      [360,480): ltc/lrc,  clip V2_CLIP_NORM (isolates the optimizer fix)
    """
    return (
        build_specs(CELLS_V2, WIRINGS, systems, seeds, clip_norm=0.0)
        + build_specs(CELLS_V2, WIRINGS, systems, seeds, clip_norm=V2_CLIP_NORM)
        + build_specs(['ltc', 'lrc'], WIRINGS, systems, seeds, clip_norm=V2_CLIP_NORM)
    )


def build_specs_v3(systems=SYSTEMS, seeds_extra=SEEDS_V3_EXTRA,
                   seeds_full=SEEDS_V3_FULL):
    """v3 matrix: forward-rollout-stability probe on the stiff ODE cells.

    Order (= SLURM array contract for v3):
      [0,480):   robustness      -- CELLS_V3 x WIRINGS x all systems x seeds 5-14
                 (baseline; extends the v1/v2 seeds 0-4 to 15 total)
      [480,600): solver-fidelity -- CELLS_V3_STIFF x WIRINGS x stiff systems
                 x seeds 0-14, ode_unfolds=24
      [600,720): training-horizon -- CELLS_V3_STIFF x WIRINGS x stiff systems
                 x seeds 0-14, batch_time=64

    Baselines for the intervention arms (ode_unfolds=6, batch_time=16) come from
    the seeds-0-4 v1/v2 runs plus the seeds-5-14 robustness block, so the
    aggregation pools results/runs + results/runs_v2 + results/runs_v3.
    """
    robustness = build_specs(CELLS_V3, WIRINGS, systems, seeds_extra, clip_norm=0.0)
    solver = build_specs(CELLS_V3_STIFF, WIRINGS, SYSTEMS_V3_STIFF, seeds_full,
                         clip_norm=0.0)
    for s in solver:
        s['ode_unfolds'] = V3_ODE_UNFOLDS
    horizon = build_specs(CELLS_V3_STIFF, WIRINGS, SYSTEMS_V3_STIFF, seeds_full,
                          clip_norm=0.0)
    for s in horizon:
        s['batch_time'] = V3_BATCH_TIME
    return robustness + solver + horizon


def build_specs_v3_1(systems=SYSTEMS, seeds=SEEDS):
    """v3.1 matrix: closed-form LRC ablation (see
    docs/superpowers/specs/2026-06-19-cfc-lrc-v3_1-design.md).

      CELLS_V3_1 x WIRINGS x systems x seeds = 4 x 2 x 6 x 5 = 240 runs
      -> results/runs_v3_1

    cfc (LTC-derived baseline) / cfc_lrc (new closed-form LRC) / cfc_pm
    (parameter-matched plain CfC, capacity control) / lrc (numerical reference).
    All on the baseline arm (clip off, default solver/horizon).
    """
    return build_specs(CELLS_V3_1, WIRINGS, systems, seeds, clip_norm=0.0)


def build_specs_v3_2(systems=SYSTEMS, seeds=SEEDS):
    """v3.2 matrix: 2x2 architectural ablation of the LRC cell.

      CELLS_V3_2 x WIRINGS x systems x seeds = 4 x 2 x 6 x 5 = 240 runs
      -> results/runs_v3_2

    {lrc, cfc_lrc, mm_lrc, cfc_mm_lrc} = {numerical, closed-form} x
    {plain, mixed-memory}. Identical hyperparameters across all four; only the
    cell architecture differs, so any pair isolates one design axis.
    """
    return build_specs(CELLS_V3_2, WIRINGS, systems, seeds, clip_norm=0.0)


def build_specs_v3_3(systems=SYSTEMS, seeds=SEEDS):
    """v3.3 matrix: parameter-matched capacity control for the v3.2 2x2.

      CELLS_V3_3 x WIRINGS x systems x seeds = 1 x 2 x 6 x 5 = 60 runs
      -> results/runs_v3_3

    lrc_pm = plain numerical LRC widened (CELL_UNITS/CELL_NCP) to >= the largest
    v3.2 fixed cell. Analyzed against the existing runs_v3_2 cells (identical
    training config), so only this control is (re)run.
    """
    return build_specs(CELLS_V3_3, WIRINGS, systems, seeds, clip_norm=0.0)


def build_specs_v4(systems=SYSTEMS, base_seeds=SEEDS_V4_BASE,
                   tail_systems=SYSTEMS_V4_TAIL, tail_seeds=SEEDS_V4_TAIL):
    """v4 matrix: cross-family generalization + classical championship.

      base: CELLS_V4 x WIRINGS x all systems x seeds 0-4 = 11 x 2 x 6 x 5 = 660
      tail: CELLS_V4 x WIRINGS x {duffing, periodic_predator_prey} x seeds 5-9
            = 11 x 2 x 2 x 5 = 220   (extra power on the divergence-prone systems)
      total = 880 runs -> results/runs_v4

    The tail uses disjoint seeds (5-9) on the two stiff systems, so its filenames
    never collide with the base (seeds 0-4). Self-contained single-dir matrix:
    every cell is (re)run at one fixed code/config, so cross-cell comparison and
    the reproducibility cross-check against v1-v3.2 need no cross-dir pooling.
    """
    base = build_specs(CELLS_V4, WIRINGS, systems, base_seeds, clip_norm=0.0)
    tail = build_specs(CELLS_V4, WIRINGS, tail_systems, tail_seeds, clip_norm=0.0)
    return base + tail


def build_specs_v5(systems=SYSTEMS, seeds=SEEDS_V5, regimes=STRESS_REGIMES_V5):
    """v5 matrix: generalization stress test over the full v4 cell set.

      per regime: CELLS_V5 x WIRINGS x all systems x seeds 0-4 = 11 x 2 x 6 x 5 = 660
      3 regimes (noise, extrapolation, ood_init)                 = 1980 runs
      -> results/runs_v5

    Order (= SLURM array contract for v5) is regime-major, then the standard
    build_specs cell-major order within each regime:
      [0,660):     noise
      [660,1320):  extrapolation
      [1320,1980): ood_init
    Each spec carries a 'stress' key naming its regime; run_one routes on it and
    result_filename appends a _stress-<regime> suffix so v5 files never collide
    with the clean v4 matrix. The clean (no-stress) baseline is v4 itself.
    """
    specs = []
    for regime in regimes:
        for s in build_specs(CELLS_V5, WIRINGS, systems, seeds, clip_norm=0.0):
            specs.append({**s, 'stress': regime})
    return specs


def build_specs_v6a(systems=SYSTEMS, seeds=SEEDS_V5, wiring_seeds=WIRING_SEEDS_V6A):
    """v6a matrix: multi-wiring-seed robustness of the noise-regime ncp ordering.

      per graph: CELLS_V6 x ['ncp'] x all systems x seeds 0-4 = 8 x 1 x 6 x 5 = 240
      4 NEW wiring graphs (7, 13, 21, 99)                       = 960 runs
      -> results/runs_v6a   (the existing seed-42 noise data lives in results/runs_v5)

    Order (= SLURM array contract) is wiring-seed-major, then cell-major. Every
    spec is the v5 NOISE regime on ncp; only ncp_wiring_seed differs, so each new
    graph is directly comparable to the seed-42 v5 anchor on (system, seed).
    """
    specs = []
    for ws in wiring_seeds:
        for s in build_specs(CELLS_V6, ['ncp'], systems, seeds, clip_norm=0.0):
            specs.append({**s, 'stress': 'noise', 'ncp_wiring_seed': ws})
    return specs


def build_specs_v6b(systems=SYSTEMS, seeds=SEEDS_V5, levels=STRESS_LEVELS_V6B):
    """v6b matrix: stress-level dose-response on ncp (floor/ceiling check).

      per (regime, new level): CELLS_V6 x ['ncp'] x systems x seeds = 8 x 6 x 5 = 240
      3 regimes x 2 new levels each                                  = 1440 runs
      -> results/runs_v6b   (clean=v4 and the mid level=v5 are the reused anchors)

    Order (= SLURM array contract) is regime-major, then level, then cell-major.
    Each spec carries the regime ('stress') plus ONE level-override key
    (stress_noise_level / stress_train_fraction / stress_ood_scale); run_one
    forwards it to generate_stress_dataset and result_filename appends a _lvl<v>
    token so the two new levels never overwrite each other or the v5 baseline.
    """
    specs = []
    for regime, regime_levels in levels.items():
        for key, val in regime_levels:
            for s in build_specs(CELLS_V6, ['ncp'], systems, seeds, clip_norm=0.0):
                specs.append({**s, 'stress': regime, key: val})
    return specs


def _eps_cells_for_task(system: str):
    """Which eps conditions run on a given task (sym/hybrid pruning, spec 5).

    The always-run interp/asym arms (A, B, D, E) run on every task. The sym arms
    (C, E_C) and hybrid arms (F, G) are restricted to the informative tasks --
    multitimescale and the stiff_linear sweep -- never the flat spiral anchor.
    """
    cells = list(CELLS_EPS_ALWAYS)
    informative = (system == 'multitimescale' or system.startswith('stiff_linear'))
    if informative:
        cells += CELLS_EPS_SYM + CELLS_EPS_HYBRID
    # preserve the canonical CELLS_EPS order for a stable array contract
    return [c for c in CELLS_EPS if c in set(cells)]


def build_specs_eps(seeds=SEEDS_EPS, wirings=WIRINGS):
    """eps matrix: liquid-elastance over-parameterization ablation.

    Order (= SLURM array contract for eps), task-block major then the canonical
    CELLS_EPS x wirings x seeds order within each block:
      1. multitimescale @ uf in {1,2,4}   -- confirmatory headline + robustness
      2. flat anchors (spiral, stiff_linear_k1) @ uf=1  -- falsification only
      3. exploratory stiff_linear_k{10,100,1000} @ uf=1 -- descriptive

    Each spec carries the dataset params that thread through run_one /
    generate_dataset and result_filename: 'ode_unfolds' (per-task), the kappa via
    the system name (stiff_linear_k<kappa>), and 'eps_jitter' (the frozen-scale
    seeded data variation). sym (C, E_C) and hybrid (F, G) arms are pruned to the
    informative tasks (multiscale + stiff), never the spiral anchor.
    """
    specs = []

    def _block(system, unfolds):
        for uf in unfolds:
            for cell in _eps_cells_for_task(system):
                for w in wirings:
                    for se in seeds:
                        specs.append({
                            'cell': cell, 'wiring': w, 'system': system,
                            'seed': se, 'clip_norm': 0.0,
                            'ode_unfolds': uf, 'eps_jitter': True,
                        })

    _block('multitimescale', EPS_HEADLINE_UNFOLDS)
    for anchor in EPS_ANCHORS:
        _block(anchor, [1])
    for expl in EPS_EXPLORATORY:
        _block(expl, [1])
    return specs


# eps §3.0 pilot: the hard go/no-go gate run BEFORE the full eps matrix. It
# measures paired-diff SD per (task, level) on the final jittered data path, so
# it needs ALL 8 conditions on EVERY pilot task (no sym/hybrid pruning -- every
# condition's SD must be estimable), at the real n_iters=2000 so the SD reflects
# converged runs. Subset: multitimescale @ uf in {1,2,4} + the two flat anchors
# (spiral, stiff_linear_k1) @ uf=1, seeds 0..7 (8 seeds), jitter ON. The
# exploratory stiff_linear_k{10,100,1000} sweep is NOT in the pilot.
SEEDS_EPS_PILOT = list(range(8))     # 8 seeds (>=8-seed pilot, spec §3.0.1)
EPS_PILOT_ANCHORS = ['spiral', 'stiff_linear_k1']


def build_specs_eps_pilot(seeds=SEEDS_EPS_PILOT, wirings=WIRINGS):
    """eps §3.0 pilot subset: ALL 8 conditions (no pruning) on the pilot tasks.

    Unlike build_specs_eps, the pilot does NOT prune sym/hybrid off the anchors:
    the SD pilot needs every condition on every task to estimate all paired-diff
    SDs that feed Delta_min and the power precompute. Order (= SLURM array
    contract for eps_pilot): multitimescale @ uf in {1,2,4} then spiral,
    stiff_linear_k1 @ uf=1; within each block the canonical CELLS_EPS x wirings x
    seeds order. Every spec carries eps_jitter=True and its ode_unfolds.
    """
    specs = []

    def _block(system, unfolds):
        for uf in unfolds:
            for cell in CELLS_EPS:               # all 8, no pruning
                for w in wirings:
                    for se in seeds:
                        specs.append({
                            'cell': cell, 'wiring': w, 'system': system,
                            'seed': se, 'clip_norm': 0.0,
                            'ode_unfolds': uf, 'eps_jitter': True,
                        })

    _block('multitimescale', EPS_HEADLINE_UNFOLDS)
    for anchor in EPS_PILOT_ANCHORS:
        _block(anchor, [1])
    return specs


def build_model(cell: str, wiring: str, ode_unfolds=None,
                ncp_wiring_seed=NCP_WIRING_SEED) -> SequentialODEFunc:
    cell_kwargs = dict(CELL_KWARGS.get(cell, {}))
    if ode_unfolds is not None:
        cell_kwargs['ode_unfolds'] = ode_unfolds
    units = CELL_UNITS.get(cell, DENSE_UNITS)
    ncp_cfg = CELL_NCP.get(cell, NCP_CONFIG)
    if wiring == 'dense':
        net = make_dense_model(cell, units=units, output_neurons=2, **cell_kwargs)
    else:
        net = make_ncp_model(cell, seed=ncp_wiring_seed, **ncp_cfg, **cell_kwargs)
    return SequentialODEFunc(net)


def evaluate_full_trajectory(model, t, y):
    """Roll out the trained model from y[0] over the full time grid."""
    y0 = tf.constant(y[0][np.newaxis, np.newaxis, :], dtype=tf.float32)
    t_tf = tf.constant(t, dtype=tf.float32)
    pred = euler_odeint(model, y0, t_tf)            # (T, 1, 1, 2)
    pred_traj = pred.numpy().reshape(len(t), -1)    # (T, 2)
    return {
        'mse': mse(y, pred_traj),
        'nrmse': nrmse(y, pred_traj),
        'trajectory_pred': pred_traj.astype(float).tolist(),
    }


def _measure_live_gate(model, t, y):
    """Live-gate diagnostic (spec 3.0.6) on the TRAINED model: CoV of the gate
    output elastance_t over the rollout, and the gradient-norm ratio through
    elastance_mapping vs. the drive Dense. Returns (cov, grad_ratio) or (nan,nan)
    if the cell has no active gate. Cheap (one short rollout + one backward)."""
    cells = _iter_lrc_cells(model)
    active = [c for c in cells if c._elastance_type in ('asymmetric', 'symmetric')
              and getattr(c, 'elastance_mapping', None) is not None]
    if not active:
        return float('nan'), float('nan')
    for c in active:
        c._capture_gate = True
    y0 = tf.constant(y[0][np.newaxis, np.newaxis, :], dtype=tf.float32)
    t_tf = tf.constant(t, dtype=tf.float32)
    # CoV of elastance_t across outer steps.
    vals = []
    state = y0
    for i in range(min(len(t) - 1, 64)):
        dt = float(t[i + 1] - t[i])
        deriv = model(t_tf[i], state)
        state = state + dt * deriv
        for c in active:
            if c._last_elastance_t is not None:
                vals.append(float(tf.math.reduce_mean(c._last_elastance_t)))
    arr = np.asarray(vals, dtype=float)
    cov = float(arr.std() / (abs(arr.mean()) + 1e-12)) if arr.size else float('nan')
    # Gradient norm through elastance_mapping vs. the largest non-elastance Dense
    # (the "drive") -- a ratio near 1 means the gate is in the optimization loop.
    with tf.GradientTape() as tape:
        pred = euler_odeint(model, y0, t_tf[: min(len(t), 64)])
        loss = tf.reduce_mean(pred ** 2)
    gs = tape.gradient(loss, model.trainable_variables)
    g_elast = g_drive = 0.0
    for g, v in zip(gs, model.trainable_variables):
        if g is None:
            continue
        gn = float(tf.reduce_sum(g ** 2))
        if 'elastance_mapping' in v.name:
            g_elast += gn
        elif 'kernel' in v.name:
            g_drive = max(g_drive, gn)
    ratio = (g_elast ** 0.5) / (g_drive ** 0.5) if g_drive > 0 else float('nan')
    for c in active:
        c._capture_gate = False
    return cov, ratio


def run_one(spec: dict, cfg: dict) -> dict:
    rng = set_global_seed(spec['seed'], deterministic_ops=cfg['deterministic'])

    # v5 stress regimes decouple the train and eval trajectories (datasets.py).
    # Absent in v1-v4 specs -> the clean single-trajectory path (t_eval is t).
    stress = spec.get('stress')
    if stress:
        data = generate_stress_dataset(spec['system'], stress,
                                       data_size=cfg['data_size'], seed=spec['seed'],
                                       noise_level=spec.get('stress_noise_level'),
                                       train_fraction=spec.get('stress_train_fraction'),
                                       ood_scale=spec.get('stress_ood_scale'))
        t, y = data['t_train'], data['y_train']
        t_eval, y_eval = data['t_eval'], data['y_eval']
        stress_y0_eval = data['y0_eval']
    else:
        # eps-ablation: when eps_jitter is set, the dataset gets the frozen-scale
        # seeded data variation (y0/kappa jitter), keyed on the spec seed so every
        # condition sharing (system, seed) sees the identical realization. Absent
        # in v1-v6 specs -> the deterministic clean trajectory (jitter OFF).
        eps_jitter = bool(spec.get('eps_jitter', False))
        t, y = generate_dataset(spec['system'], data_size=cfg['data_size'],
                                seed=spec['seed'] if eps_jitter else None,
                                jitter=eps_jitter)
        t_eval, y_eval = t, y
        stress_y0_eval = None

    # v3 per-spec overrides (absent in v1/v2 specs -> baseline behavior).
    ode_unfolds = spec.get('ode_unfolds')                  # None unless solver-fidelity arm
    batch_time = int(spec.get('batch_time', cfg['batch_time']))

    model = build_model(spec['cell'], spec['wiring'], ode_unfolds=ode_unfolds,
                        ncp_wiring_seed=int(spec.get('ncp_wiring_seed', NCP_WIRING_SEED)))
    tracker = GradientFlowTracker(log_every=cfg['grad_log_every'])

    clip_norm = float(spec.get('clip_norm', 0.0))
    t0 = time.time()
    losses = train(
        model, t, y,
        n_iters=cfg['n_iters'], batch_size=cfg['batch_size'],
        batch_time=batch_time, lr=cfg['lr'], loss=cfg['loss'],
        rng=rng, gradient_tracker=tracker,
        clip_norm=clip_norm or None,
    )
    duration = time.time() - t0

    evaluation = evaluate_full_trajectory(model, t_eval, y_eval)

    # eps-ablation: live-gate diagnostic on the trained model (B/C only). Written
    # into config so eps_analysis.py gates every Rule-4 (REDUNDANT) verdict on it.
    live_gate = {}
    if spec.get('eps_jitter'):
        cov, grad_ratio = _measure_live_gate(model, t_eval, y_eval)
        if cov == cov:   # not NaN -> the cell has an active gate
            live_gate = {'live_gate_cov': cov, 'live_gate_grad_ratio': grad_ratio}

    return {
        'schema_version': 2,
        'run': spec,
        'config': {
            **{k: cfg[k] for k in ('n_iters', 'batch_size', 'lr', 'loss',
                                   'data_size', 'deterministic')},
            'batch_time': batch_time,          # effective value (may be a v3 override)
            'dense_units': CELL_UNITS.get(spec['cell'], DENSE_UNITS),
            'ncp': CELL_NCP.get(spec['cell'], NCP_CONFIG),
            'ncp_wiring_seed': int(spec.get('ncp_wiring_seed', NCP_WIRING_SEED)),
            'cell_kwargs': CELL_KWARGS.get(spec['cell'], {}),
            'clip_norm': clip_norm,
            **({'ode_unfolds': int(ode_unfolds)} if ode_unfolds is not None else {}),
            **({'eps_jitter': True} if spec.get('eps_jitter') else {}),
            **live_gate,
            **({'stress': stress, 'stress_y0_eval': stress_y0_eval,
                'stress_level': next((v for v in (spec.get('stress_noise_level'),
                                                  spec.get('stress_train_fraction'),
                                                  spec.get('stress_ood_scale'))
                                      if v is not None), None)} if stress else {}),
        },
        'env': {
            'tensorflow': tf.__version__,
            'gpus': [d.name for d in tf.config.list_physical_devices('GPU')],
            'hostname': socket.gethostname(),
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        },
        'training': {
            'loss_history': [float(x) for x in losses],
            'initial_loss': float(losses[0]),
            'final_loss': float(losses[-1]),
            'duration_s': float(duration),
        },
        'gradient_flow': tracker.as_dict(),
        'evaluation': evaluation,
    }


def result_filename(spec: dict) -> str:
    base = f"{spec['cell']}_{spec['wiring']}_{spec['system']}_seed{spec['seed']}"
    if spec.get('stress'):
        base += f"_stress-{spec['stress']}"
    # v6a: distinguish wiring graphs so seed-7 never overwrites the seed-42 v5 file.
    if spec.get('ncp_wiring_seed') and spec['ncp_wiring_seed'] != NCP_WIRING_SEED:
        base += f"_wseed{spec['ncp_wiring_seed']}"
    # v6b: distinguish dose-response levels so the two new levels of each regime
    # never overwrite each other (or the v5 baseline level).
    # None-aware (not `or`): a literal 0.0 level must still emit a _lvl token,
    # otherwise a future 0.0 entry would silently overwrite the v5 baseline file.
    lvl = next((v for v in (spec.get('stress_noise_level'),
                            spec.get('stress_train_fraction'),
                            spec.get('stress_ood_scale')) if v is not None), None)
    if lvl is not None:
        base += f"_lvl{lvl}"
    if spec.get('clip_norm'):
        base += f"_clip{spec['clip_norm']}"
    # eps-ablation: a _uf<n> token tags every eps run with its ode_unfolds level
    # (incl. uf=1, the headline), so the per-level files never collide. The
    # kappa is already in the system name (stiff_linear_k<kappa>). Non-eps specs
    # keep the legacy _unfolds<n> token (only emitted for non-default unfolds).
    if spec.get('eps_jitter'):
        base += f"_uf{int(spec.get('ode_unfolds', 1))}"
    elif spec.get('ode_unfolds'):
        base += f"_unfolds{spec['ode_unfolds']}"
    if spec.get('batch_time'):
        base += f"_bt{spec['batch_time']}"
    return base + '.json'


def save_result(result: dict, outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, result_filename(result['run']))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)
    return path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Thesis benchmark runner (cell x wiring x system x seed)')
    sel = p.add_argument_group('run selection')
    sel.add_argument('--list', action='store_true', help='print all run specs and exit')
    sel.add_argument('--count', action='store_true', help='print number of run specs and exit')
    sel.add_argument('--index', type=int, default=None, help='run spec by index (SLURM array)')
    sel.add_argument('--all', action='store_true', help='run all specs sequentially')
    sel.add_argument('--profile', choices=['v1', 'v2', 'v3', 'v3.1', 'v3.2', 'v3.3', 'v4', 'v5', 'v6a', 'v6b', 'eps', 'eps_pilot'], default='v1',
                     help='v1: thesis matrix (240 runs, results/runs). '
                          'v2: fixed cells + clip axis (480 runs, results/runs_v2). '
                          'v3: rollout-stability probe (720 runs, results/runs_v3). '
                          'v3.1: closed-form LRC ablation (240 runs, results/runs_v3_1). '
                          'v3.2: LRC 2x2 architecture ablation (240 runs, results/runs_v3_2). '
                          'v3.3: param-matched lrc capacity control (60 runs, results/runs_v3_3). '
                          'v4: cross-family generalization + classical championship '
                          '(880 runs, results/runs_v4). '
                          'v5: generalization stress test over the v4 cell set '
                          '(noise/extrapolation/ood_init, 1980 runs, results/runs_v5). '
                          'v6a: multi-wiring-seed robustness of the noise ordering '
                          '(8 cells, 4 new graphs, 960 runs, results/runs_v6a). '
                          'v6b: stress-level dose-response sweep '
                          '(8 cells, 3 regimes x 2 new levels, 1440 runs, results/runs_v6b)')
    sel.add_argument('--clip-norm', type=float, default=0.0,
                     help='gradient clip threshold for explicit single runs '
                          '(0 = off); profile runs take it from the spec')
    # eps-ablation systems (multitimescale + the stiff_linear sweep) extend the
    # base SYSTEMS for the explicit --system / --cell paths.
    _EPS_SYSTEMS = ['multitimescale'] + [f'stiff_linear_k{k}' for k in (1, 10, 100, 1000)]
    sel.add_argument('--cell', choices=CELLS + CELLS_V2 + ['cfc_lrc', 'cfc_pm', 'cfc_mm_lrc', 'lrc_pm', 'cfc_mm_ltc', 'ctrnn'] + CELLS_EPS, default=None)
    sel.add_argument('--wiring', choices=WIRINGS, default=None)
    sel.add_argument('--system', choices=SYSTEMS + _EPS_SYSTEMS, default=None)
    sel.add_argument('--seed', type=int, default=None)
    sel.add_argument('--stress', choices=STRESS_REGIMES_V5, default=None,
                     help='v5 stress regime for an explicit single run '
                          '(profile runs take it from the spec)')

    mat = p.add_argument_group('matrix filters (apply before indexing)')
    mat.add_argument('--cells', default=None, help='comma-separated cell subset')
    # eps-ablation aliases: --conditions == --cells, --tasks == --systems, so the
    # spec's smoke command reads naturally. Both feed the same filters below.
    mat.add_argument('--conditions', default=None,
                     help='eps alias for --cells (comma-separated condition subset)')
    mat.add_argument('--systems', default=None, help='comma-separated system subset')
    mat.add_argument('--tasks', default=None,
                     help='eps alias for --systems (comma-separated task subset)')
    mat.add_argument('--wirings', default=None, help='comma-separated wiring subset')
    mat.add_argument('--seeds', default=None, help='comma-separated seeds')
    mat.add_argument('--ode-unfolds', default=None,
                     help='comma-separated eps ode_unfolds filter (e.g. "1")')
    mat.add_argument('--regimes', default=None,
                     help='comma-separated v5 stress-regime subset '
                          '(noise,extrapolation,ood_init)')

    eps = p.add_argument_group('eps-ablation smoke checks')
    eps.add_argument('--smoke', action='store_true',
                     help='eps param-audit / registration smoke matrix (no training)')
    eps.add_argument('--n-iters', type=int, default=None,
                     help='eps alias for --iters')
    eps.add_argument('--assert-param-audit', action='store_true',
                     help='eps smoke: assert built-model elastance param counts')
    eps.add_argument('--assert-live-gradient', action='store_true',
                     help='eps smoke: assert E/E_C pad gradient is not dead')
    eps.add_argument('--dry-run-live-gate', action='store_true',
                     help='eps smoke: dry-run the B/C live-gate CoV/gradient hook')

    tr = p.add_argument_group('training config')
    tr.add_argument('--iters', type=int, default=DEFAULTS['n_iters'])
    tr.add_argument('--batch-size', type=int, default=DEFAULTS['batch_size'])
    tr.add_argument('--batch-time', type=int, default=DEFAULTS['batch_time'])
    tr.add_argument('--lr', type=float, default=DEFAULTS['lr'])
    tr.add_argument('--loss', choices=['mse', 'mae'], default=DEFAULTS['loss'])
    tr.add_argument('--grad-log-every', type=int, default=DEFAULTS['grad_log_every'],
                    help='gradient-norm logging interval; 0 disables (RQ4)')
    tr.add_argument('--data-size', type=int, default=DEFAULTS['data_size'])
    tr.add_argument('--deterministic', action='store_true',
                    help='enable TF op determinism (bit-exact, slower)')
    tr.add_argument('--outdir', default=None,
                    help='default: results/runs (v1) / results/runs_v2 (v2)')
    return p.parse_args(argv)


def _build_eps_model(condition: str, wiring: str, ode_unfolds=1):
    """Build one eps condition as a SequentialODEFunc and force its weights."""
    model = build_model(condition, wiring, ode_unfolds=ode_unfolds)
    y0 = tf.zeros([1, 1, 2])
    _ = model(tf.constant(0.0), y0)   # force lazy build (D=2 state)
    return model


def _elastance_param_count(model) -> int:
    """Sum of trainable elastance_mapping + distr_shift params on a built model."""
    return int(sum(
        np.prod(v.shape) for v in model.trainable_variables
        if 'elastance_mapping' in v.name or 'distr_shift' in v.name
    ))


def _pad_param_count(model) -> int:
    return int(sum(
        np.prod(v.shape) for v in model.trainable_variables
        if 'pm_pad_mapping' in v.name or 'pm_pad_extra' in v.name
        or 'distr_shift' in v.name
    ))


def _iter_lrc_cells(model):
    """Yield every LRC_Cell instance inside a built SequentialODEFunc."""
    from src.neurons import LRC_Cell
    seen = []
    for layer in model.net.layers if hasattr(model, 'net') else []:
        cell = getattr(layer, 'cell', None)
        if isinstance(cell, LRC_Cell):
            seen.append(cell)
    return seen


def _eps_smoke(args) -> int:
    """eps pre-submission smoke matrix: param counts, registration, dead-pad /
    dead-gate instrumentation. No training (catches bugs in minutes, spec 6)."""
    wirings = (args.wirings.split(',') if args.wirings else ['dense', 'ncp'])
    print('=== eps smoke: param audit (built models, D=2) ===')
    # Expected ACTIVE elastance/pad params per wiring (built, not analytic).
    expect = {
        'dense': {'lrc_interp': 0, 'lrc_asym': 304, 'lrc_sym': 320,
                  'lrc_frozen': 0, 'lrc_pmctrl': 304, 'lrc_pmctrl_c': 320},
        'ncp':   {'lrc_interp': 0, 'lrc_asym': 450, 'lrc_sym': 476,
                  'lrc_frozen': 0, 'lrc_pmctrl': 450, 'lrc_pmctrl_c': 476},
    }
    ok = True
    for wiring in wirings:
        for cond in ['lrc_interp', 'lrc_asym', 'lrc_sym', 'lrc_frozen',
                     'lrc_pmctrl', 'lrc_pmctrl_c']:
            m = _build_eps_model(cond, wiring)
            if cond in ('lrc_pmctrl', 'lrc_pmctrl_c'):
                count = _pad_param_count(m)
            else:
                count = _elastance_param_count(m)
            exp = expect[wiring][cond]
            status = 'ok' if count == exp else 'MISMATCH'
            if count != exp:
                ok = False
            print(f'  {wiring:5} {cond:16} active_params={count:4d} '
                  f'(expect {exp}) [{status}]')
        # E == B and E_C == C per cell
        b = _pad_param_count(_build_eps_model('lrc_asym', wiring))  # 0 (no pad)
        eB = _pad_param_count(_build_eps_model('lrc_pmctrl', wiring))
        eC = _pad_param_count(_build_eps_model('lrc_pmctrl_c', wiring))
        bC = _elastance_param_count(_build_eps_model('lrc_asym', wiring))
        cC = _elastance_param_count(_build_eps_model('lrc_sym', wiring))
        print(f'  {wiring:5} E(pad)={eB} == B(elast)={bC}: '
              f'{"ok" if eB == bC else "MISMATCH"}')
        print(f'  {wiring:5} E_C(pad)={eC} == C(elast)={cC}: '
              f'{"ok" if eC == cC else "MISMATCH"}')
        if eB != bC or eC != cC:
            ok = False
        # A's elastance Dense unbuilt (param-zero baseline)
        a = _build_eps_model('lrc_interp', wiring)
        a_cells = _iter_lrc_cells(a)
        a_built = any(getattr(c.elastance_mapping, 'built', False) for c in a_cells)
        print(f'  {wiring:5} A elastance Dense unbuilt (0 params): '
              f'{"ok" if not a_built else "MISMATCH (eager-built!)"}')
        if a_built:
            ok = False
        # D frozen => elastance non-trainable
        d_cells = _iter_lrc_cells(_build_eps_model('lrc_frozen', wiring))
        d_frozen = all(not c.elastance_mapping.trainable for c in d_cells)
        print(f'  {wiring:5} D elastance non-trainable: '
              f'{"ok" if d_frozen else "MISMATCH"}')
        if not d_frozen:
            ok = False

    # hybrid runs with forget_gate=True
    print('=== eps smoke: hybrid forward pass (forget_gate=True) ===')
    for cond in ['lrc_asym_hybrid', 'lrc_interp_hybrid']:
        m = _build_eps_model(cond, 'dense')
        out = m(tf.constant(0.0), tf.zeros([1, 1, 2]))
        cells = _iter_lrc_cells(m)
        fg = all(c._forget_gate for c in cells)
        hy = all(c._ode_solver_type == 'hybrid' for c in cells)
        print(f'  {cond}: ran, forget_gate={fg}, hybrid={hy} '
              f'[{"ok" if fg and hy else "MISMATCH"}]')
        if not (fg and hy):
            ok = False

    # live-gradient diagnostic on E/E_C: pad gradient within an order of
    # magnitude of B's elastance gradient (rules out a dead pad).
    if args.assert_live_gradient:
        print('=== eps smoke: live-gradient (E/E_C pad not dead) ===')
        ratios = _eps_pad_gradient_check(wirings)
        for wiring, cond, ratio in ratios:
            # Dead-pad check: a non-dead pad has a non-negligible gradient. On the
            # UNTRAINED smoke model the additive residual naturally carries a
            # larger gradient than B's bounded sigmoid gate, so the bound here is
            # one-sided (rules out ~0); the trained-model order-of-magnitude check
            # is the analysis-time gate in the aggregator.
            status = 'ok' if ratio > 1e-3 else 'DEAD-PAD'
            print(f'  {wiring:5} {cond:14} ||dPad||/||dElast(B)|| = '
                  f'{ratio:.3g} [{status}]')

    # dry-run the live-gate CoV/gradient instrumentation on a 1-seed B/C model
    if args.dry_run_live_gate:
        print('=== eps smoke: live-gate instrumentation dry-run (B/C) ===')
        for cond in ['lrc_asym', 'lrc_sym']:
            cov, gnorm = _eps_live_gate_dryrun(cond, 'dense')
            print(f'  {cond}: CoV(elastance_t)={cov:.4g}, '
                  f'||grad elastance_mapping||={gnorm:.4g} [instrumentation ok]')

    print(f'\n=== eps smoke: {"PASS" if ok else "FAIL"} ===')
    return 0 if ok else 1


def _eps_pad_gradient_check(wirings):
    """Backprop one MSE step; return ||dPad|| / ||dElast(B)|| per (wiring,cond)."""
    out = []
    t = tf.constant(np.linspace(0, 1, 8), dtype=tf.float32)
    y = tf.constant(np.random.default_rng(0).normal(size=(8, 1, 1, 2)),
                    dtype=tf.float32)

    def grad_norm(condition, wiring, name_substr):
        m = _build_eps_model(condition, wiring)
        with tf.GradientTape() as tape:
            pred = euler_odeint(m, y[0], t)
            loss = tf.reduce_mean((pred - y) ** 2)
        gs = tape.gradient(loss, m.trainable_variables)
        tot = 0.0
        for g, v in zip(gs, m.trainable_variables):
            if g is not None and name_substr in v.name:
                tot += float(tf.reduce_sum(g ** 2))
        return tot ** 0.5

    for wiring in wirings:
        b_elast = grad_norm('lrc_asym', wiring, 'elastance_mapping')
        for cond in ['lrc_pmctrl', 'lrc_pmctrl_c']:
            pad = grad_norm(cond, wiring, 'pm_pad_mapping')
            ratio = pad / b_elast if b_elast > 0 else float('inf')
            out.append((wiring, cond, ratio))
    return out


def _eps_live_gate_dryrun(condition, wiring):
    """Capture elastance_t across a short rollout; return (CoV, grad-norm)."""
    m = _build_eps_model(condition, wiring)
    cells = _iter_lrc_cells(m)
    for c in cells:
        c._capture_gate = True
    t = tf.constant(np.linspace(0, 1, 16), dtype=tf.float32)
    y0 = tf.constant(np.random.default_rng(1).normal(size=(1, 1, 2)),
                     dtype=tf.float32)
    # Harvest elastance_t across the outer rollout (one capture per outer step).
    vals = []
    state = y0
    for i in range(len(t) - 1):
        dt = float(t[i + 1] - t[i])
        deriv = m(t[i], state)
        state = state + dt * deriv
        for c in cells:
            if c._last_elastance_t is not None:
                vals.append(float(tf.math.reduce_mean(c._last_elastance_t)))
    # Gradient norm through elastance_mapping (vs. the drive Dense scale).
    with tf.GradientTape() as tape:
        pred = euler_odeint(m, y0, t)
        loss = tf.reduce_mean(pred ** 2)
    gs = tape.gradient(loss, m.trainable_variables)
    gnorm = sum(float(tf.reduce_sum(g ** 2))
                for g, v in zip(gs, m.trainable_variables)
                if g is not None and 'elastance_mapping' in v.name) ** 0.5
    arr = np.asarray(vals, dtype=float)
    cov = float(arr.std() / (abs(arr.mean()) + 1e-12)) if arr.size else 0.0
    return cov, gnorm


def main(argv=None) -> int:
    args = parse_args(argv)

    # eps-ablation aliases: --conditions/--tasks/--n-iters map onto the canonical
    # --cells/--systems/--iters so the spec's smoke command works verbatim.
    if args.conditions and not args.cells:
        args.cells = args.conditions
    if args.tasks and not args.systems:
        args.systems = args.tasks
    if args.n_iters is not None:
        args.iters = args.n_iters

    if args.profile == 'eps' and args.smoke:
        return _eps_smoke(args)

    if args.profile == 'eps_pilot':
        specs = build_specs_eps_pilot()
    elif args.profile == 'eps':
        specs = build_specs_eps()
    elif args.profile == 'v6b':
        specs = build_specs_v6b()
    elif args.profile == 'v6a':
        specs = build_specs_v6a()
    elif args.profile == 'v5':
        specs = build_specs_v5()
    elif args.profile == 'v4':
        specs = build_specs_v4()
    elif args.profile == 'v3.3':
        specs = build_specs_v3_3()
    elif args.profile == 'v3.2':
        specs = build_specs_v3_2()
    elif args.profile == 'v3.1':
        specs = build_specs_v3_1()
    elif args.profile == 'v3':
        specs = build_specs_v3()
    elif args.profile == 'v2':
        specs = build_specs_v2()
    else:
        specs = build_specs(CELLS, WIRINGS, SYSTEMS, SEEDS)
    if args.cells:
        keep = set(args.cells.split(','))
        specs = [s for s in specs if s['cell'] in keep]
    if args.wirings:
        keep = set(args.wirings.split(','))
        specs = [s for s in specs if s['wiring'] in keep]
    if args.systems:
        keep = set(args.systems.split(','))
        specs = [s for s in specs if s['system'] in keep]
    if args.seeds:
        keep = {int(s) for s in args.seeds.split(',') if s != ''}
        specs = [s for s in specs if s['seed'] in keep]
    if args.regimes:
        keep = set(args.regimes.split(','))
        specs = [s for s in specs if s.get('stress') in keep]
    if args.ode_unfolds:
        keep = {int(u) for u in args.ode_unfolds.split(',') if u != ''}
        specs = [s for s in specs if s.get('ode_unfolds') in keep]
    if args.outdir is None:
        args.outdir = {'v2': 'results/runs_v2',
                       'v3': 'results/runs_v3',
                       'v3.1': 'results/runs_v3_1',
                       'v3.2': 'results/runs_v3_2',
                       'v3.3': 'results/runs_v3_3',
                       'v4': 'results/runs_v4',
                       'v5': 'results/runs_v5',
                       'v6a': 'results/runs_v6a',
                       'v6b': 'results/runs_v6b',
                       'eps': 'results/runs_eps',
                       'eps_pilot': 'results/runs_eps_pilot'}.get(args.profile, 'results/runs')

    if args.count:
        print(len(specs))
        return 0
    if args.list:
        for i, s in enumerate(specs):
            extra = ''
            if s.get('stress'):
                extra += f" stress={s['stress']}"
            if s.get('ode_unfolds'):
                extra += f" unfolds={s['ode_unfolds']}"
            if s.get('batch_time'):
                extra += f" bt={s['batch_time']}"
            print(f"{i:4d}  {s['cell']:<7} {s['wiring']:<6} {s['system']:<26} "
                  f"seed={s['seed']} clip={s['clip_norm']}{extra}")
        return 0

    cfg = dict(
        n_iters=args.iters, batch_size=args.batch_size, batch_time=args.batch_time,
        lr=args.lr, loss=args.loss, grad_log_every=args.grad_log_every,
        data_size=args.data_size, deterministic=args.deterministic,
    )

    if args.index is not None:
        if not (0 <= args.index < len(specs)):
            print(f'Index {args.index} out of range [0, {len(specs)})', file=sys.stderr)
            return 1
        todo = [specs[args.index]]
    elif args.all:
        todo = specs
    elif all(v is not None for v in (args.cell, args.wiring, args.system, args.seed)):
        # eps-ablation: an explicit single run under --profile eps gets the eps
        # dataset params (jitter + the requested ode_unfolds), so the spec's
        # measure-run command (--cell lrc_asym ... --ode-unfolds 1) behaves like
        # a real array run rather than a clean un-jittered single trajectory.
        eps_extra = {}
        if args.profile in ('eps', 'eps_pilot'):
            uf = int(args.ode_unfolds.split(',')[0]) if args.ode_unfolds else 1
            eps_extra = {'eps_jitter': True, 'ode_unfolds': uf}
        todo = [{'cell': args.cell, 'wiring': args.wiring,
                 'system': args.system, 'seed': args.seed,
                 'clip_norm': args.clip_norm,
                 **eps_extra,
                 **({'stress': args.stress} if args.stress else {})}]
    else:
        print('Select runs via --index, --all, or --cell/--wiring/--system/--seed '
              '(or use --list/--count).', file=sys.stderr)
        return 1

    for i, spec in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {spec['cell']} x {spec['wiring']} x "
              f"{spec['system']} (seed {spec['seed']}) ...")
        result = run_one(spec, cfg)
        path = save_result(result, args.outdir)
        ev = result['evaluation']
        print(f"  done in {result['training']['duration_s']:.1f}s — "
              f"final_loss={result['training']['final_loss']:.6f} "
              f"traj MSE={ev['mse']:.6f} NRMSE={ev['nrmse']:.4f} -> {path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
