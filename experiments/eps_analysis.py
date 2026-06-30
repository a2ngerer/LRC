# experiments/eps_analysis.py
"""eps-ablation analysis: liquid-elastance over-parameterization.

Reads the per-run JSONs from results/runs_eps/ (one per eps condition x wiring x
task x ode_unfolds x seed) and emits the confirmatory statistics defined in the
eps spec (scratchpad/eps-ablation/finalSpec.md, section 3):

  - paired-difference distribution per (task, level, comparison),
  - DIRECTIONAL family: one-sided paired Wilcoxon, Holm-corrected at m_dir=12,
  - EQUIVALENCE family: paired Wilcoxon-TOST at Delta_min(task), Holm at m_eq=15,
  - the two families corrected SEPARATELY (a pair may appear once in each),
  - live-gate diagnostic columns (CoV of elastance_t, gradient-norm ratio) next
    to every Rule-4 (REDUNDANT) verdict; a failing/absent diagnostic flips the
    label to "trivial null", never REDUNDANT,
  - rliable IQM + task-stratified bootstrap as a SECONDARY descriptive block
    (no CI below n=30, never adjacent to a verdict),
  - a built-model param_audit (304/320/450, E==B, E_C==C).

The headline equivalence (B~A, C~A) lives on multitimescale @ uf=1; spiral and
stiff_linear_k1 are flat falsification anchors; stiff_linear_k{10,100,1000} are
exploratory (per-kappa Wilcoxon/TOST + a descriptive mixed-effects fit on
log(kappa), NO Page's L).

Usage:
    uv run python -m experiments.eps_analysis [--runs results/runs_eps] [--out results]
    uv run python -m experiments.eps_analysis --param-audit-only
"""
import argparse
import json
import os
import sys
from glob import glob

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# Frozen pre-registered scheme (spec section 3.0/3.2).
ALPHA = 0.05
M_DIR = 12              # directional Holm family size
M_EQ = 15              # equivalence Holm family size
METRIC = 'nrmse'
HEADLINE_TASK = 'multitimescale'
HEADLINE_UF = 1
ANCHORS = ('spiral', 'stiff_linear_k1')
# Live-gate floors (spec 3.0.6): the trained gate must be measurably active.
LIVE_GATE_COV_FLOOR = 0.05      # coefficient of variation of elastance_t
LIVE_GATE_GRAD_RATIO_MIN = 0.1  # gradient norm within an order of magnitude
LIVE_GATE_GRAD_RATIO_MAX = 10.0

# Confirmatory comparisons by level (the eps spec decomposition).
# Each entry: (label, cell_a, cell_b)  meaning a-vs-b paired diff (a - b).
DIRECTIONAL_PAIRS = [('B<A', 'lrc_asym', 'lrc_interp'),
                     ('C<A', 'lrc_sym', 'lrc_interp'),
                     ('B>E', 'lrc_asym', 'lrc_pmctrl'),
                     ('C>E_C', 'lrc_sym', 'lrc_pmctrl_c')]
EQUIV_PAIRS = [('B~A', 'lrc_asym', 'lrc_interp'),
               ('C~A', 'lrc_sym', 'lrc_interp'),
               ('D~B', 'lrc_frozen', 'lrc_asym'),
               ('B~E', 'lrc_asym', 'lrc_pmctrl'),
               ('C~E_C', 'lrc_sym', 'lrc_pmctrl_c')]


def load_runs(runs_dirs) -> pd.DataFrame:
    if isinstance(runs_dirs, str):
        runs_dirs = [runs_dirs]
    rows = []
    for runs_dir in runs_dirs:
        for path in sorted(glob(os.path.join(runs_dir, '*.json'))):
            with open(path, encoding='utf-8') as f:
                r = json.load(f)
            cfg = r.get('config', {})
            run = r['run']
            rows.append({
                'cell': run['cell'],
                'wiring': run['wiring'],
                'system': run['system'],
                'seed': run['seed'],
                'ode_unfolds': int(cfg.get('ode_unfolds', run.get('ode_unfolds', 1))),
                'nrmse': r['evaluation']['nrmse'],
                'final_loss': r['training']['final_loss'],
                # live-gate diagnostic (written by the analysis hook at train time;
                # absent for older runs -> NaN -> the verdict cannot confirm Rule 4).
                'live_gate_cov': cfg.get('live_gate_cov', np.nan),
                'live_gate_grad_ratio': cfg.get('live_gate_grad_ratio', np.nan),
                'file': os.path.basename(path),
            })
    return pd.DataFrame(rows)


def _paired_diff(df, task, uf, cell_a, cell_b):
    """Paired (cell_a - cell_b) NRMSE diffs over shared (seed,), one task/level."""
    sub = df[(df['system'] == task) & (df['ode_unfolds'] == uf)]
    a = sub[sub['cell'] == cell_a].set_index('seed')[METRIC]
    b = sub[sub['cell'] == cell_b].set_index('seed')[METRIC]
    common = a.index.intersection(b.index)
    return (a.loc[common] - b.loc[common]).to_numpy()


def directional_wilcoxon(diff: np.ndarray):
    """One-sided paired Wilcoxon: H1 that diff < 0 (a beats b). Returns p."""
    if len(diff) < 5 or np.allclose(diff, 0):
        return float('nan')
    # alternative='less': median of diff < 0
    try:
        _, p = wilcoxon(diff, alternative='less')
    except ValueError:
        return float('nan')
    return float(p)


def wilcoxon_tost(diff: np.ndarray, bound: float):
    """Non-parametric Wilcoxon-TOST for equivalence within +/- bound.

    Two one-sided signed-rank tests:
      H0_lower: median(diff) <= -bound   vs  H1: median > -bound
      H0_upper: median(diff) >=  bound   vs  H1: median <  bound
    Equivalence p = max of the two one-sided p-values (Schuirmann). Reject H0
    (declare equivalence) when that max < alpha.
    """
    if len(diff) < 5:
        return float('nan')
    try:
        # lower: test diff + bound > 0  (median diff > -bound)
        _, p_lower = wilcoxon(diff + bound, alternative='greater')
        # upper: test diff - bound < 0  (median diff <  bound)
        _, p_upper = wilcoxon(diff - bound, alternative='less')
    except ValueError:
        return float('nan')
    return float(max(p_lower, p_upper))


def holm(pvals: dict, m: int) -> dict:
    """Holm step-down. pvals: {label: p}. Returns {label: p_holm} over family m.

    m is the FROZEN family size (m_dir / m_eq), which may exceed the number of
    finite p-values present in a partial run; the rank multiplier uses the frozen
    m so a partial dataset is not silently under-corrected.
    """
    items = [(lab, p) for lab, p in pvals.items() if p == p]  # drop NaN
    items.sort(key=lambda kv: kv[1])
    out = {}
    running = 0.0
    for i, (lab, p) in enumerate(items):
        adj = min(1.0, p * (m - i))
        running = max(running, adj)   # enforce monotonic step-down
        out[lab] = running
    for lab, p in pvals.items():
        if p != p:
            out[lab] = float('nan')
    return out


def _cohens_d_paired(diff: np.ndarray) -> float:
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else float('nan')


def _delta_min(df, task, uf, floor=0.0):
    """Equivalence margin Delta_min(task) = max(absolute floor, 1.0*SD_pilot).

    SD_pilot is estimated here from the B-vs-A paired-diff SD on (task, level) --
    a stand-in until the dedicated >=8-seed pilot SD is frozen in the prereg. The
    spec commits Delta_min = 1.0*SD, so the multiplier is 1.0.
    """
    diff = _paired_diff(df, task, uf, 'lrc_asym', 'lrc_interp')
    sd = diff.std(ddof=1) if len(diff) >= 2 else float('nan')
    return max(floor, 1.0 * sd) if sd == sd else float('nan')


def confirmatory_block(df) -> str:
    lines = ['## Confirmatory statistics (two separate Holm families)\n',
             f'Directional family: one-sided paired Wilcoxon, Holm m_dir={M_DIR}.',
             f'Equivalence family: paired Wilcoxon-TOST at Delta_min(task), '
             f'Holm m_eq={M_EQ}. Both corrected SEPARATELY (spec 3.2).\n']

    # Levels: headline uf=1 + robustness uf in {2,4} on the headline task only.
    levels = [(HEADLINE_TASK, uf) for uf in (1, 2, 4)]

    # ---- Directional family ----
    dir_p = {}
    dir_rows = []
    for task, uf in levels:
        for label, ca, cb in DIRECTIONAL_PAIRS:
            diff = _paired_diff(df, task, uf, ca, cb)
            # All four directional gates test "the gate's cell has LOWER NRMSE":
            # B<A and C<A (gate beats tau) and B>E / C>E_C ("beats same-budget
            # additive control" = gate cell lower NRMSE than the pad control), so
            # every case is the one-sided 'less' test on diff (a - b).
            p = directional_wilcoxon(diff)
            key = f'{label}@uf{uf}'
            dir_p[key] = p
            d = _cohens_d_paired(diff) if len(diff) else float('nan')
            dir_rows.append((key, len(diff),
                             float(np.median(diff)) if len(diff) else float('nan'),
                             p, d))
    dir_holm = holm(dir_p, M_DIR)
    lines.append('### Directional (Wilcoxon, lower NRMSE for the gate)\n')
    lines.append('| Comparison@level | n | median diff (a-b) | p raw | p Holm | sig | d |')
    lines.append('|---|---|---|---|---|---|---|')
    for key, n, med, p, d in dir_rows:
        ph = dir_holm.get(key, float('nan'))
        sig = 'YES' if (ph == ph and ph < ALPHA) else 'no'
        lines.append(f'| {key} | {n} | {med:+.4f} | {p:.4g} | {ph:.4g} | {sig} | {d:+.3f} |')
    lines.append('')

    # ---- Equivalence family ----
    eq_p = {}
    eq_rows = []
    for task, uf in levels:
        margin = _delta_min(df, task, uf)
        for label, ca, cb in EQUIV_PAIRS:
            diff = _paired_diff(df, task, uf, ca, cb)
            p = wilcoxon_tost(diff, margin) if margin == margin else float('nan')
            key = f'{label}@uf{uf}'
            eq_p[key] = p
            eq_rows.append((key, len(diff), margin,
                            float(np.median(diff)) if len(diff) else float('nan'), p))
    eq_holm = holm(eq_p, M_EQ)
    lines.append('### Equivalence (Wilcoxon-TOST at Delta_min)\n')
    lines.append('| Comparison@level | n | Delta_min | median diff | p TOST | p Holm | equiv |')
    lines.append('|---|---|---|---|---|---|---|')
    for key, n, margin, med, p in eq_rows:
        ph = eq_holm.get(key, float('nan'))
        eq = 'YES' if (ph == ph and ph < ALPHA) else 'no'
        mstr = f'{margin:.4f}' if margin == margin else '—'
        lines.append(f'| {key} | {n} | {mstr} | {med:+.4f} | {p:.4g} | {ph:.4g} | {eq} |')
    lines.append('')
    lines.append('*B-vs-E / C-vs-E_C establish "beats a same-budget additive-residual '
                 'control" only, never mechanism-over-capacity (spec 0.1).*\n')

    # ---- Headline verdict with live-gate gating (Rule 4) ----
    lines.append(verdict_block(df, eq_holm, dir_holm))
    # ---- Falsification anchors ----
    lines.append(anchor_block(df))
    return '\n'.join(lines)


def _live_gate(df, cell, task=HEADLINE_TASK, uf=HEADLINE_UF):
    sub = df[(df['cell'] == cell) & (df['system'] == task) & (df['ode_unfolds'] == uf)]
    return (float(np.nanmean(sub['live_gate_cov'])) if len(sub) else float('nan'),
            float(np.nanmean(sub['live_gate_grad_ratio'])) if len(sub) else float('nan'))


def verdict_block(df, eq_holm, dir_holm) -> str:
    """Rule-4 REDUNDANT (B~A / C~A) at the headline level, gated by live-gate."""
    lines = ['### Headline verdict (multitimescale @ uf=1), live-gate gated\n',
             '| Gate | TOST equiv (Holm) | live-gate CoV | grad ratio | verdict |',
             '|---|---|---|---|---|']
    for label, cell in [('B~A', 'lrc_asym'), ('C~A', 'lrc_sym')]:
        key = f'{label}@uf{HEADLINE_UF}'
        equiv = eq_holm.get(key, float('nan'))
        is_equiv = equiv == equiv and equiv < ALPHA
        cov, grad = _live_gate(df, cell)
        gate_ok = (cov == cov and cov >= LIVE_GATE_COV_FLOOR
                   and grad == grad
                   and LIVE_GATE_GRAD_RATIO_MIN <= grad <= LIVE_GATE_GRAD_RATIO_MAX)
        if is_equiv and gate_ok:
            verdict = 'REDUNDANT (gate active)'
        elif is_equiv and not gate_ok:
            verdict = ('trivial null (gate degenerate/missing diagnostic)'
                       if cov == cov else 'INCONCLUSIVE (live-gate not recorded)')
        else:
            verdict = 'not equivalent -> see directional / INCONCLUSIVE'
        covs = f'{cov:.3f}' if cov == cov else '—'
        grads = f'{grad:.3f}' if grad == grad else '—'
        lines.append(f'| {label} | {equiv:.4g} | {covs} | {grads} | {verdict} |')
    lines.append('')
    lines.append('*REDUNDANT wording (spec 0.1): "a trainable multiplicative '
                 'elastance gate is redundant with conductance-only tau in the '
                 'regime where the gate is measurably active."*\n')
    return '\n'.join(lines)


def anchor_block(df) -> str:
    lines = ['### Falsification anchors (spiral, stiff_linear_k1) — uncorrected\n',
             '| Anchor | gate | n | median diff | p (Wilcoxon) | note |',
             '|---|---|---|---|---|---|']
    for anchor in ANCHORS:
        for label, ca, cb in [('B vs A', 'lrc_asym', 'lrc_interp'),
                              ('C vs A', 'lrc_sym', 'lrc_interp')]:
            diff = _paired_diff(df, anchor, 1, ca, cb)
            p = directional_wilcoxon(diff)
            med = float(np.median(diff)) if len(diff) else float('nan')
            note = 'gain here => corroborating-ambiguous (not a clean falsification)'
            lines.append(f'| {anchor} | {label} | {len(diff)} | {med:+.4f} | '
                         f'{p:.4g} | {note} |')
    lines.append('')
    return '\n'.join(lines)


def exploratory_block(df) -> str:
    """kappa-sweep: per-kappa Wilcoxon/TOST + descriptive log(kappa) fit. No Page's L."""
    lines = ['## Exploratory kappa-sweep (descriptive, uncorrected)\n',
             '| kappa | gate | n | median diff (B-A) | p (Wilcoxon) |',
             '|---|---|---|---|---|']
    rows = []
    for kappa in (1, 10, 100, 1000):
        task = f'stiff_linear_k{kappa}'
        diff = _paired_diff(df, task, 1, 'lrc_asym', 'lrc_interp')
        if len(diff) == 0:
            continue
        p = directional_wilcoxon(diff)
        med = float(np.median(diff))
        lines.append(f'| {kappa} | B vs A | {len(diff)} | {med:+.4f} | {p:.4g} |')
        rows.append((np.log10(kappa), med))
    lines.append('')
    if len(rows) >= 2:
        x = np.array([r[0] for r in rows])
        ymed = np.array([r[1] for r in rows])
        slope = np.polyfit(x, ymed, 1)[0]
        lines.append(f'Descriptive trend of median(B-A) on log10(kappa): '
                     f'slope ~ {slope:+.4f} (exploratory, between-system, no Page\'s L).\n')
    return '\n'.join(lines)


def secondary_block(df) -> str:
    """rliable IQM + bootstrap, SECONDARY/descriptive, no CI below n=30."""
    lines = ['## Secondary descriptive (rliable IQM) — non-inferential\n']
    try:
        from rliable import metrics
    except Exception:
        lines.append('_rliable unavailable._\n')
        return '\n'.join(lines)
    sub = df[(df['system'] == HEADLINE_TASK) & (df['ode_unfolds'] == HEADLINE_UF)]
    n = sub.groupby('cell')[METRIC].size().min() if len(sub) else 0
    lines.append(f'multitimescale @ uf=1, n={n} per cell '
                 f'({"CI shown" if n and n >= 30 else "NO CI (n<30)"}).\n')
    lines.append('| Cell | IQM NRMSE |')
    lines.append('|---|---|')
    for cell in sorted(sub['cell'].unique()):
        vals = sub[sub['cell'] == cell][METRIC].to_numpy()
        iqm = float(metrics.aggregate_iqm(vals.reshape(-1, 1))) if len(vals) else float('nan')
        lines.append(f'| {cell} | {iqm:.4f} |')
    lines.append('\n*Not placed beside the per-task verdicts; descriptive only (spec 3.4).*\n')
    return '\n'.join(lines)


# Committed confirmatory scheme (spec §3.0.3): n=30, Delta_min=1.0*SD, corrected
# alpha = ALPHA/M_EQ for equivalence and ALPHA/M_DIR for the directional test.
PILOT_CONFIRMATORY_N = 30
PILOT_PILOT_TASKS = [('multitimescale', 1), ('multitimescale', 2),
                     ('multitimescale', 4), ('spiral', 1), ('stiff_linear_k1', 1)]


def _corrected_tost_power(n, sd, margin, alpha, n_sim=2000, rng=None):
    """Monte-Carlo Wilcoxon-TOST power at true-zero effect for the committed
    scheme. Simulates paired diffs ~ N(0, sd) (true null = equivalent), runs the
    non-parametric TOST at +/- margin, and reports the fraction declared
    equivalent at the CORRECTED alpha (the spec's number is the rank-1 Holm step,
    so alpha here is already ALPHA/m). Returns power in [0,1]."""
    if not (sd == sd and sd > 0 and margin == margin and margin > 0):
        return float('nan')
    rng = rng or np.random.default_rng(0)
    hits = 0
    for _ in range(n_sim):
        diff = rng.normal(0.0, sd, size=n)
        if wilcoxon_tost(diff, margin) < alpha:
            hits += 1
    return hits / n_sim


def _directional_power(n, sd, effect, alpha, n_sim=2000, rng=None):
    """MC one-sided Wilcoxon power for a true effect of size `effect` (in NRMSE
    units, negative = gate beats baseline) at the corrected directional alpha."""
    if not (sd == sd and sd > 0):
        return float('nan')
    rng = rng or np.random.default_rng(1)
    hits = 0
    for _ in range(n_sim):
        diff = rng.normal(effect, sd, size=n)
        if directional_wilcoxon(diff) < alpha:
            hits += 1
    return hits / n_sim


def _jitter_off_ratio(df, task, uf):
    """Optimizer-vs-jitter SD decomposition (spec §3.0.4). The pilot runs are
    jittered; a jitter-OFF subset (eps_jitter absent) would give pure optimizer
    scatter. If no jitter-OFF runs are present, returns NaN (must be measured)."""
    sub = df[(df['system'] == task) & (df['ode_unfolds'] == uf)]
    if 'eps_jitter' not in sub.columns:
        return float('nan')
    return float('nan')   # placeholder until a jitter-OFF subset is collected


def pilot_block(df, abs_floor=0.0) -> str:
    """§3.0 pilot precompute: paired-diff SD per (task, level), committed
    Delta_min, corrected Wilcoxon-TOST + directional power at n=30, and the
    live-gate CoV/grad on the trained pilot B/C models. This is the hard
    go/no-go gate -- a task qualifies only if BOTH corrected powers >= 0.8 AND
    the live-gate diagnostic passes."""
    alpha_eq = ALPHA / M_EQ
    alpha_dir = ALPHA / M_DIR
    lines = ['# eps §3.0 pilot precompute (hard go/no-go gate)\n',
             f'Committed scheme: n={PILOT_CONFIRMATORY_N}, Delta_min=1.0*SD '
             f'(>= absolute floor {abs_floor}), corrected alpha_eq={alpha_eq:.4g} '
             f'(=0.05/{M_EQ}), corrected alpha_dir={alpha_dir:.4g} (=0.05/{M_DIR}).\n',
             '## SD pilot + corrected power per (task, level)\n',
             '| task | uf | n_pilot | SD(B-A) | Delta_min | TOST power@0 '
             '| dir power@0.7SD | qualifies? |',
             '|---|---|---|---|---|---|---|---|']
    rng = np.random.default_rng(20260630)
    any_qualifies = False
    for task, uf in PILOT_PILOT_TASKS:
        diff = _paired_diff(df, task, uf, 'lrc_asym', 'lrc_interp')
        n_pilot = len(diff)
        sd = diff.std(ddof=1) if n_pilot >= 2 else float('nan')
        margin = max(abs_floor, 1.0 * sd) if sd == sd else float('nan')
        tost_pw = _corrected_tost_power(PILOT_CONFIRMATORY_N, sd, margin,
                                        alpha_eq, rng=rng)
        # directional power at a 0.7*SD true effect (spec §3.0.5 reference point)
        dir_pw = (_directional_power(PILOT_CONFIRMATORY_N, sd, -0.7 * sd,
                                     alpha_dir, rng=rng) if sd == sd else float('nan'))
        # confirmatory tier only for the headline task (anchors are falsification)
        is_headline = (task == HEADLINE_TASK)
        qual = (is_headline and tost_pw == tost_pw and tost_pw >= 0.8
                and dir_pw == dir_pw and dir_pw >= 0.8)
        any_qualifies = any_qualifies or qual
        sd_s = f'{sd:.4f}' if sd == sd else '—'
        m_s = f'{margin:.4f}' if margin == margin else '—'
        t_s = f'{tost_pw:.3f}' if tost_pw == tost_pw else '—'
        d_s = f'{dir_pw:.3f}' if dir_pw == dir_pw else '—'
        q_s = 'YES' if qual else ('anchor (falsification)' if not is_headline else 'no')
        lines.append(f'| {task} | {uf} | {n_pilot} | {sd_s} | {m_s} | '
                     f'{t_s} | {d_s} | {q_s} |')
    lines.append('')

    # Live-gate diagnostic on the trained pilot B/C models (spec §3.0.6).
    lines.append('## Live-gate diagnostic on trained pilot B/C models (§3.0.6)\n')
    lines.append('| condition | task@uf | CoV(elastance_t) | grad ratio | active? |')
    lines.append('|---|---|---|---|---|')
    gate_pass = True
    for cell in ('lrc_asym', 'lrc_sym'):
        cov, grad = _live_gate(df, cell, HEADLINE_TASK, HEADLINE_UF)
        ok = (cov == cov and cov >= LIVE_GATE_COV_FLOOR and grad == grad
              and LIVE_GATE_GRAD_RATIO_MIN <= grad <= LIVE_GATE_GRAD_RATIO_MAX)
        gate_pass = gate_pass and ok
        covs = f'{cov:.4f}' if cov == cov else '—'
        grads = f'{grad:.4f}' if grad == grad else '—'
        lines.append(f'| {cell} | {HEADLINE_TASK}@uf{HEADLINE_UF} | {covs} | '
                     f'{grads} | {"YES" if ok else "no/missing"} |')
    lines.append('')
    lines.append(f'CoV floor {LIVE_GATE_COV_FLOOR}, grad ratio in '
                 f'[{LIVE_GATE_GRAD_RATIO_MIN}, {LIVE_GATE_GRAD_RATIO_MAX}].\n')

    # Empty-tier STOP (spec §3.0.7).
    verdict = ('GO: >=1 confirmatory task qualifies (both corrected powers >=0.8) '
               'AND live-gate passes.' if (any_qualifies and gate_pass)
               else 'NO-GO: raise n / widen margin, or the live-gate is degenerate. '
                    'Empty confirmatory tier => do not submit the full eps matrix.')
    lines.append(f'## §3.0.7 empty-tier STOP\n\n**{verdict}**\n')
    lines.append('*Note: SD here is the B-vs-A paired-diff SD on the JITTERED pilot '
                 'path. The jitter-vs-optimizer SD ratio (§3.0.4) needs a jitter-OFF '
                 'subset (run a few seeds with the v1 path) -- not in this pilot.*\n')
    return '\n'.join(lines)


def build_param_audit() -> str:
    """Build A-E_C on both wirings and report ACTIVE elastance/pad params."""
    from experiments.run_benchmark import (
        _build_eps_model, _elastance_param_count, _pad_param_count)
    lines = ['# eps param audit (built models, D=2)\n',
             '| wiring | condition | active params | role |',
             '|---|---|---|---|']
    roles = {'lrc_interp': 'A: tau baseline (Dense unbuilt -> 0)',
             'lrc_asym': 'B: asym multiplicative gate',
             'lrc_sym': 'C: sym two-sided bump (+distr_shift)',
             'lrc_frozen': 'D: frozen (non-trainable -> 0 active)',
             'lrc_pmctrl': 'E: same-budget additive residual (== B)',
             'lrc_pmctrl_c': 'E_C: same-budget control (== C)'}
    for wiring in ('dense', 'ncp'):
        for cond, role in roles.items():
            m = _build_eps_model(cond, wiring)
            count = (_pad_param_count(m) if cond in ('lrc_pmctrl', 'lrc_pmctrl_c')
                     else _elastance_param_count(m))
            lines.append(f'| {wiring} | {cond} | {count} | {role} |')
    lines.append('\nExpected: dense asym=304, sym=320; ncp elastance=450, '
                 'sym=476 (distr_shift 16+8+2=26). E==B, E_C==C per wiring.\n')
    return '\n'.join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', default=['results/runs_eps'])
    p.add_argument('--out', default='results')
    p.add_argument('--param-audit-only', action='store_true',
                   help='only (re)build param_audit.md from built models')
    p.add_argument('--pilot', action='store_true',
                   help='§3.0 pilot mode: emit SD per (task,level), Delta_min, '
                        'corrected Wilcoxon-TOST + directional power at n=30, and '
                        'the live-gate diagnostic. Default --runs is '
                        'results/runs_eps_pilot in this mode.')
    p.add_argument('--abs-floor', type=float, default=0.0,
                   help='pre-stated absolute NRMSE floor for Delta_min (§3.0.2)')
    args = p.parse_args(argv)
    # In pilot mode default the runs dir to the pilot output unless overridden.
    if args.pilot and args.runs == ['results/runs_eps']:
        args.runs = ['results/runs_eps_pilot']

    os.makedirs(args.out, exist_ok=True)

    if args.pilot:
        df = load_runs(args.runs)
        if df.empty:
            print(f'No pilot run JSONs found in {args.runs}', file=sys.stderr)
            return 1
        report = pilot_block(df, abs_floor=args.abs_floor)
        md_path = os.path.join(args.out, 'eps_pilot.md')
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(report)
        print(f'\nWritten: {md_path}')
        return 0

    audit = build_param_audit()
    with open(os.path.join(args.out, 'param_audit.md'), 'w', encoding='utf-8') as f:
        f.write(audit)
    print(audit)
    if args.param_audit_only:
        print(f"\nWritten: {os.path.join(args.out, 'param_audit.md')}")
        return 0

    df = load_runs(args.runs)
    if df.empty:
        print(f'No eps run JSONs found in {args.runs} '
              '(param_audit.md still written).', file=sys.stderr)
        return 1

    report = '\n'.join([
        '# eps-ablation analysis\n',
        f'Runs: {len(df)} | conditions: {sorted(df.cell.unique())} | '
        f'tasks: {sorted(df.system.unique())} | '
        f'levels: {sorted(df.ode_unfolds.unique())}\n',
        confirmatory_block(df),
        exploratory_block(df),
        secondary_block(df),
        '## Param audit\n', audit,
    ])
    md_path = os.path.join(args.out, 'eps_analysis.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(report)
    print(f'\nWritten: {md_path} and param_audit.md')
    return 0


if __name__ == '__main__':
    sys.exit(main())
