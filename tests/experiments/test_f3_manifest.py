# tests/experiments/test_f3_manifest.py
# F3 campaign YAMLs (issue a2ngerer/LRC#156): configs/suite/f3_*.yaml pinned
# against the tables of the issue. Fit counts, seed lists, arm matrix, line
# flags, fixed recipes and the pass-through of stub selections, hpo/final
# tails, arm labels (with 1-epoch runner fits on stub data), the diagnostics
# pairs, array tasks and submit waves, the hpo split_fold lock of load(), the
# fallbacks, the initial weights under both optimizers and the two RMSprop
# update rules. Stub data and stub selections only; F1 selection values are
# not pinned (they live outside the repository), only their pass-through.
import fnmatch
import functools
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf
import yaml

import src.tasks.person_activity as pa_pkg
from experiments import aggregate_results as agg
from experiments import run_committee_benchmark as co_runner
from experiments import run_grid_hpo
from experiments import run_lotka_volterra_benchmark as lv_runner
from experiments import run_person_activity_benchmark as pa_runner
from experiments.suite_manifest import (_eval_jobs, _flag_value, _source_path,
                                        arms, count, expand, load, main,
                                        resources, write)
from src.benchmark.optimizers import make_optimizer
from src.benchmark.registry import CELL_REGISTRY
from tests.benchmark.test_checkpoints import _lv_argv, _pa_stub

SUITE = Path(__file__).resolve().parents[2] / 'configs' / 'suite'
PA, LV, DU, CO = 'pa', 'periodic_predator_prey', 'duffing', 'co'
F1_EPOCHS = {PA: 50, LV: 300, DU: 450, CO: 60}
BUDGET = {PA: 30000, LV: 4000, DU: 4000, CO: 30000}
DET, DIAG, LEGACY = '--deterministic', '--diagnostics', 'rmsprop_legacy'


def S(a, b):
    return list(range(a, b + 1))


def A(*specs):
    """[(arm name, 'cell/wiring')]: 'cell/wiring' names the arm cell_wiring,
    'name=cell/wiring' a named arm."""
    out = []
    for s in specs:
        name, _, cw = s.rpartition('=')
        out.append((name or cw.replace('/', '_'), cw))
    return out


FIX = ('fixed', 0.001, None)
A3 = A('lrc/ncp', 'lrc_cal/ncp', 'lrc_init/ncp')
FOLDCHECK = A(*[f'{c}_ncp_fold{k}={c}/ncp' for k in range(5)
                for c in ('lrc', 'lrc_cal')])
CO3 = A('ncp_cons=cfc_lrc/ncp', 'avg_in=cfc_lrc/ncp', 'fault_mean=cfc_lrc/ncp')


def _fold(k):
    return A(*[f'{c}_{w}_fold{k}={c}/{w}' for c, w in (
        ('gru', 'ncp'), ('cfc_lrc', 'ncp'), ('cfc_lrc', 'dense3'))])


def F1(task):
    return ('hpo_dir', f'<selection copies>/{task}_hpo')


def F3(hpo_stem):
    return ('hpo_dir', f"<{hpo_stem.replace('_', '-')} run>")


HPO = ('hpo',)
# stem: (stage, task, arms, seeds, last --epochs, lr source, fits, array tasks)
PRODUCTION = {
    'f3_a_rms_s10': ('final', PA, A3, S(10, 19), 50, FIX, 30, 30),
    'f3_a_rms_s20': ('final', PA, A3, S(20, 29), 50, FIX, 30, 30),
    'f3_a_legacy_s10': ('final', PA, A3, S(10, 19), 50, FIX, 30, 30),
    'f3_a_legacy_s20': ('final', PA, A3, S(20, 29), 50, FIX, 30, 30),
    'f3_a_budget': ('final', PA, A('lrc/ncp'), S(10, 14), 200,
                    ('fixed', 0.01, None), 5, 5),
    'f3_a_script': ('final', PA, A('lrc/ncp'), S(10, 14), 200, FIX, 5, 5),
    'f3_a_dose': ('final', PA, A('lrc_dose050/ncp', 'lrc_dose075/ncp',
                                 'lrc_dose0875/ncp'), S(10, 19), 50, FIX, 30, 30),
    'f3_a_comp': ('final', PA, A('lrc_fanin/ncp', 'lrc_ranges/ncp'), S(10, 19),
                  50, FIX, 20, 20),
    'f3_e_foldcheck': ('final', PA, FOLDCHECK, [10, 11], 50, FIX, 20, 20),
    'f3_b_co_k30': ('final', CO, CO3, S(10, 29), 60, F1('co_k3'), 60, 20),
    'f3_c_lv': ('final', LV, A('cfc_lrc/ncp', 'gru/ncp', 'lstm/ncp'), S(10, 29),
                300, F1('lv'), 60, 12),
    'f3_d_hpo_eps1': ('hpo', PA, A('cfc_lrc_eps1/ncp'), [100, 101], 50, HPO,
                      11, 1),
    'f3_d_pa_inner_outer': ('final', PA, A('cfc_lrc/ncp', 'cfc_lrc_outer/ncp'),
                            S(10, 29), 50, F1('pa'), 40, 14),
    'f3_d_pa_eps1': ('final', PA, A('cfc_lrc_eps1/ncp'), S(10, 29), 50,
                     F3('f3_d_hpo_eps1'), 20, 7),
    **{f'f3_e_pa_fold{k}': ('final', PA, _fold(k), S(10 + 5 * k, 14 + 5 * k),
                            50, F1('pa'), 15, 5) for k in range(5)},
    'f3_f_hpo_duffing_init': ('hpo', DU, A('lrc_init/ncp',
                                           'lrc_cal_duffing/ncp'),
                              [100, 101], 450, HPO, 22, 2),
    'f3_f_duffing_lrc': ('final', DU, A('lrc/ncp'), S(10, 19), 450,
                         F1('duffing'), 10, 4),
    'f3_f_duffing_init': ('final', DU, A('lrc_init/ncp'), S(10, 19), 450,
                          F3('f3_f_hpo_duffing_init'), 10, 4),
    'f3_f_duffing_cal': ('final', DU, A('lrc_cal_duffing/ncp'), S(10, 19), 450,
                         F3('f3_f_hpo_duffing_init'), 10, 4),
    'f3_g_hpo_lv': ('hpo', LV, A('cfc_lrc/cncp_ff'), [100, 101], 300, HPO,
                    11, 1),
    'f3_g_hpo_pa': ('hpo', PA, A('cfc_lrc/cncp_ff'), [100, 101], 50, HPO,
                    11, 1),
    'f3_g_lv_cncp': ('final', LV, A('cfc_lrc/cncp'), S(10, 19), 300, F1('lv'),
                     10, 2),
    'f3_g_pa_cncp': ('final', PA, A('cfc_lrc/cncp'), S(10, 19), 50, F1('pa'),
                     10, 4),
    'f3_g_lv_cncp_ff': ('final', LV, A('cfc_lrc/cncp_ff'), S(10, 19), 300,
                        F3('f3_g_hpo_lv'), 10, 2),
    'f3_g_pa_cncp_ff': ('final', PA, A('cfc_lrc/cncp_ff'), S(10, 19), 50,
                        F3('f3_g_hpo_pa'), 10, 4),
}
# fallbacks replace their regular final: stem -> (regular final, the
# neighbour arm of selection_from)
FALLBACKS = {'f3_d_pa_eps1_fb': ('f3_d_pa_eps1', 'cfc_lrc/ncp'),
             'f3_f_duffing_init_fb': ('f3_f_duffing_init', 'lrc/ncp'),
             'f3_f_duffing_cal_fb': ('f3_f_duffing_cal', 'lrc/ncp'),
             'f3_g_lv_cncp_ff_fb': ('f3_g_lv_cncp_ff', 'cfc_lrc/cncp'),
             'f3_g_pa_cncp_ff_fb': ('f3_g_pa_cncp_ff', 'cfc_lrc/cncp')}
CONDITIONAL = {
    'f3_a_diag_side_rms': ('final', PA, A3, S(10, 12), 50, FIX, 9, 9),
    'f3_a_diag_side_legacy': ('final', PA, A3, S(10, 12), 50, FIX, 9, 9),
    **{fb: (*PRODUCTION[reg][:5], F1({PA: 'pa', DU: 'duffing', LV: 'lv'}[
        PRODUCTION[reg][1]]), *PRODUCTION[reg][6:])
       for fb, (reg, _) in FALLBACKS.items()},
}
# stem: (task, source config, arms, seeds, lines, array tasks)
LAMBDAS = [0, 0.25, 0.5, 0.75, 1]
EVAL = {
    'f3_x_lambda_f1': (CO, 'co_k30_final', ['ncp_cons', 'cncp_cons'], S(0, 9),
                       20, 2),
    'f3_x_lambda_f3b': (CO, 'f3_b_co_k30', ['ncp_cons'], S(10, 29), 20, 2),
}
SW1 = {
    'f3_smoke_pa': ('final', PA, A(
        'lrc/ncp', 'lrc_cal/ncp', 'lrc_init/ncp', 'lrc_dose050/ncp',
        'lrc_dose075/ncp', 'lrc_dose0875/ncp', 'lrc_fanin/ncp',
        'lrc_ranges/ncp', 'cfc_lrc_eps1/ncp', 'cfc_lrc_outer/ncp',
        'cfc_lrc/cncp_ff'), [0], 3, FIX, 11, 5),
    'f3_smoke_pa_legacy': ('final', PA, A3, [0], 3, FIX, 3, 2),
    'f3_smoke_pa_diag': ('final', PA, A3[:2], [0], 3, FIX, 2, 1),
    'f3_smoke_pa_legacy_diag': ('final', PA, A3[:2], [0], 3, FIX, 2, 1),
    'f3_time_pa_solo': ('final', PA, A('lrc/ncp'), [0], 10, FIX, 1, 1),
    'f3_time_pa_pair': ('final', PA, A('lrc/ncp'), [0, 1], 10, FIX, 2, 1),
    'f3_smoke_pa_fold': ('final', PA, A('cfc_lrc_ncp_fold1=cfc_lrc/ncp',
                                        'lrc_cal_ncp_fold2=lrc_cal/ncp'),
                         [0], 3, FIX, 2, 2),
    'f3_smoke_lv': ('final', LV, A('lstm/ncp', 'cfc_lrc/cncp_ff'), [0], 3,
                    FIX, 2, 1),
    'f3_smoke_co': ('final', CO, CO3, [1], 3, FIX, 3, 1),
    'f3_guard_pa': ('final', PA, A('lrc/ncp'), [0], 50,
                    ('fixed', 1e-4, None), 1, 1),
    'f3_guard_lv': ('final', LV, A('cfc_lrc/ncp', 'gru/ncp', 'lstm/ncp'), [0],
                    300, F1('lv'), 3, 1),
    'f3_guard_pa_s100': ('final', PA, A('cfc_lrc/ncp', 'cfc_lrc_outer/ncp'),
                         [100], 50, F1('pa'), 2, 1),
    'f3_guard_pa_s0': ('final', PA, A('gru/ncp', 'cfc_lrc/dense3'), [0], 50,
                       F1('pa'), 2, 1),
    'f3_guard_duffing': ('final', DU, A('lrc/ncp'), [0], 450, F1('duffing'),
                         1, 1),
    # no epochs key: the selection's final_epochs apply (test 3)
    'f3_guard_co': ('final', CO, CO3, [0], None, F1('co_k3'), 3, 1),
    'f3_guard_pa_cal': ('final', PA, A('lrc_cal/ncp'), [0], 50,
                        ('fixed', 0.01, None), 1, 1),
    'f3_guard_g_lv': ('final', LV, A('cfc_lrc/cncp'), [0], 300, F1('lv'), 1,
                      1),
    'f3_guard_g_pa': ('final', PA, A('cfc_lrc/cncp'), [0], 50, F1('pa'), 1, 1),
    'f3_diag_pa_cal50': ('final', PA, A('lrc_cal/ncp'), [0], 50, FIX, 1, 1),
    'f3_diag_pa_cal50_d': ('final', PA, A('lrc_cal/ncp'), [0], 50, FIX, 1, 1),
}
SW1_EVAL = {'f3_x_lambda_probe': (CO, 'co_k30_final', ['ncp_cons'], [0], 1,
                                  1)}
TABLES = {**PRODUCTION, **CONDITIONAL, **SW1}
EVALS = {**EVAL, **SW1_EVAL}
ALL = sorted([*TABLES, *EVALS])
SMOKE_OR_TIME = {n for n in SW1 if n.startswith(('f3_smoke_', 'f3_time_'))}
DIAG_YAMLS = {n for n in TABLES if n.startswith(('f3_a_', 'f3_e_foldcheck',
                                                 'f3_smoke_pa_diag',
                                                 'f3_smoke_pa_legacy_diag',
                                                 'f3_diag_pa_cal50_d'))}
LEGACY_YAMLS = {'f3_a_legacy_s10', 'f3_a_legacy_s20', 'f3_a_script',
                'f3_a_diag_side_legacy', 'f3_smoke_pa_legacy',
                'f3_smoke_pa_legacy_diag'}
# the reference hpo YAML whose runner_args a stub selection carries (test 5)
REF_HPO = {
    **{n: 'pa_hpo' for n in (
        'f3_d_pa_inner_outer', 'f3_g_pa_cncp', 'f3_guard_pa_s100',
        'f3_guard_pa_s0', 'f3_guard_g_pa', 'f3_d_pa_eps1_fb',
        'f3_g_pa_cncp_ff_fb', *(f'f3_e_pa_fold{k}' for k in range(5)))},
    **{n: 'lv_hpo' for n in ('f3_c_lv', 'f3_g_lv_cncp', 'f3_guard_lv',
                             'f3_guard_g_lv', 'f3_g_lv_cncp_ff_fb')},
    **{n: 'duffing_hpo' for n in ('f3_f_duffing_lrc', 'f3_guard_duffing',
                                  'f3_f_duffing_init_fb',
                                  'f3_f_duffing_cal_fb')},
    'f3_d_pa_eps1': 'f3_d_hpo_eps1', 'f3_f_duffing_init': 'f3_f_hpo_duffing_init',
    'f3_f_duffing_cal': 'f3_f_hpo_duffing_init',
    'f3_g_lv_cncp_ff': 'f3_g_hpo_lv', 'f3_g_pa_cncp_ff': 'f3_g_hpo_pa',
    'f3_b_co_k30': 'co_k3_hpo', 'f3_guard_co': 'co_k3_hpo',
}
# committee selections: final_epochs per reference arm (test 3)
CO_FINAL_EPOCHS = {'ncp_cons': 60, 'avg_in': 60, 'fault_mean': 40}
LRC_NCP = {'lrc', 'lrc_cal', 'lrc_init', 'lrc_dose050', 'lrc_dose075',
           'lrc_dose0875', 'lrc_fanin', 'lrc_ranges'}

# Expected labels (issue #156, "Expected labels"; the source of test 7).
# task -> label -> the YAMLs listed for it.
_E = [f'f3_e_pa_fold{k}' for k in range(5)]
EXPECTED_LABELS = {
    PA: {
        'lrc/ncp': ['f3_a_rms_s10', 'f3_a_rms_s20', 'f3_e_foldcheck',
                    'f3_a_diag_side_rms', 'f3_guard_pa', 'f3_smoke_pa',
                    'f3_smoke_pa_diag', 'f3_time_pa_solo', 'f3_time_pa_pair'],
        'lrc_cal/ncp': ['f3_a_rms_s10', 'f3_a_rms_s20', 'f3_e_foldcheck',
                        'f3_a_diag_side_rms', 'f3_guard_pa_cal',
                        'f3_diag_pa_cal50', 'f3_diag_pa_cal50_d', 'f3_smoke_pa',
                        'f3_smoke_pa_diag', 'f3_smoke_pa_fold'],
        'lrc_init/ncp': ['f3_a_rms_s10', 'f3_a_rms_s20', 'f3_a_diag_side_rms',
                         'f3_smoke_pa'],
        'lrc/ncp [optimizer=rmsprop_legacy]': [
            'f3_a_legacy_s10', 'f3_a_legacy_s20', 'f3_a_diag_side_legacy',
            'f3_smoke_pa_legacy', 'f3_smoke_pa_legacy_diag'],
        'lrc_cal/ncp [optimizer=rmsprop_legacy]': [
            'f3_a_legacy_s10', 'f3_a_legacy_s20', 'f3_a_diag_side_legacy',
            'f3_smoke_pa_legacy', 'f3_smoke_pa_legacy_diag'],
        'lrc_init/ncp [optimizer=rmsprop_legacy]': [
            'f3_a_legacy_s10', 'f3_a_legacy_s20', 'f3_a_diag_side_legacy',
            'f3_smoke_pa_legacy'],
        'lrc/ncp [epochs=200]': ['f3_a_budget'],
        'lrc/ncp [epochs=200, optimizer=rmsprop_legacy]': ['f3_a_script'],
        **{f'lrc_dose{d}/ncp': ['f3_a_dose', 'f3_smoke_pa']
           for d in ('050', '075', '0875')},
        'lrc_fanin/ncp': ['f3_a_comp', 'f3_smoke_pa'],
        'lrc_ranges/ncp': ['f3_a_comp', 'f3_smoke_pa'],
        'cfc_lrc/ncp': ['f3_d_pa_inner_outer', *_E, 'f3_guard_pa_s100',
                        'f3_smoke_pa_fold'],
        'cfc_lrc_outer/ncp': ['f3_d_pa_inner_outer', 'f3_guard_pa_s100',
                              'f3_smoke_pa'],
        'cfc_lrc_eps1/ncp': ['f3_d_hpo_eps1', 'f3_d_pa_eps1', 'f3_d_pa_eps1_fb',
                             'f3_smoke_pa'],
        'gru/ncp': [*_E, 'f3_guard_pa_s0'],
        'cfc_lrc/dense3': [*_E, 'f3_guard_pa_s0'],
        'cfc_lrc/cncp': ['f3_g_pa_cncp', 'f3_guard_g_pa'],
        'cfc_lrc/cncp_ff': ['f3_g_hpo_pa', 'f3_g_pa_cncp_ff',
                            'f3_g_pa_cncp_ff_fb', 'f3_smoke_pa'],
    },
    LV: {
        'cfc_lrc/ncp': ['f3_c_lv', 'f3_guard_lv'],
        'gru/ncp': ['f3_c_lv', 'f3_guard_lv'],
        'lstm/ncp': ['f3_c_lv', 'f3_guard_lv', 'f3_smoke_lv'],
        'cfc_lrc/cncp': ['f3_g_lv_cncp', 'f3_guard_g_lv'],
        'cfc_lrc/cncp_ff': ['f3_g_hpo_lv', 'f3_g_lv_cncp_ff',
                            'f3_g_lv_cncp_ff_fb', 'f3_smoke_lv'],
    },
    DU: {
        'lrc/ncp': ['f3_f_duffing_lrc', 'f3_guard_duffing'],
        'lrc_init/ncp': ['f3_f_hpo_duffing_init', 'f3_f_duffing_init',
                         'f3_f_duffing_init_fb'],
        'lrc_cal_duffing/ncp': ['f3_f_hpo_duffing_init', 'f3_f_duffing_cal',
                                'f3_f_duffing_cal_fb'],
    },
    CO: {
        ('cfc_lrc', 'ncp', 'cons'): ['f3_b_co_k30', 'f3_guard_co', 'f3_smoke_co',
                                     'f3_x_lambda_f1', 'f3_x_lambda_f3b',
                                     'f3_x_lambda_probe'],
        ('cfc_lrc', 'ncp', 'avg_in'): ['f3_b_co_k30', 'f3_guard_co',
                                       'f3_smoke_co'],
        ('cfc_lrc', 'ncp', 'fault_mean'): ['f3_b_co_k30', 'f3_guard_co',
                                           'f3_smoke_co'],
        # F1 records only (f3_x_lambda_f1), no F3 fit
        ('cfc_lrc', 'cncp', 'cons'): ['f3_x_lambda_f1'],
    },
}
NO_F3_FIT = {(CO, ('cfc_lrc', 'cncp', 'cons'))}
# Submit waves (cluster/README.md, F3 wave table): part -> {YAML: tasks};
# 'X2' is the eval_x job of cluster/submit_eval_x.sh (one task).
WAVES = {
    ('SW2', 'a'): {'f3_f_hpo_duffing_init': 2, 'f3_d_hpo_eps1': 1,
                   'f3_c_lv': 12, 'f3_d_pa_inner_outer': 14,
                   'f3_b_co_k30': 20},
    ('SW2', 'b'): {'f3_a_legacy_s10': 30},
    ('SW2', 'c'): {'f3_a_rms_s10': 30},
    ('SW3', 'a'): {'f3_d_pa_eps1': 7, 'f3_f_duffing_lrc': 4,
                   'f3_f_duffing_init': 4, 'f3_f_duffing_cal': 4},
    ('SW3', 'b1'): {'f3_a_legacy_s20': 30},
    ('SW3', 'b2'): {'f3_a_rms_s20': 30},
    ('SW3', 'c'): {f'f3_e_pa_fold{k}': 5 for k in range(5)},
    ('SW4', 'a'): {'f3_e_foldcheck': 20, 'f3_a_budget': 5, 'f3_a_script': 5},
    ('SW4', 'b'): {'f3_a_dose': 30, 'f3_a_comp': 20},
    ('SW5', '-'): {'f3_x_lambda_f1': 2, 'f3_x_lambda_f3b': 2, 'X2': 1},
    ('G', '1'): {'f3_g_hpo_lv': 1, 'f3_g_hpo_pa': 1, 'f3_g_lv_cncp': 2,
                 'f3_g_pa_cncp': 4},
    ('G', '2'): {'f3_g_lv_cncp_ff': 2, 'f3_g_pa_cncp_ff': 4},
}


# --- helpers -----------------------------------------------------------------

def _raw(name):
    return load(str(SUITE / f'{name}.yaml'))


def _task(cfg):
    if cfg['task'] == 'lv':
        return DU if cfg.get('system') == 'duffing' else LV
    return cfg['task']


@functools.lru_cache(maxsize=None)
def _runner_args(hpo_stem, arm):
    """runner_args of a real selection of `arm`: what run_grid_hpo.py forwards
    to the runner from the line of the reference hpo YAML (its own parser,
    GridHPO stopped before the first fit)."""
    (line,) = [ln for ln in expand(_raw(hpo_stem)) if ln['id'] == f'hpo_{arm}']
    got = {}

    class Stop(Exception):
        pass

    class Capture:
        def __init__(self, args, runner_args):
            got['runner_args'] = runner_args
            raise Stop

    real = run_grid_hpo.GridHPO
    run_grid_hpo.GridHPO = Capture
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as d, pytest.raises(Stop):
            run_grid_hpo.main([*line['argv'][4:], '--outdir', d])
    finally:
        run_grid_hpo.GridHPO = real
    return tuple(got['runner_args'])


def _ref(cfg, name, cell, wiring):
    r = (cfg.get('selection_from') or {}).get(name)
    return (r['arm'], r['cell'], r['wiring']) if r else (name, cell, wiring)


def _stub(name, hpo_dir):
    """The YAML; with an hpo_dir placeholder, stub selections of the real
    structure for every reference arm (own or selection_from) in hpo_dir and
    the YAML pointed there. Returns (cfg, {reference arm: selection})."""
    cfg = _raw(name)
    if not cfg.get('hpo_dir'):
        return cfg, {}
    sels = {}
    for arm_name, cell, wiring, _ in arms(cfg):
        ref, rcell, rwiring = _ref(cfg, arm_name, cell, wiring)
        i = sum(map(ord, ref))
        sel = {'task': cfg['task'], 'system': cfg.get('system'), 'cell': rcell,
               'wiring': rwiring, 'param_budget': cfg.get('param_budget'),
               'lr': 1e-4 * (i % 97 + 1), 'clipnorm': [None, 1.0][i % 2],
               'tuning_seeds': [100, 101], 'grid_lr': [1e-3], 'ranking': [],
               'runner_args': list(_runner_args(REF_HPO[name], ref))}
        if cfg['task'] == 'co':
            sel['final_epochs'] = CO_FINAL_EPOCHS[ref]
        d = hpo_dir / ref
        d.mkdir(parents=True, exist_ok=True)
        (d / 'hpo_selection.json').write_text(json.dumps(sel))
        sels[ref] = sel
    return {**cfg, 'hpo_dir': str(hpo_dir)}, sels


def _expand(name, tmp_path):
    return expand(_stub(name, tmp_path / name / 'hpo')[0])


def _arm_of(cfg, line):
    """(arm name, cell, wiring, flags) of a non-eval line (by its id)."""
    by = {a[0]: a for a in arms(cfg)}
    key = line['id'][len('hpo_'):] if line['id'].startswith('hpo_') else \
        line['id'].rsplit('_seed', 1)[0]
    return by[key]


def _co_arm(argv):
    va = _flag_value(argv, '--view-average')
    if va == 'mean':
        return 'avg_in'
    if va == 'fault_mean':
        return 'fault_mean'
    assert va is None and _flag_value(argv, '--vote-lambda') is None
    assert _flag_value(argv, '--consensus') in (None, 'mean')
    return 'cons'


def _label(name, task, argv):
    """Expected label of one runner or hpo line (test 7 a)."""
    cell, wiring = _flag_value(argv, '--cell'), _flag_value(argv, '--wiring')
    if task == CO:
        return (cell, wiring, _co_arm(argv))
    tags = []
    ep = _flag_value(argv, '--epochs')
    if name not in SMOKE_OR_TIME and int(ep) != F1_EPOCHS[task]:
        tags.append(f'epochs={ep}')
    opt = _flag_value(argv, '--optimizer')
    if opt not in (None, 'rmsprop'):
        tags.append(f'optimizer={opt}')
    return f'{cell}/{wiring}' + (f" [{', '.join(tags)}]" if tags else '')


def _labels(name, tmp_path):
    """[(task, label)] of every line of a YAML; eval lines through the arms of
    their source config."""
    cfg = _raw(name)
    task = _task(cfg)
    if cfg['stage'] == 'eval':
        out = []
        for src in cfg['sources']:
            scfg = load(_source_path(src['config']))
            for (_, cell, wiring, flags), _seed in _eval_jobs(scfg, src):
                assert 'vote_lambda' not in flags and 'consensus' not in flags
                arm = {'mean': 'avg_in', 'fault_mean': 'fault_mean',
                       None: 'cons'}[flags.get('view_average')]
                out.append((task, (cell, wiring, arm)))
        return out
    return [(task, _label(name, task, ln['argv']))
            for ln in _expand(name, tmp_path)]


def _pairs(argv, drop=()):
    """Flag-value pairs of an argv: a flag with the tokens up to the next flag
    (empty for a switch); tokens before the first flag and flags in `drop`
    are left out."""
    out, cur = [], None
    for t in argv:
        if t.startswith('--'):
            cur = [t]
            out.append(cur)
        elif cur is not None:
            cur.append(t)
    return [(p[0], tuple(p[1:])) for p in out if p[0] not in drop]


def _tasks(cfg, lines, tmp_path):
    plan = dict(row.split('=', 1) for row in write(
        cfg, lines, str(tmp_path)).splitlines())
    groups = plan['SUITE_GROUPS'].strip("'").split()
    return {g: math.ceil(int(plan[f'SUITE_{g}_N'])
                         / int(plan[f'SUITE_{g}_RUNS_PER_GPU']))
            for g in groups}


# --- 1. fit counts --------------------------------------------------------------

def test_f3_files_are_the_tables():
    """59 YAMLs: 29 production, 7 conditional, 2 eval, 21 SW1."""
    assert sorted(p.stem for p in SUITE.glob('f3_*.yaml')) == ALL
    assert (len(PRODUCTION), len(CONDITIONAL), len(EVAL),
            len(SW1) + len(SW1_EVAL)) == (29, 7, 2, 21)


@pytest.mark.parametrize('name', ALL)
def test_1_fit_count_per_yaml(name, tmp_path, capsys):
    main(['--count', str(SUITE / f'{name}.yaml')])
    want = TABLES[name][6] if name in TABLES else EVALS[name][4]
    assert capsys.readouterr().out.strip() == str(want)
    lines = _expand(name, tmp_path)      # the collision guard lives here
    assert len(lines) == (want if name in EVALS or TABLES[name][0] == 'final'
                          else len(TABLES[name][2]))


def test_1_totals(capsys):
    """Production: 580 = 525 final fits + 55 planned hpo fits; block G 62 of
    them; conditional, SW1 and eval YAMLs pinned in lists of their own."""
    main(['--count', *(str(SUITE / f'{n}.yaml') for n in PRODUCTION)])
    assert capsys.readouterr().out.strip() == '580'
    hpo = sum(v[6] for v in PRODUCTION.values() if v[0] == 'hpo')
    assert (hpo, 580 - hpo) == (55, 525)
    assert sum(v[6] for n, v in PRODUCTION.items() if n.startswith('f3_g_')) == 62
    assert [CONDITIONAL[n][6] for n in sorted(CONDITIONAL)] == [
        9, 9, 20, 10, 10, 10, 10]
    guards = [n for n in SW1 if n.startswith('f3_guard_')]
    assert sum(SW1[n][6] for n in guards) == 15
    assert sum(SW1[n][6] for n in SMOKE_OR_TIME) + SW1['f3_smoke_pa_legacy'][6] \
        == 31                                # the legacy smoke is submitted twice
    assert [EVALS[n][4] for n in sorted(EVALS)] == [20, 20, 1]


# --- 2. seeds -------------------------------------------------------------------

def test_2_seed_lists():
    seeds = {n: _raw(n)['seeds'] for n in TABLES}
    for n in ('f3_a_rms_s10', 'f3_a_legacy_s10'):
        assert seeds[n] == S(10, 19)
    for n in ('f3_a_rms_s20', 'f3_a_legacy_s20'):
        assert seeds[n] == S(20, 29)
    for n in ('f3_b_co_k30', 'f3_c_lv', 'f3_d_pa_inner_outer', 'f3_d_pa_eps1'):
        assert seeds[n] == S(10, 29)
    assert seeds['f3_a_budget'] == seeds['f3_a_script'] == S(10, 14)
    assert seeds['f3_a_dose'] == seeds['f3_a_comp'] == S(10, 19)
    assert seeds['f3_e_foldcheck'] == [10, 11]
    for n, v in PRODUCTION.items():
        if n.startswith(('f3_f_', 'f3_g_')):
            assert seeds[n] == ([100, 101] if v[0] == 'hpo' else S(10, 19)), n
        if v[0] == 'hpo':
            assert seeds[n] == [100, 101], n
    # family E: fold k on seeds 10+5k..14+5k, each seed in exactly one fold
    folds = {k: seeds[f'f3_e_pa_fold{k}'] for k in range(5)}
    for k, s in folds.items():
        assert s == S(10 + 5 * k, 14 + 5 * k)
    flat = [s for f in folds.values() for s in f]
    assert sorted(flat) == S(10, 34) and len(set(flat)) == len(flat)
    for n in TABLES:
        assert seeds[n] == TABLES[n][3], n


# --- 3. arm matrix ----------------------------------------------------------------

@pytest.mark.parametrize('name', sorted(TABLES))
def test_3_arm_matrix(name, tmp_path):
    stage, task, arm_list, seeds, epochs, source = TABLES[name][:6]
    raw = yaml.safe_load((SUITE / f'{name}.yaml').read_text())
    assert 'description' not in raw and 'extra_seeds' not in raw
    assert 'config_from' not in raw
    cfg = _raw(name)
    assert (cfg['stage'], _task(cfg), cfg['seeds'], cfg.get('epochs'),
            cfg.get('param_budget')) == (stage, task, seeds,
                                         None if name == 'f3_guard_co' else
                                         epochs, BUDGET[task])
    assert [(a[0], f'{a[1]}/{a[2]}') for a in arms(cfg)] == arm_list
    assert all(a[1] in CELL_REGISTRY for a in arms(cfg))
    if source[0] == 'fixed':
        assert cfg['fixed'] == {'lr': source[1], 'clipnorm': source[2]}
        assert not cfg.get('hpo_dir')
    elif source[0] == 'hpo_dir':
        assert source[1] in cfg['hpo_dir'] and not cfg.get('fixed')
    else:
        assert not cfg.get('hpo_dir') and not cfg.get('fixed')
    lines = _expand(name, tmp_path)
    for line in lines:
        argv = line['argv']
        arm_name = _arm_of(cfg, line)[0]
        fold = _flag_value(argv, '--split-fold')
        assert fold == (arm_name[-1] if '_fold' in arm_name else None)
        if task == CO:
            assert _flag_value(argv, '--view-average') == {
                'avg_in': 'mean', 'fault_mean': 'fault_mean'}.get(arm_name)
            assert _flag_value(argv, '--corruption-seed') == (
                None if name == 'f3_guard_co' else 'seed')
            assert argv.count('--n-columns') == 1
            assert _flag_value(argv, '--n-columns') == '30'
        if epochs is not None:
            assert _flag_value(argv, '--epochs') == str(epochs)
    if name in ('f3_b_co_k30', 'f3_guard_co'):
        # stub selections with final_epochs 60, 60, 40: the campaign's
        # --epochs comes last and wins in f3_b_co_k30; f3_guard_co has none
        for line in lines:
            argv, ref = line['argv'], _arm_of(cfg, line)[0]
            got = [argv[i + 1] for i, t in enumerate(argv) if t == '--epochs']
            assert got == [str(CO_FINAL_EPOCHS[ref]),
                           *(['60'] if name == 'f3_b_co_k30' else [])]
        assert len(lines) == (60 if name == 'f3_b_co_k30' else 3)


# --- 4. line flags ------------------------------------------------------------------

@pytest.mark.parametrize('name', sorted(TABLES))
def test_4_line_flags(name, tmp_path):
    cfg = _raw(name)
    task = _task(cfg)
    for line in _expand(name, tmp_path):
        argv = line['argv']
        assert _flag_value(argv, '--param-budget') == str(BUDGET[task])
        assert argv.count(DET) == 1
        assert '--opt-epsilon' not in argv
        assert (_flag_value(argv, '--optimizer') == LEGACY) == (
            name in LEGACY_YAMLS)
        assert (DIAG in argv) == (name in DIAG_YAMLS)
        if task == CO:
            continue
        assert _flag_value(argv, '--ode-unfolds') == '6'
        assert _flag_value(argv, '--dt-scale') == 'auto'
        if name not in SMOKE_OR_TIME:
            want = 200 if name in ('f3_a_budget', 'f3_a_script') else \
                F1_EPOCHS[task]
            assert _flag_value(argv, '--epochs') == str(want)
        if task in (LV, DU):
            assert (_flag_value(argv, '--forcing'), _flag_value(
                argv, '--anchors'), _flag_value(argv, '--horizon')) == (
                'mixed', '8,16,24,32', '96')
        else:
            assert _flag_value(argv, '--forcing') is None


def test_4_diagnostics_only_on_block_a_and_the_fold_check():
    """The nine block A and fold-check YAMLs carry --diagnostics; no other
    production YAML does."""
    with_diag = {n for n in PRODUCTION if DIAG in _raw(n)['extra_args']}
    assert with_diag == {n for n in PRODUCTION
                         if n.startswith(('f3_a_', 'f3_e_foldcheck'))}
    assert len(with_diag) == 9


# --- 5. fixed recipes and the pass-through of selections ---------------------------

RUN_FLAGS = ('--cell', '--wiring', '--system', '--seed', '--lr', '--clipnorm',
             '--param-budget', '--split-fold', '--wandb')


def test_5_fixed_recipes():
    want = {**{n: (0.001, None) for n in (
        'f3_a_rms_s10', 'f3_a_rms_s20', 'f3_a_legacy_s10', 'f3_a_legacy_s20',
        'f3_a_dose', 'f3_a_comp', 'f3_e_foldcheck', 'f3_a_diag_side_rms',
        'f3_a_diag_side_legacy', 'f3_a_script', 'f3_diag_pa_cal50',
        'f3_diag_pa_cal50_d', *SMOKE_OR_TIME)},
        'f3_a_budget': (0.01, None), 'f3_guard_pa': (1e-4, None),
        'f3_guard_pa_cal': (0.01, None)}
    have = {n: (_raw(n)['fixed']['lr'], _raw(n)['fixed']['clipnorm'])
            for n in TABLES if _raw(n).get('fixed')}
    assert have == want


@pytest.mark.parametrize('name', sorted(
    n for n in TABLES if TABLES[n][5][0] == 'hpo_dir'))
@pytest.mark.parametrize('wandb', [False, True])
def test_5_selections_pass_through(name, wandb, tmp_path):
    """--lr and --clipnorm of every line are those of the stub selection of
    its reference arm; the selection's runner_args (from the reference hpo
    YAML) agree with the line as flag-value sets (committee lines exempt: K,
    epochs and the corruption draw differ from the K = 3 selection by
    design)."""
    cfg, sels = _stub(name, tmp_path / 'hpo')
    for line in expand(cfg, wandb=wandb):
        argv = line['argv']
        ref = _ref(cfg, *_arm_of(cfg, line)[:3])[0]
        sel = sels[ref]
        assert _flag_value(argv, '--lr') == f"{sel['lr']:g}"
        assert _flag_value(argv, '--clipnorm') == (
            None if sel['clipnorm'] is None else f"{sel['clipnorm']:g}")
        if cfg['task'] == 'co':
            continue
        assert set(_pairs(argv, RUN_FLAGS)) == set(
            _pairs(sel['runner_args'], ('--wandb',)))


# --- 6. hpo and final tails -----------------------------------------------------------

HPO_PAIRS = [('f3_d_hpo_eps1', 'f3_d_pa_eps1'),
             ('f3_f_hpo_duffing_init', 'f3_f_duffing_init'),
             ('f3_f_hpo_duffing_init', 'f3_f_duffing_cal'),
             ('f3_g_hpo_lv', 'f3_g_lv_cncp_ff'),
             ('f3_g_hpo_pa', 'f3_g_pa_cncp_ff')]


@pytest.mark.parametrize('hpo,final', HPO_PAIRS)
def test_6_hpo_and_final_tails_agree(hpo, final, tmp_path):
    hpo_lines = {(_flag_value(ln['argv'], '--cell'),
                  _flag_value(ln['argv'], '--wiring')): ln['argv']
                 for ln in _expand(hpo, tmp_path)}
    drop_h = ('--task', '--cell', '--wiring', '--system', '--tuning-seeds')
    drop_f = ('--cell', '--wiring', '--system', '--seed', '--lr', '--clipnorm')
    for line in _expand(final, tmp_path):
        argv = line['argv']
        key = (_flag_value(argv, '--cell'), _flag_value(argv, '--wiring'))
        assert _pairs(argv, drop_f) == _pairs(hpo_lines[key], drop_h)
        assert _flag_value(argv, '--system') == _flag_value(hpo_lines[key],
                                                            '--system')


# --- 7. labels ---------------------------------------------------------------------------

def test_7a_every_line_maps_to_a_listed_label(tmp_path):
    """Every line of the 59 YAMLs maps to a label of the list for its task,
    from a YAML the list names for that label; every label is reached."""
    got = {(task, label, name) for name in ALL
           for task, label in _labels(name, tmp_path)}
    want = {(task, label, name) for task, by in EXPECTED_LABELS.items()
            for label, names in by.items() for name in names}
    assert got == want


def test_7c_registered_tests_name_listed_labels():
    """Every F3_TESTS row that is not optional names labels of the list for
    its task (committee rows by arm name)."""
    for test, t in agg.F3_TESTS.items():
        if t['status'] == 'optional':
            continue
        if t['task'] == CO:
            names = {k[2] for k in EXPECTED_LABELS[CO]} | {
                a[0] for a in arms(_raw('f3_b_co_k30'))}
            assert {t['a'], t['b']} <= names, test
        else:
            assert {t['a'], t['b']} <= set(EXPECTED_LABELS[t['task']]), test


def _pa_stub_dt(*a, **k):
    """_pa_stub with elapsed times of 0.5 (all ones would resolve
    --dt-scale auto to 1.0 and add the dt_scale=1 tag)."""
    d = _pa_stub(*a, **k)
    for split in ('train', 'test', 'val'):
        setattr(d, f'{split}_t', np.full_like(getattr(d, f'{split}_t'), 0.5))
    return d


def _first_final_line(task, label, tmp_path):
    for name in EXPECTED_LABELS[task][label]:
        if name in EVALS or TABLES[name][0] != 'final':
            continue
        for line in _expand(name, tmp_path):
            if _label(name, task, line['argv']) == label:
                return name, line
    raise AssertionError(f'no final line for {label}')


def _strip_epochs(label):
    if isinstance(label, tuple):
        return label
    base, _, tags = label.partition(' [')
    return base, frozenset(t for t in tags.rstrip(']').split(', ')
                           if t and not t.startswith('epochs='))


F3_FIT_LABELS = [(task, label) for task, by in EXPECTED_LABELS.items()
                 for label in by if (task, label) not in NO_F3_FIT]


@pytest.mark.parametrize('task,label', F3_FIT_LABELS,
                         ids=[f'{t}:{lb}' for t, lb in F3_FIT_LABELS])
def test_7b_one_epoch_fit_carries_the_label(task, label, tmp_path, monkeypatch):
    """The first final line listed for the label, run for one epoch on stub
    data with its production flags: arm_label of the record (committee:
    cell, wiring, _co_arm_any) is the label; with the production epochs set,
    also the epochs= tag."""
    name, line = _first_final_line(task, label, tmp_path / 'sel')
    # op determinism is process-global; the record still says deterministic
    monkeypatch.setattr(tf.config.experimental, 'enable_op_determinism',
                        lambda: None)
    out = tmp_path / 'out'
    argv = [line['argv'][3], *line['argv'][4:]]
    if task in (LV, DU):
        argv += _lv_argv(out)[1:]
        runner = lv_runner
    else:
        monkeypatch.setattr(pa_pkg, 'load_person_activity', _pa_stub_dt)
        monkeypatch.setattr(co_runner, 'load_person_activity', _pa_stub_dt)
        runner = co_runner if task == CO else pa_runner
    argv += ['--epochs', '1', '--outdir', str(out)]
    monkeypatch.setattr(sys, 'argv', argv)
    runner.main()
    (path,) = [p for p in out.glob('*.json')
               if not p.name.endswith('.evalonly.json')]
    rec = json.loads(path.read_text())
    assert rec['dt_scale'] != 1
    if task == CO:
        assert (rec['cell'], rec['wiring'], agg._co_arm_any(rec)) == label
        return
    assert _strip_epochs(agg.arm_label(rec)) == _strip_epochs(label)
    rec['config']['effective']['epochs'] = int(_flag_value(
        line['argv'], '--epochs'))
    assert agg.arm_label(rec) == label, name


# --- 8. diagnostics pairs ---------------------------------------------------------------

@pytest.mark.parametrize('rms,legacy', [
    ('f3_a_rms_s10', 'f3_a_legacy_s10'), ('f3_a_rms_s20', 'f3_a_legacy_s20'),
    ('f3_a_diag_side_rms', 'f3_a_diag_side_legacy'),
    ('f3_smoke_pa_diag', 'f3_smoke_pa_legacy_diag')])
def test_8_test_pairs_share_the_diagnostics_flag(rms, legacy, tmp_path):
    """Both campaigns of a pair carry --diagnostics or neither; the lines
    differ by the optimizer only."""
    a, b = _raw(rms), _raw(legacy)
    assert (DIAG in a['extra_args']) == (DIAG in b['extra_args'])
    la, lb = _expand(rms, tmp_path), _expand(legacy, tmp_path)
    assert all(_flag_value(ln['argv'], '--optimizer') == LEGACY for ln in lb)
    assert [[t for t in ln['argv'] if t not in ('--optimizer', LEGACY)]
            for ln in lb] == [ln['argv'] for ln in la]


# --- 9. array tasks and submit waves ----------------------------------------------------

@pytest.mark.parametrize('name', ALL)
def test_9_array_tasks(name, tmp_path):
    cfg = _stub(name, tmp_path / 'hpo')[0]
    assert {'runs_per_gpu', 'walltime'} <= set(cfg['cluster'])
    if cfg.get('heavy'):
        assert 'runs_per_gpu' in cfg['heavy']['cluster']
    lines = expand(cfg)
    want = TABLES[name][7] if name in TABLES else EVALS[name][5]
    assert sum(_tasks(cfg, lines, tmp_path).values()) == want <= 50
    if cfg['task'] != 'pa' or cfg['stage'] == 'eval':
        return
    per_gpu = 2 if name in SMOKE_OR_TIME - {'f3_time_pa_solo'} else 1
    for line in lines:
        cell, wiring = (_flag_value(line['argv'], f) for f in ('--cell',
                                                              '--wiring'))
        if cell in LRC_NCP and wiring == 'ncp':
            assert resources(cfg, line['group'])['runs_per_gpu'] == per_gpu


def test_9_waves(tmp_path):
    """Every jointly submitted part has at most 50 tasks; the waves cover the
    production YAMLs once (307 tasks) and the eval YAMLs."""
    seen = []
    for part, yamls in WAVES.items():
        assert sum(yamls.values()) <= 50, part
        for name, n in yamls.items():
            if name == 'X2':
                continue
            seen.append(name)
            want = TABLES[name][7] if name in TABLES else EVALS[name][5]
            assert n == want, name
    assert sorted(seen) == sorted([*PRODUCTION, *EVAL])
    assert sum(PRODUCTION[n][7] for n in PRODUCTION) == 307
    assert sum(PRODUCTION[n][7] for n in PRODUCTION
               if not n.startswith('f3_g_')) == 293
    # SW1: every SW1 YAML once, the legacy smoke twice, the conditional
    # f3_diag_pa_cal50_d outside: 28 tasks
    sw1 = sum(v[7] for n, v in SW1.items() if n != 'f3_diag_pa_cal50_d')
    assert sw1 + SW1['f3_smoke_pa_legacy'][7] + SW1_EVAL[
        'f3_x_lambda_probe'][5] == 28


# --- 10. load() locks, lambda grids, budgets ----------------------------------------------

@pytest.mark.parametrize('over,msg', [
    ({'cells': None, 'arms': {'x_fold1': {'cell': 'cfc_lrc', 'wiring': 'cncp_ff',
                                          'split_fold': 1}}}, 'arm x_fold1'),
    ({'per_cell': {'cfc_lrc': {'split_fold': 1}}}, 'per_cell cfc_lrc'),
    ({'extra_args': ['--deterministic', '--split-fold', '1']},
     'extra_args carry --split-fold')])
def test_10_hpo_rejects_split_fold(over, msg, tmp_path):
    raw = yaml.safe_load((SUITE / 'f3_g_hpo_pa.yaml').read_text())
    path = tmp_path / 'hpo.yaml'
    path.write_text(yaml.safe_dump({k: v for k, v in {**raw, **over}.items()
                                    if v is not None}))
    with pytest.raises(SystemExit, match=f'default split.*{msg}'):
        load(str(path))


def test_10_lambda_grids_and_budgets():
    for name in EVALS:
        cfg = _raw(name)
        assert cfg['vote_lambda_override'] == LAMBDAS
        for src in cfg['sources']:
            scfg = load(_source_path(src['config']))
            flags = {a[0]: a[3] for a in arms(scfg)}
            assert all('vote_lambda' not in flags[a] for a in src['arms'])
            assert src['records_dir'].startswith('../benchmarks/')
        task, source, arm_names, seeds = EVALS[name][:4]
        (src,) = cfg['sources']
        assert (src['config'], src['arms'], src['seeds']) == (
            f'configs/suite/{source}.yaml', arm_names, seeds)
    for name in TABLES:
        if name.startswith('f3_g_'):
            assert _raw(name)['param_budget'] == BUDGET[_task(_raw(name))]


# --- 11. fallbacks -------------------------------------------------------------------------

@pytest.mark.parametrize('fb', sorted(FALLBACKS))
def test_11_fallbacks(fb):
    regular, neighbour = FALLBACKS[fb]
    a, b = _raw(fb), _raw(regular)
    (arm,) = arms(a)
    cell, wiring = neighbour.split('/')
    assert a['selection_from'] == {arm[0]: {
        'arm': f'{cell}_{wiring}', 'cell': cell, 'wiring': wiring}}
    assert [x[1:] for x in arms(a)] == [x[1:] for x in arms(b)]
    for k in ('task', 'system', 'seeds', 'epochs', 'param_budget',
              'ode_unfolds', 'dt_scale', 'extra_args', 'cluster'):
        assert a.get(k) == b.get(k), k


# --- 12. initial weights ---------------------------------------------------------------------

class _Stop(Exception):
    pass


def _initial_weights(cell, tmp_path, monkeypatch, *extra):
    monkeypatch.setattr(pa_pkg, 'load_person_activity', _pa_stub)
    monkeypatch.setattr(tf.config.experimental, 'enable_op_determinism',
                        lambda: None)
    got = []

    def fit(self, *a, **k):
        got.append(self.get_weights())
        raise _Stop

    monkeypatch.setattr(tf.keras.Model, 'fit', fit)
    monkeypatch.setattr(sys, 'argv', [
        'run_person_activity_benchmark.py', '--wiring', 'ncp', '--cell', cell,
        '--seed', '10', '--param-budget', '30000', '--ode-unfolds', '6',
        '--dt-scale', 'auto', '--deterministic', '--lr', '0.001', '--epochs',
        '50', '--outdir', str(tmp_path), *extra])
    with pytest.raises(_Stop):
        pa_runner.main()
    (weights,) = got
    return weights


@pytest.mark.parametrize('cell', ['lrc', 'lrc_cal'])
def test_12_initial_weights_follow_the_seed_only(cell, tmp_path, monkeypatch):
    """The four builds of an arm (rmsprop and rmsprop_legacy, each with and
    without --diagnostics) start from the same weights before the first
    callback and optimizer step."""
    ref = _initial_weights(cell, tmp_path / 'ref', monkeypatch)
    for i, extra in enumerate([(DIAG,), ('--optimizer', LEGACY),
                               ('--optimizer', LEGACY, DIAG)]):
        other = _initial_weights(cell, tmp_path / str(i), monkeypatch, *extra)
        assert len(other) == len(ref)
        for a, b in zip(ref, other):
            assert a.shape == b.shape and np.array_equal(a, b), extra


# --- 13. optimizers ----------------------------------------------------------------------------

@pytest.mark.parametrize('name', ['rmsprop', 'rmsprop_legacy'])
def test_13_rmsprop_update_rules(name):
    """rmsprop: lr * g * rsqrt(v + eps) (epsilon inside the root);
    rmsprop_legacy: lr * g / (sqrt(v) + eps) (outside). A large epsilon
    against small gradients separates the two."""
    lr, eps, rho = 0.01, 1e-3, 0.9
    grads = np.array([[1e-2, -3e-2], [2e-2, 5e-3], [-1e-2, 1e-2]], np.float32)
    with tf.device('/CPU:0'):
        var = tf.Variable([0.5, -0.5], dtype=tf.float32)
        opt = make_optimizer(name, lr, epsilon=eps)
        x, v = var.numpy().astype(np.float64), np.zeros(2)
        for g in grads:
            opt.apply_gradients([(tf.constant(g), var)])
            v = rho * v + (1 - rho) * g.astype(np.float64) ** 2
            if name == 'rmsprop':
                x -= lr * g / np.sqrt(v + eps)
            else:
                x -= lr * g / (np.sqrt(v) + eps)
            np.testing.assert_allclose(var.numpy(), x, rtol=1e-5)
