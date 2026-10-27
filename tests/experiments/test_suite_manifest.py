# tests/experiments/test_suite_manifest.py
# Final-suite manifest (issue a2ngerer/LRC#52): expansion count and order, the
# heavy group, the resume filter, the final-stage lookup of hpo_selection.json
# and the result names against the runners. No SLURM involved. Issue #68: every
# line passes the campaign's ode_unfolds and dt_scale, and a manifest -> runner
# smoke checks config.effective of every arm. Issue #71: the frozen protocol
# YAMLs (configs/suite/{lv,duffing,mt,pa,co}_*.yaml): fit counts per task,
# endpoint flags, per-arm runs_per_gpu, config_from, committee result names and
# resume per fit.
import fnmatch
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import yaml

from experiments import run_committee_benchmark as co_runner
from experiments import run_lotka_volterra_benchmark as lv_runner
from experiments import run_person_activity_benchmark as pa_runner
from experiments.suite_manifest import (RUNNERS, _flag_value, arms, count,
                                        expand, is_done, load, main, pending,
                                        result_name, run, write)
from src.benchmark.checkpoints import rebuild_from_result
from src.tasks.person_activity import load_person_activity
from src.tasks.person_activity.model import ODE_UNFOLD_CELLS
from tests.tasks.person_activity.test_person_activity import _write_stub

HPO = {'task': 'lv', 'system': 'duffing', 'stage': 'hpo',
       'cells': ['cfc_lrc', 'lrc'], 'wirings': ['dense3', 'ncp', 'cncp'],
       'param_budget': 4000, 'seeds': [0, 1], 'epochs': 300,
       'ode_unfolds': 6, 'dt_scale': 'auto',
       'extra_args': ['--forcing', 'mixed'], 'per_cell': {'lrc': {'ode_unfolds': 1}},
       'cluster': {'runs_per_gpu': 6, 'mem': '16G', 'walltime': '12:00:00'},
       'heavy': {'cells': ['lrc'], 'wirings': ['cncp'],
                 'cluster': {'runs_per_gpu': 2, 'walltime': '48:00:00'}}}


def _cfg(tmp_path, **over):
    path = tmp_path / 'suite.yaml'
    path.write_text(yaml.safe_dump({**HPO, **over}))
    return str(path)


def _selections(hpo_dir, arms, **over):
    for i, (cell, wiring) in enumerate(arms):
        d = hpo_dir / f'{cell}_{wiring}'
        d.mkdir(parents=True)
        (d / 'hpo_selection.json').write_text(json.dumps({
            'task': 'lv', 'system': 'duffing', 'cell': cell, 'wiring': wiring,
            'param_budget': 4000, 'lr': [1e-3, 3e-3, 1e-2, 3e-5][i % 4],
            'clipnorm': [None, 1.0][i % 2], **over}))


def test_hpo_one_line_per_arm_in_order(tmp_path, capsys):
    lines = expand(load(_cfg(tmp_path)))
    assert [line['id'] for line in lines] == [
        'hpo_cfc_lrc_dense3', 'hpo_cfc_lrc_ncp', 'hpo_cfc_lrc_cncp',
        'hpo_lrc_dense3', 'hpo_lrc_ncp', 'hpo_lrc_cncp']
    assert [line['group'] for line in lines] == ['default', 'default', 'heavy',
                                                 'heavy', 'heavy', 'heavy']
    lrc = lines[3]
    assert lrc['argv'] == [
        'uv', 'run', 'python', 'experiments/run_grid_hpo.py', '--task', 'lv',
        '--cell', 'lrc', '--wiring', 'dense3', '--system', 'duffing',
        '--tuning-seeds', '0', '1', '--param-budget', '4000', '--epochs', '300',
        '--ode-unfolds', '6', '--dt-scale', 'auto', '--anchors', '8,16,24,32',
        '--horizon', '96', '--ic-spread', '0.4', '--forcing', 'mixed',
        '--ode-unfolds', '1']
    assert (lrc['outdir'], lrc['done']) == ('hpo/lrc_dense3',
                                            'hpo/lrc_dense3/hpo_selection.json')
    assert lines[0]['argv'][-4:] == ['--ic-spread', '0.4', '--forcing', 'mixed']

    main([_cfg(tmp_path), '--count'])
    main([_cfg(tmp_path), '--count', '--group', 'heavy'])
    main([_cfg(tmp_path), '--list', '--group', 'heavy', '--wandb'])
    out = capsys.readouterr().out.splitlines()
    assert out[:2] == ['66', '44']                  # planned fits: 11 per arm
    assert [row.split('\t')[:2] for row in out[2:]] == [
        ['heavy[0]', 'hpo_cfc_lrc_cncp'], ['heavy[1]', 'hpo_lrc_dense3'],
        ['heavy[2]', 'hpo_lrc_ncp'], ['heavy[3]', 'hpo_lrc_cncp']]
    assert out[2].endswith("--wandb --outdir '<results>/hpo/cfc_lrc_cncp'")


def test_write_splits_groups_and_merges_heavy_resources(tmp_path):
    cfg = load(_cfg(tmp_path))
    plan = write(cfg, expand(cfg), str(tmp_path)).splitlines()
    assert plan == [
        'SUITE_BENCH=02_duffing', 'SUITE_STAGE=hpo', 'SUITE_COUNT=6',
        "SUITE_GROUPS='default heavy'",
        'SUITE_default_N=2', 'SUITE_default_RUNS_PER_GPU=6',
        'SUITE_default_MEM=16G', 'SUITE_default_WALLTIME=12:00:00',
        'SUITE_heavy_N=4', 'SUITE_heavy_RUNS_PER_GPU=2',
        'SUITE_heavy_MEM=16G', 'SUITE_heavy_WALLTIME=48:00:00']
    heavy = (tmp_path / 'manifest_heavy.jsonl').read_text().splitlines()
    assert [json.loads(row)['id'] for row in heavy] == [
        'hpo_cfc_lrc_cncp', 'hpo_lrc_dense3', 'hpo_lrc_ncp', 'hpo_lrc_cncp']


def test_resume_skips_lines_whose_result_exists(tmp_path, monkeypatch):
    cfg = load(_cfg(tmp_path))
    lines = expand(cfg)
    results = tmp_path / 'results'
    (results / 'hpo' / 'cfc_lrc_ncp').mkdir(parents=True)
    (results / 'hpo' / 'cfc_lrc_ncp' / 'hpo_selection.json').write_text('{}')
    (results / 'hpo' / 'lrc_ncp').mkdir(parents=True)   # killed mid-arm: not done
    assert [line['id'] for line in pending(lines, str(results))] == [
        'hpo_cfc_lrc_dense3', 'hpo_cfc_lrc_cncp', 'hpo_lrc_dense3',
        'hpo_lrc_ncp', 'hpo_lrc_cncp']

    write(cfg, lines, str(tmp_path))
    execs = []
    monkeypatch.setattr('os.execvp', lambda f, argv: execs.append(argv))
    manifest = str(tmp_path / 'manifest_default.jsonl')
    run(manifest, 1, str(results))                       # done -> skipped
    assert execs == []
    run(manifest, 0, str(results))
    assert execs[0][:4] == ['uv', 'run', 'python', 'experiments/run_grid_hpo.py']
    assert execs[0][-2:] == ['--outdir', str(results / 'hpo/cfc_lrc_dense3')]


def test_final_reads_the_selected_lr_and_clipnorm(tmp_path):
    arms = [(c, w) for c in HPO['cells'] for w in HPO['wirings']]
    _selections(tmp_path / 'hpo', arms)
    cfg = load(_cfg(tmp_path, stage='final', seeds=[0, 1, 2],
                    hpo_dir=str(tmp_path / 'hpo'), extra_args=[]))
    lines = expand(cfg)
    assert len(lines) == 6 * 3
    assert [line['id'] for line in lines[:4]] == [
        'cfc_lrc_dense3_seed0', 'cfc_lrc_dense3_seed1', 'cfc_lrc_dense3_seed2',
        'cfc_lrc_ncp_seed0']
    first, second = lines[0], lines[3]
    assert first['argv'][3:] == [
        'experiments/run_lotka_volterra_benchmark.py', '--cell', 'cfc_lrc',
        '--wiring', 'dense3', '--seed', '0', '--system', 'duffing',
        '--lr', '0.001', '--param-budget', '4000', '--epochs', '300',
        '--ode-unfolds', '6', '--dt-scale', 'auto', '--anchors', '8,16,24,32',
        '--horizon', '96', '--ic-spread', '0.4']
    assert second['argv'][second['argv'].index('--lr'):][:4] == [
        '--lr', '0.003', '--clipnorm', '1']
    assert (first['outdir'], first['done']) == ('.', 'cfc_lrc_dense3_seed0.json')
    results = tmp_path / 'results'
    results.mkdir()
    (results / 'cfc_lrc_dense3_seed1.json').write_text('{}')
    assert not is_done(lines[0], str(results)) and is_done(lines[1], str(results))
    assert len(pending(lines, str(results))) == 17


def test_final_fails_on_a_missing_or_foreign_selection(tmp_path):
    _selections(tmp_path / 'hpo', [('cfc_lrc', 'dense3')])
    cfg = load(_cfg(tmp_path, stage='final', hpo_dir=str(tmp_path / 'hpo')))
    with pytest.raises(SystemExit, match='cfc_lrc_ncp, cfc_lrc_cncp, lrc_dense3'):
        expand(cfg)
    _selections(tmp_path / 'hpo2', [('cfc_lrc', 'dense3')], param_budget=30000)
    cfg = load(_cfg(tmp_path, stage='final', hpo_dir=str(tmp_path / 'hpo2'),
                    cells=['cfc_lrc'], wirings=['dense3']))
    with pytest.raises(SystemExit, match='param_budget 30000'):
        expand(cfg)


@pytest.mark.parametrize('over, msg', [
    ({'seeds': [0, 0]}, 'two different tuning seeds'),
    ({'stage': 'final'}, 'needs hpo_dir'),
    ({'task': 'pa'}, 'lv only'),
    ({'task': 'xx'}, 'task must be one of'),
    ({'arms': {'a': {'cell': 'gru', 'wiring': 'ncp'}}}, "either 'cells' or 'arms'"),
    ({'cells': None, 'arms': {'a': {'cell': 'gru'}}}, 'needs cell and wiring'),
    ({'stage': 'final', 'hpo_dir': 'x', 'extra_seeds': {'seeds': [5], 'arms': ['gru_ncp']}},
     'extra_seeds.arms names unknown arms'),
    ({'extra_seeds': {'seeds': [5], 'arms': ['lrc_ncp']}}, 'stage final only'),
    ({'heavy': {'arms': ['lrc_dense']}}, 'heavy.arms names unknown arms'),
    ({'config_from': {'lrc_ncp': 'ltc_ncp'}}, 'config_from names unknown arms'),
    ({'config_from': {'lrc_ncp': 'lrc_dense3', 'lrc_dense3': 'lrc_cncp'}},
     'reference has no HPO'),
    ({'cell': ['gru']}, 'unknown keys'),
    ({'ode_unfolds': None}, "'ode_unfolds' is required"),
    ({'dt_scale': None}, "'dt_scale' is required"),
    ({'ode_unfolds': 0}, 'integer >= 1'),
    ({'dt_scale': 'fast'}, 'auto or a positive number'),
    ({'dt_scale': 0}, 'auto or a positive number'),
])
def test_load_rejects_bad_configs(tmp_path, over, msg):
    with pytest.raises(SystemExit, match=msg):
        load(_cfg(tmp_path, **over))


@pytest.mark.parametrize('extra', [[], ['--backbone-layers', '1'],
                                   ['--forcing', 'mixed'],
                                   ['--backbone-layers', '2', '--forcing', 'mixed']])
def test_result_name_matches_the_runners(extra):
    bb = int(extra[extra.index('--backbone-layers') + 1]) if '--backbone-layers' in extra else None
    forcing = 'mixed' if '--forcing' in extra else 'teacher'
    args = Namespace(cell='cfc_lrc', wiring='ncp', seed=3, backbone_layers=bb,
                     forcing=forcing)
    assert result_name('lv', 'cfc_lrc', 'ncp', 3, extra) == lv_runner.result_filename(args)
    pa_extra = [a for a in extra if a not in ('--forcing', 'mixed')]
    assert (result_name('pa', 'cfc_lrc', 'ncp', 3, pa_extra)
            == pa_runner.result_tag(args) + '.json')


@pytest.mark.parametrize('stage', ['hpo', 'final'])
def test_every_line_passes_ode_unfolds_and_dt_scale(tmp_path, stage):
    """No runner default is relied on: both values reach every runner call,
    and a per-cell control (k = 1, dt_scale = 1) overrides them last."""
    arms = [(c, w) for c in HPO['cells'] for w in HPO['wirings']]
    _selections(tmp_path / 'hpo', arms)
    cfg = load(_cfg(tmp_path, stage=stage, hpo_dir=str(tmp_path / 'hpo'),
                    per_cell={'cfc_lrc': {'dt_scale': 1}}))
    for line in expand(cfg):
        argv = line['argv']
        last = {f: _flag_value(argv, f) for f in ('--ode-unfolds', '--dt-scale')}
        control = _flag_value(argv, '--cell') == 'cfc_lrc'
        assert last == {'--ode-unfolds': '6',
                        '--dt-scale': '1' if control else 'auto'}


def _ode_cells(model):
    return [sub for layer in model.layers
            for sub in (layer, *layer._flatten_layers())
            if isinstance(sub, ODE_UNFOLD_CELLS)]


@pytest.mark.parametrize('task', ['lv', 'pa'])
def test_smoke_every_arm_records_dt_scale_and_ode_unfolds(task, tmp_path,
                                                          monkeypatch):
    """Manifest -> runner -> JSON for every arm of a 1-epoch campaign:
    config.effective.dt_scale is the resolved number (LV 1 / (10 / 128);
    PA 1 / median train dt) and ctrnn runs 6 sub-steps on dense3 and cncp
    (the follow-up to #56, whose test covers ncp and dense)."""
    cells, wirings = ['ctrnn', 'cfc_lrc'], ['dense3', 'cncp']
    for cell in cells:
        for wiring in wirings:
            d = tmp_path / 'hpo' / f'{cell}_{wiring}'
            d.mkdir(parents=True)
            (d / 'hpo_selection.json').write_text(json.dumps({
                'task': task, 'cell': cell, 'wiring': wiring, 'lr': 1e-3,
                'clipnorm': None}))
    if task == 'lv':
        extra, expected = ['--size', '4', '--n-trajectories', '6'], 12.8
    else:
        stub = _write_stub(tmp_path / 'ConfLongDemo_JSI.txt')
        extra = ['--size', '8', '--batch-size', '32', '--data-path', str(stub)]
        expected = 1 / np.median(
            load_person_activity(data_path=str(stub)).train_t)
    path = tmp_path / 'suite.yaml'
    path.write_text(yaml.safe_dump({
        'task': task, 'stage': 'final', 'cells': cells, 'wirings': wirings,
        'seeds': [0], 'epochs': 1, 'ode_unfolds': 6, 'dt_scale': 'auto',
        'extra_args': extra, 'hpo_dir': str(tmp_path / 'hpo')}))
    runner = {'lv': lv_runner, 'pa': pa_runner}[task]
    results = tmp_path / 'results'
    lines = expand(load(str(path)))
    assert len(lines) == 4
    for line in lines:
        assert line['argv'][:4] == ['uv', 'run', 'python', RUNNERS[task]]
        monkeypatch.setattr('sys.argv', [RUNNERS[task], *line['argv'][4:],
                                         '--outdir', str(results)])
        runner.main()
        json_path = results / line['done']
        eff = json.loads(json_path.read_text())['config']['effective']
        assert eff['dt_scale'] == eff['build']['dt_scale'] == pytest.approx(
            expected, rel=1e-12)
        if eff['build']['cell'] == 'ctrnn':
            assert eff['ode_unfolds'] == 6
            ode = _ode_cells(rebuild_from_result(str(json_path)))
            assert ode and {c.ode_unfolds for c in ode} == {6}
        else:
            assert eff['ode_unfolds'] is None


# --- frozen protocol F1-2026-10-26 (#71) -------------------------------------

SUITE = Path(__file__).resolve().parents[2] / 'configs' / 'suite'
# task -> (HPO fits, final fits incl. the extra seeds), section 3 of the freeze
FROZEN = {'lv': (308, 165), 'duffing': (319, 200), 'mt': (44, 40),
          'pa': (308, 180), 'co': (54, 110)}
# outside the per-task counts: the controls and the guard (seeds of the K = 30
# finals)
CONTROLS = ['control_lv_dt_scale1', 'control_duffing_k1', 'control_pa_k1',
            'guard_co_k30_transfer']


def _frozen(name, hpo_dir=None, final_epochs=None):
    """A frozen YAML; with hpo_dir: stub selections for every arm that runs
    HPO (lr and clip a function of the arm name), and the YAML pointed at
    them."""
    cfg = load(str(SUITE / f'{name}.yaml'))
    if hpo_dir is None:
        return cfg
    refs = cfg.get('config_from') or {}
    for arm, cell, wiring, _ in arms(cfg):
        i = sum(map(ord, arm))
        if arm in refs:
            continue
        d = hpo_dir / arm
        d.mkdir(parents=True, exist_ok=True)
        (d / 'hpo_selection.json').write_text(json.dumps({
            'task': cfg['task'], 'system': cfg.get('system'), 'cell': cell,
            'wiring': wiring, 'param_budget': cfg.get('param_budget'),
            'lr': 1e-4 * (i % 97 + 1), 'clipnorm': [None, 1.0][i % 2],
            **({'final_epochs': final_epochs} if final_epochs else {})}))
    cfg['hpo_dir'] = str(hpo_dir)
    return cfg


def _yamls(task):
    return sorted(p for p in SUITE.glob(f'{task}_*.yaml'))


@pytest.mark.parametrize('task', FROZEN)
def test_frozen_fit_counts_per_task(task, capsys):
    """--count over the task's YAMLs reproduces the frozen fit counts (473,
    519, 84, 488, 164); the controls stay out of the per-task globs."""
    stages = {'hpo': 0, 'final': 0}
    for path in _yamls(task):
        cfg = load(str(path))
        stages[cfg['stage']] += count(cfg)
    assert (stages['hpo'], stages['final']) == FROZEN[task]
    main(['--count', *map(str, _yamls(task))])
    assert capsys.readouterr().out.strip() == str(sum(FROZEN[task]))
    assert not {p.stem for p in _yamls(task)} & set(CONTROLS)


def test_frozen_arm_matrix():
    """28 / 32 / 5 / 28 / 17 arms; the hpo and final YAMLs of a task list the
    same arms; tuning seeds 100/101, final seeds 0-4 (+ 5-9)."""
    names = {n: [a[0] for a in arms(_frozen(n))] for n in (
        'lv_hpo', 'lv_final', 'duffing_hpo', 'duffing_final', 'mt_hpo',
        'mt_final', 'mt_final_static_w', 'pa_hpo', 'pa_final', 'co_k3_hpo',
        'co_k3_final', 'co_k30_final')}
    for task in ('lv', 'duffing', 'pa', 'co_k3'):
        assert names[f'{task}_hpo'] == names[f'{task}_final']
    assert len(names['lv_hpo']) == 28 and len(names['duffing_hpo']) == 32
    assert len(names['pa_hpo']) == 28 and 'mlp_dense' not in names['pa_hpo']
    assert len({*names['mt_final'], *names['mt_final_static_w']}) == 4  # + _w
    assert len(names['co_k3_hpo']) + len(names['co_k30_final']) == 17
    assert set(names['duffing_hpo']) - set(names['lv_hpo']) == {
        'cfc_static_ncp', 'ltc_ncp', 'mm_ltc_ncp', 'cfc_mm_ltc_ncp'}
    for path in SUITE.glob('*.yaml'):
        if path.stem.startswith('canary'):
            continue
        cfg = load(str(path))
        assert cfg['seeds'] == ([100, 101] if cfg['stage'] == 'hpo'
                                else list(range(len(cfg['seeds']))))
        assert (cfg.get('extra_seeds') or {}).get('seeds', [5, 6, 7, 8, 9]) == [
            5, 6, 7, 8, 9]


@pytest.mark.parametrize('name', ['lv_hpo', 'duffing_final', 'mt_hpo',
                                  'mt_final_static_w', 'pa_final',
                                  'control_lv_dt_scale1', 'control_duffing_k1',
                                  'control_pa_k1', 'co_k30_final'])
def test_frozen_lines_carry_the_protocol_flags(name, tmp_path):
    """Rollout lines: --anchors 8,16,24,32 --horizon 96 and the system's
    --ic-spread (0.3 multitimescale_lib, else 0.4), mixed forcing; lv/pa lines:
    ode_unfolds 6 and dt_scale auto unless the control changes exactly one;
    committee lines: none of these (closed-form cells, no such flags)."""
    cfg = _frozen(name, tmp_path / 'hpo')
    for line in expand(cfg):
        argv = line['argv']
        last = {f: _flag_value(argv, f) for f in (
            '--anchors', '--horizon', '--ic-spread', '--ode-unfolds',
            '--dt-scale', '--forcing')}
        if cfg['task'] == 'co':
            assert set(last.values()) == {None}
            continue
        assert last['--ode-unfolds'] == ('1' if name.endswith('_k1') else '6')
        assert last['--dt-scale'] == ('1' if name.endswith('dt_scale1') else 'auto')
        if cfg['task'] == 'pa':
            assert (last['--anchors'], last['--forcing']) == (None, None)
            continue
        spread = '0.3' if cfg.get('system') == 'multitimescale_lib' else '0.4'
        assert (last['--anchors'], last['--horizon'], last['--ic-spread'],
                last['--forcing']) == ('8,16,24,32', '96', spread, 'mixed')
        assert line['done'].endswith('_mixed.json') or line['done'].startswith('hpo/')


@pytest.mark.parametrize('name', ['pa_hpo', 'pa_final', 'lv_final',
                                  'duffing_hpo', 'mt_final'])
def test_frozen_runs_per_gpu(name, tmp_path):
    """Person Activity: the solver arms on ncp (lrc, mm_lrc, ctrnn) one fit per
    GPU, every other arm three; rollouts three per job."""
    cfg = _frozen(name, tmp_path / 'hpo')
    lines = expand(cfg)
    plan = dict(row.split('=', 1) for row in write(cfg, lines, str(tmp_path)).splitlines())
    assert plan['SUITE_default_RUNS_PER_GPU'] == '3'
    heavy = {line['argv'][line['argv'].index('--cell') + 1] + '_'
             + line['argv'][line['argv'].index('--wiring') + 1]
             for line in lines if line['group'] == 'heavy'}
    if cfg['task'] == 'pa':
        assert heavy == {'lrc_ncp', 'mm_lrc_ncp', 'ctrnn_ncp'}
        assert plan['SUITE_heavy_RUNS_PER_GPU'] == '1'
    else:
        assert not heavy and 'SUITE_heavy_N' not in plan


def test_config_from_takes_the_reference_selection(tmp_path):
    """Arms without HPO: no hpo line; their finals carry the named reference
    arm's lr/clipnorm (Duffing diagnostic arms; cfc_static_w on MT at the cfc
    width without a budget; the controls on their main arm's selection)."""
    hpo = [line['id'] for line in expand(_frozen('duffing_hpo'))]
    assert len(hpo) == 29 and not {'hpo_ltc_ncp', 'hpo_mm_ltc_ncp',
                                    'hpo_cfc_mm_ltc_ncp'} & set(hpo)
    cfg = _frozen('duffing_final', tmp_path / 'duffing')
    by_id = {line['id']: line['argv'] for line in expand(cfg)}

    def lr_clip(argv):
        return _flag_value(argv, '--lr'), _flag_value(argv, '--clipnorm')
    for arm, ref in cfg['config_from'].items():
        assert not (tmp_path / 'duffing' / arm).exists()
        assert lr_clip(by_id[f'{arm}_seed0']) == lr_clip(by_id[f'{ref}_seed0'])
        assert _flag_value(by_id[f'{arm}_seed0'], '--cell') == arm[:-len('_ncp')]
    assert len({lr_clip(a) for a in by_id.values()}) > 2

    mt = expand(_frozen('mt_final', tmp_path / 'mt'))
    static_w = expand(_frozen('mt_final_static_w', tmp_path / 'mt'))
    ref = next(line['argv'] for line in mt if line['id'] == 'cfc_static_ncp_seed0')
    assert len(static_w) == 10
    for line in static_w:
        assert lr_clip(line['argv']) == lr_clip(ref)
        assert _flag_value(line['argv'], '--size') == '24'
        assert '--param-budget' not in line['argv']


def test_static_w_size_is_the_cfc_ncp_budget_width():
    """size 24 in mt_final_static_w.yaml is cfc/ncp's 4k pick of the runner."""
    from src.tasks.lotka_volterra.model import build_lotka_volterra_model
    from src.tasks.person_activity.model import budget_sizes
    from src.wirings import size_for_budget

    def build(size):
        return build_lotka_volterra_model('ncp', 'cfc', size=size, seed=0)
    assert size_for_budget(build, 4000, budget_sizes('ncp'))[0] == 24


@pytest.mark.parametrize('flags', [
    {}, {'vote_lambda': 0}, {'consensus': 'median'}, {'view_average': 'mean'},
    {'view_average': 'fault_mean'}, {'n_columns': 1}])
@pytest.mark.parametrize('clip', [None, 1.0])
def test_co_result_name_matches_the_runner(flags, clip):
    """The committee result name as a glob over the budget-derived size
    matches the runner's result_filename, and no other arm's."""
    from experiments.suite_manifest import _flags
    defaults = dict(cell='cfc_lrc', wiring='cncp', n_columns=30, size=53.5,
                    view_mode='identical', vote_lambda=None, consensus='mean',
                    dense_vote_units=None, view_average='off', train_drop=0.0,
                    train_noise=0.0, test_corruption='dropout',
                    train_corruption='on', lr=3e-3, clipnorm=clip,
                    corruption_seed=0, seed=7, backbone_layers=1)
    names = {}
    for f in ({}, {'vote_lambda': 0}, {'consensus': 'median'},
              {'view_average': 'mean'}, {'view_average': 'fault_mean'}):
        names[json.dumps(f)] = co_runner.result_filename(Namespace(**{**defaults, **f}))
    argv = ['--n-columns', '30', '--backbone-layers', '1', '--lr', '0.003',
            *(['--clipnorm', '1'] if clip else []), *_flags(flags)]
    pattern = result_name('co', 'cfc_lrc', 'cncp', 7, argv)
    own = co_runner.result_filename(Namespace(**{**defaults, **flags, 'size': 12}))
    assert fnmatch.fnmatchcase(own, pattern)
    assert [n for n in names.values() if fnmatch.fnmatchcase(n, pattern)] == (
        [] if 'n_columns' in flags else [names[json.dumps(flags)]])


@pytest.mark.parametrize('name', ['duffing_final', 'pa_final', 'co_k30_final'])
def test_resume_per_fit_on_the_frozen_manifest(name, tmp_path, monkeypatch):
    """A re-submit into the same run folder runs only the fits without a
    result JSON (the #52 resume), on the frozen final manifests; the committee
    fits are found by their size glob; co finals take final_epochs."""
    cfg = _frozen(name, tmp_path / 'hpo', final_epochs=60)
    lines = expand(cfg)
    assert len(lines) == count(cfg)
    assert len({line['id'] for line in lines}) == len({line['done'] for line in lines}) == len(lines)
    results = tmp_path / 'results'
    results.mkdir()
    finished = lines[::7]
    for line in finished:   # what a killed array left behind
        (results / line['done'].replace('*', '40')).write_text('{}')
    todo = pending(lines, str(results))
    assert [line['id'] for line in todo] == [
        line['id'] for line in lines if line not in finished]
    if cfg['task'] == 'co':
        assert {_flag_value(line['argv'], '--epochs') for line in lines} == {'60'}

    write(cfg, lines, str(tmp_path))
    execs = []
    monkeypatch.setattr('os.execvp', lambda f, argv: execs.append(argv))
    manifest = tmp_path / f"manifest_{finished[0]['group']}.jsonl"
    rows = [json.loads(row) for row in manifest.read_text().splitlines()]
    run(str(manifest), rows.index(finished[0]), str(results))
    assert execs == []
    nxt = next(line for line in todo if line in rows)
    run(str(manifest), rows.index(nxt), str(results))
    assert execs == [[*nxt['argv'], '--outdir', str(results)]]


def test_resume_guard_fits_count_as_k30_finals(tmp_path):
    """The two transfer-guard fits are seed 0 of the K = 30 finals: resuming
    co_k30_final.yaml in the guard's run folder skips exactly them."""
    guard = expand(_frozen('guard_co_k30_transfer', tmp_path / 'hpo'))
    k30 = expand(_frozen('co_k30_final', tmp_path / 'hpo2'))
    assert [line['done'] for line in guard] == [
        line['done'] for line in k30 if line['id'] in ('ncp_cons_seed0',
                                                         'cncp_cons_seed0')]
    results = tmp_path / 'results'
    results.mkdir()
    for line in guard:
        (results / line['done'].replace('*', '41.5')).write_text('{}')
    assert len(pending(k30, str(results))) == 63
