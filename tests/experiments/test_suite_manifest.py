# tests/experiments/test_suite_manifest.py
# Final-suite manifest (issue a2ngerer/LRC#52): expansion count and order, the
# heavy group, the resume filter, the final-stage lookup of hpo_selection.json
# and the result names against the runners. No SLURM involved. Issue #68: every
# line passes the campaign's ode_unfolds and dt_scale, and a manifest -> runner
# smoke checks config.effective of every arm.
import json
from argparse import Namespace

import numpy as np
import pytest
import yaml

from experiments import run_lotka_volterra_benchmark as lv_runner
from experiments import run_person_activity_benchmark as pa_runner
from experiments.suite_manifest import (RUNNERS, _flag_value, expand, is_done,
                                        load, main, pending, result_name, run,
                                        write)
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
        '--ode-unfolds', '6', '--dt-scale', 'auto', '--ode-unfolds', '1',
        '--forcing', 'mixed']
    assert (lrc['outdir'], lrc['done']) == ('hpo/lrc_dense3',
                                            'hpo/lrc_dense3/hpo_selection.json')
    assert lines[0]['argv'][-6:] == ['--ode-unfolds', '6', '--dt-scale', 'auto',
                                     '--forcing', 'mixed']

    main([_cfg(tmp_path), '--count'])
    main([_cfg(tmp_path), '--count', '--group', 'heavy'])
    main([_cfg(tmp_path), '--list', '--group', 'heavy', '--wandb'])
    out = capsys.readouterr().out.splitlines()
    assert out[:2] == ['6', '4']
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
        '--ode-unfolds', '6', '--dt-scale', 'auto']
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
