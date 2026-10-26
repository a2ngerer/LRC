# tests/experiments/test_suite_manifest.py
# Final-suite manifest (issue a2ngerer/LRC#52): expansion count and order, the
# heavy group, the resume filter, the final-stage lookup of hpo_selection.json
# and the result names against the runners. No SLURM involved.
import json
from argparse import Namespace

import pytest
import yaml

from experiments import run_lotka_volterra_benchmark as lv_runner
from experiments import run_person_activity_benchmark as pa_runner
from experiments.suite_manifest import (expand, is_done, load, main, pending,
                                        result_name, run, write)

HPO = {'task': 'lv', 'system': 'duffing', 'stage': 'hpo',
       'cells': ['cfc_lrc', 'lrc'], 'wirings': ['dense3', 'ncp', 'cncp'],
       'param_budget': 4000, 'seeds': [0, 1], 'epochs': 300,
       'extra_args': ['--forcing', 'mixed'], 'per_cell': {'lrc': {'ode_unfolds': 6}},
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
        '--ode-unfolds', '6', '--forcing', 'mixed']
    assert (lrc['outdir'], lrc['done']) == ('hpo/lrc_dense3',
                                            'hpo/lrc_dense3/hpo_selection.json')
    assert '--ode-unfolds' not in lines[0]['argv']

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
        '--lr', '0.001', '--param-budget', '4000', '--epochs', '300']
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
