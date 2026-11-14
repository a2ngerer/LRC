# tests/experiments/test_f2_manifest.py
# F2 campaign YAMLs (issue a2ngerer/LRC#105): the snapshot of every existing
# configs/suite/*.yaml (generated from the code before the manifest
# extensions selection_from and stage eval), the two extensions, and the F2
# YAMLs pinned to protocol F2 sections 3, 4, 7 and 8.
import json
import re
import tempfile
from pathlib import Path

import pytest

from experiments.suite_manifest import count, expand, write
from tests.experiments.test_suite_manifest import SUITE, _frozen

SNAPSHOT = Path(__file__).parent / 'data' / 'suite_argv_snapshot.json'


def _snapshot(name):
    with tempfile.TemporaryDirectory() as d:
        cfg = _frozen(name)
        if cfg.get('hpo_dir'):
            cfg = _frozen(name, Path(d) / 'hpo')
        lines = expand(cfg)
        plan = write(cfg, lines, d)
    return {'count': count(cfg), 'plan': plan.splitlines(), 'lines': lines}


def test_snapshot_covers_every_pre_f2_yaml():
    names = {p.stem for p in SUITE.glob('*.yaml')
             if not p.stem.startswith(('f2_', 'f3_'))}
    assert names == set(json.loads(SNAPSHOT.read_text()))


@pytest.mark.parametrize('name', sorted(json.loads(SNAPSHOT.read_text())))
def test_existing_yaml_lines_are_unchanged(name):
    """Neither extension used: count, write plan and every line (argv, id,
    group, outdir, done) equal the snapshot taken before #105."""
    assert _snapshot(name) == json.loads(SNAPSHOT.read_text())[name]


# --- the two manifest extensions ----------------------------------------------

import yaml  # noqa: E402

from experiments.suite_manifest import (KEYS, _flag_value, arms, is_done,  # noqa: E402
                                        load, main, pending, run)

F2_LV = {'task': 'lv', 'system': 'duffing', 'stage': 'final',
         'cells': {'cfc_lrc_eps1': ['ncp'], 'lrc_init': ['ncp']},
         'param_budget': 4000, 'seeds': [0, 1], 'epochs': 450,
         'ode_unfolds': 6, 'dt_scale': 'auto',
         'extra_args': ['--forcing', 'mixed', '--deterministic']}


def _write(tmp_path, cfg, name='suite.yaml'):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


def _stub(hpo_dir, arm, cell, wiring, **over):
    d = hpo_dir / arm
    d.mkdir(parents=True, exist_ok=True)
    (d / 'hpo_selection.json').write_text(json.dumps({
        'task': 'lv', 'system': 'duffing', 'cell': cell, 'wiring': wiring,
        'param_budget': 4000, 'lr': 0.01, 'clipnorm': 1.0, **over}))


SEL_FROM = {'cfc_lrc_eps1_ncp': {'arm': 'cfc_lrc_ncp', 'cell': 'cfc_lrc',
                                 'wiring': 'ncp'},
            'lrc_init_ncp': {'arm': 'lrc_ncp', 'cell': 'lrc', 'wiring': 'ncp'}}


def test_selection_from_reads_the_reference_selection(tmp_path):
    _stub(tmp_path / 'hpo', 'cfc_lrc_ncp', 'cfc_lrc', 'ncp', lr=0.003)
    _stub(tmp_path / 'hpo', 'lrc_ncp', 'lrc', 'ncp', lr=0.01, clipnorm=None)
    cfg = load(_write(tmp_path, {**F2_LV, 'hpo_dir': str(tmp_path / 'hpo'),
                                 'selection_from': SEL_FROM}))
    by_id = {line['id']: line['argv'] for line in expand(cfg)}
    assert list(by_id) == ['cfc_lrc_eps1_ncp_seed0', 'cfc_lrc_eps1_ncp_seed1',
                           'lrc_init_ncp_seed0', 'lrc_init_ncp_seed1']
    eps1, init = by_id['cfc_lrc_eps1_ncp_seed1'], by_id['lrc_init_ncp_seed0']
    assert (_flag_value(eps1, '--lr'), _flag_value(eps1, '--clipnorm')) == ('0.003', '1')
    assert (_flag_value(init, '--lr'), '--clipnorm' in init) == ('0.01', False)
    assert _flag_value(eps1, '--cell') == 'cfc_lrc_eps1'
    assert count(cfg) == 4


@pytest.mark.parametrize('mapping, stub, msg', [
    ({'cell': 'lrc'}, {}, "cell 'cfc_lrc', campaign has 'lrc'"),
    ({'wiring': 'dense'}, {}, "wiring 'ncp', campaign has 'dense'"),
    ({}, {'param_budget': 30000}, 'param_budget 30000'),
    ({}, {'system': 'periodic_predator_prey'}, "system 'periodic_predator_prey'"),
    ({}, {'task': 'pa'}, "task 'pa'"),
])
def test_selection_from_checks_the_reference(tmp_path, mapping, stub, msg):
    """The reference's selection is checked against the mapping's cell and
    wiring and the campaign's task, system and param_budget."""
    _stub(tmp_path / 'hpo', 'cfc_lrc_ncp', 'cfc_lrc', 'ncp', **stub)
    cfg = load(_write(tmp_path, {
        **F2_LV, 'cells': {'cfc_lrc_eps1': ['ncp']},
        'hpo_dir': str(tmp_path / 'hpo'), 'selection_from': {
            'cfc_lrc_eps1_ncp': {**SEL_FROM['cfc_lrc_eps1_ncp'], **mapping}}}))
    with pytest.raises(SystemExit, match=msg):
        expand(cfg)


def test_selection_from_missing_file(tmp_path):
    cfg = load(_write(tmp_path, {**F2_LV, 'hpo_dir': str(tmp_path / 'none'),
                                 'selection_from': SEL_FROM}))
    with pytest.raises(SystemExit, match='no hpo_selection.json .* cfc_lrc_ncp, lrc_ncp'):
        expand(cfg)
    assert count(cfg) == 4                       # --count needs no selection


@pytest.mark.parametrize('over, msg', [
    ({'hpo_dir': None, 'fixed': {'lr': 0.001, 'clipnorm': None}},
     'needs stage final with hpo_dir'),
    ({'stage': 'hpo', 'hpo_dir': None, 'seeds': [100, 101]},
     'needs stage final with hpo_dir'),
    ({'selection_from': {'gru_ncp': SEL_FROM['lrc_init_ncp']}},
     'selection_from names unknown arms'),
    ({'selection_from': {'lrc_init_ncp': {'arm': 'lrc_ncp', 'cell': 'lrc'}}},
     r'maps an arm to \{arm, cell, wiring\}'),
    ({'config_from': {'lrc_init_ncp': 'cfc_lrc_eps1_ncp'}},
     'config_from or selection_from'),
])
def test_selection_from_rejections(tmp_path, over, msg):
    cfg = {**F2_LV, 'hpo_dir': 'x', 'selection_from': SEL_FROM, **over}
    with pytest.raises(SystemExit, match=msg):
        load(_write(tmp_path, {k: v for k, v in cfg.items() if v is not None}))


def _eval_sources(tmp_path, task='lv'):
    """Two final YAMLs with equal record names, and an eval YAML over both."""
    base = ({**F2_LV, 'cells': {'cfc_lrc': ['ncp']}, 'fixed': {
        'lr': 0.001, 'clipnorm': None}} if task == 'lv' else {
        'task': 'pa', 'stage': 'final', 'cells': {'cfc_lrc': ['ncp']},
        'param_budget': 30000, 'seeds': [0, 1], 'epochs': 50,
        'ode_unfolds': 6, 'dt_scale': 'auto', 'extra_args': ['--deterministic'],
        'fixed': {'lr': 0.001, 'clipnorm': None}})
    a = _write(tmp_path, base, 'src_a.yaml')
    b = _write(tmp_path, {**base, 'epochs': 900 if task == 'lv' else 100},
               'src_b.yaml')
    ev = {'task': task, 'stage': 'eval', 'sources': [
        {'config': a, 'records_dir': str(tmp_path / 'run_a' / 'results')},
        {'config': b, 'records_dir': str(tmp_path / 'run_b' / 'results'),
         'seeds': [1]}], 'cluster': {'runs_per_gpu': 2}}
    if task == 'lv':
        ev['system'] = 'duffing'
    return load(_write(tmp_path, ev, 'ev.yaml'))


def test_eval_lv_lines(tmp_path, monkeypatch, capsys):
    cfg = _eval_sources(tmp_path)
    lines = expand(cfg)
    assert count(cfg) == len(lines) == 3
    rec = 'cfc_lrc_ncp_seed1_mixed.json'
    assert [line['id'] for line in lines] == [
        'src_a/cfc_lrc_ncp_seed0_mixed.json', f'src_a/{rec}', f'src_b/{rec}']
    last = lines[2]
    assert last['argv'] == [
        'uv', 'run', 'python', 'experiments/run_lotka_volterra_benchmark.py',
        '--eval-only', str(tmp_path / 'run_b' / 'results' / rec),
        '--anchors', '8,16,24,32', '--horizon', '96']
    assert (last['outdir'], last['done'], last['group']) == (
        'src_b', f'src_b/{rec}', 'default')
    results = tmp_path / 'ev_run' / 'results'
    (results / 'src_a').mkdir(parents=True)
    (results / 'src_a' / rec).write_text('{}')
    assert [line['id'] for line in pending(lines, str(results))] == [
        lines[0]['id'], lines[2]['id']]
    plan = write(cfg, lines, str(tmp_path))
    assert 'SUITE_BENCH=02_duffing' in plan and 'SUITE_STAGE=eval' in plan
    execs = []
    monkeypatch.setattr('os.execvp', lambda f, argv: execs.append(argv))
    manifest = str(tmp_path / 'manifest_default.jsonl')
    run(manifest, 1, str(results))                 # done -> skipped
    run(manifest, 2, str(results))
    assert execs == [[*last['argv'], '--outdir', str(results / 'src_b')]]
    capsys.readouterr()
    main([str(tmp_path / 'ev.yaml'), '--list'])
    assert capsys.readouterr().out.splitlines()[0].endswith(
        "--outdir '<results>/src_a'")


def test_eval_pa_lines(tmp_path, monkeypatch):
    cfg = _eval_sources(tmp_path, 'pa')
    lines = expand(cfg)
    assert len(lines) == 3
    line = lines[2]
    rec = str(tmp_path / 'run_b' / 'results' / 'cfc_lrc_ncp_seed1.json')
    assert line['argv'] == ['uv', 'run', 'python',
                            'experiments/eval_robustness.py', '--results', rec]
    assert line['done'] == line['out'] == 'src_b/cfc_lrc_ncp_seed1.csv'
    write(cfg, lines, str(tmp_path))
    execs = []
    monkeypatch.setattr('os.execvp', lambda f, argv: execs.append(argv))
    results = tmp_path / 'ev_run' / 'results'
    run(str(tmp_path / 'manifest_default.jsonl'), 2, str(results))
    out = results / 'src_b' / 'cfc_lrc_ncp_seed1.csv'
    assert execs == [[*line['argv'], '--out', str(out)]]
    assert '--outdir' not in execs[0] and out.parent.is_dir()


@pytest.mark.parametrize('over, msg', [
    ({'seeds': [0]}, r"does not take \['seeds'\]"),
    ({'task': 'mt'}, 'needs task lv, pa or co'),
    ({'sources': []}, "'sources' is required"),
    ({'sources': [{'config': 'configs/suite/duffing_final.yaml'}]},
     'a source is'),
    ({'sources': [{'config': 'configs/suite/lv_final.yaml', 'records_dir': 'r'}]},
     'not a final stage of the same task and system'),
    ({'sources': [{'config': 'configs/suite/duffing_hpo.yaml', 'records_dir': 'r'}]},
     'not a final stage'),
    ({'sources': [{'config': 'configs/suite/duffing_final.yaml',
                   'records_dir': 'r', 'arms': ['lrc_ncp'], 'seeds': [10]}]},
     'has no fit'),
    ({'sources': [{'config': 'configs/suite/duffing_final.yaml',
                   'records_dir': 'r', 'arms': ['cfc_lrc_dense'], 'seeds': [5]}]},
     'has no fit'),
    ({'sources': [{'config': 'configs/suite/duffing_final.yaml',
                   'records_dir': 'r', 'arms': ['nope_ncp']}]},
     r"has no arms \['nope_ncp'\]"),
    ({'sources': [{'config': 'configs/suite/duffing_final.yaml',
                   'records_dir': 'r', 'arms': ['lrc_ncp', 'nope_ncp']}]},
     r"has no arms \['nope_ncp'\]"),
    ({'bogus': 1}, 'unknown keys'),
])
def test_eval_rejections(tmp_path, over, msg):
    cfg = {'task': 'lv', 'system': 'duffing', 'stage': 'eval', 'sources': [
        {'config': 'configs/suite/duffing_final.yaml', 'records_dir': 'r'}],
        **over}
    with pytest.raises(SystemExit, match=msg):
        load(_write(tmp_path, cfg))


def test_eval_rejects_a_record_listed_twice(tmp_path):
    src = {'config': 'configs/suite/duffing_final.yaml', 'records_dir': 'r',
           'arms': ['lrc_ncp'], 'seeds': [0]}
    cfg = {'task': 'lv', 'system': 'duffing', 'stage': 'eval',
           'sources': [src, dict(src)]}
    with pytest.raises(SystemExit, match='listed twice'):
        expand(load(_write(tmp_path, cfg)))


def test_eval_default_takes_every_source_record():
    """Without arms/seeds a source gives every fit of its YAML (200 Duffing
    finals of F1, extra seeds included)."""
    cfg = {'task': 'lv', 'system': 'duffing', 'stage': 'eval', 'sources': [
        {'config': 'configs/suite/duffing_final.yaml', 'records_dir': 'r'}]}
    with tempfile.TemporaryDirectory() as d:
        assert count(load(_write(Path(d), cfg))) == 200



# --- co eval lines: the vote-lambda grid at fixed weights (#137) ------------

CO_STEM = 'cfc_lrc_ncp_K30_s*_identical_lr*_clip*_cs0_seed1_bb1'


def _co_eval(tmp_path, records, **over):
    """An eval YAML over two ncp_cons fits of the K = 30 final (over: None
    drops a key)."""
    cfg = {'task': 'co', 'stage': 'eval', 'sources': [
        {'config': 'configs/suite/co_k30_final.yaml',
         'records_dir': str(records), 'arms': ['ncp_cons'], 'seeds': [0, 1]}],
        'vote_lambda_override': [0, 0.5, 1], **over}
    return load(_write(tmp_path, {k: v for k, v in cfg.items()
                                  if v is not None}, 'co_eval.yaml'))


def test_eval_co_lines(tmp_path, monkeypatch, capsys):
    """Expanding reads no record; the line carries record_glob and --out;
    at run time exactly one weights file runs, zero or two stop the line,
    a done line is skipped before the glob."""
    records = tmp_path / 'run' / 'results'
    cfg = _co_eval(tmp_path, records)
    lines = expand(cfg)
    assert count(cfg) == len(lines) == 2
    line = lines[1]
    assert line == {
        'id': f'co_k30_final/{CO_STEM}.json', 'group': 'default',
        'outdir': 'co_k30_final', 'out': 'co_k30_final',
        'done': f'co_k30_final/{CO_STEM}.lambda_grid.evalonly.json',
        'record_glob': f'{records}/{CO_STEM}.weights.h5',
        'argv': ['uv', 'run', 'python', 'experiments/run_committee_benchmark.py',
                 '--eval-only', '--vote-lambda-override', '0.0,0.5,1.0']}
    plan = write(cfg, lines, str(tmp_path))
    assert 'SUITE_BENCH=04_column_voting' in plan and 'SUITE_STAGE=eval' in plan
    manifest = str(tmp_path / 'manifest_default.jsonl')
    results = tmp_path / 'ev_run' / 'results'
    execs = []
    monkeypatch.setattr('os.execvp', lambda f, argv: execs.append(argv))
    with pytest.raises(SystemExit, match='matches 0 weight files') as e:
        run(manifest, 1, str(results))
    assert line['record_glob'] in str(e.value)
    records.mkdir(parents=True)
    fit = 'cfc_lrc_ncp_K30_s20_identical_lr0.003_clip1_cs0_seed1_bb1'
    for name in (fit + '.weights.h5', fit + '.json',
                 fit.replace('seed1', 'seed11') + '.weights.h5',
                 fit.replace('_identical', '_identical_lam0') + '.weights.h5',
                 fit.replace('_identical', '_identical_median') + '.weights.h5'):
        (records / name).write_text('')
    run(manifest, 1, str(results))
    assert execs == [[*line['argv'], '--weights',
                      str(records / (fit + '.weights.h5')),
                      '--out', str(results / 'co_k30_final')]]
    (records / (fit.replace('lr0.003_clip1', 'lr0.01_clipoff')
                + '.weights.h5')).write_text('')
    with pytest.raises(SystemExit, match='matches 2 weight files'):
        run(manifest, 1, str(results))
    done = results / 'co_k30_final' / (fit + '.lambda_grid.evalonly.json')
    done.parent.mkdir()
    done.write_text('{}')
    run(manifest, 1, str(results))                 # done -> skipped
    assert len(execs) == 1
    capsys.readouterr()
    main([str(tmp_path / 'co_eval.yaml'), '--list'])
    first = capsys.readouterr().out.splitlines()[0]
    assert '--weights' in first and "--out '<results>/co_k30_final'" in first


@pytest.mark.parametrize('over, msg', [
    ({'vote_lambda_override': None}, 'task co needs vote_lambda_override'),
    ({'vote_lambda_override': []}, 'task co needs vote_lambda_override'),
    ({'vote_lambda_override': [0, 1.5]}, r'distinct values in \[0, 1\]'),
    ({'vote_lambda_override': [0.5, 0.5]}, r'distinct values in \[0, 1\]'),
    ({'vote_lambda_override': [True]}, r'distinct values in \[0, 1\]'),
    ({'vote_lambda_override': 0.5}, r'distinct values in \[0, 1\]'),
    ({'robustness': True, 'seed_streams': 'v1'}, 'apply to task lv and pa'),
    ({'robustness': False}, 'apply to task lv and pa'),
    ({'seed_streams': 'v1'}, 'apply to task lv and pa'),
    ({'extra_args': ['--val-draws', '2']}, 'extra_args applies to task pa'),
    ({'sources': [{'config': 'configs/suite/pa_final.yaml',
                   'records_dir': 'r'}]}, 'not a final stage of the same task'),
    ({'sources': [{'config': 'configs/suite/co_k3_hpo.yaml',
                   'records_dir': 'r'}]}, 'not a final stage'),
])
def test_eval_co_rejections(tmp_path, over, msg):
    with pytest.raises(SystemExit, match=msg):
        _co_eval(tmp_path, 'r', **over)


@pytest.mark.parametrize('task, src, extra', [
    ('pa', 'pa_final', {}), ('lv', 'duffing_final', {'system': 'duffing'})])
def test_vote_lambda_override_is_co_only(tmp_path, task, src, extra):
    cfg = {'task': task, 'stage': 'eval', 'vote_lambda_override': [0, 1],
           'sources': [{'config': f'configs/suite/{src}.yaml',
                        'records_dir': 'r'}], **extra}
    with pytest.raises(SystemExit, match='applies to task co only'):
        load(_write(tmp_path, cfg))


# --- the F2 YAMLs (protocol F2 sections 3, 4, 7, 8) ---------------------------

import math  # noqa: E402

S10, S3, S5 = list(range(10)), [0, 1, 2], list(range(5))
F1_LV, F1_DU, F1_PA = '<lv-hpo run>', '<duffing-hpo run>', '<pa-hpo run>'
# name -> (stage, task, system, arms (name: cell/wiring), seeds, epochs,
#          param_budget, config source: hpo | fixed | the hpo_dir placeholder,
#          plus selection_from {arm: reference arm})
_LV, _DU = ('lv', None), ('lv', 'duffing')
F2 = {
    'f2_smoke_lv': ('final', *_LV, ['cfc_lrc/ncp_random', 'cfc_lrc/ncp', 'cfc_lrc/dense3',
                                    'cfc_lrc/cncp', 'lrc_init/ncp', 'lrc_cal_lv/ncp',
                                    'gru/ncp', 'lrc_tau/ncp', 'lrc_init_tau/ncp'],
                    [0], 3, 4000, 'fixed', {}),
    'f2_smoke_lv_w': ('final', *_LV, ['cfc_lrc/dense3'], [0], 3, None, 'fixed', {}),
    'f2_smoke_lv_c8': ('final', *_LV, ['lrc/ncp', 'lrc/dense', 'lrc_init/ncp',
                                       'lrc_cal_lv/ncp'], [0], 3, 4000, 'fixed', {}),
    'f2_smoke_duffing': ('final', *_DU, ['cfc_lrc/ncp', 'cfc_lrc/dense3', 'cfc_lrc/cncp',
                                         'gru/ncp', 'lrc_init/ncp', 'lrc_cal_duffing/ncp',
                                         'cfc_lrc_eps1/ncp'], [0], 3, 4000, 'fixed', {}),
    'f2_smoke_pa': ('final', 'pa', None, ['cfc_lrc/ncp_random', 'cfc_lrc/cncp_random',
                                          'ltc_minel/ncp'], [0], 3, 30000, 'fixed', {}),
    'f2_smoke_pa_w': ('final', 'pa', None, ['cfc_lrc/dense3'], [0], 3, None, 'fixed', {}),
    'f2_guard_lv': ('final', *_LV, ['cfc_lrc/dense3', 'cfc_lrc/ncp', 'lrc/dense', 'lrc/ncp'],
                    [0], 300, 4000, F1_LV, {}),
    'f2_guard_duffing': ('final', *_DU, ['cfc/ncp', 'cfc_lrc/ncp', 'lrc/ncp'], [0], 450,
                         4000, F1_DU, {}),
    'f2_guard_pa': ('final', 'pa', None, ['cfc_lrc/dense3', 'cfc_lrc/ncp', 'cfc_lrc/cncp'],
                    [0], 50, 30000, F1_PA, {}),
    'f2_hpo_lv': ('hpo', *_LV, ['cfc_lrc/ncp_random', 'lrc_init/ncp', 'lrc_cal_lv/ncp'],
                  [100, 101], 300, 4000, 'hpo', {}),
    'f2_hpo_lv_w': ('hpo', *_LV, ['cfc_lrc/dense3'], [100, 101], 300, None, 'hpo', {}),
    'f2_hpo_pa': ('hpo', 'pa', None, ['cfc_lrc/ncp_random'], [100, 101], 50, 30000, 'hpo', {}),
    'f2_hpo_pa_w': ('hpo', 'pa', None, ['cfc_lrc/dense3'], [100, 101], 50, None, 'hpo', {}),
    'f2_budget_lv': ('final', *_LV, ['cfc_lrc/ncp'], S10, 600, 4000, F1_LV, {}),
    'f2_budget_duffing': ('final', *_DU, ['cfc_lrc/ncp'], S10, 900, 4000, F1_DU, {}),
    'f2_budget_desc_lv': ('final', *_LV, ['cfc_lrc/dense3', 'cfc_lrc/cncp', 'gru/ncp'],
                          S10, 600, 4000, F1_LV, {}),
    'f2_budget_desc_duffing': ('final', *_DU, ['cfc_lrc/dense3', 'cfc_lrc/cncp', 'gru/ncp'],
                               S10, 900, 4000, F1_DU, {}),
    'f2_lrc_ncp_lv_s59': ('final', *_LV, ['lrc/ncp'], [5, 6, 7, 8, 9], 300, 4000, F1_LV, {}),
    'f2_b1_duffing_init': ('final', *_DU, ['lrc_init/ncp', 'lrc_cal_duffing/ncp'], S3, 450,
                           4000, F1_DU, {'lrc_init_ncp': 'lrc/ncp',
                                         'lrc_cal_duffing_ncp': 'lrc/ncp'}),
    'f2_b5_p2b_pa': ('final', 'pa', None, ['ltc_minel/ncp'], S3, 50, 30000, 'fixed', {}),
    'f2_c7_lv_lrc': ('final', *_LV, ['lrc_tau/ncp'], S3, 300, 4000, F1_LV,
                     {'lrc_tau_ncp': 'lrc/ncp'}),
    'f2_c2_eps1_duffing': ('final', *_DU, ['cfc_lrc_eps1/ncp'], S5, 450, 4000, F1_DU,
                           {'cfc_lrc_eps1_ncp': 'cfc_lrc/ncp'}),
    'f2_ba2_cncp_random_pa': ('final', 'pa', None, ['cfc_lrc/cncp_random'], S5, 50, 30000,
                              F1_PA, {'cfc_lrc_cncp_random': 'cfc_lrc/cncp'}),
    'f2_rewiring_lv_final': ('final', *_LV, ['cfc_lrc/ncp_random'], S10, 300, 4000,
                             '<f2-hpo-lv run>', {}),
    'f2_rewiring_pa_final': ('final', 'pa', None, ['cfc_lrc/ncp_random'], S10, 50, 30000,
                             '<f2-hpo-pa run>', {}),
    'f2_width_lv_final': ('final', *_LV, ['cfc_lrc/dense3'], S10, 300, None,
                          '<f2-hpo-lv-w run>', {}),
    'f2_width_pa_final': ('final', 'pa', None, ['cfc_lrc/dense3'], S10, 50, None,
                          '<f2-hpo-pa-w run>', {}),
    'f2_b2_lv_final': ('final', *_LV, ['lrc_init/ncp', 'lrc_cal_lv/ncp'], S10, 300, 4000,
                       '<f2-hpo-lv run>', {}),
    'f2_c8_lv_f1sel': ('final', *_LV, ['lrc/ncp', 'lrc/dense'], S3, 900, 4000, F1_LV, {}),
    'f2_c8_lv_f2sel': ('final', *_LV, ['lrc_init/ncp', 'lrc_cal_lv/ncp'], S3, 900, 4000,
                       '<f2-hpo-lv run>', {}),
    'f2_c7_lv_lrc_init': ('final', *_LV, ['lrc_init_tau/ncp'], S3, 300, 4000,
                          '<f2-hpo-lv run>', {'lrc_init_tau_ncp': 'lrc_init/ncp'}),
}
EVALS = {'f2_guard_eval_lv': 30, 'f2_guard_eval_duffing': 30, 'f2_guard_eval_pa': 30,
         'f2_t5_fidelity_lv': 107, 'f2_t5_fidelity_duffing': 54,
         'f2_t5_fidelity_pa': 31,
         # the F1 controls (#99): lv and duffing two lines per record
         'f2_control_eval_lv': 6, 'f2_control_eval_duffing': 10,
         'f2_control_eval_pa': 5}
# --count per YAML (protocol F2 section 8.2; smoke and eval: lines)
COUNTS = {'f2_smoke_lv': 9, 'f2_smoke_lv_w': 1, 'f2_smoke_lv_c8': 4,
          'f2_smoke_duffing': 7, 'f2_smoke_pa': 3, 'f2_smoke_pa_w': 1,
          'f2_guard_lv': 4, 'f2_guard_duffing': 3, 'f2_guard_pa': 3,
          'f2_hpo_lv': 33, 'f2_hpo_lv_w': 11, 'f2_hpo_pa': 11, 'f2_hpo_pa_w': 11,
          'f2_budget_lv': 10, 'f2_budget_duffing': 10, 'f2_budget_desc_lv': 30,
          'f2_budget_desc_duffing': 30, 'f2_lrc_ncp_lv_s59': 5,
          'f2_b1_duffing_init': 6, 'f2_b5_p2b_pa': 3, 'f2_c7_lv_lrc': 3,
          'f2_c2_eps1_duffing': 5, 'f2_ba2_cncp_random_pa': 5,
          'f2_rewiring_lv_final': 10, 'f2_rewiring_pa_final': 10,
          'f2_width_lv_final': 10, 'f2_width_pa_final': 10, 'f2_b2_lv_final': 20,
          'f2_c8_lv_f1sel': 6, 'f2_c8_lv_f2sel': 6, 'f2_c7_lv_lrc_init': 3, **EVALS}
T2 = ['f2_hpo_lv', 'f2_hpo_lv_w', 'f2_hpo_pa', 'f2_hpo_pa_w', 'f2_budget_lv',
      'f2_budget_duffing', 'f2_budget_desc_lv', 'f2_budget_desc_duffing',
      'f2_lrc_ncp_lv_s59', 'f2_b1_duffing_init', 'f2_b5_p2b_pa', 'f2_c7_lv_lrc',
      'f2_c2_eps1_duffing', 'f2_ba2_cncp_random_pa']
T3 = ['f2_rewiring_lv_final', 'f2_rewiring_pa_final', 'f2_width_lv_final',
      'f2_width_pa_final', 'f2_b2_lv_final', 'f2_c8_lv_f1sel', 'f2_c8_lv_f2sel',
      'f2_c7_lv_lrc_init']
GUARD = ['f2_guard_lv', 'f2_guard_duffing', 'f2_guard_pa']
SMOKES = [n for n in F2 if n.startswith('f2_smoke_')]
# Submit waves (protocol F2 section 8.4); SW4 is PPO (cluster/submit_mujoco.sh)
WAVES = {'SW1': SMOKES + GUARD + ['f2_guard_eval_lv', 'f2_guard_eval_duffing',
                                  'f2_guard_eval_pa'],
         'SW2': T2, 'SW3': T3,
         'SW5': ['f2_t5_fidelity_lv', 'f2_t5_fidelity_duffing', 'f2_t5_fidelity_pa']}
C8 = ['f2_c8_lv_f1sel', 'f2_c8_lv_f2sel']
WIDTH = {'f2_smoke_lv_w': '15', 'f2_hpo_lv_w': '15', 'f2_width_lv_final': '15',
         'f2_smoke_pa_w': '43', 'f2_hpo_pa_w': '43', 'f2_width_pa_final': '43'}
DET = '--deterministic'


def _f2_files():
    return sorted(p.stem for p in SUITE.glob('f2_*.yaml'))


def _stubbed(name, hpo_dir):
    """An F2 YAML with stub selections (own arms and selection_from refs)."""
    cfg = load(str(SUITE / f'{name}.yaml'))
    return _frozen(name, hpo_dir) if cfg.get('hpo_dir') else cfg


def test_f2_files_are_the_table():
    assert _f2_files() == sorted(COUNTS) == sorted([*F2, *EVALS])


@pytest.mark.parametrize('name', sorted(COUNTS))
def test_f2_count_per_yaml(name, capsys):
    main(['--count', str(SUITE / f'{name}.yaml')])
    assert capsys.readouterr().out.strip() == str(COUNTS[name])


def test_f2_fit_totals_match_protocol_8_2(capsys):
    """258 fits = guard 10 + HPO 66 + without HPO 107 + T3 75; T2 173; smoke
    25 lines, guard eval 90, T5 fidelity 192 (protocol F2 sections 7, 8.2)."""
    fits = [n for n in COUNTS if n not in SMOKES and n not in EVALS]
    main(['--count', *(str(SUITE / f'{n}.yaml') for n in fits)])
    assert capsys.readouterr().out.strip() == '258'
    hpo = sum(COUNTS[n] for n in T2 if F2[n][0] == 'hpo')
    assert (sum(COUNTS[n] for n in GUARD), hpo, sum(COUNTS[n] for n in T2) - hpo,
            sum(COUNTS[n] for n in T3)) == (10, 66, 107, 75)
    assert sum(COUNTS[n] for n in T2) == 173
    assert sum(COUNTS[n] for n in SMOKES) == 25
    assert sum(EVALS[n] for n in EVALS if 'guard' in n) == 90
    assert sum(EVALS[n] for n in EVALS if 't5' in n) == 192
    # per posten of section 8.2
    c = COUNTS
    assert (c['f2_hpo_lv'], c['f2_hpo_pa']) == (11 + 22, 11)   # rewiring + B2
    assert c['f2_hpo_lv_w'] + c['f2_hpo_pa_w'] == 22           # width HPO
    assert c['f2_rewiring_lv_final'] + c['f2_rewiring_pa_final'] == 20
    assert c['f2_width_lv_final'] + c['f2_width_pa_final'] == 20
    assert c['f2_budget_lv'] + c['f2_budget_duffing'] == 20
    assert c['f2_budget_desc_lv'] + c['f2_budget_desc_duffing'] == 60
    assert (c['f2_b1_duffing_init'], c['f2_lrc_ncp_lv_s59'], c['f2_b2_lv_final'],
            c['f2_c8_lv_f1sel'] + c['f2_c8_lv_f2sel'], c['f2_c7_lv_lrc'],
            c['f2_c7_lv_lrc_init'], c['f2_b5_p2b_pa'], c['f2_ba2_cncp_random_pa'],
            c['f2_c2_eps1_duffing']) == (6, 5, 20, 12, 3, 3, 3, 5, 5)


@pytest.mark.parametrize('name', sorted(F2))
def test_f2_arm_matrix(name):
    stage, task, system, arm_list, seeds, epochs, budget, source, sel_from = F2[name]
    raw = yaml.safe_load((SUITE / f'{name}.yaml').read_text())
    assert set(raw) <= KEYS and 'description' not in raw
    cfg = load(str(SUITE / f'{name}.yaml'))
    assert (cfg['stage'], cfg['task'], cfg.get('system'), cfg['seeds'],
            cfg['epochs'], cfg.get('param_budget')) == (
        stage, task, system, seeds, epochs, budget)
    assert sorted(f'{c}/{w}' for _, c, w, _ in arms(cfg)) == sorted(arm_list)
    assert not cfg.get('extra_seeds') and not cfg.get('config_from')
    if source == 'hpo':
        assert not cfg.get('hpo_dir') and not cfg.get('fixed')
    elif source == 'fixed':
        assert cfg['fixed'] == {'lr': 0.001, 'clipnorm': None}
        assert not cfg.get('hpo_dir')
    else:
        assert source in cfg['hpo_dir'] and not cfg.get('fixed')
    assert {a: f"{r['cell']}/{r['wiring']}" for a, r in (
        cfg.get('selection_from') or {}).items()} == sel_from
    for arm, ref in (cfg.get('selection_from') or {}).items():
        assert ref['arm'] == ref['cell'] + '_' + ref['wiring']
    assert cfg['ode_unfolds'] == 6 and cfg['dt_scale'] == 'auto'
    assert {'runs_per_gpu', 'walltime'} <= set(cfg['cluster'])
    if cfg.get('heavy'):
        assert {'runs_per_gpu'} <= set(cfg['heavy']['cluster'])


@pytest.mark.parametrize('name', sorted(COUNTS))
def test_f2_header_has_section_count_and_submit(name):
    head = (SUITE / f'{name}.yaml').read_text().split('\ntask:')[0]
    assert head.startswith('# F2 ') and 'protocol F2 section' in head
    assert f'./cluster/submit_suite.sh configs/suite/{name}.yaml' in head
    assert f'{COUNTS[name]} ' in head.replace('\n#', ' ')


@pytest.mark.parametrize('name', sorted(COUNTS))
def test_f2_lines(name, tmp_path):
    """done names unique per YAML; lv lines: anchors, horizon, ic-spread,
    mixed forcing, k = 6, dt_scale auto; pa lines: neither anchors nor
    forcing; every fit line --deterministic, the YAML's --epochs and
    --param-budget, or the fixed --size of a dense3_w YAML without a budget."""
    cfg = _stubbed(name, tmp_path / 'hpo')
    lines = expand(cfg)
    dones = [line['done'] for line in lines]
    assert len(set(dones)) == len(dones)
    assert len(lines) == (len(arms(cfg)) if cfg['stage'] == 'hpo' else COUNTS[name])
    if cfg['stage'] == 'eval':
        return
    for line in lines:
        argv = line['argv']
        flag = {f: _flag_value(argv, f) for f in (
            '--anchors', '--horizon', '--ic-spread', '--forcing', '--ode-unfolds',
            '--dt-scale', '--epochs', '--param-budget', '--size')}
        assert argv.count(DET) == 1 and '--alpha-min' not in argv
        assert (flag['--ode-unfolds'], flag['--dt-scale'], flag['--epochs']) == (
            '6', 'auto', str(cfg['epochs']))
        assert flag['--param-budget'] == (
            None if name in WIDTH else str(cfg['param_budget']))
        assert flag['--size'] == WIDTH.get(name)
        if cfg['task'] == 'pa':
            assert (flag['--anchors'], flag['--forcing']) == (None, None)
        else:
            spread = '0.4'
            assert (flag['--anchors'], flag['--horizon'], flag['--ic-spread'],
                    flag['--forcing']) == ('8,16,24,32', '96', spread, 'mixed')
        assert (_flag_value(argv, '--lr-schedule') == 'linear') == (
            name in C8 or name == 'f2_smoke_lv_c8')


def _array_tasks(name, tmp_path):
    tmp_path.mkdir()
    cfg = _stubbed(name, tmp_path / 'hpo')
    plan = dict(row.split('=', 1) for row in write(
        cfg, expand(cfg), str(tmp_path)).splitlines())
    return sum(math.ceil(int(plan[f'SUITE_{g}_N'])
                         / int(plan[f'SUITE_{g}_RUNS_PER_GPU']))
               for g in plan['SUITE_GROUPS'].strip("'").split())


def test_f2_array_tasks_per_wave(tmp_path):
    """ceil(lines / runs_per_gpu) per group as cluster/submit_suite.sh: SW2 32,
    SW3 21 (17 without C8; 32 + 17 = 49), SW1 20 and SW5 21 (protocol 8.4
    says 8 and about 3; the differences are the split smoke YAMLs and a few
    eval lines per GPU instead of one task per eval YAML); every wave <= 50."""
    tasks = {n: _array_tasks(n, tmp_path / n) for n in COUNTS}
    wave = {w: sum(tasks[n] for n in names) for w, names in WAVES.items()}
    assert wave == {'SW1': 20, 'SW2': 32, 'SW3': 21, 'SW5': 21}
    assert wave['SW3'] - sum(tasks[n] for n in C8) == 17
    assert wave['SW2'] + wave['SW3'] - sum(tasks[n] for n in C8) == 49
    assert max(wave.values()) <= 50
    assert sum(tasks[n] for n in T2 if F2[n][0] == 'hpo') == 6   # one per chain
    assert (tasks['f2_b5_p2b_pa'], tasks['f2_smoke_pa']) == (3, 2)


def _f1_lines(name, hpo_dir):
    return {line['id']: line for line in expand(_frozen(name, hpo_dir))}


@pytest.mark.parametrize('name, f1', [('f2_guard_lv', 'lv_final'),
                                      ('f2_guard_duffing', 'duffing_final'),
                                      ('f2_guard_pa', 'pa_final')])
def test_guard_lines_equal_the_f1_lines(name, f1, tmp_path):
    f1_lines = _f1_lines(f1, tmp_path / 'hpo')
    lines = expand(_frozen(name, tmp_path / 'hpo'))
    assert lines and all(line == f1_lines[line['id']] for line in lines)


@pytest.mark.parametrize('name, f1, epochs', [
    ('f2_budget_lv', 'lv_final', '600'), ('f2_budget_desc_lv', 'lv_final', '600'),
    ('f2_budget_duffing', 'duffing_final', '900'),
    ('f2_budget_desc_duffing', 'duffing_final', '900'),
    ('f2_c8_lv_f1sel', 'lv_final', '900')])
def test_budget_and_c8_lines_equal_f1_but_epochs(name, f1, epochs, tmp_path):
    """Equal to the F1 line of the same arm and seed except --epochs (and the
    C8 --lr-schedule linear); the forcing ramp of AlphaSchedule is epochs // 2."""
    from src.neurons.forced_rollout_cell import AlphaSchedule
    f1_lines = _f1_lines(f1, tmp_path / 'hpo')
    for line in expand(_frozen(name, tmp_path / 'hpo')):
        ref = f1_lines[line['id']]
        drop = ['--lr-schedule', 'linear'] if name in C8 else []
        argv = [a for a in line['argv'] if a not in drop]
        i = argv.index('--epochs')
        assert argv[i + 1] == epochs
        assert argv[:i + 1] + argv[i + 2:] == [
            a for j, a in enumerate(ref['argv']) if j != i + 1]
        assert line['done'] == ref['done']
    assert AlphaSchedule(None, int(epochs)).half == int(epochs) // 2


def test_s59_lines_equal_f1_with_the_seed_replaced(tmp_path):
    ref = _f1_lines('lv_final', tmp_path / 'hpo')['lrc_ncp_seed0']['argv']
    for line in expand(_frozen('f2_lrc_ncp_lv_s59', tmp_path / 'hpo')):
        seed = _flag_value(line['argv'], '--seed')
        i = ref.index('--seed')
        assert line['argv'] == [*ref[:i + 1], seed, *ref[i + 2:]]


def test_c2_eps1_runs_on_the_budget_not_a_fixed_size(tmp_path):
    """eps=1 is sized by --param-budget (the width search counts the frozen
    head, so the arm gets the cfc_lrc width); a fixed --size would give it
    the aggregator label [fixed width] (param_budget None)."""
    for line in expand(_stubbed('f2_c2_eps1_duffing', tmp_path / 'hpo')):
        assert _flag_value(line['argv'], '--param-budget') == '4000'
        assert '--size' not in line['argv']
        assert (_flag_value(line['argv'], '--lr'),
                _flag_value(line['argv'], '--clipnorm')) == ('0.0042', '1')


@pytest.mark.parametrize('name', sorted(EVALS))
def test_f2_eval_outputs_stay_in_their_stem_folder(name, tmp_path):
    """Every eval line writes below results/<source stem>/ of the eval run, never
    below a records_dir (the F1 / F2 run folders are read-only inputs)."""
    cfg = load(str(SUITE / f'{name}.yaml'))
    results = Path('results') / 'eval_run' / 'results'
    stems = {Path(s['config']).stem: s['records_dir'] for s in cfg['sources']}
    lines = expand(cfg)
    assert len({line['done'] for line in lines}) == len(lines) == EVALS[name]
    for line in lines:
        stem = line['outdir']
        assert stem in stems and line['done'].startswith(f'{stem}/')
        out = (results / line.get('out', line['done'])).resolve()
        assert out.is_relative_to((results / stem).resolve())
        for records in stems.values():
            assert not out.is_relative_to(Path(records).resolve())
        rec = line['argv'][5]
        assert rec.startswith(stems[stem] + '/')
    bench = {'lv': '01_lotka_volterra', 'duffing': '02_duffing',
             'pa': '03_person_activity'}[name.rsplit('_', 1)[1]]
    for records in stems.values():      # a cluster path, read at run time
        assert re.fullmatch(rf'\.\./benchmarks/{bench}/runs/<[^>]+>/results',
                            records), records
    if 'guard' in name:
        f1 = {'lv': 'lv_final', 'duffing': 'duffing_final', 'pa': 'pa_final'}[
            name.rsplit('_', 1)[1]]
        assert set(stems) == {f1}
    elif 'control' in name:
        assert set(stems) == {CONTROL_SOURCE[name]}
    else:
        assert set(stems) == T5_SOURCES[name]
        assert sum(COUNTS[s] for s in stems) == EVALS[name]


CONTROL_SOURCE = {'f2_control_eval_lv': 'control_lv_dt_scale1',
                  'f2_control_eval_duffing': 'control_duffing_k1',
                  'f2_control_eval_pa': 'control_pa_k1'}


# T5 sources (protocol F2 section 7): every non-HPO fit YAML of T1-T3
T5_SOURCES = {
    'f2_t5_fidelity_lv': {
        'f2_guard_lv', 'f2_rewiring_lv_final', 'f2_width_lv_final',
        'f2_budget_lv', 'f2_budget_desc_lv', 'f2_lrc_ncp_lv_s59',
        'f2_b2_lv_final', 'f2_c8_lv_f1sel', 'f2_c8_lv_f2sel', 'f2_c7_lv_lrc',
        'f2_c7_lv_lrc_init'},
    'f2_t5_fidelity_duffing': {
        'f2_guard_duffing', 'f2_budget_duffing', 'f2_budget_desc_duffing',
        'f2_b1_duffing_init', 'f2_c2_eps1_duffing'},
    'f2_t5_fidelity_pa': {
        'f2_guard_pa', 'f2_rewiring_pa_final', 'f2_width_pa_final',
        'f2_b5_p2b_pa', 'f2_ba2_cncp_random_pa'},
}


# --- runner acceptance: one 1-epoch fit per new arm kind ----------------------

from experiments import run_lotka_volterra_benchmark as lv_runner  # noqa: E402
from experiments import run_person_activity_benchmark as pa_runner  # noqa: E402
from experiments.suite_manifest import RUNNERS  # noqa: E402
from tests.tasks.person_activity.test_person_activity import _write_stub  # noqa: E402


def _markers(eff):
    """The F2 option keys present in one config.effective."""
    kw, build = eff.get('cell_kwargs') or {}, eff['build']
    return ({k for k in ('elastance_one', 'tau_init', 'min_elapsed',
                         'init_slice_scale') if k in kw}
            | ({'lr_schedule'} if 'lr_schedule' in eff else set())
            | ({build['wiring']} & {'ncp_random', 'cncp_random'})
            | ({'fixed_width'} if eff.get('param_budget') is None else set()))


# (YAML, arm id of the seed-0 line, expected markers); the plain cfc_lrc/ncp
# line of the same YAML carries none
ACCEPT = [
    ('f2_smoke_lv', 'cfc_lrc_ncp_random_seed0', {'ncp_random'}),
    ('f2_smoke_lv', 'lrc_tau_ncp_seed0', {'tau_init'}),
    ('f2_smoke_lv', 'lrc_init_tau_ncp_seed0', {'tau_init'}),
    ('f2_smoke_lv', 'cfc_lrc_ncp_seed0', set()),
    ('f2_smoke_lv_w', 'cfc_lrc_dense3_seed0', {'fixed_width'}),
    ('f2_smoke_lv_c8', 'lrc_ncp_seed0', {'lr_schedule'}),
    ('f2_smoke_duffing', 'cfc_lrc_eps1_ncp_seed0', {'elastance_one'}),
    ('f2_smoke_duffing', 'lrc_cal_duffing_ncp_seed0', {'init_slice_scale'}),
    ('f2_smoke_pa', 'ltc_minel_ncp_seed0', {'min_elapsed'}),
    ('f2_smoke_pa', 'cfc_lrc_cncp_random_seed0', {'cncp_random'}),
]


@pytest.mark.parametrize('name, arm_id, want', ACCEPT)
def test_runner_accepts_every_new_arm_kind(name, arm_id, want, tmp_path,
                                           monkeypatch):
    """The F2 smoke line, at 1 epoch (and a small data set), through the real
    runner: the JSON named by line['done'] is written, and the option's key
    appears in config.effective of its arm only."""
    cfg = load(str(SUITE / f'{name}.yaml'))
    line = next(x for x in expand(cfg) if x['id'] == arm_id)
    argv = line['argv'][4:]
    argv[argv.index('--epochs') + 1] = '1'
    if cfg['task'] == 'lv':
        argv += ['--n-trajectories', '6']
    else:
        stub = _write_stub(tmp_path / 'ConfLongDemo_JSI.txt')
        argv += ['--batch-size', '32', '--data-path', str(stub)]
    results = tmp_path / 'results'
    monkeypatch.setattr('sys.argv', [RUNNERS[cfg['task']], *argv,
                                     '--outdir', str(results)])
    {'lv': lv_runner, 'pa': pa_runner}[cfg['task']].main()
    eff = json.loads((results / line['done']).read_text())['config']['effective']
    assert _markers(eff) == want
    if arm_id == 'cfc_lrc_eps1_ncp_seed0':
        assert eff['build']['size'] == 23          # the cfc_lrc/ncp width


# --- control eval with robustness rows (#99) ----------------------------------

import yaml  # noqa: E402

from experiments.suite_manifest import ROBUSTNESS_RUNNER, SEED_STREAMS, load  # noqa: E402


def test_seed_streams_equal_the_robustness_script():
    from experiments import eval_robustness
    assert SEED_STREAMS == eval_robustness.SEED_STREAMS
    assert ROBUSTNESS_RUNNER.endswith('eval_robustness.py')


@pytest.mark.parametrize('name, n_records', [
    ('f2_control_eval_lv', 3), ('f2_control_eval_duffing', 5),
    ('f2_control_eval_pa', 5)])
def test_control_eval_lines(name, n_records):
    """Every control record gets the rebuild check and the robustness row,
    v1 written into each eval_robustness.py call, no training, outputs below
    results/<control stem>/; the seeds are those of the control YAML."""
    cfg = load(str(SUITE / f'{name}.yaml'))
    src = CONTROL_SOURCE[name]
    assert (cfg['robustness'], cfg['seed_streams']) == (True, 'v1')
    lines = expand(cfg)
    rob = [x for x in lines if ROBUSTNESS_RUNNER in x['argv']]
    assert len(rob) == n_records
    assert len(lines) == EVALS[name] == n_records * (1 if name.endswith('pa')
                                                     else 2)
    for x in rob:
        a = x['argv']
        assert a[a.index('--seed-streams') + 1] == 'v1'
        assert '--eval-only' not in a and x['outdir'] == src
        assert x['out'] == x['done'] and x['out'].endswith('.csv')
    if not name.endswith('pa'):
        for x in (y for y in lines if y not in rob):
            assert '--eval-only' in x['argv'] and x['done'].endswith('.json')
            assert '--seed-streams' not in x['argv']
    seeds = load(str(SUITE / f'{src}.yaml'))['seeds']
    assert len(seeds) == n_records
    assert [x['done'].split('seed')[1][0] for x in rob] == list(map(str, seeds))


def _control_eval_cfg(tmp_path, **over):
    src = tmp_path / 'src.yaml'
    src.write_text((SUITE / 'control_pa_k1.yaml').read_text())
    cfg = {'task': 'pa', 'stage': 'eval', 'sources': [
        {'config': str(src), 'records_dir': 'r'}], **over}
    p = tmp_path / 'ev.yaml'
    p.write_text(yaml.safe_dump(cfg))
    return str(p)


@pytest.mark.parametrize('over, msg', [
    ({'robustness': True}, 'needs seed_streams'),
    ({'robustness': True, 'seed_streams': 'v3'}, 'needs seed_streams'),
    ({'seed_streams': 'v1'}, 'applies with robustness'),
    ({'robustness': 'yes'}, 'true or false')])
def test_control_eval_rejections(tmp_path, over, msg):
    with pytest.raises(SystemExit, match=msg):
        load(_control_eval_cfg(tmp_path, **over))


def test_eval_without_robustness_is_unchanged(tmp_path):
    """No robustness key: the pa line has no --seed-streams, an lv source no
    second line (the T5 YAMLs keep their lines)."""
    lines = expand(load(_control_eval_cfg(tmp_path)))
    assert len(lines) == 5
    assert all('--seed-streams' not in x['argv'] for x in lines)
