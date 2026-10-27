# tests/experiments/test_run_grid_hpo.py
# Fixed-grid HPO driver (issue a2ngerer/LRC#22): stage 1/2 selection, the edge
# rule, the tie-break, resume and hpo_selection.json, driven through a fake
# runner (fake_grid_runner.sh) that reads its metrics from a score table; plus
# the tuning-seed guard and the runners' --clipnorm reaching the optimizer as
# global_clipnorm (tiny fits, issue #58).
import json
import subprocess
import sys
from pathlib import Path

import pytest

import src.tasks.person_activity as pa_pkg
from experiments import run_committee_benchmark as co_runner
from experiments import run_lotka_volterra_benchmark as lv_runner
from experiments import run_person_activity_benchmark as pa_runner
from experiments.run_grid_hpo import GRID_LR, main
from tests.benchmark.test_checkpoints import T, _capture_builds, _pa_stub

FAKE = Path(__file__).with_name('fake_grid_runner.sh')


def _hpo(tmp_path, monkeypatch, table, *extra, task='lv', seeds=(100, 101)):
    """Run the driver on the fake runner. table: {(lr, clip): (score_s0,
    score_s1[, tie_break])} for the tuning seeds (default 100 and 101). Returns the
    selection and the logged fits ([lr, clip, seed, args...] per fit)."""
    rows = [f"{lr:g} {'off' if clip is None else f'{clip:g}'} {seed} "
            f"{scores[i]} {scores[2] if len(scores) > 2 else 0.5}"
            for (lr, clip), scores in table.items()
            for i, seed in enumerate(seeds)]
    (tmp_path / 'scores.txt').write_text('\n'.join(rows) + '\n')
    monkeypatch.setenv('FAKE_SCORES', str(tmp_path / 'scores.txt'))
    monkeypatch.setenv('FAKE_LOG', str(tmp_path / 'fits.log'))
    out = tmp_path / 'hpo'
    main(['--task', task, '--cell', 'cfc_lrc', '--wiring', 'dense3',
          '--tuning-seeds', *map(str, seeds), '--outdir', str(out),
          '--runner', f'bash {FAKE}', *extra])
    log = tmp_path / 'fits.log'
    fits = [line.split() for line in log.read_text().splitlines()]
    return json.loads((out / 'hpo_selection.json').read_text()), fits


def _table(best=()):
    """The full 4 x 2 grid at score 0, the configs in best overridden."""
    return {**{(lr, clip): (0, 0) for lr in GRID_LR for clip in (None, 1.0)},
            **dict(best)}


def test_stage1_top2_then_mean_over_both_seeds(tmp_path, monkeypatch):
    # stage 1 (seed 100) ranks 1e-3/off first, the mean over seeds 100 and 101
    # flips the order: 3e-3/clip 1.0 wins (54 > 50).
    table = _table({(1e-3, None): (60, 40), (3e-3, 1.0): (58, 50)})
    sel, fits = _hpo(tmp_path, monkeypatch, table, '--system', 'duffing',
                     '--param-budget', '4000', '--epochs', '2')
    assert len(fits) == 8 + 2
    assert sorted(tuple(f[:3]) for f in fits if f[2] == '101') == [
        ('0.001', 'off', '101'), ('0.003', '1', '101')]
    assert (sel['lr'], sel['clipnorm']) == (3e-3, 1.0)
    assert [(r['lr'], r['clipnorm'], r['score']) for r in sel['ranking']] == [
        (3e-3, 1.0, 54), (1e-3, None, 50)]
    assert sel['ranking'][0]['scores'] == {'100': 58, '101': 50}
    # the fields aggregate_results.py --hpo-dir (#29) reads
    assert (sel['task'], sel['system'], sel['cell'], sel['wiring']) == (
        'lv', 'duffing', 'cfc_lrc', 'dense3')
    assert sel['grid_extended'] is False and sel['lr_at_grid_edge'] is False
    assert sel['select_on'] == 'rollout_metrics_val.vpt_multi_median'
    assert sel['tie_break'] == 'rollout_metrics_val.nrmse_32_anchor_8'
    assert sel['tuning_seeds'] == [100, 101]
    assert sel['data_seeds'] == [1100, 1101]
    assert len(sel['candidates']) == 10
    assert sel['runner_args'] == ['--epochs', '2']
    # every fit in its own folder; LV tuning draws; flags reach the runner
    assert (tmp_path / 'hpo' / 'lr0.003_clip1_seed101' / 'fake.json').is_file()
    for f in fits:
        args = ' '.join(f[3:])
        assert '--data-seed-offset' in args and '--param-budget 4000' in args
        assert '--epochs 2' in args and '--system duffing' in args
        assert ('--clipnorm 1' in args) == (f[1] == '1')
    # resume: a rerun reuses every result JSON and selects the same config
    sel2, fits2 = _hpo(tmp_path, monkeypatch, table, '--system', 'duffing',
                       '--param-budget', '4000', '--epochs', '2')
    assert len(fits2) == len(fits) and sel2['ranking'] == sel['ranking']


@pytest.mark.parametrize('edge, ext, ext_scores, lr, at_edge', [
    (1e-2, 3e-2, (90, 85), 3e-2, True),     # extension wins, still at the edge
    (1e-4, 3e-5, (2, 2), 1e-4, False),      # extension loses, edge resolved
])
def test_edge_rule_extends_once(edge, ext, ext_scores, lr, at_edge,
                                tmp_path, monkeypatch):
    table = _table()
    table[(edge, None)], table[(edge, 1.0)] = (80, 79), (70, 70)
    table[(ext, None)], table[(ext, 1.0)] = ext_scores, (3, 3)
    sel, fits = _hpo(tmp_path, monkeypatch, table)
    assert sel['grid_extended'] is True and sel['lr_at_grid_edge'] is at_edge
    assert sel['lr'] == lr and sel['grid_lr'] == sorted([*GRID_LR, ext])
    # stage 1 reruns the new point only (both clip settings), no fit twice
    assert sorted(tuple(f[:3]) for f in fits if f[0] == f'{ext:g}'
                  and f[2] == '100') == [(f'{ext:g}', '1', '100'),
                                       (f'{ext:g}', 'off', '100')]
    assert len({tuple(f[:3]) for f in fits}) == len(fits)
    assert len(fits) == 8 + 2 + 2 + (1 if lr == ext else 0)


def test_tie_break_lower_nrmse32_wins(tmp_path, monkeypatch):
    # equal multi-anchor VPT everywhere in the top 2: the lower
    # nrmse_32_anchor_8 wins although grid order (stable sort) would keep 1e-3
    # first
    table = _table({(1e-3, None): (60, 60, 0.3),
                    (3e-3, None): (60, 60, 0.2)})
    sel, _ = _hpo(tmp_path, monkeypatch, table)
    assert [r['lr'] for r in sel['ranking']] == [3e-3, 1e-3]
    assert sel['ranking'][0]['tie_break'] == pytest.approx(0.2)


def test_pa_selects_on_val_macro_f1(tmp_path, monkeypatch):
    sel, fits = _hpo(tmp_path, monkeypatch,
                     _table({(1e-3, 1.0): (0.6, 0.7)}), task='pa')
    assert (sel['lr'], sel['clipnorm']) == (1e-3, 1.0)
    assert sel['select_on'] == 'val_macro_f1' and sel['tie_break'] is None
    assert sel['system'] is None                 # #29 labels the task 'pa'
    assert not any('--data-seed-offset' in f for f in fits)


def test_select_on_override_minimize(tmp_path, monkeypatch):
    table = _table({(3e-3, None): (0, 0, 0.1)})  # lowest nrmse_32
    sel, _ = _hpo(tmp_path, monkeypatch, table, '--select-on',
                  'rollout_metrics_val.nrmse_32_median', '--minimize')
    assert (sel['lr'], sel['maximize']) == (3e-3, False)
    assert sel['tie_break'] == 'rollout_metrics_val.nrmse_32_anchor_8'


def test_select_on_keeps_the_tie_break(tmp_path, monkeypatch):
    # --select-on no longer disables the tie-break (#62): equal scores, the
    # lower nrmse_32_anchor_8 wins over grid order
    table = _table({(1e-3, None): (60, 60, 0.3),
                    (3e-3, None): (60, 60, 0.2)})
    sel, _ = _hpo(tmp_path, monkeypatch, table, '--select-on',
                  'rollout_metrics_val.vpt_median')
    assert sel['select_on'] == 'rollout_metrics_val.vpt_median'
    assert [r['lr'] for r in sel['ranking']] == [3e-3, 1e-3]


def test_failed_fit_aborts(tmp_path, monkeypatch):
    table = _table()
    del table[(1e-3, 1.0)]                       # fake runner exits 1
    with pytest.raises(subprocess.CalledProcessError):
        _hpo(tmp_path, monkeypatch, table)


def test_tuning_seeds_must_avoid_final_seeds(tmp_path, monkeypatch, capsys):
    # a tuning seed in 0..9 would share initialisation (and on PA the data)
    # with a final replicate: refused before any fit unless overridden
    with pytest.raises(SystemExit):
        _hpo(tmp_path, monkeypatch, _table(), seeds=(9, 101))
    assert 'final seed range 0..9' in capsys.readouterr().err
    assert not (tmp_path / 'fits.log').exists()
    sel, _ = _hpo(tmp_path, monkeypatch, _table({(1e-3, None): (1, 1)}),
                  '--allow-seed-overlap', seeds=(0, 1))
    assert sel['tuning_seeds'] == [0, 1] and sel['runner_args'] == []


def _assert_global_clip(built, path, value):
    opt = built[-1][0].optimizer
    assert opt.global_clipnorm == value and opt.clipnorm is None
    rec = json.loads(path.read_text())
    assert (rec['clipnorm'], rec['clip_mode']) == (value, 'global')


def test_runners_apply_global_clipnorm(tmp_path, monkeypatch):
    built = _capture_builds(monkeypatch, lv_runner, 'build_lotka_volterra_model')
    monkeypatch.setattr(sys, 'argv', [
        'run_lotka_volterra_benchmark.py', '--wiring', 'dense', '--cell', 'gru',
        '--size', '4', '--epochs', '1', '--n-trajectories', '10',
        '--seq-len', '8', '--anchors', '2,4', '--horizon', '4',
        '--clipnorm', '1.0', '--outdir', str(tmp_path)])
    lv_runner.main()
    _assert_global_clip(built, tmp_path / 'gru_dense_seed0.json', 1.0)

    monkeypatch.setattr(pa_pkg, 'load_person_activity', _pa_stub)
    built = _capture_builds(monkeypatch, pa_pkg, 'build_person_activity_model')
    monkeypatch.setattr(sys, 'argv', [
        'run_person_activity_benchmark.py', '--wiring', 'dense', '--cell', 'gru',
        '--size', '4', '--epochs', '1', '--clipnorm', '0.5',
        '--outdir', str(tmp_path / 'pa')])
    pa_runner.main()
    _assert_global_clip(built, tmp_path / 'pa' / 'gru_dense_seed0.json', 0.5)

    monkeypatch.setattr(co_runner, 'load_person_activity', _pa_stub)
    built = _capture_builds(monkeypatch, co_runner, 'build_committee_model')
    monkeypatch.setattr(sys, 'argv', [
        'run_committee_benchmark.py', '--n-columns', '2', '--size', '8',
        '--epochs', '1', '--seq-len', str(T), '--drop-fracs', '0.0',
        '--drop-seeds', '0', '--clipnorm', '2.0',
        '--outdir', str(tmp_path / 'co')])
    co_runner.main()
    (path,) = (tmp_path / 'co').glob('*.json')
    _assert_global_clip(built, path, 2.0)
