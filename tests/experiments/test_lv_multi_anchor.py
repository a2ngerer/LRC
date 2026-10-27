# tests/experiments/test_lv_multi_anchor.py
# Issue a2ngerer/LRC#62: the LV runner's multi-anchor endpoint flags
# (--anchors/--horizon/--theta, defaults 8,16,24,32 / 96 / 0.5) and
# --eval-only, which recomputes every evaluation field from weights.h5 and
# config.effective (1e-3 relative) without training.
import json
import shutil
import sys

import pytest

from experiments import run_lotka_volterra_benchmark as runner

ANCHOR_KEYS = ('vpt_multi_median', 'vpt_anchor_2', 'vpt_anchor_4',
               'nrmse_32_anchor_2', 'nrmse_32_anchor_4', 'theta_sensitivity')


def _train(tmp_path, monkeypatch, wiring, *extra):
    monkeypatch.setattr(sys, 'argv', [
        'run_lotka_volterra_benchmark.py', '--wiring', wiring, '--cell',
        'cfc_lrc', '--size', '4', '--epochs', '2', '--n-trajectories', '6',
        '--seq-len', '8', '--anchors', '2,4', '--horizon', '4',
        '--batch-size', '4', '--outdir', str(tmp_path), *extra])
    runner.main()
    (path,) = tmp_path.glob('*.json')
    return path


def _eval_only(monkeypatch, path, *extra):
    monkeypatch.setattr(sys, 'argv', [
        'run_lotka_volterra_benchmark.py', '--eval-only', str(path),
        '--anchors', '2,4', '--horizon', '4', *extra])
    runner.main()


def _assert_close(old, new, key=''):
    if isinstance(old, dict):
        assert old.keys() == new.keys(), key
        for k in old:
            _assert_close(old[k], new[k], f'{key}.{k}')
    elif isinstance(old, list):
        assert len(old) == len(new), key
        for i, (o, n) in enumerate(zip(old, new)):
            _assert_close(o, n, f'{key}[{i}]')
    elif isinstance(old, float):
        assert new == pytest.approx(old, rel=1e-3, abs=1e-6, nan_ok=True), key
    else:
        assert old == new, key


def test_flag_defaults_are_the_frozen_protocol(monkeypatch):
    seen = []
    monkeypatch.setattr(runner, 'eval_only', lambda *a: seen.append(a))
    monkeypatch.setattr(sys, 'argv', ['run_lotka_volterra_benchmark.py',
                                      '--eval-only', 'x.json'])
    runner.main()
    assert seen == [('x.json', None, (8, 16, 24, 32), 96, 0.5)]


@pytest.mark.parametrize('wiring', ['ncp', 'tbt_cncp_concat'])
def test_eval_only_reproduces_every_field(wiring, tmp_path, monkeypatch):
    """Forced view (ncp) and the closed_loop_rollout path (tbt): the JSON
    carries the anchor keys on both splits and --eval-only into another folder
    recomputes every evaluation field, training fields untouched."""
    path = _train(tmp_path / 'fit', monkeypatch, wiring)
    rec = json.loads(path.read_text())
    for split in ('val', 'test'):
        m = rec[f'rollout_metrics_{split}']
        assert set(ANCHOR_KEYS) <= set(m) and m['anchors'] == [2, 4]
    _eval_only(monkeypatch, path, '--outdir', str(tmp_path / 're'))
    again = json.loads((tmp_path / 're' / path.name).read_text())
    assert again['eval_only']['max_rel_dev'] <= 1e-3
    assert (tmp_path / 're' / again['weights_path']).resolve() == (
        path.parent / rec['weights_path']).resolve()
    del again['eval_only'], again['weights_path'], rec['weights_path']
    _assert_close(rec, again)
    assert json.loads(path.read_text()) == {**rec, 'weights_path':
                                            path.name[:-5] + '.weights.h5'}


def test_eval_only_adds_the_anchor_keys_in_place(tmp_path, monkeypatch):
    """A pre-#62 JSON (no anchor keys) re-evaluates without retraining."""
    path = _train(tmp_path, monkeypatch, 'dense3', '--forcing', 'mixed')
    rec = json.loads(path.read_text())
    stale = json.loads(path.read_text())
    for split in ('val', 'test'):
        for k in ('anchors', 'anchor_horizon', 'anchor_theta', *ANCHOR_KEYS):
            del stale[f'rollout_metrics_{split}'][k]
    path.write_text(json.dumps(stale))
    _eval_only(monkeypatch, path)
    again = json.loads(path.read_text())
    assert again['eval_only']['max_rel_dev'] <= 1e-3
    del again['eval_only']
    _assert_close(rec, again)


def test_anchor_flags_are_validated(tmp_path, monkeypatch):
    for extra in (['--seq-len', '8'],                       # 32 + 96 > 8
                  ['--seq-len', '8', '--anchors', '2,4', '--horizon', '5'],
                  ['--anchors', '-1,8'], ['--horizon', '0']):
        monkeypatch.setattr(sys, 'argv', [
            'run_lotka_volterra_benchmark.py', '--wiring', 'dense',
            '--outdir', str(tmp_path), *extra])
        with pytest.raises(SystemExit):
            runner.main()
    monkeypatch.setattr(sys, 'argv', ['run_lotka_volterra_benchmark.py'])
    with pytest.raises(SystemExit):                         # no --wiring
        runner.main()
    assert not list(tmp_path.glob('*'))


def test_eval_only_refuses_anchors_beyond_the_trajectory(tmp_path,
                                                         monkeypatch):
    path = _train(tmp_path / 'fit', monkeypatch, 'dense')
    copy = tmp_path / path.name
    shutil.copy(path, copy)
    monkeypatch.setattr(sys, 'argv', [
        'run_lotka_volterra_benchmark.py', '--eval-only', str(copy)])
    with pytest.raises(SystemExit, match='exceeds seq_len 8'):
        runner.main()
