# tests/benchmark/test_tracking.py
# Suite-level wandb metadata of the sequence runners (issue a2ngerer/LRC#53):
# every runner works with wandb absent (the tracking helpers are no-ops), a
# fake wandb module records what a tracked fit sends (project, group, job_type,
# name, tags, config, summary), and config.effective carries protocol_version
# and a config_hash that survives the JSON round trip and tells fits apart that
# differ only in a training or evaluation knob. Synthetic data only.
import json
import sys
import types

import pytest
import tensorflow as tf

from experiments import run_committee_benchmark as co_runner
from experiments import run_lotka_volterra_benchmark as lv_runner
from experiments import run_person_activity_benchmark as pa_runner
import src.tasks.person_activity as pa_pkg
from src.benchmark.checkpoints import (PROTOCOL_VERSION, config_hash,
                                       stamp_protocol)
from src.benchmark.tracking import track
from tests.benchmark.test_checkpoints import _lv_argv, _pa_stub

# the run config keys every sequence runner sends (#53)
SUITE_CONFIG_KEYS = ("cell", "wiring", "system", "seed", "data_seed", "lr",
                     "clipnorm", "param_budget", "params_effective",
                     "ode_unfolds", "backbone_layers", "elastance_type",
                     "git_sha", "protocol_version", "config_hash")


def _run_lv(tmp_path, monkeypatch, *extra):
    monkeypatch.setattr(sys, 'argv', _lv_argv(
        tmp_path, '--wiring', 'ncp', '--cell', 'cfc_lrc', '--size', '8',
        '--seed', '1', '--wandb', *extra))
    lv_runner.main()
    return tmp_path / 'cfc_lrc_ncp_seed1.json'


def _run_pa(tmp_path, monkeypatch, *extra):
    monkeypatch.setattr(pa_pkg, 'load_person_activity', _pa_stub)
    monkeypatch.setattr(sys, 'argv', [
        'run_person_activity_benchmark.py', '--wiring', 'dense3', '--cell',
        'gru', '--size', '4', '--epochs', '2', '--outdir', str(tmp_path),
        '--wandb', *extra])
    pa_runner.main()
    return tmp_path / 'gru_dense3_seed0.json'


def _run_co(tmp_path, monkeypatch, *extra):
    monkeypatch.setattr(co_runner, 'load_person_activity', _pa_stub)
    monkeypatch.setattr(sys, 'argv', [
        'run_committee_benchmark.py', '--wiring', 'ncp', '--n-columns', '2',
        '--size', '8', '--epochs', '2', '--seq-len', '32', '--vote-lambda',
        '0.5', '--outdir', str(tmp_path), '--wandb', *extra])
    co_runner.main()
    (path,) = tmp_path.glob('*.json')
    return path


# runner -> (its run, wandb job_type task)
RUNNERS = {'lv': (_run_lv, 'periodic_predator_prey'),
           'pa': (_run_pa, 'person_activity'),
           'co': (_run_co, 'committee')}


def _fake_wandb(monkeypatch):
    """Install a stand-in wandb package; returns the list of its runs."""
    runs = []

    class Run:
        def __init__(self, **init):
            self.init, self.summary, self.finished = init, {}, False

        def finish(self):
            self.finished = True

    def init(**kw):
        runs.append(Run(**kw))
        return runs[-1]

    wandb = types.ModuleType('wandb')
    wandb.init = init
    integration = types.ModuleType('wandb.integration')
    keras = types.ModuleType('wandb.integration.keras')
    keras.WandbMetricsLogger = tf.keras.callbacks.Callback
    wandb.integration, integration.keras = integration, keras
    for name, mod in (('wandb', wandb), ('wandb.integration', integration),
                      ('wandb.integration.keras', keras)):
        monkeypatch.setitem(sys.modules, name, mod)
    return runs


def _failing_init(monkeypatch, exc=RuntimeError):
    """Fake wandb whose init raises, like a service that does not come up."""
    _fake_wandb(monkeypatch)

    def init(**kw):
        raise exc('Failed to read port info after 30.0 seconds')

    monkeypatch.setattr(sys.modules['wandb'], 'init', init)


@pytest.mark.parametrize('runner', sorted(RUNNERS))
def test_runner_works_without_wandb(runner, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'wandb', None)   # import wandb fails
    path = RUNNERS[runner][0](tmp_path, monkeypatch)
    eff = json.loads(path.read_text())['config']['effective']
    assert eff['protocol_version'] == PROTOCOL_VERSION == 'F1-2026-10-26'
    # the stored hash is the hash of the stored (round-tripped) config
    assert eff['config_hash'] == config_hash(eff)


@pytest.mark.parametrize('runner', sorted(RUNNERS))
def test_runner_tracks_suite_metadata(runner, tmp_path, monkeypatch):
    runs = _fake_wandb(monkeypatch)
    monkeypatch.setenv('SUITE_CAMPAIGN', 'canary_hpo')
    monkeypatch.setenv('SUITE_STAGE', 'hpo')
    monkeypatch.setenv('RUN_GIT_SHA', 'abc123')
    run_fit, task = RUNNERS[runner]
    path = run_fit(tmp_path, monkeypatch)
    record = json.loads(path.read_text())
    eff = record['config']['effective']
    (run,) = runs
    assert run.finished
    assert {k: run.init[k] for k in ('project', 'group', 'job_type', 'name',
                                     'tags')} == {
        'project': 'thesis-benchmarks', 'group': 'canary_hpo',
        'job_type': f'{task}/hpo', 'name': path.stem,
        'tags': ['protocol:F1-2026-10-26']}
    config = run.init['config']
    assert set(SUITE_CONFIG_KEYS) <= set(config)
    for k in ('cell', 'wiring', 'seed', 'lr', 'clipnorm', 'param_budget',
              'params_effective'):
        assert config[k] == record[k], k
    for k in ('ode_unfolds', 'backbone_layers', 'elastance_type',
              'protocol_version', 'config_hash'):
        assert config[k] == eff[k], k
    assert config['git_sha'] == 'abc123'
    summary = run.summary
    if runner == 'lv':
        assert config['system'] == record['system'] == task
        assert config['data_seed'] == record['data_seed']
        test = record['rollout_metrics_test']
        assert summary['val_vpt_multi_median'] == (
            record['rollout_metrics_val']['vpt_multi_median'])
        assert summary['test_vpt_multi_median'] == test['vpt_multi_median']
        for k in ('rate_ok', 'rate_unbounded', 'rate_bounded_wrong',
                  'non_finite'):
            assert summary[f'test_{k}'] == test[k]
    elif runner == 'pa':
        assert config['system'] is None
        for k in ('val_macro_f1', 'test_macro_f1', 'best_val_epoch'):
            assert summary[k] == record[k]
    else:
        assert config['system'] is None
        for k in ('n_columns', 'rho_grid', 'fault_counts', 'vote_lambda',
                  'consensus', 'view_mode', 'noise_frac', 'train_corruption',
                  'corruption_seed', 'val_seed', 'test_seeds'):
            assert config[k] == record[k], k
        assert config['vote_lambda'] == 0.5
        assert summary['test_macro_f1_mean_grid'] == (
            record['test_macro_f1_mean_grid'])
        assert [summary[f'test_macro_f1_m{m}'] for m in
                record['fault_counts']] == record['test_macro_f1_by_fault']
        assert [summary[f'test_per_position_macro_f1_auc_m{m}'] for m in
                record['fault_counts']] == record['per_position_macro_f1_auc']


def test_track_yields_none_when_wandb_init_raises(monkeypatch, capsys):
    _failing_init(monkeypatch)
    seen = []
    with track(True, group='g', job_type='j', name='n', config={}) as run:
        seen.append(run)                                # the body runs
    assert seen == [None]
    err = capsys.readouterr().err
    assert err.count('wandb.init failed') == 1
    assert 'port info' in err


def test_track_does_not_swallow_a_keyboard_interrupt(monkeypatch):
    _failing_init(monkeypatch, KeyboardInterrupt)
    with pytest.raises(KeyboardInterrupt):
        with track(True, group='g', job_type='j', name='n', config={}):
            pytest.fail('body must not run')


@pytest.mark.parametrize('runner', sorted(RUNNERS))
def test_runner_survives_wandb_init_failure(runner, tmp_path, monkeypatch,
                                            capsys):
    run_fit = RUNNERS[runner][0]
    monkeypatch.setitem(sys.modules, 'wandb', None)     # untracked reference
    ref = json.loads(run_fit(tmp_path / 'ref', monkeypatch).read_text())
    _failing_init(monkeypatch)
    capsys.readouterr()
    path = run_fit(tmp_path / 'fail', monkeypatch)      # trains, writes JSON
    assert capsys.readouterr().err.count('wandb.init failed') == 1
    eff = json.loads(path.read_text())['config']['effective']
    assert eff['config_hash'] == ref['config']['effective']['config_hash']
    assert eff['config_hash'] == config_hash(eff)


def test_adhoc_run_falls_back_to_the_cell(tmp_path, monkeypatch):
    runs = _fake_wandb(monkeypatch)
    monkeypatch.delenv('SUITE_CAMPAIGN', raising=False)
    monkeypatch.delenv('SUITE_STAGE', raising=False)
    _run_pa(tmp_path, monkeypatch)
    assert (runs[0].init['group'], runs[0].init['job_type']) == (
        'gru', 'person_activity/adhoc')


def test_config_hash_ignores_its_own_key_and_tracks_the_config():
    eff = stamp_protocol({'task': 'lotka_volterra', 'build': {'lr': 1e-3}})
    assert eff['protocol_version'] == PROTOCOL_VERSION
    assert eff['config_hash'] == config_hash(eff) == config_hash(
        {k: v for k, v in eff.items() if k != 'config_hash'})
    other = stamp_protocol({'task': 'lotka_volterra', 'build': {'lr': 3e-3}})
    assert other['config_hash'] != eff['config_hash']


# per runner: knobs outside the builder (training, evaluation, corruption)
# that config_hash must tell apart; the rerun with no knob keeps the hash
HASH_KNOBS = {
    'lv': ((), (), ('--clipnorm', '1'), ('--epochs', '1'),
           ('--batch-size', '2'), ('--anchors', '2,3')),
    'pa': ((), (), ('--clipnorm', '1'), ('--epochs', '1'),
           ('--batch-size', '2')),
    'co': ((), (), ('--clipnorm', '1'), ('--epochs', '1'),
           ('--val-seed', '7'))}


@pytest.mark.parametrize('runner', sorted(HASH_KNOBS))
def test_config_hash_tells_training_and_evaluation_knobs_apart(
        runner, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'wandb', None)
    run_fit = RUNNERS[runner][0]
    effs = [json.loads(run_fit(tmp_path / str(i), monkeypatch, *knob)
                       .read_text())['config']['effective']
            for i, knob in enumerate(HASH_KNOBS[runner])]
    hashes = [e['config_hash'] for e in effs]
    assert hashes[0] == hashes[1]                       # identical rerun
    assert len(set(hashes[1:])) == len(hashes) - 1      # every knob differs
    assert all(e['config_hash'] == config_hash(e) for e in effs)
    if runner == 'lv':
        # --eval-only with other evaluation knobs records them, new hash
        path = tmp_path / '0' / 'cfc_lrc_ncp_seed1.json'
        eff = lv_runner.eval_only(str(path), str(tmp_path / 'ev'), (2, 3), 4,
                                  0.5)['config']['effective']
        assert eff['anchors'] == [2, 3]
        assert eff['config_hash'] == config_hash(eff) == hashes[5]
