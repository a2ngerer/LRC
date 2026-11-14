# tests/experiments/test_f3_plumbing.py
# F3 suite plumbing (issue a2ngerer/LRC#138), the runner side: the Person
# Activity test macro-F1 after every epoch and at the last one, the committee
# runner's seed-following corruption draws (--corruption-seed seed), and that
# --diagnostics leaves a Person Activity lrc/ncp fit unchanged. The manifest,
# grid-HPO and wandb parts sit with their modules' tests. Stub data only, CPU.
import json
import sys

import numpy as np
import pytest

import src.tasks.person_activity as pa_pkg
from experiments import run_committee_benchmark as co_runner
from experiments import run_person_activity_benchmark as pa_runner
from tests.benchmark.test_checkpoints import _pa_stub
from tests.benchmark.test_tracking import _fake_wandb

CorruptedBatches = co_runner.CorruptedBatches


def _pa(outdir, monkeypatch, cell, wiring, *extra, epochs=3):
    monkeypatch.setattr(pa_pkg, 'load_person_activity', _pa_stub)
    monkeypatch.setattr(sys, 'argv', [
        'run_person_activity_benchmark.py', '--wiring', wiring, '--cell', cell,
        '--size', '8', '--epochs', str(epochs), '--outdir', str(outdir),
        *extra])
    pa_runner.main()
    return json.loads((outdir / f'{cell}_{wiring}_seed0.json').read_text())


def test_pa_records_the_test_macro_f1_of_every_epoch(tmp_path, monkeypatch):
    rec = _pa(tmp_path, monkeypatch, 'gru', 'dense3', epochs=4)
    curve = rec['test_macro_f1_curve']
    assert len(curve) == len(rec['val_macro_f1_curve']) == 4
    assert rec['test_macro_f1_last'] == curve[-1]
    # the best-validation epoch keeps its value under the old key
    assert curve[rec['best_val_epoch'] - 1] == rec['test_macro_f1']
    # an addition to the record only: the hashed config is as before
    eff = rec['config']['effective']
    assert not {'test_macro_f1_curve', 'test_macro_f1_last'} & set(eff)


def test_pa_diagnostics_leave_lrc_ncp_training_unchanged(tmp_path,
                                                         monkeypatch):
    """Family A trains the lrc cells on the NCP; the RQ4 callback only reads."""
    off = _pa(tmp_path / 'off', monkeypatch, 'lrc', 'ncp', epochs=2)
    on = _pa(tmp_path / 'on', monkeypatch, 'lrc', 'ncp', '--diagnostics',
             epochs=2)
    assert 'diagnostics' not in off and 'diagnostics' in on
    for key in ('val_macro_f1_curve', 'val_accuracy_curve',
                'train_accuracy_curve', 'test_macro_f1_curve',
                'test_macro_f1', 'test_macro_f1_last', 'test_accuracy',
                'best_val_epoch', 'test_confusion_by_recording'):
        assert on[key] == off[key], key


# --- committee: --corruption-seed seed ----------------------------------------

def _co(outdir, monkeypatch, drawn, wiring, seed, *extra):
    """One 1-epoch committee fit on the stub; appends the CorruptedBatches it
    trained on to `drawn`. Returns the result record."""
    def spy(*a, **k):                  # patched over the class, never over a spy
        drawn.append(CorruptedBatches(*a, **k))
        return drawn[-1]
    monkeypatch.setattr(co_runner, 'CorruptedBatches', spy)
    monkeypatch.setattr(co_runner, 'load_person_activity', _pa_stub)
    monkeypatch.setattr(sys, 'argv', [
        'run_committee_benchmark.py', '--wiring', wiring, '--n-columns', '2',
        '--size', '8', '--epochs', '1', '--seq-len', '32', '--seed',
        str(seed), '--outdir', str(outdir), *extra])
    co_runner.main()
    return outdir, [json.loads(p.read_text()) for p in outdir.glob('*.json')]


def test_corruption_seed_follows_the_run_seed(tmp_path, monkeypatch):
    runs = _fake_wandb(monkeypatch)
    drawn = []
    fits = {}
    for wiring, seed in (('ncp', 3), ('dense', 3), ('ncp', 4)):
        out = tmp_path / f'{wiring}_{seed}'
        _, (rec,) = _co(out, monkeypatch, drawn, wiring, seed,
                        '--corruption-seed', 'seed', '--wandb')
        (path,) = out.glob('*.json')
        fits[wiring, seed] = (path.name, rec)
    # the integer reaches the name, config.effective, the record, the wandb
    # config and the draws
    for (wiring, seed), (name, rec), run, batches in zip(
            fits, fits.values(), runs, drawn):
        assert f'_cs{seed}_seed{seed}' in name
        assert rec['corruption_seed'] == seed
        assert rec['config']['effective']['corruption_seed'] == seed
        assert run.init['config']['corruption_seed'] == seed
        assert batches.seed == seed
    # two arms with one seed share the draw, two seeds do not
    views = [b.corrupted(0)[2] for b in drawn]
    np.testing.assert_array_equal(views[0], views[1])
    assert not np.array_equal(views[0], views[2])


def test_corruption_seed_default_and_integer_are_unchanged(tmp_path,
                                                           monkeypatch):
    drawn = []
    out = tmp_path / 'default'
    _, (rec,) = _co(out, monkeypatch, drawn, 'ncp', 5)
    (path,) = out.glob('*.json')
    assert '_cs0_seed5' in path.name and rec['corruption_seed'] == 0
    out = tmp_path / 'int'
    _, (rec,) = _co(out, monkeypatch, drawn, 'ncp', 5, '--corruption-seed', '2')
    (path,) = out.glob('*.json')
    assert '_cs2_seed5' in path.name and rec['corruption_seed'] == 2
    assert [b.seed for b in drawn] == [0, 2]


def test_corruption_seed_rejects_other_words(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['run_committee_benchmark.py',
                                      '--corruption-seed', 'run'])
    with pytest.raises(SystemExit):
        co_runner.main()
    assert "an int or the word 'seed'" in capsys.readouterr().err
