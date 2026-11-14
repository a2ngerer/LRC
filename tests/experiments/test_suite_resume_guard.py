"""Resume guard of cluster/submit_suite.sh --resume (#143): suite_manifest
check_resume / --check-resume, and the shell glue against a fake ssh (no
network)."""
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

from experiments.suite_manifest import check_resume, expand, load, main, write

ROOT = Path(__file__).resolve().parents[2]
BASE = {'task': 'lv', 'system': 'duffing', 'stage': 'final',
        'cells': ['lrc'], 'wirings': ['ncp'], 'seeds': [0, 1],
        'ode_unfolds': 6, 'dt_scale': 'auto',
        'fixed': {'lr': 0.001, 'clipnorm': None}}


def _yaml(tmp_path, name, **over):
    path = tmp_path / f'{name}.yaml'
    path.write_text(yaml.safe_dump({**BASE, **over}))
    return str(path)


def _manifest(tmp_path, name, group='default', wandb=False, **over):
    """manifest_<group>.jsonl of a campaign built from BASE + over."""
    cfg = load(_yaml(tmp_path, name, **over))
    out = tmp_path / name
    out.mkdir()
    write(cfg, expand(cfg, wandb=wandb), str(out))
    return str(out / f'manifest_{group}.jsonl')


@pytest.mark.parametrize('old_over, new_over, shown', [
    ({'extra_args': ['--optimizer', 'rmsprop']},
     {'extra_args': ['--optimizer', 'rmsprop_legacy']}, 'rmsprop_legacy'),
    ({'epochs': 50}, {'epochs': 200}, '200')])
def test_refuses_variants_that_share_result_names(tmp_path, old_over,
                                                  new_over, shown):
    """Two runs that differ only in the optimizer or the epoch count write
    the same file names: the resume names both lines and refuses."""
    old = _manifest(tmp_path, 'first', **old_over)
    new = _manifest(tmp_path, 'second', **new_over)
    with pytest.raises(SystemExit, match='resume refused') as err:
        check_resume(old, new)
    text = str(err.value)
    assert 'lrc_ncp_seed0 (run folder manifest) and lrc_ncp_seed0 (new' in text
    assert shown in text


def test_passes_identical_manifests(tmp_path):
    old = _manifest(tmp_path, 'first')
    assert check_resume(old, _manifest(tmp_path, 'second')) is None


def test_passes_a_longer_seed_list(tmp_path):
    old = _manifest(tmp_path, 'first')
    assert check_resume(old, _manifest(tmp_path, 'second',
                                       seeds=[0, 1, 2, 3])) is None
    # and the other way round: a smaller group
    assert check_resume(_manifest(tmp_path, 'third', seeds=[0, 1, 2, 3]),
                        old) is None


def test_passes_a_second_yaml_with_a_separate_seed_list(tmp_path):
    """E13/E16a: seeds 10-19 and 20-29 in one run folder, the second YAML
    replaces the manifest after the first array has ended."""
    first = _manifest(tmp_path, 'first', seeds=list(range(10, 20)))
    second = _manifest(tmp_path, 'second', seeds=list(range(20, 30)))
    assert check_resume(first, second) is None
    assert check_resume(second, first) is None


def test_passes_a_group_that_is_new_or_has_no_old_manifest(tmp_path):
    new = _manifest(tmp_path, 'second', group='default')
    assert check_resume(str(tmp_path / 'absent.jsonl'), new) is None
    empty = tmp_path / 'empty.jsonl'
    empty.write_text('')
    assert check_resume(str(empty), new) is None


def test_final_lines_follow_the_fetched_hpo_run(tmp_path):
    """argv of a final line takes lr and clipnorm from the arm's
    hpo_selection.json: the same fetched run re-expands to the same argv (a
    legitimate resume), another selection does not."""
    def fetched(name, lr):
        d = tmp_path / name / 'lrc_ncp'
        d.mkdir(parents=True)
        (d / 'hpo_selection.json').write_text(json.dumps({
            'task': 'lv', 'system': 'duffing', 'cell': 'lrc',
            'wiring': 'ncp', 'lr': lr, 'clipnorm': None}))
        return str(tmp_path / name)

    hpo = {'fixed': None}
    old = _manifest(tmp_path, 'first', hpo_dir=fetched('hpo_a', 0.003), **hpo)
    assert check_resume(old, _manifest(
        tmp_path, 'again', hpo_dir=str(tmp_path / 'hpo_a'), **hpo)) is None
    with pytest.raises(SystemExit, match=r'run folder: 0\.003; only in the '
                                         r'new line: 0\.01'):
        check_resume(old, _manifest(
            tmp_path, 'other', hpo_dir=fetched('hpo_b', 0.01), **hpo))


def test_refuses_a_different_wandb_flag(tmp_path):
    old = _manifest(tmp_path, 'first', wandb=True)
    with pytest.raises(SystemExit, match='--wandb'):
        check_resume(old, _manifest(tmp_path, 'second'))


def test_cli_mode(tmp_path):
    old = _manifest(tmp_path, 'first')
    assert main(['--check-resume', old, old]) is None
    with pytest.raises(SystemExit, match='resume refused'):
        main(['--check-resume', old,
              _manifest(tmp_path, 'second', epochs=200)])


# --- cluster/submit_suite.sh --resume against a fake ssh --------------------

FAKE_SSH = """#!/usr/bin/env bash
# fake ssh: log the call, drain stdin, serve $FAKE_OLD for the manifest read
echo "ssh $*" >> "$FAKE_LOG"
cat >/dev/null
case "$*" in
    *"if [ -f"*) [[ -f "$FAKE_OLD" ]] && cat "$FAKE_OLD" ;;
    *sbatch*) echo "Submitted batch job 4242" ;;
esac
exit 0
"""
FAKE_RSYNC = '#!/usr/bin/env bash\necho "rsync $*" >> "$FAKE_LOG"\n'


@pytest.fixture
def fake_cluster(tmp_path):
    if shutil.which('uv') is None:
        pytest.skip('uv not on PATH')
    bin_ = tmp_path / 'bin'
    bin_.mkdir()
    for name, body in (('ssh', FAKE_SSH), ('rsync', FAKE_RSYNC)):
        (bin_ / name).write_text(body)
        (bin_ / name).chmod((bin_ / name).stat().st_mode | stat.S_IXUSR)
    log = tmp_path / 'calls.log'
    log.write_text('')

    def submit(config, old_manifest):
        env = {**os.environ, 'PATH': f'{bin_}:{os.environ["PATH"]}',
               'CLUSTER_HOST': 'fakehost.invalid', 'FAKE_LOG': str(log),
               'FAKE_OLD': old_manifest,
               'CAMPAIGN_WORKLOG': str(tmp_path / 'no-worklog.md')}
        proc = subprocess.run(
            ['bash', 'cluster/submit_suite.sh', '--resume', config,
             '2026-11-20_run'], cwd=ROOT, env=env, text=True,
            stdin=subprocess.DEVNULL, capture_output=True)
        return proc, log.read_text().splitlines()
    return submit


def test_shell_refusal_leaves_the_run_folder_untouched(tmp_path, fake_cluster):
    old = _manifest(tmp_path, 'first', epochs=50)
    proc, calls = fake_cluster(_yaml(tmp_path, 'second', epochs=200), old)
    assert proc.returncode != 0
    assert 'resume refused' in proc.stderr
    assert 'lrc_ncp_seed0 (run folder manifest)' in proc.stderr
    # one ssh call: the read of the old manifest; no LAUNCH.md note, upload,
    # rsync or sbatch
    assert len(calls) == 1 and 'manifest_default.jsonl' in calls[0]
    assert calls[0].startswith('ssh fakehost.invalid if [ -f ')


def test_shell_resume_that_agrees_runs_to_sbatch(tmp_path, fake_cluster):
    old = _manifest(tmp_path, 'first')
    proc, calls = fake_cluster(_yaml(tmp_path, 'second', seeds=[0, 1, 2]), old)
    assert proc.returncode == 0, proc.stderr
    assert 'agrees with the manifest in the run folder' in proc.stdout
    kinds = [c.split()[0] + (' sbatch' if 'sbatch' in c else '')
             for c in calls]
    assert calls[0].startswith('ssh fakehost.invalid if [ -f ')
    assert any('LAUNCH.md' in c for c in calls[1:2])      # right after the read
    assert any('cat > ' in c and 'manifest_default.jsonl' in c for c in calls)
    assert 'rsync' in kinds and 'ssh sbatch' in kinds
    assert kinds.index('rsync') < kinds.index('ssh sbatch')
