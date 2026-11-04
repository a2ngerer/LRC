# tests/experiments/test_f2_manifest.py
# F2 campaign YAMLs (issue a2ngerer/LRC#105): the snapshot of every existing
# configs/suite/*.yaml (generated from the code before the manifest
# extensions selection_from and stage eval), the two extensions, and the F2
# YAMLs pinned to protocol F2 sections 3, 4, 7 and 8.
import json
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
    names = {p.stem for p in SUITE.glob('*.yaml') if not p.stem.startswith('f2_')}
    assert names == set(json.loads(SNAPSHOT.read_text()))


@pytest.mark.parametrize('name', sorted(json.loads(SNAPSHOT.read_text())))
def test_existing_yaml_lines_are_unchanged(name):
    """Neither extension used: count, write plan and every line (argv, id,
    group, outdir, done) equal the snapshot taken before #105."""
    assert _snapshot(name) == json.loads(SNAPSHOT.read_text())[name]
