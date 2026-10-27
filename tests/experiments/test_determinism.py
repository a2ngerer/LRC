# tests/experiments/test_determinism.py
# Issue #70: --deterministic on the sequence runners (op determinism, recorded
# in config.effective.deterministic) and the S0 probe's summary
# (experiments/determinism_probe.py). Issue #76: the committee runner likewise.
import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiments import determinism_probe as probe

ROOT = Path(__file__).resolve().parents[2]
LV = ['--wiring', 'ncp', '--cell', 'cfc_lrc', '--seed', '0', '--size', '8',
      '--n-trajectories', '10', '--seq-len', '16', '--batch-size', '4',
      '--no-save-weights', '--anchors', '2,4', '--horizon', '4']
# PA on the synthetic stub of tests/benchmark/test_checkpoints.py
PA_STUB = ("import sys; import src.tasks.person_activity as p; "
           "from tests.benchmark.test_checkpoints import _pa_stub; "
           "p.load_person_activity = _pa_stub; "
           "from experiments import run_person_activity_benchmark as r; "
           "sys.argv = ['run_person_activity_benchmark.py'] + sys.argv[1:]; "
           "r.main()")
PA = ['--wiring', 'ncp', '--cell', 'cfc_lrc', '--size', '8',
      '--no-save-weights']
# committee on the same stub: the smallest closed-form fit (K = 2, 1 epoch)
CO_STUB = ("import sys; from tests.benchmark.test_checkpoints import _pa_stub; "
           "from experiments import run_committee_benchmark as r; "
           "r.load_person_activity = _pa_stub; "
           "sys.argv = ['run_committee_benchmark.py'] + sys.argv[1:]; "
           "r.main()")
CO = ['--wiring', 'ncp', '--cell', 'cfc_lrc', '--n-columns', '2', '--size',
      '8', '--epochs', '1', '--test-seeds', '0', '--no-save-weights']


def _fit(outdir, task, *flags):
    """One runner fit in a fresh process (enable_op_determinism is global and
    cannot be switched off again); returns the result record."""
    cmd = {'lv': [sys.executable, 'experiments/run_lotka_volterra_benchmark.py',
                  *LV],
           'pa': [sys.executable, '-c', PA_STUB, *PA],
           'co': [sys.executable, '-c', CO_STUB, *CO]}[task]
    subprocess.run([*cmd, *flags, '--outdir', str(outdir)], cwd=ROOT,
                   check=True, capture_output=True)
    (path,) = Path(outdir).glob('*.json')
    return json.loads(path.read_text())


def test_lv_same_seed_reproduces_val_metric(tmp_path):
    a, b = (_fit(tmp_path / r, 'lv', '--epochs', '20', '--deterministic')
            for r in ('on1', 'on2'))
    for rec in (a, b):
        assert rec['config']['effective']['deterministic'] is True
        assert rec['wallclock_s'] > 0
    for key in ('vpt_multi_median', 'nrmse_32_anchor_4'):
        assert abs(a['rollout_metrics_val'][key]
                   - b['rollout_metrics_val'][key]) <= 1e-3
    assert abs(a['best_val_teacher_forced_mse']
               - b['best_val_teacher_forced_mse']) <= 1e-3


@pytest.mark.parametrize('flags,expected', [((), False),
                                            (('--deterministic',), True)])
def test_pa_records_setting(tmp_path, flags, expected):
    rec = _fit(tmp_path, 'pa', '--epochs', '2', *flags)
    assert rec['config']['effective']['deterministic'] is expected


@pytest.mark.parametrize('flags,expected', [((), False),
                                            (('--deterministic',), True)])
def test_co_records_setting(tmp_path, flags, expected):
    """The committee runner takes --deterministic (run_grid_hpo.py forwards
    it from the consensus campaigns) and records it in config.effective."""
    rec = _fit(tmp_path, 'co', *flags)
    assert rec['config']['effective']['deterministic'] is expected


def _record(deterministic, wall, **metrics):
    return {'wallclock_s': wall, **metrics,
            'config': {'effective': {'deterministic': deterministic}}}


def _write(outdir, task, records):
    for run, rec in records.items():
        d = outdir / task / run
        d.mkdir(parents=True)
        (d / 'fit.json').write_text(json.dumps(rec))


def test_probe_summary_reuses_fits(tmp_path, capsys):
    lv = lambda det, wall, vpt, nrmse: _record(  # noqa: E731
        det, wall, rollout_metrics_val={'vpt_multi_median': vpt,
                                        'nrmse_32_anchor_8': nrmse})
    _write(tmp_path, 'lv', {'on1': lv(True, 12.0, 40.0, 0.3),
                            'on2': lv(True, 14.0, 40.0, 0.3004),
                            'off': lv(False, 10.0, 38.0, 0.31)})
    _write(tmp_path, 'pa', {'on1': _record(True, 30.0, val_macro_f1=0.61),
                            'on2': _record(True, 30.0, val_macro_f1=0.61),
                            'off': _record(False, 20.0, val_macro_f1=0.6)})
    probe.main(['--outdir', str(tmp_path)])          # no runner call: reused
    s = json.loads((tmp_path / 'determinism_probe.json').read_text())
    lv_s, pa_s = s['tasks']['lv'], s['tasks']['pa']
    assert lv_s['ratio_on_off'] == pytest.approx(1.3)
    assert lv_s['max_abs_delta_on'] == pytest.approx(4e-4)
    assert pa_s['ratio_on_off'] == pytest.approx(1.5)
    assert s['reproducible'] and not s['enable_for_suite']
    assert 'op-determinism for the suite: OFF' in capsys.readouterr().out


def test_probe_fails_on_irreproducible_pair(tmp_path):
    _write(tmp_path, 'pa', {'on1': _record(True, 1.0, val_macro_f1=0.61),
                            'on2': _record(True, 1.0, val_macro_f1=0.6),
                            'off': _record(False, 1.0, val_macro_f1=0.6)})
    with pytest.raises(SystemExit) as e:
        probe.main(['--outdir', str(tmp_path), '--tasks', 'pa'])
    assert e.value.code == 1
    s = json.loads((tmp_path / 'determinism_probe.json').read_text())
    assert not s['reproducible'] and s['enable_for_suite']


def test_probe_rejects_fit_without_the_setting(tmp_path):
    _write(tmp_path, 'pa', {'on1': _record(False, 1.0, val_macro_f1=0.6)})
    with pytest.raises(SystemExit, match='deterministic'):
        probe.main(['--outdir', str(tmp_path), '--tasks', 'pa'])
