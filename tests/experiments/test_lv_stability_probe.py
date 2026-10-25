# tests/experiments/test_lv_stability_probe.py
# --probe-stability of the Lotka-Volterra runner (canary gate, spec 2026-10-21
# R6): the result JSON gains a 'stability' block with per-state norms at init,
# after one optimiser step and after epoch 1, the epoch-1 train loss and a
# 'finite' flag, measured on the model the runner actually trains; the same
# norms on a val and a test batch under suffixed keys (issue #19).
import json
import sys

import numpy as np
import pytest

from experiments import run_lotka_volterra_benchmark as runner

STATES = {'cncp2': ('G', 'S', 'D', 'R'),
          'cncp': ('L4', 'L23', 'L5IT', 'L5ET', 'L6CC', 'L6CT', 'Thal', 'TRN')}


@pytest.mark.parametrize('wiring', ['cncp2', 'cncp'])
def test_probe_stability_block(wiring, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', [
        'run_lotka_volterra_benchmark.py', '--wiring', wiring,
        '--cell', 'cfc_lrc', '--seed', '0', '--epochs', '2', '--size', '8',
        '--n-trajectories', '10', '--seq-len', '16', '--batch-size', '4',
        '--outdir', str(tmp_path), '--probe-stability'])
    runner.main()
    record = json.loads(
        (tmp_path / f'cfc_lrc_{wiring}_seed0.json').read_text())
    stab = record['stability']
    for key in [f'norms_{p}{s}' for p in ('init', 'step1', 'epoch1')
                for s in ('', '_val', '_test')]:
        names = [k.split('/')[-1] for k in stab[key]]
        assert names == list(STATES[wiring]), key
        assert all(np.isfinite(v) for v in stab[key].values())
    assert np.isfinite(stab['tf_loss_epoch1'])
    assert stab['finite'] is True
    # the probe reads, it does not train: one step moved the weights
    assert stab['norms_step1'] != stab['norms_init']


def test_no_stability_block_without_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', [
        'run_lotka_volterra_benchmark.py', '--wiring', 'dense', '--cell', 'gru',
        '--epochs', '1', '--size', '4', '--n-trajectories', '10',
        '--seq-len', '8', '--outdir', str(tmp_path)])
    runner.main()
    record = json.loads((tmp_path / 'gru_dense_seed0.json').read_text())
    assert 'stability' not in record
