"""Elapsed-time floor of LTC_Cell: kwarg min_elapsed and registry key ltc_minel
(issue #89, protocol F2 P2b).

On Person Activity the loader clamps negative time gaps to 0; the fused LTC
step divides by the elapsed time, so a zero gap gives a NaN state. With
min_elapsed set the step uses max(elapsed_time, min_elapsed); None (default)
leaves the cell as it was.
"""
import json
import sys

import numpy as np
import pytest
import tensorflow as tf

import src.tasks.person_activity as pa_pkg
from experiments import run_person_activity_benchmark as pa_runner
from src.benchmark.checkpoints import config_hash, rebuild_from_result
from src.benchmark.registry import KNOWN_CELLS, cell_kwargs, cell_units
from src.models.rnn_model import _CELL_REGISTRY
from src.neurons import LTC_Cell, MM_LTC_Cell
from src.tasks.person_activity.model import build_person_activity_model
from src.wirings import NCPWiring, param_counts
from tests.benchmark.test_checkpoints import _capture_builds, _pa_stub
from tests.neurons.test_lrc_init import EXISTING_ARM_HASHES

IN = 3
UNITS = 8


def _rnn(cell, t_zero=True):
    """RNN over (x, elapsed) with a zero gap in the middle of the sequence."""
    x = tf.keras.Input((4, IN))
    t = tf.keras.Input((4, 1))
    out = tf.keras.layers.RNN(cell, return_sequences=True)((x, t))
    return tf.keras.Model([x, t], out)


def _data(gap=0.0):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 4, IN)).astype(np.float32)
    t = np.full((2, 4, 1), 0.1, np.float32)
    t[:, 2] = gap
    return x, t


def _one_step(model, x, t):
    """Loss and gradients of one MSE step on the model."""
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(tf.square(model([x, t])))
    grads = tape.gradient(loss, model.trainable_variables)
    return loss.numpy(), grads


# --- the floor ----------------------------------------------------------------

def test_default_is_off_and_zero_gap_stays_nan():
    tf.keras.utils.set_random_seed(0)
    cell = LTC_Cell(UNITS)
    assert cell._min_elapsed is None
    x, t = _data(0.0)
    out = _rnn(cell)([x, t]).numpy()
    assert not np.isfinite(out).all()


def test_zero_gap_is_finite_with_the_floor_forward_and_gradient():
    tf.keras.utils.set_random_seed(0)
    model = _rnn(LTC_Cell(UNITS, min_elapsed=0.05))
    x, t = _data(0.0)
    loss, grads = _one_step(model, x, t)
    assert np.isfinite(loss)
    for g in grads:
        assert g is not None and np.isfinite(g.numpy()).all()
    # one optimizer step keeps the weights finite
    opt = tf.keras.optimizers.RMSprop(1e-3)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    assert all(np.isfinite(w).all() for w in model.get_weights())
    assert np.isfinite(_one_step(model, x, t)[0])


def test_floor_only_acts_below_the_value():
    x, t = _data(0.1)                        # every gap 0.1 >= 0.05
    outs = []
    for kw in ({}, {'min_elapsed': 0.05}):
        tf.keras.utils.set_random_seed(0)
        outs.append(_rnn(LTC_Cell(UNITS, **kw))([x, t]).numpy())
    np.testing.assert_array_equal(outs[0], outs[1])

    # a zero gap with the floor is the floor value itself
    x, t0 = _data(0.0)
    _, t05 = _data(0.05)
    outs = []
    for tt in (t0, t05):
        tf.keras.utils.set_random_seed(0)
        outs.append(_rnn(LTC_Cell(UNITS, min_elapsed=0.05))([x, tt]).numpy())
    np.testing.assert_array_equal(outs[0], outs[1])


def test_floor_applies_to_the_regular_sampling_dt():
    x = np.random.default_rng(0).normal(size=(2, 4, IN)).astype(np.float32)
    outs = []
    for kw in ({'dt': 0.05}, {'dt': 0.0, 'min_elapsed': 0.05},
               {'dt': 0.01, 'min_elapsed': 0.05}):
        tf.keras.utils.set_random_seed(0)
        cell = LTC_Cell(UNITS, **kw)
        outs.append(tf.keras.layers.RNN(cell, return_sequences=True)(x).numpy())
    assert np.isfinite(outs[1]).all()
    # python-float dt vs float32 floor tensor: equal up to rounding
    np.testing.assert_allclose(outs[0], outs[1], rtol=1e-6)
    np.testing.assert_allclose(outs[0], outs[2], rtol=1e-6)


def test_integer_dt_is_accepted():
    cell = LTC_Cell(UNITS, min_elapsed=0.05)        # default dt=1 (int)
    x = np.ones((2, 4, IN), np.float32)
    out = tf.keras.layers.RNN(cell, return_sequences=True)(x).numpy()
    assert np.isfinite(out).all()


@pytest.mark.parametrize('bad', [0, 0.0, -0.1])
def test_non_positive_floor_is_rejected(bad):
    with pytest.raises(ValueError, match='min_elapsed'):
        LTC_Cell(UNITS, min_elapsed=bad)


def test_floor_reaches_the_wired_and_mixed_memory_cells():
    x, t = _data(0.0)
    tf.keras.utils.set_random_seed(0)
    w = NCPWiring(LTC_Cell, 8, 4, 2, seed=1, min_elapsed=0.05)
    cell = w.make_cell()
    assert isinstance(cell, LTC_Cell) and cell._min_elapsed == 0.05
    assert np.isfinite(_rnn(cell)([x, t]).numpy()).all()

    tf.keras.utils.set_random_seed(0)
    mm = MM_LTC_Cell(UNITS, min_elapsed=0.05)
    out = _rnn(mm)([x, t]).numpy()
    assert np.isfinite(out).all()


# --- registry key -------------------------------------------------------------

def test_registry_key():
    assert 'ltc_minel' in KNOWN_CELLS
    assert _CELL_REGISTRY['ltc_minel'] is LTC_Cell
    assert cell_kwargs('ltc_minel') == {'min_elapsed': 0.05}
    assert cell_units('ltc_minel') == cell_units('ltc')
    assert cell_kwargs('ltc') == {}                 # unchanged


# --- the existing arms stay as they were ------------------------------------

# config_hash[:12] of {cell, cell_kwargs} per registry key before ltc_minel
# existed (the keys of tests/neurons/test_lrc_init.py plus lrc_cal_duffing).
# No protocol_version enters the dict: stamp_protocol adds it to config.effective
# before hashing, and the F2 freeze changes it.
PINNED_HASHES = {**EXISTING_ARM_HASHES, 'lrc_cal_duffing': '587764bda040'}


@pytest.mark.parametrize('cell', sorted(PINNED_HASHES))
def test_existing_arm_kwargs_keep_their_config_hash(cell):
    got = config_hash({'cell': cell, 'cell_kwargs': cell_kwargs(cell)})
    assert got[:12] == PINNED_HASHES[cell]


def test_floor_changes_the_config_hash_only_when_set():
    base = config_hash({'cell': 'ltc', 'cell_kwargs': cell_kwargs('ltc')})
    minel = config_hash({'cell': 'ltc',
                         'cell_kwargs': cell_kwargs('ltc_minel')})
    assert base[:12] == PINNED_HASHES['ltc'] and minel != base


def _fingerprint(arm, wiring):
    """(params_effective, params_raw, sum |w|, output sum) of the PA builder at
    size 8, seed 0, on a fixed input with a zero-free gap."""
    kw = {k: v for k, v in cell_kwargs(arm).items() if k != 'ode_unfolds'}
    model = build_person_activity_model(wiring, arm, size=8, seed=0,
                                        feature_size=4, seq_len=6,
                                        ode_unfolds=6, **kw)
    rng = np.random.default_rng(3)
    x = rng.normal(size=(2, 6, 4)).astype(np.float32)
    t = np.full((2, 6, 1), 0.2, np.float32)
    out = model([x, t]).numpy()
    pc = param_counts(model)
    return (pc['params_effective'], pc['params_raw'],
            sum(float(np.abs(v.numpy()).sum()) for v in model.weights),
            float(out.sum()))


# Fingerprints as built before ltc_minel existed: the parameters, RNG draws and
# forward pass of the arms that share the cell (ltc, mm_ltc) with the option off.
EXISTING_LTC_BUILDS = {
    ('ltc', 'dense'): (495, 495, 759.6936698555946, 10.123894691467285),
    ('ltc', 'ncp'): (307, 1403, 2424.778359770775, 0.3101396858692169),
    ('mm_ltc', 'dense'): (911, 911, 867.5891390442848, 0.7748282551765442),
    ('mm_ltc', 'ncp'): (555, 2747, 2669.2060384750366, -1.1357389688491821),
}


@pytest.mark.parametrize('arm,wiring', sorted(EXISTING_LTC_BUILDS))
def test_existing_arms_are_built_as_before(arm, wiring):
    got = _fingerprint(arm, wiring)
    want = EXISTING_LTC_BUILDS[(arm, wiring)]
    assert got[:2] == want[:2]
    np.testing.assert_allclose(got[2:], want[2:], rtol=1e-5)


def test_floor_adds_no_parameters_and_no_rng_draws():
    # same parameter values as ltc: the option sits in the step, not the build
    a = _fingerprint('ltc', 'ncp')
    b = _fingerprint('ltc_minel', 'ncp')
    assert a[:3] == b[:3]                 # same counts, same sum |w|
    assert a[3] == b[3]                   # gaps of 0.2 lie above the floor


# --- through the PA runner -----------------------------------------------------

def _zero_gap_stub(*a, **k):
    data = _pa_stub()
    data.train_t = data.train_t.copy()
    data.train_t[0, 3:5] = 0.0
    return data


def _pa_run(tmp_path, monkeypatch, cell, wiring='ncp', stub=_pa_stub):
    built = _capture_builds(monkeypatch, pa_pkg, 'build_person_activity_model')
    monkeypatch.setattr(pa_pkg, 'load_person_activity', stub)
    monkeypatch.setattr(sys, 'argv', [
        'runner.py', '--wiring', wiring, '--cell', cell, '--size', '8',
        '--epochs', '1', '--outdir', str(tmp_path)])
    pa_runner.main()
    (path,) = tmp_path.glob('*.json')
    return json.loads(path.read_text()), path, built[-1][0]


# config_hash[:12] of config.effective without protocol_version / config_hash for
# the ltc arm through the PA runner, as recorded before ltc_minel existed.
# stamp_protocol writes protocol_version in before hashing and the F2 freeze
# changes it, so it is taken out. A change here changes the fits this arm
# produces: re-pin only together with a new PROTOCOL_VERSION.
EXISTING_EFFECTIVE_HASHES = {'dense': '060eed250a5d', 'ncp': '037aeecd0a2c'}


@pytest.mark.parametrize('wiring', ['dense', 'ncp'])
def test_ltc_through_the_pa_runner_is_unchanged(wiring, tmp_path, monkeypatch):
    record, _, _ = _pa_run(tmp_path, monkeypatch, 'ltc', wiring)
    eff = record['config']['effective']
    assert 'min_elapsed' not in eff['cell_kwargs']
    rest = {k: v for k, v in eff.items()
            if k not in ('protocol_version', 'config_hash')}
    assert config_hash(rest)[:12] == EXISTING_EFFECTIVE_HASHES[wiring]


@pytest.mark.parametrize('wiring', ['dense', 'ncp'])
def test_ltc_minel_through_the_pa_runner(wiring, tmp_path, monkeypatch):
    record, path, model = _pa_run(tmp_path, monkeypatch, 'ltc_minel', wiring,
                                  stub=_zero_gap_stub)
    eff = record['config']['effective']
    assert eff['cell_kwargs']['min_elapsed'] == 0.05      # JSON key of the arm
    assert eff['config_hash'] == config_hash(
        {k: v for k, v in eff.items() if k != 'config_hash'})
    assert record['training_diverged'] is False
    assert all(np.isfinite(w).all() for w in model.get_weights())
    cells = [l for l in model._flatten_layers(include_self=False, recursive=True)
             if isinstance(l, LTC_Cell)]
    assert len(cells) == 1 and cells[0]._min_elapsed == 0.05
    # the rebuild from the JSON carries the floor
    (rc,) = [l for l in rebuild_from_result(str(path))._flatten_layers(
        include_self=False, recursive=True) if isinstance(l, LTC_Cell)]
    assert rc._min_elapsed == 0.05


def test_ltc_with_a_zero_gap_still_diverges_on_pa(tmp_path, monkeypatch):
    # the old behaviour stays for the key ltc (this is the P2 symptom)
    record, _, model = _pa_run(tmp_path, monkeypatch, 'ltc', 'ncp',
                               stub=_zero_gap_stub)
    assert (record['training_diverged'] is True
            or not all(np.isfinite(w).all() for w in model.get_weights()))
