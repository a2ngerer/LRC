"""Elastance held at exactly 1 in CfC_LRC_Cell: kwarg elastance_one and
registry key cfc_lrc_eps1 (issue #90, protocol F2 C-2, variant B).

With elastance_one=True the cell still builds the elastance head, so its
initializer draw is consumed and every later layer gets the same init as the
cfc_lrc arm at the same seed. The head is then frozen and the forward pass uses
eps = 1, which is plain CfC. The frozen head leaves the effective (trainable)
parameter count; the width comes from the cfc_lrc arm (same --size, or the
cfc_lrc count under --param-budget).
"""
import json
import sys

import numpy as np
import pytest
import tensorflow as tf

from experiments import run_lotka_volterra_benchmark as lv_runner
from src.benchmark.checkpoints import config_hash, rebuild_from_result
from src.benchmark.registry import KNOWN_CELLS, cell_kwargs
from src.models.rnn_model import _CELL_REGISTRY
from src.neurons import CfC_Cell, CfC_LRC_Cell
from src.tasks.lotka_volterra.model import build_lotka_volterra_model
from src.wirings import param_counts
from tests.benchmark.test_checkpoints import _capture_builds
from tests.neurons.test_ltc_min_elapsed import PINNED_HASHES

IN = 3
UNITS = 6


def _seq(cell, steps=5, batch=2, seed=1):
    x = np.random.default_rng(seed).normal(size=(batch, steps, IN))
    return tf.keras.layers.RNN(cell, return_sequences=True), x.astype(np.float32)


def _cells(model, cls=CfC_LRC_Cell):
    return [l for l in model._flatten_layers(include_self=False, recursive=True)
            if isinstance(l, cls)]


def _split(model):
    """(non-head weights, elastance-head weights) of ``model`` as arrays, each
    in model.weights order. Keras lists trainable before non-trainable
    weights, so a frozen head moves to the end of model.weights; the order
    within each part stays the build order."""
    head = {id(v) for c in _cells(model) for v in c._elastance.weights}
    rest = [v.numpy() for v in model.weights if id(v) not in head]
    heads = [v.numpy() for v in model.weights if id(v) in head]
    return rest, heads


# --- the cell -------------------------------------------------------------------

def test_default_is_off():
    assert CfC_LRC_Cell(UNITS)._elastance_one is False


@pytest.mark.parametrize('bad', [1, 'yes', None, 1.0])
def test_non_bool_is_rejected(bad):
    with pytest.raises(ValueError, match='elastance_one'):
        CfC_LRC_Cell(UNITS, elastance_one=bad)


@pytest.mark.parametrize('etype', ['asymmetric', 'symmetric'])
def test_head_is_built_and_frozen(etype):
    cell = CfC_LRC_Cell(UNITS, elastance_type=etype, elastance_one=True)
    rnn, x = _seq(cell)
    rnn(x)
    assert cell._elastance.built and not cell._elastance.trainable
    head = {id(v) for v in cell._elastance.weights}
    if etype == 'symmetric':
        head.add(id(cell._distr_shift))
    assert len(head) == 3 - (etype == 'asymmetric')
    assert not head & {id(v) for v in rnn.trainable_variables}
    assert head == {id(v) for v in rnn.non_trainable_variables}


@pytest.mark.parametrize('gate', ['inner', 'outer'])
@pytest.mark.parametrize('etype', ['asymmetric', 'symmetric'])
def test_equals_cfc_and_stays_at_one_after_a_training_step(gate, etype):
    """After an optimizer step the frozen head is unchanged and the cell output
    equals a CfC cell carrying the same (trained) weights."""
    tf.keras.utils.set_random_seed(0)
    cell = CfC_LRC_Cell(UNITS, elastance_type=etype, elastance_gate=gate,
                        elastance_one=True)
    rnn, x = _seq(cell)
    model = tf.keras.Sequential([tf.keras.Input((None, IN)), rnn])
    head0 = [w.copy() for w in cell._elastance.get_weights()]
    opt = tf.keras.optimizers.RMSprop(1e-2)
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(tf.square(model(x) - 0.3))
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    for a, b in zip(head0, cell._elastance.get_weights()):
        np.testing.assert_array_equal(a, b)

    cfc = CfC_Cell(UNITS)
    ref = tf.keras.layers.RNN(cfc, return_sequences=True)
    ref(x)
    for src in (cell._backbone[0], cell._ff1, cell._ff2, cell._time_a,
                cell._time_b):
        dst = {'backbone_0': cfc._backbone[0], 'ff1': cfc._ff1,
               'ff2': cfc._ff2, 'time_a': cfc._time_a,
               'time_b': cfc._time_b}[src.name]
        dst.set_weights(src.get_weights())
    np.testing.assert_array_equal(model(x).numpy(), ref(x).numpy())


def test_on_differs_from_cfc_lrc_forward():
    # sanity: with the head live (option off) eps < 1 changes the output
    outs = []
    for on in (False, True):
        tf.keras.utils.set_random_seed(0)
        rnn, x = _seq(CfC_LRC_Cell(UNITS, elastance_one=on))
        outs.append(rnn(x).numpy())
    assert not np.array_equal(*outs)


# --- registry key ------------------------------------------------------------

def test_registry_key():
    assert 'cfc_lrc_eps1' in KNOWN_CELLS
    assert _CELL_REGISTRY['cfc_lrc_eps1'] is CfC_LRC_Cell
    assert cell_kwargs('cfc_lrc_eps1') == {'elastance_type': 'asymmetric',
                                          'elastance_one': True}
    assert cell_kwargs('cfc_lrc') == {'elastance_type': 'asymmetric'}


# --- same init as cfc_lrc, frozen head out of the effective count ------------

WIRINGS = ['dense', 'ncp', 'cncp']


def _build(cell, wiring, size=8, seed=0):
    tf.keras.utils.set_random_seed(seed)
    model = build_lotka_volterra_model(wiring, cell, size=size, seed=seed,
                                       **cell_kwargs(cell))
    model(_lv_inputs(np.random.default_rng(0), 1, 4))
    return model


def _lv_inputs(rng, batch, steps):
    """(state, elapsed time) inputs of the LV builder."""
    return [rng.normal(size=(batch, steps, 2)).astype(np.float32),
            np.full((batch, steps, 1), 0.5, np.float32)]


@pytest.mark.parametrize('wiring', WIRINGS)
@pytest.mark.parametrize('seed', [0, 3])
def test_same_seed_same_weights_as_cfc_lrc(wiring, seed):
    lrc = _build('cfc_lrc', wiring, seed=seed)
    eps1 = _build('cfc_lrc_eps1', wiring, seed=seed)
    (a, a_head), (b, b_head) = _split(lrc), _split(eps1)
    assert len(a) == len(b) == len(lrc.weights) - len(a_head)
    assert len(a_head) == len(b_head) > 0
    # every weight but the head, the readout included
    for i, (x, y) in enumerate(zip(a, b)):
        np.testing.assert_array_equal(x, y, err_msg=str(i))
    # the head itself got the same draw too
    for x, y in zip(a_head, b_head):
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize('wiring', WIRINGS)
def test_param_counts(wiring):
    lrc = param_counts(_build('cfc_lrc', wiring))
    eps1 = param_counts(_build('cfc_lrc_eps1', wiring))
    cfc = param_counts(_build('cfc', wiring))
    assert eps1['params_raw'] == lrc['params_raw']          # head still exists
    assert eps1['params_effective'] == cfc['params_effective']
    assert eps1['params_effective'] < lrc['params_effective']


# --- existing arms stay as they were ------------------------------------------

def test_existing_arm_kwargs_keep_their_config_hash():
    pinned = {**PINNED_HASHES, 'ltc_minel': LTC_MINEL_HASH, **TAU_HASHES}
    for cell, want in pinned.items():
        got = config_hash({'cell': cell, 'cell_kwargs': cell_kwargs(cell)})
        assert got[:12] == want, cell


# recorded before elastance_one existed (keys added by #87 / #89)
LTC_MINEL_HASH = '19c6f1fb21b5'
TAU_HASHES = {'lrc_tau': '05fc13b2bbe1', 'lrc_init_tau': 'af5ac963ad54'}


def _fingerprint(cell, wiring):
    model = _build(cell, wiring)
    x = _lv_inputs(np.random.default_rng(3), 2, 6)
    pc = param_counts(model)
    return (pc['params_effective'], pc['params_raw'],
            sum(float(np.abs(v.numpy()).sum()) for v in model.weights),
            float(model(x).numpy().sum()))


# (params_effective, params_raw, sum |w|, output sum) of the LV builder at size
# 8, seed 0, recorded before elastance_one existed.
EXISTING_BUILDS = {
    ('cfc_lrc', 'dense'): (482, 482, 140.482421875, -2.4169745445251465),
    ('cfc_lrc', 'ncp'): (594, 810, 257.0027542114258, -0.019643619656562805),
    ('cfc_lrc', 'cncp'): (787, 826, 322.5928122550249, -0.6968669891357422),
    ('cfc_lrc_outer', 'dense'): (482, 482, 140.482421875, -1.6807276010513306),
    ('cfc_lrc_outer', 'ncp'): (594, 810, 257.0027542114258, 0.029824383556842804),
    ('cfc_lrc_outer', 'cncp'): (787, 826, 322.5928122550249, 0.501412570476532),
    ('cfc_mm_lrc', 'dense'): (834, 834, 195.18052196502686, -2.443121910095215),
    ('cfc_mm_lrc', 'ncp'): (866, 1514, 393.09814071655273, 0.17572790384292603),
}


@pytest.mark.parametrize('arm,wiring', sorted(EXISTING_BUILDS))
def test_existing_arms_are_built_as_before(arm, wiring):
    got = _fingerprint(arm, wiring)
    want = EXISTING_BUILDS[(arm, wiring)]
    assert got[:2] == want[:2]
    np.testing.assert_allclose(got[2:], want[2:], rtol=1e-5)


# --- through the LV runner (Duffing) -------------------------------------------

def _lv_run(tmp_path, monkeypatch, cell, wiring, *extra):
    built = _capture_builds(monkeypatch, lv_runner,
                            'build_lotka_volterra_model')
    monkeypatch.setattr(sys, 'argv', [
        'runner.py', '--wiring', wiring, '--cell', cell, '--system', 'duffing',
        '--epochs', '1', '--outdir', str(tmp_path), '--n-trajectories', '4',
        '--anchors', '2,4', '--horizon', '4', '--seq-len', '8',
        '--batch-size', '4', *extra])
    lv_runner.main()
    (path,) = tmp_path.glob('*.json')
    return json.loads(path.read_text()), path, built[-1][0]


def _rest_hash(record):
    eff = record['config']['effective']
    return config_hash({k: v for k, v in eff.items()
                        if k not in ('protocol_version', 'config_hash')})[:12]


# config_hash[:12] of config.effective without protocol_version / config_hash
# of cfc_lrc through the LV runner on Duffing (size 8), recorded before
# elastance_one existed. Re-pin only together with a new PROTOCOL_VERSION.
CFC_LRC_EFFECTIVE_HASHES = {'dense': '1365ba1b2535', 'ncp': 'e10c4a223fab'}


@pytest.mark.parametrize('wiring', ['dense', 'ncp'])
def test_cfc_lrc_through_the_lv_runner_is_unchanged(wiring, tmp_path,
                                                   monkeypatch):
    record, _, _ = _lv_run(tmp_path, monkeypatch, 'cfc_lrc', wiring,
                           '--size', '8')
    assert 'elastance_one' not in record['config']['effective']['cell_kwargs']
    assert _rest_hash(record) == CFC_LRC_EFFECTIVE_HASHES[wiring]


@pytest.mark.parametrize('wiring', ['ncp', 'cncp'])
def test_eps1_through_the_lv_runner(wiring, tmp_path, monkeypatch):
    record, path, model = _lv_run(tmp_path, monkeypatch, 'cfc_lrc_eps1',
                                  wiring, '--size', '8')
    eff = record['config']['effective']
    assert eff['cell_kwargs']['elastance_one'] is True     # label [eps=1]
    assert eff['config_hash'] == config_hash(
        {k: v for k, v in eff.items() if k != 'config_hash'})
    assert record['size'] == 8
    ref = param_counts(_build('cfc', wiring))
    assert record['params_effective'] == ref['params_effective']
    cells = _cells(model)
    assert cells and all(c._elastance_one and not c._elastance.trainable
                         for c in cells)
    # the rebuild from the JSON carries the option and the trained weights
    rebuilt = rebuild_from_result(str(path))
    assert all(c._elastance_one for c in _cells(rebuilt))
    for x, y in zip(model.get_weights(), rebuilt.get_weights()):
        np.testing.assert_array_equal(x, y)


def test_eps1_budget_width_is_the_cfc_lrc_width(tmp_path, monkeypatch):
    """Under --param-budget the eps1 arm takes the width of cfc_lrc at the same
    budget (the frozen head counts for the width search only), while the
    reported params_effective leaves the head out."""
    budget = '600'
    lrc, _, _ = _lv_run(tmp_path / 'a', monkeypatch, 'cfc_lrc', 'ncp',
                        '--param-budget', budget)
    eps1, _, _ = _lv_run(tmp_path / 'b', monkeypatch, 'cfc_lrc_eps1', 'ncp',
                         '--param-budget', budget)
    assert eps1['size'] == lrc['size']
    assert eps1['budget_deviation'] == lrc['budget_deviation']
    assert eps1['params_raw'] == lrc['params_raw']
    assert eps1['params_effective'] < lrc['params_effective']
