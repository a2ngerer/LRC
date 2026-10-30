"""Init-scaling options of LRC_Cell and the lrc_init / lrc_cal arms (issue #82)."""
import numpy as np
import pytest
import tensorflow as tf

from src.benchmark.registry import cell_kwargs
from src.neurons import LRC_Cell
from src.tasks.person_activity.model import build_person_activity_model
from src.wirings import NCPWiring, param_counts

IN = 5
# Wide enough that the fan-in rescaled column norms average close to 1.
INTER, COMMAND, MOTOR = 32, 16, 8
UNITS = INTER + COMMAND + MOTOR
DENSE_UNITS = 12
NCP_SLICES = {'motor': (0, MOTOR), 'command': (MOTOR, MOTOR + COMMAND),
              'inter': (MOTOR + COMMAND, UNITS)}


def _cell(masked, **kw):
    tf.keras.utils.set_random_seed(0)
    if masked:
        w = NCPWiring(LRC_Cell, INTER, COMMAND, MOTOR)
        cell = LRC_Cell(units=UNITS, elastance_type='asymmetric',
                        sparsity_mask=w.adjacency_mask,
                        sensory_mask=w.sensory_mask, **kw)
    else:
        cell = LRC_Cell(units=DENSE_UNITS, elastance_type='asymmetric', **kw)
    cell.build((None, IN))
    # one step right away: the elastance Dense builds lazily on the first
    # call, so it must draw from the same seeded RNG state for every cell
    cell.out = _out(cell)
    return cell


def _weights(cell):
    return {v.name.split('/')[-1]: v.numpy() for v in cell.weights}


def _out(cell):
    x = tf.constant(np.linspace(-1, 1, 2 * IN, dtype=np.float32).reshape(2, IN))
    return cell(x, [tf.ones([2, cell.units]) * 0.1])[0].numpy()


# --- (i) options off == no options ------------------------------------------

@pytest.mark.parametrize('masked', [False, True])
def test_defaults_are_byte_identical(masked, monkeypatch):
    # base: the init-scaling step removed, i.e. the cell as it was before the
    # options existed; off: the shipped cell with every option at its default
    with monkeypatch.context() as m:
        m.setattr(LRC_Cell, '_apply_init_scaling', lambda self: None)
        base = _cell(masked)
    off = _cell(masked, init_fanin_scale=None, init_slice_scale=None,
                init_slices=None)
    wb, wo = _weights(base), _weights(off)
    assert wb.keys() == wo.keys()
    assert all(np.array_equal(wb[k], wo[k]) for k in wb)
    assert np.array_equal(base.out, off.out)
    assert off._init_scale_report == {}


# --- (ii) fan-in scaling ----------------------------------------------------

def test_fanin_scaling_masked():
    base, cell = _cell(True), _cell(True, init_fanin_scale='units')
    adj, sens = cell.sparsity_mask.numpy(), cell.sensory_mask.numpy()
    fan, sfan = adj.sum(0), sens.sum(0)
    hb, h = base._params['h'].numpy(), cell._params['h'].numpy()
    shb, sh = base._params['sensory_h'].numpy(), cell._params['sensory_h'].numpy()

    on = fan > 0
    np.testing.assert_allclose(
        (h * adj)[:, on], (hb * adj)[:, on] * np.sqrt(UNITS / fan[on]),
        rtol=1e-6)
    s_on = sfan > 0
    np.testing.assert_allclose(
        (sh * sens)[:, s_on], (shb * sens)[:, s_on] * np.sqrt(UNITS / sfan[s_on]),
        rtol=1e-6)
    # zero fan-in columns (inter in h, command/motor in sensory_h) unchanged
    assert (~on).any() and (~s_on).any()
    assert np.array_equal(h[:, ~on], hb[:, ~on])
    assert np.array_equal(sh[:, ~s_on], shb[:, ~s_on])
    # orthogonal columns restricted to the fan-in keep unit norm on average
    sq = ((h * adj)[:, on] ** 2).sum(0).mean()
    assert 0.7 <= sq <= 1.3, sq
    # nothing else moves
    wb, w = _weights(base), _weights(cell)
    assert all(np.array_equal(wb[k], w[k]) for k in wb
               if k not in ('h:0', 'sensory_h:0'))


def test_fanin_scaling_dense():
    base = _weights(_cell(False))
    same = _weights(_cell(False, init_fanin_scale='input_dim'))
    assert all(np.array_equal(base[k], same[k]) for k in base)

    units = _weights(_cell(False, init_fanin_scale='units'))
    np.testing.assert_allclose(units['sensory_h:0'],
                               base['sensory_h:0'] * np.sqrt(DENSE_UNITS / IN),
                               rtol=1e-6)
    assert all(np.array_equal(base[k], units[k]) for k in base
               if k != 'sensory_h:0')


def test_fanin_scale_rejects_unknown_reference():
    with pytest.raises(ValueError, match='init_fanin_scale'):
        LRC_Cell(units=4, init_fanin_scale='fan_out')


# --- (iii) slice scaling ----------------------------------------------------

def test_slice_scaling_masked():
    scale = {'inter': 2.0, 'all': 3.0}          # command/motor fall back to 'all'
    base = _weights(_cell(True))
    cell = _cell(True, init_slice_scale=scale, init_slices=NCP_SLICES)
    w = _weights(cell)
    col = np.ones(UNITS, np.float32)
    col[slice(*NCP_SLICES['inter'])] = 2.0
    col[:MOTOR + COMMAND] = 3.0
    assert np.array_equal(w['vleak:0'], base['vleak:0'] * col)
    assert np.array_equal(w['h:0'], base['h:0'] * col[None, :])
    assert np.array_equal(w['sensory_h:0'], base['sensory_h:0'] * col[None, :])
    assert all(np.array_equal(base[k], w[k]) for k in base
               if k not in ('vleak:0', 'h:0', 'sensory_h:0'))
    assert cell._init_scale_report['slices']['motor'] == ((0, MOTOR), 3.0)


def test_slice_scaling_dense_uses_inter_factor():
    base = _weights(_cell(False))
    w = _weights(_cell(False, init_slice_scale=cell_kwargs('lrc_cal')[
        'init_slice_scale']))
    f = np.float32(7.2)
    assert np.array_equal(w['vleak:0'], base['vleak:0'] * f)
    assert np.array_equal(w['h:0'], base['h:0'] * f)
    assert np.array_equal(w['sensory_h:0'], base['sensory_h:0'] * f)


def test_slice_scaling_masked_without_slices_raises():
    with pytest.raises(ValueError, match='init_slices'):
        _cell(True, init_slice_scale={'inter': 2.0})


# --- (iv) parameter counts unchanged ----------------------------------------

@pytest.mark.parametrize('wiring', ['dense', 'ncp'])
def test_param_counts_match_lrc(wiring):
    stub = [np.zeros((1, 32, 7), np.float32), np.ones((1, 32, 1), np.float32)]
    counts = {}
    for cell in ('lrc', 'lrc_init', 'lrc_cal'):
        model = build_person_activity_model(wiring, cell, size=8, seed=0,
                                            **cell_kwargs(cell))
        model(stub)
        counts[cell] = param_counts(model)
    assert counts['lrc_init'] == counts['lrc'] == counts['lrc_cal']


# --- (v) the wiring passes init_slices only with init_slice_scale -----------

def test_ncp_make_cell_passes_slices_only_with_slice_scale():
    plain = NCPWiring(LRC_Cell, 4, 3, 2, elastance_type='asymmetric')
    assert plain.make_cell()._init_slices is None

    w = NCPWiring(LRC_Cell, 4, 3, 2, **cell_kwargs('lrc_cal'))
    assert w.make_cell()._init_slices == {
        'motor': (0, 2), 'command': (2, 5), 'inter': (5, 9)}
    assert 'init_slices' not in w.cell_kwargs


# --- (i-b) default path locked to the pre-change cell -----------------------

# sum(|w|) per weight and of the first-step output of _cell(masked) as built
# before issue #82 (int/finalsuite-2026-10-26 @ bbd65cd, same seed and
# construction). (i) compares two cells of the same code; this locks the
# default path to the old cell's values, with a tolerance for BLAS drift.
PRE_CHANGE_FINGERPRINT = {
    False: {'gleak:0': 6.194533824920654, 'vleak:0': 1.092078685760498,
            'sigma:0': 806.0675048828125, 'mu:0': 73.58831024169922,
            'w:0': 68.66522216796875, 'sensory_w:0': 36.17940139770508,
            'h:0': 34.62226867675781, 'sensory_sigma:0': 324.6983947753906,
            'sensory_mu:0': 34.35722732543945, 'sensory_h:0': 14.246881484985352,
            'input_w:0': 5.0, 'input_b:0': 0.0, 'output_w:0': 12.0,
            'output_b:0': 0.0, 'kernel:0': 46.30244445800781, 'bias:0': 0.0,
            '__out__': 1.2706618309020996},
    True: {'gleak:0': 32.20558166503906, 'vleak:0': 5.835495471954346,
           'sigma:0': 17293.14453125, 'mu:0': 1705.657470703125,
           'w:0': 1555.290283203125, 'sensory_w:0': 146.32376098632812,
           'h:0': 337.2244873046875, 'sensory_sigma:0': 1514.10595703125,
           'sensory_mu:0': 153.68374633789062, 'sensory_h:0': 30.270198822021484,
           'input_w:0': 5.0, 'input_b:0': 0.0, 'output_w:0': 56.0,
           'output_b:0': 0.0, 'kernel:0': 390.8216552734375, 'bias:0': 0.0,
           '__out__': 6.338131904602051},
}


@pytest.mark.parametrize('masked', [False, True])
def test_default_path_matches_pre_change_cell(masked):
    cell = _cell(masked)
    got = {k: float(np.abs(w).sum()) for k, w in _weights(cell).items()}
    got['__out__'] = float(np.abs(cell.out).sum())
    exp = PRE_CHANGE_FINGERPRINT[masked]
    assert got.keys() == exp.keys()
    for k, e in exp.items():
        np.testing.assert_allclose(got[k], e, rtol=1e-5, atol=1e-6, err_msg=k)
