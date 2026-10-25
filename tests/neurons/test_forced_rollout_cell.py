# tests/neurons/test_forced_rollout_cell.py
# Issue #25: the ForcedRolloutCell view shares the plain LV model's weights;
# alpha = 1 reproduces its teacher-forced predictions, alpha = 0 the reference
# closed_loop_rollout; parameter counts unchanged; StackedCell == stacked_dense;
# AlphaSchedule is linear 1 -> alpha_min over the first half, then constant.
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from experiments.run_lotka_volterra_benchmark import closed_loop_rollout
from src.neurons.forced_rollout_cell import AlphaSchedule, StackedCell
from src.tasks.lotka_volterra.model import (build_lotka_volterra_model,
                                            forced_rollout_model)
from src.wirings import param_counts

ARMS = [("cfc_lrc", "dense3"), ("gru", "ncp"), ("cfc", "dense3")]
ATOL = 1e-5


def _data(n=10, T=16):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, T, 2)).astype(np.float32)
    t = np.full((n, T, 1), 0.1, np.float32)
    return x, t


@pytest.mark.parametrize("cell,wiring", ARMS)
def test_alpha_one_is_teacher_forcing_and_alpha_zero_is_closed_loop(cell,
                                                                    wiring):
    x, t = _data()
    model = build_lotka_volterra_model(wiring, cell, size=8, seed=0)
    alpha = tf.Variable(1.0, trainable=False)
    view = forced_rollout_model(model, alpha)
    np.testing.assert_allclose(view((x, t)).numpy(), model((x, t)).numpy(),
                               atol=ATOL)
    alpha.assign(0.0)
    ref = closed_loop_rollout(model, SimpleNamespace(seq_len=x.shape[1],
                                                     test_x=x, test_t=t))
    np.testing.assert_allclose(view((x, t)).numpy(), ref[:, 1:], atol=ATOL)
    # alpha = 0 reads only y_0 of the inputs
    x2 = x.copy()
    x2[:, 1:] = 0.0
    np.testing.assert_allclose(view((x2, t)).numpy(), ref[:, 1:], atol=ATOL)


@pytest.mark.parametrize("cell,wiring", ARMS)
def test_view_adds_no_parameters(cell, wiring):
    model = build_lotka_volterra_model(wiring, cell, size=8, seed=0)
    alpha = tf.Variable(0.5, trainable=False)
    view = forced_rollout_model(model, alpha)
    assert param_counts(view) == param_counts(model)
    assert ({v.ref() for v in view.weights}
            == {v.ref() for v in model.weights})
    assert alpha.ref() not in {v.ref() for v in view.weights}


def test_stacked_cell_matches_stacked_dense():
    x, t = _data()
    model = build_lotka_volterra_model("dense3", "cfc_lrc", size=8, seed=0)
    rnns = [l for l in model.layers if isinstance(l, tf.keras.layers.RNN)]
    assert len(rnns) == 3
    stacked = tf.keras.layers.RNN(StackedCell([l.cell for l in rnns]),
                                  return_sequences=True)
    h_ref = tf.keras.Model(model.inputs, rnns[-1].output)((x, t)).numpy()
    np.testing.assert_allclose(stacked((x, t)).numpy(), h_ref, atol=ATOL)


def test_non_forceable_wiring_is_rejected():
    model = build_lotka_volterra_model("tbt_cncp_concat", "cfc_lrc", size=8,
                                       seed=0)
    with pytest.raises(ValueError, match="forced_rollout_model"):
        forced_rollout_model(model, tf.Variable(1.0, trainable=False))


def test_alpha_schedule():
    alpha = tf.Variable(1.0, trainable=False)
    sched = AlphaSchedule(alpha, epochs=200, alpha_min=0.2)
    for epoch in range(200):
        sched.on_epoch_begin(epoch)
        assert float(alpha.numpy()) == pytest.approx(sched.curve[-1])
    curve = np.array(sched.curve)
    assert curve[0] == 1.0 and curve[50] == pytest.approx(0.6)
    assert np.all(np.diff(curve[:101]) < 0)          # linear decrease ...
    np.testing.assert_allclose(np.diff(curve[:101]), -0.008)
    assert np.all(curve[100:] == 0.2)                # ... then constant
