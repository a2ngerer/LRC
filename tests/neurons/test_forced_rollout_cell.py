# tests/neurons/test_forced_rollout_cell.py
# Issue #25: the ForcedRolloutCell view shares the plain LV model's weights;
# alpha = 1 reproduces its teacher-forced predictions, alpha = 0 the reference
# closed_loop_rollout; parameter counts unchanged; StackedCell == stacked_dense;
# AlphaSchedule is linear 1 -> alpha_min over the first half, then constant.
# Issue #62: the per-step forcing input (anchored view) feeds the truth up to
# the anchor, so the state there is the teacher-forced state, then runs free.
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from experiments.run_lotka_volterra_benchmark import (anchored_rollouts,
                                                     closed_loop_rollout)
from src.neurons.forced_rollout_cell import (AlphaSchedule, ForcedRolloutCell,
                                             StackedCell)
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


def test_view_keeps_the_dt_scale_layer():
    """dt_scale != 1 (issue #28) adds a Rescaling layer on the time input; the
    view must route its time through it as well."""
    x, t = _data()
    model = build_lotka_volterra_model("dense3", "cfc_lrc", size=8, seed=0,
                                       dt_scale=2.0)
    alpha = tf.Variable(1.0, trainable=False)
    view = forced_rollout_model(model, alpha)
    np.testing.assert_allclose(view((x, t)).numpy(), model((x, t)).numpy(),
                               atol=ATOL)
    alpha.assign(0.0)
    ref = closed_loop_rollout(model, SimpleNamespace(seq_len=x.shape[1],
                                                     test_x=x, test_t=t))
    np.testing.assert_allclose(view((x, t)).numpy(), ref[:, 1:], atol=ATOL)


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


class _Sum(tf.keras.layers.AbstractRNNCell):
    """Toy core: h_t = h_{t-1} + x_t, so the state is the running input sum."""

    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return 1

    def call(self, inputs, states):
        h = states[0] + inputs[0]
        return h, [h]


def test_anchored_warm_up_ends_in_the_teacher_forced_state():
    """Two-step toy system: warm-up y_0 = 1, y_1 = 10 (anchor 1), then one
    free-run step. The state at the anchor is the teacher-forced sum 11; the
    free run feeds yhat_1 = 11 back (h = 22) and ignores y_2 = 100."""
    y = tf.constant([[[1.0], [10.0], [100.0]]])
    dt = tf.ones((1, 3, 1))
    rnn = tf.keras.layers.RNN(ForcedRolloutCell(_Sum(), [], None),
                              return_sequences=True)
    anchored = rnn((y, dt, tf.constant([[[1.0], [1.0], [0.0]]]))).numpy()
    teacher = rnn((y, dt, tf.ones((1, 3, 1)))).numpy()
    assert anchored.ravel().tolist() == [1.0, 11.0, 22.0]
    assert teacher.ravel().tolist() == [1.0, 11.0, 111.0]


@pytest.mark.parametrize("cell,wiring", ARMS)
def test_anchored_view_matches_the_reference_rollout(cell, wiring):
    x, t = _data()
    data = SimpleNamespace(seq_len=x.shape[1], test_x=x, test_t=t)
    model = build_lotka_volterra_model(wiring, cell, size=8, seed=0)
    view = forced_rollout_model(model)              # alpha None: force input
    assert len(view.inputs) == 3
    assert {v.ref() for v in view.weights} == {v.ref() for v in model.weights}
    fast = anchored_rollouts(model, data, "test", False, (0, 3, 7), 8, view)
    ref = anchored_rollouts(model, data, "test", False, (0, 3, 7), 8)
    teacher = model((x, t)).numpy()
    for a in (0, 3, 7):
        assert fast[a].shape == (len(x), 8, 2)
        np.testing.assert_allclose(fast[a], ref[a], atol=ATOL)
        # the first free-run prediction comes from the teacher-forced state
        np.testing.assert_allclose(fast[a][:, 0], teacher[:, a], atol=ATOL)
    np.testing.assert_allclose(
        fast[0], closed_loop_rollout(model, data)[:, 1:9], atol=ATOL)
    # the free run reads no ground truth after the anchor
    x2 = x.copy()
    x2[:, 4:] = 0.0
    again = anchored_rollouts(model, SimpleNamespace(
        seq_len=x.shape[1], test_x=x2, test_t=t), "test", False, (3,), 8, view)
    np.testing.assert_allclose(again[3], fast[3], atol=ATOL)
