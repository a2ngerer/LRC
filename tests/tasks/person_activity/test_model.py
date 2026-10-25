# tests/tasks/person_activity/test_model.py
# Elapsed time as an input feature for the discrete cells (issue #27):
#   - only gru / lstm are affected, continuous-time cells are unchanged;
#   - the parameter count grows by one input column per gate and layer;
#   - dt_feature=False rebuilds the previous model (dt discarded);
#   - under dense3 every layer receives dt.
import numpy as np
import pytest
import tensorflow as tf

from src.tasks.person_activity.model import (build_person_activity_model,
                                             dt_feature_applies)
from src.wirings import param_counts

T, F, C, U = 32, 7, 7, 8


def _batch(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(2, T, F)).astype(np.float32)
    t = rng.uniform(0.1, 1.0, size=(2, T, 1)).astype(np.float32)
    return x, t


def _build(wiring, cell, dt_feature, **kw):
    return build_person_activity_model(wiring, cell, size=U, seed=0,
                                       dt_feature=dt_feature, **kw)


def test_dt_feature_applies_to_discrete_cells_only():
    assert dt_feature_applies("gru") and dt_feature_applies("lstm")
    assert not dt_feature_applies("gru", False)
    assert not dt_feature_applies("cfc_lrc") and not dt_feature_applies("ltc")


@pytest.mark.parametrize("cell,wiring,extra", [
    ("gru", "dense", 3 * U),            # one kernel row per gate (3 gates)
    ("lstm", "dense", 4 * U),           # 4 gates
    ("gru", "dense3", 3 * 3 * U),       # every one of the three layers
])
def test_dt_feature_adds_one_input_column_per_gate(cell, wiring, extra):
    on, off = _build(wiring, cell, True), _build(wiring, cell, False)
    assert on.count_params() - off.count_params() == extra
    x, t = _batch()
    assert tuple(on([x, t]).shape) == tuple(off([x, t]).shape) == (2, T, C)


@pytest.mark.parametrize("wiring", ["ncp", "cncp", "cncp2"])
def test_effective_count_recounts_on_masked_wirings(wiring):
    on = param_counts(_build(wiring, "gru", True))
    off = param_counts(_build(wiring, "gru", False))
    assert on["params_effective"] > off["params_effective"]


def test_dt_feature_off_is_the_previous_model():
    """gru/dense without the flag: the plain GRU count (Keras reset_after
    bias 2 x 3u) plus the head, no Concatenate, dt discarded."""
    model = _build("dense", "gru", False)
    assert model.count_params() == 3 * U * (F + U + 2) + U * C + C
    assert not any(isinstance(l, tf.keras.layers.Concatenate)
                   for l in model.layers)
    x, t = _batch()
    assert np.allclose(model([x, t]), model([x, 3.0 * t]))


def test_continuous_time_cells_unchanged():
    kw = dict(elastance_type="asymmetric")
    on, off = (_build("dense3", "cfc_lrc", f, **kw) for f in (True, False))
    assert [w.shape for w in on.get_weights()] == \
        [w.shape for w in off.get_weights()]
    assert all(np.array_equal(a, b)
               for a, b in zip(on.get_weights(), off.get_weights()))


def test_dense3_feeds_dt_to_every_layer():
    model = _build("dense3", "gru", True)
    time_in = model.inputs[1]
    rnns = [l for l in model.layers if isinstance(l, tf.keras.layers.RNN)]
    assert len(rnns) == 3
    for layer, width in zip(rnns, (F, U, U)):
        seq = layer.input[0]
        assert seq.shape[-1] == width + 1
        concat = seq._keras_history.layer
        assert isinstance(concat, tf.keras.layers.Concatenate)
        assert any(i is time_in or i.ref() == time_in.ref()
                   for i in concat.input)
    x, t = _batch()
    assert not np.allclose(model([x, t]), model([x, 3.0 * t]))
