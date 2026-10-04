# tests/tasks/person_activity/test_person_activity.py
# Light validation of the person-activity benchmark pieces:
#   - build_person_activity_model builds and forward-passes for every
#     wiring x {cfc_lrc, gru} with the (features, time) two-input contract;
#   - gradients flow through every trainable variable;
#   - the cncp/ncp parameter counts stay comparable (fairness invariant);
#   - multi-state cells are rejected;
#   - the dataset loader returns the documented shapes (skipped when the
#     data file has not been downloaded).
import os

import numpy as np
import pytest
import tensorflow as tf

from src.wirings import effective_param_count, match_param_budget
from src.tasks.person_activity import (build_person_activity_model,
                                       load_person_activity,
                                       DEFAULT_DATA_PATH)

SEQ_LEN = 32
NUM_CLASSES = 7
FEATURES = 7


def _dummy_batch(batch=2, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(batch, SEQ_LEN, FEATURES)).astype(np.float32)
    t = rng.uniform(0.1, 1.0, size=(batch, SEQ_LEN, 1)).astype(np.float32)
    y = rng.integers(0, NUM_CLASSES, size=(batch, SEQ_LEN)).astype(np.int32)
    return x, t, y


# --- build + forward pass ---

@pytest.mark.parametrize("wiring", ["cncp", "ncp", "dense"])
@pytest.mark.parametrize("cell", ["cfc_lrc", "gru"])
def test_build_and_forward_shapes(wiring, cell):
    model = build_person_activity_model(wiring, cell, size=16, seed=0)
    x, t, _ = _dummy_batch()
    out = model([x, t])
    assert tuple(out.shape) == (2, SEQ_LEN, NUM_CLASSES)
    assert bool(tf.reduce_all(tf.math.is_finite(out)))


# --- gradient flow ---

@pytest.mark.parametrize("wiring", ["cncp", "ncp", "dense"])
def test_gradients_flow(wiring):
    model = build_person_activity_model(wiring, "cfc_lrc", size=16, seed=0)
    x, t, y = _dummy_batch()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    with tf.GradientTape() as tape:
        logits = model([x, t])
        loss = loss_fn(y, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    missing = [v.name for v, g in zip(model.trainable_variables, grads)
               if g is None]
    assert not missing, f"no gradient for: {missing}"


# --- fairness: comparable capacity at the same size knob ---

@pytest.mark.parametrize("cell", ["cfc_lrc", "gru"])
def test_budget_matched_arms_are_comparable(cell):
    """Each arm is sized on its own knob to the same EFFECTIVE budget
    (masked-off ncp synapses excluded, docs/ncp-wiring-fix-2026-09-17.md);
    the matched counts must then agree within the fair-comparison band."""
    def count(wiring):
        return lambda size: effective_param_count(
            build_person_activity_model(wiring, cell, size=size, seed=0))
    counts = {w: match_param_budget(count(w), 4000, range(2, 129))[1]
              for w in ("dense", "ncp", "cncp")}
    lo, hi = min(counts.values()), max(counts.values())
    assert hi / lo < 1.5, f"budget-matched effective counts {counts}"


# --- input validation ---

@pytest.mark.parametrize("cell", ["lstm", "mm_lrc"])
def test_multi_state_cells_rejected(cell):
    with pytest.raises(ValueError, match="single-state"):
        build_person_activity_model("ncp", cell, size=16, seed=0)


def test_unknown_wiring_rejected():
    with pytest.raises(ValueError, match="wiring"):
        build_person_activity_model("mesh", "gru", size=16, seed=0)


# --- dataset loader (needs the downloaded data file) ---

requires_data = pytest.mark.skipif(
    not os.path.isfile(DEFAULT_DATA_PATH),
    reason="person-activity data file not downloaded "
           "(run download_dataset.sh)")


@requires_data
def test_load_person_activity_shapes():
    data = load_person_activity(seq_len=SEQ_LEN)
    assert data.feature_size == FEATURES
    assert data.num_classes == NUM_CLASSES
    assert data.seq_len == SEQ_LEN
    assert data.train_x.shape[1:] == (SEQ_LEN, FEATURES)
    assert data.train_t.shape[1:] == (SEQ_LEN, 1)
    assert data.train_y.shape[1:] == (SEQ_LEN,)
    assert data.test_x.shape[1:] == (SEQ_LEN, FEATURES)
    # 80/20 split with a sensible number of sequences on both sides.
    assert data.test_x.shape[0] > 100
    assert data.train_x.shape[0] > 3 * data.test_x.shape[0]
    # labels in range
    assert data.train_y.min() >= 0
    assert data.train_y.max() < NUM_CLASSES
    # Elapsed times are finite and overwhelmingly positive. A handful of
    # NEGATIVE values are expected: the raw UCI file has out-of-order
    # timestamps, and the parsing is kept identical to the original
    # PersonData for comparability, so they are not clamped.
    assert np.isfinite(data.train_t).all()
    assert (data.train_t > 0).mean() > 0.99
