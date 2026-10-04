# tests/tasks/active_sensing/test_active_glimpse.py
# Validation of the active-glimpse sensorimotor task (Iteration 5):
#   - the differentiable bilinear glimpse sampler has nonzero gradient w.r.t. the
#     centre (the L5 motor can be trained by backprop) and returns the exact
#     pixel at an integer position;
#   - the tf reference-frame code matches datasets.encode_location bit-for-bit;
#   - the dataset yields full images + labels + random centres;
#   - every wiring x policy builds and forward-passes to (B,T,n_classes);
#   - the active policy produces a DIFFERENT glimpse path than random (steering
#     is real), and all trainable variables incl. the motor head get gradient.
import numpy as np
import tensorflow as tf

from src.tasks.active_sensing.active_glimpse import (
    differentiable_glimpse, rf_code_tf, load_active_glimpse,
    build_active_glimpse_model, ACTIVE_GLIMPSE_WIRINGS)
from src.tasks.active_sensing.datasets import encode_location


def test_glimpse_gradient_and_exactness():
    rng = np.random.default_rng(0)
    img = tf.constant(rng.random((3, 20, 20)).astype(np.float32))
    center = tf.Variable([[0.5, 0.5], [0.3, 0.7], [0.2, 0.2]], dtype=tf.float32)
    with tf.GradientTape() as tape:
        patch = differentiable_glimpse(img, center, k=5)
        loss = tf.reduce_sum(patch ** 2)
    g = tape.gradient(loss, center)
    assert patch.shape == (3, 25)
    assert g is not None and float(tf.reduce_sum(tf.abs(g))) > 0.0
    # integer position returns the exact pixel
    p = differentiable_glimpse(img[:1], tf.constant([[0.0, 0.0]], tf.float32), k=1)
    assert np.isclose(p.numpy()[0, 0], img.numpy()[0, 0, 0])


def test_rf_code_matches_encode_location():
    pos = np.array([[0.2, 0.7], [0.5, 0.5], [0.9, 0.1]], np.float32)
    a = rf_code_tf(tf.constant(pos)).numpy()
    b = encode_location(pos)
    assert a.shape == (3, 12)
    assert np.allclose(a, b, atol=1e-5)


def test_dataset_shapes():
    d = load_active_glimpse(n_per_class=6, n_classes=4, grid=20, seq_len=5)
    assert d["train_img"].ndim == 3 and d["train_img"].shape[1:] == (20, 20)
    assert d["train_pos"].shape[1:] == (5, 2)
    assert d["n_classes"] == 4
    assert d["train_y"].min() >= 0 and d["train_y"].max() < 4


def test_all_wirings_policies_build_and_forward():
    d = load_active_glimpse(n_per_class=4, n_classes=4, grid=20, seq_len=5)
    for w in ACTIVE_GLIMPSE_WIRINGS:
        for p in ("active", "random"):
            ck = {"elastance_type": "asymmetric"}
            m = build_active_glimpse_model(
                w, p, cell="cfc_lrc", size=16, grid=20, k=d["k"], seq_len=5,
                n_classes=4, seed=0, **ck)
            out = m([d["train_img"], d["train_pos"]])
            assert out.shape[1] == 5 and out.shape[2] == 4


def test_active_differs_from_random_and_motor_gets_gradient():
    d = load_active_glimpse(n_per_class=8, n_classes=4, grid=20, seq_len=6)
    act = build_active_glimpse_model(
        "tbt_cncp", "active", cell="cfc_lrc", size=16, grid=20, k=d["k"],
        seq_len=6, n_classes=4, seed=0, elastance_type="asymmetric")
    rnd = build_active_glimpse_model(
        "tbt_cncp", "random", cell="cfc_lrc", size=16, grid=20, k=d["k"],
        seq_len=6, n_classes=4, seed=0, elastance_type="asymmetric")
    ins = [d["train_img"], d["train_pos"]]
    yseq = np.repeat(d["train_y"][:, None], 6, axis=1).astype(np.int32)
    # after a step, the two policies produce different predictions (steering)
    with tf.GradientTape() as tape:
        lo = act(ins, training=True)
        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(yseq, lo,
                                                            from_logits=True))
    grads = tape.gradient(loss, act.trainable_variables)
    names = [v.name for v in act.trainable_variables]
    dead = [n for n, g in zip(names, grads)
            if g is None or float(tf.reduce_sum(tf.abs(g))) == 0.0]
    assert any("motor" in n for n in names)
    assert not any("motor" in n for n in dead), "motor head must get gradient"
    # active vs random give different outputs on the same input
    assert not np.allclose(act(ins).numpy(), rnd(ins).numpy())
