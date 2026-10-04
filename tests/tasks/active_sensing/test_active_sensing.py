# tests/tasks/active_sensing/test_active_sensing.py
# Validation of the tbt_cNCP active-sensing task + TbtCorticalColumnCell:
#   - dataset shapes/determinism for both object modes; occlusion changes only
#     the test patches;
#   - the cell outputs concat([L2/3, L5ET]) and accepts the 3-tuple input;
#   - use_location on/off actually changes the computation (the ablation is
#     real);
#   - EVERY trainable variable -- including the NEW location weights W_loc_L4 /
#     gain_loc_L4 -- receives gradient under the combined loss;
#   - the model builds + forward-passes for every wiring with the object head
#     and the optional prediction head.
import numpy as np
import pytest
import tensorflow as tf

from src.tasks.active_sensing import (load_active_sensing,
                                      build_active_sensing_model,
                                      ActiveSensingData, encode_location,
                                      ACTIVE_WIRINGS)
from src.wirings.tbt_cncp import TbtCorticalColumnCell


def _small(mode="compositional"):
    return load_active_sensing(n_per_class=6, seq_len=5, mode=mode)


# --- dataset ---

@pytest.mark.parametrize("mode", ["compositional", "shapes"])
def test_dataset_shapes(mode):
    d = _small(mode)
    assert isinstance(d, ActiveSensingData)
    assert d.num_classes == 6 and d.loc_dim == 12 and d.patch_dim == 25
    n = d.train_patch.shape[0]
    assert d.train_patch.shape == (n, 5, 25)
    assert d.train_loc.shape == (n, 5, 12)
    assert d.train_next.shape == (n, 5, 25)
    assert d.train_y.min() >= 0 and d.train_y.max() < 6


def test_dataset_deterministic():
    a, b = _small(), _small()
    assert np.array_equal(a.train_patch, b.train_patch)
    assert np.array_equal(a.test_y, b.test_y)


def test_encode_location_shape_and_locality():
    p = np.array([[0.2, 0.2], [0.21, 0.21], [0.9, 0.1]], dtype=np.float32)
    code = encode_location(p)
    assert code.shape == (3, 12)
    # nearby locations -> similar codes (grid-cell property)
    assert np.linalg.norm(code[0] - code[1]) < np.linalg.norm(code[0] - code[2])


def test_occlusion_changes_only_test():
    clean = load_active_sensing(n_per_class=6, seq_len=5)
    occ = load_active_sensing(n_per_class=6, seq_len=5, occlude=True)
    assert np.array_equal(clean.train_patch, occ.train_patch)   # train untouched
    assert not np.array_equal(clean.test_patch, occ.test_patch)  # test occluded


def test_occlude_frac_graded():
    # Graded occlusion (Iteration 3): more frac -> more blanked TEST patches;
    # train stays clean; frac 0.5 matches the occlude=True shorthand.
    clean = load_active_sensing(n_per_class=6, seq_len=5, occlude_frac=0.0)
    half = load_active_sensing(n_per_class=6, seq_len=5, occlude_frac=0.5)
    binary = load_active_sensing(n_per_class=6, seq_len=5, occlude=True)
    more = load_active_sensing(n_per_class=6, seq_len=5, occlude_frac=0.75)
    assert np.array_equal(clean.train_patch, more.train_patch)   # train untouched
    assert np.array_equal(half.test_patch, binary.test_patch)    # 0.5 == occlude
    blanked = lambda d: float((d.test_patch == 0.0).mean())
    assert blanked(clean) < blanked(half) < blanked(more)


# --- the cell ---

def _cell(use_location=True):
    return TbtCorticalColumnCell(
        cell_cls="cfc_lrc", lamina_units={"L23": 8, "L5ET": 8},
        seed=0, use_location=use_location, elastance_type="asymmetric")


def test_output_size_is_l23_plus_l5et():
    c = _cell()
    c.build((tf.TensorShape([None, 25]), tf.TensorShape([None, 1]),
             tf.TensorShape([None, 12])))
    assert c.output_size == c._units["L23"] + c._units["L5ET"]


def test_use_location_flag_changes_output():
    x = tf.random.normal((3, 25)); t = tf.ones((3, 1)); loc = tf.random.normal((3, 12))
    on, off = _cell(True), _cell(False)
    for c in (on, off):
        c.build((tf.TensorShape([None, 25]), tf.TensorShape([None, 1]),
                 tf.TensorShape([None, 12])))
    st = on.get_initial_state(batch_size=3, dtype=tf.float32)
    o_on, _ = on((x, t, loc), st)
    o_off, _ = off((x, t, loc), st)
    # different location weights + gating -> different outputs
    assert not np.allclose(o_on.numpy(), o_off.numpy())


def test_film_zero_init_is_identity_then_active():
    # FiLM (affine L6a->L4) must start at the identity -- zero-init scale/shift
    # make location irrelevant -- and become location-sensitive once the maps
    # are nonzero. One cell, compared with itself, so no cross-cell RNG coupling.
    x = tf.random.normal((3, 25)); t = tf.ones((3, 1))
    loc_a = tf.random.normal((3, 12)); loc_b = tf.random.normal((3, 12))
    c = _cell("film")
    c.build((tf.TensorShape([None, 25]), tf.TensorShape([None, 1]),
             tf.TensorShape([None, 12])))
    st = c.get_initial_state(batch_size=3, dtype=tf.float32)
    # zero-init scale/shift -> different locations produce the SAME output
    o_a0, _ = c((x, t, loc_a), st)
    o_b0, _ = c((x, t, loc_b), st)
    assert np.allclose(o_a0.numpy(), o_b0.numpy(), atol=1e-6)
    # nonzero scale/shift -> location now changes the output
    c.W_loc_scale.assign(tf.random.normal(c.W_loc_scale.shape) * 0.2)
    c.W_loc_shift.assign(tf.random.normal(c.W_loc_shift.shape) * 0.2)
    o_a1, _ = c((x, t, loc_a), st)
    o_b1, _ = c((x, t, loc_b), st)
    assert not np.allclose(o_a1.numpy(), o_b1.numpy())


def test_film_variables_get_gradient():
    # The affine FiLM maps must receive gradient from step one despite the
    # zero-init identity start (d/dW = location (x) upstream, not the weight).
    d = _small()
    m = build_active_sensing_model(
        "tbt_cncp_film", "cfc_lrc", size=16, seed=0, num_classes=d.num_classes,
        patch_dim=d.patch_dim, loc_dim=d.loc_dim, use_prediction=True,
        elastance_type="asymmetric")
    yseq = np.repeat(d.train_y[:, None], d.seq_len, axis=1).astype(np.int32)
    with tf.GradientTape() as tape:
        logits, pred = m([d.train_patch, d.train_time, d.train_loc],
                         training=True)
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
            yseq, logits, from_logits=True)) + \
            tf.reduce_mean((pred - d.train_next) ** 2)
    grads = tape.gradient(loss, m.trainable_variables)
    dead = [v.name for v, g in zip(m.trainable_variables, grads)
            if g is None or float(tf.reduce_sum(tf.abs(g))) == 0.0]
    assert dead == [], f"dead variables: {dead}"
    names = [v.name for v in m.trainable_variables]
    assert any("W_loc_scale" in n for n in names)
    assert any("W_loc_shift" in n for n in names)


# --- model build + gradients ---

@pytest.mark.parametrize("wiring", list(ACTIVE_WIRINGS))
def test_model_builds_and_forwards(wiring):
    d = _small()
    m = build_active_sensing_model(
        wiring, "cfc_lrc", size=16, seed=0, num_classes=d.num_classes,
        patch_dim=d.patch_dim, loc_dim=d.loc_dim, elastance_type="asymmetric")
    out = m([d.train_patch, d.train_time, d.train_loc])
    logits = out[0] if isinstance(out, (list, tuple)) else out
    assert logits.shape[-1] == d.num_classes
    assert logits.shape[1] == d.seq_len


def test_all_variables_get_gradient_under_combined_loss():
    d = _small()
    m = build_active_sensing_model(
        "tbt_cncp", "cfc_lrc", size=16, seed=0, num_classes=d.num_classes,
        patch_dim=d.patch_dim, loc_dim=d.loc_dim, use_prediction=True,
        elastance_type="asymmetric")
    yseq = np.repeat(d.train_y[:, None], d.seq_len, axis=1).astype(np.int32)
    with tf.GradientTape() as tape:
        logits, pred = m([d.train_patch, d.train_time, d.train_loc],
                         training=True)
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
            yseq, logits, from_logits=True)) + \
            tf.reduce_mean((pred - d.train_next) ** 2)
    grads = tape.gradient(loss, m.trainable_variables)
    dead = [v.name for v, g in zip(m.trainable_variables, grads)
            if g is None or float(tf.reduce_sum(tf.abs(g))) == 0.0]
    assert dead == [], f"dead variables: {dead}"
    names = [v.name for v in m.trainable_variables]
    assert any("W_loc_L4" in n for n in names)
    assert any("gain_loc_L4" in n for n in names)


# --- multi-column voting ---

def test_multicolumn_dataset_shape():
    d = load_active_sensing(n_per_class=4, seq_len=5, n_columns=3)
    assert d.n_columns == 3
    assert d.train_patch.ndim == 4 and d.train_patch.shape[2] == 3
    assert d.train_loc.shape[2] == 3
    # single-column path stays 3D (backward compatible)
    d1 = load_active_sensing(n_per_class=4, seq_len=5, n_columns=1)
    assert d1.train_patch.ndim == 3


def test_voting_model_builds_forwards_and_votes():
    from src.tasks.active_sensing import build_voting_model
    d = load_active_sensing(n_per_class=4, seq_len=5, n_columns=3)
    m = build_voting_model(
        3, "cfc_lrc", size=16, seed=0, num_classes=d.num_classes,
        patch_dim=d.patch_dim, loc_dim=d.loc_dim, elastance_type="asymmetric")
    out = m([d.train_patch, d.train_time, d.train_loc])
    assert out.shape[-1] == d.num_classes and out.shape[1] == d.seq_len
    # the learnable voting strength exists
    assert any("vote_raw" in v.name for v in m.trainable_variables)


def test_voting_model_film_builds_forwards_and_carries_film_maps():
    # Iteration 2: FiLM location mode inside the voting cell. Location must reach
    # each column through the affine L4 maps (W_loc_scale/shift), NOT concat.
    from src.tasks.active_sensing import build_voting_model
    d = load_active_sensing(n_per_class=4, seq_len=5, n_columns=3)
    m = build_voting_model(
        3, "cfc_lrc", size=16, seed=0, num_classes=d.num_classes,
        patch_dim=d.patch_dim, loc_dim=d.loc_dim, location_mode="film",
        elastance_type="asymmetric")
    out = m([d.train_patch, d.train_time, d.train_loc])
    assert out.shape[-1] == d.num_classes and out.shape[1] == d.seq_len
    names = [v.name for v in m.trainable_variables]
    assert any("W_loc_scale" in n for n in names)
    assert any("W_loc_shift" in n for n in names)
    assert any("vote_raw" in n for n in names)
