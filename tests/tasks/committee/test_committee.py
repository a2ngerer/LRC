"""Tests for the partial-view voting committee (Iteration 7)."""
import numpy as np
import pytest
import tensorflow as tf

from src.tasks.committee.views import (partition_masks, masked_views,
                                       noisy_views, drop_features, add_noise,
                                       make_views)
from src.tasks.committee.model import (build_committee_model, ChannelDropout,
                                       NoiseAugment, COMMITTEE_WIRINGS)
from src.wirings import effective_param_count, match_param_budget
from src.wirings.committee import CommitteeVotingCell


# --- view operators ---

def test_partition_masks_disjoint_covers_all():
    m = partition_masks(7, 3, overlap=0, seed=1)
    assert m.shape == (3, 7)
    assert m.any(axis=1).all()              # no empty column
    assert m.sum(axis=0).min() >= 1         # every feature covered by >= 1 col
    assert m.sum() == 7                     # disjoint partition: exactly F trues


def test_partition_masks_overlap_adds_features():
    base = partition_masks(7, 3, overlap=0, seed=1).sum()
    more = partition_masks(7, 3, overlap=2, seed=1).sum()
    assert more > base


def test_masked_views_zero_outside_owned_features():
    X = np.ones((2, 4, 7), np.float32)
    m = partition_masks(7, 3, seed=1)
    v = masked_views(X, m)
    assert v.shape == (2, 4, 3, 7)
    for k in range(3):
        active = (v[:, :, k, :] != 0).any(axis=(0, 1))
        assert np.array_equal(active, m[k])   # column k sees exactly its slice


def test_drop_features_zeros_whole_channels():
    X = np.ones((3, 5, 8), np.float32)
    out = drop_features(X, 0.5, seed=0)
    zeroed = (out == 0).all(axis=(0, 1))       # whole channels blanked
    assert zeroed.sum() == 4                   # round(0.5 * 8)
    assert drop_features(X, 0.0).sum() == X.sum()


def test_noisy_views_shape_and_columns_differ():
    X = np.ones((2, 3, 5), np.float32)
    v = noisy_views(X, 4, noise=0.1, seed=0)
    assert v.shape == (2, 3, 4, 5)
    assert not np.allclose(v[:, :, 0, :], v[:, :, 1, :])


def test_make_views_dispatch():
    X = np.ones((2, 3, 6), np.float32)
    vp, mp = make_views(X, 3, mode="partition", seed=0)
    assert vp.shape == (2, 3, 3, 6) and mp is not None
    vn, mn = make_views(X, 3, mode="noisy", seed=0)
    assert vn.shape == (2, 3, 3, 6) and mn is None
    with pytest.raises(ValueError):
        make_views(X, 3, mode="bogus")


# --- voting cell ---

def test_committee_cell_shapes_and_vote_in_unit_interval():
    cell = CommitteeVotingCell(n_columns=3, cell_cls="cfc_lrc", units=8,
                               elastance_type="asymmetric")
    cell.build((tf.TensorShape([None, 3, 5]), tf.TensorShape([None, 1])))
    assert cell.state_size == [8, 8, 8]
    assert cell.output_size == 8
    B = 4
    feats = tf.random.normal([B, 3, 5])
    out, ns = cell((feats, tf.ones([B, 1])),
                   cell.get_initial_state(batch_size=B))
    assert out.shape == (B, 8) and len(ns) == 3
    assert 0.0 < float(tf.nn.sigmoid(cell.vote_raw)) < 1.0


def test_committee_cell_rejects_multistate_cells():
    with pytest.raises(ValueError):
        CommitteeVotingCell(n_columns=2, cell_cls="lstm", units=4)


# --- model builder ---

@pytest.mark.parametrize("wiring", COMMITTEE_WIRINGS)
def test_model_builds_forwards_fits(wiring):
    F, C, T, K = 7, 7, 6, 3
    size = 24 if wiring == "cncp" else 32
    m = build_committee_model(wiring, K, cell="cfc_lrc", size=size,
                              feature_size=F, seq_len=T, num_classes=C,
                              elastance_type="asymmetric", seed=0)
    X = np.random.default_rng(0).standard_normal((10, T, K, F)).astype("float32")
    t = np.ones((10, T, 1), "float32")
    y = np.random.default_rng(1).integers(0, C, (10, T)).astype("int32")
    assert m.predict([X, t], verbose=0).shape == (10, T, C)
    m.fit([X, t], y, epochs=1, batch_size=5, verbose=0)


def test_params_independent_of_K_weight_sharing():
    def P(K):
        return build_committee_model(
            "cncp", K, cell="cfc_lrc", size=24, feature_size=7, seq_len=6,
            num_classes=7, elastance_type="asymmetric", seed=0).count_params()
    assert P(1) == P(2) == P(4)


def test_arms_budget_matched_on_effective_params():
    """Each arm is sized on its own knob to the same EFFECTIVE budget (masked-off
    ncp synapses excluded); the matched counts agree within the fair band."""
    def count(w):
        return lambda s: effective_param_count(build_committee_model(
            w, 2, cell="cfc_lrc", size=s, feature_size=7, seq_len=6,
            num_classes=7, elastance_type="asymmetric", seed=0))
    counts = {w: match_param_budget(count(w), 4000, range(2, 129))[1]
              for w in COMMITTEE_WIRINGS}
    lo, hi = min(counts.values()), max(counts.values())
    assert hi / lo < 1.5, f"budget-matched effective counts {counts}"


# --- consensus controls (2026-10-04) ---

def _cell_states(cell, K, F=5, B=4, seed=0):
    tf.keras.utils.set_random_seed(seed)
    cell.build((tf.TensorShape([None, K, F]), tf.TensorShape([None, 1])))
    feats = tf.random.normal([B, K, F], seed=seed)
    return cell((feats, tf.ones([B, 1])), cell.get_initial_state(batch_size=B))


def test_fixed_lambda_zero_has_no_variable_and_leaves_columns_independent():
    cell = CommitteeVotingCell(n_columns=3, cell_cls="gru", units=6,
                               vote_lambda=0.0)
    out, ns = _cell_states(cell, 3)
    assert not hasattr(cell, "vote_raw")
    assert float(cell.vote_strength) == 0.0
    # independent columns: each new state is the shared cell's own update
    tf.keras.utils.set_random_seed(0)
    feats = tf.random.normal([4, 3, 5], seed=0)
    for k in range(3):
        _, ref = cell.col((feats[:, k, :], tf.ones([4, 1])), [tf.zeros([4, 6])])
        assert np.allclose(ns[k].numpy(), ref[0].numpy(), atol=1e-6)
    assert np.allclose(out.numpy(), np.mean([s.numpy() for s in ns], axis=0),
                       atol=1e-6)


def test_mean_readout_is_independent_of_lambda():
    """mean_k((1-lam) h_k + lam hbar) == hbar: the current output does not see
    the voting strength; only the stored states do."""
    outs = []
    for lam in (0.0, 0.5, 1.0):
        cell = CommitteeVotingCell(n_columns=3, cell_cls="gru", units=6,
                                   vote_lambda=lam)
        out, ns = _cell_states(cell, 3)
        outs.append(out.numpy())
        spread = np.std([s.numpy() for s in ns], axis=0).max()
        assert (spread < 1e-6) == (lam == 1.0)      # lam=1 collapses the states
    assert np.allclose(outs[0], outs[1], atol=1e-6)
    assert np.allclose(outs[0], outs[2], atol=1e-6)


@pytest.mark.parametrize("K", [3, 4])
def test_median_consensus_matches_numpy_lower_median(K):
    from src.wirings.committee import consensus_target
    cols = [tf.constant(np.random.default_rng(k).normal(size=(2, 5)),
                        dtype=tf.float32) for k in range(K)]
    med = consensus_target(cols, "median").numpy()
    expect = np.sort(np.stack([c.numpy() for c in cols]), axis=0)[(K - 1) // 2]
    assert np.allclose(med, expect)
    cell = CommitteeVotingCell(n_columns=K, cell_cls="gru", units=6,
                               consensus="median", vote_lambda=0.0)
    out, ns = _cell_states(cell, K)
    expect = np.sort(np.stack([s.numpy() for s in ns]), axis=0)[(K - 1) // 2]
    assert np.allclose(out.numpy(), expect, atol=1e-6)


def test_invalid_consensus_and_lambda_raise():
    with pytest.raises(ValueError):
        CommitteeVotingCell(n_columns=2, cell_cls="gru", units=4, consensus="max")
    with pytest.raises(ValueError):
        CommitteeVotingCell(n_columns=2, cell_cls="gru", units=4, vote_lambda=1.5)


def test_vote_and_readout_slices_shape_the_output():
    cell = CommitteeVotingCell(n_columns=2, cell_cls="gru", units=8,
                               vote_slice=slice(2, 6), readout_slice=slice(0, 2),
                               vote_lambda=1.0)
    out, ns = _cell_states(cell, 2)
    assert cell.output_size == 2 and out.shape == (4, 2)
    a, b = ns[0].numpy(), ns[1].numpy()
    assert np.allclose(a[:, 2:6], b[:, 2:6])              # voted slice shared
    assert not np.allclose(a[:, :2], b[:, :2])            # the rest stays per column


# --- NCP committee (2026-10-04) ---

def _ncp_committee(K, vote_lambda, vote_slice, readout_slice, seed=0):
    from src.wirings import NCPWiring
    tf.keras.utils.set_random_seed(seed)
    ncp = NCPWiring(_resolve("gru"), 8, 4, 4, seed=1)
    cell = CommitteeVotingCell(n_columns=K, cell=ncp.make_cell(),
                               vote_slice=vote_slice, readout_slice=readout_slice,
                               vote_lambda=vote_lambda)
    cell.build((tf.TensorShape([None, K, 5]), tf.TensorShape([None, 1])))
    return cell


def _resolve(key):
    from src.models.rnn_model import _CELL_REGISTRY
    return _CELL_REGISTRY[key]


def _two_steps(cell, K, seed=0):
    tf.keras.utils.set_random_seed(seed)
    feats = tf.random.normal([3, 2, K, 5], seed=seed)
    states = cell.get_initial_state(batch_size=3)
    for t in range(2):
        out, states = cell((feats[:, t], tf.ones([3, 1])), states)
    return out.numpy(), [s.numpy() for s in states]


def test_ncp_committee_couples_command_not_motor():
    """ncps state order is [motor | command | inter]. Motor neurons have no
    outgoing synapses: a mixed motor state only feeds the motor neurons' own
    leak term, it never reaches the command or inter neurons, so the committee
    would share no internal representation (the non-motor states are identical
    for lambda 0 and 1). Mixing the command slice changes the later readout."""
    motor, command = slice(0, 4), slice(4, 8)
    _, s0 = _two_steps(_ncp_committee(3, 0.0, motor, motor), 3)
    _, s1 = _two_steps(_ncp_committee(3, 1.0, motor, motor), 3)
    for a, b in zip(s0, s1):
        assert np.allclose(a[:, 4:], b[:, 4:], atol=1e-6)     # command + inter
    out0, _ = _two_steps(_ncp_committee(3, 0.0, command, motor), 3)
    out1, _ = _two_steps(_ncp_committee(3, 1.0, command, motor), 3)
    assert not np.allclose(out0, out1, atol=1e-6)


def test_ncp_committee_model_builds_fits_and_is_k_independent():
    F, C, T = 7, 7, 6
    def P(K):
        return build_committee_model(
            "ncp", K, cell="cfc_lrc", size=16, feature_size=F, seq_len=T,
            num_classes=C, elastance_type="asymmetric", seed=0)
    m = P(3)
    X = np.random.default_rng(0).standard_normal((10, T, 3, F)).astype("float32")
    t = np.ones((10, T, 1), "float32")
    y = np.random.default_rng(1).integers(0, C, (10, T)).astype("int32")
    assert m.predict([X, t], verbose=0).shape == (10, T, C)
    m.fit([X, t], y, epochs=1, batch_size=5, verbose=0)
    assert effective_param_count(P(1)) == effective_param_count(P(2)) \
        == effective_param_count(m)
    assert m.layers[-1].input_shape[-1] == 8     # readout = motor slice (size//2)


def test_dense_vote_units_restricts_vote_and_readout():
    m = build_committee_model("dense", 2, cell="gru", size=12, feature_size=7,
                              seq_len=4, num_classes=7, dense_vote_units=5,
                              seed=0)
    assert m.layers[-1].input_shape[-1] == 5
    with pytest.raises(ValueError):
        build_committee_model("ncp", 2, cell="gru", size=12, dense_vote_units=5)


def test_regression_head_shape():
    m = build_committee_model("dense", 2, cell="cfc_lrc", size=16,
                              feature_size=3, seq_len=5, task="regression",
                              elastance_type="asymmetric", seed=0)
    X = np.zeros((4, 5, 2, 3), "float32")
    assert m.predict([X, np.ones((4, 5, 1), "float32")],
                     verbose=0).shape == (4, 5, 3)


def test_invalid_wiring_and_task_raise():
    with pytest.raises(ValueError):
        build_committee_model("bogus", 2)
    with pytest.raises(ValueError):
        build_committee_model("cncp", 2, task="bogus")


def test_channel_dropout_inference_noop_train_drops_whole_channels():
    cd = ChannelDropout(0.5)
    x = tf.ones([4, 3, 2, 6])
    assert np.allclose(cd(x, training=False).numpy(), 1.0)   # inference no-op
    tr = cd(x, training=True).numpy()
    per_ex = (tr == 0).all(axis=(1, 2))                      # channel off across T,K
    assert per_ex.shape == (4, 6)
    assert per_ex.any()                                      # some channel dropped


def test_add_noise_perturbs_and_zero_sigma_is_noop():
    X = np.ones((3, 4, 5), np.float32)
    assert np.array_equal(add_noise(X, 0.0), X)                # sigma 0 unchanged
    out = add_noise(X, 0.5, seed=1)
    assert out.shape == X.shape
    assert not np.allclose(out, X)                             # actually perturbed
    assert np.array_equal(add_noise(X, 0.5, seed=1),
                          add_noise(X, 0.5, seed=1))            # deterministic


def test_noise_augment_inference_noop_train_perturbs():
    na = NoiseAugment(0.5)
    x = tf.ones([4, 3, 2, 6])
    assert np.allclose(na(x, training=False).numpy(), 1.0)     # inference no-op
    tr = na(x, training=True).numpy()
    assert not np.allclose(tr, 1.0)                            # train adds noise


def test_train_noise_builds_adds_no_params():
    def P(tn):
        return build_committee_model(
            "dense", 1, cell="cfc_lrc", size=32, feature_size=7, seq_len=6,
            num_classes=7, train_noise=tn, elastance_type="asymmetric",
            seed=0).count_params()
    assert P(0.0) == P(0.5)                                    # noise aug: no params


def test_train_drop_builds_adds_no_params_and_infers_deterministically():
    def P(td):
        return build_committee_model(
            "cncp", 1, cell="cfc_lrc", size=24, feature_size=7, seq_len=6,
            num_classes=7, train_drop=td, elastance_type="asymmetric",
            seed=0).count_params()
    assert P(0.0) == P(0.3)                                  # dropout has no params
    m = build_committee_model("cncp", 1, cell="cfc_lrc", size=24, feature_size=7,
                              seq_len=6, num_classes=7, train_drop=0.3,
                              elastance_type="asymmetric", seed=0)
    X = np.zeros((5, 6, 1, 7), "float32")
    t = np.ones((5, 6, 1), "float32")
    p1 = m.predict([X, t], verbose=0)
    p2 = m.predict([X, t], verbose=0)
    assert np.allclose(p1, p2)                               # dropout off at test
