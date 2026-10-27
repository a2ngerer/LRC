"""Tests for the partial-view voting committee (Iteration 7)."""
import json
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from src.tasks.committee.views import (partition_masks, masked_views,
                                       noisy_views, drop_features, add_noise,
                                       make_views, average_views, COORDS,
                                       fault_counts, noise_sigma, corrupt_views,
                                       fault_aware_mean, channel_fault)
from src.tasks.committee.model import (build_committee_model, ChannelDropout,
                                       NoiseAugment, COMMITTEE_WIRINGS)
from src.neurons import CfC_Cell, CfC_LRC_Cell
from src.tasks.person_activity.model import (backbone_override,
                                             effective_backbone_layers)
from src.wirings import effective_param_count, match_param_budget
from experiments import run_committee_benchmark as co_runner
from experiments.run_committee_benchmark import (
    result_filename, position_metrics, steps_to, CorruptedBatches,
    GridValidation)
from src.evaluation.metrics import confusion_matrix, macro_f1
from src.wirings.committee import CommitteeVotingCell
from src.wirings.tbt_cncp import MultiColumnVotingCell


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

def _ncp_committee(K, vote_lambda, vote_slice, readout_slice, seed=0, **vote):
    from src.wirings import NCPWiring
    tf.keras.utils.set_random_seed(seed)
    ncp = NCPWiring(_resolve("gru"), 8, 4, 4, seed=1)
    cell = CommitteeVotingCell(n_columns=K, cell=ncp.make_cell(),
                               vote_slice=vote_slice, readout_slice=readout_slice,
                               vote_lambda=vote_lambda, **vote)
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


# --- folded-batch columns == per-column loop (#32) ---

def _fold_arm(arm, K, vote):
    """A built weight-shared committee cell of the given arm (random weights)."""
    tf.keras.utils.set_random_seed(0)
    if arm == "cncp":
        cell = MultiColumnVotingCell(n_columns=K, cell_cls="cfc_lrc",
                                     use_location="gate",
                                     elastance_type="asymmetric", **vote)
        cell.build((tf.TensorShape([None, K, 5]), tf.TensorShape([None, 1]),
                    tf.TensorShape([None, K, 2])))
        return cell
    if arm == "ncp":
        return _ncp_committee(K, vote_slice=slice(4, 8),
                              readout_slice=slice(0, 4), **vote)
    cell = CommitteeVotingCell(n_columns=K, cell_cls="cfc_lrc", units=8,
                               vote_slice=slice(2, 6), readout_slice=slice(0, 4),
                               elastance_type="asymmetric", **vote)
    cell.build((tf.TensorShape([None, K, 5]), tf.TensorShape([None, 1])))
    return cell


def _rollout(cell, step, inputs, states):
    """Steps `step` over inputs; returns (outputs, final states, gradients of
    a scalar loss on both w.r.t. the cell's trainable variables)."""
    with tf.GradientTape() as tape:
        outs = []
        for x in inputs:
            out, states = step(x, states)
            outs.append(out)
        loss = (tf.add_n([tf.reduce_sum(o ** 2) for o in outs])
                + tf.add_n([tf.reduce_sum(tf.sin(s)) for s in states]))
    return outs, list(states), tape.gradient(loss, cell.trainable_variables)


@pytest.mark.parametrize("vote_lambda", [0.0, 0.3, None])
@pytest.mark.parametrize("consensus", ["mean", "median"])
@pytest.mark.parametrize("K", [1, 2, 4])
@pytest.mark.parametrize("arm", ["dense", "ncp", "cncp"])
def test_folded_columns_match_column_loop(arm, K, consensus, vote_lambda):
    """One shared-cell call on the (B*K)-folded batch computes the per-column
    loop: same outputs, states and gradients over three steps from random
    states, with per-row elapsed times and per-column features/locations."""
    B, T = 3, 3
    cell = _fold_arm(arm, K, dict(vote_lambda=vote_lambda, vote_init=0.4,
                                  consensus=consensus))
    tf.keras.utils.set_random_seed(1)
    inputs = [(tf.random.normal([B, K, 5]),
               tf.random.uniform([B, 1], 0.5, 1.5))
              + ((tf.random.normal([B, K, 2]),) if arm == "cncp" else ())
              for _ in range(T)]
    s0 = [tf.random.normal(s.shape)
          for s in cell.get_initial_state(batch_size=B)]
    ref_out, ref_st, ref_g = _rollout(cell, cell._call_loop, inputs, s0)
    out, st, g = _rollout(cell, cell.call, inputs, s0)
    assert len(st) == len(ref_st) and len(g) == len(ref_g)
    for a, b in zip(ref_out + ref_st, out + st):
        np.testing.assert_allclose(b.numpy(), a.numpy(), atol=1e-5)
    # Weight gradients sum over the batch rows in a different order (one
    # (B*K)-row reduction vs K B-row ones): float32 rounding on the scale of
    # the summands, so the tolerance is relative to the tensor's largest entry.
    for a, b in zip(ref_g, g):
        assert (a is None) == (b is None)
        if a is not None:
            a = a.numpy()
            np.testing.assert_allclose(
                b.numpy(), a, atol=1e-5 * max(1.0, np.abs(a).max()))


@pytest.mark.parametrize("wiring", ["cncp", "ncp", "dense"])
def test_column_loop_flag_reaches_cell_and_matches_folded_model(wiring):
    kw = dict(cell="cfc_lrc", size=24 if wiring == "cncp" else 32,
              feature_size=7, seq_len=6, num_classes=7,
              elastance_type="asymmetric", seed=0)
    fold = build_committee_model(wiring, 3, **kw)
    loop = build_committee_model(wiring, 3, column_loop=True, **kw)
    assert loop.count_params() == fold.count_params()
    loop.set_weights(fold.get_weights())
    cells = [l.cell for m in (fold, loop) for l in m.layers
             if isinstance(l, tf.keras.layers.RNN)]
    assert [c._column_loop for c in cells] == [False, True]
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 6, 3, 7)).astype("float32")
    t = rng.uniform(0.5, 1.5, (10, 6, 1)).astype("float32")
    np.testing.assert_allclose(fold.predict([X, t], verbose=0),
                               loop.predict([X, t], verbose=0), atol=1e-5)


# --- --backbone-layers override (V2s control) ---

@pytest.mark.parametrize("backbone_layers", [None, 0, 1])
@pytest.mark.parametrize("wiring", COMMITTEE_WIRINGS)
def test_backbone_override_reaches_every_committee_arm(wiring, backbone_layers):
    kw = backbone_override("cfc_lrc", backbone_layers)
    m = build_committee_model(wiring, 2, cell="cfc_lrc", size=16, feature_size=7,
                              seq_len=6, num_classes=7, seed=0, **kw)
    depths = {len(c._backbone) for c in m._flatten_layers()
              if isinstance(c, (CfC_Cell, CfC_LRC_Cell))}
    assert depths == {effective_backbone_layers(wiring, "cfc_lrc", kw)}


def test_result_filename_backbone_suffix():
    a = SimpleNamespace(cell="cfc_lrc", wiring="cncp", n_columns=4, size=48,
                        view_mode="identical", vote_lambda=None,
                        consensus="mean", dense_vote_units=None,
                        view_average="off", train_drop=0.0, train_noise=0.0,
                        test_corruption="dropout", train_corruption="on",
                        lr=1e-3, clipnorm=None, corruption_seed=0, seed=0,
                        backbone_layers=None)
    tail = "_lr0.001_clipoff_cs0_seed0"
    base = "cfc_lrc_cncp_K4_s48_identical"
    assert result_filename(a) == base + tail + ".json"
    a.backbone_layers = 1
    assert result_filename(a) == base + tail + "_bb1.json"
    a.view_average, a.clipnorm, a.corruption_seed = "fault_mean", 1.0, 3
    assert result_filename(a) == \
        base + "_viewfm_lr0.001_clip1_cs3_seed0_bb1.json"
    a.train_corruption = "off"
    assert "_viewfm_trainclean_lr" in result_filename(a)


# --- H-d endpoints: median-input monolith, per-position curve (#33) ---

@pytest.mark.parametrize("K, n_fail", [(4, 1), (5, 2), (30, 3), (30, 14)])
def test_median_view_average_ignores_minority_column_failure(K, n_fail):
    """n_fail of K identical views zeroed (column failure, rho = n_fail / K
    < 0.5): the lower median returns the intact input, the mean shrinks it."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 6, 7)).astype(np.float32)
    v = noisy_views(X, K, noise=0.0)
    v[:, :, rng.choice(K, n_fail, replace=False)] = 0.0
    med = average_views(v, "median")
    assert med.shape == (5, 6, 1, 7)
    np.testing.assert_array_equal(med[:, :, 0], X)
    np.testing.assert_allclose(average_views(v, "mean")[:, :, 0],
                               (1 - n_fail / K) * X, atol=1e-6)
    if K % 2 == 0:              # rho = 0.5: the LOWER median breaks down
        v = noisy_views(X, K, noise=0.0)
        v[:, :, :K // 2] = 0.0
        np.testing.assert_array_equal(average_views(v, "median")[:, :, 0],
                                      np.minimum(X, 0.0))
    with pytest.raises(ValueError):
        average_views(v, "max")


def test_position_metrics_curves_and_window_scores():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 7, (40, 32))
    pred = np.where(rng.random((40, 32)) < 0.6, y, rng.integers(0, 7, (40, 32)))
    acc, f1, f1_win = position_metrics(pred, y, 7)
    assert acc.shape == f1.shape == (32,)
    assert np.isclose(acc.mean(), (pred == y).mean())
    assert np.isclose(f1[3], macro_f1(confusion_matrix(y[:, 3], pred[:, 3], 7)))
    assert np.isclose(f1_win, macro_f1(confusion_matrix(y, pred, 7)))
    assert steps_to([0.1, 0.5, 0.95, 0.9, 1.0]) == 3     # first reach of 0.9
    assert steps_to([0.0, 0.0]) == 1


# --- consensus-study corruption protocol (#55) ---

def _windows(n, T=6, seed=0):
    """(n, T, 7) windows: a one-hot sensor id and positive coordinates."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, T, 7), np.float32)
    X[np.arange(n), :, rng.integers(0, 4, n)] = 1.0
    X[..., COORDS] = rng.uniform(0.5, 2.0, (n, T, 3))
    return X


def test_fault_counts_per_K():
    assert fault_counts(3) == [0, 1, 2]
    assert fault_counts(30) == [0, 3, 9, 15, 20]
    assert fault_counts(1) == [0.0, 0.1, 0.3, 0.5, 0.67]   # probabilities


def test_noise_sigma_per_coordinate_feature():
    X = _windows(50)
    sigma = noise_sigma(X)
    assert sigma.shape == (7,) and np.all(sigma[:4] == 0)
    np.testing.assert_allclose(
        sigma[COORDS], 0.3 * X[..., COORDS].reshape(-1, 3).std(axis=0),
        rtol=1e-6)


def test_corrupt_views_exact_columns_coords_only():
    X = _windows(200)
    sigma = noise_sigma(X)
    v = corrupt_views(X, 30, 15, sigma, np.random.default_rng(0))
    assert v.shape == (200, 6, 30, 7) and v.dtype == np.float32
    failed = (v[..., COORDS] == 0).all(axis=(1, 3))          # (N, K)
    assert np.all(failed.sum(axis=1) == 15)                  # exactly m
    # the one-hot is untouched in every column
    np.testing.assert_array_equal(v[..., :4],
                                  np.repeat(X[:, :, None, :4], 30, axis=2))
    # noise on the coordinates of the intact columns, sigma per feature
    resid = (v - X[:, :, None, :])[..., COORDS][~failed[:, None, :, None]
                                                .repeat(6, 1).repeat(3, 3)]
    resid = resid.reshape(-1, 3)
    np.testing.assert_allclose(resid.std(axis=0), sigma[COORDS], rtol=0.03)
    # m = 0 without noise: K identical copies of the window
    np.testing.assert_array_equal(
        corrupt_views(X, 3, 0, np.zeros(7), np.random.default_rng(0)),
        np.repeat(X[:, :, None], 3, axis=2))
    with pytest.raises(ValueError):
        corrupt_views(X, 3, 4, sigma, np.random.default_rng(0))


def test_corrupt_views_k1_bernoulli():
    v = corrupt_views(_windows(2000), 1, 0.3, np.zeros(7),
                      np.random.default_rng(0))
    rate = (v[..., COORDS] == 0).all(axis=(1, 2, 3)).mean()
    assert abs(rate - 0.3) <= 0.03


def test_corrupt_views_seeded_and_shared():
    X, sigma = _windows(20), noise_sigma(_windows(20))

    def draw(seed):
        return corrupt_views(X, 30, 9, sigma, np.random.default_rng([seed, 9]))
    np.testing.assert_array_equal(draw(0), draw(0))
    assert not np.array_equal(draw(0), draw(1))


def test_fault_aware_mean_recovers_intact_input():
    X = _windows(40)
    sigma = noise_sigma(X)
    v = corrupt_views(X, 30, 15, sigma, np.random.default_rng(0))
    fm = fault_aware_mean(v)
    assert fm.shape == (40, 6, 1, 7)
    alive = ~(v[..., COORDS] == 0).all(axis=(1, 3))          # (N, K)
    expect = np.stack([v[n][:, alive[n]].mean(axis=1) for n in range(40)])
    np.testing.assert_allclose(fm[:, :, 0], expect, atol=1e-5)
    np.testing.assert_array_equal(fm[:, :, 0, :4], X[..., :4])
    # without noise: the clean coordinates, no (1 - m/K) shrink
    v0 = corrupt_views(X, 30, 15, np.zeros(7), np.random.default_rng(0))
    np.testing.assert_allclose(fault_aware_mean(v0)[:, :, 0], X, atol=1e-6)
    np.testing.assert_allclose(average_views(v0, "mean")[:, :, 0, COORDS],
                               0.5 * X[..., COORDS], atol=1e-6)
    # every column failed: 0 on the coordinates, the one-hot unchanged
    vk = corrupt_views(X, 30, 30, sigma, np.random.default_rng(0))
    fk = average_views(vk, "fault_mean")
    assert np.all(fk[..., COORDS] == 0)
    np.testing.assert_array_equal(fk[:, :, 0, :4], X[..., :4])


def test_lower_median_breaks_at_half():
    """K=30, m=15 on positive coordinates: the lower median of the views is
    the fault value 0 (the median arms are read on m < K/2)."""
    X = _windows(30)
    v = corrupt_views(X, 30, 15, np.zeros(7), np.random.default_rng(0))
    assert np.all(average_views(v, "median")[..., COORDS] == 0)
    v = corrupt_views(X, 30, 14, np.zeros(7), np.random.default_rng(0))
    np.testing.assert_array_equal(average_views(v, "median")[:, :, 0], X)


def test_channel_fault_zeros_one_coordinate_in_every_column():
    v = corrupt_views(_windows(10), 4, 0, np.zeros(7),
                      np.random.default_rng(0))
    out = channel_fault(v, 5)
    assert np.all(out[..., 5] == 0) and np.all(v[..., 5] > 0)   # a copy
    np.testing.assert_array_equal(np.delete(out, 5, -1), np.delete(v, 5, -1))


def _batches(view_average, seed=0, n=50):
    X = _windows(n)
    y = np.tile(np.arange(n)[:, None], (1, 6)).astype(np.int32)  # window id
    return CorruptedBatches(X, np.ones((n, 6, 1), np.float32), y, 30,
                            fault_counts(30), noise_sigma(X), seed, 16,
                            view_average)


def test_sequence_draws_identical_across_arms():
    """Same --corruption-seed, different arms: the same m sequence, column
    choices and batch order; the monolith arm averages exactly those views."""
    a, b, c = _batches("off"), _batches("mean"), _batches("off", seed=1)
    assert len(a) == 4
    for epoch in range(2):
        ms = []
        for i in range(len(a)):
            ia, ma, va = a.corrupted(i)
            ib, mb, vb = b.corrupted(i)
            np.testing.assert_array_equal(ia, ib)
            np.testing.assert_array_equal(va, vb)
            assert ma == mb
            assert np.all((va[..., COORDS] == 0).all(axis=(1, 3)).sum(1) == ma)
            (xa, _), ya = a[i]
            (xb, _), yb = b[i]
            np.testing.assert_array_equal(ya[:, 0], ia)    # the permutation
            np.testing.assert_array_equal(xa, va)
            np.testing.assert_allclose(xb, average_views(va, "mean"))
            ms.append(ma)
        assert np.array_equal(np.sort(a.perm), np.arange(50))
        a.on_epoch_end(), b.on_epoch_end(), c.on_epoch_end()
    assert not np.array_equal(a.perm, c.perm)
    assert a.epoch == 2


def test_grid_validation_selects_best_epoch_and_restores_weights():
    class Scripted(GridValidation):
        means = iter([0.2, 0.5, 0.3])

        def score(self):
            m = next(self.means)
            return [m - 0.1, m + 0.1]

    x = np.random.default_rng(0).standard_normal((32, 3)).astype(np.float32)
    model = tf.keras.Sequential([tf.keras.Input((3,)), tf.keras.layers.Dense(2)])
    model.compile("sgd", "mse")
    seen = []
    gv = Scripted([], 2, 32)
    model.fit(x, np.ones((32, 2)), epochs=3, verbose=0, callbacks=[
        gv, tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda e, logs: seen.append(
                (model.get_weights(), dict(logs))))])
    assert gv.best_epoch == 2 and gv.best == pytest.approx(0.5)
    np.testing.assert_allclose(gv.curve, [[0.1, 0.3], [0.4, 0.6], [0.2, 0.4]])
    for got, want in zip(model.get_weights(), seen[1][0]):
        np.testing.assert_array_equal(got, want)
    assert not np.allclose(model.get_weights()[0], seen[2][0][0])
    assert seen[1][1]["val_macro_f1_mean"] == pytest.approx(0.5)
    assert seen[1][1]["val_macro_f1_grid"] == pytest.approx([0.4, 0.6])


def test_steps_to_90_plateau_reference():
    """Referenced to the plateau (mean of positions 25-32), not to the last
    position alone."""
    curve = [0.1] * 24 + [0.6] * 7 + [1.0]
    assert steps_to(curve) == 25          # 0.9 * 0.65; curve[-1] gave 32
    assert steps_to([0.2] * 32) == 1


def test_build_uses_rmsprop_with_global_clipnorm():
    m = build_committee_model("dense", 2, size=4, feature_size=7, seq_len=6,
                              clipnorm=1.0, lr=3e-3)
    assert isinstance(m.optimizer, tf.keras.optimizers.RMSprop)
    assert m.optimizer.global_clipnorm == 1.0
    assert m.optimizer.learning_rate.numpy() == pytest.approx(3e-3)


def _stub(T=32):
    rng = np.random.default_rng(0)
    parts = {}
    for part, n in (("train", 20), ("val", 12), ("test", 12)):
        parts[f"{part}_x"] = _windows(n, T, seed=n)
        parts[f"{part}_t"] = np.ones((n, T, 1), np.float32)
        parts[f"{part}_y"] = rng.integers(0, 7, (n, T)).astype(np.int32)
        parts[f"{part}_rec"] = np.array(["A01", "B01"] * (n // 2))
    return SimpleNamespace(**parts, feature_size=7, num_classes=7, seq_len=T,
                           split="grouped", split_seed=0,
                           recordings={"train": ["A01"], "val": ["A01"],
                                       "test": ["A01", "B01"]})


def test_runner_writes_grid_endpoints(tmp_path, monkeypatch):
    """Two-epoch fit on a stub data set: the fault-grid endpoints, the
    selection fields and lambda; the training batches follow (epoch, batch)."""
    calls = []
    real = CorruptedBatches.__getitem__

    def spy(self, i):
        calls.append((self.epoch, i))
        return real(self, i)
    monkeypatch.setattr(CorruptedBatches, "__getitem__", spy)
    monkeypatch.setattr(co_runner, "load_person_activity",
                        lambda **kw: _stub())
    monkeypatch.setattr(sys, "argv", [
        "run_committee_benchmark.py", "--wiring", "dense", "--n-columns", "3",
        "--size", "4", "--epochs", "2", "--batch-size", "8",
        "--test-seeds", "0,1", "--outdir", str(tmp_path)])
    co_runner.main()
    # dataset() peeks batch 0 once, then 3 batches per epoch in order
    assert calls == [(0, 0), (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    (path,) = tmp_path.glob("*.json")
    assert path.name == \
        "cfc_lrc_dense_K3_s4_identical_lr0.001_clipoff_cs0_seed0.json"
    assert (tmp_path / (path.name[:-len(".json")] + ".weights.h5")).is_file()
    r = json.loads(path.read_text())
    assert r["fault_counts"] == [0, 1, 2] and r["view_mode"] == "identical"
    assert (r["batch_size"], r["lr"], r["split"]) == (8, 1e-3, "grouped")
    assert r["test_seeds"] == [0, 1] and r["val_seed"] == 100
    assert r["noise_sigma"][:4] == [0, 0, 0, 0] and min(r["noise_sigma"][4:]) > 0
    assert np.array(r["val_macro_f1_grid_curve"]).shape == (2, 3)
    assert r["best_val_epoch"] in (1, 2)
    assert r["val_macro_f1_mean"] == pytest.approx(
        np.mean(r["val_macro_f1_grid_curve"][r["best_val_epoch"] - 1]))
    assert len(r["test_macro_f1_by_fault"]) == 3
    assert r["test_macro_f1_mean_grid"] == pytest.approx(
        np.mean(r["test_macro_f1_by_fault"]))
    assert [sorted(d) for d in r["test_macro_f1_per_recording_by_fault"]] \
        == [["A01", "B01"]] * 3
    cms = np.array([[c for c in d.values()]
                    for d in r["test_confusion_by_recording_by_fault"]])
    assert cms.shape == (3, 2, 7, 7) and np.all(cms.sum(axis=(1, 2, 3))
                                                == 2 * 12 * 32)
    assert np.array(r["per_position_macro_f1"]).shape == (3, 32)
    np.testing.assert_allclose(r["per_position_macro_f1_auc"],
                               np.mean(r["per_position_macro_f1"], axis=1),
                               atol=1e-4)
    assert all(1 <= s <= 32 for s in r["steps_to_90"])
    assert sorted(r["channel_fault_macro_f1"]) == ["mean", "x", "y", "z"]
    assert 0.0 < r["vote_lambda_final"] < 1.0
    assert len(r["vote_lambda_curve"]) == 2
    assert r["vote_lambda_curve"][-1] == pytest.approx(r["vote_lambda_final"],
                                                       abs=0.05)
    assert r["wallclock_s"] >= 0 and "drop_accs" not in r


def test_runner_monolith_arm_shares_the_draws(tmp_path, monkeypatch):
    """The fault-aware-mean monolith: one model column, same file stem
    fields, the exploratory sweep only when its levels are given."""
    monkeypatch.setattr(co_runner, "load_person_activity",
                        lambda **kw: _stub())
    monkeypatch.setattr(sys, "argv", [
        "run_committee_benchmark.py", "--wiring", "dense", "--n-columns", "3",
        "--size", "4", "--epochs", "1", "--view-average", "fault_mean",
        "--drop-fracs", "0.0,0.3", "--drop-seeds", "0",
        "--no-save-weights", "--outdir", str(tmp_path)])
    co_runner.main()
    (path,) = tmp_path.glob("*.json")
    assert "_viewfm_" in path.name
    r = json.loads(path.read_text())
    assert r["model_columns"] == 1 and r["weights_path"] is None
    assert len(r["drop_accs"]) == 2 and len(r["test_macro_f1_by_fault"]) == 3


# --- eval-only rescoring (#81) ---

@pytest.fixture(scope="module")
def stub_fit(tmp_path_factory):
    """One two-epoch dense K=3 fit with its weights on the stub data, shared
    by the eval-only tests: the result JSON path."""
    out = tmp_path_factory.mktemp("fit")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(co_runner, "load_person_activity", lambda **kw: _stub())
        mp.setattr(sys, "argv", [
            "run_committee_benchmark.py", "--wiring", "dense",
            "--n-columns", "3", "--size", "4", "--epochs", "2",
            "--batch-size", "8", "--test-seeds", "0,1", "--outdir", str(out)])
        co_runner.main()
    (path,) = out.glob("*.json")
    return path


def _eval_only(monkeypatch, *args):
    monkeypatch.setattr(co_runner, "load_person_activity",
                        lambda **kw: _stub())
    monkeypatch.setattr(sys, "argv", ["run_committee_benchmark.py",
                                      "--eval-only", *args])
    return co_runner.main()


def _weights(path):
    return str(path)[:-len(".json")] + ".weights.h5"


def test_eval_only_rescores_the_stored_fit_without_training(
        stub_fit, tmp_path, monkeypatch):
    """No fit: per-seed test confusions that sum to the summed key, which
    reproduces the source record (same machine); validation draw 0 is the
    stored draw _rng(val_seed, m), draw k uses val_seed + 1000 k."""
    rec = json.loads(stub_fit.read_text())
    keys, real = [], co_runner._rng
    monkeypatch.setattr(co_runner, "_rng",
                        lambda *k: keys.append(k) or real(*k))
    monkeypatch.setattr(tf.keras.Model, "fit",
                        lambda *a, **kw: pytest.fail("eval-only trained"))
    _eval_only(monkeypatch, "--weights", _weights(stub_fit), "--val-draws",
               "2", "--out", str(tmp_path))
    r = json.loads((tmp_path / (stub_fit.stem + ".evalonly.json"))
                   .read_text())
    assert r["source_record"] == str(stub_fit)
    assert r["source_config_hash"] == rec["config"]["effective"]["config_hash"]
    assert (r["fault_counts"], r["eval_test_seeds"], r["eval_val_draws"],
            r["eval_val_draw_seeds"]) == ([0, 1, 2], [0, 1], 2, [100, 1100])
    assert sorted(set(keys)) == sorted(
        {(s, m) for s in (0, 1, 100, 1100) for m in (0, 1, 2)})
    by_seed = r["test_confusion_by_recording_by_fault_by_seed"]
    assert sorted(by_seed) == ["0", "1"]
    summed = [{rec_id: np.sum([by_seed[s][j][rec_id] for s in by_seed],
                              axis=0).tolist() for rec_id in ("A01", "B01")}
              for j in range(3)]
    assert summed == r["test_confusion_by_recording_by_fault"] \
        == rec["test_confusion_by_recording_by_fault"]
    assert r["test_macro_f1_by_fault"] == pytest.approx(
        rec["test_macro_f1_by_fault"], abs=1e-12)
    assert r["test_macro_f1_per_recording_by_fault"] == [
        pytest.approx(d, abs=1e-12)
        for d in rec["test_macro_f1_per_recording_by_fault"]]
    cms = np.array(r["val_confusion_by_fault_by_draw"])
    assert cms.shape == (2, 3, 7, 7) and np.all(cms.sum(axis=(2, 3))
                                                == 12 * 32)
    assert r["val_macro_f1_by_fault_by_draw"][0] == pytest.approx(
        rec["val_macro_f1_grid_curve"][rec["best_val_epoch"] - 1], abs=1e-12)
    assert r["val_macro_f1_by_fault_by_draw"][1] == pytest.approx(
        [macro_f1(c) for c in cms[1]])


def test_eval_only_refuses_a_foreign_config_or_weights(stub_fit, tmp_path,
                                                       monkeypatch):
    """Refused (exit 1, nothing written): a config.effective that does not
    hash to its config_hash, a weights file under another name, one with
    the right name that is not the file next to the record, and a missing
    one."""
    rec = json.loads(stub_fit.read_text())
    name, fit = rec["weights_path"], tmp_path / "fit"
    fit.mkdir()
    rec["config"]["effective"]["config_hash"] = "0" * 64
    (fit / "tampered.json").write_text(json.dumps(rec))
    shutil.copy(_weights(stub_fit), fit / name)
    shutil.copy(_weights(stub_fit), fit / "other.weights.h5")
    cases = [((fit / name, fit / "tampered.json"), "config_hash is '0000"),
             ((fit / "other.weights.h5", stub_fit), "weights_path"),
             ((fit / name, stub_fit), "weights_path"),
             ((fit / "gone" / name, stub_fit), "(or is missing)")]
    for (weights, record), why in cases:
        with pytest.raises(SystemExit) as e:
            _eval_only(monkeypatch, "--weights", str(weights), "--record",
                       str(record), "--out", str(tmp_path / "out"))
        assert "eval-only refused" in str(e.value.code)
        assert why in str(e.value.code)
    assert not (tmp_path / "out").exists()


def test_eval_only_never_writes_into_the_fit_folder(stub_fit, monkeypatch):
    """A fit folder holds exactly one result JSON (run_grid_hpo.py unpacks
    it so): --out is required, and the record's folder or a folder below it
    is refused before anything is read or written."""
    before = sorted(p.name for p in stub_fit.parent.rglob("*"))
    with pytest.raises(SystemExit) as e:
        _eval_only(monkeypatch, "--weights", _weights(stub_fit))
    assert e.value.code == 2
    for out in (stub_fit.parent, stub_fit.parent / "sub",
                stub_fit.parent / "sub" / ".."):
        with pytest.raises(SystemExit) as e:
            _eval_only(monkeypatch, "--weights", _weights(stub_fit), "--out",
                       str(out))
        assert "eval-only refused: --out" in str(e.value.code)
        assert "pass a folder outside the fit folder" in str(e.value.code)
    assert sorted(p.name for p in stub_fit.parent.rglob("*")) == before
