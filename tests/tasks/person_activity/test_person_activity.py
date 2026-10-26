# tests/tasks/person_activity/test_person_activity.py
# Light validation of the person-activity benchmark pieces:
#   - build_person_activity_model builds and forward-passes for every
#     wiring x {cfc_lrc, gru} with the (features, time) two-input contract;
#   - gradients flow through every trainable variable;
#   - the cncp/ncp parameter counts stay comparable (fairness invariant);
#   - multi-state cells are rejected;
#   - the dataset loader returns the documented shapes (skipped when the
#     data file has not been downloaded);
#   - on a stub raw file (issue #26): all recordings parsed, negative dt
#     clamped, the grouped split is disjoint by recording, person-balanced and
#     shares no time step, no window spans two recordings; macro-F1 matches
#     sklearn; a two-epoch runner run writes the val/test metric fields.
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from src.wirings import size_for_budget
from src.tasks.person_activity import (build_person_activity_model,
                                       load_person_activity,
                                       DEFAULT_DATA_PATH)
from src.evaluation.metrics import confusion_matrix, macro_f1
from src.tasks.person_activity.datasets import CLASS_MAP, SENSOR_IDS
from src.tasks.person_activity.model import WIRINGS
from experiments import run_person_activity_benchmark as pa_runner
from experiments.run_person_activity_benchmark import result_tag

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

@pytest.mark.parametrize("wiring", WIRINGS)
@pytest.mark.parametrize("cell", ["cfc_lrc", "gru"])
def test_build_and_forward_shapes(wiring, cell):
    model = build_person_activity_model(wiring, cell, size=16, seed=0)
    x, t, _ = _dummy_batch()
    out = model([x, t])
    assert tuple(out.shape) == (2, SEQ_LEN, NUM_CLASSES)
    assert bool(tf.reduce_all(tf.math.is_finite(out)))


def test_dense3_stacks_three_layers_each_fed_time():
    """The depth control: three recurrent layers, and every layer receives the
    (sequence, time) tuple so dt reaches all of them, not only the first."""
    model = build_person_activity_model("dense3", "cfc_lrc", size=8, seed=0)
    rnns = [l for l in model.layers if isinstance(l, tf.keras.layers.RNN)]
    assert len(rnns) == 3
    time_in = model.inputs[1]
    for layer in rnns:
        inp = layer.input
        assert isinstance(inp, (list, tuple)) and len(inp) == 2
        assert inp[1] is time_in or inp[1].ref() == time_in.ref()
    x, t, _ = _dummy_batch()
    # a continuous-time cell in the upper layers reacts to dt as well
    out_a = model([x, t]).numpy()
    out_b = model([x, 3.0 * t]).numpy()
    assert not np.allclose(out_a, out_b)


# --- gradient flow ---

@pytest.mark.parametrize("wiring", WIRINGS)
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
    def build(wiring):
        return lambda size: build_person_activity_model(
            wiring, cell, size=size, seed=0)
    counts = {w: size_for_budget(build(w), 4000, range(2, 129))[1]
              for w in WIRINGS}
    lo, hi = min(counts.values()), max(counts.values())
    assert hi / lo < 1.5, f"budget-matched effective counts {counts}"


def test_size_for_budget_picks_the_closest_size():
    """A model with 4*size parameters: budget 41 -> size 10 (40), not 11 (44)."""
    def build(size):
        return tf.keras.Sequential([tf.keras.layers.Dense(size, input_shape=(3,))])
    assert size_for_budget(build, 41, range(1, 50)) == (10, 40)
    assert size_for_budget(build, 43, range(1, 50)) == (11, 44)


# --- input validation ---

@pytest.mark.parametrize("cell", ["lstm", "mm_lrc"])
def test_multi_state_cells_rejected_for_cncp(cell):
    with pytest.raises(ValueError, match="single-state"):
        build_person_activity_model("cncp", cell, size=16, seed=0)


@pytest.mark.parametrize("wiring", ["dense", "dense3", "ncp"])
def test_multi_state_cells_run_on_the_non_composite_wirings(wiring):
    """lstm / mixed memory are planned on dense and NCP (thesis matrix); only
    the composite cNCP state cannot host them."""
    model = build_person_activity_model(wiring, "lstm", size=8, seed=0)
    x, t, _ = _dummy_batch()
    assert tuple(model([x, t]).shape) == (2, SEQ_LEN, NUM_CLASSES)


def test_mixed_memory_cell_runs_on_dense():
    model = build_person_activity_model("dense", "mm_lrc", size=8, seed=0,
                                        elastance_type="asymmetric")
    x, t, _ = _dummy_batch()
    assert tuple(model([x, t]).shape) == (2, SEQ_LEN, NUM_CLASSES)


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
    for part in ("train", "val", "test"):
        assert getattr(data, f"{part}_x").shape[1:] == (SEQ_LEN, FEATURES)
        assert getattr(data, f"{part}_t").shape[1:] == (SEQ_LEN, 1)
        assert getattr(data, f"{part}_y").shape[1:] == (SEQ_LEN,)
        y = getattr(data, f"{part}_y")
        assert y.min() >= 0 and y.max() < NUM_CLASSES
        assert (getattr(data, f"{part}_t") >= 0).all()   # negative dt clamped
    # 25 recordings (5 persons x 5), split 15/5/5 by recording.
    assert data.n_recordings == 25
    assert [len(data.recordings[p]) for p in ("train", "val", "test")] == \
        [15, 5, 5]
    assert data.test_x.shape[0] > 1000
    assert data.train_x.shape[0] > 2 * data.test_x.shape[0]


# --- stub raw file (no download needed) ---

STUB_ROWS = 80          # -> 3 windows of 32 per recording (stride 16)


def _write_stub(path, persons="ABCDE", recordings=5):
    """Raw-format stub. Feature x = global row index (identifies a time step),
    y = recording index; one out-of-order timestamp in recording A01."""
    labels, tags = list(CLASS_MAP), list(SENSOR_IDS)
    lines, row, ticks = [], 0, 633790226051280329
    for p in persons:
        for r in range(1, recordings + 1):
            rec_idx = len(lines) // STUB_ROWS
            for i in range(STUB_ROWS):
                ticks += 1_000_000 if row != 10 else -2_000_000   # 100 ms
                lines.append(f"{p}{r:02d},{tags[i % 4]},{ticks},date,"
                             f"{row},{rec_idx},0.0,{labels[i % len(labels)]}\n")
                row += 1
    path.write_text("".join(lines))
    return path


@pytest.fixture
def stub(tmp_path):
    return str(_write_stub(tmp_path / "ConfLongDemo_JSI.txt"))


def test_stub_keeps_all_recordings_and_clamps_negative_dt(stub):
    data = load_person_activity(data_path=stub)
    assert data.n_recordings == 25
    recs = set(data.train_rec) | set(data.val_rec) | set(data.test_rec)
    assert "E05" in recs and len(recs) == 25          # last one not dropped
    assert data.n_negative_dt == 1
    assert all((getattr(data, f"{p}_t") >= 0).all()
               for p in ("train", "val", "test"))


def test_grouped_split_disjoint_and_person_balanced(stub):
    data = load_person_activity(data_path=stub, split="grouped", split_seed=0)
    parts = data.recordings
    sets = [set(parts[p]) for p in ("train", "val", "test")]
    assert sum(map(len, sets)) == len(set.union(*sets)) == 25
    for person in "ABCDE":
        assert [sum(r[0] == person for r in parts[p])
                for p in ("train", "val", "test")] == [3, 1, 1]
    for p in ("train", "val", "test"):
        assert set(getattr(data, f"{p}_rec")) == set(parts[p])
    assert load_person_activity(data_path=stub, split_seed=0).recordings == parts
    assert load_person_activity(data_path=stub, split_seed=1).recordings != parts


def test_no_window_spans_two_recordings(stub):
    data = load_person_activity(data_path=stub)
    for p in ("train", "val", "test"):
        rec_feature = getattr(data, f"{p}_x")[:, :, 5]
        assert (rec_feature == rec_feature[:, :1]).all()


def _shared_steps(a, b):
    return len(set(a[:, :, 4].ravel()) & set(b[:, :, 4].ravel()))


def test_grouped_split_shares_no_time_steps(stub):
    g = load_person_activity(data_path=stub, split="grouped")
    assert _shared_steps(g.train_x, g.test_x) == 0
    assert _shared_steps(g.train_x, g.val_x) == 0
    assert _shared_steps(g.val_x, g.test_x) == 0
    # the legacy window split leaks, and has no separate validation set
    r = load_person_activity(data_path=stub, split="random_window")
    assert _shared_steps(r.train_x, r.test_x) > 0
    assert np.array_equal(r.val_x, r.test_x) and r.recordings is None


def test_unknown_split_rejected(stub):
    with pytest.raises(ValueError, match="split"):
        load_person_activity(data_path=stub, split="by_person")


def test_macro_f1_matches_sklearn():
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, NUM_CLASSES, size=(50, SEQ_LEN))
    y_pred = np.where(rng.random(y_true.shape) < 0.6, y_true,
                      rng.integers(0, NUM_CLASSES, size=y_true.shape))
    # class 6 absent from labels and predictions -> left out of the mean
    y_true[y_true == 6], y_pred[y_pred == 6] = 0, 1
    cm = confusion_matrix(y_true, y_pred, NUM_CLASSES)
    assert cm.sum() == y_true.size
    assert macro_f1(cm) == pytest.approx(
        f1_score(y_true.ravel(), y_pred.ravel(), average="macro"))


def test_runner_writes_val_and_test_metrics(stub, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "run_person_activity_benchmark.py", "--wiring", "dense",
        "--cell", "gru", "--size", "4", "--epochs", "2", "--batch-size", "32",
        "--data-path", stub, "--outdir", str(tmp_path)])
    pa_runner.main()
    record = json.loads((tmp_path / "gru_dense_seed0.json").read_text())
    assert record["split"] == "grouped" and record["split_seed"] == 0
    assert record["n_recordings"] == 25 and record["n_negative_dt"] == 1
    assert "best_test_accuracy" not in record
    assert record["dt_feature"] is True          # gru, default --dt-feature on
    curve = record["val_macro_f1_curve"]
    assert len(curve) == 2
    assert record["best_val_epoch"] == int(np.argmax(curve)) + 1
    assert record["val_macro_f1"] == max(curve)
    assert 0.0 <= record["test_macro_f1"] <= 1.0
    per_rec = record["test_macro_f1_per_recording"]
    assert sorted(per_rec) == sorted(record["recordings"]["test"])
    assert all(0.0 <= v <= 1.0 for v in per_rec.values())
    cms = record["test_confusion_by_recording"]
    assert sorted(cms) == sorted(record["recordings"]["test"])
    total = np.sum([np.asarray(c) for c in cms.values()], axis=0)
    assert total.sum() == record["test_sequences"] * SEQ_LEN
    assert record["test_accuracy"] == pytest.approx(
        np.trace(total) / total.sum())
    assert record["test_macro_f1"] == pytest.approx(macro_f1(total))


def test_result_tag_backbone_suffix():
    a = SimpleNamespace(cell="cfc_lrc", wiring="ncp", seed=2,
                        backbone_layers=None)
    assert result_tag(a) == "cfc_lrc_ncp_seed2"
    a.backbone_layers = 1
    assert result_tag(a) == "cfc_lrc_ncp_seed2_bb1"


# --- --no-self-recurrence (issue #23 stability pilot) ---

def test_runner_passes_no_self_recurrence_to_the_ncp(stub, tmp_path,
                                                     monkeypatch):
    """Flag -> builder -> NCP sub-cells (fewer effective params at the same
    size) -> JSON, on the stub raw file so the dataset file is not needed."""
    def run(*flags):
        monkeypatch.setattr(sys, "argv", [
            "run_person_activity_benchmark.py", "--wiring", "ncp", "--cell",
            "cfc_lrc", "--epochs", "1", "--size", "8", "--data-path", stub,
            "--outdir", str(tmp_path), *flags])
        pa_runner.main()
        return json.loads((tmp_path / "cfc_lrc_ncp_seed0.json").read_text())
    on, off = run(), run("--no-self-recurrence")
    assert on["self_recurrent"] is True and off["self_recurrent"] is False
    assert on["params_effective"] > off["params_effective"]
