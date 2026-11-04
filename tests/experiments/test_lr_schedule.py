# tests/experiments/test_lr_schedule.py
# Linearly decaying learning rate of the LV runner (issue #88, C8): the
# optimizer's lr at the first, middle and last epoch, the runner flags and the
# result JSON, the teacher-forcing ramp, and the unchanged default path (no new
# key, same config_hash at a fixed protocol_version).
import pytest
import tensorflow as tf

from src.benchmark.checkpoints import config_hash
from src.benchmark.optimizers import LinearLRDecay, make_optimizer
from tests.experiments.test_init_flags import _run

NEW_KEYS = {"lr_schedule", "lr_end_fraction", "lr_curve"}

# config_hash[:12] of config.effective without protocol_version / config_hash
# for the default argv of _run, recorded on main before this option existed.
# stamp_protocol hashes protocol_version in, so the comparison leaves it out;
# re-pin only together with a change of the fits these arms produce.
DEFAULT_EFFECTIVE_HASHES = {("ncp", "lrc"): "3c82f5b67486",
                            ("dense", "cfc_lrc"): "7e9d24bfcbad"}


def _lr_per_epoch(optimizer_name, epochs, **schedule):
    """The optimizer's lr as seen at the start of every epoch of a real fit."""
    model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(2,))])
    model.compile(make_optimizer(optimizer_name, 1e-2), "mse")
    seen = []
    tap = tf.keras.callbacks.LambdaCallback(on_epoch_begin=lambda e, logs: seen.append(
        float(tf.keras.backend.get_value(model.optimizer.learning_rate))))
    model.fit(tf.zeros((4, 2)), tf.zeros((4, 1)), epochs=epochs, verbose=0,
              callbacks=[LinearLRDecay(1e-2, epochs, **schedule), tap])
    return seen


@pytest.mark.parametrize("optimizer", ["rmsprop", "rmsprop_legacy", "adam"])
@pytest.mark.parametrize("end", [0.0, 0.1])
def test_optimizer_lr_first_middle_last_epoch(optimizer, end):
    epochs = 10
    seen = _lr_per_epoch(optimizer, epochs, end_fraction=end)
    expected = [1e-2 * (1 - (1 - end) * e / epochs) for e in range(epochs)]
    assert seen == pytest.approx(expected, rel=1e-6)
    assert seen[0] == pytest.approx(1e-2)                       # first
    assert seen[epochs // 2] == pytest.approx(1e-2 * (1 - (1 - end) / 2))
    assert seen[-1] == pytest.approx(1e-2 * (1 - (1 - end) * 0.9))  # last
    assert all(a > b for a, b in zip(seen, seen[1:]))           # strictly falling


def test_schedule_reaches_the_end_fraction_after_the_budget():
    cb = LinearLRDecay(1e-2, 10, end_fraction=0.25)
    assert cb.value(10) == pytest.approx(0.25e-2)
    assert cb.value(0) == pytest.approx(1e-2)


def test_clipnorm_survives_the_schedule():
    model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(2,))])
    model.compile(make_optimizer("rmsprop", 1e-2), "mse")
    model.optimizer.global_clipnorm = 1.0
    model.fit(tf.ones((4, 2)), tf.ones((4, 1)), epochs=3, verbose=0,
              callbacks=[LinearLRDecay(1e-2, 3)])
    assert model.optimizer.global_clipnorm == 1.0
    assert type(model.optimizer) is tf.keras.optimizers.RMSprop


@pytest.mark.parametrize("wiring,cell", list(DEFAULT_EFFECTIVE_HASHES))
def test_default_run_adds_no_key_and_keeps_config_hash(
        wiring, cell, tmp_path, monkeypatch):
    record, _, _ = _run("lv", wiring, tmp_path, monkeypatch, cell=cell)
    eff = record["config"]["effective"]
    assert not NEW_KEYS & (set(record) | set(eff) | set(eff["build"]))
    rest = {k: v for k, v in eff.items()
            if k not in ("protocol_version", "config_hash")}
    assert config_hash(rest)[:12] == DEFAULT_EFFECTIVE_HASHES[(wiring, cell)]


def test_explicit_constant_equals_the_default(tmp_path, monkeypatch):
    record, _, _ = _run("lv", "ncp", tmp_path, monkeypatch,
                        "--lr-schedule", "constant")
    eff = record["config"]["effective"]
    assert not NEW_KEYS & (set(record) | set(eff) | set(eff["build"]))
    assert eff["config_hash"] == config_hash(
        {k: v for k, v in eff.items() if k != "config_hash"})
    rest = {k: v for k, v in eff.items()
            if k not in ("protocol_version", "config_hash")}
    assert config_hash(rest)[:12] == DEFAULT_EFFECTIVE_HASHES[("ncp", "lrc")]


def test_linear_run_records_schedule_and_changes_only_that(tmp_path, monkeypatch):
    base, _, _ = _run("lv", "ncp", tmp_path / "a", monkeypatch)
    record, _, model = _run("lv", "ncp", tmp_path / "b", monkeypatch,
                            "--lr-schedule", "linear", "--epochs", "4",
                            "--lr", "0.01")
    eff = record["config"]["effective"]
    assert eff["lr_schedule"] == "linear" and eff["lr_end_fraction"] == 0.0
    assert record["lr_schedule"] == "linear" and record["lr_end_fraction"] == 0.0
    assert record["lr_curve"] == pytest.approx([0.01, 0.0075, 0.005, 0.0025])
    # build, i.e. architecture and initialisation, is the constant run's
    base_eff = base["config"]["effective"]
    assert {**eff["build"], "lr": None} == {**base_eff["build"], "lr": None}
    assert eff["config_hash"] != base_eff["config_hash"]
    assert float(tf.keras.backend.get_value(model.optimizer.learning_rate)
                 ) == pytest.approx(0.0025)


def test_end_fraction_flag(tmp_path, monkeypatch):
    record, _, _ = _run("lv", "ncp", tmp_path, monkeypatch,
                        "--lr-schedule", "linear", "--lr-end-fraction", "0.5",
                        "--epochs", "2", "--lr", "0.01")
    assert record["lr_end_fraction"] == 0.5
    assert record["lr_curve"] == pytest.approx([0.01, 0.0075])


def test_decay_spans_the_whole_budget_beyond_the_forcing_ramp(
        tmp_path, monkeypatch):
    # the alpha ramp ends after epochs // 2 = 2 epochs; the lr keeps falling
    record, _, model = _run("lv", "ncp", tmp_path, monkeypatch,
                            "--forcing", "mixed", "--lr-schedule", "linear",
                            "--epochs", "4", "--lr", "0.01")
    assert record["alpha_curve"][2:] == pytest.approx([0.2, 0.2])
    assert record["lr_curve"] == pytest.approx([0.01, 0.0075, 0.005, 0.0025])
    assert float(tf.keras.backend.get_value(model.optimizer.learning_rate)
                 ) == pytest.approx(0.0025)


@pytest.mark.parametrize("argv", [["--lr-end-fraction", "0.1"],
                                  ["--lr-schedule", "linear",
                                   "--lr-end-fraction", "1.5"],
                                  ["--lr-schedule", "linear",
                                   "--lr-end-fraction", "-0.1"]])
def test_invalid_end_fraction_is_rejected(argv, tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        _run("lv", "ncp", tmp_path, monkeypatch, *argv)
