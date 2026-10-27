# tests/experiments/test_registry_cell_kwargs.py
# Cell kwargs come from src.benchmark.registry in the LV, PA and committee
# runners (issue a2ngerer/LRC#51): every LRC-family arm builds with
# elastance_type='asymmetric' (mixed-memory wrappers forward it to the inner
# cell), cfc_lrc_outer carries elastance_gate='outer', and --elastance-type
# still overrides. Each case runs the real runner main() for one epoch on
# synthetic data and reads both the result JSON and the built cells.
import json
import sys

import numpy as np
import pytest

import src.tasks.person_activity as pa_pkg
from experiments import run_committee_benchmark as co_runner
from experiments import run_lotka_volterra_benchmark as lv_runner
from experiments import run_person_activity_benchmark as pa_runner
from src.benchmark import registry
from src.neurons.mixed_memory_cell import MixedMemoryCell
from tests.benchmark.test_checkpoints import T, _capture_builds, _pa_stub

ASYM_CELLS = ("lrc", "lrc_pm", "mm_lrc", "cfc_lrc", "cfc_lrc_outer",
              "cfc_mm_lrc")
MULTI_STATE = ("mm_lrc", "cfc_mm_lrc")       # committee rejects them
PLAIN_CELLS = ("ctrnn", "ltc", "cfc", "cfc_static", "gru", "lstm", "mm_ltc",
               "cfc_mm_ltc")


def _run(runner, cell, tmp_path, monkeypatch, *extra):
    """One-epoch run of `runner` ('lv', 'pa', 'committee') on a dense `cell`
    arm; returns (result record, built model)."""
    common = ["--cell", cell, "--epochs", "1", "--outdir", str(tmp_path),
              *extra]
    if runner == "lv":
        built = _capture_builds(monkeypatch, lv_runner,
                                "build_lotka_volterra_model")
        argv = ["--wiring", "dense", "--size", "4", "--n-trajectories", "4",
                "--anchors", "2,4", "--horizon", "4",
                "--seq-len", "8", "--batch-size", "4"]
        main = lv_runner.main
    elif runner == "pa":
        built = _capture_builds(monkeypatch, pa_pkg,
                                "build_person_activity_model")
        monkeypatch.setattr(pa_pkg, "load_person_activity", _pa_stub)
        argv = ["--wiring", "dense", "--size", "4"]
        main = pa_runner.main
    else:
        built = _capture_builds(monkeypatch, co_runner, "build_committee_model")
        monkeypatch.setattr(co_runner, "load_person_activity", _pa_stub)
        argv = ["--wiring", "dense", "--n-columns", "2", "--size", "4",
                "--seq-len", str(T), "--drop-fracs", "0.0", "--drop-seeds",
                "0"]
        main = co_runner.main
    monkeypatch.setattr(sys, "argv", ["runner.py", *argv, *common])
    main()
    (path,) = tmp_path.glob("*.json")
    return json.loads(path.read_text()), built[-1][0]


def _elastance_cells(model):
    """Every built cell carrying an elastance mechanism (inner cells of the
    mixed-memory wrappers included)."""
    return [l for l in model._flatten_layers(include_self=False, recursive=True)
            if hasattr(l, "_elastance_type")]


RUNNER_CELLS = [(r, c) for r in ("lv", "pa", "committee") for c in ASYM_CELLS
                if not (r == "committee" and c in MULTI_STATE)]


@pytest.mark.parametrize("runner,cell", RUNNER_CELLS)
def test_lrc_family_builds_asymmetric(runner, cell, tmp_path, monkeypatch):
    record, model = _run(runner, cell, tmp_path, monkeypatch)
    eff = record["config"]["effective"]
    assert eff["elastance_type"] == "asymmetric"
    assert eff["cell_kwargs"]["elastance_type"] == "asymmetric"
    cells = _elastance_cells(model)
    assert cells and all(c._elastance_type == "asymmetric" for c in cells)
    if cell in MULTI_STATE:      # the wrapper forwards to its inner LRC cell
        (wrapper,) = [l for l in model._flatten_layers()
                      if isinstance(l, MixedMemoryCell)]
        assert wrapper._inner._elastance_type == "asymmetric"
    gate = "outer" if cell == "cfc_lrc_outer" else None
    assert eff["cell_kwargs"].get("elastance_gate") == gate
    if cell.startswith("cfc_lrc"):
        assert all(c._elastance_gate == (gate or "inner") for c in cells)


@pytest.mark.parametrize("runner", ["lv", "pa", "committee"])
def test_elastance_type_flag_overrides_registry(runner, tmp_path, monkeypatch):
    record, model = _run(runner, "lrc", tmp_path, monkeypatch,
                         "--elastance-type", "symmetric")
    assert record["config"]["effective"]["elastance_type"] == "symmetric"
    assert [c._elastance_type for c in _elastance_cells(model)] == ["symmetric"]


@pytest.mark.parametrize("runner", ["lv", "pa"])
def test_outer_gate_changes_the_fit(runner, tmp_path, monkeypatch):
    """Same seed, one epoch: cfc_lrc_outer must not reproduce cfc_lrc."""
    stub = _pa_stub()
    losses = {}
    for cell in ("cfc_lrc", "cfc_lrc_outer"):
        out = tmp_path / cell
        record, model = _run(runner, cell, out, monkeypatch)
        losses[cell] = (record["val_loss_curve"] if runner == "lv" else
                        model.evaluate((stub.train_x, stub.train_t),
                                       stub.train_y, verbose=0)[0])
    assert not np.allclose(losses["cfc_lrc"], losses["cfc_lrc_outer"])


def test_registry_kwargs_table():
    for cell in ("lrc", "lrc_ar", *ASYM_CELLS):
        assert registry.cell_kwargs(cell)["elastance_type"] == "asymmetric"
    assert registry.cell_kwargs("cfc_lrc_outer")["elastance_gate"] == "outer"
    for cell in PLAIN_CELLS:
        assert registry.cell_kwargs(cell) == {}
        # the --elastance-type override never reaches cells without one
        assert registry.cell_kwargs(cell, "symmetric") == {}
    assert registry.cell_kwargs("lrc", "symmetric") == {
        "elastance_type": "symmetric"}
