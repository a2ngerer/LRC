# tests/experiments/test_check_gate_sign.py
# Issue a2ngerer/LRC#98: eval-only sign check of the gate time coefficient
# t_a from saved weights. Tiny synthetic fits of cfc, cfc_lrc and
# cfc_lrc_outer on the three runners (rollout, Person Activity stub,
# committee stub); no real data, no real results.
import json
import sys

import numpy as np
import pandas as pd
import pytest

import src.tasks.person_activity as pa_pkg
from experiments import check_gate_sign as gs
from experiments import run_committee_benchmark as co_runner
from experiments import run_lotka_volterra_benchmark as lv_runner
from experiments import run_person_activity_benchmark as pa_runner
from src.benchmark.checkpoints import rebuild_from_result
from src.neurons.cfc_cell import CfC_Cell
from src.neurons.cfc_lrc_cell import CfC_LRC_Cell
from tests.benchmark.test_checkpoints import T, _pa_stub


@pytest.fixture(scope="module")
def fits(tmp_path_factory):
    """One tiny saved fit per runner and target cell, plus a gru fit that
    the check skips."""
    d = tmp_path_factory.mktemp("fits")
    mp = pytest.MonkeyPatch()
    mp.setattr(pa_pkg, "load_person_activity", _pa_stub)
    mp.setattr(co_runner, "load_person_activity", _pa_stub)
    mp.setattr(gs, "load_person_activity", _pa_stub)
    runs = [
        (lv_runner, ["run_lotka_volterra_benchmark.py", "--wiring", "ncp",
                     "--cell", "cfc_lrc_outer", "--size", "8", "--epochs", "2",
                     "--n-trajectories", "6", "--seq-len", "8", "--anchors",
                     "2,4", "--horizon", "4", "--batch-size", "4"]),
        (pa_runner, ["run_person_activity_benchmark.py", "--wiring", "dense3",
                     "--cell", "cfc", "--size", "4", "--epochs", "2"]),
        (pa_runner, ["run_person_activity_benchmark.py", "--wiring", "dense",
                     "--cell", "gru", "--size", "4", "--epochs", "1"]),
        (co_runner, ["run_committee_benchmark.py", "--n-columns", "2",
                     "--size", "8", "--epochs", "2", "--seq-len", str(T),
                     "--drop-fracs", "0.0", "--drop-seeds", "0"]),
    ]
    for runner, argv in runs:
        mp.setattr(sys, "argv", [*argv, "--outdir", str(d)])
        runner.main()
    yield d
    mp.undo()


def _cells(model):
    return [m for m in model.submodules
            if isinstance(m, (CfC_Cell, CfC_LRC_Cell))]


def test_one_row_per_target_fit_and_summary(fits, tmp_path):
    """Rows for cfc, cfc_lrc and cfc_lrc_outer on all three runners, the gru
    fit skipped; shares in [0, 1], diff = trained - init."""
    out, summ = tmp_path / "rows.csv", tmp_path / "summary.csv"
    gs.main(["--results", str(fits / "*.json"), "--out", str(out),
             "--summary", str(summ)])
    rows = pd.read_csv(out)
    assert sorted(zip(rows.task, rows.cell)) == [
        ("co", "cfc_lrc"), ("lv", "cfc_lrc_outer"), ("pa", "cfc")]
    for c in ("neg_share_trained", "neg_share_init"):
        assert rows[c].between(0, 1).all()
    np.testing.assert_allclose(
        rows.neg_share_diff, rows.neg_share_trained - rows.neg_share_init)
    assert set(rows.system) == {"periodic_predator_prey", "pa", "co"}
    summary = pd.read_csv(summ)
    assert len(summary) == 3 and (summary.n_seeds == 1).all()
    np.testing.assert_allclose(summary.median_neg_share_diff,
                               summary.diff_ci_lo)       # one seed


@pytest.mark.parametrize("name", ["cfc_lrc_outer_ncp_seed0.json",
                                  "cfc_dense3_seed0.json"])
def test_counts_every_unit_step_and_window(fits, name):
    """n_values = windows x time steps x units summed over every gate cell
    (three stacked layers on dense3, the NCP groups on ncp)."""
    path = str(fits / name)
    record = json.load(open(path))
    task = "lv" if "rollout_metrics_test" in record else "pa"
    inputs = gs.test_inputs(record, task)
    model = rebuild_from_result(path)
    cells = _cells(model)
    assert len(cells) >= (3 if "dense3" in name else 1)
    _, n = gs.negative_share(model, inputs)
    x = inputs[0]
    assert n == x.shape[0] * x.shape[1] * sum(c.units for c in cells)


def test_reads_the_time_a_head(fits):
    """t_a forced negative through the time_a bias (kernel zero) gives share
    1, forced positive gives 0; time_b stays untouched, so the hook reads
    t_a, not t_b."""
    path = str(fits / "cfc_lrc_outer_ncp_seed0.json")
    record = json.load(open(path))
    inputs = gs.test_inputs(record, "lv")
    for bias, share in ((-5.0, 1.0), (5.0, 0.0)):
        model = rebuild_from_result(path)
        for cell in _cells(model):
            k, b = cell._time_a.get_weights()
            cell._time_a.set_weights([np.zeros_like(k), np.full_like(b, bias)])
        assert gs.negative_share(model, inputs)[0] == share


def test_untrained_rebuild_is_the_seeded_init(fits):
    """rebuild_from_result(untrained=True) is the builder's init of the
    fit's seed: twice the same weights, not the trained ones."""
    path = str(fits / "cfc_dense3_seed0.json")
    a, b = (rebuild_from_result(path, untrained=True) for _ in range(2))
    trained = rebuild_from_result(path)
    for wa, wb, wt in zip(a.get_weights(), b.get_weights(),
                          trained.get_weights()):
        np.testing.assert_array_equal(wa, wb)
    assert any(not np.array_equal(wa, wt) for wa, wt in
               zip(a.get_weights(), trained.get_weights()))


def test_summary_median_bootstrap_and_reading():
    """Median over seeds, seed-bootstrap interval around it, and the C-1
    reading only when trained median > 0.5, diff median > 0 and lo > 0."""
    rows = pd.DataFrame({
        "task": "lv", "system": "duffing", "arm": ["a"] * 5 + ["b"] * 5,
        "neg_share_trained": [0.7, 0.8, 0.75, 0.9, 0.85,
                              0.4, 0.6, 0.5, 0.45, 0.55],
        "neg_share_init": [0.5] * 10})
    rows["neg_share_diff"] = rows.neg_share_trained - rows.neg_share_init
    s = gs.summarise(rows).set_index("arm")
    assert s.loc["a", "median_neg_share_trained"] == pytest.approx(0.8)
    assert s.loc["a", "median_neg_share_diff"] == pytest.approx(0.3)
    assert 0.2 - 1e-9 <= s.loc["a", "diff_ci_lo"] <= 0.3 <= s.loc["a",
                                                                "diff_ci_hi"]
    assert s.loc["a", "sign_mostly_negative"]
    assert s.loc["b", "diff_ci_lo"] < 0 < s.loc["b", "diff_ci_hi"]
    assert not s.loc["b", "sign_mostly_negative"]
