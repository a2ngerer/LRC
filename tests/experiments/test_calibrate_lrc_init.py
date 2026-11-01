# tests/experiments/test_calibrate_lrc_init.py
# The LSUV calibration script behind lrc_cal / lrc_cal_lv / lrc_cal_duffing
# (issue #86): it reproduces the stored factors of Person Activity and
# Lotka-Volterra, the Duffing key carries its output, and the stored values
# are the square roots of the LSUV factors to one decimal. CPU, forward passes
# only; the calibrations run a few seconds to minutes (slow).
import os

import numpy as np
import pytest

from experiments.calibrate_lrc_init import calibrate, stored_scale
from src.benchmark.registry import cell_kwargs
from src.tasks.person_activity.datasets import DEFAULT_DATA_PATH

# LSUV factors (std units, two decimals) of the layer-wise calibration of the
# fresh lrc/ncp network, model seed 0, batch 0, ode_unfolds 6, dt_scale auto:
# Person Activity (size 119) and Lotka-Volterra (size 43).
PA_FACTORS = {"inter": 51.31, "command": 43.26, "motor": 43.40}
LV_FACTORS = {"inter": 28.00, "command": 18.42, "motor": 14.38}
# The pinned factors carry two decimals (relative rounding <= 1e-4); the rest of
# the tolerance covers BLAS differences between machines, which stay orders of
# magnitude below it (the calibration is a handful of float32 forward passes).
RTOL = 5e-3

requires_data = pytest.mark.skipif(
    not os.path.isfile(DEFAULT_DATA_PATH),
    reason="person-activity data file not downloaded (run download_dataset.sh)")


def _assert_factors(got, want):
    assert set(got) == {"inter", "command", "motor"}
    for layer, f in want.items():
        assert got[layer] == pytest.approx(f, rel=RTOL), layer


@pytest.mark.parametrize("cell,factors", [("lrc_cal", PA_FACTORS),
                                          ("lrc_cal_lv", LV_FACTORS)])
def test_stored_values_are_rounded_square_roots(cell, factors):
    assert stored_scale(factors) == cell_kwargs(cell)["init_slice_scale"]


@pytest.mark.slow
def test_reproduces_the_lv_factors():
    r = calibrate("lv", size=43, log=lambda *_: None)
    _assert_factors(r["factors"], LV_FACTORS)
    assert r["stored"] == cell_kwargs("lrc_cal_lv")["init_slice_scale"]


@pytest.mark.slow
@requires_data
def test_reproduces_the_pa_factors():
    r = calibrate("pa", size=119, log=lambda *_: None)
    _assert_factors(r["factors"], PA_FACTORS)
    assert r["stored"] == cell_kwargs("lrc_cal")["init_slice_scale"]


@pytest.mark.slow
def test_lrc_cal_duffing_carries_the_script_output():
    """The documented command (--task lv --system duffing --seed 0, ncp, suite
    budget): the registry holds its stored values; an arm built with them
    needs no further scaling on the same batch."""
    r = calibrate("lv", system="duffing", log=lambda *_: None)
    stored = cell_kwargs("lrc_cal_duffing")["init_slice_scale"]
    assert r["size"] == 43
    for layer, v in stored.items():
        # stored to one decimal: within the rounding half-width (+ slack)
        assert np.sqrt(r["factors"][layer]) == pytest.approx(v, abs=0.06), layer
    again = calibrate("lv", system="duffing", size=r["size"],
                      cell="lrc_cal_duffing", log=lambda *_: None)
    assert again["factors"] == {"inter": 1.0, "command": 1.0, "motor": 1.0}
