# tests/neurons/test_cfc_static_cell.py
# cfc_static (issue #31): cfc with per-unit time constants instead of the
# t_a/t_b heads -- the H-a control that differs from cfc only in the gate.
from bisect import bisect_right
from functools import cache

import numpy as np
import pytest
import tensorflow as tf

from src.models.rnn_model import _CELL_REGISTRY
from src.neurons import CfC_Cell, CfCStaticCell
from src.tasks.lotka_volterra import build_lotka_volterra_model
from src.tasks.person_activity.model import effective_backbone_layers
from src.wirings import (NCPLayeredCell, effective_param_count,
                         match_param_budget, param_counts)
from src.wirings.ncp import CFC_NCP_CELLS, SEQUENTIAL_NCP_CELLS


def _cfc_cells(model):
    return [l for l in model._flatten_layers() if isinstance(l, CfC_Cell)]


def _effective_size(cell, var):
    """Entries of ``var`` that the wiring mask leaves on."""
    mask = cell._masked_vars.get(var.ref())
    return int(np.prod(var.shape)) if mask is None else int(np.count_nonzero(mask))


def test_registered_and_in_the_cfc_ncp_family():
    assert _CELL_REGISTRY["cfc_static"] is CfCStaticCell
    # subclass of CfC_Cell: ncp backbone default and layered pass apply as for cfc
    assert issubclass(CfCStaticCell, CFC_NCP_CELLS)
    assert issubclass(CfCStaticCell, SEQUENTIAL_NCP_CELLS)
    assert effective_backbone_layers("ncp", "cfc_static", {}) == \
        effective_backbone_layers("ncp", "cfc", {})


@pytest.mark.parametrize("wiring", ["dense", "dense3", "ncp"])
def test_param_count_is_cfc_minus_the_time_head_kernels(wiring):
    cfc = build_lotka_volterra_model(wiring, "cfc", size=8, seed=0)
    static = build_lotka_volterra_model(wiring, "cfc_static", size=8, seed=0)
    head_kernels = sum(_effective_size(c, h.kernel) for c in _cfc_cells(cfc)
                       for h in (c._time_a, c._time_b))
    assert (param_counts(cfc)["params_effective"]
            - param_counts(static)["params_effective"]) == head_kernels
    if wiring == "ncp":
        rnn, = (l for l in static.layers if isinstance(l, tf.keras.layers.RNN))
        assert isinstance(rnn.cell, NCPLayeredCell)
    x = np.random.default_rng(0).normal(size=(2, 5, 2)).astype(np.float32)
    out = static([x, np.full((2, 5, 1), 0.1, np.float32)])
    assert out.shape == (2, 5, 2) and np.isfinite(out.numpy()).all()


@pytest.mark.parametrize("backbone_layers", [0, 1])
def test_gate_is_independent_of_the_input(backbone_layers):
    """ff1 = 0 and ff2 = 1 make the new state equal to the gate."""
    def gate(cell, x, h, dt):
        return cell.call((tf.constant(x), tf.fill([1, 1], dt)),
                         [tf.constant(h)])[0]

    rng = np.random.default_rng(0)
    xs = rng.normal(size=(2, 1, 3)).astype(np.float32)
    hs = rng.normal(size=(2, 1, 4)).astype(np.float32)
    for cls in (CfCStaticCell, CfC_Cell):
        cell = cls(4, backbone_layers=backbone_layers)
        cell.build((None, 3))
        for head, bias in ((cell._ff1, 0.0), (cell._ff2, 1.0)):
            k, b = head.get_weights()
            head.set_weights([k * 0.0, b * 0.0 + bias])
        if cls is CfCStaticCell:
            cell.t_a.assign(rng.normal(size=4).astype(np.float32))
            cell.t_b.assign(rng.normal(size=4).astype(np.float32))
        g0, g1 = (gate(cell, xs[i], hs[i], 0.3) for i in range(2))
        if cls is CfCStaticCell:
            np.testing.assert_allclose(g0, g1, atol=1e-7)
            assert not np.allclose(g0, gate(cell, xs[0], hs[0], 1.5))  # dt matters
        else:  # the check detects an input-dependent gate
            assert not np.allclose(g0, g1)


def test_shared_weights_start_identical_to_cfc():
    cfc = build_lotka_volterra_model("ncp", "cfc", size=8, seed=3)
    static = build_lotka_volterra_model("ncp", "cfc_static", size=8, seed=3)
    pairs = [(c._ff1, s._ff1) for c, s in zip(_cfc_cells(cfc), _cfc_cells(static))]
    pairs.append((cfc.get_layer("next_state"), static.get_layer("next_state")))
    for a, b in pairs:
        for wa, wb in zip(a.get_weights(), b.get_weights()):
            np.testing.assert_array_equal(wa, wb)


def test_get_config_round_trip():
    cell = CfCStaticCell(6, backbone_units=10, backbone_layers=2, dt=0.5)
    clone = CfCStaticCell.from_config(cell.get_config())
    assert clone.get_config() == cell.get_config()
    for c in (cell, clone):
        c.build((None, 3))
    assert ([w.shape for w in clone.weights] == [w.shape for w in cell.weights])


@pytest.mark.parametrize("budget", [4000, 30000])
@pytest.mark.parametrize("wiring", ["dense", "dense3", "ncp"])
def test_budget_sizing_matches_cfc(wiring, budget):
    """The runner's --param-budget pick (size_for_budget) lands within 5 %."""
    def deviation(cell):
        @cache
        def count(size):
            n = effective_param_count(build_lotka_volterra_model(
                wiring, cell, size=size, seed=0))
            tf.keras.backend.clear_session()
            return n
        # ponytail: size_for_budget scans every size from 2 (minutes at 30k);
        # counts are non-decreasing, so its pick is one of the two sizes around
        # the bisection point and match_param_budget chooses between them.
        sizes = range(2, 257)
        i = bisect_right(sizes, budget, key=count)
        n = match_param_budget(count, budget, sizes[max(i - 1, 0):i + 1])[1]
        return abs(n - budget) / budget
    assert deviation("cfc_static") <= 0.05 and deviation("cfc") <= 0.05
