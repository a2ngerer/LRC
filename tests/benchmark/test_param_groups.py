# tests/benchmark/test_param_groups.py
# Parameter groups of the RQ4 diagnostics (issue a2ngerer/LRC#30): every
# trainable variable lands in exactly one group, none in 'other', for the eight
# thesis cells under dense and ncp.
import pytest

from src.evaluation.param_groups import group_parameters
from src.tasks.lotka_volterra.model import build_lotka_volterra_model

CELLS = ("lstm", "gru", "ctrnn", "ltc", "lrc", "cfc", "cfc_lrc", "mm_lrc")


def _build(cell, wiring):
    # the LV runner's cell_kwargs for the LRC family
    kw = {"elastance_type": "asymmetric"} if cell in ("lrc", "cfc_lrc") else {}
    return build_lotka_volterra_model(wiring, cell, size=8, **kw)


@pytest.mark.parametrize("wiring", ("dense", "ncp"))
@pytest.mark.parametrize("cell", CELLS)
def test_every_variable_in_exactly_one_group(cell, wiring):
    model = _build(cell, wiring)
    groups = group_parameters(model)
    assigned = [v.ref() for vs in groups.values() for v in vs]
    assert len(assigned) == len(set(assigned))
    assert set(assigned) == {v.ref() for v in model.trainable_variables}
    assert "other" not in groups
    assert {"timescale", "readout"} <= set(groups)


def test_table_spot_checks():
    names = {g: {v.name.split("/")[-1] for v in vs}
             for g, vs in group_parameters(_build("ctrnn", "dense")).items()}
    assert names == {"timescale": {"tau:0"}, "recurrent": {"W_h:0", "b:0"},
                     "input": {"W_x:0"},
                     "readout": {"kernel:0", "bias:0"}}
    groups = group_parameters(_build("cfc_lrc", "dense"))
    # backbone_0 (2 vars), ff1/ff2 -> recurrent, time_a/time_b/elastance
    assert len(groups["backbone"]) == 2 and len(groups["recurrent"]) == 4
    assert len(groups["timescale"]) == 6
