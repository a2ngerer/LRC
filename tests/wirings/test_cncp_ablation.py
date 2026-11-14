# tests/wirings/test_cncp_ablation.py
# Issue a2ngerer/LRC#136: eval-only path ablation of the cNCP at fixed
# weights. Python flags on CorticalColumnCell (no new variables), switched by
# set_cncp_ablation after the weights are loaded: all off is the plain model
# bit for bit, each flag equals the reference model with the path's weights
# zeroed by hand, cncp_ff (no such paths) raises. Tiny sizes, CPU only.
import numpy as np
import pytest
import tensorflow as tf

from src.tasks.lotka_volterra.model import (build_lotka_volterra_model,
                                            forced_rollout_model)
from src.wirings.cncp import CorticalColumnCell, set_cncp_ablation

# the weights a flag removes, as the reference model zeroes them by hand
ZEROED = {'delayed': ('edges/M_L5ET_L5IT', 'edges/M_L6CC_L23'),
          'apical': ('gain_L23', 'gain_L5ET'),
          'relay': ('edges/M_Thal_L4',)}


def _model(wiring='cncp'):
    """LV model with every weight redrawn at scale 0.5 (seed 0), so each path
    moves the output well above float noise (the init gain is 0.01)."""
    model = build_lotka_volterra_model(wiring, 'cfc_lrc', size=8, seed=0)
    rng = np.random.default_rng(0)
    model.set_weights([0.5 * rng.standard_normal(w.shape).astype(np.float32)
                       for w in model.get_weights()])
    return model


def _inputs():
    rng = np.random.default_rng(1)
    return (rng.normal(size=(3, 10, 2)).astype(np.float32),
            np.full((3, 10, 1), 0.1, np.float32))


def _cell(model):
    (cell,) = [c for c in model._flatten_layers(include_self=True)
               if isinstance(c, CorticalColumnCell)]
    return cell


def _zero(model, paths):
    cell = _cell(model)
    for p in paths:
        for name in ZEROED[p]:
            var = (cell.edges[name.split('/')[1]].W if '/' in name
                   else getattr(cell, name))
            var.assign(tf.zeros_like(var))


def _out(model, x):
    return model(x, training=False).numpy()


@pytest.mark.parametrize('wiring', ['cncp', 'cncp_random'])
def test_each_flag_equals_the_zeroed_reference(wiring):
    """Every flag (and all three) reproduces the reference model whose path
    weights are zero; each changes the output; switching back off restores
    the plain model bit for bit. cncp_random cuts the edges by name, where
    the swap moved them."""
    x = _inputs()
    model, plain = _model(wiring), _out(_model(wiring), x)
    n_vars = len(model.weights)
    set_cncp_ablation(model)
    assert np.array_equal(_out(model, x), plain)
    for paths in (['delayed'], ['apical'], ['relay'], list(ZEROED)):
        set_cncp_ablation(model, **{p: True for p in paths})
        ablated = _out(model, x)
        ref = _model(wiring)
        _zero(ref, paths)
        np.testing.assert_array_equal(ablated, _out(ref, x), err_msg=paths)
        assert np.abs(ablated - plain).max() > 1e-4, paths
    set_cncp_ablation(model)
    assert np.array_equal(_out(model, x), plain)
    assert len(model.weights) == n_vars           # no variable was added


def test_flags_reach_the_view_and_reset_the_predict_cache():
    """The forced-rollout view shares the cell, so a view built before the
    switch follows it; model.predict retraces after the switch."""
    x = _inputs()
    model = _model()
    view = forced_rollout_model(model, tf.Variable(1.0, trainable=False))
    before, view_before = model.predict(x, verbose=0), _out(view, x)
    np.testing.assert_allclose(view_before, before, atol=1e-5)
    set_cncp_ablation(view, relay=True)
    assert model.predict_function is not None     # only the view's dropped
    set_cncp_ablation(model, relay=True)
    assert model.predict_function is None
    after = model.predict(x, verbose=0)
    assert np.abs(after - before).max() > 1e-4
    np.testing.assert_allclose(_out(view, x), after, atol=1e-5)
    set_cncp_ablation(model)
    np.testing.assert_array_equal(model.predict(x, verbose=0), before)


def test_cncp_ff_and_non_cncp_models_raise():
    """cncp_ff has none of the three paths: each flag raises, all off is
    allowed (the ablation 'none'); a model without a cNCP cell raises."""
    model = _model('cncp_ff')
    for p in ZEROED:
        with pytest.raises(ValueError, match=p):
            set_cncp_ablation(model, **{p: True})
    assert not any(getattr(_cell(model), f'ablate_{p}') for p in ZEROED)
    set_cncp_ablation(model)
    with pytest.raises(ValueError, match='no CorticalColumnCell'):
        set_cncp_ablation(build_lotka_volterra_model('ncp', 'cfc_lrc', size=8))
