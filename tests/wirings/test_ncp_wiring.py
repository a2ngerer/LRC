"""Tests for the single-cell NCP wiring (docs/ncp-wiring-fix-2026-09-17.md).

Covers the mask STRUCTURE (what the NCP graph allows), the gradient consequence
(masked-off weights never train), that the single cell consumes elapsed_time,
that every registered cell builds under the wiring, and that the deprecated
stacked approximation still builds.
"""
import numpy as np
import pytest
import tensorflow as tf

from src.benchmark.registry import KNOWN_CELLS, cell_kwargs
from src.models.rnn_model import _CELL_REGISTRY
from src.neurons import LRC_Cell, LSTM_Cell, CTRNN_Cell, LTC_Cell
from src.wirings import (NCPWiring, NCPStackedWiring, SparseLinear,
                         effective_param_count)

INTER, COMMAND, MOTOR, INPUT_DIM = 8, 6, 4, 3
UNITS = INTER + COMMAND + MOTOR


@pytest.fixture
def wiring():
    return NCPWiring(LRC_Cell, INTER, COMMAND, MOTOR, seed=42)


def _index_sets(w):
    g = w.graph(INPUT_DIM)
    return (np.array(g._inter_neurons), np.array(g._command_neurons),
            np.array(g._motor_neurons))


# --- (a) mask structure ----------------------------------------------------

def test_no_inter_or_motor_self_recurrence(wiring):
    adj = wiring.adjacency_mask(INPUT_DIM)
    inter, command, motor = _index_sets(wiring)
    assert adj[np.ix_(inter, inter)].sum() == 0
    assert adj[np.ix_(motor, motor)].sum() == 0


def test_motor_units_receive_only_from_command(wiring):
    adj = wiring.adjacency_mask(INPUT_DIM)
    inter, command, motor = _index_sets(wiring)
    incoming = adj[:, motor]
    assert incoming.sum() > 0
    assert incoming[command].sum() == incoming.sum()


def test_command_recurrence_is_non_empty(wiring):
    adj = wiring.adjacency_mask(INPUT_DIM)
    _, command, _ = _index_sets(wiring)
    assert adj[np.ix_(command, command)].sum() > 0


def test_sensory_mask_matches_ncps_sensory_adjacency(wiring):
    g = wiring.graph(INPUT_DIM)
    expected = (np.abs(g.sensory_adjacency_matrix) > 0).astype(np.float32)
    np.testing.assert_array_equal(wiring.sensory_mask(INPUT_DIM), expected)
    # sensory synapses reach the inter neurons only
    inter, command, motor = _index_sets(wiring)
    sens = wiring.sensory_mask(INPUT_DIM)
    assert sens[:, inter].sum() == sens.sum() > 0


def test_masks_have_the_right_shapes(wiring):
    assert wiring.units == UNITS
    assert wiring.adjacency_mask(INPUT_DIM).shape == (UNITS, UNITS)
    assert wiring.sensory_mask(INPUT_DIM).shape == (INPUT_DIM, UNITS)


# --- (b) gradients ---------------------------------------------------------

def test_masked_off_weight_has_exactly_zero_gradient(wiring):
    model = wiring.build_model()
    x = tf.random.stateless_normal([2, 4, INPUT_DIM], seed=(1, 2))
    y = tf.zeros([2, 4, MOTOR])
    cell = model.layers[0].cell
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean((model(x) - y) ** 2)
    grads = dict(zip([v.name for v in model.trainable_variables],
                     tape.gradient(loss, model.trainable_variables)))
    adj = wiring.adjacency_mask(INPUT_DIM)
    off = np.argwhere(adj == 0)
    assert len(off) > 0
    for name, g in grads.items():
        if g is not None and tuple(g.shape) == (UNITS, UNITS) and 'sigma' not in name:
            g = g.numpy()
            assert np.all(g[off[:, 0], off[:, 1]] == 0.0), name


def test_masked_weight_stays_zero_influence_after_training(wiring):
    model = wiring.build_model()
    model.compile(optimizer='adam', loss='mse')
    x = tf.random.stateless_normal([4, 5, INPUT_DIM], seed=(3, 4))
    model.fit(x, tf.zeros([4, 5, MOTOR]), epochs=1, verbose=0)
    cell = model.layers[0].cell
    adj = wiring.adjacency_mask(INPUT_DIM)
    masked = (cell._mask(cell._params['w'], cell.sparsity_mask).numpy())[adj == 0]
    assert np.all(masked == 0.0)


def test_effective_param_count_excludes_masked_entries(wiring):
    model = wiring.build_model()
    model(tf.zeros([1, 2, INPUT_DIM]))
    eff = effective_param_count(model)
    assert 0 < eff < model.count_params()


# --- (c) elapsed_time reaches the single cell ------------------------------

def test_elapsed_time_changes_the_output():
    w = NCPWiring(LTC_Cell, INTER, COMMAND, MOTOR, seed=7)
    cell = w.make_cell()
    rnn = tf.keras.layers.RNN(cell, return_sequences=True)
    x = tf.ones([1, 3, INPUT_DIM])
    slow = rnn((x, tf.fill([1, 3, 1], 0.1)))
    fast = rnn((x, tf.fill([1, 3, 1], 2.0)))
    assert not np.allclose(slow.numpy(), fast.numpy())


# --- (d) every registered cell builds under the wiring ---------------------

@pytest.mark.parametrize('cell_key', KNOWN_CELLS)
def test_registry_cell_builds_under_ncp(cell_key):
    cell_cls = _CELL_REGISTRY[cell_key]
    kwargs = cell_kwargs(cell_key)
    if cell_key == 'cfc_pm':
        # cfc_pm's whole point is a wider backbone; under a sparse wiring the
        # first backbone layer carries the mask and must be units wide.
        with pytest.raises(ValueError, match='backbone_units'):
            NCPWiring(cell_cls, INTER, COMMAND, MOTOR,
                      **kwargs).build_model()(tf.zeros([1, 2, INPUT_DIM]))
        return
    model = NCPWiring(cell_cls, INTER, COMMAND, MOTOR, **kwargs).build_model()
    assert model(tf.zeros([2, 5, INPUT_DIM])).shape == (2, 5, MOTOR)


# --- (e) unmasked cells are untouched --------------------------------------

@pytest.mark.parametrize('cell_cls', [LRC_Cell, LTC_Cell, CTRNN_Cell, LSTM_Cell])
def test_dense_cell_is_unmasked(cell_cls):
    cell = cell_cls(units=5)
    cell.build((None, 3))
    assert not cell.is_masked
    assert cell.sparsity_mask is None and cell.concat_mask is None
    assert cell.masked_off_count() == 0


# --- (f) the deprecated stacked wiring still works -------------------------

def test_ncp_stacked_still_builds():
    model = NCPStackedWiring(LSTM_Cell, INTER, COMMAND, MOTOR).build_model()
    assert len(model.layers) == 5
    assert model(tf.zeros([2, 5, INPUT_DIM])).shape == (2, 5, MOTOR)


def test_sparse_linear_respects_mask():
    """Weights at mask=0 positions produce zero output."""
    mask = np.zeros((4, 6), dtype=np.float32)
    mask[:, :3] = 1.0   # only first 3 outputs connected
    layer = SparseLinear(units=6, mask=mask)
    out = layer(tf.ones([1, 1, 4])).numpy()
    assert layer(tf.zeros([2, 5, 4])).shape == (2, 5, 6)
    assert np.all(out[..., 3:] == 0.0)


# --- (g) the input reaches the motor neurons within ONE call ----------------

def _motor_grad(cell, x):
    """d(sum of motor state) / d(input) for one call from the zero state."""
    x = tf.Variable(x)
    init = cell.get_initial_state(batch_size=1)
    init = list(init) if isinstance(init, (list, tuple)) else [init]
    with tf.GradientTape() as tape:
        _, states = cell(x, init)
        motor = tf.reduce_sum(states[0][:, :MOTOR])
    return tape.gradient(motor, x)


@pytest.mark.parametrize('cell_cls', [LTC_Cell, LRC_Cell, CTRNN_Cell])
def test_ode_cell_motor_sees_input_after_unfolds(cell_cls):
    """Synchronous ODE cells (ncps wired LTCCell semantics): one sub-step moves
    a signal one synapse, so ode_unfolds=1 leaves the motor neurons blind to
    the input and ode_unfolds=6 does not."""
    x = [[0.5, -1.0, 2.0]]
    blind = _motor_grad(NCPWiring(cell_cls, INTER, COMMAND, MOTOR, seed=1,
                                  ode_unfolds=1).make_cell(), x)
    assert blind is None or np.all(blind.numpy() == 0.0)
    cell = NCPWiring(cell_cls, INTER, COMMAND, MOTOR, seed=1,
                     ode_unfolds=6).make_cell()
    assert np.any(_motor_grad(cell, x).numpy() != 0.0)
    # and a changed input changes the motor output of the same fresh cell
    s0 = [tf.zeros([1, UNITS])]
    a = cell(tf.constant(x), s0)[1][0][:, :MOTOR]
    b = cell(tf.constant([[-2.0, 3.0, -1.0]]), s0)[1][0][:, :MOTOR]
    assert not np.array_equal(a.numpy(), b.numpy())


@pytest.mark.parametrize('cell_key', ['gru', 'lstm', 'cfc', 'cfc_lrc'])
def test_layered_cell_motor_sees_input_in_one_call(cell_key):
    """Closed-form / discrete cells run the sequential layer pass (ncps
    WiredCfCCell): non-zero, input-dependent motor output in ONE call."""
    from src.wirings import NCPLayeredCell
    w = NCPWiring(_CELL_REGISTRY[cell_key], INTER, COMMAND, MOTOR, seed=1,
                  **cell_kwargs(cell_key))
    model = w.build_model()
    assert isinstance(model.layers[0].cell, NCPLayeredCell)
    a = model(tf.constant([[[0.5, -1.0, 2.0]]])).numpy()
    b = model(tf.constant([[[-2.0, 3.0, -1.0]]])).numpy()
    assert a.shape == (1, 1, MOTOR)
    assert np.any(a != 0.0) and not np.allclose(a, b)
    assert effective_param_count(model) < model.count_params()


def test_layered_cell_mask_structure():
    """Per-layer masks: no inter-inter / motor-motor recurrence, no input to
    command or motor, command fed by inter + command, motor by command only."""
    w = NCPWiring(_CELL_REGISTRY['gru'], INTER, COMMAND, MOTOR, seed=42)
    cell = w.make_cell()
    cell.build((None, INPUT_DIM))
    # (sensory mask, recurrent mask) handed to each layer's sub-cell
    inter, command, motor = ((c._sensory_mask_src, c._sparsity_mask_src)
                             for c in cell._cells)
    adj = w.adjacency_mask(INPUT_DIM)
    i, c, m = _index_sets(w)
    assert inter[0].shape == (INPUT_DIM, INTER)
    assert np.all(inter[1] == 0)
    assert np.all(motor[1] == 0)
    np.testing.assert_array_equal(command[0], adj[np.ix_(i, c)])
    np.testing.assert_array_equal(command[1], adj[np.ix_(c, c)])
    np.testing.assert_array_equal(motor[0], adj[np.ix_(c, m)])
    assert command[1].sum() > 0


def test_effective_param_count_excludes_sparse_linear_mask():
    model = NCPStackedWiring(LSTM_Cell, INTER, COMMAND, MOTOR).build_model()
    model(tf.zeros([1, 2, INPUT_DIM]))
    dead = sum(int((l.mask.numpy() == 0).sum()) for l in model.layers
               if isinstance(l, SparseLinear))
    assert dead > 0
    assert effective_param_count(model) == model.count_params() - dead


# --- CfC family under ncp: no lateral mixing (ncps WiredCfCCell) -----------

@pytest.mark.parametrize('cell', ['cfc', 'cfc_lrc', 'cfc_lrc_outer'])
def test_cfc_ncp_subcells_have_no_lateral_kernel(cell):
    """backbone_layers=0: every trainable kernel of an inter / motor sub-cell is
    an (input_dim + units, units) map carrying the concat mask, whose recurrent
    rows (intra-layer adjacency) are all zero -- no weight connects two inter
    or two motor neurons."""
    from src.wirings import NCPLayeredCell
    w = NCPWiring(_CELL_REGISTRY[cell], INTER, COMMAND, MOTOR, seed=42,
                  **cell_kwargs(cell))
    model = w.build_model()
    model(tf.zeros([2, 3, INPUT_DIM]))
    layered = model.layers[0].cell
    assert isinstance(layered, NCPLayeredCell)
    for sub in (layered._cells[0], layered._cells[2]):        # inter, motor
        assert sub._backbone == []
        n_in = sub.input_dim
        kernels = [v for v in sub.trainable_variables if 'kernel' in v.name]
        assert kernels
        for k in kernels:
            assert tuple(k.shape) == (n_in + sub.units, sub.units)
            mask = np.asarray(sub._masked_vars[k.ref()])
            assert mask[n_in:].sum() == 0


def test_cfc_dense_keeps_backbone():
    from src.neurons import CfC_Cell
    model = tf.keras.Sequential([tf.keras.layers.RNN(CfC_Cell(units=5))])
    model(tf.zeros([1, 2, INPUT_DIM]))
    assert len(model.layers[0].cell._backbone) == 1
