"""Degree-preserving random rewiring of the NCP and cNCP graphs (issue #17).

Covers the swap engines (degrees, pass order, determinism), the ``ncp_random``
wiring (same degrees and effective parameter count as ``ncp`` of the same seed,
generalised read set of NCPLayeredCell) and the ``cncp_random`` null model (the
one admissible group-level swap, signs, delay types, shapes, parameter count).
"""
import numpy as np
import pytest
import tensorflow as tf

from src.benchmark.registry import cell_kwargs
from src.models.rnn_model import _CELL_REGISTRY
from src.neurons import GRU_Cell
from src.tasks.person_activity.model import lamina_sizes, scaled_lamina_units
from src.wirings import (CorticalColumnCell, NCPLayeredCell, NCPWiring,
                         SignedSparseLinear, effective_param_count,
                         rewire_degree_preserving, rewire_group_edges)
from src.wirings.cncp import (DELAYED_BASAL, NODE_ORDER, _GROUP_EDGES,
                              _PASS_ORDER, group_edges)
from src.wirings.rewiring import admissible_group_swaps

INTER, COMMAND, MOTOR, INPUT_DIM = 8, 6, 4, 3
UNITS = INTER + COMMAND + MOTOR
LAYER_CELLS = ['gru', 'lstm', 'cfc', 'cfc_lrc', 'cfc_lrc_outer']
ODE_CELLS = ['ltc', 'lrc']


# --- swap engine -------------------------------------------------------------

def _layered_graph(seed=0, sizes=(4, 6, 6, 4), p=0.4):
    """Random graph over layers 0..3 with forward and intra-layer edges."""
    rng = np.random.default_rng(seed)
    layer_of = np.repeat(np.arange(len(sizes)), sizes)
    allowed = layer_of[:, None] <= layer_of[None, :]
    adj = (rng.random((len(layer_of),) * 2) < p) & allowed
    adj[:, layer_of == 0] = False                  # layer 0 is source-only
    return adj, layer_of


def test_engine_keeps_degrees_and_pass_order():
    adj, layer_of = _layered_graph()
    new, overlap = rewire_degree_preserving(adj, layer_of, seed=1,
                                            n_swaps=10 * int(adj.sum()))
    np.testing.assert_array_equal(new.sum(0), adj.sum(0))        # in-degree
    np.testing.assert_array_equal(new.sum(1), adj.sum(1))        # out-degree
    src, dst = np.nonzero(new)
    assert np.all(layer_of[src] <= layer_of[dst])
    assert new.sum() == adj.sum()
    assert 0.0 <= overlap < 1.0
    assert overlap == pytest.approx((new & adj).sum() / adj.sum())


def test_engine_zero_swaps_is_the_identity():
    adj, layer_of = _layered_graph()
    new, overlap = rewire_degree_preserving(adj, layer_of, seed=1, n_swaps=0)
    np.testing.assert_array_equal(new, adj)
    assert overlap == 1.0


def test_engine_is_deterministic_per_seed():
    adj, layer_of = _layered_graph()
    n = 10 * int(adj.sum())
    a = rewire_degree_preserving(adj, layer_of, 3, n)[0]
    b = rewire_degree_preserving(adj, layer_of, 3, n)[0]
    c = rewire_degree_preserving(adj, layer_of, 4, n)[0]
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_engine_keeps_self_loops_fixed():
    adj, layer_of = _layered_graph()
    adj[np.diag_indices(len(adj))] = False
    loops = [5, 9, 12]
    adj[loops, loops] = True
    new, _ = rewire_degree_preserving(adj, layer_of, 2, 10 * int(adj.sum()))
    np.testing.assert_array_equal(np.diag(new), np.diag(adj))
    np.testing.assert_array_equal(new.sum(0), adj.sum(0))
    np.testing.assert_array_equal(new.sum(1), adj.sum(1))


def test_engine_rejects_a_backward_edge():
    adj, layer_of = _layered_graph()
    adj[10, 5] = True                               # layer 2 -> layer 1
    with pytest.raises(ValueError, match="backwards"):
        rewire_degree_preserving(adj, layer_of, 0, 10)


def test_more_swaps_move_further_from_the_original():
    adj, layer_of = _layered_graph()
    few = rewire_degree_preserving(adj, layer_of, 1, 5)[1]
    many = rewire_degree_preserving(adj, layer_of, 1, 10 * int(adj.sum()))[1]
    assert many < few < 1.0


# --- ncp_random --------------------------------------------------------------

def _combined(wiring, input_dim=INPUT_DIM):
    """Source-and-state adjacency, node order [input | units]."""
    sens, adj = wiring.sensory_mask(input_dim), wiring.adjacency_mask(input_dim)
    n, u = input_dim, wiring.units
    full = np.zeros((n + u, n + u), bool)
    full[:n, n:], full[n:, n:] = sens > 0, adj > 0
    return full


def _pass_layers(wiring, input_dim=INPUT_DIM):
    g = wiring.graph(input_dim)
    layer_of = np.zeros(input_dim + wiring.units, int)
    for k, idx in enumerate((g._inter_neurons, g._command_neurons,
                             g._motor_neurons), 1):
        layer_of[input_dim + np.asarray(idx)] = k
    return layer_of


@pytest.mark.parametrize('seed', [0, 1, 7])
@pytest.mark.parametrize('sizes,input_dim', [((INTER, COMMAND, MOTOR), 3),
                                              ((16, 8, 8), 7)])
def test_ncp_random_keeps_degrees_and_forward_edges(sizes, input_dim, seed):
    plain = NCPWiring(GRU_Cell, *sizes, seed=seed)
    rand = NCPWiring(GRU_Cell, *sizes, seed=seed, rewire=True)
    a, b = _combined(plain, input_dim), _combined(rand, input_dim)
    np.testing.assert_array_equal(b.sum(0), a.sum(0))
    np.testing.assert_array_equal(b.sum(1), a.sum(1))
    layer_of = _pass_layers(rand, input_dim)
    src, dst = np.nonzero(b)
    assert np.all(layer_of[src] <= layer_of[dst])           # no backward edge
    assert not np.array_equal(a, b)
    assert rand.rewire_overlap(input_dim) == pytest.approx((a & b).sum() / a.sum())
    assert plain.rewire_overlap(input_dim) is None


def test_ncp_random_overlap_is_not_above_the_random_expectation():
    """After 10 x edges swaps the chain has mixed: the overlap with the
    original is no higher than between independent rewirings (the random
    expectation), which is far below 1."""
    sizes, input_dim = (16, 8, 8), 7
    plain = NCPWiring(GRU_Cell, *sizes, seed=0)
    a = _combined(plain, input_dim)
    samples = [_combined(NCPWiring(GRU_Cell, *sizes, seed=0, rewire=True),
                         input_dim)]
    for s in range(1, 6):                       # same NCP graph, other chains
        samples.append(rewire_degree_preserving(
            a, _pass_layers(plain, input_dim), 100 + s, 10 * int(a.sum()))[0])
    to_original = np.mean([(x & a).sum() / a.sum() for x in samples])
    pairwise = np.mean([(x & y).sum() / a.sum()
                        for i, x in enumerate(samples) for y in samples[:i]])
    assert to_original < 0.5
    assert to_original < pairwise + 0.05


@pytest.mark.parametrize('seed', [0, 3])
@pytest.mark.parametrize('cell', LAYER_CELLS + ODE_CELLS)
def test_ncp_random_effective_params_equal_ncp(cell, seed):
    def count(rewire):
        w = NCPWiring(_CELL_REGISTRY[cell], INTER, COMMAND, MOTOR, seed=seed,
                      rewire=rewire, **cell_kwargs(cell))
        model = w.build_model()
        model(tf.zeros([1, 2, INPUT_DIM]))
        return effective_param_count(model)
    assert count(True) == count(False)


@pytest.mark.parametrize('cell', ODE_CELLS)
def test_ncp_random_ode_cells_run_on_the_rewired_masks(cell):
    """The synchronous single cell of the ODE cells gets the rewired masks, not
    the plain NCP ones."""
    kw = cell_kwargs(cell)
    rand = NCPWiring(_CELL_REGISTRY[cell], INTER, COMMAND, MOTOR, seed=1,
                     rewire=True, **kw)
    plain = NCPWiring(_CELL_REGISTRY[cell], INTER, COMMAND, MOTOR, seed=1, **kw)
    model = rand.build_model()
    model(tf.zeros([1, 2, INPUT_DIM]))
    sub = model.layers[0].cell
    np.testing.assert_array_equal(sub.sparsity_mask.numpy(),
                                  rand.adjacency_mask(INPUT_DIM))
    np.testing.assert_array_equal(sub.sensory_mask.numpy(),
                                  rand.sensory_mask(INPUT_DIM))
    assert not np.array_equal(sub.sparsity_mask.numpy(),
                              plain.adjacency_mask(INPUT_DIM))
    assert rand.sensory_mask(INPUT_DIM)[:, :COMMAND + MOTOR].sum() > 0  # beyond inter


def test_ncp_random_raises_when_no_swap_is_admissible():
    with pytest.raises(ValueError, match="no edge swap"):
        NCPWiring(GRU_Cell, 2, 1, 1, seed=2, rewire=True).adjacency_mask(1)


# --- generalised read set of NCPLayeredCell ----------------------------------

@pytest.mark.parametrize('seed', range(6))
@pytest.mark.parametrize('sizes,input_dim', [((INTER, COMMAND, MOTOR), 3),
                                              ((16, 8, 8), 7), ((5, 3, 2), 1)])
def test_plain_ncp_layers_read_only_their_predecessor(sizes, input_dim, seed):
    """No-op of the generalisation: in the NCP graph the sensory input reaches
    the inter layer only and each later layer is fed by the layer before it, so
    the read set and the input masks are those of the previous-layer-only
    cell."""
    w = NCPWiring(GRU_Cell, *sizes, seed=seed)
    cell = w.make_cell()
    cell.build((None, input_dim))
    assert cell._reads == [[0], [1], [2]]
    g, adj = w.graph(input_dim), w.adjacency_mask(input_dim)
    inter, command, motor = (np.asarray(i) for i in
                             (g._inter_neurons, g._command_neurons,
                              g._motor_neurons))
    masks = [c._sensory_mask_src for c in cell._cells]
    np.testing.assert_array_equal(masks[0], w.sensory_mask(input_dim)[:, inter])
    np.testing.assert_array_equal(masks[1], adj[np.ix_(inter, command)])
    np.testing.assert_array_equal(masks[2], adj[np.ix_(command, motor)])


@pytest.mark.parametrize('cell_key', ['gru', 'cfc_lrc'])
def test_plain_ncp_cell_matches_the_previous_layer_only_reference(cell_key):
    """The call graph of the plain NCP is the previous-layer-only loop: feed
    each sub-cell the new state of the layer before it (x for the first)."""
    w = NCPWiring(_CELL_REGISTRY[cell_key], INTER, COMMAND, MOTOR, seed=2,
                  **cell_kwargs(cell_key))
    layered = w.make_cell()
    layered.build((None, INPUT_DIM))
    x = tf.random.stateless_normal([3, INPUT_DIM], seed=(5, 6))
    states = [tf.random.stateless_normal([3, UNITS], seed=(7, 8))]
    out, new = layered(x, states)
    v, outs = x, []
    for sl, sub in zip(layered._slices, layered._cells):
        v, _ = sub(v, [states[0][:, sl]])
        outs.append(v)
    np.testing.assert_array_equal(out.numpy(),
                                  tf.concat(outs[::-1], axis=-1).numpy())


def test_rewired_layered_cell_loses_no_edge():
    """Every edge of the rewired graph is a weight of some sub-cell: the input
    masks of the layers (over x and the earlier layers) and their intra-layer
    masks reproduce the rewired sensory and adjacency masks exactly."""
    w = NCPWiring(GRU_Cell, 16, 8, 8, seed=1, rewire=True)
    cell = w.make_cell()
    cell.build((None, 7))
    sens, adj = w.sensory_mask(7), w.adjacency_mask(7)
    got_sens, got_adj = np.zeros_like(sens), np.zeros_like(adj)
    widths = [7] + [sl.stop - sl.start for sl in cell._slices]
    node_sets = [None] + [np.arange(sl.start, sl.stop) for sl in cell._slices]
    for sub, sl, reads in zip(cell._cells, cell._slices, cell._reads):
        cols = np.arange(sl.start, sl.stop)
        rows = np.cumsum([0] + [widths[i] for i in reads])
        for src, lo, hi in zip(reads, rows[:-1], rows[1:]):
            block = sub._sensory_mask_src[lo:hi]
            if src == 0:
                got_sens[:, cols] = block
            else:
                got_adj[np.ix_(node_sets[src], cols)] = block
        got_adj[np.ix_(cols, cols)] = sub._sparsity_mask_src
    np.testing.assert_array_equal(got_sens, sens)
    np.testing.assert_array_equal(got_adj, adj)
    assert any(r != [i] for i, r in enumerate(cell._reads))   # reads beyond predecessor


class _FixedMasks(NCPWiring):
    """NCPWiring over hand-made masks (2 + 2 + 2 units, one input)."""

    def __init__(self, sens, adj, **kw):
        super().__init__(GRU_Cell, 2, 2, 2, seed=0, **kw)
        self._fixed = (np.asarray(sens, np.float32), np.asarray(adj, np.float32))

    def sensory_mask(self, input_dim):
        return self._fixed[0]

    def adjacency_mask(self, input_dim):
        return self._fixed[1]


@pytest.mark.parametrize('edge,layer,reads', [
    ('sensory->motor', 2, [0, 2]),
    ('inter->motor', 2, [1, 2]),
    ('sensory->command', 1, [0, 1])])
def test_layer_reads_beyond_its_predecessor(edge, layer, reads):
    """ncps order is [motor(0,1) | command(2,3) | inter(4,5)]. A layer fed by
    the input, or by a layer other than its predecessor, sees it in one call --
    the previous-layer-only cell dropped these edges."""
    sens, adj = np.zeros((1, 6)), np.zeros((6, 6))
    sens[0, 4] = 1                                  # x -> inter 4 (drives it)
    if edge == 'sensory->motor':
        sens[0, 0] = 1
    elif edge == 'inter->motor':
        adj[4, 0] = 1
    else:
        sens[0, 2] = 1
    cell = _FixedMasks(sens, adj, self_recurrent=False).make_cell()
    cell.build((None, 1))
    assert cell._reads[layer] == reads
    target = cell._slices[layer]
    zero = [tf.zeros([1, 6])]
    lo = cell(tf.constant([[-1.0]]), zero)[0][:, target].numpy()
    hi = cell(tf.constant([[1.0]]), zero)[0][:, target].numpy()
    assert not np.allclose(lo, hi)


def test_unconnected_layer_ignores_the_input():
    """Control of the test above: without the extra edge the motor layer does
    not see x (its only source is the empty command -> motor mask)."""
    sens, adj = np.zeros((1, 6)), np.zeros((6, 6))
    sens[0, 4] = 1
    cell = _FixedMasks(sens, adj, self_recurrent=False).make_cell()
    cell.build((None, 1))
    zero = [tf.zeros([1, 6])]
    lo = cell(tf.constant([[-1.0]]), zero)[0][:, :2].numpy()
    hi = cell(tf.constant([[1.0]]), zero)[0][:, :2].numpy()
    np.testing.assert_array_equal(lo, hi)


# --- cncp_random -------------------------------------------------------------

def _group_graph(units=None, sign_constraint=True):
    blocks, kinds = group_edges(sign_constraint)
    cell = CorticalColumnCell('gru')
    width = {**cell._units, **(units or {}), 'in': None}
    rank = {g: i for i, g in enumerate(_PASS_ORDER)}
    return blocks, kinds, [(s, d) for _, s, d in blocks], width, rank


def test_exactly_one_group_swap_is_admissible_in_the_default_graph():
    blocks, kinds, edges, width, rank = _group_graph()
    pairs = admissible_group_swaps(edges, kinds, width, rank)
    names = [{blocks[i][0], blocks[j][0]} for i, j in pairs]
    assert names == [{'M_L6CC_L23', 'M_Thal_L4'}]


def test_group_inventory_matches_the_cell():
    """The inventory behind the swap rule lists exactly the edge classes the
    cell instantiates, with the source and target widths of their masks."""
    cell = CorticalColumnCell('gru')
    cell.build((None, 5))
    blocks, _ = group_edges()
    assert {n for n, _, _ in blocks} == set(cell._active_edges())
    u = {**cell._units, 'in': 5}
    for name, layer in cell.edges.items():
        srcs = [s for n, s, _ in blocks if n == name]
        dst = {d for n, _, d in blocks if n == name}.pop()
        assert layer._mask_np.shape == (sum(u[s] for s in srcs), u[dst])


def test_cncp_random_swaps_the_targets_of_the_two_delayed_feedbacks():
    cell = CorticalColumnCell('gru', seed=4, rewire=True)
    assert dict((n, (s, d)) for n, s, d in cell._delayed) == {
        'M_L5ET_L5IT': ('L5ET', 'L5IT'),
        'M_L6CC_L23': ('L6CC', 'L4'),
        'M_Thal_L4': ('Thal', 'L23')}
    assert cell.rewire_overlap == pytest.approx(17 / 19)
    plain = CorticalColumnCell('gru', seed=4)
    assert plain.rewire_overlap is None
    assert dict((n, (s, d)) for n, s, d in plain._delayed) == DELAYED_BASAL


def test_cncp_random_keeps_degrees_signs_delay_types_and_masks():
    blocks, kinds, edges, width, rank = _group_graph()
    new, overlap = rewire_group_edges(edges, kinds, width, rank, 0, 190)
    assert overlap < 1.0
    for deg in (0, 1):                                 # out- then in-degree
        for g in set(width):
            assert (sum(e[deg] == g for e in new)
                    == sum(e[deg] == g for e in edges)), (deg, g)
    for (old_s, old_d), (s, d) in zip(edges, new):
        assert s == old_s and width[d] == width[old_d]   # source and mask shape
    assert len(set(new)) == len(new)                   # no parallel edge
    assert all(s != d for s, d in new)                 # no self-loop
    moved = {blocks[i][0] for i, (a, b) in enumerate(zip(edges, new)) if a != b}
    assert moved == {'M_L6CC_L23', 'M_Thal_L4'}


def test_cncp_random_leaves_signs_and_same_step_edges_alone():
    cell = CorticalColumnCell('gru', seed=2, rewire=True)
    cell.build((None, 2))
    assert isinstance(cell.edges['M_TRN_Thal'], SignedSparseLinear)
    moved = {n for n, s, d in cell._delayed if (s, d) != DELAYED_BASAL[n]}
    assert moved == {'M_L6CC_L23', 'M_Thal_L4'}
    blocks, kinds = group_edges()
    assert [b[0] for b, k in zip(blocks, kinds) if k[2] == '-'] == ['M_TRN_Thal']


def test_group_inventory_delay_labels_match_what_the_cell_reads():
    """The S / D label of every edge class in the inventory is the one the cell
    implements: a delayed edge is fed from the previous-state tensors only, a
    same-step edge from values computed in this step (or the input)."""
    cell = CorticalColumnCell('gru')
    cell.build((None, 2))
    seen = {}

    class Tap:
        def __init__(self, name, layer):
            self.name, self.layer = name, layer

        def __call__(self, x):
            seen[self.name] = np.unique(np.round(x.numpy(), 5))
            return self.layer(x)

    cell.edges = {n: Tap(n, l) for n, l in cell.edges.items()}
    fills = [10.0 + k for k in range(len(NODE_ORDER))]
    states = [tf.fill([1, u], f) for u, f in zip(cell.state_size, fills)]
    cell(tf.ones([1, 2]), states)
    for name, (srcs, dst, delay, role) in _GROUP_EDGES.items():
        from_previous = bool(np.all(np.isin(seen[name], fills)))
        assert from_previous == (delay == 'D'), name


@pytest.mark.parametrize('size', [None, 9.5])
@pytest.mark.parametrize('cell_key', ['gru', 'cfc_lrc'])
def test_cncp_random_equals_cncp_in_weights_and_effective_params(cell_key, size):
    units = None if size is None else scaled_lamina_units(size)

    def build(rewire):
        tf.keras.utils.set_random_seed(0)
        cell = CorticalColumnCell(cell_key, seed=3, rewire=rewire,
                                  lamina_units=units, **cell_kwargs(cell_key))
        model = tf.keras.Sequential([tf.keras.layers.RNN(
            cell, return_sequences=True)])
        out = model(tf.random.stateless_normal([2, 5, 4], seed=(1, 2)))
        return model, out
    (plain, out_p), (rand, out_r) = build(False), build(True)
    assert effective_param_count(rand) == effective_param_count(plain)
    for a, b in zip(plain.get_weights(), rand.get_weights()):
        np.testing.assert_array_equal(a, b)
    assert not np.allclose(out_p.numpy(), out_r.numpy())


def test_cncp_random_is_buildable_where_the_two_target_widths_agree():
    """Over the whole lamina budget scan: cncp_random exists exactly where L4
    and L23 are equally wide (the other sizes raise), with one admissible swap
    above lamina size 5 (sizes 2, 4 and 5 have a second pair)."""
    blocks, kinds, edges, _, rank = _group_graph()
    for size in lamina_sizes():
        units = scaled_lamina_units(size)
        pairs = admissible_group_swaps(edges, kinds, {**units, 'in': None}, rank)
        assert (len(pairs) >= 1) == (units['L4'] == units['L23']), size
        assert len(pairs) <= 1 or size <= 5, size
        if units['L4'] == units['L23']:
            assert CorticalColumnCell(
                'gru', rewire=True, lamina_units=units).rewire_overlap > 0.8
        else:
            with pytest.raises(ValueError, match="no degree-"):
                CorticalColumnCell('gru', rewire=True, lamina_units=units)


def _l4_l23_after_one_step(cell, node):
    """New (L4, L23) states from zero states, except ``node`` set to one."""
    states = cell.get_initial_state(batch_size=1)
    if node is not None:
        k = NODE_ORDER.index(node)
        states[k] = tf.ones_like(states[k])
    _, new = cell(tf.ones([1, 2]), states)
    return new[0].numpy(), new[1].numpy()


def test_cncp_random_reads_the_previous_state_of_the_new_source():
    """M_L6CC_L23 and M_Thal_L4 swap targets: the previous L6CC state drives
    L23 in cncp but L4 in cncp_random, the previous Thal state L4 in cncp but
    L23 in cncp_random."""
    for rewire, target in ((False, {'L6CC': 'L23', 'Thal': 'L4'}),
                           (True, {'L6CC': 'L4', 'Thal': 'L23'})):
        tf.keras.utils.set_random_seed(0)
        cell = CorticalColumnCell('gru', seed=3, rewire=rewire)
        cell.build((None, 2))
        base = _l4_l23_after_one_step(cell, None)
        for node, hit in target.items():
            out = _l4_l23_after_one_step(cell, node)
            l4, l23 = (not np.array_equal(o, b) for o, b in zip(out, base))
            # L23 reads the new L4 state, so it moves in both cases
            assert (l4, l23) == (hit == 'L4', True), (rewire, node)


@pytest.mark.parametrize('flags', [dict(feedforward_only=True),
                                   dict(sensory_route='thalamic'),
                                   dict(divisive_inhibition=True),
                                   dict(combiner='additive')])
def test_cncp_random_is_defined_for_the_default_graph_only(flags):
    with pytest.raises(ValueError, match="default graph"):
        CorticalColumnCell('gru', rewire=True, **flags)


def test_cncp_random_raises_when_no_swap_is_admissible():
    """Unequal target widths (L4 8 / L23 9) and unequal source widths
    (L6CC 5 / Thal 4) leave no shape-preserving swap."""
    with pytest.raises(ValueError, match="no degree-"):
        CorticalColumnCell('gru', rewire=True,
                           lamina_units={'L23': 9, 'L6CC': 5})
