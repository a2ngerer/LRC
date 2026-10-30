"""Degree-preserving random rewiring of the NCP and cNCP graphs (issue #17).

Null models for the question "does the role structure of the wiring matter, or
only its sparsity and state dimension?". Both are Maslov-Sneppen edge-swap
chains: two edges (a, b) and (c, d) trade targets, giving (a, d) and (c, b),
which keeps the in- and out-degree of every node. A proposal is skipped when it
would create a self-loop or a parallel edge, or when the graph-specific rule
of the caller rejects it.

* ``rewire_degree_preserving``: neuron level (NCP). Edges stay forward in pass
  order or intra-layer.
* ``rewire_group_edges``: node-group level (cNCP). Edges keep their kind (delay
  type, role, sign) and their mask shape.
"""
import numpy as np

# n_swaps = SWAPS_PER_EDGE x number of edges (issue #17).
SWAPS_PER_EDGE = 10
_MAX_ROUNDS = 1000


def _propose(edges, present, ok, i, j):
    """New endpoints of edge slots i and j if trading targets is admissible.

    ``ok(i, j)`` is the caller's rule on the current edges: False to reject,
    'dst' to give slot i the target of slot j and vice versa, 'src' to give the
    slots the sources instead. Both modes yield the edge set {(a, d), (c, b)};
    they differ in which slot (and therefore which weight mask) lands where.
    """
    (a, b), (c, d) = edges[i], edges[j]
    if a == d or c == b or (a, d) in present or (c, b) in present:
        return None
    mode = ok(i, j)
    if mode == 'dst':
        return (a, d), (c, b)
    if mode == 'src':
        return (c, b), (a, d)
    return None


def _swap_chain(edges, ok, rng, n_swaps):
    """``n_swaps`` swap proposals on ``edges`` (list of (src, dst), in place)."""
    if len(edges) < 2:
        return
    present = set(edges)
    for i, j in rng.integers(0, len(edges), size=(n_swaps, 2)).tolist():
        new = _propose(edges, present, ok, i, j)
        if new is not None:
            present -= {edges[i], edges[j]}
            present |= set(new)
            edges[i], edges[j] = new


def rewire_degree_preserving(adjacency, layer_of, seed, n_swaps):
    """Rewire a layered digraph keeping every node's in- and out-degree.

    Args:
        adjacency: (n, n) bool/0-1 array, ``adjacency[s, t]`` = edge s -> t.
                   Source-only nodes (sensory inputs) are rows with an empty
                   column.
        layer_of:  (n,) pass-order layer of every node. Edges must satisfy
                   ``layer_of[s] <= layer_of[t]`` (forward in pass order or
                   intra-layer), and so does every edge of the result.
        seed:      seed of the swap chain.
        n_swaps:   number of swap proposals (issue #17: 10 x edges).

    Existing self-loops are kept as they are and take no part in the swaps.

    Returns:
        ``(rewired, overlap)``: the new bool adjacency and the fraction of the
        original edges that are still present (1.0 = unchanged).
    """
    adj = np.asarray(adjacency) != 0
    layer = np.asarray(layer_of)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1] or layer.shape != (len(adj),):
        raise ValueError("adjacency must be (n, n) and layer_of (n,)")
    src, dst = np.nonzero(adj)
    if np.any(layer[src] > layer[dst]):
        raise ValueError("adjacency has an edge backwards in pass order")
    loop = src == dst
    edges = list(zip(src[~loop].tolist(), dst[~loop].tolist()))

    def ok(i, j):
        return ('dst' if layer[edges[i][0]] <= layer[edges[j][1]]
                and layer[edges[j][0]] <= layer[edges[i][1]] else False)

    _swap_chain(edges, ok, np.random.default_rng(seed), n_swaps)
    out = np.zeros_like(adj)
    out[src[loop], dst[loop]] = True
    if edges:
        out[tuple(np.array(edges).T)] = True
    return out, float((out & adj).sum() / max(adj.sum(), 1))


def _group_rule(edges, kinds, width, rank):
    """The swap rule of ``rewire_group_edges`` as an ``ok(i, j)`` for _propose."""
    def ok(i, j):
        (a, b), (c, d) = edges[i], edges[j]
        if kinds[i] != kinds[j]:
            return False
        if kinds[i][0] == 'S' and not (rank[a] < rank[d] and rank[c] < rank[b]):
            return False
        if width[b] == width[d]:
            return 'dst'
        return 'src' if width[a] == width[c] else False
    return ok


def admissible_group_swaps(edges, kinds, width, rank):
    """Slot pairs (i, j), j < i, whose targets ``rewire_group_edges`` may trade
    in the given graph."""
    edges = [tuple(e) for e in edges]
    ok, present = _group_rule(edges, kinds, width, rank), set(edges)
    return [(i, j) for i in range(len(edges)) for j in range(i)
            if _propose(edges, present, ok, i, j) is not None]


def rewire_group_edges(edges, kinds, width, rank, seed, n_swaps):
    """Rewire the node-group graph of a composite cell (cNCP).

    Args:
        edges:  list of (src group, dst group), one entry per edge slot.
        kinds:  per slot a hashable kind (delay type, role, sign). Only slots
                of the same kind trade targets, so the kind of every slot is
                kept.
        width:  group -> unit count. A swap needs equal target widths or equal
                source widths, so every slot keeps its mask shape and the
                effective parameter count is unchanged.
        rank:   group -> position in the pass. A slot of delay type 'S'
                (kind[0], reads the current step) stays forward in rank; 'D'
                slots read the previous state and are unconstrained.
        seed, n_swaps: as in ``rewire_degree_preserving``.

    The result is deliberately forced away from the input: this graph space can
    be as small as two states, so a chain that ends on the original is
    extended until it differs.

    Returns:
        ``(new_edges, overlap)``: the endpoints of every slot after rewiring
        and the fraction of the original group edges still present. Overlap 1.0
        means no swap is admissible.
    """
    edges = [tuple(e) for e in edges]
    start = list(edges)
    if not admissible_group_swaps(edges, kinds, width, rank):
        return start, 1.0
    ok, rng = _group_rule(edges, kinds, width, rank), np.random.default_rng(seed)
    for _ in range(_MAX_ROUNDS):
        _swap_chain(edges, ok, rng, n_swaps)
        if set(edges) != set(start):
            break
    return edges, len(set(edges) & set(start)) / len(start)
