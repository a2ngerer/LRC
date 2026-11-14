"""Per-lamina state diagnostics, hop gains and path lengths of saved
rollout fits (issue #136, eval-only).

Per result JSON in --results (periodic_predator_prey and duffing fits of
cncp, cncp_random, cncp_ff, ncp, ncp_random and dense3) the model is rebuilt
twice, with its trained weights and as rebuild_from_result(untrained=True)
(the same-seed initial draw), and run as the one-step map of
rq4_diagnostics.OneStep on the teacher-forced (alpha = 1) trajectories of all
validation windows from the zero state. --out gets one row per (fit,
weights in {trained, untrained}):

  * blocks: the cNCP laminae, one block per node of NODE_ORDER cut at the
    cumulative offsets of the cell's state sizes (one.sizes[0]); cncp_ff
    keeps its four built nodes only (rq4_diagnostics.active_state). NCP:
    inter / command / motor, dense3: layer0..2 (rq4_diagnostics.state_blocks);
  * full-state and per-block diagnostics: hnorm_mean (||S_t||_2 on the
    active indices; cncp_ff without its four pass-through slots),
    hnorm_<block>_mean, jac1_median (spectral norm of the active S-block of
    the one-step Jacobian J_t = dZ_{t+1}/dZ_t) and jac1_<block>_median (the
    block's diagonal, i.e. its retention); medians over sequences and steps;
  * hop gains along the same-step chain, HOPS (target <- source), in two
    definitions:
      (a) gain_<target>_from_<source>: per-unit std over the validation
          windows of the state at the last step, averaged over the node's
          units, target / source (the quantity of calibrate_lrc_init.py);
      (b) jac1_<target>_from_<source>: spectral norm of the block
          [target, source] of J_t (D = 1), median over sequences and steps.
          A feed-forward hop is same-step, so the block contains the
          retention of the source node (target_{t+1} reads source_{t+1},
          which carries source_t). Blocks are never multiplied; the
          end-to-end entry is the single block [readout node, first node]
          (CHAIN).

--paths gets the path lengths input -> readout node (cNCP: L5ET, NCP:
motor, dense3: layer2) on the node-group level, one row per fit: sp_same /
lp_same, the shortest / longest path over same-step edges, and sp_all, the
shortest path including the delayed edges (edges: group_edges and the built
cell's delayed-edge targets for the cNCP, the built NCP masks with
same-step = forward in the pass order inter < command < motor, three
stacked layers for dense3).

Writes the two CSVs only, never into a source folder or record;
experiments/rq4_diagnostics.py is unchanged. Skipped with a message: other
tasks, systems and wirings.

Usage:
    uv run python experiments/eval_cncp_gains.py \
        --results 'results/<run>/**/*.json' --out gains.csv --paths paths.csv
"""
import argparse
import functools
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.eval_robustness import SYSTEMS  # noqa: E402
from experiments.rq4_diagnostics import (OneStep, active_state,  # noqa: E402
                                         state_blocks)
from experiments.run_lotka_volterra_benchmark import (  # noqa: E402
    load_effective_data)
from src.benchmark.checkpoints import config_hash, rebuild_from_result  # noqa: E402
from src.evaluation.gradient_flow import ncp_state_slices  # noqa: E402
from src.tasks.person_activity.model import (CNCP_WIRINGS,  # noqa: E402
                                             NCP_WIRINGS, ncp_layer_sizes)
from src.wirings.cncp import NODE_ORDER, group_edges  # noqa: E402
from src.wirings.ncp import NCPLayeredCell  # noqa: E402

_CNCP_HOPS = (("L23", "L4"), ("L5IT", "L23"), ("L5ET", "L23"),
              ("L5ET", "L5IT"))
_NCP_HOPS = (("command", "inter"), ("motor", "command"))
HOPS = {**dict.fromkeys(CNCP_WIRINGS, _CNCP_HOPS),
        **dict.fromkeys(NCP_WIRINGS, _NCP_HOPS),
        "dense3": (("layer1", "layer0"), ("layer2", "layer1"))}
# (first node, readout node) of each wiring's chain
CHAIN = {**dict.fromkeys(CNCP_WIRINGS, ("L4", "L5ET")),
         **dict.fromkeys(NCP_WIRINGS, ("inter", "motor")),
         "dense3": ("layer0", "layer2")}
NCP_ORDER = ("inter", "command", "motor")       # pass order of NCPLayeredCell


def laminar_blocks(record, one):
    """{node: S indices} of the cNCP laminae in NODE_ORDER, cut at the
    cumulative offsets of one.sizes[0]; cncp_ff keeps the four built nodes
    (active_state)."""
    sizes = one.sizes[0]
    if len(one.sizes) != 1 or len(sizes) != len(NODE_ORDER):
        raise ValueError(f"not a cNCP state: sizes {one.sizes}")
    bounds = np.cumsum([0, *sizes])
    active = set(active_state(record, one).tolist())
    return {node: np.arange(a, b) for node, a, b
            in zip(NODE_ORDER, bounds[:-1], bounds[1:]) if a in active}


def blocks(record, one):
    """The state blocks of the wiring: laminae (cNCP) or layers (NCP,
    dense3)."""
    if record["wiring"] in CNCP_WIRINGS:
        return laminar_blocks(record, one)
    return state_blocks(record, one)


def block_norms(jacs, rows, cols):
    """Spectral norm of the block J[rows, cols] (rows = target, cols =
    source) of every Jacobian in ``jacs`` (..., Dz, Dz)."""
    block = tf.gather(tf.gather(jacs, rows, axis=-2), cols, axis=-1)
    return tf.linalg.svd(block, compute_uv=False)[..., 0].numpy()


def std_gain(states, target, source):
    """Definition (a): mean per-unit std over the sequences of the state at
    the last step, target block / source block."""
    sd = states[:, -1].std(axis=0)
    with np.errstate(all="ignore"):
        return float(sd[target].mean() / sd[source].mean())


def diagnostics(record, model, data):
    """The gain columns of one rebuilt ``model`` (trained or untrained)."""
    xs, dts = data.val_x, data.val_t
    one = OneStep(model)
    preds, states, jacs = one.trajectory(xs, dts, alpha=1.0)
    np.testing.assert_allclose(preds, model.predict((xs, dts), verbose=0),
                               rtol=1e-3, atol=1e-4,
                               err_msg="one-step map at alpha 1 != model")
    blk, full = blocks(record, one), active_state(record, one)
    jacs = tf.constant(jacs)
    row = {"n_seq": len(xs), "seq_len": xs.shape[1], "state_dim": len(full),
           "hnorm_mean": float(np.linalg.norm(states[..., full],
                                              axis=-1).mean()),
           "jac1_median": float(np.median(block_norms(jacs, full, full)))}
    for name, idx in blk.items():
        row[f"hnorm_{name}_mean"] = float(
            np.linalg.norm(states[..., idx], axis=-1).mean())
        row[f"jac1_{name}_median"] = float(np.median(
            block_norms(jacs, idx, idx)))
    first, readout = CHAIN[record["wiring"]]
    for t, s in HOPS[record["wiring"]]:
        row[f"gain_{t}_from_{s}"] = std_gain(states, blk[t], blk[s])
    for t, s in (*HOPS[record["wiring"]], (readout, first)):
        row[f"jac1_{t}_from_{s}"] = float(np.median(
            block_norms(jacs, blk[t], blk[s])))
    return row


# --- path lengths ---------------------------------------------------------- #

def cncp_edges(cell):
    """[(source group, target group, delay 'S'|'D')] of a built
    CorticalColumnCell: the group edges of its instantiated edge classes, the
    delayed basal ones at the cell's (possibly rewired) endpoints."""
    if cell._thalamic:
        raise ValueError("path lengths are defined for the direct route")
    moved = {name: (src, dst) for name, src, dst in cell._delayed}
    blocks_, kinds = group_edges(cell._sign_constraint)
    return [(*moved.get(name, (src, dst)), delay)
            for (name, src, dst), (delay, _, _) in zip(blocks_, kinds)
            if name in cell.edges]


def ncp_edges(sensory, adjacency, spans):
    """Group edges of an NCP from its masks (``spans``: layer -> (a, b) in
    the state order): input -> layer where the sensory block is non-zero,
    layer -> layer where the adjacency block is; same-step if forward in
    NCP_ORDER, else delayed (intra-layer recurrence included)."""
    edges = [("in", layer, "S") for layer, (a, b) in spans.items()
             if sensory[:, a:b].any()]
    for src, (a, b) in spans.items():
        for dst, (c, d) in spans.items():
            if adjacency[a:b, c:d].any():
                forward = NCP_ORDER.index(src) < NCP_ORDER.index(dst)
                edges.append((src, dst, "S" if forward else "D"))
    return edges


def dense3_edges():
    edges = [("in", "layer0", "S"), ("layer0", "layer1", "S"),
             ("layer1", "layer2", "S")]
    return edges + [(f"layer{i}", f"layer{i}", "D") for i in range(3)]


def path_lengths(edges, source, target):
    """(shortest, longest) path over the same-step edges and the shortest
    path over all edges from ``source`` to ``target``, in edges (None if
    unreachable). The same-step edges form a DAG (pass order)."""
    def adj(delays):
        out = {}
        for s, t, d in edges:
            if d in delays and s != t:
                out.setdefault(s, []).append(t)
        return out

    def shortest(a):
        dist, todo = {source: 0}, [source]
        for node in todo:                   # BFS: todo grows while iterated
            for nxt in a.get(node, ()):
                if nxt not in dist:
                    dist[nxt] = dist[node] + 1
                    todo.append(nxt)
        return dist.get(target)

    same = adj("S")

    @functools.cache
    def longest(node):
        if node == target:
            return 0
        tails = [n for n in map(longest, same.get(node, ())) if n is not None]
        return 1 + max(tails) if tails else None

    return shortest(same), longest(source), shortest(adj("SD"))


def model_edges(record, model):
    """(edges, readout node) of the rebuilt model's wiring."""
    rnn = next(l for l in model.layers if isinstance(l, tf.keras.layers.RNN))
    cell, wiring = rnn.cell, record["wiring"]
    if wiring in CNCP_WIRINGS:
        return cncp_edges(cell), "L5ET"
    if wiring == "dense3":
        return dense3_edges(), "layer2"
    if isinstance(cell, NCPLayeredCell):       # closed-form / discrete cells
        d = int(model.inputs[0].shape[-1])
        sens = cell._wiring.sensory_mask(d)
        adj = cell._wiring.adjacency_mask(d)
    else:                                      # one ODE cell, masks built
        sens, adj = cell.sensory_mask.numpy(), cell.sparsity_mask.numpy()
    spans = ncp_state_slices(*ncp_layer_sizes(record["size"]))
    return ncp_edges(sens, adj, spans), "motor"


# --- driver ---------------------------------------------------------------- #

def evaluate(path):
    """(gain rows, path row) of the result JSON at ``path``, or ([], None)
    if unsupported."""
    with open(path) as f:
        record = json.load(f)
    eff = record.get("config", {}).get("effective", {}) \
        if isinstance(record, dict) else {}
    if (eff.get("task") != "lotka_volterra"
            or record["system"] not in SYSTEMS
            or record["wiring"] not in HOPS):
        print(f"skip {path}: no hop gains for this record")
        return [], None
    data = load_effective_data(eff)
    base = {"file": path, "system": record["system"],
            "wiring": record["wiring"], "cell": record["cell"],
            "seed": record["seed"],
            "config_effective_sha256": config_hash(eff),
            "protocol_version": eff.get("protocol_version")}
    rows = []
    for weights, untrained in (("trained", False), ("untrained", True)):
        model = rebuild_from_result(path, untrained=untrained)
        rows.append({**base, "weights": weights,
                     **diagnostics(record, model, data)})
    edges, readout = model_edges(record, model)
    sp, lp, sp_all = path_lengths(edges, "in", readout)
    print(f"gains {path}: {len(rows)} rows, paths {sp}/{lp}/{sp_all}")
    return rows, {**base, "readout": readout, "sp_same": sp, "lp_same": lp,
                  "sp_all": sp_all}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", required=True, nargs="+",
                    help="LV result JSONs or globs (quote them; ** recurses), "
                         "weights next to them")
    ap.add_argument("--out", default="cncp_gains.csv",
                    help="CSV with one row per (fit, trained|untrained)")
    ap.add_argument("--paths", default="cncp_paths.csv",
                    help="CSV with the path lengths, one row per fit")
    args = ap.parse_args(argv)
    paths = sorted({p for g in args.results
                    for p in glob.glob(g, recursive=True)})
    sources = {os.path.dirname(os.path.abspath(p)) for p in paths}
    for out in (args.out, args.paths):
        if os.path.dirname(os.path.abspath(out)) in sources:
            raise SystemExit(f"{out} lies in a source folder; the gains "
                             "script never writes next to the records")
    rows, path_rows = [], []
    for p in paths:
        r, pr = evaluate(p)
        rows += r
        path_rows += [pr] if pr else []
    if not rows:
        raise SystemExit(f"no supported result JSON matches {args.results!r}")
    pd.DataFrame(rows).to_csv(args.out, index=False)
    pd.DataFrame(path_rows).to_csv(args.paths, index=False)
    print(f"wrote {len(rows)} rows to {args.out} and {len(path_rows)} to "
          f"{args.paths}")


if __name__ == "__main__":
    main()
