"""RQ4 post-hoc diagnostics (issue #30): Jacobian retention and hidden-state
norms of trained Lotka-Volterra fits, rebuilt from result JSON + weights (#18).

Per result JSON matched by --results, one CSV row:
  * the model is rebuilt (rebuild_from_result) and run as a one-step map: the
    cells of its RNN layers in model order, then the layers after the last RNN
    (motor slice, head) on one time step; the map must reproduce the model's
    teacher-forced predictions, otherwise the script stops;
  * trajectories on the first 4 validation sequences: (a) teacher forced, (b)
    closed loop (inputs from closed_loop_rollout, the prediction is fed back);
  * one-step Jacobians J_t = dS_{t+1}/dS_t, S = every state slot of every RNN
    layer concatenated. Closed loop: J is taken over (S, x), so the feedback
    through the head is included, and the S-block of the product is read;
  * ||dS_{t+D}/dS_t||_2 (spectral norm of J_{t+D-1}...J_t) for D in 1, 4, 16,
    median and max over sequences and anchors t; ||S_t||_2 mean and max over
    sequences and t. dense3: additionally per RNN layer, ncp: per
    inter/command/motor block (blocks of the full Jacobian product);
  * --hessian-top (exploratory, off by default, not a pre-registered outcome):
    top Hessian eigenvalue of the training loss on the first 16 training
    sequences (the runner's default fixed batch), 20 power-iteration HVPs.

Skipped with a message: tbt_cncp* (their location input is a function of the
state outside the graph) and non-LV results (no closed loop).

Usage:
    uv run python experiments/rq4_diagnostics.py \
        --results 'results/01_lotka_volterra/<run>/results/*.json' --out rq4.csv
"""
import argparse
import glob
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_lotka_volterra_benchmark import closed_loop_rollout  # noqa: E402
from src.benchmark.checkpoints import rebuild_from_result  # noqa: E402
from src.tasks.lotka_volterra.datasets import load_lotka_volterra  # noqa: E402
from src.tasks.active_sensing.model import TBT_MODES  # noqa: E402
from src.tasks.person_activity.model import ncp_layer_sizes  # noqa: E402

DELTAS = (1, 4, 16)   # powers of two: the products are built by doubling
N_SEQ = 4
UNSUPPORTED = tuple(TBT_MODES)
HESSIAN_BATCH = 16    # the LV runner's default --batch-size
HVP_STEPS = 20


class OneStep:
    """A chain model (RNN layers fed (previous output, time), then per-step
    layers) as the map (x_t, dt_t, S_t) -> (prediction_t, S_{t+1})."""

    def __init__(self, model):
        self.rnns = [l for l in model.layers
                     if isinstance(l, tf.keras.layers.RNN)]
        self.post = model.layers[model.layers.index(self.rnns[-1]) + 1:]
        self.sizes = [[int(n) for n in np.atleast_1d(r.cell.state_size)]
                      for r in self.rnns]
        self.dim = sum(map(sum, self.sizes))
        self._jac = tf.function(self._jac_eager)

    def step(self, x, dt, s):
        parts = tf.split(s, [n for sz in self.sizes for n in sz], axis=-1)
        h, new = x, []
        for rnn, sz in zip(self.rnns, self.sizes):
            states, parts = parts[:len(sz)], parts[len(sz):]
            h, ns = rnn.cell((h, dt), states)
            new += tf.nest.flatten(ns)
        for layer in self.post:
            h = layer(h)
        return h, tf.concat(new, axis=-1)

    def _jac_eager(self, x, dt, s, closed):
        z = tf.concat([s, x], axis=-1) if closed else s
        with tf.GradientTape() as tape:
            tape.watch(z)
            y, s1 = self.step(z[:, self.dim:] if closed else x, dt,
                              z[:, :self.dim])
            out = tf.concat([s1, y], axis=-1) if closed else s1
        return y, s1, tape.batch_jacobian(out, z)

    def trajectory(self, xs, dts, closed):
        """Inputs (B, T, F) -> predictions (B, T, F), states S_1..S_T
        (B, T, D) and one-step Jacobians (T, B, Dz, Dz). All cells start
        from zero states (as in the RNN layers)."""
        s = tf.zeros([xs.shape[0], self.dim])
        preds, states, jacs = [], [], []
        for t in range(xs.shape[1]):
            y, s, j = self._jac(tf.constant(xs[:, t], tf.float32),
                                tf.constant(dts[:, t], tf.float32),
                                s, closed)
            preds.append(y.numpy())
            states.append(s.numpy())
            jacs.append(j.numpy())
        return np.stack(preds, 1), np.stack(states, 1), np.stack(jacs)


def jacobian_products(jacs):
    """{D: (T-D+1, B, Dz, Dz) products J_{t+D-1}...J_t} for D in DELTAS.
    TF batched matmul/SVD: an order of magnitude faster than numpy here."""
    out, prod, d = {}, tf.constant(jacs), 1
    while True:
        if d in DELTAS:
            out[d] = prod
        if d >= max(DELTAS):
            return out
        prod = tf.matmul(prod[d:], prod[:-d])   # P^{2d}_t = P^d_{t+d} P^d_t
        d *= 2


def state_blocks(record, one):
    """{block: state indices} of the per-layer blocks (dense3, ncp), else {}."""
    if record["wiring"] == "dense3":
        bounds = np.cumsum([0] + [sum(sz) for sz in one.sizes])
        return {f"layer{i}": np.arange(a, b)
                for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:]))}
    if record["wiring"] == "ncp":
        inter, command, motor = ncp_layer_sizes(record["size"])
        units = inter + command + motor        # ncps order [motor|command|inter]
        spans = {"motor": (0, motor), "command": (motor, motor + command),
                 "inter": (motor + command, units)}
        n_slots = len(one.sizes[0])
        return {name: np.concatenate([np.arange(a, b) + k * units
                                      for k in range(n_slots)])
                for name, (a, b) in spans.items()}
    return {}


def _stats(a):
    a = np.asarray(a).ravel()
    return (float(np.median(a)), float(np.max(a))) if a.size else (np.nan,
                                                                    np.nan)


def _spectral(prod, idx):
    block = tf.gather(tf.gather(prod, idx, axis=-2), idx, axis=-1)
    return tf.linalg.svd(block, compute_uv=False)[..., 0].numpy()


def trajectory_columns(prefix, states, jacs, dim, blocks):
    """CSV columns of one trajectory (prefix 'tf' or 'cl')."""
    row = {}
    norms = np.linalg.norm(states, axis=-1)                   # (B, T)
    row[f"{prefix}_hnorm_mean"] = float(norms.mean())
    row[f"{prefix}_hnorm_max"] = float(norms.max())
    for name, idx in blocks.items():
        row[f"{prefix}_hnorm_{name}_mean"] = float(
            np.linalg.norm(states[..., idx], axis=-1).mean())
    full = np.arange(dim)
    for d, prod in jacobian_products(jacs).items():
        row[f"{prefix}_jac{d}_median"], row[f"{prefix}_jac{d}_max"] = _stats(
            _spectral(prod, full))
        for name, idx in blocks.items():
            row[f"{prefix}_jac{d}_{name}_median"] = _stats(
                _spectral(prod, idx))[0]
    return row


def hessian_top(model, inputs, y, steps=HVP_STEPS, seed=0):
    """Largest-magnitude Hessian eigenvalue of the loss on (inputs, y), by
    power iteration with ``steps`` Hessian-vector products."""
    variables = model.trainable_variables

    @tf.function
    def hvp(v):
        with tf.GradientTape() as outer:
            with tf.GradientTape() as inner:
                loss = model.loss(y, model(inputs, training=False))
            g = inner.gradient(loss, variables,
                               unconnected_gradients=tf.UnconnectedGradients.ZERO)
            gv = tf.add_n([tf.reduce_sum(a * b) for a, b in zip(g, v)])
        return outer.gradient(gv, variables,
                              unconnected_gradients=tf.UnconnectedGradients.ZERO)

    rng = tf.random.Generator.from_seed(seed)
    v = [rng.normal(w.shape) for w in variables]
    eig = np.nan
    for _ in range(steps):
        norm = tf.linalg.global_norm(v)
        v = [a / norm for a in v]
        hv = hvp(v)
        eig = float(tf.add_n([tf.reduce_sum(a * b) for a, b in zip(v, hv)]))
        v = hv
    return eig


def _traced(model):
    """model as one graph with a free time axis: closed_loop_rollout calls it
    on T growing histories, which runs ~6x slower eagerly."""
    call = tf.function(lambda inputs: model(inputs, training=False),
                       input_signature=[tuple(tf.TensorSpec(i.shape, i.dtype)
                                              for i in model.inputs)])
    return lambda inputs, training=False: call(tuple(inputs))


def diagnose(path, hessian=False):
    """One CSV row for the result JSON at ``path``, or None if unsupported."""
    with open(path) as f:
        record = json.load(f)
    eff = record.get("config", {}).get("effective", {})
    if eff.get("task") != "lotka_volterra" or record["wiring"] in UNSUPPORTED:
        print(f"skip {path}: task {eff.get('task')!r} / wiring "
              f"{record.get('wiring')!r} not supported")
        return None
    model = rebuild_from_result(path)
    data = load_lotka_volterra(n_trajectories=eff["n_trajectories"],
                               seq_len=eff["seq_len"], split=tuple(eff["split"]),
                               seed=eff["data_seed"], system=eff["system"])
    xs, dts = data.val_x[:N_SEQ], data.val_t[:N_SEQ]
    one = OneStep(model)

    preds, tf_states, tf_jacs = one.trajectory(xs, dts, closed=False)
    np.testing.assert_allclose(preds, model.predict((xs, dts), verbose=0),
                               rtol=1e-3, atol=1e-4,
                               err_msg="one-step map != model (not a chain?)")
    gen = closed_loop_rollout(_traced(model), SimpleNamespace(
        seq_len=data.seq_len, val_x=xs, val_t=dts), split="val")
    preds, cl_states, cl_jacs = one.trajectory(gen[:, :-1], dts, closed=True)
    np.testing.assert_allclose(preds, gen[:, 1:], rtol=1e-3, atol=1e-4,
                               err_msg="closed-loop one-step map != rollout")

    blocks = state_blocks(record, one)
    row = {"file": os.path.basename(path),
           **{k: record.get(k) for k in ("system", "wiring", "cell", "seed",
                                         "size", "params_effective")},
           "n_seq": len(xs), "seq_len": data.seq_len, "state_dim": one.dim,
           **trajectory_columns("tf", tf_states, tf_jacs, one.dim, blocks),
           **trajectory_columns("cl", cl_states, cl_jacs, one.dim, blocks)}
    if hessian:
        b = HESSIAN_BATCH
        row["hessian_top"] = hessian_top(
            model, (tf.constant(data.train_x[:b]), tf.constant(data.train_t[:b])),
            tf.constant(data.train_y[:b]))
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", required=True,
                    help="glob of LV result JSONs (quote it), weights next to them")
    ap.add_argument("--out", default="rq4_diagnostics.csv")
    ap.add_argument("--hessian-top", action="store_true",
                    help="also the top Hessian eigenvalue (20 HVPs, exploratory)")
    args = ap.parse_args(argv)
    paths = sorted(glob.glob(args.results))
    rows = [r for r in (diagnose(p, args.hessian_top) for p in paths) if r]
    if not rows:
        raise SystemExit(f"no supported result JSON matches {args.results!r}")
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {len(rows)} rows ({len(paths) - len(rows)} skipped) to "
          f"{args.out}")


if __name__ == "__main__":
    main()
