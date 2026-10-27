"""RQ4 post-hoc diagnostics (issues #30, #65): Jacobian retention and
hidden-state norms of trained Lotka-Volterra fits, rebuilt from result JSON +
weights (#18).

Per result JSON matched by --results:
  * the model is rebuilt (rebuild_from_result) and wrapped in its
    ForcedRolloutCell view (forced_rollout_model, #25), the object mixed
    forcing trains; the wrapper cell, fed the time input through the model's
    own dt_scale layer, is run as a one-step map (y_t, dt_t, Z_t) ->
    (yhat_t, Z_{t+1}), Z = the core cell's state slots S followed by the
    fed-back prediction yhat_{t-1}. Protocol freeze F1-2026-10-26 (section 4,
    RQ4) fixes the chain on this wrapper at alpha = 0;
  * trajectories on the first 4 validation sequences, both from the zero
    state: 'cl' = the wrapper at alpha = 0 (free run from y_0, the RQ4
    chain), 'tf' = the wrapper at alpha = 1 (teacher forced, the plain
    model). The map must reproduce the plain model (alpha = 1) and the
    wrapper view (alpha = 0), otherwise the script stops;
  * one-step Jacobians J_t = dZ_{t+1}/dZ_t by autodiff through the wrapper
    (the feedback through the readout is part of Z); the S-block of the
    products is read: ||dS_{t+D}/dS_t||_2 (spectral norm of the S-block of
    J_{t+D-1}...J_t) for D in 1, 4, 16, median and max over sequences and
    anchors t; ||S_t||_2 mean and max over sequences and t. dense3:
    additionally per RNN layer, ncp: per inter/command/motor block. The
    stateless mlp control (#60) never reads S, so its retention is 0;
  * --hessian-top (exploratory, off by default, not a pre-registered outcome):
    top Hessian eigenvalue of the training loss on the first 16 training
    sequences (the runner's default fixed batch), 20 power-iteration HVPs.

Output: the result JSON gets the key rq4_diagnostics (the values above plus
config_effective_sha256, the hash of the config.effective they were computed
from) and is written into --outdir (default: in place) through the eval-only
write path of run_lotka_volterra_benchmark.py; --out also collects one CSV
row per fit.

Skipped with a message: wirings without a ForcedRolloutCell view (tbt_cncp*,
cncp2_multi) and non-LV results (no closed loop).

Usage:
    uv run python experiments/rq4_diagnostics.py \
        --results 'results/01_lotka_volterra/<run>/results/*.json' --out rq4.csv
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_lotka_volterra_benchmark import (  # noqa: E402
    load_effective_data, write_result)
from src.benchmark.checkpoints import config_hash, rebuild_from_result  # noqa: E402
from src.neurons.forced_rollout_cell import StackedCell  # noqa: E402
from src.tasks.lotka_volterra.model import (  # noqa: E402
    FORCEABLE_WIRINGS, forced_rollout_model)
from src.tasks.person_activity.model import ncp_layer_sizes  # noqa: E402

DELTAS = (1, 4, 16)   # powers of two: the products are built by doubling
N_SEQ = 4
HESSIAN_BATCH = 16    # the LV runner's default --batch-size
HVP_STEPS = 20


class OneStep:
    """The ForcedRolloutCell view of a forceable model (forced_rollout_model)
    as the map (y_t, dt_t, Z_t) -> (yhat_t, Z_{t+1}); Z = core state slots S
    (dim) then yhat_{t-1}. dt_t is the raw time input: it passes the model's
    dt_scale layer, as in the view. The wrapper's 'started' flag (0 before the
    first step, constant 1 after it) is an input, not part of Z."""

    def __init__(self, model):
        self.alpha = tf.Variable(0.0, trainable=False,
                                 dtype=tf.keras.backend.floatx())
        self.view = forced_rollout_model(model, self.alpha)
        rnn = next(l for l in self.view.layers
                   if isinstance(l, tf.keras.layers.RNN))
        self.cell = rnn.cell
        scale = [l for l in self.view.layers if l.name == "dt_scale"]
        self.scale = scale[0] if scale else tf.identity
        core = self.cell.core
        self.sizes = [[int(n) for n in np.atleast_1d(c.state_size)]
                      for c in (core.cells if isinstance(core, StackedCell)
                                else [core])]
        self.dim = sum(map(sum, self.sizes))
        self._slots = [n for sz in self.sizes for n in sz] + [
            self.cell.output_size]
        self._jac = tf.function(self._jac_eager)

    def step(self, y, dt, z, started):
        states = tf.split(z, self._slots, axis=-1) + [started]
        y_hat, new = self.cell((y, self.scale(dt)), states)
        return y_hat, tf.concat(new[:-1], axis=-1)

    def _jac_eager(self, y, dt, z, started):
        with tf.GradientTape() as tape:
            tape.watch(z)
            y_hat, z1 = self.step(y, dt, z, started)
        return y_hat, z1, tape.batch_jacobian(z1, z)

    def trajectory(self, xs, dts, alpha):
        """Inputs (B, T, F) at forcing ``alpha`` -> predictions (B, T, F),
        core states S_1..S_T (B, T, dim) and one-step Jacobians (T, B, Dz,
        Dz) over Z. Starts from the zero state, as the RNN layers do."""
        self.alpha.assign(alpha)
        dtype = tf.keras.backend.floatx()
        z = tf.zeros([xs.shape[0], sum(self._slots)], dtype)
        started = tf.zeros([xs.shape[0], 1], dtype)
        preds, states, jacs = [], [], []
        for t in range(xs.shape[1]):
            y, z, j = self._jac(tf.constant(xs[:, t], dtype),
                                tf.constant(dts[:, t], dtype), z, started)
            started = tf.ones_like(started)
            preds.append(y.numpy())
            states.append(z.numpy()[:, :self.dim])
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


def diagnose(path, hessian=False):
    """The rq4_diagnostics dict of the result JSON at ``path``, or None if
    unsupported."""
    with open(path) as f:
        record = json.load(f)
    eff = record.get("config", {}).get("effective", {})
    if (eff.get("task") != "lotka_volterra"
            or record["wiring"] not in FORCEABLE_WIRINGS):
        print(f"skip {path}: task {eff.get('task')!r} / wiring "
              f"{record.get('wiring')!r} not supported")
        return None
    model = rebuild_from_result(path)
    data = load_effective_data(eff)
    xs, dts = data.val_x[:N_SEQ], data.val_t[:N_SEQ]
    one = OneStep(model)

    preds, tf_states, tf_jacs = one.trajectory(xs, dts, alpha=1.0)
    np.testing.assert_allclose(preds, model.predict((xs, dts), verbose=0),
                               rtol=1e-3, atol=1e-4,
                               err_msg="one-step map at alpha 1 != model")
    preds, cl_states, cl_jacs = one.trajectory(xs, dts, alpha=0.0)
    np.testing.assert_allclose(preds, one.view.predict((xs, dts), verbose=0),
                               rtol=1e-3, atol=1e-4,
                               err_msg="one-step map at alpha 0 != wrapper")

    blocks = state_blocks(record, one)
    diag = {"config_effective_sha256": config_hash(eff),
            "n_seq": len(xs), "seq_len": data.seq_len, "state_dim": one.dim,
            **trajectory_columns("tf", tf_states, tf_jacs, one.dim, blocks),
            **trajectory_columns("cl", cl_states, cl_jacs, one.dim, blocks)}
    if hessian:
        b = HESSIAN_BATCH
        diag["hessian_top"] = hessian_top(
            model, (tf.constant(data.train_x[:b]), tf.constant(data.train_t[:b])),
            tf.constant(data.train_y[:b]))
    return diag


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", required=True,
                    help="glob of LV result JSONs (quote it), weights next to them")
    ap.add_argument("--outdir", default=None,
                    help="write the result JSONs with rq4_diagnostics here; "
                         "default: in place")
    ap.add_argument("--out", default="rq4_diagnostics.csv",
                    help="CSV with one row per diagnosed fit")
    ap.add_argument("--hessian-top", action="store_true",
                    help="also the top Hessian eigenvalue (20 HVPs, exploratory)")
    args = ap.parse_args(argv)
    paths = sorted(glob.glob(args.results))
    rows = []
    for path in paths:
        diag = diagnose(path, args.hessian_top)
        if diag is None:
            continue
        with open(path) as f:
            record = json.load(f)
        record["rq4_diagnostics"] = diag
        print(f"rq4 {path} -> {write_result(record, path, args.outdir)}")
        rows.append({"file": os.path.basename(path),
                     **{k: record.get(k) for k in (
                         "system", "wiring", "cell", "seed", "size",
                         "params_effective")}, **diag})
    if not rows:
        raise SystemExit(f"no supported result JSON matches {args.results!r}")
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {len(rows)} rows ({len(paths) - len(rows)} skipped) to "
          f"{args.out}")


if __name__ == "__main__":
    main()
