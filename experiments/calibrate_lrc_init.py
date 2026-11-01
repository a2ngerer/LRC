"""Layer-wise unit-variance (LSUV) calibration of the lrc_cal init factors (issue #86).

The registry keys ``lrc_cal`` (Person Activity), ``lrc_cal_lv`` (Lotka-Volterra)
and ``lrc_cal_duffing`` carry an ``init_slice_scale``: one factor per NCP layer
that multiplies ``vleak``, ``h`` and ``sensory_h`` of that layer at init. The
factors are the square roots of an LSUV calibration of the fresh ``lrc``
network on one batch of the task, stored to one decimal. This script runs that
calibration and prints the values to paste:

    uv run python experiments/calibrate_lrc_init.py --task pa --seed 0
    uv run python experiments/calibrate_lrc_init.py --task lv --seed 0
    uv run python experiments/calibrate_lrc_init.py --task lv --system duffing --seed 0

CPU only, forward passes only: no training. The model is built as the runners
build ``lrc`` (asymmetric elastance, ode_unfolds 6, dt_scale auto, self
recurrence) at the size the parameter budget of the suite picks. The batch is
the first ``--batch-size`` train windows (Person Activity, grouped split seed
0) or train trajectories (LV / Duffing, data seed DATA_SEED_BASE + seed); the
time input is held at the batch median.

Calibration, in the order inter, command, motor: the std over the batch of the
state at the last step, averaged over the layer's units, is compared with a
reference (inter: 0.1 x the input spread at the last step; command: the inter
std; motor: the command std). Outside the accepted band, ``vleak`` and the
layer's synaptic columns (``sensory_h`` for inter, ``h`` for command and motor)
are scaled by the square root of target / ratio (at most 20 per round, at most
10 rounds). A round that pushes mean |tanh(g)| of the layer above 0.9 is undone
and ends that layer.
"""
import argparse

import numpy as np
import tensorflow as tf

from src.benchmark.registry import cell_kwargs as registry_cell_kwargs
from src.evaluation.gradient_flow import ncp_state_slices
from src.tasks.lotka_volterra.datasets import DATA_SEED_BASE, load_lotka_volterra
from src.tasks.lotka_volterra.model import build_lotka_volterra_model
from src.tasks.person_activity.datasets import load_person_activity
from src.tasks.person_activity.model import (
    DEFAULT_ODE_UNFOLDS, budget_sizes, build_person_activity_model,
    dt_scale_arg, ncp_layer_sizes, resolve_dt_scale)
from src.wirings import size_for_budget

LAYERS = ("inter", "command", "motor")        # calibration order
TARGET = 0.55                                 # hop ratio command/inter, motor/command
HOP_BAND = (0.3, 1.0)
INTER_BAND = (0.7, 1.4)                       # inter std / (0.1 x input spread)
SAT_MAX = 0.9
MAX_SCALE = 20.0
ROUNDS = 10
# Suite budgets (configs/suite/pa_final.yaml, lv_final.yaml, duffing_final.yaml)
# and the runners' batch sizes.
BUDGET = {"pa": 30000, "lv": 4000}
BATCH = {"pa": 128, "lv": 16}


def _sigmoid(a):
    return 1.0 / (1.0 + np.exp(-a))


def load_task(task, system, seed, dt_scale, batch_size, cell="lrc"):
    """``(build(wiring, size, **cell_kwargs), x, t, dt_scale)`` of one task:
    ``x`` / ``t`` are the first ``batch_size`` train inputs / elapsed times."""
    if task == "pa":
        data = load_person_activity(split="grouped", split_seed=0)
        dt_scale = resolve_dt_scale(dt_scale, np.median(data.train_t))

        def build(wiring, size, **kw):
            return build_person_activity_model(
                wiring, cell, size=size, seed=seed,
                num_classes=data.num_classes, feature_size=data.feature_size,
                seq_len=data.seq_len, dt_feature=False, dt_scale=dt_scale,
                self_recurrent=True, **kw)
    else:
        data = load_lotka_volterra(n_trajectories=80, seq_len=128,
                                   seed=DATA_SEED_BASE + seed, system=system)
        dt_scale = resolve_dt_scale(dt_scale, data.dt)

        def build(wiring, size, **kw):
            return build_lotka_volterra_model(
                wiring, cell, size=size, seed=seed,
                feature_size=data.feature_size, loc_dim=data.loc_dim,
                dt_scale=dt_scale, self_recurrent=True, **kw)
    return (build, data.train_x[:batch_size].astype(np.float32),
            data.train_t[:batch_size].astype(np.float32), dt_scale)


def _abs_tanh_g(cell, state, x_last, slices):
    """Mean |tanh(g)| per layer at the last step, g the total conductance of
    the cell's state equation (the saturation guard of the calibration)."""
    p = {k: v.numpy().astype(np.float64) for k, v in cell._params.items()}
    hm, sm = cell.sparsity_mask.numpy(), cell.sensory_mask.numpy()
    v = state[:, -2].astype(np.float64)
    x = x_last * p["input_w"] + p["input_b"]
    ssyn = _sigmoid(p["sensory_sigma"] * (x[:, :, None] - p["sensory_mu"]))
    syn = _sigmoid(p["sigma"] * (v[:, :, None] - p["mu"]))
    g = (p["gleak"] + (p["h"] * hm * syn).sum(1)
         + (p["sensory_h"] * sm * ssyn).sum(1))
    return {k: float(np.abs(np.tanh(g[:, a:b])).mean())
            for k, (a, b) in slices.items()}


def _scale_cols(var, a, b, f):
    m = np.ones(var.shape[-1], np.float32)
    m[a:b] = f
    var.assign(var * m)


def lsuv(model, x, t, slices, log=print):
    """LSUV calibration of a fresh NCP ``lrc`` model on the batch ``x`` / ``t``
    (time held at its median). The cell is scaled in place. Returns the
    cumulative factor per layer (std units; the stored init_slice_scale value
    is its square root) and the per-round trace."""
    rnn = next(l for l in model.layers if isinstance(l, tf.keras.layers.RNN))
    cell = rnn.cell
    sub = tf.keras.Model(model.inputs, rnn.output)
    fwd = tf.function(lambda a, b: sub([a, b]))
    t_const = np.full_like(t, np.median(t))
    s_in = float(x[:, -1].std(0).mean())
    params = cell._params
    plan = [("inter", params["sensory_h"], lambda s: 0.1 * s_in, 1.0, INTER_BAND),
            ("command", params["h"], lambda s: s["inter"], TARGET, HOP_BAND),
            ("motor", params["h"], lambda s: s["command"], TARGET, HOP_BAND)]
    fac, trace = {}, []
    for layer, var, ref, target, (lo, hi) in plan:
        fac[layer] = last = 1.0
        a, b = slices[layer]

        def scale(f):
            _scale_cols(var, a, b, np.sqrt(f))
            _scale_cols(params["vleak"], a, b, np.sqrt(f))
            fac[layer] *= f

        for rd in range(ROUNDS):
            state = fwd(x, t_const).numpy()
            sd = state[:, -1].std(0)
            std = {k: float(sd[s:e].mean()) for k, (s, e) in slices.items()}
            ratio = std[layer] / ref(std)
            sat = _abs_tanh_g(cell, state, x[:, -1], slices)[layer]
            trace.append(dict(layer=layer, round=rd, ratio=ratio, std=std[layer],
                              abs_tanh_g=sat, factor=fac[layer]))
            log(f"   lsuv {layer:7s} round {rd} ratio {ratio:.3e} "
                f"std {std[layer]:.3e} |tanh g| {sat:.3f} cum {fac[layer]:.2f}")
            if sat > SAT_MAX and rd > 0:               # undo the last step, stop
                scale(1.0 / last)
                trace.append(dict(layer=layer, round=rd, event="saturation_undo"))
                break
            if lo <= ratio <= hi:
                break
            last = float(np.clip(target / ratio, 1 / MAX_SCALE, MAX_SCALE))
            scale(last)
    return fac, trace


def stored_scale(factors):
    """The init_slice_scale value of the registry: sqrt of the LSUV factor,
    one decimal."""
    return {k: round(float(np.sqrt(f)), 1) for k, f in factors.items()}


def calibrate(task, wiring="ncp", system="periodic_predator_prey", seed=0,
              size=None, param_budget=None, ode_unfolds=DEFAULT_ODE_UNFOLDS,
              dt_scale="auto", batch_size=None, cell="lrc", log=print):
    """Calibrate ``lrc`` on ``task`` ('pa' or 'lv'); the width is ``size`` or
    the knob closest to ``param_budget`` (default: the suite budget of the
    task). ``cell`` names another registry arm to start from (a lrc_cal_* arm
    comes out near factor 1 when its stored factors fit the graph). Returns a
    dict with size, factors, stored (init_slice_scale), trace."""
    kw = dict(registry_cell_kwargs(cell), ode_unfolds=ode_unfolds)
    build, x, t, dt_scale = load_task(task, system, seed, dt_scale,
                                      batch_size or BATCH[task], cell)
    if size is None:
        size, _ = size_for_budget(lambda s: build(wiring, s, **kw),
                                  param_budget or BUDGET[task],
                                  budget_sizes(wiring))
    model = build(wiring, size, **kw)
    slices = ncp_state_slices(*ncp_layer_sizes(size))
    factors, trace = lsuv(model, x, t, slices, log)
    return dict(task=task, system=system if task == "lv" else None,
                wiring=wiring, seed=seed, size=size, dt_scale=dt_scale,
                factors=factors, stored=stored_scale(factors), trace=trace)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--task", required=True, choices=("pa", "lv"),
                    help="pa: Person Activity; lv: Lotka-Volterra / Duffing")
    ap.add_argument("--wiring", default="ncp", choices=("ncp",))
    ap.add_argument("--system", default="periodic_predator_prey",
                    help="lv only: periodic_predator_prey (default) or duffing")
    ap.add_argument("--seed", type=int, default=0)
    cap = ap.add_mutually_exclusive_group()
    cap.add_argument("--size", type=int, default=None)
    cap.add_argument("--param-budget", type=int, default=None,
                     help=f"effective parameter budget (default: {BUDGET})")
    ap.add_argument("--ode-unfolds", type=int, default=DEFAULT_ODE_UNFOLDS)
    ap.add_argument("--dt-scale", type=dt_scale_arg, default="auto")
    ap.add_argument("--batch-size", type=int, default=None,
                    help=f"default per task: {BATCH}")
    args = ap.parse_args()
    r = calibrate(args.task, args.wiring, args.system, args.seed, args.size,
                  args.param_budget, args.ode_unfolds, args.dt_scale,
                  args.batch_size)
    where = f"{args.task}" + (f"/{args.system}" if args.task == "lv" else "")
    print(f"\n{where} {args.wiring} seed {args.seed}: size {r['size']}, "
          f"dt_scale {r['dt_scale']:g}")
    for layer in LAYERS:
        print(f"  {layer:7s} LSUV factor {r['factors'][layer]:7.2f}   "
              f"stored (sqrt, 1 decimal) {r['stored'][layer]}")
    print(f"init_slice_scale = {r['stored']}")


if __name__ == "__main__":
    main()
