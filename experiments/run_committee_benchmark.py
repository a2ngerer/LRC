"""Train one (wiring, K, seed) voting committee on person-activity under the
consensus-study corruption protocol (Chapter 5, protocol freeze
F1-2026-10-26, issue #55).

Protocol: every column receives the same window (--view-mode identical);
training batches are corrupted on the fly (CorruptedBatches): per batch m is
drawn uniformly from the K-specific fault grid (fault_counts of --rho-grid),
Gaussian noise of --noise-frac times the training std goes on the
coordinates, and exactly m columns per window lose their coordinates for the
whole window. All draws come from --corruption-seed, so every arm sees the same
m sequence, column choices and batch order. Each epoch is scored on fixed
validation draws for every m of the grid (GridValidation, --val-seed); the
epoch with the best mean validation macro-F1 over the grid is kept. The test
split is scored per m over --test-seeds fixed draws (pooled window macro-F1,
per-recording confusions, per-position macro-F1 curve) plus the correlated
channel fault (one coordinate zeroed in every column).

Headline comparisons (all arms at matched EFFECTIVE parameters; size the arms
with --param-budget or pass a per-arm --size):
  - learned consensus vs the controls at the SAME K and views: --vote-lambda 0
    (columns independent, readout averages), --consensus median, and
    --view-average mean|median (one cell on the coordinate-wise mean or lower
    median of the K views, the input-averaging monolith); K=1 is the single
    monolith on the full input.
  - cncp committee (cortical columns, L2/3 vote) vs ncp committee (command-slice
    vote, motor readout) vs dense committee (hidden-state vote, or a leading
    subspace with --dense-vote-units): is the column type or the consensus the
    source of a gain?

Every fault count also records the macro-F1 per window position t =
1..seq_len (fresh state per window): how fast the committee reaches its
decision (normalised area under the curve, steps_to_90 against the plateau of
the last eight positions).

Exploratory July 2026 path, NOT part of the suite: --view-mode
partition|noisy, the train-time layers --train-drop / --train-noise and the
test-time sweep over --drop-fracs (or --test-corruption noise with
--noise-sigmas), which only runs when its levels are given and writes the
drop_* fields.

--wandb logs the fit to Weights & Biases (src/benchmark/tracking.py track_fit,
#53): K, the rho grid, vote_lambda, consensus, view_mode and the corruption
settings in the run config, the test macro-F1 per fault count in the run
summary.

--deterministic enables tf.config.experimental.enable_op_determinism() before
the model is built, as in the LV and PA runners (#70); the record carries it as
config.effective.deterministic (protocol freeze F1-2026-10-26 l, #76).

--eval-only --weights <stem>.weights.h5 re-scores a stored fit without
training (#81, eval_only): per-seed test confusions and further validation
draws, written to <stem>.evalonly.json; its own flags, see --eval-only --help.

Usage:
    uv run python experiments/run_committee_benchmark.py \
        --wiring ncp --n-columns 30 --param-budget 30000 --seed 0
    uv run python experiments/run_committee_benchmark.py \
        --wiring ncp --n-columns 30 --param-budget 30000 --view-average mean
    uv run python experiments/run_committee_benchmark.py --eval-only \
        --weights results/.../<stem>.weights.h5 --val-draws 5 --out <folder>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import tensorflow as tf

from src.evaluation.metrics import confusion_matrix, macro_f1
from src.wirings import param_counts, size_for_budget
from src.benchmark.checkpoints import (config_hash, rebuild_from_result,
                                       save_weights, stamp_protocol)
from src.benchmark.tracking import keras_callbacks, log_summary, track_fit
from src.benchmark.registry import cell_kwargs as registry_cell_kwargs
from src.tasks.person_activity.datasets import load_person_activity
from src.tasks.committee.views import (
    COORDS, make_views, average_views, drop_features, add_noise,
    partition_masks, fault_counts, noise_sigma, corrupt_views, channel_fault)
from src.tasks.committee.model import build_committee_model, COMMITTEE_WIRINGS
from src.tasks.person_activity.model import (
    backbone_override, backbone_suffix, budget_sizes, effective_backbone_layers,
    effective_ode_unfolds)


DEFAULT_SIZE = 48
BUDGET_TOLERANCE = 0.05
PLATEAU = 8          # steps_to_90 reference: mean of the last 8 positions (25-32)
VAL_DRAW_STRIDE = 1000   # --eval-only: validation draw k: val_seed + 1000 k


def position_metrics(pred, y, num_classes):
    """Per window position t: accuracy (T,) and macro-F1 (T,) of the
    predicted classes pred (N,T) against y (N,T); plus the window macro-F1
    pooled over all positions."""
    cms = np.stack([confusion_matrix(y[:, t], pred[:, t], num_classes)
                    for t in range(y.shape[1])])            # (T,C,C)
    acc = np.trace(cms, axis1=1, axis2=2) / cms.sum(axis=(1, 2))
    return acc, np.array([macro_f1(c) for c in cms]), macro_f1(cms.sum(axis=0))


def steps_to(curve, frac=0.9, plateau=PLATEAU):
    """First window position (1-based) where the curve reaches frac of its
    plateau, the mean of its last `plateau` positions (25-32 at seq_len 32);
    the plateau mean never exceeds the curve maximum, so it is always reached."""
    curve = np.asarray(curve)
    return int(np.argmax(curve >= frac * curve[-plateau:].mean())) + 1


def _rng(seed, *keys):
    """np.random.Generator keyed on (seed, *keys); a fault key that is a K=1
    probability enters as round(1000 p)."""
    return np.random.default_rng(
        [seed, *(int(k) if float(k).is_integer() else int(round(1000 * k))
                 for k in keys)])


def _average(views, how):
    return views if how == "off" else average_views(views, how)


def view_transform(mode, n_features, n_columns, overlap=0, noise=0.1, seed=1):
    """(views, rng) -> views: the exploratory view modes on top of the
    corrupted identical copies (identical: unchanged)."""
    if mode == "identical":
        return lambda v, rng: v
    if mode == "partition":
        masks = partition_masks(n_features, n_columns, overlap=overlap,
                                seed=seed).astype(np.float32)
        return lambda v, rng: v * masks
    return lambda v, rng: v + rng.normal(0.0, noise, v.shape).astype(np.float32)


class CorruptedBatches(tf.keras.utils.Sequence):
    """Training batches corrupted on the fly, identical across arms.

    Batch i of epoch e: windows perm_e[i*b:(i+1)*b] with perm_e from
    default_rng([seed, e]); m uniform over the fault grid, then corrupt_views
    and the view transform, all from default_rng([seed, e, i]); monolith arms
    (view_average != 'off') average the K corrupted views last, so every arm
    with the same seed and K trains on the same draws. Fit on dataset() and
    call on_epoch_end after every epoch (the runner's callback).
    """

    def __init__(self, x, t, y, n_columns, grid, sigma, seed, batch_size,
                 view_average="off", transform=None):
        super().__init__()
        self.x, self.t, self.y = x, t, y
        self.K, self.grid, self.sigma = n_columns, list(grid), sigma
        self.seed, self.batch_size = seed, batch_size
        self.view_average = view_average
        self.transform = transform or (lambda v, rng: v)
        self.epoch = 0
        self.perm = _rng(seed, 0).permutation(len(x))

    def __len__(self):
        return -(-len(self.x) // self.batch_size)

    def corrupted(self, i):
        """(window indices, fault count m, (B,T,K,F) views before averaging)
        of batch i in the current epoch."""
        idx = self.perm[i * self.batch_size:(i + 1) * self.batch_size]
        rng = _rng(self.seed, self.epoch, i)
        m = self.grid[rng.integers(len(self.grid))]
        v = corrupt_views(self.x[idx], self.K, m, self.sigma, rng)
        return idx, m, self.transform(v, rng)

    def __getitem__(self, i):
        idx, _, v = self.corrupted(i)
        return (_average(v, self.view_average), self.t[idx]), self.y[idx]

    def on_epoch_end(self):
        self.epoch += 1
        self.perm = _rng(self.seed, self.epoch).permutation(len(self.x))

    def dataset(self):
        """The batches in order as a tf.data.Dataset with static shapes:
        Keras' Sequence adapter keeps only the tensor ranks, and the folded
        committee cells need the feature width."""
        (v, t), y = self[0]

        def spec(a):
            return tf.TensorSpec((None, *a.shape[1:]), a.dtype)
        return tf.data.Dataset.from_generator(
            lambda: (self[i] for i in range(len(self))),
            output_signature=((spec(v), spec(t)), spec(y)))


class GridValidation(tf.keras.callbacks.Callback):
    """Per epoch: pooled macro-F1 on each of the fixed validation sets (one per
    fault count m), logged as val_macro_f1_grid (list) and val_macro_f1_mean;
    keeps the weights of the epoch with the best mean and restores them in
    on_train_end. sets: [((views, t), y)] in grid order."""

    def __init__(self, sets, num_classes, batch_size):
        super().__init__()
        self.sets, self.C, self.batch_size = sets, num_classes, batch_size
        self.curve, self.best_epoch, self.best = [], None, -np.inf
        self.best_weights = None

    def score(self):
        return [macro_f1(confusion_matrix(
            y, self.model.predict(x, batch_size=self.batch_size,
                                  verbose=0).argmax(-1), self.C))
            for x, y in self.sets]

    def on_epoch_end(self, epoch, logs=None):
        grid = [float(f) for f in self.score()]
        mean = float(np.mean(grid))
        self.curve.append(grid)
        if logs is not None:
            logs["val_macro_f1_grid"] = grid
            logs["val_macro_f1_mean"] = mean
        if mean > self.best:
            self.best, self.best_epoch = mean, epoch + 1
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


def vote_strength(model):
    """The voting strength lambda of the model's committee cell (learned or
    fixed), or None for an arm without a state vote (cncp2_multi)."""
    for layer in model.layers:
        cell = getattr(layer, "cell", None)
        if hasattr(cell, "vote_strength"):
            return float(cell.vote_strength)
    return None


def result_filename(args):
    """Result JSON name: arm, view mode, lr, clip, corruption seed and seed; a
    set --backbone-layers adds `_bb{N}` (V2 vs V2s)."""
    td = f"_td{args.train_drop}" if args.train_drop > 0.0 else ""
    tn = f"_tn{args.train_noise}" if args.train_noise > 0.0 else ""
    tc = "_tcnoise" if args.test_corruption == "noise" else ""
    tr = "_trainclean" if args.train_corruption == "off" else ""
    lam = "" if args.vote_lambda is None else f"_lam{args.vote_lambda:g}"
    med = "_median" if args.consensus == "median" else ""
    dvu = f"_dvu{args.dense_vote_units}" if args.dense_vote_units else ""
    avg = {"off": "", "mean": "_viewavg", "median": "_viewmed",
           "fault_mean": "_viewfm"}[args.view_average]
    clip = "off" if args.clipnorm is None else f"{args.clipnorm:g}"
    return (f"{args.cell}_{args.wiring}_K{args.n_columns}_s{args.size}_"
            f"{args.view_mode}{lam}{med}{dvu}{avg}{td}{tn}{tc}{tr}"
            f"_lr{args.lr:g}_clip{clip}_cs{args.corruption_seed}"
            f"_seed{args.seed}{backbone_suffix(args.backbone_layers)}.json")


def _floats(s):
    return [float(x) for x in s.split(",") if x.strip()]


def eval_only(argv):
    """--eval-only (#81): re-score a stored fit without training.

    The model is rebuilt from the record's config.effective with its weights
    (rebuild_from_result); views, fault grid and batch size come from
    config.effective, the noise scale from the record's noise_sigma. Refused
    (exit 1) when config.effective does not hash to its config_hash or the
    weights file is not the record's weights_path. The draws are the
    runner's: test _rng(seed, m) per test seed; validation draw k
    _rng(val_seed + VAL_DRAW_STRIDE * k, m), so draw 0 is the stored
    validation draw and draws k >= 1 (val_seed + 1000, + 2000, ...) differ
    from the test seeds 0-4 and from val_seed. Writes
    <out>/<stem>.evalonly.json (confusions: rows true, columns predicted):

      test_confusion_by_recording_by_fault_by_seed
          {str(test seed): [{recording: confusion} per fault count]}: one
          seed's entry has the layout of test_confusion_by_recording_by_fault
      test_confusion_by_recording_by_fault, test_macro_f1_by_fault,
      test_macro_f1_per_recording_by_fault
          from the per-seed confusions summed over eval_test_seeds, as the
          runner computes them (comparable with the source record's keys)
      val_confusion_by_fault_by_draw
          [[confusion per fault count] per draw], draw k from
          eval_val_draw_seeds[k]; val_macro_f1_by_fault_by_draw its macro-F1

    plus source_record, source_config_hash, fault_counts, eval_test_seeds,
    eval_val_draws and eval_val_draw_seeds."""
    ap = argparse.ArgumentParser(
        prog="run_committee_benchmark.py --eval-only",
        description="Re-score a stored committee fit without training (#81).")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--weights", required=True,
                    help="the fit's <stem>.weights.h5")
    ap.add_argument("--record", default=None,
                    help="the fit's result JSON (default: <stem>.json next "
                         "to the weights)")
    ap.add_argument("--test-seeds", default=None,
                    help="test draws per fault count (default: the record's "
                         "test_seeds)")
    ap.add_argument("--val-draws", type=int, default=1,
                    help="validation draws per fault count; draw 0 is the "
                         "stored one")
    ap.add_argument("--out", default=None,
                    help="output folder (default: the record's folder; pass "
                         "one outside a read-only results tree)")
    ap.add_argument("--data-path", default=None)
    args = ap.parse_args(argv)
    if args.val_draws < 1:
        ap.error("--val-draws must be >= 1")
    path = args.record or args.weights.removesuffix(".weights.h5") + ".json"
    with open(path) as f:
        record = json.load(f)
    eff, name = record["config"]["effective"], os.path.basename(args.weights)
    if eff.get("config_hash") != config_hash(eff):
        sys.exit(f"eval-only refused: config.effective of {path} hashes to "
                 f"{config_hash(eff)}, its config_hash is "
                 f"{eff.get('config_hash')!r}")
    if name != record.get("weights_path") or not os.path.samefile(
            args.weights, os.path.join(os.path.dirname(path), name)):
        sys.exit(f"eval-only refused: {args.weights} is not the weights_path "
                 f"{record.get('weights_path')!r} of {path}")
    if eff.get("deterministic"):
        tf.config.experimental.enable_op_determinism()
    model = rebuild_from_result(path)
    v, b = eff["views"], eff["build"]
    d = load_person_activity(seq_len=b["seq_len"], data_path=args.data_path,
                             split=eff["split"], split_seed=eff["split_seed"])
    K, C = v["n_columns"], d.num_classes
    grid = fault_counts(K, v["rho_grid"])
    sigma = np.asarray(record["noise_sigma"], np.float32)
    transform = view_transform(v["mode"], d.feature_size, K, v["overlap"],
                               v["noise"], v["seed"])

    def predict(x, t, m, seed):
        rng = _rng(seed, m)
        views = _average(transform(corrupt_views(x, K, m, sigma, rng), rng),
                         v["average"])
        return model.predict((views, t), batch_size=eff["batch_size"],
                             verbose=0).argmax(-1)

    seeds = ([int(s) for s in args.test_seeds.split(",")]
             if args.test_seeds else eff["test_seeds"])
    recs = [str(r) for r in np.unique(d.test_rec)]
    by_seed = {}
    for s in seeds:
        by_seed[str(s)] = []
        for m in grid:
            pred = predict(d.test_x, d.test_t, m, s)
            by_seed[str(s)].append({r: confusion_matrix(
                d.test_y[d.test_rec == r], pred[d.test_rec == r], C).tolist()
                for r in recs})
    total = np.sum([[[c[r] for r in recs] for c in cs]
                    for cs in by_seed.values()], axis=0)  # (fault, rec, C, C)
    val_seeds = [eff["val_seed"] + VAL_DRAW_STRIDE * k
                 for k in range(args.val_draws)]
    val_cm = [[confusion_matrix(d.val_y, predict(d.val_x, d.val_t, m, vs), C)
               for m in grid] for vs in val_seeds]
    out = {
        "source_record": path, "source_config_hash": eff["config_hash"],
        "fault_counts": grid, "eval_test_seeds": seeds,
        "eval_val_draws": args.val_draws, "eval_val_draw_seeds": val_seeds,
        "test_confusion_by_recording_by_fault_by_seed": by_seed,
        "test_confusion_by_recording_by_fault": [
            {r: c.tolist() for r, c in zip(recs, f)} for f in total],
        "test_macro_f1_by_fault": [macro_f1(f.sum(axis=0)) for f in total],
        "test_macro_f1_per_recording_by_fault": [
            {r: macro_f1(c) for r, c in zip(recs, f)} for f in total],
        "val_confusion_by_fault_by_draw": [[c.tolist() for c in cs]
                                           for cs in val_cm],
        "val_macro_f1_by_fault_by_draw": [[macro_f1(c) for c in cs]
                                          for cs in val_cm]}
    dest = os.path.join(args.out or os.path.dirname(os.path.abspath(path)),
                        os.path.basename(path)[:-len(".json")]
                        + ".evalonly.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as f:
        json.dump(out, f)
    same = (out["test_confusion_by_recording_by_fault"]
            == record.get("test_confusion_by_recording_by_fault"))
    print(f"eval-only {path}: test macro-F1 by fault "
          f"{[round(x, 4) for x in out['test_macro_f1_by_fault']]}, summed "
          f"test confusions {'equal' if same else 'differ from'} the "
          f"record's; validation draw 0 macro-F1 "
          f"{[round(x, 4) for x in out['val_macro_f1_by_fault_by_draw'][0]]}"
          f" -> {dest}")
    return out


def main():
    if "--eval-only" in sys.argv[1:]:
        return eval_only(sys.argv[1:])
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiring", default="cncp", choices=COMMITTEE_WIRINGS)
    ap.add_argument("--n-columns", type=int, default=4)
    cap = ap.add_mutually_exclusive_group()
    cap.add_argument("--size", type=int, default=None,
                     help="the arm's width knob (see build_committee_model); "
                          f"default {DEFAULT_SIZE} when no budget is given")
    cap.add_argument("--param-budget", type=int, default=None,
                     help="size the arm to this EFFECTIVE parameter budget "
                          "(closest size on the arm's own knob; weight sharing "
                          "makes the count independent of K)")
    ap.add_argument("--cell", default="cfc_lrc")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="RMSprop learning rate")
    ap.add_argument("--clipnorm", type=float, default=None,
                    help="clip the global gradient norm to this value (the "
                         "optimizer's global_clipnorm); default: no clipping")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--data-path", default=None,
                    help="person-activity CSV (default: data/person/ under "
                         "the repo root)")
    ap.add_argument("--split", default="grouped",
                    choices=("grouped", "random_window"),
                    help="grouped: whole recordings to train/val/test "
                         "(--split-seed); random_window: the original window "
                         "split, val = test (literature anchor only)")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--view-mode", default="identical",
                    choices=("identical", "partition", "noisy"),
                    help="identical (suite): every column sees the same "
                         "window; partition / noisy: exploratory July path")
    ap.add_argument("--rho-grid", default="0,0.1,0.3,0.5,0.67",
                    help="fault fractions; the fault grid is the sorted, "
                         "de-duplicated round(rho*K) (K=1: the rhos, a "
                         "per-window failure probability)")
    ap.add_argument("--noise-frac", type=float, default=0.3,
                    help="coordinate noise sigma as a fraction of the "
                         "training std of each coordinate")
    ap.add_argument("--train-corruption", default="on", choices=("on", "off"),
                    help="on: per-batch fault count from the grid and "
                         "coordinate noise; off: clean identical views")
    ap.add_argument("--corruption-seed", type=int, default=0,
                    help="seeds the training draws and the batch order "
                         "(shared by every arm)")
    ap.add_argument("--val-seed", type=int, default=100,
                    help="seeds the fixed validation draws per fault count")
    ap.add_argument("--test-seeds", default="0,1,2,3,4",
                    help="fixed test draws per fault count, pooled")
    ap.add_argument("--overlap", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.1,
                    help="--view-mode noisy: per-column noise on top")
    ap.add_argument("--view-seed", type=int, default=1,
                    help="fixes the feature partition (same masks train+test)")
    ap.add_argument("--vote-lambda", type=float, default=None,
                    help="fixed voting strength in [0, 1] (default: learned); "
                         "0 = columns independent, the consensus control")
    ap.add_argument("--consensus", default="mean", choices=("mean", "median"),
                    help="consensus and readout statistic across columns")
    ap.add_argument("--dense-vote-units", type=int, default=None,
                    help="dense only: vote on / read out the leading n units "
                         "(the subspace-voting control for the cncp L2/3 vote)")
    ap.add_argument("--view-average", default="off", nargs="?", const="mean",
                    choices=("off", "mean", "median", "fault_mean"),
                    help="monolith control: collapse the K corrupted views to "
                         "their coordinate-wise mean, lower median or "
                         "fault-aware mean, then run a single column (K=1 "
                         "model); bare flag = mean")
    ap.add_argument("--train-drop", type=float, default=0.0,
                    help="exploratory: train-time channel-dropout rate")
    ap.add_argument("--train-noise", type=float, default=0.0,
                    help="exploratory: train-time Gaussian-noise sigma")
    ap.add_argument("--test-corruption", default="dropout",
                    choices=("dropout", "noise"),
                    help="exploratory sweep family: dropout over --drop-fracs "
                         "(runs only when given) or noise over --noise-sigmas")
    ap.add_argument("--noise-sigmas", default="0.0,0.25,0.5,1.0",
                    help="exploratory: test-time Gaussian-noise sigmas")
    ap.add_argument("--drop-fracs", default="",
                    help="exploratory: test-time channel-dropout fractions; "
                         "empty = no sweep")
    ap.add_argument("--drop-seeds", default="0,1,2,3,4",
                    help="exploratory: channel-choice seeds per level")
    ap.add_argument("--outdir", default="results/committee")
    ap.add_argument("--elastance-type", default=None,
                    help="override the registry elastance_type of the "
                         "LRC-family cells; default: the registry value")
    ap.add_argument("--backbone-layers", type=int, default=None,
                    help="CfC-family backbone depth for EVERY wiring (overrides "
                         "the default 1 of every wiring; 0 = no backbone); "
                         "default: unchanged per-wiring behaviour")
    ap.add_argument("--column-loop", action="store_true",
                    help="cncp/ncp/dense: step the K columns in the per-column "
                         "loop instead of the folded batch (same function, "
                         "slower; the #32 equivalence canary)")
    ap.add_argument("--verbose", type=int, default=0)
    ap.add_argument("--save-weights", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="write <stem>.weights.h5 (best epoch) next to the "
                         "result JSON")
    ap.add_argument("--wandb", action="store_true",
                    help="log this run to Weights & Biases (needs the "
                         "'tracking' extra; use WANDB_MODE=offline on cluster)")
    ap.add_argument("--deterministic", action="store_true",
                    help="tf.config.experimental.enable_op_determinism() "
                         "before model construction (#76)")
    args = ap.parse_args()
    if args.size is None and args.param_budget is None:
        args.size = DEFAULT_SIZE
    if args.wiring == "cncp2_multi" and args.vote_lambda is not None:
        # Independent columns read out by mean/median, no state vote.
        warnings.warn("--vote-lambda does not apply to cncp2_multi; ignored")
        args.vote_lambda = None
    if args.view_average == "median" and args.view_mode == "partition":
        warnings.warn("--view-average median on partition views: a feature "
                      "owned by fewer than half of the columns has median 0; "
                      "use --view-mode noisy")

    if args.deterministic:
        tf.config.experimental.enable_op_determinism()
    cell_kwargs = registry_cell_kwargs(args.cell, args.elastance_type)
    cell_kwargs.update(backbone_override(args.cell, args.backbone_layers))
    bb_effective = effective_backbone_layers(args.wiring, args.cell, cell_kwargs)

    d = load_person_activity(seq_len=args.seq_len, data_path=args.data_path,
                             split=args.split, split_seed=args.split_seed)
    F, C, K = d.feature_size, d.num_classes, args.n_columns
    rho_grid = _floats(args.rho_grid)
    grid = fault_counts(K, rho_grid)
    sigma = noise_sigma(d.train_x, args.noise_frac)
    test_seeds = [int(s) for s in args.test_seeds.split(",")]
    transform = view_transform(args.view_mode, F, K, args.overlap, args.noise,
                               args.view_seed)

    def draw(x, m, rng):
        """The model input for one fixed draw: corrupt, view mode, average."""
        return _average(transform(corrupt_views(x, K, m, sigma, rng), rng),
                        args.view_average)

    # The model's column count: K, or 1 for the input-averaging monolith.
    n_cols = 1 if args.view_average != "off" else K

    print(f"committee {args.wiring} K={K} size={args.size} "
          f"lambda={'learned' if args.vote_lambda is None else args.vote_lambda} "
          f"consensus={args.consensus} view-average={args.view_average} "
          f"backbone_layers={args.backbone_layers} (effective {bb_effective}): "
          f"train {d.train_x.shape[0]} / val {d.val_x.shape[0]} / test "
          f"{d.test_x.shape[0]} ({d.split}), F={F} C={C} mode={args.view_mode} "
          f"fault grid {grid} sigma {np.round(sigma[COORDS], 4).tolist()}",
          flush=True)

    def build(size):
        return build_committee_model(
            args.wiring, n_cols, cell=args.cell, size=size,
            feature_size=F, seq_len=args.seq_len, num_classes=C, lr=args.lr,
            clipnorm=args.clipnorm, seed=args.seed,
            vote_lambda=args.vote_lambda, consensus=args.consensus,
            dense_vote_units=args.dense_vote_units,
            train_drop=args.train_drop, train_noise=args.train_noise,
            column_loop=args.column_loop, **cell_kwargs)

    # Capacity: a fixed width knob, or the knob closest to the effective
    # parameter budget for THIS arm.
    budget_dev, over_budget = None, False
    if args.param_budget is not None:
        args.size, n_eff = size_for_budget(build, args.param_budget,
                                           budget_sizes(args.wiring))
        budget_dev = abs(n_eff - args.param_budget) / args.param_budget
        over_budget = budget_dev > BUDGET_TOLERANCE
        flag = "  ABOVE TOLERANCE" if over_budget else ""
        print(f"budget {args.param_budget}: size={args.size} "
              f"params_effective={n_eff} (dev {budget_dev:.1%}){flag}")
    model = build(args.size)
    pc = param_counts(model)
    params = pc['params_effective']

    effective = stamp_protocol({
        "task": "committee", "param_budget": args.param_budget,
        "deterministic": args.deterministic,
        "backbone_layers": bb_effective,
        # training, data split and corruption knobs outside the builder
        # (#53 config_hash)
        "epochs": args.epochs, "batch_size": args.batch_size,
        "split": d.split, "split_seed": d.split_seed,
        "train_corruption": args.train_corruption,
        "corruption_seed": args.corruption_seed, "val_seed": args.val_seed,
        "test_seeds": test_seeds,
        # no --ode-unfolds here: the cell constructor default applies
        "ode_unfolds": effective_ode_unfolds(args.cell, cell_kwargs),
        "views": dict(n_columns=K, mode=args.view_mode,
                      overlap=args.overlap, noise=args.noise,
                      seed=args.view_seed, average=args.view_average,
                      rho_grid=rho_grid, noise_frac=args.noise_frac),
        "build": dict(wiring=args.wiring, n_columns=n_cols,
                      cell=args.cell, size=args.size, feature_size=F,
                      seq_len=args.seq_len, num_classes=C, lr=args.lr,
                      clipnorm=args.clipnorm,
                      seed=args.seed, vote_lambda=args.vote_lambda,
                      consensus=args.consensus,
                      dense_vote_units=args.dense_vote_units,
                      train_drop=args.train_drop,
                      train_noise=args.train_noise,
                      column_loop=args.column_loop),
        "elastance_type": cell_kwargs.get("elastance_type"),
        "cell_kwargs": cell_kwargs})
    wandb_config = {
        "task": "committee", "system": None, "cell": args.cell,
        "wiring": args.wiring, "seed": args.seed, "data_seed": d.split_seed,
        "split": d.split, "n_columns": K, "model_columns": n_cols,
        "size": args.size, "param_budget": args.param_budget,
        "budget_deviation": budget_dev,
        "budget_above_tolerance": over_budget, "lr": args.lr,
        "clipnorm": args.clipnorm, "batch_size": args.batch_size,
        "epochs": args.epochs, "view_mode": args.view_mode,
        "view_average": args.view_average, "vote_lambda": args.vote_lambda,
        "consensus": args.consensus,
        "dense_vote_units": args.dense_vote_units,
        # corruption protocol (issue #55)
        "rho_grid": rho_grid, "fault_counts": grid,
        "noise_frac": args.noise_frac,
        "train_corruption": args.train_corruption,
        "corruption_seed": args.corruption_seed, "val_seed": args.val_seed,
        "test_seeds": test_seeds, **pc}

    on = args.train_corruption == "on"
    train = CorruptedBatches(
        d.train_x, d.train_t, d.train_y, K, grid if on else [0],
        sigma if on else np.zeros_like(sigma), args.corruption_seed,
        args.batch_size, args.view_average, transform)
    gv = GridValidation(
        [((draw(d.val_x, m, _rng(args.val_seed, m)), d.val_t), d.val_y)
         for m in grid], C, args.batch_size)
    lam_curve = []
    t0 = time.time()
    def epoch_end(epoch, logs):
        train.on_epoch_end()
        lam_curve.append(vote_strength(model))

    def predict(views, t):
        return model.predict((views, t), batch_size=args.batch_size,
                             verbose=0).argmax(-1)

    with track_fit(args.wandb, task="committee",
                   name=result_filename(args)[:-len(".json")],
                   config=wandb_config, effective=effective) as run_:
        hist = model.fit(train.dataset(), epochs=args.epochs,
                         verbose=args.verbose,
                         callbacks=[gv, tf.keras.callbacks.LambdaCallback(
                             on_epoch_end=epoch_end), *keras_callbacks(run_)])
        wall = time.time() - t0      # gv restored the best-epoch weights
        diverged = not np.isfinite(hist.history["loss"][-1])
        best = gv.best_epoch or len(gv.curve)

        # Test per fault count: pooled over --test-seeds fixed draws;
        # confusions per position (curve) and per recording (bootstrap units).
        T, recs = d.test_y.shape[1], np.unique(d.test_rec)
        by_fault, per_rec, cm_rec, pos_f1, acc_m0 = [], [], [], [], None
        for m in grid:
            cms = np.zeros((T, C, C), np.int64)
            rec_cms = {str(r): np.zeros((C, C), np.int64) for r in recs}
            for s in test_seeds:
                pred = predict(draw(d.test_x, m, _rng(s, m)), d.test_t)
                cms += np.stack([confusion_matrix(d.test_y[:, t], pred[:, t],
                                                  C) for t in range(T)])
                for r in recs:
                    sel = d.test_rec == r
                    rec_cms[str(r)] += confusion_matrix(d.test_y[sel],
                                                        pred[sel], C)
            pooled = cms.sum(axis=0)
            if acc_m0 is None:
                acc_m0 = float(np.trace(pooled) / pooled.sum())
            by_fault.append(macro_f1(pooled))
            per_rec.append({r: macro_f1(c) for r, c in rec_cms.items()})
            cm_rec.append({r: c.tolist() for r, c in rec_cms.items()})
            pos_f1.append(np.array([macro_f1(c) for c in cms]))
        log_summary(run_, {
            "best_val_epoch": best,
            "val_macro_f1_mean": gv.best if gv.best_epoch else float("nan"),
            "test_macro_f1_mean_grid": float(np.mean(by_fault)),
            **{f"test_macro_f1_m{m}": f for m, f in zip(grid, by_fault)},
            **{f"test_per_position_macro_f1_auc_m{m}": float(c.mean())
               for m, c in zip(grid, pos_f1)},
            "vote_lambda_final": vote_strength(model)})

    # Correlated failure: one coordinate zeroed in every column, at m = 0.
    chan = {}
    for name, ch in zip("xyz", range(F)[COORDS]):
        cm = np.zeros((C, C), np.int64)
        for s in test_seeds:
            v = channel_fault(corrupt_views(d.test_x, K, 0, sigma, _rng(s, 0)),
                              ch)
            cm += confusion_matrix(d.test_y, predict(
                _average(transform(v, _rng(s, 0)), args.view_average),
                d.test_t), C)
        chan[name] = macro_f1(cm)
    chan["mean"] = float(np.mean(list(chan.values())))

    steps90 = [steps_to(c) for c in pos_f1]
    lam_final = vote_strength(model)
    print(f"  params {params} | best epoch {best} val mean "
          f"{gv.best:.3f} | test macro-F1 by fault "
          f"{[f'{m}:{f:.3f}' for m, f in zip(grid, by_fault)]} | channel "
          f"{chan['mean']:.3f} | steps_to_90 {steps90} | lambda {lam_final}")

    record = {
        "task": "person_activity", "wiring": args.wiring,
        "n_columns": K, "size": args.size,
        "param_budget": args.param_budget, "budget_deviation": budget_dev,
        "budget_above_tolerance": over_budget,
        "cell": args.cell, "seed": args.seed, "epochs": args.epochs,
        "lr": args.lr, "batch_size": args.batch_size,
        "clipnorm": args.clipnorm,
        "clip_mode": None if args.clipnorm is None else "global", **pc,
        "feature_size": F, "num_classes": C, "view_mode": args.view_mode,
        "overlap": args.overlap, "train_drop": args.train_drop,
        "train_noise": args.train_noise, "test_corruption": args.test_corruption,
        "vote_lambda": args.vote_lambda, "consensus": args.consensus,
        "dense_vote_units": args.dense_vote_units,
        "view_average": args.view_average, "model_columns": n_cols,
        "backbone_layers": args.backbone_layers,
        "backbone_layers_effective": bb_effective,
        **({"coupling": "none", "readout": args.consensus}
           if args.wiring == "cncp2_multi" else {}),
        "split": d.split, "split_seed": d.split_seed,
        "recordings": d.recordings,
        # corruption protocol (issue #55)
        "rho_grid": rho_grid, "fault_counts": grid,
        "noise_frac": args.noise_frac,
        "noise_sigma": [float(x) for x in sigma],
        "train_corruption": args.train_corruption,
        "corruption_seed": args.corruption_seed, "val_seed": args.val_seed,
        "test_seeds": test_seeds,
        # selection: best mean validation macro-F1 over the grid (1-based
        # epoch; the curve is epochs x fault counts)
        "best_val_epoch": best,
        "val_macro_f1_grid_curve": gv.curve,
        "val_macro_f1_mean": gv.best if gv.best_epoch else float("nan"),
        # test at the best epoch, one entry per fault count (grid order)
        "clean_accuracy": acc_m0,
        "test_macro_f1_by_fault": by_fault,
        "test_macro_f1_mean_grid": float(np.mean(by_fault)),
        "test_macro_f1_per_recording_by_fault": per_rec,
        # rows = true, columns = predicted, summed over the test seeds
        "test_confusion_by_recording_by_fault": cm_rec,
        # one length-seq_len curve per fault count; the area is normalised
        # (mean over positions), steps are 1-based positions against the
        # plateau of the last PLATEAU positions
        "per_position_macro_f1": [[round(float(v), 4) for v in c]
                                  for c in pos_f1],
        "per_position_macro_f1_auc": [float(c.mean()) for c in pos_f1],
        "steps_to_90": steps90,
        "channel_fault_macro_f1": chan,
        "vote_lambda_final": lam_final,
        "vote_lambda_curve": lam_curve,
        "wallclock_s": round(wall, 1),
        "training_diverged": diverged,
        "config": {"effective": effective},
    }

    # Exploratory July sweep (not part of the suite): whole-channel dropout or
    # additive noise on the clean window, then the view mode of make_views.
    if args.test_corruption == "noise":
        levels, corrupt = _floats(args.noise_sigmas), add_noise
    else:
        levels, corrupt = _floats(args.drop_fracs), drop_features
    if levels:
        drop_seeds = [int(s) for s in args.drop_seeds.split(",")]

        def views(X):
            v, _ = make_views(X, K, mode=args.view_mode, overlap=args.overlap,
                              noise=args.noise, seed=args.view_seed)
            return _average(v, args.view_average)
        accs, stds, f1s = [], [], []
        for lv in levels:
            a, _, w = (np.array(m) for m in zip(*[position_metrics(
                predict(views(corrupt(d.test_x, lv, seed=s)), d.test_t),
                d.test_y, C) for s in drop_seeds]))
            accs.append(float(a.mean()))
            stds.append(float(a.mean(axis=1).std()))
            f1s.append(float(w.mean()))
        record.update({
            "drop_fracs": levels, "drop_accs": [round(a, 4) for a in accs],
            "drop_stds": [round(s, 4) for s in stds], "drop_seeds": drop_seeds,
            "drop_macro_f1": [round(f, 4) for f in f1s]})

    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, result_filename(args))
    record["weights_path"] = save_weights(model, path, args.save_weights)
    with open(path, "w") as f:
        json.dump(record, f)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
