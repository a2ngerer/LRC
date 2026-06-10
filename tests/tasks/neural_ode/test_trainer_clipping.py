import numpy as np
import tensorflow as tf

from src.models import make_dense_model
from src.tasks.neural_ode.datasets import generate_dataset
from src.tasks.neural_ode.ode_model import SequentialODEFunc
from src.tasks.neural_ode.trainer import train
from src.evaluation import GradientFlowTracker
from src.utils import set_global_seed


def _tiny_run(clip_norm, tracker=None, seed=0):
    rng = set_global_seed(seed)
    t, y = generate_dataset('spiral', data_size=60)
    model = SequentialODEFunc(make_dense_model('gru', units=4, output_neurons=2))
    losses = train(model, t, y, n_iters=3, batch_size=4, batch_time=8,
                   lr=1e-3, rng=rng, gradient_tracker=tracker,
                   clip_norm=clip_norm, verbose=False)
    return losses


def test_clip_norm_none_is_default_path():
    """clip_norm=None reproduces the deterministic v1 behavior."""
    assert _tiny_run(clip_norm=None) == _tiny_run(clip_norm=None)


def test_clip_norm_changes_training():
    """A tiny clip threshold must alter the loss trajectory."""
    baseline = _tiny_run(clip_norm=None)
    clipped = _tiny_run(clip_norm=1e-6)
    assert baseline != clipped


def test_tracker_records_clip_norms():
    tracker = GradientFlowTracker(log_every=1)
    _tiny_run(clip_norm=0.5, tracker=tracker)
    d = tracker.as_dict()
    assert d['clip']['iterations'] == [1, 2, 3]
    assert len(d['clip']['pre_clip_norm']) == 3
    assert all(c == 0.5 for c in d['clip']['clip_norm'])
    assert all(n >= 0 for n in d['clip']['pre_clip_norm'])


def test_tracker_clip_block_empty_without_clipping():
    tracker = GradientFlowTracker(log_every=1)
    _tiny_run(clip_norm=None, tracker=tracker)
    d = tracker.as_dict()
    assert d['clip']['iterations'] == []
