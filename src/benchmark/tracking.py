"""Optional Weights & Biases experiment tracking for the benchmark runners.

wandb is an optional dependency (``uv sync --extra tracking``). Every helper
here degrades to a no-op when wandb is not installed or tracking was not
requested, so smoke tests and the pytest suite keep working without a wandb
install or login. On the cluster set ``WANDB_MODE=offline`` and ``wandb sync``
the run directories from the login node (compute nodes have no internet).

Usage (Keras backend)::

    with track(args.wandb, group=cell, job_type=wiring, name=tag,
               config={...}) as run:
        model.fit(..., callbacks=keras_callbacks(run))

Usage (custom training loop)::

    with track(...) as run:
        train(..., log_fn=step_logger(run))
"""
from __future__ import annotations

from contextlib import contextmanager

DEFAULT_PROJECT = "thesis-benchmarks"


def _wandb_available() -> bool:
    try:
        import wandb  # noqa: F401
    except ImportError:
        return False
    return True


@contextmanager
def track(enabled, *, group, job_type, name, config, project=DEFAULT_PROJECT):
    """Yield an active wandb run, or ``None`` when tracking is off/unavailable.

    ``group``/``job_type`` drive the dashboard grouping (group=cell,
    job_type=wiring); the seed and hyper-parameters belong in ``config`` (tags
    do not group). The run is always finished on exit -- including on an
    exception -- so a crashed trial still flushes its offline data.
    """
    run = None
    if enabled and _wandb_available():
        import wandb
        run = wandb.init(project=project, group=group, job_type=job_type,
                         name=name, config=config, reinit=True)
    try:
        yield run
    finally:
        if run is not None:
            run.finish()


def keras_callbacks(run):
    """Return ``[WandbMetricsLogger]`` for an active run, else ``[]``.

    Ready to splice into ``model.fit(callbacks=keras_callbacks(run))``.
    """
    if run is None:
        return []
    from wandb.integration.keras import WandbMetricsLogger
    return [WandbMetricsLogger()]


def step_logger(run):
    """Return a ``log_fn(step, loss)`` for the custom loop, or ``None``.

    Matches the optional ``log_fn`` hook in ``neural_ode.trainer.train`` so the
    trainer itself never imports wandb.
    """
    if run is None:
        return None

    def _log(step, loss):
        run.log({"loss": loss}, step=step)

    return _log
