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

The sequence runners (LV, PA, committee) open their run with ``track_fit``
(#53): one project for the whole final suite, group = the campaign
(``SUITE_CAMPAIGN``, the suite YAML stem exported by cluster/submit_suite.sh;
fallback the cell), job_type = ``<task>/<stage>`` (task: the LV system,
person_activity or committee; stage: ``SUITE_STAGE``, default adhoc), run name
= the result JSON stem, tag ``protocol:<protocol_version>``; the config adds
the config.effective identifiers (EFFECTIVE_KEYS) and the git SHA. At the end
of the fit ``log_summary`` writes the primary metric to the run summary.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

DEFAULT_PROJECT = "thesis-benchmarks"
# config.effective keys every sequence-runner run config carries (#53)
EFFECTIVE_KEYS = ("ode_unfolds", "backbone_layers", "elastance_type",
                  "protocol_version", "config_hash")


def _wandb_available() -> bool:
    try:
        import wandb  # noqa: F401
    except ImportError:
        return False
    return True


@contextmanager
def track(enabled, *, group, job_type, name, config, project=DEFAULT_PROJECT,
          tags=None):
    """Yield an active wandb run, or ``None`` when tracking is off/unavailable.

    ``group``/``job_type`` drive the dashboard grouping (the sequence runners
    via ``track_fit``; the legacy callers run_benchmark.py and
    run_lotka_volterra_hpo.py pass group=cell, job_type=wiring); the seed and
    hyper-parameters belong in ``config`` (tags do not group). The run is always finished on exit -- including on an
    exception -- so a crashed trial still flushes its offline data.
    """
    run = None
    if enabled and _wandb_available():
        import wandb
        run = wandb.init(project=project, group=group, job_type=job_type,
                         name=name, config=config, tags=tags, reinit=True)
    try:
        yield run
    finally:
        if run is not None:
            run.finish()


def track_fit(enabled, *, task, name, config, effective):
    """``track`` with the suite metadata of one sequence-runner fit (module
    docstring); ``config`` holds the runner's own fields (cell, wiring, system,
    seed, data_seed, lr, clipnorm, param_budget, params_effective, ...)."""
    from src.benchmark.manifest import _git_sha
    config = {**config, **{k: effective.get(k) for k in EFFECTIVE_KEYS},
              "git_sha": _git_sha() if enabled else None}
    group = os.environ.get("SUITE_CAMPAIGN") or config["cell"]
    stage = os.environ.get("SUITE_STAGE") or "adhoc"
    return track(enabled, group=group, job_type=f"{task}/{stage}", name=name,
                 config=config,
                 tags=[f"protocol:{effective['protocol_version']}"])


def log_summary(run, metrics):
    """End-of-fit metrics into the run summary; no-op without a run."""
    if run is not None:
        run.summary.update(metrics)


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
