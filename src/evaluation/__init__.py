from .metrics import mse, nrmse

__all__ = ["mse", "nrmse", "GradientFlowTracker"]


def __getattr__(name):
    # GradientFlowTracker needs TensorFlow: load it on first access so the numpy
    # metrics (metrics, rollout_metrics) import without TF (CPU aggregation).
    if name == "GradientFlowTracker":
        from .gradient_flow import GradientFlowTracker
        return GradientFlowTracker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
