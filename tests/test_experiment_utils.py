import numpy as np

from scripts.experiment_utils import bootstrap_metric_ci


def test_bootstrap_metric_ci_shape():
    vals = [0.1, 0.2, 0.3, 0.4]
    out = bootstrap_metric_ci(vals, seed=42, n_bootstrap=50)
    assert "mean" in out and "ci_low" in out and "ci_high" in out
    assert out["ci_low"] <= out["mean"] <= out["ci_high"]


def test_bootstrap_metric_ci_empty():
    out = bootstrap_metric_ci([], seed=42, n_bootstrap=10)
    assert out["mean"] == 0.0
    assert out["ci_low"] == 0.0
    assert out["ci_high"] == 0.0
