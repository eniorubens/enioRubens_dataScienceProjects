"""
test_evaluation.py — Tests for CLTVEvaluator.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.evaluation import CLTVEvaluator


@pytest.fixture
def evaluator() -> CLTVEvaluator:
    return CLTVEvaluator()


# ── RMSE ─────────────────────────────────────────────────────────────────────

def test_compute_rmse_perfect_prediction(evaluator):
    """RMSE must be 0 when actual == predicted."""
    values = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert evaluator.compute_rmse(values, values) == pytest.approx(0.0)


def test_compute_rmse_known_value(evaluator):
    """RMSE for a known case: actual=[0,1], predicted=[1,0] → RMSE=1."""
    actual = pd.Series([0.0, 1.0])
    predicted = pd.Series([1.0, 0.0])
    assert evaluator.compute_rmse(actual, predicted) == pytest.approx(1.0)


# ── Pearson independence check ────────────────────────────────────────────────

def test_pearson_check_returns_required_keys(evaluator):
    """pearson_independence_check must return pearson_r, p_value, assumption_holds."""
    rng = np.random.default_rng(1)
    rfm = pd.DataFrame(
        {
            "customer_id": range(30),
            "frequency": rng.integers(1, 10, size=30).astype(float),
            "monetary_value": rng.uniform(10, 300, size=30),
        }
    )
    result = evaluator.pearson_independence_check(rfm)
    for key in ("pearson_r", "p_value", "assumption_holds"):
        assert key in result


def test_pearson_check_correlated_data(evaluator):
    """assumption_holds must be False for strongly correlated data."""
    n = 50
    x = np.arange(1, n + 1, dtype=float)
    rfm = pd.DataFrame(
        {
            "customer_id": range(n),
            "frequency": x,
            "monetary_value": x * 100.0,
        }
    )
    result = evaluator.pearson_independence_check(rfm)
    assert result["assumption_holds"] is False


def test_pearson_check_uncorrelated_data(evaluator):
    """assumption_holds must be True for independent data."""
    rng = np.random.default_rng(42)
    n = 100
    rfm = pd.DataFrame(
        {
            "customer_id": range(n),
            "frequency": rng.integers(1, 10, size=n).astype(float),
            "monetary_value": rng.uniform(10, 300, size=n),
        }
    )
    result = evaluator.pearson_independence_check(rfm)
    assert result["assumption_holds"] is True


# ── calibration_plot ──────────────────────────────────────────────────────────

def test_calibration_plot_returns_axes(evaluator, fitted_model, sample_calibration_holdout):
    """calibration_plot must return a matplotlib Axes object."""
    import matplotlib.pyplot as plt

    ax = evaluator.calibration_plot(fitted_model, sample_calibration_holdout)
    assert hasattr(ax, "get_xlabel")
    plt.close("all")


# ── evaluate ─────────────────────────────────────────────────────────────────

def test_evaluate_returns_complete_dict(evaluator, fitted_model, sample_calibration_holdout):
    """evaluate() must return a dict with all required keys."""
    result = evaluator.evaluate(fitted_model, sample_calibration_holdout)
    required_keys = {
        "rmse_frequency",
        "mae_frequency",
        "pearson_r",
        "pearson_p",
        "assumption_holds",
        "n_customers_holdout",
    }
    assert required_keys <= set(result.keys())
    assert isinstance(result["rmse_frequency"], float)
    assert isinstance(result["n_customers_holdout"], int)
