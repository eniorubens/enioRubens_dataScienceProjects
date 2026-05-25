"""
test_cltv_model.py — Tests for CLTVModel.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.cltv_model import CLTVModel


# ── Fit ──────────────────────────────────────────────────────────────────────

def test_fit_returns_self(sample_modeling_base):
    """fit() must return self to support method chaining."""
    model = CLTVModel(fit_method="map")
    result = model.fit(sample_modeling_base)
    assert result is model


def test_fit_sets_is_fitted(sample_modeling_base):
    """is_fitted() must return True after fit()."""
    model = CLTVModel(fit_method="map")
    assert not model.is_fitted()
    model.fit(sample_modeling_base)
    assert model.is_fitted()


# ── Predict before fit ───────────────────────────────────────────────────────

def test_predict_before_fit_raises(sample_modeling_base):
    """Calling predict_cltv() before fit() must raise RuntimeError."""
    model = CLTVModel()
    with pytest.raises(RuntimeError, match="fit\\(\\)"):
        model.predict_cltv(sample_modeling_base)


# ── predict_cltv ─────────────────────────────────────────────────────────────

def test_predict_cltv_output_columns(fitted_model, sample_modeling_base):
    """predict_cltv() must include all expected output columns."""
    result = fitted_model.predict_cltv(sample_modeling_base, t=90)
    for col in ("customer_id", "predicted_purchases", "probability_alive", "expected_spend", "cltv_90d"):
        assert col in result.columns, f"Missing column: {col}"


def test_predict_cltv_positive_values(fitted_model, sample_modeling_base):
    """CLTV values must be >= 0 for all customers."""
    result = fitted_model.predict_cltv(sample_modeling_base, t=180)
    assert (result["cltv_180d"] >= 0).all()


# ── predict_probability_alive ────────────────────────────────────────────────

def test_predict_probability_alive_bounds(fitted_model, sample_modeling_base):
    """P(alive) must be in [0, 1] for all customers."""
    probs = fitted_model.predict_probability_alive(sample_modeling_base)
    assert (probs >= 0).all()
    assert (probs <= 1).all()


# ── predict_purchases ────────────────────────────────────────────────────────

def test_predict_purchases_positive(fitted_model, sample_modeling_base):
    """Predicted purchases must be >= 0 for all customers."""
    preds = fitted_model.predict_purchases(sample_modeling_base, t=180)
    assert (preds >= 0).all()


# ── Validation ───────────────────────────────────────────────────────────────

def test_validate_rfm_raises_on_missing_columns(sample_modeling_base):
    """validate_rfm_input() must raise ValueError with a descriptive message."""
    model = CLTVModel()
    bad_df = sample_modeling_base.drop(columns=["recency"])
    with pytest.raises(ValueError, match="recency"):
        model.validate_rfm_input(bad_df)


# ── get_model_params ─────────────────────────────────────────────────────────

def test_get_model_params_returns_dict(fitted_model):
    """get_model_params() must return {"bgm": {...}, "gg": {...}}."""
    params = fitted_model.get_model_params()
    assert isinstance(params, dict)
    assert "bgm" in params
    assert "gg" in params
    assert isinstance(params["bgm"], dict)
    assert isinstance(params["gg"], dict)


# ── Save / Load ───────────────────────────────────────────────────────────────

def test_save_creates_expected_files(fitted_model, tmp_path):
    """save() must create bgm_model.nc, gg_model.nc, and metadata.json."""
    fitted_model.save(tmp_path)
    assert (tmp_path / "bgm_model.nc").exists()
    assert (tmp_path / "gg_model.nc").exists()
    assert (tmp_path / "metadata.json").exists()


def test_load_restores_model(fitted_model, sample_modeling_base, tmp_path):
    """load() after save() must produce numerically equivalent predictions."""
    fitted_model.save(tmp_path)
    loaded = CLTVModel.load(tmp_path)

    assert loaded.is_fitted()

    original_preds = fitted_model.predict_purchases(sample_modeling_base, t=90)
    loaded_preds = loaded.predict_purchases(sample_modeling_base, t=90)

    np.testing.assert_allclose(
        original_preds.values,
        loaded_preds.values,
        rtol=1e-4,
        err_msg="Loaded model predictions differ from original",
    )
