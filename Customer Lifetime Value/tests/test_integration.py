"""
test_integration.py — End-to-end pipeline tests.
"""

from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd
import pytest


# ── Full pipeline ─────────────────────────────────────────────────────────────

def test_full_pipeline_raw_to_cltv(tmp_path):
    """
    End-to-end test: synthetic transactions → CLTV predictions → segmentation.
    Validates that the full pipeline runs without errors and produces
    correct shapes and types.
    """
    from src.cltv_model import CLTVModel
    from src.evaluation import CLTVEvaluator
    from src.preprocessing import OnlineRetailPreprocessor
    from src.segmentation import CustomerSegmenter

    # Create a synthetic CSV spanning ~12 months for 15 customers
    rng = np.random.default_rng(77)
    rows = []
    for cid in [str(10000 + i) for i in range(15)]:
        n_invoices = rng.integers(3, 10)
        for _ in range(n_invoices):
            offset = rng.integers(0, 365)
            date = (pd.Timestamp("2010-12-01") + pd.Timedelta(days=int(offset))).strftime("%m/%d/%y %H:%M")
            rows.append(
                {
                    "InvoiceNo": f"INV{cid}{offset}",
                    "Quantity": int(rng.integers(1, 10)),
                    "InvoiceDate": date,
                    "UnitPrice": float(rng.uniform(1, 50)),
                    "CustomerID": cid,
                }
            )

    csv_path = tmp_path / "transactions.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    preprocessor = OnlineRetailPreprocessor()
    rfm, modeling_base, summary = preprocessor.prepare(csv_path)

    assert len(rfm) > 0
    assert len(modeling_base) > 0
    assert set(rfm.columns) >= {"customer_id", "frequency", "recency", "T", "monetary_value"}

    model = CLTVModel(fit_method="map")
    model.fit(modeling_base)
    assert model.is_fitted()

    cltv_df = model.predict_cltv(modeling_base, t=90)
    assert len(cltv_df) == len(modeling_base)
    assert "cltv_90d" in cltv_df.columns
    assert (cltv_df["cltv_90d"] >= 0).all()

    segmenter = CustomerSegmenter()
    segmented, threshold = segmenter.segment(cltv_df, cltv_col="cltv_90d")
    assert "marketing_action" in segmented.columns
    assert segmented["marketing_action"].isna().sum() == 0


def test_save_load_cycle(fitted_model, sample_modeling_base, tmp_path):
    """
    Save model to tmp_path, reload, verify predictions are numerically equivalent.
    Tolerance: 1e-4 relative.
    """
    fitted_model.save(tmp_path)

    from src.cltv_model import CLTVModel

    loaded = CLTVModel.load(tmp_path)
    assert loaded.is_fitted()

    orig = fitted_model.predict_purchases(sample_modeling_base, t=90).values
    reloaded = loaded.predict_purchases(sample_modeling_base, t=90).values
    np.testing.assert_allclose(orig, reloaded, rtol=1e-4)


# ── Flask API ─────────────────────────────────────────────────────────────────

@pytest.fixture
def flask_app(fitted_model):
    """
    Build a Flask test client by directly injecting the fitted model.
    This avoids file-system loading and module-level side effects.
    """
    import serve_model as sm

    # Directly inject the fitted model into the server module
    original_model = sm._model
    original_error = sm._model_error
    sm._model = fitted_model
    sm._model_error = None

    sm.app.config["TESTING"] = True
    with sm.app.test_client() as client:
        yield client

    # Restore original state
    sm._model = original_model
    sm._model_error = original_error


def test_flask_api_health_endpoint(flask_app):
    """GET /health must return 200 and status 'ok'."""
    response = flask_app.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"


def test_flask_api_predict_single(flask_app):
    """POST /predict/single must return cltv, segment, and marketing_action."""
    payload = {
        "customer_id": "17850",
        "frequency": 4,
        "recency": 120.0,
        "T": 300.0,
        "monetary_value": 25.50,
        "t": 180,
    }
    response = flask_app.post("/predict/single", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    for key in ("customer_id", "cltv", "cltv_segment", "marketing_action"):
        assert key in data, f"Missing key: {key}"


def test_flask_api_predict_batch(flask_app):
    """POST /predict/batch must return as many results as inputs."""
    payload = [
        {"customer_id": f"C{i:03d}", "frequency": i + 1, "recency": float(10 * i + 5), "T": float(10 * i + 50), "monetary_value": 30.0 + i, "t": 90}
        for i in range(3)
    ]
    response = flask_app.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data, list)
    assert len(data) == len(payload)
