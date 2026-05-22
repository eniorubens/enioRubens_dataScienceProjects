"""Tests for the FastAPI inference endpoints in serve_model.py."""
import pytest

SAMPLE_CUSTOMER = {
    "tenure":           3,
    "Contract":         "Month-to-month",
    "InternetService":  "Fiber optic",
    "MonthlyCharges":   85.50,
    "TotalCharges":     256.5,
}

LONG_TENURE_CUSTOMER = {
    "tenure":           60,
    "Contract":         "Two year",
    "InternetService":  "DSL",
    "MonthlyCharges":   45.0,
    "TotalCharges":     2700.0,
}


class TestHealthEndpoint:
    def test_returns_200(self, api_client):
        assert api_client.get("/health").status_code == 200

    def test_model_loaded_true(self, api_client):
        data = api_client.get("/health").json()
        assert data["model_loaded"] is True

    def test_status_ok(self, api_client):
        data = api_client.get("/health").json()
        assert data["status"] == "ok"


class TestPredictEndpoint:
    def test_returns_200(self, api_client):
        assert api_client.post("/predict", json=SAMPLE_CUSTOMER).status_code == 200

    def test_response_schema(self, api_client):
        data = api_client.post("/predict", json=SAMPLE_CUSTOMER).json()
        assert {"churn", "churn_probability", "threshold", "risk_level"}.issubset(data)

    def test_churn_is_binary(self, api_client):
        data = api_client.post("/predict", json=SAMPLE_CUSTOMER).json()
        assert data["churn"] in (0, 1)

    def test_probability_in_range(self, api_client):
        data = api_client.post("/predict", json=SAMPLE_CUSTOMER).json()
        assert 0.0 <= data["churn_probability"] <= 1.0

    def test_risk_level_valid(self, api_client):
        data = api_client.post("/predict", json=SAMPLE_CUSTOMER).json()
        assert data["risk_level"] in ("low", "medium", "high")

    def test_custom_threshold_reflected(self, api_client):
        payload = {**SAMPLE_CUSTOMER, "threshold": 0.01}
        data    = api_client.post("/predict", json=payload).json()
        assert data["threshold"] == pytest.approx(0.01)

    def test_threshold_01_predicts_churn(self, api_client):
        payload = {**SAMPLE_CUSTOMER, "threshold": 0.01}
        data    = api_client.post("/predict", json=payload).json()
        assert data["churn"] == 1


class TestBatchPredictEndpoint:
    def test_returns_list(self, api_client):
        resp = api_client.post("/batch_predict", json=[SAMPLE_CUSTOMER, LONG_TENURE_CUSTOMER])
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_batch_length_matches_input(self, api_client):
        resp = api_client.post("/batch_predict", json=[SAMPLE_CUSTOMER, LONG_TENURE_CUSTOMER])
        assert len(resp.json()) == 2

    def test_empty_batch_returns_422(self, api_client):
        assert api_client.post("/batch_predict", json=[]).status_code == 422


class TestModelInfoEndpoint:
    def test_returns_200(self, api_client):
        assert api_client.get("/model/info").status_code == 200

    def test_has_expected_keys(self, api_client):
        data = api_client.get("/model/info").json()
        assert {"estimator", "threshold", "feature_count", "test_metrics"}.issubset(data)

    def test_threshold_value(self, api_client):
        data = api_client.get("/model/info").json()
        assert data["threshold"] == pytest.approx(0.52)
