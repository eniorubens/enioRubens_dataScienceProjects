"""Shared pytest fixtures for all test modules."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churn_project.data import read_telecom_data, split_telecom_dataset

DATA_PATH = Path(__file__).parent.parent / "dataset" / "telco.csv"


def _make_synthetic_telco(n: int = 400, seed: int = 42) -> pd.DataFrame:
    """Minimal Telco-like dataset for CI — same schema as the real CSV."""
    rng      = np.random.default_rng(seed)
    internet = rng.choice([True, False], size=n, p=[0.80, 0.20])
    phone    = rng.choice([True, False], size=n, p=[0.90, 0.10])
    tenure   = rng.integers(0, 73, size=n)
    monthly  = rng.uniform(18.0, 120.0, size=n).round(2)
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.25, 0.20]
    )
    churn = rng.uniform(size=n) < np.where(contract == "Month-to-month", 0.40, 0.10)

    def inet_svc(has, yes_no=True):
        opts = (["Yes", "No"] if yes_no else ["DSL", "Fiber optic"])
        return [rng.choice(opts) if h else ("No internet service" if yes_no else "No")
                for h in has]

    return pd.DataFrame({
        "customerID":       [f"{i:04d}-SYNTH" for i in range(n)],
        "gender":           rng.choice(["Male", "Female"], size=n),
        "SeniorCitizen":    rng.choice([0, 1], size=n, p=[0.84, 0.16]),
        "Partner":          rng.choice(["Yes", "No"], size=n),
        "Dependents":       rng.choice(["Yes", "No"], size=n),
        "tenure":           tenure,
        "PhoneService":     np.where(phone, "Yes", "No"),
        "MultipleLines":    [rng.choice(["Yes", "No"]) if p else "No phone service"
                             for p in phone],
        "InternetService":  inet_svc(internet, yes_no=False),
        "OnlineSecurity":   inet_svc(internet),
        "OnlineBackup":     inet_svc(internet),
        "DeviceProtection": inet_svc(internet),
        "TechSupport":      inet_svc(internet),
        "StreamingTV":      inet_svc(internet),
        "StreamingMovies":  inet_svc(internet),
        "Contract":         contract,
        "PaperlessBilling": rng.choice(["Yes", "No"], size=n),
        "PaymentMethod":    rng.choice(
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)", "Credit card (automatic)"], size=n
        ),
        "MonthlyCharges":   monthly,
        "TotalCharges":     (monthly * tenure).round(2),
        "Churn":            np.where(churn, "Yes", "No"),
    })


@pytest.fixture(scope="session")
def df():
    if DATA_PATH.exists():
        return read_telecom_data(str(DATA_PATH))
    return _make_synthetic_telco()


@pytest.fixture(scope="session")
def splits(df):
    return split_telecom_dataset(df, random_state=738)


@pytest.fixture(scope="session")
def trained_pipeline(splits):
    """Fast pipeline (n_estimators=10) used for shape/behaviour tests."""
    import category_encoders as ce
    from imblearn.ensemble import BalancedRandomForestClassifier
    from sklearn.feature_selection import SelectFromModel
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MaxAbsScaler, Normalizer

    brfc = BalancedRandomForestClassifier(n_estimators=10, max_depth=5, random_state=738)
    pipeline = Pipeline([
        ("encoder",    ce.SumEncoder()),
        ("normalizer", Normalizer()),
        ("scaler",     MaxAbsScaler()),
        ("selector",   SelectFromModel(brfc, threshold="mean")),
        ("clf",        BalancedRandomForestClassifier(
            n_estimators=10, max_depth=5, random_state=738
        )),
    ])
    pipeline.fit(splits["train_features"], splits["train_labels"])
    return pipeline


@pytest.fixture(scope="session")
def artifact(splits, trained_pipeline):
    X_train = splits["train_features"]
    y_train = splits["train_labels"]

    col_defaults = {}
    for col in X_train.columns:
        if X_train[col].dtype == object:
            col_defaults[col] = X_train[col].mode()[0]
        else:
            col_defaults[col] = float(X_train[col].median())

    return {
        "pipeline":        trained_pipeline,
        "feature_columns": list(X_train.columns),
        "column_defaults": col_defaults,
        "threshold":       0.52,
        "trained_at":      "2026-01-01T00:00:00+00:00",
        "feature_count":   int(X_train.shape[1]),
        "churn_rate":      float(y_train.mean()),
        "estimator_name":  "BalancedRandomForestClassifier",
        "test_metrics":    {"recall_macro": 0.764, "roc_auc": 0.835},
    }


@pytest.fixture
def api_client(monkeypatch, artifact):
    import serve_model
    from fastapi.testclient import TestClient

    monkeypatch.setattr(serve_model, "_load_model", lambda: None)
    serve_model.ARTIFACT.clear()
    serve_model.ARTIFACT.update(artifact)
    with TestClient(serve_model.app) as client:
        yield client
