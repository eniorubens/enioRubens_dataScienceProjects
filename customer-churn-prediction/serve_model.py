"""FastAPI inference server for the telecom churn prediction pipeline.

Start:
    python serve_model.py
    # or: uvicorn serve_model:app --reload

Endpoints:
    GET  /health          — service + model status
    POST /predict         — single-customer prediction
    POST /batch_predict   — batch predictions
    GET  /model/info      — training metadata and metrics
"""
from __future__ import annotations

import gzip
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path("models/churn_pipeline.pkl.gz")
ARTIFACT: dict[str, Any] = {}


def _load_model() -> None:
    if not MODEL_PATH.exists():
        return
    with gzip.open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    ARTIFACT.update(data)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(
    title="Churn Prediction API",
    description="Telecom customer churn — BalancedRandomForestClassifier",
    version="1.0.0",
    lifespan=lifespan,
)


class CustomerFeatures(BaseModel):
    gender:           Optional[str]   = None
    SeniorCitizen:    Optional[int]   = None
    Partner:          Optional[str]   = None
    Dependents:       Optional[str]   = None
    tenure:           Optional[int]   = None
    PhoneService:     Optional[str]   = None
    MultipleLines:    Optional[str]   = None
    InternetService:  Optional[str]   = None
    OnlineSecurity:   Optional[str]   = None
    OnlineBackup:     Optional[str]   = None
    DeviceProtection: Optional[str]   = None
    TechSupport:      Optional[str]   = None
    StreamingTV:      Optional[str]   = None
    StreamingMovies:  Optional[str]   = None
    Contract:         Optional[str]   = None
    PaperlessBilling: Optional[str]   = None
    PaymentMethod:    Optional[str]   = None
    MonthlyCharges:   Optional[float] = None
    TotalCharges:     Optional[float] = None
    threshold: float = Field(default=0.52, ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    churn:             int
    churn_probability: float
    threshold:         float
    risk_level:        str


def _to_dataframe(features: CustomerFeatures) -> pd.DataFrame:
    data     = features.model_dump(exclude={"threshold"})
    defaults = ARTIFACT["column_defaults"]
    cols     = ARTIFACT["feature_columns"]
    row = {col: (data[col] if data.get(col) is not None else defaults.get(col)) for col in cols}
    return pd.DataFrame([row])


def _risk_level(prob: float) -> str:
    if prob >= 0.7:
        return "high"
    if prob >= 0.4:
        return "medium"
    return "low"


def _check_model() -> None:
    if "pipeline" not in ARTIFACT:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train_model.py first.")


@app.get("/health")
def health():
    loaded = "pipeline" in ARTIFACT
    return {
        "status":      "ok" if loaded else "degraded",
        "model_loaded": loaded,
        "threshold":   ARTIFACT.get("threshold"),
        "trained_at":  ARTIFACT.get("trained_at"),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    _check_model()
    df    = _to_dataframe(customer)
    proba = float(ARTIFACT["pipeline"].predict_proba(df)[0, 1])
    churn = int(proba >= customer.threshold)
    return PredictionResponse(
        churn=churn,
        churn_probability=round(proba, 4),
        threshold=customer.threshold,
        risk_level=_risk_level(proba),
    )


@app.post("/batch_predict")
def batch_predict(customers: list[CustomerFeatures]):
    _check_model()
    if not customers:
        raise HTTPException(status_code=422, detail="Batch must not be empty.")
    results = []
    for customer in customers:
        df    = _to_dataframe(customer)
        proba = float(ARTIFACT["pipeline"].predict_proba(df)[0, 1])
        churn = int(proba >= customer.threshold)
        results.append({
            "churn":             churn,
            "churn_probability": round(proba, 4),
            "threshold":         customer.threshold,
            "risk_level":        _risk_level(proba),
        })
    return results


@app.get("/model/info")
def model_info():
    _check_model()
    return {
        "estimator":     ARTIFACT.get("estimator_name"),
        "feature_count": ARTIFACT.get("feature_count"),
        "churn_rate":    ARTIFACT.get("churn_rate"),
        "threshold":     ARTIFACT.get("threshold"),
        "trained_at":    ARTIFACT.get("trained_at"),
        "test_metrics":  ARTIFACT.get("test_metrics"),
    }


if __name__ == "__main__":
    uvicorn.run("serve_model:app", host="0.0.0.0", port=8000, reload=False)
