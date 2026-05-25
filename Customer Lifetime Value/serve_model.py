"""
serve_model.py — Flask API for CLTV scoring.

Usage
-----
    python serve_model.py
    python serve_model.py --models models/ --port 5000

Requires trained models in models/ (run train_model.py first).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from flask import Flask, jsonify, request

# Allow MODEL_DIR to be patched by tests before module-level load
MODEL_DIR = os.environ.get("CLTV_MODEL_DIR", "models/")

app = Flask(__name__)

# ── State ─────────────────────────────────────────────────────────────────────
_model = None
_model_error: str | None = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("serve_model")


def _load_model(model_dir: str) -> None:
    """Attempt to load models from model_dir on startup."""
    global _model, _model_error
    try:
        from src.cltv_model import CLTVModel

        _model = CLTVModel.load(model_dir)
        logger.info("Model loaded from %s", model_dir)
        _model_error = None
    except FileNotFoundError as exc:
        _model = None
        _model_error = str(exc)
        logger.error("Model load failed: %s", exc)
    except Exception as exc:
        _model = None
        _model_error = str(exc)
        logger.error("Unexpected error loading model: %s", exc)


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_customer_payload(data: dict) -> str | None:
    """Return an error message string, or None if valid."""
    required = {"customer_id", "frequency", "recency", "T", "monetary_value"}
    missing = required - set(data.keys())
    if missing:
        return f"Missing fields: {sorted(missing)}"
    if data["frequency"] < 0:
        return "frequency must be >= 0"
    if data["recency"] < 0:
        return "recency must be >= 0"
    if data["T"] < data["recency"]:
        return "T must be >= recency"
    if data.get("monetary_value", 1) <= 0 and data["frequency"] > 0:
        return "monetary_value must be > 0 for repeat customers"
    t = data.get("t", 180)
    if not (1 <= t <= 730):
        return "t must be between 1 and 730 days"
    return None


def _predict_single_customer(data: dict) -> dict:
    """Run prediction for a single customer payload dict."""
    import pandas as pd
    from src.segmentation import CustomerSegmenter

    rfm = pd.DataFrame(
        [
            {
                "customer_id": str(data["customer_id"]),
                "frequency": float(data["frequency"]),
                "recency": float(data["recency"]),
                "T": float(data["T"]),
                "monetary_value": float(data.get("monetary_value", 0.0)),
            }
        ]
    )
    t = int(data.get("t", 180))
    cltv_col = f"cltv_{t}d"

    cltv_df = _model.predict_cltv(rfm, t=t)

    row = cltv_df.iloc[0]
    segmenter = CustomerSegmenter()
    seg_df, _ = segmenter.segment(cltv_df, cltv_col=cltv_col)
    seg_row = seg_df.iloc[0]

    expected_spend = None if pd.isna(row.get("expected_spend", None)) else float(row["expected_spend"])

    return {
        "customer_id": str(data["customer_id"]),
        "predicted_purchases": float(row["predicted_purchases"]),
        "probability_alive": float(row["probability_alive"]),
        "expected_spend": expected_spend,
        "cltv": float(row[cltv_col]),
        "cltv_segment": str(seg_row["cltv_segment"]),
        "marketing_action": str(seg_row["marketing_action"]),
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Health check — always returns 200 with model status."""
    loaded = _model is not None and _model.is_fitted()
    payload = {"status": "ok", "model_loaded": loaded}
    if loaded and _model.metadata:
        payload["fit_date"] = _model.metadata.get("fit_date", "unknown")
        payload["fit_method"] = _model.metadata.get("fit_method", "unknown")
    return jsonify(payload), 200


@app.route("/model/info", methods=["GET"])
def model_info():
    """Return model metadata and parameters."""
    if _model is None:
        return jsonify({"error": _model_error or "Model not loaded.", "status": 503}), 503

    info = dict(_model.metadata)
    return jsonify(info), 200


@app.route("/predict/single", methods=["POST"])
def predict_single():
    """Score a single customer."""
    if _model is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first.", "status": 503}), 503

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "Empty request body.", "status": 400}), 400

    error = _validate_customer_payload(data)
    if error:
        return jsonify({"error": error, "status": 400}), 400

    try:
        result = _predict_single_customer(data)
        return jsonify(result), 200
    except Exception as exc:
        logger.exception("Prediction failed")
        return jsonify({"error": str(exc), "status": 500}), 500


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """Score a list of customers."""
    if _model is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first.", "status": 503}), 503

    records = request.get_json(force=True)
    if not isinstance(records, list):
        return jsonify({"error": "Request body must be a JSON array.", "status": 400}), 400
    if len(records) == 0:
        return jsonify([]), 200

    results = []
    for i, rec in enumerate(records):
        error = _validate_customer_payload(rec)
        if error:
            return jsonify({"error": f"Record {i}: {error}", "status": 400}), 400
        try:
            results.append(_predict_single_customer(rec))
        except Exception as exc:
            logger.exception("Batch prediction failed at record %d", i)
            return jsonify({"error": f"Record {i}: {exc}", "status": 500}), 500

    return jsonify(results), 200


# ── Startup ───────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve CLTV predictions via Flask.")
    parser.add_argument("--models", default=MODEL_DIR, help="Directory with trained models")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="0.0.0.0")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    MODEL_DIR = args.models
    _load_model(MODEL_DIR)
    app.run(host=args.host, port=args.port, debug=False)
else:
    # When imported (e.g., by tests), use the module-level MODEL_DIR
    _load_model(MODEL_DIR)
