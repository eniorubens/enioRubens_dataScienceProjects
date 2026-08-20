"""Framework-neutral reporting tables for the S5 artifact."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import polars as pl

from .temporal_split import MODELED_FAMILIES


@dataclass(frozen=True, slots=True)
class S5ReportTables:
    """Hold Polars tables derived from one complete S5 artifact."""

    candidate_summary: pl.DataFrame
    critical_confusions: pl.DataFrame
    per_class: pl.DataFrame
    selection_status: str
    recommended_candidate: str | None
    reference_status: str
    reference_parity: pl.DataFrame
    deferred_candidates: pl.DataFrame


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    """Require a mapping at one artifact path."""

    if not isinstance(value, Mapping):
        raise ValueError(
            f"S5 artifact field is not an object: {name}"
        )
    return value


def validate_s5_artifact(payload: Mapping[str, Any]) -> None:
    """Validate a complete S5 payload and its claim boundary.

    Args:
        payload: Parsed S5 JSON object.

    Raises:
        ValueError: If required fields or selection safeguards are invalid.
    """

    if payload.get("schema_version") != "s5-estimator-benchmark-v1":
        raise ValueError("Unexpected S5 artifact schema version")
    if payload.get("complete") is not True:
        raise ValueError("S5 reporting requires a complete artifact")
    if payload.get("claim_boundary") != "NO_TEST_STRESS_OR_MONITOR_ACCESS":
        raise ValueError("S5 artifact claim boundary is invalid")
    allowed = {
        "DIAGNOSTIC_ONLY",
        "REFERENCE_NOT_VERIFIED",
        "NO_ELIGIBLE_ESTIMATOR",
        "ELIGIBLE_ESTIMATOR",
    }
    if payload.get("selection_status") not in allowed:
        raise ValueError("S5 artifact selection status is invalid")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("S5 artifact must contain candidate results")
    names: set[str] = set()
    for candidate in candidates:
        item = _mapping(candidate, "candidates[]")
        name = str(item.get("name", ""))
        if not name or name in names:
            raise ValueError("S5 candidate names must be unique and non-empty")
        names.add(name)
        metrics = _mapping(item.get("metrics"), f"candidates[{name}].metrics")
        _mapping(metrics.get("per_class"), f"candidates[{name}].metrics.per_class")
        _mapping(item.get("diagnostics"), f"candidates[{name}].diagnostics")
        gates = _mapping(item.get("gates"), f"candidates[{name}].gates")
        if set(gates) != {"global_macro_f1", "critical_f1", "critical_precision"}:
            raise ValueError(f"Incomplete S5 gates for {name}")
        if "runtime_seconds" not in item or "convergence_warnings" not in item:
            raise ValueError(f"S5 runtime evidence is incomplete for {name}")
    deferred = _mapping(payload.get("deferred_candidates"), "deferred_candidates")
    if "logistic_regression_saga" not in deferred:
        raise ValueError("S5 deferred LogisticRegression(saga) is missing")
    parity = _mapping(
        payload.get("reference_reproduction"),
        "reference_reproduction",
    )
    if payload["selection_status"] == "ELIGIBLE_ESTIMATOR":
        if parity.get("passed") is not True:
            raise ValueError(
                "S5 cannot recommend an estimator without reference parity"
            )
        if payload.get("recommended_candidate") not in names:
            raise ValueError("S5 recommended candidate is absent")
    elif payload.get("recommended_candidate") is not None:
        raise ValueError("S5 non-selection artifact cannot recommend a candidate")


def _load(path: str | Path) -> dict[str, Any]:
    """Load and validate one S5 artifact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("S5 artifact must be a JSON object")
    validate_s5_artifact(payload)
    return payload


def _diagnostic_focus(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the strongest critical-class candidate for diagnostics."""

    return sorted(
        payload["candidates"],
        key=lambda item: (
            -item["metrics"]["per_class"]["debt_credit_management"]["f1"],
            -item["metrics"]["macro_f1"],
            item["name"],
        ),
    )[0]


def build_s5_report_tables(payload: Mapping[str, Any]) -> S5ReportTables:
    """Build Polars tables from one validated S5 payload.

    Args:
        payload: Parsed complete S5 artifact.

    Returns:
        Candidate, confusion, per-class, and deferred-candidate tables.
    """

    validate_s5_artifact(payload)
    focus = _diagnostic_focus(payload)
    summary_rows = []
    per_class_rows = []
    for candidate in payload["candidates"]:
        metrics = candidate["metrics"]
        critical = metrics["per_class"]["debt_credit_management"]
        summary_rows.append({
            "candidate": candidate["name"],
            "estimator": candidate["estimator"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "critical_f1": critical["f1"],
            "critical_precision": critical["precision"],
            "critical_recall": critical["recall"],
            "runtime_seconds": candidate["runtime_seconds"],
            "convergence_warning_count": len(candidate["convergence_warnings"]),
            "eligible": candidate["eligible"],
            "recommended": candidate["name"] == payload.get("recommended_candidate"),
            "diagnostic_focus": candidate["name"] == focus["name"],
        })
        if candidate["name"] == focus["name"]:
            for label in MODELED_FAMILIES:
                values = metrics["per_class"][label]
                per_class_rows.append({
                    "candidate": candidate["name"],
                    "product_family": label,
                    "precision": values["precision"],
                    "recall": values["recall"],
                    "f1": values["f1"],
                    "support": values["support"],
                })
    diagnostics = focus["diagnostics"]
    confusion_rows = []
    for item in diagnostics["critical_false_negatives"]:
        confusion_rows.append({
            "candidate": focus["name"],
            "diagnostic": "critical_false_negative",
            "true_class": "debt_credit_management",
            "predicted_class": item["predicted_class"],
            "count": item["count"],
            "rate": item["rate"],
        })
    for item in diagnostics["critical_false_positives"]:
        confusion_rows.append({
            "candidate": focus["name"],
            "diagnostic": "critical_false_positive",
            "true_class": item["true_class"],
            "predicted_class": "debt_credit_management",
            "count": item["count"],
            "rate": item["rate"],
        })
    for item in diagnostics["top_confusions"]:
        confusion_rows.append({
            "candidate": focus["name"],
            "diagnostic": "top_global_confusion",
            "true_class": item["true_class"],
            "predicted_class": item["predicted_class"],
            "count": item["count"],
            "rate": item["rate"],
        })
    reference = payload["reference_reproduction"]
    parity_rows = []
    reference_values = reference.get("reference", {})
    actual_values = reference.get("actual", {})
    delta_values = reference.get("deltas", {})
    tolerance_value = reference.get("tolerance")
    reference_status = str(reference.get("status", "NOT_CHECKED"))
    for metric in (
        "macro_f1",
        "critical_precision",
        "critical_recall",
        "critical_f1",
    ):
        reference_metric = reference_values.get(metric)
        actual_metric = actual_values.get(metric)
        delta_metric = delta_values.get(metric)
        tolerance = (
            float(tolerance_value)
            if tolerance_value is not None
            else None
        )
        delta = float(delta_metric) if delta_metric is not None else None
        passed = (
            delta is not None
            and tolerance is not None
            and abs(delta) <= tolerance
        )
        status = (
            "PASSED" if passed
            else "FAILED" if delta is not None and tolerance is not None
            else reference_status
        )
        parity_rows.append({
            "metric": metric,
            "reference": (
                float(reference_metric) if reference_metric is not None else None
            ),
            "actual": float(actual_metric) if actual_metric is not None else None,
            "delta": delta,
            "tolerance": tolerance,
            "passed": passed,
            "status": status,
        })
    deferred_rows = [
        {"candidate": name, **dict(values)}
        for name, values in payload["deferred_candidates"].items()
    ]
    return S5ReportTables(
        candidate_summary=pl.DataFrame(summary_rows),
        critical_confusions=pl.DataFrame(confusion_rows),
        per_class=pl.DataFrame(per_class_rows),
        selection_status=str(payload["selection_status"]),
        recommended_candidate=payload.get("recommended_candidate"),
        reference_status=str(reference["status"]),
        reference_parity=pl.DataFrame(parity_rows),
        deferred_candidates=pl.DataFrame(deferred_rows),
    )


def load_s5_report_tables(artifact_path: str | Path) -> S5ReportTables:
    """Load, validate, and tabulate a complete S5 artifact.

    Args:
        artifact_path: Path to the complete S5 JSON artifact.

    Returns:
        Framework-neutral Polars report tables.
    """

    return build_s5_report_tables(_load(artifact_path))
