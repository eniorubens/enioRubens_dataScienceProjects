"""Framework-neutral reporting tables for the S4 artifact."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import polars as pl


@dataclass(frozen=True, slots=True)
class S4ReportTables:
    """Hold Polars tables derived from one complete S4 JSON artifact."""

    candidate_summary: pl.DataFrame
    critical_confusions: pl.DataFrame
    per_class: pl.DataFrame
    selection_status: str
    diagnostic_focus_candidate: str


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    """Require a mapping at one artifact path."""

    if not isinstance(value, Mapping):
        raise ValueError(f"S4 artifact field is not an object: {name}")
    return value


def validate_s4_artifact(payload: Mapping[str, Any]) -> None:
    """Validate the complete S4 reporting payload.

    Args:
        payload: Parsed S4 JSON object.

    Raises:
        ValueError: If required fields or the sealed boundary are invalid.
    """

    if payload.get("schema_version") != "s4-experiment-v1":
        raise ValueError("Unexpected S4 artifact schema version")
    if payload.get("complete") is not True:
        raise ValueError("S4 reporting requires a complete artifact")
    if payload.get("claim_boundary") != "NO_TEST_STRESS_OR_MONITOR_ACCESS":
        raise ValueError("S4 artifact claim boundary is invalid")
    if payload.get("selection_status") not in {
        "NO_ELIGIBLE_CHALLENGER",
        "ELIGIBLE_CHALLENGER",
    }:
        raise ValueError("S4 artifact selection status is invalid")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("S4 artifact must contain candidate results")
    names = set()
    for candidate in candidates:
        item = _mapping(candidate, "candidates[]")
        name = str(item.get("name", ""))
        if not name or name in names:
            raise ValueError("S4 candidate names must be unique and non-empty")
        names.add(name)
        metrics = _mapping(item.get("metrics"), f"candidates[{name}].metrics")
        _mapping(metrics.get("per_class"), f"candidates[{name}].metrics.per_class")
        _mapping(item.get("diagnostics"), f"candidates[{name}].diagnostics")
        gates = _mapping(item.get("gates"), f"candidates[{name}].gates")
        if set(gates) != {
            "global_macro_f1",
            "critical_f1",
            "critical_precision",
        }:
            raise ValueError(f"Incomplete S4 gates for {name}")
    recommended = payload.get("recommended_candidate")
    if recommended is not None and recommended not in names:
        raise ValueError("S4 recommended candidate is absent")
    if payload["selection_status"] == "NO_ELIGIBLE_CHALLENGER" and recommended:
        raise ValueError("No-eligible S4 artifact cannot recommend a candidate")


def _load(path: str | Path) -> dict[str, Any]:
    """Load and validate one S4 JSON artifact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("S4 artifact must be a JSON object")
    validate_s4_artifact(payload)
    return payload


def _recommended_candidate(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return the candidate selected by the frozen gates, if one exists."""

    candidates = payload["candidates"]
    recommended = payload.get("recommended_candidate")
    if recommended is None:
        return None
    return next(item for item in candidates if item["name"] == recommended)


def _diagnostic_focus_candidate(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the best critical-class candidate for diagnostic tables only."""

    candidates = payload["candidates"]
    return sorted(
        candidates,
        key=lambda item: (
            -item["metrics"]["per_class"]["debt_credit_management"]["f1"],
            -item["metrics"]["macro_f1"],
            item["name"],
        ),
    )[0]


def build_s4_report_tables(payload: Mapping[str, Any]) -> S4ReportTables:
    """Build Polars reporting tables from one validated S4 payload.

    Args:
        payload: Parsed complete S4 artifact.

    Returns:
        Candidate, critical-confusion, and per-class Polars tables.
    """

    validate_s4_artifact(payload)
    recommended = _recommended_candidate(payload)
    diagnostic_focus = _diagnostic_focus_candidate(payload)
    summary_rows = []
    per_class_rows = []
    for candidate in payload["candidates"]:
        metrics = candidate["metrics"]
        critical = metrics["per_class"]["debt_credit_management"]
        summary_rows.append(
            {
                "candidate": candidate["name"],
                "representation": candidate["representation"],
                "class_weight": candidate["class_weight"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "critical_f1": critical["f1"],
                "critical_precision": critical["precision"],
                "critical_recall": critical["recall"],
                "eligible": candidate["eligible"],
                "recommended": (
                    recommended is not None
                    and candidate["name"] == recommended["name"]
                ),
                "diagnostic_focus": candidate["name"] == diagnostic_focus["name"],
            }
        )
        if candidate["name"] == diagnostic_focus["name"]:
            for label, values in metrics["per_class"].items():
                per_class_rows.append(
                    {
                        "candidate": candidate["name"],
                        "product_family": label,
                        "precision": values["precision"],
                        "recall": values["recall"],
                        "f1": values["f1"],
                        "support": values["support"],
                    }
                )
    diagnostics = diagnostic_focus["diagnostics"]
    confusion_rows = []
    for item in diagnostics["critical_false_negatives"]:
        confusion_rows.append(
            {
                "candidate": diagnostic_focus["name"],
                "diagnostic": "critical_false_negative",
                "true_class": "debt_credit_management",
                "predicted_class": item["predicted_class"],
                "count": item["count"],
                "rate": item["rate"],
            }
        )
    for item in diagnostics["critical_false_positives"]:
        confusion_rows.append(
            {
                "candidate": diagnostic_focus["name"],
                "diagnostic": "critical_false_positive",
                "true_class": item["true_class"],
                "predicted_class": "debt_credit_management",
                "count": item["count"],
                "rate": item["rate"],
            }
        )
    for item in diagnostics["top_confusions"]:
        confusion_rows.append(
            {
                "candidate": diagnostic_focus["name"],
                "diagnostic": "top_global_confusion",
                "true_class": item["true_class"],
                "predicted_class": item["predicted_class"],
                "count": item["count"],
                "rate": item["rate"],
            }
        )
    confusion_schema = {
        "candidate": pl.String,
        "diagnostic": pl.String,
        "true_class": pl.String,
        "predicted_class": pl.String,
        "count": pl.Int64,
        "rate": pl.Float64,
    }
    return S4ReportTables(
        candidate_summary=pl.DataFrame(summary_rows),
        critical_confusions=pl.DataFrame(
            confusion_rows,
            schema=confusion_schema,
        ),
        per_class=pl.DataFrame(per_class_rows),
        selection_status=str(payload["selection_status"]),
        diagnostic_focus_candidate=str(diagnostic_focus["name"]),
    )


def load_s4_report_tables(artifact_path: str | Path) -> S4ReportTables:
    """Load, validate, and tabulate a complete S4 artifact.

    Args:
        artifact_path: S4 JSON artifact path.

    Returns:
        Framework-neutral Polars tables.
    """

    return build_s4_report_tables(_load(artifact_path))
