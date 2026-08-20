"""Framework-neutral Polars reporting for the S6 artifact."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import polars as pl

from .temporal_split import MODELED_FAMILIES


CRITICAL_CLASS = "debt_credit_management"


@dataclass(frozen=True, slots=True)
class S6ReportTables:
    """Hold Polars tables and selection fields derived from S6."""

    calibration_summary: pl.DataFrame
    outer_summary: pl.DataFrame
    statuses: pl.DataFrame
    critical_confusions: pl.DataFrame
    per_class: pl.DataFrame
    selection_status: str
    recommended_candidate: str | None
    diagnostic_focus: str | None


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    """Require an object at one artifact path."""

    if not isinstance(value, Mapping):
        raise ValueError(f"S6 artifact field is not an object: {name}")
    return value


def validate_s6_artifact(payload: Mapping[str, Any]) -> None:
    """Validate S6 completeness, statuses, and sealed-boundary claims.

    Args:
        payload: Parsed S6 artifact.

    Raises:
        ValueError: If the payload is incomplete or violates S6 safeguards.
    """

    if payload.get("schema_version") != "s6-calibrated-classical-v1":
        raise ValueError("Unexpected S6 artifact schema version")
    if payload.get("complete") is not True:
        raise ValueError("S6 reporting requires a complete artifact")
    if payload.get("claim_boundary") != "NO_TEST_STRESS_OR_MONITOR_ACCESS":
        raise ValueError("S6 claim boundary is invalid")
    if tuple(payload.get("sealed_partitions", ())) != (
        "test",
        "stress",
        "monitor",
    ):
        raise ValueError("S6 sealed partitions are invalid")
    statuses = {
        "DIAGNOSTIC_ONLY",
        "RECOMMENDED",
        "NO_OUTER_GATE_PASS",
        "NO_ELIGIBLE_CALIBRATED_CANDIDATE",
    }
    if payload.get("selection_status") not in statuses:
        raise ValueError("S6 selection status is invalid")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 5:
        raise ValueError("S6 must contain exactly five candidates")
    names: set[str] = set()
    for candidate in candidates:
        item = _mapping(candidate, "candidates[]")
        name = str(item.get("name", ""))
        if not name or name in names:
            raise ValueError("S6 candidate names must be unique")
        names.add(name)
        calibration = _mapping(item.get("calibration"), f"{name}.calibration")
        _mapping(calibration.get("baseline_threshold_zero"), f"{name}.baseline")
        _mapping(calibration.get("selected"), f"{name}.selected")
        if "runtime_seconds" not in item or "convergence_warnings" not in item:
            raise ValueError(f"S6 runtime evidence is incomplete for {name}")
    outer = _mapping(payload.get("outer_summary"), "outer_summary")
    if payload.get("outer_evaluated_candidate") != outer.get("candidate"):
        raise ValueError("S6 outer candidate declaration is inconsistent")
    if payload.get("run_mode") == "smoke":
        if payload.get("selection_status") != "DIAGNOSTIC_ONLY":
            raise ValueError("S6 smoke cannot promote a candidate")
        if payload.get("recommended_candidate") is not None:
            raise ValueError("S6 smoke cannot recommend a candidate")
    elif payload.get("selection_status") == "RECOMMENDED":
        if payload.get("recommended_candidate") != outer.get("candidate"):
            raise ValueError("S6 recommendation must be the outer candidate")
    elif payload.get("recommended_candidate") is not None:
        raise ValueError("S6 non-recommendation artifact cannot recommend")


def _load(path: str | Path) -> dict[str, Any]:
    """Load and validate one S6 JSON artifact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("S6 artifact must be a JSON object")
    validate_s6_artifact(payload)
    return payload


def _diagnostic_confusions(
    metrics: Mapping[str, Any], candidate: str
) -> list[dict[str, Any]]:
    """Build critical and global confusion rows from stored outer metrics."""

    matrix = metrics["confusion_matrix"]
    critical_index = MODELED_FAMILIES.index(CRITICAL_CLASS)
    total_errors = sum(
        matrix[actual][predicted]
        for actual in range(len(MODELED_FAMILIES))
        for predicted in range(len(MODELED_FAMILIES))
        if actual != predicted
    )
    rows = []
    for index, label in enumerate(MODELED_FAMILIES):
        if index != critical_index and matrix[critical_index][index]:
            rows.append({
                "candidate": candidate,
                "diagnostic": "critical_false_negative",
                "true_class": CRITICAL_CLASS,
                "predicted_class": label,
                "count": matrix[critical_index][index],
            })
        if index != critical_index and matrix[index][critical_index]:
            rows.append({
                "candidate": candidate,
                "diagnostic": "critical_false_positive",
                "true_class": label,
                "predicted_class": CRITICAL_CLASS,
                "count": matrix[index][critical_index],
            })
    for actual, actual_label in enumerate(MODELED_FAMILIES):
        for predicted, predicted_label in enumerate(MODELED_FAMILIES):
            count = matrix[actual][predicted]
            if actual != predicted and count:
                rows.append({
                    "candidate": candidate,
                    "diagnostic": "top_global_confusion",
                    "true_class": actual_label,
                    "predicted_class": predicted_label,
                    "count": count,
                    "error_rate": count / total_errors if total_errors else 0.0,
                })
    return rows


def build_s6_report_tables(payload: Mapping[str, Any]) -> S6ReportTables:
    """Build Polars tables from one validated S6 artifact.

    Args:
        payload: Parsed complete S6 artifact.

    Returns:
        Calibration, outer, confusion, and per-class Polars tables.
    """

    validate_s6_artifact(payload)
    calibration_rows = []
    for candidate in payload["candidates"]:
        calibration = candidate["calibration"]
        selected = calibration["selected"]
        baseline = calibration["baseline_threshold_zero"]
        selected_metrics = selected["metrics"]
        baseline_metrics = baseline["metrics"]
        selected_critical = selected_metrics["per_class"][CRITICAL_CLASS]
        baseline_critical = baseline_metrics["per_class"][CRITICAL_CLASS]
        calibration_rows.append({
            "candidate": candidate["name"],
            "estimator": candidate["estimator"],
            "baseline_macro_f1": baseline_metrics["macro_f1"],
            "baseline_critical_f1": baseline_critical["f1"],
            "baseline_critical_precision": baseline_critical["precision"],
            "selected_threshold": selected["threshold"],
            "selected_macro_f1": selected_metrics["macro_f1"],
            "selected_critical_f1": selected_critical["f1"],
            "selected_critical_precision": selected_critical["precision"],
            "selected_critical_recall": selected_critical["recall"],
            "gate_count": selected["gates"]["gate_count"],
            "eligible": selected["gates"]["eligible"],
            "runtime_seconds": candidate["runtime_seconds"],
        })
    outer = payload["outer_summary"]
    outer_metrics = outer["metrics"]
    outer_critical = outer_metrics["per_class"][CRITICAL_CLASS]
    outer_rows = [{
        "candidate": outer["candidate"],
        "threshold": outer["threshold"],
        "macro_f1": outer_metrics["macro_f1"],
        "weighted_f1": outer_metrics["weighted_f1"],
        "balanced_accuracy": outer_metrics["balanced_accuracy"],
        "critical_f1": outer_critical["f1"],
        "critical_precision": outer_critical["precision"],
        "critical_recall": outer_critical["recall"],
        "eligible": outer["gates"]["eligible"],
    }]
    per_class_rows = []
    for label in MODELED_FAMILIES:
        values = outer_metrics["per_class"][label]
        per_class_rows.append({
            "candidate": outer["candidate"],
            "product_family": label,
            "precision": values["precision"],
            "recall": values["recall"],
            "f1": values["f1"],
            "support": values["support"],
        })
    statuses = pl.DataFrame([{
        "run_mode": payload["run_mode"],
        "selection_status": payload["selection_status"],
        "recommended_candidate": payload.get("recommended_candidate"),
        "diagnostic_focus": payload.get("diagnostic_focus"),
        "outer_evaluated_candidate": payload["outer_evaluated_candidate"],
    }])
    return S6ReportTables(
        calibration_summary=pl.DataFrame(calibration_rows),
        outer_summary=pl.DataFrame(outer_rows),
        statuses=statuses,
        critical_confusions=pl.DataFrame(
            _diagnostic_confusions(outer_metrics, outer["candidate"])
        ),
        per_class=pl.DataFrame(per_class_rows),
        selection_status=str(payload["selection_status"]),
        recommended_candidate=payload.get("recommended_candidate"),
        diagnostic_focus=payload.get("diagnostic_focus"),
    )


def load_s6_report_tables(artifact_path: str | Path) -> S6ReportTables:
    """Load, validate, and tabulate one complete S6 artifact.

    Args:
        artifact_path: Path to the S6 JSON artifact.

    Returns:
        Framework-neutral Polars report tables.
    """

    return build_s6_report_tables(_load(artifact_path))
