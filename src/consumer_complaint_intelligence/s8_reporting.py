"""Small reporting tables for the cached S8 confirmatory artifact."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import polars as pl

from .s8 import (
    CRITICAL_CLASS,
    INPUT_LANGUAGE,
    MODELED_FAMILIES,
    S8_RESULT_SCHEMA,
    SEALED_PARTITIONS,
    validate_s8_manifest,
)


@dataclass(frozen=True, slots=True)
class S8ReportTables:
    """Hold the aggregate tables exposed by the S8 notebook."""

    statuses: pl.DataFrame
    primary_summary: pl.DataFrame
    per_class: pl.DataFrame
    confidence_intervals: pl.DataFrame
    scientific_operational: pl.DataFrame
    audit_counts: pl.DataFrame


def _load_json(path: str | Path) -> dict[str, Any]:
    """Load one cached S8 JSON object."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("S8 result must be a JSON object")
    return payload


def validate_s8_result(payload: Mapping[str, Any]) -> None:
    """Validate the aggregate-only result before building tables."""

    if payload.get("schema_version") != S8_RESULT_SCHEMA:
        raise ValueError("Unexpected S8 result schema")
    if payload.get("complete") is not True:
        raise ValueError("S8 reporting requires a complete result")
    if payload.get("confirmatory") is not True:
        raise ValueError("S8 result must identify its confirmatory nature")
    if payload.get("deploy") is not False:
        raise ValueError("S8 result cannot authorize deployment")
    if tuple(payload.get("remaining_sealed", ())) != SEALED_PARTITIONS:
        raise ValueError("S8 sealed boundary is invalid")
    model = payload.get("model", {})
    if model.get("input_language") != INPUT_LANGUAGE:
        raise ValueError("S8 language contract is invalid")
    if tuple(model.get("classes", ())) != tuple(MODELED_FAMILIES):
        raise ValueError("S8 class order is invalid")
    if model.get("critical_class") != CRITICAL_CLASS:
        raise ValueError("S8 critical class is invalid")
    primary = payload.get("primary")
    operational = payload.get("operational_secondary")
    if not isinstance(primary, Mapping) or not isinstance(operational, Mapping):
        raise ValueError("S8 primary and operational evidence are required")
    if operational.get("excluded_from_decision") is not True:
        raise ValueError("Operational evidence must be excluded from decision")
    metrics = primary.get("metrics", {})
    if set(metrics.get("per_class", {})) != set(MODELED_FAMILIES):
        raise ValueError("S8 per-class labels are invalid")
    serialized = json.dumps(payload, ensure_ascii=False).lower()
    for forbidden in ("narrative", "complaint id", "individual_scores"):
        if forbidden in serialized:
            raise ValueError(f"S8 result contains forbidden value: {forbidden}")


def build_s8_report_tables(payload: Mapping[str, Any]) -> S8ReportTables:
    """Build six compact Polars tables from aggregate evidence."""

    validate_s8_result(payload)
    primary = payload["primary"]
    primary_metrics = primary["metrics"]
    critical = primary_metrics["per_class"][CRITICAL_CLASS]
    gates = primary["gates"]
    primary_summary = pl.DataFrame([{
        "status": payload["status"],
        "confirmed": payload["confirmed"],
        "macro_f1": primary_metrics["macro_f1"],
        "critical_f1": critical["f1"],
        "critical_precision": critical["precision"],
        "critical_recall": critical["recall"],
        "gate_count": gates["gate_count"],
        "gates_passed": gates["passed"],
    }])
    per_class = pl.DataFrame([
        {
            "product_family": label,
            **primary_metrics["per_class"][label],
        }
        for label in MODELED_FAMILIES
    ])
    intervals = primary["confidence_intervals"]["intervals"]
    confidence_intervals = pl.DataFrame([
        {"metric": label, **intervals[label]}
        for label in ("macro_f1", "critical_f1", "critical_precision")
    ])
    operational = payload["operational_secondary"]["metrics"]
    scientific_operational = pl.DataFrame([
        {
            "view": "scientific_primary",
            "row_count": primary_metrics["row_count"],
            "macro_f1": primary_metrics["macro_f1"],
            "critical_f1": primary_metrics["critical_f1"],
            "critical_precision": primary_metrics["critical_precision"],
            "included_in_decision": True,
        },
        {
            "view": "operational_secondary",
            "row_count": operational["row_count"],
            "macro_f1": operational["macro_f1"],
            "critical_f1": operational["critical_f1"],
            "critical_precision": operational["critical_precision"],
            "included_in_decision": False,
        },
    ])
    counts = payload["scope_counts"]
    audit_counts = pl.DataFrame([
        {"count": label, "value": value}
        for label, value in counts.items()
        if isinstance(value, (int, float, str, bool))
    ])
    statuses = pl.DataFrame([{
        "status": payload["status"],
        "test_opened": payload["test_opened"],
        "confirmatory": payload["confirmatory"],
        "confirmed": payload["confirmed"],
        "deploy": payload["deploy"],
        "input_language": payload["model"]["input_language"],
        "remaining_sealed": ",".join(payload["remaining_sealed"]),
    }])
    return S8ReportTables(
        statuses=statuses,
        primary_summary=primary_summary,
        per_class=per_class,
        confidence_intervals=confidence_intervals,
        scientific_operational=scientific_operational,
        audit_counts=audit_counts,
    )


def load_s8_report_tables(
    result_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    config_path: str | Path | None = None,
) -> S8ReportTables:
    """Load cached S8 tables and validate its manifest when supplied.

    Args:
        result_path: Aggregate S8 result JSON path.
        manifest_path: Full-run manifest path, when available.
        config_path: Frozen S8 protocol path required with a manifest.

    Returns:
        Compact Polars tables for notebook or application presentation.
    """

    result_file = Path(result_path).expanduser().resolve()
    payload = _load_json(result_file)
    if manifest_path is not None:
        if config_path is None:
            raise ValueError("config_path is required with manifest_path")
        manifest = _load_json(manifest_path)
        validate_s8_manifest(manifest, result_file, config_path)
    return build_s8_report_tables(payload)
