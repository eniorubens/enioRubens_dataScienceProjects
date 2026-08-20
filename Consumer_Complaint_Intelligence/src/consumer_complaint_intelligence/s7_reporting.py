"""Thin Polars reporting for the cached S7 result and public manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import polars as pl

from .s7 import (
    CRITICAL_CLASS,
    INPUT_LANGUAGE,
    S7_RESULT_SCHEMA,
    SEALED_PARTITIONS,
    validate_s7_manifest,
)
from .temporal_split import MODELED_FAMILIES


@dataclass(frozen=True, slots=True)
class S7ReportTables:
    """Hold the small Polars tables exposed by the S7 notebook."""

    statuses: pl.DataFrame
    calibration_summary: pl.DataFrame
    per_class: pl.DataFrame


def _load_json(path: str | Path) -> dict[str, Any]:
    """Load one UTF-8 JSON object."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("S7 JSON artifact must be an object")
    return payload


def validate_s7_result(payload: Mapping[str, Any]) -> None:
    """Validate the cached result before constructing report tables."""

    if payload.get("schema_version") != S7_RESULT_SCHEMA:
        raise ValueError("Unexpected S7 result schema")
    if payload.get("complete") is not True:
        raise ValueError("S7 reporting requires a complete result")
    if tuple(payload.get("sealed_partitions", ())) != SEALED_PARTITIONS:
        raise ValueError("S7 result sealed boundary is invalid")
    if payload.get("input_language") != INPUT_LANGUAGE:
        raise ValueError("S7 result language is invalid")
    calibration = payload.get("calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError("S7 calibration evidence is missing")
    selected = calibration.get("selected")
    if not isinstance(selected, Mapping):
        raise ValueError("S7 selected threshold evidence is missing")
    metrics = selected.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("S7 calibration metrics are missing")
    per_class = metrics.get("per_class")
    if not isinstance(per_class, Mapping):
        raise ValueError("S7 per-class metrics are missing")
    if set(per_class) != set(MODELED_FAMILIES):
        raise ValueError("S7 per-class labels are invalid")


def build_s7_report_tables(payload: Mapping[str, Any]) -> S7ReportTables:
    """Build status, calibration, and per-class Polars tables.

    Args:
        payload: Complete cached S7 result.

    Returns:
        Three small tables for notebook or application presentation.
    """

    validate_s7_result(payload)
    selected = payload["calibration"]["selected"]
    metrics = selected["metrics"]
    critical = metrics["per_class"][CRITICAL_CLASS]
    calibration_summary = pl.DataFrame([{
        "status": payload["status"],
        "candidate": "linear_svc_c_0_3_balanced",
        "threshold": selected["threshold"],
        "macro_f1": metrics["macro_f1"],
        "critical_f1": critical["f1"],
        "critical_precision": critical["precision"],
        "critical_recall": critical["recall"],
        "gates_passed": selected["gates"]["eligible"],
        "runtime_seconds": payload["runtime_seconds"],
    }])
    per_class = pl.DataFrame([
        {
            "product_family": label,
            "precision": metrics["per_class"][label]["precision"],
            "recall": metrics["per_class"][label]["recall"],
            "f1": metrics["per_class"][label]["f1"],
            "support": metrics["per_class"][label]["support"],
        }
        for label in MODELED_FAMILIES
    ])
    statuses = pl.DataFrame([{
        "status": payload["status"],
        "run_mode": payload["run_mode"],
        "development_only": payload["development_only"],
        "deploy": payload["deploy"],
        "confirmatory": payload["confirmatory"],
        "input_language": payload["input_language"],
        "validation_role": payload["validation_role"],
        "validation_independence": payload["validation_independence"],
    }])
    return S7ReportTables(
        statuses=statuses,
        calibration_summary=calibration_summary,
        per_class=per_class,
    )


def load_s7_report_tables(
    result_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    bundle_path: str | Path | None = None,
    config_path: str | Path | None = None,
) -> S7ReportTables:
    """Load cached S7 tables and validate the manifest when available.

    Args:
        result_path: Cached scientific result path.
        manifest_path: Optional public manifest path.
        bundle_path: Bundle path required when a manifest is supplied.
        config_path: Optional frozen configuration path.

    Returns:
        Polars tables for status and calibration evidence.
    """

    result_file = Path(result_path).expanduser().resolve()
    payload = _load_json(result_file)
    validate_s7_result(payload)
    if manifest_path is not None:
        if bundle_path is None:
            raise ValueError("bundle_path is required with manifest_path")
        manifest_file = Path(manifest_path).expanduser().resolve()
        manifest = _load_json(manifest_file)
        validate_s7_manifest(
            manifest,
            bundle_path,
            result_file,
            config_path,
        )
    return build_s7_report_tables(payload)
