"""Build framework-neutral tables from the cached S3 evidence artifact."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl


@dataclass(frozen=True)
class S3EvidenceTables:
    """Hold the S3 learning curve and final per-class evidence tables."""

    curve: pl.DataFrame
    per_class: pl.DataFrame


def build_s3_evidence_tables(
    payload: Mapping[str, Any],
) -> S3EvidenceTables:
    """Build typed Polars tables from a completed S3 artifact payload.

    Args:
        payload: Parsed JSON object from the completed S3 artifact.

    Returns:
        Learning-curve and final per-class evidence tables.

    Raises:
        ValueError: If the payload is incomplete or has an invalid structure.
    """

    _validate_payload(payload)
    points = payload["points"]
    curve_rows = [
        _build_curve_row(fraction, point) for fraction, point in points.items()
    ]
    curve = pl.DataFrame(curve_rows).sort("fraction")

    final_key = max(points, key=_fraction_value)
    final_per_class = _require_mapping(
        points[final_key]["sgd_logistic"]["scientific"]["per_class"],
        f"points[{final_key!r}].sgd_logistic.scientific.per_class",
    )
    if not final_per_class:
        raise ValueError("Final S3 point has an empty scientific per_class mapping")

    class_rows = []
    for label, metrics in final_per_class.items():
        metric_mapping = _require_mapping(
            metrics,
            f"points[{final_key!r}].scientific.per_class[{label!r}]",
        )
        missing = {
            name
            for name in ("precision", "recall", "f1", "support")
            if name not in metric_mapping
        }
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(
                f"Missing per-class metric(s) for {label!r}: {names}"
            )
        class_rows.append(
            {
                "product_family": label,
                "precision": metric_mapping["precision"],
                "recall": metric_mapping["recall"],
                "f1": metric_mapping["f1"],
                "support": metric_mapping["support"],
            }
        )

    per_class = pl.DataFrame(class_rows).sort("product_family")
    return S3EvidenceTables(curve=curve, per_class=per_class)


def load_s3_evidence_tables(artifact_path: str | Path) -> S3EvidenceTables:
    """Load and transform a completed S3 evidence JSON artifact.

    Args:
        artifact_path: Path to the JSON artifact produced by S3.

    Returns:
        Learning-curve and final per-class evidence tables.

    Raises:
        FileNotFoundError: If the artifact path does not exist or is not a file.
        ValueError: If the file is not valid JSON or fails S3 validation.
    """

    path = Path(artifact_path)
    if not path.exists():
        raise FileNotFoundError(f"S3 evidence artifact does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"S3 evidence artifact is not a file: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Could not read S3 evidence artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid S3 evidence JSON in {path}") from exc

    if not isinstance(payload, Mapping):
        raise ValueError("S3 evidence JSON must contain an object at the root")
    return build_s3_evidence_tables(payload)


def _validate_payload(payload: Mapping[str, Any]) -> None:
    """Validate the structural contract required by S3 reporting."""

    if payload.get("complete") is not True:
        raise ValueError("S3 evidence artifact must have complete=True")
    points = _require_mapping(payload.get("points"), "points")
    if not points:
        raise ValueError("S3 evidence artifact must contain non-empty points")

    for fraction, point in points.items():
        point_mapping = _require_mapping(point, f"points[{fraction!r}]")
        model = _require_mapping(
            point_mapping.get("sgd_logistic"),
            f"points[{fraction!r}].sgd_logistic",
        )
        scientific = _require_mapping(
            model.get("scientific"),
            f"points[{fraction!r}].sgd_logistic.scientific",
        )
        _require_mapping(
            scientific.get("per_class"),
            f"points[{fraction!r}].sgd_logistic.scientific.per_class",
        )
        _fraction_value(fraction)
        if "train_groups" not in point_mapping:
            raise ValueError(f"Missing train_groups for points[{fraction!r}]")


def _build_curve_row(fraction: Any, point: Any) -> dict[str, Any]:
    """Extract one learning-curve row from an S3 point."""

    point_mapping = _require_mapping(point, f"points[{fraction!r}]")
    model = _require_mapping(
        point_mapping["sgd_logistic"],
        f"points[{fraction!r}].sgd_logistic",
    )
    scientific = _require_mapping(
        model["scientific"],
        f"points[{fraction!r}].sgd_logistic.scientific",
    )
    for name in ("macro_f1", "balanced_accuracy"):
        if name not in scientific:
            raise ValueError(
                f"Missing scientific metric {name!r} for points[{fraction!r}]"
            )
    per_class = _require_mapping(
        scientific["per_class"],
        f"points[{fraction!r}].sgd_logistic.scientific.per_class",
    )
    critical = _require_mapping(
        per_class.get("debt_credit_management"),
        f"points[{fraction!r}].per_class[debt_credit_management]",
    )
    if "f1" not in critical:
        raise ValueError(
            f"Missing debt_credit_management f1 for points[{fraction!r}]"
        )
    operational = model.get("operational_all_text") or {}
    operational = _require_mapping(
        operational,
        f"points[{fraction!r}].sgd_logistic.operational_all_text",
    )
    return {
        "fraction": _fraction_value(fraction),
        "train_groups": point_mapping["train_groups"],
        "macro_f1": scientific["macro_f1"],
        "balanced_accuracy": scientific["balanced_accuracy"],
        "debt_credit_management_f1": critical["f1"],
        "operational_macro_f1": operational.get("macro_f1"),
    }


def _fraction_value(value: Any) -> float:
    """Convert a point key to a finite numeric fraction."""

    try:
        fraction = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid S3 point fraction: {value!r}") from exc
    if not fraction == fraction or fraction in (float("inf"), float("-inf")):
        raise ValueError(f"Invalid non-finite S3 point fraction: {value!r}")
    return fraction


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    """Return a mapping or raise a contextual validation error."""

    if not isinstance(value, Mapping):
        raise ValueError(f"S3 evidence field {label} must be an object")
    return value
