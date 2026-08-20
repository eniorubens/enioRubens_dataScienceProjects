"""Frozen V2 development contract and framework-neutral evaluation helpers."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .s6 import CRITICAL_CLASS, MODELED_FAMILIES


DEFAULT_CONFIG = "config/v2_development_protocol.json"
INPUT_LANGUAGE = "en-US"
ALLOWED_PARTITIONS = ("train", "validation")
FORBIDDEN_PARTITIONS = ("test", "stress", "monitor")
V2_SCHEMA_VERSION = "v2-post-confirmation-development-v1-1"


def _expected_protocol() -> dict[str, Any]:
    """Return the complete frozen V2 contract for strict validation."""

    return {
        "schema_version": V2_SCHEMA_VERSION,
        "cycle_id": "consumer-complaint-intelligence-v2",
        "stage": "V2.1-D0",
        "approval": {
            "status": "FROZEN_FOR_V2_DEVELOPMENT",
            "approved_on": "2026-08-17",
            "approved_by": "user",
        },
        "revision": {
            "revision_id": "V2.1",
            "supersedes_schema_version": (
                "v2-post-confirmation-development-v1"
            ),
            "adr": "docs/ADR-011-v2-1-calibration-window-revision.md",
            "reason": (
                "d1_degenerate_null_all_candidates_selected_zero_override_sentinel"
            ),
            "change": "move_stage_a_calibration_to_a_later_harder_window",
            "final_development_iteration": True,
            "zero_override_candidate_is_not_selectable": True,
        },
        "objective": {
            "primary": (
                "raise critical-class recall while preserving "
                "all three scientific gates"
            ),
            "critical_class": CRITICAL_CLASS,
            "input_language": INPUT_LANGUAGE,
            "deployment_authorized": False,
        },
        "baseline_s8": {
            "status": "NOT_CONFIRMED",
            "manifest": {
                "path": "config/s8_results.json",
                "sha256": (
                    "E9301499F73BC0619A477C6925BE35E1ED67928D308748E3755B901D659F455A"
                ),
                "size_bytes": 2298,
            },
            "result": {
                "path": "temp/s8/s8_results.json",
                "sha256": (
                    "B72927A7D44BDFDF37536B586EB41A16ECBAA1B2A53416BB293FC669C9002D6A"
                ),
                "size_bytes": 11915,
            },
            "aggregate_metrics": {
                "macro_f1": 0.7182138908021497,
                "critical_precision": 0.4046153846153846,
                "critical_recall": 0.18920863309352517,
                "critical_f1": 0.25784313725490193,
            },
            "permitted_use": "motivation_and_immutable_baseline_only",
            "forbidden_use": [
                "candidate_selection",
                "threshold_selection",
                "hyperparameter_selection",
                "test_narrative_inspection",
                "test_prediction_inspection",
            ],
        },
        "development_windows": {
            "inner_fit": {
                "partition": "train",
                "start": "2023-08-01",
                "end": "2024-06-30",
            },
            "inner_calibration": {
                "partition": "validation",
                "start": "2024-07-01",
                "end": "2024-09-30",
            },
            "outer_evaluation": {
                "partition": "validation",
                "start": "2024-10-01",
                "end": "2024-12-31",
            },
        },
        "data_boundary": {
            "scientific_cache": "temp/s3/scientific.parquet",
            "allowed_partitions": list(ALLOWED_PARTITIONS),
            "forbidden_partitions": list(FORBIDDEN_PARTITIONS),
            "future_confirmation": {
                "partition": "stress",
                "start": "2025-07-01",
                "end": "2025-12-31",
                "status": "SEALED",
            },
            "future_monitoring": {
                "partition": "monitor",
                "start": "2026-01-01",
                "end": "2026-12-31",
                "status": "SEALED",
            },
        },
        "architecture": {
            "type": "hierarchical_critical_override",
            "stage_a": "binary_critical_detector",
            "stage_b": "frozen_s7_multiclass_fallback",
            "decision_rule": (
                "stage_a_overrides_stage_b_only_at_a_development_calibrated_threshold"
            ),
            "stage_b_bundle": {
                "path": "artifacts/s7/consumer_complaint_classifier_s7.joblib",
                "sha256": (
                    "5B302E1D958EC99A2ECB3C8B2DA5E5F52F613C5112153A76086FC5ACB4931049"
                ),
                "size_bytes": 3268327,
                "fit_scope": {
                    "partition": "train",
                    "start": "2023-08-01",
                    "end": "2024-06-30",
                },
            },
        },
        "classical_challenge": {
            "representations": [
                "word_tfidf_1_2_40000",
                "word_char_tfidf_union_40000_60000",
            ],
            "estimator": {
                "class": "LinearSVC",
                "C": [0.1, 0.3, 1.0],
                "tol": 0.0001,
                "max_iter": 5000,
                "dual": "auto",
                "random_state": 42,
            },
            "balance_strategies": {
                "weighted_full": {"class_weight": "balanced"},
                "random_over": {
                    "sampling_strategy": [0.05, 0.1, 0.2],
                    "shrinkage": None,
                    "class_weight": None,
                    "sparse_materialization": (
                        "row_index_multiplicity_as_integer_sample_weight"
                    ),
                },
                "hard_negative": {
                    "positive_policy": "all_development_critical_groups",
                    "hard_negative_source": (
                        "inner_fit_out_of_fold_predictions_only"
                    ),
                    "oof": {
                        "splitter": "StratifiedKFold",
                        "n_splits": 3,
                        "shuffle": True,
                        "random_state": 42,
                        "pipeline_refit_per_fold": True,
                        "representation": "word_tfidf_1_2_40000",
                        "C": 0.3,
                        "class_weight": "balanced",
                    },
                    "hard_negative_per_positive": 10,
                    "background_negative_per_positive": 5,
                    "candidate_class_weight": None,
                },
            },
            "resampling_scope": "fit_fold_only",
            "group_identity_required": True,
            "maximum_completed_candidates": 30,
        },
        "transformer_challenge": {
            "status": "DEFERRED_UNTIL_CLASSICAL_WINNER",
            "task": "binary_critical_detector",
            "maximum_models": 1,
            "requires_separate_frozen_execution_config": True,
        },
        "selection": {
            "scientific_gates": {
                "global_macro_f1_min": 0.69,
                "critical_f1_min": 0.2715,
                "critical_precision_min": 0.2,
            },
            "development_safety_margins": {
                "global_macro_f1_min": 0.7,
                "critical_f1_min": 0.29,
                "critical_precision_min": 0.22,
            },
            "required_margin_gate_count": 3,
            "ranking_after_margin": [
                "critical_f1_desc",
                "global_macro_f1_desc",
                "critical_precision_desc",
                "lower_runtime_asc",
            ],
            "threshold_source": "inner_calibration_only",
            "outer_evaluation_can_select_candidate": True,
            "s8_can_select_candidate": False,
        },
        "execution": {
            "default_run_mode": "disabled",
            "smoke_is_diagnostic_only": True,
            "random_state": 42,
            "batch_size": 4096,
            "no_sealed_partition_unlock_in_development_code": True,
        },
        "seven_step_cycle": [
            "freeze_v1_baseline_and_v2_protocol",
            "build_hierarchical_critical_detector",
            "compare_resampling_and_hard_negatives",
            "challenge_best_classical_with_compact_transformer",
            "apply_safety_margin_and_select_v2_candidate",
            "freeze_v2_package",
            "open_stress_2025_h2_once_under_a_new_confirmatory_protocol",
        ],
    }


def _read_json(path: Path) -> dict[str, Any]:
    """Read one UTF-8 JSON object and reject non-object documents."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read V2 protocol: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("V2 protocol must be a JSON object")
    return payload


_DEVELOPMENT_WINDOW_ROLES = ("inner_fit", "inner_calibration", "outer_evaluation")
_FALLBACK_EXEMPT_ROLE = "inner_fit"


def _parse_window(window: Any, role: str) -> dict[str, Any]:
    """Parse and strictly validate one partition-and-date development window."""

    if not isinstance(window, Mapping):
        raise ValueError(f"V2 development window is not an object: {role}")
    partition = str(window.get("partition"))
    if partition not in ALLOWED_PARTITIONS:
        raise ValueError(f"V2 development window partition is invalid: {role}")
    start = date.fromisoformat(str(window.get("start")))
    end = date.fromisoformat(str(window.get("end")))
    if start > end:
        raise ValueError(f"V2 development window start is after end: {role}")
    return {"partition": partition, "start": start, "end": end}


def _windows_intersect(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    """Return whether two parsed partition-and-date windows share any day."""

    return (
        first["partition"] == second["partition"]
        and first["start"] <= second["end"]
        and second["start"] <= first["end"]
    )


def _validate_development_windows(
    windows: Any,
    fallback_fit_scope: Any,
) -> None:
    """Enforce strict window ordering and a fallback-safe calibration boundary."""

    if not isinstance(windows, Mapping) or set(windows) != set(
        _DEVELOPMENT_WINDOW_ROLES
    ):
        raise ValueError("V2 development windows must define exactly three roles")
    parsed = {
        role: _parse_window(windows[role], role) for role in _DEVELOPMENT_WINDOW_ROLES
    }
    if parsed["inner_fit"]["end"] >= parsed["inner_calibration"]["start"]:
        raise ValueError(
            "V2 development windows must be ordered: inner_fit before "
            "inner_calibration"
        )
    if parsed["inner_calibration"]["end"] >= parsed["outer_evaluation"]["start"]:
        raise ValueError(
            "V2 development windows must be ordered: inner_calibration before "
            "outer_evaluation"
        )
    scope = _parse_window(fallback_fit_scope, "stage_b_bundle.fit_scope")
    for role in _DEVELOPMENT_WINDOW_ROLES:
        if role == _FALLBACK_EXEMPT_ROLE:
            continue
        if _windows_intersect(parsed[role], scope):
            raise ValueError(
                f"V2 {role} window intersects the frozen S7 fallback fit scope"
            )


@dataclass(frozen=True, slots=True)
class V2Protocol:
    """Represent the validated and immutable V2 development contract."""

    payload: Mapping[str, Any]

    def validate(self) -> None:
        """Validate every frozen value and the development boundary."""

        architecture = self.payload.get("architecture")
        stage_b_bundle = (
            architecture.get("stage_b_bundle")
            if isinstance(architecture, Mapping)
            else None
        )
        fit_scope = (
            stage_b_bundle.get("fit_scope")
            if isinstance(stage_b_bundle, Mapping)
            else None
        )
        _validate_development_windows(
            self.payload.get("development_windows"), fit_scope
        )
        actual = json.loads(json.dumps(self.payload, sort_keys=True))
        if actual != _expected_protocol():
            raise ValueError("V2 protocol differs from the frozen contract")
        if self.allowed_partitions != ALLOWED_PARTITIONS:
            raise ValueError("V2 allowed partition boundary is invalid")
        if any(
            partition not in FORBIDDEN_PARTITIONS
            for partition in self.forbidden_partitions
        ):
            raise ValueError("V2 forbidden partition boundary is invalid")

    @property
    def critical_class(self) -> str:
        """Return the frozen critical class label."""

        return str(self.payload["objective"]["critical_class"])

    @property
    def allowed_partitions(self) -> tuple[str, ...]:
        """Return partitions available to development code."""

        return tuple(self.payload["data_boundary"]["allowed_partitions"])

    @property
    def forbidden_partitions(self) -> tuple[str, ...]:
        """Return partitions unavailable to development code."""

        return tuple(self.payload["data_boundary"]["forbidden_partitions"])

    @property
    def scientific_gates(self) -> Mapping[str, float]:
        """Return the three frozen scientific gate limits."""

        return self.payload["selection"]["scientific_gates"]

    @property
    def safety_margins(self) -> Mapping[str, float]:
        """Return the three frozen development safety limits."""

        return self.payload["selection"]["development_safety_margins"]

    def to_dict(self) -> dict[str, Any]:
        """Return a detached copy of the validated protocol."""

        return copy.deepcopy(dict(self.payload))


def load_v2_protocol(path: str | Path = DEFAULT_CONFIG) -> V2Protocol:
    """Load and strictly validate the frozen V2 development protocol.

    Args:
        path: Path to ``v2_development_protocol.json``.

    Returns:
        A validated V2 protocol object.

    Raises:
        ValueError: If the file is not the exact frozen contract.
    """

    return validate_v2_protocol(_read_json(Path(path).expanduser().resolve()))


def validate_v2_protocol(payload: Mapping[str, Any]) -> V2Protocol:
    """Validate an in-memory V2 protocol against the frozen contract.

    Args:
        payload: JSON-compatible protocol mapping to validate.

    Returns:
        A validated immutable V2 protocol object.

    Raises:
        ValueError: If the mapping is not the exact frozen contract.
    """

    if not isinstance(payload, Mapping):
        raise ValueError("V2 protocol must be a mapping")
    protocol = V2Protocol(copy.deepcopy(dict(payload)))
    protocol.validate()
    return protocol


def _safe_relative_path(root: Path, value: str) -> Path:
    """Resolve one protocol artifact path without allowing path traversal."""

    candidate = Path(value)
    if candidate.is_absolute() or candidate.drive or ".." in candidate.parts:
        raise ValueError("Protocol artifact paths must be project-relative")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Protocol artifact path escapes the project root")
    return resolved


def _file_signature(path: Path) -> dict[str, Any]:
    """Return the size and uppercase SHA256 digest of one artifact."""

    if not path.is_file():
        raise ValueError(f"Protocol artifact is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest().upper(), "size_bytes": path.stat().st_size}


def validate_baseline_artifacts(
    protocol: V2Protocol,
    project_root: str | Path,
    *,
    validate_s8: bool = True,
    validate_s7: bool = True,
) -> dict[str, dict[str, Any]]:
    """Validate optional S8 and S7 hashes without opening dataset files.

    Args:
        protocol: Validated V2 development protocol.
        project_root: Project root containing the configured artifacts.
        validate_s8: Validate the S8 manifest and result when true.
        validate_s7: Validate the frozen S7 bundle when true.

    Returns:
        Actual signatures keyed by artifact role.

    Raises:
        ValueError: If a requested artifact is absent or tampered.
    """

    protocol.validate()
    root = Path(project_root).expanduser().resolve()
    signatures: dict[str, dict[str, Any]] = {}
    targets: list[tuple[str, Mapping[str, Any]]] = []
    if validate_s8:
        baseline = protocol.payload["baseline_s8"]
        targets.extend((name, baseline[name]) for name in ("manifest", "result"))
    if validate_s7:
        targets.append(
            ("s7_bundle", protocol.payload["architecture"]["stage_b_bundle"])
        )
    for role, expected in targets:
        actual = _file_signature(_safe_relative_path(root, str(expected["path"])))
        if actual != {
            "sha256": str(expected["sha256"]),
            "size_bytes": int(expected["size_bytes"]),
        }:
            raise ValueError(f"V2 baseline artifact signature mismatch: {role}")
        signatures[role] = actual
    return signatures


def require_development_partition(partition: str) -> str:
    """Accept only train or validation for V2 development execution.

    Args:
        partition: Partition identifier supplied by development code.

    Returns:
        The validated partition identifier.

    Raises:
        ValueError: If the partition is sealed or otherwise unsupported.
    """

    if partition not in ALLOWED_PARTITIONS:
        if partition in FORBIDDEN_PARTITIONS:
            raise ValueError(f"Sealed partition is unavailable in V2: {partition}")
        raise ValueError(f"Unsupported V2 development partition: {partition}")
    return partition


def make_binary_critical_target(
    labels: Sequence[str],
    critical_class: str = CRITICAL_CLASS,
) -> tuple[int, ...]:
    """Create a deterministic binary target for the critical class.

    Args:
        labels: Multiclass labels from an allowed development partition.
        critical_class: Label mapped to one; all other labels map to zero.

    Returns:
        Integer targets in the original order.

    Raises:
        ValueError: If labels are not a non-empty sequence of strings.
    """

    if isinstance(labels, (str, bytes)):
        raise ValueError("labels must be a sequence of strings")
    values = tuple(labels)
    if not values or not all(isinstance(label, str) for label in values):
        raise ValueError("labels must be a non-empty sequence of strings")
    if critical_class != CRITICAL_CLASS:
        raise ValueError("critical_class differs from the V2 contract")
    unknown = set(values) - set(MODELED_FAMILIES)
    if unknown:
        raise ValueError(f"labels contain unknown V2 classes: {sorted(unknown)}")
    return tuple(int(label == critical_class) for label in values)


def make_critical_target(
    labels: Sequence[str],
    critical_class: str = CRITICAL_CLASS,
) -> tuple[int, ...]:
    """Alias for :func:`make_binary_critical_target`."""

    return make_binary_critical_target(labels, critical_class)


def combine_detector_with_fallback(
    detector_decisions: Sequence[bool],
    fallback_labels: Sequence[str],
    critical_class: str = CRITICAL_CLASS,
) -> tuple[str, ...]:
    """Combine binary detector decisions with frozen multiclass fallback.

    Args:
        detector_decisions: True when the calibrated detector selects critical.
        fallback_labels: Labels produced by the S7 multiclass fallback.
        critical_class: Label emitted for a positive detector decision.

    Returns:
        Hierarchical labels in the original order.

    Raises:
        ValueError: If lengths, booleans, labels, or the critical class are invalid.
    """

    decisions = tuple(detector_decisions)
    fallbacks = tuple(fallback_labels)
    if len(decisions) != len(fallbacks) or not decisions:
        raise ValueError("detector decisions and fallback labels must align")
    if critical_class != CRITICAL_CLASS:
        raise ValueError("critical_class differs from the V2 contract")
    if not all(isinstance(value, (bool, np.bool_)) for value in decisions):
        raise ValueError("detector decisions must be boolean")
    if not all(isinstance(value, str) and value for value in fallbacks):
        raise ValueError("fallback labels must be non-empty strings")
    unknown = set(fallbacks) - set(MODELED_FAMILIES)
    if unknown:
        raise ValueError(
            f"fallback labels contain unknown V2 classes: {sorted(unknown)}"
        )
    return tuple(
        critical_class if bool(decision) else fallback
        for decision, fallback in zip(decisions, fallbacks)
    )


def _metric_limits(
    limits: Mapping[str, float],
) -> dict[str, float]:
    """Normalize and strictly validate the three metric limits."""

    expected = {
        "global_macro_f1_min",
        "critical_f1_min",
        "critical_precision_min",
    }
    if set(limits) != expected:
        raise ValueError("V2 metric limits must contain exactly three gates")
    normalized = {key: float(value) for key, value in limits.items()}
    if not all(np.isfinite(value) for value in normalized.values()):
        raise ValueError("V2 metric limits must be finite")
    if not all(0.0 <= value <= 1.0 for value in normalized.values()):
        raise ValueError("V2 metric limits must be between zero and one")
    return normalized


def _evaluate_limits(
    metrics: Mapping[str, float],
    limits: Mapping[str, float],
) -> dict[str, Any]:
    """Compare metrics against exactly three lower-bound limits."""

    normalized_limits = _metric_limits(limits)
    metric_names = {
        "global_macro_f1_min": "macro_f1",
        "critical_f1_min": "critical_f1",
        "critical_precision_min": "critical_precision",
    }
    missing = set(metric_names.values()) - set(metrics)
    if missing:
        raise ValueError(f"Missing V2 metrics: {sorted(missing)}")
    values = {name: float(metrics[name]) for name in metric_names.values()}
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("V2 metrics must be finite")
    if not all(0.0 <= value <= 1.0 for value in values.values()):
        raise ValueError("V2 metrics must be between zero and one")
    checks = {
        metric_names[limit_name]: values[metric_names[limit_name]] >= limit
        for limit_name, limit in normalized_limits.items()
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "gate_count": int(sum(checks.values())),
        "required_gate_count": 3,
        "limits": normalized_limits,
        "values": values,
    }


def calculate_scientific_gates(
    metrics: Mapping[str, float],
    protocol: V2Protocol | Mapping[str, Any],
) -> dict[str, Any]:
    """Calculate the three simultaneous scientific gates."""

    validated = (
        protocol
        if isinstance(protocol, V2Protocol)
        else validate_v2_protocol(protocol)
    )
    validated.validate()
    limits = validated.scientific_gates
    return _evaluate_limits(metrics, limits)


def calculate_safety_margins(
    metrics: Mapping[str, float],
    protocol: V2Protocol | Mapping[str, Any],
) -> dict[str, Any]:
    """Calculate the three development safety margins and metric headroom."""

    validated = (
        protocol
        if isinstance(protocol, V2Protocol)
        else validate_v2_protocol(protocol)
    )
    validated.validate()
    limits = validated.safety_margins
    result = _evaluate_limits(metrics, limits)
    result["headroom"] = {
        name: result["values"][name] - result["limits"][limit_name]
        for limit_name, name in {
            "global_macro_f1_min": "macro_f1",
            "critical_f1_min": "critical_f1",
            "critical_precision_min": "critical_precision",
        }.items()
    }
    return result


def evaluate_scientific_gates(
    metrics: Mapping[str, float],
    protocol: V2Protocol | Mapping[str, Any],
) -> dict[str, Any]:
    """Alias for :func:`calculate_scientific_gates`."""

    return calculate_scientific_gates(metrics, protocol)


def evaluate_safety_margins(
    metrics: Mapping[str, float],
    protocol: V2Protocol | Mapping[str, Any],
) -> dict[str, Any]:
    """Alias for :func:`calculate_safety_margins`."""

    return calculate_safety_margins(metrics, protocol)
