"""S6 inner-calibrated classical models for sparse complaint text."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC

from .s3 import read_scientific_frame
from .temporal_split import MODELED_FAMILIES
from .tracking import NullTracker, Tracker


S6_SCHEMA_VERSION = "s6-calibrated-classical-v1"
S6_CODE_SCHEMA = "s6-runtime-v1"
DEVELOPMENT_PARTITIONS = ("train", "validation")
SEALED_PARTITIONS = ("test", "stress", "monitor")
CRITICAL_CLASS = "debt_credit_management"
DEFAULT_BATCH_SIZE = 4096


@dataclass(frozen=True, slots=True)
class S6RepresentationConfig:
    """Describe the frozen S3/S4/S5 word representation."""

    analyzer: str
    ngram_range: tuple[int, int]
    max_features: int
    min_df: int
    max_df: float
    sublinear_tf: bool
    dtype: str

    def validate(self) -> None:
        """Validate exact compatibility with the prior word baseline."""

        expected = {
            "analyzer": "word",
            "ngram_range": (1, 2),
            "max_features": 40000,
            "min_df": 2,
            "max_df": 0.98,
            "sublinear_tf": True,
            "dtype": "float32",
        }
        if asdict(self) != expected:
            raise ValueError("S6 representation differs from the frozen word TF-IDF")


@dataclass(frozen=True, slots=True)
class S6CandidateConfig:
    """Describe one frozen S6 estimator candidate."""

    name: str
    estimator: str
    parameters: Mapping[str, Any]

    def validate(self) -> None:
        """Validate the estimator and its exact frozen parameters."""

        supported = {
            "ridge_balanced": "RidgeClassifier",
            "logistic_regression_saga_balanced": "LogisticRegression",
            "linear_svc_c_0_3_balanced": "LinearSVC",
            "linear_svc_c_1_balanced": "LinearSVC",
            "linear_svc_c_3_balanced": "LinearSVC",
        }
        if supported.get(self.name) != self.estimator:
            raise ValueError(f"Unsupported S6 candidate: {self.name}")
        if self.estimator == "RidgeClassifier":
            expected = {
                "alpha": 1.0,
                "class_weight": "balanced",
                "tol": 0.001,
                "solver": "lsqr",
            }
        elif self.estimator == "LogisticRegression":
            expected = {
                "solver": "saga",
                "C": 1.0,
                "class_weight": "balanced",
                "max_iter": 200,
                "tol": 0.001,
                "random_state": 42,
            }
        else:
            c_value = float(self.parameters.get("C", 0.0))
            expected = {
                "C": c_value,
                "class_weight": "balanced",
                "tol": 0.0001,
                "max_iter": 5000,
                "dual": "auto",
                "random_state": 42,
            }
        if dict(self.parameters) != expected:
            raise ValueError(f"S6 parameters are not frozen for {self.name}")


@dataclass(frozen=True, slots=True)
class S6GateConfig:
    """Define the frozen S6 eligibility gates."""

    global_macro_f1_min: float
    critical_f1_min: float
    critical_precision_min: float


@dataclass(frozen=True, slots=True)
class S6ExperimentConfig:
    """Represent the complete frozen S6 experiment configuration."""

    schema_version: str
    status: str
    approved_on: str
    scientific_cache: str
    fit_partition: str
    inner_fit: Mapping[str, str]
    inner_calibration: Mapping[str, str]
    outer_evaluation: Mapping[str, str]
    sealed_partitions: tuple[str, ...]
    critical_class: str
    representation: S6RepresentationConfig
    candidates: tuple[S6CandidateConfig, ...]
    gates: S6GateConfig
    random_state: int
    run_defaults: Mapping[str, Any]

    def validate(self) -> None:
        """Validate the frozen dates, boundaries, candidates, and gates."""

        if self.schema_version != S6_SCHEMA_VERSION:
            raise ValueError("Unexpected S6 schema version")
        if self.status != "FROZEN_FOR_S6_DEVELOPMENT":
            raise ValueError("S6 protocol is not frozen for development")
        if self.fit_partition != "train":
            raise ValueError("S6 must fit only the train partition")
        expected_scopes = {
            "inner_fit": {
                "partition": "train",
                "start": "2023-08-01",
                "end": "2024-04-30",
            },
            "inner_calibration": {
                "partition": "train",
                "start": "2024-05-01",
                "end": "2024-06-30",
            },
            "outer_evaluation": {
                "partition": "validation",
                "start": "2024-07-01",
                "end": "2024-12-31",
            },
        }
        actual_scopes = {
            "inner_fit": dict(self.inner_fit),
            "inner_calibration": dict(self.inner_calibration),
            "outer_evaluation": dict(self.outer_evaluation),
        }
        if actual_scopes != expected_scopes:
            raise ValueError("S6 date scopes differ from the frozen protocol")
        if self.sealed_partitions != SEALED_PARTITIONS:
            raise ValueError("S6 sealed partition boundary is invalid")
        if self.critical_class != CRITICAL_CLASS:
            raise ValueError("S6 critical class is invalid")
        self.representation.validate()
        expected = {
            "ridge_balanced",
            "logistic_regression_saga_balanced",
            "linear_svc_c_0_3_balanced",
            "linear_svc_c_1_balanced",
            "linear_svc_c_3_balanced",
        }
        actual = {candidate.name for candidate in self.candidates}
        if actual != expected or len(actual) != len(self.candidates):
            raise ValueError("S6 candidate set differs from the frozen design")
        for candidate in self.candidates:
            candidate.validate()


def _read_json(path: Path) -> dict[str, Any]:
    """Read one UTF-8 JSON object."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def load_s6_config(path: str | Path) -> S6ExperimentConfig:
    """Load and validate the frozen S6 configuration.

    Args:
        path: Path to ``s6_calibrated_classical.json``.

    Returns:
        A validated S6 configuration.
    """

    payload = _read_json(Path(path).expanduser().resolve())
    representation = payload["representation"]
    candidates = tuple(
        S6CandidateConfig(
            name=name,
            estimator=str(values["estimator"]),
            parameters={
                key: value for key, value in values.items() if key != "estimator"
            },
        )
        for name, values in payload["candidates"].items()
    )
    config = S6ExperimentConfig(
        schema_version=str(payload["schema_version"]),
        status=str(payload["status"]),
        approved_on=str(payload["approved_on"]),
        scientific_cache=str(payload["scientific_cache"]),
        fit_partition=str(payload["fit_partition"]),
        inner_fit={str(key): str(value) for key, value in payload["inner_fit"].items()},
        inner_calibration={
            str(key): str(value)
            for key, value in payload["inner_calibration"].items()
        },
        outer_evaluation={
            str(key): str(value)
            for key, value in payload["outer_evaluation"].items()
        },
        sealed_partitions=tuple(str(x) for x in payload["sealed_partitions"]),
        critical_class=str(payload["critical_class"]),
        representation=S6RepresentationConfig(
            analyzer=str(representation["analyzer"]),
            ngram_range=tuple(int(x) for x in representation["ngram_range"]),
            max_features=int(representation["max_features"]),
            min_df=int(representation["min_df"]),
            max_df=float(representation["max_df"]),
            sublinear_tf=bool(representation["sublinear_tf"]),
            dtype=str(representation["dtype"]),
        ),
        candidates=candidates,
        gates=S6GateConfig(**{
            "global_macro_f1_min": float(payload["gates"]["global_macro_f1_min"]),
            "critical_f1_min": float(payload["gates"]["critical_f1_min"]),
            "critical_precision_min": float(
                payload["gates"]["critical_precision_min"]
            ),
        }),
        random_state=int(payload["random_state"]),
        run_defaults=payload["run_defaults"],
    )
    config.validate()
    return config


def validate_scientific_cache(frame: Any) -> None:
    """Validate the cache schema and reject every sealed partition.

    Args:
        frame: Arrow table loaded from the S3 scientific cache.
    """

    expected = {
        "Complaint ID",
        "received_date",
        "product_family",
        "normalized_group_hash",
        "normalized_length",
        "partition_name",
        "narrative",
    }
    if set(frame.column_names) != expected:
        raise ValueError("Scientific cache columns do not match the S6 contract")
    partitions = {str(value.as_py()) for value in frame["partition_name"]}
    if not partitions.issubset(set(DEVELOPMENT_PARTITIONS)):
        raise ValueError("S6 scientific cache contains a sealed partition")
    labels = {str(value.as_py()) for value in frame["product_family"]}
    if not labels.issubset(set(MODELED_FAMILIES)):
        raise ValueError("S6 cache contains an out-of-scope class")
    identities: set[tuple[str, int]] = set()
    identifiers: set[int] = set()
    for group_hash, length, complaint_id in zip(
        frame["normalized_group_hash"],
        frame["normalized_length"],
        frame["Complaint ID"],
    ):
        identity = (str(group_hash.as_py()), int(length.as_py()))
        identifier = int(complaint_id.as_py())
        if identity in identities or identifier in identifiers:
            raise ValueError("S6 cache must contain unique groups and IDs")
        identities.add(identity)
        identifiers.add(identifier)


def _date(value: Any) -> str:
    """Normalize an Arrow scalar date to ISO date text."""

    return str(value.as_py())[:10]


def _indices_for_scope(
    frame: Any,
    partition: str,
    start: str,
    end: str,
    max_per_class: int | None,
    random_state: int,
) -> list[int]:
    """Select deterministic per-class rows within one approved date scope."""

    if partition not in DEVELOPMENT_PARTITIONS:
        raise ValueError("S6 scope cannot access a sealed partition")
    if max_per_class is not None and max_per_class <= 0:
        raise ValueError("max_per_class must be positive")
    by_class: dict[str, list[int]] = {label: [] for label in MODELED_FAMILIES}
    for index in range(len(frame)):
        if str(frame["partition_name"][index].as_py()) != partition:
            continue
        received = _date(frame["received_date"][index])
        if start <= received <= end:
            label = str(frame["product_family"][index].as_py())
            by_class[label].append(index)
    selected: list[int] = []
    for label in MODELED_FAMILIES:
        ordered = sorted(
            by_class[label],
            key=lambda index: hashlib.sha256(
                (
                    f"{random_state}|{label}|"
                    f"{frame['Complaint ID'][index].as_py()}"
                ).encode()
            ).hexdigest(),
        )
        selected.extend(ordered[:max_per_class])
    return sorted(selected)


def _texts(frame: Any, indices: Sequence[int]) -> list[str]:
    """Extract narratives for Arrow row indices."""

    return [str(frame["narrative"][index].as_py()) for index in indices]


def _labels(frame: Any, indices: Sequence[int]) -> list[str]:
    """Extract product labels for Arrow row indices."""

    return [str(frame["product_family"][index].as_py()) for index in indices]


def _metrics_from_matrix(matrix: np.ndarray) -> dict[str, Any]:
    """Calculate metrics from a fixed-order confusion matrix."""

    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros(len(MODELED_FAMILIES)),
        where=predicted != 0,
    )
    recall = np.divide(
        true_positive, support, out=np.zeros(len(MODELED_FAMILIES)), where=support != 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(len(MODELED_FAMILIES)),
        where=(precision + recall) != 0,
    )
    total = int(support.sum())
    if total == 0:
        raise ValueError("Cannot calculate metrics for an empty scope")
    return {
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.dot(f1, support) / total),
        "balanced_accuracy": float(recall.mean()),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(MODELED_FAMILIES)
        },
        "row_count": total,
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def _gates(metrics: Mapping[str, Any], config: S6GateConfig) -> dict[str, Any]:
    """Evaluate the three frozen gates for one metric bundle."""

    critical = metrics["per_class"][CRITICAL_CLASS]
    values = {
        "global_macro_f1": float(metrics["macro_f1"]),
        "critical_f1": float(critical["f1"]),
        "critical_precision": float(critical["precision"]),
    }
    limits = {
        "global_macro_f1": config.global_macro_f1_min,
        "critical_f1": config.critical_f1_min,
        "critical_precision": config.critical_precision_min,
    }
    checks = {name: values[name] >= limits[name] for name in values}
    return {
        **checks,
        "gate_count": int(sum(checks.values())),
        "eligible": bool(all(checks.values())),
        "values": values,
        "limits": limits,
    }


def _scores_in_family_order(estimator: Any, scores: Any) -> np.ndarray:
    """Reorder decision columns from estimator classes to modeled families."""

    classes = [str(value) for value in estimator.classes_]
    if set(classes) != set(MODELED_FAMILIES) or len(classes) != len(MODELED_FAMILIES):
        raise ValueError("Estimator classes do not match MODELED_FAMILIES")
    positions = [classes.index(label) for label in MODELED_FAMILIES]
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(classes):
        raise ValueError("Estimator decision_function has an invalid shape")
    return values[:, positions]


def _margin_predictions(
    scores: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return predictions and margins using the critical-class rule."""

    critical_index = MODELED_FAMILIES.index(CRITICAL_CLASS)
    noncritical = [
        index for index, label in enumerate(MODELED_FAMILIES)
        if label != CRITICAL_CLASS
    ]
    noncritical_scores = scores[:, noncritical]
    best_noncritical = np.argmax(noncritical_scores, axis=1)
    margins = scores[:, critical_index] - noncritical_scores.max(axis=1)
    predicted = np.asarray(
        [MODELED_FAMILIES[noncritical[index]] for index in best_noncritical],
        dtype=object,
    )
    predicted[margins >= threshold] = CRITICAL_CLASS
    return predicted, margins


def _matrix_for_predictions(
    labels: Sequence[str], predictions: Sequence[str]
) -> np.ndarray:
    """Build a fixed-order confusion matrix from labels and predictions."""

    positions = {label: index for index, label in enumerate(MODELED_FAMILIES)}
    matrix = np.zeros((len(MODELED_FAMILIES), len(MODELED_FAMILIES)), dtype=np.int64)
    for actual, predicted in zip(labels, predictions):
        if actual not in positions or str(predicted) not in positions:
            raise ValueError("Labels or predictions fall outside modeled families")
        matrix[positions[actual], positions[str(predicted)]] += 1
    return matrix


def search_thresholds_exact(
    labels: Sequence[str],
    scores: np.ndarray,
    gates: S6GateConfig,
) -> dict[str, Any]:
    """Search all unique margins with an incremental confusion matrix.

    Args:
        labels: Calibration labels in modeled-family order.
        scores: Decision scores already reordered by modeled families.
        gates: Frozen S6 gates.

    Returns:
        Baseline threshold-zero metrics and the selected threshold result.
    """

    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != len(MODELED_FAMILIES):
        raise ValueError("scores must have one column per modeled family")
    if len(labels) != len(scores) or len(scores) == 0:
        raise ValueError("Threshold search requires aligned non-empty arrays")
    _, margins = _margin_predictions(scores, 0.0)
    if not np.isfinite(margins).all():
        raise ValueError("Margins must be finite")
    no_critical_threshold = float(np.nextafter(np.max(margins), np.inf))
    noncritical, _ = _margin_predictions(scores, no_critical_threshold)
    baseline_matrix = _matrix_for_predictions(
        labels, _margin_predictions(scores, 0.0)[0]
    )
    baseline_metrics = _metrics_from_matrix(baseline_matrix)
    baseline = {
        "threshold": 0.0,
        "metrics": baseline_metrics,
        "gates": _gates(baseline_metrics, gates),
    }
    critical_index = MODELED_FAMILIES.index(CRITICAL_CLASS)
    base_matrix = _matrix_for_predictions(labels, noncritical)
    order = np.argsort(margins)[::-1]
    unique = np.unique(margins)
    no_critical = float(np.nextafter(np.max(margins), np.inf))
    threshold_values = [no_critical] + [float(value) for value in unique]
    selected: dict[str, Any] | None = None
    threshold_count = 0
    matrix = base_matrix.copy()
    cursor = 0
    for threshold in sorted(threshold_values, reverse=True):
        while cursor < len(order) and margins[order[cursor]] >= threshold:
            row = int(order[cursor])
            noncritical_prediction = int(
                np.argmax(
                    np.delete(scores[row], critical_index)
                )
            )
            noncritical_indices = [
                index for index in range(len(MODELED_FAMILIES))
                if index != critical_index
            ]
            actual_index = MODELED_FAMILIES.index(str(labels[row]))
            matrix[actual_index, noncritical_indices[noncritical_prediction]] -= 1
            matrix[actual_index, critical_index] += 1
            cursor += 1
        metrics = _metrics_from_matrix(matrix)
        candidate = {
            "threshold": float(threshold),
            "metrics": metrics,
            "gates": _gates(metrics, gates),
        }
        threshold_count += 1
        if selected is None or _threshold_sort_key(candidate) > _threshold_sort_key(
            selected
        ):
            selected = candidate
    if selected is None:
        raise RuntimeError("S6 threshold search produced no candidate")
    return {
        "baseline_threshold_zero": baseline,
        "selected": selected,
        "threshold_count": threshold_count,
    }


def _threshold_sort_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return the deterministic threshold-selection key."""

    metrics = item["metrics"]
    critical = metrics["per_class"][CRITICAL_CLASS]
    gate_info = item["gates"]
    threshold = float(item["threshold"])
    return (
        int(gate_info["eligible"]),
        int(gate_info["gate_count"]),
        float(critical["f1"]),
        float(metrics["macro_f1"]),
        float(critical["precision"]),
        -abs(threshold),
        -threshold,
    )


def _candidate_sort_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return the deterministic candidate-selection key."""

    metrics = item["calibration"]["selected"]["metrics"]
    critical = metrics["per_class"][CRITICAL_CLASS]
    gate_info = item["calibration"]["selected"]["gates"]
    return (
        -int(gate_info["eligible"]),
        -int(gate_info["gate_count"]),
        -float(critical["f1"]),
        -float(metrics["macro_f1"]),
        -float(critical["precision"]),
        item["name"],
    )


def _vectorizer(config: S6RepresentationConfig) -> TfidfVectorizer:
    """Build the one frozen word TF-IDF vectorizer."""

    return TfidfVectorizer(
        analyzer=config.analyzer,
        ngram_range=config.ngram_range,
        max_features=config.max_features,
        min_df=config.min_df,
        max_df=config.max_df,
        sublinear_tf=config.sublinear_tf,
        dtype=np.float32,
    )


def _build_estimator(candidate: S6CandidateConfig) -> Any:
    """Instantiate one frozen S6 candidate."""

    parameters = dict(candidate.parameters)
    if candidate.estimator == "RidgeClassifier":
        return RidgeClassifier(**parameters)
    if candidate.estimator == "LogisticRegression":
        return LogisticRegression(**parameters)
    if candidate.estimator == "LinearSVC":
        return LinearSVC(**parameters)
    raise ValueError(f"Unsupported S6 estimator: {candidate.estimator}")


def _fit_scores(
    candidate: S6CandidateConfig,
    x_fit: Any,
    x_calibration: Any,
    labels: Sequence[str],
) -> tuple[Any, np.ndarray, list[dict[str, str]]]:
    """Fit one candidate and return reordered calibration scores."""

    estimator = _build_estimator(candidate)
    captured: list[dict[str, str]] = []
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(x_fit, labels)
        for record in records:
            if issubclass(record.category, ConvergenceWarning):
                captured.append({
                    "category": record.category.__name__,
                    "message": str(record.message),
                })
    scores = _scores_in_family_order(
        estimator, estimator.decision_function(x_calibration)
    )
    return estimator, scores, captured


def _evaluate_outer(
    estimator: Any,
    vectorizer: TfidfVectorizer,
    frame: Any,
    indices: Sequence[int],
    threshold: float,
    batch_size: int,
) -> dict[str, Any]:
    """Evaluate exactly one fitted estimator on outer validation batches."""

    matrix = np.zeros((len(MODELED_FAMILIES), len(MODELED_FAMILIES)), dtype=np.int64)
    for start in range(0, len(indices), batch_size):
        batch = indices[start : start + batch_size]
        transformed = vectorizer.transform(_texts(frame, batch))
        scores = _scores_in_family_order(
            estimator, estimator.decision_function(transformed)
        )
        predicted, _ = _margin_predictions(scores, threshold)
        matrix += _matrix_for_predictions(_labels(frame, batch), predicted)
    return _metrics_from_matrix(matrix)


def _file_signature(path: Path) -> dict[str, Any]:
    """Return stable file metadata and digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
        "sha256": digest.hexdigest().upper(),
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one JSON artifact atomically with real UTF-8 text."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _signature(
    cache_path: Path,
    config_path: Path,
    mode: str,
    max_per_class: int | None,
    batch_size: int,
) -> dict[str, Any]:
    """Build the S6 cache signature without reading sealed partitions."""

    return {
        "schema_version": S6_SCHEMA_VERSION,
        "code_schema": S6_CODE_SCHEMA,
        "scientific_cache": _file_signature(cache_path),
        "config": _file_signature(config_path),
        "run": {
            "mode": mode,
            "max_per_class": max_per_class,
            "batch_size": batch_size,
        },
        "boundary": {
            "inner_fit": "train:2023-08-01..2024-04-30",
            "inner_calibration": "train:2024-05-01..2024-06-30",
            "outer": "validation:2024-07-01..2024-12-31",
            "sealed": list(SEALED_PARTITIONS),
        },
    }


def _base_payload(
    config: S6ExperimentConfig,
    signature: Mapping[str, Any],
    mode: str,
    fit_rows: int,
    calibration_rows: int,
    outer_rows: int,
) -> dict[str, Any]:
    """Create the incremental artifact envelope."""

    return {
        "schema_version": S6_SCHEMA_VERSION,
        "status": "RUNNING",
        "complete": False,
        "run_mode": mode,
        "claim_boundary": "NO_TEST_STRESS_OR_MONITOR_ACCESS",
        "sealed_partitions": list(config.sealed_partitions),
        "signature": signature,
        "execution_attempts": 1,
        "vectorizer_fit_count": 0,
        "candidates": [],
        "calibration_summary": [],
        "outer_summary": None,
        "outer_evaluated_candidate": None,
        "selection_status": "RUNNING",
        "recommended_candidate": None,
        "diagnostic_focus": None,
        "resume": {
            "resumed": False,
            "reused_candidate_count": 0,
        },
        "scope_rows": {
            "inner_fit": fit_rows,
            "inner_calibration": calibration_rows,
            "outer_validation": outer_rows,
        },
        "resources": {"batch_size": signature["run"]["batch_size"]},
    }


def _validate_resume_prefix(
    payload: Mapping[str, Any],
    candidates: Sequence[S6CandidateConfig],
) -> None:
    """Validate that persisted candidates form one frozen-order prefix."""

    persisted = payload.get("candidates")
    if not isinstance(persisted, list):
        raise ValueError("S6 resume artifact candidates must be a list")
    if len(persisted) > len(candidates):
        raise ValueError("S6 resume artifact has too many candidates")
    expected_names = [candidate.name for candidate in candidates]
    persisted_names = []
    for item in persisted:
        if not isinstance(item, Mapping) or not item.get("name"):
            raise ValueError("S6 resume candidate entry is invalid")
        persisted_names.append(str(item["name"]))
        if "calibration" not in item:
            raise ValueError("S6 resume candidate metrics are incomplete")
    if persisted_names != expected_names[: len(persisted_names)]:
        raise ValueError("S6 resume candidates are not a frozen-order prefix")
    if len(set(persisted_names)) != len(persisted_names):
        raise ValueError("S6 resume candidates must be unique")


def run_s6(
    scientific_cache_path: str | Path,
    artifact_path: str | Path,
    config_path: str | Path,
    *,
    smoke_max_per_class: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    tracker: Tracker | None = None,
) -> dict[str, Any]:
    """Run S6 calibration and one outer validation evaluation.

    Args:
        scientific_cache_path: Development-only S3 scientific cache.
        artifact_path: Incremental atomic S6 result artifact.
        config_path: Frozen S6 configuration.
        smoke_max_per_class: Deterministic per-class cap for diagnostic smoke.
        batch_size: Batch size used for outer transformations.
        tracker: Optional framework-neutral tracker.

    Returns:
        Complete S6 artifact payload.

    Raises:
        ValueError: If the cache, configuration, or boundaries are invalid.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    mode = "smoke" if smoke_max_per_class is not None else "full"
    cache = Path(scientific_cache_path).expanduser().resolve()
    artifact = Path(artifact_path).expanduser().resolve()
    config_file = Path(config_path).expanduser().resolve()
    config = load_s6_config(config_file)
    frame = read_scientific_frame(cache)
    validate_scientific_cache(frame)
    fit_indices = _indices_for_scope(
        frame, "train", "2023-08-01", "2024-04-30", smoke_max_per_class,
        config.random_state,
    )
    calibration_indices = _indices_for_scope(
        frame, "train", "2024-05-01", "2024-06-30", smoke_max_per_class,
        config.random_state,
    )
    outer_indices = _indices_for_scope(
        frame, "validation", "2024-07-01", "2024-12-31", smoke_max_per_class,
        config.random_state,
    )
    if not fit_indices or not calibration_indices or not outer_indices:
        raise ValueError("S6 scopes must all contain rows")
    signature = _signature(
        cache, config_file, mode, smoke_max_per_class, batch_size
    )
    cached = None
    if artifact.exists():
        try:
            cached = _read_json(artifact)
        except (OSError, json.JSONDecodeError):
            cached = None
        if (
            cached
            and cached.get("complete") is True
            and cached.get("signature") == signature
        ):
            return cached
    is_resume = bool(
        cached
        and cached.get("complete") is False
        and cached.get("signature") == signature
    )
    if is_resume:
        _validate_resume_prefix(cached, config.candidates)
        payload = cached
        previous_status = str(payload.get("status", "UNKNOWN"))
        previous_count = len(payload["candidates"])
        payload["execution_attempts"] = int(
            payload.get("execution_attempts", 1)
        ) + 1
        payload["resume"] = {
            "resumed": True,
            "reused_candidate_count": previous_count,
            "previous_status": previous_status,
        }
        payload["status"] = "RUNNING"
        payload["complete"] = False
        payload["outer_summary"] = None
        payload["outer_evaluated_candidate"] = None
        payload["recommended_candidate"] = None
        payload["diagnostic_focus"] = None
    else:
        payload = _base_payload(
            config, signature, mode, len(fit_indices), len(calibration_indices),
            len(outer_indices),
        )
    _write_json_atomic(artifact, payload)
    tracker_instance = tracker or NullTracker()
    tracker_instance.log_params({
        "schema_version": S6_SCHEMA_VERSION,
        "mode": mode,
        "resumed": is_resume,
    })
    vectorizer = _vectorizer(config.representation)
    x_fit = vectorizer.fit_transform(_texts(frame, fit_indices))
    x_calibration = vectorizer.transform(_texts(frame, calibration_indices))
    y_fit = _labels(frame, fit_indices)
    y_calibration = _labels(frame, calibration_indices)
    payload["vectorizer_fit_count"] = int(
        payload.get("vectorizer_fit_count", 0)
    ) + 1
    payload.setdefault("resources", {})["vocabulary_size"] = len(
        vectorizer.vocabulary_
    )
    _write_json_atomic(artifact, payload)
    best_result: dict[str, Any] | None = None
    best_estimator: Any | None = None
    try:
        persisted_count = len(payload["candidates"])
        candidate_by_name = {
            candidate.name: candidate for candidate in config.candidates
        }
        if persisted_count:
            best_result = sorted(
                payload["candidates"], key=_candidate_sort_key
            )[0]
            best_candidate = candidate_by_name[best_result["name"]]
            best_estimator, _, _ = _fit_scores(
                best_candidate, x_fit, x_calibration, y_fit
            )
            gc.collect()
        for candidate in config.candidates[persisted_count:]:
            started = time.perf_counter()
            estimator, scores, convergence = _fit_scores(
                candidate, x_fit, x_calibration, y_fit,
            )
            calibration = search_thresholds_exact(
                y_calibration, scores, config.gates,
            )
            result = {
                "name": candidate.name,
                "estimator": candidate.estimator,
                "parameters": dict(candidate.parameters),
                "calibration": {
                    "baseline_threshold_zero": calibration["baseline_threshold_zero"],
                    "selected": calibration["selected"],
                    "threshold_count": calibration["threshold_count"],
                },
                "runtime_seconds": float(time.perf_counter() - started),
                "convergence_warnings": convergence,
            }
            payload["candidates"].append(result)
            payload["calibration_summary"] = list(payload["candidates"])
            if best_result is None or _candidate_sort_key(result) < _candidate_sort_key(
                best_result
            ):
                previous_estimator = best_estimator
                best_estimator = estimator
                best_result = result
                if previous_estimator is not None:
                    del previous_estimator
            else:
                del estimator
            gc.collect()
            _write_json_atomic(artifact, payload)
        if best_result is None or best_estimator is None:
            raise RuntimeError("S6 produced no fitted candidate")
        selected = best_result["calibration"]["selected"]
        threshold = float(selected["threshold"])
        outer_metrics = _evaluate_outer(
            best_estimator, vectorizer, frame, outer_indices, threshold, batch_size,
        )
        outer_gates = _gates(outer_metrics, config.gates)
        payload["outer_summary"] = {
            "candidate": best_result["name"],
            "threshold": threshold,
            "metrics": outer_metrics,
            "gates": outer_gates,
        }
        payload["outer_evaluated_candidate"] = best_result["name"]
        diagnostic = sorted(payload["candidates"], key=_candidate_sort_key)[0]
        payload["diagnostic_focus"] = diagnostic["name"]
        calibration_eligible = bool(selected["gates"]["eligible"])
        if mode == "smoke":
            payload["status"] = "DIAGNOSTIC_ONLY"
            payload["selection_status"] = "DIAGNOSTIC_ONLY"
        elif calibration_eligible and outer_gates["eligible"]:
            payload["status"] = "DEVELOPMENT_COMPLETE"
            payload["selection_status"] = "RECOMMENDED"
            payload["recommended_candidate"] = best_result["name"]
        elif calibration_eligible:
            payload["status"] = "DEVELOPMENT_COMPLETE"
            payload["selection_status"] = "NO_OUTER_GATE_PASS"
        else:
            payload["status"] = "DEVELOPMENT_COMPLETE"
            payload["selection_status"] = "NO_ELIGIBLE_CALIBRATED_CANDIDATE"
        payload["complete"] = True
        tracker_instance.log_metrics({
            "outer_macro_f1": float(outer_metrics["macro_f1"]),
            "outer_critical_f1": float(
                outer_metrics["per_class"][CRITICAL_CLASS]["f1"]
            ),
        })
        _write_json_atomic(artifact, payload)
        tracker_instance.log_artifact(artifact)
        return payload
    except Exception as error:
        payload["status"] = "ERROR"
        payload["complete"] = False
        payload["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        _write_json_atomic(artifact, payload)
        raise
    finally:
        if best_estimator is not None:
            del best_estimator
        del x_fit
        del x_calibration
        gc.collect()
        tracker_instance.close()


def run_s6_smoke(
    scientific_cache_path: str | Path,
    artifact_path: str | Path,
    config_path: str | Path,
    *,
    max_per_class: int = 8,
    tracker: Tracker | None = None,
) -> dict[str, Any]:
    """Run deterministic S6 smoke diagnostics without promotion."""

    return run_s6(
        scientific_cache_path,
        artifact_path,
        config_path,
        smoke_max_per_class=max_per_class,
        tracker=tracker,
    )
