"""S5 development-only benchmark of sparse-text estimators."""

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
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

from .s3 import read_scientific_frame
from .temporal_split import MODELED_FAMILIES
from .tracking import NullTracker, Tracker


S5_SCHEMA_VERSION = "s5-estimator-benchmark-v1"
S5_CODE_SCHEMA = "s5-runtime-v1"
DEVELOPMENT_PARTITIONS = ("train", "validation")
SEALED_PARTITIONS = ("test", "stress", "monitor")
CRITICAL_CLASS = "debt_credit_management"
DEFAULT_BATCH_SIZE = 4_096
DEFAULT_REFERENCE_TOLERANCE = 1e-6


@dataclass(frozen=True, slots=True)
class S5RepresentationConfig:
    """Describe the single frozen S5 TF-IDF representation."""

    analyzer: str
    ngram_range: tuple[int, int]
    max_features: int
    min_df: int
    max_df: float
    sublinear_tf: bool
    dtype: str

    def validate(self) -> None:
        """Validate the representation against the S3/S4 word baseline."""

        expected = {
            "analyzer": "word",
            "ngram_range": (1, 2),
            "max_features": 40_000,
            "min_df": 2,
            "max_df": 0.98,
            "sublinear_tf": True,
            "dtype": "float32",
        }
        actual = asdict(self)
        if actual != expected:
            raise ValueError("S5 representation must exactly match S3/S4 word baseline")


@dataclass(frozen=True, slots=True)
class S5CandidateConfig:
    """Describe one frozen S5 estimator candidate."""

    name: str
    estimator: str
    parameters: Mapping[str, Any]

    def validate(self) -> None:
        """Validate the supported local estimator contract."""

        supported = {
            "sgd_log_loss_balanced_reference": "SGDClassifier",
            "linear_svc_balanced": "LinearSVC",
            "complement_nb_balanced": "ComplementNB",
        }
        if supported.get(self.name) != self.estimator:
            raise ValueError(f"Unsupported S5 candidate: {self.name}")
        if self.estimator == "SGDClassifier":
            expected = {
                "loss": "log_loss",
                "class_weight": "balanced",
                "max_iter": 500,
                "tol": 0.001,
                "random_state": 42,
            }
        elif self.estimator == "LinearSVC":
            expected = {
                "C": 1.0,
                "class_weight": "balanced",
                "tol": 0.0001,
                "max_iter": 5000,
                "random_state": 42,
                "dual": "auto",
            }
        else:
            expected = {
                "alpha": 1.0,
                "norm": False,
                "sample_weight": "compute_sample_weight('balanced', y_train)",
            }
        if dict(self.parameters) != expected:
            raise ValueError(f"S5 parameters are not frozen for {self.name}")


@dataclass(frozen=True, slots=True)
class S5GateConfig:
    """Define the S4 gates reused by S5."""

    global_macro_f1_min: float
    critical_f1_min: float
    critical_precision_min: float


@dataclass(frozen=True, slots=True)
class S5ExperimentConfig:
    """Represent the complete frozen S5 experiment configuration."""

    schema_version: str
    status: str
    approved_on: str
    fit_partition: str
    evaluation_partition: str
    sealed_partitions: tuple[str, ...]
    random_state: int
    critical_class: str
    gates: S5GateConfig
    representation: S5RepresentationConfig
    candidates: tuple[S5CandidateConfig, ...]
    deferred_candidates: Mapping[str, Mapping[str, Any]]
    reference_reproduction: Mapping[str, Any]
    run_defaults: Mapping[str, Any]

    def validate(self) -> None:
        """Validate the frozen protocol and candidate inventory."""

        if self.schema_version != S5_SCHEMA_VERSION:
            raise ValueError("Unexpected S5 schema version")
        if self.status != "FROZEN_FOR_S5_DEVELOPMENT":
            raise ValueError("S5 experiment is not frozen for development")
        if (self.fit_partition, self.evaluation_partition) != (
            "train",
            "validation",
        ):
            raise ValueError("S5 must fit on train and evaluate on validation")
        if tuple(self.sealed_partitions) != SEALED_PARTITIONS:
            raise ValueError("S5 sealed partition boundary is invalid")
        if self.critical_class != CRITICAL_CLASS:
            raise ValueError("S5 critical class is invalid")
        self.representation.validate()
        expected = {
            "sgd_log_loss_balanced_reference",
            "linear_svc_balanced",
            "complement_nb_balanced",
        }
        actual = {candidate.name for candidate in self.candidates}
        if actual != expected or len(actual) != len(self.candidates):
            raise ValueError("S5 local candidate set differs from the frozen design")
        for candidate in self.candidates:
            candidate.validate()
        if "logistic_regression_saga" not in self.deferred_candidates:
            raise ValueError("S5 must record deferred LogisticRegression(saga)")


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object from UTF-8."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def load_s5_config(path: str | Path) -> S5ExperimentConfig:
    """Load and validate the frozen S5 configuration.

    Args:
        path: Path to ``s5_estimator_benchmark.json``.

    Returns:
        Validated S5 configuration.
    """

    payload = _read_json(Path(path).expanduser().resolve())
    representation = S5RepresentationConfig(
        analyzer=str(payload["representation"]["analyzer"]),
        ngram_range=tuple(int(x) for x in payload["representation"]["ngram_range"]),
        max_features=int(payload["representation"]["max_features"]),
        min_df=int(payload["representation"]["min_df"]),
        max_df=float(payload["representation"]["max_df"]),
        sublinear_tf=bool(payload["representation"]["sublinear_tf"]),
        dtype=str(payload["representation"]["dtype"]),
    )
    candidates = tuple(
        S5CandidateConfig(
            name=name,
            estimator=str(values["estimator"]),
            parameters={
                key: value
                for key, value in values.items()
                if key != "estimator"
            },
        )
        for name, values in payload["candidates"].items()
    )
    gates = S5GateConfig(**{
        "global_macro_f1_min": float(payload["gates"]["global_macro_f1_min"]),
        "critical_f1_min": float(payload["gates"]["critical_f1_min"]),
        "critical_precision_min": float(payload["gates"]["critical_precision_min"]),
    })
    config = S5ExperimentConfig(
        schema_version=str(payload["schema_version"]),
        status=str(payload["status"]),
        approved_on=str(payload["approved_on"]),
        fit_partition=str(payload["fit_partition"]),
        evaluation_partition=str(payload["evaluation_partition"]),
        sealed_partitions=tuple(str(x) for x in payload["sealed_partitions"]),
        random_state=int(payload["random_state"]),
        critical_class=str(payload["critical_class"]),
        gates=gates,
        representation=representation,
        candidates=candidates,
        deferred_candidates=payload["deferred_candidates"],
        reference_reproduction=payload["reference_reproduction"],
        run_defaults=payload["run_defaults"],
    )
    config.validate()
    return config


def validate_scientific_cache(frame: Any) -> None:
    """Validate the scientific cache and reject sealed partitions.

    Args:
        frame: Arrow table loaded from the S3 scientific cache.

    Raises:
        ValueError: If columns, labels, partitions, groups, or IDs are invalid.
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
        raise ValueError("Scientific cache columns do not match the S5 contract")
    partitions = {str(value.as_py()) for value in frame["partition_name"]}
    if not partitions.issubset(set(DEVELOPMENT_PARTITIONS)):
        raise ValueError("S5 scientific cache contains a sealed partition")
    labels = {str(value.as_py()) for value in frame["product_family"]}
    if not labels.issubset(set(MODELED_FAMILIES)):
        raise ValueError("S5 scientific cache contains an out-of-scope class")
    identities: set[tuple[str, int]] = set()
    ids: set[int] = set()
    for group_hash, length, complaint_id in zip(
        frame["normalized_group_hash"],
        frame["normalized_length"],
        frame["Complaint ID"],
    ):
        identity = (str(group_hash.as_py()), int(length.as_py()))
        identifier = int(complaint_id.as_py())
        if identity in identities or identifier in ids:
            raise ValueError("S5 scientific cache must contain unique groups and IDs")
        identities.add(identity)
        ids.add(identifier)


def _file_sha256(path: Path) -> str:
    """Return an uppercase SHA256 digest for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _file_signature(path: Path) -> dict[str, Any]:
    """Return stable metadata for one cache or reference artifact."""

    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
        "sha256": _file_sha256(path),
    }


def _signature(
    cache_path: Path,
    config_path: Path,
    run_params: Mapping[str, Any],
    reference_path: Path | None,
) -> dict[str, Any]:
    """Build the complete S5 cache signature."""

    signature: dict[str, Any] = {
        "schema_version": S5_SCHEMA_VERSION,
        "code_schema": S5_CODE_SCHEMA,
        "scientific_cache": _file_signature(cache_path),
        "config": _file_signature(config_path),
        "run": dict(run_params),
        "boundary": {
            "fit": "train",
            "evaluation": "validation",
            "sealed": list(SEALED_PARTITIONS),
        },
    }
    if reference_path is not None and reference_path.exists():
        signature["s4_reference"] = _file_signature(reference_path)
    return signature


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Persist one JSON artifact atomically as UTF-8."""

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


def _indices_for_partition(
    frame: Any,
    partition: str,
    max_per_class: int | None,
    random_state: int,
) -> list[int]:
    """Select deterministic per-class row indices within development data."""

    if partition not in DEVELOPMENT_PARTITIONS:
        raise ValueError("S5 index selection cannot access sealed partitions")
    by_class: dict[str, list[int]] = {label: [] for label in MODELED_FAMILIES}
    for index in range(len(frame)):
        if str(frame["partition_name"][index].as_py()) == partition:
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
        if max_per_class is not None:
            if max_per_class <= 0:
                raise ValueError("max_per_class must be positive")
            ordered = ordered[:max_per_class]
        selected.extend(ordered)
    return sorted(selected)


def _texts(frame: Any, indices: Sequence[int]) -> list[str]:
    """Extract narratives for selected Arrow rows."""

    column = frame["narrative"]
    return [str(column[int(index)].as_py()) for index in indices]


def _labels(frame: Any, indices: Sequence[int]) -> list[str]:
    """Extract product labels for selected Arrow rows."""

    column = frame["product_family"]
    return [str(column[int(index)].as_py()) for index in indices]


def _metrics_from_matrix(matrix: np.ndarray) -> dict[str, Any]:
    """Calculate fixed-order macro, weighted, and per-class metrics."""

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
        true_positive,
        support,
        out=np.zeros(len(MODELED_FAMILIES)),
        where=support != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(len(MODELED_FAMILIES)),
        where=(precision + recall) != 0,
    )
    total = int(support.sum())
    if total == 0:
        raise ValueError("Cannot calculate S5 metrics for empty validation")
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


def _diagnostics(matrix: np.ndarray) -> dict[str, Any]:
    """Build critical-class and global confusion diagnostics."""

    critical_index = MODELED_FAMILIES.index(CRITICAL_CLASS)
    critical_support = int(matrix[critical_index].sum())
    false_negatives = [
        {
            "predicted_class": label,
            "count": int(matrix[critical_index, index]),
            "rate": float(matrix[critical_index, index] / critical_support),
        }
        for index, label in enumerate(MODELED_FAMILIES)
        if index != critical_index and matrix[critical_index, index]
    ]
    false_negatives.sort(key=lambda item: (-item["count"], item["predicted_class"]))
    false_positives = []
    for index, label in enumerate(MODELED_FAMILIES):
        if index == critical_index:
            continue
        count = int(matrix[index, critical_index])
        if count:
            support = int(matrix[index].sum())
            false_positives.append({
                "true_class": label,
                "count": count,
                "rate": float(count / support) if support else 0.0,
            })
    false_positives.sort(key=lambda item: (-item["count"], item["true_class"]))
    total_errors = int(matrix.sum() - np.trace(matrix))
    top_confusions = [
        {
            "true_class": actual,
            "predicted_class": predicted,
            "count": int(matrix[actual_index, predicted_index]),
            "rate": (
                float(matrix[actual_index, predicted_index] / total_errors)
                if total_errors
                else 0.0
            ),
        }
        for actual_index, actual in enumerate(MODELED_FAMILIES)
        for predicted_index, predicted in enumerate(MODELED_FAMILIES)
        if actual_index != predicted_index and matrix[actual_index, predicted_index]
    ]
    top_confusions.sort(
        key=lambda item: (
            -item["count"],
            item["true_class"],
            item["predicted_class"],
        )
    )
    return {
        "critical_false_negatives": false_negatives,
        "critical_false_positives": false_positives,
        "top_confusions": top_confusions,
    }


def _evaluate(
    frame: Any,
    indices: Sequence[int],
    vectorizer: TfidfVectorizer,
    estimator: Any,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate one estimator in batches without retaining predictions."""

    positions = {label: index for index, label in enumerate(MODELED_FAMILIES)}
    matrix = np.zeros((len(MODELED_FAMILIES), len(MODELED_FAMILIES)), dtype=np.int64)
    for start in range(0, len(indices), batch_size):
        batch = indices[start : start + batch_size]
        actual = _labels(frame, batch)
        predicted = estimator.predict(vectorizer.transform(_texts(frame, batch)))
        for truth, prediction in zip(actual, predicted):
            if str(prediction) not in positions:
                raise ValueError("Estimator emitted a label outside S5")
            matrix[positions[truth], positions[str(prediction)]] += 1
    return _metrics_from_matrix(matrix), _diagnostics(matrix)


def _vectorizer(config: S5RepresentationConfig) -> TfidfVectorizer:
    """Build the one frozen S3/S4-compatible word vectorizer."""

    return TfidfVectorizer(
        analyzer=config.analyzer,
        ngram_range=config.ngram_range,
        max_features=config.max_features,
        min_df=config.min_df,
        max_df=config.max_df,
        sublinear_tf=config.sublinear_tf,
        dtype=np.float32,
    )


def _build_estimator(candidate: S5CandidateConfig) -> Any:
    """Instantiate one frozen estimator without fitting it."""

    parameters = dict(candidate.parameters)
    if candidate.estimator == "SGDClassifier":
        return SGDClassifier(**parameters)
    if candidate.estimator == "LinearSVC":
        return LinearSVC(**parameters)
    if candidate.estimator == "ComplementNB":
        parameters.pop("sample_weight")
        return ComplementNB(**parameters)
    raise ValueError(f"Unsupported estimator: {candidate.estimator}")


def _fit_estimator(
    candidate: S5CandidateConfig,
    x_train: Any,
    y_train: Sequence[str],
) -> tuple[Any, list[dict[str, str]]]:
    """Fit one candidate and capture only convergence warnings."""

    estimator = _build_estimator(candidate)
    captured: list[dict[str, str]] = []
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always", ConvergenceWarning)
        if candidate.estimator == "ComplementNB":
            sample_weight = compute_sample_weight("balanced", y_train)
            estimator.fit(x_train, y_train, sample_weight=sample_weight)
        else:
            estimator.fit(x_train, y_train)
        for record in records:
            if issubclass(record.category, ConvergenceWarning):
                captured.append({
                    "category": record.category.__name__,
                    "message": str(record.message),
                })
    return estimator, captured


def _candidate_result(
    candidate: S5CandidateConfig,
    metrics: dict[str, Any],
    diagnostics: dict[str, Any],
    train_rows: int,
    validation_rows: int,
    vocabulary_size: int,
    runtime_seconds: float,
    convergence_warnings: Sequence[Mapping[str, str]],
    config: S5ExperimentConfig,
) -> dict[str, Any]:
    """Assemble metrics, diagnostics, resource timing, and frozen gates."""

    critical = metrics["per_class"][config.critical_class]
    gates = {
        "global_macro_f1": (
            metrics["macro_f1"] >= config.gates.global_macro_f1_min
        ),
        "critical_f1": critical["f1"] >= config.gates.critical_f1_min,
        "critical_precision": (
            critical["precision"] >= config.gates.critical_precision_min
        ),
    }
    return {
        "name": candidate.name,
        "estimator": candidate.estimator,
        "parameters": dict(candidate.parameters),
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "vocabulary_size": vocabulary_size,
        "runtime_seconds": float(runtime_seconds),
        "convergence_warnings": list(convergence_warnings),
        "metrics": metrics,
        "diagnostics": diagnostics,
        "gates": gates,
        "eligible": all(gates.values()),
    }


def _default_reference(config_path: Path) -> Path:
    """Resolve the conventional S4 artifact beside the project config."""

    return config_path.parent.parent / "temp" / "s4" / "s4_results.json"


def _reference_metrics(path: Path, candidate_name: str) -> dict[str, float]:
    """Read the S4 reference metrics required for reproduction."""

    payload = _read_json(path)
    for candidate in payload.get("candidates", []):
        if candidate.get("name") == candidate_name:
            metrics = candidate["metrics"]
            per_class = metrics.get("per_class", metrics)
            critical = per_class[CRITICAL_CLASS]
            return {
                "macro_f1": float(metrics["macro_f1"]),
                "critical_precision": float(critical["precision"]),
                "critical_recall": float(critical["recall"]),
                "critical_f1": float(critical["f1"]),
            }
    raise ValueError("S4 reference candidate is absent")


def _check_reference(
    results: Sequence[Mapping[str, Any]],
    reference_path: Path | None,
    config: S5ExperimentConfig,
    smoke: bool,
) -> dict[str, Any]:
    """Compare the full-run SGD result with the real S4 artifact."""

    if smoke:
        return {"status": "NOT_CHECKED_SMOKE", "passed": False, "deltas": {}}
    if reference_path is None or not reference_path.exists():
        return {"status": "REFERENCE_NOT_PROVIDED", "passed": False, "deltas": {}}
    reference = _reference_metrics(
        reference_path,
        str(config.reference_reproduction["candidate"]),
    )
    result = next(
        (item for item in results if item["name"] == "sgd_log_loss_balanced_reference"),
        None,
    )
    if result is None:
        return {"status": "REFERENCE_RESULT_MISSING", "passed": False, "deltas": {}}
    result_per_class = result["metrics"].get(
        "per_class", result["metrics"]
    )
    critical = result_per_class[CRITICAL_CLASS]
    actual = {
        "macro_f1": float(result["metrics"]["macro_f1"]),
        "critical_precision": float(critical["precision"]),
        "critical_recall": float(critical["recall"]),
        "critical_f1": float(critical["f1"]),
    }
    deltas = {key: actual[key] - reference[key] for key in reference}
    tolerance = float(
        config.reference_reproduction.get(
            "max_absolute_metric_delta",
            DEFAULT_REFERENCE_TOLERANCE,
        )
    )
    passed = all(abs(delta) <= tolerance for delta in deltas.values())
    return {
        "status": "PASSED" if passed else "FAILED",
        "passed": passed,
        "tolerance": tolerance,
        "reference": reference,
        "actual": actual,
        "deltas": deltas,
    }


def _progress_payload(
    config: S5ExperimentConfig,
    signature: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    fit_event: Mapping[str, Any] | None,
    status: str,
    complete: bool,
    parity: Mapping[str, Any],
    error: Exception | None = None,
) -> dict[str, Any]:
    """Build a serializable incremental S5 artifact."""

    payload: dict[str, Any] = {
        "schema_version": S5_SCHEMA_VERSION,
        "status": status,
        "complete": complete,
        "claim_boundary": "NO_TEST_STRESS_OR_MONITOR_ACCESS",
        "selection_status": "IN_PROGRESS" if not complete else None,
        "recommended_candidate": None,
        "signature": dict(signature),
        "config": json.loads(json.dumps(asdict(config))),
        "candidates": list(results),
        "deferred_candidates": dict(config.deferred_candidates),
        "completed_candidates": [item["name"] for item in results],
        "vectorizer_fit_count": 1 if fit_event else 0,
        "representation_fit_event": fit_event,
        "reference_reproduction": dict(parity),
    }
    if error is not None:
        payload["error"] = {"type": type(error).__name__, "message": str(error)}
    return payload


def run_s5(
    scientific_cache_path: str | Path,
    artifact_path: str | Path,
    config_path: str | Path,
    *,
    reference_artifact_path: str | Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    smoke_max_per_class: int | None = None,
    tracker: Tracker | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Run the S5 development benchmark on train and validation only.

    Args:
        scientific_cache_path: S3 scientific cache path.
        artifact_path: Incremental JSON artifact destination.
        config_path: Frozen S5 configuration path.
        reference_artifact_path: Optional real S4 artifact for parity checking.
        batch_size: Validation transform/prediction batch size.
        smoke_max_per_class: Deterministic diagnostic cap per class.
        tracker: Optional MLflow-compatible tracker.
        force_refresh: Ignore a matching complete artifact.

    Returns:
        Complete or diagnostic-only artifact payload.

    Raises:
        ValueError: If the protocol or inputs violate the S5 boundary.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if smoke_max_per_class is not None and smoke_max_per_class <= 0:
        raise ValueError("smoke_max_per_class must be positive")
    cache_path = Path(scientific_cache_path).expanduser().resolve()
    artifact = Path(artifact_path).expanduser().resolve()
    experiment_path = Path(config_path).expanduser().resolve()
    reference_path = (
        Path(reference_artifact_path).expanduser().resolve()
        if reference_artifact_path is not None
        else _default_reference(experiment_path)
    )
    if not cache_path.exists() or not experiment_path.exists():
        raise FileNotFoundError("S5 scientific cache and config are required")
    config = load_s5_config(experiment_path)
    run_params = {
        "batch_size": batch_size,
        "smoke_max_per_class": smoke_max_per_class,
        "diagnostic_only": smoke_max_per_class is not None,
        "reference_artifact_path": (
            str(reference_path) if reference_path.exists() else None
        ),
    }
    signature = _signature(
        cache_path,
        experiment_path,
        run_params,
        reference_path if reference_path.exists() else None,
    )
    if not force_refresh and artifact.exists():
        cached = _read_json(artifact)
        if cached.get("complete") is True and cached.get("signature") == signature:
            return cached
    results: list[dict[str, Any]] = []
    fit_event: dict[str, Any] | None = None
    parity: dict[str, Any] = {"status": "NOT_CHECKED", "passed": False, "deltas": {}}
    tracker_instance = tracker or NullTracker()
    _write_json_atomic(
        artifact,
        _progress_payload(
            config,
            signature,
            results,
            fit_event,
            "IN_PROGRESS",
            False,
            parity,
        ),
    )
    failure: Exception | None = None
    complete = False
    try:
        frame = read_scientific_frame(cache_path)
        validate_scientific_cache(frame)
        train_indices = _indices_for_partition(
            frame,
            "train",
            smoke_max_per_class,
            config.random_state,
        )
        validation_indices = _indices_for_partition(
            frame,
            "validation",
            smoke_max_per_class,
            config.random_state,
        )
        train_labels = _labels(frame, train_indices)
        if not train_indices or not validation_indices:
            raise ValueError("S5 train and validation must both be non-empty")
        tracker_instance.log_params({
            "schema_version": S5_SCHEMA_VERSION,
            "train_rows": len(train_indices),
            "validation_rows": len(validation_indices),
            "diagnostic_only": smoke_max_per_class is not None,
        })
        vectorizer = _vectorizer(config.representation)
        x_train = vectorizer.fit_transform(_texts(frame, train_indices))
        fit_event = {
            "representation": "word",
            "candidate_names": [candidate.name for candidate in config.candidates],
            "fit_transform_count": 1,
            "vocabulary_size": len(vectorizer.vocabulary_),
        }
        _write_json_atomic(
            artifact,
            _progress_payload(
                config,
                signature,
                results,
                fit_event,
                "IN_PROGRESS",
                False,
                parity,
            ),
        )
        for candidate in config.candidates:
            started = time.perf_counter()
            warnings_seen: list[dict[str, str]] = []
            estimator = None
            try:
                estimator, warnings_seen = _fit_estimator(
                    candidate,
                    x_train,
                    train_labels,
                )
                metrics, diagnostics = _evaluate(
                    frame,
                    validation_indices,
                    vectorizer,
                    estimator,
                    batch_size,
                )
                runtime = time.perf_counter() - started
                results.append(_candidate_result(
                    candidate,
                    metrics,
                    diagnostics,
                    len(train_indices),
                    len(validation_indices),
                    len(vectorizer.vocabulary_),
                    runtime,
                    warnings_seen,
                    config,
                ))
                tracker_instance.log_metrics(
                    {
                        f"{candidate.name}.macro_f1": metrics["macro_f1"],
                        f"{candidate.name}.critical_f1": metrics[
                            "per_class"
                        ][CRITICAL_CLASS]["f1"],
                        f"{candidate.name}.runtime_seconds": runtime,
                    }
                )
                _write_json_atomic(
                    artifact,
                    _progress_payload(
                        config,
                        signature,
                        results,
                        fit_event,
                        "IN_PROGRESS",
                        False,
                        parity,
                    ),
                )
            finally:
                del estimator
                gc.collect()
        parity = _check_reference(
            results,
            reference_path if reference_path.exists() else None,
            config,
            smoke_max_per_class is not None,
        )
        eligible = [item for item in results if item["eligible"]]
        ranked = sorted(eligible, key=lambda item: (
            -item["metrics"]["per_class"][CRITICAL_CLASS]["f1"],
            -item["metrics"]["macro_f1"],
            item["name"],
        ))
        recommendation_allowed = (
            smoke_max_per_class is None and parity.get("passed") is True
        )
        recommended = ranked[0]["name"] if ranked and recommendation_allowed else None
        if smoke_max_per_class is not None:
            status = "DIAGNOSTIC_ONLY"
            selection_status = "DIAGNOSTIC_ONLY"
        elif not parity.get("passed"):
            status = "DEVELOPMENT_COMPLETE_REFERENCE_NOT_VERIFIED"
            selection_status = "REFERENCE_NOT_VERIFIED"
        elif recommended is None:
            status = "DEVELOPMENT_COMPLETE"
            selection_status = "NO_ELIGIBLE_ESTIMATOR"
        else:
            status = "DEVELOPMENT_COMPLETE"
            selection_status = "ELIGIBLE_ESTIMATOR"
        payload = _progress_payload(
            config,
            signature,
            results,
            fit_event,
            status,
            True,
            parity,
        )
        payload["selection_status"] = selection_status
        payload["recommended_candidate"] = recommended
        payload["resources"] = {"candidate_count": len(results)}
        _write_json_atomic(artifact, payload)
        complete = True
        return payload
    except Exception as error:
        failure = error
        _write_json_atomic(
            artifact,
            _progress_payload(
                config,
                signature,
                results,
                fit_event,
                "ERROR",
                False,
                parity,
                error,
            ),
        )
        raise
    finally:
        try:
            tracker_instance.close()
        except Exception as close_error:
            if complete:
                _write_json_atomic(
                    artifact,
                    _progress_payload(
                        config,
                        signature,
                        results,
                        fit_event,
                        "ERROR",
                        False,
                        parity,
                        close_error,
                    ),
                )
                raise
            if failure is None:
                    _write_json_atomic(
                        artifact,
                        _progress_payload(
                            config,
                            signature,
                            results,
                            fit_event,
                            "ERROR",
                            False,
                            parity,
                            close_error,
                        ),
                    )


def run_s5_smoke(
    scientific_cache_path: str | Path,
    artifact_path: str | Path,
    config_path: str | Path,
    *,
    max_per_class: int = 8,
    tracker: Tracker | None = None,
) -> dict[str, Any]:
    """Run a deterministic, explicitly diagnostic-only S5 smoke test."""

    return run_s5(
        scientific_cache_path,
        artifact_path,
        config_path,
        smoke_max_per_class=max_per_class,
        tracker=tracker,
    )
