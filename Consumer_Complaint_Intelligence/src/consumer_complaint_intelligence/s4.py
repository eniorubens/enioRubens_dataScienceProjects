"""S4 error diagnostics and controlled text-representation challenge."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_class_weight

from .s3 import read_scientific_frame
from .temporal_split import MODELED_FAMILIES
from .tracking import NullTracker, Tracker


S4_SCHEMA_VERSION = "s4-experiment-v1"
S4_CODE_SCHEMA = "s4-runtime-v1"
DEVELOPMENT_PARTITIONS = ("train", "validation")
SEALED_PARTITIONS = ("test", "stress", "monitor")
CRITICAL_CLASS = "debt_credit_management"
DEFAULT_BATCH_SIZE = 4_096


@dataclass(frozen=True, slots=True)
class S4CandidateConfig:
    """Describe one frozen S4 vectorizer and classifier candidate."""

    name: str
    representation: str
    analyzer: str
    ngram_range: tuple[int, int]
    max_features: int
    min_df: int
    max_df: float
    sublinear_tf: bool
    dtype: str
    classifier: str
    loss: str
    class_weight: str
    max_iter: int
    tol: float

    def validate(self) -> None:
        """Validate the candidate against the supported S4 search space."""

        if self.representation not in {"word", "char_wb"}:
            raise ValueError(f"Unsupported representation: {self.representation}")
        if self.analyzer != self.representation:
            raise ValueError("Candidate analyzer must match its representation")
        if self.classifier != "SGDClassifier" or self.loss != "log_loss":
            raise ValueError("S4 supports only the frozen SGD log_loss baseline")
        if self.class_weight not in {"balanced", "sqrt_balanced"}:
            raise ValueError("Unsupported S4 class-weight scheme")
        if self.max_features <= 0 or self.min_df <= 0:
            raise ValueError("TF-IDF limits must be positive")
        if self.max_iter <= 0 or self.tol <= 0:
            raise ValueError("Classifier limits must be positive")


@dataclass(frozen=True, slots=True)
class S4GateConfig:
    """Define the frozen development eligibility gates."""

    global_macro_f1_min: float
    critical_f1_min: float
    critical_precision_min: float


@dataclass(frozen=True, slots=True)
class S4ExperimentConfig:
    """Represent the complete versioned S4 experiment configuration."""

    schema_version: str
    status: str
    approved_on: str
    fit_partition: str
    evaluation_partition: str
    sealed_partitions: tuple[str, ...]
    random_state: int
    critical_class: str
    gates: S4GateConfig
    candidates: tuple[S4CandidateConfig, ...]
    weight_formula: str

    def validate(self) -> None:
        """Validate the frozen experiment status and candidate set."""

        if self.schema_version != S4_SCHEMA_VERSION:
            raise ValueError("Unexpected S4 schema version")
        if self.status != "FROZEN_FOR_S4_DEVELOPMENT":
            raise ValueError("S4 experiment is not frozen for development")
        if (self.fit_partition, self.evaluation_partition) != (
            "train",
            "validation",
        ):
            raise ValueError("S4 must fit on train and evaluate on validation")
        if tuple(self.sealed_partitions) != SEALED_PARTITIONS:
            raise ValueError("S4 sealed partition boundary is invalid")
        if self.critical_class != CRITICAL_CLASS:
            raise ValueError("S4 critical class is invalid")
        if self.random_state < 0:
            raise ValueError("random_state must be non-negative")
        expected = {
            "word_balanced_reference",
            "word_sqrt_balanced",
            "char_wb_balanced",
            "char_wb_sqrt_balanced",
        }
        actual = {candidate.name for candidate in self.candidates}
        if actual != expected:
            raise ValueError("S4 candidate set differs from the frozen design")
        if len(self.candidates) != len(actual):
            raise ValueError("S4 candidate names must be unique")
        for candidate in self.candidates:
            candidate.validate()


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object and reject non-object payloads."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def load_s4_config(path: str | Path) -> S4ExperimentConfig:
    """Load and validate the frozen S4 experiment configuration.

    Args:
        path: Versioned S4 JSON configuration path.

    Returns:
        Validated experiment configuration.
    """

    payload = _read_json(Path(path).expanduser().resolve())
    candidates = tuple(
        S4CandidateConfig(
            name=name,
            representation=str(values["representation"]),
            analyzer=str(values["analyzer"]),
            ngram_range=tuple(int(value) for value in values["ngram_range"]),
            max_features=int(values["max_features"]),
            min_df=int(values["min_df"]),
            max_df=float(values["max_df"]),
            sublinear_tf=bool(values["sublinear_tf"]),
            dtype=str(values["dtype"]),
            classifier=str(values["classifier"]),
            loss=str(values["loss"]),
            class_weight=str(values["class_weight"]),
            max_iter=int(values["max_iter"]),
            tol=float(values["tol"]),
        )
        for name, values in payload["candidates"].items()
    )
    gates = S4GateConfig(
        global_macro_f1_min=float(payload["gates"]["global_macro_f1_min"]),
        critical_f1_min=float(payload["gates"]["critical_f1_min"]),
        critical_precision_min=float(
            payload["gates"]["critical_precision_min"]
        ),
    )
    config = S4ExperimentConfig(
        schema_version=str(payload["schema_version"]),
        status=str(payload["status"]),
        approved_on=str(payload["approved_on"]),
        fit_partition=str(payload["fit_partition"]),
        evaluation_partition=str(payload["evaluation_partition"]),
        sealed_partitions=tuple(str(value) for value in payload["sealed_partitions"]),
        random_state=int(payload["random_state"]),
        critical_class=str(payload["critical_class"]),
        gates=gates,
        candidates=candidates,
        weight_formula=str(payload["weight_formula"]),
    )
    config.validate()
    return config


def sqrt_balanced_weights(
    labels: Sequence[str], classes: Sequence[str] = MODELED_FAMILIES
) -> dict[str, float]:
    """Return sample-mean-one square-root balanced class weights.

    The balanced weight is ``n / (k * n_i)``. The S4 challenge uses its
    square root and normalizes it by ``sum(p_i * sqrt(weight_i))`` so the
    resulting average sample weight remains one.

    Args:
        labels: Training labels used to estimate class frequencies.
        classes: Fixed ordered class universe.

    Returns:
        Mapping from class name to normalized square-root weight.

    Raises:
        ValueError: If a class is absent or labels are empty.
    """

    if not labels:
        raise ValueError("Cannot calculate weights from empty labels")
    class_array = np.asarray(tuple(classes), dtype=object)
    label_array = np.asarray([str(label) for label in labels], dtype=object)
    observed = set(label_array.tolist())
    missing = set(class_array.tolist()).difference(observed)
    if missing:
        raise ValueError(f"Training labels miss classes: {sorted(missing)}")
    balanced = compute_class_weight(
        class_weight="balanced", classes=class_array, y=label_array
    )
    square_root = np.sqrt(np.asarray(balanced, dtype=float))
    probabilities = np.asarray(
        [np.mean(label_array == label) for label in class_array], dtype=float
    )
    normalization = float(np.dot(probabilities, square_root))
    if normalization <= 0:
        raise ValueError("Invalid square-root weight normalization")
    return {
        str(label): float(weight / normalization)
        for label, weight in zip(class_array, square_root)
    }


def validate_scientific_cache(frame: Any) -> None:
    """Validate the Arrow scientific cache and its development boundary.

    Args:
        frame: Arrow table loaded from the S3 scientific cache.

    Raises:
        ValueError: If the cache contains sealed rows, duplicates, or invalid
            labels.
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
    actual = set(frame.column_names)
    if actual != expected:
        raise ValueError("Scientific cache columns do not match the S4 contract")
    partitions = {str(value.as_py()) for value in frame["partition_name"]}
    if not partitions.issubset(set(DEVELOPMENT_PARTITIONS)):
        raise ValueError("S4 scientific cache contains a sealed partition")
    labels = {str(value.as_py()) for value in frame["product_family"]}
    if not labels.issubset(set(MODELED_FAMILIES)):
        raise ValueError("S4 scientific cache contains an out-of-scope class")
    identities = set()
    ids = set()
    for group_hash, length, complaint_id in zip(
        frame["normalized_group_hash"],
        frame["normalized_length"],
        frame["Complaint ID"],
    ):
        identity = (str(group_hash.as_py()), int(length.as_py()))
        identifier = int(complaint_id.as_py())
        if identity in identities or identifier in ids:
            raise ValueError("S4 scientific cache must contain unique groups and IDs")
        identities.add(identity)
        ids.add(identifier)


def _file_sha256(path: Path) -> str:
    """Return one file's uppercase SHA256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _signature(cache_path: Path, config_path: Path) -> dict[str, Any]:
    """Build the S4 cache invalidation signature."""

    return {
        "schema_version": S4_SCHEMA_VERSION,
        "code_schema": S4_CODE_SCHEMA,
        "scientific_cache": {
            "path": str(cache_path.resolve()),
            "size": cache_path.stat().st_size,
            "mtime_ns": cache_path.stat().st_mtime_ns,
        },
        "config": {
            "path": str(config_path.resolve()),
            "sha256": _file_sha256(config_path),
        },
        "boundary": {
            "fit": "train",
            "evaluation": "validation",
            "sealed": list(SEALED_PARTITIONS),
        },
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Persist an artifact atomically as UTF-8 JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _progress_payload(
    config: S4ExperimentConfig,
    signature: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    fit_events: Sequence[Mapping[str, Any]],
    status: str,
    complete: bool,
    error: Exception | None = None,
) -> dict[str, Any]:
    """Build an incremental artifact payload without runtime objects."""

    payload: dict[str, Any] = {
        "schema_version": S4_SCHEMA_VERSION,
        "status": status,
        "complete": complete,
        "claim_boundary": "NO_TEST_STRESS_OR_MONITOR_ACCESS",
        "recommended_candidate": None,
        "selection_status": (
            "IN_PROGRESS" if status == "IN_PROGRESS" else "ERROR"
        ) if not complete else None,
        "signature": dict(signature),
        "config": json.loads(json.dumps(asdict(config))),
        "candidates": list(results),
        "completed_candidates": [item["name"] for item in results],
        "vectorizer_fit_count": len(fit_events),
        "representation_fit_events": list(fit_events),
    }
    if error is not None:
        payload["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    return payload


def _indices_for_partition(
    frame: Any,
    partition: str,
    max_per_class: int | None,
    random_state: int,
) -> list[int]:
    """Select deterministic per-class Arrow row indices for a run."""

    if partition not in DEVELOPMENT_PARTITIONS:
        raise ValueError("S4 index selection cannot access sealed partitions")
    by_class: dict[str, list[int]] = {label: [] for label in MODELED_FAMILIES}
    labels = frame["product_family"]
    partitions = frame["partition_name"]
    identifiers = frame["Complaint ID"]
    for index in range(len(frame)):
        if str(partitions[index].as_py()) == partition:
            label = str(labels[index].as_py())
            by_class[label].append(index)
    selected: list[int] = []
    for label in MODELED_FAMILIES:
        ordered = sorted(
            by_class[label],
            key=lambda index: hashlib.sha256(
                f"{random_state}|{label}|{identifiers[index].as_py()}".encode()
            ).hexdigest(),
        )
        if max_per_class is not None:
            if max_per_class <= 0:
                raise ValueError("max_per_class must be positive")
            ordered = ordered[:max_per_class]
        selected.extend(ordered)
    return sorted(selected)


def _texts(frame: Any, indices: Sequence[int]) -> list[str]:
    """Extract bounded narrative text from Arrow row indices."""

    column = frame["narrative"]
    return [str(column[int(index)].as_py()) for index in indices]


def _labels(frame: Any, indices: Sequence[int]) -> list[str]:
    """Extract bounded labels from Arrow row indices."""

    column = frame["product_family"]
    return [str(column[int(index)].as_py()) for index in indices]


def _metrics_from_matrix(matrix: np.ndarray) -> dict[str, Any]:
    """Calculate fixed-label metrics from a confusion matrix."""

    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros(len(MODELED_FAMILIES), dtype=float),
        where=predicted != 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros(len(MODELED_FAMILIES), dtype=float),
        where=support != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(len(MODELED_FAMILIES), dtype=float),
        where=(precision + recall) != 0,
    )
    total = int(support.sum())
    if total == 0:
        raise ValueError("Cannot calculate S4 metrics for empty validation")
    per_class = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(MODELED_FAMILIES)
    }
    return {
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.dot(f1, support) / total),
        "balanced_accuracy": float(recall.mean()),
        "per_class": per_class,
        "row_count": total,
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def _diagnostics(matrix: np.ndarray) -> dict[str, Any]:
    """Build critical-class and global aggregate confusion diagnostics."""

    critical_index = MODELED_FAMILIES.index(CRITICAL_CLASS)
    critical_support = int(matrix[critical_index].sum())
    false_negatives = []
    for index, label in enumerate(MODELED_FAMILIES):
        if index == critical_index:
            continue
        count = int(matrix[critical_index, index])
        if count:
            false_negatives.append(
                {
                    "predicted_class": label,
                    "count": count,
                    "rate": float(count / critical_support),
                }
            )
    false_negatives.sort(key=lambda item: (-item["count"], item["predicted_class"]))
    false_positives = []
    for index, label in enumerate(MODELED_FAMILIES):
        if index == critical_index:
            continue
        true_support = int(matrix[index].sum())
        count = int(matrix[index, critical_index])
        if count:
            false_positives.append(
                {
                    "true_class": label,
                    "count": count,
                    "rate": float(count / true_support) if true_support else 0.0,
                }
            )
    false_positives.sort(key=lambda item: (-item["count"], item["true_class"]))
    total_errors = int(matrix.sum() - np.trace(matrix))
    top_confusions = []
    for actual_index, actual in enumerate(MODELED_FAMILIES):
        for predicted_index, predicted in enumerate(MODELED_FAMILIES):
            count = int(matrix[actual_index, predicted_index])
            if actual_index != predicted_index and count:
                top_confusions.append(
                    {
                        "true_class": actual,
                        "predicted_class": predicted,
                        "count": count,
                        "rate": float(count / total_errors)
                        if total_errors
                        else 0.0,
                    }
                )
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
    estimator: SGDClassifier,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate a candidate in batches without retaining predictions."""

    positions = {label: index for index, label in enumerate(MODELED_FAMILIES)}
    matrix = np.zeros((len(MODELED_FAMILIES), len(MODELED_FAMILIES)), dtype=np.int64)
    for start in range(0, len(indices), batch_size):
        batch = indices[start:start + batch_size]
        actual = _labels(frame, batch)
        predicted = estimator.predict(vectorizer.transform(_texts(frame, batch)))
        for truth, prediction in zip(actual, predicted):
            try:
                matrix[positions[truth], positions[str(prediction)]] += 1
            except KeyError as error:
                raise ValueError("Estimator emitted a label outside S4") from error
    return _metrics_from_matrix(matrix), _diagnostics(matrix)


def _vectorizer(candidate: S4CandidateConfig) -> TfidfVectorizer:
    """Build a vectorizer from one frozen candidate configuration."""

    return TfidfVectorizer(
        analyzer=candidate.analyzer,
        ngram_range=candidate.ngram_range,
        max_features=candidate.max_features,
        min_df=candidate.min_df,
        max_df=candidate.max_df,
        sublinear_tf=candidate.sublinear_tf,
        dtype=np.float32,
    )


def _class_weight(candidate: S4CandidateConfig, labels: Sequence[str]) -> Any:
    """Resolve the frozen balanced or square-root-balanced weight scheme."""

    if candidate.class_weight == "balanced":
        return "balanced"
    return sqrt_balanced_weights(labels)


def _candidate_result(
    candidate: S4CandidateConfig,
    metrics: dict[str, Any],
    diagnostics: dict[str, Any],
    train_rows: int,
    validation_rows: int,
    vocabulary_size: int,
    config: S4ExperimentConfig,
) -> dict[str, Any]:
    """Assemble one candidate result and evaluate all frozen gates."""

    critical = metrics["per_class"][config.critical_class]
    gates = {
        "global_macro_f1": metrics["macro_f1"] >= config.gates.global_macro_f1_min,
        "critical_f1": critical["f1"] >= config.gates.critical_f1_min,
        "critical_precision": (
            critical["precision"] >= config.gates.critical_precision_min
        ),
    }
    return {
        "name": candidate.name,
        "representation": candidate.representation,
        "class_weight": candidate.class_weight,
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "vocabulary_size": vocabulary_size,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "gates": gates,
        "eligible": all(gates.values()),
    }


def run_s4(
    scientific_cache_path: str | Path,
    artifact_path: str | Path,
    config_path: str | Path,
    tracker: Tracker | None = None,
    force_refresh: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    smoke_max_per_class: int | None = None,
) -> dict[str, Any]:
    """Run or load the S4 development-only representation challenge.

    Args:
        scientific_cache_path: S3 scientific Arrow/Parquet cache.
        artifact_path: Incremental S4 JSON artifact destination.
        config_path: Frozen S4 experiment configuration.
        tracker: Optional framework-neutral experiment tracker.
        force_refresh: Recompute even when the artifact signature matches.
        batch_size: Maximum validation rows processed per prediction batch.
        smoke_max_per_class: Deterministic per-class cap for diagnostics only.

    Returns:
        Complete S4 result payload. No estimator is persisted.

    Raises:
        ValueError: If the cache boundary or run arguments are invalid.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if smoke_max_per_class is not None and smoke_max_per_class <= 0:
        raise ValueError("smoke_max_per_class must be positive")
    cache_path = Path(scientific_cache_path).expanduser().resolve()
    artifact = Path(artifact_path).expanduser().resolve()
    experiment_path = Path(config_path).expanduser().resolve()
    if not cache_path.exists() or not experiment_path.exists():
        raise FileNotFoundError("S4 scientific cache and config are required")
    config = load_s4_config(experiment_path)
    signature = _signature(cache_path, experiment_path)
    signature["run"] = {
        "batch_size": batch_size,
        "smoke_max_per_class": smoke_max_per_class,
        "diagnostic_only": smoke_max_per_class is not None,
    }
    if not force_refresh and artifact.exists():
        cached = _read_json(artifact)
        if cached.get("complete") is True and cached.get("signature") == signature:
            return cached
    results: list[dict[str, Any]] = []
    fit_events: list[dict[str, Any]] = []
    tracker_instance = tracker or NullTracker()
    _write_json_atomic(
        artifact,
        _progress_payload(
            config,
            signature,
            results,
            fit_events,
            "IN_PROGRESS",
            False,
        ),
    )
    failure: Exception | None = None
    complete = False
    try:
        frame = read_scientific_frame(cache_path)
        validate_scientific_cache(frame)
        train_indices = _indices_for_partition(
            frame, "train", smoke_max_per_class, config.random_state
        )
        validation_indices = _indices_for_partition(
            frame, "validation", smoke_max_per_class, config.random_state
        )
        train_labels = _labels(frame, train_indices)
        if not train_indices or not validation_indices:
            raise ValueError("S4 train and validation must both be non-empty")
        tracker_instance.log_params(
            {
                "schema_version": S4_SCHEMA_VERSION,
                "train_rows": len(train_indices),
                "validation_rows": len(validation_indices),
                "diagnostic_only": smoke_max_per_class is not None,
            }
        )
        for representation in ("word", "char_wb"):
            group = [
                candidate
                for candidate in config.candidates
                if candidate.representation == representation
            ]
            vectorizer = _vectorizer(group[0])
            x_train = vectorizer.fit_transform(_texts(frame, train_indices))
            fit_events.append(
                {
                    "representation": representation,
                    "candidate_names": [item.name for item in group],
                    "fit_transform_count": 1,
                }
            )
            _write_json_atomic(
                artifact,
                _progress_payload(
                    config,
                    signature,
                    results,
                    fit_events,
                    "IN_PROGRESS",
                    False,
                ),
            )
            for candidate in group:
                estimator = SGDClassifier(
                    loss=candidate.loss,
                    class_weight=_class_weight(candidate, train_labels),
                    max_iter=candidate.max_iter,
                    tol=candidate.tol,
                    random_state=config.random_state,
                )
                estimator.fit(x_train, train_labels)
                metrics, diagnostics = _evaluate(
                    frame,
                    validation_indices,
                    vectorizer,
                    estimator,
                    batch_size,
                )
                results.append(
                    _candidate_result(
                        candidate,
                        metrics,
                        diagnostics,
                        len(train_indices),
                        len(validation_indices),
                        len(vectorizer.vocabulary_),
                        config,
                    )
                )
                _write_json_atomic(
                    artifact,
                    _progress_payload(
                        config,
                        signature,
                        results,
                        fit_events,
                        "IN_PROGRESS",
                        False,
                    ),
                )
                tracker_instance.log_metrics(
                    {
                        f"{candidate.name}.macro_f1": metrics["macro_f1"],
                        f"{candidate.name}.critical_f1": metrics["per_class"][
                            CRITICAL_CLASS
                        ]["f1"],
                    }
                )
            del x_train
            del vectorizer
        eligible = [result for result in results if result["eligible"]]
        ranked = sorted(
            eligible,
            key=lambda result: (
                -result["metrics"]["per_class"][CRITICAL_CLASS]["f1"],
                -result["metrics"]["macro_f1"],
                result["name"],
            ),
        )
        recommended = ranked[0]["name"] if ranked else None
        payload = _progress_payload(
            config,
            signature,
            results,
            fit_events,
            "DIAGNOSTIC_ONLY" if smoke_max_per_class else "DEVELOPMENT_COMPLETE",
            True,
        )
        payload["selection_status"] = (
            "NO_ELIGIBLE_CHALLENGER"
            if recommended is None
            else "ELIGIBLE_CHALLENGER"
        )
        payload["recommended_candidate"] = recommended
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
                fit_events,
                "ERROR",
                False,
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
                        fit_events,
                        "ERROR",
                        False,
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
                        fit_events,
                        "ERROR",
                        False,
                        close_error,
                    ),
                )


def run_s4_smoke(
    scientific_cache_path: str | Path,
    artifact_path: str | Path,
    config_path: str | Path,
    max_per_class: int = 8,
    tracker: Tracker | None = None,
) -> dict[str, Any]:
    """Run a deterministic, explicitly diagnostic-only S4 smoke test."""

    return run_s4(
        scientific_cache_path,
        artifact_path,
        config_path,
        tracker=tracker,
        smoke_max_per_class=max_per_class,
    )
