"""Classical V2 benchmark runner with a sealed development boundary.

The runner owns the D1 candidate comparison only. It never reads the raw
dataset, never unlocks a sealed partition, and persists aggregate evidence
without narratives, identifiers, individual margins, or fitted estimators.
"""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from .s3 import DATASET_COLUMNS
from .s6 import CRITICAL_CLASS, MODELED_FAMILIES
from .s7 import load_s7_predictor
from .v2_detector import (
    HARD_NEGATIVE,
    RANDOM_OVER,
    WEIGHTED_FULL,
    WORD_CHAR_TFIDF_ALIAS,
    WORD_TFIDF_ALIAS,
    build_estimator,
    build_vectorizer,
    combine_detector_with_fallback,
    count_override_decisions,
    search_detector_threshold_exact,
)
from .v2_protocol import (
    DEFAULT_CONFIG,
    V2Protocol,
    calculate_safety_margins,
    calculate_scientific_gates,
    load_v2_protocol,
    validate_baseline_artifacts,
)


RESULT_SCHEMA = "v2-classical-benchmark-v1"
MANIFEST_SCHEMA = "v2-classical-results-manifest-v1"
CODE_SCHEMA = "v2-benchmark-runtime-v3"
DEFAULT_CACHE = "temp/s3/scientific.parquet"
DEFAULT_ARTIFACT = "temp/v2/v2_classical_benchmark.json"
DEFAULT_MANIFEST = "config/v2_classical_results.json"
DEFAULT_BATCH_SIZE = 4096
RANDOM_STATE = 42
REPRESENTATIONS = (WORD_TFIDF_ALIAS, WORD_CHAR_TFIDF_ALIAS)
CV_FOLDS = 3
_EXPECTED_COLUMNS = tuple(DATASET_COLUMNS)
_READ_COLUMNS = (
    "received_date",
    "product_family",
    "normalized_group_hash",
    "normalized_length",
    "partition_name",
    "narrative",
)
_FORBIDDEN_RESULT_KEYS = {
    "ids",
    "indices",
    "narratives",
    "texts",
    "scores",
    "models",
    "model",
}


@dataclass(frozen=True, slots=True)
class _Scope:
    """Hold one in-memory development scope during a full run."""

    texts: tuple[str, ...]
    labels: tuple[str, ...]
    groups: tuple[tuple[str, int], ...]


@dataclass(slots=True)
class _RepresentationCache:
    """Reuse one fitted representation across all candidates."""

    vectorizer: Any
    fit_matrix: Any
    calibration_matrix: Any


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object from disk."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object")
    return payload


def _sha256(path: Path) -> str:
    """Return an uppercase SHA256 digest for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _signature(path: Path) -> dict[str, Any]:
    """Return a portable file signature without an absolute path."""

    if not path.is_file():
        raise ValueError(f"Required artifact is missing: {path}")
    return {"sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _relative(path: Path, root: Path) -> str:
    """Return one project-relative POSIX path."""

    return path.resolve().relative_to(root.resolve()).as_posix()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one JSON payload through a same-directory temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with open(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=True, indent=2, sort_keys=True)
            stream.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_path(root: Path, value: str) -> Path:
    """Resolve one project-relative protocol path."""

    if not isinstance(value, str) or not value:
        raise ValueError("Configured paths must be non-empty strings")
    candidate = Path(value)
    if candidate.is_absolute() or candidate.drive or ".." in candidate.parts:
        raise ValueError("Configured paths must be project-relative")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("Configured path escapes the project root")
    return resolved


def _protocol_signature(protocol_path: Path) -> str:
    """Return the protocol hash used by the cache signature."""

    return _sha256(protocol_path)


def _candidate_catalog() -> tuple[dict[str, Any], ...]:
    """Build the exact frozen 30-candidate catalog."""

    candidates: list[dict[str, Any]] = []
    for representation in REPRESENTATIONS:
        for c_value in (0.1, 0.3, 1.0):
            candidates.append(_candidate(representation, c_value, WEIGHTED_FULL))
            for ratio in (0.05, 0.1, 0.2):
                candidates.append(
                    _candidate(
                        representation,
                        c_value,
                        RANDOM_OVER,
                        sampling_strategy=ratio,
                    )
                )
            candidates.append(_candidate(representation, c_value, HARD_NEGATIVE))
    if len(candidates) != 30:
        raise AssertionError("The frozen V2 catalog must contain 30 candidates")
    return tuple(candidates)


def _candidate(
    representation: str,
    c_value: float,
    strategy: str,
    *,
    sampling_strategy: float | None = None,
) -> dict[str, Any]:
    """Create one aggregate-safe candidate descriptor."""

    suffix = strategy
    if sampling_strategy is not None:
        suffix = f"{strategy}_{sampling_strategy:g}"
    name = f"{representation}_c_{c_value:g}_{suffix}"
    return {
        "candidate_id": name,
        "representation": representation,
        "C": float(c_value),
        "balance_strategy": strategy,
        "sampling_strategy": sampling_strategy,
    }


def candidate_catalog() -> tuple[dict[str, Any], ...]:
    """Return a detached copy of the exact 30-candidate catalog."""

    return tuple(copy.deepcopy(item) for item in _candidate_catalog())


def _validate_result_privacy(payload: Mapping[str, Any]) -> None:
    """Reject persisted aggregate payloads containing row-level evidence."""

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                normalized = str(key).lower()
                if normalized in _FORBIDDEN_RESULT_KEYS:
                    raise ValueError(f"Private V2 field persisted at {path}.{key}")
                visit(child, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(payload, "result")


def _result_signature(
    protocol_path: Path,
    cache_signature: Mapping[str, Any],
    s7_signatures: Mapping[str, Mapping[str, Any]],
) -> str:
    """Build the full-run signature from code and frozen file hashes."""

    value = {
        "code_schema": CODE_SCHEMA,
        "protocol": _signature(protocol_path),
        "cache": dict(cache_signature),
        "s7": {key: dict(item) for key, item in sorted(s7_signatures.items())},
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True).encode("utf-8")
    ).hexdigest().upper()


def _date_value(value: Any) -> date:
    """Convert one parquet date scalar to an ISO date."""

    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _scope_windows(protocol: V2Protocol) -> dict[str, tuple[str, date, date]]:
    """Return the three frozen temporal windows."""

    windows = protocol.payload["development_windows"]
    result = {}
    for role, window in windows.items():
        result[role] = (
            str(window["partition"]),
            date.fromisoformat(str(window["start"])),
            date.fromisoformat(str(window["end"])),
        )
    return result


def _read_development_cache(
    cache_path: Path,
    protocol: V2Protocol,
    batch_size: int,
) -> dict[str, _Scope]:
    """Read only the frozen S3 cache and split it by frozen dates."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    parquet = pq.ParquetFile(cache_path)
    if tuple(parquet.schema_arrow.names) != _EXPECTED_COLUMNS:
        raise ValueError("Scientific cache columns differ from the S3 contract")
    windows = _scope_windows(protocol)
    buckets: dict[str, list[tuple[str, str, tuple[str, int]]]] = {
        role: [] for role in windows
    }
    seen_groups: set[tuple[str, int]] = set()
    for batch in parquet.iter_batches(
        columns=list(_READ_COLUMNS), batch_size=batch_size
    ):
        values = batch.to_pydict()
        for index, raw_partition in enumerate(values["partition_name"]):
            partition = str(raw_partition)
            if partition not in protocol.allowed_partitions:
                raise ValueError("Scientific cache contains a sealed partition")
            received = _date_value(values["received_date"][index])
            group = (
                str(values["normalized_group_hash"][index]),
                int(values["normalized_length"][index]),
            )
            if group in seen_groups:
                raise ValueError("Scientific cache groups must be unique")
            seen_groups.add(group)
            role = None
            for candidate_role, (expected_partition, start, end) in windows.items():
                if partition == expected_partition and start <= received <= end:
                    role = candidate_role
                    break
            if role is None:
                raise ValueError("Scientific cache row is outside frozen dates")
            text = values["narrative"][index]
            label = values["product_family"][index]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Scientific cache narrative is invalid")
            if str(label) not in MODELED_FAMILIES:
                raise ValueError("Scientific cache contains an unknown family")
            buckets[role].append((text, str(label), group))
    result = {}
    for role, rows in buckets.items():
        if not rows:
            raise ValueError(f"Frozen development scope is empty: {role}")
        result[role] = _Scope(
            tuple(item[0] for item in rows),
            tuple(item[1] for item in rows),
            tuple(item[2] for item in rows),
        )
    return result


def _prediction_labels(batch: Any) -> tuple[str, ...]:
    """Extract labels from an S7 PredictionBatch or a test double."""

    predictions = getattr(batch, "predictions", batch)
    labels = []
    for item in predictions:
        labels.append(str(getattr(item, "label", item)))
    if any(label not in MODELED_FAMILIES for label in labels):
        raise ValueError("Fallback returned a label outside the modeled families")
    return tuple(labels)


def _fallback_labels(
    predictor: Any,
    scope: _Scope,
    batch_size: int,
) -> tuple[str, ...]:
    """Predict the frozen S7 fallback in bounded batches."""

    labels: list[str] = []
    for start in range(0, len(scope.texts), batch_size):
        batch = scope.texts[start : start + batch_size]
        labels.extend(
            _prediction_labels(predictor.predict(batch, input_language="en-US"))
        )
    if len(labels) != len(scope.texts):
        raise ValueError("Fallback prediction count differs from the cache")
    return tuple(labels)


def generate_hard_negative_indices(
    texts: Sequence[str],
    labels: Sequence[str],
    *,
    hard_negative_per_positive: int = 10,
    background_per_positive: int = 5,
    n_splits: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> tuple[int, ...]:
    """Generate one deterministic OOF hard-negative training pool.

    The returned indices are intentionally an in-memory development object.
    Callers must not persist them. Every critical example is retained; the
    non-critical pool contains the highest OOF critical margins followed by a
    deterministic background sample.

    Args:
        texts: Inner-fit English narratives.
        labels: Multiclass labels aligned with ``texts``.
        hard_negative_per_positive: Hard negatives retained per critical row.
        background_per_positive: Background negatives retained per critical row.
        n_splits: Number of deterministic stratified OOF folds.
        random_state: Frozen splitter seed.

    Returns:
        Stable row positions for the in-memory hard-negative pool.

    Raises:
        ValueError: If the binary OOF problem cannot be stratified.
    """

    values = tuple(texts)
    actual = tuple(labels)
    if len(values) != len(actual) or not values:
        raise ValueError("Hard-negative inputs must be non-empty and aligned")
    if any(not isinstance(text, str) or not text.strip() for text in values):
        raise ValueError("Hard-negative texts must be non-empty strings")
    unknown = set(actual) - set(MODELED_FAMILIES)
    if unknown:
        raise ValueError(f"Hard-negative labels are unknown: {sorted(unknown)}")
    if hard_negative_per_positive <= 0 or background_per_positive <= 0:
        raise ValueError("Hard-negative sampling counts must be positive")
    target = np.asarray(
        [int(label == CRITICAL_CLASS) for label in actual], dtype=np.int8
    )
    positives = np.flatnonzero(target == 1)
    negatives = np.flatnonzero(target == 0)
    if not len(positives) or not len(negatives):
        raise ValueError("Hard-negative OOF requires both binary classes")
    if min(np.bincount(target)) < n_splits:
        raise ValueError("Hard-negative OOF needs enough rows per class")
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    margins = np.full(len(values), np.nan, dtype=np.float64)
    for train_indices, test_indices in splitter.split(values, target):
        pipeline = Pipeline(
            [
                ("tfidf", build_vectorizer(WORD_TFIDF_ALIAS)),
                ("classifier", build_estimator(0.3, WEIGHTED_FULL)),
            ]
        )
        pipeline.fit([values[index] for index in train_indices], target[train_indices])
        margins[test_indices] = pipeline.decision_function(
            [values[index] for index in test_indices]
        )
    if not np.isfinite(margins).all():
        raise ValueError("OOF hard-negative margins are incomplete")
    positive_count = len(positives)
    hard_count = min(
        len(negatives), positive_count * hard_negative_per_positive
    )
    hard_order = sorted(
        (int(index) for index in negatives),
        key=lambda index: (-float(margins[index]), index),
    )
    hard = hard_order[:hard_count]
    hard_set = set(hard)
    remaining = [index for index in hard_order if index not in hard_set]
    background_count = min(len(remaining), positive_count * background_per_positive)
    background = sorted(
        remaining,
        key=lambda index: (
            hashlib.sha256(f"{random_state}:{index}".encode()).hexdigest(),
            index,
        ),
    )[:background_count]
    selected = sorted({*map(int, positives), *hard, *background})
    return tuple(selected)


def _fit_representation(
    representation: str,
    fit_scope: _Scope,
    calibration_scope: _Scope,
) -> _RepresentationCache:
    """Fit one reusable vectorizer and transform fit/calibration scopes."""

    vectorizer = build_vectorizer(representation)
    fit_matrix = vectorizer.fit_transform(fit_scope.texts)
    calibration_matrix = vectorizer.transform(calibration_scope.texts)
    return _RepresentationCache(vectorizer, fit_matrix, calibration_matrix)


def _fit_candidate(
    descriptor: Mapping[str, Any],
    cache: _RepresentationCache,
    fit_scope: _Scope,
    hard_indices: Sequence[int],
) -> Any:
    """Fit one candidate from reusable sparse matrices."""

    strategy = str(descriptor["balance_strategy"])
    target = np.asarray(
        [int(label == CRITICAL_CLASS) for label in fit_scope.labels], dtype=np.int8
    )
    matrix = cache.fit_matrix
    if strategy == HARD_NEGATIVE:
        matrix = matrix[list(hard_indices)]
        target = target[list(hard_indices)]
    estimator = build_estimator(float(descriptor["C"]), strategy)
    sample_weight = None
    if strategy == RANDOM_OVER:
        sampler = RandomOverSampler(
            sampling_strategy=float(descriptor["sampling_strategy"]),
            random_state=RANDOM_STATE,
        )
        row_indices = np.arange(len(target), dtype=np.int64).reshape(-1, 1)
        sampled_indices, _ = sampler.fit_resample(row_indices, target)
        sample_weight = np.bincount(
            sampled_indices.ravel(), minlength=len(target)
        ).astype(np.float64)
    estimator.fit(matrix, target, sample_weight=sample_weight)
    return estimator


def _outer_metrics(
    matrix: np.ndarray,
) -> dict[str, Any]:
    """Calculate fixed-order aggregate metrics from one confusion matrix."""

    from .v2_detector import _metrics_from_confusion

    return _metrics_from_confusion(matrix)


def _fallback_only_metrics(
    labels: Sequence[str], fallback_labels: Sequence[str]
) -> dict[str, Any]:
    """Score the pure S7 fallback alone, with no stage-A detector applied."""

    matrix = confusion_matrix(
        labels, fallback_labels, labels=list(MODELED_FAMILIES)
    ).astype(np.int64)
    return _outer_metrics(matrix)


def _evaluate_outer(
    representation_cache: _RepresentationCache,
    estimators: Mapping[str, Any],
    thresholds: Mapping[str, float],
    outer_scope: _Scope,
    fallback_labels: Sequence[str],
    batch_size: int,
) -> dict[str, dict[str, Any]]:
    """Evaluate every representation candidate on shared outer batches."""

    matrices = {
        name: np.zeros((len(MODELED_FAMILIES), len(MODELED_FAMILIES)), dtype=np.int64)
        for name in estimators
    }
    override_decisions = {name: 0 for name in estimators}
    effective_overrides = {name: 0 for name in estimators}
    class_order = list(MODELED_FAMILIES)
    for start in range(0, len(outer_scope.texts), batch_size):
        stop = min(start + batch_size, len(outer_scope.texts))
        transformed = representation_cache.vectorizer.transform(
            outer_scope.texts[start:stop]
        )
        actual = outer_scope.labels[start:stop]
        fallback = fallback_labels[start:stop]
        for name, estimator in estimators.items():
            margins = estimator.decision_function(transformed)
            decisions = np.asarray(margins) >= float(thresholds[name])
            predictions = combine_detector_with_fallback(decisions, fallback)
            matrices[name] += confusion_matrix(
                actual, predictions, labels=class_order
            ).astype(np.int64)
            batch_override, batch_effective = count_override_decisions(
                decisions, fallback
            )
            override_decisions[name] += batch_override
            effective_overrides[name] += batch_effective
    return {
        name: {
            "metrics": _outer_metrics(matrix),
            "override_decisions": override_decisions[name],
            "effective_overrides": effective_overrides[name],
        }
        for name, matrix in matrices.items()
    }


def _run_candidates(
    scopes: Mapping[str, _Scope],
    fallback: Mapping[str, Sequence[str]],
    protocol: V2Protocol,
    *,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Run the 30 candidates and return aggregate evidence and baselines."""

    fit_scope = scopes["inner_fit"]
    calibration_scope = scopes["inner_calibration"]
    outer_scope = scopes["outer_evaluation"]
    hard_indices = generate_hard_negative_indices(
        fit_scope.texts, fit_scope.labels
    )
    fallback_baseline = {
        "inner_calibration": _fallback_only_metrics(
            calibration_scope.labels, fallback["inner_calibration"]
        ),
        "outer_evaluation": _fallback_only_metrics(
            outer_scope.labels, fallback["outer_evaluation"]
        ),
    }
    fallback_outer_critical_f1 = float(
        fallback_baseline["outer_evaluation"]["critical_f1"]
    )
    descriptors = _candidate_catalog()
    results: list[dict[str, Any]] = []
    for representation in REPRESENTATIONS:
        cache = _fit_representation(representation, fit_scope, calibration_scope)
        estimators: dict[str, Any] = {}
        thresholds: dict[str, float] = {}
        calibration_by_name: dict[str, dict[str, Any]] = {}
        candidate_runtime: dict[str, float] = {}
        for descriptor in descriptors:
            if descriptor["representation"] != representation:
                continue
            name = str(descriptor["candidate_id"])
            candidate_started = time.perf_counter()
            estimator = _fit_candidate(descriptor, cache, fit_scope, hard_indices)
            calibration_margins = estimator.decision_function(
                cache.calibration_matrix
            )
            threshold_result = search_detector_threshold_exact(
                calibration_scope.labels,
                calibration_margins,
                fallback["inner_calibration"],
                protocol,
            )
            selected = threshold_result["selected"]
            estimators[name] = estimator
            thresholds[name] = float(selected["threshold"])
            calibration_by_name[name] = {
                "threshold": float(selected["threshold"]),
                "threshold_count": int(threshold_result["threshold_count"]),
                "metrics": selected["metrics"],
                "gates": selected["gates"],
                "override_decisions": int(selected["override_decisions"]),
                "effective_overrides": int(selected["effective_overrides"]),
            }
            candidate_runtime[name] = float(
                time.perf_counter() - candidate_started
            )
        outer_started = time.perf_counter()
        outer_by_name = _evaluate_outer(
            cache,
            estimators,
            thresholds,
            outer_scope,
            fallback["outer_evaluation"],
            batch_size,
        )
        shared_outer_seconds = float(time.perf_counter() - outer_started)
        allocated_outer_seconds = shared_outer_seconds / len(estimators)
        for descriptor in descriptors:
            if descriptor["representation"] != representation:
                continue
            name = str(descriptor["candidate_id"])
            outer_entry = outer_by_name[name]
            metrics = outer_entry["metrics"]
            gates = calculate_scientific_gates(metrics, protocol)
            safety = calculate_safety_margins(metrics, protocol)
            critical_f1_vs_fallback = (
                float(metrics["critical_f1"]) - fallback_outer_critical_f1
            )
            results.append(
                {
                    "candidate_id": name,
                    "parameters": {
                        key: value
                        for key, value in descriptor.items()
                        if key != "candidate_id"
                    },
                    "calibration": calibration_by_name[name],
                    "outer": {
                        "metrics": metrics,
                        "gates": gates,
                        "safety": safety,
                        "override_decisions": int(
                            outer_entry["override_decisions"]
                        ),
                        "effective_overrides": int(
                            outer_entry["effective_overrides"]
                        ),
                        "critical_f1_vs_fallback": critical_f1_vs_fallback,
                    },
                    "runtime_seconds": float(
                        candidate_runtime[name] + allocated_outer_seconds
                    ),
                    "runtime_components": {
                        "fit_calibration_seconds": candidate_runtime[name],
                        "allocated_outer_seconds": allocated_outer_seconds,
                    },
                }
            )
        del cache, estimators, thresholds, calibration_by_name, outer_by_name
    results.sort(key=lambda item: item["candidate_id"])
    selection = _select_candidate(results)
    positive_count = sum(label == CRITICAL_CLASS for label in fit_scope.labels)
    evidence = {
        "positive_groups": int(positive_count),
        "hard_negative_groups": int(len(hard_indices) - positive_count),
        "selected_candidate": selection["selected_candidate"],
        "margin_eligible_count": selection["margin_eligible_count"],
        "effective_eligible_count": selection["effective_eligible_count"],
        "fallback_beating_eligible_count": (
            selection["fallback_beating_eligible_count"]
        ),
    }
    extra = {
        "fallback_baseline": fallback_baseline,
        "degenerate_null": selection["degenerate_null"],
        "selection_blocked_reason": selection["selection_blocked_reason"],
    }
    return results, evidence, extra


def _select_candidate(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select only among candidates that clear every development-side rule.

    Three filters apply in order: the candidate must pass all three safety
    margins; it must have at least one effective outer override, because a
    candidate with zero effective overrides is byte-identical to the frozen
    S7 fallback rather than a V2 model; and its outer critical F1 must
    strictly exceed the pure-fallback baseline, because passing the frozen
    absolute margins stops being informative once the fallback alone clears
    them on its own.

    Args:
        results: Every candidate result, each already carrying
            ``outer.effective_overrides`` and ``outer.critical_f1_vs_fallback``.

    Returns:
        A mapping with the selected candidate id (or ``None``), the
        per-stage funnel counts, whether every candidate is a degenerate
        fallback clone, and a machine-readable block reason.
    """

    margin_eligible = [
        item
        for item in results
        if item["outer"]["safety"].get("passed") is True
        and item["outer"]["safety"].get("gate_count") == 3
    ]
    effective_eligible = [
        item
        for item in margin_eligible
        if int(item["outer"]["effective_overrides"]) > 0
    ]
    fallback_beating_eligible = [
        item
        for item in effective_eligible
        if float(item["outer"]["critical_f1_vs_fallback"]) > 0.0
    ]
    ranked = sorted(
        fallback_beating_eligible,
        key=lambda item: (
            item["outer"]["metrics"]["critical_f1"],
            item["outer"]["metrics"]["macro_f1"],
            item["outer"]["metrics"]["critical_precision"],
            -item["runtime_seconds"],
        ),
        reverse=True,
    )
    selected = ranked[0]["candidate_id"] if ranked else None
    degenerate_null = all(
        int(item["outer"]["effective_overrides"]) == 0 for item in results
    )
    return {
        "selected_candidate": selected,
        "margin_eligible_count": len(margin_eligible),
        "effective_eligible_count": len(effective_eligible),
        "fallback_beating_eligible_count": len(fallback_beating_eligible),
        "degenerate_null": degenerate_null,
        "selection_blocked_reason": _selection_blocked_reason(
            selected, margin_eligible, effective_eligible, fallback_beating_eligible
        ),
    }


def _selection_blocked_reason(
    selected: str | None,
    margin_eligible: Sequence[Mapping[str, Any]],
    effective_eligible: Sequence[Mapping[str, Any]],
    fallback_beating_eligible: Sequence[Mapping[str, Any]],
) -> str | None:
    """Explain, in one machine-readable token, why no candidate was chosen.

    Args:
        selected: The chosen candidate id, or ``None`` if selection failed.
        margin_eligible: Candidates passing all three safety margins.
        effective_eligible: Of those, candidates with a real outer override.
        fallback_beating_eligible: Of those, candidates beating the fallback.

    Returns:
        ``None`` when a candidate was selected, otherwise the reason the
        eligible pool emptied out. Reasons are checked from the weakest
        rule to the strongest, so the strongest applicable reason wins when
        more than one would independently explain an empty pool.

    Raises:
        AssertionError: If an eligible candidate exists but none was chosen.
    """

    if selected is not None:
        return None
    if not margin_eligible:
        return "no_candidate_passed_safety_margins"
    if not effective_eligible:
        return "all_candidates_zero_effective_overrides"
    if not fallback_beating_eligible:
        return "no_candidate_beat_the_fallback_baseline"
    raise AssertionError("An eligible candidate exists but none was selected")


def _base_result(
    signature: str,
    protocol: V2Protocol,
    *,
    diagnostic_only: bool,
    opened_at: str | None = None,
    attempts: int = 1,
) -> dict[str, Any]:
    """Create an aggregate-only result envelope."""

    return {
        "schema_version": RESULT_SCHEMA,
        "code_schema": CODE_SCHEMA,
        "signature": signature,
        "complete": False,
        "status": "DIAGNOSTIC_ONLY" if diagnostic_only else "RUNNING",
        "diagnostic_only": diagnostic_only,
        "input_language": "en-US",
        "critical_class": CRITICAL_CLASS,
        "allowed_partitions": list(protocol.allowed_partitions),
        "sealed_partitions": list(protocol.forbidden_partitions),
        "sealed_access": {name: False for name in protocol.forbidden_partitions},
        "attempts": attempts,
        "opened_at": opened_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "candidate_catalog_count": 30,
        "candidates": None,
        "hard_negative": None,
        "fallback_baseline": None,
        "degenerate_null": None,
        "selection_blocked_reason": None,
        "selected": None,
        "runtime_seconds": None,
    }


def _complete_result(
    base: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Any],
    extra: Mapping[str, Any],
    runtime_seconds: float,
) -> dict[str, Any]:
    """Finalize one aggregate result and enforce its privacy boundary."""

    result = dict(base)
    result.update(
        {
            "complete": True,
            "status": "DIAGNOSTIC_ONLY" if result["diagnostic_only"] else "COMPLETE",
            "candidates": [dict(item) for item in candidates],
            "hard_negative": dict(evidence),
            "selected": evidence["selected_candidate"],
            "fallback_baseline": dict(extra["fallback_baseline"]),
            "degenerate_null": bool(extra["degenerate_null"]),
            "selection_blocked_reason": extra["selection_blocked_reason"],
            "runtime_seconds": float(runtime_seconds),
        }
    )
    _validate_result_privacy(result)
    return result


def _resume_marker(path: Path, signature: str, protocol: V2Protocol) -> dict[str, Any]:
    """Create or resume the single aggregate execution marker."""

    if not path.exists():
        result = _base_result(signature, protocol, diagnostic_only=False)
        _write_json_atomic(path, result)
        return result
    previous = _read_json(path)
    if previous.get("complete") is True:
        if previous.get("signature") != signature:
            raise ValueError("Complete V2 result has a stale signature")
        if previous.get("code_schema") != CODE_SCHEMA:
            raise ValueError("Complete V2 result has a stale code schema")
        return previous
    if previous.get("signature") != signature:
        raise ValueError("Incomplete V2 result has a stale signature")
    if previous.get("code_schema") != CODE_SCHEMA:
        raise ValueError("Incomplete V2 result has a stale code schema")
    if previous.get("candidates") is not None:
        raise ValueError("Partial V2 candidates must not be persisted")
    result = _base_result(
        signature,
        protocol,
        diagnostic_only=False,
        opened_at=str(previous["opened_at"]),
        attempts=int(previous.get("attempts", 1)) + 1,
    )
    _write_json_atomic(path, result)
    return result


def _s7_paths(root: Path) -> tuple[Path, Path, Path]:
    """Return the frozen S7 bundle, public manifest, and result paths."""

    return (
        root / "artifacts" / "s7" / "consumer_complaint_classifier_s7.joblib",
        root / "config" / "s7_results.json",
        root / "temp" / "s7" / "s7_results.json",
    )


def _s7_signatures(root: Path) -> dict[str, dict[str, Any]]:
    """Hash the portable S7 serving inputs used by the fallback."""

    bundle, manifest, result = _s7_paths(root)
    return {
        "bundle": _signature(bundle),
        "manifest": _signature(manifest),
        "result": _signature(result),
    }


def _publish_manifest(
    root: Path,
    protocol_path: Path,
    cache_path: Path,
    artifact_path: Path,
    manifest_path: Path,
    s7_signatures: Mapping[str, Mapping[str, Any]],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish a portable manifest after a complete full result."""

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "stage": "V2-D1",
        "status": result["status"],
        "complete": True,
        "diagnostic_only": False,
        "signature": result["signature"],
        "protocol": {
            "path": _relative(protocol_path, root),
            **_signature(protocol_path),
        },
        "cache": {"path": _relative(cache_path, root), **_signature(cache_path)},
        "s7": {
            key: {"path": _relative(path, root), **dict(signature)}
            for key, path, signature in (
                ("bundle", _s7_paths(root)[0], s7_signatures["bundle"]),
                ("manifest", _s7_paths(root)[1], s7_signatures["manifest"]),
                ("result", _s7_paths(root)[2], s7_signatures["result"]),
            )
        },
        "artifact": {
            "path": _relative(artifact_path, root),
            **_signature(artifact_path),
        },
        "sealed_access": {name: False for name in ("test", "stress", "monitor")},
        "selected": result["selected"],
        "degenerate_null": result["degenerate_null"],
        "selection_blocked_reason": result["selection_blocked_reason"],
        "candidate_count": len(result["candidates"]),
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest


def validate_v2_manifest(
    manifest_path: str | Path,
    result_path: str | Path | None = None,
    protocol_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Validate a complete portable V2 result manifest and artifact.

    Args:
        manifest_path: Manifest JSON path.
        result_path: Optional explicit aggregate result path.
        protocol_path: Frozen V2 protocol path.

    Returns:
        The validated manifest payload.

    Raises:
        ValueError: If a manifest, hash, signature, or privacy invariant fails.
    """

    manifest_file = Path(manifest_path).expanduser().resolve()
    protocol_file = Path(protocol_path).expanduser().resolve()
    root = protocol_file.parent.parent.resolve()
    manifest = _read_json(manifest_file)
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("Unexpected V2 manifest schema")
    if (
        manifest.get("complete") is not True
        or manifest.get("diagnostic_only") is not False
    ):
        raise ValueError("V2 manifest is not a complete full result")
    if any(manifest.get("sealed_access", {}).values()):
        raise ValueError("V2 manifest claims sealed-partition access")
    for role in ("protocol", "cache", "artifact"):
        item = manifest.get(role, {})
        path = _safe_path(root, item.get("path"))
        if (
            item.get("sha256") != _sha256(path)
            or item.get("size_bytes") != path.stat().st_size
        ):
            raise ValueError(f"V2 manifest {role} hash is invalid")
    s7_paths = _s7_paths(root)
    for role, path in zip(("bundle", "manifest", "result"), s7_paths):
        item = manifest.get("s7", {}).get(role, {})
        if _safe_path(root, item.get("path")) != path.resolve():
            raise ValueError(f"V2 manifest S7 {role} path is invalid")
        if (
            item.get("sha256") != _sha256(path)
            or item.get("size_bytes") != path.stat().st_size
        ):
            raise ValueError(f"V2 manifest S7 {role} hash is invalid")
    artifact = _safe_path(root, manifest["artifact"].get("path"))
    if result_path is not None and artifact != Path(result_path).expanduser().resolve():
        raise ValueError("V2 manifest artifact path differs from requested result")
    result = _read_json(artifact)
    if (
        result.get("schema_version") != RESULT_SCHEMA
        or result.get("complete") is not True
    ):
        raise ValueError("V2 aggregate result is incomplete or stale")
    if result.get("signature") != manifest.get("signature"):
        raise ValueError("V2 result signature differs from manifest")
    if len(result.get("candidates", ())) != 30:
        raise ValueError("V2 result does not contain exactly 30 candidates")
    _validate_result_privacy(result)
    expected_s7 = {
        role: {
            "sha256": manifest["s7"][role]["sha256"],
            "size_bytes": manifest["s7"][role]["size_bytes"],
        }
        for role in ("bundle", "manifest", "result")
    }
    expected = _result_signature(
        protocol_file,
        _signature(_safe_path(root, manifest["cache"].get("path"))),
        expected_s7,
    )
    if result["signature"] != expected:
        raise ValueError("V2 result signature cannot be reproduced")
    _validate_complete_result(result, expected)
    if manifest.get("selected") != result.get("selected"):
        raise ValueError("V2 manifest selected candidate differs from result")
    if manifest.get("degenerate_null") != result.get("degenerate_null"):
        raise ValueError("V2 manifest degenerate-null flag differs from result")
    if manifest.get("selection_blocked_reason") != result.get(
        "selection_blocked_reason"
    ):
        raise ValueError("V2 manifest selection-blocked reason differs from result")
    return manifest


def _validate_complete_result(
    result: Mapping[str, Any], signature: str
) -> None:
    """Validate complete aggregate evidence before accepting a cache hit."""

    if result.get("code_schema") != CODE_SCHEMA:
        raise ValueError("V2 result code schema is stale")
    if result.get("signature") != signature:
        raise ValueError("V2 result signature is stale")
    if result.get("status") != "COMPLETE":
        raise ValueError("V2 full result status is invalid")
    if result.get("diagnostic_only") is not False:
        raise ValueError("V2 full result cannot be diagnostic-only")
    if len(result.get("candidates", ())) != 30:
        raise ValueError("V2 full result must contain 30 candidates")
    expected_ids = {item["candidate_id"] for item in _candidate_catalog()}
    actual_ids = {item.get("candidate_id") for item in result["candidates"]}
    if actual_ids != expected_ids:
        raise ValueError("V2 result candidate catalog differs from protocol")
    selection = _select_candidate(result["candidates"])
    selected = result.get("selected")
    if selected != selection["selected_candidate"]:
        raise ValueError("V2 selected candidate is inconsistent with ranking")
    evidence = result.get("hard_negative", {})
    if evidence.get("margin_eligible_count") != selection["margin_eligible_count"]:
        raise ValueError("V2 eligible candidate count is inconsistent")
    if (
        evidence.get("effective_eligible_count")
        != selection["effective_eligible_count"]
    ):
        raise ValueError("V2 effective-override eligible count is inconsistent")
    if (
        evidence.get("fallback_beating_eligible_count")
        != selection["fallback_beating_eligible_count"]
    ):
        raise ValueError("V2 fallback-beating eligible count is inconsistent")
    if evidence.get("selected_candidate") != selection["selected_candidate"]:
        raise ValueError("V2 hard-negative evidence has a stale selection")
    if result.get("degenerate_null") != selection["degenerate_null"]:
        raise ValueError("V2 degenerate-null flag is inconsistent")
    if (
        result.get("selection_blocked_reason")
        != selection["selection_blocked_reason"]
    ):
        raise ValueError("V2 selection-blocked reason is inconsistent")
    if selection["degenerate_null"] and selected is not None:
        raise ValueError("V2 result selected a candidate despite a null result")
    _validate_fallback_baseline(result)
    if result.get("sealed_access") != {
        "test": False,
        "stress": False,
        "monitor": False,
    }:
        raise ValueError("V2 result sealed boundary is invalid")


def _validate_fallback_baseline(result: Mapping[str, Any]) -> None:
    """Validate the fallback-only baseline block and its published deltas."""

    baseline = result.get("fallback_baseline")
    if not isinstance(baseline, Mapping) or set(baseline) != {
        "inner_calibration",
        "outer_evaluation",
    }:
        raise ValueError("V2 result fallback baseline is malformed")
    outer_baseline = baseline["outer_evaluation"]
    if not isinstance(outer_baseline, Mapping) or "critical_f1" not in outer_baseline:
        raise ValueError("V2 result outer fallback baseline is malformed")
    outer_fallback_f1 = float(outer_baseline["critical_f1"])
    for item in result.get("candidates", ()):
        outer = item.get("outer", {})
        expected_delta = float(outer["metrics"]["critical_f1"]) - outer_fallback_f1
        if outer.get("critical_f1_vs_fallback") != expected_delta:
            raise ValueError("V2 candidate fallback delta is inconsistent")


def _cached_full(
    artifact_path: Path,
    manifest_path: Path,
    protocol_path: Path,
    cache_path: Path,
    signature: str,
) -> dict[str, Any] | None:
    """Return a valid cache hit and repair only a missing/tampered manifest."""

    if not artifact_path.exists():
        return None
    result = _read_json(artifact_path)
    if result.get("complete") is not True or result.get("signature") != signature:
        return None
    _validate_result_privacy(result)
    _validate_complete_result(result, signature)
    try:
        validate_v2_manifest(manifest_path, artifact_path, protocol_path)
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        root = protocol_path.parent.parent.resolve()
        _publish_manifest(
            root,
            protocol_path,
            cache_path,
            artifact_path,
            manifest_path,
            _s7_signatures(root),
            result,
        )
    return result


def _synthetic_scopes() -> tuple[dict[str, _Scope], dict[str, tuple[str, ...]]]:
    """Build small deterministic smoke data without reading project files."""

    scopes: dict[str, _Scope] = {}
    for role, repeats in (
        ("inner_fit", 30),
        ("inner_calibration", 20),
        ("outer_evaluation", 20),
    ):
        texts = []
        labels = []
        groups = []
        for label_index, label in enumerate(MODELED_FAMILIES):
            label_repeats = repeats if label != CRITICAL_CLASS else 3
            for repeat in range(label_repeats):
                texts.append(f"{label} complaint marker {label_index} {repeat}")
                labels.append(label)
                groups.append((f"{role}-{label_index}-{repeat}", 1))
        scopes[role] = _Scope(tuple(texts), tuple(labels), tuple(groups))
    fallback = {role: scope.labels for role, scope in scopes.items()}
    return scopes, fallback


def run_v2_benchmark(
    mode: str = "disabled",
    *,
    project_root: str | Path | None = None,
    protocol_path: str | Path = DEFAULT_CONFIG,
    cache_path: str | Path = DEFAULT_CACHE,
    artifact_path: str | Path = DEFAULT_ARTIFACT,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Run the disabled, synthetic smoke, or real V2 classical benchmark.

    Args:
        mode: ``disabled``, ``smoke``, or ``full``.
        project_root: Project root for relative paths.
        protocol_path: Frozen V2 protocol JSON path.
        cache_path: Development-only S3 scientific parquet path.
        artifact_path: Aggregate restartable result path.
        manifest_path: Complete-run public manifest path.
        batch_size: Bounded fallback and outer evaluation batch size.

    Returns:
        Aggregate-only benchmark result.

    Raises:
        ValueError: If the mode, boundary, hashes, or data contract is invalid.
    """

    if mode not in {"disabled", "smoke", "full"}:
        raise ValueError("mode must be disabled, smoke, or full")
    if mode == "disabled":
        return {"status": "DISABLED", "complete": False, "diagnostic_only": True}
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    root = Path(
        project_root or Path(protocol_path).expanduser().resolve().parent.parent
    )
    root = root.expanduser().resolve()
    protocol_file = (
        _safe_path(root, str(protocol_path))
        if not Path(protocol_path).is_absolute()
        else Path(protocol_path).resolve()
    )
    protocol = load_v2_protocol(protocol_file)
    if mode == "smoke":
        scopes, fallback = _synthetic_scopes()
        started = time.perf_counter()
        candidates, evidence, extra = _run_candidates(
            scopes, fallback, protocol, batch_size=batch_size
        )
        return _complete_result(
            _base_result("smoke", protocol, diagnostic_only=True),
            candidates,
            evidence,
            extra,
            time.perf_counter() - started,
        )
    cache_file = (
        _safe_path(root, str(cache_path))
        if not Path(cache_path).is_absolute()
        else Path(cache_path).resolve()
    )
    artifact_file = (
        _safe_path(root, str(artifact_path))
        if not Path(artifact_path).is_absolute()
        else Path(artifact_path).resolve()
    )
    manifest_file = (
        _safe_path(root, str(manifest_path))
        if not Path(manifest_path).is_absolute()
        else Path(manifest_path).resolve()
    )
    expected_cache = _safe_path(root, DEFAULT_CACHE)
    if cache_file != expected_cache:
        raise ValueError("V2 full mode accepts only temp/s3/scientific.parquet")
    if not cache_file.is_file():
        raise FileNotFoundError(cache_file)
    baseline = validate_baseline_artifacts(protocol, root)
    s7_signatures = _s7_signatures(root)
    signature = _result_signature(protocol_file, _signature(cache_file), s7_signatures)
    cached = _cached_full(
        artifact_file, manifest_file, protocol_file, cache_file, signature
    )
    if cached is not None:
        return cached
    marker = _resume_marker(artifact_file, signature, protocol)
    if marker.get("complete") is True:
        return marker
    bundle, s7_manifest, s7_result = _s7_paths(root)
    predictor = load_s7_predictor(bundle, s7_manifest, s7_result)
    scopes = _read_development_cache(cache_file, protocol, batch_size)
    fallback = {
        role: _fallback_labels(predictor, scopes[role], batch_size)
        for role in ("inner_calibration", "outer_evaluation")
    }
    started = time.perf_counter()
    candidates, evidence, extra = _run_candidates(
        scopes, fallback, protocol, batch_size=batch_size
    )
    result = _complete_result(
        marker,
        candidates,
        evidence,
        extra,
        time.perf_counter() - started,
    )
    _write_json_atomic(artifact_file, result)
    _publish_manifest(
        root,
        protocol_file,
        cache_file,
        artifact_file,
        manifest_file,
        s7_signatures,
        result,
    )
    validate_v2_manifest(manifest_file, artifact_file, protocol_file)
    return result


def run_v2_benchmark_smoke() -> dict[str, Any]:
    """Run the synthetic diagnostic mode without real project files."""

    return run_v2_benchmark("smoke")


__all__ = [
    "CODE_SCHEMA",
    "DEFAULT_ARTIFACT",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CACHE",
    "DEFAULT_MANIFEST",
    "MANIFEST_SCHEMA",
    "RESULT_SCHEMA",
    "candidate_catalog",
    "generate_hard_negative_indices",
    "run_v2_benchmark",
    "run_v2_benchmark_smoke",
    "validate_v2_manifest",
]
