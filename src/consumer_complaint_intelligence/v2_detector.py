"""Framework-neutral V2 binary detector for the critical complaint class."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from .s6 import CRITICAL_CLASS, MODELED_FAMILIES
from .v2_protocol import (
    V2Protocol,
    calculate_scientific_gates,
    load_v2_protocol,
    make_binary_critical_target,
    require_development_partition,
)


WORD_TFIDF_ALIAS = "word_tfidf_1_2_40000"
WORD_CHAR_TFIDF_ALIAS = "word_char_tfidf_union_40000_60000"
REPRESENTATION_ALIASES = (WORD_TFIDF_ALIAS, WORD_CHAR_TFIDF_ALIAS)
WEIGHTED_FULL = "weighted_full"
RANDOM_OVER = "random_over"
HARD_NEGATIVE = "hard_negative"
BALANCE_STRATEGIES = (WEIGHTED_FULL, RANDOM_OVER, HARD_NEGATIVE)
SAMPLING_STRATEGIES = (0.05, 0.1, 0.2)
RANDOM_STATE = 42
INPUT_LANGUAGE = "en-US"


def _as_texts(texts: Sequence[str], name: str) -> tuple[str, ...]:
    """Validate and normalize a non-empty sequence of English narratives."""

    if isinstance(texts, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of texts")
    values = tuple(texts)
    if not values or not all(isinstance(value, str) for value in values):
        raise ValueError(f"{name} must contain non-empty strings")
    if not all(value.strip() for value in values):
        raise ValueError(f"{name} contains an empty narrative")
    return values


def _as_labels(labels: Sequence[str], name: str) -> tuple[str, ...]:
    """Validate and normalize labels in the frozen nine-family order."""

    if isinstance(labels, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of labels")
    values = tuple(labels)
    if not values or not all(isinstance(value, str) for value in values):
        raise ValueError(f"{name} must contain non-empty strings")
    unknown = set(values) - set(MODELED_FAMILIES)
    if unknown:
        raise ValueError(f"{name} contains unknown modeled classes: {sorted(unknown)}")
    return values


def _validate_language(input_language: str) -> str:
    """Require the frozen English input-language contract."""

    if input_language != INPUT_LANGUAGE:
        raise ValueError("V2 detector accepts input_language=en-US only")
    return input_language


_DEVELOPMENT_ROLE_BY_DETECTOR_ROLE = {
    "fit": "inner_fit",
    "inner_calibration": "inner_calibration",
    "outer": "outer_evaluation",
}


def role_partition_map(protocol: V2Protocol | None = None) -> dict[str, str]:
    """Derive each detector role's required partition from the protocol.

    The map is derived from ``development_windows`` on every call instead of
    being restated as a second, hand-maintained literal, so it cannot drift
    from the frozen contract in ``config/v2_development_protocol.json``.

    Args:
        protocol: Validated V2 protocol, or the frozen default when omitted.

    Returns:
        A mapping from detector role (``fit``, ``inner_calibration``,
        ``outer``) to its required development partition.

    Raises:
        ValueError: If the protocol is missing an expected development
            window.
    """

    validated = protocol if protocol is not None else load_v2_protocol()
    windows = validated.payload["development_windows"]
    try:
        return {
            role: str(windows[window_role]["partition"])
            for role, window_role in _DEVELOPMENT_ROLE_BY_DETECTOR_ROLE.items()
        }
    except KeyError as error:
        raise ValueError(
            "V2 protocol is missing an expected development window"
        ) from error


def _validate_partition(partition: str, role: str) -> str:
    """Validate one explicit development partition and its experiment role."""

    require_development_partition(partition)
    expected = role_partition_map()
    if role not in expected:
        raise ValueError(f"Unsupported V2 detector partition role: {role}")
    if partition != expected[role]:
        raise ValueError(
            f"{role} requires partition {expected[role]!r}, got {partition!r}"
        )
    return partition


def _validate_aligned(
    texts: Sequence[str],
    labels: Sequence[str],
    partition: str,
    role: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Validate one explicitly partitioned text/label array pair."""

    _validate_partition(partition, role)
    normalized_texts = _as_texts(texts, f"{role}_texts")
    normalized_labels = _as_labels(labels, f"{role}_labels")
    if len(normalized_texts) != len(normalized_labels):
        raise ValueError(f"{role} texts and labels must have equal length")
    return normalized_texts, normalized_labels


def _validate_fallback(
    fallback_labels: Sequence[str],
    expected_length: int,
    name: str,
) -> tuple[str, ...]:
    """Validate aggregate-only fallback labels supplied by frozen S7."""

    values = _as_labels(fallback_labels, name)
    if len(values) != expected_length:
        raise ValueError(f"{name} must align with the corresponding labels")
    return values


def build_vectorizer(alias: str) -> TfidfVectorizer | FeatureUnion:
    """Build one contract-defined sparse text representation.

    Args:
        alias: Frozen representation alias from the V2 protocol.

    Returns:
        An unfitted word TF-IDF vectorizer or word-plus-character union.

    Raises:
        ValueError: If ``alias`` is not defined by the protocol.
    """

    common = {
        "min_df": 2,
        "max_df": 0.98,
        "sublinear_tf": True,
        "dtype": np.float32,
    }
    if alias == WORD_TFIDF_ALIAS:
        return TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=40000,
            **common,
        )
    if alias == WORD_CHAR_TFIDF_ALIAS:
        word = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=40000,
            **common,
        )
        char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=60000,
            **common,
        )
        return FeatureUnion([("word", word), ("char", char)])
    raise ValueError(f"Unsupported V2 representation alias: {alias}")


def build_estimator(
    c_value: float,
    balance_strategy: str,
) -> LinearSVC:
    """Build the contract-defined binary LinearSVC estimator.

    Args:
        c_value: Regularization value from the V2 candidate grid.
        balance_strategy: Frozen weighted, oversampled, or hard-negative mode.

    Returns:
        An unfitted binary LinearSVC.

    Raises:
        ValueError: If a candidate parameter is outside the frozen contract.
    """

    if float(c_value) not in (0.1, 0.3, 1.0):
        raise ValueError("V2 LinearSVC C must be one of 0.1, 0.3, or 1.0")
    if balance_strategy not in BALANCE_STRATEGIES:
        raise ValueError(f"Unsupported V2 balance strategy: {balance_strategy}")
    return LinearSVC(
        C=float(c_value),
        class_weight="balanced" if balance_strategy == WEIGHTED_FULL else None,
        tol=0.0001,
        max_iter=5000,
        dual="auto",
        random_state=RANDOM_STATE,
    )


def _validate_sampling_strategy(value: float) -> float:
    """Validate one frozen random-oversampling ratio."""

    value = float(value)
    if value not in SAMPLING_STRATEGIES:
        raise ValueError("sampling_strategy must be 0.05, 0.1, or 0.2")
    return value


@dataclass(slots=True)
class V2CriticalDetector:
    """Fit and serve a binary critical-class detector without data access."""

    representation_alias: str
    balance_strategy: str
    c_value: float
    sampling_strategy: float | None = None
    random_state: int = RANDOM_STATE
    vectorizer: TfidfVectorizer | FeatureUnion | None = None
    estimator: LinearSVC | None = None
    fit_rows_before: int = 0
    fit_rows_after: int = 0
    fit_runtime_seconds: float = 0.0

    def __post_init__(self) -> None:
        """Validate immutable candidate parameters before fitting."""

        if self.representation_alias not in REPRESENTATION_ALIASES:
            raise ValueError("Unsupported V2 representation alias")
        if self.balance_strategy not in BALANCE_STRATEGIES:
            raise ValueError("Unsupported V2 balance strategy")
        if self.random_state != RANDOM_STATE:
            raise ValueError("V2 random_state must be 42")
        if self.balance_strategy == RANDOM_OVER:
            if self.sampling_strategy is None:
                raise ValueError("random_over requires sampling_strategy")
            self.sampling_strategy = _validate_sampling_strategy(
                self.sampling_strategy
            )
        elif self.sampling_strategy is not None:
            raise ValueError(
                "Only random_over candidates can define sampling_strategy"
            )
        build_estimator(self.c_value, self.balance_strategy)

    @property
    def fitted(self) -> bool:
        """Return whether vectorizer and estimator have both been fitted."""

        return self.vectorizer is not None and self.estimator is not None

    def fit(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
        *,
        partition: str,
        input_language: str = INPUT_LANGUAGE,
    ) -> "V2CriticalDetector":
        """Fit the detector on one explicitly validated development partition.

        Args:
            texts: English narratives already loaded by the caller.
            labels: Nine-family labels aligned with ``texts``.
            partition: Must be ``train`` for the inner-fit role.

        Returns:
            This fitted detector.

        Raises:
            ValueError: If the arrays or partition are invalid.
        """

        _validate_language(input_language)
        normalized_texts, normalized_labels = _validate_aligned(
            texts, labels, partition, "fit"
        )
        target = np.asarray(
            make_binary_critical_target(normalized_labels), dtype=np.int8
        )
        if np.unique(target).size != 2:
            raise ValueError("Detector fit requires critical and non-critical labels")
        started = time.perf_counter()
        vectorizer = build_vectorizer(self.representation_alias)
        matrix = vectorizer.fit_transform(normalized_texts)
        estimator = build_estimator(self.c_value, self.balance_strategy)
        sampler = None
        fit_rows_after = int(matrix.shape[0])
        sample_weight = None
        if self.balance_strategy == RANDOM_OVER:
            from imblearn.over_sampling import RandomOverSampler

            sampler = RandomOverSampler(
                sampling_strategy=float(self.sampling_strategy),
                random_state=self.random_state,
            )
            row_indices = np.arange(len(target), dtype=np.int64).reshape(-1, 1)
            sampled_indices, _ = sampler.fit_resample(row_indices, target)
            sample_weight = np.bincount(
                sampled_indices.ravel(), minlength=len(target)
            ).astype(np.float64)
            fit_rows_after = int(len(sampled_indices))
        estimator.fit(matrix, target, sample_weight=sample_weight)
        self.vectorizer = vectorizer
        self.estimator = estimator
        self.fit_rows_before = int(matrix.shape[0])
        self.fit_rows_after = fit_rows_after
        self.fit_runtime_seconds = float(time.perf_counter() - started)
        return self

    def _decision_scores(self, texts: Sequence[str]) -> np.ndarray:
        """Transform texts and return internal LinearSVC decision margins."""

        if not self.fitted:
            raise RuntimeError("Detector must be fitted before evaluation")
        values = _as_texts(texts, "texts")
        transformed = self.vectorizer.transform(values)
        scores = np.asarray(self.estimator.decision_function(transformed))
        if scores.ndim != 1 or len(scores) != len(values):
            raise ValueError("Binary detector must return one margin per text")
        if not np.isfinite(scores).all():
            raise ValueError("Detector margins must be finite")
        return scores.astype(np.float64, copy=False)

    def predict(
        self,
        texts: Sequence[str],
        threshold: float,
        *,
        input_language: str = INPUT_LANGUAGE,
    ) -> tuple[bool, ...]:
        """Predict binary stage-A decisions using a margin threshold.

        Args:
            texts: English narratives already loaded by the caller.
            threshold: Decision-margin threshold, not a probability.

        Returns:
            Boolean critical decisions in the input order.
        """

        _validate_language(input_language)
        if not np.isfinite(float(threshold)):
            raise ValueError("threshold must be finite")
        scores = self._decision_scores(texts)
        return tuple(bool(score >= threshold) for score in scores)

    def decision_function(
        self,
        texts: Sequence[str],
        *,
        input_language: str = INPUT_LANGUAGE,
    ) -> np.ndarray:
        """Return LinearSVC margins for framework interoperability."""

        _validate_language(input_language)
        return self._decision_scores(texts)

    def parameters(self) -> dict[str, Any]:
        """Return candidate parameters without fitted data or individual scores."""

        return {
            "representation_alias": self.representation_alias,
            "balance_strategy": self.balance_strategy,
            "C": float(self.c_value),
            "sampling_strategy": self.sampling_strategy,
            "fit_rows_before": self.fit_rows_before,
            "fit_rows_after": self.fit_rows_after,
            "estimator": {
                "class": "LinearSVC",
                "tol": 0.0001,
                "max_iter": 5000,
                "dual": "auto",
                "random_state": self.random_state,
                "class_weight": (
                    "balanced" if self.balance_strategy == WEIGHTED_FULL else None
                ),
            },
        }


def combine_detector_with_fallback(
    detector_decisions: Sequence[bool],
    fallback_labels: Sequence[str],
) -> tuple[str, ...]:
    """Apply stage A over the frozen S7 fallback without losing critical labels."""

    decisions = tuple(detector_decisions)
    fallbacks = _as_labels(fallback_labels, "fallback_labels")
    if len(decisions) != len(fallbacks) or not decisions:
        raise ValueError("detector decisions and fallback labels must align")
    if not all(isinstance(value, (bool, np.bool_)) for value in decisions):
        raise ValueError("detector decisions must be boolean")
    return tuple(
        CRITICAL_CLASS if bool(decision) else fallback
        for decision, fallback in zip(decisions, fallbacks)
    )


def count_override_decisions(
    decisions: Sequence[bool] | np.ndarray,
    fallback_labels: Sequence[str],
) -> tuple[int, int]:
    """Count raw and effective stage-A override decisions.

    A raw override decision is any row where the margin cleared the
    threshold. An effective override is the subset of those rows whose
    frozen fallback label was not already the critical class -- the only
    rows where the combined hierarchical output actually differs from the
    fallback alone.

    Args:
        decisions: Boolean stage-A decisions, one per row.
        fallback_labels: Frozen S7 fallback labels aligned with ``decisions``.

    Returns:
        A ``(override_decisions, effective_overrides)`` count pair.

    Raises:
        ValueError: If ``decisions`` and ``fallback_labels`` are misaligned.
    """

    decision_array = np.asarray(decisions, dtype=bool)
    fallback_array = np.asarray(fallback_labels)
    if decision_array.shape != fallback_array.shape:
        raise ValueError("decisions and fallback_labels must align")
    override_decisions = int(np.count_nonzero(decision_array))
    effective_mask = decision_array & (fallback_array != CRITICAL_CLASS)
    effective_overrides = int(np.count_nonzero(effective_mask))
    return override_decisions, effective_overrides


def _aggregate_metrics(
    labels: Sequence[str], predictions: Sequence[str]
) -> dict[str, Any]:
    """Return only aggregate multiclass and critical-class metrics."""

    actual = _as_labels(labels, "labels")
    predicted = _as_labels(predictions, "predictions")
    if len(actual) != len(predicted):
        raise ValueError("labels and predictions must align")
    matrix = confusion_matrix(actual, predicted, labels=list(MODELED_FAMILIES))
    return _metrics_from_confusion(matrix)


def _metrics_from_confusion(matrix: np.ndarray) -> dict[str, Any]:
    """Calculate all V2 aggregate metrics from a fixed-order matrix."""

    matrix = np.asarray(matrix, dtype=np.int64)
    expected = (len(MODELED_FAMILIES), len(MODELED_FAMILIES))
    if matrix.shape != expected or np.any(matrix < 0):
        raise ValueError("V2 confusion matrix is invalid")
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    total = int(support.sum())
    if total <= 0:
        raise ValueError("Cannot calculate metrics for an empty scope")
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros(len(MODELED_FAMILIES), dtype=np.float64),
        where=predicted != 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros(len(MODELED_FAMILIES), dtype=np.float64),
        where=support != 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(len(MODELED_FAMILIES), dtype=np.float64),
        where=(precision + recall) != 0,
    )
    critical_index = MODELED_FAMILIES.index(CRITICAL_CLASS)
    return {
        "row_count": total,
        "accuracy": float(true_positive.sum() / total),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.dot(f1, support) / total),
        "balanced_accuracy": float(recall.mean()),
        "critical_precision": float(precision[critical_index]),
        "critical_recall": float(recall[critical_index]),
        "critical_f1": float(f1[critical_index]),
        "critical_support": int(support[critical_index]),
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def search_detector_threshold_exact(
    labels: Sequence[str],
    detector_scores: Sequence[float],
    fallback_labels: Sequence[str],
    protocol: V2Protocol | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Search every detector margin and select the deterministic best threshold.

    Args:
        labels: Actual nine-family labels from inner calibration.
        detector_scores: Binary LinearSVC margins, never probabilities.
        fallback_labels: Frozen S7 labels aligned with ``labels``.
        protocol: Validated V2 protocol, or the frozen file by default.

    Returns:
        Selected threshold, aggregate metrics, gates, override decision
        counts, and candidate count.

    Raises:
        ValueError: If arrays are invalid or non-finite.
    """

    actual = _as_labels(labels, "inner_calibration_labels")
    fallback = _validate_fallback(
        fallback_labels, len(actual), "inner_calibration_fallback_labels"
    )
    scores = np.asarray(detector_scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) != len(actual) or not len(scores):
        raise ValueError("detector_scores must be a non-empty aligned vector")
    if not np.isfinite(scores).all():
        raise ValueError("detector_scores must be finite margins")
    validated = protocol if isinstance(protocol, V2Protocol) else (
        load_v2_protocol() if protocol is None else protocol
    )
    positions = {label: index for index, label in enumerate(MODELED_FAMILIES)}
    matrix = confusion_matrix(
        actual,
        fallback,
        labels=list(MODELED_FAMILIES),
    ).astype(np.int64)
    order = np.argsort(scores, kind="stable")[::-1]
    thresholds = [float(np.nextafter(np.max(scores), np.inf))]
    thresholds.extend(float(value) for value in np.unique(scores)[::-1])
    selected: dict[str, Any] | None = None
    cursor = 0
    for threshold in thresholds:
        while cursor < len(order) and scores[order[cursor]] >= threshold:
            row = int(order[cursor])
            predicted_label = fallback[row]
            if predicted_label != CRITICAL_CLASS:
                actual_index = positions[actual[row]]
                predicted_index = positions[predicted_label]
                critical_index = positions[CRITICAL_CLASS]
                matrix[actual_index, predicted_index] -= 1
                matrix[actual_index, critical_index] += 1
            cursor += 1
        metrics = _metrics_from_confusion(matrix)
        gates = calculate_scientific_gates(metrics, validated)
        candidate = {
            "threshold": threshold,
            "metrics": metrics,
            "gates": gates,
        }
        key = (
            int(gates["passed"]),
            int(gates["gate_count"]),
            metrics["critical_f1"],
            metrics["macro_f1"],
            metrics["critical_precision"],
            threshold,
        )
        if selected is None or key > selected["_sort_key"]:
            candidate["_sort_key"] = key
            selected = candidate
    if selected is None:
        raise RuntimeError("Threshold search produced no candidate")
    selected.pop("_sort_key")
    final_decisions = scores >= float(selected["threshold"])
    override_decisions, effective_overrides = count_override_decisions(
        final_decisions, fallback
    )
    selected["override_decisions"] = override_decisions
    selected["effective_overrides"] = effective_overrides
    return {"selected": selected, "threshold_count": len(thresholds)}


def evaluate_detector(
    detector: V2CriticalDetector,
    texts: Sequence[str],
    labels: Sequence[str],
    fallback_labels: Sequence[str],
    *,
    partition: str,
    threshold: float,
    protocol: V2Protocol | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a fitted detector and frozen fallback on one allowed partition."""

    role = "outer" if partition == "validation" else "inner_calibration"
    normalized_texts, actual = _validate_aligned(
        texts, labels, partition, role
    )
    fallback = _validate_fallback(fallback_labels, len(actual), "fallback_labels")
    scores = detector._decision_scores(normalized_texts)
    predictions = combine_detector_with_fallback(scores >= float(threshold), fallback)
    metrics = _aggregate_metrics(actual, predictions)
    validated = protocol if isinstance(protocol, V2Protocol) else (
        load_v2_protocol() if protocol is None else protocol
    )
    return {
        "partition": partition,
        "threshold": float(threshold),
        "metrics": metrics,
        "gates": calculate_scientific_gates(metrics, validated),
    }


def fit_and_evaluate(
    fit_texts: Sequence[str],
    fit_labels: Sequence[str],
    calibration_texts: Sequence[str],
    calibration_labels: Sequence[str],
    calibration_fallback_labels: Sequence[str],
    outer_texts: Sequence[str],
    outer_labels: Sequence[str],
    outer_fallback_labels: Sequence[str],
    *,
    fit_partition: str,
    calibration_partition: str,
    outer_partition: str,
    representation_alias: str,
    balance_strategy: str,
    c_value: float,
    sampling_strategy: float | None = None,
    protocol: V2Protocol | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit one candidate, calibrate on inner data, and evaluate outer data.

    Returns only aggregate metrics, frozen parameters, threshold metadata, and
    runtime. No narrative, identifier, or individual score is returned.
    """

    _validate_partition(fit_partition, "fit")
    _validate_partition(calibration_partition, "inner_calibration")
    _validate_partition(outer_partition, "outer")
    validated = protocol if isinstance(protocol, V2Protocol) else (
        load_v2_protocol() if protocol is None else protocol
    )
    started = time.perf_counter()
    detector = V2CriticalDetector(
        representation_alias=representation_alias,
        balance_strategy=balance_strategy,
        c_value=c_value,
        sampling_strategy=sampling_strategy,
    ).fit(fit_texts, fit_labels, partition=fit_partition)
    calibration_values, calibration_actual = _validate_aligned(
        calibration_texts, calibration_labels, calibration_partition,
        "inner_calibration",
    )
    calibration_fallback = _validate_fallback(
        calibration_fallback_labels,
        len(calibration_actual),
        "calibration_fallback_labels",
    )
    calibration_scores = detector._decision_scores(calibration_values)
    threshold_result = search_detector_threshold_exact(
        calibration_actual,
        calibration_scores,
        calibration_fallback,
        validated,
    )
    threshold = float(threshold_result["selected"]["threshold"])
    outer = evaluate_detector(
        detector,
        outer_texts,
        outer_labels,
        outer_fallback_labels,
        partition=outer_partition,
        threshold=threshold,
        protocol=validated,
    )
    return {
        "parameters": detector.parameters(),
        "calibration": {
            "threshold": threshold,
            "threshold_count": threshold_result["threshold_count"],
            "metrics": threshold_result["selected"]["metrics"],
            "gates": threshold_result["selected"]["gates"],
        },
        "outer": outer,
        "runtime_seconds": float(time.perf_counter() - started),
    }


def run_v2_detector_smoke() -> dict[str, Any]:
    """Run a deterministic nine-class, in-memory diagnostic smoke test."""

    texts: list[str] = []
    labels: list[str] = []
    for label in MODELED_FAMILIES:
        for repeat in range(6):
            texts.append(f"{label} complaint topic {label} repeat {repeat}")
            labels.append(label)
    calibration_texts = [
        f"{label} complaint topic {label} calibration {repeat}"
        for label in MODELED_FAMILIES
        for repeat in range(3)
    ]
    calibration_labels = [
        label for label in MODELED_FAMILIES for _ in range(3)
    ]
    fallback = list(calibration_labels)
    critical_index = calibration_labels.index(CRITICAL_CLASS)
    fallback[critical_index] = MODELED_FAMILIES[0]
    detector = V2CriticalDetector(
        WORD_TFIDF_ALIAS, WEIGHTED_FULL, 0.3
    ).fit(texts, labels, partition="train")
    scores = detector._decision_scores(calibration_texts)
    threshold_result = search_detector_threshold_exact(
        calibration_labels, scores, fallback
    )
    selected = threshold_result["selected"]
    predictions = combine_detector_with_fallback(
        scores >= selected["threshold"], fallback
    )
    preserved = fallback[critical_index] == CRITICAL_CLASS or (
        combine_detector_with_fallback(
            (False,), (CRITICAL_CLASS,)
        )[0] == CRITICAL_CLASS
    )
    recovered = predictions[critical_index] == CRITICAL_CLASS
    return {
        "status": "DIAGNOSTIC_ONLY",
        "parameters": detector.parameters(),
        "metrics": selected["metrics"],
        "gates": selected["gates"],
        "threshold": selected["threshold"],
        "threshold_count": threshold_result["threshold_count"],
        "checks": {
            "all_nine_classes_present": len(set(labels)) == len(MODELED_FAMILIES),
            "fallback_critical_preserved": bool(preserved),
            "stage_a_recovered_critical": bool(recovered),
        },
        "runtime_seconds": float(detector.fit_runtime_seconds),
    }


__all__ = [
    "BALANCE_STRATEGIES",
    "HARD_NEGATIVE",
    "RANDOM_OVER",
    "REPRESENTATION_ALIASES",
    "V2CriticalDetector",
    "WEIGHTED_FULL",
    "WORD_CHAR_TFIDF_ALIAS",
    "WORD_TFIDF_ALIAS",
    "build_estimator",
    "build_vectorizer",
    "combine_detector_with_fallback",
    "count_override_decisions",
    "evaluate_detector",
    "fit_and_evaluate",
    "role_partition_map",
    "run_v2_detector_smoke",
    "search_detector_threshold_exact",
]
