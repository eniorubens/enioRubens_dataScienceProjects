"""V2 frozen hierarchical package guarded by an exact D1 reproduction gate.

Step 6 of the seven-step cycle freezes the V2 package: the fitted stage-A
critical detector, its calibrated threshold, and a hash reference to the
frozen S7 fallback serving as stage B. D1 persisted no fitted estimator, so
the selected candidate is refitted here under D1's exact code path -- the
same cache, the same frozen S7 fallback labels, the same deterministic
hard-negative pool, the same representation, the same threshold search, and
the same batched outer evaluation.

Refitting creates the possibility that the frozen artifact differs from the
object D1 measured, so the freeze is conditional on a reproduction gate with
no numeric tolerance at all. Every published comparison is recorded as a
named boolean. If any one of them fails the outcome is
``REPRODUCTION_MISMATCH``: the divergence is published as evidence and no
joblib bundle is written. Nothing is ever frozen silently over numbers that
do not match.

The runner never reads the raw dataset, never unlocks a sealed partition,
and persists aggregate evidence only -- no narratives, identifiers, row
indices, individual margins, or hard-negative row positions.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from .contracts import Prediction, PredictionBatch
from .s6 import CRITICAL_CLASS
from .s7 import load_s7_predictor
from .v2_benchmark import DEFAULT_CACHE as BENCHMARK_DEFAULT_CACHE
from .v2_benchmark import (
    _evaluate_outer,
    _fallback_labels,
    _fallback_only_metrics,
    _fit_candidate,
    _fit_representation,
    _prediction_labels,
    _read_development_cache,
    _read_json,
    _relative,
    _s7_paths,
    _safe_path,
    _sha256,
    _signature,
    _validate_result_privacy,
    _write_json_atomic,
    generate_hard_negative_indices,
)
from .v2_detector import (
    HARD_NEGATIVE,
    WORD_CHAR_TFIDF_ALIAS,
    combine_detector_with_fallback,
    search_detector_threshold_exact,
)
from .v2_protocol import DEFAULT_CONFIG as V2_PROTOCOL_DEFAULT_CONFIG
from .v2_protocol import (
    V2Protocol,
    calculate_safety_margins,
    calculate_scientific_gates,
    load_v2_protocol,
)
from .v2_transformer import hard_negative_pool_signature


V2_PACKAGE_CONFIG_SCHEMA = "v2-frozen-package-config-v1"
V2_PACKAGE_RESULT_SCHEMA = "v2-package-v1"
V2_PACKAGE_MANIFEST_SCHEMA = "v2-package-results-manifest-v1"
V2_BUNDLE_SCHEMA = "v2-model-bundle-v1"
V2_PACKAGE_CODE_SCHEMA = "v2-package-runtime-v1"
MODEL_VERSION = "consumer-complaint-detector-v2"
INPUT_LANGUAGE = "en-US"
PACKAGE_STAGE = "V2.1-P"
PACKAGE_STATUS = "FROZEN_FOR_CONFIRMATION"
DEFAULT_PACKAGE_CONFIG = "config/v2_frozen_package.json"
DEFAULT_PACKAGE_BUNDLE = "artifacts/v2/consumer_complaint_detector_v2.joblib"
DEFAULT_PACKAGE_ARTIFACT = "temp/v2/v2_package.json"
DEFAULT_PACKAGE_MANIFEST = "config/v2_results.json"
DEFAULT_BATCH_SIZE = 4096
OUTCOME_FROZEN = "PACKAGE_FROZEN"
OUTCOME_MISMATCH = "REPRODUCTION_MISMATCH"
OUTCOME_DIAGNOSTIC = "DIAGNOSTIC_ONLY"
D1_POSITIVE_GROUPS = 946
D1_HARD_NEGATIVE_GROUPS = 14190
D1_POOL_ROWS = 15136
SEALED_PARTITIONS = ("test", "stress", "monitor")

_REQUIRED_CONFIG_KEYS = (
    "schema_version",
    "stage",
    "adr",
    "status",
    "approved_on",
    "protocol",
    "model_version",
    "critical_class",
    "input_language",
    "architecture",
    "candidate",
    "fit_scope",
    "calibration_scope",
    "outer_scope",
    "hard_negative_pool",
    "safety_margin",
    "reproduction_gate",
    "fallback_baseline_outer",
    "provenance",
    "boundary",
    "deployment",
    "outputs",
    "run_defaults",
)
_REQUIRED_GATE_CHECKS = (
    "calibrated_threshold",
    "calibration_confusion_matrix",
    "calibration_override_decisions",
    "calibration_effective_overrides",
    "outer_confusion_matrix",
    "outer_override_decisions",
    "outer_effective_overrides",
    "hard_negative_pool_counts",
)
_PROVENANCE_ROLES = (
    ("cache", ("cache",)),
    ("d1_artifact", ("d1", "artifact")),
    ("d1_manifest", ("d1", "manifest")),
    ("d2_artifact", ("d2", "artifact")),
    ("d2_manifest", ("d2", "manifest")),
    ("s7_bundle", ("s7", "bundle")),
    ("s7_manifest", ("s7", "manifest")),
    ("s7_result", ("s7", "result")),
)
_EXPECTED_SCALAR_SOURCES = {
    "threshold": ("calibration", ("threshold",), float),
    "calibration_override_decisions": (
        "calibration",
        ("override_decisions",),
        int,
    ),
    "calibration_effective_overrides": (
        "calibration",
        ("effective_overrides",),
        int,
    ),
    "calibration_row_count": ("calibration", ("metrics", "row_count"), int),
    "outer_override_decisions": ("outer", ("override_decisions",), int),
    "outer_effective_overrides": ("outer", ("effective_overrides",), int),
    "outer_row_count": ("outer", ("metrics", "row_count"), int),
    "outer_critical_f1": ("outer", ("metrics", "critical_f1"), float),
    "outer_critical_precision": (
        "outer",
        ("metrics", "critical_precision"),
        float,
    ),
    "outer_critical_recall": ("outer", ("metrics", "critical_recall"), float),
    "outer_macro_f1": ("outer", ("metrics", "macro_f1"), float),
    "outer_critical_support": ("outer", ("metrics", "critical_support"), int),
}
_WORD_BRANCH_PARAMS = {
    "analyzer": "word",
    "ngram_range": (1, 2),
    "max_features": 40000,
    "min_df": 2,
    "max_df": 0.98,
    "sublinear_tf": True,
    "dtype": np.float32,
}
_CHAR_BRANCH_PARAMS = {
    "analyzer": "char_wb",
    "ngram_range": (3, 5),
    "max_features": 60000,
    "min_df": 2,
    "max_df": 0.98,
    "sublinear_tf": True,
    "dtype": np.float32,
}
_ESTIMATOR_PARAMS = {
    "C": 1.0,
    "class_weight": None,
    "tol": 0.0001,
    "max_iter": 5000,
    "dual": "auto",
    "random_state": 42,
}


def default_project_root() -> Path:
    """Return the project root derived from this module's own location.

    The process working directory is never consulted. A staged execution
    environment such as a Kaggle kernel runs with an unrelated working
    directory while the project tree lives elsewhere, so every path in this
    module resolves against either an explicit ``project_root`` or the
    directory two levels above the installed package.

    Returns:
        Absolute path to the project root containing ``config`` and ``src``.
    """

    return Path(__file__).resolve().parents[2]


def _resolve_root(project_root: str | Path | None) -> Path:
    """Resolve an optional project root without ever using the working directory.

    Args:
        project_root: Explicit project root, or ``None`` for the package's own
            location.

    Returns:
        The absolute project root.
    """

    if project_root is None:
        return default_project_root()
    return Path(project_root).expanduser().resolve()


def _resolve_path(root: Path, value: str | Path | None, default: str) -> Path:
    """Resolve one optional project-relative or absolute override path.

    Args:
        root: Project root used to resolve relative paths.
        value: Caller-supplied override, or ``None`` to use ``default``.
        default: Project-relative default path.

    Returns:
        The resolved absolute path.
    """

    candidate = value if value is not None else default
    if Path(candidate).is_absolute():
        return Path(candidate).expanduser().resolve()
    return _safe_path(root, str(candidate))


def _dig(payload: Mapping[str, Any], keys: Sequence[str]) -> Any:
    """Read one nested value from a mapping by an ordered key path.

    Args:
        payload: Mapping to traverse.
        keys: Ordered keys naming the nested value.

    Returns:
        The nested value.

    Raises:
        ValueError: If any key on the path is absent or not a mapping.
    """

    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            raise ValueError(f"Missing required key path: {'.'.join(keys)}")
        current = current[key]
    return current


@dataclass(frozen=True, slots=True)
class V2PackageConfig:
    """Hold the frozen, approved V2 package execution contract.

    Attributes:
        payload: The validated JSON configuration.
        path: Absolute path the configuration was loaded from.
        signature: ``{"sha256": ..., "size_bytes": ...}`` for ``path``.
    """

    payload: Mapping[str, Any]
    path: Path
    signature: Mapping[str, Any]

    @property
    def schema_version(self) -> str:
        """Return the frozen configuration schema version."""

        return str(self.payload["schema_version"])

    @property
    def status(self) -> str:
        """Return the frozen package status."""

        return str(self.payload["status"])

    @property
    def approved_on(self) -> str:
        """Return the approval date recorded by the governing ADR."""

        return str(self.payload["approved_on"])

    @property
    def adr(self) -> str:
        """Return the project-relative path of the governing ADR."""

        return str(self.payload["adr"])

    @property
    def candidate(self) -> Mapping[str, Any]:
        """Return the pinned D1 candidate descriptor."""

        return dict(self.payload["candidate"])

    @property
    def candidate_id(self) -> str:
        """Return the pinned D1 candidate identifier."""

        return str(self.payload["candidate"]["candidate_id"])

    @property
    def architecture(self) -> Mapping[str, Any]:
        """Return the frozen hierarchical architecture block."""

        return dict(self.payload["architecture"])

    @property
    def hard_negative_pool(self) -> Mapping[str, Any]:
        """Return the frozen hard-negative pool contract."""

        return dict(self.payload["hard_negative_pool"])

    @property
    def reproduction_gate(self) -> Mapping[str, Any]:
        """Return the frozen reproduction-gate contract."""

        return dict(self.payload["reproduction_gate"])

    @property
    def expected(self) -> Mapping[str, Any]:
        """Return the pre-registered scalar expectations of the gate."""

        return dict(self.payload["reproduction_gate"]["expected"])

    @property
    def provenance(self) -> Mapping[str, Any]:
        """Return the pinned provenance block."""

        return dict(self.payload["provenance"])

    @property
    def boundary(self) -> Mapping[str, Any]:
        """Return the declared partition and persistence boundary."""

        return dict(self.payload["boundary"])

    @property
    def outputs(self) -> Mapping[str, Any]:
        """Return the declared bundle, artifact, and manifest paths."""

        return dict(self.payload["outputs"])

    @property
    def batch_size(self) -> int:
        """Return the frozen default batch size."""

        return int(self.payload["run_defaults"]["batch_size"])

    def descriptor(self) -> dict[str, Any]:
        """Build the ``_fit_candidate`` descriptor for the pinned candidate.

        Returns:
            A benchmark-compatible candidate descriptor carrying the frozen
            representation alias, regularization, balance strategy, and
            sampling strategy.
        """

        candidate = self.payload["candidate"]
        return {
            "candidate_id": str(candidate["candidate_id"]),
            "representation": str(candidate["representation"]),
            "C": float(candidate["C"]),
            "balance_strategy": str(candidate["balance_strategy"]),
            "sampling_strategy": candidate["sampling_strategy"],
        }

    def validate(self) -> None:
        """Validate every frozen value, boundary, and declared output path.

        Raises:
            ValueError: If any frozen value diverges from the approved
                package contract in ``docs/ADR-013-v2-frozen-package.md``.
        """

        missing = [key for key in _REQUIRED_CONFIG_KEYS if key not in self.payload]
        if missing:
            raise ValueError(
                f"V2 package config is missing required keys: {sorted(missing)}"
            )
        if self.schema_version != V2_PACKAGE_CONFIG_SCHEMA:
            raise ValueError("Unexpected V2 package configuration schema")
        if str(self.payload["stage"]) != PACKAGE_STAGE:
            raise ValueError("V2 package config stage is invalid")
        if self.status != PACKAGE_STATUS:
            raise ValueError("V2 package config is not frozen for confirmation")
        if str(self.payload["model_version"]) != MODEL_VERSION:
            raise ValueError("V2 package model version is invalid")
        if str(self.payload["critical_class"]) != CRITICAL_CLASS:
            raise ValueError("V2 package critical class is invalid")
        if str(self.payload["input_language"]) != INPUT_LANGUAGE:
            raise ValueError("V2 package input language must be en-US")
        self._validate_architecture()
        self._validate_candidate()
        self._validate_pool()
        self._validate_gate()
        self._validate_boundary()
        self._validate_outputs()
        if self.batch_size <= 0:
            raise ValueError("V2 package batch_size must be positive")

    def _validate_architecture(self) -> None:
        """Validate the frozen hierarchical architecture declarations."""

        architecture = self.payload["architecture"]
        expected = {
            "type": "hierarchical_critical_override",
            "stage_a": "binary_critical_detector",
            "stage_b": "frozen_s7_multiclass_fallback",
            "combination": "critical_override_at_or_above_calibrated_threshold",
            "score_kind": "critical_margin",
            "threshold_source": "inner_calibration_only",
            "threshold_search": "search_detector_threshold_exact",
            "no_refit_after_calibration": True,
        }
        if dict(architecture) != expected:
            raise ValueError("V2 package architecture differs from the contract")

    def _validate_candidate(self) -> None:
        """Validate the pinned D1 candidate descriptor."""

        candidate = self.payload["candidate"]
        if str(candidate["representation"]) != WORD_CHAR_TFIDF_ALIAS:
            raise ValueError("V2 package representation is not the D1 winner")
        if str(candidate["balance_strategy"]) != HARD_NEGATIVE:
            raise ValueError("V2 package balance strategy is not hard_negative")
        if float(candidate["C"]) != 1.0:
            raise ValueError("V2 package candidate C must be 1.0")
        if candidate["sampling_strategy"] is not None:
            raise ValueError("V2 package candidate must not resample")
        if int(candidate["random_state"]) != 42:
            raise ValueError("V2 package random_state must be 42")
        if str(candidate["selected_by"]) != "V2.1-D1":
            raise ValueError("V2 package candidate was not selected by D1")

    def _validate_pool(self) -> None:
        """Validate the frozen hard-negative pool contract and its counts."""

        pool = self.payload["hard_negative_pool"]
        if str(pool["source"]) != "generate_hard_negative_indices":
            raise ValueError("V2 package hard-negative source is invalid")
        if int(pool["hard_negative_per_positive"]) != 10:
            raise ValueError("V2 package hard-negative ratio is invalid")
        if int(pool["background_negative_per_positive"]) != 5:
            raise ValueError("V2 package background ratio is invalid")
        if int(pool["oof_n_splits"]) != 3 or int(pool["oof_random_state"]) != 42:
            raise ValueError("V2 package hard-negative OOF contract is invalid")
        if int(pool["positive_groups"]) != D1_POSITIVE_GROUPS:
            raise ValueError("V2 package positive pool count is not the D1 count")
        if int(pool["hard_negative_groups"]) != D1_HARD_NEGATIVE_GROUPS:
            raise ValueError("V2 package negative pool count is not the D1 count")
        if int(pool["row_count"]) != D1_POOL_ROWS:
            raise ValueError("V2 package pool row count is not the D1 count")
        if pool["persisted"] is not False:
            raise ValueError("V2 package must never persist the hard-negative pool")

    def _validate_gate(self) -> None:
        """Validate the reproduction-gate contract and its expectation keys."""

        gate = self.payload["reproduction_gate"]
        if gate.get("required") is not True:
            raise ValueError("V2 package reproduction gate must be required")
        if str(gate.get("comparison")) != "exact_no_tolerance":
            raise ValueError("V2 package reproduction gate must be exact")
        if gate.get("publishes_bundle_on_mismatch") is not False:
            raise ValueError("V2 package must not publish a bundle on mismatch")
        if str(gate.get("on_mismatch")) != OUTCOME_MISMATCH:
            raise ValueError("V2 package mismatch outcome name is invalid")
        if tuple(gate.get("checks", ())) != _REQUIRED_GATE_CHECKS:
            raise ValueError("V2 package reproduction checks differ from contract")
        if str(gate.get("source_of_truth")) != str(
            self.payload["provenance"]["d1"]["artifact"]["path"]
        ):
            raise ValueError("V2 package gate source of truth is not the D1 artifact")
        expected = gate.get("expected")
        if not isinstance(expected, Mapping):
            raise ValueError("V2 package gate expectations are missing")
        if set(expected) != set(_EXPECTED_SCALAR_SOURCES):
            raise ValueError("V2 package gate expectation keys differ from contract")

    def _validate_boundary(self) -> None:
        """Validate the sealed-partition and persistence boundary."""

        boundary = self.payload["boundary"]
        if tuple(boundary.get("allowed_partitions", ())) != ("train", "validation"):
            raise ValueError("V2 package allowed partitions are invalid")
        if tuple(boundary.get("sealed_partitions", ())) != SEALED_PARTITIONS:
            raise ValueError("V2 package sealed partition boundary is invalid")
        if boundary.get("no_sealed_partition_unlock_in_development_code") is not True:
            raise ValueError("V2 package must forbid sealed-partition unlocking")
        if boundary.get("persists_fitted_weights") is not True:
            raise ValueError("V2 package must declare persisted fitted weights")
        for field in (
            "persists_narratives_or_identifiers",
            "persists_row_indices",
            "persists_individual_margins",
        ):
            if boundary.get(field) is not False:
                raise ValueError(f"V2 package boundary flag must be false: {field}")
        if self.payload["deployment"].get("deployment_authorized") is not False:
            raise ValueError("V2 package must not authorize deployment")
        if str(self.payload["deployment"].get("status")) != PACKAGE_STATUS:
            raise ValueError("V2 package deployment status is invalid")

    def _validate_outputs(self) -> None:
        """Validate the declared cache, output paths, and protocol schema."""

        if str(self.payload["provenance"]["cache"]["path"]) != BENCHMARK_DEFAULT_CACHE:
            raise ValueError("V2 package accepts only temp/s3/scientific.parquet")
        outputs = self.payload["outputs"]
        expected = {
            "bundle": DEFAULT_PACKAGE_BUNDLE,
            "artifact": DEFAULT_PACKAGE_ARTIFACT,
            "manifest": DEFAULT_PACKAGE_MANIFEST,
        }
        if {key: str(value) for key, value in outputs.items()} != expected:
            raise ValueError("V2 package output paths differ from the contract")
        protocol = self.payload["protocol"]
        if str(protocol["path"]) != V2_PROTOCOL_DEFAULT_CONFIG:
            raise ValueError("V2 package protocol path is invalid")
        if tuple(protocol.get("seven_step_cycle_index", ())) != (5, 6):
            raise ValueError("V2 package must execute cycle steps five and six")


def load_v2_package_config(
    path: str | Path = DEFAULT_PACKAGE_CONFIG,
    *,
    project_root: str | Path | None = None,
) -> V2PackageConfig:
    """Load and strictly validate the frozen V2 package configuration.

    Pinned artifact hashes are deliberately not verified here: verifying them
    reads a 152 MB development cache. Use :func:`verify_package_provenance`
    for that, which every execution path calls before fitting anything.

    Args:
        path: Absolute path, or a project-relative path resolved against
            ``project_root``.
        project_root: Project root used for relative paths. Defaults to the
            installed package location, never the working directory.

    Returns:
        A validated V2 package configuration.

    Raises:
        ValueError: If the document is not the exact approved contract.
    """

    root = _resolve_root(project_root)
    resolved = _resolve_path(root, path, DEFAULT_PACKAGE_CONFIG)
    payload = _read_json(resolved)
    config = V2PackageConfig(
        payload=payload, path=resolved, signature=_signature(resolved)
    )
    config.validate()
    return config


def verify_package_provenance(
    config: V2PackageConfig,
    *,
    project_root: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Verify every pinned protocol and provenance hash under one root.

    Args:
        config: Validated frozen package configuration.
        project_root: Project root holding the pinned artifacts. Defaults to
            the installed package location, never the working directory.

    Returns:
        Actual ``{"path", "sha256", "size_bytes"}`` signatures keyed by role.

    Raises:
        ValueError: If a pinned artifact is absent, resized, or rehashed.
    """

    root = _resolve_root(project_root)
    targets: list[tuple[str, Mapping[str, Any]]] = [
        ("protocol", config.payload["protocol"])
    ]
    provenance = config.provenance
    for role, keys in _PROVENANCE_ROLES:
        targets.append((role, _dig(provenance, keys)))
    verified: dict[str, dict[str, Any]] = {}
    for role, pinned in targets:
        path = _safe_path(root, str(pinned["path"]))
        actual = _signature(path)
        expected = {
            "sha256": str(pinned["sha256"]),
            "size_bytes": int(pinned["size_bytes"]),
        }
        if actual != expected:
            raise ValueError(f"V2 package pinned artifact has drifted: {role}")
        verified[role] = {"path": _relative(path, root), **actual}
    return verified


def _locate_d1_record(
    payload: Mapping[str, Any], candidate_id: str
) -> dict[str, Any]:
    """Locate and shape the D1 record proving the selected candidate.

    Args:
        payload: The complete D1 classical benchmark artifact.
        candidate_id: The pinned candidate identifier.

    Returns:
        The candidate's calibration and outer blocks together with the D1
        run's hard-negative pool counts.

    Raises:
        ValueError: If the artifact is stale, incomplete, or does not carry
            exactly one record for ``candidate_id``.
    """

    if payload.get("complete") is not True:
        raise ValueError("D1 classical artifact is incomplete")
    if payload.get("selected") != candidate_id:
        raise ValueError("D1 classical artifact selected a different candidate")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("D1 classical artifact carries no candidate list")
    matches = [
        item
        for item in candidates
        if isinstance(item, Mapping) and item.get("candidate_id") == candidate_id
    ]
    if len(matches) != 1:
        raise ValueError("D1 candidate record is missing or duplicated")
    record = matches[0]
    pool = payload.get("hard_negative")
    if not isinstance(pool, Mapping):
        raise ValueError("D1 hard-negative evidence is missing")
    calibration = record.get("calibration")
    outer = record.get("outer")
    if not isinstance(calibration, Mapping) or not isinstance(outer, Mapping):
        raise ValueError("D1 candidate record is malformed")
    return {
        "candidate_id": candidate_id,
        "calibration": dict(calibration),
        "outer": dict(outer),
        "hard_negative": {
            "positive_groups": int(pool["positive_groups"]),
            "hard_negative_groups": int(pool["hard_negative_groups"]),
        },
    }


def _read_d1_record(
    config: V2PackageConfig,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Read the pinned D1 artifact and return the selected candidate record.

    The pinned hash is verified before the file is parsed, so a tampered D1
    artifact can never become the source of truth for the gate.

    Args:
        config: Validated frozen package configuration.
        project_root: Project root holding the pinned D1 artifact.

    Returns:
        The shaped D1 record for the pinned candidate.

    Raises:
        ValueError: If the artifact drifted, is stale, or lacks the record.
    """

    root = _resolve_root(project_root)
    pinned = _dig(config.provenance, ("d1", "artifact"))
    path = _safe_path(root, str(pinned["path"]))
    actual = _signature(path)
    if actual != {
        "sha256": str(pinned["sha256"]),
        "size_bytes": int(pinned["size_bytes"]),
    }:
        raise ValueError("V2 package pinned artifact has drifted: d1_artifact")
    return _locate_d1_record(_read_json(path), config.candidate_id)


def _matrix_difference(
    observed: Any, expected: Any
) -> dict[str, Any]:
    """Summarize a confusion-matrix divergence without any row-level data.

    Only shape and cell-difference aggregates are returned. Neither matrix
    is echoed, so the published divergence stays an aggregate summary.

    Args:
        observed: Refitted confusion matrix as a nested integer list.
        expected: D1's published confusion matrix as a nested integer list.

    Returns:
        A shape and cell-difference summary of the two matrices.
    """

    try:
        first = np.asarray(observed, dtype=np.int64)
        second = np.asarray(expected, dtype=np.int64)
    except (TypeError, ValueError):
        return {"comparable": False, "reason": "matrix_is_not_integer_shaped"}
    if first.shape != second.shape:
        return {
            "comparable": False,
            "reason": "matrix_shape_differs",
            "observed_shape": list(first.shape),
            "expected_shape": list(second.shape),
        }
    difference = np.abs(first - second)
    return {
        "comparable": True,
        "shape": list(first.shape),
        "mismatched_cells": int(np.count_nonzero(difference)),
        "total_absolute_difference": int(difference.sum()),
        "max_absolute_cell_difference": int(difference.max()),
        "observed_total": int(first.sum()),
        "expected_total": int(second.sum()),
    }


def _matrices_equal(observed: Any, expected: Any) -> bool:
    """Compare two confusion matrices element-wise with integer equality."""

    try:
        first = np.asarray(observed, dtype=np.int64)
        second = np.asarray(expected, dtype=np.int64)
    except (TypeError, ValueError):
        return False
    if first.shape != second.shape:
        return False
    return bool(np.array_equal(first, second))


def _evaluate_reproduction_gate(
    observed: Mapping[str, Any],
    d1_record: Mapping[str, Any],
    config: V2PackageConfig,
) -> dict[str, Any]:
    """Compare the refit against D1 and the frozen config with no tolerance.

    Every comparison is recorded as its own named boolean. Failures also
    publish the observed and expected values, as scalars or as a
    matrix-difference summary, so a divergence is usable evidence without
    exposing a single row of data.

    Args:
        observed: Aggregate evidence produced by the local refit.
        d1_record: The shaped D1 record for the pinned candidate.
        config: Validated frozen package configuration.

    Returns:
        The reproduction-gate block, carrying ``checks``, ``passed``,
        ``failed_checks``, and per-failure ``divergences``.

    Raises:
        ValueError: If a required check name is not evaluated.
    """

    checks: dict[str, bool] = {}
    divergences: dict[str, Any] = {}

    def record_scalar(name: str, actual: Any, expected: Any) -> None:
        """Record one exact scalar comparison and its divergence."""

        passed = actual == expected
        checks[name] = bool(passed)
        if not passed:
            divergences[name] = {"observed": actual, "expected": expected}

    def record_matrix(name: str, actual: Any, expected: Any) -> None:
        """Record one exact matrix comparison and its difference summary."""

        passed = _matrices_equal(actual, expected)
        checks[name] = bool(passed)
        if not passed:
            divergences[name] = _matrix_difference(actual, expected)

    calibration = observed["calibration"]
    outer = observed["outer"]
    pool = observed["hard_negative"]
    d1_calibration = d1_record["calibration"]
    d1_outer = d1_record["outer"]
    d1_pool = d1_record["hard_negative"]

    record_scalar(
        "calibrated_threshold",
        float(calibration["threshold"]),
        float(d1_calibration["threshold"]),
    )
    record_matrix(
        "calibration_confusion_matrix",
        calibration["metrics"]["confusion_matrix"],
        d1_calibration["metrics"]["confusion_matrix"],
    )
    record_scalar(
        "calibration_override_decisions",
        int(calibration["override_decisions"]),
        int(d1_calibration["override_decisions"]),
    )
    record_scalar(
        "calibration_effective_overrides",
        int(calibration["effective_overrides"]),
        int(d1_calibration["effective_overrides"]),
    )
    record_matrix(
        "outer_confusion_matrix",
        outer["metrics"]["confusion_matrix"],
        d1_outer["metrics"]["confusion_matrix"],
    )
    record_scalar(
        "outer_override_decisions",
        int(outer["override_decisions"]),
        int(d1_outer["override_decisions"]),
    )
    record_scalar(
        "outer_effective_overrides",
        int(outer["effective_overrides"]),
        int(d1_outer["effective_overrides"]),
    )
    observed_pool = (
        int(pool["positive_groups"]),
        int(pool["hard_negative_groups"]),
    )
    d1_pool_counts = (
        int(d1_pool["positive_groups"]),
        int(d1_pool["hard_negative_groups"]),
    )
    record_scalar(
        "hard_negative_pool_counts", list(observed_pool), list(d1_pool_counts)
    )
    configured_pool = config.hard_negative_pool
    config_pool_counts = (
        int(configured_pool["positive_groups"]),
        int(configured_pool["hard_negative_groups"]),
    )
    record_scalar(
        "config_expected_hard_negative_pool",
        list(observed_pool),
        list(config_pool_counts),
    )
    expected_scalars = config.expected
    for key, (block, keys, caster) in _EXPECTED_SCALAR_SOURCES.items():
        actual = caster(_dig(observed[block], keys))
        record_scalar(
            f"config_expected_{key}", actual, caster(expected_scalars[key])
        )
    missing = [name for name in _REQUIRED_GATE_CHECKS if name not in checks]
    if missing:
        raise ValueError(f"V2 reproduction gate skipped checks: {sorted(missing)}")
    failed = sorted(name for name, value in checks.items() if not value)
    gate = config.reproduction_gate
    return {
        "required": True,
        "comparison": str(gate["comparison"]),
        "source_of_truth": str(gate["source_of_truth"]),
        "candidate_id": config.candidate_id,
        "required_checks": list(_REQUIRED_GATE_CHECKS),
        "checks": dict(checks),
        "check_count": len(checks),
        "passed": not failed,
        "failed_checks": failed,
        "divergences": divergences,
        "fallback_environment": str(gate.get("fallback_environment", "")),
    }


@dataclass(frozen=True, slots=True)
class V2ModelBundle:
    """Own the fitted stage-A detector and the exact V2 serving invariants.

    Attributes:
        vectorizer: The fitted word-plus-character TF-IDF union.
        estimator: The fitted binary critical LinearSVC.
        threshold: The margin threshold calibrated on inner calibration only.
        critical_class: The one class stage A may emit.
        model_version: The immutable V2 model version.
        input_language: The required narrative language contract.
        schema_version: The bundle schema version.
    """

    vectorizer: FeatureUnion
    estimator: LinearSVC
    threshold: float
    critical_class: str = CRITICAL_CLASS
    model_version: str = MODEL_VERSION
    input_language: str = INPUT_LANGUAGE
    schema_version: str = V2_BUNDLE_SCHEMA

    def validate(self) -> None:
        """Validate the serialized representation, estimator, and contract.

        Raises:
            ValueError: If any frozen serving invariant fails.
        """

        if self.schema_version != V2_BUNDLE_SCHEMA:
            raise ValueError("Unexpected V2 bundle schema")
        if self.input_language != INPUT_LANGUAGE:
            raise ValueError("V2 bundle input language must be en-US")
        if self.model_version != MODEL_VERSION:
            raise ValueError("Unexpected V2 model version")
        if self.critical_class != CRITICAL_CLASS:
            raise ValueError("V2 bundle critical class is invalid")
        if isinstance(self.threshold, bool) or not isinstance(
            self.threshold, (int, float, np.floating, np.integer)
        ):
            raise ValueError("V2 bundle threshold must be a real number")
        if not np.isfinite(float(self.threshold)):
            raise ValueError("V2 threshold must be finite")
        self._validate_vectorizer()
        self._validate_estimator()

    def _validate_vectorizer(self) -> None:
        """Validate the fitted word and character TF-IDF union branches."""

        if not isinstance(self.vectorizer, FeatureUnion):
            raise ValueError("V2 bundle vectorizer must be a FeatureUnion")
        branches = dict(self.vectorizer.transformer_list)
        if tuple(name for name, _ in self.vectorizer.transformer_list) != (
            "word",
            "char",
        ):
            raise ValueError("V2 vectorizer branches must be word then char")
        for name, expected in (
            ("word", _WORD_BRANCH_PARAMS),
            ("char", _CHAR_BRANCH_PARAMS),
        ):
            params = branches[name].get_params()
            for key, value in expected.items():
                if params.get(key) != value:
                    raise ValueError(
                        f"V2 vectorizer invariant failed: {name}.{key}"
                    )

    def _validate_estimator(self) -> None:
        """Validate the fitted binary LinearSVC and its class order."""

        if not isinstance(self.estimator, LinearSVC):
            raise ValueError("V2 bundle estimator must be a LinearSVC")
        params = self.estimator.get_params()
        for key, value in _ESTIMATOR_PARAMS.items():
            if params.get(key) != value:
                raise ValueError(f"V2 estimator invariant failed: {key}")
        classes = getattr(self.estimator, "classes_", None)
        if classes is None:
            raise ValueError("V2 bundle estimator is not fitted")
        if tuple(int(value) for value in np.asarray(classes).ravel()) != (0, 1):
            raise ValueError("V2 estimator classes must be the binary [0, 1]")


def _validate_batch(
    texts: Sequence[str], input_language: str, expected_language: str
) -> tuple[str, ...]:
    """Validate one serving batch exactly as ``S7Predictor.predict`` does.

    Args:
        texts: Candidate narratives supplied by the caller.
        input_language: Language declared by the caller.
        expected_language: The bundle's required language contract.

    Returns:
        The validated narratives in request order.

    Raises:
        ValueError: If the batch, text, or language contract is invalid.
    """

    if input_language != expected_language:
        raise ValueError("V2 predictor accepts input_language=en-US only")
    if isinstance(texts, (str, bytes)):
        raise ValueError("texts must be a sequence of strings")
    try:
        values = tuple(texts)
    except TypeError as error:
        raise ValueError("texts must be a sequence of strings") from error
    if not values:
        raise ValueError("texts must contain at least one item")
    if not all(isinstance(text, str) for text in values):
        raise ValueError("texts must contain only strings")
    if any(not text.strip() for text in values):
        raise ValueError("texts must not contain empty narratives")
    return values


class V2Predictor:
    """Apply the frozen hierarchical V2 package behind the neutral contract.

    Stage A scores a critical margin from the frozen bundle. Where the margin
    reaches the calibrated threshold the critical class is emitted; every
    other row keeps the frozen S7 fallback's multiclass label.
    """

    def __init__(self, bundle: V2ModelBundle, fallback: Any) -> None:
        """Initialize a predictor after validating its bundle and fallback.

        Args:
            bundle: The frozen stage-A bundle.
            fallback: The frozen S7 stage-B predictor.

        Raises:
            ValueError: If the bundle or the fallback contract is invalid.
        """

        bundle.validate()
        if not callable(getattr(fallback, "predict", None)):
            raise ValueError("V2 fallback must expose a predict method")
        self._bundle = bundle
        self._fallback = fallback

    @property
    def model_version(self) -> str:
        """Return the immutable V2 model version exposed by the bundle."""

        return self._bundle.model_version

    @property
    def input_language(self) -> str:
        """Return the required language contract for incoming narratives."""

        return self._bundle.input_language

    @property
    def threshold(self) -> float:
        """Return the frozen stage-A critical-margin threshold."""

        return float(self._bundle.threshold)

    @property
    def fallback_model_version(self) -> str:
        """Return the stage-B model version, or ``unknown`` when absent."""

        return str(getattr(self._fallback, "model_version", "unknown"))

    def decision_margins(
        self,
        texts: Sequence[str],
        *,
        input_language: str = INPUT_LANGUAGE,
    ) -> np.ndarray:
        """Return stage-A critical margins for framework interoperability.

        Args:
            texts: Non-empty English complaint narratives.
            input_language: Language declared by the caller.

        Returns:
            One finite float64 margin per narrative, in request order.

        Raises:
            ValueError: If the batch or the margins are invalid.
        """

        values = _validate_batch(
            texts, input_language, self._bundle.input_language
        )
        matrix = self._bundle.vectorizer.transform(values)
        margins = np.asarray(
            self._bundle.estimator.decision_function(matrix), dtype=np.float64
        )
        if margins.ndim != 1 or len(margins) != len(values):
            raise ValueError("Binary detector must return one margin per text")
        if not np.isfinite(margins).all():
            raise ValueError("V2 stage-A margins must be finite")
        return margins

    def predict(
        self,
        texts: Sequence[str],
        *,
        input_language: str = INPUT_LANGUAGE,
    ) -> PredictionBatch:
        """Predict an ordered batch through the hierarchical override rule.

        Args:
            texts: Non-empty English complaint narratives.
            input_language: Language declared by the caller.

        Returns:
            Predictions carrying the stage-A margin as ``score`` and
            ``score_kind=critical_margin`` metadata.

        Raises:
            ValueError: If the batch, text, or language contract is invalid.
        """

        values = _validate_batch(
            texts, input_language, self._bundle.input_language
        )
        margins = self.decision_margins(values, input_language=input_language)
        decisions = margins >= float(self._bundle.threshold)
        fallback_labels = _prediction_labels(
            self._fallback.predict(values, input_language=input_language)
        )
        if len(fallback_labels) != len(values):
            raise ValueError("Fallback prediction count differs from the batch")
        labels = combine_detector_with_fallback(decisions, fallback_labels)
        metadata = {
            "score_kind": "critical_margin",
            "threshold": float(self._bundle.threshold),
            "input_language": self._bundle.input_language,
            "stage": "hierarchical_critical_override",
            "fallback_model_version": self.fallback_model_version,
        }
        predictions = tuple(
            Prediction(
                label=str(label),
                score=float(margin),
                model_version=self._bundle.model_version,
                metadata=dict(metadata),
            )
            for label, margin in zip(labels, margins)
        )
        return PredictionBatch(predictions=predictions)


def _dump_joblib_atomic(bundle: V2ModelBundle, path: Path) -> None:
    """Persist one joblib bundle through a same-directory atomic replacement.

    Args:
        bundle: The validated frozen bundle to persist.
        path: Destination joblib path.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        joblib.dump(bundle, temporary, compress=3)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _refit_selected_candidate(
    scopes: Mapping[str, Any],
    fallback: Mapping[str, Sequence[str]],
    protocol: V2Protocol,
    config: V2PackageConfig,
    *,
    batch_size: int,
) -> tuple[dict[str, Any], V2ModelBundle]:
    """Refit the pinned D1 candidate along D1's exact code path.

    The representation is fitted on the complete ``inner_fit`` scope, never
    on the hard-negative subset; only the estimator sees the pool. The
    threshold is searched on ``inner_calibration`` alone, and the outer
    window is evaluated in the same bounded batches D1 used.

    Args:
        scopes: The three frozen development scopes.
        fallback: Frozen S7 fallback labels for calibration and outer scopes.
        protocol: Validated V2 development protocol.
        config: Validated frozen package configuration.
        batch_size: Bounded outer evaluation batch size.

    Returns:
        A ``(observed, bundle)`` pair, where ``observed`` is aggregate-only
        evidence and ``bundle`` is the unvalidated candidate package.

    Raises:
        ValueError: If ``batch_size`` is not positive.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    fit_scope = scopes["inner_fit"]
    calibration_scope = scopes["inner_calibration"]
    outer_scope = scopes["outer_evaluation"]
    hard_indices = generate_hard_negative_indices(
        fit_scope.texts, fit_scope.labels
    )
    positive_count = sum(label == CRITICAL_CLASS for label in fit_scope.labels)
    descriptor = config.descriptor()
    name = str(descriptor["candidate_id"])
    representation = _fit_representation(
        str(descriptor["representation"]), fit_scope, calibration_scope
    )
    estimator = _fit_candidate(
        descriptor, representation, fit_scope, hard_indices
    )
    threshold_result = search_detector_threshold_exact(
        calibration_scope.labels,
        estimator.decision_function(representation.calibration_matrix),
        fallback["inner_calibration"],
        protocol,
    )
    selected = threshold_result["selected"]
    threshold = float(selected["threshold"])
    calibration_block = {
        "threshold": threshold,
        "threshold_count": int(threshold_result["threshold_count"]),
        "metrics": selected["metrics"],
        "gates": selected["gates"],
        "override_decisions": int(selected["override_decisions"]),
        "effective_overrides": int(selected["effective_overrides"]),
    }
    outer_entry = _evaluate_outer(
        representation,
        {name: estimator},
        {name: threshold},
        outer_scope,
        fallback["outer_evaluation"],
        batch_size,
    )[name]
    fallback_baseline = {
        "inner_calibration": _fallback_only_metrics(
            calibration_scope.labels, fallback["inner_calibration"]
        ),
        "outer_evaluation": _fallback_only_metrics(
            outer_scope.labels, fallback["outer_evaluation"]
        ),
    }
    outer_metrics = outer_entry["metrics"]
    outer_block = {
        "metrics": outer_metrics,
        "gates": calculate_scientific_gates(outer_metrics, protocol),
        "safety": calculate_safety_margins(outer_metrics, protocol),
        "override_decisions": int(outer_entry["override_decisions"]),
        "effective_overrides": int(outer_entry["effective_overrides"]),
        "critical_f1_vs_fallback": (
            float(outer_metrics["critical_f1"])
            - float(fallback_baseline["outer_evaluation"]["critical_f1"])
        ),
    }
    observed = {
        "candidate_id": name,
        "calibration": calibration_block,
        "outer": outer_block,
        "hard_negative": {
            "positive_groups": int(positive_count),
            "hard_negative_groups": int(len(hard_indices) - positive_count),
            "pool_rows": int(len(hard_indices)),
            "pool_signature": hard_negative_pool_signature(hard_indices),
        },
        "fallback_baseline": fallback_baseline,
    }
    bundle = V2ModelBundle(
        vectorizer=representation.vectorizer,
        estimator=estimator,
        threshold=threshold,
        critical_class=CRITICAL_CLASS,
        model_version=MODEL_VERSION,
        input_language=INPUT_LANGUAGE,
    )
    return observed, bundle


def _package_signature(
    config: V2PackageConfig,
    provenance: Mapping[str, Mapping[str, Any]],
) -> str:
    """Build the full package signature from code, config, and frozen hashes.

    Args:
        config: Validated frozen package configuration.
        provenance: Verified signatures of every pinned input.

    Returns:
        An uppercase SHA256 hex digest identifying this execution's inputs.
    """

    value = {
        "code_schema": V2_PACKAGE_CODE_SCHEMA,
        "frozen_config": dict(config.signature),
        "provenance": {
            key: dict(item) for key, item in sorted(provenance.items())
        },
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True).encode("utf-8")
    ).hexdigest().upper()


def _base_result(
    config: V2PackageConfig,
    protocol: V2Protocol,
    *,
    run_mode: str,
    diagnostic_only: bool,
    opened_at: str,
    signature: str,
) -> dict[str, Any]:
    """Create the aggregate-only V2 package result envelope.

    Args:
        config: Validated frozen package configuration.
        protocol: Validated V2 development protocol.
        run_mode: ``full`` or ``smoke``.
        diagnostic_only: Whether the run may not publish a package.
        opened_at: UTC timestamp the run opened at.
        signature: Input signature for this execution.

    Returns:
        The result envelope shared by both execution modes.
    """

    return {
        "schema_version": V2_PACKAGE_RESULT_SCHEMA,
        "code_schema": V2_PACKAGE_CODE_SCHEMA,
        "stage": PACKAGE_STAGE,
        "adr": config.adr,
        "approved_on": config.approved_on,
        "status": OUTCOME_DIAGNOSTIC if diagnostic_only else "COMPLETE",
        "outcome": OUTCOME_DIAGNOSTIC,
        "complete": False,
        "frozen": False,
        "diagnostic_only": diagnostic_only,
        "run_mode": run_mode,
        "opened_at": opened_at,
        "runtime_seconds": None,
        "signature": signature,
        "input_language": INPUT_LANGUAGE,
        "critical_class": config.payload["critical_class"],
        "model_version": MODEL_VERSION,
        "allowed_partitions": list(protocol.allowed_partitions),
        "sealed_partitions": list(protocol.forbidden_partitions),
        "sealed_access": {name: False for name in protocol.forbidden_partitions},
        "architecture": config.architecture,
        "candidate": config.candidate,
        "fit_scope": dict(config.payload["fit_scope"]),
        "calibration_scope": dict(config.payload["calibration_scope"]),
        "outer_scope": dict(config.payload["outer_scope"]),
        "boundary": config.boundary,
        "deployment": dict(config.payload["deployment"]),
        "safety_margin": dict(config.payload["safety_margin"]),
        "frozen_config": dict(config.signature),
    }


def _smoke_result(
    config: V2PackageConfig,
    protocol: V2Protocol,
    provenance: Mapping[str, Mapping[str, Any]],
    d1_record: Mapping[str, Any],
    fallback_model_version: str,
    *,
    opened_at: str,
    runtime_seconds: float,
) -> dict[str, Any]:
    """Build the diagnostic smoke result, which fits and writes nothing.

    Args:
        config: Validated frozen package configuration.
        protocol: Validated V2 development protocol.
        provenance: Verified signatures of every pinned input.
        d1_record: The shaped D1 record for the pinned candidate.
        fallback_model_version: Model version reported by the S7 fallback.
        opened_at: UTC timestamp the run opened at.
        runtime_seconds: Diagnostic wall-clock duration.

    Returns:
        An aggregate-only diagnostic result.
    """

    result = _base_result(
        config,
        protocol,
        run_mode="smoke",
        diagnostic_only=True,
        opened_at=opened_at,
        signature="smoke",
    )
    result.update(
        {
            "complete": True,
            "runtime_seconds": float(runtime_seconds),
            "persisted": False,
            "fitted": False,
            "checks": {
                "frozen_config_validated": True,
                "protocol_hash_verified": "protocol" in provenance,
                "provenance_hashes_verified": all(
                    role in provenance for role, _ in _PROVENANCE_ROLES
                ),
                "d1_candidate_record_located": (
                    d1_record["candidate_id"] == config.candidate_id
                ),
                "d1_candidate_record_parsed": True,
                "s7_fallback_predictor_loaded": bool(fallback_model_version),
                "sealed_boundary_declared": True,
            },
            "provenance": {
                key: dict(item) for key, item in sorted(provenance.items())
            },
            "d1_reference": {
                "candidate_id": d1_record["candidate_id"],
                "calibration_threshold": float(
                    d1_record["calibration"]["threshold"]
                ),
                "calibration_override_decisions": int(
                    d1_record["calibration"]["override_decisions"]
                ),
                "calibration_effective_overrides": int(
                    d1_record["calibration"]["effective_overrides"]
                ),
                "outer_override_decisions": int(
                    d1_record["outer"]["override_decisions"]
                ),
                "outer_effective_overrides": int(
                    d1_record["outer"]["effective_overrides"]
                ),
                "positive_groups": int(
                    d1_record["hard_negative"]["positive_groups"]
                ),
                "hard_negative_groups": int(
                    d1_record["hard_negative"]["hard_negative_groups"]
                ),
            },
            "fallback_model_version": fallback_model_version,
            "bundle": {"persisted": False},
        }
    )
    _validate_result_privacy(result)
    return result


def _complete_result(
    base: Mapping[str, Any],
    observed: Mapping[str, Any],
    gate: Mapping[str, Any],
    provenance: Mapping[str, Mapping[str, Any]],
    bundle_metadata: Mapping[str, Any],
    fallback_model_version: str,
    runtime_seconds: float,
) -> dict[str, Any]:
    """Finalize one full result and enforce its aggregate privacy boundary.

    Args:
        base: The shared result envelope.
        observed: Aggregate evidence produced by the local refit.
        gate: The evaluated reproduction gate.
        provenance: Verified signatures of every pinned input.
        bundle_metadata: Bundle persistence metadata.
        fallback_model_version: Model version reported by the S7 fallback.
        runtime_seconds: Refit and evaluation wall-clock duration.

    Returns:
        The complete aggregate-only result.

    Raises:
        ValueError: If a forbidden row-level key reached the payload.
    """

    frozen = bool(gate["passed"])
    result = dict(base)
    result.update(
        {
            "complete": True,
            "frozen": frozen,
            "status": "COMPLETE",
            "outcome": OUTCOME_FROZEN if frozen else OUTCOME_MISMATCH,
            "runtime_seconds": float(runtime_seconds),
            "calibration": dict(observed["calibration"]),
            "outer": dict(observed["outer"]),
            "hard_negative": dict(observed["hard_negative"]),
            "fallback_baseline": dict(observed["fallback_baseline"]),
            "reproduction_gate": dict(gate),
            "provenance": {
                key: dict(item) for key, item in sorted(provenance.items())
            },
            "model_spec": {
                "model_version": MODEL_VERSION,
                "input_language": INPUT_LANGUAGE,
                "critical_class": CRITICAL_CLASS,
                "score_kind": "critical_margin",
                "stage": "hierarchical_critical_override",
                "threshold": float(observed["calibration"]["threshold"]),
                "fallback_model_version": fallback_model_version,
            },
            "bundle": dict(bundle_metadata),
        }
    )
    _validate_result_privacy(result)
    return result


def _publish_package_manifest(
    root: Path,
    config: V2PackageConfig,
    artifact_path: Path,
    bundle_path: Path | None,
    manifest_path: Path,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish the portable V2 package manifest with every recorded hash.

    Args:
        root: Project root every recorded path is relative to.
        config: Validated frozen package configuration.
        artifact_path: The published aggregate result path.
        bundle_path: The persisted joblib path, or ``None`` on mismatch.
        manifest_path: Destination manifest path.
        result: The complete aggregate result.

    Returns:
        The published manifest payload.
    """

    provenance = result["provenance"]
    manifest = {
        "schema_version": V2_PACKAGE_MANIFEST_SCHEMA,
        "stage": PACKAGE_STAGE,
        "status": result["status"],
        "outcome": result["outcome"],
        "complete": True,
        "frozen": bool(result["frozen"]),
        "diagnostic_only": False,
        "adr": config.adr,
        "approved_on": config.approved_on,
        "signature": result["signature"],
        "deployment_authorized": False,
        "package_status": PACKAGE_STATUS,
        "frozen_config": {
            "path": _relative(config.path, root),
            **dict(config.signature),
        },
        "protocol": dict(provenance["protocol"]),
        "cache": dict(provenance["cache"]),
        "d1": {
            "artifact": dict(provenance["d1_artifact"]),
            "manifest": dict(provenance["d1_manifest"]),
        },
        "d2": {
            "artifact": dict(provenance["d2_artifact"]),
            "manifest": dict(provenance["d2_manifest"]),
        },
        "s7": {
            "bundle": dict(provenance["s7_bundle"]),
            "manifest": dict(provenance["s7_manifest"]),
            "result": dict(provenance["s7_result"]),
        },
        "artifact": {
            "path": _relative(artifact_path, root),
            **_signature(artifact_path),
        },
        "bundle": (
            {
                "path": _relative(bundle_path, root),
                **_signature(bundle_path),
            }
            if bundle_path is not None
            else None
        ),
        "sealed_access": {name: False for name in SEALED_PARTITIONS},
        "model_spec": dict(result["model_spec"]),
        "reproduction_gate_passed": bool(result["reproduction_gate"]["passed"]),
        "failed_checks": list(result["reproduction_gate"]["failed_checks"]),
        "candidate_id": config.candidate_id,
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest


def validate_v2_manifest(
    manifest_path: str | Path, artifact_path: str | Path
) -> dict[str, Any]:
    """Validate a published V2 package manifest, artifact, and bundle.

    Every recorded hash is recomputed, both schema versions are checked, the
    artifact's privacy boundary is re-enforced, and the frozen flag, outcome,
    and bundle presence are cross-checked against the stored reproduction
    gate. A frozen manifest must record a bundle; a mismatch manifest must
    record none.

    Args:
        manifest_path: Manifest JSON path.
        artifact_path: Aggregate package result JSON path.

    Returns:
        The validated manifest payload.

    Raises:
        ValueError: If a hash, schema, or cross-check invariant fails.
    """

    manifest_file = Path(manifest_path).expanduser().resolve()
    artifact_file = Path(artifact_path).expanduser().resolve()
    manifest = _read_json(manifest_file)
    if manifest.get("schema_version") != V2_PACKAGE_MANIFEST_SCHEMA:
        raise ValueError("Unexpected V2 package manifest schema")
    if (
        manifest.get("complete") is not True
        or manifest.get("diagnostic_only") is not False
    ):
        raise ValueError("V2 package manifest is not a complete full result")
    if any(manifest.get("sealed_access", {}).values()):
        raise ValueError("V2 package manifest claims sealed-partition access")
    if manifest.get("deployment_authorized") is not False:
        raise ValueError("V2 package manifest cannot authorize deployment")
    root = manifest_file.parent.parent.resolve()
    for role in ("frozen_config", "protocol", "cache", "artifact"):
        _require_signature(root, manifest.get(role), role)
    for group in ("d1", "d2"):
        for role in ("artifact", "manifest"):
            _require_signature(
                root, manifest.get(group, {}).get(role), f"{group}.{role}"
            )
    for role, path in zip(("bundle", "manifest", "result"), _s7_paths(root)):
        item = manifest.get("s7", {}).get(role, {})
        if _safe_path(root, str(item.get("path"))) != path.resolve():
            raise ValueError(f"V2 package manifest S7 {role} path is invalid")
        _require_signature(root, item, f"s7.{role}")
    if _safe_path(root, str(manifest["artifact"].get("path"))) != artifact_file:
        raise ValueError("V2 package manifest artifact path differs from result")
    result = _read_json(artifact_file)
    if result.get("schema_version") != V2_PACKAGE_RESULT_SCHEMA:
        raise ValueError("V2 package result schema is invalid")
    if result.get("code_schema") != V2_PACKAGE_CODE_SCHEMA:
        raise ValueError("V2 package result code schema is stale")
    if result.get("complete") is not True:
        raise ValueError("V2 package result is incomplete")
    if result.get("signature") != manifest.get("signature"):
        raise ValueError("V2 package result signature differs from manifest")
    _validate_result_privacy(result)
    _validate_manifest_outcome(manifest, result, root)
    return manifest


def _require_signature(
    root: Path, item: Any, role: str
) -> None:
    """Recompute and compare one recorded path, hash, and size.

    Args:
        root: Project root every recorded path is relative to.
        item: The recorded ``{"path", "sha256", "size_bytes"}`` mapping.
        role: Role name used in error messages.

    Raises:
        ValueError: If the record is absent or the file has drifted.
    """

    if not isinstance(item, Mapping):
        raise ValueError(f"V2 package manifest {role} metadata is missing")
    path = _safe_path(root, str(item.get("path")))
    if (
        item.get("sha256") != _sha256(path)
        or item.get("size_bytes") != path.stat().st_size
    ):
        raise ValueError(f"V2 package manifest {role} hash is invalid")


def _validate_manifest_outcome(
    manifest: Mapping[str, Any], result: Mapping[str, Any], root: Path
) -> None:
    """Cross-check the outcome, frozen flag, bundle, and gate consistency.

    Args:
        manifest: The published manifest payload.
        result: The published aggregate result.
        root: Project root every recorded path is relative to.

    Raises:
        ValueError: If any published outcome invariant is inconsistent.
    """

    gate = result.get("reproduction_gate")
    if not isinstance(gate, Mapping):
        raise ValueError("V2 package result carries no reproduction gate")
    checks = gate.get("checks")
    if not isinstance(checks, Mapping) or not checks:
        raise ValueError("V2 package reproduction gate has no checks")
    missing = [name for name in _REQUIRED_GATE_CHECKS if name not in checks]
    if missing:
        raise ValueError("V2 package reproduction gate is missing checks")
    passed = all(bool(value) for value in checks.values())
    if gate.get("passed") is not passed:
        raise ValueError("V2 package reproduction gate flag is inconsistent")
    expected_outcome = OUTCOME_FROZEN if passed else OUTCOME_MISMATCH
    if result.get("outcome") != expected_outcome:
        raise ValueError("V2 package outcome disagrees with the gate")
    if result.get("frozen") is not passed:
        raise ValueError("V2 package frozen flag disagrees with the gate")
    if manifest.get("outcome") != result.get("outcome"):
        raise ValueError("V2 package manifest outcome differs from result")
    if manifest.get("frozen") is not bool(result.get("frozen")):
        raise ValueError("V2 package manifest frozen flag differs from result")
    if manifest.get("reproduction_gate_passed") is not passed:
        raise ValueError("V2 package manifest gate flag differs from result")
    bundle_metadata = manifest.get("bundle")
    if passed:
        if not isinstance(bundle_metadata, Mapping):
            raise ValueError("A frozen V2 package manifest must record a bundle")
        _require_signature(root, bundle_metadata, "bundle")
        recorded = result.get("bundle", {})
        if recorded.get("sha256") != bundle_metadata.get("sha256"):
            raise ValueError("V2 package result bundle hash differs from manifest")
    else:
        if bundle_metadata is not None:
            raise ValueError("A mismatched V2 package must not record a bundle")
        if result.get("bundle", {}).get("persisted") is not False:
            raise ValueError("A mismatched V2 package must not persist a bundle")
    model_spec = manifest.get("model_spec", {})
    if model_spec.get("model_version") != MODEL_VERSION:
        raise ValueError("V2 package manifest model version is invalid")
    if model_spec.get("input_language") != INPUT_LANGUAGE:
        raise ValueError("V2 package manifest input language is invalid")
    threshold = float(result["calibration"]["threshold"])
    if float(model_spec.get("threshold")) != threshold:
        raise ValueError("V2 package manifest threshold differs from result")


def load_v2_predictor(
    project_root: str | Path | None = None,
) -> V2Predictor:
    """Load the frozen V2 package only after validating its public manifest.

    Args:
        project_root: Project root holding the manifest, artifact, bundle,
            and frozen S7 package. Defaults to the installed package
            location, never the working directory.

    Returns:
        A validated hierarchical predictor suitable for the neutral contract.

    Raises:
        ValueError: If the manifest, hashes, outcome, or bundle are invalid.
    """

    root = _resolve_root(project_root)
    manifest_file = _safe_path(root, DEFAULT_PACKAGE_MANIFEST)
    manifest = _read_json(manifest_file)
    if manifest.get("outcome") != OUTCOME_FROZEN:
        raise ValueError("V2 package is not frozen and cannot be loaded")
    artifact_file = _safe_path(root, str(manifest["artifact"]["path"]))
    validate_v2_manifest(manifest_file, artifact_file)
    bundle_file = _safe_path(root, str(manifest["bundle"]["path"]))
    bundle = joblib.load(bundle_file)
    if not isinstance(bundle, V2ModelBundle):
        raise ValueError("V2 joblib does not contain a V2ModelBundle")
    bundle.validate()
    model_spec = manifest.get("model_spec", {})
    if model_spec.get("model_version") != bundle.model_version:
        raise ValueError("V2 model version differs from manifest")
    if model_spec.get("input_language") != bundle.input_language:
        raise ValueError("V2 input language differs from manifest")
    if float(model_spec.get("threshold")) != float(bundle.threshold):
        raise ValueError("V2 threshold differs from manifest")
    s7_bundle, s7_manifest, s7_result = _s7_paths(root)
    fallback = load_s7_predictor(s7_bundle, s7_manifest, s7_result)
    return V2Predictor(bundle, fallback)


def _run(
    mode: str,
    *,
    project_root: str | Path | None,
    config_path: str | Path | None,
    protocol_path: str | Path | None,
    artifact_path: str | Path | None,
    bundle_path: str | Path | None,
    manifest_path: str | Path | None,
    batch_size: int | None,
) -> dict[str, Any]:
    """Execute either the diagnostic smoke flow or the full freeze flow.

    Args:
        mode: ``full`` or ``smoke``.
        project_root: Project root for every relative path.
        config_path: Frozen package config path override.
        protocol_path: Frozen V2 protocol path override.
        artifact_path: Aggregate result path override.
        bundle_path: Persisted joblib path override.
        manifest_path: Public manifest path override.
        batch_size: Bounded outer evaluation batch size override.

    Returns:
        The aggregate-only package result.

    Raises:
        ValueError: If the mode, batch size, boundary, or hashes are invalid.
    """

    if mode not in {"full", "smoke"}:
        raise ValueError("mode must be full or smoke")
    root = _resolve_root(project_root)
    config = load_v2_package_config(
        config_path if config_path is not None else DEFAULT_PACKAGE_CONFIG,
        project_root=root,
    )
    protocol_file = _resolve_path(
        root, protocol_path, str(config.payload["protocol"]["path"])
    )
    protocol = load_v2_protocol(protocol_file)
    resolved_batch_size = int(
        batch_size if batch_size is not None else config.batch_size
    )
    if resolved_batch_size <= 0:
        raise ValueError("batch_size must be positive")
    opened_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    provenance = verify_package_provenance(config, project_root=root)
    d1_record = _read_d1_record(config, project_root=root)
    s7_bundle, s7_manifest, s7_result = _s7_paths(root)
    started = time.perf_counter()
    predictor = load_s7_predictor(s7_bundle, s7_manifest, s7_result)
    fallback_model_version = str(
        getattr(predictor, "model_version", "unknown")
    )
    if mode == "smoke":
        return _smoke_result(
            config,
            protocol,
            provenance,
            d1_record,
            fallback_model_version,
            opened_at=opened_at,
            runtime_seconds=time.perf_counter() - started,
        )
    scopes = _read_development_cache(
        _safe_path(root, BENCHMARK_DEFAULT_CACHE), protocol, resolved_batch_size
    )
    fallback = {
        role: _fallback_labels(predictor, scopes[role], resolved_batch_size)
        for role in ("inner_calibration", "outer_evaluation")
    }
    observed, bundle = _refit_selected_candidate(
        scopes, fallback, protocol, config, batch_size=resolved_batch_size
    )
    gate = _evaluate_reproduction_gate(observed, d1_record, config)
    artifact_file = _resolve_path(root, artifact_path, DEFAULT_PACKAGE_ARTIFACT)
    bundle_file = _resolve_path(root, bundle_path, DEFAULT_PACKAGE_BUNDLE)
    manifest_file = _resolve_path(root, manifest_path, DEFAULT_PACKAGE_MANIFEST)
    signature = _package_signature(config, provenance)
    base = _base_result(
        config,
        protocol,
        run_mode="full",
        diagnostic_only=False,
        opened_at=opened_at,
        signature=signature,
    )
    if gate["passed"]:
        bundle.validate()
        if float(bundle.threshold) != float(observed["calibration"]["threshold"]):
            raise ValueError("V2 bundle threshold differs from the calibration")
        _dump_joblib_atomic(bundle, bundle_file)
        bundle_metadata = {
            "persisted": True,
            "path": _relative(bundle_file, root),
            **_signature(bundle_file),
        }
        persisted_bundle: Path | None = bundle_file
    else:
        bundle_metadata = {
            "persisted": False,
            "reason": OUTCOME_MISMATCH,
            "failed_checks": list(gate["failed_checks"]),
        }
        persisted_bundle = None
    result = _complete_result(
        base,
        observed,
        gate,
        provenance,
        bundle_metadata,
        fallback_model_version,
        time.perf_counter() - started,
    )
    _write_json_atomic(artifact_file, result)
    _publish_package_manifest(
        root, config, artifact_file, persisted_bundle, manifest_file, result
    )
    validate_v2_manifest(manifest_file, artifact_file)
    return result


def run_v2_package(
    mode: str,
    *,
    project_root: str | Path | None = None,
    batch_size: int | None = None,
    config_path: str | Path | None = None,
    protocol_path: str | Path | None = None,
    artifact_path: str | Path | None = None,
    bundle_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Freeze the V2 package, or run its diagnostic preflight.

    ``full`` verifies every pinned hash, reads the frozen S3 development
    cache, scores the frozen S7 fallback, rebuilds the deterministic
    hard-negative pool, refits the pinned D1 candidate, calibrates the
    threshold on inner calibration only, evaluates the outer window, and
    applies the exact reproduction gate. The joblib bundle is persisted only
    when every check passes; otherwise the outcome is
    ``REPRODUCTION_MISMATCH`` and the divergence is still published.

    ``smoke`` is diagnostic only. It validates the frozen configuration,
    verifies every pinned hash, confirms the D1 candidate record is
    locatable and parseable, and confirms the frozen S7 fallback loads. It
    fits nothing and writes nothing.

    Args:
        mode: ``full`` or ``smoke``.
        project_root: Project root for every relative path. Defaults to the
            installed package location, never the working directory.
        batch_size: Bounded outer evaluation batch size.
        config_path: Frozen package config path override.
        protocol_path: Frozen V2 protocol path override.
        artifact_path: Aggregate result path override.
        bundle_path: Persisted joblib path override.
        manifest_path: Public manifest path override.

    Returns:
        The aggregate-only package result.

    Raises:
        ValueError: If the mode, batch size, boundary, or hashes are invalid.
    """

    return _run(
        mode,
        project_root=project_root,
        config_path=config_path,
        protocol_path=protocol_path,
        artifact_path=artifact_path,
        bundle_path=bundle_path,
        manifest_path=manifest_path,
        batch_size=batch_size,
    )


def run_v2_package_smoke(
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run the diagnostic package preflight without fitting or writing.

    Args:
        project_root: Project root for every relative path. Forwarded
            unchanged so a staged kernel never resolves against its own
            working directory.

    Returns:
        The diagnostic-only package preflight result.
    """

    return run_v2_package("smoke", project_root=project_root)


__all__ = [
    "DEFAULT_PACKAGE_ARTIFACT",
    "DEFAULT_PACKAGE_BUNDLE",
    "DEFAULT_PACKAGE_CONFIG",
    "DEFAULT_PACKAGE_MANIFEST",
    "MODEL_VERSION",
    "OUTCOME_FROZEN",
    "OUTCOME_MISMATCH",
    "V2_BUNDLE_SCHEMA",
    "V2_PACKAGE_CODE_SCHEMA",
    "V2_PACKAGE_CONFIG_SCHEMA",
    "V2_PACKAGE_MANIFEST_SCHEMA",
    "V2_PACKAGE_RESULT_SCHEMA",
    "V2ModelBundle",
    "V2PackageConfig",
    "V2Predictor",
    "default_project_root",
    "load_v2_package_config",
    "load_v2_predictor",
    "run_v2_package",
    "run_v2_package_smoke",
    "validate_v2_manifest",
    "verify_package_provenance",
]
