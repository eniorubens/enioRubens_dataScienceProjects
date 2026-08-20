"""S7 frozen LinearSVC package and framework-neutral prediction boundary."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from .contracts import Prediction, PredictionBatch
from .s3 import read_scientific_frame
from .s6 import (
    CRITICAL_CLASS,
    DEFAULT_BATCH_SIZE,
    S6GateConfig,
    _indices_for_scope,
    _labels,
    _margin_predictions,
    _scores_in_family_order,
    _texts,
    search_thresholds_exact,
    validate_scientific_cache,
)
from .temporal_split import MODELED_FAMILIES


S7_SCHEMA_VERSION = "s7-frozen-package-v1"
S7_RESULT_SCHEMA = "s7-results-v1"
S7_MANIFEST_SCHEMA = "s7-results-manifest-v1"
S7_BUNDLE_SCHEMA = "s7-model-bundle-v1"
S7_CODE_SCHEMA = "s7-runtime-v2"
INPUT_LANGUAGE = "en-US"
MODEL_VERSION = "consumer-complaint-classifier-s7"
DEVELOPMENT_PARTITIONS = ("train", "validation")
SEALED_PARTITIONS = ("test", "stress", "monitor")
DEFAULT_CONFIG = "config/s7_frozen_package.json"
DEFAULT_RESULT = "temp/s7/s7_results.json"
DEFAULT_SMOKE_RESULT = "temp/s7/s7_smoke_results.json"
DEFAULT_BUNDLE = "artifacts/s7/consumer_complaint_classifier_s7.joblib"
@dataclass(frozen=True, slots=True)
class S7Config:
    """Represent the immutable S7 fit and calibration contract."""

    schema_version: str
    status: str
    approved_on: str
    scientific_cache: str
    fit_partition: str
    fit_scope: Mapping[str, str]
    calibration_scope: Mapping[str, str]
    sealed_partitions: tuple[str, ...]
    critical_class: str
    input_language: str
    representation: Mapping[str, Any]
    estimator: Mapping[str, Any]
    threshold_policy: Mapping[str, Any]
    gates: Mapping[str, float]
    random_state: int
    run_defaults: Mapping[str, Any]

    def validate(self) -> None:
        """Validate every frozen S7 value and partition boundary."""

        if self.schema_version != S7_SCHEMA_VERSION:
            raise ValueError("Unexpected S7 configuration schema")
        if self.status != "FROZEN_FOR_FINAL_FIT":
            raise ValueError("S7 configuration is not frozen for final fit")
        if self.approved_on != "2026-08-16":
            raise ValueError("Unexpected S7 approval date")
        if self.scientific_cache != "temp/s3/scientific.parquet":
            raise ValueError("S7 scientific cache path is invalid")
        if self.fit_partition != "train":
            raise ValueError("S7 fit partition must be train")
        if self.sealed_partitions != SEALED_PARTITIONS:
            raise ValueError("S7 sealed partition boundary is invalid")
        if self.critical_class != CRITICAL_CLASS:
            raise ValueError("S7 critical class is invalid")
        if self.input_language != INPUT_LANGUAGE:
            raise ValueError("S7 input language must be en-US")
        expected_fit = {
            "partition": "train",
            "start": "2023-08-01",
            "end": "2024-06-30",
        }
        expected_calibration = {
            "partition": "validation",
            "start": "2024-07-01",
            "end": "2024-12-31",
        }
        if dict(self.fit_scope) != expected_fit:
            raise ValueError("S7 fit scope is invalid")
        if dict(self.calibration_scope) != expected_calibration:
            raise ValueError("S7 calibration scope is invalid")
        expected_representation = {
            "analyzer": "word",
            "ngram_range": [1, 2],
            "max_features": 40000,
            "min_df": 2,
            "max_df": 0.98,
            "sublinear_tf": True,
            "dtype": "float32",
        }
        if dict(self.representation) != expected_representation:
            raise ValueError("S7 representation is not the frozen word TF-IDF")
        expected_estimator = {
            "name": "linear_svc_c_0_3_balanced",
            "class": "LinearSVC",
            "C": 0.3,
            "class_weight": "balanced",
            "tol": 0.0001,
            "max_iter": 5000,
            "dual": "auto",
            "random_state": 42,
        }
        if dict(self.estimator) != expected_estimator:
            raise ValueError("S7 estimator is not the frozen LinearSVC")
        expected_policy = {
            "method": "search_thresholds_exact",
            "score_rule": "critical_margin",
            "calibration_source": "validation_only",
            "no_refit_after_calibration": True,
        }
        if dict(self.threshold_policy) != expected_policy:
            raise ValueError("S7 threshold policy differs from the frozen contract")
        expected_gates = {
            "global_macro_f1_min": 0.69,
            "critical_f1_min": 0.2715,
            "critical_precision_min": 0.2,
        }
        if dict(self.gates) != expected_gates:
            raise ValueError("S7 gates differ from the frozen contract")
        if self.random_state != 42:
            raise ValueError("S7 random_state must be 42")
        defaults = dict(self.run_defaults)
        if set(defaults) != {"batch_size", "smoke_max_per_class"}:
            raise ValueError("S7 run_defaults are incomplete")
        if any(int(value) <= 0 for value in defaults.values()):
            raise ValueError("S7 run_defaults must be positive")


def _read_json(path: Path) -> dict[str, Any]:
    """Read one UTF-8 JSON object from disk."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def load_s7_config(path: str | Path) -> S7Config:
    """Load and validate the frozen S7 configuration.

    Args:
        path: Path to ``s7_frozen_package.json``.

    Returns:
        Validated S7 configuration.
    """

    payload = _read_json(Path(path).expanduser().resolve())
    config = S7Config(
        schema_version=str(payload["schema_version"]),
        status=str(payload["status"]),
        approved_on=str(payload["approved_on"]),
        scientific_cache=str(payload["scientific_cache"]),
        fit_partition=str(payload["fit_partition"]),
        fit_scope={str(k): str(v) for k, v in payload["fit_scope"].items()},
        calibration_scope={
            str(k): str(v) for k, v in payload["calibration_scope"].items()
        },
        sealed_partitions=tuple(str(x) for x in payload["sealed_partitions"]),
        critical_class=str(payload["critical_class"]),
        input_language=str(payload["input_language"]),
        representation=payload["representation"],
        estimator=payload["estimator"],
        threshold_policy=payload["threshold_policy"],
        gates={str(k): float(v) for k, v in payload["gates"].items()},
        random_state=int(payload["random_state"]),
        run_defaults=payload["run_defaults"],
    )
    config.validate()
    return config


def _sha256(path: Path) -> str:
    """Return the uppercase SHA256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _file_signature(path: Path) -> dict[str, Any]:
    """Return path, size, and digest metadata for one file."""

    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _project_relative(path: Path, project_root: Path) -> str:
    """Return a portable project-relative path when the file is inside it."""

    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.name


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one JSON payload atomically with real UTF-8 text."""

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


def _dump_joblib_atomic(bundle: S7ModelBundle, path: Path) -> None:
    """Persist one joblib bundle through a same-directory atomic replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        joblib.dump(bundle, temporary, compress=3)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _signature(
    cache: Path,
    config: Path,
    mode: str,
    max_per_class: int | None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> str:
    """Build a cache signature from inputs and execution mode."""

    content = {
        "code_schema": S7_CODE_SCHEMA,
        "cache_sha256": _sha256(cache),
        "config_sha256": _sha256(config),
        "mode": mode,
        "max_per_class": max_per_class,
        "batch_size": batch_size,
    }
    return hashlib.sha256(
        json.dumps(content, sort_keys=True).encode("utf-8")
    ).hexdigest().upper()


def _vectorizer(config: S7Config) -> TfidfVectorizer:
    """Build the frozen word TF-IDF vectorizer."""

    representation = config.representation
    return TfidfVectorizer(
        analyzer=str(representation["analyzer"]),
        ngram_range=tuple(int(x) for x in representation["ngram_range"]),
        max_features=int(representation["max_features"]),
        min_df=int(representation["min_df"]),
        max_df=float(representation["max_df"]),
        sublinear_tf=bool(representation["sublinear_tf"]),
        dtype=np.float32,
    )


def _estimator(config: S7Config) -> LinearSVC:
    """Build the one frozen LinearSVC estimator."""

    parameters = {
        key: value
        for key, value in config.estimator.items()
        if key not in {"name", "class"}
    }
    return LinearSVC(**parameters)


def _scope_indices(
    frame: Any,
    scope: Mapping[str, str],
    max_per_class: int | None,
    random_state: int,
) -> list[int]:
    """Resolve one approved S7 date scope without opening sealed data."""

    return _indices_for_scope(
        frame,
        scope["partition"],
        scope["start"],
        scope["end"],
        max_per_class,
        random_state,
    )


@dataclass(frozen=True, slots=True)
class S7ModelBundle:
    """Own the fitted estimator and the exact S7 serving invariants."""

    vectorizer: TfidfVectorizer
    estimator: LinearSVC
    threshold: float
    classes: tuple[str, ...]
    critical_class: str
    model_version: str
    input_language: str
    schema_version: str = S7_BUNDLE_SCHEMA

    def validate(self) -> None:
        """Validate serialized vectorizer, estimator, classes, and language."""

        if self.schema_version != S7_BUNDLE_SCHEMA:
            raise ValueError("Unexpected S7 bundle schema")
        if self.input_language != INPUT_LANGUAGE:
            raise ValueError("S7 bundle input language must be en-US")
        if self.model_version != MODEL_VERSION:
            raise ValueError("Unexpected S7 model version")
        if self.critical_class != CRITICAL_CLASS:
            raise ValueError("S7 bundle critical class is invalid")
        if self.classes != tuple(MODELED_FAMILIES):
            raise ValueError("S7 bundle class order is invalid")
        estimator_classes = tuple(str(value) for value in self.estimator.classes_)
        if estimator_classes != self.classes:
            raise ValueError("S7 estimator class order differs from bundle")
        if not np.isfinite(self.threshold):
            raise ValueError("S7 threshold must be finite")
        params = self.vectorizer.get_params()
        expected = {
            "analyzer": "word",
            "ngram_range": (1, 2),
            "max_features": 40000,
            "min_df": 2,
            "max_df": 0.98,
            "sublinear_tf": True,
            "dtype": np.float32,
        }
        for key, value in expected.items():
            if params.get(key) != value:
                raise ValueError(f"S7 vectorizer invariant failed: {key}")
        estimator_params = self.estimator.get_params()
        expected_estimator = {
            "C": 0.3,
            "class_weight": "balanced",
            "tol": 0.0001,
            "max_iter": 5000,
            "dual": "auto",
            "random_state": 42,
        }
        for key, value in expected_estimator.items():
            if estimator_params.get(key) != value:
                raise ValueError(f"S7 estimator invariant failed: {key}")


class S7Predictor:
    """Apply the frozen S7 bundle through the framework-neutral contract."""

    def __init__(self, bundle: S7ModelBundle) -> None:
        """Initialize a predictor after validating its bundle."""

        bundle.validate()
        self._bundle = bundle

    @property
    def model_version(self) -> str:
        """Return the immutable model version exposed by the bundle."""

        return self._bundle.model_version

    @property
    def input_language(self) -> str:
        """Return the required language contract for incoming narratives."""

        return self._bundle.input_language

    def predict(
        self,
        texts: Sequence[str],
        *,
        input_language: str = INPUT_LANGUAGE,
    ) -> PredictionBatch:
        """Predict an ordered batch using the critical-margin override.

        Args:
            texts: Non-empty English complaint narratives.
            input_language: Language declared by the caller.

        Returns:
            Predictions with ``score_kind=critical_margin`` metadata.

        Raises:
            ValueError: If the batch, text, or language contract is invalid.
        """

        if input_language != self._bundle.input_language:
            raise ValueError("S7 predictor accepts input_language=en-US only")
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
        matrix = self._bundle.vectorizer.transform(values)
        scores = _scores_in_family_order(
            self._bundle.estimator,
            self._bundle.estimator.decision_function(matrix),
        )
        labels, margins = _margin_predictions(scores, self._bundle.threshold)
        predictions = tuple(
            Prediction(
                label=str(label),
                score=float(margin),
                model_version=self._bundle.model_version,
                metadata={
                    "score_kind": "critical_margin",
                    "threshold": float(self._bundle.threshold),
                    "input_language": self._bundle.input_language,
                },
            )
            for label, margin in zip(labels, margins)
        )
        return PredictionBatch(predictions=predictions)


def _fit_and_calibrate(
    frame: Any,
    config: S7Config,
    fit_indices: Sequence[int],
    calibration_indices: Sequence[int],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[S7ModelBundle, dict[str, Any]]:
    """Fit on train and calibrate the threshold once on validation."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    vectorizer = _vectorizer(config)
    x_fit = vectorizer.fit_transform(_texts(frame, fit_indices))
    estimator = _estimator(config)
    estimator.fit(x_fit, _labels(frame, fit_indices))
    del x_fit
    gc.collect()
    score_batches = []
    for start in range(0, len(calibration_indices), batch_size):
        batch = calibration_indices[start : start + batch_size]
        x_calibration = vectorizer.transform(_texts(frame, batch))
        score_batches.append(
            _scores_in_family_order(
                estimator, estimator.decision_function(x_calibration)
            )
        )
        del x_calibration
    scores = np.concatenate(score_batches, axis=0)
    gates = S6GateConfig(
        float(config.gates["global_macro_f1_min"]),
        float(config.gates["critical_f1_min"]),
        float(config.gates["critical_precision_min"]),
    )
    calibration = search_thresholds_exact(
        _labels(frame, calibration_indices), scores, gates
    )
    selected = calibration["selected"]
    bundle = S7ModelBundle(
        vectorizer=vectorizer,
        estimator=estimator,
        threshold=float(selected["threshold"]),
        classes=tuple(MODELED_FAMILIES),
        critical_class=CRITICAL_CLASS,
        model_version=MODEL_VERSION,
        input_language=INPUT_LANGUAGE,
    )
    bundle.validate()
    return bundle, calibration


def _base_result(
    config: S7Config,
    signature: str,
    mode: str,
    fit_count: int,
    calibration_count: int,
) -> dict[str, Any]:
    """Create the development-only result envelope before fitting."""

    return {
        "schema_version": S7_RESULT_SCHEMA,
        "code_schema": S7_CODE_SCHEMA,
        "status": "RUNNING",
        "run_mode": mode,
        "complete": False,
        "signature": signature,
        "development_only": True,
        "deploy": False,
        "confirmatory": False,
        "sealed": False,
        "claim_boundary": "NO_TEST_STRESS_OR_MONITOR_ACCESS",
        "sealed_partitions": list(config.sealed_partitions),
        "scientific_cache": config.scientific_cache,
        "fit_partition": config.fit_partition,
        "fit_scope": dict(config.fit_scope),
        "calibration_scope": dict(config.calibration_scope),
        "fit_row_count": fit_count,
        "calibration_row_count": calibration_count,
        "model_version": MODEL_VERSION,
        "input_language": INPUT_LANGUAGE,
        "random_state": config.random_state,
        "gates": dict(config.gates),
        "threshold_policy": dict(config.threshold_policy),
        "run_defaults": dict(config.run_defaults),
        "validation_role": "FINAL_CALIBRATION_ONLY",
        "validation_independence": (
            "NOT_INDEPENDENT_EVIDENCE_AFTER_FINAL_CALIBRATION"
        ),
        "validation_reuse_note_pt_br": (
            "A validation foi reutilizada agora apenas para a calibração final "
            "do limiar e não é evidência independente."
        ),
        "validation_reuse_note_en_us": (
            "Validation was reused now only for final threshold calibration and "
            "is not independent evidence."
        ),
        "bundle": None,
        "calibration": None,
        "runtime_seconds": None,
    }


def _publish_manifest(
    result_path: Path,
    bundle_path: Path,
    config_path: Path,
    manifest_path: Path,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish a non-circular manifest after result and bundle are complete."""

    bundle_signature = _file_signature(bundle_path)
    result_signature = _file_signature(result_path)
    config_signature = _file_signature(config_path)
    project_root = config_path.parent.parent
    calibration = result["calibration"]
    selected = calibration["selected"]
    status = str(result["status"])
    if status == "packaged_for_confirmation":
        interpretation_pt_br = (
            "Pacote de desenvolvimento pronto para confirmação. A validation "
            "foi usada somente para calibrar o limiar final; portanto, não há "
            "evidência independente nesta etapa. Test, stress e monitor seguem "
            "selados."
        )
    else:
        interpretation_pt_br = (
            "O limiar final não passou os três gates. O pacote permanece apenas "
            "em desenvolvimento e não avança para confirmação."
        )
    if status == "packaged_for_confirmation":
        interpretation_en_us = (
            "Development package ready for confirmation. Validation was used "
            "only to calibrate the final threshold, so this stage has no "
            "independent evidence. Test, stress, and monitor remain sealed."
        )
    else:
        interpretation_en_us = (
            "The final threshold did not pass all three gates. The package "
            "remains development-only and does not advance to confirmation."
        )
    manifest = {
        "schema_version": S7_MANIFEST_SCHEMA,
        "stage": "S7",
        "status": status,
        "development_only": True,
        "deploy": False,
        "confirmatory": False,
        "sealed": False,
        "claim_boundary": "NO_TEST_STRESS_OR_MONITOR_ACCESS",
        "sealed_partitions": list(SEALED_PARTITIONS),
        "approved_on": "2026-08-16",
        "frozen_config": {
            "path": _project_relative(config_path, project_root),
            "sha256": config_signature["sha256"],
            "size_bytes": config_signature["size_bytes"],
        },
        "bundle": {
            "path": _project_relative(bundle_path, project_root),
            "sha256": bundle_signature["sha256"],
            "size_bytes": bundle_signature["size_bytes"],
        },
        "result": {
            "path": _project_relative(result_path, project_root),
            "sha256": result_signature["sha256"],
            "size_bytes": result_signature["size_bytes"],
        },
        "model": {
            "model_version": MODEL_VERSION,
            "input_language": INPUT_LANGUAGE,
            "classes": list(MODELED_FAMILIES),
            "critical_class": CRITICAL_CLASS,
            "threshold": selected["threshold"],
            "score_kind": "critical_margin",
        },
        "fit_scope": result["fit_scope"],
        "calibration_scope": result["calibration_scope"],
        "validation_role": result["validation_role"],
        "validation_independence": result["validation_independence"],
        "validation_reuse_note_pt_br": result["validation_reuse_note_pt_br"],
        "validation_reuse_note_en_us": result["validation_reuse_note_en_us"],
        "runtime_seconds": result["runtime_seconds"],
        "calibration_gate_passed": bool(
            selected["gates"]["eligible"]
        ),
        "calibration": calibration,
        "interpretation_pt_br": interpretation_pt_br,
        "interpretation_en_us": interpretation_en_us,
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest


def validate_s7_manifest(
    manifest: Mapping[str, Any],
    bundle_path: str | Path,
    result_path: str | Path,
    config_path: str | Path | None = None,
) -> None:
    """Validate S7 status, hashes, and serving invariants before loading."""

    if manifest.get("schema_version") != S7_MANIFEST_SCHEMA:
        raise ValueError("Unexpected S7 manifest schema")
    if manifest.get("status") not in {
        "packaged_for_confirmation",
        "calibration_gate_failed",
    }:
        raise ValueError("S7 manifest status is invalid")
    for field in ("development_only", "sealed"):
        if manifest.get(field) is not (True if field == "development_only" else False):
            raise ValueError(f"Invalid S7 manifest flag: {field}")
    if manifest.get("deploy") is not False:
        raise ValueError("S7 manifest cannot be deployable")
    if manifest.get("confirmatory") is not False:
        raise ValueError("S7 manifest cannot be confirmatory")
    if tuple(manifest.get("sealed_partitions", ())) != SEALED_PARTITIONS:
        raise ValueError("S7 manifest sealed partition boundary is invalid")
    bundle_metadata = manifest.get("bundle")
    result_metadata = manifest.get("result")
    if not isinstance(bundle_metadata, Mapping):
        raise ValueError("S7 bundle metadata is missing")
    if not isinstance(result_metadata, Mapping):
        raise ValueError("S7 result metadata is missing")
    config_metadata = manifest.get("frozen_config")
    model_metadata = manifest.get("model")
    if not isinstance(config_metadata, Mapping):
        raise ValueError("S7 frozen config metadata is missing")
    if not isinstance(model_metadata, Mapping):
        raise ValueError("S7 model metadata is missing")
    bundle = Path(bundle_path).expanduser().resolve()
    result = Path(result_path).expanduser().resolve()
    config = (
        Path(config_path).expanduser().resolve()
        if config_path is not None
        else Path(str(config_metadata["path"])).expanduser().resolve()
    )
    if not bundle.exists() or not result.exists():
        raise ValueError("S7 bundle and result must both exist")
    if not config.exists():
        raise ValueError("S7 frozen config must exist")
    if _sha256(config) != str(config_metadata["sha256"]):
        raise ValueError("S7 frozen config hash does not match manifest")
    if config.stat().st_size != int(config_metadata["size_bytes"]):
        raise ValueError("S7 frozen config size does not match manifest")
    load_s7_config(config)
    if _sha256(bundle) != str(bundle_metadata["sha256"]):
        raise ValueError("S7 bundle hash does not match manifest")
    if bundle.stat().st_size != int(bundle_metadata["size_bytes"]):
        raise ValueError("S7 bundle size does not match manifest")
    if _sha256(result) != str(result_metadata["sha256"]):
        raise ValueError("S7 result hash does not match manifest")
    if result.stat().st_size != int(result_metadata["size_bytes"]):
        raise ValueError("S7 result size does not match manifest")
    result_payload = _read_json(result)
    if result_payload.get("schema_version") != S7_RESULT_SCHEMA:
        raise ValueError("S7 result schema is invalid")
    if result_payload.get("code_schema") != S7_CODE_SCHEMA:
        raise ValueError("S7 result code schema is stale")
    if result_payload.get("bundle", {}).get("sha256") != bundle_metadata["sha256"]:
        raise ValueError("S7 result bundle hash differs from manifest")
    if result_payload.get("complete") is not True:
        raise ValueError("S7 result is incomplete")
    if result_payload.get("status") != manifest.get("status"):
        raise ValueError("S7 result status differs from manifest")
    for field in ("development_only", "deploy", "confirmatory", "sealed"):
        if result_payload.get(field) != manifest.get(field):
            raise ValueError(f"S7 result flag differs from manifest: {field}")
    if result_payload.get("input_language") != INPUT_LANGUAGE:
        raise ValueError("S7 result input language is invalid")
    if tuple(result_payload.get("sealed_partitions", ())) != SEALED_PARTITIONS:
        raise ValueError("S7 result sealed partition boundary is invalid")
    if result_payload.get("model_version") != MODEL_VERSION:
        raise ValueError("S7 result model version is invalid")
    gate_passed = bool(
        result_payload["calibration"]["selected"]["gates"]["eligible"]
    )
    if result_payload.get("calibration_gate_passed") != gate_passed:
        raise ValueError("S7 calibration gate status is inconsistent")
    expected_status = (
        "packaged_for_confirmation" if gate_passed else "calibration_gate_failed"
    )
    if manifest.get("status") != expected_status:
        raise ValueError("S7 manifest status does not match calibration gates")
    if model_metadata.get("model_version") != MODEL_VERSION:
        raise ValueError("S7 model version metadata is invalid")
    if model_metadata.get("input_language") != INPUT_LANGUAGE:
        raise ValueError("S7 model language metadata is invalid")
    if model_metadata.get("score_kind") != "critical_margin":
        raise ValueError("S7 score kind metadata is invalid")
    if tuple(model_metadata.get("classes", ())) != tuple(MODELED_FAMILIES):
        raise ValueError("S7 model classes metadata is invalid")
    if model_metadata.get("critical_class") != CRITICAL_CLASS:
        raise ValueError("S7 model critical class metadata is invalid")
    threshold = result_payload["calibration"]["selected"]["threshold"]
    if float(model_metadata.get("threshold")) != float(threshold):
        raise ValueError("S7 model threshold metadata is invalid")


def load_s7_predictor(
    bundle_path: str | Path,
    manifest_path: str | Path,
    result_path: str | Path,
) -> S7Predictor:
    """Load a hashed S7 bundle only after validating its public manifest.

    Args:
        bundle_path: Persisted joblib bundle path.
        manifest_path: Public S7 manifest path.
        result_path: Scientific S7 result path referenced by the manifest.

    Returns:
        A validated predictor suitable for ``PredictionService`` or Flask.
    """

    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = _read_json(manifest_file)
    if manifest.get("status") != "packaged_for_confirmation":
        raise ValueError("S7 bundle is not eligible for serving")
    project_root = manifest_file.parent.parent
    config_file = project_root / manifest["frozen_config"]["path"]
    if not config_file.exists():
        config_file = Path.cwd() / manifest["frozen_config"]["path"]
    validate_s7_manifest(manifest, bundle_path, result_path, config_file)
    bundle = joblib.load(Path(bundle_path).expanduser().resolve())
    if not isinstance(bundle, S7ModelBundle):
        raise ValueError("S7 joblib does not contain an S7ModelBundle")
    bundle.validate()
    model = manifest.get("model", {})
    if model.get("model_version") != bundle.model_version:
        raise ValueError("S7 model version differs from manifest")
    if model.get("input_language") != bundle.input_language:
        raise ValueError("S7 input language differs from manifest")
    if tuple(model.get("classes", ())) != bundle.classes:
        raise ValueError("S7 class order differs from manifest")
    if float(model.get("threshold")) != bundle.threshold:
        raise ValueError("S7 threshold differs from manifest")
    return S7Predictor(bundle)


def _run(
    scientific_cache_path: str | Path,
    artifact_path: str | Path,
    config_path: str | Path,
    *,
    mode: str,
    max_per_class: int | None,
    bundle_path: str | Path,
    manifest_path: str | Path | None,
    batch_size: int,
) -> dict[str, Any]:
    """Execute either the full package flow or a diagnostic smoke flow."""

    cache = Path(scientific_cache_path).expanduser().resolve()
    artifact = Path(artifact_path).expanduser().resolve()
    config_file = Path(config_path).expanduser().resolve()
    bundle_file = Path(bundle_path).expanduser().resolve()
    config = load_s7_config(config_file)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    signature = _signature(cache, config_file, mode, max_per_class, batch_size)
    if artifact.exists():
        cached = _read_json(artifact)
        if cached.get("complete") is True and cached.get("signature") == signature:
            if mode == "full":
                bundle_meta = cached.get("bundle", {})
                if bundle_file.exists() and _sha256(bundle_file) == bundle_meta.get(
                    "sha256"
                ):
                    if manifest_path is None:
                        raise ValueError("Full S7 runs require a public manifest path")
                    manifest_file = Path(manifest_path).expanduser().resolve()
                    if manifest_file.exists():
                        manifest = _read_json(manifest_file)
                        try:
                            validate_s7_manifest(
                                manifest,
                                bundle_file,
                                artifact,
                                config_file,
                            )
                        except ValueError:
                            _publish_manifest(
                                artifact,
                                bundle_file,
                                config_file,
                                manifest_file,
                                cached,
                            )
                    else:
                        _publish_manifest(
                            artifact,
                            bundle_file,
                            config_file,
                            manifest_file,
                            cached,
                        )
                    return cached
            else:
                return cached
    frame = read_scientific_frame(cache)
    validate_scientific_cache(frame)
    fit_indices = _scope_indices(
        frame, config.fit_scope, max_per_class, config.random_state
    )
    calibration_indices = _scope_indices(
        frame, config.calibration_scope, max_per_class, config.random_state
    )
    if not fit_indices or not calibration_indices:
        raise ValueError("S7 fit and calibration scopes must contain rows")
    result = _base_result(
        config, signature, mode, len(fit_indices), len(calibration_indices)
    )
    _write_json_atomic(artifact, result)
    started = time.perf_counter()
    bundle, calibration = _fit_and_calibrate(
        frame, config, fit_indices, calibration_indices, batch_size
    )
    result["calibration"] = calibration
    result["runtime_seconds"] = float(time.perf_counter() - started)
    if mode == "smoke":
        result["status"] = "DIAGNOSTIC_ONLY"
        result["complete"] = True
        result["bundle"] = {
            "persisted": False,
            "final_bundle_untouched": True,
            "model_version": MODEL_VERSION,
        }
        _write_json_atomic(artifact, result)
        return result
    bundle_file.parent.mkdir(parents=True, exist_ok=True)
    _dump_joblib_atomic(bundle, bundle_file)
    bundle_signature = _file_signature(bundle_file)
    selected = calibration["selected"]
    result["status"] = (
        "packaged_for_confirmation"
        if selected["gates"]["eligible"]
        else "calibration_gate_failed"
    )
    result["calibration_gate_passed"] = bool(selected["gates"]["eligible"])
    result["complete"] = True
    result["bundle"] = {
        "persisted": True,
        "path": _project_relative(bundle_file, config_file.parent.parent),
        "sha256": bundle_signature["sha256"],
        "size_bytes": bundle_signature["size_bytes"],
    }
    _write_json_atomic(artifact, result)
    if manifest_path is None:
        raise ValueError("Full S7 runs require a public manifest path")
    _publish_manifest(
        artifact, bundle_file, config_file, Path(manifest_path).resolve(), result
    )
    return result


def run_s7(
    scientific_cache_path: str | Path,
    artifact_path: str | Path,
    config_path: str | Path,
    *,
    bundle_path: str | Path = DEFAULT_BUNDLE,
    manifest_path: str | Path = "config/s7_results.json",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Run the full S7 fit, final calibration, and bundle publication."""

    return _run(
        scientific_cache_path,
        artifact_path,
        config_path,
        mode="full",
        max_per_class=None,
        bundle_path=bundle_path,
        manifest_path=manifest_path,
        batch_size=batch_size,
    )


def run_s7_smoke(
    scientific_cache_path: str | Path,
    artifact_path: str | Path,
    config_path: str | Path,
    *,
    max_per_class: int = 8,
    bundle_path: str | Path = DEFAULT_BUNDLE,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Run diagnostic S7 smoke without persisting or replacing the final bundle."""

    return _run(
        scientific_cache_path,
        artifact_path,
        config_path,
        mode="smoke",
        max_per_class=max_per_class,
        bundle_path=bundle_path,
        manifest_path=None,
        batch_size=batch_size,
    )
