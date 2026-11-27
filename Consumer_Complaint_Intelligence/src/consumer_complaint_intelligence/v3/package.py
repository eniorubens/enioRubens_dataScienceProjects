"""Freeze the selected V3 detector behind an exact E3 reproduction gate."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np

from .e3_reporting import audit_v3_e3_results
from .protocol import load_v3_protocol
from .rolling_probes import (
    CRITICAL_CLASS,
    _binary_target,
    _candidate_estimator,
    _compare_with_s7_f3,
    _fold_indices,
    _load_development_data,
    _mine_oof_margins,
    _score_sparse,
    evaluate_probabilities,
    load_rolling_config,
    sample_fit_pool,
    select_threshold,
)


CONFIG_SCHEMA = "v3-frozen-package-config-v1"
RESULT_SCHEMA = "v3-package-result-v1"
MANIFEST_SCHEMA = "v3-package-results-manifest-v1"
BUNDLE_SCHEMA = "v3-critical-detector-bundle-v1"
RUNTIME_SCHEMA = "v3-critical-detector-runtime-v1"
DEFAULT_CONFIG = "config/v3/package.json"
OUTCOME_FROZEN = "PACKAGE_FROZEN"
OUTCOME_MISMATCH = "REPRODUCTION_MISMATCH"
SELECTED_CANDIDATE = "tfidf_word_char_ratio_1_10"
SEALED_PARTITIONS = ("test", "stress", "monitor")
_FORBIDDEN_RESULT_KEYS = {
    "ids",
    "indices",
    "texts",
    "narratives",
    "scores",
    "probabilities",
    "margins",
    "vectorizer",
    "estimator",
    "calibrator",
}


@dataclass
class FrozenV3Detector:
    """Portable fitted critical detector without fallback duplication."""

    schema_version: str
    model_version: str
    critical_class: str
    input_language: str
    vectorizer: Any
    estimator: Any
    calibrator: Any
    threshold: float
    selected_candidate_id: str
    source_result_sha256: str

    def predict_probabilities(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 4096,
    ) -> np.ndarray:
        """Return calibrated critical probabilities in bounded batches."""

        values = list(texts)
        if not values:
            return np.empty(0, dtype=np.float64)
        if batch_size <= 0:
            raise ValueError("V3 detector batch_size must be positive")
        if any(not isinstance(text, str) for text in values):
            raise TypeError("V3 detector inputs must be strings")
        output = np.empty(len(values), dtype=np.float64)
        for start in range(0, len(values), batch_size):
            end = min(start + batch_size, len(values))
            matrix = self.vectorizer.transform(values[start:end])
            margins = self.estimator.decision_function(matrix)
            output[start:end] = self.calibrator.predict_proba(
                np.asarray(margins, dtype=np.float64).reshape(-1, 1)
            )[:, 1]
        return output

    def predict_critical(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 4096,
    ) -> np.ndarray:
        """Return the frozen critical override decisions."""

        return self.predict_probabilities(
            texts,
            batch_size=batch_size,
        ) >= float(self.threshold)


def _canonical_detector_type() -> type[FrozenV3Detector]:
    """Return the importable runtime class even under ``python -m``."""

    if FrozenV3Detector.__module__ != "__main__":
        return FrozenV3Detector
    module = importlib.import_module(
        "consumer_complaint_intelligence.v3.package"
    )
    return module.FrozenV3Detector


def _read_json(path: Path) -> dict[str, Any]:
    """Read one UTF-8 JSON object."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read V3-P evidence: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("V3-P JSON evidence must be an object")
    return payload


def _sha256(path: Path) -> str:
    """Return an uppercase SHA-256 for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _signature(path: Path) -> dict[str, Any]:
    """Return one portable file signature."""

    if not path.is_file():
        raise FileNotFoundError(path)
    return {"sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _safe_path(root: Path, relative: str) -> Path:
    """Resolve a project-relative path without allowing escape."""

    candidate = Path(relative)
    if candidate.is_absolute() or candidate.drive or ".." in candidate.parts:
        raise ValueError("V3-P paths must be project-relative")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("V3-P path escapes the project root")
    return resolved


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write aggregate JSON atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_joblib(path: Path, value: Any) -> None:
    """Write one fitted bundle atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        joblib.dump(value, temporary, compress=3)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_privacy(payload: Mapping[str, Any]) -> None:
    """Reject row-level evidence from public V3-P JSON outputs."""

    def visit(value: Any, location: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                normalized = str(key).lower()
                if normalized in _FORBIDDEN_RESULT_KEYS:
                    raise ValueError(f"Private V3-P field at {location}.{key}")
                visit(child, f"{location}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{location}[{index}]")

    visit(payload, "result")


def validate_package_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact frozen V3-P execution contract."""

    block = json.loads(json.dumps(payload))
    if block.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("V3-P config schema is invalid")
    if block.get("stage") != "V3-P":
        raise ValueError("V3-P config stage is invalid")
    if block.get("status") != "FROZEN_FOR_V3_PACKAGE_EXECUTION":
        raise ValueError("V3-P config is not frozen for execution")
    if block.get("candidate", {}).get("candidate_id") != SELECTED_CANDIDATE:
        raise ValueError("V3-P selected candidate differs")
    if block.get("candidate", {}).get("negative_ratio") != 10:
        raise ValueError("V3-P negative ratio differs")
    sample = block.get("sample", {})
    expected_sample = {
        "positive_groups": 946,
        "hard_negative_groups": 6622,
        "background_negative_groups": 2838,
        "negative_groups": 9460,
        "row_count": 10406,
        "signature": (
            "C20F9823FB5C55D6CAC4AC8243CD5FC3AE7B6DD45F77048F60A7770F768D6138"
        ),
        "persisted": False,
    }
    if sample != expected_sample:
        raise ValueError("V3-P sample contract differs")
    gate = block.get("reproduction_gate", {})
    if gate.get("required") is not True:
        raise ValueError("V3-P reproduction gate must be required")
    if gate.get("comparison") != "exact_no_tolerance":
        raise ValueError("V3-P reproduction comparison differs")
    if gate.get("publishes_bundle_on_mismatch") is not False:
        raise ValueError("V3-P may not publish a mismatched bundle")
    expected = gate.get("expected", {})
    threshold = expected.get("calibration", {}).get("threshold")
    if threshold != 0.10162120744876312:
        raise ValueError("V3-P threshold differs")
    boundary = block.get("boundary", {})
    if tuple(boundary.get("sealed_partitions", ())) != SEALED_PARTITIONS:
        raise ValueError("V3-P sealed partitions differ")
    if boundary.get("no_sealed_partition_unlock_in_package_code") is not True:
        raise ValueError("V3-P sealed partition lock differs")
    if block.get("deployment", {}).get("deployment_authorized") is not False:
        raise ValueError("V3-P may not authorize deployment")
    if block.get("deployment", {}).get("confirmation_claim_authorized") is not False:
        raise ValueError("V3-P may not claim confirmation")
    return block


def load_package_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load and validate the V3-P config."""

    return validate_package_config(_read_json(Path(path)))


def _verify_provenance(root: Path, config: Mapping[str, Any]) -> None:
    """Require every frozen input to match its recorded signature."""

    for name, evidence in config["provenance"].items():
        path = _safe_path(root, str(evidence["path"]))
        observed = _signature(path)
        expected = {
            "sha256": evidence["sha256"],
            "size_bytes": evidence["size_bytes"],
        }
        if observed != expected:
            raise ValueError(f"V3-P provenance differs: {name}")


def _fit_frozen_detector(
    data: Any,
    sample_indices: np.ndarray,
    calibration_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    *,
    batch_rows: int,
    source_result_sha256: str,
) -> tuple[FrozenV3Detector, dict[str, Any], dict[str, Any], np.ndarray]:
    """Fit the selected candidate and reproduce its future evidence."""

    from sklearn.linear_model import LogisticRegression

    from consumer_complaint_intelligence.v2_detector import (
        WORD_CHAR_TFIDF_ALIAS,
        build_vectorizer,
    )

    target = _binary_target(data.labels[sample_indices])
    vectorizer = build_vectorizer(WORD_CHAR_TFIDF_ALIAS)
    fit_matrix = vectorizer.fit_transform(data.texts[sample_indices].tolist())
    estimator = _candidate_estimator(balanced=False)
    estimator.fit(fit_matrix, target)
    del fit_matrix
    gc.collect()
    calibration_margins = _score_sparse(
        vectorizer,
        estimator,
        data.texts[calibration_indices],
        batch_rows,
    )
    evaluation_margins = _score_sparse(
        vectorizer,
        estimator,
        data.texts[evaluation_indices],
        batch_rows,
    )
    calibration_target = _binary_target(data.labels[calibration_indices])
    evaluation_target = _binary_target(data.labels[evaluation_indices])
    calibrator = LogisticRegression(
        C=1_000_000.0,
        solver="lbfgs",
        class_weight=None,
        max_iter=1000,
        random_state=42,
    )
    calibrator.fit(calibration_margins.reshape(-1, 1), calibration_target)
    calibration_probabilities = calibrator.predict_proba(
        calibration_margins.reshape(-1, 1)
    )[:, 1]
    evaluation_probabilities = calibrator.predict_proba(
        evaluation_margins.reshape(-1, 1)
    )[:, 1]
    threshold = select_threshold(calibration_target, calibration_probabilities)
    evaluation = evaluate_probabilities(
        evaluation_target,
        evaluation_probabilities,
        threshold["threshold"],
    )
    decisions = evaluation_probabilities >= threshold["threshold"]
    detector_type = _canonical_detector_type()
    detector = detector_type(
        schema_version=RUNTIME_SCHEMA,
        model_version="consumer-complaint-detector-v3",
        critical_class=CRITICAL_CLASS,
        input_language="en-US",
        vectorizer=vectorizer,
        estimator=estimator,
        calibrator=calibrator,
        threshold=float(threshold["threshold"]),
        selected_candidate_id=SELECTED_CANDIDATE,
        source_result_sha256=source_result_sha256,
    )
    del (
        calibration_margins,
        evaluation_margins,
        calibration_probabilities,
        evaluation_probabilities,
    )
    gc.collect()
    return detector, threshold, evaluation, decisions


def _exact_check(name: str, observed: Any, expected: Any) -> dict[str, Any]:
    """Return one exact reproduction check without publishing private data."""

    return {
        "name": name,
        "passed": observed == expected,
        "observed": observed,
        "expected": expected,
    }


def _evaluate_reproduction(
    config: Mapping[str, Any],
    audit: Mapping[str, Any],
    sampling: Mapping[str, Any],
    calibration: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    system_comparison: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate all public exact checks guarding bundle publication."""

    expected_sample = config["sample"]
    observed_counts = {
        "positive_groups": sampling["positive_groups"],
        "hard_negative_groups": sampling["hard_negative_groups"],
        "background_negative_groups": sampling["background_negative_groups"],
        "negative_groups": sampling["negative_groups"],
        "row_count": sampling["sample_rows"],
    }
    expected_counts = {
        key: expected_sample[key]
        for key in (
            "positive_groups",
            "hard_negative_groups",
            "background_negative_groups",
            "negative_groups",
            "row_count",
        )
    }
    expected = config["reproduction_gate"]["expected"]
    checks = [
        _exact_check(
            "selected_candidate",
            audit["selected_candidate_id"],
            SELECTED_CANDIDATE,
        ),
        _exact_check("sample_counts", observed_counts, expected_counts),
        _exact_check(
            "sample_signature",
            sampling["sample_signature"],
            expected_sample["signature"],
        ),
        _exact_check(
            "calibration_block",
            dict(calibration),
            expected["calibration"],
        ),
        _exact_check(
            "evaluation_block",
            dict(evaluation),
            expected["evaluation"],
        ),
        _exact_check(
            "f3_system_comparison",
            dict(system_comparison),
            expected["f3_system_comparison"],
        ),
    ]
    return {
        "comparison": "exact_no_tolerance",
        "required_check_count": len(checks),
        "passed_check_count": sum(check["passed"] for check in checks),
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }


def load_v3_detector(
    project_root: str | Path = ".",
    *,
    bundle_path: str = "artifacts/v3/consumer_complaint_detector_v3.joblib",
    manifest_path: str = "config/v3/package_results_manifest.json",
) -> FrozenV3Detector:
    """Load and validate the frozen V3 critical detector."""

    root = Path(project_root).expanduser().resolve()
    bundle_file = _safe_path(root, bundle_path)
    manifest = _read_json(_safe_path(root, manifest_path))
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("V3-P package manifest schema is invalid")
    if manifest.get("outcome") != OUTCOME_FROZEN:
        raise ValueError("V3-P package is not frozen")
    if manifest.get("bundle", {}).get("sha256") != _sha256(bundle_file):
        raise ValueError("V3-P bundle hash differs")
    if manifest.get("bundle", {}).get("size_bytes") != bundle_file.stat().st_size:
        raise ValueError("V3-P bundle size differs")
    detector = joblib.load(bundle_file)
    if not isinstance(detector, FrozenV3Detector):
        raise ValueError("V3-P bundle type is invalid")
    if detector.schema_version != RUNTIME_SCHEMA:
        raise ValueError("V3-P runtime schema differs")
    if detector.selected_candidate_id != SELECTED_CANDIDATE:
        raise ValueError("V3-P runtime candidate differs")
    return detector


def run_v3_package(
    project_root: str | Path = ".",
    *,
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Reproduce V3-E3 F3 and freeze the selected fitted detector."""

    root = Path(project_root).expanduser().resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config = load_package_config(config_file)
    _verify_provenance(root, config)
    outputs = config["outputs"]
    bundle_path = _safe_path(root, outputs["bundle"])
    result_path = _safe_path(root, outputs["result"])
    manifest_path = _safe_path(root, outputs["manifest"])
    if any(path.exists() for path in (bundle_path, result_path, manifest_path)):
        raise FileExistsError("V3-P output already exists; overwrite is forbidden")

    started = time.perf_counter()
    audit = audit_v3_e3_results(root)
    rolling_config = load_rolling_config(
        _safe_path(root, config["provenance"]["rolling_config"]["path"])
    )
    data = _load_development_data(root, rolling_config)
    protocol = load_v3_protocol(
        _safe_path(root, config["provenance"]["development_protocol"]["path"])
    )
    f3 = next(
        fold
        for fold in protocol.payload["rolling_origin"]["folds"]
        if fold["fold_id"] == "F3"
    )
    indices = _fold_indices(data, f3)
    print(
        "V3-P regenerating F3 OOF hard-negative evidence; "
        "sealed partitions remain inaccessible",
        flush=True,
    )
    oof_margins = _mine_oof_margins(data, indices["fit"], "F3-P")
    sample, sampling = sample_fit_pool(
        data,
        indices["fit"],
        oof_margins,
        fold_id="F3",
        ratio=10,
        hard_per_positive=7,
        background_per_positive=3,
    )
    del oof_margins
    gc.collect()
    detector, calibration, evaluation, decisions = _fit_frozen_detector(
        data,
        sample,
        indices["calibration"],
        indices["evaluation"],
        batch_rows=int(config["run_defaults"]["transform_batch_rows"]),
        source_result_sha256=config["provenance"]["e3_result"]["sha256"],
    )
    system_comparison = _compare_with_s7_f3(
        root,
        data,
        indices["evaluation"],
        decisions,
        int(config["run_defaults"]["s7_prediction_batch_rows"]),
    )
    gate = _evaluate_reproduction(
        config,
        audit,
        sampling,
        calibration,
        evaluation,
        system_comparison,
    )
    frozen = bool(gate["passed"])
    outcome = OUTCOME_FROZEN if frozen else OUTCOME_MISMATCH
    bundle_signature = None
    if frozen:
        _atomic_joblib(bundle_path, detector)
        bundle_signature = {
            "path": outputs["bundle"],
            **_signature(bundle_path),
        }
    result = {
        "schema_version": RESULT_SCHEMA,
        "stage": "V3-P",
        "status": "FROZEN_FOR_CONFIRMATION" if frozen else outcome,
        "outcome": outcome,
        "complete": True,
        "development_only": True,
        "confirmatory": False,
        "deploy": False,
        "selected_candidate_id": SELECTED_CANDIDATE,
        "model_version": config["model_version"],
        "sample": {
            "positive_groups": sampling["positive_groups"],
            "hard_negative_groups": sampling["hard_negative_groups"],
            "background_negative_groups": sampling[
                "background_negative_groups"
            ],
            "negative_groups": sampling["negative_groups"],
            "row_count": sampling["sample_rows"],
            "sample_signature": sampling["sample_signature"],
            "persisted": False,
        },
        "calibration": calibration,
        "evaluation": evaluation,
        "f3_system_comparison": system_comparison,
        "reproduction_gate": gate,
        "bundle": bundle_signature,
        "config": {
            "path": config_file.relative_to(root).as_posix(),
            **_signature(config_file),
        },
        "runtime_seconds": float(time.perf_counter() - started),
        "boundary": {
            "allowed_partitions": ["train", "validation"],
            "sealed_partitions_accessed": False,
            "narratives_or_identifiers_persisted": False,
            "row_indices_persisted": False,
            "individual_scores_persisted": False,
            "deployment_authorized": False,
            "confirmation_claim_authorized": False,
        },
    }
    _validate_privacy(result)
    _atomic_json(result_path, result)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "stage": "V3-P",
        "status": result["status"],
        "outcome": outcome,
        "complete": True,
        "development_only": True,
        "confirmatory": False,
        "deploy": False,
        "selected_candidate_id": SELECTED_CANDIDATE,
        "reproduction_gate_passed": frozen,
        "result": {
            "path": outputs["result"],
            **_signature(result_path),
        },
        "bundle": bundle_signature,
        "config": result["config"],
        "sealed_partitions_accessed": False,
        "confirmation_claim_authorized": False,
        "next_step_authorized": False,
    }
    _validate_privacy(manifest)
    _atomic_json(manifest_path, manifest)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    """Run V3-P only through the explicit ``--run`` gate."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    arguments = parser.parse_args(argv)
    if not arguments.run:
        raise SystemExit("V3-P is disabled by default; pass --run explicitly")
    result = run_v3_package()
    print(
        json.dumps(
            {
                "stage": result["stage"],
                "outcome": result["outcome"],
                "status": result["status"],
                "reproduction_gate_passed": result["reproduction_gate"][
                    "passed"
                ],
                "runtime_seconds": result["runtime_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "FrozenV3Detector",
    "load_package_config",
    "load_v3_detector",
    "run_v3_package",
    "validate_package_config",
]
