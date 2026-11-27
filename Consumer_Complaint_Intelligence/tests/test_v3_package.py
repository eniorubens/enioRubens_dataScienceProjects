"""Directed tests for the frozen V3-P package boundary."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from consumer_complaint_intelligence.v3 import package as v3_package


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "config" / "v3" / "package.json"
BUNDLE = ROOT / "artifacts" / "v3" / "consumer_complaint_detector_v3.joblib"


def test_frozen_package_config_loads_and_pins_selected_candidate() -> None:
    """Pin the exact candidate, sample, threshold, and sealed boundary."""

    config = v3_package.load_package_config(CONFIG)
    assert config["candidate"]["candidate_id"] == (
        "tfidf_word_char_ratio_1_10"
    )
    assert config["sample"]["row_count"] == 10_406
    assert config["sample"]["signature"] == (
        "C20F9823FB5C55D6CAC4AC8243CD5FC3AE7B6DD45F77048F60A7770F768D6138"
    )
    assert config["reproduction_gate"]["expected"]["calibration"][
        "threshold"
    ] == 0.10162120744876312
    assert config["boundary"]["sealed_partitions"] == [
        "test",
        "stress",
        "monitor",
    ]
    assert config["deployment"]["confirmation_claim_authorized"] is False


def test_package_config_rejects_candidate_threshold_or_boundary_drift() -> None:
    """Reject post-selection scientific changes before expensive execution."""

    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    changed_candidate = copy.deepcopy(payload)
    changed_candidate["candidate"]["candidate_id"] = "another"
    with pytest.raises(ValueError, match="candidate"):
        v3_package.validate_package_config(changed_candidate)
    changed_threshold = copy.deepcopy(payload)
    changed_threshold["reproduction_gate"]["expected"]["calibration"][
        "threshold"
    ] = 0.2
    with pytest.raises(ValueError, match="threshold"):
        v3_package.validate_package_config(changed_threshold)
    changed_boundary = copy.deepcopy(payload)
    changed_boundary["boundary"]["sealed_partitions"] = []
    with pytest.raises(ValueError, match="sealed"):
        v3_package.validate_package_config(changed_boundary)


def test_frozen_detector_has_batch_invariant_calibrated_predictions() -> None:
    """Keep runtime probability and override decisions independent of batching."""

    texts = [
        "debt collector called repeatedly",
        "mortgage payment was applied",
        "credit card debt is not mine",
        "bank account transfer problem",
        "collector reported a false debt",
        "mortgage escrow question",
    ]
    target = np.asarray([1, 0, 1, 0, 1, 0], dtype=np.int8)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)
    estimator = LinearSVC(C=1.0, dual="auto", random_state=42)
    estimator.fit(matrix, target)
    margins = estimator.decision_function(matrix)
    calibrator = LogisticRegression(C=1_000_000.0, random_state=42)
    calibrator.fit(margins.reshape(-1, 1), target)
    detector = v3_package.FrozenV3Detector(
        schema_version=v3_package.RUNTIME_SCHEMA,
        model_version="test",
        critical_class=v3_package.CRITICAL_CLASS,
        input_language="en-US",
        vectorizer=vectorizer,
        estimator=estimator,
        calibrator=calibrator,
        threshold=0.5,
        selected_candidate_id=v3_package.SELECTED_CANDIDATE,
        source_result_sha256="A" * 64,
    )
    first = detector.predict_probabilities(texts, batch_size=1)
    full = detector.predict_probabilities(texts, batch_size=len(texts))
    assert np.array_equal(first, full)
    assert np.array_equal(
        detector.predict_critical(texts, batch_size=2),
        full >= 0.5,
    )
    assert detector.predict_probabilities([]).shape == (0,)
    with pytest.raises(ValueError, match="batch_size"):
        detector.predict_probabilities(texts, batch_size=0)
    with pytest.raises(TypeError, match="strings"):
        detector.predict_probabilities(["valid", 1])


def test_exact_reproduction_gate_rejects_any_aggregate_difference() -> None:
    """Never freeze a bundle over approximately matching E3 evidence."""

    config = v3_package.load_package_config(CONFIG)
    expected = config["reproduction_gate"]["expected"]
    sample = {
        "positive_groups": 946,
        "hard_negative_groups": 6622,
        "background_negative_groups": 2838,
        "negative_groups": 9460,
        "sample_rows": 10406,
        "sample_signature": config["sample"]["signature"],
    }
    audit = {"selected_candidate_id": v3_package.SELECTED_CANDIDATE}
    passed = v3_package._evaluate_reproduction(
        config,
        audit,
        sample,
        expected["calibration"],
        expected["evaluation"],
        expected["f3_system_comparison"],
    )
    assert passed["passed"] is True
    changed = copy.deepcopy(expected["evaluation"])
    changed["true_positive"] += 1
    failed = v3_package._evaluate_reproduction(
        config,
        audit,
        sample,
        expected["calibration"],
        changed,
        expected["f3_system_comparison"],
    )
    assert failed["passed"] is False
    assert [
        check["name"] for check in failed["checks"] if not check["passed"]
    ] == ["evaluation_block"]


def test_public_result_privacy_guard_rejects_individual_evidence() -> None:
    """Keep narratives, positions, and individual outputs out of JSON."""

    v3_package._validate_privacy({"status": "safe", "metrics": {"f1": 0.2}})
    for key in ("texts", "indices", "scores", "probabilities", "estimator"):
        with pytest.raises(ValueError, match="Private V3-P"):
            v3_package._validate_privacy({key: []})


def test_cli_requires_explicit_execution_gate() -> None:
    """Prevent an accidental expensive package reconstruction."""

    with pytest.raises(SystemExit, match="disabled by default"):
        v3_package.main([])


@pytest.mark.skipif(not BUNDLE.is_file(), reason="local V3-P bundle unavailable")
def test_real_frozen_bundle_loads_and_matches_manifest() -> None:
    """Exercise the finished local package after the reproduction run."""

    detector = v3_package.load_v3_detector(ROOT)
    assert type(detector).__module__ == (
        "consumer_complaint_intelligence.v3.package"
    )
    assert detector.selected_candidate_id == v3_package.SELECTED_CANDIDATE
    assert detector.threshold == 0.10162120744876312
    assert detector.critical_class == "debt_credit_management"
