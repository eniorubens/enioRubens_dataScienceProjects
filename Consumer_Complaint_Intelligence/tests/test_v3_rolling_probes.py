"""Directed tests for the frozen V3-E3 rolling-probe mechanics."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from consumer_complaint_intelligence.v3 import rolling_probes as rolling


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "config" / "v3" / "rolling_probes.json"


def _synthetic_data() -> rolling.DevelopmentData:
    """Return a deterministic fit scope with two background strata."""

    rows = 120
    labels = np.asarray(
        [rolling.CRITICAL_CLASS] * 6
        + ["credit_reporting"] * 57
        + ["mortgage"] * 57,
        dtype=object,
    )
    dates = np.asarray(
        ["2024-01-01"] * 60 + ["2024-02-01"] * 60,
        dtype="datetime64[D]",
    )
    return rolling.DevelopmentData(
        dates=dates,
        labels=labels,
        group_hashes=np.asarray([f"group-{index:03d}" for index in range(rows)]),
        group_lengths=np.arange(rows, dtype=np.int64) + 10,
        partitions=np.asarray(["train"] * rows, dtype=object),
        texts=np.asarray([f"synthetic complaint {index}" for index in range(rows)]),
        embeddings=np.ones((rows, 384), dtype=np.float16),
    )


def test_frozen_config_loads_with_exact_operational_contract() -> None:
    """Pin the details that were unspecified before E3 execution."""

    config = rolling.load_rolling_config(CONFIG)
    assert config["status"] == "FROZEN_FOR_EXECUTION"
    assert config["hard_negative_mining"]["representation"] == (
        "v2_word_char_tfidf_union_40000_60000"
    )
    assert config["hard_negative_mining"]["class_weight"] == "balanced"
    assert config["classifier"]["class_weight"] is None
    assert config["sampling"]["negative_ratios"] == [5, 10, 15]
    assert config["calibration"]["critical_precision_floor"] == 0.4
    assert config["calibration"]["evaluation_oracle_threshold_forbidden"] is True
    assert config["selection"]["s7_forbidden_folds"] == ["F1", "F2"]


def test_config_rejects_precision_or_sampling_drift() -> None:
    """Reject scientific changes after the operational contract is frozen."""

    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    changed_floor = copy.deepcopy(payload)
    changed_floor["calibration"]["critical_precision_floor"] = 0.2
    with pytest.raises(ValueError, match="precision floor"):
        rolling.validate_rolling_config(changed_floor)
    changed_ratio = copy.deepcopy(payload)
    changed_ratio["sampling"]["negative_ratios"] = [1]
    with pytest.raises(ValueError, match="negative ratios"):
        rolling.validate_rolling_config(changed_ratio)


@pytest.mark.parametrize(
    ("ratio", "hard", "background"),
    [(5, 3, 2), (10, 7, 3), (15, 10, 5)],
)
def test_sampling_keeps_all_positives_and_exact_negative_composition(
    ratio: int,
    hard: int,
    background: int,
) -> None:
    """Build exact, deterministic, unique fit pools for every frozen ratio."""

    data = _synthetic_data()
    fit = np.arange(len(data.labels), dtype=np.int64)
    margins = np.linspace(-1.0, 1.0, len(fit))
    first, evidence = rolling.sample_fit_pool(
        data,
        fit,
        margins,
        fold_id="F1",
        ratio=ratio,
        hard_per_positive=hard,
        background_per_positive=background,
    )
    second, repeated = rolling.sample_fit_pool(
        data,
        fit,
        margins,
        fold_id="F1",
        ratio=ratio,
        hard_per_positive=hard,
        background_per_positive=background,
    )
    assert np.array_equal(first, second)
    assert evidence == repeated
    assert evidence["positive_groups"] == 6
    assert evidence["hard_negative_groups"] == 6 * hard
    assert evidence["background_negative_groups"] == 6 * background
    assert evidence["negative_groups"] == 6 * ratio
    assert evidence["sample_rows"] == 6 * (ratio + 1)
    assert len(np.unique(first)) == len(first)
    assert set(range(6)).issubset(set(first.tolist()))
    assert len(evidence["sample_signature"]) == 64


def test_largest_remainder_allocation_is_exact_and_deterministic() -> None:
    """Preserve proportional month-family background support without replacement."""

    sizes = {("2024-01", "a"): 51, ("2024-01", "b"): 29, ("2024-02", "a"): 20}
    allocation = rolling._allocate_largest_remainder(sizes, 37)
    assert sum(allocation.values()) == 37
    assert all(0 <= allocation[key] <= sizes[key] for key in sizes)
    assert allocation == rolling._allocate_largest_remainder(sizes, 37)


def test_threshold_maximizes_f1_subject_to_precision_floor() -> None:
    """Select only from calibration and enforce deterministic tie-breaking."""

    target = np.asarray([1, 0, 1, 0, 1, 0], dtype=np.int8)
    probabilities = np.asarray([0.9, 0.8, 0.7, 0.4, 0.3, 0.1])
    selected = rolling.select_threshold(target, probabilities, 0.4)
    assert selected["precision_floor_met"] is True
    assert selected["critical_precision"] >= 0.4
    observed = rolling.evaluate_probabilities(
        target,
        probabilities,
        selected["threshold"],
    )
    assert observed["critical_f1"] == pytest.approx(selected["critical_f1"])
    assert observed["critical_recall"] == pytest.approx(
        selected["critical_recall"]
    )


def test_evaluation_floor_failure_sets_ranking_recall_to_zero() -> None:
    """Never use an evaluation-oracle threshold to rescue a precision failure."""

    target = np.asarray([1, 0, 0, 0], dtype=np.int8)
    probabilities = np.asarray([0.7, 0.9, 0.8, 0.6])
    metrics = rolling.evaluate_probabilities(target, probabilities, 0.6, 0.4)
    assert metrics["critical_precision"] == pytest.approx(0.25)
    assert metrics["critical_recall"] == pytest.approx(1.0)
    assert metrics["precision_floor_met"] is False
    assert metrics["critical_recall_at_precision_floor"] == 0.0


def test_sigmoid_is_fitted_on_calibration_and_returns_probabilities() -> None:
    """Calibrate one-dimensional margins without exposing a fitted object."""

    calibration_margins = np.asarray([-2, -1, 0, 1, 2, 3], dtype=float)
    calibration_target = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int8)
    evaluation_margins = np.asarray([-3, 0.5, 4], dtype=float)
    calibration, evaluation = rolling.calibrate_sigmoid(
        calibration_margins,
        calibration_target,
        evaluation_margins,
    )
    assert np.all((calibration >= 0) & (calibration <= 1))
    assert np.all((evaluation >= 0) & (evaluation <= 1))
    assert np.all(np.diff(evaluation) > 0)


def test_privacy_guard_rejects_row_level_outputs() -> None:
    """Reject individual evidence before result publication."""

    rolling._validate_privacy({"status": "safe", "metrics": {"f1": 0.2}})
    with pytest.raises(ValueError, match="Private V3-E3"):
        rolling._validate_privacy({"scores": [0.1, 0.2]})
    with pytest.raises(ValueError, match="Private V3-E3"):
        rolling._validate_privacy({"nested": {"embeddings": [[1.0]]}})


def test_cli_is_disabled_without_explicit_run_gate() -> None:
    """Prevent accidental scientific execution from a plain module call."""

    with pytest.raises(SystemExit, match="disabled by default"):
        rolling.main([])
