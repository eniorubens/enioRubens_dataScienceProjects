"""Test the text-only Kaggle V2-D1 import and reporting module."""

import json
import tempfile
import unittest
from pathlib import Path

from consumer_complaint_intelligence.v2_import import (
    load_stress_payload,
    read_kaggle_log,
    render_d2_import_report,
    render_kaggle_import_report,
    render_package_import_report,
    render_stress_import_report,
    summarize_d2_manifest,
    summarize_d2_result,
    summarize_kaggle_log,
    summarize_package_manifest,
    summarize_package_result,
    summarize_stress_manifest,
    summarize_stress_result,
    summarize_v2_manifest,
    summarize_v2_result,
)


def _candidate(
    candidate_id: str,
    outer_metrics: dict,
    *,
    threshold: float = 0.5,
    safety_passed: bool = True,
) -> dict:
    """Build one small synthetic candidate record for tests."""

    return {
        "candidate_id": candidate_id,
        "calibration": {
            "threshold": threshold,
            "metrics": {
                "critical_f1": 0.9,
                "critical_precision": 0.8,
                "critical_recall": 0.95,
                "macro_f1": 0.85,
            },
        },
        "outer": {
            "metrics": outer_metrics,
            "safety": {"passed": safety_passed},
        },
    }


def _degenerate_result() -> dict:
    """Build a small result where every candidate shares outer metrics."""

    shared = {
        "critical_f1": 0.3,
        "critical_precision": 0.4,
        "critical_recall": 0.25,
        "macro_f1": 0.7,
    }
    candidates = [
        _candidate("cand_a", dict(shared), threshold=0.1),
        _candidate("cand_b", dict(shared), threshold=0.2),
        _candidate("cand_c", dict(shared), threshold=0.3),
    ]
    return {
        "status": "COMPLETE",
        "complete": True,
        "runtime_seconds": 12.5,
        "selected": "cand_a",
        "candidates": candidates,
    }


def _healthy_result() -> dict:
    """Build a small result where candidates differ on outer metrics."""

    candidates = [
        _candidate(
            "cand_a",
            {
                "critical_f1": 0.3,
                "critical_precision": 0.4,
                "critical_recall": 0.25,
                "macro_f1": 0.7,
            },
            threshold=0.1,
        ),
        _candidate(
            "cand_b",
            {
                "critical_f1": 0.5,
                "critical_precision": 0.6,
                "critical_recall": 0.45,
                "macro_f1": 0.75,
            },
            threshold=0.2,
        ),
    ]
    return {
        "status": "COMPLETE",
        "complete": True,
        "runtime_seconds": 8.25,
        "selected": "cand_b",
        "candidates": candidates,
    }


class ReadKaggleLogTests(unittest.TestCase):
    """Verify JSON-array log reconstruction and the plain-text fallback."""

    def test_reconstructs_lines_from_chunked_data_fields(self) -> None:
        """Rebuild full lines even when data chunks split mid-line."""

        entries = [
            {"stream_name": "stdout", "time": 0.1, "data": "hello "},
            {"stream_name": "stdout", "time": 0.2, "data": "world\n"},
            {"stream_name": "stdout", "time": 0.3, "data": "second line\n"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "log.json"
            path.write_text(json.dumps(entries), encoding="utf-8")

            text = read_kaggle_log(path)

        self.assertEqual(text, "hello world\nsecond line")

    def test_plain_text_fallback_when_not_json(self) -> None:
        """Return the raw text when the file is not the JSON-array format."""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "log.txt"
            path.write_text("line one\nline two\n", encoding="utf-8")

            text = read_kaggle_log(path)

        self.assertEqual(text, "line one\nline two")

    def test_missing_file_raises_value_error(self) -> None:
        """Reject a log path that does not exist."""

        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.log"
            with self.assertRaisesRegex(ValueError, "missing"):
                read_kaggle_log(missing)

    def test_empty_file_raises_value_error(self) -> None:
        """Reject an empty log file."""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.log"
            path.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "empty"):
                read_kaggle_log(path)


class SummarizeKaggleLogTests(unittest.TestCase):
    """Verify stream filtering and tail truncation on the log text."""

    def _write_log(self, directory: Path) -> Path:
        """Write a small two-stream chunked log fixture."""

        entries = [
            {"stream_name": "stderr", "time": 0.1, "data": "warn one\n"},
            {"stream_name": "stdout", "time": 0.2, "data": "out "},
            {"stream_name": "stdout", "time": 0.3, "data": "one\n"},
            {"stream_name": "stderr", "time": 0.4, "data": "warn two\n"},
            {"stream_name": "stdout", "time": 0.5, "data": "out two\n"},
        ]
        path = directory / "log.json"
        path.write_text(json.dumps(entries), encoding="utf-8")
        return path

    def test_filters_by_stream_name(self) -> None:
        """Keep only entries from the requested stream."""

        with tempfile.TemporaryDirectory() as directory:
            path = self._write_log(Path(directory))
            text = summarize_kaggle_log(path, stream="stdout")

        self.assertEqual(text, "out one\nout two")

    def test_tail_keeps_last_n_lines(self) -> None:
        """Keep only the trailing lines when tail is given."""

        with tempfile.TemporaryDirectory() as directory:
            path = self._write_log(Path(directory))
            text = summarize_kaggle_log(path, tail=1)

        self.assertEqual(text, "out two")

    def test_non_positive_tail_returns_no_lines(self) -> None:
        """Treat a non-positive tail as an empty selection."""

        with tempfile.TemporaryDirectory() as directory:
            path = self._write_log(Path(directory))
            text = summarize_kaggle_log(path, tail=0)

        self.assertEqual(text, "")


class SummarizeV2ResultTests(unittest.TestCase):
    """Verify the degeneracy check and defensive result rendering."""

    def test_flags_degenerate_outer_metrics(self) -> None:
        """Warn loudly when every candidate shares outer metrics."""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(_degenerate_result()), encoding="utf-8")

            text = summarize_v2_result(path)

        self.assertIn("WARNING", text)
        self.assertIn("IDENTICAL OUTER METRICS", text)
        self.assertIn("differentiated itself from the fallback", text)
        self.assertIn("distinct outer metric tuples: 1", text)
        self.assertIn("distinct calibration thresholds: 3", text)

    def test_does_not_flag_healthy_candidates(self) -> None:
        """Stay silent when candidates differentiate on outer metrics."""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(_healthy_result()), encoding="utf-8")

            text = summarize_v2_result(path)

        self.assertNotIn("WARNING", text)
        self.assertIn("distinct outer metric tuples: 2", text)

    def test_single_candidate_is_never_flagged_degenerate(self) -> None:
        """Never fire the warning when only one candidate exists."""

        payload = _degenerate_result()
        payload["candidates"] = payload["candidates"][:1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            text = summarize_v2_result(path)

        self.assertNotIn("WARNING", text)

    def test_renders_partially_shaped_result_without_raising(self) -> None:
        """Render an older/partial shape defensively instead of raising."""

        payload = {
            "status": "COMPLETE",
            "candidates": [{"candidate_id": "solo"}],
            "selected": "solo",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            text = summarize_v2_result(path)

        self.assertIn("candidate_id: solo", text)
        self.assertIn("N/A", text)

    def test_missing_result_file_raises_value_error(self) -> None:
        """Reject a result path that does not exist."""

        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing_result.json"
            with self.assertRaisesRegex(ValueError, "missing"):
                summarize_v2_result(missing)


class SummarizeV2ManifestTests(unittest.TestCase):
    """Verify the small published manifest renders as readable text."""

    def test_renders_nested_keys_and_values(self) -> None:
        """Render nested mappings indented under their parent key."""

        payload = {
            "status": "COMPLETE",
            "artifact": {"path": "a.json", "size_bytes": 10},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            text = summarize_v2_manifest(path)

        self.assertIn("status: COMPLETE", text)
        self.assertIn("artifact:", text)
        self.assertIn("path: a.json", text)


class RenderKaggleImportReportTests(unittest.TestCase):
    """Verify report composition and graceful skipping of optional parts."""

    def test_skips_missing_optional_sections_without_raising(self) -> None:
        """Skip an absent log path and a missing manifest file cleanly."""

        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory) / "result.json"
            result_path.write_text(
                json.dumps(_healthy_result()), encoding="utf-8"
            )

            report = render_kaggle_import_report(
                result_path,
                log_path=None,
                manifest_path=Path(directory) / "missing_manifest.json",
            )

        self.assertIn("=== KAGGLE LOG", report)
        self.assertIn("skipped: no path provided", report)
        self.assertIn("=== PUBLISHED MANIFEST ===", report)
        self.assertIn("skipped: file not found", report)
        self.assertIn("=== V2 BENCHMARK RESULT ===", report)
        self.assertIn("cand_b", report)

    def test_renders_all_sections_when_present(self) -> None:
        """Render all three sections when every path is provided."""

        entries = [{"stream_name": "stdout", "time": 0.1, "data": "hi\n"}]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            result_path = base / "result.json"
            result_path.write_text(
                json.dumps(_healthy_result()), encoding="utf-8"
            )
            log_path = base / "log.json"
            log_path.write_text(json.dumps(entries), encoding="utf-8")
            manifest_path = base / "manifest.json"
            manifest_path.write_text(
                json.dumps({"status": "COMPLETE"}), encoding="utf-8"
            )

            report = render_kaggle_import_report(
                result_path,
                log_path=log_path,
                manifest_path=manifest_path,
            )

        self.assertIn("hi", report)
        self.assertIn("cand_b", report)
        self.assertIn("status: COMPLETE", report)


def _d2_seed(seed: int, critical_f1: float) -> dict:
    """Build one small synthetic D2 seed-replicate record for tests."""

    return {
        "seed": seed,
        "calibration": {
            "threshold": 0.1,
            "metrics": {"critical_f1": critical_f1},
            "override_decisions": 10,
            "effective_overrides": 8,
        },
        "outer": {
            "metrics": {
                "critical_f1": critical_f1,
                "critical_precision": 0.4,
                "critical_recall": 0.35,
                "macro_f1": 0.73,
            },
            "override_decisions": 12,
            "effective_overrides": 9,
            "safety": {"passed": True},
        },
        "runtime_seconds": 100.0,
    }


def _d2_result(
    *,
    outcome: str = "TRANSFORMER_DISPLACES_CLASSICAL",
    meets_displacement_threshold: bool = True,
    reported_seed: int = 43,
    seed_f1s: tuple = (0.40, 0.45, 0.50),
) -> dict:
    """Build a small, fully-shaped synthetic D2 result for tests.

    ``seed_f1s`` gives the outer critical_f1 for seeds 42, 43, and 44
    in order; the default puts the median (0.45) on seed 43, while the
    best score (0.50) sits on seed 44, so tests can prove the reported
    marker follows the pre-registered seed and not the maximum.
    """

    return {
        "schema_version": "v2-transformer-challenge-v1",
        "status": "COMPLETE",
        "complete": True,
        "diagnostic_only": False,
        "runtime_seconds": 3600.0,
        "allowed_partitions": ["train", "validation"],
        "sealed_partitions": ["test", "stress", "monitor"],
        "sealed_access": {"monitor": False, "stress": False, "test": False},
        "execution_config_signature": "DEADBEEF",
        "model": {
            "model_id": "distilbert-base-uncased",
            "resolved_revision": "abc123",
        },
        "incumbent": {
            "candidate_id": (
                "word_char_tfidf_union_40000_60000_c_1_hard_negative"
            ),
            "artifact": {"sha256": "A0ED", "size_bytes": 219597},
            "outer": {
                "critical_f1": 0.386899,
                "critical_precision": 0.4375,
                "critical_recall": 0.346789,
                "macro_f1": 0.731214,
            },
        },
        "fallback_baseline": {
            "inner_calibration": {
                "critical_f1": 0.3,
                "critical_precision": 0.4,
                "critical_recall": 0.25,
                "macro_f1": 0.7,
            },
            "outer_evaluation": {
                "critical_f1": 0.339665,
                "critical_precision": 0.434286,
                "critical_recall": 0.278899,
                "macro_f1": 0.725816,
            },
        },
        "hard_negative": {
            "positive_groups": 946,
            "hard_negative_groups": 14190,
            "pool_rows": 15136,
            "pool_signature": "POOLSIG",
        },
        "seeds": [
            _d2_seed(42, seed_f1s[0]),
            _d2_seed(43, seed_f1s[1]),
            _d2_seed(44, seed_f1s[2]),
        ],
        "reported": {
            "aggregation": "median_of_outer_critical_f1",
            "seed": reported_seed,
            "critical_f1_vs_fallback": 0.11,
            "critical_f1_vs_incumbent": 0.063,
        },
        "seed_spread": {
            "critical_f1_min": min(seed_f1s),
            "critical_f1_median": sorted(seed_f1s)[1],
            "critical_f1_max": max(seed_f1s),
            "critical_f1_range": max(seed_f1s) - min(seed_f1s),
        },
        "decision": {
            "pre_registered": True,
            "passes_margins": True,
            "has_effective_overrides": True,
            "beats_fallback": True,
            "meets_precision_floor": True,
            "meets_displacement_threshold": meets_displacement_threshold,
            "displacement_bar": 0.4255986693961106,
            "displacement_increment": 0.0387,
            "precision_floor": 0.4342857142857143,
            "outcome": outcome,
            "blocked_reason": None,
        },
        "signature": "SIG",
    }


class SummarizeD2ResultTests(unittest.TestCase):
    """Verify the D2 header block renders the key run identity fields."""

    def test_renders_schema_status_model_and_partitions(self) -> None:
        """Cover schema, status, model identity, and the partitions."""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(_d2_result()), encoding="utf-8")

            text = summarize_d2_result(path)

        self.assertIn("v2-transformer-challenge-v1", text)
        self.assertIn("status: COMPLETE", text)
        self.assertIn("distilbert-base-uncased", text)
        self.assertIn("resolved_revision: abc123", text)
        self.assertIn("allowed_partitions: train, validation", text)
        self.assertIn("sealed_partitions: test, stress, monitor", text)

    def test_missing_result_file_raises_value_error(self) -> None:
        """Reject a D2 result path that does not exist."""

        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing_d2_result.json"
            with self.assertRaisesRegex(ValueError, "missing"):
                summarize_d2_result(missing)


class RenderD2ImportReportTests(unittest.TestCase):
    """Verify the full D2 report composition and its defensive reading."""

    def test_includes_incumbent_all_seeds_and_outcome(self) -> None:
        """Cover the incumbent id, every seed, and the final outcome."""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(_d2_result()), encoding="utf-8")

            report = render_d2_import_report(path)

        self.assertIn(
            "word_char_tfidf_union_40000_60000_c_1_hard_negative", report
        )
        self.assertIn("42", report)
        self.assertIn("43", report)
        self.assertIn("44", report)
        self.assertIn("TRANSFORMER_DISPLACES_CLASSICAL", report)

    def test_reported_seed_marker_follows_median_not_best(self) -> None:
        """Mark the pre-registered median seed, not the best-scoring one."""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            # Median (0.45) is seed 43; best (0.50) is seed 44.
            payload = _d2_result(reported_seed=43, seed_f1s=(0.40, 0.45, 0.50))
            path.write_text(json.dumps(payload), encoding="utf-8")

            report = render_d2_import_report(path)

        # The table row marker column is followed by two spaces then the
        # seed value; the legend line ("* marks the...") has only one
        # space after the asterisk, so this substring is row-specific.
        self.assertIn("  *  43", report)
        self.assertNotIn("  *  42", report)
        self.assertNotIn("  *  44", report)
        self.assertIn("pre-registered median-aggregation seed", report)
        self.assertIn("not the best seed", report)

    def test_classical_winner_stands_renders_explicit_line(self) -> None:
        """State plainly that a non-displacing result is still valid."""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            payload = _d2_result(
                outcome="CLASSICAL_WINNER_STANDS",
                meets_displacement_threshold=False,
            )
            path.write_text(json.dumps(payload), encoding="utf-8")

            report = render_d2_import_report(path)

        self.assertIn("CLASSICAL_WINNER_STANDS", report)
        self.assertIn("valid, published result", report)
        self.assertIn("D1 classical winner remains the V2 candidate", report)

    def test_renders_result_missing_many_keys_without_raising(self) -> None:
        """Render a sparse/malformed result instead of crashing."""

        payload = {"schema_version": "v2-transformer-challenge-v1"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            report = render_d2_import_report(path)

        self.assertIn("N/A", report)
        self.assertIn("UNKNOWN", report)
        self.assertIn("(no seeds available)", report)

    def test_skips_missing_optional_sections_without_raising(self) -> None:
        """Skip an absent log path and a missing manifest file cleanly."""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(_d2_result()), encoding="utf-8")

            report = render_d2_import_report(
                path,
                log_path=None,
                manifest_path=Path(directory) / "missing_manifest.json",
            )

        self.assertIn("=== KAGGLE LOG", report)
        self.assertIn("skipped: no path provided", report)
        self.assertIn("=== D2 PUBLISHED MANIFEST ===", report)
        self.assertIn("skipped: file not found", report)


class SummarizeD2ManifestTests(unittest.TestCase):
    """Verify the small published D2 manifest renders as readable text."""

    def test_renders_outcome_and_reported_seed(self) -> None:
        """Render the pre-registered outcome and the reported seed."""

        payload = {
            "schema_version": "v2-transformer-results-manifest-v1",
            "outcome": "TRANSFORMER_DISPLACES_CLASSICAL",
            "reported_seed": 43,
            "critical_f1_vs_incumbent": 0.063,
            "execution_config": {"stage": "V2.1-D2"},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            text = summarize_d2_manifest(path)

        self.assertIn("outcome: TRANSFORMER_DISPLACES_CLASSICAL", text)
        self.assertIn("reported_seed: 43", text)


_PACKAGE_CANDIDATE_ID = "word_char_tfidf_union_40000_60000_c_1_hard_negative"
_PACKAGE_REQUIRED_CHECKS = (
    "calibrated_threshold",
    "calibration_confusion_matrix",
    "calibration_override_decisions",
    "calibration_effective_overrides",
    "outer_confusion_matrix",
    "outer_override_decisions",
    "outer_effective_overrides",
    "hard_negative_pool_counts",
)
_PACKAGE_MATRIX = [[777777, 1], [2, 3]]


def _package_gate(*, passed: bool = True) -> dict:
    """Build a reproduction-gate block that passes or diverges."""

    checks = {name: True for name in _PACKAGE_REQUIRED_CHECKS}
    checks["config_expected_outer_critical_f1"] = True
    failed: list = []
    divergences: dict = {}
    if not passed:
        checks["outer_confusion_matrix"] = False
        checks["outer_effective_overrides"] = False
        failed = ["outer_confusion_matrix", "outer_effective_overrides"]
        divergences = {
            "outer_effective_overrides": {"observed": 81, "expected": 82},
            "outer_confusion_matrix": {
                "comparable": True,
                "shape": [9, 9],
                "mismatched_cells": 4,
                "total_absolute_difference": 6,
                "max_absolute_cell_difference": 2,
                "observed_total": 127706,
                "expected_total": 127706,
            },
        }
    return {
        "required": True,
        "comparison": "exact_no_tolerance",
        "source_of_truth": "temp/v2/v2_classical_benchmark.json",
        "candidate_id": _PACKAGE_CANDIDATE_ID,
        "required_checks": list(_PACKAGE_REQUIRED_CHECKS),
        "checks": checks,
        "check_count": len(checks),
        "passed": passed,
        "failed_checks": failed,
        "divergences": divergences,
        "fallback_environment": "kaggle_image_used_by_v2_1_d1",
    }


def _package_result(*, passed: bool = True) -> dict:
    """Build a small, fully-shaped synthetic V2 package result.

    The confusion matrices carry the sentinel ``777777`` so a test can
    prove no matrix cell ever reaches the rendered report.
    """

    gate = _package_gate(passed=passed)
    bundle = (
        {
            "persisted": True,
            "path": "artifacts/v2/consumer_complaint_detector_v2.joblib",
            "sha256": "BUNDLESHA",
            "size_bytes": 4211337,
        }
        if passed
        else {
            "persisted": False,
            "reason": "REPRODUCTION_MISMATCH",
            "failed_checks": list(gate["failed_checks"]),
        }
    )
    return {
        "schema_version": "v2-package-v1",
        "code_schema": "v2-package-runtime-v1",
        "stage": "V2.1-P",
        "adr": "docs/ADR-013-v2-frozen-package.md",
        "status": "COMPLETE",
        "outcome": "PACKAGE_FROZEN" if passed else "REPRODUCTION_MISMATCH",
        "complete": True,
        "frozen": passed,
        "diagnostic_only": False,
        "run_mode": "full",
        "runtime_seconds": 6120.5,
        "signature": "PACKAGESIG",
        "sealed_partitions": ["test", "stress", "monitor"],
        "sealed_access": {"monitor": False, "stress": False, "test": False},
        "candidate": {
            "candidate_id": _PACKAGE_CANDIDATE_ID,
            "selected_by": "V2.1-D1",
            "balance_strategy": "hard_negative",
            "C": 1.0,
            "random_state": 42,
        },
        "fit_scope": {"role": "inner_fit", "partition": "train"},
        "calibration_scope": {
            "role": "inner_calibration",
            "partition": "validation",
        },
        "outer_scope": {
            "role": "outer_evaluation",
            "partition": "validation",
        },
        "calibration": {
            "threshold": -0.13949530151425016,
            "override_decisions": 57,
            "effective_overrides": 16,
            "metrics": {
                "row_count": 118274,
                "critical_f1": 0.23221,
                "confusion_matrix": _PACKAGE_MATRIX,
            },
        },
        "outer": {
            "override_decisions": 258,
            "effective_overrides": 82,
            "critical_f1_vs_fallback": 0.047233864926836844,
            "metrics": {
                "row_count": 127706,
                "critical_f1": 0.38689866939611056,
                "critical_precision": 0.4375,
                "critical_recall": 0.3467889908256881,
                "macro_f1": 0.731214101791537,
                "critical_support": 545,
                "confusion_matrix": _PACKAGE_MATRIX,
            },
            "safety": {
                "passed": True,
                "values": {"critical_f1": 0.38689866939611056},
            },
        },
        "fallback_baseline": {
            "outer_evaluation": {
                "critical_f1": 0.3396648044692737,
                "critical_precision": 0.4342857142857143,
                "critical_recall": 0.27889908256880735,
                "macro_f1": 0.7258162180952451,
            },
        },
        "safety_margin": {
            "required_gate_count": 3,
            "limits": {"critical_f1_min": 0.29},
            "values": {"critical_f1": 0.38689866939611056},
            "headroom": {"critical_f1": 0.09689866939611058},
            "passed": True,
            "measured_on": "outer_evaluation",
            "evidence_status": "development_optimistic_not_independent",
        },
        "hard_negative": {
            "positive_groups": 946,
            "hard_negative_groups": 14190,
            "pool_rows": 15136,
            "pool_signature": "POOLSIG",
        },
        "boundary": {
            "allowed_partitions": ["train", "validation"],
            "persists_fitted_weights": True,
            "persists_narratives_or_identifiers": False,
            "persists_row_indices": False,
        },
        "deployment": {
            "deployment_authorized": False,
            "status": "FROZEN_FOR_CONFIRMATION",
            "next_step": "open_stress_2025_h2_once",
        },
        "provenance": {
            "d1_artifact": {
                "path": "temp/v2/v2_classical_benchmark.json",
                "sha256": "A0ED",
                "size_bytes": 219597,
            },
        },
        "reproduction_gate": gate,
        "bundle": bundle,
    }


def _package_smoke_result() -> dict:
    """Build a diagnostic smoke result with no gate and no evidence."""

    return {
        "schema_version": "v2-package-v1",
        "stage": "V2.1-P",
        "status": "DIAGNOSTIC_ONLY",
        "outcome": "DIAGNOSTIC_ONLY",
        "complete": True,
        "frozen": False,
        "diagnostic_only": True,
        "run_mode": "smoke",
        "runtime_seconds": 12.5,
        "signature": "smoke",
        "sealed_access": {"monitor": False, "stress": False, "test": False},
        "checks": {"frozen_config_validated": True},
        "candidate": {"candidate_id": _PACKAGE_CANDIDATE_ID},
        "bundle": {"persisted": False},
    }


def _write_json(directory: str, name: str, payload: dict) -> Path:
    """Write one small JSON fixture and return its path."""

    path = Path(directory) / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class SummarizePackageResultTests(unittest.TestCase):
    """Verify the frozen-package report and its unmissable outcome."""

    def test_frozen_result_names_the_pass_and_the_bundle(self) -> None:
        """Confirm a PACKAGE_FROZEN run reports the persisted bundle."""

        with tempfile.TemporaryDirectory() as directory:
            path = _write_json(directory, "result.json", _package_result())

            text = summarize_package_result(path)

        self.assertIn("outcome: PACKAGE_FROZEN", text)
        self.assertIn("frozen: True", text)
        self.assertIn("PACKAGE FROZEN: every reproduction check passed", text)
        self.assertIn("passed: PASS", text)
        self.assertIn("persisted: True", text)
        self.assertIn(
            "path: artifacts/v2/consumer_complaint_detector_v2.joblib", text
        )
        self.assertIn("sha256: BUNDLESHA", text)
        self.assertIn("size_bytes: 4211337", text)
        self.assertNotIn("REPRODUCTION MISMATCH", text)

    def test_canonical_checks_are_marked_and_listed_first(self) -> None:
        """Mark every required check and leave the extra ones unmarked."""

        with tempfile.TemporaryDirectory() as directory:
            path = _write_json(directory, "result.json", _package_result())

            text = summarize_package_result(path)

        self.assertIn("  *  calibrated_threshold", text)
        self.assertIn("  *  outer_confusion_matrix", text)
        self.assertNotIn("  *  config_expected_outer_critical_f1", text)
        self.assertIn("config_expected_outer_critical_f1", text)
        self.assertIn("canonical check named in required_checks", text)

    def test_mismatch_result_withholds_the_bundle_and_shows_divergence(
        self,
    ) -> None:
        """Never let a REPRODUCTION_MISMATCH read as a successful freeze."""

        with tempfile.TemporaryDirectory() as directory:
            path = _write_json(
                directory, "result.json", _package_result(passed=False)
            )

            text = summarize_package_result(path)

        self.assertIn("outcome: REPRODUCTION_MISMATCH", text)
        self.assertIn("frozen: False", text)
        self.assertIn("REPRODUCTION MISMATCH: THE PACKAGE WAS NOT FROZEN", text)
        self.assertIn("NO joblib bundle was written", text)
        self.assertIn("passed: FAIL", text)
        self.assertIn("outer_confusion_matrix", text)
        self.assertIn("outer_effective_overrides", text)
        self.assertIn("observed: 81", text)
        self.assertIn("expected: 82", text)
        self.assertIn("mismatched_cells: 4", text)
        self.assertIn("max_absolute_cell_difference: 2", text)
        self.assertIn("persisted: False", text)
        self.assertIn("reason: REPRODUCTION_MISMATCH", text)
        self.assertIn("No fitted artifact was written", text)
        self.assertNotIn(
            "path: artifacts/v2/consumer_complaint_detector_v2.joblib", text
        )
        self.assertNotIn("PACKAGE FROZEN: every reproduction check", text)

    def test_never_renders_a_confusion_matrix(self) -> None:
        """Keep every confusion-matrix cell out of the rendered report."""

        for payload in (_package_result(), _package_result(passed=False)):
            with tempfile.TemporaryDirectory() as directory:
                path = _write_json(directory, "result.json", payload)

                text = summarize_package_result(path)

            self.assertNotIn("777777", text)

    def test_integer_counts_never_render_with_decimals(self) -> None:
        """Render counts as integers, never as six-decimal metrics."""

        with tempfile.TemporaryDirectory() as directory:
            path = _write_json(directory, "result.json", _package_result())

            text = summarize_package_result(path)

        self.assertIn("effective_overrides: 82", text)
        self.assertIn("override_decisions: 258", text)
        self.assertIn("critical_support: 545", text)
        self.assertIn("row_count: 127706", text)
        self.assertIn("pool_rows: 15136", text)
        self.assertIn("positive_groups: 946", text)
        for forbidden in (
            "82.000000",
            "258.000000",
            "545.000000",
            "127706.000000",
            "15136.000000",
            "946.000000",
        ):
            self.assertNotIn(forbidden, text)

    def test_published_false_is_never_rendered_as_not_available(self) -> None:
        """Distinguish an absent boolean field from a published False."""

        payload = _package_result(passed=False)
        payload["sealed_access"] = False
        with tempfile.TemporaryDirectory() as directory:
            path = _write_json(directory, "result.json", payload)

            text = summarize_package_result(path)

        self.assertIn("sealed_access: False", text)
        self.assertNotIn("sealed_access: N/A", text)
        self.assertIn("deployment_authorized: False", text)
        self.assertIn("persists_narratives_or_identifiers: False", text)
        self.assertNotIn("deployment_authorized: N/A", text)

    def test_safety_margin_is_declared_development_optimistic(self) -> None:
        """Say plainly that the margin is not independent evidence."""

        with tempfile.TemporaryDirectory() as directory:
            path = _write_json(directory, "result.json", _package_result())

            text = summarize_package_result(path)

        self.assertIn("SAFETY MARGIN", text)
        self.assertIn("evidence_status:", text)
        self.assertIn("development-optimistic", text)
        self.assertIn("not independent evidence", text)
        self.assertIn("measured_on: outer_evaluation", text)

    def test_smoke_mode_artifact_renders_without_raising(self) -> None:
        """Render a diagnostic-only artifact that carries no gate."""

        with tempfile.TemporaryDirectory() as directory:
            path = _write_json(
                directory, "result.json", _package_smoke_result()
            )

            text = summarize_package_result(path)

        self.assertIn("DIAGNOSTIC PREFLIGHT ONLY", text)
        self.assertIn("run_mode: smoke", text)
        self.assertIn("(no checks available)", text)
        self.assertIn("persisted: False", text)
        self.assertNotIn("PACKAGE FROZEN: every reproduction check", text)

    def test_renders_partially_shaped_result_without_raising(self) -> None:
        """Render an artifact missing nearly every block instead of raising."""

        payload = {"schema_version": "v2-package-v1"}
        with tempfile.TemporaryDirectory() as directory:
            path = _write_json(directory, "result.json", payload)

            text = summarize_package_result(path)

        self.assertIn("NOT FROZEN", text)
        self.assertIn("outcome: UNKNOWN", text)
        self.assertIn("N/A", text)
        self.assertIn("(not available)", text)

    def test_missing_result_file_raises_value_error(self) -> None:
        """Reject a package result path that does not exist."""

        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing_package.json"
            with self.assertRaisesRegex(ValueError, "missing"):
                summarize_package_result(missing)

    def test_recomputed_margin_is_shown_next_to_the_frozen_one(self) -> None:
        """Publish the refit's own margin, not only the pinned D1 one."""

        with tempfile.TemporaryDirectory() as directory:
            path = _write_json(directory, "result.json", _package_result())

            text = summarize_package_result(path)

        self.assertIn("recomputed by this run (outer.safety)", text)
        self.assertIn("agrees with the frozen margin on every value", text)

    def test_recomputed_margin_names_a_divergent_value(self) -> None:
        """Name every value where the refit left the frozen margin."""

        payload = _package_result(passed=False)
        payload["outer"]["safety"]["values"]["critical_f1"] = 0.4
        payload["outer"]["safety"]["passed"] = False
        with tempfile.TemporaryDirectory() as directory:
            path = _write_json(directory, "result.json", payload)

            text = summarize_package_result(path)

        self.assertIn("DIFFERS from the frozen margin at: critical_f1", text)

    def test_absent_recomputed_margin_is_silent(self) -> None:
        """Skip the comparison when this run recomputed no margin."""

        payload = _package_result()
        payload["outer"].pop("safety")
        with tempfile.TemporaryDirectory() as directory:
            path = _write_json(directory, "result.json", payload)

            text = summarize_package_result(path)

        self.assertNotIn("recomputed by this run", text)
        self.assertIn("SAFETY MARGIN (DEVELOPMENT)", text)


class SummarizePackageManifestTests(unittest.TestCase):
    """Verify the small published package manifest renders as text."""

    def test_renders_outcome_frozen_flag_and_bundle_record(self) -> None:
        """Render the outcome, the frozen flag, and the bundle record."""

        payload = {
            "schema_version": "v2-package-results-manifest-v1",
            "outcome": "REPRODUCTION_MISMATCH",
            "frozen": False,
            "reproduction_gate_passed": False,
            "failed_checks": ["outer_confusion_matrix"],
            "bundle": None,
            "artifact": {"path": "temp/v2/v2_package.json"},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = _write_json(directory, "manifest.json", payload)

            text = summarize_package_manifest(path)

        self.assertIn("outcome: REPRODUCTION_MISMATCH", text)
        self.assertIn("frozen: False", text)
        self.assertIn("reproduction_gate_passed: False", text)
        self.assertIn("failed_checks: outer_confusion_matrix", text)
        self.assertIn("path: temp/v2/v2_package.json", text)


class RenderPackageImportReportTests(unittest.TestCase):
    """Verify package report composition and graceful skipping."""

    def test_renders_all_sections_when_present(self) -> None:
        """Render the result, the manifest, and the log together."""

        entries = [{"stream_name": "stdout", "time": 0.1, "data": "frozen\n"}]
        with tempfile.TemporaryDirectory() as directory:
            result_path = _write_json(
                directory, "v2_package.json", _package_result()
            )
            manifest_path = _write_json(
                directory,
                "v2_results.json",
                {"outcome": "PACKAGE_FROZEN", "frozen": True},
            )
            log_path = Path(directory) / "package.log"
            log_path.write_text(json.dumps(entries), encoding="utf-8")

            report = render_package_import_report(
                result_path,
                log_path=log_path,
                manifest_path=manifest_path,
            )

        self.assertIn("=== V2.1-P FROZEN PACKAGE RESULT ===", report)
        self.assertIn("=== V2.1-P PUBLISHED MANIFEST ===", report)
        self.assertIn("=== KAGGLE LOG (last 40 lines) ===", report)
        self.assertIn("outcome: PACKAGE_FROZEN", report)
        self.assertIn("frozen", report)

    def test_skips_missing_optional_sections_without_raising(self) -> None:
        """Skip an absent log path and a missing manifest file cleanly."""

        with tempfile.TemporaryDirectory() as directory:
            result_path = _write_json(
                directory, "v2_package.json", _package_result(passed=False)
            )

            report = render_package_import_report(
                result_path,
                log_path=None,
                manifest_path=Path(directory) / "missing_manifest.json",
            )

        self.assertIn("skipped: no path provided", report)
        self.assertIn("skipped: file not found", report)
        self.assertIn("REPRODUCTION MISMATCH", report)

    def test_missing_result_file_is_skipped_not_raised(self) -> None:
        """Stay runnable before the Kaggle artifacts have been placed."""

        with tempfile.TemporaryDirectory() as directory:
            report = render_package_import_report(
                Path(directory) / "v2_package.json"
            )

        self.assertIn("=== V2.1-P FROZEN PACKAGE RESULT ===", report)
        self.assertIn("skipped: file not found", report)


def _stress_arm_metrics(
    critical_f1: float,
    *,
    macro_f1: float = 0.70,
    critical_precision: float = 0.45,
    critical_recall: float = 0.35,
) -> dict:
    """Build one small synthetic scientific-view metrics block."""

    return {
        "macro_f1": macro_f1,
        "critical_f1": critical_f1,
        "critical_precision": critical_precision,
        "critical_recall": critical_recall,
    }


def _stress_gate(name: str, *, observed, limit, strict: bool, passed: bool) -> dict:
    """Build one small synthetic gate-result record for tests."""

    return {
        "name": name,
        "observed": observed,
        "limit": limit,
        "strict": strict,
        "passed": passed,
    }


def _stress_result(
    *,
    status: str = "CONFIRMED",
    confirmed: bool = True,
    passed_count: int = 4,
    v2_critical_f1: float = 0.42,
    s7_critical_f1: float = 0.36,
    agrees_in_sign: bool = True,
    observed_gain: float | None = None,
    include_bootstrap: bool = True,
) -> dict:
    """Build a small, fully-shaped synthetic V2.1-C stress result."""

    gain = v2_critical_f1 - s7_critical_f1
    if observed_gain is None:
        observed_gain = gain
    paired_gate_passed = gain > 0
    gates_results = [
        _stress_gate(
            "macro_f1_min", observed=0.70, limit=0.69,
            strict=False, passed=True,
        ),
        _stress_gate(
            "critical_f1_min", observed=v2_critical_f1, limit=0.2715,
            strict=False, passed=True,
        ),
        _stress_gate(
            "critical_precision_min", observed=0.45, limit=0.20,
            strict=False, passed=True,
        ),
        _stress_gate(
            "paired_critical_f1_gain_min", observed=gain, limit=0.0,
            strict=True, passed=paired_gate_passed,
        ),
    ]
    if passed_count < 4:
        # Force exactly the requested number of PASS entries so the
        # fixture stays internally consistent with passed_count.
        for entry in gates_results:
            entry["passed"] = True
        for entry in gates_results[: len(gates_results) - passed_count]:
            entry["passed"] = False
    payload = {
        "schema_version": "v2-stress-confirmatory-results-v1",
        "code_schema": "v2-stress-runtime-v1",
        "stage": "V2.1-C",
        "adr": "ADR-014",
        "complete": True,
        "status": status,
        "confirmed": confirmed,
        "deploy": False,
        "confirmatory": True,
        "stress_opened": True,
        "opened_at": "2026-08-19T12:00:00Z",
        "stress_scope": {"start": "2025-07-01", "end": "2025-12-31"},
        "remaining_sealed": ["monitor"],
        "model": {
            "model_version": "consumer-complaint-detector-v2",
            "fallback_model_version": "consumer-complaint-classifier-s7",
            "threshold": -0.13949530151425016,
            "critical_class": "debt_credit_management",
            "input_language": "en-US",
            "combination": (
                "critical_override_at_or_above_calibrated_threshold"
            ),
        },
        "scope_counts": {
            "clean_unique_groups": 273480,
            "s2_difference": {
                "s2_novel_unique_groups": 270279,
                "reconstructed_novel_unique_groups": 270100,
                "difference": 179,
            },
        },
        "primary": {
            "view": "scientific",
            "rows": 270100,
            "arms": {
                "v2_combined": {
                    "confusion": [[1] * 9 for _ in range(9)],
                    "metrics": _stress_arm_metrics(v2_critical_f1),
                },
                "s7_fallback_alone": {
                    "confusion": [[1] * 9 for _ in range(9)],
                    "metrics": _stress_arm_metrics(s7_critical_f1),
                },
            },
            "paired": {
                "critical_f1_gain": gain,
                "macro_f1_gain": 0.01,
                "critical_precision_gain": 0.02,
                "critical_recall_gain": gain,
            },
            "override": {
                "override_decisions": 500,
                "effective_overrides": 150,
            },
        },
        "operational_secondary": {
            "view": "operational",
            "rows": 400000,
            "arms": {
                "v2_combined": {
                    "confusion": [[1] * 9 for _ in range(9)],
                    "metrics": _stress_arm_metrics(v2_critical_f1 - 0.02),
                },
                "s7_fallback_alone": {
                    "confusion": [[1] * 9 for _ in range(9)],
                    "metrics": _stress_arm_metrics(s7_critical_f1 - 0.02),
                },
            },
            "paired": {
                "critical_f1_gain": gain,
                "macro_f1_gain": 0.0,
                "critical_precision_gain": 0.0,
                "critical_recall_gain": 0.0,
            },
            "override": {
                "override_decisions": 700,
                "effective_overrides": 210,
            },
        },
        "gates": {
            "required_gate_count": 4,
            "passed_count": passed_count,
            "passed": passed_count == 4,
            "results": gates_results,
        },
        "expectation": {
            "development_paired_gain": 0.047233864926836844,
            "observed_paired_gain": observed_gain,
            "agrees_in_sign": agrees_in_sign,
        },
        "s2_evidence": {"stress_novel_unique_groups": 270279},
        "provenance": {"raw_sha256": "DEADBEEF"},
        "execution_attempts": [{"attempt": 1, "outcome": "ok"}],
        "signature": "STRESSSIG",
    }
    if include_bootstrap:
        payload["bootstrap"] = {
            "replicates": 2000,
            "seed": 42,
            "confidence_level": 0.95,
            "diagnostic_only": True,
            "v2_combined": {
                "critical_f1": [v2_critical_f1 - 0.02, v2_critical_f1 + 0.02],
            },
            "s7_fallback_alone": {
                "critical_f1": [s7_critical_f1 - 0.02, s7_critical_f1 + 0.02],
            },
            "paired_critical_f1_gain": [gain - 0.03, gain + 0.03],
        }
    return payload


def _stress_manifest(
    *,
    status: str = "CONFIRMED",
    confirmed: bool = True,
    signature: str = "STRESSSIG",
) -> dict:
    """Build a small, fully-shaped synthetic published stress manifest."""

    return {
        "schema_version": "v2-stress-results-manifest-v1",
        "stage": "V2.1-C",
        "status": status,
        "confirmed": confirmed,
        "deploy": False,
        "signature": signature,
        "opened_at": "2026-08-19T12:00:00Z",
    }


def _stress_protocol() -> dict:
    """Build a small, fully-shaped synthetic frozen stress protocol."""

    return {
        "schema_version": "v2-stress-confirmatory-protocol-v1",
        "stage": "V2.1-C",
        "adr": "ADR-014",
        "stress_scope": {"start": "2025-07-01", "end": "2025-12-31"},
        "remaining_sealed": ["monitor"],
        "expectation": {
            "development_paired_gain": 0.047233864926836844,
        },
        "gates": {
            "required_gate_count": 4,
            "macro_f1_min": 0.69,
            "critical_f1_min": 0.2715,
            "critical_precision_min": 0.2,
            "paired_critical_f1_gain_min": 0.0,
        },
    }


class LoadStressPayloadTests(unittest.TestCase):
    """Verify the tolerant loader used by the stress notebook."""

    def test_missing_file_returns_empty_mapping(self) -> None:
        """Return an empty dict instead of raising for an absent file."""

        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.json"
            self.assertEqual(load_stress_payload(missing), {})

    def test_empty_file_returns_empty_mapping(self) -> None:
        """Return an empty dict instead of raising for a blank file."""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.json"
            path.write_text("", encoding="utf-8")
            self.assertEqual(load_stress_payload(path), {})

    def test_loads_a_valid_json_object(self) -> None:
        """Parse a present, non-empty JSON object file normally."""

        with tempfile.TemporaryDirectory() as directory:
            path = _write_json(directory, "result.json", {"status": "CONFIRMED"})
            self.assertEqual(load_stress_payload(path), {"status": "CONFIRMED"})

    def test_non_object_json_raises_value_error(self) -> None:
        """Reject a present, non-empty file that is not a JSON object."""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "list.json"
            path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "JSON object"):
                load_stress_payload(path)


class SummarizeStressResultTests(unittest.TestCase):
    """Verify the structured dict summary of a stress result payload."""

    def test_extracts_verdict_gates_arms_and_paired_gain(self) -> None:
        """Cover the verdict, gate counts, both arms, and the paired gain."""

        summary = summarize_stress_result(_stress_result())

        self.assertEqual(summary["status"], "CONFIRMED")
        self.assertTrue(summary["confirmed"])
        self.assertFalse(summary["deploy"])
        self.assertEqual(summary["gates"]["passed_count"], 4)
        self.assertEqual(summary["gates"]["required_gate_count"], 4)
        self.assertAlmostEqual(summary["paired"]["critical_f1_gain"], 0.06)
        self.assertAlmostEqual(
            summary["arms"]["v2_combined"]["critical_f1"], 0.42
        )
        self.assertAlmostEqual(
            summary["arms"]["s7_fallback_alone"]["critical_f1"], 0.36
        )
        self.assertEqual(summary["override"]["effective_overrides"], 150)

    def test_renders_partially_shaped_payload_without_raising(self) -> None:
        """Return None-valued fields instead of raising on a sparse payload."""

        summary = summarize_stress_result({"status": "NOT_CONFIRMED"})

        self.assertEqual(summary["status"], "NOT_CONFIRMED")
        self.assertIsNone(summary["confirmed"])
        self.assertIsNone(summary["gates"]["passed_count"])
        self.assertEqual(summary["arms"]["v2_combined"], {})

    def test_empty_payload_does_not_raise(self) -> None:
        """Return an all-placeholder summary for a completely empty payload."""

        summary = summarize_stress_result({})

        self.assertIsNone(summary["status"])
        self.assertEqual(summary["remaining_sealed"], [])


class SummarizeStressManifestTests(unittest.TestCase):
    """Verify the structured dict summary of a stress manifest payload."""

    def test_extracts_identity_fields(self) -> None:
        """Cover status, confirmed, signature, and schema version."""

        summary = summarize_stress_manifest(_stress_manifest())

        self.assertEqual(summary["status"], "CONFIRMED")
        self.assertTrue(summary["confirmed"])
        self.assertEqual(summary["signature"], "STRESSSIG")
        self.assertEqual(
            summary["schema_version"], "v2-stress-results-manifest-v1"
        )

    def test_empty_payload_does_not_raise(self) -> None:
        """Return an all-None summary for a completely empty manifest."""

        summary = summarize_stress_manifest({})

        self.assertIsNone(summary["status"])
        self.assertIsNone(summary["signature"])


class RenderStressImportReportTests(unittest.TestCase):
    """Verify the nine-block V2.1-C stress confirmatory report."""

    def test_confirmed_render_shows_the_verdict_and_all_gates_passing(
        self,
    ) -> None:
        """Cover the happy-path CONFIRMED verdict and its gate table."""

        report = render_stress_import_report(
            _stress_result(), _stress_manifest(), _stress_protocol()
        )

        self.assertIn("status: CONFIRMED", report)
        self.assertIn("confirmed: True", report)
        self.assertIn("deploy: False", report)
        self.assertIn("gates_passed: 4 / 4", report)
        self.assertIn(
            "CONFIRMED: the frozen V2 package passed all pre-registered "
            "gates",
            report,
        )
        self.assertNotIn("AT LEAST ONE GATE FAILED", report)

    def test_not_confirmed_render_shows_the_failure_unmistakably(self) -> None:
        """Cover a NOT_CONFIRMED verdict with a failing paired gate."""

        result = _stress_result(
            status="NOT_CONFIRMED",
            confirmed=False,
            passed_count=3,
            v2_critical_f1=0.30,
            s7_critical_f1=0.36,
        )

        report = render_stress_import_report(
            result, _stress_manifest(status="NOT_CONFIRMED", confirmed=False),
            _stress_protocol(),
        )

        self.assertIn("status: NOT_CONFIRMED", report)
        self.assertIn("confirmed: False", report)
        self.assertIn(
            "NOT_CONFIRMED: at least one pre-registered gate failed", report
        )
        self.assertIn("FAIL", report)
        self.assertIn("AT LEAST ONE GATE FAILED", report)

    def test_missing_metrics_block_renders_without_raising(self) -> None:
        """Render a payload with no metrics/arms blocks at all."""

        sparse_result = {
            "schema_version": "v2-stress-confirmatory-results-v1",
            "status": "UNKNOWN",
        }

        report = render_stress_import_report(
            sparse_result, {}, _stress_protocol()
        )

        self.assertIn("N/A", report)
        self.assertIn("(no gate results available)", report)
        self.assertIn("(not available)", report)

    def test_completely_empty_payloads_render_without_raising(self) -> None:
        """Render three empty mappings -- the pre-run notebook state."""

        report = render_stress_import_report({}, {}, {})

        self.assertIn("V2.1-C STRESS CONFIRMATORY EVALUATION", report)
        self.assertIn("NO SEALED VERDICT YET", report)

    def test_paired_block_reports_the_gain_correctly_including_negative(
        self,
    ) -> None:
        """Cover a positive gain and, separately, a negative gain."""

        positive = render_stress_import_report(
            _stress_result(v2_critical_f1=0.42, s7_critical_f1=0.36),
            _stress_manifest(),
            _stress_protocol(),
        )
        self.assertIn("paired critical_f1_gain:       0.060000", positive)

        negative = render_stress_import_report(
            _stress_result(
                status="NOT_CONFIRMED",
                confirmed=False,
                passed_count=3,
                v2_critical_f1=0.30,
                s7_critical_f1=0.36,
            ),
            _stress_manifest(status="NOT_CONFIRMED", confirmed=False),
            _stress_protocol(),
        )
        self.assertIn("paired critical_f1_gain:       -0.060000", negative)
        self.assertIn("drift-controlled comparison", negative)

    def test_cross_window_caveat_is_always_present(self) -> None:
        """Require the fixed caveat on confirmed, not-confirmed, and empty."""

        for report in (
            render_stress_import_report(
                _stress_result(), _stress_manifest(), _stress_protocol()
            ),
            render_stress_import_report(
                _stress_result(status="NOT_CONFIRMED", confirmed=False),
                _stress_manifest(status="NOT_CONFIRMED", confirmed=False),
                _stress_protocol(),
            ),
            render_stress_import_report({}, {}, {}),
        ):
            self.assertIn(
                "absolute metrics on stress are NOT comparable to", report
            )
            self.assertIn("drift-controlled", report)

    def test_manifest_result_disagreement_is_surfaced(self) -> None:
        """Flag a manifest whose status disagrees with the sealed result."""

        result = _stress_result(status="CONFIRMED", confirmed=True)
        stale_manifest = _stress_manifest(
            status="NOT_CONFIRMED", confirmed=False, signature="OLDSIG"
        )

        report = render_stress_import_report(
            result, stale_manifest, _stress_protocol()
        )

        self.assertIn("MANIFEST DISAGREES WITH RESULT AT", report)
        self.assertIn("status", report)
        self.assertIn("confirmed", report)
        self.assertIn("signature", report)

    def test_manifest_result_agreement_is_stated_plainly(self) -> None:
        """State plain agreement when the manifest matches the result."""

        report = render_stress_import_report(
            _stress_result(), _stress_manifest(), _stress_protocol()
        )

        self.assertIn(
            "manifest_agreement: agrees on status/confirmed/signature",
            report,
        )
        self.assertNotIn("MANIFEST DISAGREES", report)

    def test_expectation_block_flags_a_sign_disagreement(self) -> None:
        """Flag it explicitly when observed and development gains diverge."""

        result = _stress_result(
            v2_critical_f1=0.30,
            s7_critical_f1=0.36,
            agrees_in_sign=False,
            observed_gain=-0.06,
            status="NOT_CONFIRMED",
            confirmed=False,
            passed_count=3,
        )

        report = render_stress_import_report(
            result,
            _stress_manifest(status="NOT_CONFIRMED", confirmed=False),
            _stress_protocol(),
        )

        self.assertIn("EXPECTATION CHECK (DIAGNOSTIC, NOT A GATE)", report)
        self.assertIn(
            "development_paired_gain (pre-registered): 0.047234", report
        )
        self.assertIn("SIGN DISAGREEMENT", report)
        self.assertIn("is not a gate", report)

    def test_expectation_falls_back_to_protocol_before_the_result_exists(
        self,
    ) -> None:
        """Show the pre-registered gain from the protocol when unsealed."""

        report = render_stress_import_report({}, {}, _stress_protocol())

        self.assertIn(
            "development_paired_gain (pre-registered): 0.047234", report
        )

    def test_both_arms_block_shows_bootstrap_intervals(self) -> None:
        """Cover both arms' metrics and their bootstrap intervals."""

        report = render_stress_import_report(
            _stress_result(), _stress_manifest(), _stress_protocol()
        )

        self.assertIn("BOTH ARMS (SCIENTIFIC VIEW)", report)
        self.assertIn("v2_combined:", report)
        self.assertIn("s7_fallback_alone:", report)
        self.assertIn("95% CI: [0.400000, 0.440000]", report)

    def test_operational_view_is_marked_excluded_from_the_decision(
        self,
    ) -> None:
        """State plainly that the operational view cannot change the gate."""

        report = render_stress_import_report(
            _stress_result(), _stress_manifest(), _stress_protocol()
        )

        self.assertIn("OPERATIONAL SECONDARY VIEW", report)
        self.assertIn(
            "NOT part of the decision: gates are evaluated on the "
            "scientific view only.",
            report,
        )

    def test_scope_counts_include_the_s2_difference_block(self) -> None:
        """Cover the scope counts and the nested s2_difference block."""

        report = render_stress_import_report(
            _stress_result(), _stress_manifest(), _stress_protocol()
        )

        self.assertIn("SCOPE COUNTS", report)
        self.assertIn("clean_unique_groups: 273480", report)
        self.assertIn("s2_difference:", report)
        self.assertIn("s2_novel_unique_groups: 270279", report)

    def test_integrity_block_carries_schema_and_signature(self) -> None:
        """Cover the schema versions, opened_at, signature, and sealed set."""

        report = render_stress_import_report(
            _stress_result(), _stress_manifest(), _stress_protocol()
        )

        self.assertIn("INTEGRITY", report)
        self.assertIn(
            "schema_version: v2-stress-confirmatory-results-v1", report
        )
        self.assertIn("code_schema: v2-stress-runtime-v1", report)
        self.assertIn("signature: STRESSSIG", report)
        self.assertIn("remaining_sealed: monitor", report)


if __name__ == "__main__":
    unittest.main()
