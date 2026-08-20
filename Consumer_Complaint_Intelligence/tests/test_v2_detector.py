"""Tests for the V2 binary critical detector."""

from __future__ import annotations

import unittest

import numpy as np
from imblearn.over_sampling import RandomOverSampler
from scipy import sparse

from consumer_complaint_intelligence import v2_detector as detector_module
from consumer_complaint_intelligence.s6 import CRITICAL_CLASS, MODELED_FAMILIES
from consumer_complaint_intelligence.v2_protocol import calculate_scientific_gates
from consumer_complaint_intelligence.v2_detector import (
    HARD_NEGATIVE,
    RANDOM_OVER,
    V2CriticalDetector,
    WEIGHTED_FULL,
    WORD_CHAR_TFIDF_ALIAS,
    WORD_TFIDF_ALIAS,
    build_estimator,
    build_vectorizer,
    combine_detector_with_fallback,
    count_override_decisions,
    fit_and_evaluate,
    role_partition_map,
    run_v2_detector_smoke,
    search_detector_threshold_exact,
)


def _development_data(
    repeats: int = 4,
) -> tuple[list[str], list[str]]:
    """Build deterministic in-memory data covering all modeled families."""

    texts = []
    labels = []
    for label in MODELED_FAMILIES:
        for repeat in range(repeats):
            texts.append(
                f"complaint {label} topic {label} marker {repeat}"
            )
            labels.append(label)
    return texts, labels


class V2DetectorTests(unittest.TestCase):
    """Verify V2 representations, fitting, thresholding, and privacy."""

    def test_representations_are_sparse_and_contract_shaped(self) -> None:
        """Build both frozen representations as sparse feature matrices."""

        texts, _ = _development_data()
        word = build_vectorizer(WORD_TFIDF_ALIAS)
        word_matrix = word.fit_transform(texts)
        union = build_vectorizer(WORD_CHAR_TFIDF_ALIAS)
        union_matrix = union.fit_transform(texts)
        self.assertTrue(sparse.issparse(word_matrix))
        self.assertTrue(sparse.issparse(union_matrix))
        self.assertEqual(word.ngram_range, (1, 2))
        self.assertEqual(word.max_features, 40000)
        self.assertEqual(union.transformer_list[0][0], "word")
        self.assertEqual(union.transformer_list[1][0], "char")

    def test_estimator_parameters_follow_the_protocol(self) -> None:
        """Keep weighted and oversampling estimator parameters distinct."""

        weighted = build_estimator(0.3, WEIGHTED_FULL)
        oversampled = build_estimator(1.0, RANDOM_OVER)
        hard_negative = build_estimator(0.1, HARD_NEGATIVE)
        self.assertEqual(weighted.class_weight, "balanced")
        self.assertIsNone(oversampled.class_weight)
        self.assertIsNone(hard_negative.class_weight)
        self.assertEqual(weighted.max_iter, 5000)
        self.assertEqual(weighted.random_state, 42)
        with self.assertRaises(ValueError):
            build_estimator(0.2, WEIGHTED_FULL)

    def test_random_oversampling_occurs_only_during_fit(self) -> None:
        """Fit expands aggregate counts while decision calls only score."""

        texts, labels = _development_data(6)
        detector = V2CriticalDetector(
            WORD_TFIDF_ALIAS,
            RANDOM_OVER,
            0.3,
            sampling_strategy=0.2,
        ).fit(texts, labels, partition="train")
        self.assertEqual(detector.fit_rows_before, len(labels))
        self.assertGreater(detector.fit_rows_after, detector.fit_rows_before)
        self.assertFalse(hasattr(detector, "sampler"))
        scores = detector.decision_function(texts[:5])
        self.assertEqual(scores.shape, (5,))
        with self.assertRaises(ValueError):
            detector.decision_function(texts[:1], input_language="pt-BR")

    def test_integer_weights_match_materialized_random_oversampling(self) -> None:
        """Match explicit sparse row duplication with integer weights."""

        texts, labels = _development_data(8)
        target = np.asarray(
            [int(label == CRITICAL_CLASS) for label in labels], dtype=np.int8
        )
        vectorizer = build_vectorizer(WORD_TFIDF_ALIAS)
        matrix = vectorizer.fit_transform(texts)
        sampler = RandomOverSampler(sampling_strategy=0.2, random_state=42)
        duplicated_matrix, duplicated_target = sampler.fit_resample(
            matrix, target
        )
        materialized = build_estimator(0.3, RANDOM_OVER)
        materialized.fit(duplicated_matrix, duplicated_target)
        weighted = V2CriticalDetector(
            WORD_TFIDF_ALIAS,
            RANDOM_OVER,
            0.3,
            sampling_strategy=0.2,
        ).fit(texts, labels, partition="train")
        np.testing.assert_allclose(
            materialized.decision_function(matrix),
            weighted.decision_function(texts),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_threshold_search_is_exact_margin_search_and_deterministic(self) -> None:
        """Select only unique margins and never interpret them as probabilities."""

        labels = list(MODELED_FAMILIES)
        fallback = list(labels)
        fallback[0] = MODELED_FAMILIES[1]
        scores = np.asarray([-2.0, -1.0, 0.2, 0.2, 1.4, -0.5, 0.7, -0.1, 0.4])
        first = search_detector_threshold_exact(labels, scores, fallback)
        second = search_detector_threshold_exact(labels, scores, fallback)
        selected = first["selected"]
        possible = set(float(value) for value in np.unique(scores))
        possible.add(float(np.nextafter(scores.max(), np.inf)))
        self.assertIn(selected["threshold"], possible)
        self.assertEqual(first, second)
        self.assertEqual(first["threshold_count"], len(possible))
        self.assertNotIn("probability", str(first).lower())

    def test_incremental_threshold_search_matches_brute_force(self) -> None:
        """Match an explicit prediction rebuild at every unique margin."""

        labels = [label for label in MODELED_FAMILIES for _ in range(4)]
        fallback = [
            MODELED_FAMILIES[(index + 2) % len(MODELED_FAMILIES)]
            for index in range(len(labels))
        ]
        scores = np.asarray(
            [((index * 7) % 13 - 6) / 5 for index in range(len(labels))],
            dtype=np.float64,
        )
        incremental = search_detector_threshold_exact(
            labels, scores, fallback
        )["selected"]
        thresholds = [float(np.nextafter(scores.max(), np.inf))]
        thresholds.extend(float(value) for value in np.unique(scores))
        selected = None
        selected_key = None
        for threshold in thresholds:
            predictions = combine_detector_with_fallback(
                scores >= threshold, fallback
            )
            metrics = detector_module._aggregate_metrics(labels, predictions)
            gates = calculate_scientific_gates(
                metrics, detector_module.load_v2_protocol()
            )
            key = (
                int(gates["passed"]),
                int(gates["gate_count"]),
                metrics["critical_f1"],
                metrics["macro_f1"],
                metrics["critical_precision"],
                threshold,
            )
            if selected_key is None or key > selected_key:
                override_decisions, effective_overrides = count_override_decisions(
                    scores >= threshold, fallback
                )
                selected = {
                    "threshold": threshold,
                    "metrics": metrics,
                    "gates": gates,
                    "override_decisions": override_decisions,
                    "effective_overrides": effective_overrides,
                }
                selected_key = key
        self.assertEqual(incremental, selected)

    def test_threshold_boundary_preserves_fallback_critical_and_recovers_one(
        self,
    ) -> None:
        """Prove false stage-A decisions preserve and true decisions recover."""

        preserved = combine_detector_with_fallback(
            (False,), (CRITICAL_CLASS,)
        )
        recovered = combine_detector_with_fallback(
            (True,), (MODELED_FAMILIES[0],)
        )
        self.assertEqual(preserved, (CRITICAL_CLASS,))
        self.assertEqual(recovered, (CRITICAL_CLASS,))

    def test_partition_and_alignment_boundaries_are_strict(self) -> None:
        """Reject sealed partitions, wrong roles, mismatched arrays, and bad labels."""

        texts, labels = _development_data(2)
        detector = V2CriticalDetector(
            WORD_TFIDF_ALIAS, WEIGHTED_FULL, 0.3
        )
        with self.assertRaises(ValueError):
            detector.fit(texts, labels, partition="test")
        with self.assertRaises(ValueError):
            detector.fit(texts, labels[:-1], partition="train")
        with self.assertRaises(ValueError):
            detector.fit(texts, labels, partition="validation")
        with self.assertRaises(ValueError):
            combine_detector_with_fallback((True,), ("unknown",))

    def test_smoke_is_deterministic_nine_class_and_hierarchical(self) -> None:
        """Exercise the complete diagnostic smoke without reading project data."""

        first = run_v2_detector_smoke()
        second = run_v2_detector_smoke()
        self.assertEqual(first["status"], "DIAGNOSTIC_ONLY")
        self.assertTrue(first["checks"]["all_nine_classes_present"])
        self.assertTrue(first["checks"]["fallback_critical_preserved"])
        self.assertTrue(first["checks"]["stage_a_recovered_critical"])
        self.assertEqual(first["metrics"], second["metrics"])
        self.assertEqual(first["threshold"], second["threshold"])

    def test_fit_and_evaluate_returns_only_aggregate_evidence(self) -> None:
        """Keep candidate evaluation free of texts, identifiers, and scores."""

        fit_texts, fit_labels = _development_data(4)
        calibration_texts, calibration_labels = _development_data(3)
        outer_texts, outer_labels = _development_data(3)
        result = fit_and_evaluate(
            fit_texts,
            fit_labels,
            calibration_texts,
            calibration_labels,
            calibration_labels,
            outer_texts,
            outer_labels,
            outer_labels,
            fit_partition="train",
            calibration_partition="validation",
            outer_partition="validation",
            representation_alias=WORD_TFIDF_ALIAS,
            balance_strategy=WEIGHTED_FULL,
            c_value=0.3,
        )
        self.assertIn("metrics", result["calibration"])
        self.assertIn("metrics", result["outer"])
        self.assertIn("runtime_seconds", result)
        serialized = str(result).lower()
        self.assertNotIn(fit_texts[0].lower(), serialized)
        self.assertNotIn("complaint topic", serialized)
        self.assertNotIn("complaint id", serialized)
        self.assertNotIn("decision_function", serialized)
        self.assertNotIn("individual", serialized)

    def test_override_counts_distinguish_effective_from_raw(self) -> None:
        """Count every firing decision, but only non-critical fallbacks."""

        decisions = (True, True, False, True)
        fallback = (
            CRITICAL_CLASS,
            MODELED_FAMILIES[0],
            MODELED_FAMILIES[1],
            MODELED_FAMILIES[0],
        )
        override_decisions, effective_overrides = count_override_decisions(
            decisions, fallback
        )
        self.assertEqual(override_decisions, 3)
        self.assertEqual(effective_overrides, 2)
        with self.assertRaises(ValueError):
            count_override_decisions(decisions[:-1], fallback)

    def test_threshold_search_reports_override_counts_for_final_choice(
        self,
    ) -> None:
        """Report override counts for the selected threshold, not mid-loop state."""

        labels = list(MODELED_FAMILIES)
        fallback = list(labels)
        fallback[0] = MODELED_FAMILIES[1]
        scores = np.asarray([-2.0, -1.0, 0.2, 0.2, 1.4, -0.5, 0.7, -0.1, 0.4])
        result = search_detector_threshold_exact(labels, scores, fallback)
        selected = result["selected"]
        decisions = scores >= float(selected["threshold"])
        expected_override, expected_effective = count_override_decisions(
            decisions, fallback
        )
        self.assertEqual(selected["override_decisions"], expected_override)
        self.assertEqual(selected["effective_overrides"], expected_effective)
        self.assertLessEqual(
            selected["effective_overrides"], selected["override_decisions"]
        )
        self.assertGreaterEqual(selected["effective_overrides"], 0)

    def test_role_partition_map_matches_protocol_json(self) -> None:
        """Derive the role/partition map from the protocol instead of a literal."""

        protocol = detector_module.load_v2_protocol()
        windows = protocol.payload["development_windows"]
        expected = {
            "fit": windows["inner_fit"]["partition"],
            "inner_calibration": windows["inner_calibration"]["partition"],
            "outer": windows["outer_evaluation"]["partition"],
        }
        self.assertEqual(role_partition_map(protocol), expected)
        self.assertEqual(role_partition_map(), expected)
        self.assertEqual(
            expected,
            {
                "fit": "train",
                "inner_calibration": "validation",
                "outer": "validation",
            },
        )


if __name__ == "__main__":
    unittest.main()
