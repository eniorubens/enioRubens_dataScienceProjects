"""Synthetic S3 tests for privacy, groups, caches, and baseline execution."""

import json
import hashlib
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from consumer_complaint_intelligence.s3 import BaselineConfig
from consumer_complaint_intelligence.s3 import FrozenS3Protocol
from consumer_complaint_intelligence.s3 import assert_development_partitions
from consumer_complaint_intelligence.s3 import build_learning_curve_subsets
from consumer_complaint_intelligence.s3 import build_learning_curve_indices
from consumer_complaint_intelligence.s3 import build_or_load_development_dataset
from consumer_complaint_intelligence.s3 import build_or_load_scientific_cache
from consumer_complaint_intelligence.s3 import calculate_metrics
from consumer_complaint_intelligence.s3 import iter_operational_validation_batches
from consumer_complaint_intelligence.s3 import prepare_scientific_split
from consumer_complaint_intelligence.s3 import read_scientific_frame
from consumer_complaint_intelligence.s3 import read_development_rows
from consumer_complaint_intelligence.s3 import run_s3_baseline
from consumer_complaint_intelligence.s3 import run_s3_full
from consumer_complaint_intelligence.s3 import validate_frozen_report
from consumer_complaint_intelligence.s3 import _stratified_limit_rows
from consumer_complaint_intelligence.temporal_split import MODELED_FAMILIES


class S3Tests(unittest.TestCase):
    """Verify S3 behavior with small synthetic data only."""

    def _assert_boundary(self, boundary: dict[str, object]) -> None:
        """Assert the explicit development, fit, evaluation boundary."""

        self.assertEqual(
            boundary,
            {
                "development_partitions": ["train", "validation"],
                "fit_partition": "train",
                "evaluation_partition": "validation",
                "sealed_partitions": ["test", "stress", "monitor"],
            },
        )
        sealed = set(boundary["sealed_partitions"])
        self.assertFalse(
            sealed.intersection(boundary["development_partitions"])
        )
        self.assertNotIn(boundary["fit_partition"], sealed)
        self.assertNotIn(boundary["evaluation_partition"], sealed)
        self.assertNotIn("training_partitions", boundary)

    def _protocol(self) -> FrozenS3Protocol:
        """Return the approved protocol without a corpus-bound digest."""

        return FrozenS3Protocol(
            protocol_id="post_2023_taxonomy",
            windows={
                "train": {"start": "2023-08-01", "end": "2024-06-30"},
                "validation": {
                    "start": "2024-07-01",
                    "end": "2024-12-31",
                },
                "test": {"start": "2025-01-01", "end": "2025-06-30"},
                "stress": {
                    "start": "2025-07-01",
                    "end": "2025-12-31",
                },
                "monitor": {
                    "start": "2026-01-01",
                    "end": "2026-12-31",
                },
            },
            modeled_families=tuple(MODELED_FAMILIES),
            rare_family="other_financial_services",
            fingerprint_version=(
                "lowercase-trim-whitespace-collapse-md5-length-v1"
            ),
            report_sha256=None,
            index_sha256=None,
            approval_status="FROZEN_FOR_S3_DEVELOPMENT",
            approved_on="2026-08-15",
        )

    def _rows(self) -> list[dict[str, object]]:
        """Create two clean rows per class and one ambiguous group."""

        rows: list[dict[str, object]] = []
        complaint_id = 1
        for family in MODELED_FAMILIES:
            rows.append(
                {
                    "Complaint ID": complaint_id,
                    "received_date": "2024-01-01",
                    "product_family": family,
                    "normalized_group_hash": f"train-{family}",
                    "normalized_length": 10,
                    "partition_name": "train",
                    "narrative": f"train narrative {family}",
                }
            )
            complaint_id += 1
            rows.append(
                {
                    "Complaint ID": complaint_id,
                    "received_date": "2024-08-01",
                    "product_family": family,
                    "normalized_group_hash": f"validation-{family}",
                    "normalized_length": 11,
                    "partition_name": "validation",
                    "narrative": f"validation narrative {family}",
                }
            )
            complaint_id += 1
        rows.extend(
            [
                {
                    "Complaint ID": 9000,
                    "received_date": "2024-02-01",
                    "product_family": "debt_credit_management",
                    "normalized_group_hash": "ambiguous",
                    "normalized_length": 9,
                    "partition_name": "train",
                    "narrative": "ambiguous old label",
                },
                {
                    "Complaint ID": 9001,
                    "received_date": "2024-08-01",
                    "product_family": "credit_reporting",
                    "normalized_group_hash": "ambiguous",
                    "normalized_length": 9,
                    "partition_name": "validation",
                    "narrative": "ambiguous new label",
                },
            ]
        )
        return rows

    def test_ambiguous_groups_are_excluded_before_representative_selection(self):
        """Count conflicting labels and never choose the smaller ID silently."""

        split = prepare_scientific_split(self._rows())
        summary = split["summary"]
        self.assertEqual(summary["label_ambiguous_groups"], 1)
        self.assertEqual(summary["label_ambiguous_by_partition"]["train"], 1)
        self.assertEqual(summary["label_ambiguous_by_partition"]["validation"], 1)
        ids = {
            int(row["Complaint ID"]) for row in split["train"]
        }
        self.assertNotIn(9000, ids)
        self.assertNotIn(9001, {
            int(row["Complaint ID"])
            for row in split["validation_scientific"]
        })

    def test_sealed_partition_guard_rejects_any_narrative_request(self):
        """Reject test, stress, and monitor before data access."""

        for partition in ("test", "stress", "monitor"):
            with self.assertRaises(ValueError):
                assert_development_partitions((partition,))

    def test_validation_smoke_limit_is_stratified_and_deterministic(self):
        """Keep all available classes under a bounded validation limit."""

        rows = self._rows()
        validation = [row for row in rows if row["partition_name"] == "validation"]
        validation.extend(
            dict(row, **{"Complaint ID": 10000 + index})
            for index, row in enumerate(tuple(validation))
        )
        first = _stratified_limit_rows(validation, len(MODELED_FAMILIES))
        second = _stratified_limit_rows(validation, len(MODELED_FAMILIES))
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(MODELED_FAMILIES))
        self.assertEqual(
            {row["product_family"] for row in first}, set(MODELED_FAMILIES)
        )

    def test_frozen_report_digest_is_verified(self):
        """Accept the exact approved report and reject a tampered digest."""

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pilot.json"
            payload = {
                "report": {
                    "candidates": [
                        {
                            "candidate": {"name": "post_2023_taxonomy"},
                            "candidate_status": "PASS",
                            "eligible_class_count": 9,
                        }
                    ]
                }
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest().upper()
            protocol = replace(self._protocol(), report_sha256=digest)
            self.assertEqual(
                validate_frozen_report(protocol, path)["candidate_status"],
                "PASS",
            )
            path.write_text(json.dumps({"tampered": True}), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_frozen_report(protocol, path)

    def test_large_list_loading_requires_explicit_override(self):
        """Refuse a full list load when the configured memory guard is crossed."""

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "development.parquet"
            pq.write_table(pa.Table.from_pylist(self._rows()), path)
            with self.assertRaises(ValueError):
                run_s3_baseline(path, Path(temporary) / "s3.json",
                                max_loaded_rows=1)

    def test_learning_curve_is_deterministic_and_keeps_all_classes(self):
        """Keep deterministic class-stratified group samples at every point."""

        rows = [row for row in self._rows() if row["partition_name"] == "train"]
        first = build_learning_curve_subsets(rows, (0.25, 1.0), 7)
        second = build_learning_curve_subsets(rows, (0.25, 1.0), 7)
        self.assertEqual(first, second)
        for subset in first.values():
            self.assertEqual(
                {row["product_family"] for row in subset},
                set(MODELED_FAMILIES),
            )

    def test_metrics_are_serializable(self):
        """Expose stable primary and secondary metrics."""

        metrics = calculate_metrics(
            ["credit_reporting", "debt_collection"],
            ["credit_reporting", "credit_reporting"],
        )
        self.assertIn("macro_f1", metrics)
        self.assertIn("balanced_accuracy", metrics)
        self.assertEqual(set(metrics["per_class"]), set(MODELED_FAMILIES))
        json.dumps(metrics)

    def test_dataset_cache_is_development_only_and_invalidates(self):
        """Join only train and validation, then refresh after source changes."""

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source_rows = []
            index_rows = []
            for row in self._rows():
                source_rows.append(
                    {
                        "Complaint ID": row["Complaint ID"],
                        "Date received": row["received_date"],
                        "Product": row["product_family"],
                        "Issue": "synthetic",
                        "Consumer complaint narrative": row["narrative"],
                    }
                )
                index_rows.append(
                    {
                        "Complaint ID": row["Complaint ID"],
                        "received_date": row["received_date"],
                        "raw_product": row["product_family"],
                        "product_family": row["product_family"],
                        "raw_issue": "synthetic",
                        "normalized_group_hash": row[
                            "normalized_group_hash"
                        ],
                        "normalized_length": row["normalized_length"],
                        "is_modeled_family": True,
                    }
                )
            source = directory / "source.parquet"
            index = directory / "index.parquet"
            output = directory / "development.parquet"
            pq.write_table(pa.Table.from_pylist(source_rows), source)
            pq.write_table(pa.Table.from_pylist(index_rows), index)
            first = build_or_load_development_dataset(
                source, index, output, self._protocol()
            )
            second = build_or_load_development_dataset(
                source, index, output, self._protocol()
            )
            self.assertEqual(first["status"], "refreshed")
            self.assertEqual(second["status"], "hit")
            rows = read_development_rows(output)
            self.assertTrue(rows)
            self.assertTrue(
                {row["partition_name"] for row in rows}
                <= {"train", "validation"}
            )
            os.utime(source, None)
            refreshed = build_or_load_development_dataset(
                source, index, output, self._protocol()
            )
            self.assertEqual(refreshed["status"], "refreshed")

    def test_smoke_executes_without_real_corpus(self):
        """Run the bounded classifier and cache an incremental JSON result."""

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "development.parquet"
            artifact = Path(temporary) / "s3.json"
            pq.write_table(pa.Table.from_pylist(self._rows()), path)
            result = run_s3_baseline(
                path,
                artifact,
                config=BaselineConfig(
                    max_features=500,
                    min_df=1,
                    max_iter=20,
                    fractions=(1.0,),
                ),
            )
            self.assertTrue(result["complete"])
            self.assertIn("1", result["points"])
            self.assertIn("sgd_logistic", result["points"]["1"])
            self._assert_boundary(result["protocol_boundary"])

            artifact_payload = json.loads(artifact.read_text(encoding="utf-8"))
            artifact_payload["protocol_boundary"] = {
                "training_partitions": ["train", "validation"],
                "sealed_partitions": ["test", "stress", "monitor"],
            }
            artifact.write_text(
                json.dumps(artifact_payload), encoding="utf-8"
            )
            migrated = run_s3_baseline(
                path,
                artifact,
                config=BaselineConfig(
                    max_features=500,
                    min_df=1,
                    max_iter=20,
                    fractions=(1.0,),
                ),
            )
            self._assert_boundary(migrated["protocol_boundary"])
            persisted = json.loads(artifact.read_text(encoding="utf-8"))
            self._assert_boundary(persisted["protocol_boundary"])

    def test_scientific_cache_resolves_ambiguity_and_validation_novelty(self):
        """Cache only clean representatives and report audit counts."""

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "development.parquet"
            cache = directory / "scientific.parquet"
            pq.write_table(pa.Table.from_pylist(self._rows()), source)
            result = build_or_load_scientific_cache(source, cache)
            frame = read_scientific_frame(cache)
            rows = frame.to_pylist()
            self.assertEqual(result["summary"]["ambiguous_groups"], 1)
            self.assertEqual(result["summary"]["ambiguous_rows"], 2)
            self.assertEqual(len(rows), 18)
            self.assertNotIn(9000, {row["Complaint ID"] for row in rows})
            self.assertNotIn(9001, {row["Complaint ID"] for row in rows})
            self.assertEqual(
                result["summary"]["raw_rows_by_partition"],
                {"train": 10, "validation": 10},
            )

    def test_train_representative_is_selected_within_train_partition(self):
        """Keep a train group when validation has the lower complaint ID."""

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "development.parquet"
            cache = directory / "scientific.parquet"
            rows = self._rows()
            rows.extend(
                [
                    {
                        "Complaint ID": 10_002,
                        "received_date": "2024-01-01",
                        "product_family": "credit_reporting",
                        "normalized_group_hash": "lower-validation-id",
                        "normalized_length": 13,
                        "partition_name": "train",
                        "narrative": "train representative",
                    },
                    {
                        "Complaint ID": 10_001,
                        "received_date": "2024-08-01",
                        "product_family": "credit_reporting",
                        "normalized_group_hash": "lower-validation-id",
                        "normalized_length": 13,
                        "partition_name": "validation",
                        "narrative": "validation duplicate",
                    },
                ]
            )
            pq.write_table(pa.Table.from_pylist(rows), source)
            build_or_load_scientific_cache(source, cache)
            ids = {
                int(row["Complaint ID"])
                for row in read_scientific_frame(cache).to_pylist()
            }
            self.assertIn(10_002, ids)
            self.assertNotIn(10_001, ids)

    def test_scientific_cache_invalidates_and_batches_operational_rows(self):
        """Refresh the scientific cache and stream validation in small batches."""

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "development.parquet"
            cache = directory / "scientific.parquet"
            rows = self._rows()
            pq.write_table(pa.Table.from_pylist(rows), source)
            first = build_or_load_scientific_cache(source, cache)
            second = build_or_load_scientific_cache(source, cache)
            self.assertEqual(first["status"], "refreshed")
            self.assertEqual(second["status"], "hit")
            rows.append(
                {
                    "Complaint ID": 20_000,
                    "received_date": "2024-08-01",
                    "product_family": "credit_reporting",
                    "normalized_group_hash": "new-validation",
                    "normalized_length": 12,
                    "partition_name": "validation",
                    "narrative": "new validation narrative",
                }
            )
            pq.write_table(pa.Table.from_pylist(rows), source)
            refreshed = build_or_load_scientific_cache(source, cache)
            self.assertEqual(refreshed["status"], "refreshed")
            batches = list(iter_operational_validation_batches(source, 3))
            self.assertGreater(len(batches), 1)
            self.assertEqual(
                sum(len(labels) for _, labels in batches),
                sum(row["partition_name"] == "validation" for row in rows),
            )

    def test_full_operational_metrics_are_final_point_only(self):
        """Evaluate all-text validation only at the final curve fraction."""

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "development.parquet"
            artifact = directory / "full.json"
            scientific = directory / "scientific.parquet"
            pq.write_table(pa.Table.from_pylist(self._rows()), source)
            result = run_s3_full(
                source,
                artifact,
                scientific,
                config=BaselineConfig(
                    max_features=100,
                    min_df=1,
                    max_iter=20,
                    fractions=(0.5, 1.0),
                ),
                batch_size=3,
                memory_budget_gb=1.0,
            )
            self.assertTrue(result["complete"])
            self._assert_boundary(result["protocol_boundary"])
            artifact_payload = json.loads(artifact.read_text(encoding="utf-8"))
            artifact_payload["protocol_boundary"] = {
                "training_partitions": ["train", "validation"],
                "sealed_partitions": ["test", "stress", "monitor"],
            }
            artifact.write_text(
                json.dumps(artifact_payload), encoding="utf-8"
            )
            migrated = run_s3_full(
                source,
                artifact,
                scientific,
                config=BaselineConfig(
                    max_features=100,
                    min_df=1,
                    max_iter=20,
                    fractions=(0.5, 1.0),
                ),
                batch_size=3,
                memory_budget_gb=1.0,
            )
            self._assert_boundary(migrated["protocol_boundary"])
            persisted = json.loads(artifact.read_text(encoding="utf-8"))
            self._assert_boundary(persisted["protocol_boundary"])
            frame = read_scientific_frame(scientific)
            train_ids = {
                int(value.as_py())
                for index, value in enumerate(frame["Complaint ID"])
                if frame["partition_name"][index].as_py() == "train"
            }
            curve = build_learning_curve_indices(frame, (0.5, 1.0))
            for indices in curve.values():
                selected_ids = {
                    int(frame["Complaint ID"][int(index)].as_py())
                    for index in indices
                }
                self.assertTrue(selected_ids.issubset(train_ids))
            self.assertNotIn(
                "operational_all_text", result["points"]["0.5"]["dummy"]
            )
            self.assertIn(
                "operational_all_text", result["points"]["1"]["dummy"]
            )
            self.assertEqual(
                result["points"]["1"]["dummy"]["operational_all_text"][
                    "row_count"
                ],
                10,
            )


if __name__ == "__main__":
    unittest.main()
