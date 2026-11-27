"""Synthetic tests for the bounded V3 E0 protocol draft."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import consumer_complaint_intelligence.v3_protocol as v3_protocol


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "config" / "v3_development_protocol.json"


class V3ProtocolTests(unittest.TestCase):
    """Verify the draft V3 protocol does not authorize later work."""

    def test_draft_protocol_loads_without_freezing_or_promotion(self) -> None:
        """Load the draft and expose its non-promoted state."""

        protocol = v3_protocol.load_v3_protocol(CONFIG)
        self.assertEqual(protocol.status, "DRAFT_FOR_USER_REVIEW")
        self.assertFalse(protocol.payload["protocol_status"]["frozen"])
        self.assertFalse(
            protocol.payload["protocol_status"]["promotion_authorized"]
        )
        self.assertFalse(
            protocol.payload["protocol_status"]["deployment_authorized"]
        )
        self.assertEqual(protocol.execution_authorized_stage_ids, ())

    def test_candidate_roles_and_model_ids_are_exact(self) -> None:
        """Verify the exact V3 model catalog and role ordering."""

        protocol = v3_protocol.load_v3_protocol(CONFIG)
        expected = (
            ("incumbent", "sklearn.LinearSVC"),
            (
                "primary_frozen_embedding",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
            (
                "direct_frozen_embedding_competitor",
                "BAAI/bge-small-en-v1.5",
            ),
            ("score_fusion", "v3_score_fusion"),
            (
                "quality_ceiling",
                "sentence-transformers/all-mpnet-base-v2",
            ),
            (
                "optional_supervised_fine_tune",
                "winner_from_frozen_embedding_stage",
            ),
        )
        self.assertEqual(protocol.candidate_models, expected)

    def test_input_language_contract_is_en_us(self) -> None:
        """Keep the estimator input contract explicitly en-US."""

        protocol = v3_protocol.load_v3_protocol(CONFIG)
        self.assertEqual(protocol.input_language, "en-US")

        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        payload["objective"]["input_language"] = "pt-BR"
        with self.assertRaisesRegex(ValueError, "en-US"):
            v3_protocol.validate_v3_protocol(payload)

    def test_kaggle_gpu_policy_is_t4_only_and_rejects_p100(self) -> None:
        """Accept Kaggle T4 names and reject the prior P100 failure path."""

        protocol = v3_protocol.load_v3_protocol(CONFIG)
        self.assertEqual(
            v3_protocol.require_kaggle_t4_gpu("NVIDIA T4", protocol),
            "NVIDIA T4",
        )
        self.assertEqual(
            v3_protocol.require_kaggle_t4_gpu("Tesla T4", protocol),
            "Tesla T4",
        )
        for gpu_name in ("Tesla P100", "NVIDIA P100", "sm_60"):
            with self.assertRaises(ValueError):
                v3_protocol.require_kaggle_t4_gpu(gpu_name, protocol)

        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        payload["execution"]["accepted_gpu_names"].append("Tesla P100")
        with self.assertRaises(ValueError):
            v3_protocol.validate_v3_protocol(payload)

    def test_e1_authorizes_only_synthetic_smoke_design_review(self) -> None:
        """Authorize E1 design review without execution or dataset loading."""

        protocol = v3_protocol.load_v3_protocol(CONFIG)
        action = v3_protocol.require_e1_smoke_design_review(
            protocol,
            stage_id="E1",
            action="synthetic_model_loading_smoke_design_review",
            uses_project_dataset=False,
        )
        self.assertEqual(action, "synthetic_model_loading_smoke_design_review")
        self.assertEqual(
            protocol.payload["authorization"]["execution_authorized_actions"],
            [],
        )

        with self.assertRaises(ValueError):
            v3_protocol.require_e1_smoke_design_review(
                protocol,
                stage_id="E1",
                action="model_training",
                uses_project_dataset=False,
            )
        with self.assertRaises(ValueError):
            v3_protocol.require_e1_smoke_design_review(
                protocol,
                stage_id="E1",
                action="synthetic_model_loading_smoke_design_review",
                uses_project_dataset=True,
            )

    def test_later_stage_sequence_is_declared_but_not_authorized(self) -> None:
        """Keep E2 through packaging planned and blocked by review gates."""

        protocol = v3_protocol.load_v3_protocol(CONFIG)
        self.assertEqual(protocol.design_authorized_stage_ids, ("E1",))
        self.assertEqual(protocol.execution_authorized_stage_ids, ())
        stage_ids = tuple(stage["id"] for stage in protocol.payload["stages"])
        self.assertEqual(
            stage_ids,
            ("E0", "E1", "E2", "E3", "E4", "E5", "E6", "P"),
        )

        e1 = protocol.payload["stages"][1]
        self.assertTrue(e1["design_authorized"])
        self.assertFalse(e1["execution_authorized"])

        for stage in protocol.payload["stages"][2:]:
            self.assertFalse(stage["design_authorized"])
            self.assertFalse(stage["execution_authorized"])
            self.assertIn("blocked_until", stage)

        with self.assertRaises(ValueError):
            v3_protocol.require_e1_smoke_design_review(
                protocol,
                stage_id="E2",
                action="synthetic_model_loading_smoke_design_review",
                uses_project_dataset=False,
            )

    def test_consumed_and_monitor_partitions_are_distinctly_rejected(self) -> None:
        """Reject archived test/stress and diagnostic-only monitor."""

        self.assertEqual(
            v3_protocol.require_development_partition("train"),
            "train",
        )
        self.assertEqual(
            v3_protocol.require_development_partition("validation"),
            "validation",
        )
        for partition in ("test", "stress"):
            with self.assertRaisesRegex(ValueError, "Consumed partition"):
                v3_protocol.require_development_partition(partition)

        with self.assertRaisesRegex(ValueError, "Monitor partition"):
            v3_protocol.require_development_partition("monitor")

        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(
            payload["data_boundary"]["consumed_archived_partitions"],
            ["test", "stress"],
        )
        self.assertEqual(
            payload["data_boundary"]["never_reusable_in_v3_partitions"],
            ["test", "stress"],
        )
        self.assertEqual(
            payload["data_boundary"]["monitor_partition"]["status"],
            "SEALED_DIAGNOSTIC_ONLY",
        )
        self.assertNotIn("sealed_use", payload["data_boundary"])

    def test_mutated_candidate_catalog_is_rejected(self) -> None:
        """Reject role, model-id, and E0-training drift in the catalog."""

        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        changed_model = copy.deepcopy(payload)
        changed_model["candidate_catalog"][1]["model_id"] = "other/model"
        with self.assertRaises(ValueError):
            v3_protocol.validate_v3_protocol(changed_model)

        training_enabled = copy.deepcopy(payload)
        training_enabled["candidate_catalog"][2]["training_allowed_in_e0"] = True
        with self.assertRaises(ValueError):
            v3_protocol.validate_v3_protocol(training_enabled)

    def test_e1_synthetic_contract_and_long_text_strategy_are_strict(
        self,
    ) -> None:
        """Reject project data in E1 and preserve the later E4 text test."""

        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        dataset_enabled = copy.deepcopy(payload)
        dataset_enabled["e1_smoke"]["project_dataset_required"] = True
        with self.assertRaises(ValueError):
            v3_protocol.validate_v3_protocol(dataset_enabled)

        execution_enabled = copy.deepcopy(payload)
        execution_enabled["e1_smoke"]["execution_allowed"] = True
        with self.assertRaises(ValueError):
            v3_protocol.validate_v3_protocol(execution_enabled)

        missing_long_text_gate = copy.deepcopy(payload)
        missing_long_text_gate["long_text_strategy"]["required_stage"] = "E3"
        with self.assertRaises(ValueError):
            v3_protocol.validate_v3_protocol(missing_long_text_gate)

    def test_legacy_blanket_sealed_use_and_execution_auth_are_rejected(
        self,
    ) -> None:
        """Reject ambiguous sealed-use and old execution-like fields."""

        payload = json.loads(CONFIG.read_text(encoding="utf-8"))

        blanket_rule = copy.deepcopy(payload)
        blanket_rule["data_boundary"]["sealed_use"] = (
            "diagnostic_only_after_separate_user_approval"
        )
        with self.assertRaisesRegex(ValueError, "blanket sealed_use"):
            v3_protocol.validate_v3_protocol(blanket_rule)

        old_authorization = copy.deepcopy(payload)
        old_authorization["authorization"]["authorized_stage_ids"] = ["E1"]
        with self.assertRaisesRegex(ValueError, "design and execution"):
            v3_protocol.validate_v3_protocol(old_authorization)

        execution_authorized = copy.deepcopy(payload)
        execution_authorized["authorization"]["execution_authorized_actions"] = [
            "synthetic_model_loading_smoke_execution"
        ]
        with self.assertRaises(ValueError):
            v3_protocol.validate_v3_protocol(execution_authorized)

    def test_invalid_json_shape_is_rejected_deterministically(self) -> None:
        """Reject non-object protocol files through the public loader."""

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "v3.json"
            path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "JSON object"):
                v3_protocol.load_v3_protocol(path)


if __name__ == "__main__":
    unittest.main()
