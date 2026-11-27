"""Tests for the synthetic-only V3-E1 GPU smoke boundary."""

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from consumer_complaint_intelligence.v3 import smoke


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "config" / "v3" / "smoke.json"
PROTOCOL = ROOT / "config" / "v3" / "development_protocol.json"


class _FakeEmbeddings:
    """Expose only the aggregate properties consumed by the smoke."""

    def __init__(self, rows: int) -> None:
        self.shape = (rows, 384)
        self.dtype = "float32"


class _FakeEncoder:
    """Return bounded shape metadata instead of real embeddings."""

    def encode(self, texts, **kwargs):
        del kwargs
        return _FakeEmbeddings(len(texts))


class _FakeCuda:
    """Implement the CUDA measurement surface used by the smoke."""

    @staticmethod
    def empty_cache() -> None:
        """Match the torch CUDA API."""

    @staticmethod
    def reset_peak_memory_stats() -> None:
        """Match the torch CUDA API."""

    @staticmethod
    def synchronize() -> None:
        """Match the torch CUDA API."""

    @staticmethod
    def max_memory_allocated() -> int:
        """Return one deterministic diagnostic value."""

        return 128 * 1024**2

    @staticmethod
    def max_memory_reserved() -> int:
        """Return one deterministic diagnostic value."""

        return 192 * 1024**2

    @staticmethod
    def get_device_properties(index: int):
        """Return a synthetic 4 GiB CUDA device."""

        if index != 0:
            raise ValueError("unexpected fake CUDA index")
        return SimpleNamespace(name="Synthetic CUDA", total_memory=4096 * 1024**2)


FAKE_TORCH = SimpleNamespace(
    cuda=_FakeCuda(),
    version=SimpleNamespace(cuda="13.0"),
)


class V3SmokeTests(unittest.TestCase):
    """Keep E1 technical, synthetic, pinned, and non-selecting."""

    def setUp(self) -> None:
        """Load a detached smoke payload for mutation tests."""

        self.payload = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_config_is_synthetic_only_and_pinned(self) -> None:
        """Load the exact 10k-row synthetic MiniLM smoke."""

        config = smoke.load_smoke_config(CONFIG)
        self.assertEqual(config.row_count, 10000)
        self.assertEqual(config.batch_sizes, (8, 16, 32, 64, 128))
        self.assertEqual(
            config.revision,
            "bc57282bc374d33e0d6c4de27f12dc1c2a87f37a",
        )

    def test_scientific_data_or_selection_flags_are_rejected(self) -> None:
        """Prevent a smoke config from becoming a scientific runner."""

        for key in ("selection_authorized", "scientific_execution_authorized"):
            changed = copy.deepcopy(self.payload)
            changed[key] = True
            with self.assertRaises(ValueError):
                smoke.validate_smoke_config(changed)

        changed = copy.deepcopy(self.payload)
        changed["input"]["scientific_partition_access"] = True
        with self.assertRaisesRegex(ValueError, "scientific partitions"):
            smoke.validate_smoke_config(changed)

    def test_synthetic_texts_are_deterministic_and_span_lengths(self) -> None:
        """Cover short, medium, and deliberately truncated-risk profiles."""

        first = smoke.build_synthetic_texts(6)
        second = smoke.build_synthetic_texts(6)
        self.assertEqual(first, second)
        lengths = [len(value.split()) for value in first[:3]]
        self.assertLess(lengths[0], lengths[1])
        self.assertLess(lengths[1], lengths[2])

    def test_fake_smoke_writes_only_aggregate_diagnostic_result(self) -> None:
        """Exercise orchestration without torch, a model download, or data."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "config" / "v3"
            target.mkdir(parents=True)
            shutil.copyfile(CONFIG, target / "smoke.json")
            shutil.copyfile(PROTOCOL, target / "development_protocol.json")

            def loader(config, cache):
                self.assertTrue(cache.is_relative_to(root))
                self.assertEqual(config.model_id, self.payload["model"]["model_id"])
                return _FakeEncoder(), FAKE_TORCH

            synthetic = tuple("synthetic complaint" for _ in range(10000))
            with mock.patch.object(smoke, "build_synthetic_texts", return_value=synthetic):
                result = smoke.run_synthetic_gpu_smoke(
                    root,
                    encoder_loader=loader,
                )

            self.assertEqual(result["status"], "TECHNICAL_DIAGNOSTIC_ONLY")
            self.assertFalse(result["selection_authorized"])
            self.assertFalse(result["scientific_partition_access"])
            self.assertEqual(result["selected_batch_size"], 128)
            self.assertEqual(result["full_benchmark"]["embedding_shape"], [10000, 384])
            persisted = json.loads(
                (root / "temp" / "v3" / "v3_e1_gpu_smoke.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(persisted["input"]["source"], (
                "deterministic_synthetic_text_only"
            ))
            self.assertNotIn("texts", persisted)
            self.assertNotIn("embeddings", persisted)

    def test_persisted_smoke_manifest_matches_private_aggregate_result(self) -> None:
        """Verify the published technical result hash and privacy contract."""

        manifest = smoke.validate_smoke_manifest(
            ROOT / "config" / "v3" / "smoke_results.json",
            ROOT,
        )
        self.assertTrue(manifest["complete"])
        self.assertFalse(manifest["selection_authorized"])
        self.assertFalse(manifest["scientific_partition_access"])

    def test_individual_outputs_are_rejected_from_result(self) -> None:
        """Reject texts, labels, predictions, scores, or embeddings by key."""

        config = smoke.load_smoke_config(CONFIG)
        result = json.loads(
            (ROOT / "temp" / "v3" / "v3_e1_gpu_smoke.json").read_text(
                encoding="utf-8"
            )
        )
        result["texts"] = ["not allowed"]
        with self.assertRaisesRegex(ValueError, "individual model data"):
            smoke.validate_smoke_result(result, config)


if __name__ == "__main__":
    unittest.main()
