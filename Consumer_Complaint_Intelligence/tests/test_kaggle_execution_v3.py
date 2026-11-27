"""Operational checks for the V3 E1 Kaggle smoke scaffold.

These tests intentionally avoid torch, sentence-transformers, and network
access. Runtime behavior is exercised with small fakes so the execution
contract stays reviewable before any Kaggle run is approved.
"""

from __future__ import annotations

import copy
import inspect
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

from consumer_complaint_intelligence import v3_kaggle as v3


ROOT = Path(__file__).parents[1]
CONFIG_PATH = ROOT / "config" / "v3_e1_execution.json"
KERNEL_DIR = ROOT / "kaggle" / "kernel-v3-e1"
DATASET_DIR = ROOT / "kaggle" / "dataset-v3-e1"


def _load_config() -> dict:
    """Return the draft V3 E1 config from disk."""

    return v3.load_v3_e1_config(CONFIG_PATH)


class FakeCuda:
    """Small CUDA surface used by the fake torch module."""

    def is_available(self) -> bool:
        """Pretend CUDA is available."""

        return True

    def get_device_name(self, index: int) -> str:
        """Return the configured fake GPU name."""

        _ = index
        return self.device_name

    def get_device_capability(self, index: int) -> tuple[int, int]:
        """Return T4 compute capability for the fake GPU."""

        _ = index
        return self.capability

    def synchronize(self) -> None:
        """Mirror torch.cuda.synchronize without work."""

    def max_memory_allocated(self) -> int:
        """Return a small fake peak memory value."""

        return 2 * 1024 * 1024


class FakeEmbedding:
    """Small embedding layer surface used by the fake torch module."""

    def __init__(self, rows: int, dims: int) -> None:
        """Store layer shape for parity with torch.nn.Embedding."""

        self.rows = rows
        self.dims = dims

    def to(self, device: str) -> "FakeEmbedding":
        """Return self after accepting a fake device."""

        _ = device
        return self

    def __call__(self, indexes: np.ndarray) -> np.ndarray:
        """Return a zero tensor shaped like an embedding lookup."""

        return np.zeros((*indexes.shape, self.dims), dtype=np.float32)


class FakeNN:
    """Namespace holding the fake embedding class."""

    Embedding = FakeEmbedding


class FakeTorch:
    """Tiny torch-compatible surface for the V3 E1 GPU probe."""

    __version__ = "fake-torch"
    long = "long"

    class version:
        """Fake torch.version namespace."""

        cuda = "12.4"

    def __init__(
        self,
        device_name: str = "NVIDIA Tesla T4",
        capability: tuple[int, int] = (7, 5),
    ) -> None:
        """Create a fake torch module reporting one device name."""

        self.cuda = FakeCuda()
        self.cuda.device_name = device_name
        self.cuda.capability = capability
        self.nn = FakeNN()

    def device(self, name: str) -> str:
        """Return the requested fake device name."""

        return name

    def ones(self, shape: tuple[int, int], device: str) -> np.ndarray:
        """Return a fake tensor of ones."""

        _ = device
        return np.ones(shape, dtype=np.float32)

    def zeros(
        self,
        shape: tuple[int, int],
        dtype: str,
        device: str,
    ) -> np.ndarray:
        """Return a fake integer tensor of zeros."""

        _ = (dtype, device)
        return np.zeros(shape, dtype=np.int64)

    def mm(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Run a real NumPy matrix multiplication."""

        return left @ right


class FakeSentenceModel:
    """Deterministic fake sentence encoder returning unit vectors."""

    def __init__(self, model_id: str, device: str, revision: str) -> None:
        """Store model construction arguments for inspection."""

        self.model_id = model_id
        self.device = device
        self.revision = revision

    def encode(
        self,
        texts: list[str],
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> np.ndarray:
        """Return deterministic normalized vectors without persisting text."""

        self._assert_encode_options(
            normalize_embeddings,
            convert_to_numpy,
            show_progress_bar,
        )
        rows = np.zeros((len(texts), 384), dtype=np.float32)
        rows[:, 0] = 1.0
        return rows

    def _assert_encode_options(
        self,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> None:
        self._last_options = (
            normalize_embeddings,
            convert_to_numpy,
            show_progress_bar,
        )


class ConfigContractTests(unittest.TestCase):
    """Verify the V3 E1 config is a draft scaffold, not an approval."""

    def test_config_is_draft_and_uses_only_two_smoke_models(self) -> None:
        """Require draft status, T4, and the approved 384-dim candidates."""

        config = _load_config()
        self.assertEqual(config["status"], "DRAFT")
        self.assertIs(config["approved_for_kaggle_execution"], False)
        self.assertEqual(config["kaggle"]["machine_shape"], "NvidiaTeslaT4")
        self.assertEqual(
            tuple(model["model_id"] for model in config["models"]),
            v3.V3_E1_ALLOWED_MODELS,
        )
        self.assertTrue(
            all(model["expected_embedding_dim"] == 384 for model in config["models"])
        )

    def test_config_rejects_status_approval_drift(self) -> None:
        """Reject one-field drift between status and approval."""

        config = _load_config()
        config["approved_for_kaggle_execution"] = True
        with self.assertRaises(ValueError):
            v3.validate_v3_e1_config(config)
        config = _approved_config()
        config["approved_for_kaggle_execution"] = False
        with self.assertRaises(ValueError):
            v3.validate_v3_e1_config(config)

    def test_draft_config_blocks_public_runtime_before_model_load(self) -> None:
        """Ensure approval failure happens before any model factory is called."""

        called = []

        def factory(*args, **kwargs):
            called.append((args, kwargs))
            raise AssertionError("model factory must not be called")

        with self.assertRaises(PermissionError):
            v3.run_v3_e1_smoke(
                _load_config(),
                model_factory=factory,
                torch_module=FakeTorch(),
                revision_resolver=lambda model_id: "sha",
            )
        self.assertEqual(called, [])


class RuntimeGateTests(unittest.TestCase):
    """Verify the GPU gate rejects non-T4 runtime conditions."""

    def test_t4_fake_gpu_passes_real_kernel_probe_contract(self) -> None:
        """Accept a fake T4 only after matmul and embedding probes pass."""

        report = v3.assert_v3_e1_t4_gpu(_approved_config(), FakeTorch())
        self.assertTrue(report["cuda_usable"])
        self.assertTrue(report["device_is_t4"])
        self.assertTrue(report["matmul_probe_ok"])
        self.assertTrue(report["embedding_probe_ok"])

    def test_p100_config_is_rejected_even_if_fake_cuda_works(self) -> None:
        """Reject forbidden Kaggle P100 before relying on CUDA success."""

        config = _approved_config()
        config["kaggle"]["machine_shape"] = "NvidiaTeslaP100"
        with self.assertRaises(ValueError):
            v3.assert_v3_e1_t4_gpu(config, FakeTorch())

    def test_non_t4_device_name_is_rejected(self) -> None:
        """Require the observed GPU name to exactly match an accepted T4."""

        with self.assertRaises(ValueError):
            v3.assert_v3_e1_t4_gpu(
                _approved_config(),
                FakeTorch("NVIDIA Tesla P100"),
            )

    def test_t4_substring_device_name_is_rejected(self) -> None:
        """Reject names that merely contain the T4 token."""

        with self.assertRaises(ValueError):
            v3.assert_v3_e1_t4_gpu(
                _approved_config(),
                FakeTorch("NVIDIA Tesla T4 Preview"),
            )

    def test_wrong_compute_capability_is_rejected(self) -> None:
        """Require the observed compute capability to be sm_75."""

        with self.assertRaises(ValueError):
            v3.assert_v3_e1_t4_gpu(
                _approved_config(),
                FakeTorch(capability=(8, 0)),
            )


class SmokeResultTests(unittest.TestCase):
    """Verify approved smoke output remains aggregate-only."""

    def test_approved_smoke_uses_fake_models_and_validates_output(self) -> None:
        """Run the full scaffold with fakes and validate the result contract."""

        loaded = []

        def factory(
            model_id: str,
            device: str,
            revision: str,
        ) -> FakeSentenceModel:
            loaded.append((model_id, device, revision))
            return FakeSentenceModel(model_id, device, revision)

        config = _approved_config()
        with mock.patch.object(
            v3,
            "package_version_report",
            return_value=_package_report(config),
        ):
            result = v3.run_v3_e1_smoke(
                config,
                model_factory=factory,
                torch_module=FakeTorch(),
                revision_resolver=lambda model_id: f"sha-{model_id}",
            )
        summary = v3.validate_v3_e1_result(result, config)
        self.assertEqual(summary["status"], "VALID")
        self.assertEqual(len(loaded), 2)
        self.assertTrue(all(call[2].startswith("sha-") for call in loaded))
        self.assertNotIn("\"synthetic_batch\":", json.dumps(result))
        self.assertNotIn("\"embeddings\"", json.dumps(result))

    def test_missing_revision_fails_before_model_factory(self) -> None:
        """Require resolved HF SHA before constructing a model."""

        called = []
        with self.assertRaises(ValueError):
            v3.smoke_one_v3_e1_model(
                _approved_config()["models"][0],
                _approved_config()["synthetic_batch"],
                _approved_config()["checks"],
                lambda *args, **kwargs: called.append((args, kwargs)),
                lambda model_id: "",
            )
        self.assertEqual(called, [])

    def test_result_validation_rejects_persisted_text(self) -> None:
        """Reject accidental text leakage in aggregate evidence."""

        result = {
            "schema_version": v3.V3_E1_SCHEMA,
            "stage": "V3-E1",
            "status": "COMPLETE",
            "complete": True,
            "texts": ["should not be here"],
        }
        with self.assertRaises(ValueError):
            v3.validate_v3_e1_result(result)

    def test_result_validation_rejects_metric_and_package_drift(self) -> None:
        """Reject shape, norm, determinism, and package-version drift."""

        result = _valid_result(_approved_config())
        result["models"][0]["observed_shape"] = [99, 384]
        with self.assertRaises(ValueError):
            v3.validate_v3_e1_result(result, _approved_config())
        result = _valid_result(_approved_config())
        result["models"][0]["norm_min"] = 0.5
        with self.assertRaises(ValueError):
            v3.validate_v3_e1_result(result, _approved_config())
        result = _valid_result(_approved_config())
        result["models"][0]["deterministic_max_abs_diff"] = 0.1
        with self.assertRaises(ValueError):
            v3.validate_v3_e1_result(result, _approved_config())
        result = _valid_result(_approved_config())
        result["environment"]["packages"]["sentence-transformers"] = "0.0.0"
        with self.assertRaises(ValueError):
            v3.validate_v3_e1_result(result, _approved_config())
        result = _valid_result(_approved_config())
        result["models"][0]["resolved_hf_revision_sha"] = " "
        with self.assertRaises(ValueError):
            v3.validate_v3_e1_result(result, _approved_config())

    def test_result_validation_rejects_gpu_and_output_contract_drift(self) -> None:
        """Reject non-strict T4 identity and aggregate flag drift."""

        result = _valid_result(_approved_config())
        result["environment"]["gpu"]["device_name"] = "NVIDIA Tesla T4 Preview"
        with self.assertRaises(ValueError):
            v3.validate_v3_e1_result(result, _approved_config())
        result = _valid_result(_approved_config())
        result["environment"]["gpu"]["compute_capability"] = "sm_80"
        with self.assertRaises(ValueError):
            v3.validate_v3_e1_result(result, _approved_config())
        result = _valid_result(_approved_config())
        result["output_contract"]["persisted_embeddings"] = True
        with self.assertRaises(ValueError):
            v3.validate_v3_e1_result(result, _approved_config())


class BundleContractTests(unittest.TestCase):
    """Verify the V3 E1 Kaggle bundle stays tiny and data-free."""

    def test_bundle_manifest_contains_only_allowed_files(self) -> None:
        """Require only license, pyproject, config, package init, and helper."""

        manifest = v3.build_v3_e1_bundle_manifest(ROOT)
        self.assertFalse(manifest["contains_project_data"])
        self.assertFalse(manifest["contains_scientific_cache"])
        self.assertFalse(manifest["contains_artifacts"])
        self.assertEqual(
            tuple(sorted(manifest["files"])),
            tuple(sorted(v3.V3_E1_BUNDLE_FILES)),
        )
        for name in manifest["files"]:
            parts = set(Path(name).parts)
            forbidden = parts.intersection(v3.V3_E1_FORBIDDEN_BUNDLE_PARTS)
            self.assertEqual(forbidden, set())

    def test_write_bundle_creates_manifest_and_zip(self) -> None:
        """Create a review bundle in a temporary directory."""

        with tempfile.TemporaryDirectory() as directory:
            summary = v3.write_v3_e1_bundle(ROOT, directory)
            self.assertTrue(Path(summary["zip_path"]).is_file())
            self.assertTrue(Path(summary["manifest_path"]).is_file())
            self.assertEqual(summary["file_count"], len(v3.V3_E1_BUNDLE_FILES))


class KaggleMetadataTests(unittest.TestCase):
    """Verify Kaggle metadata pins the required private T4 execution."""

    def test_dataset_metadata_uses_expected_slug(self) -> None:
        """Require the exact V3 E1 dataset slug."""

        meta = json.loads(
            (DATASET_DIR / "dataset-metadata.json").read_text(encoding="utf-8")
        )
        self.assertEqual(meta["id"], "eniorubens/cci-v3-e1-bundle")
        self.assertEqual(meta["licenses"], [{"name": "other"}])
        self.assertIn("MIT-licensed", meta["description"])

    def test_kernel_metadata_is_private_t4_and_internet_enabled(self) -> None:
        """Require T4, internet, and the V3 E1 code bundle as source."""

        meta = json.loads(
            (KERNEL_DIR / "kernel-metadata.json").read_text(encoding="utf-8")
        )
        self.assertEqual(meta["machine_shape"], "NvidiaTeslaT4")
        self.assertIs(meta["is_private"], True)
        self.assertIs(meta["enable_gpu"], True)
        self.assertIs(meta["enable_internet"], True)
        self.assertEqual(meta["dataset_sources"], ["eniorubens/cci-v3-e1-bundle"])
        self.assertNotIn("P100", json.dumps(meta))


class NotebookContractTests(unittest.TestCase):
    """Verify the Kaggle notebook is compact orchestration only."""

    def test_notebook_parses_and_preserves_gate_order(self) -> None:
        """Require approval and T4 gates before the smoke runner call."""

        notebook = json.loads(
            (KERNEL_DIR / "v3_e1_embedding_smoke.ipynb").read_text(
                encoding="utf-8"
            )
        )
        code_cells = [
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        code = "\n".join(code_cells)
        bundle = code.index("stage_v3_e1_code")
        approval = code.index("require_v3_e1_approval")
        gpu = code.index("assert_v3_e1_t4_gpu")
        runner = code.index("run_v3_e1_smoke")
        self.assertLess(bundle, approval)
        self.assertLess(approval, gpu)
        self.assertLess(gpu, runner)
        self.assertNotIn("SentenceTransformer", code)
        self.assertNotIn("all-mpnet", code)


class ScriptContractTests(unittest.TestCase):
    """Verify local scripts preserve the draft execution lock."""

    def test_push_script_checks_token_and_refuses_draft_before_build(self) -> None:
        """Require KGAT token check and draft refusal before bundle build."""

        script = (ROOT / "scripts" / "push_kaggle_v3_e1.ps1").read_text(
            encoding="utf-8"
        )
        self.assertIn(".kaggle\\access_token", script)
        self.assertIn("KGAT_", script)
        token_read = script.index("Get-Content -LiteralPath $tokenPath")
        approval = script.index("approved_for_kaggle_execution")
        build = script.index("build_kaggle_v3_e1_bundle.py")
        publish = script.index("Publish-Dataset")
        self.assertLess(approval, build)
        self.assertLess(approval, publish)
        self.assertLess(approval, token_read)
        self.assertIn("Remove-Variable -Name token -Force", script)
        self.assertNotIn("Write-Output $token", script)


class PublicApiDocumentationTests(unittest.TestCase):
    """Keep the new V3 E1 public Python API documented."""

    def test_public_v3_kaggle_definitions_have_docstrings(self) -> None:
        """Require non-empty docstrings on public V3 E1 functions/classes."""

        missing = []
        for name, member in inspect.getmembers(v3):
            if name.startswith("_"):
                continue
            if inspect.isfunction(member) or inspect.isclass(member):
                if getattr(member, "__module__", None) == v3.__name__:
                    if not inspect.getdoc(member):
                        missing.append(f"{v3.__name__}.{name}")
        self.assertEqual(missing, [])


def _approved_config() -> dict:
    """Return a copy of the V3 E1 config with execution approval enabled."""

    config = copy.deepcopy(_load_config())
    config["approved_for_kaggle_execution"] = True
    config["status"] = v3.V3_E1_APPROVED_STATUS
    return config


def _package_report(config: dict) -> dict[str, str]:
    """Return the configured package pin report for fake smoke tests."""

    return {
        "torch": "fake-torch",
        "transformers": "fake-transformers",
        "sentence-transformers": config["package_policy"]["sentence-transformers"],
        "huggingface_hub": "fake-hub",
    }


def _valid_result(config: dict) -> dict:
    """Return one config-aligned aggregate V3 E1 result."""

    batch_size = len(config["synthetic_batch"])
    models = []
    for model in config["models"]:
        models.append(
            {
                "candidate_id": model["candidate_id"],
                "model_id": model["model_id"],
                "expected_embedding_dim": model["expected_embedding_dim"],
                "observed_shape": [batch_size, 384],
                "synthetic_batch_size": batch_size,
                "all_finite": True,
                "norm_min": 1.0,
                "norm_mean": 1.0,
                "norm_max": 1.0,
                "deterministic_max_abs_diff": 0.0,
                "determinism_tolerance": 0.000001,
                "duration_seconds": 0.1,
                "peak_cuda_memory_mb": 1.0,
                "resolved_hf_revision_sha": f"sha-{model['candidate_id']}",
                "passed": True,
            }
        )
    return {
        "schema_version": v3.V3_E1_SCHEMA,
        "stage": "V3-E1",
        "status": "COMPLETE",
        "language": "en-US",
        "approved_for_kaggle_execution": True,
        "environment": {
            "python_version": "3.11",
            "packages": _package_report(config),
            "gpu": {
                "cuda_available": True,
                "cuda_usable": True,
                "device_name": "NVIDIA Tesla T4",
                "compute_capability": "sm_75",
                "device_is_t4": True,
                "matmul_probe_ok": True,
                "embedding_probe_ok": True,
                "cuda_error": None,
                "cuda_runtime_version": "12.4",
                "torch_version": "fake-torch",
            },
        },
        "models": models,
        "complete": True,
        "output_contract": {
            "aggregate_only": True,
            "persisted_embeddings": False,
            "persisted_texts": False,
            "persisted_model_cache": False,
            "persisted_fitted_weights": False,
        },
    }


if __name__ == "__main__":
    unittest.main()
