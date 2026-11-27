"""Kaggle scaffolding for the V3 E1 embedding smoke experiment.

The E1 stage is deliberately narrow: it checks that the selected Kaggle
T4 runtime can load two frozen sentence-transformer encoders and produce
stable normalized embeddings for a synthetic en-US batch. It never stages
project data, trains a model, sets thresholds, or promotes an estimator.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

V3_E1_SCHEMA = "cci-v3-e1-execution-v1"
V3_E1_MANIFEST_SCHEMA = "cci-v3-e1-bundle-manifest-v1"
V3_E1_CONFIG = "config/v3_e1_execution.json"
V3_E1_BUNDLE_MANIFEST = "v3_e1_bundle_manifest.json"
V3_E1_BUNDLE_ZIP = "cci-v3-e1-bundle.zip"
V3_E1_OUTPUT = "v3_e1_smoke.json"
V3_E1_DRAFT_STATUS = "DRAFT"
V3_E1_APPROVED_STATUS = "APPROVED_FOR_MANUAL_KAGGLE_RUN"
V3_E1_ALLOWED_MODELS = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
)
V3_E1_T4_DEVICE_NAMES = ("Tesla T4", "NVIDIA Tesla T4", "NVIDIA T4")
V3_E1_T4_COMPUTE_CAPABILITY = "sm_75"
V3_E1_BUNDLE_FILES = (
    "LICENSE",
    "pyproject.toml",
    V3_E1_CONFIG,
    "src/consumer_complaint_intelligence/__init__.py",
    "src/consumer_complaint_intelligence/v3_kaggle.py",
)
V3_E1_FORBIDDEN_BUNDLE_PARTS = {
    ".git",
    ".ipynb_checkpoints",
    "artifacts",
    "data",
    "notebooks",
    "temp",
    "__pycache__",
}
V3_E1_RESULT_FORBIDDEN_KEYS = {
    "embedding",
    "embeddings",
    "model_cache",
    "synthetic_batch",
    "texts",
    "vectors",
}

ProgressCallback = Callable[[str], None]
ModelFactory = Callable[..., Any]
RevisionResolver = Callable[[str], str | None]


def load_v3_e1_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the V3 E1 execution scaffold.

    Args:
        path: JSON config path.

    Returns:
        Parsed execution config.

    Raises:
        ValueError: If the config is not a valid E1 scaffold.
    """

    config_path = Path(path).expanduser().resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    validate_v3_e1_config(payload)
    return payload


def validate_v3_e1_config(config: Mapping[str, Any]) -> None:
    """Validate the static V3 E1 execution contract.

    Args:
        config: Parsed execution config.

    Raises:
        ValueError: If a required gate, model, or output contract drifts.
    """

    if config.get("schema_version") != V3_E1_SCHEMA:
        raise ValueError("Unexpected V3 E1 schema_version")
    if config.get("stage") != "V3-E1":
        raise ValueError("V3 E1 config must declare stage V3-E1")
    _validate_status_approval_pair(config)
    if config.get("language") != "en-US":
        raise ValueError("V3 E1 accepts synthetic en-US text only")
    kaggle = _mapping(config, "kaggle")
    if kaggle.get("machine_shape") != "NvidiaTeslaT4":
        raise ValueError("V3 E1 must pin Kaggle machine_shape to T4")
    if kaggle.get("internet_enabled") is not True:
        raise ValueError("V3 E1 requires Kaggle internet for HF downloads")
    accelerator = _mapping(config, "accelerator")
    accepted = tuple(accelerator.get("accepted_device_names", ()))
    if accepted != V3_E1_T4_DEVICE_NAMES:
        raise ValueError("V3 E1 must list exact accepted T4 device names")
    if accelerator.get("required_compute_capability") != V3_E1_T4_COMPUTE_CAPABILITY:
        raise ValueError("V3 E1 must require T4 compute capability sm_75")
    if "NvidiaTeslaP100" not in accelerator.get(
        "forbidden_machine_shapes", ()
    ):
        raise ValueError("V3 E1 must explicitly forbid Kaggle P100")
    batch = config.get("synthetic_batch")
    if not isinstance(batch, list) or not batch:
        raise ValueError("V3 E1 synthetic_batch must be a non-empty list")
    if not all(isinstance(text, str) and text for text in batch):
        raise ValueError("V3 E1 synthetic_batch must contain non-empty text")
    models = config.get("models")
    if not isinstance(models, list):
        raise ValueError("V3 E1 models must be a list")
    model_ids = tuple(model.get("model_id") for model in models)
    if model_ids != V3_E1_ALLOWED_MODELS:
        raise ValueError(f"Unexpected V3 E1 smoke model list: {model_ids}")
    for model in models:
        if int(model.get("expected_embedding_dim", 0)) != 384:
            raise ValueError("V3 E1 candidate embedding dim must be 384")
    output = _mapping(config, "output_contract")
    for key in (
        "persist_embeddings",
        "persist_texts",
        "persist_model_cache",
        "persist_fitted_weights",
    ):
        if output.get(key) is not False:
            raise ValueError(f"V3 E1 output contract must disable {key}")
    package_policy = _mapping(config, "package_policy")
    if package_policy.get("sentence-transformers") != "5.6.1":
        raise ValueError("V3 E1 must pin sentence-transformers==5.6.1")
    if package_policy.get("install_sentence_transformers_without_deps") is not True:
        raise ValueError("V3 E1 must install sentence-transformers with no deps")


def _validate_status_approval_pair(config: Mapping[str, Any]) -> None:
    status = config.get("status")
    approved_status = config.get("approved_status")
    approved = config.get("approved_for_kaggle_execution")
    if approved_status != V3_E1_APPROVED_STATUS:
        raise ValueError("V3 E1 approved_status drifted")
    if not isinstance(approved, bool):
        raise ValueError("V3 E1 approval flag must be boolean")
    allowed = {V3_E1_DRAFT_STATUS, V3_E1_APPROVED_STATUS}
    if status not in allowed:
        raise ValueError(f"Unexpected V3 E1 status: {status}")
    expected = status == V3_E1_APPROVED_STATUS
    if approved is not expected:
        raise ValueError("V3 E1 status and approval flag disagree")


def require_v3_e1_approval(config: Mapping[str, Any]) -> None:
    """Fail unless the V3 E1 scaffold has explicit execution approval.

    Args:
        config: Parsed execution config.

    Raises:
        PermissionError: If the config is still a draft.
    """

    if config.get("approved_for_kaggle_execution") is not True:
        raise PermissionError(
            "V3 E1 Kaggle execution is not approved. Keep "
            "approved_for_kaggle_execution=false until user review."
        )


def package_version_report() -> dict[str, str | None]:
    """Record package versions required by the V3 E1 evidence contract.

    Returns:
        Mapping from distribution name to the installed version or None.
    """

    packages = {
        "torch": "torch",
        "transformers": "transformers",
        "sentence-transformers": "sentence-transformers",
        "huggingface_hub": "huggingface-hub",
    }
    return {
        output_name: _installed_version(distribution_name)
        for output_name, distribution_name in packages.items()
    }


def report_v3_e1_gpu(torch_module: Any | None = None) -> dict[str, Any]:
    """Report whether the current runtime is a usable NVIDIA T4.

    Args:
        torch_module: Optional fake or imported torch module.

    Returns:
        Aggregate GPU report with real kernel-probe outcomes.
    """

    report: dict[str, Any] = {
        "cuda_available": False,
        "cuda_usable": False,
        "device_name": None,
        "compute_capability": None,
        "device_is_t4": False,
        "matmul_probe_ok": False,
        "embedding_probe_ok": False,
        "cuda_error": None,
        "cuda_runtime_version": None,
        "torch_version": None,
    }
    torch_obj = torch_module
    if torch_obj is None:
        try:
            import torch as torch_obj
        except Exception as error:  # noqa: BLE001
            report["cuda_error"] = f"torch import failed: {error}"
            return report
    report["torch_version"] = str(getattr(torch_obj, "__version__", None))
    torch_version = getattr(torch_obj, "version", None)
    report["cuda_runtime_version"] = str(getattr(torch_version, "cuda", None))
    try:
        report["cuda_available"] = bool(torch_obj.cuda.is_available())
    except Exception as error:  # noqa: BLE001
        report["cuda_error"] = f"cuda probe failed: {error}"
        return report
    if not report["cuda_available"]:
        return report
    try:
        report["device_name"] = str(torch_obj.cuda.get_device_name(0))
        major, minor = torch_obj.cuda.get_device_capability(0)
        report["compute_capability"] = f"sm_{major}{minor}"
        report["device_is_t4"] = (
            report["device_name"] in V3_E1_T4_DEVICE_NAMES
            and report["compute_capability"] == V3_E1_T4_COMPUTE_CAPABILITY
        )
    except Exception as error:  # noqa: BLE001
        report["cuda_error"] = f"device query failed: {error}"
        return report
    try:
        device = torch_obj.device("cuda:0")
        probe = torch_obj.ones((8, 8), device=device)
        torch_obj.mm(probe, probe)
        report["matmul_probe_ok"] = True
        table = torch_obj.nn.Embedding(16, 4).to(device)
        indexes = torch_obj.zeros(
            (2, 3), dtype=torch_obj.long, device=device
        )
        table(indexes)
        report["embedding_probe_ok"] = True
        if hasattr(torch_obj.cuda, "synchronize"):
            torch_obj.cuda.synchronize()
        report["cuda_usable"] = True
    except Exception as error:  # noqa: BLE001
        report["cuda_error"] = f"{type(error).__name__}: {error}"
    return report


def assert_v3_e1_t4_gpu(
    config: Mapping[str, Any],
    torch_module: Any | None = None,
) -> dict[str, Any]:
    """Fail fast unless Kaggle is pinned to a working NVIDIA T4.

    Args:
        config: Parsed execution config.
        torch_module: Optional fake or imported torch module.

    Returns:
        Successful GPU report.

    Raises:
        ValueError: If the configured or observed accelerator is not T4.
    """

    validate_v3_e1_config(config)
    shape = _mapping(config, "kaggle").get("machine_shape")
    accelerator = _mapping(config, "accelerator")
    if shape in set(accelerator.get("forbidden_machine_shapes", ())):
        raise ValueError(f"V3 E1 refuses forbidden accelerator: {shape}")
    if shape != accelerator.get("required_machine_shape"):
        raise ValueError(f"V3 E1 requires Kaggle T4, got: {shape}")
    report = report_v3_e1_gpu(torch_module)
    if not report["cuda_available"]:
        raise ValueError(f"V3 E1 requires CUDA; got: {report}")
    if not report["cuda_usable"]:
        raise ValueError(f"V3 E1 CUDA kernel probe failed: {report}")
    accepted_names = set(accelerator.get("accepted_device_names", ()))
    if report["device_name"] not in accepted_names:
        raise ValueError(f"V3 E1 requires an exact T4 name, got: {report}")
    if report["compute_capability"] != accelerator.get(
        "required_compute_capability"
    ):
        raise ValueError(f"V3 E1 requires T4 sm_75 capability, got: {report}")
    return report


def resolve_hf_model_revision(model_id: str) -> str | None:
    """Resolve a Hugging Face model revision SHA at runtime.

    Args:
        model_id: Hugging Face model identifier.

    Returns:
        Resolved commit SHA when available, otherwise None.
    """

    try:
        from huggingface_hub import model_info
    except Exception:
        return None
    try:
        info = model_info(model_id)
    except Exception:
        return None
    sha = getattr(info, "sha", None)
    return str(sha) if sha else None


def assert_v3_e1_sentence_transformers_pin(config: Mapping[str, Any]) -> None:
    """Fail before model loading if the sentence-transformers pin drifts.

    Args:
        config: Parsed execution config.

    Raises:
        ValueError: If the installed distribution is not the configured pin.
    """

    expected = str(_mapping(config, "package_policy")["sentence-transformers"])
    actual = _installed_version("sentence-transformers")
    if actual != expected:
        raise ValueError(
            "V3 E1 requires sentence-transformers=="
            f"{expected}; installed={actual}"
        )


def smoke_one_v3_e1_model(
    model_config: Mapping[str, Any],
    synthetic_batch: list[str],
    checks: Mapping[str, Any],
    model_factory: ModelFactory,
    revision_resolver: RevisionResolver,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run deterministic aggregate checks for one sentence encoder.

    Args:
        model_config: Candidate model config.
        synthetic_batch: Small neutral en-US batch.
        checks: Numeric validation tolerances.
        model_factory: Callable compatible with ``SentenceTransformer``.
        revision_resolver: Callable returning a model revision SHA.
        progress: Optional callback for notebook-visible progress.

    Returns:
        Aggregate-only evidence for this model.

    Raises:
        ValueError: If shape, finiteness, norms, or determinism fail.
    """

    model_id = str(model_config["model_id"])
    expected_dim = int(model_config["expected_embedding_dim"])
    if progress:
        progress(f"Loading {model_id}")
    start = time.perf_counter()
    _reset_peak_cuda_memory()
    revision = revision_resolver(model_id)
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError(f"Missing resolved HF revision SHA for {model_id}")
    model = model_factory(model_id, device="cuda", revision=revision)
    first = _encode(model, synthetic_batch, checks)
    second = _encode(model, synthetic_batch, checks)
    duration = time.perf_counter() - start
    observed = np.asarray(first, dtype=np.float32)
    repeated = np.asarray(second, dtype=np.float32)
    _validate_embedding_arrays(observed, repeated, expected_dim, checks)
    norms = np.linalg.norm(observed, axis=1)
    max_abs_diff = float(np.max(np.abs(observed - repeated)))
    peak_memory = _peak_cuda_memory_mb()
    result = {
        "candidate_id": model_config.get("candidate_id"),
        "model_id": model_id,
        "expected_embedding_dim": expected_dim,
        "observed_shape": list(observed.shape),
        "synthetic_batch_size": len(synthetic_batch),
        "all_finite": bool(np.isfinite(observed).all()),
        "norm_min": float(np.min(norms)),
        "norm_mean": float(np.mean(norms)),
        "norm_max": float(np.max(norms)),
        "deterministic_max_abs_diff": max_abs_diff,
        "determinism_tolerance": float(
            checks["determinism_tolerance_max_abs_diff"]
        ),
        "duration_seconds": round(duration, 6),
        "peak_cuda_memory_mb": peak_memory,
        "resolved_hf_revision_sha": revision.strip(),
        "passed": True,
    }
    if progress:
        progress(f"Completed {model_id}")
    return result


def run_v3_e1_smoke(
    config: Mapping[str, Any],
    output_path: str | Path | None = None,
    model_factory: ModelFactory | None = None,
    torch_module: Any | None = None,
    revision_resolver: RevisionResolver | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run the approved V3 E1 smoke test and write aggregate evidence.

    Args:
        config: Parsed execution config.
        output_path: Optional JSON path for aggregate evidence.
        model_factory: Optional fake or ``SentenceTransformer`` factory.
        torch_module: Optional fake or imported torch module.
        revision_resolver: Optional Hugging Face revision resolver.
        progress: Optional callback for notebook-visible progress.

    Returns:
        Aggregate-only V3 E1 smoke result.

    Raises:
        PermissionError: If execution is not explicitly approved.
        ValueError: If runtime gates or model checks fail.
    """

    validate_v3_e1_config(config)
    require_v3_e1_approval(config)
    gpu_report = assert_v3_e1_t4_gpu(config, torch_module)
    factory = model_factory
    if factory is None:
        assert_v3_e1_sentence_transformers_pin(config)
        from sentence_transformers import SentenceTransformer

        factory = SentenceTransformer
    resolver = revision_resolver or resolve_hf_model_revision
    batch = list(config["synthetic_batch"])
    checks = _mapping(config, "checks")
    if progress:
        progress("V3 E1 approval and T4 GPU gates passed")
    models = [
        smoke_one_v3_e1_model(
            model,
            batch,
            checks,
            factory,
            resolver,
            progress,
        )
        for model in config["models"]
    ]
    result = {
        "schema_version": V3_E1_SCHEMA,
        "stage": "V3-E1",
        "status": "COMPLETE",
        "language": "en-US",
        "approved_for_kaggle_execution": True,
        "environment": {
            "python_version": platform.python_version(),
            "packages": package_version_report(),
            "gpu": gpu_report,
        },
        "models": models,
        "complete": all(bool(model["passed"]) for model in models),
        "output_contract": {
            "aggregate_only": True,
            "persisted_embeddings": False,
            "persisted_texts": False,
            "persisted_model_cache": False,
            "persisted_fitted_weights": False,
        },
    }
    validate_v3_e1_result(result, config)
    if output_path is not None:
        target = Path(output_path).expanduser().resolve()
        target.write_text(
            json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    return result


def validate_v3_e1_result(
    result: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate downloaded V3 E1 aggregate evidence.

    Args:
        result: Parsed aggregate output.
        config: Optional execution config for candidate comparison.

    Returns:
        Small validation summary.

    Raises:
        ValueError: If the result is incomplete or contains forbidden detail.
    """

    _reject_forbidden_result_keys(result)
    if result.get("schema_version") != V3_E1_SCHEMA:
        raise ValueError("Unexpected V3 E1 result schema")
    if result.get("stage") != "V3-E1":
        raise ValueError("Unexpected V3 E1 result stage")
    if result.get("status") != "COMPLETE" or result.get("complete") is not True:
        raise ValueError("V3 E1 result is not complete")
    expected_output = {
        "aggregate_only": True,
        "persisted_embeddings": False,
        "persisted_texts": False,
        "persisted_model_cache": False,
        "persisted_fitted_weights": False,
    }
    if result.get("output_contract") != expected_output:
        raise ValueError("V3 E1 result output contract drifted")
    if config is not None:
        validate_v3_e1_config(config)
        expected_batch_size = len(config["synthetic_batch"])
        checks = _mapping(config, "checks")
        expected_models = config["models"]
    else:
        expected_batch_size = None
        checks = None
        expected_models = [{"model_id": model_id} for model_id in V3_E1_ALLOWED_MODELS]
    models = result.get("models")
    if not isinstance(models, list) or len(models) != len(expected_models):
        raise ValueError("V3 E1 result has an unexpected model count")
    for expected_config, model in zip(expected_models, models):
        expected_id = expected_config["model_id"]
        if model.get("model_id") != expected_id:
            raise ValueError("V3 E1 result model order drifted")
        expected_dim = int(expected_config.get("expected_embedding_dim", 384))
        if expected_batch_size is None:
            expected_shape = model.get("observed_shape")
        else:
            expected_shape = [expected_batch_size, expected_dim]
        if model.get("observed_shape") != expected_shape:
            raise ValueError("V3 E1 result has wrong embedding dimension")
        if model.get("all_finite") is not True:
            raise ValueError(f"V3 E1 result has non-finite embeddings: {expected_id}")
        if checks is not None:
            _validate_result_model_metrics(model, checks)
        if model.get("passed") is not True:
            raise ValueError(f"V3 E1 model did not pass: {expected_id}")
        sha = model.get("resolved_hf_revision_sha")
        if not isinstance(sha, str) or not sha.strip():
            raise ValueError(f"Missing HF revision SHA for {expected_id}")
    gpu = _mapping(_mapping(result, "environment"), "gpu")
    if not _is_strict_result_gpu_valid(gpu, config):
        raise ValueError("V3 E1 result was not produced on a usable T4")
    if config is not None:
        packages = _mapping(_mapping(result, "environment"), "packages")
        expected_pin = _mapping(config, "package_policy")["sentence-transformers"]
        if packages.get("sentence-transformers") != expected_pin:
            raise ValueError("V3 E1 result sentence-transformers pin drifted")
    return {
        "status": "VALID",
        "model_count": len(models),
        "gpu_device_name": gpu.get("device_name"),
    }


def _validate_result_model_metrics(
    model: Mapping[str, Any],
    checks: Mapping[str, Any],
) -> None:
    tolerance = float(checks["norm_tolerance"])
    for key in ("norm_min", "norm_mean", "norm_max"):
        value = float(model[key])
        if abs(value - 1.0) > tolerance:
            raise ValueError(f"V3 E1 result {key} exceeded tolerance")
    limit = float(checks["determinism_tolerance_max_abs_diff"])
    if float(model["deterministic_max_abs_diff"]) > limit:
        raise ValueError("V3 E1 result determinism tolerance drifted")


def _is_strict_result_gpu_valid(
    gpu: Mapping[str, Any],
    config: Mapping[str, Any] | None,
) -> bool:
    if not all(
        bool(gpu.get(key))
        for key in ("cuda_available", "cuda_usable", "matmul_probe_ok")
    ):
        return False
    if not bool(gpu.get("embedding_probe_ok")):
        return False
    if config is None:
        accepted_names = set(V3_E1_T4_DEVICE_NAMES)
        capability = V3_E1_T4_COMPUTE_CAPABILITY
    else:
        accelerator = _mapping(config, "accelerator")
        accepted_names = set(accelerator.get("accepted_device_names", ()))
        capability = str(accelerator.get("required_compute_capability"))
    return (
        gpu.get("device_name") in accepted_names
        and gpu.get("compute_capability") == capability
        and gpu.get("device_is_t4") is True
    )


def build_v3_e1_bundle_manifest(root: str | Path) -> dict[str, Any]:
    """Build the manifest for the tiny V3 E1 Kaggle code bundle.

    Args:
        root: Project root.

    Returns:
        Manifest with hashes for review and Kaggle staging.

    Raises:
        ValueError: If any source path is missing or forbidden.
    """

    base = Path(root).expanduser().resolve()
    config = load_v3_e1_config(base / V3_E1_CONFIG)
    files = tuple(sorted(V3_E1_BUNDLE_FILES))
    for name in files:
        _assert_allowed_bundle_path(name)
        if not (base / name).is_file():
            raise ValueError(f"V3 E1 bundle file is missing: {name}")
    return {
        "schema_version": V3_E1_MANIFEST_SCHEMA,
        "stage": "V3-E1",
        "purpose": "synthetic_embedding_smoke_code_bundle",
        "contains_project_data": False,
        "contains_scientific_cache": False,
        "contains_artifacts": False,
        "approved_for_kaggle_execution": bool(
            config["approved_for_kaggle_execution"]
        ),
        "files": {name: _file_signature(base / name) for name in files},
        "forbidden_path_parts": sorted(V3_E1_FORBIDDEN_BUNDLE_PARTS),
    }


def write_v3_e1_bundle(
    root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write the V3 E1 code bundle zip and manifest.

    Args:
        root: Project root.
        output_dir: Directory receiving the Kaggle dataset files.

    Returns:
        Summary with paths, file count, and zip size.
    """

    base = Path(root).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest = build_v3_e1_bundle_manifest(base)
    manifest_text = json.dumps(
        manifest, ensure_ascii=True, indent=2, sort_keys=True
    )
    zip_path = destination / V3_E1_BUNDLE_ZIP
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        _write_zip_text(archive, V3_E1_BUNDLE_MANIFEST, manifest_text + "\n")
        for name in sorted(manifest["files"]):
            _write_zip_file(archive, base / name, name)
    manifest_path = destination / V3_E1_BUNDLE_MANIFEST
    manifest_path.write_text(manifest_text + "\n", encoding="utf-8")
    return {
        "zip_path": str(zip_path),
        "manifest_path": str(manifest_path),
        "zip_size_bytes": zip_path.stat().st_size,
        "file_count": len(manifest["files"]),
    }


def resolve_v3_e1_bundle_root(
    input_root: str | Path,
    work_root: str | Path,
) -> Path:
    """Locate or extract the V3 E1 Kaggle code bundle.

    Args:
        input_root: Kaggle dataset input directory.
        work_root: Writable Kaggle working directory.

    Returns:
        Directory containing the extracted bundle manifest.

    Raises:
        ValueError: If the bundle cannot be found.
    """

    source = Path(input_root)
    manifests = sorted(source.rglob(V3_E1_BUNDLE_MANIFEST))
    if manifests:
        return manifests[0].parent
    zips = sorted(source.rglob(V3_E1_BUNDLE_ZIP))
    if not zips:
        raise ValueError(f"No V3 E1 bundle found under: {source}")
    target = Path(work_root) / "v3_e1_bundle"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zips[0]) as archive:
        archive.extractall(target)
    if not (target / V3_E1_BUNDLE_MANIFEST).is_file():
        raise ValueError("Extracted V3 E1 bundle manifest is missing")
    return target


def stage_v3_e1_code(bundle_root: str | Path, work_root: str | Path) -> Path:
    """Verify the extracted V3 E1 code bundle and return its root.

    Args:
        bundle_root: Extracted bundle directory.
        work_root: Writable Kaggle working directory, kept for symmetry.

    Returns:
        Verified project root to prepend to ``sys.path``.

    Raises:
        ValueError: If any file hash differs from the manifest.
    """

    root = Path(bundle_root).expanduser().resolve()
    _ = Path(work_root).expanduser().resolve()
    manifest_path = root / V3_E1_BUNDLE_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != V3_E1_MANIFEST_SCHEMA:
        raise ValueError("Unexpected V3 E1 bundle manifest schema")
    for name, expected in manifest.get("files", {}).items():
        _assert_allowed_bundle_path(name)
        if _file_signature(root / name) != expected:
            raise ValueError(f"V3 E1 bundle hash mismatch: {name}")
    return root


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected mapping at {key}")
    return value


def _installed_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _encode(
    model: Any,
    texts: list[str],
    checks: Mapping[str, Any],
) -> np.ndarray:
    return np.asarray(
        model.encode(
            texts,
            normalize_embeddings=bool(checks["normalize_embeddings"]),
            convert_to_numpy=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )


def _validate_embedding_arrays(
    observed: np.ndarray,
    repeated: np.ndarray,
    expected_dim: int,
    checks: Mapping[str, Any],
) -> None:
    if observed.shape != repeated.shape:
        raise ValueError("Repeated embedding shape changed")
    if observed.ndim != 2 or observed.shape[1] != expected_dim:
        raise ValueError(f"Unexpected embedding shape: {observed.shape}")
    if not np.isfinite(observed).all() or not np.isfinite(repeated).all():
        raise ValueError("Embeddings contain non-finite values")
    norms = np.linalg.norm(observed, axis=1)
    tolerance = float(checks["norm_tolerance"])
    if not np.all(np.abs(norms - 1.0) <= tolerance):
        raise ValueError(f"Normalized embedding norms drifted: {norms}")
    max_abs_diff = float(np.max(np.abs(observed - repeated)))
    limit = float(checks["determinism_tolerance_max_abs_diff"])
    if max_abs_diff > limit:
        raise ValueError(f"Embedding determinism exceeded {limit}")


def _peak_cuda_memory_mb() -> float | None:
    try:
        import torch
    except Exception:
        return None
    try:
        value = torch.cuda.max_memory_allocated()
    except Exception:
        return None
    return round(float(value) / (1024.0 * 1024.0), 6)


def _reset_peak_cuda_memory() -> None:
    try:
        import torch
    except Exception:
        return
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        return


def _reject_forbidden_result_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in V3_E1_RESULT_FORBIDDEN_KEYS:
                raise ValueError(f"V3 E1 result contains forbidden key: {key}")
            _reject_forbidden_result_keys(child)
    elif isinstance(value, list):
        for child in value:
            _reject_forbidden_result_keys(child)


def _assert_allowed_bundle_path(name: str) -> None:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"V3 E1 bundle refuses unsafe path: {name}")
    if "\\" in name:
        raise ValueError(f"V3 E1 bundle path must use POSIX separators: {name}")
    lowered = {part.lower() for part in Path(name).parts}
    forbidden = lowered.intersection(V3_E1_FORBIDDEN_BUNDLE_PARTS)
    if forbidden:
        raise ValueError(f"V3 E1 bundle refuses forbidden path: {name}")


def _file_signature(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise ValueError(f"Required V3 E1 file is missing: {target}")
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "sha256": digest.hexdigest().upper(),
        "size_bytes": target.stat().st_size,
    }


def _write_zip_text(
    archive: zipfile.ZipFile,
    arcname: str,
    content: str,
) -> None:
    info = zipfile.ZipInfo(arcname)
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.compress_type = zipfile.ZIP_DEFLATED
    archive.writestr(info, content.encode("utf-8"))


def _write_zip_file(
    archive: zipfile.ZipFile,
    source: Path,
    arcname: str,
) -> None:
    info = zipfile.ZipInfo(arcname)
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.compress_type = zipfile.ZIP_DEFLATED
    archive.writestr(info, source.read_bytes())


__all__ = [
    "V3_E1_ALLOWED_MODELS",
    "V3_E1_APPROVED_STATUS",
    "V3_E1_BUNDLE_FILES",
    "V3_E1_BUNDLE_MANIFEST",
    "V3_E1_BUNDLE_ZIP",
    "V3_E1_CONFIG",
    "V3_E1_DRAFT_STATUS",
    "V3_E1_OUTPUT",
    "V3_E1_SCHEMA",
    "V3_E1_T4_COMPUTE_CAPABILITY",
    "V3_E1_T4_DEVICE_NAMES",
    "assert_v3_e1_t4_gpu",
    "assert_v3_e1_sentence_transformers_pin",
    "build_v3_e1_bundle_manifest",
    "load_v3_e1_config",
    "package_version_report",
    "report_v3_e1_gpu",
    "require_v3_e1_approval",
    "resolve_hf_model_revision",
    "resolve_v3_e1_bundle_root",
    "run_v3_e1_smoke",
    "smoke_one_v3_e1_model",
    "stage_v3_e1_code",
    "validate_v3_e1_config",
    "validate_v3_e1_result",
    "write_v3_e1_bundle",
]
