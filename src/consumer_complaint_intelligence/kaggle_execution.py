"""Operational staging and execution helpers for the V2-D1 Kaggle run.

ADR-010 moves the full classical benchmark to Kaggle after two local
executions aborted on low memory. This module owns the operational side
only: bundle manifests, hash verification, staging, and bounded execution.
It adds no scientific behavior and no sealed-partition access; the
benchmark itself remains ``v2_benchmark.run_v2_benchmark``.
"""

from __future__ import annotations

import datetime
import hashlib
import importlib.metadata
import json
import platform
import shutil
import zipfile
from pathlib import Path
from typing import Any, Mapping

BUNDLE_MANIFEST_NAME = "kaggle_bundle_manifest.json"
BUNDLE_SCHEMA = "v2-kaggle-bundle-manifest-v1"
BUNDLE_ZIP_NAME = "cci-v2-bundle.zip"
CACHE_FILE_NAME = "scientific.parquet"
CACHE_RELATIVE_PATH = "temp/s3/scientific.parquet"
SOURCE_PACKAGE_RELATIVE = "src/consumer_complaint_intelligence"
REQUIRED_PINS = {
    "scikit-learn": "1.9.0",
    "imbalanced-learn": "0.14.2",
}
RECORDED_PACKAGES = (
    "scikit-learn",
    "imbalanced-learn",
    "numpy",
    "scipy",
    "pyarrow",
    "joblib",
)
BUNDLE_FILES = (
    "pyproject.toml",
    "config/v2_development_protocol.json",
    "config/s8_results.json",
    "config/s7_frozen_package.json",
    "config/s7_results.json",
    "temp/s8/s8_results.json",
    "temp/s7/s7_results.json",
    "artifacts/s7/consumer_complaint_classifier_s7.joblib",
    "config/v2_d2_execution.json",
    "config/v2_classical_results.json",
    "temp/v2/v2_classical_benchmark.json",
    "config/v2_frozen_package.json",
    "config/v2_transformer_results.json",
    "temp/v2/v2_transformer_challenge.json",
)
OUTPUT_FILES = (
    "temp/v2/v2_classical_benchmark.json",
    "config/v2_classical_results.json",
)
D2_OUTPUT_FILES = (
    "temp/v2/v2_transformer_challenge.json",
    "config/v2_transformer_results.json",
)
PACKAGE_OUTPUT_FILES = (
    "temp/v2/v2_package.json",
    "config/v2_results.json",
)
PACKAGE_BUNDLE_FILE = "artifacts/v2/consumer_complaint_detector_v2.joblib"


def file_signature(path: str | Path) -> dict[str, Any]:
    """Return the uppercase SHA256 digest and byte size of one file.

    Args:
        path: File to hash.

    Returns:
        Mapping with ``sha256`` and ``size_bytes``.

    Raises:
        ValueError: If the file does not exist.
    """

    target = Path(path)
    if not target.is_file():
        raise ValueError(f"Required artifact is missing: {target}")
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "sha256": digest.hexdigest().upper(),
        "size_bytes": target.stat().st_size,
    }


def verify_signature(path: str | Path, expected: Mapping[str, Any]) -> None:
    """Compare one file against its expected signature.

    Args:
        path: File to verify.
        expected: Mapping with ``sha256`` and ``size_bytes``.

    Raises:
        ValueError: If the file is absent or its signature differs.
    """

    actual = file_signature(path)
    wanted = {
        "sha256": str(expected["sha256"]),
        "size_bytes": int(expected["size_bytes"]),
    }
    if actual != wanted:
        raise ValueError(f"Bundle artifact signature mismatch: {path}")


def environment_report(
    pins: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Describe the running interpreter against the required pins.

    Args:
        pins: Required package versions; defaults to ``REQUIRED_PINS``.

    Returns:
        Report with the python version, installed versions, and match flag.
    """

    wanted = dict(REQUIRED_PINS if pins is None else pins)
    installed: dict[str, str | None] = {}
    for name in RECORDED_PACKAGES:
        try:
            installed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed[name] = None
    matches = {
        name: installed.get(name) == version
        for name, version in wanted.items()
    }
    return {
        "python_version": platform.python_version(),
        "required_pins": wanted,
        "installed": installed,
        "pins_match": all(matches.values()),
        "pin_matches": matches,
    }


def assert_pinned_environment(pins: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Fail fast when a required package pin is not satisfied.

    Args:
        pins: Required package versions; defaults to ``REQUIRED_PINS``.

    Returns:
        The successful environment report.

    Raises:
        ValueError: If any required pin is absent or mismatched.
    """

    report = environment_report(pins)
    if not report["pins_match"]:
        raise ValueError(f"Environment pins are not satisfied: {report}")
    return report


def _source_files(root: Path) -> tuple[str, ...]:
    """List the project-relative python files of the shipped package."""

    package_dir = root / SOURCE_PACKAGE_RELATIVE
    if not package_dir.is_dir():
        raise ValueError(f"Source package is missing: {package_dir}")
    return tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in package_dir.glob("*.py")
        )
    )


def build_bundle_manifest(root: str | Path) -> dict[str, Any]:
    """Validate the frozen baseline and describe every bundle file.

    Args:
        root: Project root containing the frozen V2 artifacts.

    Returns:
        Bundle manifest payload with per-file signatures and pins.

    Raises:
        ValueError: If a baseline hash check or a required pin fails.
    """

    from .v2_protocol import load_v2_protocol, validate_baseline_artifacts

    base = Path(root).expanduser().resolve()
    assert_pinned_environment()
    protocol = load_v2_protocol(base / "config" / "v2_development_protocol.json")
    validate_baseline_artifacts(protocol, base)
    files = tuple(BUNDLE_FILES) + _source_files(base)
    created = datetime.datetime.now(datetime.timezone.utc)
    return {
        "schema_version": BUNDLE_SCHEMA,
        "stage": "V2.1-D1+D2+P",
        "purpose": "kaggle_execution_bundle",
        "adr": "docs/ADR-010-v2-post-confirmation-cycle.md",
        "created_utc": created.isoformat(timespec="seconds"),
        "environment": environment_report(),
        "files": {name: file_signature(base / name) for name in files},
        "scientific_cache": {
            "path": CACHE_RELATIVE_PATH,
            **file_signature(base / CACHE_RELATIVE_PATH),
        },
        "sealed_partitions": ["test", "stress", "monitor"],
        "contains_data": False,
    }


def write_bundle(root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Write the reproducible Kaggle bundle zip and its local manifest.

    Args:
        root: Project root containing the frozen V2 artifacts.
        output_dir: Directory receiving the zip and manifest copy.

    Returns:
        Summary with the zip path, manifest path, and file count.

    Raises:
        ValueError: If a baseline hash check or a required pin fails.
    """

    base = Path(root).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest = build_bundle_manifest(base)
    manifest_text = json.dumps(
        manifest, ensure_ascii=True, indent=2, sort_keys=True
    )
    zip_path = destination / BUNDLE_ZIP_NAME
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(BUNDLE_MANIFEST_NAME, manifest_text + "\n")
        for name in manifest["files"]:
            archive.write(base / name, name)
    manifest_path = destination / BUNDLE_MANIFEST_NAME
    manifest_path.write_text(manifest_text + "\n", encoding="utf-8")
    return {
        "zip_path": str(zip_path),
        "manifest_path": str(manifest_path),
        "zip_size_bytes": zip_path.stat().st_size,
        "file_count": len(manifest["files"]),
        "scientific_cache": manifest["scientific_cache"],
    }


def read_bundle_manifest(bundle_root: str | Path) -> dict[str, Any]:
    """Read and minimally validate one extracted bundle manifest.

    Args:
        bundle_root: Directory containing the extracted bundle tree.

    Returns:
        The parsed bundle manifest payload.

    Raises:
        ValueError: If the manifest is absent or has the wrong schema.
    """

    manifest_path = Path(bundle_root) / BUNDLE_MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(f"Bundle manifest is missing: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Bundle manifest must be a JSON object")
    if payload.get("schema_version") != BUNDLE_SCHEMA:
        raise ValueError("Unexpected bundle manifest schema")
    return payload


def resolve_bundle_root(input_root: str | Path, work_root: str | Path) -> Path:
    """Locate the bundle tree, extracting the zip when still archived.

    Args:
        input_root: Kaggle input dataset directory for the bundle.
        work_root: Writable directory used when extraction is needed.

    Returns:
        Directory that directly contains the bundle manifest.

    Raises:
        ValueError: If neither an extracted tree nor a zip is present.
    """

    source = Path(input_root)
    manifests = [
        path
        for path in sorted(source.rglob(BUNDLE_MANIFEST_NAME))
        if (path.parent / "src").is_dir()
    ]
    if manifests:
        return manifests[0].parent
    zips = sorted(source.rglob(BUNDLE_ZIP_NAME))
    if zips:
        target = Path(work_root) / "bundle_extracted"
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zips[0]) as archive:
            archive.extractall(target)
        if (target / BUNDLE_MANIFEST_NAME).is_file():
            return target
        raise ValueError("Extracted bundle does not contain its manifest")
    raise ValueError(f"No bundle manifest or zip found under: {source}")


def stage_project(
    bundle_root: str | Path,
    cache_file: str | Path,
    work_root: str | Path,
) -> dict[str, Any]:
    """Assemble and verify one runnable project tree for the benchmark.

    Args:
        bundle_root: Directory that directly contains the bundle manifest.
        cache_file: Uploaded development-only scientific parquet file.
        work_root: Writable project root receiving the staged tree.

    Returns:
        Staging report with the work root and verified file count.

    Raises:
        ValueError: If any staged file fails its manifest signature.
    """

    source = Path(bundle_root).expanduser().resolve()
    target = Path(work_root).expanduser().resolve()
    manifest = read_bundle_manifest(source)
    target.mkdir(parents=True, exist_ok=True)
    for name, expected in manifest["files"].items():
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, destination)
        verify_signature(destination, expected)
    cache_destination = target / CACHE_RELATIVE_PATH
    cache_destination.parent.mkdir(parents=True, exist_ok=True)
    cache_source = Path(cache_file).expanduser().resolve()
    if cache_source != cache_destination:
        shutil.copyfile(cache_source, cache_destination)
    verify_signature(cache_destination, manifest["scientific_cache"])
    return {
        "status": "STAGED",
        "work_root": str(target),
        "verified_files": len(manifest["files"]) + 1,
        "scientific_cache": manifest["scientific_cache"],
    }


def preflight(work_root: str | Path) -> dict[str, Any]:
    """Prove the staged tree can serve S7 and run the synthetic smoke.

    Args:
        work_root: Staged project root produced by ``stage_project``.

    Returns:
        Report with baseline, S7 load, and smoke statuses.

    Raises:
        ValueError: If a baseline hash, the S7 load, or the smoke fails.
    """

    from .s7 import load_s7_predictor
    from .v2_benchmark import run_v2_benchmark
    from .v2_protocol import load_v2_protocol, validate_baseline_artifacts

    root = Path(work_root).expanduser().resolve()
    protocol_path = root / "config" / "v2_development_protocol.json"
    protocol = load_v2_protocol(protocol_path)
    signatures = validate_baseline_artifacts(protocol, root)
    load_s7_predictor(
        root / "artifacts" / "s7" / "consumer_complaint_classifier_s7.joblib",
        root / "config" / "s7_results.json",
        root / "temp" / "s7" / "s7_results.json",
    )
    smoke = run_v2_benchmark("smoke", protocol_path=protocol_path)
    if smoke.get("status") != "DIAGNOSTIC_ONLY" or not smoke.get("complete"):
        raise ValueError(f"V2 smoke did not complete: {smoke.get('status')}")
    return {
        "status": "READY",
        "baseline_artifacts": sorted(signatures),
        "s7_loaded": True,
        "smoke_status": smoke["status"],
        "smoke_candidate_count": len(smoke.get("candidates") or ()),
    }


def run_full(
    work_root: str | Path,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Run the real V2-D1 classical benchmark inside the staged tree.

    Args:
        work_root: Staged project root produced by ``stage_project``.
        batch_size: Optional bounded batch size override.

    Returns:
        Aggregate-only benchmark result from ``run_v2_benchmark``.
    """

    from .v2_benchmark import DEFAULT_BATCH_SIZE, run_v2_benchmark

    return run_v2_benchmark(
        "full",
        project_root=Path(work_root).expanduser().resolve(),
        batch_size=DEFAULT_BATCH_SIZE if batch_size is None else batch_size,
    )


def _collect_named_outputs(
    work_root: str | Path,
    destination: str | Path,
    names: tuple[str, ...],
) -> tuple[str, ...]:
    """Copy a fixed set of published output files to a retrieval directory."""

    root = Path(work_root).expanduser().resolve()
    target = Path(destination).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in names:
        source = root / name
        if not source.is_file():
            raise ValueError(f"Expected published output is missing: {source}")
        output = target / source.name
        shutil.copyfile(source, output)
        copied.append(str(output))
    return tuple(copied)


def collect_outputs(
    work_root: str | Path,
    destination: str | Path,
) -> tuple[str, ...]:
    """Copy the published benchmark evidence to a retrieval directory.

    Args:
        work_root: Staged project root after a complete full run.
        destination: Directory that survives the Kaggle session.

    Returns:
        Destination paths of the copied artifact and manifest.

    Raises:
        ValueError: If a published output file is missing.
    """

    return _collect_named_outputs(work_root, destination, OUTPUT_FILES)


def preflight_d2(work_root: str | Path) -> dict[str, Any]:
    """Prove the staged tree matches the pinned D1 incumbent and can run D2.

    Args:
        work_root: Staged project root produced by ``stage_project``.

    Returns:
        Report with status ``READY``, the incumbent candidate id, and the
        D2 synthetic smoke status.

    Raises:
        ValueError: If a baseline hash, an incumbent hash, the S7 load, or
            the D2 smoke fails.
    """

    from .s7 import load_s7_predictor
    from .v2_protocol import load_v2_protocol, validate_baseline_artifacts
    from .v2_transformer import run_v2_transformer_smoke

    root = Path(work_root).expanduser().resolve()
    protocol_path = root / "config" / "v2_development_protocol.json"
    protocol = load_v2_protocol(protocol_path)
    signatures = validate_baseline_artifacts(protocol, root)
    load_s7_predictor(
        root / "artifacts" / "s7" / "consumer_complaint_classifier_s7.joblib",
        root / "config" / "s7_results.json",
        root / "temp" / "s7" / "s7_results.json",
    )
    d2_config_path = root / "config" / "v2_d2_execution.json"
    if not d2_config_path.is_file():
        raise ValueError(f"D2 execution config is missing: {d2_config_path}")
    d2_config = json.loads(d2_config_path.read_text(encoding="utf-8"))
    incumbent = d2_config.get("incumbent")
    if not isinstance(incumbent, Mapping) or not incumbent.get("candidate_id"):
        raise ValueError("D2 execution config is missing an incumbent candidate")
    candidate_id = str(incumbent["candidate_id"])
    for role in ("artifact", "manifest"):
        pinned = incumbent.get(role)
        if not isinstance(pinned, Mapping) or "path" not in pinned:
            raise ValueError(f"D2 execution config is missing incumbent.{role}")
        verify_signature(root / str(pinned["path"]), pinned)
    smoke = run_v2_transformer_smoke(root)
    if smoke.get("status") != "DIAGNOSTIC_ONLY" or not smoke.get("complete"):
        raise ValueError(f"V2 D2 smoke did not complete: {smoke.get('status')}")
    return {
        "status": "READY",
        "baseline_artifacts": sorted(signatures),
        "s7_loaded": True,
        "incumbent_candidate_id": candidate_id,
        "smoke_status": smoke["status"],
    }


def run_full_d2(
    work_root: str | Path,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Run the real V2-D2 transformer challenge inside the staged tree.

    Args:
        work_root: Staged project root produced by ``stage_project``.
        batch_size: Optional bounded batch size override.

    Returns:
        Aggregate-only challenge result from
        ``run_v2_transformer_challenge``.
    """

    from .v2_transformer import run_v2_transformer_challenge

    root = Path(work_root).expanduser().resolve()
    if batch_size is None:
        return run_v2_transformer_challenge("full", project_root=root)
    return run_v2_transformer_challenge(
        "full", project_root=root, batch_size=batch_size
    )


def collect_outputs_d2(
    work_root: str | Path,
    destination: str | Path,
) -> tuple[str, ...]:
    """Copy the published D2 transformer evidence to a retrieval directory.

    Args:
        work_root: Staged project root after a complete D2 full run.
        destination: Directory that survives the Kaggle session.

    Returns:
        Destination paths of the copied artifact and manifest.

    Raises:
        ValueError: If a published output file is missing.
    """

    return _collect_named_outputs(work_root, destination, D2_OUTPUT_FILES)


def preflight_package(work_root: str | Path) -> dict[str, Any]:
    """Prove the staged tree can freeze the V2 package before fitting.

    Args:
        work_root: Staged project root produced by ``stage_project``.

    Returns:
        Report with status ``READY``, the pinned candidate id, and the
        diagnostic smoke checks.

    Raises:
        ValueError: If a pinned hash, the S7 load, or the smoke fails.
    """

    from .v2_package import run_v2_package_smoke

    root = Path(work_root).expanduser().resolve()
    config_path = root / "config" / "v2_frozen_package.json"
    if not config_path.is_file():
        raise ValueError(f"Frozen package config is missing: {config_path}")
    smoke = run_v2_package_smoke(root)
    if smoke.get("status") != "DIAGNOSTIC_ONLY" or not smoke.get("complete"):
        raise ValueError(f"V2 package smoke did not complete: {smoke.get('status')}")
    checks = dict(smoke.get("checks") or {})
    failed = sorted(name for name, value in checks.items() if not value)
    if failed:
        raise ValueError(f"V2 package smoke checks failed: {failed}")
    return {
        "status": "READY",
        "candidate_id": smoke["candidate"]["candidate_id"],
        "fallback_model_version": smoke["fallback_model_version"],
        "provenance_verified": sorted(smoke["provenance"]),
        "checks": checks,
        "smoke_status": smoke["status"],
    }


def run_full_package(
    work_root: str | Path,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Run the real V2 package freeze inside the staged tree.

    Args:
        work_root: Staged project root produced by ``stage_project``.
        batch_size: Optional bounded batch size override.

    Returns:
        Aggregate-only freeze result from ``run_v2_package``.
    """

    from .v2_package import run_v2_package

    root = Path(work_root).expanduser().resolve()
    if batch_size is None:
        return run_v2_package("full", project_root=root)
    return run_v2_package("full", project_root=root, batch_size=batch_size)


def collect_outputs_package(
    work_root: str | Path,
    destination: str | Path,
) -> tuple[str, ...]:
    """Copy the published V2 package evidence to a retrieval directory.

    The fitted joblib bundle is copied only when the reproduction gate
    passed and the freeze actually persisted it. A run that ended in
    ``REPRODUCTION_MISMATCH`` still returns its two evidence files.

    Args:
        work_root: Staged project root after a complete package run.
        destination: Directory that survives the Kaggle session.

    Returns:
        Destination paths of the copied evidence and, when frozen, bundle.

    Raises:
        ValueError: If a published evidence file is missing.
    """

    copied = list(
        _collect_named_outputs(work_root, destination, PACKAGE_OUTPUT_FILES)
    )
    root = Path(work_root).expanduser().resolve()
    bundle = root / PACKAGE_BUNDLE_FILE
    if bundle.is_file():
        target = Path(destination).expanduser().resolve() / bundle.name
        shutil.copyfile(bundle, target)
        copied.append(str(target))
    return tuple(copied)


def _installed_version(name: str) -> str | None:
    """Return one installed package version, or None when it is absent."""

    try:
        return importlib.metadata.version(name)
    except Exception:
        return None


def report_gpu() -> dict[str, Any]:
    """Describe local GPU availability for the D2 notebook to print.

    Never raises: it degrades to a not-available report when torch is
    absent or GPU introspection itself fails.

    ``cuda_available`` only means torch found a device. It does NOT mean
    the device can run anything: a torch build without kernels compiled
    for the device's compute capability reports the GPU as available and
    then fails on the first real kernel launch. ``cuda_usable`` therefore
    launches a tiny matmul and an embedding lookup, which is what
    actually distinguishes a working accelerator from an unusable one.

    Returns:
        Mapping with ``cuda_available``, ``cuda_usable``, ``device_name``,
        ``compute_capability``, ``cuda_error``, ``torch_version``, and
        ``transformers_version``.
    """

    report: dict[str, Any] = {
        "cuda_available": False,
        "cuda_usable": False,
        "device_name": None,
        "compute_capability": None,
        "cuda_error": None,
        "torch_version": None,
        "transformers_version": _installed_version("transformers"),
    }
    try:
        import torch
    except Exception as error:  # noqa: BLE001 -- reported, never raised
        report["cuda_error"] = f"torch import failed: {error}"
        return report
    report["torch_version"] = str(getattr(torch, "__version__", None))
    try:
        report["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as error:  # noqa: BLE001 -- reported, never raised
        report["cuda_error"] = f"cuda probe failed: {error}"
        return report
    if not report["cuda_available"]:
        return report
    try:
        report["device_name"] = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        report["compute_capability"] = f"sm_{major}{minor}"
    except Exception as error:  # noqa: BLE001 -- reported, never raised
        report["cuda_error"] = f"device query failed: {error}"
        return report
    try:
        device = torch.device("cuda:0")
        probe = torch.ones((8, 8), device=device)
        torch.mm(probe, probe)
        table = torch.nn.Embedding(16, 4).to(device)
        table(torch.zeros((2, 3), dtype=torch.long, device=device))
        torch.cuda.synchronize()
        report["cuda_usable"] = True
    except Exception as error:  # noqa: BLE001 -- reported, never raised
        report["cuda_error"] = f"{type(error).__name__}: {error}"
    return report


def assert_usable_gpu() -> dict[str, Any]:
    """Fail fast when the accelerator cannot actually run a kernel.

    Returns:
        The successful GPU report.

    Raises:
        ValueError: If no CUDA device is present, or if a device is
            present but cannot execute a kernel. The message carries the
            device name, its compute capability, and the underlying CUDA
            error, which is what identifies an architecture the installed
            torch build no longer ships kernels for.
    """

    report = report_gpu()
    if not report["cuda_available"]:
        raise ValueError(f"D2 requires a CUDA device; got: {report}")
    if not report["cuda_usable"]:
        raise ValueError(
            "CUDA device is present but cannot execute a kernel. "
            f"device={report['device_name']} "
            f"capability={report['compute_capability']} "
            f"torch={report['torch_version']} "
            f"error={report['cuda_error']}"
        )
    return report


__all__ = [
    "BUNDLE_FILES",
    "BUNDLE_MANIFEST_NAME",
    "BUNDLE_SCHEMA",
    "BUNDLE_ZIP_NAME",
    "CACHE_FILE_NAME",
    "CACHE_RELATIVE_PATH",
    "D2_OUTPUT_FILES",
    "PACKAGE_BUNDLE_FILE",
    "PACKAGE_OUTPUT_FILES",
    "assert_usable_gpu",
    "OUTPUT_FILES",
    "RECORDED_PACKAGES",
    "REQUIRED_PINS",
    "assert_pinned_environment",
    "build_bundle_manifest",
    "collect_outputs",
    "collect_outputs_d2",
    "collect_outputs_package",
    "environment_report",
    "file_signature",
    "preflight",
    "preflight_d2",
    "preflight_package",
    "read_bundle_manifest",
    "report_gpu",
    "resolve_bundle_root",
    "run_full",
    "run_full_d2",
    "run_full_package",
    "stage_project",
    "verify_signature",
    "write_bundle",
]
