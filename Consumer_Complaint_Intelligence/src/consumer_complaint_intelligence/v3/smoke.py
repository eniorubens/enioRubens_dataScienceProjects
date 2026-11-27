"""Synthetic-only V3-E1 CUDA smoke benchmark for the pinned MiniLM encoder."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .protocol import load_v3_protocol


DEFAULT_CONFIG = "config/v3/smoke.json"
SMOKE_SCHEMA = "v3-e1-gpu-smoke-v1"
RESULT_SCHEMA = "v3-e1-gpu-smoke-result-v1"
MANIFEST_SCHEMA = "v3-e1-gpu-smoke-results-manifest-v1"
_FORBIDDEN_RESULT_KEYS = {
    "text",
    "texts",
    "narrative",
    "narratives",
    "label",
    "labels",
    "prediction",
    "predictions",
    "score",
    "scores",
    "embedding",
    "embeddings",
}


def _exact_keys(value: Any, expected: set[str], name: str) -> Mapping[str, Any]:
    """Require one mapping with exactly the expected keys."""

    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"V3 smoke {name} keys differ from the contract")
    return value


def _safe_project_path(project_root: Path, value: str) -> Path:
    """Resolve a project-relative smoke path without traversal."""

    candidate = Path(value)
    if candidate.is_absolute() or candidate.drive or ".." in candidate.parts:
        raise ValueError("V3 smoke paths must be project-relative")
    resolved = (project_root / candidate).resolve()
    if not resolved.is_relative_to(project_root):
        raise ValueError("V3 smoke path escapes the project root")
    return resolved


@dataclass(frozen=True, slots=True)
class SmokeConfig:
    """Validated synthetic-only V3-E1 configuration."""

    payload: Mapping[str, Any]

    @property
    def model_id(self) -> str:
        """Return the pinned Hugging Face model identifier."""

        return str(self.payload["model"]["model_id"])

    @property
    def revision(self) -> str:
        """Return the pinned Hugging Face commit."""

        return str(self.payload["model"]["revision"])

    @property
    def row_count(self) -> int:
        """Return the full synthetic benchmark row count."""

        return int(self.payload["input"]["row_count"])

    @property
    def batch_sizes(self) -> tuple[int, ...]:
        """Return increasing candidate batch sizes."""

        return tuple(int(value) for value in self.payload["benchmark"]["batch_sizes"])


def validate_smoke_config(payload: Mapping[str, Any]) -> SmokeConfig:
    """Validate a smoke that cannot read data or select a model."""

    block = _exact_keys(
        payload,
        {
            "schema_version",
            "stage",
            "status",
            "model",
            "input",
            "benchmark",
            "paths",
            "selection_authorized",
            "scientific_execution_authorized",
        },
        "top-level",
    )
    if block["schema_version"] != SMOKE_SCHEMA or block["stage"] != "V3-E1":
        raise ValueError("V3 smoke schema or stage is invalid")
    if block["status"] != "TECHNICAL_DIAGNOSTIC_ONLY":
        raise ValueError("V3 smoke must remain diagnostic only")
    if block["selection_authorized"] is not False:
        raise ValueError("V3 smoke may not select a model")
    if block["scientific_execution_authorized"] is not False:
        raise ValueError("V3 smoke may not claim scientific execution")

    model = _exact_keys(
        block["model"],
        {
            "model_id",
            "revision",
            "backend",
            "device",
            "torch_dtype",
            "max_seq_length",
            "normalize_embeddings",
        },
        "model",
    )
    if model != {
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "bc57282bc374d33e0d6c4de27f12dc1c2a87f37a",
        "backend": "torch",
        "device": "cuda",
        "torch_dtype": "float16",
        "max_seq_length": 256,
        "normalize_embeddings": False,
    }:
        raise ValueError("V3 smoke model differs from the pinned MiniLM")

    source = _exact_keys(
        block["input"],
        {
            "source",
            "row_count",
            "wordpiece_risk_profile",
            "scientific_partition_access",
        },
        "input",
    )
    if source["source"] != "deterministic_synthetic_text_only":
        raise ValueError("V3 smoke may use synthetic text only")
    if source["scientific_partition_access"] is not False:
        raise ValueError("V3 smoke may not access scientific partitions")
    if int(source["row_count"]) != 10000:
        raise ValueError("V3 smoke row count must remain 10000")

    benchmark = _exact_keys(
        block["benchmark"],
        {
            "batch_sizes",
            "probe_rows_per_batch_size",
            "warmup_rows",
            "stop_after_cuda_oom",
            "synchronize_cuda_for_timing",
        },
        "benchmark",
    )
    batches = tuple(int(value) for value in benchmark["batch_sizes"])
    if batches != (8, 16, 32, 64, 128):
        raise ValueError("V3 smoke batch-size catalog differs")
    if int(benchmark["probe_rows_per_batch_size"]) != 512:
        raise ValueError("V3 smoke probe size differs")
    if int(benchmark["warmup_rows"]) != 128:
        raise ValueError("V3 smoke warmup size differs")
    if benchmark["stop_after_cuda_oom"] is not True:
        raise ValueError("V3 smoke must stop batch growth after CUDA OOM")
    if benchmark["synchronize_cuda_for_timing"] is not True:
        raise ValueError("V3 smoke must synchronize CUDA timing")

    _exact_keys(block["paths"], {"model_cache", "result"}, "paths")
    return SmokeConfig(json.loads(json.dumps(block)))


def load_smoke_config(path: str | Path = DEFAULT_CONFIG) -> SmokeConfig:
    """Load one V3-E1 smoke configuration."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read V3 smoke config: {path}") from error
    return validate_smoke_config(payload)


def build_synthetic_texts(row_count: int) -> tuple[str, ...]:
    """Build deterministic English texts spanning three length profiles."""

    if row_count <= 0:
        raise ValueError("Synthetic smoke row_count must be positive")
    stems = (
        "consumer reports an account dispute and requests investigation",
        "payment history contains an unfamiliar balance and servicing issue",
        "credit file shows information that the consumer cannot recognize",
    )
    repeats = (2, 12, 44)
    return tuple(
        " ".join([stems[index % len(stems)]] * repeats[index % len(repeats)])
        for index in range(row_count)
    )


def _text_signature(texts: Sequence[str]) -> str:
    """Hash synthetic inputs without publishing their contents."""

    digest = hashlib.sha256()
    for value in texts:
        digest.update(value.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest().upper()


def _package_versions() -> dict[str, str]:
    """Return the packages material to the smoke result."""

    names = (
        "torch",
        "sentence-transformers",
        "transformers",
        "numpy",
        "scikit-learn",
    )
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _load_real_encoder(config: SmokeConfig, cache_folder: Path) -> tuple[Any, Any]:
    """Load the pinned encoder lazily, after all boundary checks."""

    import torch
    from sentence_transformers import SentenceTransformer

    if not torch.cuda.is_available():
        raise RuntimeError("V3-E1 requires a CUDA device")
    encoder = SentenceTransformer(
        config.model_id,
        revision=config.revision,
        cache_folder=str(cache_folder),
        device="cuda",
        backend="torch",
        trust_remote_code=False,
        model_kwargs={"torch_dtype": torch.float16},
    )
    encoder.max_seq_length = int(config.payload["model"]["max_seq_length"])
    return encoder, torch


def _encode(
    encoder: Any,
    texts: Sequence[str],
    batch_size: int,
    normalize_embeddings: bool,
) -> Any:
    """Encode one bounded sequence without retaining outputs after measurement."""

    return encoder.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )


def _cuda_measurement(
    encoder: Any,
    torch_module: Any,
    texts: Sequence[str],
    batch_size: int,
    normalize_embeddings: bool,
) -> dict[str, Any]:
    """Measure one synchronized CUDA embedding pass."""

    cuda = torch_module.cuda
    cuda.empty_cache()
    cuda.reset_peak_memory_stats()
    cuda.synchronize()
    started = time.perf_counter()
    embeddings = _encode(
        encoder, texts, batch_size, normalize_embeddings=normalize_embeddings
    )
    cuda.synchronize()
    elapsed = time.perf_counter() - started
    shape = tuple(int(value) for value in embeddings.shape)
    dtype = str(embeddings.dtype)
    del embeddings
    return {
        "batch_size": int(batch_size),
        "row_count": len(texts),
        "elapsed_seconds": elapsed,
        "texts_per_second": len(texts) / elapsed,
        "embedding_shape": list(shape),
        "output_dtype": dtype,
        "peak_allocated_mib": cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": cuda.max_memory_reserved() / 1024**2,
    }


def _is_cuda_oom(error: RuntimeError) -> bool:
    """Return whether one runtime error is a CUDA out-of-memory condition."""

    message = str(error).lower()
    return "cuda" in message and "out of memory" in message


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON result atomically in its target directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _walk_keys(value: Any) -> set[str]:
    """Return normalized keys from a nested JSON-compatible value."""

    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            keys.add(str(key).lower())
            keys.update(_walk_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(_walk_keys(item))
    return keys


def validate_smoke_result(
    payload: Mapping[str, Any],
    config: SmokeConfig | Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one aggregate smoke result and reject individual outputs."""

    validated_config = (
        config if isinstance(config, SmokeConfig) else validate_smoke_config(config)
    )
    if _walk_keys(payload) & _FORBIDDEN_RESULT_KEYS:
        raise ValueError("V3 smoke result contains individual model data")
    block = _exact_keys(
        payload,
        {
            "schema_version",
            "stage",
            "status",
            "complete",
            "selection_authorized",
            "scientific_partition_access",
            "model",
            "environment",
            "input",
            "batch_probes",
            "selected_batch_size",
            "full_benchmark",
            "protocol_sha256",
        },
        "result",
    )
    if block["schema_version"] != RESULT_SCHEMA or block["stage"] != "V3-E1":
        raise ValueError("V3 smoke result schema or stage is invalid")
    if block["status"] != "TECHNICAL_DIAGNOSTIC_ONLY":
        raise ValueError("V3 smoke result must remain diagnostic only")
    if block["complete"] is not True:
        raise ValueError("V3 smoke result is incomplete")
    if block["selection_authorized"] is not False:
        raise ValueError("V3 smoke result may not authorize selection")
    if block["scientific_partition_access"] is not False:
        raise ValueError("V3 smoke result may not report scientific access")
    model = block["model"]
    if model.get("model_id") != validated_config.model_id:
        raise ValueError("V3 smoke result model identifier differs")
    if model.get("revision") != validated_config.revision:
        raise ValueError("V3 smoke result model revision differs")
    source = block["input"]
    if source.get("source") != "deterministic_synthetic_text_only":
        raise ValueError("V3 smoke result input source is invalid")
    if int(source.get("row_count", -1)) != validated_config.row_count:
        raise ValueError("V3 smoke result row count differs")
    digest = str(source.get("sha256", ""))
    is_upper_hex = all(
        character in "0123456789ABCDEF" for character in digest
    )
    if len(digest) != 64 or not is_upper_hex:
        raise ValueError("V3 smoke result input hash is invalid")

    selected = int(block["selected_batch_size"])
    if selected not in validated_config.batch_sizes:
        raise ValueError("V3 smoke selected batch size is unregistered")
    full = block["full_benchmark"]
    if int(full.get("batch_size", -1)) != selected:
        raise ValueError("V3 smoke full benchmark batch differs")
    if list(full.get("embedding_shape", ())) != [validated_config.row_count, 384]:
        raise ValueError("V3 smoke embedding shape differs")
    if float(full.get("elapsed_seconds", 0.0)) <= 0.0:
        raise ValueError("V3 smoke elapsed time is invalid")
    if float(full.get("texts_per_second", 0.0)) <= 0.0:
        raise ValueError("V3 smoke throughput is invalid")
    protocol_digest = str(block["protocol_sha256"])
    if len(protocol_digest) != 64:
        raise ValueError("V3 smoke protocol hash is invalid")
    return json.loads(json.dumps(block))


def validate_smoke_manifest(
    manifest_path: str | Path,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    """Validate the persisted result signature and privacy boundary."""

    root = Path(project_root).expanduser().resolve()
    path = Path(manifest_path)
    if not path.is_absolute():
        path = root / path
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read V3 smoke manifest: {path}") from error
    block = _exact_keys(
        manifest,
        {
            "schema_version",
            "stage",
            "status",
            "complete",
            "selection_authorized",
            "scientific_partition_access",
            "result",
            "summary",
            "interpretation",
        },
        "manifest",
    )
    if block["schema_version"] != MANIFEST_SCHEMA:
        raise ValueError("V3 smoke manifest schema is invalid")
    if block["status"] != "TECHNICAL_DIAGNOSTIC_ONLY":
        raise ValueError("V3 smoke manifest status is invalid")
    for key, expected in {
        "complete": True,
        "selection_authorized": False,
        "scientific_partition_access": False,
    }.items():
        if block[key] is not expected:
            raise ValueError(f"V3 smoke manifest flag is invalid: {key}")
    result_meta = _exact_keys(
        block["result"], {"path", "sha256", "size_bytes"}, "manifest.result"
    )
    result_path = _safe_project_path(root, str(result_meta["path"]))
    raw = result_path.read_bytes()
    if len(raw) != int(result_meta["size_bytes"]):
        raise ValueError("V3 smoke result size differs from the manifest")
    if hashlib.sha256(raw).hexdigest().upper() != str(result_meta["sha256"]):
        raise ValueError("V3 smoke result hash differs from the manifest")
    result = json.loads(raw.decode("utf-8"))
    config = load_smoke_config(root / "config/v3/smoke.json")
    validate_smoke_result(result, config)
    summary = block["summary"]
    if summary.get("revision") != config.revision:
        raise ValueError("V3 smoke manifest revision differs")
    if int(summary.get("synthetic_rows", -1)) != config.row_count:
        raise ValueError("V3 smoke manifest row count differs")
    if int(summary.get("selected_batch_size", -1)) != int(
        result["selected_batch_size"]
    ):
        raise ValueError("V3 smoke manifest selected batch differs")
    return json.loads(json.dumps(block))


def run_synthetic_gpu_smoke(
    project_root: str | Path = ".",
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    encoder_loader: Callable[[SmokeConfig, Path], tuple[Any, Any]] | None = None,
) -> dict[str, Any]:
    """Run V3-E1 without reading any project dataset or model label."""

    root = Path(project_root).expanduser().resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config = load_smoke_config(config_file)
    protocol = load_v3_protocol(root / "config/v3/development_protocol.json")
    minilm = protocol.payload["screening"]["representations"][1]
    if (config.model_id, config.revision) != (
        minilm["model_id"],
        minilm["revision"],
    ):
        raise ValueError("V3 smoke model differs from the V3-E0 protocol")

    cache = _safe_project_path(root, str(config.payload["paths"]["model_cache"]))
    result_path = _safe_project_path(root, str(config.payload["paths"]["result"]))
    cache.mkdir(parents=True, exist_ok=True)
    texts = build_synthetic_texts(config.row_count)
    loader = encoder_loader or _load_real_encoder
    encoder, torch_module = loader(config, cache)
    normalize = bool(config.payload["model"]["normalize_embeddings"])
    warmup_rows = int(config.payload["benchmark"]["warmup_rows"])
    _encode(encoder, texts[:warmup_rows], config.batch_sizes[0], normalize)
    torch_module.cuda.synchronize()

    probe_rows = int(config.payload["benchmark"]["probe_rows_per_batch_size"])
    probes: list[dict[str, Any]] = []
    selected_batch: int | None = None
    for batch_size in config.batch_sizes:
        try:
            measurement = _cuda_measurement(
                encoder,
                torch_module,
                texts[:probe_rows],
                batch_size,
                normalize,
            )
        except RuntimeError as error:
            if not _is_cuda_oom(error):
                raise
            probes.append({"batch_size": batch_size, "status": "CUDA_OOM"})
            torch_module.cuda.empty_cache()
            break
        measurement["status"] = "PASS"
        probes.append(measurement)
        selected_batch = batch_size
    if selected_batch is None:
        raise RuntimeError("V3-E1 found no safe CUDA batch size")

    full = _cuda_measurement(
        encoder,
        torch_module,
        texts,
        selected_batch,
        normalize,
    )
    device = torch_module.cuda.get_device_properties(0)
    result = {
        "schema_version": RESULT_SCHEMA,
        "stage": "V3-E1",
        "status": "TECHNICAL_DIAGNOSTIC_ONLY",
        "complete": True,
        "selection_authorized": False,
        "scientific_partition_access": False,
        "model": {
            "model_id": config.model_id,
            "revision": config.revision,
            "max_seq_length": int(config.payload["model"]["max_seq_length"]),
            "torch_dtype": config.payload["model"]["torch_dtype"],
        },
        "environment": {
            "python": platform.python_version(),
            "packages": _package_versions(),
            "device": str(device.name),
            "total_vram_mib": int(device.total_memory / 1024**2),
            "cuda_runtime": str(torch_module.version.cuda),
        },
        "input": {
            "source": "deterministic_synthetic_text_only",
            "row_count": len(texts),
            "sha256": _text_signature(texts),
        },
        "batch_probes": probes,
        "selected_batch_size": selected_batch,
        "full_benchmark": full,
        "protocol_sha256": protocol.protocol_sha256,
    }
    validate_smoke_result(result, config)
    _atomic_json(result_path, result)
    return result


def main() -> None:
    """Run the synthetic smoke and print its aggregate result."""

    result = run_synthetic_gpu_smoke()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
