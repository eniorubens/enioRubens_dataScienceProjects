"""Execution cache for heavy notebook runs shared across languages.

A run is identified by a name, a human-chosen version and a fingerprint that
combines the configuration, the input data and the package source code. The
language of the notebook never enters the fingerprint, so a notebook in one
language can train and the notebook in the other language can load the result.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import time
import uuid
from collections.abc import Callable
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path

import pandas as pd

_VALID_MODES = ("auto", "train", "load")
_PACKAGES = ("pandas", "numpy", "scikit-learn", "xgboost", "lightgbm", "catboost")


def _package_version(name: str) -> str | None:
    """Return the installed version of a distribution or ``None``."""
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _hash_frame(frame: pd.DataFrame) -> str:
    """Return a deterministic SHA-256 digest of a frame (values, index, columns)."""
    digest = hashlib.sha256()
    digest.update(json.dumps([str(c) for c in frame.columns]).encode("utf-8"))
    digest.update(json.dumps([str(t) for t in frame.dtypes]).encode("utf-8"))
    digest.update(str(frame.shape).encode("utf-8"))
    row_hashes = pd.util.hash_pandas_object(frame, index=True)
    digest.update(row_hashes.to_numpy(dtype="uint64").tobytes())
    return digest.hexdigest()


def _hash_package(package_dir: Path) -> str:
    """Return a SHA-256 digest of every ``*.py`` file of the package."""
    digest = hashlib.sha256()
    files = sorted(
        (p for p in package_dir.rglob("*.py") if "__pycache__" not in p.parts),
        key=lambda p: p.relative_to(package_dir).as_posix(),
    )
    for path in files:
        digest.update(path.relative_to(package_dir).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def run_fingerprint(
    config: dict,
    frames: dict[str, pd.DataFrame] | None = None,
    package_dir: Path | None = None,
) -> str:
    """Build a stable SHA-256 fingerprint for a run.

    Parameters
    ----------
    config:
        JSON-serializable configuration (run version, fold parameters, models,
        feature lists, random state). Keys are sorted before hashing.
    frames:
        Optional named input frames. Each one contributes its values, index,
        column names and dtypes.
    package_dir:
        Directory of the ``ecom_delivery_sla`` package whose ``*.py`` files are
        hashed. Defaults to the installed package directory.

    Returns
    -------
    str
        Hexadecimal SHA-256 digest. The notebook language is not an input.
    """
    package_dir = Path(package_dir) if package_dir is not None else Path(__file__).parent
    digest = hashlib.sha256()
    digest.update(
        json.dumps(config, sort_keys=True, default=str, ensure_ascii=True).encode("utf-8")
    )
    for key in sorted(frames or {}):
        digest.update(b"\0frame\0" + str(key).encode("utf-8"))
        digest.update(_hash_frame(frames[key]).encode("utf-8"))
    digest.update(b"\0code\0" + _hash_package(package_dir).encode("utf-8"))
    return digest.hexdigest()


def _default_cache_root() -> Path:
    """Return ``<project root>/outputs/runs`` found by walking up to ``data/``."""
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "data").exists():
            return candidate / "outputs" / "runs"
    return current / "outputs" / "runs"


def _load_run(run_dir: Path, keys: list[str]) -> dict[str, pd.DataFrame]:
    """Read the pickled frames of a cached run."""
    return {key: pd.read_pickle(run_dir / f"{key}.pkl") for key in keys}


def _save_run(
    run_dir: Path,
    results: dict[str, pd.DataFrame],
    metadata: dict,
) -> None:
    """Write a run atomically: temporary sibling directory, then rename."""
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = run_dir.parent / f".{run_dir.name}.tmp-{uuid.uuid4().hex[:8]}"
    old_dir = run_dir.parent / f".{run_dir.name}.old-{uuid.uuid4().hex[:8]}"
    tmp_dir.mkdir()
    try:
        for key, frame in results.items():
            frame.to_pickle(tmp_dir / f"{key}.pkl")
            frame.to_csv(tmp_dir / f"{key}.csv", index=False)
        (tmp_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        if run_dir.exists():
            run_dir.rename(old_dir)
        tmp_dir.rename(run_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if old_dir.exists() and not run_dir.exists():
            old_dir.rename(run_dir)
        raise
    shutil.rmtree(old_dir, ignore_errors=True)


def load_run_unchecked(
    name: str,
    run_version: str,
    cache_root: Path | None = None,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Load a saved run without comparing fingerprints.

    This supports runs with frozen semantics (for example the single final
    evaluation), where the stored result must be used even when the current
    configuration, data or package code would produce a different fingerprint.

    Parameters
    ----------
    name:
        Run name, first directory level under ``cache_root``.
    run_version:
        Version, second directory level.
    cache_root:
        Base directory. Defaults to ``<project root>/outputs/runs``.

    Returns
    -------
    tuple
        ``(results, metadata)``; ``metadata["source"]`` is ``"loaded"``.

    Raises
    ------
    FileNotFoundError
        If the run directory has no ``metadata.json`` or lacks a result file.
    """
    root = Path(cache_root) if cache_root is not None else _default_cache_root()
    run_dir = root / name / run_version
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No cached run at {run_dir}")
    cached_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    keys = list(cached_meta.get("keys", []))
    if not all((run_dir / f"{key}.pkl").exists() for key in keys):
        raise FileNotFoundError(f"Incomplete cache in {run_dir}")
    return _load_run(run_dir, keys), {**cached_meta, "source": "loaded"}


def cached_run(
    name: str,
    run_version: str,
    config: dict,
    compute_fn: Callable[[], dict[str, pd.DataFrame]],
    frames: dict[str, pd.DataFrame] | None = None,
    mode: str = "auto",
    cache_root: Path | None = None,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Compute a heavy run or load it from the on-disk cache.

    Parameters
    ----------
    name:
        Run name, first directory level under ``cache_root``.
    run_version:
        Human-chosen version, second directory level.
    config:
        Configuration included in the fingerprint.
    compute_fn:
        Zero-argument callable returning ``{key: DataFrame}``.
    frames:
        Input frames included in the fingerprint.
    mode:
        ``"auto"`` loads a cache with the same fingerprint, otherwise computes
        and saves; ``"train"`` always computes and overwrites; ``"load"`` only
        loads.
    cache_root:
        Base directory. Defaults to ``<project root>/outputs/runs``.

    Returns
    -------
    tuple
        ``(results, metadata)``; ``metadata["source"]`` is ``"loaded"`` or
        ``"computed"``.

    Raises
    ------
    ValueError
        If ``mode`` is unknown, or in ``"load"`` mode when the cached
        fingerprint differs from the current one.
    FileNotFoundError
        In ``"load"`` mode when no cache exists.
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")
    root = Path(cache_root) if cache_root is not None else _default_cache_root()
    run_dir = root / name / run_version
    fingerprint = run_fingerprint(config, frames)
    meta_path = run_dir / "metadata.json"

    if mode != "train" and meta_path.exists():
        cached_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        keys = list(cached_meta.get("keys", []))
        files_ok = all((run_dir / f"{k}.pkl").exists() for k in keys)
        if cached_meta.get("fingerprint") == fingerprint and files_ok:
            results = _load_run(run_dir, keys)
            return results, {**cached_meta, "source": "loaded"}
        if mode == "load":
            if cached_meta.get("fingerprint") != fingerprint:
                raise ValueError(
                    f"Cached fingerprint {str(cached_meta.get('fingerprint'))[:8]} differs "
                    f"from current {fingerprint[:8]} in {run_dir}; use mode='train'."
                )
            raise FileNotFoundError(f"Incomplete cache in {run_dir}")
    if mode == "load":
        raise FileNotFoundError(f"No cached run at {run_dir}")

    started = time.perf_counter()
    results = compute_fn()
    elapsed = time.perf_counter() - started
    metadata = {
        "name": name,
        "run_version": run_version,
        "fingerprint": fingerprint,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed, 3),
        "python": platform.python_version(),
        **{pkg: _package_version(pkg) for pkg in _PACKAGES},
        "config": config,
        "keys": list(results),
    }
    _save_run(run_dir, results, metadata)
    return results, {**metadata, "source": "computed"}


def describe_run(metadata: dict, language: str = "pt") -> str:
    """Return a human-readable provenance line for a cached run.

    Parameters
    ----------
    metadata:
        Metadata returned by :func:`cached_run`.
    language:
        ``"pt"`` or ``"en"``.

    Returns
    -------
    str
        One-line description of where the results came from.
    """
    path = f"outputs/runs/{metadata['name']}/{metadata['run_version']}"
    created = datetime.fromisoformat(metadata["created_at"]).strftime("%Y-%m-%d %H:%M")
    minutes = float(metadata["elapsed_seconds"]) / 60.0
    short = str(metadata["fingerprint"])[:8]
    loaded = metadata.get("source") == "loaded"
    if language == "pt":
        minutes_txt = f"{minutes:.1f}".replace(".", ",")
        if loaded:
            return (
                f"Resultados carregados de {path} (gerados em {created}, "
                f"{minutes_txt} min de treino; fingerprint {short})"
            )
        return (
            f"Resultados treinados agora e salvos em {path} ({created}, "
            f"{minutes_txt} min de treino; fingerprint {short})"
        )
    if loaded:
        return (
            f"Results loaded from {path} (generated on {created}, "
            f"{minutes:.1f} min of training; fingerprint {short})"
        )
    return (
        f"Results trained now and saved to {path} ({created}, "
        f"{minutes:.1f} min of training; fingerprint {short})"
    )
