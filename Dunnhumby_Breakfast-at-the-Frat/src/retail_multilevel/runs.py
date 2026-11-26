"""Experiment runs stored on disk: fit once by script, read many times by the notebooks.

Layout::

    outputs/runs/<experiment>/<run_id>/
        metadata.json      config, fingerprint, inputs, versions, metrics; written last
        tables/<name>.parquet
        json/<name>.json
        files/...          fitted models (e.g. R .rds) and any other artefact
    outputs/runs/LEDGER.jsonl   one line per completed run

A run is reused when its fingerprint matches: the fingerprint combines the configuration,
the content of the input files and the package source code. A run is only visible once
``metadata.json`` exists, so an interrupted fit never looks complete. The notebook language
never enters the fingerprint; EN and PT notebooks load the same run.
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path

import pandas as pd

from retail_multilevel.paths import OUTPUTS_DIR

RUNS_DIR = OUTPUTS_DIR / "runs"
PACKAGE_DIR = Path(__file__).resolve().parent
_VERSIONS = ("pandas", "numpy", "statsmodels", "rpy2", "gpboost", "lightgbm", "optuna")
_MODES = ("auto", "train", "load")


def _hash_files(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(p) for p in paths):
        digest.update(path.name.encode())
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _hash_package(package_dir: Path) -> str:
    files = sorted(p for p in package_dir.rglob("*.py") if "__pycache__" not in p.parts)
    digest = hashlib.sha256()
    for p in files:
        digest.update(p.relative_to(package_dir).as_posix().encode() + b"\0" + p.read_bytes() + b"\0")
    return digest.hexdigest()


def module_files(*modules) -> list[Path]:
    """Return the source files of imported ``modules``, for ``code_files`` in :func:`get_or_fit`."""
    return [Path(m.__file__).resolve() for m in modules]


def fingerprint(
    config: dict,
    inputs: Iterable[Path] = (),
    package_dir: Path | None = PACKAGE_DIR,
    code_files: Iterable[Path] | None = None,
) -> str:
    """Return a SHA-256 fingerprint of ``config`` (JSON), input file contents and code.

    The code part is ``code_files`` when given (only the modules the experiment depends on,
    so editing e.g. plotting code does not invalidate a fit); otherwise every ``*.py`` of
    ``package_dir``. Pass ``package_dir=None`` and no ``code_files`` to leave code out.
    """
    if code_files is not None:
        code = _hash_files(code_files)
    else:
        code = _hash_package(package_dir) if package_dir is not None else ""
    parts = {
        "config": json.dumps(config, sort_keys=True, default=str),
        "inputs": _hash_files(inputs),
        "package": code,
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


def _versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {"python": platform.python_version()}
    for name in _VERSIONS:
        try:
            out[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            out[name] = None
    return out


@dataclass
class Run:
    """One experiment run directory. Artefacts are written through its methods."""

    path: Path
    metadata: dict = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        """Directory name of the run."""
        return self.path.name

    @property
    def is_complete(self) -> bool:
        """Whether ``metadata.json`` was written (the run finished)."""
        return (self.path / "metadata.json").exists()

    def file(self, name: str) -> Path:
        """Return a path under ``files/`` for an arbitrary artefact (e.g. a saved R model)."""
        p = self.path / "files" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def save_table(self, name: str, df: pd.DataFrame) -> Path:
        """Save a DataFrame as ``tables/<name>.parquet``."""
        p = self.path / "tables" / f"{name}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p, index=False)
        return p

    def load_table(self, name: str) -> pd.DataFrame:
        """Load ``tables/<name>.parquet``."""
        return pd.read_parquet(self.path / "tables" / f"{name}.parquet")

    def save_json(self, name: str, obj) -> Path:
        """Save a JSON-serialisable object as ``json/<name>.json``."""
        p = self.path / "json" / f"{name}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
        return p

    def load_json(self, name: str):
        """Load ``json/<name>.json``."""
        return json.loads((self.path / "json" / f"{name}.json").read_text(encoding="utf-8"))

    @property
    def metrics(self) -> dict:
        """Metrics recorded when the run was finalised."""
        return self.metadata.get("metrics", {})


def _complete_runs(experiment: str, root: Path) -> list[Run]:
    exp_dir = root / experiment
    if not exp_dir.is_dir():
        return []
    runs = []
    for d in sorted(exp_dir.iterdir()):
        meta = d / "metadata.json"
        if meta.exists():
            runs.append(Run(d, json.loads(meta.read_text(encoding="utf-8"))))
    return runs


def load_run(experiment: str, run_id: str | None = None, root: Path = RUNS_DIR) -> Run:
    """Load a completed run (the latest one if ``run_id`` is None). Used by the notebooks."""
    runs = _complete_runs(experiment, root)
    if run_id is not None:
        runs = [r for r in runs if r.run_id == run_id]
    if not runs:
        raise FileNotFoundError(
            f"No completed run for experiment {experiment!r}"
            + (f" with id {run_id!r}" if run_id else "")
            + f" under {root}. Run the corresponding script in scripts/ first."
        )
    return max(runs, key=lambda r: r.metadata["finished_at"])


def find_run(experiment: str, fp: str, root: Path = RUNS_DIR) -> Run | None:
    """Return the latest completed run of ``experiment`` with fingerprint ``fp``, if any."""
    matches = [r for r in _complete_runs(experiment, root) if r.metadata.get("fingerprint") == fp]
    return max(matches, key=lambda r: r.metadata["finished_at"]) if matches else None


def get_or_fit(
    experiment: str,
    config: dict,
    fit: Callable[[Run], dict | None],
    inputs: Iterable[Path] = (),
    mode: str = "auto",
    root: Path = RUNS_DIR,
    package_dir: Path | None = PACKAGE_DIR,
    code_files: Iterable[Path] | None = None,
) -> Run:
    """Reuse a matching run or fit a new one. Used by the scripts, never by the notebooks.

    Parameters
    ----------
    experiment
        Experiment name (directory under ``root``).
    config
        JSON-serialisable configuration of the experiment.
    fit
        ``fit(run)`` writes the artefacts through ``run`` and returns a metrics dict (or None).
    inputs
        Input files whose content enters the fingerprint (e.g. the panel Parquet).
    mode
        ``'auto'`` reuses a run with the same fingerprint, else fits; ``'train'`` always fits
        (explicit request); ``'load'`` never fits and fails if no matching run exists.
    root, package_dir
        Storage root and package whose code enters the fingerprint.
    code_files
        Source files the experiment depends on (see :func:`module_files`); when given, they
        replace the whole-package hash, so unrelated edits do not trigger a refit.
    """
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    inputs = [Path(p) for p in inputs]
    fp = fingerprint(config, inputs, package_dir, code_files)
    if mode != "train":
        existing = find_run(experiment, fp, root)
        if existing is not None:
            return existing
        if mode == "load":
            raise FileNotFoundError(f"No run of {experiment!r} matches the current config, inputs and code.")

    started = datetime.now()
    base = root / experiment / f"{started:%Y%m%d-%H%M%S}-{fp[:8]}"
    path, k = base, 1
    while path.exists():  # forced refits of the same config within one second
        k += 1
        path = base.with_name(f"{base.name}-{k}")
    run = Run(path)
    run.path.mkdir(parents=True, exist_ok=False)
    metrics = fit(run) or {}
    finished = datetime.now()
    run.metadata = {
        "experiment": experiment,
        "run_id": run.run_id,
        "fingerprint": fp,
        "config": config,
        "inputs": [str(p) for p in inputs],
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="microseconds"),
        "duration_s": round((finished - started).total_seconds(), 2),
        "versions": _versions(),
        "metrics": metrics,
    }
    (run.path / "metadata.json").write_text(json.dumps(run.metadata, indent=2, default=str), encoding="utf-8")
    with (root / "LEDGER.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {k: run.metadata[k] for k in ("experiment", "run_id", "fingerprint", "finished_at", "metrics")},
                default=str,
            )
            + "\n"
        )
    return run
