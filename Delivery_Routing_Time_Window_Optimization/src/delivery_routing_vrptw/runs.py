"""Resume guard shared by the runner scripts.

A runner that resumes into an existing ``results.csv`` must only reuse rows produced by the same effective
configuration. The guard fingerprints that configuration (SHA-256 of a canonical JSON), refuses to resume
into a folder with a different fingerprint, and refuses legacy folders (results without a fingerprint)
unless explicitly allowed. Metadata is written only after the check passed.

The fingerprint includes the number of worker processes: with wall-clock time limits the process count
changes results (CPU contention; block B4 exists to compare 4 vs 8 processes). Timestamps and paths are not
part of it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .solver import SolverConfig

FINGERPRINT_COL = "config_fingerprint"
METADATA_NAME = "run_metadata.json"
LEGACY_METADATA_NAME = "run_metadata.legacy.json"
RESULTS_NAME = "results.csv"


class ResumeError(RuntimeError):
    """Raised when resuming into an output folder would mix results of different configurations."""


def add_resume_args(parser) -> None:
    """Add ``--allow-legacy-resume`` to an ``argparse`` parser."""
    parser.add_argument(
        "--allow-legacy-resume", action="store_true",
        help="resume into a folder whose results carry no configuration fingerprint (frozen/legacy runs); "
             "without it such a folder is refused",
    )


def config_fingerprint(config: dict) -> str:
    """Return a stable SHA-256 hex digest of the effective configuration.

    The dictionary is serialised as canonical JSON (sorted keys, no whitespace), so the digest does not
    depend on key order. Only values that determine results belong in it: the number of worker processes
    is one (wall-clock limits make results depend on CPU contention; B4 compares 4 vs 8), timestamps and paths
    are not.

    Parameters
    ----------
    config : dict
        JSON-serialisable effective configuration.

    Returns
    -------
    str
        64-character hexadecimal digest.
    """
    text = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fixed_cost_fields(config: SolverConfig) -> dict:
    """Return the integer scale and fixed-cost rule of a solver configuration (v1 absolute, v2 relative ``k``)."""
    relative = config.fixed_cost_k is not None
    return {
        "scale": config.scale,
        "fixed_cost_rule": "relative_k_scale_Dub" if relative else "absolute",
        "fixed_cost_k": config.fixed_cost_k,
        "vehicle_fixed_cost": None if relative else config.vehicle_fixed_cost,
    }


def solver_fields(config: SolverConfig, with_time_limit: bool = True) -> dict:
    """Return the result-determining fields of a solver configuration (see :func:`fixed_cost_fields`).

    ``with_time_limit=False`` leaves the budget out when it is a cell key of the runner (a time sweep).
    """
    fields = {"first_solution": config.first_solution, "metaheuristic": config.metaheuristic}
    if with_time_limit:
        fields["time_limit_s"] = config.time_limit_s
    return {**fields, **fixed_cost_fields(config)}


def file_fingerprint(path: str | Path) -> str:
    """Return the SHA-256 of a file's bytes (an input that determines results, such as a reference plan file)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _has_rows(results: Path) -> bool:
    return results.exists() and results.stat().st_size > 0


def check_resume(
    out_dir: str | Path, fingerprint: str, allow_legacy: bool = False, results_name: str = RESULTS_NAME
) -> str:
    """Check that the output folder can be resumed with this configuration; never modifies anything.

    A folder with a non-empty results file requires ``run_metadata.json`` with the same fingerprint and,
    when the results have a ``config_fingerprint`` column, every row with that fingerprint. Folders without
    a fingerprint (legacy) are refused unless ``allow_legacy``.

    Parameters
    ----------
    out_dir : str or pathlib.Path
        Output folder of the run.
    fingerprint : str
        Fingerprint of the current effective configuration.
    allow_legacy : bool, default False
        Accept results and metadata that carry no fingerprint (resume at the user's own risk).
    results_name : str, default "results.csv"
        Results file inside ``out_dir``.

    Returns
    -------
    str
        ``"fresh"`` (nothing to resume), ``"resume"`` (same fingerprint) or ``"legacy"`` (accepted legacy folder).

    Raises
    ------
    ResumeError
        On a fingerprint mismatch, or on a legacy folder without ``allow_legacy``.
    """
    out_dir = Path(out_dir)
    results = out_dir / results_name
    if not _has_rows(results):
        return "fresh"
    hint = "choose a new --out (the existing results are neither reused nor overwritten)"
    meta_path = out_dir / METADATA_NAME
    meta_fp = None
    if meta_path.exists():
        try:
            meta_fp = json.loads(meta_path.read_text(encoding="utf-8")).get(FINGERPRINT_COL)
        except json.JSONDecodeError:
            meta_fp = None
    header = pd.read_csv(results, nrows=0).columns
    legacy = meta_fp is None or FINGERPRINT_COL not in header
    if meta_fp is not None and meta_fp != fingerprint:
        raise ResumeError(
            f"refused: {out_dir} was produced by a different configuration "
            f"(fingerprint {meta_fp[:12]}... != {fingerprint[:12]}...); {hint}."
        )
    if FINGERPRINT_COL in header:
        found = pd.read_csv(results, usecols=[FINGERPRINT_COL], dtype=str)[FINGERPRINT_COL]
        other = found.dropna()[found.dropna() != fingerprint]
        if len(other):
            raise ResumeError(
                f"refused: {len(other)} row(s) of {results} carry a different configuration fingerprint; {hint}."
            )
        legacy = legacy or bool(found.isna().any())
    if legacy:
        if not allow_legacy:
            raise ResumeError(
                f"refused: {out_dir} has results without a configuration fingerprint (legacy folder), so they "
                f"cannot be checked against the current configuration; {hint}, or pass --allow-legacy-resume "
                "to resume it knowingly."
            )
        return "legacy"
    return "resume"


def start_run(
    out_dir: str | Path,
    fields: dict,
    metadata: dict,
    allow_legacy: bool = False,
    results_name: str = RESULTS_NAME,
) -> str:
    """Check the resume guard and only then write ``run_metadata.json`` (with the fingerprint).

    Parameters
    ----------
    out_dir : str or pathlib.Path
        Output folder (created after the check passed).
    fields : dict
        Effective configuration that determines results; fingerprinted and stored in the metadata.
    metadata : dict
        Run metadata; ``config_fingerprint`` and ``config_fingerprint_fields`` are added.
    allow_legacy : bool, default False
        Accept a legacy folder; its old metadata is then kept as ``run_metadata.legacy.json``.
    results_name : str, default "results.csv"
        Results file inside ``out_dir``.

    Returns
    -------
    str
        The fingerprint.

    Raises
    ------
    ResumeError
        See :func:`check_resume`; nothing is written in that case.
    """
    out_dir = Path(out_dir)
    fingerprint = config_fingerprint(fields)
    status = check_resume(out_dir, fingerprint, allow_legacy, results_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / METADATA_NAME
    extra = {FINGERPRINT_COL: fingerprint, "config_fingerprint_fields": fields}
    if status == "legacy":
        legacy_path = out_dir / LEGACY_METADATA_NAME
        if meta_path.exists() and not legacy_path.exists():
            legacy_path.write_text(meta_path.read_text(encoding="utf-8"), encoding="utf-8")
        extra["legacy_resume"] = True
    meta_path.write_text(json.dumps({**metadata, **extra}, indent=2), encoding="utf-8")
    return fingerprint


def append_rows(results: str | Path, frame: pd.DataFrame, fingerprint: str) -> None:
    """Append rows to the results CSV, stamping ``config_fingerprint`` and writing the header if needed.

    A legacy file without the fingerprint column is appended unstamped (a new column would misalign it).

    Parameters
    ----------
    results : str or pathlib.Path
        Results CSV.
    frame : pandas.DataFrame
        Rows to append.
    fingerprint : str
        Fingerprint of the current configuration.
    """
    results = Path(results)
    new_file = not _has_rows(results)
    frame = frame.copy()
    if new_file or FINGERPRINT_COL in pd.read_csv(results, nrows=0).columns:
        frame[FINGERPRINT_COL] = fingerprint
    frame.to_csv(results, mode="a", header=new_file, index=False)
