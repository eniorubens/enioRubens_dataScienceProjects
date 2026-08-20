"""V2.1-C one-shot confirmatory stress evaluation of the frozen V2 package.

Stage V2.1-C is the exact analogue of the S8 confirmatory evaluation, run
once on the sealed ``stress`` partition (2025-07-01..2025-12-31) instead of
``test``. Every batch scores both arms from the identical rows in one pass:
the frozen V2 hierarchical predictor (``v2_combined``) and the frozen S7
fallback alone (``s7_fallback_alone``). Only a joint truth/v2/s7 accumulator
and two scalar override counters are kept -- no narrative, identifier,
per-row score, or per-row margin is ever persisted.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import duckdb
import numpy as np

from .s7 import load_s7_predictor, validate_s7_manifest
from .s8 import (
    CRITICAL_CLASS,
    INPUT_LANGUAGE,
    bootstrap_confidence_intervals,
    metrics_from_confusion,
)
from .temporal_split import MODELED_FAMILIES
from .v2_detector import combine_detector_with_fallback, count_override_decisions
from .v2_package import load_v2_predictor, validate_v2_manifest


V2_STRESS_SCHEMA_VERSION = "v2-stress-confirmatory-protocol-v1"
V2_STRESS_RESULT_SCHEMA = "v2-stress-confirmatory-results-v1"
V2_STRESS_MANIFEST_SCHEMA = "v2-stress-confirmatory-manifest-v1"
V2_STRESS_CODE_SCHEMA = "v2-stress-runtime-v1"
STRESS_UNLOCK_ENV = "V2_STRESS_UNLOCK"
STRESS_PARTITION = "stress"
SEALED_PARTITIONS = ("monitor",)
DEFAULT_BATCH_SIZE = 4096
DEFAULT_MEMORY_LIMIT = "4GB"
DEFAULT_BOOTSTRAP_REPLICATES = 2000
DEFAULT_BOOTSTRAP_SEED = 42
DEFAULT_CONFIDENCE_LEVEL = 0.95
UNLOCK_SHA256 = (
    "4E43E6D5E549495BC4BA358B3C2040792024C1A5226DDC43529836F46DEC1A1F"
)
DEFAULT_CONFIG = Path("config/v2_stress_protocol.json")
DEFAULT_RESULT = Path("temp/v2/v2_stress_results.json")
DEFAULT_MANIFEST = Path("config/v2_stress_results.json")
_ALLOWED_TEXT_KEYS = {
    "test_all_text",
    "novel_text",
    "stress_all_text",
    "stress_novel_text",
}
FORBIDDEN_RESULT_KEY_TERMS = (
    "narrative",
    "text",
    "texts",
    "complaint id",
    "complaint_id",
    "individual score",
    "individual_score",
    "margin",
    "margins",
)


def _sha256(path: Path) -> str:
    """Return one file's uppercase SHA256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _file_signature(path: Path) -> dict[str, Any]:
    """Return the portable size and digest metadata for one file."""

    return {
        "path": str(path).replace("\\", "/"),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _validate_expected_signature(
    path: Path,
    expected: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    """Validate one file's frozen digest and optional byte size."""

    if not path.exists():
        raise FileNotFoundError(f"Frozen {label} is missing: {path}")
    actual = _file_signature(path)
    if actual["sha256"] != str(expected["sha256"]).upper():
        raise ValueError(f"Frozen {label} hash changed")
    expected_size = expected.get("size_bytes")
    if expected_size is not None and actual["size_bytes"] != int(expected_size):
        raise ValueError(f"Frozen {label} size changed")
    return actual


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one JSON object through a same-directory atomic replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object and reject non-object payloads."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("V2 stress JSON artifact must be an object")
    return payload


def _relative(path: Path, root: Path) -> str:
    """Return a portable path relative to the project root."""

    return path.resolve().relative_to(root.resolve()).as_posix()


def _quote_identifier(identifier: str) -> str:
    """Quote a configured DuckDB identifier."""

    if not identifier or "\x00" in identifier:
        raise ValueError("Invalid DuckDB identifier")
    return '"' + identifier.replace('"', '""') + '"'


def _default_project_root() -> Path:
    """Return the project root derived from this module's own location."""

    return Path(__file__).resolve().parents[2]


class V2StressConfig:
    """Represent the frozen V2.1-C protocol and its data-access boundary."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        """Initialize a validated configuration payload."""

        self.payload = dict(payload)
        self.validate_shape()

    @property
    def source(self) -> Mapping[str, Any]:
        """Return frozen source paths, columns, and hashes."""

        return self.payload["source"]

    @property
    def v2_freeze(self) -> Mapping[str, Any]:
        """Return the four frozen V2 package artifact signatures."""

        return self.payload["v2_freeze"]

    @property
    def s7_freeze(self) -> Mapping[str, Any]:
        """Return the four frozen S7 artifact signatures."""

        return self.payload["s7_freeze"]

    @property
    def gates(self) -> Mapping[str, float]:
        """Return the four simultaneous scientific gates."""

        return self.payload["gates"]

    @property
    def stress_scope(self) -> Mapping[str, str]:
        """Return the single confirmatory stress window."""

        return self.payload["stress_scope"]

    @property
    def model(self) -> Mapping[str, Any]:
        """Return the frozen hierarchical model contract."""

        return self.payload["model"]

    def validate_shape(self) -> None:
        """Validate immutable protocol values without reading dataset rows."""

        if self.payload.get("schema_version") != V2_STRESS_SCHEMA_VERSION:
            raise ValueError("Unexpected V2 stress protocol schema")
        if self.payload.get("status") != "FROZEN_FOR_CONFIRMATORY_TEST":
            raise ValueError("V2 stress protocol is not frozen")
        if self.payload.get("approved_on") != "2026-08-19":
            raise ValueError("Unexpected V2 stress approval date")
        if self.payload.get("approved_by") != "user":
            raise ValueError("V2 stress approval must be explicit user approval")
        if self.payload.get("confirmatory_partition") != STRESS_PARTITION:
            raise ValueError("V2 stress must use the stress partition")
        expected_scope = {"start": "2025-07-01", "end": "2025-12-31"}
        if dict(self.stress_scope) != expected_scope:
            raise ValueError("V2 stress scope differs from the frozen window")
        if tuple(self.payload.get("remaining_sealed", ())) != SEALED_PARTITIONS:
            raise ValueError("V2 stress sealed boundary is invalid")
        source = self.source
        if source.get("stress_start") != expected_scope["start"]:
            raise ValueError("V2 stress source start differs from stress_scope")
        if source.get("stress_end") != expected_scope["end"]:
            raise ValueError("V2 stress source end differs from stress_scope")
        model = self.model
        if model.get("input_language") != INPUT_LANGUAGE:
            raise ValueError("V2 stress input language must be en-US")
        if tuple(model.get("classes", ())) != tuple(MODELED_FAMILIES):
            raise ValueError("V2 stress class order differs from MODELED_FAMILIES")
        if model.get("critical_class") != CRITICAL_CLASS:
            raise ValueError("V2 stress critical class is invalid")
        if float(model.get("threshold")) != -0.13949530151425016:
            raise ValueError("V2 stress threshold differs from the frozen V2 package")
        if model.get("score_kind") != "critical_margin":
            raise ValueError("V2 stress score kind is invalid")
        if model.get("model_version") != "consumer-complaint-detector-v2":
            raise ValueError("V2 stress model version is invalid")
        if model.get("fallback_model_version") != "consumer-complaint-classifier-s7":
            raise ValueError("V2 stress fallback model version is invalid")
        if model.get("combination") != (
            "critical_override_at_or_above_calibrated_threshold"
        ):
            raise ValueError("V2 stress combination rule is invalid")
        gates = self.gates
        if int(gates.get("required_gate_count")) != 4:
            raise ValueError("V2 stress must require exactly four gates")
        if float(gates.get("macro_f1_min")) != 0.69:
            raise ValueError("V2 stress macro F1 gate limit is invalid")
        if float(gates.get("critical_f1_min")) != 0.2715:
            raise ValueError("V2 stress critical F1 gate limit is invalid")
        if float(gates.get("critical_precision_min")) != 0.2:
            raise ValueError("V2 stress critical precision gate limit is invalid")
        if float(gates.get("paired_critical_f1_gain_min")) != 0.0:
            raise ValueError("V2 stress paired gain gate limit is invalid")
        if gates.get("paired_gain_strict") is not True:
            raise ValueError("V2 stress paired gain gate must be strict")
        arms = self.payload.get("arms", {})
        if arms.get("primary") != "v2_combined":
            raise ValueError("V2 stress primary arm must be v2_combined")
        if arms.get("control") != "s7_fallback_alone":
            raise ValueError("V2 stress control arm must be s7_fallback_alone")
        if arms.get("scored_in_one_pass") is not True:
            raise ValueError("V2 stress arms must be scored in one pass")
        if int(arms.get("seal_openings", -1)) != 1:
            raise ValueError("V2 stress must record exactly one seal opening")
        bootstrap = self.payload.get("bootstrap", {})
        if bootstrap != {
            "replicates": DEFAULT_BOOTSTRAP_REPLICATES,
            "seed": DEFAULT_BOOTSTRAP_SEED,
            "confidence_level": DEFAULT_CONFIDENCE_LEVEL,
            "diagnostic_only": True,
            "includes_paired_difference": True,
        }:
            raise ValueError("V2 stress bootstrap configuration is invalid")
        access = self.payload.get("access", {})
        if access.get("batch_size") != DEFAULT_BATCH_SIZE:
            raise ValueError("V2 stress batch size is invalid")
        if access.get("memory_limit") != DEFAULT_MEMORY_LIMIT:
            raise ValueError("V2 stress memory limit is invalid")
        if access.get("threads") != 1:
            raise ValueError("V2 stress must use one DuckDB thread")
        if access.get("unlock_env") != STRESS_UNLOCK_ENV:
            raise ValueError("V2 stress unlock environment variable is invalid")
        if access.get("unlock_sha256") != UNLOCK_SHA256:
            raise ValueError("V2 stress unlock digest is invalid")
        temp_directory = access.get("temp_directory")
        if not isinstance(temp_directory, str) or Path(temp_directory).is_absolute():
            raise ValueError("V2 stress DuckDB temp directory must be relative")
        if ".." in Path(temp_directory).parts:
            raise ValueError("V2 stress DuckDB temp directory escapes the project")
        boundary = self.payload.get("boundary", {})
        if tuple(boundary.get("reads_partitions", ())) != (STRESS_PARTITION,):
            raise ValueError("V2 stress boundary must read only stress")
        if tuple(boundary.get("sealed_after_run", ())) != SEALED_PARTITIONS:
            raise ValueError("V2 stress boundary sealed-after-run is invalid")
        for field in (
            "persists_narratives_or_identifiers",
            "persists_individual_scores",
            "persists_stress_cache",
        ):
            if boundary.get(field) is not False:
                raise ValueError(f"V2 stress boundary flag must be false: {field}")
        if boundary.get("aggregate_only") is not True:
            raise ValueError("V2 stress boundary must be aggregate only")
        deployment = self.payload.get("deployment", {})
        if deployment.get("deployment_authorized") is not False:
            raise ValueError("V2 stress must not authorize deployment")
        if deployment.get("confirmation_never_authorizes_deployment") is not True:
            raise ValueError("V2 stress must declare confirmation never deploys")
        outputs = self.payload.get("outputs", {})
        if outputs.get("result") != DEFAULT_RESULT.as_posix():
            raise ValueError("V2 stress result output path is invalid")
        if outputs.get("manifest") != DEFAULT_MANIFEST.as_posix():
            raise ValueError("V2 stress manifest output path is invalid")
        run_defaults = self.payload.get("run_defaults", {})
        if run_defaults.get("batch_size") != DEFAULT_BATCH_SIZE:
            raise ValueError("V2 stress run default batch size is invalid")
        if run_defaults.get("memory_limit") != DEFAULT_MEMORY_LIMIT:
            raise ValueError("V2 stress run default memory limit is invalid")
        if run_defaults.get("threads") != 1:
            raise ValueError("V2 stress run defaults must use one thread")
        expectation = self.payload.get("expectation", {})
        if expectation.get("diagnostic_only") is not True:
            raise ValueError("V2 stress expectation must be diagnostic only")
        if expectation.get("not_a_gate") is not True:
            raise ValueError("V2 stress expectation must not be a gate")
        promotion = self.payload.get("partition_promotion", {})
        if promotion.get("would_pass_pilot_criteria") is not True:
            raise ValueError("V2 stress partition promotion evidence is invalid")
        if promotion.get("monitor_remains_diagnostic") is not True:
            raise ValueError("V2 stress must keep monitor diagnostic")
        for key in (
            "source_path",
            "index_path",
            "s2_report",
            "s3_protocol",
            "v2_config",
            "v2_manifest",
            "v2_result",
            "v2_bundle",
            "s7_config",
            "s7_manifest",
            "s7_result",
            "s7_bundle",
        ):
            if not isinstance(self.payload.get("paths", {}).get(key), str):
                raise ValueError(f"V2 stress path is missing: {key}")

    def to_dict(self) -> dict[str, Any]:
        """Return a detached serializable configuration."""

        return json.loads(json.dumps(self.payload))


def load_v2_stress_config(path: str | Path = DEFAULT_CONFIG) -> V2StressConfig:
    """Load and validate the frozen V2.1-C protocol configuration."""

    return V2StressConfig(_read_json(Path(path).expanduser().resolve()))


def validate_frozen_metadata(
    config: V2StressConfig,
    project_root: str | Path,
    *,
    include_raw: bool = False,
) -> dict[str, Any]:
    """Validate frozen metadata and hashes without querying dataset rows.

    Args:
        config: Frozen V2.1-C configuration.
        project_root: Project directory containing the relative artifacts.
        include_raw: Permit raw-source metadata hashing after the unlock guard.

    Returns:
        Validated V2 and S7 freeze signatures plus S2 evidence metadata.
    """

    root = Path(project_root).expanduser().resolve()
    v2_signatures: dict[str, Any] = {}
    for key, expected in config.v2_freeze.items():
        path = root / str(expected["path"])
        v2_signatures[key] = _validate_expected_signature(path, expected, f"V2 {key}")
    s7_signatures: dict[str, Any] = {}
    for key, expected in config.s7_freeze.items():
        path = root / str(expected["path"])
        s7_signatures[key] = _validate_expected_signature(path, expected, f"S7 {key}")
    source = config.source
    source_path = root / config.payload["paths"]["source_path"]
    index_path = root / config.payload["paths"]["index_path"]
    _validate_expected_signature(
        index_path,
        {
            "sha256": source["index_sha256"],
            "size_bytes": source["index_size_bytes"],
        },
        "S2 modeling index",
    )
    s2_report = root / config.payload["paths"]["s2_report"]
    _validate_expected_signature(
        s2_report,
        {
            "sha256": source["s2_report_sha256"],
            "size_bytes": source["s2_report_size_bytes"],
        },
        "S2 pilot report",
    )
    report = _read_json(s2_report)
    s3_protocol = root / config.payload["paths"]["s3_protocol"]
    _validate_expected_signature(
        s3_protocol,
        {
            "sha256": source["s3_protocol_sha256"],
            "size_bytes": source["s3_protocol_size_bytes"],
        },
        "S3 protocol",
    )
    candidate = next(
        (
            item
            for item in report.get("report", report).get("candidates", [])
            if item.get("candidate", {}).get("name") == "post_2023_taxonomy"
        ),
        None,
    )
    if not candidate or candidate.get("candidate_status") != "PASS":
        raise ValueError("S2 post_2023_taxonomy evidence is not PASS")
    if candidate.get("eligible_class_count") != len(MODELED_FAMILIES):
        raise ValueError("S2 evidence does not cover nine classes")
    if include_raw:
        raw = _validate_expected_signature(
            source_path,
            {
                "sha256": source["raw_sha256"],
                "size_bytes": source["raw_size_bytes"],
            },
            "raw CFPB source",
        )
        v2_signatures["raw_source"] = raw
    return {
        "v2": v2_signatures,
        "s7": s7_signatures,
        "s2": {
            "protocol_id": "post_2023_taxonomy",
            "candidate_status": candidate["candidate_status"],
            "eligible_class_count": candidate["eligible_class_count"],
            "s3_protocol_sha256": source["s3_protocol_sha256"],
            "evidence": config.payload["s2_evidence"],
        },
    }


def require_confirmatory_unlock(config: V2StressConfig) -> None:
    """Require the one-time full-run token without exposing its plaintext."""

    token = os.environ.get(STRESS_UNLOCK_ENV)
    if not token or hashlib.sha256(token.encode("utf-8")).hexdigest().upper() != (
        config.payload["access"]["unlock_sha256"]
    ):
        raise PermissionError(
            "V2 stress full mode requires the approved unlock token"
        )


def evaluate_gates(
    v2_metrics: Mapping[str, Any],
    paired_gain: float,
    gates: Mapping[str, float],
) -> dict[str, Any]:
    """Evaluate the four simultaneous V2.1-C confirmatory gates.

    Three gates are absolute limits on the ``v2_combined`` arm's scientific
    metrics; the fourth is the strict paired critical-F1 gain over
    ``s7_fallback_alone``, which must be positive, not merely non-negative.
    All four are evaluated on the scientific primary view only.

    Args:
        v2_metrics: Aggregate metrics for the ``v2_combined`` arm, as
            returned by :func:`metrics_from_confusion`.
        paired_gain: ``v2_combined`` critical F1 minus ``s7_fallback_alone``
            critical F1, from the same joint sample.
        gates: Frozen gate limits from the protocol.

    Returns:
        ``{"required_gate_count", "passed_count", "passed", "results"}``,
        where ``results`` carries one named check per gate.
    """

    checks = (
        ("macro_f1", float(v2_metrics["macro_f1"]),
         float(gates["macro_f1_min"]), False),
        ("critical_f1", float(v2_metrics["critical_f1"]),
         float(gates["critical_f1_min"]), False),
        ("critical_precision", float(v2_metrics["critical_precision"]),
         float(gates["critical_precision_min"]), False),
        ("paired_critical_f1_gain", float(paired_gain),
         float(gates["paired_critical_f1_gain_min"]), True),
    )
    results = []
    for name, observed, limit, strict in checks:
        passed = observed > limit if strict else observed >= limit
        results.append({
            "name": name,
            "observed": observed,
            "limit": limit,
            "strict": strict,
            "passed": bool(passed),
        })
    passed_count = sum(1 for item in results if item["passed"])
    return {
        "required_gate_count": 4,
        "passed_count": passed_count,
        "passed": passed_count == 4,
        "results": results,
    }


def paired_bootstrap_interval(
    joint: Sequence[Sequence[Sequence[int]]],
    *,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> dict[str, Any]:
    """Bootstrap the paired critical-F1 gain from one joint truth/v2/s7 table.

    Each replicate draws one multinomial sample over the full ``(9, 9, 9)``
    joint table, using the observed cell probabilities, and marginalizes it
    to both arms' confusion matrices before differencing their critical F1.
    Both arms are always scored from the identical resampled rows, so their
    correlation is preserved. Resampling each arm's confusion matrix
    independently -- the way :func:`bootstrap_confidence_intervals` does for
    a single arm -- would discard that correlation and understate how
    tightly the two arms actually move together; it is deliberately not
    used for this paired difference.

    The procedure is diagnostic only: it never changes gate decisions.

    Args:
        joint: The observed ``(9, 9, 9)`` truth/v2_label/s7_label counts.
        replicates: Number of multinomial replicates.
        seed: Deterministic RNG seed.
        confidence_level: Two-sided percentile interval width.

    Returns:
        ``{"method", "replicates", "seed", "confidence_level",
        "diagnostic_only", "lower", "upper"}`` for the paired gain.

    Raises:
        ValueError: If ``replicates``, ``confidence_level``, or the joint
            table shape is invalid.
    """

    if replicates <= 0 or not 0 < confidence_level < 1:
        raise ValueError("Invalid V2 stress bootstrap parameters")
    matrix = np.asarray(joint, dtype=np.int64)
    size = len(MODELED_FAMILIES)
    if matrix.shape != (size, size, size):
        raise ValueError("V2 stress joint table shape is invalid")
    total = int(matrix.sum())
    if total <= 0:
        raise ValueError("V2 stress joint table is empty")
    probabilities = matrix.reshape(-1).astype(np.float64) / total
    rng = np.random.default_rng(seed)
    gains = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        sampled = rng.multinomial(total, probabilities).reshape(size, size, size)
        v2_metrics = metrics_from_confusion(sampled.sum(axis=2))
        s7_metrics = metrics_from_confusion(sampled.sum(axis=1))
        gains[replicate] = v2_metrics["critical_f1"] - s7_metrics["critical_f1"]
    alpha = (1 - confidence_level) / 2
    return {
        "method": "paired_multinomial_from_joint_truth_v2_s7_table",
        "replicates": replicates,
        "seed": seed,
        "confidence_level": confidence_level,
        "diagnostic_only": True,
        "lower": float(np.quantile(gains, alpha)),
        "upper": float(np.quantile(gains, 1 - alpha)),
    }


def _scope_counts(
    connection: duckdb.DuckDBPyConnection,
    index_path: Path,
    config: V2StressConfig,
) -> dict[str, Any]:
    """Materialize only hash and label scope state in DuckDB."""

    source = config.source
    families = ", ".join("?" for _ in MODELED_FAMILIES)
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE v2c_index AS
        SELECT "Complaint ID", received_date, product_family,
            normalized_group_hash, normalized_length
        FROM read_parquet(?)
        WHERE product_family IN (""" + families + ")",
        [str(index_path), *MODELED_FAMILIES],
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE v2c_test_base AS
        SELECT i.*
        FROM v2c_index AS i
        WHERE TRY_CAST(i.received_date AS DATE) BETWEEN ? AND ?
          AND NOT EXISTS (
            SELECT 1 FROM v2c_index AS prior
            WHERE TRY_CAST(prior.received_date AS DATE) < ?
              AND prior.normalized_group_hash = i.normalized_group_hash
              AND prior.normalized_length = i.normalized_length
          )
        """,
        [
            source["stress_start"],
            source["stress_end"],
            source["stress_start"],
        ],
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE v2c_clean_groups AS
        SELECT normalized_group_hash, normalized_length,
            min(product_family) AS product_family
        FROM v2c_test_base
        GROUP BY normalized_group_hash, normalized_length
        HAVING count(DISTINCT product_family) = 1
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE v2c_primary AS
        SELECT * EXCLUDE (row_number)
        FROM (
            SELECT b.*, row_number() OVER (
                PARTITION BY b.normalized_group_hash, b.normalized_length
                ORDER BY b."Complaint ID"
            ) AS row_number
            FROM v2c_test_base AS b
            INNER JOIN v2c_clean_groups AS g USING (
                normalized_group_hash, normalized_length
            )
        )
        WHERE row_number = 1
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE v2c_operational AS
        SELECT b.* FROM v2c_test_base AS b
        INNER JOIN v2c_clean_groups AS g USING (
            normalized_group_hash, normalized_length
        )
        """
    )
    result = connection.execute(
        """
        SELECT
            (SELECT count(*) FROM v2c_test_base) AS novel_test_lines,
            (SELECT count(*) FROM v2c_primary) AS primary_representatives,
            (SELECT count(*) FROM v2c_operational) AS operational_lines,
            (SELECT count(DISTINCT (normalized_group_hash, normalized_length))
             FROM v2c_test_base) AS novel_unique_groups,
            (SELECT count(*) FROM v2c_clean_groups) AS clean_unique_groups,
            (SELECT count(*) FROM v2c_clean_groups
             WHERE product_family = ?) AS critical_novel_groups,
            (SELECT count(DISTINCT (normalized_group_hash, normalized_length))
             FROM v2c_index
             WHERE TRY_CAST(received_date AS DATE) BETWEEN ? AND ?)
             AS all_test_unique_groups,
            (SELECT count(*) FROM v2c_index
             WHERE TRY_CAST(received_date AS DATE) BETWEEN ? AND ?)
             AS test_all_text,
            (SELECT count(*) - count(DISTINCT (
                normalized_group_hash, normalized_length
            )) FROM v2c_primary) AS scientific_duplicate_representatives,
            (SELECT count(*) FROM v2c_test_base b
             WHERE NOT EXISTS (
                 SELECT 1 FROM v2c_clean_groups g
                 WHERE g.normalized_group_hash = b.normalized_group_hash
                   AND g.normalized_length = b.normalized_length
             )) AS ambiguous_test_lines,
            (SELECT count(*) FROM v2c_index i
             WHERE TRY_CAST(i.received_date AS DATE) < ?) AS prior_modeled_lines,
            (SELECT count(DISTINCT (i.normalized_group_hash, i.normalized_length))
             FROM v2c_index i
             WHERE TRY_CAST(i.received_date AS DATE) < ?) AS prior_modeled_groups
        """,
        [
            CRITICAL_CLASS,
            source["stress_start"],
            source["stress_end"],
            source["stress_start"],
            source["stress_end"],
            source["stress_start"],
            source["stress_start"],
        ],
    ).fetchone()
    names = (
        "novel_test_lines",
        "primary_representatives",
        "operational_lines",
        "novel_unique_groups",
        "clean_unique_groups",
        "critical_novel_groups",
        "all_test_unique_groups",
        "test_all_text",
        "scientific_duplicate_representatives",
        "ambiguous_test_lines",
        "prior_modeled_lines",
        "prior_modeled_groups",
    )
    counts = dict(zip(names, (int(value) for value in result)))
    counts["ambiguous_unique_groups"] = (
        counts["novel_unique_groups"] - counts["clean_unique_groups"]
    )
    counts["seen_previously_test_lines"] = (
        counts["test_all_text"] - counts["novel_test_lines"]
    )
    counts["seen_previously_test_groups"] = (
        counts["all_test_unique_groups"] - counts["novel_unique_groups"]
    )
    return counts


def _iter_raw_batches(
    connection: duckdb.DuckDBPyConnection,
    source_path: Path,
    scope_table: str,
    config: V2StressConfig,
) -> Iterable[tuple[list[str], list[str]]]:
    """Stream raw narratives and labels for one already-defined scope."""

    text_column = _quote_identifier(config.source["text_column"])
    query = (
        "SELECT s.product_family, CAST(r."
        + text_column
        + " AS VARCHAR) AS narrative FROM read_parquet(?) r "
        "INNER JOIN "
        + scope_table
        + ' s ON r."Complaint ID" = s."Complaint ID" '
        'ORDER BY s."Complaint ID"'
    )
    reader = connection.execute(query, [str(source_path)]).to_arrow_reader(
        DEFAULT_BATCH_SIZE
    )
    for batch in reader:
        labels = [str(value) for value in batch.column("product_family").to_pylist()]
        raw_texts = batch.column("narrative").to_pylist()
        if any(value is None or not str(value).strip() for value in raw_texts):
            raise ValueError(
                "V2 stress raw narrative join returned an empty narrative"
            )
        texts = [str(value) for value in raw_texts]
        yield labels, texts


def _score_scope_two_arms(
    connection: duckdb.DuckDBPyConnection,
    source_path: Path,
    scope_table: str,
    v2_predictor: Any,
    s7_predictor: Any,
    threshold: float,
    config: V2StressConfig,
) -> tuple[np.ndarray, int, int, int]:
    """Score one scope into a joint truth/v2/s7 accumulator, in one pass.

    Both arms are scored from the identical batch of rows: the frozen S7
    fallback produces one multiclass label per row, the frozen V2 stage-A
    margin is thresholded into a boolean override decision, and
    :func:`combine_detector_with_fallback` produces the ``v2_combined``
    label. Every row updates exactly one ``(truth, v2_label, s7_label)``
    cell, so the two arms can never see a different sample of rows.

    Args:
        connection: Open DuckDB connection with the scope tables built.
        source_path: Raw dataset parquet path.
        scope_table: Already-materialized scope temp table name.
        v2_predictor: Frozen V2 hierarchical predictor (stage-A margins).
        s7_predictor: Frozen S7 fallback predictor.
        threshold: Frozen stage-A critical-margin threshold.
        config: Validated V2 stress configuration.

    Returns:
        A ``(joint, override_decisions, effective_overrides, rows)`` tuple,
        where ``joint`` is an int64 array of shape ``(9, 9, 9)`` indexed
        ``[truth, v2_label, s7_label]`` over ``MODELED_FAMILIES`` order.

    Raises:
        ValueError: If a batch is misaligned or returns an unknown label.
    """

    positions = {label: index for index, label in enumerate(MODELED_FAMILIES)}
    size = len(MODELED_FAMILIES)
    joint = np.zeros((size, size, size), dtype=np.int64)
    override_decisions = 0
    effective_overrides = 0
    rows = 0
    for labels, texts in _iter_raw_batches(
        connection, source_path, scope_table, config
    ):
        if len(labels) != len(texts):
            raise ValueError(
                f"V2 stress {scope_table} batch labels/texts differ in length"
            )
        unknown_labels = set(labels).difference(positions)
        if unknown_labels:
            raise ValueError(
                f"V2 stress {scope_table} contains unknown truth labels: "
                f"{sorted(unknown_labels)}"
            )
        s7_predicted = s7_predictor.predict(texts, input_language=INPUT_LANGUAGE)
        s7_predictions = getattr(s7_predicted, "predictions", None)
        if s7_predictions is None or len(s7_predictions) != len(labels):
            raise ValueError(
                f"V2 stress {scope_table} S7 predictor returned an invalid "
                "batch length"
            )
        s7_labels: list[str] = []
        for item in s7_predictions:
            label = getattr(item, "label", None)
            if label not in positions:
                raise ValueError(
                    f"V2 stress {scope_table} S7 predictor returned unknown "
                    f"label: {label!r}"
                )
            s7_labels.append(label)
        margins = v2_predictor.decision_margins(texts, input_language=INPUT_LANGUAGE)
        if len(margins) != len(labels):
            raise ValueError(
                f"V2 stress {scope_table} V2 predictor returned an invalid "
                "batch length"
            )
        decisions = np.asarray(margins, dtype=np.float64) >= float(threshold)
        v2_labels = combine_detector_with_fallback(decisions, s7_labels)
        batch_override, batch_effective = count_override_decisions(
            decisions, s7_labels
        )
        override_decisions += batch_override
        effective_overrides += batch_effective
        for truth, v2_label, s7_label in zip(labels, v2_labels, s7_labels):
            joint[positions[truth], positions[v2_label], positions[s7_label]] += 1
            rows += 1
    return joint, override_decisions, effective_overrides, rows


def _validate_scope_row_count(scope_table: str, observed: int, expected: int) -> None:
    """Reject a raw join that silently loses or duplicates scoped rows."""

    if observed != expected:
        raise ValueError(
            f"V2 stress {scope_table} raw join rows {observed} differ from "
            f"expected {expected}"
        )


def _view_block(
    view: str,
    joint: np.ndarray,
    rows: int,
    override_decisions: int,
    effective_overrides: int,
) -> dict[str, Any]:
    """Marginalize one joint accumulator into the published view block.

    Args:
        view: ``scientific`` or ``operational``.
        joint: The accumulated ``(9, 9, 9)`` truth/v2/s7 joint counts.
        rows: Total scored rows in this view.
        override_decisions: Raw stage-A override decisions in this view.
        effective_overrides: Effective stage-A overrides in this view.

    Returns:
        The two-arm result block for this view.
    """

    v2_confusion = joint.sum(axis=2)
    s7_confusion = joint.sum(axis=1)
    v2_metrics = metrics_from_confusion(v2_confusion)
    s7_metrics = metrics_from_confusion(s7_confusion)
    return {
        "view": view,
        "rows": int(rows),
        "arms": {
            "v2_combined": {
                "confusion": v2_confusion.tolist(),
                "metrics": v2_metrics,
            },
            "s7_fallback_alone": {
                "confusion": s7_confusion.tolist(),
                "metrics": s7_metrics,
            },
        },
        "paired": {
            "critical_f1_gain": (
                float(v2_metrics["critical_f1"])
                - float(s7_metrics["critical_f1"])
            ),
            "macro_f1_gain": (
                float(v2_metrics["macro_f1"]) - float(s7_metrics["macro_f1"])
            ),
            "critical_precision_gain": (
                float(v2_metrics["critical_precision"])
                - float(s7_metrics["critical_precision"])
            ),
            "critical_recall_gain": (
                float(v2_metrics["critical_recall"])
                - float(s7_metrics["critical_recall"])
            ),
        },
        "override": {
            "override_decisions": int(override_decisions),
            "effective_overrides": int(effective_overrides),
        },
    }


def _check_result_privacy(
    payload: Mapping[str, Any],
    forbidden_texts: Iterable[str] = (),
) -> None:
    """Reject raw text, identifiers, individual scores, or margins."""

    forbidden_values = tuple(value for value in forbidden_texts if value)

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                lowered = str(key).lower().replace("-", "_")
                score_key = lowered in {"score", "scores"} or (
                    "individual_score" in lowered
                )
                text_key = "text" in lowered and lowered not in _ALLOWED_TEXT_KEYS
                exact_key = lowered in FORBIDDEN_RESULT_KEY_TERMS
                if score_key or text_key or exact_key:
                    raise ValueError(
                        f"V2 stress result contains forbidden key: {path}.{key}"
                    )
                visit(item, f"{path}.{key}")
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
            return
        if isinstance(value, str) and any(
            text in value for text in forbidden_values
        ):
            raise ValueError(f"V2 stress result contains input text at {path}")

    visit(payload, "result")


def _signature(
    config_path: Path,
    source: Mapping[str, Any],
    v2_freeze: Mapping[str, Any],
    s7_freeze: Mapping[str, Any],
) -> str:
    """Create a stable run signature from protocol and frozen metadata only."""

    payload = {
        "code_schema": V2_STRESS_CODE_SCHEMA,
        "config": _file_signature(config_path),
        "source": dict(source),
        "v2_freeze": dict(v2_freeze),
        "s7_freeze": dict(s7_freeze),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest().upper()


def _require_relative_path(value: Any, label: str) -> str:
    """Validate and normalize one project-relative manifest path."""

    if not isinstance(value, str) or not value:
        raise ValueError(f"V2 stress manifest {label} path is invalid")
    path = Path(value)
    windows_path = path.drive or value.startswith("\\")
    if path.is_absolute() or windows_path or ".." in path.parts:
        raise ValueError(f"V2 stress manifest {label} path must be relative")
    return value.replace("\\", "/")


def _validate_complete_result(
    payload: Mapping[str, Any],
    signature: str | None = None,
) -> None:
    """Validate a complete aggregate result without accessing dataset files."""

    if payload.get("schema_version") != V2_STRESS_RESULT_SCHEMA:
        raise ValueError("Unexpected V2 stress result schema")
    if payload.get("code_schema") != V2_STRESS_CODE_SCHEMA:
        raise ValueError("V2 stress result code schema is stale")
    if payload.get("complete") is not True:
        raise ValueError("V2 stress result is not complete")
    if signature is not None and payload.get("signature") != signature:
        raise ValueError("V2 stress result signature differs from frozen protocol")
    if payload.get("status") not in {"CONFIRMED", "NOT_CONFIRMED"}:
        raise ValueError("V2 stress complete result status is invalid")
    if payload.get("confirmatory") is not True:
        raise ValueError("V2 stress result confirmatory flag is invalid")
    if payload.get("deploy") is not False:
        raise ValueError("V2 stress result cannot authorize deployment")
    if payload.get("stress_opened") is not True:
        raise ValueError("V2 stress result must record stress_opened=true")
    if tuple(payload.get("remaining_sealed", ())) != SEALED_PARTITIONS:
        raise ValueError("V2 stress result sealed boundary is invalid")
    if dict(payload.get("stress_scope", {})) != {
        "start": "2025-07-01",
        "end": "2025-12-31",
    }:
        raise ValueError("V2 stress result stress scope is invalid")
    model = payload.get("model", {})
    if model.get("input_language") != INPUT_LANGUAGE:
        raise ValueError("V2 stress result language is invalid")
    if model.get("critical_class") != CRITICAL_CLASS:
        raise ValueError("V2 stress result critical class is invalid")
    primary = payload.get("primary")
    operational = payload.get("operational_secondary")
    if not isinstance(primary, Mapping) or not isinstance(operational, Mapping):
        raise ValueError("V2 stress result aggregate views are missing")
    if primary.get("view") != "scientific":
        raise ValueError("V2 stress primary view label is invalid")
    if operational.get("view") != "operational":
        raise ValueError("V2 stress operational view label is invalid")
    for view_name, view in (
        ("primary", primary),
        ("operational_secondary", operational),
    ):
        arms = view.get("arms", {})
        if set(arms) != {"v2_combined", "s7_fallback_alone"}:
            raise ValueError(f"V2 stress {view_name} arms are incomplete")
        for arm_name, arm in arms.items():
            metrics = arm.get("metrics", {})
            if set(metrics.get("per_class", {})) != set(MODELED_FAMILIES):
                raise ValueError(
                    f"V2 stress {view_name}.{arm_name} per-class support is "
                    "incomplete"
                )
    gates = payload.get("gates", {})
    if gates.get("required_gate_count") != 4:
        raise ValueError("V2 stress gate count contract is invalid")
    if gates.get("passed_count") != sum(
        1 for item in gates.get("results", []) if item.get("passed")
    ):
        raise ValueError("V2 stress gate passed count is inconsistent")
    if payload.get("confirmed") is not bool(gates.get("passed")):
        raise ValueError("V2 stress confirmed flag is inconsistent with gates")
    expected_status = "CONFIRMED" if payload["confirmed"] else "NOT_CONFIRMED"
    if payload["status"] != expected_status:
        raise ValueError("V2 stress status is inconsistent with gates")
    strict_results = [
        item
        for item in gates.get("results", [])
        if item.get("name") == "paired_critical_f1_gain"
    ]
    if len(strict_results) != 1 or strict_results[0].get("strict") is not True:
        raise ValueError("V2 stress paired gate must be strict")
    bootstrap = payload.get("bootstrap", {})
    if bootstrap.get("diagnostic_only") is not True:
        raise ValueError("V2 stress bootstrap must be diagnostic only")
    _check_result_privacy(payload)


def validate_v2_stress_manifest(
    manifest: Mapping[str, Any],
    result_path: str | Path,
    config_path: str | Path,
) -> None:
    """Validate the complete portable V2.1-C manifest and aggregate result."""

    result_file = Path(result_path).expanduser().resolve()
    config_file = Path(config_path).expanduser().resolve()
    config = load_v2_stress_config(config_file)
    root = config_file.parent.parent.resolve()
    if manifest.get("schema_version") != V2_STRESS_MANIFEST_SCHEMA:
        raise ValueError("Unexpected V2 stress manifest schema")
    if manifest.get("stage") != "V2.1-C":
        raise ValueError("V2 stress manifest stage is invalid")
    if manifest.get("status") not in {"CONFIRMED", "NOT_CONFIRMED"}:
        raise ValueError("V2 stress manifest status is invalid")
    if manifest.get("confirmatory") is not True:
        raise ValueError("V2 stress manifest confirmatory flag is invalid")
    if manifest.get("deploy") is not False:
        raise ValueError("V2 stress manifest cannot authorize deployment")
    if manifest.get("stress_opened") is not True:
        raise ValueError("V2 stress manifest must record stress_opened=true")
    if tuple(manifest.get("remaining_sealed", ())) != SEALED_PARTITIONS:
        raise ValueError("V2 stress manifest sealed boundary is invalid")
    protocol = manifest.get("protocol", {})
    result_meta = manifest.get("result", {})
    if _require_relative_path(protocol.get("path"), "protocol") != _relative(
        config_file, root
    ):
        raise ValueError("V2 stress manifest protocol path is invalid")
    if _require_relative_path(result_meta.get("path"), "result") != _relative(
        result_file, root
    ):
        raise ValueError("V2 stress manifest result path is invalid")
    protocol_actual = _file_signature(config_file)
    if protocol.get("sha256") != protocol_actual["sha256"]:
        raise ValueError("V2 stress manifest protocol hash is invalid")
    if protocol.get("size_bytes") != protocol_actual["size_bytes"]:
        raise ValueError("V2 stress manifest protocol size is invalid")
    result_actual = _file_signature(result_file)
    if result_meta.get("sha256") != result_actual["sha256"]:
        raise ValueError("V2 stress manifest result hash is invalid")
    if result_meta.get("size_bytes") != result_actual["size_bytes"]:
        raise ValueError("V2 stress manifest result size is invalid")
    if manifest.get("v2_freeze") != config.v2_freeze:
        raise ValueError("V2 stress manifest V2 freeze differs from protocol")
    if manifest.get("s7_freeze") != config.s7_freeze:
        raise ValueError("V2 stress manifest S7 freeze differs from protocol")
    expected_external = {
        "raw_source": (
            config.payload["paths"]["source_path"],
            config.source["raw_sha256"],
            config.source["raw_size_bytes"],
        ),
        "index": (
            config.payload["paths"]["index_path"],
            config.source["index_sha256"],
            config.source["index_size_bytes"],
        ),
        "s3_protocol": (
            config.payload["paths"]["s3_protocol"],
            config.source["s3_protocol_sha256"],
            config.source["s3_protocol_size_bytes"],
        ),
    }
    for key, (path, digest, size) in expected_external.items():
        item = manifest.get(key, {})
        if _require_relative_path(item.get("path"), key) != path:
            raise ValueError(f"V2 stress manifest {key} path is invalid")
        if item.get("sha256") != digest or item.get("size_bytes") != size:
            raise ValueError(f"V2 stress manifest {key} metadata is invalid")
    payload = _read_json(result_file)
    signature = _signature(
        config_file, config.source, config.v2_freeze, config.s7_freeze
    )
    _validate_complete_result(payload, signature)
    if payload["model"]["threshold"] != config.payload["model"]["threshold"]:
        raise ValueError("V2 stress manifest model threshold differs from protocol")
    if payload["gates"]["required_gate_count"] != int(
        config.gates["required_gate_count"]
    ):
        raise ValueError("V2 stress manifest gate contract differs from protocol")
    if manifest.get("status") != payload.get("status"):
        raise ValueError("V2 stress manifest status differs from result")
    if manifest.get("confirmed") != payload.get("confirmed"):
        raise ValueError("V2 stress manifest decision differs from result")
    if manifest.get("opened_at") != payload.get("opened_at"):
        raise ValueError("V2 stress manifest opened_at differs from result")
    if manifest.get("execution_attempts") != payload.get("execution_attempts"):
        raise ValueError(
            "V2 stress manifest execution attempts differ from result"
        )


def _cached_result(
    result_path: Path,
    manifest_path: Path,
    signature: str,
    *,
    config_path: Path,
) -> dict[str, Any] | None:
    """Return or repair a complete cache hit without dataset access."""

    if not result_path.exists():
        return None
    payload = _read_json(result_path)
    if payload.get("complete") is not True or payload.get("signature") != signature:
        return None
    config = load_v2_stress_config(config_path)
    _validate_complete_result(payload, signature)
    try:
        manifest = _read_json(manifest_path)
        validate_v2_stress_manifest(manifest, result_path, config_path)
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        _publish_manifest(result_path, manifest_path, config_path, config, payload)
    return payload


def _resume_attempts(result_path: Path, signature: str) -> tuple[int, str | None]:
    """Validate an incomplete marker and return its next attempt number."""

    if not result_path.exists():
        return 1, None
    payload = _read_json(result_path)
    if payload.get("complete") is True:
        if payload.get("signature") != signature:
            raise ValueError("Complete V2 stress result has a stale signature")
        return int(payload.get("execution_attempts", 1)), None
    if payload.get("signature") != signature:
        raise ValueError(
            "Incomplete V2 stress result cannot resume under new hashes"
        )
    if payload.get("code_schema") != V2_STRESS_CODE_SCHEMA:
        raise ValueError("Incomplete V2 stress result cannot resume under new code")
    if payload.get("primary") is not None:
        raise ValueError("V2 stress partial primary metrics must not be persisted")
    if payload.get("operational_secondary") is not None:
        raise ValueError(
            "V2 stress partial operational metrics must not be persisted"
        )
    opened_at = payload.get("opened_at")
    if not isinstance(opened_at, str) or not opened_at.endswith("Z"):
        raise ValueError("Incomplete V2 stress result has no valid opened_at")
    return int(payload.get("execution_attempts", 1)) + 1, opened_at


def _base_result(
    signature: str,
    config: V2StressConfig,
    evidence: Mapping[str, Any],
    execution_attempts: int = 1,
    opened_at: str | None = None,
) -> dict[str, Any]:
    """Create the aggregate-only V2.1-C result envelope."""

    model = config.payload["model"]
    return {
        "schema_version": V2_STRESS_RESULT_SCHEMA,
        "code_schema": V2_STRESS_CODE_SCHEMA,
        "stage": "V2.1-C",
        "adr": "ADR-014",
        "signature": signature,
        "complete": False,
        "status": "RUNNING",
        "confirmed": None,
        "deploy": False,
        "confirmatory": True,
        "stress_opened": True,
        "opened_at": opened_at or time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
        "execution_attempts": execution_attempts,
        "stress_scope": dict(config.stress_scope),
        "remaining_sealed": list(SEALED_PARTITIONS),
        "model": {
            "model_version": str(model["model_version"]),
            "fallback_model_version": str(model["fallback_model_version"]),
            "threshold": float(model["threshold"]),
            "critical_class": str(model["critical_class"]),
            "input_language": str(model["input_language"]),
            "combination": str(model["combination"]),
        },
        "s2_evidence": evidence["s2"],
        "provenance": {
            "v2_freeze": evidence["v2"],
            "s7_freeze": evidence["s7"],
        },
        "scope_counts": None,
        "primary": None,
        "operational_secondary": None,
        "gates": None,
        "bootstrap": None,
        "expectation": None,
    }


def _error_result(result: Mapping[str, Any], error: Exception) -> dict[str, Any]:
    """Create an incomplete error marker without persisting partial metrics."""

    failed = dict(result)
    failed["status"] = "ERROR"
    failed["complete"] = False
    failed["primary"] = None
    failed["operational_secondary"] = None
    failed["scope_counts"] = None
    failed["gates"] = None
    failed["bootstrap"] = None
    failed["expectation"] = None
    failed["confirmed"] = None
    failed["error"] = {
        "type": type(error).__name__,
        "stage": "confirmatory_execution",
    }
    return failed


def _publish_manifest(
    result_path: Path,
    manifest_path: Path,
    config_path: Path,
    config: V2StressConfig,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish the post-full V2.1-C manifest with all frozen digests."""

    root = config_path.parent.parent
    protocol_signature = _file_signature(config_path)
    protocol_signature["path"] = _relative(config_path, root)
    manifest = {
        "schema_version": V2_STRESS_MANIFEST_SCHEMA,
        "stage": "V2.1-C",
        "status": result["status"],
        "confirmatory": True,
        "confirmed": result["confirmed"],
        "deploy": False,
        "stress_opened": True,
        "opened_at": result["opened_at"],
        "execution_attempts": result["execution_attempts"],
        "remaining_sealed": list(SEALED_PARTITIONS),
        "protocol": protocol_signature,
        "result": {
            **_file_signature(result_path),
            "path": _relative(result_path, root),
        },
        "v2_freeze": dict(config.v2_freeze),
        "s7_freeze": dict(config.s7_freeze),
        "raw_source": {
            "path": config.payload["paths"]["source_path"],
            "sha256": config.source["raw_sha256"],
            "size_bytes": config.source["raw_size_bytes"],
        },
        "index": {
            "path": config.payload["paths"]["index_path"],
            "sha256": config.source["index_sha256"],
            "size_bytes": config.source["index_size_bytes"],
        },
        "s3_protocol": {
            "path": config.payload["paths"]["s3_protocol"],
            "sha256": config.source["s3_protocol_sha256"],
            "size_bytes": config.source["s3_protocol_size_bytes"],
        },
        "gates": result["gates"],
        "scientific_view": "primary_only",
        "operational_view": "secondary_diagnostic_only",
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest


def _run_full(
    project_root: Path,
    config_path: Path,
    result_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Run the one-time stress evaluation after all guards pass."""

    config = load_v2_stress_config(config_path)
    signature = _signature(
        config_path, config.source, config.v2_freeze, config.s7_freeze
    )
    cached = _cached_result(
        result_path, manifest_path, signature, config_path=config_path
    )
    if cached is not None:
        return cached
    execution_attempts, opened_at = _resume_attempts(result_path, signature)
    metadata = validate_frozen_metadata(config, project_root, include_raw=False)
    require_confirmatory_unlock(config)
    result = _base_result(
        signature,
        config,
        metadata,
        execution_attempts=execution_attempts,
        opened_at=opened_at,
    )
    try:
        _write_json_atomic(result_path, result)
        metadata = validate_frozen_metadata(config, project_root, include_raw=True)
        source_path = project_root / config.payload["paths"]["source_path"]
        index_path = project_root / config.payload["paths"]["index_path"]
        v2_predictor = load_v2_predictor(project_root)
        s7_bundle = project_root / config.payload["paths"]["s7_bundle"]
        s7_manifest = project_root / config.payload["paths"]["s7_manifest"]
        s7_result = project_root / config.payload["paths"]["s7_result"]
        s7_predictor = load_s7_predictor(s7_bundle, s7_manifest, s7_result)
        threshold = float(config.payload["model"]["threshold"])
        spill_root = (
            project_root / config.payload["access"]["temp_directory"]
        ).resolve()
        spill_root.mkdir(parents=True, exist_ok=True)
        spill_directory = spill_root / f"run-{os.getpid()}-{execution_attempts}"
        spill_directory.mkdir(parents=True, exist_ok=False)
        connection = duckdb.connect()
        try:
            spill_literal = str(spill_directory).replace("'", "''")
            connection.execute(f"SET temp_directory = '{spill_literal}'")
            memory_limit = str(config.payload["access"]["memory_limit"])
            connection.execute(f"SET memory_limit = '{memory_limit}'")
            connection.execute("SET threads = 1")
            connection.execute("SET preserve_insertion_order = false")
            counts = _scope_counts(connection, index_path, config)
            (
                primary_joint,
                primary_override,
                primary_effective,
                primary_rows,
            ) = _score_scope_two_arms(
                connection,
                source_path,
                "v2c_primary",
                v2_predictor,
                s7_predictor,
                threshold,
                config,
            )
            (
                operational_joint,
                operational_override,
                operational_effective,
                operational_rows,
            ) = _score_scope_two_arms(
                connection,
                source_path,
                "v2c_operational",
                v2_predictor,
                s7_predictor,
                threshold,
                config,
            )
        finally:
            connection.close()
            if spill_directory.parent.resolve() == spill_root:
                shutil.rmtree(spill_directory, ignore_errors=True)
        _validate_scope_row_count(
            "v2c_primary", primary_rows, counts["primary_representatives"]
        )
        _validate_scope_row_count(
            "v2c_operational", operational_rows, counts["operational_lines"]
        )
        support = primary_joint.sum(axis=(1, 2))
        if np.any(support <= 0):
            raise ValueError(
                "V2 stress primary scientific view does not support all "
                "nine classes"
            )
        result["scope_counts"] = {
            **counts,
            "primary_scored_rows": primary_rows,
            "operational_scored_rows": operational_rows,
            "s2_difference": {
                "expected": {
                    "test_all_text": config.payload["s2_evidence"][
                        "stress_all_text"
                    ],
                    "novel_text": config.payload["s2_evidence"][
                        "stress_novel_text"
                    ],
                    "novel_unique_groups": config.payload["s2_evidence"][
                        "stress_novel_unique_groups"
                    ],
                    "critical_novel_groups": config.payload["s2_evidence"][
                        "critical_novel_groups"
                    ],
                },
                "observed": {
                    "test_all_text": counts["test_all_text"],
                    "novel_text": counts["novel_test_lines"],
                    "novel_unique_groups": counts["novel_unique_groups"],
                    "critical_novel_groups": counts["critical_novel_groups"],
                },
            },
        }
        observed = result["scope_counts"]["s2_difference"]["observed"]
        expected = result["scope_counts"]["s2_difference"]["expected"]
        result["scope_counts"]["s2_difference"]["delta"] = {
            key: int(observed[key]) - int(expected[key]) for key in expected
        }
        result["primary"] = _view_block(
            "scientific",
            primary_joint,
            primary_rows,
            primary_override,
            primary_effective,
        )
        result["operational_secondary"] = _view_block(
            "operational",
            operational_joint,
            operational_rows,
            operational_override,
            operational_effective,
        )
        result["gates"] = evaluate_gates(
            result["primary"]["arms"]["v2_combined"]["metrics"],
            result["primary"]["paired"]["critical_f1_gain"],
            config.gates,
        )
        bootstrap_config = config.payload["bootstrap"]
        replicates = int(bootstrap_config["replicates"])
        seed = int(bootstrap_config["seed"])
        confidence_level = float(bootstrap_config["confidence_level"])
        primary_v2_confusion = primary_joint.sum(axis=2)
        primary_s7_confusion = primary_joint.sum(axis=1)
        v2_intervals = bootstrap_confidence_intervals(
            primary_v2_confusion,
            replicates=replicates,
            seed=seed,
            confidence_level=confidence_level,
        )["intervals"]
        s7_intervals = bootstrap_confidence_intervals(
            primary_s7_confusion,
            replicates=replicates,
            seed=seed,
            confidence_level=confidence_level,
        )["intervals"]
        paired_interval = paired_bootstrap_interval(
            primary_joint,
            replicates=replicates,
            seed=seed,
            confidence_level=confidence_level,
        )
        result["bootstrap"] = {
            "replicates": replicates,
            "seed": seed,
            "confidence_level": confidence_level,
            "diagnostic_only": True,
            "v2_combined": {
                metric: [item["lower"], item["upper"]]
                for metric, item in v2_intervals.items()
            },
            "s7_fallback_alone": {
                metric: [item["lower"], item["upper"]]
                for metric, item in s7_intervals.items()
            },
            "paired_critical_f1_gain": [
                paired_interval["lower"], paired_interval["upper"]
            ],
        }
        expectation = dict(config.payload["expectation"])
        observed_gain = float(result["primary"]["paired"]["critical_f1_gain"])
        development_gain = float(expectation["development_paired_gain"])
        expectation["observed_paired_gain"] = observed_gain
        expectation["agrees_in_sign"] = (
            (observed_gain > 0) == (development_gain > 0)
        )
        result["expectation"] = expectation
        result["confirmed"] = bool(result["gates"]["passed"])
        result["status"] = "CONFIRMED" if result["confirmed"] else "NOT_CONFIRMED"
        result["complete"] = True
        _check_result_privacy(result)
    except Exception as error:
        spill_candidate = locals().get("spill_directory")
        if isinstance(spill_candidate, Path) and spill_candidate.exists():
            if spill_candidate.parent.resolve() == (
                project_root / config.payload["access"]["temp_directory"]
            ).resolve():
                shutil.rmtree(spill_candidate, ignore_errors=True)
        failed = _error_result(result, error)
        _check_result_privacy(failed)
        _write_json_atomic(result_path, failed)
        raise
    _write_json_atomic(result_path, result)
    _publish_manifest(result_path, manifest_path, config_path, config, result)
    return result


def run_v2_stress(
    *,
    project_root: str | Path,
    config_path: str | Path = DEFAULT_CONFIG,
    result_path: str | Path = DEFAULT_RESULT,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    run_mode: str = "disabled",
) -> dict[str, Any]:
    """Run V2.1-C in disabled, diagnostic smoke, or guarded full mode.

    ``full`` is the only mode that can access the real index or raw source.
    """

    if run_mode == "disabled":
        return {"status": "DISABLED", "confirmatory": True, "deploy": False}
    if run_mode == "smoke":
        return run_v2_stress_smoke(project_root)
    if run_mode != "full":
        raise ValueError("V2 stress run_mode must be disabled, smoke, or full")
    return _run_full(
        Path(project_root).expanduser().resolve(),
        Path(config_path).expanduser().resolve(),
        Path(result_path).expanduser().resolve(),
        Path(manifest_path).expanduser().resolve(),
    )


def run_v2_stress_smoke(
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate config, hashes, upstream manifests, and predictors only.

    This never opens the stress partition and never requires the unlock
    token: it verifies that the frozen V2 and S7 packages are internally
    consistent and loadable, using only artifacts already unsealed for
    confirmation (never ``stress`` or ``monitor``).

    Args:
        project_root: Project root holding the frozen artifacts. Defaults
            to the installed package location, never the working directory.

    Returns:
        A diagnostic-only preflight result.

    Raises:
        ValueError: If the config, a hash, or a manifest is invalid.
    """

    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else _default_project_root()
    )
    config = load_v2_stress_config(root / DEFAULT_CONFIG)
    metadata = validate_frozen_metadata(config, root, include_raw=False)
    v2_manifest_path = root / config.payload["paths"]["v2_manifest"]
    v2_result_path = root / config.payload["paths"]["v2_result"]
    validate_v2_manifest(v2_manifest_path, v2_result_path)
    s7_manifest_path = root / config.payload["paths"]["s7_manifest"]
    s7_manifest = _read_json(s7_manifest_path)
    s7_config_path = root / config.payload["paths"]["s7_config"]
    s7_bundle_path = root / config.payload["paths"]["s7_bundle"]
    s7_result_path = root / config.payload["paths"]["s7_result"]
    validate_s7_manifest(
        s7_manifest, s7_bundle_path, s7_result_path, s7_config_path
    )
    v2_predictor = load_v2_predictor(root)
    s7_predictor = load_s7_predictor(s7_bundle_path, s7_manifest_path, s7_result_path)
    model = config.payload["model"]
    return {
        "schema_version": V2_STRESS_RESULT_SCHEMA,
        "code_schema": V2_STRESS_CODE_SCHEMA,
        "status": "DIAGNOSTIC_ONLY",
        "complete": True,
        "confirmatory": True,
        "confirmed": None,
        "deploy": False,
        "stress_opened": False,
        "checks": {
            "config_validated": True,
            "v2_freeze_verified": bool(metadata["v2"]),
            "s7_freeze_verified": bool(metadata["s7"]),
            "s2_candidate_status": metadata["s2"]["candidate_status"],
            "v2_manifest_valid": True,
            "s7_manifest_valid": True,
            "v2_predictor_loaded": (
                v2_predictor.model_version == model["model_version"]
            ),
            "s7_predictor_loaded": (
                s7_predictor.model_version == model["fallback_model_version"]
            ),
        },
    }
