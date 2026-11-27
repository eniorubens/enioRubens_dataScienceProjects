"""Strict, execution-blocking contract for the proposed V3 experiment."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_CONFIG = "config/v3/development_protocol.json"
DEFAULT_MANIFEST = "config/v3/development_protocol_manifest.json"
CRITICAL_CLASS = "debt_credit_management"
ALLOWED_PARTITIONS = ("train", "validation")
FORBIDDEN_PARTITIONS = ("test", "stress", "monitor")
FOLD_IDS = ("F1", "F2", "F3")
NEGATIVE_RATIOS = (5, 10, 15)
_FROZEN_STATUS = "FROZEN_FOR_V3_DEVELOPMENT"
_DRAFT_STATUS = "PROPOSED_FOR_USER_REVIEW"
_FROZEN_SCHEMA = "v3-development-protocol-v1"
_DRAFT_SCHEMA = "v3-development-protocol-v1-draft"
_SHA_PATTERN = re.compile(r"[0-9a-fA-F]{40}")

_TOP_LEVEL_KEYS = {
    "schema_version",
    "cycle_id",
    "stage",
    "approval",
    "objective",
    "data_boundary",
    "rolling_origin",
    "screening",
    "sampling",
    "calibration",
    "metrics",
    "selection",
    "long_text_challenge",
    "execution",
    "stages",
}

_EXPECTED_WINDOWS = {
    "F1": {
        "fit": ("train", "2023-08-01", "2023-12-31", 140310, 305),
        "calibration": ("train", "2024-01-01", "2024-02-29", 60936, 207),
        "evaluation": ("train", "2024-03-01", "2024-04-30", 70573, 219),
    },
    "F2": {
        "fit": ("train", "2023-08-01", "2024-02-29", 201246, 512),
        "calibration": ("train", "2024-03-01", "2024-04-30", 70573, 219),
        "evaluation": ("train", "2024-05-01", "2024-06-30", 73733, 215),
    },
    "F3": {
        "fit": ("train", "2023-08-01", "2024-06-30", 345552, 946),
        "calibration": (
            "validation",
            "2024-07-01",
            "2024-09-30",
            118274,
            347,
        ),
        "evaluation": (
            "validation",
            "2024-10-01",
            "2024-12-31",
            127706,
            545,
        ),
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    """Read one UTF-8 JSON object."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read V3 protocol: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("V3 protocol must be a JSON object")
    return payload


def _exact_keys(value: Any, expected: set[str], name: str) -> Mapping[str, Any]:
    """Require a mapping with exactly the expected keys."""

    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"V3 {name} keys differ from the protocol")
    return value


def _parse_scope(scope: Any, name: str) -> dict[str, Any]:
    """Validate one dated development scope and its aggregate supports."""

    values = _exact_keys(
        scope,
        {"partition", "start", "end", "rows", "critical_rows"},
        name,
    )
    partition = str(values["partition"])
    require_development_partition(partition)
    start = date.fromisoformat(str(values["start"]))
    end = date.fromisoformat(str(values["end"]))
    rows = int(values["rows"])
    critical_rows = int(values["critical_rows"])
    if start > end:
        raise ValueError(f"V3 {name} starts after it ends")
    if rows <= 0 or critical_rows <= 0 or critical_rows >= rows:
        raise ValueError(f"V3 {name} supports are invalid")
    return {
        "partition": partition,
        "start": start,
        "end": end,
        "rows": rows,
        "critical_rows": critical_rows,
    }


def _validate_folds(rolling: Any) -> None:
    """Enforce the three exact chronological expanding folds."""

    block = _exact_keys(
        rolling,
        {
            "folds",
            "fit_policy",
            "calibration_and_evaluation_prevalence",
            "later_folds_may_reuse_earlier_evaluation_as_fit",
        },
        "rolling_origin",
    )
    if block["fit_policy"] != "expanding_origin":
        raise ValueError("V3 rolling origin must use expanding fits")
    if block["calibration_and_evaluation_prevalence"] != "natural_untouched":
        raise ValueError("V3 later windows must retain natural prevalence")
    if block["later_folds_may_reuse_earlier_evaluation_as_fit"] is not True:
        raise ValueError("V3 rolling reuse policy differs from the contract")
    folds = block["folds"]
    if not isinstance(folds, Sequence) or isinstance(folds, (str, bytes)):
        raise ValueError("V3 folds must be a sequence")
    if tuple(item.get("fold_id") for item in folds) != FOLD_IDS:
        raise ValueError("V3 fold identifiers or ordering are invalid")

    previous_fit_end: date | None = None
    for item in folds:
        fold = _exact_keys(
            item,
            {"fold_id", "fit", "calibration", "evaluation"},
            "fold",
        )
        fold_id = str(fold["fold_id"])
        parsed = {
            role: _parse_scope(fold[role], f"{fold_id}.{role}")
            for role in ("fit", "calibration", "evaluation")
        }
        if parsed["fit"]["end"] >= parsed["calibration"]["start"]:
            raise ValueError(f"V3 {fold_id} fit overlaps calibration")
        if parsed["calibration"]["end"] >= parsed["evaluation"]["start"]:
            raise ValueError(f"V3 {fold_id} calibration overlaps evaluation")
        if parsed["fit"]["start"] != date(2023, 8, 1):
            raise ValueError("V3 expanding folds must share the initial origin")
        if previous_fit_end is not None and parsed["fit"]["end"] <= previous_fit_end:
            raise ValueError("V3 fit windows do not expand")
        previous_fit_end = parsed["fit"]["end"]
        for role, expected in _EXPECTED_WINDOWS[fold_id].items():
            observed = parsed[role]
            actual = (
                observed["partition"],
                observed["start"].isoformat(),
                observed["end"].isoformat(),
                observed["rows"],
                observed["critical_rows"],
            )
            if actual != expected:
                raise ValueError(f"V3 {fold_id}.{role} differs from audited scope")


def _validate_boundary(boundary: Any) -> None:
    """Reject any weakening of the development-only data boundary."""

    block = _exact_keys(
        boundary,
        {
            "scientific_cache",
            "allowed_partitions",
            "forbidden_partitions",
            "partition_status",
            "group_identity",
            "one_scientific_representative_per_group",
            "individual_text_or_prediction_persistence",
        },
        "data_boundary",
    )
    if tuple(block["allowed_partitions"]) != ALLOWED_PARTITIONS:
        raise ValueError("V3 allowed partitions differ from the boundary")
    if tuple(block["forbidden_partitions"]) != FORBIDDEN_PARTITIONS:
        raise ValueError("V3 forbidden partitions differ from the boundary")
    statuses = _exact_keys(
        block["partition_status"], set(FORBIDDEN_PARTITIONS), "partition_status"
    )
    if "CONSUMED" not in str(statuses["test"]):
        raise ValueError("V3 test must remain consumed")
    if "CONSUMED" not in str(statuses["stress"]):
        raise ValueError("V3 stress must remain consumed")
    if "SEALED" not in str(statuses["monitor"]):
        raise ValueError("V3 monitor must remain sealed")
    if tuple(block["group_identity"]) != (
        "normalized_group_hash",
        "normalized_length",
    ):
        raise ValueError("V3 group identity differs from the audited identity")
    if block["one_scientific_representative_per_group"] is not True:
        raise ValueError("V3 requires one scientific representative per group")
    if block["individual_text_or_prediction_persistence"] is not False:
        raise ValueError("V3 may not persist individual text or predictions")


def _validate_representations(screening: Any) -> None:
    """Require the initial TF-IDF control and frozen MiniLM challenger."""

    block = _exact_keys(
        screening,
        {"task", "representations", "deferred_challengers", "classifier"},
        "screening",
    )
    if block["task"] != "binary_critical_detector":
        raise ValueError("V3 screening task must remain binary")
    representations = block["representations"]
    if not isinstance(representations, list) or len(representations) != 2:
        raise ValueError("V3 must define exactly two initial representations")
    if [item.get("candidate_id") for item in representations] != [
        "tfidf_word_char_control",
        "minilm_frozen_primary",
    ]:
        raise ValueError("V3 initial representation catalog differs")
    minilm = representations[1]
    if minilm.get("model_id") != "sentence-transformers/all-MiniLM-L6-v2":
        raise ValueError("V3 primary encoder must be all-MiniLM-L6-v2")
    if minilm.get("encoder_trainable") is not False:
        raise ValueError("V3 E3 MiniLM encoder must remain frozen")
    revision = minilm.get("revision")
    status = str(minilm.get("revision_status"))
    if revision is None:
        if status != "MUST_RESOLVE_BEFORE_PROTOCOL_FREEZE":
            raise ValueError("V3 unresolved MiniLM revision status is invalid")
    elif not isinstance(revision, str) or _SHA_PATTERN.fullmatch(revision) is None:
        raise ValueError("V3 MiniLM revision must be a 40-character commit hash")

    classifier = block["classifier"]
    expected_classifier = {
        "class": "LinearSVC",
        "C": 1.0,
        "tol": 0.0001,
        "max_iter": 5000,
        "dual": "auto",
        "class_weight": None,
        "random_state": 42,
    }
    if classifier != expected_classifier:
        raise ValueError("V3 linear classifier differs from the contract")


def _validate_sampling(sampling: Any) -> None:
    """Enforce fit-only, unique-positive, controlled negative sampling."""

    block = _exact_keys(
        sampling,
        {
            "scope",
            "positive_policy",
            "negative_ratios_per_positive",
            "negative_composition",
            "hard_negatives",
            "background_negatives",
            "same_sampled_group_ids_across_representations",
            "oversampling_before_temporal_split",
            "resampling_calibration_or_evaluation",
        },
        "sampling",
    )
    if block["scope"] != "fit_window_only":
        raise ValueError("V3 resampling must be restricted to each fit window")
    if block["positive_policy"] != "all_unique_critical_groups_without_replacement":
        raise ValueError("V3 positive policy differs from the contract")
    if tuple(block["negative_ratios_per_positive"]) != NEGATIVE_RATIOS:
        raise ValueError("V3 negative ratios must be 1:5, 1:10, and 1:15")
    composition = block["negative_composition"]
    expected_counts = {"5": (3, 2), "10": (7, 3), "15": (10, 5)}
    if not isinstance(composition, Mapping) or set(composition) != set(expected_counts):
        raise ValueError("V3 negative composition keys are invalid")
    for ratio, (hard, background) in expected_counts.items():
        item = composition[ratio]
        observed = (
            int(item.get("hard", -1)),
            int(item.get("background", -1)),
        )
        if observed != (hard, background):
            raise ValueError(f"V3 negative composition differs for ratio {ratio}")
        if hard + background != int(ratio):
            raise ValueError(f"V3 negative composition does not sum to {ratio}")
    if block["same_sampled_group_ids_across_representations"] is not True:
        raise ValueError("V3 representations must use identical sampled groups")
    if block["oversampling_before_temporal_split"] is not False:
        raise ValueError("V3 forbids oversampling before temporal splitting")
    if block["resampling_calibration_or_evaluation"] is not False:
        raise ValueError("V3 forbids resampling later windows")


@dataclass(frozen=True, slots=True)
class V3Protocol:
    """Validated V3 development contract, still blocked while proposed."""

    payload: Mapping[str, Any]

    @property
    def approval_status(self) -> str:
        """Return the current protocol approval state."""

        return str(self.payload["approval"]["status"])

    @property
    def is_frozen(self) -> bool:
        """Return whether scientific execution has been explicitly authorized."""

        return self.approval_status == _FROZEN_STATUS

    @property
    def protocol_sha256(self) -> str:
        """Return a deterministic digest of the normalized protocol payload."""

        normalized = json.dumps(
            self.payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(normalized).hexdigest().upper()

    @property
    def folds(self) -> tuple[Mapping[str, Any], ...]:
        """Return detached rolling-fold mappings."""

        return tuple(copy.deepcopy(self.payload["rolling_origin"]["folds"]))

    def validate_for_scientific_execution(self) -> None:
        """Reject execution until approval and encoder revision are frozen."""

        if not self.is_frozen:
            raise ValueError("V3 protocol is not frozen for scientific execution")
        approval = self.payload["approval"]
        if not approval.get("approved_on") or approval.get("approved_by") != "user":
            raise ValueError("V3 frozen approval metadata is incomplete")
        minilm = self.payload["screening"]["representations"][1]
        if _SHA_PATTERN.fullmatch(str(minilm.get("revision"))) is None:
            raise ValueError("V3 MiniLM revision is not frozen")
        if minilm.get("revision_status") != "FROZEN_COMMIT_HASH":
            raise ValueError("V3 MiniLM revision status is not frozen")

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible copy."""

        return copy.deepcopy(dict(self.payload))


def validate_v3_protocol(payload: Mapping[str, Any]) -> V3Protocol:
    """Validate the complete proposed or frozen V3 protocol."""

    block = _exact_keys(payload, _TOP_LEVEL_KEYS, "top-level")
    if block["cycle_id"] != "consumer-complaint-intelligence-v3":
        raise ValueError("V3 cycle identifier is invalid")
    if block["stage"] != "V3-E0":
        raise ValueError("V3 stage identifier is invalid")
    approval = _exact_keys(
        block["approval"],
        {"status", "proposed_on", "approved_on", "approved_by"},
        "approval",
    )
    if approval["status"] not in {_DRAFT_STATUS, _FROZEN_STATUS}:
        raise ValueError("V3 approval status is invalid")
    expected_schema = (
        _FROZEN_SCHEMA if approval["status"] == _FROZEN_STATUS else _DRAFT_SCHEMA
    )
    if block["schema_version"] != expected_schema:
        raise ValueError("V3 schema version differs from its approval status")
    if approval["status"] == _DRAFT_STATUS and (
        approval["approved_on"] is not None or approval["approved_by"] is not None
    ):
        raise ValueError("V3 draft may not claim approval metadata")
    objective = block["objective"]
    if objective.get("critical_class") != CRITICAL_CLASS:
        raise ValueError("V3 critical class differs from the project target")
    if objective.get("deployment_authorized") is not False:
        raise ValueError("V3 development may not authorize deployment")
    if objective.get("confirmation_claim_authorized") is not False:
        raise ValueError("V3 development may not claim confirmation")

    _validate_boundary(block["data_boundary"])
    _validate_folds(block["rolling_origin"])
    _validate_representations(block["screening"])
    _validate_sampling(block["sampling"])

    calibration = block["calibration"]
    if float(calibration.get("critical_precision_floor", -1.0)) != 0.4:
        raise ValueError("V3 proposed precision floor must be exactly 0.40")
    selection = block["selection"]
    if int(selection.get("required_fold_count", 0)) != 3:
        raise ValueError("V3 selection must require all three folds")
    if selection.get("monitor_can_select_candidate") is not False:
        raise ValueError("V3 monitor partition may not select a candidate")
    historical = selection.get("historical_s7_use")
    if historical != {
        "F1": "FORBIDDEN_IN_SAMPLE_WITH_FROZEN_FALLBACK",
        "F2": "FORBIDDEN_IN_SAMPLE_WITH_FROZEN_FALLBACK",
        "F3": "ALLOWED_FOR_DEVELOPMENT_SYSTEM_COMPARISON",
    }:
        raise ValueError("V3 historical S7 fold boundary differs")
    execution = block["execution"]
    if execution.get("default_run_mode") != "disabled":
        raise ValueError("V3 default execution mode must remain disabled")
    if execution.get("no_sealed_partition_unlock_in_development_code") is not True:
        raise ValueError("V3 development boundary must prohibit sealed unlocks")
    cache_partitions = tuple(
        execution.get("embedding_cache_allowed_partitions", ())
    )
    if cache_partitions != ALLOWED_PARTITIONS:
        raise ValueError("V3 embedding cache boundary is invalid")

    protocol = V3Protocol(copy.deepcopy(dict(block)))
    if protocol.is_frozen:
        protocol.validate_for_scientific_execution()
    return protocol


def load_v3_protocol(path: str | Path = DEFAULT_CONFIG) -> V3Protocol:
    """Load and validate the V3 protocol without authorizing execution."""

    return validate_v3_protocol(_read_json(Path(path).expanduser().resolve()))


def _safe_project_path(root: Path, value: str) -> Path:
    """Resolve one project-relative protocol path without traversal."""

    candidate = Path(value)
    if candidate.is_absolute() or candidate.drive or ".." in candidate.parts:
        raise ValueError("V3 protocol paths must be project-relative")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("V3 protocol path escapes the project root")
    return resolved


def validate_protocol_manifest(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    """Validate the file and normalized hashes of the frozen V3-E0 contract."""

    root = Path(project_root).expanduser().resolve()
    path = Path(manifest_path)
    if not path.is_absolute():
        path = root / path
    manifest = _read_json(path)
    block = _exact_keys(
        manifest,
        {
            "schema_version",
            "cycle_id",
            "stage",
            "status",
            "approved_on",
            "approved_by",
            "protocol",
            "authorized_next_stages",
            "forbidden_partitions",
            "deployment_authorized",
            "confirmation_claim_authorized",
        },
        "manifest",
    )
    if block["schema_version"] != "v3-development-protocol-manifest-v1":
        raise ValueError("V3 protocol manifest schema is invalid")
    if block["cycle_id"] != "consumer-complaint-intelligence-v3":
        raise ValueError("V3 protocol manifest cycle is invalid")
    if block["stage"] != "V3-E0" or block["status"] != _FROZEN_STATUS:
        raise ValueError("V3 protocol manifest stage or status is invalid")
    if block["approved_by"] != "user" or not block["approved_on"]:
        raise ValueError("V3 protocol manifest approval is incomplete")
    if tuple(block["forbidden_partitions"]) != FORBIDDEN_PARTITIONS:
        raise ValueError("V3 protocol manifest partition boundary is invalid")
    if block["deployment_authorized"] is not False:
        raise ValueError("V3 protocol manifest may not authorize deployment")
    if block["confirmation_claim_authorized"] is not False:
        raise ValueError("V3 protocol manifest may not claim confirmation")
    if tuple(block["authorized_next_stages"]) != (
        "V3-E2_DEVELOPMENT_EMBEDDING_CACHE",
        "V3-E3_ROLLING_LINEAR_PROBES",
    ):
        raise ValueError("V3 protocol manifest next stages are invalid")

    metadata = _exact_keys(
        block["protocol"],
        {"path", "sha256", "normalized_sha256", "size_bytes"},
        "manifest.protocol",
    )
    protocol_path = _safe_project_path(root, str(metadata["path"]))
    raw = protocol_path.read_bytes()
    if len(raw) != int(metadata["size_bytes"]):
        raise ValueError("V3 protocol file size differs from the manifest")
    if hashlib.sha256(raw).hexdigest().upper() != str(metadata["sha256"]):
        raise ValueError("V3 protocol file hash differs from the manifest")
    protocol = load_v3_protocol(protocol_path)
    protocol.validate_for_scientific_execution()
    if protocol.protocol_sha256 != str(metadata["normalized_sha256"]):
        raise ValueError("V3 normalized protocol hash differs from the manifest")
    return copy.deepcopy(dict(block))


def require_development_partition(partition: str) -> str:
    """Accept only the two partitions authorized for V3 development."""

    if partition in FORBIDDEN_PARTITIONS:
        raise ValueError(f"Partition is unavailable to V3 development: {partition}")
    if partition not in ALLOWED_PARTITIONS:
        raise ValueError(f"Unsupported V3 development partition: {partition}")
    return partition


__all__ = [
    "ALLOWED_PARTITIONS",
    "CRITICAL_CLASS",
    "DEFAULT_CONFIG",
    "DEFAULT_MANIFEST",
    "FORBIDDEN_PARTITIONS",
    "NEGATIVE_RATIOS",
    "V3Protocol",
    "load_v3_protocol",
    "require_development_partition",
    "validate_protocol_manifest",
    "validate_v3_protocol",
]
