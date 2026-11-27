"""Draft V3 development protocol validation helpers."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


DEFAULT_CONFIG = "config/v3_development_protocol.json"
INPUT_LANGUAGE = "en-US"
ALLOWED_PARTITIONS = ("train", "validation")
CONSUMED_ARCHIVED_PARTITIONS = ("test", "stress")
MONITOR_PARTITION = "monitor"
NEVER_SELECTABLE_PARTITIONS = ("test", "stress", "monitor")
V3_SCHEMA_VERSION = "v3-development-protocol-draft-v1"
DESIGN_AUTHORIZED_E1_ACTION = "synthetic_model_loading_smoke_design_review"
REQUIRED_GPU = "NVIDIA T4"
FORBIDDEN_GPU_TOKENS = ("P100", "sm_60")

REQUIRED_CANDIDATES = (
    ("incumbent", "sklearn.LinearSVC"),
    ("primary_frozen_embedding", "sentence-transformers/all-MiniLM-L6-v2"),
    ("direct_frozen_embedding_competitor", "BAAI/bge-small-en-v1.5"),
    ("score_fusion", "v3_score_fusion"),
    ("quality_ceiling", "sentence-transformers/all-mpnet-base-v2"),
    ("optional_supervised_fine_tune", "winner_from_frozen_embedding_stage"),
)

STAGE_ORDER = ("E0", "E1", "E2", "E3", "E4", "E5", "E6", "P")


def _read_json(path: Path) -> dict[str, Any]:
    """Read one UTF-8 JSON object and reject non-object documents."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read V3 protocol: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("V3 protocol must be a JSON object")
    return payload


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    """Return a mapping value or fail with a clear protocol message."""

    if not isinstance(value, Mapping):
        raise ValueError(f"V3 protocol section must be an object: {name}")
    return value


def _require_exact_tuple(
    values: Any,
    expected: tuple[str, ...],
    name: str,
) -> tuple[str, ...]:
    """Validate a JSON string list against an exact ordered tuple."""

    if not isinstance(values, list) or not all(
        isinstance(value, str) for value in values
    ):
        raise ValueError(f"V3 protocol field must be a string list: {name}")
    actual = tuple(values)
    if actual != expected:
        raise ValueError(f"V3 protocol field differs from contract: {name}")
    return actual


def _require_contains(
    values: Any,
    required: tuple[str, ...],
    name: str,
) -> tuple[str, ...]:
    """Validate that a JSON string list contains all required values."""

    if not isinstance(values, list) or not all(
        isinstance(value, str) for value in values
    ):
        raise ValueError(f"V3 protocol field must be a string list: {name}")
    actual = tuple(values)
    missing = set(required) - set(actual)
    if missing:
        raise ValueError(
            f"V3 protocol field is missing required values: {name}"
        )
    return actual


def _validate_status(payload: Mapping[str, Any]) -> None:
    """Validate that the E0 protocol remains a draft, not a promotion."""

    status = _require_mapping(payload.get("protocol_status"), "protocol_status")
    if payload.get("schema_version") != V3_SCHEMA_VERSION:
        raise ValueError("V3 protocol schema version is invalid")
    if payload.get("stage") != "V3-E0":
        raise ValueError("V3 protocol must remain at stage V3-E0")
    if status.get("status") != "DRAFT_FOR_USER_REVIEW":
        raise ValueError("V3 protocol must be a draft for user review")
    blocked_flags = (
        "frozen",
        "approved_for_execution",
        "promotion_authorized",
        "deployment_authorized",
    )
    if any(bool(status.get(flag)) for flag in blocked_flags):
        raise ValueError("V3 E0 cannot freeze, execute, promote, or deploy")
    if status.get("scientific_cycle") != "V3":
        raise ValueError("V3 scientific cycle label is required")
    if status.get("review_required_before_freeze") is not True:
        raise ValueError("V3 protocol must require review before freezing")


def _validate_language(payload: Mapping[str, Any]) -> None:
    """Validate the explicit model-input language contract."""

    objective = _require_mapping(payload.get("objective"), "objective")
    if objective.get("input_language") != INPUT_LANGUAGE:
        raise ValueError("V3 input language must be en-US")


def _validate_data_boundary(payload: Mapping[str, Any]) -> None:
    """Validate development-only partition boundaries."""

    boundary = _require_mapping(payload.get("data_boundary"), "data_boundary")
    if "sealed_use" in boundary:
        raise ValueError("V3 must not use one blanket sealed_use rule")
    allowed = _require_exact_tuple(
        boundary.get("allowed_partitions"),
        ALLOWED_PARTITIONS,
        "data_boundary.allowed_partitions",
    )
    consumed = _require_exact_tuple(
        boundary.get("consumed_archived_partitions"),
        CONSUMED_ARCHIVED_PARTITIONS,
        "data_boundary.consumed_archived_partitions",
    )
    never_reusable = _require_exact_tuple(
        boundary.get("never_reusable_in_v3_partitions"),
        CONSUMED_ARCHIVED_PARTITIONS,
        "data_boundary.never_reusable_in_v3_partitions",
    )
    monitor = _require_mapping(
        boundary.get("monitor_partition"), "data_boundary.monitor_partition"
    )
    if monitor.get("partition") != MONITOR_PARTITION:
        raise ValueError("V3 monitor partition boundary is invalid")
    if monitor.get("status") != "SEALED_DIAGNOSTIC_ONLY":
        raise ValueError("V3 monitor partition must remain diagnostic-only")
    if monitor.get("selectable") is not False:
        raise ValueError("V3 monitor partition must never be selectable")
    never_selectable = _require_exact_tuple(
        boundary.get("never_selectable_partitions"),
        NEVER_SELECTABLE_PARTITIONS,
        "data_boundary.never_selectable_partitions",
    )
    selection = _require_exact_tuple(
        boundary.get("selection_partitions"),
        ALLOWED_PARTITIONS,
        "data_boundary.selection_partitions",
    )
    if set(allowed) & set(never_selectable):
        raise ValueError("V3 allowed and sealed partitions must be disjoint")
    if set(selection) - set(allowed):
        raise ValueError("V3 selection partitions must be development-only")
    if set(consumed) != set(never_reusable):
        raise ValueError("V3 consumed partitions must be archived permanently")


def _validate_authorization(payload: Mapping[str, Any]) -> None:
    """Validate that E0 authorizes design review, not execution."""

    auth = _require_mapping(payload.get("authorization"), "authorization")
    if "authorized_stage_ids" in auth or "authorized_actions" in auth:
        raise ValueError("V3 must separate design and execution authorization")
    _require_exact_tuple(
        auth.get("design_authorized_stage_ids"),
        ("E1",),
        "authorization.design_authorized_stage_ids",
    )
    _require_exact_tuple(
        auth.get("design_authorized_actions"),
        (DESIGN_AUTHORIZED_E1_ACTION,),
        "authorization.design_authorized_actions",
    )
    _require_exact_tuple(
        auth.get("execution_authorized_stage_ids"),
        (),
        "authorization.execution_authorized_stage_ids",
    )
    _require_exact_tuple(
        auth.get("execution_authorized_actions"),
        (),
        "authorization.execution_authorized_actions",
    )
    forbidden_execution = _require_contains(
        auth.get("forbidden_execution_actions"),
        (
            "synthetic_model_loading_smoke_execution",
            "kaggle_execution",
            "model_training",
            "threshold_selection",
            "candidate_promotion",
            "sealed_data_access",
            "project_dataset_loading",
        ),
        "authorization.forbidden_execution_actions",
    )
    forbidden = _require_contains(
        auth.get("forbidden_actions"),
        (
            "model_training",
            "threshold_selection",
            "candidate_promotion",
            "kaggle_execution_before_user_approval",
            "sealed_data_access",
            "project_dataset_loading",
        ),
        "authorization.forbidden_actions",
    )
    if DESIGN_AUTHORIZED_E1_ACTION in forbidden:
        raise ValueError("V3 design action cannot also be forbidden")
    if DESIGN_AUTHORIZED_E1_ACTION in forbidden_execution:
        raise ValueError("V3 design action cannot be a forbidden execution")
    if auth.get("e1_execution_requires_separate_user_approval") is not True:
        raise ValueError("V3 E1 execution must require separate approval")


def _validate_execution(payload: Mapping[str, Any]) -> None:
    """Validate the Kaggle T4-only accelerated execution rule."""

    execution = _require_mapping(payload.get("execution"), "execution")
    if execution.get("accelerated_platform") != "kaggle":
        raise ValueError("V3 accelerated platform must be Kaggle")
    if execution.get("required_gpu") != REQUIRED_GPU:
        raise ValueError("V3 accelerated execution requires NVIDIA T4")
    accepted = _require_contains(
        execution.get("accepted_gpu_names"),
        (REQUIRED_GPU,),
        "execution.accepted_gpu_names",
    )
    forbidden = _require_contains(
        execution.get("forbidden_gpu_tokens"),
        FORBIDDEN_GPU_TOKENS,
        "execution.forbidden_gpu_tokens",
    )
    if any(token in name for token in forbidden for name in accepted):
        raise ValueError("V3 accepted GPU names include a forbidden GPU")


def _candidate_key(candidate: Mapping[str, Any]) -> tuple[str, str]:
    """Return the identity key of a V3 model catalog entry."""

    return str(candidate.get("role")), str(candidate.get("model_id"))


def _validate_candidates(payload: Mapping[str, Any]) -> None:
    """Validate exact V3 candidate roles and model identifiers."""

    candidates = payload.get("candidate_catalog")
    if not isinstance(candidates, list):
        raise ValueError("V3 candidate catalog must be a list")
    keys: list[tuple[str, str]] = []
    for index, candidate in enumerate(candidates):
        candidate_map = _require_mapping(candidate, f"candidate_catalog[{index}]")
        keys.append(_candidate_key(candidate_map))
        if bool(candidate_map.get("training_allowed_in_e0")):
            raise ValueError("V3 E0 must not authorize candidate training")
    if tuple(keys) != REQUIRED_CANDIDATES:
        raise ValueError("V3 candidate catalog differs from the contract")


def _validate_long_text_strategy(payload: Mapping[str, Any]) -> None:
    """Validate that long-text handling is deferred to the E4 comparison."""

    strategy = _require_mapping(
        payload.get("long_text_strategy"), "long_text_strategy"
    )
    if strategy.get("required_stage") != "E4":
        raise ValueError("V3 long-text strategy must be evaluated in E4")
    if strategy.get("must_compare_before_promotion") is not True:
        raise ValueError("V3 long-text strategy must precede promotion")
    primary = _require_mapping(
        strategy.get("primary_model_wordpiece_limit"),
        "long_text_strategy.primary_model_wordpiece_limit",
    )
    competitor = _require_mapping(
        strategy.get("direct_competitor_token_limit"),
        "long_text_strategy.direct_competitor_token_limit",
    )
    if primary.get("model_id") != "sentence-transformers/all-MiniLM-L6-v2":
        raise ValueError("V3 long-text primary model is invalid")
    if int(primary.get("max_wordpieces", -1)) != 256:
        raise ValueError("V3 MiniLM wordpiece limit must be explicit")
    if competitor.get("model_id") != "BAAI/bge-small-en-v1.5":
        raise ValueError("V3 long-text competitor model is invalid")
    if int(competitor.get("max_tokens", -1)) != 512:
        raise ValueError("V3 BGE token limit must be explicit")


def _validate_stages(payload: Mapping[str, Any]) -> None:
    """Validate stage ordering and bounded E1-only authorization."""

    stages = payload.get("stages")
    if not isinstance(stages, list):
        raise ValueError("V3 stages must be a list")
    stage_maps = [
        _require_mapping(stage, f"stages[{index}]")
        for index, stage in enumerate(stages)
    ]
    stage_ids = tuple(str(stage.get("id")) for stage in stage_maps)
    if stage_ids != STAGE_ORDER:
        raise ValueError("V3 stage order differs from the contract")
    design_authorized = tuple(
        str(stage.get("id"))
        for stage in stage_maps
        if stage.get("design_authorized")
    )
    execution_authorized = tuple(
        str(stage.get("id"))
        for stage in stage_maps
        if stage.get("execution_authorized")
    )
    if design_authorized != ("E1",):
        raise ValueError("V3 E0 may authorize only E1 design review")
    if execution_authorized:
        raise ValueError("V3 E0 draft must not authorize execution")
    for stage in stage_maps:
        stage_id = str(stage.get("id"))
        if stage_id not in ("E0", "E1") and "blocked_until" not in stage:
            raise ValueError(f"V3 later stage lacks sequencing guard: {stage_id}")
    e1 = stage_maps[1]
    if e1.get("uses_project_dataset") is not False:
        raise ValueError("V3 E1 must not require the project dataset")


def _validate_e1_smoke(payload: Mapping[str, Any]) -> None:
    """Validate the E1 synthetic-only smoke-test boundary."""

    smoke = _require_mapping(payload.get("e1_smoke"), "e1_smoke")
    if smoke.get("stage_id") != "E1":
        raise ValueError("V3 smoke authorization must target E1")
    if smoke.get("synthetic_texts_only") is not True:
        raise ValueError("V3 E1 smoke must use synthetic texts only")
    if smoke.get("project_dataset_required") is not False:
        raise ValueError("V3 E1 smoke must not require project data")
    if smoke.get("design_action") != DESIGN_AUTHORIZED_E1_ACTION:
        raise ValueError("V3 E1 smoke design action is invalid")
    if smoke.get("execution_allowed") is not False:
        raise ValueError("V3 E1 smoke execution is not authorized")


@dataclass(frozen=True, slots=True)
class V3Protocol:
    """Represent the validated draft V3 development protocol."""

    payload: Mapping[str, Any]

    def validate(self) -> None:
        """Validate every bounded E0 rule in the draft protocol."""

        _validate_status(self.payload)
        _validate_language(self.payload)
        _validate_data_boundary(self.payload)
        _validate_authorization(self.payload)
        _validate_execution(self.payload)
        _validate_candidates(self.payload)
        _validate_long_text_strategy(self.payload)
        _validate_stages(self.payload)
        _validate_e1_smoke(self.payload)

    @property
    def status(self) -> str:
        """Return the draft protocol status."""

        return str(self.payload["protocol_status"]["status"])

    @property
    def input_language(self) -> str:
        """Return the V3 model input-language contract."""

        return str(self.payload["objective"]["input_language"])

    @property
    def candidate_models(self) -> tuple[tuple[str, str], ...]:
        """Return candidate role and model-id pairs in protocol order."""

        return tuple(
            _candidate_key(candidate)
            for candidate in self.payload["candidate_catalog"]
        )

    @property
    def design_authorized_stage_ids(self) -> tuple[str, ...]:
        """Return stage identifiers authorized for E0 design review."""

        auth = self.payload["authorization"]
        return tuple(
            str(stage_id) for stage_id in auth["design_authorized_stage_ids"]
        )

    @property
    def execution_authorized_stage_ids(self) -> tuple[str, ...]:
        """Return execution stage identifiers authorized by the E0 draft."""

        auth = self.payload["authorization"]
        return tuple(
            str(stage_id) for stage_id in auth["execution_authorized_stage_ids"]
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a detached copy of the draft protocol payload."""

        return copy.deepcopy(dict(self.payload))


def load_v3_protocol(path: str | Path = DEFAULT_CONFIG) -> V3Protocol:
    """Load and strictly validate the draft V3 development protocol.

    Args:
        path: Path to ``v3_development_protocol.json``.

    Returns:
        A validated draft V3 protocol object.

    Raises:
        ValueError: If the file cannot be read or violates the E0 contract.
    """

    return validate_v3_protocol(_read_json(Path(path).expanduser().resolve()))


def validate_v3_protocol(payload: Mapping[str, Any]) -> V3Protocol:
    """Validate an in-memory V3 protocol mapping.

    Args:
        payload: JSON-compatible protocol mapping to validate.

    Returns:
        A validated immutable protocol object.

    Raises:
        ValueError: If the mapping violates the bounded V3 E0 contract.
    """

    if not isinstance(payload, Mapping):
        raise ValueError("V3 protocol must be a mapping")
    protocol = V3Protocol(copy.deepcopy(dict(payload)))
    protocol.validate()
    return protocol


def require_development_partition(partition: str) -> str:
    """Accept only train or validation for V3 development execution.

    Args:
        partition: Partition identifier supplied by development code.

    Returns:
        The validated development partition.

    Raises:
        ValueError: If the partition is sealed or unsupported.
    """

    if partition in ALLOWED_PARTITIONS:
        return partition
    if partition in CONSUMED_ARCHIVED_PARTITIONS:
        raise ValueError(f"Consumed partition is archived for V3: {partition}")
    if partition == MONITOR_PARTITION:
        raise ValueError("Monitor partition is diagnostic-only in V3")
    raise ValueError(f"Unsupported V3 development partition: {partition}")


def require_kaggle_t4_gpu(gpu_name: str, protocol: V3Protocol) -> str:
    """Accept only the configured Kaggle T4 GPU name.

    Args:
        gpu_name: Device name observed by an external runner.
        protocol: Validated V3 protocol.

    Returns:
        The accepted device name.

    Raises:
        ValueError: If the device is missing, P100-like, or not T4.
    """

    protocol.validate()
    name = str(gpu_name).strip()
    if not name:
        raise ValueError("V3 Kaggle GPU name is required")
    tokens = protocol.payload["execution"]["forbidden_gpu_tokens"]
    if any(str(token) in name for token in tokens):
        raise ValueError(f"V3 Kaggle GPU is forbidden: {name}")
    accepted = tuple(protocol.payload["execution"]["accepted_gpu_names"])
    if name not in accepted:
        raise ValueError(f"V3 Kaggle GPU must be T4, not: {name}")
    return name


def require_e1_smoke_design_review(
    protocol: V3Protocol,
    *,
    stage_id: str,
    action: str,
    uses_project_dataset: bool,
    partition: str | None = None,
) -> str:
    """Authorize only the synthetic E1 smoke scaffold design review.

    Args:
        protocol: Validated V3 protocol.
        stage_id: Requested V3 stage identifier.
        action: Requested action identifier.
        uses_project_dataset: Whether the request would read project data.
        partition: Optional data partition requested by the caller.

    Returns:
        The design-review action name.

    Raises:
        ValueError: If the request exceeds the bounded E0 design review.
    """

    protocol.validate()
    if stage_id not in protocol.design_authorized_stage_ids:
        raise ValueError(f"V3 stage is not design-authorized: {stage_id}")
    if action != DESIGN_AUTHORIZED_E1_ACTION:
        raise ValueError(f"V3 action is not design-authorized: {action}")
    if uses_project_dataset:
        raise ValueError("V3 E1 design review must not load project data")
    if partition is not None:
        require_development_partition(partition)
    return action
