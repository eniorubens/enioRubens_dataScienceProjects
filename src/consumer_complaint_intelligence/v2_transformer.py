"""D2 compact-transformer challenge runner for the pinned classical winner.

The runner owns exactly one comparison: hold the D1 winner's fit scope,
hard-negative pool, calibration window, outer window, and hierarchical
architecture fixed, and swap only the stage-A model family for a fine-tuned
``distilbert-base-uncased`` binary detector. It never reads the raw dataset,
never unlocks a sealed partition, and persists aggregate evidence only --
no narratives, identifiers, per-row scores, or fitted weights.

``torch`` and ``transformers`` are imported lazily, inside
:func:`build_transformer_scorer` only, so the rest of this module -- and the
full ``full``/``smoke`` orchestration in :func:`run_v2_transformer_challenge`
when a ``scorer_factory`` is injected -- imports and runs without either
dependency installed.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from sklearn.metrics import confusion_matrix

from .s6 import CRITICAL_CLASS, MODELED_FAMILIES
from .s7 import load_s7_predictor
from .v2_benchmark import DEFAULT_CACHE as BENCHMARK_DEFAULT_CACHE
from .v2_benchmark import (
    _fallback_labels,
    _fallback_only_metrics,
    _outer_metrics,
    _read_development_cache,
    _s7_paths,
    _s7_signatures,
    _safe_path,
    _sha256,
    _signature,
    _synthetic_scopes,
    _validate_result_privacy,
    _write_json_atomic,
    generate_hard_negative_indices,
)
from .v2_detector import (
    combine_detector_with_fallback,
    count_override_decisions,
    search_detector_threshold_exact,
)
from .v2_protocol import DEFAULT_CONFIG as V2_PROTOCOL_DEFAULT_CONFIG
from .v2_protocol import (
    V2Protocol,
    calculate_safety_margins,
    calculate_scientific_gates,
    load_v2_protocol,
    validate_baseline_artifacts,
)


D2_RESULT_SCHEMA = "v2-transformer-challenge-v1"
D2_MANIFEST_SCHEMA = "v2-transformer-results-manifest-v1"
D2_CODE_SCHEMA = "v2-transformer-runtime-v1"
DEFAULT_EXECUTION_CONFIG = "config/v2_d2_execution.json"
DEFAULT_D2_ARTIFACT = "temp/v2/v2_transformer_challenge.json"
DEFAULT_D2_MANIFEST = "config/v2_transformer_results.json"
DEFAULT_BATCH_SIZE = 4096
_D2_EXECUTION_SCHEMA = "v2-d2-execution-config-v1"
_D2_FROZEN_SEEDS = (42, 43, 44)


def _read_json(path: Path) -> dict[str, Any]:
    """Read one UTF-8 JSON object from disk.

    Args:
        path: Path to a JSON document.

    Returns:
        The parsed JSON object.

    Raises:
        ValueError: If the document does not decode to a JSON object.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object")
    return payload


def _relative(path: Path, root: Path) -> str:
    """Return one project-relative POSIX path.

    Args:
        path: Absolute path to convert.
        root: Project root the path must live under.

    Returns:
        A POSIX-style path relative to ``root``.
    """

    return path.resolve().relative_to(root.resolve()).as_posix()


def _resolve_path(root: Path, value: str | Path | None, default: str) -> Path:
    """Resolve one optional project-relative or absolute override path.

    Args:
        root: Project root used to resolve relative paths.
        value: Caller-supplied override, or ``None`` to use ``default``.
        default: Project-relative default path.

    Returns:
        The resolved absolute path.
    """

    candidate = value if value is not None else default
    if Path(candidate).is_absolute():
        return Path(candidate).expanduser().resolve()
    return _safe_path(root, str(candidate))


@dataclass(frozen=True, slots=True)
class D2ExecutionConfig:
    """Hold the frozen, pre-registered D2 execution configuration.

    Attributes:
        payload: The validated JSON configuration.
        path: Absolute path the configuration was loaded from.
        signature: ``{"sha256": ..., "size_bytes": ...}`` for ``path``.
    """

    payload: Mapping[str, Any]
    path: Path
    signature: dict[str, Any]

    @property
    def model_id(self) -> str:
        """Return the frozen compact-transformer model identifier."""

        return str(self.payload["model"]["model_id"])

    @property
    def seeds(self) -> tuple[int, ...]:
        """Return the three pre-registered replicate seeds."""

        return tuple(int(value) for value in self.payload["seeds"]["values"])

    @property
    def max_length(self) -> int:
        """Return the frozen tokenizer maximum sequence length."""

        return int(self.payload["tokenizer"]["max_length"])

    @property
    def epochs(self) -> int:
        """Return the frozen fine-tuning epoch count."""

        return int(self.payload["training"]["epochs"])

    @property
    def learning_rate(self) -> float:
        """Return the frozen AdamW learning rate."""

        return float(self.payload["training"]["learning_rate"])

    @property
    def weight_decay(self) -> float:
        """Return the frozen AdamW weight decay."""

        return float(self.payload["training"]["weight_decay"])

    @property
    def warmup_ratio(self) -> float:
        """Return the frozen linear-schedule warmup ratio."""

        return float(self.payload["training"]["warmup_ratio"])

    @property
    def max_grad_norm(self) -> float:
        """Return the frozen gradient-clipping norm."""

        return float(self.payload["training"]["max_grad_norm"])

    @property
    def train_batch_size(self) -> int:
        """Return the frozen training batch size."""

        return int(self.payload["training"]["train_batch_size"])

    @property
    def eval_batch_size(self) -> int:
        """Return the frozen evaluation batch size."""

        return int(self.payload["training"]["eval_batch_size"])

    @property
    def displacement_bar(self) -> float:
        """Return the pre-registered outer critical-F1 displacement bar."""

        return float(self.payload["decision_rule"]["displacement_bar"])

    @property
    def displacement_increment(self) -> float:
        """Return the pre-registered minimum critical-F1 improvement."""

        return float(self.payload["decision_rule"]["displacement_increment"])

    @property
    def precision_floor(self) -> float:
        """Return the pre-registered minimum outer critical precision."""

        return float(self.payload["decision_rule"]["precision_floor"])

    @property
    def incumbent_outer(self) -> dict[str, Any]:
        """Return the pinned D1 incumbent's outer metric vector."""

        return dict(self.payload["incumbent"]["outer"])

    @property
    def incumbent_candidate_id(self) -> str:
        """Return the pinned D1 incumbent candidate identifier."""

        return str(self.payload["incumbent"]["candidate_id"])

    @property
    def incumbent_artifact(self) -> dict[str, Any]:
        """Return the pinned D1 classical artifact path and signature."""

        return dict(self.payload["incumbent"]["artifact"])

    @property
    def incumbent_manifest(self) -> dict[str, Any]:
        """Return the pinned D1 classical manifest path and signature."""

        return dict(self.payload["incumbent"]["manifest"])

    @property
    def decision_conditions(self) -> tuple[str, ...]:
        """Return the pre-registered decision-rule condition names in order."""

        return tuple(
            str(item) for item in self.payload["decision_rule"]["conditions"]
        )


def load_d2_execution_config(
    path: str | Path = DEFAULT_EXECUTION_CONFIG,
) -> D2ExecutionConfig:
    """Load and strictly validate the frozen D2 execution configuration.

    Args:
        path: Path to ``v2_d2_execution.json``.

    Returns:
        A validated D2 execution configuration.

    Raises:
        ValueError: If the schema, seeds, maximum model count, or
            pre-registration flag diverge from the frozen D2 contract.
    """

    resolved = Path(path).expanduser().resolve()
    payload = _read_json(resolved)
    if payload.get("schema_version") != _D2_EXECUTION_SCHEMA:
        raise ValueError("Unexpected D2 execution config schema")
    seeds = payload.get("seeds", {}).get("values")
    if list(seeds or []) != list(_D2_FROZEN_SEEDS):
        raise ValueError("D2 execution config seeds must be exactly 42, 43, 44")
    if int(payload.get("model", {}).get("maximum_models", 0)) != 1:
        raise ValueError("D2 execution config must fix maximum_models to 1")
    if payload.get("decision_rule", {}).get("pre_registered") is not True:
        raise ValueError("D2 decision rule must be pre-registered")
    return D2ExecutionConfig(
        payload=payload, path=resolved, signature=_signature(resolved)
    )


def hard_negative_pool_signature(indices: Sequence[int]) -> str:
    """Fingerprint the hard-negative training pool by its row positions.

    This proves the transformer trained on exactly the same rows as the D1
    classical winner: the pool is built once and reused across every seed.

    Args:
        indices: Row positions returned by ``generate_hard_negative_indices``.

    Returns:
        An uppercase SHA256 hex digest of the sorted, comma-joined indices.
    """

    ordered = sorted(int(value) for value in indices)
    joined = ",".join(str(value) for value in ordered)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest().upper()


def build_transformer_scorer(
    fit_texts: Sequence[str],
    fit_targets: Sequence[int],
    seed: int,
    config: D2ExecutionConfig,
) -> Callable[[Sequence[str]], np.ndarray]:
    """Fine-tune one compact transformer and return a margin-scoring closure.

    This is the only function in the D2 runner that touches ``torch`` or
    ``transformers``; both are imported lazily inside this function body so
    the rest of the module, and every other call path, stays torch-free.
    Training uses a plain PyTorch loop rather than ``transformers.Trainer``
    to avoid the ``accelerate`` dependency and keep the loop fully
    deterministic under a seeded numpy shuffle.

    Args:
        fit_texts: Hard-negative pool narratives, identical across seeds.
        fit_targets: Binary critical targets aligned with ``fit_texts``.
        seed: One of the three pre-registered replicate seeds.
        config: Frozen D2 execution configuration.

    Returns:
        A callable that tokenizes input texts in bounded evaluation batches
        and returns float64 margins, ``logit_critical - logit_non_critical``.
        The callable also carries a ``resolved_revision`` attribute recording
        the model revision actually loaded, or ``"unknown"`` on failure.

    Raises:
        ValueError: If ``fit_texts`` and ``fit_targets`` are misaligned.
    """

    import random

    import torch
    from torch.optim import AdamW
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    texts = tuple(fit_texts)
    targets = tuple(int(value) for value in fit_targets)
    if len(texts) != len(targets) or not texts:
        raise ValueError("fit_texts and fit_targets must be non-empty and aligned")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(config.model_id, do_lower_case=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_id, num_labels=2
    )
    model.to(device)
    model.train()

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    steps_per_epoch = max(1, -(-len(texts) // config.train_batch_size))
    total_steps = config.epochs * steps_per_epoch
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)
    rng = np.random.default_rng(seed)
    row_indices = np.arange(len(texts))

    for _ in range(config.epochs):
        order = rng.permutation(row_indices)
        for start in range(0, len(order), config.train_batch_size):
            batch_index = order[start : start + config.train_batch_size]
            batch_texts = [texts[int(index)] for index in batch_index]
            batch_targets = [targets[int(index)] for index in batch_index]
            encoded = tokenizer(
                batch_texts,
                max_length=config.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            labels_tensor = torch.tensor(
                batch_targets, dtype=torch.long, device=device
            )
            optimizer.zero_grad()
            with torch.autocast(
                device_type="cuda" if use_cuda else "cpu",
                dtype=torch.float16,
                enabled=use_cuda,
            ):
                outputs = model(**encoded, labels=labels_tensor)
                loss = outputs.loss
            if use_cuda:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.max_grad_norm
                )
                optimizer.step()
            scheduler.step()

    model.eval()
    try:
        resolved_revision = str(getattr(model.config, "_name_or_path", "unknown"))
    except Exception:  # noqa: BLE001 -- revision is best-effort metadata
        resolved_revision = "unknown"
    if not resolved_revision:
        resolved_revision = "unknown"

    def _score(batch_texts: Sequence[str]) -> np.ndarray:
        """Score a batch of texts into critical-minus-non-critical margins."""

        values = tuple(batch_texts)
        if not values:
            return np.zeros((0,), dtype=np.float64)
        pieces: list[np.ndarray] = []
        for start in range(0, len(values), config.eval_batch_size):
            chunk = values[start : start + config.eval_batch_size]
            encoded = tokenizer(
                list(chunk),
                max_length=config.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                logits = model(**encoded).logits
            margin = logits[:, 1] - logits[:, 0]
            pieces.append(margin.detach().to(torch.float64).cpu().numpy())
        return np.concatenate(pieces)

    _score.resolved_revision = resolved_revision  # type: ignore[attr-defined]
    return _score


def _synthetic_scorer_factory(
    fit_texts: Sequence[str],
    fit_targets: Sequence[int],
    seed: int,
    config: D2ExecutionConfig,
) -> Callable[[Sequence[str]], np.ndarray]:
    """Score margins deterministically without touching torch.

    Used only by ``smoke`` mode so the D2 diagnostic never imports torch.
    Assigns a strongly positive margin to any text containing the frozen
    critical-class marker used by ``v2_benchmark._synthetic_scopes``, offset
    slightly by ``seed`` so replicate seeds remain distinguishable.

    Args:
        fit_texts: Synthetic pool narratives, validated for alignment only.
        fit_targets: Synthetic binary targets, validated for alignment only.
        seed: Replicate seed, folded into the deterministic margin.
        config: Frozen D2 execution configuration (unused by this stub).

    Returns:
        A callable scoring a batch of texts into float64 margins.

    Raises:
        ValueError: If ``fit_texts`` and ``fit_targets`` are misaligned.
    """

    del config
    if len(tuple(fit_texts)) != len(tuple(fit_targets)):
        raise ValueError("fit_texts and fit_targets must align")

    def _score(batch_texts: Sequence[str]) -> np.ndarray:
        offset = float(seed) * 1e-6
        return np.asarray(
            [
                (5.0 if CRITICAL_CLASS in text else -5.0) + offset
                for text in batch_texts
            ],
            dtype=np.float64,
        )

    _score.resolved_revision = "synthetic"  # type: ignore[attr-defined]
    return _score


def _score_margins(
    scorer: Callable[[Sequence[str]], np.ndarray],
    texts: Sequence[str],
    batch_size: int,
) -> np.ndarray:
    """Score one scope's texts in bounded batches and concatenate margins."""

    pieces: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        stop = min(start + batch_size, len(texts))
        margins = np.asarray(scorer(texts[start:stop]), dtype=np.float64)
        pieces.append(margins)
    if not pieces:
        return np.zeros((0,), dtype=np.float64)
    return np.concatenate(pieces)


def _score_outer_block(
    scorer: Callable[[Sequence[str]], np.ndarray],
    outer_scope: Any,
    fallback_labels: Sequence[str],
    threshold: float,
    batch_size: int,
    protocol: V2Protocol,
) -> dict[str, Any]:
    """Score the outer scope in batches and accumulate one confusion matrix."""

    class_order = list(MODELED_FAMILIES)
    matrix = np.zeros((len(class_order), len(class_order)), dtype=np.int64)
    override_decisions = 0
    effective_overrides = 0
    for start in range(0, len(outer_scope.texts), batch_size):
        stop = min(start + batch_size, len(outer_scope.texts))
        chunk_texts = outer_scope.texts[start:stop]
        chunk_fallback = fallback_labels[start:stop]
        margins = np.asarray(scorer(chunk_texts), dtype=np.float64)
        decisions = margins >= float(threshold)
        predictions = combine_detector_with_fallback(decisions, chunk_fallback)
        actual = outer_scope.labels[start:stop]
        matrix += confusion_matrix(
            actual, predictions, labels=class_order
        ).astype(np.int64)
        batch_override, batch_effective = count_override_decisions(
            decisions, chunk_fallback
        )
        override_decisions += batch_override
        effective_overrides += batch_effective
    metrics = _outer_metrics(matrix)
    return {
        "metrics": metrics,
        "gates": calculate_scientific_gates(metrics, protocol),
        "safety": calculate_safety_margins(metrics, protocol),
        "override_decisions": int(override_decisions),
        "effective_overrides": int(effective_overrides),
    }


def _execute_seeds(
    scopes: Mapping[str, Any],
    fallback: Mapping[str, Sequence[str]],
    protocol: V2Protocol,
    config: D2ExecutionConfig,
    *,
    batch_size: int,
    scorer_factory: Callable[
        [Sequence[str], Sequence[int], int, D2ExecutionConfig],
        Callable[[Sequence[str]], np.ndarray],
    ],
    seeds: Sequence[int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Build the D2 hard-negative pool once and run every replicate seed.

    The pool is generated a single time from ``inner_fit`` and reused across
    every seed, so only the transformer fine-tune varies between replicates.

    Args:
        scopes: The three frozen development scopes.
        fallback: Frozen S7 fallback labels for calibration and outer scopes.
        protocol: Validated V2 development protocol.
        config: Validated D2 execution configuration.
        batch_size: Bounded scoring batch size.
        scorer_factory: Builds one seed's margin-scoring closure.
        seeds: Override for the seeds actually executed. Defaults to every
            seed in ``config.seeds``; ``smoke`` mode passes a single seed.

    Returns:
        Per-seed result blocks, hard-negative pool evidence, and the
        fallback-only baseline metrics on both calibration and outer scopes.
    """

    fit_scope = scopes["inner_fit"]
    calibration_scope = scopes["inner_calibration"]
    outer_scope = scopes["outer_evaluation"]
    fallback_baseline = {
        "inner_calibration": _fallback_only_metrics(
            calibration_scope.labels, fallback["inner_calibration"]
        ),
        "outer_evaluation": _fallback_only_metrics(
            outer_scope.labels, fallback["outer_evaluation"]
        ),
    }
    hard_indices = generate_hard_negative_indices(
        fit_scope.texts,
        fit_scope.labels,
        hard_negative_per_positive=10,
        background_per_positive=5,
        n_splits=3,
        random_state=42,
    )
    positive_count = sum(label == CRITICAL_CLASS for label in fit_scope.labels)
    pool_texts = [fit_scope.texts[index] for index in hard_indices]
    pool_labels = [fit_scope.labels[index] for index in hard_indices]
    pool_targets = [int(label == CRITICAL_CLASS) for label in pool_labels]
    hard_negative_evidence = {
        "positive_groups": int(positive_count),
        "hard_negative_groups": int(len(hard_indices) - positive_count),
        "pool_rows": int(len(hard_indices)),
        "pool_signature": hard_negative_pool_signature(hard_indices),
    }
    seed_results: list[dict[str, Any]] = []
    for seed in (seeds if seeds is not None else config.seeds):
        scorer = scorer_factory(pool_texts, pool_targets, int(seed), config)
        calibration_margins = _score_margins(
            scorer, calibration_scope.texts, batch_size
        )
        threshold_result = search_detector_threshold_exact(
            calibration_scope.labels,
            calibration_margins,
            fallback["inner_calibration"],
            protocol,
        )
        selected = threshold_result["selected"]
        threshold = float(selected["threshold"])
        calibration_block = {
            "threshold": threshold,
            "threshold_count": int(threshold_result["threshold_count"]),
            "metrics": selected["metrics"],
            "gates": selected["gates"],
            "override_decisions": int(selected["override_decisions"]),
            "effective_overrides": int(selected["effective_overrides"]),
        }
        outer_block = _score_outer_block(
            scorer,
            outer_scope,
            fallback["outer_evaluation"],
            threshold,
            batch_size,
            protocol,
        )
        seed_results.append(
            {
                "seed": int(seed),
                "resolved_revision": str(
                    getattr(scorer, "resolved_revision", "unknown")
                ),
                "calibration": calibration_block,
                "outer": outer_block,
            }
        )
    return seed_results, hard_negative_evidence, fallback_baseline


def _select_reported_seed(
    seed_results: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pick the seed attaining the median outer critical F1, never the best.

    Args:
        seed_results: Per-seed result blocks, each with ``outer.metrics``.

    Returns:
        A ``(reported_seed_block, seed_spread)`` pair, where ``seed_spread``
        publishes the minimum, median, maximum, and range of outer critical
        F1 across every executed seed.

    Raises:
        ValueError: If ``seed_results`` is empty.
    """

    if not seed_results:
        raise ValueError("D2 seed results must be non-empty")
    ordered = sorted(
        seed_results,
        key=lambda item: (
            float(item["outer"]["metrics"]["critical_f1"]),
            int(item["seed"]),
        ),
    )
    values = [float(item["outer"]["metrics"]["critical_f1"]) for item in ordered]
    median_index = len(ordered) // 2
    reported = dict(ordered[median_index])
    spread = {
        "critical_f1_min": values[0],
        "critical_f1_median": values[median_index],
        "critical_f1_max": values[-1],
        "critical_f1_range": values[-1] - values[0],
    }
    return reported, spread


def _evaluate_decision_rule(
    reported_seed_block: Mapping[str, Any],
    fallback_outer_critical_f1: float,
    config: D2ExecutionConfig,
) -> dict[str, Any]:
    """Evaluate the five pre-registered D2 displacement conditions.

    Args:
        reported_seed_block: The reported seed's ``outer`` block.
        fallback_outer_critical_f1: The pure S7 fallback's outer critical F1.
        config: Validated D2 execution configuration.

    Returns:
        Every condition boolean, plus ``outcome`` in
        ``{"TRANSFORMER_DISPLACES_CLASSICAL", "CLASSICAL_WINNER_STANDS"}``
        and ``blocked_reason`` (``None``, or the first failed condition).

    Raises:
        ValueError: If the configured condition set does not match the
            five conditions this runner implements.
    """

    outer = reported_seed_block["outer"]
    checks = {
        "passes_three_of_three_development_safety_margins": (
            outer["safety"].get("passed") is True
            and outer["safety"].get("gate_count") == 3
        ),
        "effective_overrides_greater_than_zero": (
            int(outer["effective_overrides"]) > 0
        ),
        "outer_critical_f1_greater_than_fallback_baseline": (
            float(outer["metrics"]["critical_f1"]) > fallback_outer_critical_f1
        ),
        "outer_critical_precision_at_least_precision_floor": (
            float(outer["metrics"]["critical_precision"])
            >= config.precision_floor
        ),
        "outer_critical_f1_at_least_displacement_bar": (
            float(outer["metrics"]["critical_f1"]) >= config.displacement_bar
        ),
    }
    order = config.decision_conditions
    if set(order) != set(checks):
        raise ValueError("D2 decision-rule condition order is unrecognized")
    blocked_reason = next((name for name in order if not checks[name]), None)
    outcome = (
        "TRANSFORMER_DISPLACES_CLASSICAL"
        if blocked_reason is None
        else "CLASSICAL_WINNER_STANDS"
    )
    return {
        **checks,
        "pre_registered": True,
        "displacement_bar": config.displacement_bar,
        "displacement_increment": config.displacement_increment,
        "precision_floor": config.precision_floor,
        "outcome": outcome,
        "blocked_reason": blocked_reason,
    }


_KEY_RENAMES = {"model": "model_spec", "models": "models_spec"}


def _rename_forbidden_keys(value: Any) -> Any:
    """Recursively rename dict keys banned by the reused privacy validator.

    ``_validate_result_privacy`` (reused from ``v2_benchmark``) rejects any
    persisted key literally named ``model`` or ``models`` anywhere in the
    result tree, because that denylist exists to keep fitted estimators out
    of aggregate evidence. The frozen D2 execution config legitimately has
    a static ``model`` metadata section (model id, revision, parameter
    count -- no fitted weights), so it is renamed rather than dropped when
    the config payload is echoed into the artifact.

    Args:
        value: Any JSON-compatible value to sanitize.

    Returns:
        A deep copy with every forbidden key renamed to a safe alias.
    """

    if isinstance(value, Mapping):
        return {
            _KEY_RENAMES.get(str(key).lower(), key): _rename_forbidden_keys(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_rename_forbidden_keys(item) for item in value]
    return value


def _finalize_result(
    *,
    protocol: V2Protocol,
    config: D2ExecutionConfig,
    seed_results: Sequence[Mapping[str, Any]],
    hard_negative_evidence: Mapping[str, Any],
    fallback_baseline: Mapping[str, Any],
    incumbent_block: Mapping[str, Any],
    runtime_seconds: float,
    opened_at: str,
    diagnostic_only: bool,
    signature: str,
) -> dict[str, Any]:
    """Aggregate seeds, evaluate the decision rule, and seal the D2 result."""

    reported_seed_block, seed_spread = _select_reported_seed(seed_results)
    fallback_outer_f1 = float(fallback_baseline["outer_evaluation"]["critical_f1"])
    incumbent_outer_f1 = float(incumbent_block["outer"]["critical_f1"])
    reported_outer = reported_seed_block["outer"]
    critical_f1_vs_fallback = (
        float(reported_outer["metrics"]["critical_f1"]) - fallback_outer_f1
    )
    critical_f1_vs_incumbent = (
        float(reported_outer["metrics"]["critical_f1"]) - incumbent_outer_f1
    )
    reported = {
        "aggregation": "median_of_three_outer_critical_f1",
        "seed": reported_seed_block["seed"],
        "calibration": reported_seed_block["calibration"],
        "outer": reported_outer,
        "critical_f1_vs_fallback": critical_f1_vs_fallback,
        "critical_f1_vs_incumbent": critical_f1_vs_incumbent,
        "gates": reported_outer["gates"],
        "safety": reported_outer["safety"],
    }
    decision = _evaluate_decision_rule(
        reported_seed_block, fallback_outer_f1, config
    )
    result = {
        "schema_version": D2_RESULT_SCHEMA,
        "code_schema": D2_CODE_SCHEMA,
        "status": "DIAGNOSTIC_ONLY" if diagnostic_only else "COMPLETE",
        "complete": True,
        "diagnostic_only": diagnostic_only,
        "opened_at": opened_at,
        "runtime_seconds": float(runtime_seconds),
        "input_language": "en-US",
        "critical_class": protocol.critical_class,
        "allowed_partitions": list(protocol.allowed_partitions),
        "sealed_partitions": list(protocol.forbidden_partitions),
        "sealed_access": {name: False for name in protocol.forbidden_partitions},
        "execution_config": _rename_forbidden_keys(dict(config.payload)),
        "execution_config_signature": dict(config.signature),
        "model_spec": {
            "model_id": config.model_id,
            "resolved_revision": reported_seed_block["resolved_revision"],
        },
        "incumbent": dict(incumbent_block),
        "fallback_baseline": dict(fallback_baseline),
        "hard_negative": dict(hard_negative_evidence),
        "seeds": [dict(item) for item in seed_results],
        "reported": reported,
        "seed_spread": seed_spread,
        "decision": decision,
        "signature": signature,
    }
    _validate_result_privacy(result)
    return result


def _d2_signature(
    config: D2ExecutionConfig,
    protocol_path: Path,
    cache_signature: Mapping[str, Any],
    s7_signatures: Mapping[str, Mapping[str, Any]],
    incumbent_artifact_signature: Mapping[str, Any],
    incumbent_manifest_signature: Mapping[str, Any],
) -> str:
    """Build the full D2 run signature from code, config, and frozen hashes."""

    value = {
        "code_schema": D2_CODE_SCHEMA,
        "execution_config": dict(config.signature),
        "protocol": _signature(protocol_path),
        "cache": dict(cache_signature),
        "s7": {key: dict(item) for key, item in sorted(s7_signatures.items())},
        "incumbent_artifact": dict(incumbent_artifact_signature),
        "incumbent_manifest": dict(incumbent_manifest_signature),
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True).encode("utf-8")
    ).hexdigest().upper()


def _publish_d2_manifest(
    root: Path,
    protocol_path: Path,
    config: D2ExecutionConfig,
    cache_path: Path,
    artifact_path: Path,
    manifest_path: Path,
    s7_signatures: Mapping[str, Mapping[str, Any]],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish a portable D2 manifest after a complete full result."""

    manifest = {
        "schema_version": D2_MANIFEST_SCHEMA,
        "stage": "V2.1-D2",
        "status": result["status"],
        "complete": True,
        "diagnostic_only": False,
        "signature": result["signature"],
        "protocol": {
            "path": _relative(protocol_path, root),
            **_signature(protocol_path),
        },
        "execution_config": {
            "path": _relative(config.path, root),
            **dict(config.signature),
        },
        "cache": {"path": _relative(cache_path, root), **_signature(cache_path)},
        "s7": {
            key: {"path": _relative(path, root), **dict(signature)}
            for key, path, signature in (
                ("bundle", _s7_paths(root)[0], s7_signatures["bundle"]),
                ("manifest", _s7_paths(root)[1], s7_signatures["manifest"]),
                ("result", _s7_paths(root)[2], s7_signatures["result"]),
            )
        },
        "artifact": {
            "path": _relative(artifact_path, root),
            **_signature(artifact_path),
        },
        "sealed_access": {name: False for name in ("test", "stress", "monitor")},
        "outcome": result["decision"]["outcome"],
        "reported_seed": result["reported"]["seed"],
        "critical_f1_vs_incumbent": result["reported"]["critical_f1_vs_incumbent"],
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest


def validate_d2_manifest(
    manifest_path: str | Path, artifact_path: str | Path
) -> dict[str, Any]:
    """Validate a complete portable D2 result manifest and artifact.

    Mirrors :func:`v2_benchmark.validate_v2_manifest`: re-hashes the
    artifact and every referenced input, checks schemas, cross-checks the
    outcome and reported seed against the stored decision, and re-derives
    ``critical_f1_vs_fallback`` and ``critical_f1_vs_incumbent`` from the
    stored metrics so a tampered artifact fails.

    Args:
        manifest_path: Manifest JSON path.
        artifact_path: Aggregate D2 result JSON path.

    Returns:
        The validated manifest payload.

    Raises:
        ValueError: If a hash, schema, or cross-check invariant fails.
    """

    manifest_file = Path(manifest_path).expanduser().resolve()
    artifact_file = Path(artifact_path).expanduser().resolve()
    manifest = _read_json(manifest_file)
    if manifest.get("schema_version") != D2_MANIFEST_SCHEMA:
        raise ValueError("Unexpected D2 manifest schema")
    if (
        manifest.get("complete") is not True
        or manifest.get("diagnostic_only") is not False
    ):
        raise ValueError("D2 manifest is not a complete full result")
    if any(manifest.get("sealed_access", {}).values()):
        raise ValueError("D2 manifest claims sealed-partition access")
    root = manifest_file.parent.parent.resolve()
    for role in ("protocol", "execution_config", "cache", "artifact"):
        item = manifest.get(role, {})
        path = _safe_path(root, item.get("path"))
        if (
            item.get("sha256") != _sha256(path)
            or item.get("size_bytes") != path.stat().st_size
        ):
            raise ValueError(f"D2 manifest {role} hash is invalid")
    s7_paths = _s7_paths(root)
    for role, path in zip(("bundle", "manifest", "result"), s7_paths):
        item = manifest.get("s7", {}).get(role, {})
        if _safe_path(root, item.get("path")) != path.resolve():
            raise ValueError(f"D2 manifest S7 {role} path is invalid")
        if (
            item.get("sha256") != _sha256(path)
            or item.get("size_bytes") != path.stat().st_size
        ):
            raise ValueError(f"D2 manifest S7 {role} hash is invalid")
    artifact = _safe_path(root, manifest["artifact"].get("path"))
    if artifact != artifact_file:
        raise ValueError("D2 manifest artifact path differs from requested result")
    result = _read_json(artifact_file)
    if (
        result.get("schema_version") != D2_RESULT_SCHEMA
        or result.get("complete") is not True
    ):
        raise ValueError("D2 aggregate result is incomplete or stale")
    if result.get("signature") != manifest.get("signature"):
        raise ValueError("D2 result signature differs from manifest")
    _validate_result_privacy(result)
    reported = result.get("reported", {})
    if manifest.get("outcome") != result.get("decision", {}).get("outcome"):
        raise ValueError("D2 manifest outcome differs from result decision")
    if manifest.get("reported_seed") != reported.get("seed"):
        raise ValueError("D2 manifest reported seed differs from result")
    fallback_outer_f1 = float(
        result["fallback_baseline"]["outer_evaluation"]["critical_f1"]
    )
    reported_f1 = float(reported["outer"]["metrics"]["critical_f1"])
    expected_vs_fallback = reported_f1 - fallback_outer_f1
    if reported.get("critical_f1_vs_fallback") != expected_vs_fallback:
        raise ValueError("D2 result critical_f1_vs_fallback is inconsistent")
    incumbent_f1 = float(result["incumbent"]["outer"]["critical_f1"])
    expected_vs_incumbent = reported_f1 - incumbent_f1
    if reported.get("critical_f1_vs_incumbent") != expected_vs_incumbent:
        raise ValueError("D2 result critical_f1_vs_incumbent is inconsistent")
    if manifest.get("critical_f1_vs_incumbent") != reported.get(
        "critical_f1_vs_incumbent"
    ):
        raise ValueError("D2 manifest critical_f1_vs_incumbent differs from result")
    return manifest


def run_v2_transformer_challenge(
    mode: str,
    *,
    project_root: str | Path | None = None,
    config_path: str | Path | None = None,
    protocol_path: str | Path | None = None,
    batch_size: int | None = None,
    scorer_factory: (
        Callable[
            [Sequence[str], Sequence[int], int, D2ExecutionConfig],
            Callable[[Sequence[str]], np.ndarray],
        ]
        | None
    ) = None,
) -> dict[str, Any]:
    """Run the D2 compact-transformer challenge in ``full`` or ``smoke`` mode.

    ``full`` mode pins the D1 classical winner by hash, reads the frozen S3
    development cache, scores the frozen S7 fallback, builds the D1
    hard-negative pool once, fine-tunes one seed at a time, and evaluates
    the pre-registered decision rule on the seed attaining the median outer
    critical F1. ``smoke`` mode runs the identical mechanics on tiny
    synthetic in-memory data with a single seed and never touches real
    project files or the real incumbent pin.

    Args:
        mode: ``full`` for the real D1-pinned run, or ``smoke`` for a
            synthetic, torch-free diagnostic.
        project_root: Project root for relative paths. Defaults to the
            current working directory.
        config_path: Frozen D2 execution config path.
        protocol_path: Frozen V2 development protocol path.
        batch_size: Bounded scoring batch size for calibration and outer
            evaluation.
        scorer_factory: Dependency-injection seam building one seed's
            margin-scoring closure, ``(fit_texts, fit_targets, seed,
            config) -> scorer``. Defaults to :func:`build_transformer_scorer`
            in ``full`` mode and to a deterministic, torch-free stub in
            ``smoke`` mode. Supplying it explicitly keeps every torch
            import out of the call path, which is how tests exercise the
            whole pipeline without torch installed.

    Returns:
        The aggregate-only D2 challenge result.

    Raises:
        ValueError: If ``mode`` is invalid, ``batch_size`` is not positive,
            or the pinned D1 incumbent has drifted from the frozen
            execution configuration.
    """

    if mode not in {"full", "smoke"}:
        raise ValueError("mode must be full or smoke")
    resolved_batch_size = int(batch_size or DEFAULT_BATCH_SIZE)
    if resolved_batch_size <= 0:
        raise ValueError("batch_size must be positive")
    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else Path.cwd()
    )
    config_file = _resolve_path(root, config_path, DEFAULT_EXECUTION_CONFIG)
    protocol_file = _resolve_path(
        root, protocol_path, V2_PROTOCOL_DEFAULT_CONFIG
    )
    protocol = load_v2_protocol(protocol_file)
    config = load_d2_execution_config(config_file)
    opened_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    if mode == "smoke":
        scopes, fallback = _synthetic_scopes()
        resolved_scorer_factory = scorer_factory or _synthetic_scorer_factory
        started = time.perf_counter()
        seed_results, hard_negative_evidence, fallback_baseline = _execute_seeds(
            scopes,
            fallback,
            protocol,
            config,
            batch_size=resolved_batch_size,
            scorer_factory=resolved_scorer_factory,
            seeds=(config.seeds[0],),
        )
        runtime_seconds = time.perf_counter() - started
        incumbent_block = {
            "candidate_id": config.incumbent_candidate_id,
            "outer": config.incumbent_outer,
            "artifact": dict(config.incumbent_artifact),
            "manifest": dict(config.incumbent_manifest),
        }
        return _finalize_result(
            protocol=protocol,
            config=config,
            seed_results=seed_results,
            hard_negative_evidence=hard_negative_evidence,
            fallback_baseline=fallback_baseline,
            incumbent_block=incumbent_block,
            runtime_seconds=runtime_seconds,
            opened_at=opened_at,
            diagnostic_only=True,
            signature="smoke",
        )

    validate_baseline_artifacts(protocol, root)
    incumbent_artifact_path = _safe_path(
        root, str(config.incumbent_artifact["path"])
    )
    incumbent_manifest_path = _safe_path(
        root, str(config.incumbent_manifest["path"])
    )
    actual_artifact_signature = _signature(incumbent_artifact_path)
    actual_manifest_signature = _signature(incumbent_manifest_path)
    expected_artifact_signature = {
        "sha256": str(config.incumbent_artifact["sha256"]),
        "size_bytes": int(config.incumbent_artifact["size_bytes"]),
    }
    expected_manifest_signature = {
        "sha256": str(config.incumbent_manifest["sha256"]),
        "size_bytes": int(config.incumbent_manifest["size_bytes"]),
    }
    if actual_artifact_signature != expected_artifact_signature:
        raise ValueError("D2 pinned incumbent classical artifact has drifted")
    if actual_manifest_signature != expected_manifest_signature:
        raise ValueError("D2 pinned incumbent classical manifest has drifted")

    cache_file = _safe_path(root, BENCHMARK_DEFAULT_CACHE)
    if not cache_file.is_file():
        raise FileNotFoundError(cache_file)
    scopes = _read_development_cache(cache_file, protocol, resolved_batch_size)
    bundle, s7_manifest, s7_result = _s7_paths(root)
    predictor = load_s7_predictor(bundle, s7_manifest, s7_result)
    fallback = {
        role: _fallback_labels(predictor, scopes[role], resolved_batch_size)
        for role in ("inner_calibration", "outer_evaluation")
    }
    s7_signatures = _s7_signatures(root)
    resolved_scorer_factory = scorer_factory or build_transformer_scorer
    started = time.perf_counter()
    seed_results, hard_negative_evidence, fallback_baseline = _execute_seeds(
        scopes,
        fallback,
        protocol,
        config,
        batch_size=resolved_batch_size,
        scorer_factory=resolved_scorer_factory,
    )
    runtime_seconds = time.perf_counter() - started
    incumbent_block = {
        "candidate_id": config.incumbent_candidate_id,
        "outer": config.incumbent_outer,
        "artifact": {
            "path": str(config.incumbent_artifact["path"]),
            **actual_artifact_signature,
        },
        "manifest": {
            "path": str(config.incumbent_manifest["path"]),
            **actual_manifest_signature,
        },
    }
    signature = _d2_signature(
        config,
        protocol_file,
        _signature(cache_file),
        s7_signatures,
        actual_artifact_signature,
        actual_manifest_signature,
    )
    result = _finalize_result(
        protocol=protocol,
        config=config,
        seed_results=seed_results,
        hard_negative_evidence=hard_negative_evidence,
        fallback_baseline=fallback_baseline,
        incumbent_block=incumbent_block,
        runtime_seconds=runtime_seconds,
        opened_at=opened_at,
        diagnostic_only=False,
        signature=signature,
    )
    artifact_file = _safe_path(root, DEFAULT_D2_ARTIFACT)
    manifest_file = _safe_path(root, DEFAULT_D2_MANIFEST)
    _write_json_atomic(artifact_file, result)
    _publish_d2_manifest(
        root,
        protocol_file,
        config,
        cache_file,
        artifact_file,
        manifest_file,
        s7_signatures,
        result,
    )
    validate_d2_manifest(manifest_file, artifact_file)
    return result


def run_v2_transformer_smoke(
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run the D2 synthetic diagnostic mode without real project files.

    Returns:
        The synthetic, single-seed, diagnostic-only D2 result.
    """

    return run_v2_transformer_challenge(
        "smoke", project_root=project_root
    )


__all__ = [
    "D2_CODE_SCHEMA",
    "D2_MANIFEST_SCHEMA",
    "D2_RESULT_SCHEMA",
    "DEFAULT_D2_ARTIFACT",
    "DEFAULT_D2_MANIFEST",
    "DEFAULT_EXECUTION_CONFIG",
    "D2ExecutionConfig",
    "build_transformer_scorer",
    "hard_negative_pool_signature",
    "load_d2_execution_config",
    "run_v2_transformer_challenge",
    "run_v2_transformer_smoke",
    "validate_d2_manifest",
]
