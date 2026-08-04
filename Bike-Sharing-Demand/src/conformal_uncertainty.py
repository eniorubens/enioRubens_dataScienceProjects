"""Prequential temporal conformal calibration for notebook 07.

The workflow consumes only the frozen development OOF predictions produced by
notebook 06.  It does not fit or call an estimator, and every interval at time
``t`` is constructed exclusively from scores observed before ``t`` inside the
same temporal fold.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import mlflow
import numpy as np
import pandas as pd

from src.environment import describe_environment, describe_git_source_state, require_environment
from src.normal_operations import (
    DEFAULT_SELECTION_FOLD_WEIGHTS,
    DEFAULT_SELECTION_TEST_YEARS,
    DEFAULT_STRESS_TEST_YEARS,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_RUNTIME_ROOT = (
    _PROJECT_ROOT / "dataset" / "normal_operations" / "uncertainty_experiments_v1"
)
DEFAULT_SOURCE_MANIFEST_PATH = DEFAULT_SOURCE_RUNTIME_ROOT / "uncertainty_experiments_manifest.json"
DEFAULT_SOURCE_PREDICTIONS_PATH = DEFAULT_SOURCE_RUNTIME_ROOT / "development_oof_predictions.csv"
DEFAULT_RUNTIME_ROOT = _PROJECT_ROOT / "dataset" / "normal_operations" / "conformal_uncertainty_v1"
DEFAULT_SMOKE_RUNTIME_ROOT = (
    _PROJECT_ROOT / "dataset" / "normal_operations" / "conformal_uncertainty_v1_smoke"
)
DEFAULT_TRACKING_URI = f"file:{_PROJECT_ROOT / 'mlruns'}"

CONFORMAL_CODE_VERSION = "conformal_uncertainty_v2_prequential"
CONFORMAL_STATUS = "experimental_uncertainty_calibrator"

SOURCE_USECOLS: Tuple[str, ...] = (
    "timestamp",
    "weekday",
    "hour",
    "Rush_Period",
    "Seasons",
    "Rainfall Cat",
    "experiment_id",
    "fold",
    "test_year",
    "fold_role",
    "selection_eligible",
    "y_true",
    "y_pred",
    "residual",
    "predicted_scale",
)

_RESULT_TABLE_FILES = {
    "input_audit": "input_audit.csv",
    "method_specs": "method_specs.csv",
    "fold_metrics": "fold_metrics.csv",
    "aggregate_metrics": "aggregate_metrics.csv",
    "stress_metrics": "stress_metrics.csv",
    "scale_diagnostics": "scale_diagnostics.csv",
    "rolling_coverage": "rolling_coverage.csv.gz",
    "segment_metrics": "segment_metrics.csv",
    "bootstrap_metrics": "bootstrap_metrics.csv",
    "decision_table": "decision_table.csv",
    "aci_alpha_trace": "aci_alpha_trace.csv.gz",
    "predictions": "conformal_oof_predictions.csv.gz",
}


@dataclass
class ConformalMethodSpec:
    """One declared calibrator or sensitivity variant."""

    candidate_id: str
    method_id: str
    label: str
    calibration_window: str
    nonconformity: str
    adaptive_alpha: bool
    uses_e4_scale: bool
    status: str = "planned"
    notes: str = ""


@dataclass
class ConformalUncertaintyConfig:
    """Declaration surface for notebook 07."""

    run_mode: str = "smoke"
    source_manifest_path: Path = DEFAULT_SOURCE_MANIFEST_PATH
    source_predictions_path: Path = DEFAULT_SOURCE_PREDICTIONS_PATH
    runtime_root: Optional[Path] = None
    tracking_uri: Optional[str] = DEFAULT_TRACKING_URI
    experiment_name: str = "bike_sharing_demand_v4_conformal_uncertainty"
    point_experiment_id: str = "E0"
    scale_experiment_id: str = "E4"
    selection_test_years: Sequence[int] = DEFAULT_SELECTION_TEST_YEARS
    stress_test_years: Sequence[int] = DEFAULT_STRESS_TEST_YEARS
    fold_weights: Sequence[float] = DEFAULT_SELECTION_FOLD_WEIGHTS
    interval_coverages: Sequence[float] = (0.80, 0.90, 0.95)
    primary_coverage: float = 0.90
    warmup_hours: int = 168
    rolling_window_hours: int = 2160
    recency_half_life_hours: Sequence[int] = (168, 720, 2160)
    aci_gamma_values: Sequence[float] = (0.001, 0.005, 0.01)
    aci_alpha_bounds: Tuple[float, float] = (0.001, 0.5)
    scale_epsilon: float = 1e-6
    global_min_history: int = 168
    group_min_history: int = 96
    bootstrap_block_hours: int = 168
    smoke_bootstrap_repetitions: int = 50
    full_bootstrap_repetitions: int = 500
    smoke_fold_limit: int = 2
    smoke_hours_per_fold: int = 720
    coverage_tolerance: float = 0.02
    maximum_fold_shortfall: float = 0.05
    segment_min_n: int = 168
    maximum_segment_shortfall: float = 0.10
    log_to_mlflow: bool = True
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.run_mode not in {"smoke", "full"}:
            raise ValueError("run_mode must be 'smoke' or 'full'.")
        if self.warmup_hours < 1 or self.rolling_window_hours < self.warmup_hours:
            raise ValueError("rolling_window_hours must be at least warmup_hours.")
        if self.global_min_history < 1 or self.group_min_history < 1:
            raise ValueError("history thresholds must be positive.")
        if self.smoke_fold_limit < 1 or self.smoke_hours_per_fold <= self.warmup_hours:
            raise ValueError("smoke mode must contain folds and observations after warm-up.")
        if len(self.fold_weights) != len(self.selection_test_years):
            raise ValueError("fold_weights must match selection_test_years.")
        if not all(0.0 < float(level) < 1.0 for level in self.interval_coverages):
            raise ValueError("interval_coverages must lie in (0, 1).")
        if not any(np.isclose(self.primary_coverage, level) for level in self.interval_coverages):
            raise ValueError("primary_coverage must be one of interval_coverages.")
        if not all(float(value) > 0.0 for value in self.recency_half_life_hours):
            raise ValueError("recency half-lives must be positive.")
        if not all(float(value) > 0.0 for value in self.aci_gamma_values):
            raise ValueError("ACI gamma values must be positive.")
        lower, upper = self.aci_alpha_bounds
        if not 0.0 < lower < upper < 1.0:
            raise ValueError("aci_alpha_bounds must satisfy 0 < lower < upper < 1.")
        if self.runtime_root is None:
            self.runtime_root = (
                DEFAULT_SMOKE_RUNTIME_ROOT if self.run_mode == "smoke" else DEFAULT_RUNTIME_ROOT
            )
        self.source_manifest_path = Path(self.source_manifest_path)
        self.source_predictions_path = Path(self.source_predictions_path)
        self.runtime_root = Path(self.runtime_root)

    @property
    def fold_limit(self) -> Optional[int]:
        return self.smoke_fold_limit if self.run_mode == "smoke" else None

    @property
    def bootstrap_repetitions(self) -> int:
        return (
            self.smoke_bootstrap_repetitions
            if self.run_mode == "smoke"
            else self.full_bootstrap_repetitions
        )


@dataclass
class ConformalCalibrationResults:
    """Displayable and persisted results from notebook 07."""

    config: ConformalUncertaintyConfig
    source_manifest: Mapping[str, Any]
    input_audit: pd.DataFrame
    specs: List[ConformalMethodSpec]
    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    aggregate_metrics: pd.DataFrame
    stress_metrics: pd.DataFrame
    scale_diagnostics: pd.DataFrame
    rolling_coverage: pd.DataFrame
    segment_metrics: pd.DataFrame
    bootstrap_metrics: pd.DataFrame
    decision_table: pd.DataFrame
    aci_alpha_trace: pd.DataFrame
    manifest_path: Path
    artifacts: Dict[str, str] = field(default_factory=dict)

    @property
    def is_smoke(self) -> bool:
        return self.config.run_mode == "smoke"


def _format_float_id(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def conformal_method_specs(
    config: Optional[ConformalUncertaintyConfig] = None,
) -> List[ConformalMethodSpec]:
    """Return every pre-registered main candidate and sensitivity variant."""
    config = config or ConformalUncertaintyConfig(log_to_mlflow=False)
    specs = [
        ConformalMethodSpec(
            "U0",
            "U0",
            "Conformal expansível simétrico",
            "expanding_within_fold",
            "absolute_residual",
            False,
            False,
        ),
        ConformalMethodSpec(
            "U1",
            "U1",
            "Conformal rolling assimétrico",
            f"rolling_{config.rolling_window_hours}h",
            "signed_residual",
            False,
            False,
        ),
    ]
    for half_life in config.recency_half_life_hours:
        specs.append(
            ConformalMethodSpec(
                f"U2_h{int(half_life)}",
                "U2",
                f"Conformal ponderado por recência, meia-vida {int(half_life)}h",
                f"rolling_{config.rolling_window_hours}h",
                "recency_weighted_absolute_residual",
                False,
                False,
                notes=f"half_life_hours={int(half_life)}",
            )
        )
    specs.append(
        ConformalMethodSpec(
            "U3",
            "U3",
            "Conformal normalizado pela escala E4",
            f"rolling_{config.rolling_window_hours}h",
            "absolute_residual_over_e4_scale",
            False,
            True,
        )
    )
    for gamma in config.aci_gamma_values:
        suffix = _format_float_id(float(gamma))
        specs.extend(
            [
                ConformalMethodSpec(
                    f"U4a_g{suffix}",
                    "U4a",
                    f"ACI sobre o erro absoluto, gamma={gamma:g}",
                    f"rolling_{config.rolling_window_hours}h",
                    "adaptive_absolute_residual",
                    True,
                    False,
                    notes=f"gamma={gamma:g}",
                ),
                ConformalMethodSpec(
                    f"U4b_g{suffix}",
                    "U4b",
                    f"ACI normalizado pela escala E4, gamma={gamma:g}",
                    f"rolling_{config.rolling_window_hours}h",
                    "adaptive_normalized_residual",
                    True,
                    True,
                    notes=f"gamma={gamma:g}",
                ),
            ]
        )
    specs.append(
        ConformalMethodSpec(
            "U5",
            "U5",
            "Conformal rolling assimétrico e hierárquico por regime",
            f"rolling_{config.rolling_window_hours}h",
            "hierarchical_signed_residual",
            False,
            False,
        )
    )
    return specs


def load_source_manifest(path: Path = DEFAULT_SOURCE_MANIFEST_PATH) -> Mapping[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_source_predictions(
    config: ConformalUncertaintyConfig, manifest: Mapping[str, Any]
) -> Path:
    configured = Path(config.source_predictions_path)
    if configured.exists():
        return configured.resolve()
    declared = manifest.get("artifacts", {}).get("predictions")
    if not declared:
        raise FileNotFoundError("The source manifest does not declare its predictions artifact.")
    declared_path = Path(declared)
    candidates = [declared_path, Path(config.source_manifest_path).parent / declared_path.name]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError("The frozen notebook 06 predictions artifact could not be resolved.")


def validate_source_manifest(
    config: ConformalUncertaintyConfig, manifest: Mapping[str, Any]
) -> None:
    """Fail closed when the frozen source contract is incomplete or incompatible."""
    required = (
        "code_version",
        "run_mode",
        "dataset_fingerprint",
        "regime_fingerprint",
        "cv_strategy",
        "cv_strategy_version",
    )
    missing = [field for field in required if not manifest.get(field)]
    if missing:
        raise ValueError(f"Source manifest missing required fields: {missing}")
    if manifest["run_mode"] != "full":
        raise ValueError("Notebook 07 requires the full notebook 06 OOF artifact.")
    if "normal_operations" not in str(manifest["cv_strategy_version"]):
        raise ValueError("Source CV strategy is not the normal-operations protocol.")
    _resolve_source_predictions(config, manifest)


def source_artifact_hashes(
    config: ConformalUncertaintyConfig, manifest: Optional[Mapping[str, Any]] = None
) -> pd.DataFrame:
    manifest = manifest or load_source_manifest(config.source_manifest_path)
    validate_source_manifest(config, manifest)
    predictions_path = _resolve_source_predictions(config, manifest)
    paths = [Path(config.source_manifest_path).resolve(), predictions_path]
    return pd.DataFrame(
        [
            {
                "artifact": "source_manifest" if index == 0 else "source_predictions",
                "path": str(path),
                "bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
            for index, path in enumerate(paths)
        ]
    )


def load_conformal_calibration_results(
    config: ConformalUncertaintyConfig,
    results_manifest_path: Optional[Path] = None,
) -> ConformalCalibrationResults:
    """Reconstruct notebook 07 reports from persisted conformal artifacts.

    The replay is read-only and never recalculates a calibrator. It fails
    closed if the configuration, source hashes, or conformal code contract no
    longer matches the persisted full run.
    """
    require_environment()
    root = Path(config.runtime_root)
    manifest_path = Path(results_manifest_path or root / "conformal_uncertainty_manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Conformal calibration manifest not found at '{manifest_path}'. "
            "Run the explicit calibration workflow before requesting replay."
        )

    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    saved_config = saved.get("config", {})
    current_config = json.loads(json.dumps(asdict(config), default=str))
    mismatches = {}
    if saved.get("code_version") != CONFORMAL_CODE_VERSION:
        mismatches["code_version"] = {
            "expected": CONFORMAL_CODE_VERSION,
            "saved": saved.get("code_version"),
        }
    if saved_config != current_config:
        mismatches["config"] = {"expected": current_config, "saved": saved_config}
    if mismatches:
        raise ValueError(
            "Persisted conformal results are incompatible with the current contract: "
            + json.dumps(mismatches, sort_keys=True)
        )

    source_manifest = load_source_manifest(config.source_manifest_path)
    validate_source_manifest(config, source_manifest)
    current_audit = source_artifact_hashes(config, source_manifest)
    saved_audit = pd.DataFrame(saved.get("input_hashes", []))
    audit_columns = ["artifact", "bytes", "sha256"]
    if saved_audit.empty or not current_audit[audit_columns].equals(saved_audit[audit_columns]):
        raise ValueError("Persisted conformal results do not match the current source hashes.")

    table_paths = {name: root / filename for name, filename in _RESULT_TABLE_FILES.items()}
    missing = [str(path) for path in table_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Persisted conformal results are incomplete; missing: " + ", ".join(missing)
        )

    tables = {name: pd.read_csv(path, low_memory=False) for name, path in table_paths.items()}
    for name in ("predictions", "rolling_coverage", "aci_alpha_trace"):
        if "timestamp" in tables[name]:
            tables[name]["timestamp"] = pd.to_datetime(tables[name]["timestamp"])
    specs = [ConformalMethodSpec(**payload) for payload in saved.get("methods", [])]
    if not specs:
        raise ValueError("Persisted conformal manifest contains no method specifications.")

    artifacts = {name: str(path) for name, path in table_paths.items()}
    artifacts["manifest"] = str(manifest_path)
    return ConformalCalibrationResults(
        config=config,
        source_manifest=source_manifest,
        input_audit=tables["input_audit"],
        specs=specs,
        predictions=tables["predictions"],
        fold_metrics=tables["fold_metrics"],
        aggregate_metrics=tables["aggregate_metrics"],
        stress_metrics=tables["stress_metrics"],
        scale_diagnostics=tables["scale_diagnostics"],
        rolling_coverage=tables["rolling_coverage"],
        segment_metrics=tables["segment_metrics"],
        bootstrap_metrics=tables["bootstrap_metrics"],
        decision_table=tables["decision_table"],
        aci_alpha_trace=tables["aci_alpha_trace"],
        manifest_path=manifest_path,
        artifacts=artifacts,
    )


def load_point_and_scale_predictions(
    config: ConformalUncertaintyConfig,
    manifest: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """Read the frozen CSV once and align E0 and E4 by exact temporal keys."""
    manifest = manifest or load_source_manifest(config.source_manifest_path)
    validate_source_manifest(config, manifest)
    path = _resolve_source_predictions(config, manifest)
    frame = pd.read_csv(path, usecols=list(SOURCE_USECOLS), parse_dates=["timestamp"])
    keys = ["fold", "test_year", "timestamp"]
    point = frame.loc[frame["experiment_id"].eq(config.point_experiment_id)].copy()
    scale = frame.loc[frame["experiment_id"].eq(config.scale_experiment_id)].copy()
    if point.duplicated(keys).any() or scale.duplicated(keys).any():
        raise ValueError("E0 or E4 contains duplicate temporal keys.")
    scale = scale[keys + ["y_true", "fold_role", "predicted_scale"]].rename(
        columns={
            "y_true": "y_true_e4",
            "fold_role": "fold_role_e4",
            "predicted_scale": "predicted_scale_e4",
        }
    )
    point = point.drop(columns=["experiment_id", "predicted_scale"])
    merged = point.merge(scale, on=keys, how="outer", validate="one_to_one", indicator=True)
    if not merged["_merge"].eq("both").all():
        raise ValueError("E0 and E4 temporal keys do not align one-to-one.")
    if not np.allclose(
        merged["y_true"].to_numpy(dtype=float),
        merged["y_true_e4"].to_numpy(dtype=float),
        equal_nan=True,
    ):
        raise ValueError("E0 and E4 disagree on y_true for aligned timestamps.")
    if not merged["fold_role"].astype(str).eq(merged["fold_role_e4"].astype(str)).all():
        raise ValueError("E0 and E4 disagree on fold_role.")
    computed_residual = merged["y_true"].to_numpy(dtype=float) - merged["y_pred"].to_numpy(
        dtype=float
    )
    if not np.allclose(
        computed_residual,
        merged["residual"].to_numpy(dtype=float),
        rtol=1e-8,
        atol=1e-8,
        equal_nan=True,
    ):
        raise ValueError("Persisted E0 residual is inconsistent with y_true - y_pred.")
    merged["predicted_scale"] = pd.to_numeric(merged["predicted_scale_e4"], errors="coerce")
    merged["scale_valid"] = np.isfinite(merged["predicted_scale"]) & merged["predicted_scale"].gt(
        0.0
    )
    return (
        merged.drop(columns=["_merge", "y_true_e4", "fold_role_e4", "predicted_scale_e4"])
        .rename(columns={"Rainfall Cat": "Rainfall_Cat"})
        .sort_values(keys)
        .reset_index(drop=True)
    )


def _finite_conformal_quantile(scores: Iterable[float], alpha: float) -> float:
    clean = np.asarray(list(scores), dtype=float)
    clean = np.sort(clean[np.isfinite(clean)])
    if clean.size == 0:
        return float("nan")
    rank = int(np.ceil((clean.size + 1) * (1.0 - float(alpha))))
    return float(clean[min(max(rank, 1), clean.size) - 1])


def _asymmetric_residual_bounds(residuals: Iterable[float], coverage: float) -> Tuple[float, float]:
    clean = np.asarray(list(residuals), dtype=float)
    clean = np.sort(clean[np.isfinite(clean)])
    if clean.size == 0:
        return float("nan"), float("nan")
    tail = (1.0 - float(coverage)) / 2.0
    lower_rank = int(np.floor((clean.size + 1) * tail))
    upper_rank = int(np.ceil((clean.size + 1) * (1.0 - tail)))
    lower = clean[min(max(lower_rank, 1), clean.size) - 1]
    upper = clean[min(max(upper_rank, 1), clean.size) - 1]
    return float(lower), float(upper)


def _weighted_quantile(values: Iterable[float], weights: Iterable[float], q: float) -> float:
    values_array = np.asarray(list(values), dtype=float)
    weights_array = np.asarray(list(weights), dtype=float)
    valid = np.isfinite(values_array) & np.isfinite(weights_array) & (weights_array > 0.0)
    if not valid.any():
        return float("nan")
    values_array = values_array[valid]
    weights_array = weights_array[valid]
    order = np.argsort(values_array, kind="mergesort")
    values_array = values_array[order]
    weights_array = weights_array[order]
    cumulative = np.cumsum(weights_array)
    target = float(q) * cumulative[-1]
    index = int(np.searchsorted(cumulative, target, side="left"))
    return float(values_array[min(index, values_array.size - 1)])


def _winkler_score(y_true: float, lower: float, upper: float, coverage: float) -> float:
    alpha = 1.0 - float(coverage)
    score = upper - lower
    if y_true < lower:
        score += (2.0 / alpha) * (lower - y_true)
    elif y_true > upper:
        score += (2.0 / alpha) * (y_true - upper)
    return float(score)


def _fold_frames(frame: pd.DataFrame, config: ConformalUncertaintyConfig) -> List[pd.DataFrame]:
    metadata = (
        frame[["fold", "test_year", "fold_role", "timestamp"]]
        .groupby(["fold", "test_year", "fold_role"], as_index=False)["timestamp"]
        .min()
        .sort_values("timestamp")
    )
    if config.run_mode == "smoke":
        metadata = metadata.loc[metadata["fold_role"].eq("selection")].head(config.smoke_fold_limit)
    frames = []
    for row in metadata.itertuples(index=False):
        fold_frame = frame.loc[
            frame["fold"].eq(row.fold)
            & frame["test_year"].eq(row.test_year)
            & frame["fold_role"].eq(row.fold_role)
        ].sort_values("timestamp")
        if config.run_mode == "smoke":
            fold_frame = fold_frame.head(config.smoke_hours_per_fold)
        frames.append(fold_frame.reset_index(drop=True))
    return frames


def _interval_record(
    row: Any,
    candidate_id: str,
    method_id: str,
    coverage: float,
    lower: float,
    upper: float,
    calibration_size: int,
    status: str,
    alpha_used: Optional[float] = None,
    alpha_after: Optional[float] = None,
    fallback_used: bool = False,
    hierarchy_level: str = "",
    hierarchy_key: str = "",
) -> Dict[str, Any]:
    available = bool(np.isfinite(lower) and np.isfinite(upper))
    y_true = float(row.y_true)
    if available:
        lower = max(0.0, float(lower))
        upper = max(lower, float(upper))
        covered = bool(lower <= y_true <= upper)
        lower_miss = bool(y_true < lower)
        upper_miss = bool(y_true > upper)
        width = float(upper - lower)
        winkler = _winkler_score(y_true, lower, upper, coverage)
    else:
        lower = upper = covered = lower_miss = upper_miss = width = winkler = np.nan
    return {
        "candidate_id": candidate_id,
        "method_id": method_id,
        "coverage": float(coverage),
        "timestamp": row.timestamp,
        "fold": int(row.fold),
        "test_year": int(row.test_year),
        "fold_role": row.fold_role,
        "weekday": row.weekday,
        "hour": int(row.hour),
        "Rush_Period": row.Rush_Period,
        "Seasons": row.Seasons,
        "Rainfall Cat": row.Rainfall_Cat,
        "y_true": y_true,
        "y_pred": float(row.y_pred),
        "predicted_scale": float(row.predicted_scale)
        if np.isfinite(row.predicted_scale)
        else np.nan,
        "scale_valid": bool(row.scale_valid),
        "lower": lower,
        "upper": upper,
        "interval_available": available,
        "status": status,
        "calibration_size": int(calibration_size),
        "alpha_used": alpha_used,
        "alpha_after": alpha_after,
        "fallback_used": bool(fallback_used),
        "hierarchy_level": hierarchy_level,
        "hierarchy_key": hierarchy_key,
        "covered": covered,
        "lower_miss": lower_miss,
        "upper_miss": upper_miss,
        "width": width,
        "winkler_score": winkler,
    }


def _rain_value(row: Any) -> str:
    return str(row.Rainfall_Cat)


def _run_u0(frame: pd.DataFrame, config: ConformalUncertaintyConfig) -> pd.DataFrame:
    records = []
    for fold_frame in _fold_frames(frame, config):
        history: List[float] = []
        for row in fold_frame.itertuples(index=False, name="Observation"):
            for coverage in config.interval_coverages:
                if len(history) < config.warmup_hours:
                    lower = upper = np.nan
                    status = "warmup"
                else:
                    width = _finite_conformal_quantile(history, 1.0 - coverage)
                    lower, upper = row.y_pred - width, row.y_pred + width
                    status = "ok"
                records.append(
                    _interval_record(row, "U0", "U0", coverage, lower, upper, len(history), status)
                )
            history.append(abs(float(row.residual)))
    return pd.DataFrame(records)


def _run_u1(frame: pd.DataFrame, config: ConformalUncertaintyConfig) -> pd.DataFrame:
    records = []
    for fold_frame in _fold_frames(frame, config):
        history: Deque[float] = deque(maxlen=config.rolling_window_hours)
        for row in fold_frame.itertuples(index=False, name="Observation"):
            for coverage in config.interval_coverages:
                if len(history) < config.warmup_hours:
                    lower = upper = np.nan
                    status = "warmup"
                else:
                    residual_lower, residual_upper = _asymmetric_residual_bounds(history, coverage)
                    lower = row.y_pred + residual_lower
                    upper = row.y_pred + residual_upper
                    status = "ok"
                records.append(
                    _interval_record(row, "U1", "U1", coverage, lower, upper, len(history), status)
                )
            history.append(float(row.residual))
    return pd.DataFrame(records)


def _run_u2(
    frame: pd.DataFrame, config: ConformalUncertaintyConfig, half_life: int
) -> pd.DataFrame:
    candidate_id = f"U2_h{int(half_life)}"
    records = []
    for fold_frame in _fold_frames(frame, config):
        history: Deque[Tuple[pd.Timestamp, float]] = deque(maxlen=config.rolling_window_hours)
        for row in fold_frame.itertuples(index=False, name="Observation"):
            if len(history) >= config.warmup_hours:
                timestamps = np.array([item[0].value for item in history], dtype=np.int64)
                values = np.array([item[1] for item in history], dtype=float)
                ages = (row.timestamp.value - timestamps) / 3.6e12
                weights = np.power(0.5, ages / float(half_life))
            else:
                values = weights = np.asarray([], dtype=float)
            for coverage in config.interval_coverages:
                if len(history) < config.warmup_hours:
                    lower = upper = np.nan
                    status = "warmup"
                else:
                    width = _weighted_quantile(values, weights, coverage)
                    lower, upper = row.y_pred - width, row.y_pred + width
                    status = "ok"
                records.append(
                    _interval_record(
                        row,
                        candidate_id,
                        "U2",
                        coverage,
                        lower,
                        upper,
                        len(history),
                        status,
                    )
                )
            history.append((row.timestamp, abs(float(row.residual))))
    return pd.DataFrame(records)


def _run_u3(frame: pd.DataFrame, config: ConformalUncertaintyConfig) -> pd.DataFrame:
    records = []
    for fold_frame in _fold_frames(frame, config):
        normalized: Deque[float] = deque(maxlen=config.rolling_window_hours)
        absolute: Deque[float] = deque(maxlen=config.rolling_window_hours)
        for row in fold_frame.itertuples(index=False, name="Observation"):
            for coverage in config.interval_coverages:
                fallback = False
                if len(absolute) < config.warmup_hours:
                    lower = upper = np.nan
                    status = "warmup"
                    size = len(absolute)
                elif row.scale_valid and len(normalized) >= config.warmup_hours:
                    q_value = _finite_conformal_quantile(normalized, 1.0 - coverage)
                    width = q_value * float(row.predicted_scale)
                    lower, upper = row.y_pred - width, row.y_pred + width
                    status = "ok"
                    size = len(normalized)
                else:
                    width = _finite_conformal_quantile(absolute, 1.0 - coverage)
                    lower, upper = row.y_pred - width, row.y_pred + width
                    status = "fallback_absolute"
                    fallback = True
                    size = len(absolute)
                records.append(
                    _interval_record(
                        row,
                        "U3",
                        "U3",
                        coverage,
                        lower,
                        upper,
                        size,
                        status,
                        fallback_used=fallback,
                    )
                )
            absolute.append(abs(float(row.residual)))
            if row.scale_valid:
                denominator = max(float(row.predicted_scale), config.scale_epsilon)
                normalized.append(abs(float(row.residual)) / denominator)
    return pd.DataFrame(records)


def _run_u4(
    frame: pd.DataFrame,
    config: ConformalUncertaintyConfig,
    normalized_method: bool,
    gamma: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    method_id = "U4b" if normalized_method else "U4a"
    candidate_id = f"{method_id}_g{_format_float_id(float(gamma))}"
    records = []
    trace = []
    lower_bound, upper_bound = config.aci_alpha_bounds
    for fold_frame in _fold_frames(frame, config):
        histories = {
            float(coverage): deque(maxlen=config.rolling_window_hours)
            for coverage in config.interval_coverages
        }
        absolute: Deque[float] = deque(maxlen=config.rolling_window_hours)
        alpha_state = {
            float(coverage): 1.0 - float(coverage) for coverage in config.interval_coverages
        }
        for row in fold_frame.itertuples(index=False, name="Observation"):
            for coverage in config.interval_coverages:
                coverage = float(coverage)
                history = histories[coverage]
                alpha_before = alpha_state[coverage]
                fallback = False
                if len(absolute) < config.warmup_hours:
                    lower = upper = np.nan
                    status = "warmup"
                    size = len(absolute)
                elif normalized_method and (
                    not row.scale_valid or len(history) < config.warmup_hours
                ):
                    width = _finite_conformal_quantile(absolute, alpha_before)
                    lower, upper = row.y_pred - width, row.y_pred + width
                    status = "fallback_absolute"
                    fallback = True
                    size = len(absolute)
                else:
                    q_value = _finite_conformal_quantile(history, alpha_before)
                    width = q_value * float(row.predicted_scale) if normalized_method else q_value
                    lower, upper = row.y_pred - width, row.y_pred + width
                    status = "ok"
                    size = len(history)
                available = np.isfinite(lower) and np.isfinite(upper)
                if available:
                    bounded_lower = max(0.0, float(lower))
                    error = float(not (bounded_lower <= float(row.y_true) <= float(upper)))
                    alpha_after = float(
                        np.clip(
                            alpha_before + float(gamma) * ((1.0 - coverage) - error),
                            lower_bound,
                            upper_bound,
                        )
                    )
                    alpha_state[coverage] = alpha_after
                else:
                    error = np.nan
                    alpha_after = alpha_before
                records.append(
                    _interval_record(
                        row,
                        candidate_id,
                        method_id,
                        coverage,
                        lower,
                        upper,
                        size,
                        status,
                        alpha_used=alpha_before,
                        alpha_after=alpha_after,
                        fallback_used=fallback,
                    )
                )
                trace.append(
                    {
                        "candidate_id": candidate_id,
                        "coverage": coverage,
                        "fold": int(row.fold),
                        "test_year": int(row.test_year),
                        "fold_role": row.fold_role,
                        "timestamp": row.timestamp,
                        "alpha_before": alpha_before,
                        "error_indicator": error,
                        "alpha_after": alpha_after,
                    }
                )
            absolute.append(abs(float(row.residual)))
            for coverage in config.interval_coverages:
                if normalized_method:
                    if row.scale_valid:
                        score = abs(float(row.residual)) / max(
                            float(row.predicted_scale), config.scale_epsilon
                        )
                        histories[float(coverage)].append(score)
                else:
                    histories[float(coverage)].append(abs(float(row.residual)))
    return pd.DataFrame(records), pd.DataFrame(trace)


def _hierarchy_keys(row: Any) -> List[Tuple[str, str]]:
    rain = _rain_value(row)
    rush = str(row.Rush_Period)
    season = str(row.Seasons)
    return [
        ("rush_rain_season", f"{rush}|{rain}|{season}"),
        ("rush_rain", f"{rush}|{rain}"),
        ("rush", rush),
        ("global", "global"),
    ]


def _run_u5(frame: pd.DataFrame, config: ConformalUncertaintyConfig) -> pd.DataFrame:
    records = []
    for fold_frame in _fold_frames(frame, config):
        histories: Dict[Tuple[str, str], Deque[float]] = defaultdict(
            lambda: deque(maxlen=config.rolling_window_hours)
        )
        for row in fold_frame.itertuples(index=False, name="Observation"):
            keys = _hierarchy_keys(row)
            chosen_level = chosen_key = ""
            chosen_history: Optional[Deque[float]] = None
            global_history = histories[("global", "global")]
            if len(global_history) >= config.global_min_history:
                for level, key in keys[:-1]:
                    history = histories[(level, key)]
                    if len(history) >= config.group_min_history:
                        chosen_level, chosen_key, chosen_history = level, key, history
                        break
                if chosen_history is None:
                    chosen_level, chosen_key, chosen_history = (
                        "global",
                        "global",
                        global_history,
                    )
            for coverage in config.interval_coverages:
                if chosen_history is None:
                    lower = upper = np.nan
                    size = len(global_history)
                    status = "warmup"
                    fallback = False
                else:
                    residual_lower, residual_upper = _asymmetric_residual_bounds(
                        chosen_history, coverage
                    )
                    lower = row.y_pred + residual_lower
                    upper = row.y_pred + residual_upper
                    size = len(chosen_history)
                    fallback = chosen_level != "rush_rain_season"
                    status = "fallback_hierarchy" if fallback else "ok"
                records.append(
                    _interval_record(
                        row,
                        "U5",
                        "U5",
                        coverage,
                        lower,
                        upper,
                        size,
                        status,
                        fallback_used=fallback,
                        hierarchy_level=chosen_level,
                        hierarchy_key=chosen_key,
                    )
                )
            residual = float(row.residual)
            for level, key in keys:
                histories[(level, key)].append(residual)
    return pd.DataFrame(records)


def _worst_miss_streak(values: Iterable[bool]) -> int:
    worst = current = 0
    for covered in values:
        current = 0 if bool(covered) else current + 1
        worst = max(worst, current)
    return int(worst)


def calibration_fold_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ["candidate_id", "method_id", "coverage", "fold"]
    for keys, group in predictions.groupby(group_columns, sort=False):
        candidate_id, method_id, coverage, fold = keys
        scored = group.loc[group["interval_available"]].sort_values("timestamp")
        base = {
            "candidate_id": candidate_id,
            "method_id": method_id,
            "coverage": float(coverage),
            "fold": int(fold),
            "test_year": int(group["test_year"].iloc[0]),
            "fold_role": group["fold_role"].iloc[0],
            "n_total": int(len(group)),
            "n_scored": int(len(scored)),
            "n_warmup": int((~group["interval_available"]).sum()),
            "maximum_calibration_size": int(group["calibration_size"].max()),
        }
        if scored.empty:
            rows.append(
                {
                    **base,
                    "empirical_coverage": np.nan,
                    "coverage_error": np.nan,
                    "absolute_coverage_error": np.nan,
                    "mean_width": np.nan,
                    "median_width": np.nan,
                    "winkler_score": np.nan,
                    "lower_miss_rate": np.nan,
                    "upper_miss_rate": np.nan,
                    "fallback_rate": np.nan,
                    "worst_miss_streak": np.nan,
                }
            )
            continue
        empirical = float(scored["covered"].astype(bool).mean())
        rows.append(
            {
                **base,
                "empirical_coverage": empirical,
                "coverage_error": empirical - float(coverage),
                "absolute_coverage_error": abs(empirical - float(coverage)),
                "mean_width": float(scored["width"].mean()),
                "median_width": float(scored["width"].median()),
                "winkler_score": float(scored["winkler_score"].mean()),
                "lower_miss_rate": float(scored["lower_miss"].astype(bool).mean()),
                "upper_miss_rate": float(scored["upper_miss"].astype(bool).mean()),
                "fallback_rate": float(scored["fallback_used"].astype(bool).mean()),
                "worst_miss_streak": _worst_miss_streak(scored["covered"]),
            }
        )
    return pd.DataFrame(rows)


def calibration_aggregate_metrics(
    fold_metrics: pd.DataFrame, config: ConformalUncertaintyConfig
) -> pd.DataFrame:
    weights_by_year = dict(zip(config.selection_test_years, config.fold_weights))
    rows = []
    selection = fold_metrics.loc[fold_metrics["fold_role"].eq("selection")]
    metrics = (
        "empirical_coverage",
        "coverage_error",
        "absolute_coverage_error",
        "mean_width",
        "median_width",
        "winkler_score",
        "lower_miss_rate",
        "upper_miss_rate",
        "fallback_rate",
    )
    for keys, group in selection.groupby(["candidate_id", "method_id", "coverage"], sort=False):
        candidate_id, method_id, coverage = keys
        valid = group.dropna(subset=["empirical_coverage"])
        if valid.empty:
            continue
        weights = np.asarray(
            [weights_by_year.get(int(year), 1.0) for year in valid["test_year"]], dtype=float
        )
        row = {
            "candidate_id": candidate_id,
            "method_id": method_id,
            "coverage": float(coverage),
            "n_folds_calibrated": int(len(valid)),
            "n_scored": int(valid["n_scored"].sum()),
            "coverage_std_between_folds": float(valid["empirical_coverage"].std(ddof=0)),
            "worst_fold_coverage": float(valid["empirical_coverage"].min()),
        }
        for metric in metrics:
            row[f"{metric}_weighted"] = float(np.average(valid[metric], weights=weights))
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(["coverage", "winkler_score_weighted"])
        .reset_index(drop=True)
    )


def scale_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    selection = frame.loc[frame["fold_role"].eq("selection") & frame["scale_valid"]].copy()
    selection["absolute_error"] = selection["residual"].abs()
    scopes = [("overall", selection)] + [
        (f"fold_{int(fold)}", group) for fold, group in selection.groupby("fold", sort=True)
    ]
    for scope, group in scopes:
        if group.empty:
            continue
        group = group.copy()
        group["scale_decile"] = pd.qcut(
            group["predicted_scale"], q=10, labels=False, duplicates="drop"
        )
        deciles = (
            group.groupby("scale_decile", observed=True)
            .agg(
                n=("absolute_error", "size"),
                mean_predicted_scale=("predicted_scale", "mean"),
                mean_absolute_error=("absolute_error", "mean"),
            )
            .reset_index()
        )
        differences = deciles["mean_absolute_error"].diff().dropna()
        monotonic_rate = float((differences >= 0.0).mean()) if not differences.empty else np.nan
        spearman = float(group["predicted_scale"].corr(group["absolute_error"], method="spearman"))
        for row in deciles.itertuples(index=False):
            rows.append(
                {
                    "scope": scope,
                    "scale_decile": int(row.scale_decile) + 1,
                    "n": int(row.n),
                    "mean_predicted_scale": float(row.mean_predicted_scale),
                    "mean_absolute_error": float(row.mean_absolute_error),
                    "spearman_scale_abs_error": spearman,
                    "monotonic_growth_rate": monotonic_rate,
                }
            )
    return pd.DataFrame(rows)


def rolling_coverage_metrics(
    predictions: pd.DataFrame, config: ConformalUncertaintyConfig
) -> pd.DataFrame:
    primary = predictions.loc[
        np.isclose(predictions["coverage"], config.primary_coverage)
        & predictions["interval_available"]
    ].copy()
    rows = []
    for keys, group in primary.groupby(["candidate_id", "fold"], sort=False):
        candidate_id, fold = keys
        group = group.sort_values("timestamp")
        covered = group["covered"].astype(float)
        for window in (168, 720):
            rolling = covered.rolling(window, min_periods=window).mean()
            valid = rolling.notna()
            rows.append(
                pd.DataFrame(
                    {
                        "candidate_id": candidate_id,
                        "fold": int(fold),
                        "test_year": int(group["test_year"].iloc[0]),
                        "fold_role": group["fold_role"].iloc[0],
                        "timestamp": group.loc[valid, "timestamp"].to_numpy(),
                        "window_hours": window,
                        "rolling_coverage": rolling.loc[valid].to_numpy(),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _demand_bands(frame: pd.DataFrame) -> pd.Series:
    labels = ["muito_baixa", "baixa", "media", "alta", "muito_alta"]
    try:
        return pd.qcut(frame["y_pred"], q=5, labels=labels, duplicates="drop").astype(str)
    except ValueError:
        return pd.Series("indefinida", index=frame.index)


def segment_calibration_metrics(
    predictions: pd.DataFrame, config: ConformalUncertaintyConfig
) -> pd.DataFrame:
    frame = predictions.loc[
        np.isclose(predictions["coverage"], config.primary_coverage)
        & predictions["interval_available"]
        & predictions["fold_role"].eq("selection")
    ].copy()
    frame["predicted_demand_band"] = ""
    for _fold, index in frame.groupby("fold").groups.items():
        frame.loc[index, "predicted_demand_band"] = _demand_bands(frame.loc[index]).to_numpy()
    frame["friday_18"] = np.where(
        frame["weekday"].eq(4) & frame["hour"].eq(18), "sexta_18h", "demais_horas"
    )
    rows = []
    for segment_column in (
        "Rush_Period",
        "Seasons",
        "Rainfall Cat",
        "predicted_demand_band",
        "friday_18",
        "hierarchy_level",
    ):
        for keys, group in frame.groupby(["candidate_id", segment_column], dropna=False):
            candidate_id, segment_value = keys
            n = len(group)
            observed = float(group["covered"].astype(bool).mean())
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "segment_type": segment_column,
                    "segment_value": str(segment_value),
                    "n": int(n),
                    "empirical_coverage": observed,
                    "coverage_error": observed - config.primary_coverage,
                    "mean_width": float(group["width"].mean()),
                    "winkler_score": float(group["winkler_score"].mean()),
                    "fallback_rate": float(group["fallback_used"].astype(bool).mean()),
                    "segment_alert": bool(
                        n >= config.segment_min_n
                        and observed < config.primary_coverage - config.maximum_segment_shortfall
                    ),
                }
            )
    return pd.DataFrame(rows)


def _sample_blocks(values: np.ndarray, block_hours: int, rng: np.random.Generator) -> np.ndarray:
    if values.size == 0:
        return values
    starts = np.arange(max(values.size - block_hours + 1, 1))
    blocks = []
    while sum(len(block) for block in blocks) < values.size:
        start = int(rng.choice(starts))
        blocks.append(values[start : start + block_hours])
    return np.concatenate(blocks)[: values.size]


def block_bootstrap_coverage(
    predictions: pd.DataFrame, config: ConformalUncertaintyConfig
) -> pd.DataFrame:
    frame = predictions.loc[
        np.isclose(predictions["coverage"], config.primary_coverage)
        & predictions["interval_available"]
        & predictions["fold_role"].eq("selection")
    ].copy()
    weights_by_year = dict(zip(config.selection_test_years, config.fold_weights))
    rng = np.random.default_rng(config.random_state)
    rows = []
    for candidate_id, candidate in frame.groupby("candidate_id", sort=False):
        fold_values = []
        for (_fold, year), group in candidate.groupby(["fold", "test_year"], sort=True):
            fold_values.append(
                (
                    int(year),
                    group.sort_values("timestamp")["covered"].astype(float).to_numpy(),
                )
            )
        observed = np.average(
            [values.mean() for _year, values in fold_values],
            weights=[weights_by_year.get(year, 1.0) for year, _values in fold_values],
        )
        replicates = []
        for _ in range(config.bootstrap_repetitions):
            fold_means = [
                _sample_blocks(values, config.bootstrap_block_hours, rng).mean()
                for _year, values in fold_values
            ]
            replicates.append(
                np.average(
                    fold_means,
                    weights=[weights_by_year.get(year, 1.0) for year, _values in fold_values],
                )
            )
        rows.append(
            {
                "candidate_id": candidate_id,
                "coverage": config.primary_coverage,
                "observed_coverage": float(observed),
                "ci_lower": float(np.quantile(replicates, 0.025)),
                "ci_upper": float(np.quantile(replicates, 0.975)),
                "bootstrap_repetitions": int(config.bootstrap_repetitions),
                "block_hours": int(config.bootstrap_block_hours),
            }
        )
    return pd.DataFrame(rows)


def experimental_decision_table(
    aggregate: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    segments: pd.DataFrame,
    config: ConformalUncertaintyConfig,
) -> pd.DataFrame:
    primary = aggregate.loc[np.isclose(aggregate["coverage"], config.primary_coverage)].copy()
    rows = []
    for row in primary.itertuples(index=False):
        candidate_folds = fold_metrics.loc[
            fold_metrics["candidate_id"].eq(row.candidate_id)
            & np.isclose(fold_metrics["coverage"], config.primary_coverage)
            & fold_metrics["fold_role"].eq("selection")
        ]
        worst_fold = float(candidate_folds["empirical_coverage"].min())
        candidate_segments = segments.loc[segments["candidate_id"].eq(row.candidate_id)]
        segment_alerts = int(candidate_segments["segment_alert"].sum())
        feasible = bool(
            config.run_mode == "full"
            and abs(row.coverage_error_weighted) <= config.coverage_tolerance
            and worst_fold >= config.primary_coverage - config.maximum_fold_shortfall
        )
        rows.append(
            {
                "candidate_id": row.candidate_id,
                "method_id": row.method_id,
                "empirical_coverage": row.empirical_coverage_weighted,
                "coverage_error": row.coverage_error_weighted,
                "worst_fold_coverage": worst_fold,
                "winkler_score": row.winkler_score_weighted,
                "mean_width": row.mean_width_weighted,
                "coverage_std_between_folds": row.coverage_std_between_folds,
                "fallback_rate": row.fallback_rate_weighted,
                "segment_alerts": segment_alerts,
                "coverage_gate": feasible,
            }
        )
    decision = pd.DataFrame(rows)
    if decision.empty:
        return decision
    decision["experimental_rank"] = np.nan
    feasible = decision.loc[decision["coverage_gate"]].sort_values(
        [
            "winkler_score",
            "mean_width",
            "coverage_std_between_folds",
            "fallback_rate",
        ]
    )
    decision.loc[feasible.index, "experimental_rank"] = np.arange(1, len(feasible) + 1)
    return decision.sort_values(
        ["coverage_gate", "experimental_rank", "winkler_score"],
        ascending=[False, True, True],
        na_position="last",
    ).reset_index(drop=True)


def _persist_results(results: ConformalCalibrationResults) -> Path:
    root = Path(results.config.runtime_root)
    root.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "input_audit": root / "input_audit.csv",
        "method_specs": root / "method_specs.csv",
        "fold_metrics": root / "fold_metrics.csv",
        "aggregate_metrics": root / "aggregate_metrics.csv",
        "stress_metrics": root / "stress_metrics.csv",
        "scale_diagnostics": root / "scale_diagnostics.csv",
        "rolling_coverage": root / "rolling_coverage.csv.gz",
        "segment_metrics": root / "segment_metrics.csv",
        "bootstrap_metrics": root / "bootstrap_metrics.csv",
        "decision_table": root / "decision_table.csv",
        "aci_alpha_trace": root / "aci_alpha_trace.csv.gz",
        "predictions": root / "conformal_oof_predictions.csv.gz",
    }
    results.input_audit.to_csv(artifacts["input_audit"], index=False)
    pd.DataFrame([asdict(spec) for spec in results.specs]).to_csv(
        artifacts["method_specs"], index=False
    )
    results.fold_metrics.to_csv(artifacts["fold_metrics"], index=False)
    results.aggregate_metrics.to_csv(artifacts["aggregate_metrics"], index=False)
    results.stress_metrics.to_csv(artifacts["stress_metrics"], index=False)
    results.scale_diagnostics.to_csv(artifacts["scale_diagnostics"], index=False)
    results.rolling_coverage.to_csv(artifacts["rolling_coverage"], index=False, compression="gzip")
    results.segment_metrics.to_csv(artifacts["segment_metrics"], index=False)
    results.bootstrap_metrics.to_csv(artifacts["bootstrap_metrics"], index=False)
    results.decision_table.to_csv(artifacts["decision_table"], index=False)
    results.aci_alpha_trace.to_csv(artifacts["aci_alpha_trace"], index=False, compression="gzip")
    results.predictions.to_csv(artifacts["predictions"], index=False, compression="gzip")
    results.artifacts.update({name: str(path) for name, path in artifacts.items()})
    manifest = {
        "code_version": CONFORMAL_CODE_VERSION,
        "run_mode": results.config.run_mode,
        "status": CONFORMAL_STATUS,
        "source_code_version": results.source_manifest.get("code_version"),
        "source_dataset_fingerprint": results.source_manifest.get("dataset_fingerprint"),
        "source_regime_fingerprint": results.source_manifest.get("regime_fingerprint"),
        "cv_strategy": results.source_manifest.get("cv_strategy"),
        "cv_strategy_version": results.source_manifest.get("cv_strategy_version"),
        "input_hashes": results.input_audit.to_dict(orient="records"),
        "environment": describe_environment(),
        "git_source": describe_git_source_state(),
        "operational_contract": {
            "forecast_horizon": "rolling_one_step_ahead",
            "warmup_hours": results.config.warmup_hours,
            "observation_at_t_updates": "t_plus_1_only",
            "state_reset": "every_fold_and_regime",
            "stress_policy": "reported_separately_and_never_ranked",
        },
        "config": asdict(results.config),
        "methods": [asdict(spec) for spec in results.specs],
        "artifacts": results.artifacts,
    }
    manifest_path = root / "conformal_uncertainty_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    results.artifacts["manifest"] = str(manifest_path)
    results.manifest_path = manifest_path
    return manifest_path


def _log_to_mlflow(results: ConformalCalibrationResults) -> None:
    if not results.config.log_to_mlflow:
        return
    mlflow.set_tracking_uri(results.config.tracking_uri)
    mlflow.set_experiment(results.config.experiment_name)
    with mlflow.start_run(run_name=f"notebook07_{results.config.run_mode}"):
        mlflow.set_tags(
            {
                "code_version": CONFORMAL_CODE_VERSION,
                "run_mode": results.config.run_mode,
                "status": CONFORMAL_STATUS,
                "source_dataset_fingerprint": results.source_manifest.get("dataset_fingerprint"),
            }
        )
        for row in results.decision_table.itertuples(index=False):
            with mlflow.start_run(run_name=row.candidate_id, nested=True):
                mlflow.log_param("method_id", row.method_id)
                mlflow.log_metric("empirical_coverage", float(row.empirical_coverage))
                mlflow.log_metric("mean_width", float(row.mean_width))
                mlflow.log_metric("winkler_score", float(row.winkler_score))
                mlflow.log_metric("fallback_rate", float(row.fallback_rate))
                mlflow.log_metric("coverage_gate", int(row.coverage_gate))
        mlflow.log_artifact(results.artifacts["manifest"])


def run_conformal_uncertainty_experiments(
    config: ConformalUncertaintyConfig,
    predictions: Optional[pd.DataFrame] = None,
) -> ConformalCalibrationResults:
    """Run U0-U5 on frozen development OOF predictions and persist the audit."""
    require_environment()
    source_manifest = load_source_manifest(config.source_manifest_path)
    validate_source_manifest(config, source_manifest)
    input_audit = source_artifact_hashes(config, source_manifest)
    if predictions is None:
        predictions = load_point_and_scale_predictions(config, source_manifest)
    base = predictions.copy(deep=True)
    prediction_frames = [_run_u0(base, config), _run_u1(base, config)]
    for half_life in config.recency_half_life_hours:
        prediction_frames.append(_run_u2(base, config, int(half_life)))
    prediction_frames.append(_run_u3(base, config))
    alpha_frames = []
    for gamma in config.aci_gamma_values:
        for normalized_method in (False, True):
            candidate, trace = _run_u4(base, config, normalized_method, float(gamma))
            prediction_frames.append(candidate)
            alpha_frames.append(trace)
    prediction_frames.append(_run_u5(base, config))
    conformal_predictions = pd.concat(prediction_frames, ignore_index=True)
    alpha_trace = pd.concat(alpha_frames, ignore_index=True)
    fold_metrics = calibration_fold_metrics(conformal_predictions)
    aggregate = calibration_aggregate_metrics(fold_metrics, config)
    stress = fold_metrics.loc[fold_metrics["fold_role"].eq("stress")].copy()
    scale = scale_diagnostics(base)
    rolling = rolling_coverage_metrics(conformal_predictions, config)
    segments = segment_calibration_metrics(conformal_predictions, config)
    bootstrap = block_bootstrap_coverage(conformal_predictions, config)
    decision = experimental_decision_table(aggregate, fold_metrics, segments, config)
    specs = conformal_method_specs(config)
    for spec in specs:
        spec.status = "executed"
    results = ConformalCalibrationResults(
        config=config,
        source_manifest=source_manifest,
        input_audit=input_audit,
        specs=specs,
        predictions=conformal_predictions,
        fold_metrics=fold_metrics,
        aggregate_metrics=aggregate,
        stress_metrics=stress,
        scale_diagnostics=scale,
        rolling_coverage=rolling,
        segment_metrics=segments,
        bootstrap_metrics=bootstrap,
        decision_table=decision,
        aci_alpha_trace=alpha_trace,
        manifest_path=Path(config.runtime_root) / "conformal_uncertainty_manifest.json",
    )
    _persist_results(results)
    _log_to_mlflow(results)
    return results
