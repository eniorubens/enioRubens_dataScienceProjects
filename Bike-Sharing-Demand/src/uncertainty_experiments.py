"""Controlled residual and uncertainty experiments for notebook 06.

This module deliberately stays on the development split. It rebuilds the
registered CatBoost champion inside each temporal fold, adds controlled
ablation variants, and keeps any successor claim experimental.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.diagnostic import het_arch

from src.cv import _resolve_timestamps
from src.environment import describe_environment, describe_git_source_state, require_environment
from src.feature_engineering import (
    EXPERIMENTAL_CATEGORICAL_FEATURES,
    build_preprocessing_pipeline,
    build_residual_uncertainty_feature_pipeline,
)
from src.model_selection_workflow import (
    DevelopmentData,
    ModelSelectionConfig,
    prepare_development_data,
)
from src.modeling_pipeline import (
    CATEGORICAL_FEATURES,
    ESTIMATOR_CLASSES,
    NUMERICAL_FEATURES,
    SEARCH_PROFILE_REFINED,
    TARGET_STRATEGY_ROBUST_TREND,
    PipelineSpec,
    build_dynamic_pipeline,
)
from src.normal_operations import (
    DEFAULT_EXCLUSION_END,
    DEFAULT_EXCLUSION_START,
    DEFAULT_SELECTION_FOLD_WEIGHTS,
    DEFAULT_SELECTION_TEST_YEARS,
    DEFAULT_STRESS_TEST_YEARS,
    REGIME_POLICY_NORMAL_OPERATIONS,
)
from src.optimizer import suppress_category_encoder_intercept_warning
from src.probabilistic_modeling import (
    DEFAULT_INTERVAL_COVERAGES,
    CatBoostResidualUncertaintyRegressor,
    as_probabilistic_trend_regressor,
    lognormal_demand_distribution,
)
from src.temporal_optimizer import CODE_VERSION, CV_STRATEGY_NAME, CV_STRATEGY_VERSION
from src.trend import RobustTrendResidualRegressor

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST_PATH = (
    _PROJECT_ROOT / "dataset" / "normal_operations" / "candidates_v4" / "candidates_manifest.json"
)
DEFAULT_RUNTIME_ROOT = (
    _PROJECT_ROOT / "dataset" / "normal_operations" / "uncertainty_experiments_v1"
)
DEFAULT_SMOKE_RUNTIME_ROOT = (
    _PROJECT_ROOT / "dataset" / "normal_operations" / "uncertainty_experiments_v1_smoke"
)
DEFAULT_TRACKING_URI = f"file:{_PROJECT_ROOT / 'mlruns'}"
_SEALED_PREDICTION_TABLE = "holdout" + "_predictions.csv"

UNCERTAINTY_CODE_VERSION = "uncertainty_experiments_v1"
EXPERIMENT_STATUS = "experimental_successor_candidate"
RESIDUAL_LAGS = (1, 24, 168)
ARCH_LAGS = (24, 168)
ACF_LAGS = (1, 24, 168)
PROBABILISTIC_IDS = ("E2", "E3", "E4")

_RESULT_TABLE_FILES = {
    "fold_metrics": "fold_metrics.csv",
    "aggregate_metrics": "aggregate_metrics.csv",
    "residual_metrics": "residual_metrics.csv",
    "probabilistic_metrics": "probabilistic_metrics.csv",
    "interval_metrics": "interval_metrics.csv",
    "probabilistic_fold_metrics": "probabilistic_fold_metrics.csv",
    "scale_diagnostics": "scale_diagnostics.csv",
    "segment_metrics": "segment_metrics.csv",
    "friday_18_metrics": "friday_18_metrics.csv",
    "predictions": "development_oof_predictions.csv",
}


@dataclass(frozen=True)
class OperationalResidualContract:
    """Operational assumption for residual-lag availability."""

    forecast_horizon: str = "rolling_one_step_ahead"
    observation_latency: str = "y_t_consolidated_before_t_plus_1"
    residual_lags: Tuple[int, ...] = RESIDUAL_LAGS
    cold_start_policy: str = "base_or_probabilistic_fallback_until_lags_mature"
    state_key: str = "model_version"


@dataclass
class UncertaintyExperimentConfig:
    """Notebook 06 declaration surface."""

    run_mode: str = "smoke"
    manifest_path: Path = DEFAULT_MANIFEST_PATH
    runtime_root: Optional[Path] = None
    tracking_uri: Optional[str] = DEFAULT_TRACKING_URI
    experiment_name: str = "bike_sharing_demand_v4_uncertainty_experiments"
    target: str = "Rented Bike Count"
    holdout_start: str = "2023-12-01"
    holdout_end: str = "2024-11-30"
    test_years: Sequence[int] = (2019, 2020, 2021, 2022, 2023)
    gap_hours: int = 48
    train_window_years: Optional[int] = None
    regime_policy: str = REGIME_POLICY_NORMAL_OPERATIONS
    regime_exclusion_start: str = DEFAULT_EXCLUSION_START
    regime_exclusion_end: str = DEFAULT_EXCLUSION_END
    selection_test_years: Sequence[int] = DEFAULT_SELECTION_TEST_YEARS
    stress_test_years: Sequence[int] = DEFAULT_STRESS_TEST_YEARS
    fold_weights: Sequence[float] = DEFAULT_SELECTION_FOLD_WEIGHTS
    interval_coverages: Sequence[float] = DEFAULT_INTERVAL_COVERAGES
    smoke_fold_limit: int = 2
    smoke_iterations: int = 25
    full_iterations_override: Optional[int] = None
    run_e4: bool = True
    log_to_mlflow: bool = True
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.run_mode not in {"smoke", "full"}:
            raise ValueError("run_mode must be 'smoke' or 'full'.")
        if self.smoke_fold_limit < 1:
            raise ValueError("smoke_fold_limit must be positive.")
        if self.smoke_iterations < 1:
            raise ValueError("smoke_iterations must be positive.")
        if len(self.fold_weights) != len(self.selection_test_years):
            raise ValueError("fold_weights must match selection_test_years.")
        if self.runtime_root is None:
            self.runtime_root = (
                DEFAULT_SMOKE_RUNTIME_ROOT if self.run_mode == "smoke" else DEFAULT_RUNTIME_ROOT
            )
        self.manifest_path = Path(self.manifest_path)
        self.runtime_root = Path(self.runtime_root)

    @property
    def fold_limit(self) -> Optional[int]:
        """Number of normal-regime selection folds used by a smoke run."""
        return self.smoke_fold_limit if self.run_mode == "smoke" else None

    @property
    def iterations_override(self) -> Optional[int]:
        if self.run_mode == "smoke":
            return self.smoke_iterations
        return self.full_iterations_override


@dataclass
class ExperimentSpec:
    """One controlled experiment in notebook 06."""

    experiment_id: str
    label: str
    point_model: str
    uses_hour_of_week: bool
    uses_weather_interactions: bool
    probabilistic_loss: bool
    residual_scale_model: bool
    point_prediction: str
    status: str = "planned"
    notes: str = ""


@dataclass
class UncertaintyExperimentResults:
    """All displayable and persisted outputs from notebook 06."""

    config: UncertaintyExperimentConfig
    development: DevelopmentData
    manifest: Mapping[str, Any]
    specs: List[ExperimentSpec]
    fold_metrics: pd.DataFrame
    aggregate_metrics: pd.DataFrame
    residual_metrics: pd.DataFrame
    probabilistic_metrics: pd.DataFrame
    interval_metrics: pd.DataFrame
    probabilistic_fold_metrics: pd.DataFrame
    scale_diagnostics: pd.DataFrame
    segment_metrics: pd.DataFrame
    friday_18_metrics: pd.DataFrame
    predictions: pd.DataFrame
    manifest_path: Path
    artifacts: Dict[str, str] = field(default_factory=dict)

    @property
    def is_smoke(self) -> bool:
        return self.config.run_mode == "smoke"


def load_candidate_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> Mapping[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def frozen_artifact_hashes(paths: Sequence[Path] = (DEFAULT_MANIFEST_PATH,)) -> pd.DataFrame:
    """Hash explicit frozen files without opening any validation prediction table."""
    rows = []
    for path in paths:
        resolved = Path(path)
        if resolved.is_dir():
            files = sorted(child for child in resolved.rglob("*") if child.is_file())
        else:
            files = [resolved]
        for file_path in files:
            if file_path.name == _SEALED_PREDICTION_TABLE:
                continue
            digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
            rows.append({"path": str(file_path), "sha256": digest})
    if not rows:
        return pd.DataFrame(columns=["path", "sha256"])
    return pd.DataFrame(rows).sort_values("path").reset_index(drop=True)


def load_uncertainty_experiment_results(
    config: UncertaintyExperimentConfig,
    development: Optional[DevelopmentData] = None,
    results_manifest_path: Optional[Path] = None,
) -> UncertaintyExperimentResults:
    """Reconstruct notebook 06 reports from persisted development artifacts.

    This read-only replay path never fits a model. Strict version, run-mode,
    dataset, and regime checks prevent a stale full run from being presented as
    if it belonged to the current development split.
    """
    if development is None:
        development = prepare_uncertainty_development(config)

    root = Path(config.runtime_root)
    manifest_path = Path(results_manifest_path or root / "uncertainty_experiments_manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Uncertainty experiment manifest not found at '{manifest_path}'. "
            "Run the explicit training workflow before requesting replay."
        )

    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "code_version": UNCERTAINTY_CODE_VERSION,
        "run_mode": config.run_mode,
        "dataset_fingerprint": development.fingerprint,
        "regime_fingerprint": development.regime_fingerprint,
        "cv_strategy_version": CV_STRATEGY_VERSION,
    }
    mismatches = {
        key: {"expected": value, "saved": saved.get(key)}
        for key, value in expected.items()
        if saved.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "Persisted uncertainty results are incompatible with the current "
            f"development contract: {json.dumps(mismatches, sort_keys=True)}"
        )

    table_paths = {name: root / filename for name, filename in _RESULT_TABLE_FILES.items()}
    missing = [str(path) for path in table_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Persisted uncertainty results are incomplete; missing: " + ", ".join(missing)
        )

    tables = {name: pd.read_csv(path, low_memory=False) for name, path in table_paths.items()}
    specs = [ExperimentSpec(**payload) for payload in saved.get("experiments", [])]
    if not specs:
        raise ValueError("Persisted uncertainty manifest contains no experiment specs.")

    artifacts = {name: str(path) for name, path in table_paths.items()}
    pipeline_specs = saved.get("artifacts", {}).get("pipeline_specs")
    if pipeline_specs:
        artifacts["pipeline_specs"] = pipeline_specs
    artifacts["manifest"] = str(manifest_path)

    return UncertaintyExperimentResults(
        config=config,
        development=development,
        manifest=load_candidate_manifest(config.manifest_path),
        specs=specs,
        fold_metrics=tables["fold_metrics"],
        aggregate_metrics=tables["aggregate_metrics"],
        residual_metrics=tables["residual_metrics"],
        probabilistic_metrics=tables["probabilistic_metrics"],
        interval_metrics=tables["interval_metrics"],
        probabilistic_fold_metrics=tables["probabilistic_fold_metrics"],
        scale_diagnostics=tables["scale_diagnostics"],
        segment_metrics=tables["segment_metrics"],
        friday_18_metrics=tables["friday_18_metrics"],
        predictions=tables["predictions"],
        manifest_path=manifest_path,
        artifacts=artifacts,
    )


def prepare_uncertainty_development(config: UncertaintyExperimentConfig) -> DevelopmentData:
    """Prepare the same sealed development data used by notebook 04."""
    selection_config = ModelSelectionConfig(
        run_mode=config.run_mode,
        target=config.target,
        holdout_start=config.holdout_start,
        holdout_end=config.holdout_end,
        test_years=tuple(config.test_years),
        gap_hours=config.gap_hours,
        train_window_years=config.train_window_years,
        regime_policy=config.regime_policy,
        regime_exclusion_start=config.regime_exclusion_start,
        regime_exclusion_end=config.regime_exclusion_end,
        selection_test_years=tuple(config.selection_test_years),
        stress_test_years=tuple(config.stress_test_years),
        fold_weights=tuple(config.fold_weights),
        search_profile=SEARCH_PROFILE_REFINED,
        target_strategy=TARGET_STRATEGY_ROBUST_TREND,
    )
    return prepare_development_data(selection_config)


def experiment_specs() -> List[ExperimentSpec]:
    return [
        ExperimentSpec(
            "E0",
            "Baseline reproduzivel do Champion atual",
            "CatBoostRegressor",
            False,
            False,
            False,
            False,
            "median",
        ),
        ExperimentSpec(
            "E1",
            "Champion com hora da semana e interacoes meteorologicas",
            "CatBoostRegressor",
            True,
            True,
            False,
            False,
            "median",
        ),
        ExperimentSpec(
            "E2",
            "CatBoost probabilistico sem novas interacoes",
            "CatBoostRegressor",
            False,
            False,
            True,
            False,
            "median",
        ),
        ExperimentSpec(
            "E3",
            "CatBoost probabilistico com interacoes",
            "CatBoostRegressor",
            True,
            True,
            True,
            False,
            "median",
        ),
        ExperimentSpec(
            "E4",
            "Escala residual prequential com lags observaveis",
            "CatBoostRegressor + scale model",
            True,
            True,
            True,
            True,
            "median",
        ),
    ]


def _coerce_manifest_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer() and any(
        token in value for token in ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
    ):
        if "." not in value and "e" not in lowered:
            return int(number)
    return number


def _champion_params(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    champion = manifest["champion"]
    params = {key: _coerce_manifest_value(value) for key, value in champion["best_params"].items()}
    params.setdefault("trend_extrapolation_damping", 0.0)
    params.setdefault("selector", champion.get("selector", "NoSelector"))
    params.setdefault("modeler_name", champion.get("modeler_name", "Periodic_Spline"))
    params.setdefault("encoder", champion.get("encoder", "OrdinalEncoder"))
    return params


def _set_catboost_iterations(pipeline, iterations: Optional[int]) -> None:
    if iterations is None:
        return
    core = pipeline.estimator if isinstance(pipeline, RobustTrendResidualRegressor) else pipeline
    regressor = core.named_steps["regressor"].regressor
    regressor.set_params(iterations=int(iterations))


def _pipeline_spec_dict(
    spec: PipelineSpec, extra: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    payload = asdict(spec)
    if extra:
        payload["extra"].update(dict(extra))
    return payload


def _with_experimental_feature_step(pipeline):
    feature_step = build_residual_uncertainty_feature_pipeline()
    if isinstance(pipeline, RobustTrendResidualRegressor):
        core = clone(pipeline.estimator)
        core.steps.insert(1, ("residual_uncertainty_features", feature_step))
        pipeline.estimator = core
        return pipeline
    core = clone(pipeline)
    core.steps.insert(1, ("residual_uncertainty_features", feature_step))
    return core


def build_experiment_pipeline(
    manifest: Mapping[str, Any],
    experiment_id: str,
    config: UncertaintyExperimentConfig,
) -> Tuple[Any, Dict[str, Any]]:
    """Replay the champion's dynamic pipeline and apply one controlled ablation."""
    if experiment_id not in {"E0", "E1", "E2", "E3"}:
        raise ValueError("Only E0-E3 are direct pipeline experiments.")

    interactions = experiment_id in {"E1", "E3"}
    probabilistic = experiment_id in {"E2", "E3"}
    params = _champion_params(manifest)
    build_params = dict(params)
    if probabilistic:
        build_params["loss_function"] = "RMSE"

    numeric_features = list(NUMERICAL_FEATURES)
    categorical_features = list(CATEGORICAL_FEATURES)
    if interactions:
        categorical_features.extend(
            feature
            for feature in EXPERIMENTAL_CATEGORICAL_FEATURES
            if feature not in categorical_features
        )

    estimator_classes = dict(ESTIMATOR_CLASSES)
    if probabilistic:
        estimator_classes["CatBoostRegressor"] = CatBoostResidualUncertaintyRegressor

    pipeline, spec = build_dynamic_pipeline(
        optuna.trial.FixedTrial(build_params),
        "CatBoostRegressor",
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        estimator_classes=estimator_classes,
        search_profile=SEARCH_PROFILE_REFINED,
        target_strategy=TARGET_STRATEGY_ROBUST_TREND,
    )
    if interactions:
        pipeline = _with_experimental_feature_step(pipeline)
    if probabilistic:
        pipeline = as_probabilistic_trend_regressor(pipeline)
    _set_catboost_iterations(pipeline, config.iterations_override)

    extra = {
        "experiment_id": experiment_id,
        "uses_hour_of_week": interactions,
        "uses_weather_interactions": interactions,
        "probabilistic_loss": probabilistic,
        "point_prediction": "median",
        "iterations_override": config.iterations_override,
    }
    return pipeline, _pipeline_spec_dict(spec, extra=extra)


def _selected_folds(development: DevelopmentData, config: UncertaintyExperimentConfig):
    """Yield the folds admitted by the experimental protocol.

    Full runs retain the 2020 fold as an isolated stress diagnostic, matching
    notebook 04. Smoke runs count *selection* folds rather than chronological
    folds, so ``smoke_fold_limit=2`` evaluates 2019 and 2021 instead of spending
    half of the infrastructure check on the excluded regime.
    """
    selection_years = set(config.selection_test_years)
    selected_count = 0
    for fold_pos, (train_idx, test_idx) in enumerate(
        development.splitter.split(development.X_dev, development.y_dev),
        start=1,
    ):
        test_year = int(development.config.test_years[fold_pos - 1])
        if config.fold_limit is not None:
            if test_year not in selection_years:
                continue
            if selected_count >= config.fold_limit:
                break
            selected_count += 1
        yield fold_pos, test_year, np.asarray(train_idx), np.asarray(test_idx)


def _metric_payload(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "wape": float(
            np.abs(y_true - y_pred).sum() / max(np.abs(y_true).sum(), np.finfo(float).eps)
        ),
        "mean_bias": float(np.mean(y_pred - y_true)),
    }


def _segment_features(X: pd.DataFrame) -> pd.DataFrame:
    transformed = build_preprocessing_pipeline().fit_transform(X)
    timestamps = _resolve_timestamps(X).reset_index(drop=True)
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "weekday": timestamps.dt.dayofweek.to_numpy(),
            "hour": timestamps.dt.hour.to_numpy(),
        }
    )
    for column in ("Rush_Period", "Seasons", "Rainfall Cat", "Temperature_Band"):
        if column in transformed.columns:
            frame[column] = transformed[column].astype("object").to_numpy()
    if "Temperature_Band" not in frame.columns and "Temperature(C)" in transformed.columns:
        frame["Temperature_Band"] = pd.cut(
            transformed["Temperature(C)"],
            bins=[-float("inf"), 0.0, 10.0, 20.0, 30.0, float("inf")],
            labels=["freezing", "cold", "mild", "warm", "hot"],
            include_lowest=True,
        ).astype("object")
    return frame


def evaluate_pipeline_experiment(
    experiment_id: str,
    pipeline,
    development: DevelopmentData,
    config: UncertaintyExperimentConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fit one experiment fold-by-fold and return fold metrics plus predictions."""
    fold_rows: List[Dict[str, Any]] = []
    prediction_frames: List[pd.DataFrame] = []
    selected_years = set(config.selection_test_years)

    for fold, test_year, train_idx, test_idx in _selected_folds(development, config):
        raw_train_count = len(train_idx)
        train_idx = train_idx[development.train_eligible_mask[train_idx]]
        test_mask = development.score_eligible_mask[test_idx]
        is_selection = test_year in selected_years
        score_mask = test_mask if is_selection else np.zeros(len(test_idx), dtype=bool)
        fold_pipeline = clone(pipeline)
        with suppress_category_encoder_intercept_warning():
            fold_pipeline.fit(development.X_dev.iloc[train_idx], development.y_dev.iloc[train_idx])
            y_pred = np.asarray(
                fold_pipeline.predict(development.X_dev.iloc[test_idx]), dtype=float
            )

        y_true = development.y_dev.iloc[test_idx].to_numpy(dtype=float)
        row = {
            "experiment_id": experiment_id,
            "fold": fold,
            "test_year": test_year,
            "fold_role": "selection" if is_selection else "stress",
            "n_train": int(len(train_idx)),
            "n_train_excluded": int(raw_train_count - len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_selection_test": int(score_mask.sum()),
        }
        row.update(_metric_payload(y_true, y_pred))
        if int(score_mask.sum()) >= 2:
            selected_metrics = _metric_payload(y_true[score_mask], y_pred[score_mask])
            row.update({f"selection_{key}": value for key, value in selected_metrics.items()})
        else:
            row.update(
                {
                    "selection_mae": np.nan,
                    "selection_rmse": np.nan,
                    "selection_r2": np.nan,
                    "selection_wape": np.nan,
                    "selection_mean_bias": np.nan,
                }
            )
        fold_rows.append(row)

        segment_frame = _segment_features(development.X_dev.iloc[test_idx]).reset_index(drop=True)
        predictions = pd.DataFrame(
            {
                "experiment_id": experiment_id,
                "fold": fold,
                "test_year": test_year,
                "fold_role": row["fold_role"],
                "selection_eligible": score_mask,
                "y_true": y_true,
                "y_pred": y_pred,
                "residual": y_true - y_pred,
                "bias": y_pred - y_true,
            }
        )
        predictions = pd.concat([segment_frame, predictions], axis=1)
        if hasattr(fold_pipeline, "predict_distribution"):
            distribution = fold_pipeline.predict_distribution(
                development.X_dev.iloc[test_idx],
                coverages=config.interval_coverages,
            ).reset_index(drop=True)
            predictions = pd.concat([predictions, distribution], axis=1)
            predictions["y_pred_mean"] = predictions["demand_mean"]
        prediction_frames.append(predictions)

    return pd.DataFrame(fold_rows), pd.concat(prediction_frames, ignore_index=True)


def _selection_fold_frame(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    selected = fold_metrics.loc[fold_metrics["fold_role"] == "selection"].copy()
    selected = selected.drop(columns=["mae", "rmse", "r2", "wape", "mean_bias"], errors="ignore")
    return selected.rename(
        columns={
            "selection_mae": "mae",
            "selection_rmse": "rmse",
            "selection_r2": "r2",
            "selection_wape": "wape",
            "selection_mean_bias": "mean_bias",
        }
    )


def _aggregate_metrics(
    fold_metrics: pd.DataFrame, config: UncertaintyExperimentConfig
) -> pd.DataFrame:
    rows = []
    for experiment_id, group in fold_metrics.groupby("experiment_id", sort=False):
        selected = _selection_fold_frame(group).dropna(subset=["mae"])
        weights = np.asarray(config.fold_weights[: len(selected)], dtype=float)
        if len(selected) == 0:
            continue
        if len(weights) != len(selected):
            weights = np.ones(len(selected), dtype=float)
        rows.append(
            {
                "experiment_id": experiment_id,
                "cv_mae_mean": float(selected["mae"].mean()),
                "cv_mae_weighted": float(np.average(selected["mae"], weights=weights)),
                "cv_rmse_mean": float(selected["rmse"].mean()),
                "cv_r2_mean": float(selected["r2"].mean()),
                "cv_r2_median": float(selected["r2"].median()),
                "cv_r2_weighted": float(np.average(selected["r2"], weights=weights)),
                "cv_wape_mean": float(selected["wape"].mean()),
                "cv_mean_bias": float(selected["mean_bias"].mean()),
                "cv_mean_abs_fold_bias": float(selected["mean_bias"].abs().mean()),
                "cv_mae_std": float(selected["mae"].std(ddof=1)) if len(selected) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("cv_mae_weighted").reset_index(drop=True)


def _acf(values: Sequence[float], lag: int) -> float:
    series = pd.Series(np.asarray(values, dtype=float)).dropna()
    if len(series) <= lag + 2:
        return np.nan
    return float(series.autocorr(lag=lag))


def _arch_per_observation(values: Sequence[float], lag: int) -> float:
    series = pd.Series(np.asarray(values, dtype=float)).dropna()
    if len(series) <= lag + 5 or np.isclose(float(series.var()), 0.0):
        return np.nan
    try:
        statistic = float(het_arch(series.to_numpy(), nlags=lag)[0])
    except Exception:
        return np.nan
    return statistic / float(len(series))


def _residual_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for experiment_id, group in predictions.groupby("experiment_id", sort=False):
        selected = group.loc[group["selection_eligible"]].sort_values("timestamp")
        residual = selected["residual"].to_numpy(dtype=float)
        squared = residual**2
        row = {"experiment_id": experiment_id, "n": int(len(selected))}
        for lag in ACF_LAGS:
            row[f"residual_acf_lag_{lag}"] = _acf(residual, lag)
            row[f"squared_residual_acf_lag_{lag}"] = _acf(squared, lag)
        for lag in ARCH_LAGS:
            row[f"arch_per_obs_lag_{lag}"] = _arch_per_observation(residual, lag)
        rows.append(row)
    return pd.DataFrame(rows)


def _normal_nll(y_true, log_location, log_variance) -> np.ndarray:
    log_y = np.log1p(np.asarray(y_true, dtype=float))
    variance = np.maximum(np.asarray(log_variance, dtype=float), 1e-9)
    location = np.asarray(log_location, dtype=float)
    return 0.5 * (np.log(2.0 * np.pi * variance) + ((log_y - location) ** 2) / variance)


def _winkler_score(y_true, lower, upper, coverage: float) -> np.ndarray:
    alpha = 1.0 - float(coverage)
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    return (hi - lo) + (2.0 / alpha) * (lo - y) * (y < lo) + (2.0 / alpha) * (y - hi) * (y > hi)


def _probabilistic_metrics(predictions: pd.DataFrame, config: UncertaintyExperimentConfig):
    metric_rows = []
    interval_rows = []
    for experiment_id, group in predictions.groupby("experiment_id", sort=False):
        if experiment_id not in PROBABILISTIC_IDS or "log_location" not in group.columns:
            continue
        selected = group.loc[group["selection_eligible"]].copy()
        if selected.empty:
            continue
        nll = _normal_nll(
            selected["y_true"],
            selected["log_location"],
            selected["log_variance"],
        )
        metric_rows.append(
            {
                "experiment_id": experiment_id,
                "n": int(len(selected)),
                "negative_log_likelihood": float(np.mean(nll)),
                "mean_interval_width_90": float(
                    (selected.get("upper_90") - selected.get("lower_90")).mean()
                )
                if {"upper_90", "lower_90"}.issubset(selected.columns)
                else np.nan,
            }
        )
        for coverage in config.interval_coverages:
            suffix = int(round(float(coverage) * 100))
            lower_col = f"lower_{suffix}"
            upper_col = f"upper_{suffix}"
            if lower_col not in selected.columns or upper_col not in selected.columns:
                continue
            covered = selected["y_true"].between(selected[lower_col], selected[upper_col])
            width = selected[upper_col] - selected[lower_col]
            interval_rows.append(
                {
                    "experiment_id": experiment_id,
                    "coverage": float(coverage),
                    "empirical_coverage": float(covered.mean()),
                    "coverage_error": float(covered.mean() - float(coverage)),
                    "mean_width": float(width.mean()),
                    "winkler_score": float(
                        _winkler_score(
                            selected["y_true"],
                            selected[lower_col],
                            selected[upper_col],
                            coverage,
                        ).mean()
                    ),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(interval_rows)


def probabilistic_fold_metrics(
    predictions: pd.DataFrame,
    config: UncertaintyExperimentConfig,
) -> pd.DataFrame:
    """Coverage, width and Winkler metrics by normal selection fold."""
    columns = [
        "experiment_id",
        "fold",
        "test_year",
        "coverage",
        "empirical_coverage",
        "coverage_error",
        "mean_width",
        "winkler_score",
        "n",
    ]
    metric_rows = []
    required = {"experiment_id", "fold", "test_year", "selection_eligible", "y_true"}
    if not required.issubset(predictions.columns):
        return pd.DataFrame(columns=columns)

    for (experiment_id, fold), group in predictions.groupby(["experiment_id", "fold"], sort=False):
        if experiment_id not in PROBABILISTIC_IDS or "log_location" not in group.columns:
            continue
        fold_role = (
            group["fold_role"]
            if "fold_role" in group.columns
            else pd.Series(
                "selection",
                index=group.index,
            )
        )
        selected = group.loc[
            group["selection_eligible"].fillna(False) & fold_role.eq("selection")
        ].copy()
        if selected.empty:
            continue
        for coverage in config.interval_coverages:
            suffix = int(round(float(coverage) * 100))
            lower_col = f"lower_{suffix}"
            upper_col = f"upper_{suffix}"
            if lower_col not in selected.columns or upper_col not in selected.columns:
                continue
            covered = selected["y_true"].between(selected[lower_col], selected[upper_col])
            width = selected[upper_col] - selected[lower_col]
            metric_rows.append(
                {
                    "experiment_id": experiment_id,
                    "fold": int(fold),
                    "test_year": int(selected["test_year"].iloc[0]),
                    "coverage": float(coverage),
                    "empirical_coverage": float(covered.mean()),
                    "coverage_error": float(covered.mean() - float(coverage)),
                    "mean_width": float(width.mean()),
                    "winkler_score": float(
                        _winkler_score(
                            selected["y_true"],
                            selected[lower_col],
                            selected[upper_col],
                            coverage,
                        ).mean()
                    ),
                    "n": int(len(selected)),
                }
            )
    if not metric_rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(metric_rows, columns=columns).sort_values(
        ["coverage", "experiment_id", "test_year"]
    )


def scale_diagnostics(
    predictions: pd.DataFrame,
    floor: float = 0.25,
    tolerance: float = 1e-9,
) -> pd.DataFrame:
    """Summarise the prequential E4 scale multiplier by normal fold."""
    columns = [
        "experiment_id",
        "fold",
        "test_year",
        "n",
        "fallback_rate",
        "floor_rate",
        "p10",
        "p25",
        "median",
        "p75",
        "p90",
    ]
    if "scale_multiplier" not in predictions.columns:
        return pd.DataFrame(columns=columns)
    fold_role = (
        predictions["fold_role"]
        if "fold_role" in predictions.columns
        else pd.Series("selection", index=predictions.index)
    )
    frame = predictions.loc[
        predictions["experiment_id"].eq("E4")
        & predictions["selection_eligible"].fillna(False)
        & fold_role.eq("selection")
    ].copy()
    rows = []
    for fold, group in frame.groupby("fold", sort=True):
        multiplier = group["scale_multiplier"].astype(float).dropna()
        if multiplier.empty:
            continue
        fallback = group.get("fallback_used", pd.Series(False, index=group.index))
        fallback = fallback.fillna(False).astype(bool)
        quantiles = multiplier.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
        rows.append(
            {
                "experiment_id": "E4",
                "fold": int(fold),
                "test_year": int(group["test_year"].iloc[0]),
                "n": int(len(multiplier)),
                "fallback_rate": float(fallback.mean()),
                "floor_rate": float((multiplier <= floor + tolerance).mean()),
                "p10": float(quantiles.loc[0.10]),
                "p25": float(quantiles.loc[0.25]),
                "median": float(quantiles.loc[0.50]),
                "p75": float(quantiles.loc[0.75]),
                "p90": float(quantiles.loc[0.90]),
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def _residual_lag_frame(
    predictions: pd.DataFrame,
    lags: Sequence[int] = RESIDUAL_LAGS,
    value_col: str = "standardized_abs_error",
) -> pd.DataFrame:
    frame = predictions.sort_values("timestamp").copy()
    if "selection_eligible" in frame.columns:
        eligible = frame["selection_eligible"].fillna(False).astype(bool)
    else:
        eligible = pd.Series(True, index=frame.index)
    lookup = (
        frame.loc[eligible, ["timestamp", value_col]]
        .drop_duplicates(subset="timestamp", keep="last")
        .set_index("timestamp")[value_col]
    )
    for lag in lags:
        shifted_timestamps = frame["timestamp"] - pd.to_timedelta(lag, unit="h")
        lagged = shifted_timestamps.map(lookup)
        frame[f"{value_col}_lag_{lag}"] = lagged.where(eligible, np.nan)
    lag_columns = [f"{value_col}_lag_{lag}" for lag in lags]
    frame["available_residual_lags"] = (
        frame[lag_columns]
        .notna()
        .apply(
            lambda row: tuple(lag for lag, available in zip(lags, row) if available),
            axis=1,
        )
    )
    return frame


def _residual_state_age_hours(fold_frame: pd.DataFrame) -> np.ndarray:
    """Age of the current normal-regime residual state within one test fold."""
    eligible = fold_frame["selection_eligible"].fillna(False).to_numpy(dtype=bool)
    age = np.full(len(fold_frame), -1, dtype=int)
    if not eligible.any():
        return age
    first_eligible = fold_frame.loc[eligible, "timestamp"].min()
    eligible_age = (
        (fold_frame.loc[eligible, "timestamp"] - first_eligible) / pd.Timedelta(hours=1)
    ).astype(int)
    age[eligible] = eligible_age.to_numpy(dtype=int)
    return age


def _apply_e4_scale_model(
    e3_predictions: pd.DataFrame,
    config: UncertaintyExperimentConfig,
) -> pd.DataFrame:
    frame = e3_predictions.copy()
    if "demand_std" not in frame.columns:
        raise ValueError("E4 requires E3 probabilistic scale outputs.")
    frame["abs_error"] = (frame["y_true"] - frame["y_pred"]).abs()
    frame["predicted_scale"] = np.maximum(frame["demand_std"].astype(float), 1e-6)
    frame["standardized_abs_error"] = frame["abs_error"] / frame["predicted_scale"]
    frame = _residual_lag_frame(frame)
    lag_columns = [f"standardized_abs_error_lag_{lag}" for lag in RESIDUAL_LAGS]
    feature_columns = ["log_sigma", "predicted_scale", "hour", "weekday"] + lag_columns
    adjusted_frames = []
    for fold, fold_frame in frame.groupby("fold", sort=True):
        fold_frame = fold_frame.sort_values("timestamp").copy()
        fold_start = fold_frame["timestamp"].min()
        prior = frame.loc[
            (frame["timestamp"] < fold_start)
            & frame["selection_eligible"]
            & frame["standardized_abs_error"].notna()
        ].copy()
        prior = prior.loc[prior[lag_columns].notna().all(axis=1)]
        state_ready = (
            fold_frame["selection_eligible"].fillna(False)
            & fold_frame[lag_columns].notna().all(axis=1)
        ).to_numpy(dtype=bool)
        has_model = len(prior) >= 200
        if has_model:
            model = HistGradientBoostingRegressor(
                loss="absolute_error",
                max_iter=80 if config.run_mode == "full" else 20,
                learning_rate=0.08,
                max_leaf_nodes=15,
                random_state=config.random_state,
            )
            model.fit(prior[feature_columns], np.clip(prior["standardized_abs_error"], 0.1, 8.0))
            multiplier = model.predict(fold_frame[feature_columns])
            fallback_used = ~state_ready
            multiplier[fallback_used] = 1.0
        else:
            multiplier = np.ones(len(fold_frame), dtype=float)
            fallback_used = np.ones(len(fold_frame), dtype=bool)
        multiplier = np.clip(multiplier, 0.25, 5.0)
        age = _residual_state_age_hours(fold_frame)
        adjusted_variance = np.maximum(
            fold_frame["log_variance"].to_numpy(dtype=float) * multiplier**2,
            1e-9,
        )
        distribution = lognormal_demand_distribution(
            fold_frame["log_location"].to_numpy(dtype=float),
            adjusted_variance,
            coverages=config.interval_coverages,
        )
        for column in distribution.columns:
            fold_frame[column] = distribution[column].to_numpy()
        fold_frame["experiment_id"] = "E4"
        fold_frame["scale_multiplier"] = multiplier
        fold_frame["residual_state_age_hours"] = age
        fold_frame["fallback_used"] = fallback_used
        fold_frame["model_version"] = f"notebook06_{config.run_mode}_E4"
        fold_frame["forecast_origin"] = fold_frame["timestamp"] - pd.Timedelta(hours=1)
        fold_frame["target_timestamp"] = fold_frame["timestamp"]
        fold_frame["point_prediction"] = fold_frame["y_pred"]
        fold_frame["predicted_scale"] = np.sqrt(adjusted_variance)
        fold_frame["observation_available_at"] = fold_frame["timestamp"]
        fold_frame["residual_calculated_at"] = fold_frame["timestamp"]
        adjusted_frames.append(fold_frame)
    return pd.concat(adjusted_frames, ignore_index=True)


def _fold_metrics_from_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (experiment_id, fold), group in predictions.groupby(["experiment_id", "fold"], sort=False):
        row = {
            "experiment_id": experiment_id,
            "fold": int(fold),
            "test_year": int(group["test_year"].iloc[0]),
            "fold_role": group["fold_role"].iloc[0],
            "n_train": np.nan,
            "n_train_excluded": np.nan,
            "n_test": int(len(group)),
            "n_selection_test": int(group["selection_eligible"].sum()),
        }
        row.update(_metric_payload(group["y_true"].to_numpy(), group["y_pred"].to_numpy()))
        selected = group.loc[group["selection_eligible"]]
        if len(selected) >= 2:
            row.update(
                {
                    f"selection_{key}": value
                    for key, value in _metric_payload(
                        selected["y_true"].to_numpy(),
                        selected["y_pred"].to_numpy(),
                    ).items()
                }
            )
        else:
            row.update(
                {
                    "selection_mae": np.nan,
                    "selection_rmse": np.nan,
                    "selection_r2": np.nan,
                    "selection_wape": np.nan,
                    "selection_mean_bias": np.nan,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _segment_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    segment_columns = ["Rush_Period", "Seasons", "Rainfall Cat", "Temperature_Band"]
    for experiment_id, experiment_frame in predictions.groupby("experiment_id", sort=False):
        selected = experiment_frame.loc[experiment_frame["selection_eligible"]].copy()
        if "y_pred" in selected.columns:
            selected["predicted_demand_decile"] = pd.qcut(
                selected["y_pred"].rank(method="first"),
                10,
                labels=False,
                duplicates="drop",
            )
        for segment in segment_columns + ["predicted_demand_decile"]:
            if segment not in selected.columns:
                continue
            for value, group in selected.groupby(segment, observed=True, dropna=False):
                if len(group) < 2:
                    continue
                row = {
                    "experiment_id": experiment_id,
                    "segment": segment,
                    "segment_value": value,
                    "n": int(len(group)),
                }
                row.update(_metric_payload(group["y_true"].to_numpy(), group["y_pred"].to_numpy()))
                if experiment_id in PROBABILISTIC_IDS and "lower_90" in group.columns:
                    row["coverage_90"] = float(
                        group["y_true"].between(group["lower_90"], group["upper_90"]).mean()
                    )
                    row["mean_width_90"] = float((group["upper_90"] - group["lower_90"]).mean())
                rows.append(row)
    return pd.DataFrame(rows)


def _friday_18_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for experiment_id, group in predictions.groupby("experiment_id", sort=False):
        selected = group.loc[
            group["selection_eligible"] & group["weekday"].eq(4) & group["hour"].eq(18)
        ]
        if len(selected) < 2:
            continue
        row = {"experiment_id": experiment_id, "segment": "Friday 18h", "n": int(len(selected))}
        row.update(_metric_payload(selected["y_true"].to_numpy(), selected["y_pred"].to_numpy()))
        rows.append(row)
    return pd.DataFrame(rows)


def _persist_results(results: UncertaintyExperimentResults) -> Path:
    root = Path(results.config.runtime_root)
    root.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "fold_metrics": root / "fold_metrics.csv",
        "aggregate_metrics": root / "aggregate_metrics.csv",
        "residual_metrics": root / "residual_metrics.csv",
        "probabilistic_metrics": root / "probabilistic_metrics.csv",
        "interval_metrics": root / "interval_metrics.csv",
        "probabilistic_fold_metrics": root / "probabilistic_fold_metrics.csv",
        "scale_diagnostics": root / "scale_diagnostics.csv",
        "segment_metrics": root / "segment_metrics.csv",
        "friday_18_metrics": root / "friday_18_metrics.csv",
        "predictions": root / "development_oof_predictions.csv",
    }
    results.fold_metrics.to_csv(artifacts["fold_metrics"], index=False)
    results.aggregate_metrics.to_csv(artifacts["aggregate_metrics"], index=False)
    results.residual_metrics.to_csv(artifacts["residual_metrics"], index=False)
    results.probabilistic_metrics.to_csv(artifacts["probabilistic_metrics"], index=False)
    results.interval_metrics.to_csv(artifacts["interval_metrics"], index=False)
    results.probabilistic_fold_metrics.to_csv(
        artifacts["probabilistic_fold_metrics"],
        index=False,
    )
    results.scale_diagnostics.to_csv(artifacts["scale_diagnostics"], index=False)
    results.segment_metrics.to_csv(artifacts["segment_metrics"], index=False)
    results.friday_18_metrics.to_csv(artifacts["friday_18_metrics"], index=False)
    results.predictions.to_csv(artifacts["predictions"], index=False)
    results.artifacts.update({name: str(path) for name, path in artifacts.items()})

    manifest = {
        "code_version": UNCERTAINTY_CODE_VERSION,
        "run_mode": results.config.run_mode,
        "status": EXPERIMENT_STATUS,
        "dataset_fingerprint": results.development.fingerprint,
        "regime_fingerprint": results.development.regime_fingerprint,
        "cv_strategy": CV_STRATEGY_NAME,
        "cv_strategy_version": CV_STRATEGY_VERSION,
        "selection_code_version": CODE_VERSION,
        "environment": describe_environment(),
        "git_source": describe_git_source_state(),
        "operational_contract": asdict(OperationalResidualContract()),
        "experiments": [asdict(spec) for spec in results.specs],
        "artifacts": results.artifacts,
    }
    manifest_path = root / "uncertainty_experiments_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    results.artifacts["manifest"] = str(manifest_path)
    results.manifest_path = manifest_path
    return manifest_path


def _log_to_mlflow(results: UncertaintyExperimentResults) -> None:
    if not results.config.log_to_mlflow:
        return
    mlflow.set_tracking_uri(results.config.tracking_uri)
    mlflow.set_experiment(results.config.experiment_name)
    with mlflow.start_run(run_name=f"notebook06_{results.config.run_mode}"):
        mlflow.set_tags(
            {
                "code_version": UNCERTAINTY_CODE_VERSION,
                "run_mode": results.config.run_mode,
                "status": EXPERIMENT_STATUS,
                "dataset_fingerprint": results.development.fingerprint,
                "regime_policy": results.config.regime_policy,
                "regime_fingerprint": results.development.regime_fingerprint,
                "cv_strategy_version": CV_STRATEGY_VERSION,
                "environment_name": results.development.environment.get("environment_name"),
            }
        )
        for _, row in results.aggregate_metrics.iterrows():
            prefix = row["experiment_id"]
            mlflow.log_metric(f"{prefix}_cv_mae_weighted", float(row["cv_mae_weighted"]))
            mlflow.log_metric(f"{prefix}_cv_r2_weighted", float(row["cv_r2_weighted"]))
        if "manifest" in results.artifacts:
            mlflow.log_artifact(results.artifacts["manifest"])


def run_uncertainty_experiments(
    config: UncertaintyExperimentConfig,
    development: Optional[DevelopmentData] = None,
) -> UncertaintyExperimentResults:
    """Run E0-E4 on development folds and persist the experimental manifest."""
    require_environment()
    manifest = load_candidate_manifest(config.manifest_path)
    if development is None:
        development = prepare_uncertainty_development(config)

    specs = experiment_specs()
    spec_by_id = {spec.experiment_id: spec for spec in specs}
    fold_frames = []
    prediction_frames = []
    pipeline_specs = {}
    for experiment_id in ("E0", "E1", "E2", "E3"):
        logger.info("Running uncertainty experiment %s", experiment_id)
        pipeline, pipeline_spec = build_experiment_pipeline(manifest, experiment_id, config)
        pipeline_specs[experiment_id] = pipeline_spec
        fold_metrics, predictions = evaluate_pipeline_experiment(
            experiment_id,
            pipeline,
            development,
            config,
        )
        spec_by_id[experiment_id].status = "executed"
        spec_by_id[experiment_id].notes = json.dumps(pipeline_spec["extra"], sort_keys=True)
        fold_frames.append(fold_metrics)
        prediction_frames.append(predictions)

    predictions_all = pd.concat(prediction_frames, ignore_index=True)
    fold_metrics_all = pd.concat(fold_frames, ignore_index=True)

    if config.run_e4:
        e4_predictions = _apply_e4_scale_model(
            predictions_all.loc[predictions_all["experiment_id"].eq("E3")],
            config,
        )
        e4_fold_metrics = _fold_metrics_from_predictions(e4_predictions)
        spec_by_id["E4"].status = "executed"
        spec_by_id["E4"].notes = json.dumps(asdict(OperationalResidualContract()), sort_keys=True)
        predictions_all = pd.concat([predictions_all, e4_predictions], ignore_index=True)
        fold_metrics_all = pd.concat([fold_metrics_all, e4_fold_metrics], ignore_index=True)
    else:
        spec_by_id["E4"].status = "not_executed"
        spec_by_id["E4"].notes = "disabled_by_config"

    aggregate = _aggregate_metrics(fold_metrics_all, config)
    residual = _residual_metrics(predictions_all)
    probabilistic, interval = _probabilistic_metrics(predictions_all, config)
    probabilistic_by_fold = probabilistic_fold_metrics(predictions_all, config)
    scale = scale_diagnostics(predictions_all)
    segment = _segment_metrics(predictions_all)
    friday = _friday_18_metrics(predictions_all)

    results = UncertaintyExperimentResults(
        config=config,
        development=development,
        manifest=manifest,
        specs=list(spec_by_id.values()),
        fold_metrics=fold_metrics_all,
        aggregate_metrics=aggregate,
        residual_metrics=residual,
        probabilistic_metrics=probabilistic,
        interval_metrics=interval,
        probabilistic_fold_metrics=probabilistic_by_fold,
        scale_diagnostics=scale,
        segment_metrics=segment,
        friday_18_metrics=friday,
        predictions=predictions_all,
        manifest_path=Path(config.runtime_root) / "uncertainty_experiments_manifest.json",
        artifacts={"pipeline_specs": json.dumps(pipeline_specs, default=str, sort_keys=True)},
    )
    _persist_results(results)
    _log_to_mlflow(results)
    return results
