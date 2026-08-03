"""Class-based MLflow integration for Bike Sharing Demand v3 (regression).

Complements src/tracking.py (procedural write primitives) with:
  - ExperimentConfig: dataclass for experiment-level settings.
  - RegressionModelMetrics: dataclass for 3-tier regression metrics (cv/holdout/loso).
  - MLflowTracker: stateful class for run logging, querying, and registration.
  - MLflowExperimentOrchestrator: multi-estimator experiment coordinator.

Design contract:
  - Write operations delegate to src.tracking (single source of truth for MLflow calls).
  - Read/query operations use mlflow.search_runs + MlflowClient directly here.
  - src/tracking.py is preserved intact; this module adds, never replaces.

Key difference from classification (Customer Churn):
  - 3-tier metrics: cv / holdout / loso  (vs. train/test)
  - Best model by ascending metric (lower RMSE = better, not higher recall)
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from src.tracking import log_estimator_run, setup_mlflow
from src.utils import MLRUNS_DIR

logger = logging.getLogger(__name__)


# ── Dataclasses ────────────────────────────────────────────────────────────────


@dataclass
class ExperimentConfig:
    """Configuration for an MLflow experiment (BSD regression defaults)."""

    project_name: str = "bike_sharing_demand_v3"
    experiment_name: str = "bike_sharing_demand_v3"
    description: str = (
        "Seoul Hourly Bike Demand — time-series regression with "
        "temporal holdout and Leave-One-Season-Out validation"
    )
    tags: Dict[str, str] = field(
        default_factory=lambda: {
            "project_type": "regression",
            "domain": "bike_sharing",
            "cv_strategy": "TimeSeriesSplit+LOSO",
            "measurement_era": "v3",
        }
    )
    tracking_uri: Optional[str] = None  # None → uses MLRUNS_DIR from utils.py

    def __post_init__(self) -> None:
        if not self.project_name:
            raise ValueError("project_name é obrigatório")
        if not self.experiment_name:
            raise ValueError("experiment_name é obrigatório")


@dataclass
class ExperimentConfigV4(ExperimentConfig):
    """MLflow experiment config for v4 temporal model selection.

    Selection is CV-only: ``selection_metric`` is ``cv_mae_mean``, and
    ``final_holdout`` is recorded purely as a tag (for traceability), never
    read back by any selection query.
    """

    project_name: str = "bike_sharing_demand_v4"
    experiment_name: str = "bike_sharing_demand_v4_model_selection"
    description: str = (
        "Seoul Hourly Bike Demand v4 (2015-2024) — expanding meteorological-year "
        "CV model selection; final holdout (2023-12-01/2024-11-30) sealed and "
        "never accessed in this experiment"
    )
    tags: Dict[str, str] = field(
        default_factory=lambda: {
            "project_type": "regression",
            "domain": "bike_sharing",
            "cv_strategy": "ForwardMeteorologicalYearSplit",
            "measurement_era": "v4",
            "final_holdout": "2023-12-01/2024-11-30",
            "selection_metric": "cv_mae_mean",
        }
    )


@dataclass
class RegressionModelMetrics:
    """Container for 3-tier regression metrics (cv / holdout / loso).

    Attributes
    ----------
    estimator_name:
        e.g. 'XGBRegressor'
    params:
        Optuna best params dict.
    cv_metrics:
        Inner CV metrics. Expected keys: 'r2_mean', 'rmse_mean', 'mae_mean'.
        log_estimator_run prefixes these with 'cv_' → stored as cv_r2_mean etc.
    holdout_metrics:
        Temporal holdout metrics. Expected keys: 'r2', 'rmse', 'mae'.
        log_estimator_run prefixes these with 'holdout_' → stored as holdout_r2 etc.
    loso_df:
        Optional LOSO DataFrame (cols: Season, MAE, RMSE, R2).
        Not included in to_dict() — passed through to log_estimator_run as-is,
        which iterates it and generates loso_{season}_{metric} entries.
    """

    estimator_name: str
    params: Dict[str, Any]
    cv_metrics: Dict[str, float]
    holdout_metrics: Dict[str, float]
    loso_df: Optional[pd.DataFrame] = None

    def to_dict(self) -> Dict[str, float]:
        """Return flat scalar dict with MLflow storage-side metric names.

        Keys match exactly what mlflow.search_runs() exposes (without 'metrics.' prefix):
          cv_r2_mean, cv_rmse_mean, cv_mae_mean,
          holdout_r2, holdout_rmse, holdout_mae

        loso_df is intentionally excluded — the write layer handles its iteration.
        """
        result: Dict[str, float] = {}
        for k, v in self.cv_metrics.items():
            result[f"cv_{k}"] = float(v)
        for k, v in self.holdout_metrics.items():
            result[f"holdout_{k}"] = float(v)
        return result


# ── MLflowTracker ──────────────────────────────────────────────────────────────


class MLflowTracker:
    """Stateful orchestrator for MLflow experiment runs (regression).

    Separates concerns:
      - Write path: delegates to src.tracking (atomic write primitives).
      - Read/query path: uses mlflow.search_runs + MlflowClient.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self._client: Optional[MlflowClient] = None
        logger.info("MLflowTracker inicializado para '%s'", config.project_name)

    @property
    def client(self) -> MlflowClient:
        """Lazy MlflowClient, initialized after setup_experiment() sets the URI."""
        if self._client is None:
            self._client = MlflowClient()
        return self._client

    def setup_experiment(self) -> str:
        """Configure MLflow URI, create/retrieve experiment, return experiment_id.

        Delegates URI and experiment creation to tracking.setup_mlflow() to avoid
        duplicating that logic. Then sets experiment-level description tag.

        ``self.config.experiment_name`` is passed through explicitly and
        actually activated — previously only ``tracking_uri`` was forwarded,
        so ``setup_mlflow`` always activated its own hardcoded v3 experiment
        and any custom ``experiment_name`` here was created but never made
        the active one, silently sending subsequent runs to v3.
        """
        setup_mlflow(self.config.tracking_uri, self.config.experiment_name)

        experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(self.config.experiment_name)
            experiment = mlflow.get_experiment(experiment_id)

        experiment_id = experiment.experiment_id

        try:
            self.client.set_experiment_tag(experiment_id, "description", self.config.description)
            for tag_key, tag_val in self.config.tags.items():
                self.client.set_experiment_tag(experiment_id, tag_key, tag_val)
        except Exception as exc:
            logger.warning("Could not set experiment tags: %s", exc)

        logger.info("Experiment '%s' ready (id=%s)", self.config.experiment_name, experiment_id)
        return experiment_id

    def log_model_run(
        self,
        metrics: RegressionModelMetrics,
        measurement_era: str = "v3",
        artifacts_dir: str = "./dataset",
        model_object: Optional[Any] = None,
        run_name: Optional[str] = None,
    ) -> str:
        """Log one estimator run to MLflow. Returns the run_id.

        Delegates entirely to log_estimator_run() from src.tracking — no direct
        mlflow calls here, keeping the write path in one place.

        Parameters
        ----------
        metrics:
            RegressionModelMetrics with cv, holdout, and optional loso data.
        measurement_era:
            Tag applied to the run (e.g. 'v3').
        artifacts_dir:
            Directory containing metric_dataframe.csv artifact.
        model_object:
            Optional fitted model. If provided, logs via mlflow.sklearn.log_model().
            Currently a no-op (future extension) — log_estimator_run does not yet
            log model artifacts. Included to keep the signature stable.
        run_name:
            Display name in the MLflow UI. Defaults to estimator_name when None.
            Use to label batches (e.g. '1st run', '2nd run post-tuning').
        """
        run_id = log_estimator_run(
            estimator_name=metrics.estimator_name,
            params=metrics.params,
            inner_cv_metrics=metrics.cv_metrics,
            holdout_metrics=metrics.holdout_metrics,
            loso_df=metrics.loso_df,
            artifacts_dir=artifacts_dir,
            measurement_era=measurement_era,
            run_name=run_name,
        )

        if model_object is not None:
            logger.warning(
                "model_object foi fornecido mas log_estimator_run não loga artefatos "
                "de modelo ainda. Use mlflow.sklearn.log_model() manualmente por ora."
            )

        return run_id

    def log_from_metric_dataframe(
        self,
        estimator_name: str,
        metric_dataframe: pd.DataFrame,
        run_name: Optional[str] = None,
    ) -> str:
        """Extract metrics for one estimator from metric_dataframe and log to MLflow.

        Designed for incremental logging: call immediately after each estimator
        finishes training, before the next one starts.

        Parameters
        ----------
        estimator_name:
            e.g. 'XGBRegressor'. Must match the Estimator level of metric_dataframe.
        metric_dataframe:
            MultiIndex DataFrame (Estimator, Optimization, Pre-Process Pipeline).
            Expected rows per estimator: 'Optuna-InnerCV' and 'Temporal-Holdout'.
        run_name:
            Display label in MLflow UI (e.g. '1st run'). Defaults to estimator_name.

        Returns
        -------
        str
            The MLflow run_id.
        """
        try:
            cv_row = metric_dataframe.loc[(estimator_name, "Optuna-InnerCV")].iloc[0]
            ho_row = metric_dataframe.loc[(estimator_name, "Temporal-Holdout")].iloc[0]
        except (KeyError, IndexError) as exc:
            available = metric_dataframe.index.get_level_values("Estimator").unique().tolist()
            raise KeyError(
                f"Estimator '{estimator_name}' not found in metric_dataframe. "
                f"Available: {available}"
            ) from exc

        cv_metrics = {
            "r2_mean": float(cv_row["Test R2"]),
            "rmse_mean": float(cv_row["Test RMSE"]),
            "mae_mean": float(cv_row["Test MAE"]),
        }
        holdout_metrics = {
            "r2": float(ho_row["Test R2"]),
            "rmse": float(ho_row["Test RMSE"]),
            "mae": float(ho_row["Test MAE"]),
        }
        try:
            params = ast.literal_eval(str(ho_row["Parameters"]))
        except Exception:
            params = {}

        metrics_obj = RegressionModelMetrics(
            estimator_name=estimator_name,
            params=params,
            cv_metrics=cv_metrics,
            holdout_metrics=holdout_metrics,
        )
        return self.log_model_run(metrics_obj, run_name=run_name)

    def get_best_model(
        self,
        metric_name: str = "holdout_rmse",
        ascending: bool = True,
    ) -> Dict[str, Any]:
        """Query experiment runs and return the best run as a dict.

        Parameters
        ----------
        metric_name:
            MLflow metric to rank by (storage name, e.g. 'holdout_rmse').
        ascending:
            True for error metrics (lower = better: RMSE, MAE).
            False for quality metrics (higher = better: R2).

        Returns
        -------
        dict with keys: run_id, estimator, measurement_era, metric_name,
        metric_value, all_metrics, all_tags, all_params
        """
        runs_df = mlflow.search_runs(
            experiment_names=[self.config.experiment_name],
        )

        # Exclude LOSO-only runs in pandas (MLflow's != filter also excludes
        # runs where the tag is absent, which would drop normal estimator runs)
        if "tags.evaluation" in runs_df.columns:
            runs_df = runs_df[
                runs_df["tags.evaluation"].isna() | (runs_df["tags.evaluation"] != "LOSO")
            ]

        metric_col = f"metrics.{metric_name}"
        if metric_col not in runs_df.columns:
            raise ValueError(
                f"Métrica '{metric_name}' não encontrada nas runs. "
                f"Colunas disponíveis: {[c for c in runs_df.columns if c.startswith('metrics.')]}"
            )

        valid = runs_df.dropna(subset=[metric_col])
        if valid.empty:
            raise RuntimeError(
                f"Nenhuma run com a métrica '{metric_name}' encontrada no experimento "
                f"'{self.config.experiment_name}'."
            )

        best_row = valid.sort_values(metric_col, ascending=ascending).iloc[0]

        metrics_cols = {
            col.replace("metrics.", ""): best_row[col]
            for col in best_row.index
            if col.startswith("metrics.") and pd.notna(best_row[col])
        }
        tags_cols = {
            col.replace("tags.", ""): best_row[col]
            for col in best_row.index
            if col.startswith("tags.") and pd.notna(best_row[col])
        }
        params_cols = {
            col.replace("params.", ""): best_row[col]
            for col in best_row.index
            if col.startswith("params.") and pd.notna(best_row[col])
        }

        return {
            "run_id": best_row["run_id"],
            "estimator": tags_cols.get("estimator", "unknown"),
            "measurement_era": tags_cols.get("measurement_era", "unknown"),
            "metric_name": metric_name,
            "metric_value": best_row[metric_col],
            "all_metrics": metrics_cols,
            "all_tags": tags_cols,
            "all_params": params_cols,
        }

    def get_best_model_by_cv(
        self,
        dataset_fingerprint: str,
        metric_name: str = "cv_mae_mean",
        cv_strategy: str = "ForwardMeteorologicalYearSplit",
        run_mode: str = "full",
        cv_strategy_version: Optional[str] = None,
        code_version: Optional[str] = None,
        environment_name: Optional[str] = None,
        environment_fingerprint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Select the best v4 run using only CV metrics — never holdout.

        This is the auxiliary read API; the authoritative selection lives in
        :func:`src.temporal_optimizer.select_champion_and_challengers`. It
        applies the *same* fail-closed filter, because an auxiliary query that
        is merely permissive is not a convenience but a second, weaker way to
        arrive at the wrong run: an earlier version filtered only on
        ``cv_strategy`` and an optional fingerprint, so it would happily return
        a smoke run, a run from a superseded code version, or a run produced
        under a different Python environment.

        Every provenance tag is matched exactly — run mode, code version,
        environment name and fingerprint, dataset fingerprint, CV strategy and
        its version — and the model must be logged and verified. A tag that is
        absent from the whole experiment raises rather than being skipped:
        silently dropping a filter is exactly the failure mode this guards
        against. ``dataset_fingerprint`` is required for the same reason.
        """
        from src.environment import ENVIRONMENT_NAME
        from src.environment import environment_fingerprint as current_environment_fingerprint
        from src.temporal_optimizer import CODE_VERSION, CV_STRATEGY_VERSION

        if "holdout" in metric_name.lower():
            raise ValueError(
                f"get_best_model_by_cv must not select by a holdout metric (got '{metric_name}')."
            )

        runs_df = mlflow.search_runs(experiment_names=[self.config.experiment_name])
        if runs_df.empty:
            raise RuntimeError(
                f"Nenhuma run encontrada no experimento '{self.config.experiment_name}'."
            )

        expected = {
            "cv_strategy": cv_strategy,
            "cv_strategy_version": cv_strategy_version or CV_STRATEGY_VERSION,
            "code_version": code_version or CODE_VERSION,
            "dataset_fingerprint": dataset_fingerprint,
            "run_mode": run_mode,
            "environment_name": environment_name or ENVIRONMENT_NAME,
            "environment_fingerprint": (
                environment_fingerprint or current_environment_fingerprint()
            ),
            "model_logged": "true",
            "model_artifact_verified": "true",
        }
        missing = [tag for tag in expected if f"tags.{tag}" not in runs_df.columns]
        if missing:
            raise ValueError(
                f"O experimento '{self.config.experiment_name}' não possui nenhuma run com "
                f"a(s) tag(s) de proveniência {missing}. A consulta é fail-closed: registre "
                "as runs com src.tracking.log_temporal_model_run antes de consultá-las."
            )

        eligible = runs_df.dropna(subset=[f"tags.{tag}" for tag in expected])
        for tag, value in expected.items():
            eligible = eligible[eligible[f"tags.{tag}"] == value]
        if "status" in eligible.columns:
            eligible = eligible[eligible["status"] == "FINISHED"]

        metric_col = f"metrics.{metric_name}"
        if metric_col not in eligible.columns:
            raise ValueError(
                f"Métrica '{metric_name}' não encontrada. Colunas disponíveis: "
                f"{[c for c in runs_df.columns if c.startswith('metrics.')]}"
            )

        valid = eligible.dropna(subset=[metric_col])
        if valid.empty:
            raise RuntimeError(
                f"Nenhuma run concluída em '{self.config.experiment_name}' satisfaz "
                f"{expected} e possui a métrica '{metric_name}'."
            )

        best_row = valid.sort_values(metric_col, ascending=True).iloc[0]
        return {
            "run_id": best_row["run_id"],
            "estimator": best_row.get("tags.estimator", "unknown"),
            "metric_name": metric_name,
            "metric_value": float(best_row[metric_col]),
            "run_mode": best_row.get("tags.run_mode"),
            "code_version": best_row.get("tags.code_version"),
            "environment_name": best_row.get("tags.environment_name"),
            "environment_fingerprint": best_row.get("tags.environment_fingerprint"),
            "dataset_fingerprint": best_row.get("tags.dataset_fingerprint"),
        }

    def get_experiment_summary(self) -> pd.DataFrame:
        """Return a clean summary DataFrame of all non-LOSO experiment runs.

        Columns: run_id, estimator, measurement_era,
                 cv_rmse, cv_r2,
                 holdout_rmse, holdout_r2, holdout_mae,
                 loso_mean_rmse, loso_mean_r2, loso_mean_mae

        Sorted by holdout_rmse ascending (best model first).
        Runs with evaluation='LOSO' are excluded.
        """
        runs_df = mlflow.search_runs(
            experiment_names=[self.config.experiment_name],
        )

        # Exclude LOSO-only runs in pandas (same reason as get_best_model)
        if "tags.evaluation" in runs_df.columns:
            runs_df = runs_df[
                runs_df["tags.evaluation"].isna() | (runs_df["tags.evaluation"] != "LOSO")
            ]

        if runs_df.empty:
            return pd.DataFrame()

        col_map = {
            "run_id": "run_id",
            "tags.estimator": "estimator",
            "tags.measurement_era": "measurement_era",
            "metrics.cv_rmse_mean": "cv_rmse",
            "metrics.cv_r2_mean": "cv_r2",
            "metrics.holdout_rmse": "holdout_rmse",
            "metrics.holdout_r2": "holdout_r2",
            "metrics.holdout_mae": "holdout_mae",
            "metrics.loso_mean_rmse": "loso_mean_rmse",
            "metrics.loso_mean_r2": "loso_mean_r2",
            "metrics.loso_mean_mae": "loso_mean_mae",
        }

        available = {k: v for k, v in col_map.items() if k in runs_df.columns}
        summary = runs_df[list(available.keys())].rename(columns=available)

        if "holdout_rmse" in summary.columns:
            summary = summary.sort_values("holdout_rmse", ascending=True)

        return summary.reset_index(drop=True)

    def register_best_model(
        self,
        model_name: str = "bike_sharing_demand_regressor",
        stage: str = "Staging",
        metric_name: str = "holdout_rmse",
    ) -> str:
        """Register the best model run in the MLflow Model Registry.

        Returns the run_id of the best model.

        Note: Full model artifact registration requires model_object to be passed
        in log_model_run(). Currently logs a warning and returns the run_id only,
        since tracking.py does not call mlflow.sklearn.log_model() yet.
        """
        best = self.get_best_model(metric_name=metric_name, ascending=True)
        run_id = best["run_id"]

        logger.warning(
            "register_best_model: artefato de modelo não disponível ainda (tracking.py "
            "não usa mlflow.sklearn.log_model). Passe model_object para log_model_run() "
            "para habilitar registro completo no Model Registry. run_id=%s",
            run_id,
        )
        return run_id


# ── MLflowExperimentOrchestrator ───────────────────────────────────────────────


class MLflowExperimentOrchestrator:
    """High-level coordinator for multi-estimator MLflow experiments.

    Bridges RegressionOptimizer results and MLflowTracker.
    Does not own the Optuna study — receives already-optimized results.
    """

    def __init__(self, tracker: MLflowTracker) -> None:
        self.tracker = tracker

    def run_full_experiment(
        self,
        estimator_configs: List[Dict[str, Any]],
        measurement_era: str = "v3",
        artifacts_dir: str = "./dataset",
        register_best: bool = False,
        model_name: str = "bike_sharing_demand_regressor",
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Log MLflow runs for multiple pre-optimized estimators.

        Parameters
        ----------
        estimator_configs:
            List of dicts, each with keys:
              - 'estimator_name': str
              - 'params': Dict[str, Any]
              - 'cv_metrics': Dict[str, float]
              - 'holdout_metrics': Dict[str, float]
              - 'loso_df': Optional[pd.DataFrame]
        measurement_era:
            Tag applied to all runs.
        artifacts_dir:
            Directory containing metric_dataframe.csv.
        register_best:
            If True, calls register_best_model() after all runs are logged.
        model_name:
            Registry model name (used only if register_best=True).

        Returns
        -------
        summary_df: pd.DataFrame
            Result of tracker.get_experiment_summary().
        run_ids: Dict[str, str]
            Mapping of estimator_name → run_id.
        """
        self.tracker.setup_experiment()

        run_ids: Dict[str, str] = {}

        for cfg in estimator_configs:
            metrics = RegressionModelMetrics(
                estimator_name=cfg["estimator_name"],
                params=cfg.get("params", {}),
                cv_metrics=cfg.get("cv_metrics", {}),
                holdout_metrics=cfg.get("holdout_metrics", {}),
                loso_df=cfg.get("loso_df"),
            )
            run_id = self.tracker.log_model_run(
                metrics=metrics,
                measurement_era=measurement_era,
                artifacts_dir=artifacts_dir,
            )
            run_ids[cfg["estimator_name"]] = run_id
            logger.info("Logged: %s → run_id=%s", cfg["estimator_name"], run_id)

        summary_df = self.tracker.get_experiment_summary()

        if register_best:
            self.tracker.register_best_model(model_name=model_name)

        return summary_df, run_ids
