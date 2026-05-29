"""
persistence.py
--------------
Model and metric persistence layer.

Design
~~~~~~
``ModelStore`` is an abstract interface with a single method ``save``.
``CsvModelStore`` implements it using gzip-pickle + CSV.
``MlflowModelStore`` implements it logging to an MLflow experiment.

Switching backends only requires changing the store constructor — no other
code changes are needed.
"""
from __future__ import annotations

import gzip
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .scoring import build_business_weight_config, compute_business_score


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class ModelStore(ABC):
    """Abstract base class for all model / metric storage backends."""

    @abstractmethod
    def save(
        self,
        description: str,
        data_model: str,
        encoder: str,
        pipeline_obj: Any,
        scores: dict[str, Any],
        params: dict[str, Any] | None,
        metric_df: pd.DataFrame | None = None,
        overwrite_existing: bool = True,
    ) -> pd.DataFrame:
        """
        Persist *pipeline_obj* and its evaluation scores.

        Parameters
        ----------
        description : str
            Estimator / experiment label.
        data_model : str
            Optimisation strategy description.
        encoder : str
            Pre-processing pipeline description.
        pipeline_obj : Any
            Fitted sklearn-compatible pipeline.
        scores : dict[str, Any]
            Cross-validation scores dictionary.
        params : dict[str, Any] | None
            Hyperparameter dictionary.
        metric_df : pd.DataFrame | None
            In-memory metrics DataFrame (used as fallback when no CSV exists).
        overwrite_existing : bool, default=True
            Replace an existing row that matches the same identity triple.

        Returns
        -------
        pd.DataFrame
            Updated metrics DataFrame.
        """

    @abstractmethod
    def load(self, description: str, data_model: str, encoder: str) -> Any:
        """
        Recover a previously saved pipeline by its identity triple.

        Parameters
        ----------
        description : str
            Estimator label used when the pipeline was saved.
        data_model : str
            Optimisation strategy label used when the pipeline was saved.
        encoder : str
            Pre-processing pipeline label used when the pipeline was saved.

        Returns
        -------
        Any
            The fitted sklearn-compatible pipeline.
        """

    @abstractmethod
    def best_run(self, metric: str = "test_recall") -> dict[str, str]:
        """
        Return the identity triple and metric value for the highest-scoring run.

        Parameters
        ----------
        metric : str, default="test_recall"
            Metric key to maximise.  Uses MLflow convention (lower-case,
            underscore-separated), e.g. ``"test_recall"``, ``"test_roc_auc"``,
            ``"business_score"``.

        Returns
        -------
        dict
            Keys: ``"description"``, ``"data_model"``, ``"encoder"``,
            and the requested *metric* key with its float value.
        """

    @abstractmethod
    def find_run(
        self,
        estimator: str | None = None,
        data_model: str | None = None,
        metric: str = "test_recall",
    ) -> dict[str, str]:
        """
        Find the best run matching optional tag filters.

        Parameters
        ----------
        estimator : str | None, default=None
            Filter by estimator name.  ``None`` means no filter.
        data_model : str | None, default=None
            Filter by optimisation strategy label.  ``None`` means no filter.
        metric : str, default="test_recall"
            Metric to maximise when selecting among matching runs.

        Returns
        -------
        dict
            Keys: ``"description"``, ``"data_model"``, ``"encoder"``,
            and the requested *metric* key with its float value.
        """

    @abstractmethod
    def load_metrics(self) -> pd.DataFrame:
        """
        Return all stored metrics as a DataFrame in the standard schema.

        The schema matches the ``metric_dataframe`` produced by
        :func:`~opt_binary_clf_pipe.train_all_models`, so existing ``.loc[...]`` filters
        work unchanged after a session restart.

        Returns
        -------
        pd.DataFrame
            All experiment results with columns matching ``_ALL_COLS``.
        """

    @abstractmethod
    def save_threshold_runs(
        self,
        df_metric: pd.DataFrame,
        thresholds: list[float],
        baseline_estimator: str = "LogisticRegression",
        baseline_encoder: str = "OrdinalEncoder",
        baseline_data_model: str = "Baseline",
        threshold_data_model: str = "Baseline + Threshold tuning",
        metric_df: pd.DataFrame | None = None,
        overwrite_existing: bool = True,
    ) -> pd.DataFrame:
        """
        Persist threshold-tuning experiments for a baseline estimator.

        Each threshold in *thresholds* is stored as a separate experiment run
        (named ``"<estimator> | <threshold_data_model> | <encoder> | thr=<t>"``).
        Training metrics and timing are inherited from the baseline row.
        No pipeline artefact is created — threshold tuning only shifts the
        decision boundary of the already-stored baseline model.

        Parameters
        ----------
        df_metric : pd.DataFrame
            Threshold-sweep results.  Required columns:
            ``Threshold``, ``Roc_auc``, ``Accuracy``,
            ``Precision_macro``, ``Recall_macro``, ``F1_macro``.
        thresholds : list[float]
            Subset of thresholds from *df_metric* to persist.
        baseline_estimator : str, default="LogisticRegression"
        baseline_encoder : str, default="OrdinalEncoder"
        baseline_data_model : str, default="Baseline"
        threshold_data_model : str, default="Baseline + Threshold tuning"
        metric_df : pd.DataFrame | None, default=None
            In-memory metrics DataFrame passed to the return value.
            When ``None`` the backend is queried.
        overwrite_existing : bool, default=True

        Returns
        -------
        pd.DataFrame
            Updated metrics DataFrame in the standard schema.
        """


# ---------------------------------------------------------------------------
# CSV + gzip-pickle backend
# ---------------------------------------------------------------------------

_INDEX_COLS = ["Estimator", "Optimization/Data model", "Pre-Process Pipeline"]

_METRIC_COLS = [
    "Train Roc auc",
    "Test Roc auc",
    "Train Balanced Accuracy",
    "Test Balanced Accuracy",
    "Train Recall",
    "Test Recall",
    "Train Precision",
    "Test Precision",
    "Train F1",
    "Test F1",
    "Parameters",
    "Fit Time",
    "Score Time",
    "Total Time",
    "Business Score",
    "Business Weights",
]

_ALL_COLS = _INDEX_COLS + _METRIC_COLS

# Maps MLflow-style metric keys → CSV column names
_METRIC_COL_MAP: dict[str, str] = {
    "test_recall":            "Test Recall",
    "test_roc_auc":           "Test Roc auc",
    "test_balanced_accuracy": "Test Balanced Accuracy",
    "test_precision":         "Test Precision",
    "test_f1":                "Test F1",
    "train_recall":           "Train Recall",
    "train_roc_auc":          "Train Roc auc",
    "business_score":         "Business Score",
}


class CsvModelStore(ModelStore):
    """
    Save metrics to a CSV file and serialize pipelines as gzip-pickle.

    Parameters
    ----------
    metric_path : str, default="./dataset/metric_dataframe.csv"
        Path to the CSV file that accumulates experiment results.
    model_dir : str, default="./models/"
        Directory where serialized pipeline files are written.
    w_recall : float, default=7
        Business-score recall weight.
    w_precision : float, default=3
        Business-score precision weight.
    w_time : float, default=0
        Business-score time-penalty weight.
    time_reference : float | None, default=1.0
        Reference time for time-penalty scaling.
    """

    def __init__(
        self,
        metric_path: str = "./dataset/metric_dataframe.csv",
        model_dir: str = "./models/",
        w_recall: float = 7,
        w_precision: float = 3,
        w_time: float = 0,
        time_reference: float | None = 1.0,
    ) -> None:
        self.metric_path = Path(metric_path)
        self.model_dir = Path(model_dir)
        self.w_recall = w_recall
        self.w_precision = w_precision
        self.w_time = w_time
        self.time_reference = time_reference

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        description: str,
        data_model: str,
        encoder: str,
        pipeline_obj: Any,
        scores: dict[str, Any],
        params: dict[str, Any] | None,
        metric_df: pd.DataFrame | None = None,
        overwrite_existing: bool = True,
    ) -> pd.DataFrame:
        """See :class:`ModelStore` for full parameter documentation."""
        metric_df = self._load_or_init(metric_df)
        row_identity, metric_df = self._resolve_identity(
            description, data_model, encoder, metric_df, overwrite_existing
        )
        row = self._build_row(row_identity, scores, params)
        metric_df = self._append_row(metric_df, row)
        self._flush_csv(metric_df)
        self._serialize_pipeline(pipeline_obj, description, data_model, row_identity)
        return metric_df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_or_init(self, metric_df: pd.DataFrame | None) -> pd.DataFrame:
        """Load CSV if it exists; otherwise use *metric_df* or create empty."""
        if self.metric_path.exists():
            return pd.read_csv(self.metric_path)

        if metric_df is None:
            df = pd.DataFrame(columns=_ALL_COLS)
        else:
            df = metric_df.copy()
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()

        missing = [c for c in _INDEX_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"metric_df is missing identity columns: {missing}")

        for col in _ALL_COLS:
            if col not in df.columns:
                df[col] = np.nan

        return df

    def _resolve_identity(
        self,
        description: str,
        data_model: str,
        encoder: str,
        metric_df: pd.DataFrame,
        overwrite_existing: bool,
    ) -> tuple[dict[str, str], pd.DataFrame]:
        """Build the identity triple, handling duplicates."""
        identity = {
            "Estimator": description,
            "Optimization/Data model": data_model,
            "Pre-Process Pipeline": encoder,
        }
        mask = self._identity_mask(metric_df, identity)

        if mask.any() and not overwrite_existing:
            suffix = 1
            while mask.any():
                identity["Pre-Process Pipeline"] = f"{encoder} | run={suffix}"
                mask = self._identity_mask(metric_df, identity)
                suffix += 1

        if mask.any():
            metric_df = metric_df.loc[~mask].copy()

        return identity, metric_df

    @staticmethod
    def _identity_mask(df: pd.DataFrame, identity: dict[str, str]) -> pd.Series:
        return (
            (df["Estimator"] == identity["Estimator"])
            & (df["Optimization/Data model"] == identity["Optimization/Data model"])
            & (df["Pre-Process Pipeline"] == identity["Pre-Process Pipeline"])
        )

    def _build_row(
        self,
        identity: dict[str, str],
        scores: dict[str, Any],
        params: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Compute all metric values and return a flat row dict."""
        from json import dumps

        try:
            params_str = dumps(params, default=str) if params is not None else ""
        except Exception:
            params_str = ""

        fit_time = float(np.mean(scores["fit_time"]))
        score_time = float(np.mean(scores["score_time"]))

        business_score = compute_business_score(
            recall=float(np.mean(scores["test_recall_macro"])),
            precision=float(np.mean(scores["test_precision_macro"])),
            fit_time=fit_time,
            score_time=score_time,
            w_recall=self.w_recall,
            w_precision=self.w_precision,
            w_time=self.w_time,
            time_reference=self.time_reference,
        )

        business_weights = build_business_weight_config(
            w_recall=self.w_recall,
            w_precision=self.w_precision,
            w_time=self.w_time,
            time_reference=self.time_reference,
        )

        return {
            **identity,
            "Train Roc auc": float(np.mean(scores["train_roc_auc"])),
            "Test Roc auc": float(np.mean(scores["test_roc_auc"])),
            "Train Balanced Accuracy": float(np.mean(scores["train_balanced_accuracy"])),
            "Test Balanced Accuracy": float(np.mean(scores["test_balanced_accuracy"])),
            "Train Recall": float(np.mean(scores["train_recall_macro"])),
            "Test Recall": float(np.mean(scores["test_recall_macro"])),
            "Train Precision": float(np.mean(scores["train_precision_macro"])),
            "Test Precision": float(np.mean(scores["test_precision_macro"])),
            "Train F1": float(np.mean(scores["train_f1_macro"])),
            "Test F1": float(np.mean(scores["test_f1_macro"])),
            "Parameters": params_str,
            "Fit Time": fit_time,
            "Score Time": score_time,
            "Total Time": fit_time + score_time,
            "Business Score": business_score,
            "Business Weights": business_weights,
        }

    @staticmethod
    def _append_row(metric_df: pd.DataFrame, row: dict) -> pd.DataFrame:
        new_row = pd.DataFrame([row])
        if metric_df.empty:
            result = new_row
        else:
            result = pd.concat([metric_df, new_row], ignore_index=True)
        result = result[_ALL_COLS]
        result.sort_values(_INDEX_COLS, inplace=True, ignore_index=True)
        return result

    def _flush_csv(self, metric_df: pd.DataFrame) -> None:
        self.metric_path.parent.mkdir(parents=True, exist_ok=True)
        metric_df.to_csv(self.metric_path, index=False)

    def _serialize_pipeline(
        self,
        pipeline_obj: Any,
        description: str,
        data_model: str,
        identity: dict[str, str],
    ) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)

        safe_encoder = (
            str(identity["Pre-Process Pipeline"])
            .replace(" ", "")
            .replace("/", "-")
            .replace("\\", "-")
            .replace("|", "-")
            .replace(":", "-")
        )
        filename = f"{description}_{data_model}_{safe_encoder}.pkl.gz"
        with gzip.open(self.model_dir / filename, "wb") as fh:
            pickle.dump(pipeline_obj, fh)

    # ------------------------------------------------------------------
    # Load API
    # ------------------------------------------------------------------

    def load(self, description: str, data_model: str, encoder: str) -> Any:
        """See :class:`ModelStore` for full parameter documentation."""
        safe_encoder = self._safe_part(encoder)
        model_path = self.model_dir / f"{description}_{data_model}_{safe_encoder}.pkl.gz"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Identity: {description!r} | {data_model!r} | {encoder!r}"
            )
        with gzip.open(model_path, "rb") as fh:
            return pickle.load(fh)

    def best_run(self, metric: str = "test_recall") -> dict[str, str]:
        """See :class:`ModelStore` for full parameter documentation."""
        col = _METRIC_COL_MAP.get(metric, metric)
        if not self.metric_path.exists():
            raise FileNotFoundError(f"Metric CSV not found: {self.metric_path}")
        df = pd.read_csv(self.metric_path)
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not found in metric CSV. Available: {list(df.columns)}")
        best = df.loc[df[col].idxmax()]
        return {
            "description": best["Estimator"],
            "data_model":  best["Optimization/Data model"],
            "encoder":     best["Pre-Process Pipeline"],
            metric:        float(best[col]),
        }

    def find_run(
        self,
        estimator: str | None = None,
        data_model: str | None = None,
        metric: str = "test_recall",
    ) -> dict[str, str]:
        """See :class:`ModelStore` for full parameter documentation."""
        col = _METRIC_COL_MAP.get(metric, metric)
        df = self.load_metrics()
        if estimator is not None:
            df = df[df["Estimator"] == estimator]
        if data_model is not None:
            df = df[df["Optimization/Data model"] == data_model]
        if df.empty:
            raise FileNotFoundError(
                f"No run found: estimator={estimator!r}, data_model={data_model!r}"
            )
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not in metrics. Available: {list(df.columns)}")
        best = df.loc[df[col].idxmax()]
        return {
            "description": best["Estimator"],
            "data_model":  best["Optimization/Data model"],
            "encoder":     best["Pre-Process Pipeline"],
            metric:        float(best[col]),
        }

    def load_metrics(self) -> pd.DataFrame:
        """See :class:`ModelStore` for full parameter documentation."""
        if self.metric_path.exists():
            return pd.read_csv(self.metric_path)
        return pd.DataFrame(columns=_ALL_COLS)

    def save_threshold_runs(
        self,
        df_metric: pd.DataFrame,
        thresholds: list[float],
        baseline_estimator: str = "LogisticRegression",
        baseline_encoder: str = "OrdinalEncoder",
        baseline_data_model: str = "Baseline",
        threshold_data_model: str = "Baseline + Threshold tuning",
        metric_df: pd.DataFrame | None = None,
        overwrite_existing: bool = True,
    ) -> pd.DataFrame:
        """See :class:`ModelStore` for full parameter documentation."""
        from json import dumps, loads

        base = self.load_metrics() if metric_df is None else metric_df.copy()
        if isinstance(base.index, pd.MultiIndex):
            base = base.reset_index()

        mask = (
            (base["Estimator"] == baseline_estimator)
            & (base["Optimization/Data model"] == baseline_data_model)
            & (base["Pre-Process Pipeline"] == baseline_encoder)
        )
        if not mask.any():
            raise KeyError(
                f"Baseline row not found in metrics: "
                f"({baseline_estimator!r}, {baseline_data_model!r}, {baseline_encoder!r})"
            )
        bl = base.loc[mask].iloc[0]
        fit_time = bl.get("Fit Time", np.nan)
        score_time = bl.get("Score Time", np.nan)
        total_time = bl.get("Total Time", np.nan)
        try:
            baseline_params = loads(bl.get("Parameters", "{}") or "{}")
        except Exception:
            baseline_params = {}

        business_weights = build_business_weight_config(
            w_recall=self.w_recall, w_precision=self.w_precision,
            w_time=self.w_time, time_reference=self.time_reference,
        )

        for thr in thresholds:
            match = df_metric.loc[np.isclose(df_metric["Threshold"], thr)]
            if match.empty:
                continue
            r = match.iloc[0]
            enc = f"{baseline_encoder} | thr={thr:.2f}"

            bs = compute_business_score(
                recall=float(r["Recall_macro"]),
                precision=float(r["Precision_macro"]),
                fit_time=float(fit_time) if pd.notna(fit_time) else 0.0,
                score_time=float(score_time) if pd.notna(score_time) else 0.0,
                w_recall=self.w_recall, w_precision=self.w_precision,
                w_time=self.w_time, time_reference=self.time_reference,
            )
            row = {
                "Estimator": baseline_estimator,
                "Optimization/Data model": threshold_data_model,
                "Pre-Process Pipeline": enc,
                "Train Roc auc": bl.get("Train Roc auc", np.nan),
                "Test Roc auc": float(r["Roc_auc"]),
                "Train Balanced Accuracy": bl.get("Train Balanced Accuracy", np.nan),
                "Test Balanced Accuracy": float(r["Accuracy"]),
                "Train Recall": bl.get("Train Recall", np.nan),
                "Test Recall": float(r["Recall_macro"]),
                "Train Precision": bl.get("Train Precision", np.nan),
                "Test Precision": float(r["Precision_macro"]),
                "Train F1": bl.get("Train F1", np.nan),
                "Test F1": float(r["F1_macro"]),
                "Parameters": dumps({**baseline_params, "threshold": float(thr)}, default=str),
                "Fit Time": fit_time,
                "Score Time": score_time,
                "Total Time": total_time,
                "Business Score": bs,
                "Business Weights": business_weights,
            }
            dup = (
                (base["Estimator"] == baseline_estimator)
                & (base["Optimization/Data model"] == threshold_data_model)
                & (base["Pre-Process Pipeline"] == enc)
            )
            if dup.any():
                if not overwrite_existing:
                    continue
                base = base.loc[~dup].copy()
            base = pd.concat([base, pd.DataFrame([row])], ignore_index=True)

        base = base[[c for c in _ALL_COLS if c in base.columns]]
        base.sort_values(_INDEX_COLS, inplace=True, ignore_index=True)
        self._flush_csv(base)
        return base

    @staticmethod
    def _safe_part(value: str) -> str:
        return (
            str(value)
            .replace(" ", "")
            .replace("/", "-")
            .replace("\\", "-")
            .replace("|", "-")
            .replace(":", "-")
        )


# ---------------------------------------------------------------------------
# MLflow backend
# ---------------------------------------------------------------------------

class MlflowModelStore(ModelStore):
    """
    Log metrics and pipeline artifacts to an MLflow experiment.

    Each call to :meth:`save` starts a new MLflow run (deleting any prior run
    with the same name when *overwrite_existing* is ``True``).  The method also
    returns a one-row :class:`~pandas.DataFrame` in the same schema as
    :class:`CsvModelStore` so downstream notebook code works identically
    regardless of which backend is active.

    Parameters
    ----------
    experiment_name : str, default="opt-binary-clf-experiment"
        MLflow experiment name (created automatically if it does not exist).
    tracking_uri : str | None, default=None
        MLflow tracking server URI, e.g. ``"http://127.0.0.1:5000"``.
        When ``None`` the MLflow default is used (local ``./mlruns`` folder).
        Set this to the running server address so runs appear in the UI.
    w_recall : float, default=7
        Business-score recall weight.
    w_precision : float, default=3
        Business-score precision weight.
    w_time : float, default=0
        Business-score time-penalty weight.
    time_reference : float | None, default=1.0
        Reference time for time-penalty scaling.

    Notes
    -----
    ``mlflow`` is imported lazily inside :meth:`save` so that the package can
    be imported without MLflow installed when only ``CsvModelStore`` is used.
    """

    def __init__(
        self,
        experiment_name: str = "opt-binary-clf-experiment",
        tracking_uri: str | None = None,
        w_recall: float = 7,
        w_precision: float = 3,
        w_time: float = 0,
        time_reference: float | None = 1.0,
    ) -> None:
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.w_recall = w_recall
        self.w_precision = w_precision
        self.w_time = w_time
        self.time_reference = time_reference

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        description: str,
        data_model: str,
        encoder: str,
        pipeline_obj: Any,
        scores: dict[str, Any],
        params: dict[str, Any] | None,
        metric_df: pd.DataFrame | None = None,
        overwrite_existing: bool = True,
    ) -> pd.DataFrame:
        """See :class:`ModelStore` for full parameter documentation."""
        import mlflow
        import mlflow.sklearn

        if self.tracking_uri is not None:
            mlflow.set_tracking_uri(self.tracking_uri)

        mlflow.set_experiment(self.experiment_name)

        run_name = f"{description} | {data_model} | {encoder}"

        fit_time = float(np.mean(scores["fit_time"]))
        score_time = float(np.mean(scores["score_time"]))

        business_score = compute_business_score(
            recall=float(np.mean(scores["test_recall_macro"])),
            precision=float(np.mean(scores["test_precision_macro"])),
            fit_time=fit_time,
            score_time=score_time,
            w_recall=self.w_recall,
            w_precision=self.w_precision,
            w_time=self.w_time,
            time_reference=self.time_reference,
        )
        business_weights = build_business_weight_config(
            w_recall=self.w_recall,
            w_precision=self.w_precision,
            w_time=self.w_time,
            time_reference=self.time_reference,
        )

        mlflow_metrics = {
            "train_roc_auc": float(np.mean(scores["train_roc_auc"])),
            "test_roc_auc": float(np.mean(scores["test_roc_auc"])),
            "train_balanced_accuracy": float(np.mean(scores["train_balanced_accuracy"])),
            "test_balanced_accuracy": float(np.mean(scores["test_balanced_accuracy"])),
            "train_recall": float(np.mean(scores["train_recall_macro"])),
            "test_recall": float(np.mean(scores["test_recall_macro"])),
            "train_precision": float(np.mean(scores["train_precision_macro"])),
            "test_precision": float(np.mean(scores["test_precision_macro"])),
            "train_f1": float(np.mean(scores["train_f1_macro"])),
            "test_f1": float(np.mean(scores["test_f1_macro"])),
            "fit_time": fit_time,
            "score_time": score_time,
            "total_time": fit_time + score_time,
            "business_score": business_score,
        }

        if overwrite_existing:
            self._delete_existing_run(run_name, self.tracking_uri)

        with mlflow.start_run(run_name=run_name):
            mlflow.log_metrics(mlflow_metrics)
            mlflow.set_tags({
                "estimator": description,
                "data_model": data_model,
                "encoder": encoder,
                "business_weights": business_weights,
            })
            if params:
                safe_params = {k: str(v)[:500] for k, v in params.items()}
                mlflow.log_params(safe_params)
            mlflow.sklearn.log_model(pipeline_obj, artifact_path="pipeline")

        return self._build_dataframe_row(
            description, data_model, encoder,
            mlflow_metrics, params, business_score, business_weights,
            fit_time, score_time, metric_df,
        )

    # ------------------------------------------------------------------
    # Load API
    # ------------------------------------------------------------------

    def load(self, description: str, data_model: str, encoder: str) -> Any:
        """See :class:`ModelStore` for full parameter documentation."""
        import mlflow
        import mlflow.sklearn

        if self.tracking_uri is not None:
            mlflow.set_tracking_uri(self.tracking_uri)

        run_name = f"{description} | {data_model} | {encoder}"
        runs = mlflow.search_runs(
            experiment_names=[self.experiment_name],
            filter_string=f"tags.`mlflow.runName` = '{run_name}'",
            max_results=1,
        )
        if runs.empty:
            raise FileNotFoundError(
                f"No MLflow run found for: {run_name!r}\n"
                f"Experiment: {self.experiment_name!r}"
            )
        run_id = runs.iloc[0]["run_id"]
        return mlflow.sklearn.load_model(f"runs:/{run_id}/pipeline")

    def best_run(self, metric: str = "test_recall") -> dict[str, str]:
        """See :class:`ModelStore` for full parameter documentation."""
        import mlflow

        if self.tracking_uri is not None:
            mlflow.set_tracking_uri(self.tracking_uri)

        runs = mlflow.search_runs(
            experiment_names=[self.experiment_name],
            order_by=[f"metrics.{metric} DESC"],
            max_results=1,
        )
        if runs.empty:
            raise RuntimeError(
                f"No runs found in experiment {self.experiment_name!r}."
            )
        row = runs.iloc[0]
        return {
            "description": row["tags.estimator"],
            "data_model":  row["tags.data_model"],
            "encoder":     row["tags.encoder"],
            metric:        float(row[f"metrics.{metric}"]),
        }

    def find_run(
        self,
        estimator: str | None = None,
        data_model: str | None = None,
        metric: str = "test_recall",
    ) -> dict[str, str]:
        """See :class:`ModelStore` for full parameter documentation."""
        import mlflow

        if self.tracking_uri is not None:
            mlflow.set_tracking_uri(self.tracking_uri)

        filters: list[str] = []
        if estimator is not None:
            filters.append(f"tags.estimator = '{estimator}'")
        if data_model is not None:
            filters.append(f"tags.data_model = '{data_model}'")
        filter_string = " AND ".join(filters)

        runs = mlflow.search_runs(
            experiment_names=[self.experiment_name],
            filter_string=filter_string,
            order_by=[f"metrics.{metric} DESC"],
            max_results=1,
        )
        if runs.empty:
            raise FileNotFoundError(
                f"No MLflow run found: estimator={estimator!r}, data_model={data_model!r}\n"
                f"Experiment: {self.experiment_name!r}"
            )
        row = runs.iloc[0]
        return {
            "description": row["tags.estimator"],
            "data_model":  row["tags.data_model"],
            "encoder":     row["tags.encoder"],
            metric:        float(row[f"metrics.{metric}"]),
        }

    def load_metrics(self) -> pd.DataFrame:
        """See :class:`ModelStore` for full parameter documentation."""
        import mlflow

        if self.tracking_uri is not None:
            mlflow.set_tracking_uri(self.tracking_uri)

        runs = mlflow.search_runs(
            experiment_names=[self.experiment_name],
            max_results=10_000,
        )
        if runs.empty:
            return pd.DataFrame(columns=_ALL_COLS)

        rows = []
        for _, row in runs.iterrows():
            rows.append({
                "Estimator":                  row.get("tags.estimator", ""),
                "Optimization/Data model":    row.get("tags.data_model", ""),
                "Pre-Process Pipeline":       row.get("tags.encoder", ""),
                "Train Roc auc":              row.get("metrics.train_roc_auc"),
                "Test Roc auc":               row.get("metrics.test_roc_auc"),
                "Train Balanced Accuracy":    row.get("metrics.train_balanced_accuracy"),
                "Test Balanced Accuracy":     row.get("metrics.test_balanced_accuracy"),
                "Train Recall":               row.get("metrics.train_recall"),
                "Test Recall":                row.get("metrics.test_recall"),
                "Train Precision":            row.get("metrics.train_precision"),
                "Test Precision":             row.get("metrics.test_precision"),
                "Train F1":                   row.get("metrics.train_f1"),
                "Test F1":                    row.get("metrics.test_f1"),
                "Parameters":                 "",
                "Fit Time":                   row.get("metrics.fit_time"),
                "Score Time":                 row.get("metrics.score_time"),
                "Total Time":                 row.get("metrics.total_time"),
                "Business Score":             row.get("metrics.business_score"),
                "Business Weights":           row.get("tags.business_weights", ""),
            })

        df = pd.DataFrame(rows)
        df = df[[c for c in _ALL_COLS if c in df.columns]]
        df.sort_values(_INDEX_COLS, inplace=True, ignore_index=True)
        return df

    def save_threshold_runs(
        self,
        df_metric: pd.DataFrame,
        thresholds: list[float],
        baseline_estimator: str = "LogisticRegression",
        baseline_encoder: str = "OrdinalEncoder",
        baseline_data_model: str = "Baseline",
        threshold_data_model: str = "Baseline + Threshold tuning",
        metric_df: pd.DataFrame | None = None,
        overwrite_existing: bool = True,
    ) -> pd.DataFrame:
        """See :class:`ModelStore` for full parameter documentation."""
        import mlflow
        from json import dumps, loads

        if self.tracking_uri is not None:
            mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        # Inherit timing and params from baseline run in MLflow
        all_stored = self.load_metrics()
        bl_mask = (
            (all_stored["Estimator"] == baseline_estimator)
            & (all_stored["Optimization/Data model"] == baseline_data_model)
            & (all_stored["Pre-Process Pipeline"] == baseline_encoder)
        )
        if bl_mask.any():
            bl = all_stored.loc[bl_mask].iloc[0]
            fit_time   = float(bl.get("Fit Time", 0.0))
            score_time = float(bl.get("Score Time", 0.0))
            total_time = float(bl.get("Total Time", 0.0))
            try:
                baseline_params = loads(bl.get("Parameters", "{}") or "{}")
            except Exception:
                baseline_params = {}
            train_metrics = {
                "train_roc_auc":           bl.get("Train Roc auc", np.nan),
                "train_balanced_accuracy": bl.get("Train Balanced Accuracy", np.nan),
                "train_recall":            bl.get("Train Recall", np.nan),
                "train_precision":         bl.get("Train Precision", np.nan),
                "train_f1":                bl.get("Train F1", np.nan),
            }
        else:
            fit_time = score_time = total_time = 0.0
            baseline_params = {}
            train_metrics = {k: np.nan for k in [
                "train_roc_auc", "train_balanced_accuracy",
                "train_recall", "train_precision", "train_f1",
            ]}

        business_weights = build_business_weight_config(
            w_recall=self.w_recall, w_precision=self.w_precision,
            w_time=self.w_time, time_reference=self.time_reference,
        )

        result_df = metric_df.copy() if metric_df is not None else all_stored.copy()

        for thr in thresholds:
            match = df_metric.loc[np.isclose(df_metric["Threshold"], thr)]
            if match.empty:
                continue
            r = match.iloc[0]
            enc = f"{baseline_encoder} | thr={thr:.2f}"
            run_name = f"{baseline_estimator} | {threshold_data_model} | {enc}"

            bs = compute_business_score(
                recall=float(r["Recall_macro"]),
                precision=float(r["Precision_macro"]),
                fit_time=fit_time, score_time=score_time,
                w_recall=self.w_recall, w_precision=self.w_precision,
                w_time=self.w_time, time_reference=self.time_reference,
            )

            if overwrite_existing:
                self._delete_existing_run(run_name, self.tracking_uri)

            with mlflow.start_run(run_name=run_name):
                mlflow.log_metrics({
                    **{k: float(v) for k, v in train_metrics.items() if pd.notna(v)},
                    "test_roc_auc":           float(r["Roc_auc"]),
                    "test_balanced_accuracy": float(r["Accuracy"]),
                    "test_recall":            float(r["Recall_macro"]),
                    "test_precision":         float(r["Precision_macro"]),
                    "test_f1":                float(r["F1_macro"]),
                    "fit_time":               fit_time,
                    "score_time":             score_time,
                    "total_time":             total_time,
                    "business_score":         float(bs),
                })
                mlflow.log_params({"threshold": float(thr)})
                mlflow.set_tags({
                    "estimator":       baseline_estimator,
                    "data_model":      threshold_data_model,
                    "encoder":         enc,
                    "business_weights": business_weights,
                })

            threshold_params = {**baseline_params, "threshold": float(thr)}
            mlflow_metrics = {
                **train_metrics,
                "test_roc_auc":           float(r["Roc_auc"]),
                "test_balanced_accuracy": float(r["Accuracy"]),
                "test_recall":            float(r["Recall_macro"]),
                "test_precision":         float(r["Precision_macro"]),
                "test_f1":                float(r["F1_macro"]),
                "fit_time": fit_time, "score_time": score_time,
                "total_time": total_time, "business_score": float(bs),
            }
            result_df = self._build_dataframe_row(
                description=baseline_estimator,
                data_model=threshold_data_model,
                encoder=enc,
                mlflow_metrics=mlflow_metrics,
                params=threshold_params,
                business_score=float(bs),
                business_weights=business_weights,
                fit_time=fit_time,
                score_time=score_time,
                metric_df=result_df,
            )

        return result_df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _delete_existing_run(self, run_name: str, tracking_uri: str | None) -> None:
        import mlflow

        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)

        existing = mlflow.search_runs(
            experiment_names=[self.experiment_name],
            filter_string=f"tags.`mlflow.runName` = '{run_name}'",
            max_results=1,
        )
        if not existing.empty:
            mlflow.delete_run(existing.iloc[0]["run_id"])

    @staticmethod
    def _build_dataframe_row(
        description: str,
        data_model: str,
        encoder: str,
        mlflow_metrics: dict[str, float],
        params: dict[str, Any] | None,
        business_score: float,
        business_weights: str,
        fit_time: float,
        score_time: float,
        metric_df: pd.DataFrame | None,
    ) -> pd.DataFrame:
        from json import dumps

        try:
            params_str = dumps(params, default=str) if params is not None else ""
        except Exception:
            params_str = ""

        row = {
            "Estimator": description,
            "Optimization/Data model": data_model,
            "Pre-Process Pipeline": encoder,
            "Train Roc auc": mlflow_metrics["train_roc_auc"],
            "Test Roc auc": mlflow_metrics["test_roc_auc"],
            "Train Balanced Accuracy": mlflow_metrics["train_balanced_accuracy"],
            "Test Balanced Accuracy": mlflow_metrics["test_balanced_accuracy"],
            "Train Recall": mlflow_metrics["train_recall"],
            "Test Recall": mlflow_metrics["test_recall"],
            "Train Precision": mlflow_metrics["train_precision"],
            "Test Precision": mlflow_metrics["test_precision"],
            "Train F1": mlflow_metrics["train_f1"],
            "Test F1": mlflow_metrics["test_f1"],
            "Parameters": params_str,
            "Fit Time": fit_time,
            "Score Time": score_time,
            "Total Time": fit_time + score_time,
            "Business Score": business_score,
            "Business Weights": business_weights,
        }

        new_row = pd.DataFrame([row])[_ALL_COLS]

        if metric_df is None or metric_df.empty:
            return new_row

        mask = (
            (metric_df.get("Estimator", pd.Series(dtype=str)) == description)
            & (metric_df.get("Optimization/Data model", pd.Series(dtype=str)) == data_model)
            & (metric_df.get("Pre-Process Pipeline", pd.Series(dtype=str)) == encoder)
        )
        base = metric_df.loc[~mask].copy()
        result = pd.concat([base, new_row], ignore_index=True)
        result = result[[c for c in _ALL_COLS if c in result.columns]]
        result.sort_values(_INDEX_COLS, inplace=True, ignore_index=True)
        return result
