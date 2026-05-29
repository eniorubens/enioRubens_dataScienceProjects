"""
trainer.py
----------
High-level orchestration functions that tie together baseline training
and Optuna-optimised pipeline search.
"""
from __future__ import annotations

import gc
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from .estimators import define_estimators, get_estimator
from .optimizer import Optimizer
from .persistence import CsvModelStore, ModelStore
from .pipeline_builder import build_baseline_preprocessor
from .scoring import get_classification_scoring


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def run_baseline_logistic_regression(
    train_features: pd.DataFrame,
    train_labels: np.ndarray,
    cv: object,
    max_iter: int = 1000,
    random_seed: int = 42,
) -> tuple:
    """
    Train a baseline LogisticRegression pipeline and return raw CV results.

    Unlike :func:`run_baseline_estimator`, this function does **not** persist
    anything — the caller is responsible for calling ``store.save()`` with the
    returned scores and best estimator.

    Parameters
    ----------
    train_features : pd.DataFrame
        Training feature matrix.
    train_labels : np.ndarray
        Training target vector.
    cv : cross-validation splitter
    max_iter : int, default=1000
        Maximum number of LogisticRegression iterations.
    random_seed : int, default=42
        Reproducibility seed.

    Returns
    -------
    model : sklearn Pipeline
        Unfitted pipeline (useful for display / inspection).
    scores : dict
        Raw output of :func:`~sklearn.model_selection.cross_validate`.
    best_estimator : sklearn Pipeline
        Fitted pipeline from the CV fold with the highest test recall macro.
    """
    from sklearn.linear_model import LogisticRegression

    model = Pipeline(
        steps=[
            ("preprocessor", build_baseline_preprocessor()),
            ("classifier", LogisticRegression(
                max_iter=max_iter,
                random_state=random_seed,
            )),
        ]
    )

    scores = cross_validate(
        model,
        train_features,
        train_labels,
        cv=cv,
        return_train_score=True,
        return_estimator=True,
        scoring=get_classification_scoring(),
        verbose=0,
        n_jobs=-1,
    )

    best_idx = int(np.argmax(scores["test_recall_macro"]))
    best_estimator = scores["estimator"][best_idx]

    return model, scores, best_estimator


def run_baseline_estimator(
    estimator_name: str,
    train_features: pd.DataFrame,
    train_labels: np.ndarray,
    cv: object,
    random_seed: int,
    model_store: ModelStore,
    metric_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Train and persist a non-optimised baseline estimator.

    Uses ``OrdinalEncoder`` preprocessing and no resampling.

    Parameters
    ----------
    estimator_name : str
        Classifier class name.
    train_features : pd.DataFrame
        Training feature matrix.
    train_labels : np.ndarray
        Training target vector.
    cv : cross-validation splitter
    random_seed : int
        Reproducibility seed.
    model_store : ModelStore
        Storage backend for metrics and serialized pipelines.
    metric_df : pd.DataFrame
        Accumulated metrics DataFrame.

    Returns
    -------
    pd.DataFrame
        Updated metrics DataFrame.
    """
    (
        linear_estimators, ensemble_estimators, tree_estimators,
        neural_estimators, xgb_estimators, lgbm_estimators,
        catboost_estimators, extra_estimators, _,
    ) = define_estimators()

    estimator_cls = get_estimator(
        estimator_name,
        linear_estimators, ensemble_estimators, tree_estimators,
        neural_estimators, xgb_estimators, lgbm_estimators, catboost_estimators,
        extra_estimators,
    )

    baseline_params: dict = {}
    try:
        baseline_params["random_state"] = random_seed
        estimator = estimator_cls(**baseline_params)
    except TypeError:
        baseline_params = {}
        estimator = estimator_cls()

    model = Pipeline(
        steps=[
            ("preprocessor", build_baseline_preprocessor()),
            ("classifier", estimator),
        ]
    )

    scores = cross_validate(
        model,
        train_features,
        train_labels,
        cv=cv,
        return_train_score=True,
        return_estimator=True,
        scoring=get_classification_scoring(),
        verbose=0,
        n_jobs=-1,
    )

    best_idx = int(np.argmax(scores["test_recall_macro"]))
    best_estimator = scores["estimator"][best_idx]

    return model_store.save(
        description=estimator_name,
        data_model="Baseline",
        encoder="OrdinalEncoder",
        pipeline_obj=best_estimator,
        scores=scores,
        params=baseline_params,
        metric_df=metric_df,
    )


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def train_all_models(
    train_features: pd.DataFrame,
    train_labels: np.ndarray,
    test_features: pd.DataFrame,
    test_labels: np.ndarray,
    cv: object,
    x_reference: pd.DataFrame,
    metric_df: pd.DataFrame,
    random_seed: int,
    model_store: ModelStore | None = None,
    ratio_min: float | None = None,
    ratio_max: float | None = None,
    estimators_to_run: Iterable[str] | None = None,
    fixed_resampler: str = "",
    trials: int = 50,
    run_baseline: bool = True,
    run_optuna: bool = True,
    param_space: object | None = None,
) -> pd.DataFrame:
    """
    Train multiple estimators through baseline and Optuna-optimised pipelines.

    Parameters
    ----------
    train_features : pd.DataFrame
        Training feature matrix.
    train_labels : np.ndarray
        Training target vector.
    test_features : pd.DataFrame
        Test feature matrix.
    test_labels : np.ndarray
        Test target vector.
    cv : cross-validation splitter
    x_reference : pd.DataFrame
        Reference DataFrame for dtype inspection.
    metric_df : pd.DataFrame
        Accumulated metrics DataFrame.
    random_seed : int
        Reproducibility seed.
    model_store : ModelStore | None, default=None
        Storage backend. Defaults to :class:`~opt_binary_clf_pipe.CsvModelStore`
        with standard paths when ``None``.
    ratio_min : float | None, default=None
        Minimum ``scale_pos_weight`` for XGBClassifier.
    ratio_max : float | None, default=None
        Maximum ``scale_pos_weight`` for XGBClassifier.
    estimators_to_run : Iterable[str] | None, default=None
        Restrict to a subset of estimators.
    fixed_resampler : str, default=""
        Fixed over-sampler passed to the over-sampling Optimizer.
    trials : int, default=50
        Optuna trial budget per estimator and mode.
    run_baseline : bool, default=True
        Whether to run the baseline step.
    run_optuna : bool, default=True
        Whether to run Optuna optimisation.
    param_space : callable or dict | None, default=None
        Custom hyperparameter space forwarded to all ``Optimizer`` instances.

    Returns
    -------
    pd.DataFrame
        Updated metrics DataFrame.
    """
    if model_store is None:
        model_store = CsvModelStore()

    (_, _, _, _, _, _, _, _, all_estimators) = define_estimators()

    estimators = list(estimators_to_run) if estimators_to_run is not None else list(all_estimators)

    optuna_steps = 2 if run_optuna else 0
    total_steps = len(estimators) * (int(run_baseline) + optuna_steps)

    progress_bar = tqdm(total=total_steps, unit=" step", position=0, desc="Training pipelines")

    try:
        for estimator_name in estimators:
            progress_bar.set_description(estimator_name)

            if run_baseline:
                metric_df = run_baseline_estimator(
                    estimator_name=estimator_name,
                    train_features=train_features,
                    train_labels=train_labels,
                    cv=cv,
                    random_seed=random_seed,
                    model_store=model_store,
                    metric_df=metric_df,
                )
                progress_bar.update(1)

            if run_optuna:
                _common = dict(
                    estimator_name=estimator_name,
                    train_features=train_features,
                    train_labels=train_labels,
                    cv=cv,
                    random_seed=random_seed,
                    x_reference=x_reference,
                    model_store=model_store,
                    ratio_min=ratio_min,
                    ratio_max=ratio_max,
                    trials=trials,
                    test_features=test_features,
                    test_labels=test_labels,
                    metric_df=metric_df,
                    param_space=param_space,
                )

                balanced_opt = Optimizer(balanced=True, resampler="", sampling_method="", **_common)
                updated = balanced_opt.optimize()
                if updated is not None:
                    metric_df = updated
                progress_bar.update(1)

                over_opt = Optimizer(balanced=False, resampler=fixed_resampler, sampling_method="Over", **_common)
                updated = over_opt.optimize()
                if updated is not None:
                    metric_df = updated
                progress_bar.update(1)

            gc.collect()

    finally:
        progress_bar.close()

    return metric_df
