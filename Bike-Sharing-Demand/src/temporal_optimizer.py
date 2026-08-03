"""Leakage-safe temporal model selection for Bike-Sharing-Demand v4.

Orchestration only. The architecture being optimized — the estimator-
conditioned dynamic pipeline — lives in :mod:`src.modeling_pipeline`, which
both this optimizer and the v3 reference ``RegressionOptimizer`` describe;
nothing about how a pipeline is assembled is decided here.

What this module adds on top of that contract is the temporal discipline:

* the development data, the cross-validator and every path arrive through the
  constructor, never through module-level globals;
* there is no constructor parameter through which holdout data could be
  passed, so the guarantee is enforced by ``TypeError`` rather than by
  convention;
* studies are partitioned by ``run_mode`` — a smoke run and a full run never
  share an Optuna study, an artifact directory, or a champion query;
* champion selection is fail-closed on dataset fingerprint, CV strategy
  version, code version and run mode, and never reads a holdout metric;
* the fold protocol has exactly one implementation
  (:func:`temporal_cv_fold_results`), shared by the Optuna objective, by the
  winner's diagnostics and by the refit that produces the artifact, so the
  metric that selected a model always describes the model that was saved.

The subprocess plumbing for the per-trial wall-clock guard and the
transient-versus-structural error classification are imported unchanged from
:mod:`src.optimizer`: they are generic in what they run, so they needed no
adaptation.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils import estimator_html_repr

from src.environment import (
    ENVIRONMENT_NAME,
    describe_git_source_state,
    environment_fingerprint,
    require_environment,
)
from src.modeling_pipeline import (
    BOOSTING_CAP_RATIO,
    BOOSTING_BUDGET_FIXED,
    CATEGORICAL_FEATURES,
    ESTIMATOR_CLASSES,
    NUMERICAL_FEATURES,
    PipelineSpec,
    SEARCH_PROFILE_BROAD,
    TARGET_STRATEGY_DIRECT,
    build_dynamic_pipeline,
    count_output_features,
    estimator_family,
    modeler_space,
    selector_space,
    validate_combination,
)
from src.optimizer import (
    _is_transient_error,
    _run_in_subprocess_with_timeout,
    suppress_category_encoder_intercept_warning,
)
from src.tracking import params_hash, pipeline_provenance
from src.trend import RobustTrendResidualRegressor

logger = logging.getLogger(__name__)

# Bumped from v3 when the search timeout became cumulative and hard at the
# current-trial boundary. Previous studies used a per-invocation soft timeout,
# so resuming them would silently measure a different budget contract.
CODE_VERSION = "temporal_optimizer_v7"
CV_STRATEGY_NAME = "ForwardMeteorologicalYearSplit"
CV_STRATEGY_VERSION = "ForwardMeteorologicalYearSplit_v3_normal_operations"

RUN_MODE_SMOKE = "smoke"
RUN_MODE_FULL = "full"
RUN_MODES: Tuple[str, ...] = (RUN_MODE_SMOKE, RUN_MODE_FULL)

# A smoke run exists to prove the machinery end to end, never to rank models.
MAX_SMOKE_TRIALS = 2

# Which limit ended a study. Optuna's ``optimize`` takes both ``n_trials`` and
# ``timeout`` and returns on whichever arrives first, without saying which —
# so the distinction is reconstructed from the trial count and recorded, since
# a study cut short by the clock has a different standing from one that
# exhausted its budget.
TERMINATION_TRIAL_LIMIT = "trial_limit"
TERMINATION_STUDY_TIMEOUT = "study_timeout"
TERMINATION_REASONS: Tuple[str, ...] = (TERMINATION_TRIAL_LIMIT, TERMINATION_STUDY_TIMEOUT)

STUDY_ELAPSED_SECONDS_ATTR = "cumulative_search_elapsed_seconds"
STUDY_TIMEOUT_REACHED_ATTR = "search_timeout_reached"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STUDIES_DIR = _PROJECT_ROOT / "optuna_studies"
DEFAULT_CANDIDATES_ROOT = _PROJECT_ROOT / "dataset"
DEFAULT_INVALID_CONFIGS_PATH = _PROJECT_ROOT / "dataset" / "invalid_configs_v4.csv"

_INVALID_CONFIG_COLS = [
    "estimator",
    "modeler_name",
    "encoder",
    "selector",
    "dataset_fingerprint",
    "cv_version",
    "code_version",
    "run_mode",
    "reason",
]
_INVALID_CONFIG_KEYS = [column for column in _INVALID_CONFIG_COLS if column != "reason"]

# Same set as src.optimizer._EARLY_STOPPING_ESTIMATORS. HistGradientBoosting
# is absent on purpose: its native early stopping is pinned off in the shared
# hyperparameter space because sklearn carves that validation slice at random.
_EARLY_STOPPING_ESTIMATORS = ("XGBRegressor", "LGBMRegressor", "CatBoostRegressor")

DEFAULT_ESTIMATORS: List[str] = [
    "DummyRegressor",
    "Ridge",
    "HistGradientBoostingRegressor",
    "XGBRegressor",
    "LGBMRegressor",
]

# Tags that must match exactly before a run may be considered for champion or
# challenger. Absence of any of them makes selection fail rather than guess.
# ``environment_name`` and ``environment_fingerprint`` are part of the set
# because a run produced by a different interpreter was produced by a different
# numerical stack, whatever its metrics say.
REQUIRED_SELECTION_TAGS: Tuple[str, ...] = (
    "dataset_fingerprint",
    "cv_strategy_version",
    "code_version",
    "run_mode",
    "model_logged",
    "model_artifact_verified",
    "environment_name",
    "environment_fingerprint",
    "git_commit",
    "git_source_dirty",
    "git_source_status_hash",
    "git_source_fingerprint",
    "regime_policy",
    "regime_fingerprint",
)


# ---------------------------------------------------------------------------
# Dataset fingerprint
# ---------------------------------------------------------------------------


def dataset_fingerprint(X: pd.DataFrame, y: pd.Series) -> str:
    """Hash the exact development dataset: values, order, schema and dtypes.

    Every ingredient that could change what a study means is folded in — the
    column names and their order, the declared dtypes, the row-wise hash of
    every feature value (which includes the timestamp columns and, because it
    is computed row by row in position order, the row ordering itself), and
    the target values. Changing a single cell of ``X`` while leaving the
    shape, the column names and ``y`` untouched therefore produces a
    different fingerprint, which is what stops a stale Optuna study or a
    stale MLflow run from being silently reused against different data.
    """
    hasher = hashlib.sha256()
    hasher.update(repr(list(X.columns)).encode())
    hasher.update(repr([str(dtype) for dtype in X.dtypes]).encode())
    hasher.update(repr(X.shape).encode())
    hasher.update(pd.util.hash_pandas_object(X, index=True).to_numpy().tobytes())
    hasher.update(repr(y.name).encode())
    hasher.update(pd.util.hash_pandas_object(y, index=True).to_numpy().tobytes())
    return hasher.hexdigest()[:16]


# ---------------------------------------------------------------------------
# The temporal CV protocol — one implementation, used by search and by refit
# ---------------------------------------------------------------------------

# Early-stopping carve geometry, shared by every caller so that the number a
# fold discovers is the number every later stage uses.
ES_VAL_FRACTION = 0.12
ES_GAP = 48
ES_ROUNDS = 30

# How the per-fold boosting budgets are collapsed into the single budget of
# the artifact refit on the whole development set. The median is used rather
# than the mean or the maximum: the folds' training windows differ in length
# by design (the window expands), so their best iteration counts differ in
# scale, and a median is the aggregate least disturbed by the one fold whose
# early stopping fired unusually late.
ITERATION_AGGREGATION = "median"


def _iteration_param(regressor) -> Optional[str]:
    """Name of the boosting-budget parameter of ``regressor``, if it has one."""
    name = type(regressor).__name__
    if name in ("XGBRegressor", "LGBMRegressor"):
        return "n_estimators"
    if name == "CatBoostRegressor":
        return "iterations"
    return None


def _core_pipeline(pipeline):
    """Return the dynamic sklearn pipeline inside an optional trend wrapper."""

    if isinstance(pipeline, RobustTrendResidualRegressor):
        return getattr(pipeline, "estimator_", pipeline.estimator)
    return pipeline


def iteration_ceiling(pipeline) -> Optional[int]:
    """The boosting budget a fold's early stopping is allowed to reach.

    Read from the pipeline rather than assumed, so the ceiling recorded on a
    fold is the one that fold actually ran under — including when a caller
    pinned a different value than :data:`~src.modeling_pipeline.BOOSTING_CEILING`.
    """
    try:
        regressor = _core_pipeline(pipeline).named_steps["regressor"].regressor
    except (AttributeError, KeyError):
        return None
    if _iteration_param(regressor) is None:
        return None
    value = regressor.get_params().get(_iteration_param(regressor))
    return None if value is None else int(value)


def hit_iteration_ceiling(
    best_iteration: Optional[int],
    ceiling: Optional[int],
    ratio: float = BOOSTING_CAP_RATIO,
) -> bool:
    """Whether a fold's discovered budget is close enough to its ceiling to be suspect.

    Early stopping reports the iteration with the best validation score, so a
    number just under the ceiling is indistinguishable from a fit that was
    still improving when the budget ran out. Treating anything at or above
    ``ratio`` of the ceiling as a hit is deliberately conservative: the
    consequence of a false positive is a warning, the consequence of a false
    negative is a model frozen at an arbitrary size.
    """
    if not best_iteration or not ceiling:
        return False
    return int(best_iteration) >= ratio * int(ceiling)


def set_iteration_budget(pipeline, n_iterations: Optional[int]) -> Optional[int]:
    """Pin the boosting budget of ``pipeline``'s regressor, returning what was set.

    Returns ``None`` when the estimator has no boosting budget or when
    ``n_iterations`` is not usable, in which case the sampled ``n_estimators``
    is left untouched.
    """
    if not n_iterations or n_iterations < 1:
        return None
    regressor = _core_pipeline(pipeline).named_steps["regressor"].regressor
    param = _iteration_param(regressor)
    if param is None:
        return None
    regressor.set_params(**{param: int(n_iterations)})
    return int(n_iterations)


def _discover_best_iteration(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    train_idx,
    es_val_fraction: float = ES_VAL_FRACTION,
    es_gap: int = ES_GAP,
    es_rounds: int = ES_ROUNDS,
) -> Optional[int]:
    """Find the boosting budget on a temporal tail of ``train_idx`` alone.

    The fold's own test block is never touched here: the tail is carved from
    the *end* of the training window, separated from the fitting portion by
    ``es_gap`` rows, and the preprocessing plus the target transform are fit
    on the earlier portion only. What comes back is a number of iterations,
    normalised across libraries — XGBoost reports a zero-based
    ``best_iteration`` and LightGBM a one-based ``best_iteration_``.
    """
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from sklearn.base import clone as _clone
    from sklearn.pipeline import Pipeline as _Pipeline
    from xgboost import XGBRegressor

    n_val = max(int(len(train_idx) * es_val_fraction), 50)
    if len(train_idx) <= n_val + es_gap:
        raise ValueError(
            f"Fold train window has only {len(train_idx)} rows — too small to carve a "
            f"{n_val}-row early-stopping slice plus a {es_gap}-row gap."
        )
    val_idx = train_idx[-n_val:]
    fit_idx = train_idx[: -(n_val + es_gap)]

    carve_pipeline = _clone(_core_pipeline(pipeline))
    if isinstance(pipeline, RobustTrendResidualRegressor):
        trend = _clone(pipeline)
        trend.fit_trend(X.iloc[fit_idx], y.iloc[fit_idx])
        y_fit_source = trend.transform_target(X.iloc[fit_idx], y.iloc[fit_idx])
        y_val_source = trend.transform_target(X.iloc[val_idx], y.iloc[val_idx])
    else:
        y_fit_source = y.iloc[fit_idx].to_numpy()
        y_val_source = y.iloc[val_idx].to_numpy()

    preprocessing = _Pipeline(carve_pipeline.steps[:-1])
    X_fit = preprocessing.fit_transform(X.iloc[fit_idx], y_fit_source)
    X_val = preprocessing.transform(X.iloc[val_idx])

    ttr = carve_pipeline.named_steps["regressor"]
    y_fit_raw = np.asarray(y_fit_source).reshape(-1, 1)
    y_val_raw = np.asarray(y_val_source).reshape(-1, 1)
    if ttr.transformer is not None:
        target_transformer = _clone(ttr.transformer)
        target_transformer.fit(y_fit_raw)
        y_fit = target_transformer.transform(y_fit_raw).ravel()
        y_val = target_transformer.transform(y_val_raw).ravel()
    else:
        y_fit = y_fit_raw.ravel()
        y_val = y_val_raw.ravel()

    regressor = _clone(ttr.regressor)
    if isinstance(regressor, XGBRegressor):
        regressor.set_params(early_stopping_rounds=es_rounds)
        regressor.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
        best = getattr(regressor, "best_iteration", None)
        return None if best is None else int(best) + 1
    if isinstance(regressor, LGBMRegressor):
        import lightgbm

        # eval_metric="l1" guards against trial params carrying an invalid
        # `metric` value: the callback needs one valid eval metric or the fit
        # raises.
        regressor.fit(
            X_fit,
            y_fit,
            eval_set=[(X_val, y_val)],
            eval_metric="l1",
            callbacks=[lightgbm.early_stopping(stopping_rounds=es_rounds, verbose=False)],
        )
        best = getattr(regressor, "best_iteration_", None)
        return None if not best else int(best)
    if isinstance(regressor, CatBoostRegressor):
        regressor.set_params(
            early_stopping_rounds=es_rounds,
            use_best_model=True,
            allow_writing_files=False,
        )
        regressor.fit(X_fit, y_fit, eval_set=(X_val, y_val), verbose=False)
        best = regressor.get_best_iteration()
        return None if best is None or best < 0 else int(best) + 1
    return None


def temporal_cv_fold_results(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    early_stopping: bool = False,
    return_predictions: bool = False,
    train_eligible_mask: Optional[Sequence[bool]] = None,
    score_eligible_mask: Optional[Sequence[bool]] = None,
    selection_test_years: Optional[Sequence[int]] = None,
    es_val_fraction: float = ES_VAL_FRACTION,
    es_gap: int = ES_GAP,
    es_rounds: int = ES_ROUNDS,
) -> List[Dict[str, Any]]:
    """Evaluate ``pipeline`` fold by fold — the single protocol of this project.

    This is what the Optuna objective scores, what the winning trial's
    diagnostics are computed from, and what determines the boosting budget of
    the persisted artifact. Having one implementation is the point: when the
    search scored a model trained one way and the artifact was refit another
    way, the reported metric described a model that was never saved.

    For every fold, when ``early_stopping`` is on, a temporal tail of the
    training window discovers the boosting budget (see
    :func:`_discover_best_iteration`); the whole pipeline is then refit on the
    *entire* training window with that budget pinned, and only then is the
    fold's test block predicted. Without early stopping the fold is a plain
    fit on the whole training window. Either way the fold's test block is used
    for nothing but scoring.
    """
    from sklearn.base import clone as _clone

    train_mask = (
        np.ones(len(X), dtype=bool)
        if train_eligible_mask is None
        else np.asarray(train_eligible_mask, dtype=bool)
    )
    score_mask = (
        np.ones(len(X), dtype=bool)
        if score_eligible_mask is None
        else np.asarray(score_eligible_mask, dtype=bool)
    )
    if train_mask.shape != (len(X),) or score_mask.shape != (len(X),):
        raise ValueError("Regime masks must contain exactly one value per development row.")

    declared_test_years = tuple(getattr(cv, "test_years", ()))
    selected_years = (
        set(declared_test_years)
        if selection_test_years is None
        else {int(year) for year in selection_test_years}
    )

    def metrics(y_true, y_pred, prefix: str = "") -> Dict[str, float]:
        return {
            f"{prefix}mae": float(mean_absolute_error(y_true, y_pred)),
            f"{prefix}rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            f"{prefix}r2": float(r2_score(y_true, y_pred)),
            f"{prefix}wape": float(
                np.abs(y_true - y_pred).sum() / max(np.abs(y_true).sum(), np.finfo(float).eps)
            ),
            f"{prefix}mean_bias": float(np.mean(y_pred - y_true)),
        }

    results: List[Dict[str, Any]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        raw_train_count = len(train_idx)
        train_idx = np.asarray(train_idx)[train_mask[np.asarray(train_idx)]]
        test_idx = np.asarray(test_idx)
        if len(train_idx) < 2:
            raise ValueError(
                f"Fold {fold_idx} has fewer than two eligible training rows "
                "after applying the regime policy."
            )
        test_year = (
            int(declared_test_years[fold_idx - 1])
            if fold_idx <= len(declared_test_years)
            else fold_idx
        )
        is_selection_fold = test_year in selected_years
        selection_positions = (
            score_mask[test_idx] if is_selection_fold else np.zeros(len(test_idx), dtype=bool)
        )
        if is_selection_fold and int(selection_positions.sum()) < 2:
            raise ValueError(
                f"Selection fold {test_year} has fewer than two normal-regime test rows."
            )

        fold_pipeline = _clone(pipeline)
        best_iteration = None
        ceiling = None
        cap_hit = False
        if early_stopping:
            ceiling = iteration_ceiling(fold_pipeline)
            best_iteration = _discover_best_iteration(
                fold_pipeline, X, y, train_idx, es_val_fraction, es_gap, es_rounds
            )
            cap_hit = hit_iteration_ceiling(best_iteration, ceiling)
            if cap_hit:
                logger.warning(
                    "Fold %d stopped at iteration %s of a %s ceiling — the budget, not "
                    "the validation loss, may have ended the fit.",
                    fold_idx,
                    best_iteration,
                    ceiling,
                )
            set_iteration_budget(fold_pipeline, best_iteration)

        with suppress_category_encoder_intercept_warning():
            fold_pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
            y_test_pred = fold_pipeline.predict(X.iloc[test_idx])
            y_train_pred = fold_pipeline.predict(X.iloc[train_idx])

        y_test_true = y.iloc[test_idx].to_numpy()
        y_train_true = y.iloc[train_idx].to_numpy()

        fold: Dict[str, Any] = {
            "fold": fold_idx,
            "test_year": test_year,
            "fold_role": "selection" if is_selection_fold else "stress",
            "n_train": int(len(train_idx)),
            "n_train_excluded": int(raw_train_count - len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_selection_test": int(selection_positions.sum()),
            "best_iteration": best_iteration,
            "iteration_ceiling": ceiling,
            "best_iteration_cap_hit": bool(cap_hit),
            "train_r2": float(r2_score(y_train_true, y_train_pred)),
        }
        fold.update(metrics(y_test_true, y_test_pred))
        if is_selection_fold:
            fold.update(
                metrics(
                    y_test_true[selection_positions],
                    np.asarray(y_test_pred)[selection_positions],
                    prefix="selection_",
                )
            )
        else:
            fold.update(
                {
                    "selection_mae": np.nan,
                    "selection_rmse": np.nan,
                    "selection_r2": np.nan,
                    "selection_wape": np.nan,
                    "selection_mean_bias": np.nan,
                }
            )
        if return_predictions:
            fold["test_index"] = np.asarray(test_idx)
            fold["y_true"] = y_test_true
            fold["y_pred"] = np.asarray(y_test_pred)
            fold["selection_test_mask"] = selection_positions
        results.append(fold)
    return results


def selection_fold_metrics(fold_frame: pd.DataFrame) -> pd.DataFrame:
    """Return the normal-regime fold metrics in the canonical summary schema."""
    selected = fold_frame.loc[fold_frame["fold_role"] == "selection"].copy()
    if selected.empty:
        raise ValueError("The temporal protocol produced no selection folds.")
    rename = {
        "selection_mae": "mae",
        "selection_rmse": "rmse",
        "selection_r2": "r2",
        "selection_wape": "wape",
        "selection_mean_bias": "mean_bias",
    }
    selected.drop(columns=list(rename.values()), errors="ignore", inplace=True)
    return selected.rename(columns=rename)


def summarize_cv_fold_metrics(
    fold_frame: pd.DataFrame,
    fold_weights: Sequence[float],
) -> Dict[str, float]:
    """Summarize temporal folds with both central and robustness diagnostics.

    The weighted metrics use the same recency weights as the Optuna objective.
    The median R² exposes whether a poor mean is concentrated in one anomalous
    fold, while the mean absolute fold bias measures systematic over- or
    under-prediction without allowing opposite fold biases to cancel out.
    """
    required = {"mae", "rmse", "r2", "wape", "mean_bias"}
    missing = sorted(required.difference(fold_frame.columns))
    if missing:
        raise ValueError(f"fold_frame is missing required metrics: {missing}.")

    weights = np.asarray(fold_weights, dtype=float)
    if (
        weights.shape != (len(fold_frame),)
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0)
    ):
        raise ValueError(
            "fold_weights must contain one finite positive value for every temporal fold."
        )

    return {
        "cv_mae_mean": float(fold_frame["mae"].mean()),
        "cv_mae_weighted": float(np.average(fold_frame["mae"], weights=weights)),
        "cv_rmse_mean": float(fold_frame["rmse"].mean()),
        "cv_r2_mean": float(fold_frame["r2"].mean()),
        "cv_r2_median": float(fold_frame["r2"].median()),
        "cv_r2_weighted": float(np.average(fold_frame["r2"], weights=weights)),
        "cv_wape_mean": float(fold_frame["wape"].mean()),
        "cv_mean_bias": float(fold_frame["mean_bias"].mean()),
        "cv_mean_abs_fold_bias": float(fold_frame["mean_bias"].abs().mean()),
        "cv_mae_std": float(fold_frame["mae"].std()),
    }


def aggregate_iteration_budget(best_iterations: Sequence[Optional[int]]) -> Optional[int]:
    """Collapse the per-fold boosting budgets into the budget of the final refit.

    The rule is :data:`ITERATION_AGGREGATION` — the median of the folds that
    actually stopped early, rounded to an integer. ``None`` means no fold
    produced a budget, and the sampled ``n_estimators`` stands.
    """
    values = [int(value) for value in best_iterations if value]
    if not values:
        return None
    return int(round(float(np.median(values))))


def summarize_iteration_truncation(folds: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarise how often early stopping ran into its ceiling across the folds.

    Truncation is called *systematic* when at least half of the folds that
    produced a budget hit the ceiling: an isolated fold reaching it is a
    plausible property of that year's data, whereas half of them reaching it
    says the ceiling — not the data — is choosing the model size, and the
    aggregate budget the artifact inherits would be an artefact of the limit.
    """
    cap_hits = [bool(fold.get("best_iteration_cap_hit")) for fold in folds]
    ceilings = [fold.get("iteration_ceiling") for fold in folds if fold.get("iteration_ceiling")]
    n_with_budget = sum(1 for fold in folds if fold.get("best_iteration"))
    n_cap_hit = sum(cap_hits)
    return {
        "cap_hits_by_fold": cap_hits,
        "n_folds_cap_hit": n_cap_hit,
        "n_folds_with_budget": n_with_budget,
        "iteration_ceiling": int(ceilings[0]) if ceilings else None,
        "systematic": bool(n_with_budget and n_cap_hit * 2 >= n_with_budget),
    }


def _run_temporal_cv_in_subprocess(
    pipeline,
    X,
    y,
    cv,
    early_stopping,
    train_eligible_mask,
    score_eligible_mask,
    selection_test_years,
    es_val_fraction,
    es_gap,
    es_rounds,
    result_queue,
) -> None:
    """Target of the timeout-guarded subprocess: one pass of the shared protocol.

    Runs in a freshly spawned interpreter, so every input arrives explicitly.
    Predictions are deliberately not returned — the parent only needs the
    per-fold metrics, and shipping five folds of predictions back through the
    queue would cost more than recomputing them for the one winning trial.
    """
    try:
        import joblib

        with joblib.parallel_backend("threading"):
            folds = temporal_cv_fold_results(
                pipeline,
                X,
                y,
                cv,
                early_stopping=early_stopping,
                return_predictions=False,
                train_eligible_mask=train_eligible_mask,
                score_eligible_mask=score_eligible_mask,
                selection_test_years=selection_test_years,
                es_val_fraction=es_val_fraction,
                es_gap=es_gap,
                es_rounds=es_rounds,
            )
        result_queue.put(("ok", folds))
    except Exception as exc:
        result_queue.put(("error", str(exc)[:2000]))


def temporal_cv_with_timeout(
    pipeline,
    X,
    y,
    cv,
    timeout: float,
    early_stopping: bool = False,
    train_eligible_mask: Optional[Sequence[bool]] = None,
    score_eligible_mask: Optional[Sequence[bool]] = None,
    selection_test_years: Optional[Sequence[int]] = None,
    es_val_fraction: float = ES_VAL_FRACTION,
    es_gap: int = ES_GAP,
    es_rounds: int = ES_ROUNDS,
):
    """Run :func:`temporal_cv_fold_results` under a wall-clock guard.

    Returns ``(folds, None)`` on success or ``(None, error_message)`` on
    timeout or failure — never raises, so the objective does not need to wrap
    the timeout mechanics in its own try/except.
    """
    status, payload = _run_in_subprocess_with_timeout(
        _run_temporal_cv_in_subprocess,
        (
            pipeline,
            X,
            y,
            cv,
            early_stopping,
            train_eligible_mask,
            score_eligible_mask,
            selection_test_years,
            es_val_fraction,
            es_gap,
            es_rounds,
        ),
        timeout,
    )
    if status == "ok":
        return payload, None
    if status == "timeout":
        return None, f"Trial exceeded timeout of {timeout}s and was terminated"
    return None, payload


# ---------------------------------------------------------------------------
# Post-study diagnostics
# ---------------------------------------------------------------------------


@dataclass
class FoldEvaluation:
    """Diagnostics for the winning trial, computed only from development folds.

    Holdout data reaches none of this: every metric here comes from the
    expanding CV's own test blocks, and ``fitted_pipeline`` is refit on the
    full development set under the same protocol the folds used.
    """

    best_params: dict
    spec: PipelineSpec
    fold_metrics: pd.DataFrame
    seasonal_metrics: pd.DataFrame
    extreme_metrics: pd.DataFrame
    fitted_pipeline: Any
    trials_completed: int = 0
    cv_metrics: Dict[str, float] = field(default_factory=dict)
    best_iterations_by_fold: List[Optional[int]] = field(default_factory=list)
    final_n_estimators: Optional[int] = None
    iteration_aggregation: str = ITERATION_AGGREGATION
    cap_hits_by_fold: List[bool] = field(default_factory=list)
    n_folds_cap_hit: int = 0
    n_folds_with_budget: int = 0
    iteration_ceiling: Optional[int] = None
    systematic_truncation: bool = False


# ---------------------------------------------------------------------------
# TemporalRegressionOptimizer
# ---------------------------------------------------------------------------


class TemporalRegressionOptimizer:
    """Optuna search over the dynamic pipeline, under an expanding temporal CV.

    Parameters
    ----------
    estimator:
        One of the keys of :data:`src.modeling_pipeline.ESTIMATOR_CLASSES`.
    X_dev, y_dev:
        Development data only, already sealed away from the final holdout by
        :func:`src.cv.split_dev_holdout`.
    cv:
        An ``ExpandingMeteorologicalYearSplit`` (or any compatible
        ``BaseCrossValidator``).
    run_mode:
        ``"smoke"`` or ``"full"``. Part of the study name and of the storage
        file name, so the two never share trials; ``"smoke"`` is additionally
        capped at :data:`MAX_SMOKE_TRIALS` trials in total, counted across
        re-executions rather than per call.
    trials:
        Total trial budget for the study. ``optimize`` tops the study up to
        this number instead of appending to it, so re-running a notebook does
        not grow a study past its configured size.
    trial_timeout, study_timeout:
        Wall-clock guards, in seconds.
    numeric_features, categorical_features:
        Candidate feature lists; default to the shared v4 lists.
    studies_dir, invalid_configs_path:
        Persistence locations. The blocklist is keyed on pipeline *shape*
        (estimator, modeler, encoder, selector) plus dataset fingerprint, CV
        version, code version and run mode — never on the numeric
        hyperparameters that vary within a shape.
    seed:
        TPESampler seed.

    Notes
    -----
    There is deliberately no holdout parameter. Passing ``X_holdout=`` or
    ``y_holdout=`` raises ``TypeError`` for an unexpected keyword argument.
    """

    def __init__(
        self,
        estimator: str,
        X_dev: pd.DataFrame,
        y_dev: pd.Series,
        cv,
        run_mode: str = RUN_MODE_FULL,
        trials: int = 50,
        trial_timeout: float = 600.0,
        study_timeout: Optional[float] = 3600.0,
        numeric_features: Sequence[str] = tuple(NUMERICAL_FEATURES),
        categorical_features: Sequence[str] = tuple(CATEGORICAL_FEATURES),
        studies_dir: Path = DEFAULT_STUDIES_DIR,
        invalid_configs_path: Path = DEFAULT_INVALID_CONFIGS_PATH,
        seed: int = 42,
        fold_weights: Optional[Sequence[float]] = None,
        search_profile: str = SEARCH_PROFILE_BROAD,
        target_strategy: str = TARGET_STRATEGY_DIRECT,
        enqueued_trials: Optional[Sequence[Mapping[str, Any]]] = None,
        train_eligible_mask: Optional[Sequence[bool]] = None,
        score_eligible_mask: Optional[Sequence[bool]] = None,
        selection_test_years: Optional[Sequence[int]] = None,
        regime_policy: str = "all_observed",
        regime_fingerprint_value: str = "all_observed",
    ) -> None:
        # Before anything persistent exists: an Optuna study created under the
        # wrong interpreter would carry that interpreter's numbers forever, and
        # the storage file does not record which one produced them.
        require_environment()

        if estimator not in ESTIMATOR_CLASSES:
            raise ValueError(
                f"Unknown estimator '{estimator}'. Available: {sorted(ESTIMATOR_CLASSES)}"
            )
        if run_mode not in RUN_MODES:
            raise ValueError(f"run_mode must be one of {RUN_MODES}, got '{run_mode}'.")

        self.estimator = estimator
        self.family = estimator_family(estimator)
        self.X_dev = X_dev
        self.y_dev = y_dev
        self.cv = cv
        self.run_mode = run_mode
        self.trials = min(trials, MAX_SMOKE_TRIALS) if run_mode == RUN_MODE_SMOKE else trials
        self.trial_timeout = trial_timeout
        self.study_timeout = study_timeout
        self.numeric_features = list(numeric_features)
        self.categorical_features = list(categorical_features)
        self.studies_dir = Path(studies_dir)
        self.invalid_configs_path = Path(invalid_configs_path)
        self.seed = seed
        n_folds = cv.get_n_splits(X_dev)
        declared_test_years = tuple(int(year) for year in getattr(cv, "test_years", ()))
        self.selection_test_years = (
            declared_test_years
            if selection_test_years is None
            else tuple(int(year) for year in selection_test_years)
        )
        if declared_test_years and not set(self.selection_test_years).issubset(declared_test_years):
            raise ValueError("selection_test_years must be a subset of the CV test years.")
        ordered_selection_years = tuple(
            year for year in declared_test_years if year in set(self.selection_test_years)
        )
        if declared_test_years and self.selection_test_years != ordered_selection_years:
            raise ValueError(
                "selection_test_years must follow the CV test-year order because "
                "fold_weights are positional."
            )
        n_selection_folds = len(self.selection_test_years) if declared_test_years else n_folds
        weights = (
            np.ones(n_selection_folds, dtype=float)
            if fold_weights is None
            else np.asarray(fold_weights)
        )
        if (
            weights.shape != (n_selection_folds,)
            or not np.all(np.isfinite(weights))
            or np.any(weights <= 0)
        ):
            raise ValueError(
                f"fold_weights must contain {n_selection_folds} finite positive values; got "
                f"{list(fold_weights) if fold_weights is not None else fold_weights}."
            )
        self.fold_weights = tuple(float(value) for value in weights)
        self.train_eligible_mask = (
            np.ones(len(X_dev), dtype=bool)
            if train_eligible_mask is None
            else np.asarray(train_eligible_mask, dtype=bool)
        )
        self.score_eligible_mask = (
            np.ones(len(X_dev), dtype=bool)
            if score_eligible_mask is None
            else np.asarray(score_eligible_mask, dtype=bool)
        )
        if self.train_eligible_mask.shape != (len(X_dev),) or self.score_eligible_mask.shape != (
            len(X_dev),
        ):
            raise ValueError("Regime masks must contain one value per development row.")
        self.regime_policy = regime_policy
        self.regime_fingerprint = regime_fingerprint_value
        self.search_profile = search_profile
        self.target_strategy = target_strategy
        self.enqueued_trials = tuple(dict(params) for params in (enqueued_trials or ()))
        self.dataset_fingerprint = dataset_fingerprint(X_dev, y_dev)
        self.environment_fingerprint = environment_fingerprint()
        cv_payload = {
            "class": type(cv).__name__,
            "test_years": list(getattr(cv, "test_years", [])),
            "gap": getattr(cv, "gap", None),
            "max_train_years": getattr(cv, "max_train_years", None),
            "fold_weights": self.fold_weights,
            "selection_test_years": self.selection_test_years,
            "regime_policy": self.regime_policy,
            "regime_fingerprint": self.regime_fingerprint,
        }
        self.cv_fingerprint = hashlib.sha256(
            json.dumps(cv_payload, sort_keys=True).encode()
        ).hexdigest()[:12]
        self.sampler = optuna.samplers.TPESampler(seed=seed)
        self.termination_reason: Optional[str] = None
        self.elapsed_seconds: float = 0.0
        self.cumulative_elapsed_seconds: float = 0.0
        self._search_deadline: Optional[float] = None

        if self.invalid_configs_path.exists():
            self.invalid_df = pd.read_csv(self.invalid_configs_path)
        else:
            self.invalid_df = pd.DataFrame(columns=_INVALID_CONFIG_COLS)

    # -- identity -----------------------------------------------------------------

    @property
    def study_name(self) -> str:
        """Deterministic study identity: estimator, mode, versions and both fingerprints.

        The environment fingerprint is part of the name so that trials measured
        under one numerical stack are never appended to, or resumed from, a
        study built under another. The cost is that upgrading scikit-learn or a
        booster starts a fresh study; that is the correct outcome, because the
        old trials no longer describe what a new trial would measure.
        """
        return "__".join(
            [
                self.estimator,
                self.run_mode,
                CV_STRATEGY_VERSION,
                self.cv_fingerprint,
                self.dataset_fingerprint,
                CODE_VERSION,
                self.search_profile,
                self.target_strategy,
                self.environment_fingerprint,
            ]
        )

    @property
    def storage_url(self) -> str:
        """SQLite URL, with one database file per run mode."""
        self.studies_dir.mkdir(parents=True, exist_ok=True)
        db_path = (self.studies_dir / f"bike_sharing_demand_v4_{self.run_mode}.db").as_posix()
        return f"sqlite:///{db_path}"

    @property
    def modeler_space(self) -> List[str]:
        """Representation strategies available to this estimator's family."""
        return modeler_space(self.estimator, self.search_profile)

    @property
    def selector_space(self) -> List[str]:
        """Feature selectors available to this estimator's family."""
        return selector_space(self.estimator, self.search_profile)

    def _weights_for(self, n_folds: int) -> np.ndarray:
        """Return objective weights, tolerating shortened equal-weight test doubles."""

        weights = np.asarray(self.fold_weights, dtype=float)
        if len(weights) == n_folds:
            return weights
        if np.allclose(weights, weights[0]):
            return np.ones(n_folds, dtype=float)
        raise ValueError(
            f"CV returned {n_folds} folds but {len(weights)} non-uniform fold weights "
            "were configured."
        )

    # -- invalid-config blocklist ---------------------------------------------------

    def _config_signature(self, spec: PipelineSpec) -> dict:
        return {
            "estimator": self.estimator,
            "modeler_name": spec.modeler_name,
            "encoder": spec.encoder,
            "selector": spec.selector,
            "dataset_fingerprint": self.dataset_fingerprint,
            "cv_version": CV_STRATEGY_VERSION,
            "code_version": CODE_VERSION,
            "run_mode": self.run_mode,
        }

    def _was_invalid_config(self, spec: PipelineSpec) -> bool:
        if self.invalid_df.empty:
            return False
        signature = self._config_signature(spec)
        mask = pd.Series(True, index=self.invalid_df.index)
        for key, value in signature.items():
            if key not in self.invalid_df.columns:
                return False
            mask &= self.invalid_df[key] == value
        return bool(mask.any())

    def _register_invalid_config(self, spec: PipelineSpec, reason: str = "") -> None:
        signature = self._config_signature(spec)
        signature["reason"] = reason
        self.invalid_df = pd.concat([self.invalid_df, pd.DataFrame([signature])], ignore_index=True)
        self.invalid_df.drop_duplicates(subset=_INVALID_CONFIG_KEYS, inplace=True)

    def flush_invalid_configs(self) -> None:
        """Persist the blocklist of structurally invalid pipeline shapes."""
        self.invalid_configs_path.parent.mkdir(parents=True, exist_ok=True)
        self.invalid_df.to_csv(self.invalid_configs_path, index=False)

    # -- pipeline construction ------------------------------------------------------

    def build_pipeline(self, trial) -> Tuple[Any, PipelineSpec]:
        """Assemble the pipeline for ``trial`` from the shared dynamic contract.

        Also used to *replay* a finished trial: Optuna's ``FrozenTrial``
        returns each recorded value instead of resampling, so handing
        ``study.best_trial`` here rebuilds the winning pipeline exactly.
        """
        return build_dynamic_pipeline(
            trial,
            self.estimator,
            numeric_features=self.numeric_features,
            categorical_features=self.categorical_features,
            search_profile=self.search_profile,
            target_strategy=self.target_strategy,
        )

    # -- Optuna objective -----------------------------------------------------------

    @property
    def uses_early_stopping(self) -> bool:
        """Whether this estimator's folds discover their own boosting budget."""
        return self.estimator in _EARLY_STOPPING_ESTIMATORS

    def uses_early_stopping_for(self, spec: PipelineSpec) -> bool:
        """Whether this particular trial delegates its budget to a temporal tail."""
        return self.uses_early_stopping and (spec.boosting_budget_strategy != BOOSTING_BUDGET_FIXED)

    def objective(self, trial) -> float:
        """Weighted MAE across the declared normal-regime selection folds.

        The folds are produced by :func:`temporal_cv_fold_results`, the same
        function used by :meth:`evaluate_best`. Therefore the value ranked by
        Optuna is the declared weighted aggregate of the normal-regime MAEs
        later reported for the winner, and the boosting contract rewarded by
        the search is the contract carried by the persisted artifact.
        """
        pipeline, spec = self.build_pipeline(trial)

        invalid_reason = validate_combination(spec)
        if invalid_reason is not None:
            trial.set_user_attr("failed_reason", invalid_reason)
            self._register_invalid_config(spec, reason=invalid_reason)
            return float("inf")

        if self._was_invalid_config(spec):
            trial.set_user_attr("failed_reason", "Known invalid/slow configuration — skipped")
            return float("inf")

        for key, value in spec.as_tags().items():
            trial.set_user_attr(key, value)

        effective_timeout = self.trial_timeout
        if self._search_deadline is not None:
            remaining_seconds = self._search_deadline - time.monotonic()
            if remaining_seconds <= 0:
                raise optuna.TrialPruned("Cumulative study time budget exhausted")
            effective_timeout = min(effective_timeout, remaining_seconds)

        try:
            folds, err = temporal_cv_with_timeout(
                pipeline,
                self.X_dev,
                self.y_dev,
                self.cv,
                timeout=effective_timeout,
                early_stopping=self.uses_early_stopping_for(spec),
                train_eligible_mask=self.train_eligible_mask,
                score_eligible_mask=self.score_eligible_mask,
                selection_test_years=self.selection_test_years,
            )

            if err is not None:
                trial.set_user_attr("failed_reason", err)
                if not _is_transient_error(err):
                    self._register_invalid_config(spec, reason=err)
                return float("inf")

            frame = pd.DataFrame(folds)
            selected_frame = selection_fold_metrics(frame)
            summary = summarize_cv_fold_metrics(
                selected_frame,
                self._weights_for(len(selected_frame)),
            )
            mae_weighted = summary["cv_mae_weighted"]
            train_r2 = float(frame["train_r2"].mean())
            test_r2 = summary["cv_r2_mean"]
            selection_folds = [fold for fold in folds if fold["fold_role"] == "selection"]
            best_iterations = [fold["best_iteration"] for fold in selection_folds]
            final_iterations = (
                aggregate_iteration_budget(best_iterations)
                if self.uses_early_stopping_for(spec)
                else iteration_ceiling(pipeline)
            )

            for metric_name, metric_value in summary.items():
                trial.set_user_attr(metric_name, metric_value)
            trial.set_user_attr("fold_weights", list(self.fold_weights))
            trial.set_user_attr("train_r2_mean", train_r2)
            trial.set_user_attr("r2_gap", train_r2 - test_r2)
            trial.set_user_attr(
                "fold_mae",
                [float(value) for value in selected_frame["mae"]],
            )
            trial.set_user_attr("best_iterations_by_fold", best_iterations)
            trial.set_user_attr("final_n_estimators", final_iterations)
            truncation = summarize_iteration_truncation(selection_folds)
            trial.set_user_attr("n_folds_cap_hit", truncation["n_folds_cap_hit"])
            trial.set_user_attr("systematic_truncation", truncation["systematic"])
            return mae_weighted
        except Exception as exc:
            reason = str(exc)[:2000]
            trial.set_user_attr("failed_reason", reason)
            if not _is_transient_error(reason):
                self._register_invalid_config(spec, reason=reason)
            return float("inf")

    # -- study lifecycle ------------------------------------------------------------

    def remaining_trials(self, study: "optuna.Study") -> int:
        """Trials still owed to reach the configured total for this study.

        Only trials that have actually started consume the budget. Optuna
        represents enqueued seeds as ``WAITING`` before
        :meth:`Study.optimize` runs them; counting those rows as completed
        would leave a freshly seeded smoke study with no best trial. Failed,
        pruned, running and completed trials have started and therefore count.
        """
        waiting = optuna.trial.TrialState.WAITING
        n_started = sum(getattr(trial, "state", None) != waiting for trial in study.trials)
        return max(0, self.trials - n_started)

    def _log_study_health(self, study: "optuna.Study") -> dict:
        states: Dict[str, int] = {}
        for trial in study.trials:
            states[trial.state.name] = states.get(trial.state.name, 0) + 1
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        finite = [t for t in completed if t.value is not None and np.isfinite(t.value)]
        logger.info(
            "[%s/%s] trial states: %s (%d finite of %d completed)",
            self.estimator,
            self.run_mode,
            states,
            len(finite),
            len(completed),
        )
        if not finite:
            raise RuntimeError(
                f"[{self.estimator}] the study has no finite completed trial "
                f"(states={states}) — no valid configuration was found."
            )
        return {"states": states, "n_finite": len(finite), "n_completed": len(completed)}

    def optimize(self) -> "optuna.Study":
        """Create or resume the persistent study and top it up to ``self.trials``.

        The search stops at whichever limit is reached first — the configured
        trial total or ``study_timeout`` — and which one it was is recorded in
        :attr:`termination_reason` so that a study truncated by the clock is
        never mistaken for one that exhausted its budget.
        """
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage_url,
            load_if_exists=True,
            direction="minimize",
            sampler=self.sampler,
        )
        for params in self.enqueued_trials:
            study.enqueue_trial(
                dict(params),
                user_attrs={"seeded_configuration": True},
                skip_if_exists=True,
            )
        pending = self.remaining_trials(study)
        previous_elapsed = float(study.user_attrs.get(STUDY_ELAPSED_SECONDS_ATTR, 0.0))
        timeout_already_reached = bool(study.user_attrs.get(STUDY_TIMEOUT_REACHED_ATTR, False))
        self.cumulative_elapsed_seconds = previous_elapsed
        remaining_seconds = (
            None
            if self.study_timeout is None
            else max(0.0, float(self.study_timeout) - previous_elapsed)
        )
        if pending == 0:
            logger.info(
                "[%s/%s] study already holds %d of %d configured trials — nothing to run.",
                self.estimator,
                self.run_mode,
                len(study.trials),
                self.trials,
            )
            self.elapsed_seconds = 0.0
            self.termination_reason = TERMINATION_TRIAL_LIMIT
        elif remaining_seconds == 0.0:
            logger.info(
                "[%s/%s] cumulative study timeout already exhausted after %.1fs "
                "— no additional trials will be started.",
                self.estimator,
                self.run_mode,
                previous_elapsed,
            )
            self.elapsed_seconds = 0.0
            self.termination_reason = TERMINATION_STUDY_TIMEOUT
        else:
            if timeout_already_reached:
                logger.info(
                    "[%s/%s] resuming a previously timed-out study because the configured "
                    "cumulative budget increased from %.1fs to %s.",
                    self.estimator,
                    self.run_mode,
                    previous_elapsed,
                    "unlimited" if self.study_timeout is None else f"{self.study_timeout:.1f}s",
                )
            start = time.monotonic()
            self._search_deadline = None if remaining_seconds is None else start + remaining_seconds
            try:
                study.optimize(
                    self.objective,
                    n_trials=pending,
                    timeout=remaining_seconds,
                )
            finally:
                self.elapsed_seconds = time.monotonic() - start
                self.cumulative_elapsed_seconds = previous_elapsed + self.elapsed_seconds
                study.set_user_attr(
                    STUDY_ELAPSED_SECONDS_ATTR,
                    self.cumulative_elapsed_seconds,
                )
                self._search_deadline = None
                self.flush_invalid_configs()
            self.termination_reason = (
                TERMINATION_TRIAL_LIMIT
                if self.remaining_trials(study) == 0
                else TERMINATION_STUDY_TIMEOUT
            )
            study.set_user_attr(
                STUDY_TIMEOUT_REACHED_ATTR,
                self.termination_reason == TERMINATION_STUDY_TIMEOUT,
            )
            logger.info(
                "[%s/%s] ran up to %d trial(s) in %.1fs (%.1fs cumulative; "
                "configured total %d, stopped by %s)",
                self.estimator,
                self.run_mode,
                pending,
                self.elapsed_seconds,
                self.cumulative_elapsed_seconds,
                self.trials,
                self.termination_reason,
            )
        self._log_study_health(study)
        return study

    # -- winning pipeline -----------------------------------------------------------

    def evaluate_best(self, study: "optuna.Study") -> FoldEvaluation:
        """Re-run the winning trial's folds for diagnostics, then refit on all of development.

        The fold pass goes through :func:`temporal_cv_fold_results`, the very
        function the objective scored, so the configured weighted mean of
        ``fold_metrics["mae"]`` reproduces ``study.best_value``. The final refit inherits the folds'
        aggregated boosting budget (:data:`ITERATION_AGGREGATION`) instead of
        the raw sampled ``n_estimators``, which is what makes the persisted
        artifact the same model the search measured.

        Every metric here is a development-fold metric; the artifact refit on
        the whole development set is never scored in this module.
        """
        best_trial = study.best_trial
        pipeline, spec = self.build_pipeline(best_trial)

        folds = temporal_cv_fold_results(
            pipeline,
            self.X_dev,
            self.y_dev,
            self.cv,
            early_stopping=self.uses_early_stopping_for(spec),
            return_predictions=True,
            train_eligible_mask=self.train_eligible_mask,
            score_eligible_mask=self.score_eligible_mask,
            selection_test_years=self.selection_test_years,
        )

        fold_rows: List[dict] = []
        season_rows: List[dict] = []
        extreme_rows: List[dict] = []

        for fold in folds:
            fold_idx = fold["fold"]
            y_true, y_pred = fold["y_true"], fold["y_pred"]
            test_idx = fold["test_index"]

            fold_rows.append(
                {
                    "fold": fold_idx,
                    "test_year": fold["test_year"],
                    "fold_role": fold["fold_role"],
                    "n_train": fold["n_train"],
                    "n_train_excluded": fold["n_train_excluded"],
                    "n_test": fold["n_test"],
                    "n_selection_test": fold["n_selection_test"],
                    "mae": fold["mae"],
                    "rmse": fold["rmse"],
                    "r2": fold["r2"],
                    "selection_mae": fold["selection_mae"],
                    "selection_rmse": fold["selection_rmse"],
                    "selection_r2": fold["selection_r2"],
                    "selection_wape": fold["selection_wape"],
                    "selection_mean_bias": fold["selection_mean_bias"],
                    "wape": fold.get(
                        "wape",
                        float(
                            np.abs(y_true - y_pred).sum()
                            / max(np.abs(y_true).sum(), np.finfo(float).eps)
                        ),
                    ),
                    "mean_bias": fold.get("mean_bias", float(np.mean(y_pred - y_true))),
                    "best_iteration": fold["best_iteration"],
                    "iteration_ceiling": fold["iteration_ceiling"],
                    "best_iteration_cap_hit": fold["best_iteration_cap_hit"],
                }
            )

            selection_mask = fold["selection_test_mask"]
            selected_true = y_true[selection_mask]
            selected_pred = y_pred[selection_mask]
            selected_idx = test_idx[selection_mask]

            if "Seasons" in self.X_dev.columns and len(selected_idx):
                seasons = self.X_dev["Seasons"].iloc[selected_idx].to_numpy()
                for season in pd.unique(seasons):
                    mask = seasons == season
                    if mask.sum() < 30:
                        continue
                    season_rows.append(
                        {
                            "fold": fold_idx,
                            "season": season,
                            "n": int(mask.sum()),
                            "mae": mean_absolute_error(
                                selected_true[mask],
                                selected_pred[mask],
                            ),
                            "rmse": float(
                                np.sqrt(
                                    mean_squared_error(
                                        selected_true[mask],
                                        selected_pred[mask],
                                    )
                                )
                            ),
                            "r2": r2_score(
                                selected_true[mask],
                                selected_pred[mask],
                            ),
                        }
                    )

            if "Temperature(C)" in self.X_dev.columns and len(selected_idx):
                temps = self.X_dev["Temperature(C)"].iloc[selected_idx].to_numpy()
                low_thr, high_thr = np.quantile(temps, [0.05, 0.95])
                for label, mask in (
                    ("cold_extreme", temps <= low_thr),
                    ("hot_extreme", temps >= high_thr),
                ):
                    if mask.sum() < 30:
                        continue
                    extreme_rows.append(
                        {
                            "fold": fold_idx,
                            "band": label,
                            "n": int(mask.sum()),
                            "mae": mean_absolute_error(
                                selected_true[mask],
                                selected_pred[mask],
                            ),
                            "rmse": float(
                                np.sqrt(
                                    mean_squared_error(
                                        selected_true[mask],
                                        selected_pred[mask],
                                    )
                                )
                            ),
                        }
                    )

        selection_folds = [fold for fold in folds if fold["fold_role"] == "selection"]
        best_iterations = [fold["best_iteration"] for fold in selection_folds]
        final_n_estimators = (
            aggregate_iteration_budget(best_iterations)
            if self.uses_early_stopping_for(spec)
            else iteration_ceiling(pipeline)
        )
        truncation = summarize_iteration_truncation(selection_folds)
        if truncation["systematic"]:
            logger.warning(
                "[%s] %d of %d folds with a boosting budget stopped at their %s-iteration "
                "ceiling. The aggregated budget of %s describes the limit, not the data; "
                "raise the ceiling or investigate before freezing this candidate.",
                self.estimator,
                truncation["n_folds_cap_hit"],
                truncation["n_folds_with_budget"],
                truncation["iteration_ceiling"],
                final_n_estimators,
            )

        final_pipeline = clone(pipeline)
        set_iteration_budget(final_pipeline, final_n_estimators)
        final_X = self.X_dev.iloc[self.train_eligible_mask]
        final_y = self.y_dev.iloc[self.train_eligible_mask]
        with suppress_category_encoder_intercept_warning():
            final_pipeline.fit(final_X, final_y)
        spec.n_features_selected = count_output_features(final_pipeline, final_X)
        spec.extra["final_n_estimators"] = final_n_estimators
        spec.extra["iteration_aggregation"] = ITERATION_AGGREGATION
        spec.extra["iteration_ceiling"] = truncation["iteration_ceiling"]
        spec.extra["n_folds_cap_hit"] = truncation["n_folds_cap_hit"]

        fold_frame = pd.DataFrame(fold_rows)
        selected_frame = selection_fold_metrics(fold_frame)
        cv_metrics = summarize_cv_fold_metrics(
            selected_frame,
            self._weights_for(len(selected_frame)),
        )
        cv_metrics["r2_gap"] = best_trial.user_attrs.get("r2_gap")

        return FoldEvaluation(
            best_params=dict(best_trial.params),
            spec=spec,
            fold_metrics=fold_frame,
            seasonal_metrics=pd.DataFrame(season_rows),
            extreme_metrics=pd.DataFrame(extreme_rows),
            fitted_pipeline=final_pipeline,
            trials_completed=len(study.trials),
            cv_metrics={k: v for k, v in cv_metrics.items() if v is not None},
            best_iterations_by_fold=best_iterations,
            final_n_estimators=final_n_estimators,
            cap_hits_by_fold=truncation["cap_hits_by_fold"],
            n_folds_cap_hit=truncation["n_folds_cap_hit"],
            n_folds_with_budget=truncation["n_folds_with_budget"],
            iteration_ceiling=truncation["iteration_ceiling"],
            systematic_truncation=truncation["systematic"],
        )


# ---------------------------------------------------------------------------
# Champion / challenger selection — fail-closed, CV-only
# ---------------------------------------------------------------------------


def _candidate_from_row(row: pd.Series, metric_name: str) -> Dict[str, Any]:
    """Turn one ``mlflow.search_runs`` row into a manifest-shaped candidate."""
    best_params = {
        key[len("params.") :]: value
        for key, value in row.items()
        if key.startswith("params.") and pd.notna(value)
    }
    candidate: Dict[str, Any] = {
        "run_id": row["run_id"],
        "estimator": row.get("tags.estimator", "unknown"),
        "run_mode": row.get("tags.run_mode"),
        "cv_strategy_version": row.get("tags.cv_strategy_version"),
        "code_version": row.get("tags.code_version"),
        "dataset_fingerprint": row.get("tags.dataset_fingerprint"),
        "params_hash": row.get("tags.params_hash"),
        "pipeline_spec_hash": row.get("tags.pipeline_spec_hash"),
        "model_logged": row.get("tags.model_logged"),
        "model_artifact_verified": row.get("tags.model_artifact_verified"),
        "environment_name": row.get("tags.environment_name"),
        "environment_fingerprint": row.get("tags.environment_fingerprint"),
        "python_version": row.get("tags.python_version"),
        "model_uri": f"runs:/{row['run_id']}/model",
        "best_params": best_params,
        metric_name: float(row[f"metrics.{metric_name}"]),
    }
    for spec_key in (
        "modeler_name",
        "encoder",
        "scaler",
        "normalizer",
        "selector",
        "target_transform",
        "boosting_budget_strategy",
        "regime_policy",
        "regime_fingerprint",
        "search_profile",
        "family",
        "n_features_selected",
        "termination_reason",
        "best_iterations_by_fold",
        "final_n_estimators",
        "iteration_aggregation",
        "iteration_ceiling",
        "boosting_cap_hits_by_fold",
        "systematic_truncation",
    ):
        candidate[spec_key] = row.get(f"tags.{spec_key}")
    for metric_key in (
        "cv_mae_mean",
        "cv_mae_weighted",
        "cv_rmse_mean",
        "cv_r2_mean",
        "cv_r2_median",
        "cv_r2_weighted",
        "cv_wape_mean",
        "cv_mean_bias",
        "cv_mean_abs_fold_bias",
        "cv_mae_std",
        "r2_gap",
    ):
        value = row.get(f"metrics.{metric_key}")
        if value is not None and pd.notna(value):
            candidate[metric_key] = float(value)
    for count_key in (
        "trials_planned",
        "trials_completed",
        "n_folds_cap_hit",
        "n_folds_with_budget",
    ):
        value = row.get(f"metrics.{count_key}")
        if value is not None and pd.notna(value):
            candidate[count_key] = int(value)
    return candidate


def select_champion_and_challengers(
    tracker,
    dataset_fingerprint_value: str,
    n_challengers: int = 2,
    metric_name: str = "cv_mae_mean",
    run_mode: str = RUN_MODE_FULL,
    cv_strategy_version: str = CV_STRATEGY_VERSION,
    code_version: str = CODE_VERSION,
    environment_name: str = ENVIRONMENT_NAME,
    environment_fingerprint_value: Optional[str] = None,
    cv_fingerprint_value: Optional[str] = None,
    regime_policy: str = "all_observed",
    regime_fingerprint_value: str = "all_observed",
) -> Dict[str, Any]:
    """Rank finished runs by ascending ``metric_name`` and pick champion plus runners-up.

    Selection is fail-closed. A run only competes if it carries an exact match
    on every provenance tag — dataset fingerprint, CV strategy version, code
    version, run mode, environment name and environment fingerprint — and has
    status ``FINISHED``. A run missing any of those tags is dropped, and if the
    tag is absent from the whole experiment the call raises instead of silently
    ranking everything. This is what stops a smoke run, a run from an older code
    version, a run trained on different data, or a run produced under a
    different Python environment from ever being returned as a champion.

    No holdout metric is read, and ``metric_name`` is rejected outright if it
    names one.
    """
    if "holdout" in metric_name.lower():
        raise ValueError(f"Champion selection must not use a holdout metric (got '{metric_name}').")
    if run_mode not in RUN_MODES:
        raise ValueError(f"run_mode must be one of {RUN_MODES}, got '{run_mode}'.")

    experiment_name = tracker.config.experiment_name
    runs_df = mlflow.search_runs(experiment_names=[experiment_name])
    if runs_df.empty:
        raise RuntimeError(f"No runs found in experiment '{experiment_name}'.")

    missing_tags = [tag for tag in REQUIRED_SELECTION_TAGS if f"tags.{tag}" not in runs_df.columns]
    if missing_tags:
        raise ValueError(
            f"Experiment '{experiment_name}' has no run carrying the required provenance "
            f"tag(s) {missing_tags}. Selection is fail-closed: re-log the runs with "
            "src.tracking.log_temporal_model_run before selecting a champion."
        )

    expected = {
        "dataset_fingerprint": dataset_fingerprint_value,
        "cv_strategy_version": cv_strategy_version,
        "code_version": code_version,
        "run_mode": run_mode,
        "environment_name": environment_name,
        "environment_fingerprint": environment_fingerprint_value or environment_fingerprint(),
        "regime_policy": regime_policy,
        "regime_fingerprint": regime_fingerprint_value,
    }
    if cv_fingerprint_value is not None:
        column = "tags.cv_fingerprint"
        if column not in runs_df.columns:
            raise ValueError(
                "The experiment has no cv_fingerprint tag, so differently weighted "
                "or bounded temporal protocols cannot be separated safely."
            )
        expected["cv_fingerprint"] = cv_fingerprint_value
    git_state = describe_git_source_state()
    expected["git_source_fingerprint"] = git_state["git_source_fingerprint"]
    if run_mode == RUN_MODE_FULL:
        # A definitive candidate must have a retrievable *and* self-contained
        # model: notebook 05 receives a model URI, and a run whose log_model
        # call failed would hand it a URI that does not resolve, while one whose
        # artifact carries no code path or a foreign requirement would hand it a
        # URI that resolves only inside this working copy.
        expected["model_logged"] = "true"
        expected["model_artifact_verified"] = "true"
        expected["git_source_dirty"] = "false"
    eligible = runs_df.dropna(subset=[f"tags.{tag}" for tag in REQUIRED_SELECTION_TAGS])
    for tag, value in expected.items():
        eligible = eligible[eligible[f"tags.{tag}"] == value]
    if "status" in eligible.columns:
        eligible = eligible[eligible["status"] == "FINISHED"]

    metric_col = f"metrics.{metric_name}"
    if metric_col not in eligible.columns:
        raise ValueError(
            f"Metric '{metric_name}' is not present in experiment '{experiment_name}'."
        )
    eligible = eligible.dropna(subset=[metric_col]).sort_values(metric_col, ascending=True)
    if eligible.empty:
        raise RuntimeError(
            f"No FINISHED run in '{experiment_name}' matches {expected} and carries "
            f"metric '{metric_name}'."
        )

    champion = _candidate_from_row(eligible.iloc[0], metric_name)
    challengers = [
        _candidate_from_row(row, metric_name)
        for _, row in eligible.iloc[1 : 1 + n_challengers].iterrows()
    ]
    return {
        "champion": champion,
        "challengers": challengers,
        "run_mode": run_mode,
        "selection_metric": metric_name,
        "dataset_fingerprint": dataset_fingerprint_value,
        "cv_strategy_version": cv_strategy_version,
        "code_version": code_version,
        "environment_name": expected["environment_name"],
        "environment_fingerprint": expected["environment_fingerprint"],
        "regime_policy": expected["regime_policy"],
        "regime_fingerprint": expected["regime_fingerprint"],
    }


# ---------------------------------------------------------------------------
# Candidate freezing — keyed on run_id, provisional in smoke mode
# ---------------------------------------------------------------------------


def candidates_dir_for(run_mode: str, root: Path = DEFAULT_CANDIDATES_ROOT) -> Path:
    """Directory holding the frozen candidates of ``run_mode``."""
    if run_mode not in RUN_MODES:
        raise ValueError(f"run_mode must be one of {RUN_MODES}, got '{run_mode}'.")
    suffix = "" if run_mode == RUN_MODE_FULL else f"_{run_mode}"
    return Path(root) / f"candidates_v4{suffix}"


# What the pipeline's own stamp must reproduce from the selected run's tags.
_PROVENANCE_CHECKS: Tuple[Tuple[str, str], ...] = (
    ("source_run_id", "run_id"),
    ("best_params_hash", "params_hash"),
    ("pipeline_spec_hash", "pipeline_spec_hash"),
    ("code_version", "code_version"),
    ("dataset_fingerprint", "dataset_fingerprint"),
)


def _validate_candidate_pipeline(candidate: Dict[str, Any], pipeline: Any) -> None:
    """Assert that ``pipeline`` really is the artifact of ``candidate``'s run.

    The decisive check is the provenance stamp written onto the object at
    logging time by :func:`src.tracking.stamp_pipeline_provenance`: its
    ``source_run_id``, parameter hash, spec hash, code version and dataset
    fingerprint must all reproduce the selected run's tags. This is what an
    earlier version got wrong — it re-hashed the run's *own* logged parameters,
    which only proved that the run's metadata was self-consistent and would
    have accepted any pipeline whatsoever presented under that ``run_id``.

    An unstamped pipeline is refused outright rather than waved through: the
    stamp is written by the same call that logs the run, so its absence means
    the object did not come from one.

    The regressor-class check against the ``estimator`` tag is kept as a
    cheap, independent second opinion.
    """
    run_id = candidate["run_id"]
    core_pipeline = pipeline
    if isinstance(pipeline, RobustTrendResidualRegressor):
        try:
            core_pipeline = pipeline.estimator_
        except AttributeError as exc:
            raise ValueError(
                f"Robust trend pipeline for run {run_id} is not fitted: its "
                "dynamic estimator_ is missing."
            ) from exc

    try:
        regressor = core_pipeline.named_steps["regressor"].regressor
    except (AttributeError, KeyError) as exc:
        raise ValueError(
            f"Pipeline for run {run_id} has no dynamic 'regressor' step — it "
            "cannot be the artifact of a v4 model-selection run."
        ) from exc

    actual = type(regressor).__name__
    expected = candidate.get("estimator")
    if expected and actual != expected:
        raise ValueError(
            f"Pipeline/run mismatch for run {run_id}: the run is tagged estimator="
            f"'{expected}' but the pipeline's regressor is a '{actual}'."
        )

    provenance = pipeline_provenance(pipeline)
    if provenance is None:
        raise ValueError(
            f"Pipeline for run {run_id} carries no provenance stamp. Only a pipeline "
            "stamped by src.tracking.log_temporal_model_run can be proven to belong "
            "to the run whose metric selected it."
        )

    mismatches = []
    for stamp_key, tag_key in _PROVENANCE_CHECKS:
        expected_value = candidate.get(tag_key)
        if expected_value is None or (
            isinstance(expected_value, float) and pd.isna(expected_value)
        ):
            continue
        if str(provenance.get(stamp_key)) != str(expected_value):
            mismatches.append(f"{stamp_key}={provenance.get(stamp_key)!r} != {expected_value!r}")
    if mismatches:
        raise ValueError(
            f"Pipeline provenance mismatch for run {run_id}: " + "; ".join(mismatches) + ". "
            "The pipeline offered for freezing was produced by a different run, a "
            "different parameter set or a different code version."
        )

    logged_hash = params_hash(candidate.get("best_params", {}))
    tagged_hash = candidate.get("params_hash")
    if tagged_hash and logged_hash != tagged_hash:
        raise ValueError(
            f"Parameter metadata is inconsistent for run {run_id}: the run is tagged "
            f"params_hash={tagged_hash} but its logged parameters hash to {logged_hash}."
        )


def _truncation_reason(candidate: Dict[str, Any]) -> Optional[str]:
    """Describe systematic boosting truncation in ``candidate``, or ``None``.

    Reads the run's own tags rather than recomputing anything, so the check
    also applies to a champion selected across sessions whose folds were never
    in this process's memory.
    """
    n_cap_hit = candidate.get("n_folds_cap_hit")
    n_with_budget = candidate.get("n_folds_with_budget")
    if not n_cap_hit or not n_with_budget:
        return None
    if int(n_cap_hit) * 2 < int(n_with_budget):
        return None
    return (
        f"systematic boosting truncation: {int(n_cap_hit)} of {int(n_with_budget)} folds "
        f"stopped at the {candidate.get('iteration_ceiling')}-iteration ceiling, so the frozen "
        f"budget of {candidate.get('final_n_estimators')} records where the limit was, not where "
        "the validation loss stopped improving"
    )


def _resolve_candidate_pipeline(
    candidate: Dict[str, Any], pipelines_by_run_id: Dict[str, Any]
) -> Tuple[Any, str]:
    """Fetch the pipeline belonging to ``candidate``'s run, by ``run_id`` only.

    The in-memory mapping is consulted first, then the run's own MLflow model
    artifact. The second path is what makes selection across sessions safe: a
    champion logged by an earlier execution has no in-memory pipeline, and
    falling back to its estimator *name* would be exactly the substitution
    this function exists to prevent.
    """
    run_id = candidate["run_id"]
    if run_id in pipelines_by_run_id:
        return pipelines_by_run_id[run_id], "memory"

    model_uri = candidate.get("model_uri") or f"runs:/{run_id}/model"
    try:
        import mlflow.sklearn

        return mlflow.sklearn.load_model(model_uri), "mlflow"
    except Exception as exc:
        raise KeyError(
            f"No pipeline could be resolved for run_id {run_id} "
            f"({candidate.get('estimator')}): it was not supplied in memory and loading "
            f"'{model_uri}' failed ({exc}). Freezing is keyed on run_id so that the "
            "persisted artifact provably belongs to the selected run."
        ) from exc


def freeze_candidates(
    selection: Dict[str, Any],
    pipelines_by_run_id: Dict[str, Any],
    run_mode: str,
    candidates_root: Path = DEFAULT_CANDIDATES_ROOT,
    allow_boosting_truncation: bool = False,
) -> Path:
    """Persist the selected pipelines and write the hand-off manifest.

    Pipelines are looked up by ``run_id``, never by estimator name: two runs
    of the same estimator (a resumed study, a re-executed notebook) would
    otherwise be indistinguishable and the manifest could point at a pipeline
    belonging to a different run than the metric that selected it. A run whose
    pipeline is not in ``pipelines_by_run_id`` is loaded from its own MLflow
    model artifact; if that also fails, the call raises rather than
    substituting anything.

    In ``"smoke"`` mode the artifacts go to their own directory, every file is
    prefixed ``provisional_``, and the manifest is marked ``provisional`` and
    named ``candidates_manifest_smoke.json`` — the definitive manifest that
    notebook 05 consumes is only ever written by a ``"full"`` run.

    Every ``"full"`` finalist showing systematic boosting truncation is
    refused: its frozen tree count would describe the ceiling rather than the
    data, and notebook 05 would validate a model whose size nobody chose.
    Raising the ceiling and re-running is the fix;
    ``allow_boosting_truncation=True`` is the deliberate override for a case
    that has actually been investigated.
    """
    if run_mode not in RUN_MODES:
        raise ValueError(f"run_mode must be one of {RUN_MODES}, got '{run_mode}'.")
    if selection.get("run_mode") != run_mode:
        raise ValueError(
            f"Refusing to freeze: the selection was made in run_mode "
            f"'{selection.get('run_mode')}' but freezing was requested as '{run_mode}'."
        )

    selected_entries = [("champion", selection["champion"])] + [
        ("challenger", candidate) for candidate in selection.get("challengers", [])
    ]
    truncations = [
        {
            "role": role,
            "estimator": candidate.get("estimator"),
            "run_id": candidate.get("run_id"),
            "reason": reason,
        }
        for role, candidate in selected_entries
        if (reason := _truncation_reason(candidate))
    ]
    if truncations and run_mode == RUN_MODE_FULL and not allow_boosting_truncation:
        detail = "; ".join(
            f"{item['role']} {item['estimator']} (run {item['run_id']}): {item['reason']}"
            for item in truncations
        )
        raise RuntimeError(
            "Refusing to freeze definitive candidates because at least one finalist "
            f"has systematic boosting truncation — {detail}. Raise the ceiling and "
            "re-run, or pass allow_boosting_truncation=True once every affected "
            "candidate has been investigated."
        )

    provisional = run_mode != RUN_MODE_FULL
    target_dir = candidates_dir_for(run_mode, candidates_root)
    target_dir.mkdir(parents=True, exist_ok=True)
    prefix = "provisional_" if provisional else ""

    manifest: Dict[str, Any] = {
        "run_mode": run_mode,
        "provisional": provisional,
        "selection_metric": selection.get("selection_metric"),
        "cv_strategy": CV_STRATEGY_NAME,
        "cv_strategy_version": selection.get("cv_strategy_version"),
        "code_version": selection.get("code_version"),
        "dataset_fingerprint": selection.get("dataset_fingerprint"),
        "environment_name": selection.get("environment_name"),
        "environment_fingerprint": selection.get("environment_fingerprint"),
        "regime_policy": selection.get("regime_policy"),
        "regime_fingerprint": selection.get("regime_fingerprint"),
        "boosting_truncation": (
            "; ".join(item["reason"] for item in truncations) if truncations else None
        ),
        "boosting_truncations": truncations,
        "champion": dict(selection["champion"]),
        "challengers": [dict(c) for c in selection.get("challengers", [])],
    }

    entries = [("champion", manifest["champion"])] + [
        ("challenger", candidate) for candidate in manifest["challengers"]
    ]
    for role, candidate in entries:
        run_id = candidate["run_id"]
        pipeline, source = _resolve_candidate_pipeline(candidate, pipelines_by_run_id)
        _validate_candidate_pipeline(candidate, pipeline)
        candidate["pipeline_source"] = source

        filename = f"{prefix}{role}_{candidate.get('estimator')}_{run_id}_pipeline.pkl.gz"
        path = target_dir / filename
        with gzip.open(path, "wb") as handle:
            pickle.dump(pipeline, handle)
        candidate["artifact_path"] = str(path)

    manifest_name = (
        "candidates_manifest.json" if not provisional else ("candidates_manifest_smoke.json")
    )
    manifest_path = target_dir / manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest_path


def pipeline_html_repr(pipeline) -> str:
    """Thin wrapper around ``sklearn.utils.estimator_html_repr`` for artifact logging."""
    return estimator_html_repr(pipeline)
