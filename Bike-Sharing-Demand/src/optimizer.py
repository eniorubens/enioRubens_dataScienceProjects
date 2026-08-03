"""RegressionOptimizer and helpers.

Faithful port of notebook cell [137] (exec_count=79, ~1440 lines).

REGRA DE OURO: port fiel, não reescrita criativa.
- Nenhum search space foi alterado.
- Nenhum transformer ou CV foi trocado.
- Mutações in-place e side-effects preservados (§4).
- globals() utilizados pelo notebook (X_train_opt, y_train_opt, ts_cv,
  X_holdout, y_holdout, max_label, metric_dataframe, start_time,
  pipeline_winner, essemble_estimators, linear_estimators, …) são mantidos
  como referências globais, exatamente como na fonte.

O notebook importa este módulo e popula os globals antes de chamar optimize().
"""

from __future__ import annotations

import gc
import gzip
import logging
import multiprocessing
import pickle
import time
import warnings
from contextlib import contextmanager
from json import dumps
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import optuna
import catboost
import lightgbm
import xgboost
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy.stats import yeojohnson, yeojohnson_normmax
from sklearn import ensemble, linear_model, neighbors, neural_network, svm, tree
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import (
    ColumnTransformer,
    TransformedTargetRegressor,
    make_column_selector,
    make_column_transformer,
)
from sklearn.ensemble import (
    BaggingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    SequentialFeatureSelector,
    mutual_info_regression,
)
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import (
    BayesianRidge,
    Lasso,
    Ridge,
    SGDRegressor,
    TweedieRegressor,
)
from sklearn.metrics import (
    PredictionErrorDisplay,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    PolynomialFeatures,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    SplineTransformer,
    StandardScaler,
)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from category_encoders import (  # type: ignore[import]
    BackwardDifferenceEncoder,
    BaseNEncoder,
    CountEncoder,
    GLMMEncoder,
    GrayEncoder,
    HashingEncoder,
    HelmertEncoder,
    JamesSteinEncoder,
    LeaveOneOutEncoder,
    MEstimateEncoder,
    OrdinalEncoder,
    PolynomialEncoder,
    QuantileEncoder,
    SumEncoder,
)
from feature_engine.encoding import (  # type: ignore[import]
    CountFrequencyEncoder,
    DecisionTreeEncoder,
    MeanEncoder,
    RareLabelEncoder,
)
from feature_engine.transformation import YeoJohnsonTransformer  # type: ignore[import]

from src.periodic_features import (
    CosTransformer,
    CustomPreprocessorWithNystroem,
    DebugTransformer,
    PeriodicSplineTransformer,
    SinTransformer,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Trial/study timeout guard + invalid-config blocklist
# ---------------------------------------------------------------------------

DEFAULT_TRIAL_TIMEOUT = 1800.0  # 30 min — hard cap per individual trial
DEFAULT_STUDY_TIMEOUT = 14400.0  # 4h — cap for the whole study (safety net)
DEFAULT_DETAILED_EVAL_TIMEOUT = 2 * DEFAULT_TRIAL_TIMEOUT  # 1h — hard cap for the
# post-study re-evaluation of the winning pipeline in save_model_and_metrics_regression:
# a cross_validate call (same cost as one trial) *plus* a manual per-fold OOF refit
# loop (same folds again) *plus* one final full-training-data fit against the holdout.
# Since the winning config already survived trial_timeout during the search, its
# single-CV cost is bounded by trial_timeout — budgeting 2x that covers the doubled
# CV cost with headroom for the extra final fit, without leaving this phase
# unbounded like before (a slow winner, e.g. Interactions_with_Kernels +
# SequentialFeatureSelector, could previously blow the per-estimator wall-clock
# budget with no cap at all).

INVALID_CONFIGS_PATH = _PROJECT_ROOT / "dataset" / "invalid_configs.csv"
_INVALID_CONFIG_COLS = ["estimator", "modeler_name", "encoder", "selector", "reason"]

# Error signatures that indicate a transient execution/infra failure (e.g. a bug
# in the timeout-guard subprocess plumbing) rather than a genuine, deterministic
# incompatibility between a pipeline shape and the data (e.g. "Cannot predict
# random effects from singular covariance structure", "max_features == X, must
# be <= Y"). Failures matching these markers must NOT be persisted to
# INVALID_CONFIGS_PATH: unlike structural incompatibilities, they don't
# reliably recur once the underlying bug is fixed, and permanently blocklisting
# them silently shrinks Optuna's effective search space forever.
_TRANSIENT_ERROR_MARKERS = ("daemonic processes are not allowed to have children",)


def _is_transient_error(reason: str) -> bool:
    """Whether ``reason`` looks like a transient infra failure, not a genuine
    pipeline-shape incompatibility (see ``_TRANSIENT_ERROR_MARKERS``)."""
    reason_lower = (reason or "").lower()
    return any(marker in reason_lower for marker in _TRANSIENT_ERROR_MARKERS)


def purge_transient_invalid_configs(path: Path = INVALID_CONFIGS_PATH) -> int:
    """Remove transient-infra-failure rows from the persisted invalid-config blocklist.

    One-time migration for entries written before the transient/genuine
    distinction existed (e.g. "daemonic processes are not allowed to have
    children" rows from the nested-multiprocessing bug). Returns the number
    of rows removed. No-op (returns 0) if the file doesn't exist.
    """
    if not path.exists():
        return 0

    df = pd.read_csv(path)
    if df.empty:
        return 0

    transient_mask = df["reason"].apply(_is_transient_error)
    removed = int(transient_mask.sum())
    if removed:
        df.loc[~transient_mask].to_csv(path, index=False)
    return removed


# ---------------------------------------------------------------------------
# Global estimator category lists (populated by define_estimator)
# ---------------------------------------------------------------------------
tree_estimators: List[str] = []
essemble_estimators: List[str] = []
linear_estimators: List[str] = []
lightgbm_estimators: List[str] = ["LGBMRegressor"]
catboost_estimators: List[str] = ["CatBoostRegressor"]
xgboost_estimators: List[str] = ["XGBRegressor"]
sgdregressor_estimators: List[str] = ["SGDRegressor"]
neighbor_estimators: List[str] = ["KNeighborsRegressor"]
svr_estimator: List[str] = ["SVR"]
neural_network_estimator: List[str] = []

# Shared globals populated by the notebook before calling optimize()
start_time: float = 0
X_train_opt = None
y_train_opt = None
ts_cv = None
X_holdout = None
y_holdout = None
max_label: float = 1.0
metric_dataframe = None
pipeline_winner = None


# ---------------------------------------------------------------------------
# Context manager — suppress category_encoders FutureWarning (cell [137])
# ---------------------------------------------------------------------------


@contextmanager
def suppress_category_encoder_intercept_warning():
    """Suppress category_encoders contrast-encoder intercept FutureWarning locally."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Intercept column might not be added anymore in future releases.*",
            category=FutureWarning,
            module=r"category_encoders\.base_contrast_encoder",
        )
        yield


# ---------------------------------------------------------------------------
# HashingEncoder without category_encoders' internal multiprocessing.Manager()
# ---------------------------------------------------------------------------


class _SingleProcessHashingEncoder(HashingEncoder):
    """HashingEncoder that never touches multiprocessing.

    ``category_encoders.HashingEncoder._transform`` unconditionally creates a
    ``multiprocessing.Manager()`` (which itself spawns a manager server
    process) before it even checks ``max_process`` — so ``max_process=1``
    does NOT avoid the spawn. Since each trial's ``cross_validate`` already
    runs inside a ``daemon=True`` subprocess (see
    ``_run_in_subprocess_with_timeout``), and daemonic processes cannot spawn
    children, every trial sampling ``HashingEncoder`` crashed with "daemonic
    processes are not allowed to have children" regardless of max_process.
    This override replicates exactly what the multi-process path computes
    (single call to ``hashing_trick`` over the whole frame — equivalent to
    what ``require_data`` does when ``max_process == 1``, just without the
    Manager()-mediated chunking) so results are unchanged.
    """

    def _transform(self, X):
        return self.hashing_trick(
            X, hashing_method=self.hash_method, N=self.n_components, cols=self.cols
        )


# ---------------------------------------------------------------------------
# save_model_and_metrics_regression (cell [137])
# ---------------------------------------------------------------------------


def save_model_and_metrics_regression(
    description: str,
    preprocessing: str,
    pipe: object,
    regressor: object,
    target_transformer: object,
    params: dict,
    feature_selector: object = None,
    timeout: float = DEFAULT_DETAILED_EVAL_TIMEOUT,
):
    """Evaluate inner CV on temporal training data and report final holdout performance.

    Uses module-level globals: X_train_opt, y_train_opt, ts_cv,
    X_holdout, y_holdout, max_label, metric_dataframe, start_time.
    These are set by the notebook before calling optimizer.optimize().

    The CV metrics, out-of-fold predictions, and final holdout fit are all
    produced by a single subprocess call guarded by ``timeout`` — this used to
    run unguarded in-process, so a slow winning pipeline had no cap on how long
    this post-study step could take (see DEFAULT_DETAILED_EVAL_TIMEOUT). The CV
    metrics and the out-of-fold predictions come from a *single* per-fold pass
    (each fold is fit exactly once) — this used to be two separate full CV
    passes (a cross_validate call, then a second manual refit loop just for the
    OOF predictions needed by the left-hand plot), doubling the CV cost for no
    reason.
    """
    global start_time

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    estimator = Pipeline(
        steps=[
            ("transformer", pipe),
            ("imputer", SimpleImputer(strategy="median")),
            (
                "selector",
                clone(feature_selector) if feature_selector is not None else "passthrough",
            ),
            (
                "regressor",
                TransformedTargetRegressor(
                    regressor=clone(regressor),
                    transformer=clone(target_transformer),
                ),
            ),
        ]
    )

    metrics, y_pred_cv, final_estimator, y_train_fit, y_holdout_pred = _detailed_fit_with_timeout(
        estimator,
        X_train_opt,
        y_train_opt,
        X_holdout,
        ts_cv,
        timeout=timeout,
    )

    valid_mask = ~np.isnan(y_pred_cv)

    PredictionErrorDisplay.from_predictions(
        y_true=y_train_opt.iloc[valid_mask] * max_label,
        y_pred=y_pred_cv[valid_mask] * max_label,
        kind="actual_vs_predicted",
        ax=axes[0],
        scatter_kwargs={"alpha": 0.1, "color": "tab:blue"},
        line_kwargs={"color": "tab:red"},
    )
    axes[0].set_title(f"Inner CV (Train Window): {description}")

    inner_scores = {
        "R2": f"{np.abs(np.mean(metrics['test_r2'] * 100)):.4f}% +- {np.std(metrics['test_r2']):.4f}",
        "MAE": f"{np.abs(np.mean(metrics['test_neg_mean_absolute_error'] * 100)):.4f}% +- {np.std(metrics['test_neg_mean_absolute_error']):.4f}",
        "MSE": f"{np.abs(np.mean(metrics['test_neg_mean_squared_error'] * 100)):.4f}% +- {np.std(metrics['test_neg_mean_squared_error']):.4f}",
        "RMSE": f"{np.abs(np.mean(metrics['test_neg_root_mean_squared_error'] * 100)):.4f}% +- {np.std(metrics['test_neg_root_mean_squared_error']):.4f}",
    }

    inner_metrics_text = "\n".join([f"{name}: {score}" for name, score in inner_scores.items()])
    axes[0].text(
        0.03,
        0.95,
        inner_metrics_text,
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="black"),
    )

    holdout_mae = mean_absolute_error(y_holdout, y_holdout_pred)
    holdout_mse = mean_squared_error(y_holdout, y_holdout_pred)
    holdout_rmse = np.sqrt(holdout_mse)
    holdout_r2 = r2_score(y_holdout, y_holdout_pred)

    train_mae_final = mean_absolute_error(y_train_opt, y_train_fit)
    train_mse_final = mean_squared_error(y_train_opt, y_train_fit)
    train_rmse_final = np.sqrt(train_mse_final)
    train_r2_final = r2_score(y_train_opt, y_train_fit)

    PredictionErrorDisplay.from_predictions(
        y_true=y_holdout * max_label,
        y_pred=y_holdout_pred * max_label,
        kind="actual_vs_predicted",
        ax=axes[1],
        scatter_kwargs={"alpha": 0.2, "color": "tab:green"},
        line_kwargs={"color": "tab:red"},
    )
    axes[1].set_title(f"Final Temporal Holdout: {description}")

    holdout_scores = {
        "R2": f"{holdout_r2 * 100:.4f}%",
        "MAE": f"{holdout_mae * 100:.4f}%",
        "MSE": f"{holdout_mse * 100:.4f}%",
        "RMSE": f"{holdout_rmse * 100:.4f}%",
    }

    holdout_metrics_text = "\n".join([f"{name}: {score}" for name, score in holdout_scores.items()])
    axes[1].text(
        0.03,
        0.95,
        holdout_metrics_text,
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="black"),
    )

    elapsed_time = time.time() - start_time
    fig.suptitle(
        f"Single Predictor : {description}\nEvaluation in {elapsed_time:.2f} seconds",
        fontsize=18,
        fontweight="bold",
    )

    metric_dataframe.loc[(description, "Optuna-InnerCV", preprocessing), :] = [
        np.mean(-metrics["train_neg_mean_absolute_error"]),
        np.std(-metrics["train_neg_mean_absolute_error"]),
        np.mean(-metrics["test_neg_mean_absolute_error"]),
        np.std(-metrics["test_neg_mean_absolute_error"]),
        np.mean(-metrics["train_neg_mean_squared_error"]),
        np.std(-metrics["train_neg_mean_squared_error"]),
        np.mean(-metrics["test_neg_mean_squared_error"]),
        np.std(-metrics["test_neg_mean_squared_error"]),
        np.mean(-metrics["train_neg_root_mean_squared_error"]),
        np.std(-metrics["train_neg_root_mean_squared_error"]),
        np.mean(-metrics["test_neg_root_mean_squared_error"]),
        np.std(-metrics["test_neg_root_mean_squared_error"]),
        np.mean(metrics["train_r2"]),
        np.std(metrics["train_r2"]),
        np.mean(metrics["test_r2"]),
        np.std(metrics["test_r2"]),
        str(params),
    ]

    metric_dataframe.loc[(description, "Temporal-Holdout", preprocessing), :] = [
        train_mae_final,
        0.0,
        holdout_mae,
        0.0,
        train_mse_final,
        0.0,
        holdout_mse,
        0.0,
        train_rmse_final,
        0.0,
        holdout_rmse,
        0.0,
        train_r2_final,
        0.0,
        holdout_r2,
        0.0,
        str(params),
    ]

    metric_dataframe.to_csv(_PROJECT_ROOT / "dataset" / "metric_dataframe.csv", index=True)

    with gzip.open(_PROJECT_ROOT / "models" / f"{description}_estimator.pkl.gz", "wb") as f:
        pickle.dump(regressor, f)

    with gzip.open(_PROJECT_ROOT / "models" / f"{description}_pipeline.pkl.gz", "wb") as f:
        pickle.dump(pipe, f)

    with gzip.open(_PROJECT_ROOT / "models" / f"{description}_transformer.pkl.gz", "wb") as f:
        pickle.dump(target_transformer, f)

    with gzip.open(_PROJECT_ROOT / "models" / f"{description}_full_pipeline.pkl.gz", "wb") as f:
        pickle.dump(final_estimator, f)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()

    print(f"Evaluation completed in {elapsed_time:.2f} seconds")

    return metrics


# ---------------------------------------------------------------------------
# define_estimator / get_regressor (cell [137])
# ---------------------------------------------------------------------------


def define_estimator() -> List[str]:
    """Populate module-level estimator category lists and return a combined list."""
    global tree_estimators, essemble_estimators, linear_estimators, lightgbm_estimators, catboost_estimators, xgboost_estimators, sgdregressor_estimators, neighbor_estimators, svr_estimator, neural_network_estimator

    tree_estimators = [e for e in list(set(tree.__all__)) if "Regressor" in e]
    essemble_estimators = [e for e in list(set(ensemble.__all__)) if "Regressor" in e]
    linear_estimators = [
        e
        for e in list(set(linear_model.__all__))
        if ("Regressor" in e) or ("Regression" in e) or ("BayesianRidge" in e)
    ] + ["Ridge", "Lasso"]

    lightgbm_estimators = ["LGBMRegressor"]
    catboost_estimators = ["CatBoostRegressor"]
    xgboost_estimators = ["XGBRegressor"]
    sgdregressor_estimators = ["SGDRegressor"]
    neighbor_estimators = ["KNeighborsRegressor"]
    svr_estimator = ["SVR"]
    neural_network_estimator = [e for e in list(set(neural_network.__all__)) if "Regressor" in e]

    estimators = (
        tree_estimators
        + essemble_estimators
        + linear_estimators
        + lightgbm_estimators
        + catboost_estimators
        + xgboost_estimators
        + neighbor_estimators
        + svr_estimator
        + neural_network_estimator
    )
    return estimators


def get_regressor(estimator_name: str):
    """Return the regressor class for a given estimator name string."""
    define_estimator()

    if estimator_name in essemble_estimators:
        return getattr(ensemble, estimator_name)

    elif estimator_name in catboost_estimators:
        return getattr(catboost, estimator_name)

    elif estimator_name in xgboost_estimators:
        return getattr(xgboost, estimator_name)

    elif estimator_name in linear_estimators:
        return getattr(linear_model, estimator_name)

    elif estimator_name in lightgbm_estimators:
        return getattr(lightgbm, estimator_name)

    elif estimator_name in neighbor_estimators:
        return getattr(neighbors, estimator_name)

    elif estimator_name in tree_estimators:
        return getattr(tree, estimator_name)

    elif estimator_name in svr_estimator:
        return getattr(svm, estimator_name)

    elif estimator_name in neural_network_estimator:
        return getattr(neural_network, estimator_name)


# ---------------------------------------------------------------------------
# Subprocess-based hard timeout for a single trial's cross_validate call
# ---------------------------------------------------------------------------


def _run_in_subprocess_with_timeout(target, args, timeout, on_progress=None, poll_interval=5.0):
    """Run ``target(*args, result_queue)`` in a spawned subprocess, killing it if it
    exceeds ``timeout``.

    Polls the queue in ``poll_interval``-second slices (instead of one big blocking
    ``join(timeout)``) so the target can optionally report intermediate progress via
    ``result_queue.put(("progress", message))`` — each such message is forwarded to
    ``on_progress`` and does NOT count as the final result. Targets that never send
    progress messages (e.g. the per-trial cross_validate call) behave identically to
    the old single-join implementation.

    Returns ``(status, payload)`` where status is one of ``"ok"``, ``"timeout"``,
    or ``"error"``. Shared low-level plumbing for both the per-trial and the
    post-study "detailed evaluation" timeout guards.
    """
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=target, args=(*args, q), daemon=True)
    p.start()

    deadline = time.time() + timeout
    result = None

    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        try:
            status, payload = q.get(timeout=min(poll_interval, remaining))
        except Exception:
            if not p.is_alive():
                break
            continue
        if status == "progress":
            if on_progress is not None:
                on_progress(payload)
            continue
        result = (status, payload)
        break

    if result is not None:
        p.join(5)
        q.close()
        q.join_thread()
        return result

    if p.is_alive():
        p.terminate()
        p.join(5)
        if p.is_alive():
            p.kill()
            p.join()
        q.close()
        q.join_thread()
        return "timeout", f"exceeded timeout of {timeout}s and was terminated"

    q.close()
    q.join_thread()
    return "error", "Subprocess finished but produced no result (crashed or queue error)"


def _run_cross_validate_in_subprocess(estimator, X, y, cv, scoring, result_queue) -> None:
    """Target for the timeout-guarded subprocess.

    Runs in a freshly spawned interpreter — must not rely on module globals
    (``X_train_opt`` etc. are unset there). All inputs are passed explicitly.
    """
    try:
        with suppress_category_encoder_intercept_warning(), joblib.parallel_backend("threading"):
            cv_results = cross_validate(
                estimator,
                X,
                y,
                cv=cv,
                scoring=scoring,
                error_score="raise",
                return_train_score=True,
            )
        result_queue.put(("ok", cv_results))
    except Exception as exc:
        result_queue.put(("error", str(exc)[:2000]))


def _cross_validate_with_timeout(estimator, X, y, cv, scoring, timeout):
    """Run cross_validate in a subprocess, killing it if it exceeds ``timeout``.

    Returns ``(cv_results, None)`` on success or ``(None, error_message)``
    on timeout/failure — never raises, so callers don't need a try/except
    around the timeout mechanics themselves.
    """
    status, payload = _run_in_subprocess_with_timeout(
        _run_cross_validate_in_subprocess,
        (estimator, X, y, cv, scoring),
        timeout,
    )
    if status == "ok":
        return payload, None
    if status == "timeout":
        return None, f"Trial exceeded timeout of {timeout}s and was terminated"
    return None, payload


# ---------------------------------------------------------------------------
# Temporally-ordered early stopping for XGBRegressor/LGBMRegressor trials
# ---------------------------------------------------------------------------

# Estimators whose Optuna trials get a per-fold early-stopping validation carve
# (see _run_cross_validate_with_early_stopping_in_subprocess). Neither XGB nor
# LGBM had any early stopping before, so both trained their full sampled
# n_estimators every trial regardless of whether later trees were still
# improving generalization — a plausible contributor to their ~0.12-0.14
# train/test R2 gap. HistGradientBoostingRegressor already has native
# early_stopping in its search space and is left out of this path.
_EARLY_STOPPING_ESTIMATORS = ("XGBRegressor", "LGBMRegressor")


def _run_cross_validate_with_early_stopping_in_subprocess(
    estimator,
    X,
    y,
    cv,
    es_val_fraction,
    es_gap,
    es_rounds,
    result_queue,
) -> None:
    """Target for the timeout-guarded subprocess — early-stopping-aware variant
    of ``_run_cross_validate_in_subprocess`` for XGBRegressor/LGBMRegressor trials.

    For each ``cv`` fold, carves a temporally-ordered tail of that fold's own
    ``train_idx`` (with a small gap, mirroring ``make_temporal_holdout_split``'s
    ``holdout_gap`` convention) as an early-stopping validation slice, fits the
    pipeline's preprocessing steps + regressor on the earlier remainder, and
    monitors early stopping on that slice. The fold's real ``test_idx`` is then
    scored with that same early-stopped model — the set of rows Optuna's
    objective ultimately sees (the fold's test partition) is unchanged; only
    the model producing those predictions now stops before it starts
    memorizing the training window.

    Runs in a freshly spawned interpreter — must not rely on module globals.
    Returns a dict shaped like ``sklearn.model_selection.cross_validate``'s
    output (``return_train_score=True``, scoring=[neg_mean_absolute_error,
    neg_root_mean_squared_error, r2]) plus a ``best_iterations`` list, so
    ``objective()`` can treat both CV paths uniformly.
    """
    try:
        with suppress_category_encoder_intercept_warning(), joblib.parallel_backend("threading"):
            fold_scores = {
                "train_neg_mean_absolute_error": [],
                "test_neg_mean_absolute_error": [],
                "train_neg_root_mean_squared_error": [],
                "test_neg_root_mean_squared_error": [],
                "train_r2": [],
                "test_r2": [],
            }
            best_iterations: list = []

            for fold_train_idx, fold_test_idx in cv.split(X, y):
                n_val = max(int(len(fold_train_idx) * es_val_fraction), 50)
                if len(fold_train_idx) <= n_val + es_gap:
                    raise ValueError(
                        f"Fold train window has only {len(fold_train_idx)} rows — too small "
                        f"to carve a {n_val}-row early-stopping slice plus a {es_gap}-row gap."
                    )
                es_val_idx = fold_train_idx[-n_val:]
                es_fit_idx = fold_train_idx[: -(n_val + es_gap)]

                fold_estimator = clone(estimator)
                preproc = Pipeline(fold_estimator.steps[:-1])
                X_fit_t = preproc.fit_transform(X.iloc[es_fit_idx], y.iloc[es_fit_idx])
                X_val_t = preproc.transform(X.iloc[es_val_idx])
                X_test_t = preproc.transform(X.iloc[fold_test_idx])

                ttr = fold_estimator.named_steps["regressor"]
                y_fit_raw = y.iloc[es_fit_idx].to_numpy().reshape(-1, 1)
                y_val_raw = y.iloc[es_val_idx].to_numpy().reshape(-1, 1)

                if ttr.transformer is not None:
                    target_transformer = clone(ttr.transformer)
                    target_transformer.fit(y_fit_raw)
                    y_fit_t = target_transformer.transform(y_fit_raw).ravel()
                    y_val_t = target_transformer.transform(y_val_raw).ravel()
                else:
                    target_transformer = None
                    y_fit_t = y_fit_raw.ravel()
                    y_val_t = y_val_raw.ravel()

                regressor = clone(ttr.regressor)
                if isinstance(regressor, XGBRegressor):
                    regressor.set_params(early_stopping_rounds=es_rounds)
                    regressor.fit(X_fit_t, y_fit_t, eval_set=[(X_val_t, y_val_t)], verbose=False)
                    best_iterations.append(getattr(regressor, "best_iteration", None))
                elif isinstance(regressor, LGBMRegressor):
                    # eval_metric="l1" guards against trial params carrying an
                    # invalid `metric` value: the early-stopping callback needs at
                    # least one valid eval metric or every fit raises.
                    regressor.fit(
                        X_fit_t,
                        y_fit_t,
                        eval_set=[(X_val_t, y_val_t)],
                        eval_metric="l1",
                        callbacks=[
                            lightgbm.early_stopping(stopping_rounds=es_rounds, verbose=False)
                        ],
                    )
                    best_iterations.append(getattr(regressor, "best_iteration_", None))
                else:
                    regressor.fit(X_fit_t, y_fit_t)
                    best_iterations.append(None)

                y_fit_pred_t = regressor.predict(X_fit_t)
                y_test_pred_t = regressor.predict(X_test_t)
                if target_transformer is not None:
                    y_fit_pred = target_transformer.inverse_transform(
                        y_fit_pred_t.reshape(-1, 1)
                    ).ravel()
                    y_test_pred = target_transformer.inverse_transform(
                        y_test_pred_t.reshape(-1, 1)
                    ).ravel()
                else:
                    y_fit_pred = y_fit_pred_t
                    y_test_pred = y_test_pred_t

                y_fit_true = y.iloc[es_fit_idx].to_numpy()
                y_test_true = y.iloc[fold_test_idx].to_numpy()

                train_mse = mean_squared_error(y_fit_true, y_fit_pred)
                test_mse = mean_squared_error(y_test_true, y_test_pred)
                fold_scores["train_neg_mean_absolute_error"].append(
                    -mean_absolute_error(y_fit_true, y_fit_pred)
                )
                fold_scores["test_neg_mean_absolute_error"].append(
                    -mean_absolute_error(y_test_true, y_test_pred)
                )
                fold_scores["train_neg_root_mean_squared_error"].append(-np.sqrt(train_mse))
                fold_scores["test_neg_root_mean_squared_error"].append(-np.sqrt(test_mse))
                fold_scores["train_r2"].append(r2_score(y_fit_true, y_fit_pred))
                fold_scores["test_r2"].append(r2_score(y_test_true, y_test_pred))

            cv_results = {name: np.array(values) for name, values in fold_scores.items()}
            cv_results["best_iterations"] = best_iterations
        result_queue.put(("ok", cv_results))
    except Exception as exc:
        result_queue.put(("error", str(exc)[:2000]))


def _cross_validate_with_early_stopping_timeout(
    estimator,
    X,
    y,
    cv,
    timeout,
    es_val_fraction=0.12,
    es_gap=48,
    es_rounds=30,
):
    """Early-stopping-aware counterpart of ``_cross_validate_with_timeout``.

    Same contract: returns ``(cv_results, None)`` on success or
    ``(None, error_message)`` on timeout/failure.
    """
    status, payload = _run_in_subprocess_with_timeout(
        _run_cross_validate_with_early_stopping_in_subprocess,
        (estimator, X, y, cv, es_val_fraction, es_gap, es_rounds),
        timeout,
    )
    if status == "ok":
        return payload, None
    if status == "timeout":
        return None, f"Trial exceeded timeout of {timeout}s and was terminated"
    return None, payload


def _run_detailed_fit_in_subprocess(estimator, X, y, X_holdout, cv, result_queue) -> None:
    """Target for the timeout-guarded subprocess used by ``save_model_and_metrics_regression``.

    Computes the winning pipeline's inner-CV metrics *and* out-of-fold
    predictions from a single per-fold pass (each fold fit exactly once), then
    does one final fit on the whole training window scored against the
    temporal holdout.

    This used to be TWO separate full CV passes over the same folds: a
    ``cross_validate`` call for the metrics, followed by a manual refit loop
    that repeated the exact same fold fits just to get out-of-fold prediction
    arrays for the left-hand plot (``cross_validate`` only returns scores, not
    predictions). Since both need the same per-fold fit, merging them into one
    loop halves the CV-related cost of this phase — cross_val_predict isn't an
    option here because TimeSeriesSplit-style folds don't cover every row (the
    earliest rows are never in any test fold), which is exactly why the
    original code used a manual loop with a NaN-filled array in the first
    place.

    Emits a ("progress", message) on ``result_queue`` after each of the two
    remaining stages so the caller can see *which* one is slow instead of the
    whole call being an opaque multi-hour black box: the final full-fit is
    often the single most expensive step since it sees more rows than any
    individual CV fold, and if the winning selector is RFE/SelectFromModel
    wrapping the same (possibly slow) base estimator, both stages internally
    fit it many more times over.
    """
    try:
        with suppress_category_encoder_intercept_warning(), joblib.parallel_backend("threading"):
            stage_start = time.time()

            fold_metrics = {
                "train_neg_mean_absolute_error": [],
                "test_neg_mean_absolute_error": [],
                "train_neg_mean_squared_error": [],
                "test_neg_mean_squared_error": [],
                "train_neg_root_mean_squared_error": [],
                "test_neg_root_mean_squared_error": [],
                "train_r2": [],
                "test_r2": [],
            }
            y_pred_cv = np.full(shape=len(y), fill_value=np.nan, dtype=float)

            for fold_train_idx, fold_test_idx in cv.split(X, y):
                fold_estimator = clone(estimator)
                fold_estimator.fit(X.iloc[fold_train_idx], y.iloc[fold_train_idx])

                y_train_true = y.iloc[fold_train_idx]
                y_test_true = y.iloc[fold_test_idx]
                y_train_pred = fold_estimator.predict(X.iloc[fold_train_idx])
                y_test_pred = fold_estimator.predict(X.iloc[fold_test_idx])
                y_pred_cv[fold_test_idx] = y_test_pred

                train_mse = mean_squared_error(y_train_true, y_train_pred)
                test_mse = mean_squared_error(y_test_true, y_test_pred)

                fold_metrics["train_neg_mean_absolute_error"].append(
                    -mean_absolute_error(y_train_true, y_train_pred)
                )
                fold_metrics["test_neg_mean_absolute_error"].append(
                    -mean_absolute_error(y_test_true, y_test_pred)
                )
                fold_metrics["train_neg_mean_squared_error"].append(-train_mse)
                fold_metrics["test_neg_mean_squared_error"].append(-test_mse)
                fold_metrics["train_neg_root_mean_squared_error"].append(-np.sqrt(train_mse))
                fold_metrics["test_neg_root_mean_squared_error"].append(-np.sqrt(test_mse))
                fold_metrics["train_r2"].append(r2_score(y_train_true, y_train_pred))
                fold_metrics["test_r2"].append(r2_score(y_test_true, y_test_pred))

            metrics = {name: np.array(values) for name, values in fold_metrics.items()}
            result_queue.put(
                (
                    "progress",
                    f"single-pass per-fold CV (metrics + OOF predictions) done in {time.time() - stage_start:.1f}s",
                )
            )

            stage_start = time.time()
            final_estimator = clone(estimator)
            final_estimator.fit(X, y)
            y_train_fit = final_estimator.predict(X)
            y_holdout_pred = final_estimator.predict(X_holdout)
            result_queue.put(
                (
                    "progress",
                    f"final full-train fit + holdout predict done in {time.time() - stage_start:.1f}s",
                )
            )

        result_queue.put(("ok", (metrics, y_pred_cv, final_estimator, y_train_fit, y_holdout_pred)))
    except Exception as exc:
        result_queue.put(("error", str(exc)[:2000]))


def _detailed_fit_with_timeout(estimator, X, y, X_holdout, cv, timeout):
    """Run the detailed post-study re-evaluation in a subprocess, killing it if it
    exceeds ``timeout``.

    Unlike ``_cross_validate_with_timeout``, this raises on timeout/failure:
    there's no "just skip this trial" fallback once we've already committed to
    the winning config — the caller (``save_model_and_metrics_regression``)
    previously had no error handling here either, since ``error_score="raise"``
    meant a failing cross_validate call already propagated.
    """
    status, payload = _run_in_subprocess_with_timeout(
        _run_detailed_fit_in_subprocess,
        (estimator, X, y, X_holdout, cv),
        timeout,
        on_progress=lambda msg: logger.info("[detailed-eval] %s", msg),
    )
    if status == "ok":
        return payload
    if status == "timeout":
        raise TimeoutError(
            f"Detailed evaluation of the winning pipeline exceeded timeout of {timeout}s and was terminated."
        )
    raise RuntimeError(f"Detailed evaluation of the winning pipeline failed: {payload}")


# ---------------------------------------------------------------------------
# Module-level helper for the Pairwise_Interactions Hour x Weekday cross term
# (must be a picklable module-level function, not a lambda/bound method,
# since trials run inside a spawned subprocess via _cross_validate_with_timeout)
# ---------------------------------------------------------------------------


def _is_weekday_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df["Weekday"].eq("Weekday").astype(float).to_frame()


# ---------------------------------------------------------------------------
# RegressionOptimizer (cell [137])
# ---------------------------------------------------------------------------


class RegressionOptimizer:
    """Hyperparameter optimiser for regression models using Optuna.

    Attributes
    ----------
    estimator : str
        Name of the regression model to optimise.
    trials : int
        Number of Optuna trials.
    sampler : optuna.samplers.TPESampler
        TPE sampler with seed=42 for reproducibility.
    """

    # Single source of truth for the base feature columns fed into every
    # ColumnTransformer branch of modeling_transformers(). get_encoder() reuses
    # _CATEGORICAL_FEATURES so the feature_engine encoders (MeanEncoder,
    # CountFrequencyEncoder) never expect columns the pipeline never gives them.
    _NUMERICAL_FEATURES = [
        "Temperature(C)",
        "Dew point temperature(C)",
        "Ground Temp(C)",
        "Humidity(%)",
        "Solar Radiation (MJ/m2)",
        "Wind speed (m/s)",
        "Month",
        "Hour",
    ]
    _CATEGORICAL_FEATURES = [
        "Holiday",
        "Seasons",
        "Functioning Day",
        "Weekday",
        "Rainfall Cat",
        "Snowfall Cat",
        "WeekStatus",
        "Time_Period",
        "Rush_Hour",
        "Rush_Period",
        "DayNumberOnWeek",
    ]

    # Preprocessing branches that expand dimensionality a lot (Nystroem produces
    # 300 components; PolynomialFeatures(degree=2) and the Hour interaction
    # splines add dozens more) — never let the selector search skip feature
    # selection entirely for these.
    _HIGH_DIM_MODELERS = {"Interactions_with_Kernels", "Polynomial", "Pairwise_Interactions"}

    def __init__(
        self,
        estimator: str,
        trials: int = 100,
        trial_timeout: float = DEFAULT_TRIAL_TIMEOUT,
        study_timeout: Optional[float] = DEFAULT_STUDY_TIMEOUT,
        detailed_timeout: float = DEFAULT_DETAILED_EVAL_TIMEOUT,
    ) -> None:
        self.estimator = estimator
        self.trials = trials
        self.trial_timeout = trial_timeout
        self.study_timeout = study_timeout
        self.detailed_timeout = detailed_timeout
        self.sampler = optuna.samplers.TPESampler(seed=42)
        define_estimator()

        if INVALID_CONFIGS_PATH.exists():
            self.invalid_df = pd.read_csv(INVALID_CONFIGS_PATH)
        else:
            self.invalid_df = pd.DataFrame(columns=_INVALID_CONFIG_COLS)

    def _config_signature(self, modeler_name, encoder_name, selector_name) -> dict:
        """Build a config signature ignoring numeric hyperparameters (alphas, sfs_n_features, ...)."""
        return {
            "estimator": self.estimator,
            "modeler_name": modeler_name or "None",
            "encoder": encoder_name or "None",
            "selector": selector_name or "None",
        }

    def was_invalid_config(self, modeler_name, encoder_name, selector_name) -> bool:
        """Check whether this pipeline shape previously timed out or errored."""
        if self.invalid_df.empty:
            return False

        sig = self._config_signature(modeler_name, encoder_name, selector_name)
        mask = (
            (self.invalid_df["estimator"] == sig["estimator"])
            & (self.invalid_df["modeler_name"] == sig["modeler_name"])
            & (self.invalid_df["encoder"] == sig["encoder"])
            & (self.invalid_df["selector"] == sig["selector"])
        )
        return bool(mask.any())

    def register_invalid_config(
        self, modeler_name, encoder_name, selector_name, reason: str = ""
    ) -> None:
        """Record a pipeline shape as invalid so future trials skip it."""
        sig = self._config_signature(modeler_name, encoder_name, selector_name)
        sig["reason"] = reason

        self.invalid_df = pd.concat([self.invalid_df, pd.DataFrame([sig])], ignore_index=True)
        self.invalid_df.drop_duplicates(
            subset=["estimator", "modeler_name", "encoder", "selector"], inplace=True
        )

    def flush_invalid_configs(self) -> None:
        """Persist the invalid-config blocklist to disk."""
        INVALID_CONFIGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.invalid_df.to_csv(INVALID_CONFIGS_PATH, index=False)

    def is_weekday(self, x: str) -> bool:
        return x == "Weekday"

    def modeling_transformers(self, trial) -> Tuple[Any, str, Any]:
        """Suggest preprocessing pipeline strategy for the current trial."""
        define_estimator()
        modeling_params: dict = {}

        if self.estimator in essemble_estimators:
            modeling_params = {
                "modeler_name": trial.suggest_categorical(
                    "modeler_name",
                    (
                        "Normalizers",
                        "Polynomial",
                        "linear_modeling",
                        "Sin_Cos",
                        "Time_steps_as_categories",
                        "Periodic_Spline",
                        "Interactions_with_Kernels",
                    ),
                )
            }

        elif self.estimator in lightgbm_estimators:
            modeling_params = {
                "modeler_name": trial.suggest_categorical(
                    "modeler_name",
                    ("linear_modeling", "Sin_Cos", "Time_steps_as_categories", "Periodic_Spline"),
                )
            }

        elif self.estimator in catboost_estimators:
            modeling_params = {
                "modeler_name": trial.suggest_categorical(
                    "modeler_name",
                    ("linear_modeling", "Sin_Cos", "Time_steps_as_categories", "Periodic_Spline"),
                )
            }

        elif self.estimator in sgdregressor_estimators:
            modeling_params = {
                "modeler_name": trial.suggest_categorical(
                    "modeler_name",
                    [
                        "linear_modeling",
                        "Time_steps_as_categories",
                        "Sin_Cos",
                        "Periodic_Spline",
                        "Pairwise_Interactions",
                        "Interactions_with_Kernels",
                    ],
                )
            }

        elif self.estimator in xgboost_estimators:
            modeling_params = {
                "modeler_name": trial.suggest_categorical(
                    "modeler_name",
                    (
                        "Periodic_Spline",
                        "linear_modeling",
                        "Sin_Cos",
                        "Time_steps_as_categories",
                        "Interactions_with_Kernels",
                        "Polynomial",
                        "Normalizers",
                    ),
                )
            }

        elif self.estimator in linear_estimators:
            modeling_params = {
                "modeler_name": trial.suggest_categorical(
                    "modeler_name",
                    [
                        "linear_modeling",
                        "Time_steps_as_categories",
                        "Sin_Cos",
                        "Periodic_Spline",
                        "Pairwise_Interactions",
                        "Interactions_with_Kernels",
                    ],
                )
            }

        elif self.estimator in neural_network_estimator:
            modeling_params = {
                "modeler_name": trial.suggest_categorical(
                    "modeler_name",
                    [
                        "linear_modeling",
                        "Time_steps_as_categories",
                        "Sin_Cos",
                        "Periodic_Spline",
                        "Pairwise_Interactions",
                        "Interactions_with_Kernels",
                    ],
                )
            }

        else:
            modeling_params = {
                "modeler_name": trial.suggest_categorical(
                    "modeler_name",
                    [
                        "linear_modeling",
                        "Time_steps_as_categories",
                        "Sin_Cos",
                        "Periodic_Spline",
                        "Pairwise_Interactions",
                    ],
                )
            }

        numerical_features = self._NUMERICAL_FEATURES
        categorical_features = self._CATEGORICAL_FEATURES

        scaler, scaler_name = self.get_standardization(trial)
        encoder_class, encoder_name = self.get_encoder(trial)

        if modeling_params["modeler_name"] == "linear_modeling":
            return (
                ColumnTransformer(
                    transformers=[
                        ("num", scaler, numerical_features),
                        ("categorical", encoder_class, categorical_features),
                    ],
                    remainder="drop",
                ),
                f"{encoder_name} {scaler_name}",
                scaler,
            )

        elif modeling_params["modeler_name"] == "Sin_Cos":
            encoder_class, encoder_name = self.get_encoder(trial)
            return (
                ColumnTransformer(
                    transformers=[
                        ("categorical", encoder_class, categorical_features),
                        ("month_sin", SinTransformer(12), ["Month"]),
                        ("month_cos", CosTransformer(12), ["Month"]),
                        ("hour_sin", SinTransformer(24), ["Hour"]),
                        ("hour_cos", CosTransformer(24), ["Hour"]),
                    ],
                    remainder="drop",
                ),
                f'{modeling_params["modeler_name"]} {scaler_name} {encoder_name}',
                scaler,
            )

        elif modeling_params["modeler_name"] == "Time_steps_as_categories":
            return (
                ColumnTransformer(
                    transformers=[
                        ("categorical", encoder_class, categorical_features + ["Hour", "Month"]),
                    ],
                    remainder="drop",
                ),
                f'{modeling_params["modeler_name"]} {scaler_name} {encoder_name}',
                scaler,
            )

        elif modeling_params["modeler_name"] == "Periodic_Spline":
            encoder_class, encoder_name = self.get_encoder(trial)
            return (
                ColumnTransformer(
                    transformers=[
                        ("categorical", encoder_class, categorical_features),
                        ("cyclic_month", PeriodicSplineTransformer(12, n_splines=6), ["Month"]),
                        ("cyclic_hour", PeriodicSplineTransformer(24, n_splines=12), ["Hour"]),
                    ],
                    remainder="drop",
                ),
                f'{modeling_params["modeler_name"]} {scaler_name} {encoder_name}',
                scaler,
            )

        elif modeling_params["modeler_name"] == "Pairwise_Interactions":
            encoder_class, encoder_name = self.get_encoder(trial)

            hour_workday_interaction = make_pipeline(
                ColumnTransformer(
                    transformers=[
                        ("cyclic_hour", PeriodicSplineTransformer(24, n_splines=8), ["Hour"]),
                        (
                            "is_weekday",
                            FunctionTransformer(_is_weekday_frame),
                            ["Weekday"],
                        ),
                    ]
                ),
                PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
            )

            cyclic_spline_transformer = ColumnTransformer(
                transformers=[
                    ("categorical", encoder_class, categorical_features),
                    ("cyclic_month", PeriodicSplineTransformer(12, n_splines=6), ["Month"]),
                    ("cyclic_hour", PeriodicSplineTransformer(24, n_splines=12), ["Hour"]),
                ],
                remainder="drop",
            )

            return (
                FeatureUnion(
                    [
                        ("marginal", cyclic_spline_transformer),
                        ("interactions", hour_workday_interaction),
                    ]
                ),
                f'{modeling_params["modeler_name"]} {encoder_name} {scaler_name} Periodic_Spline Polynom_Transformation',
                scaler,
            )

        elif modeling_params["modeler_name"] == "Interactions_with_Kernels":
            encoder_class, encoder_name = self.get_encoder(trial)

            column_transf = categorical_features + numerical_features
            custom_preprocessor_with_nystroem = CustomPreprocessorWithNystroem(
                encoder_class, categorical_features, scaler
            )

            return (
                ColumnTransformer(
                    transformers=[
                        ("custom_preprocessor", custom_preprocessor_with_nystroem, column_transf)
                    ]
                ),
                f'{modeling_params["modeler_name"]} {encoder_name} Periodic_Spline {scaler_name}',
                scaler,
            )

        elif modeling_params["modeler_name"] == "Normalizers":
            normalizer, normalizer_name = self.get_normalization(trial)

            if normalizer_name == "Yeo-Johnson":
                scaler, scaler_name = StandardScaler(), "StandardScaler"

            encoder_class, encoder_name = self.get_encoder(trial)

            num_transformer = Pipeline(
                steps=[
                    ("normalizer", normalizer),
                    ("scaler", scaler),
                ]
            )

            return (
                ColumnTransformer(
                    transformers=[
                        ("num", num_transformer, numerical_features),
                        ("categorical", encoder_class, categorical_features),
                    ],
                    remainder="drop",
                    verbose_feature_names_out=False,
                ),
                f'{modeling_params["modeler_name"]} {normalizer_name} {scaler_name} {encoder_name}',
                num_transformer,
            )

        elif modeling_params["modeler_name"] == "Polynomial":
            normalizer, normalizer_name = self.get_normalization(trial)

            if normalizer_name == "Yeo-Johnson":
                scaler, scaler_name = StandardScaler(), "StandardScaler"

            num_transformer = Pipeline(
                steps=[
                    ("normalizer", normalizer),
                    ("scaler", scaler),
                    ("polynomial", PolynomialFeatures(degree=2)),
                ]
            )

            num_transformer_label = Pipeline(
                steps=[
                    ("normalizer", normalizer),
                    ("scaler", scaler),
                ]
            )

            return (
                ColumnTransformer(
                    transformers=[
                        ("num", num_transformer, numerical_features),
                        ("categorical", encoder_class, categorical_features),
                    ],
                    remainder="drop",
                    verbose_feature_names_out=False,
                ),
                f'{modeling_params["modeler_name"]} {normalizer_name} {scaler_name} PolynomialFeatures(degree=2) {encoder_name}',
                num_transformer_label,
            )

    def get_parameters(self, estimator: str, trial) -> dict:
        """Suggest hyperparameters for the given estimator."""
        if "RandomForestRegressor" == estimator:
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 400, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 25),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 1.0]),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 50, 500, log=True),
                "min_impurity_decrease": trial.suggest_float(
                    "min_impurity_decrease", 1e-6, 1e-3, log=True
                ),
                "random_state": 42,
                "n_jobs": -1,
            }

        elif "BaggingRegressor" in self.estimator:
            return {
                "n_estimators": trial.suggest_int("br_n_estimators", 10, 100),
                "max_samples": trial.suggest_float("br_max_samples", 0.1, 1.0),
                "max_features": trial.suggest_float("br_max_features", 0.1, 1.0),
                "bootstrap": trial.suggest_categorical("br_bootstrap", [True, False]),
                "bootstrap_features": trial.suggest_categorical(
                    "br_bootstrap_features", [True, False]
                ),
                "estimator": RandomForestRegressor(
                    n_estimators=150,
                    max_depth=15,
                    min_samples_split=2,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    random_state=42,
                ),
                "random_state": 42,
            }

        elif "HistGradientBoostingRegressor" == estimator:
            return {
                "learning_rate": trial.suggest_categorical("learning_rate", [0.01, 0.005, 0.001]),
                "max_iter": trial.suggest_int("max_iter", 100, 2000, log=True),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 30, 50, log=True),
                "max_depth": trial.suggest_int("max_depth", 4, 6, step=1),
                # Ceiling widened from 40: the past winning trial picked min_samples_leaf=39
                # (90% of the old ceiling), the same boundary-seeking pattern seen in
                # XGB/LGBM — worth testing whether even larger (more conservative) leaves
                # help, given HGB still shows a real train/test gap despite already-shallow
                # trees. Note: HGB's internal early_stopping/validation_fraction below uses
                # a random (non-temporal) split of its training data — sklearn exposes no
                # API for a custom/ordered eval set here, so this is a known, documented
                # limitation rather than something this search-space change addresses.
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 30, 60, step=1),
                "l2_regularization": trial.suggest_categorical(
                    "l2_regularization", [0.01, 0.05, 0.1]
                ),
                "early_stopping": True,
                "n_iter_no_change": 5,
                "validation_fraction": 0.3,
                "random_state": 42,
            }

        elif "Ridge" == estimator:
            return {
                "alpha": trial.suggest_float("alpha", 1e-06, 25.0, log=True),
            }

        elif "Lasso" == self.estimator:
            return {
                "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
                "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
                "max_iter": trial.suggest_int("max_iter", 100, 5000),
                "tol": trial.suggest_float("tol", 1e-4, 1e-1, log=True),
                "random_state": 42,
            }

        elif "LGBMRegressor" == self.estimator:
            return {
                "objective": "regression",
                # "l1" (MAE) — aligned with the study objective (mean CV MAE) and,
                # unlike the previous "r2" (not a LightGBM metric), a valid eval
                # metric for the early-stopping callback: with an invalid metric
                # every fit with eval_set raised "For early stopping, at least one
                # dataset and eval metric is required for evaluation", zeroing out
                # entire studies.
                "metric": "l1",
                "boosting_type": "gbdt",
                "force_col_wise": True,
                "num_leaves": trial.suggest_int("num_leaves", 2, 256),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                # Floor raised from 5: the past winning trial picked min_child_samples=7,
                # near the old floor — the MAE-only objective was actively selecting the
                # least-regularized corner of this parameter.
                "min_child_samples": trial.suggest_int("min_child_samples", 15, 100),
                "subsample": trial.suggest_float("subsample", 0.1, 1),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1),
                # Floor raised from 1e-9: the past winning trial picked reg_alpha~3.7e-9
                # (effectively zero) despite the range allowing up to 100 — same
                # boundary-seeking-toward-zero-regularization pattern as min_child_samples.
                # reg_lambda's past winning value (0.01) was NOT at an extreme, so its
                # range is left unchanged.
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 100.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-9, 100.0, log=True),
                "max_depth": trial.suggest_int("max_depth", -1, 8),
                "verbosity": -1,
                "random_state": 42,
            }

        elif "CatboostRegressor" in self.estimator:
            return {
                "iterations": trial.suggest_int("iterations", 100, 1000),
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "random_strength": trial.suggest_int("random_strength", 0, 100),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
                "border_count": trial.suggest_int("border_count", 1, 255),
                "loss_function": "RMSE",
                "verbose": False,
                "task_type": "GPU",
            }

        elif "XGBRegressor" in self.estimator:
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                # [3, 8]: the first cap ([3, 6], down from [3, 10]) proved too
                # tight — the winner pinned depth at the floor (3) and holdout R2
                # dropped ~0.04 while the train/test gap barely moved. With the
                # temporal early stopping below now guarding memorization (plus
                # min_child_weight and log-scaled reg terms), depth no longer has
                # to carry the regularization burden on its own.
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "gamma": trial.suggest_float("gamma", 0, 5),
                # min_child_weight was previously absent from the search space
                # (implicit default=1, effectively unconstrained) — a standard
                # XGB regularizer worth exposing to the search.
                "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
                # reg_alpha/reg_lambda switched from uniform[0,5] to log-scale with a
                # wider ceiling on reg_lambda: past winners landed at reg_lambda~4.4/5.0
                # (87% of the old ceiling), a boundary-seeking signal that the search
                # wants more L2 regularization than the old range allowed.
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 20.0, log=True),
                "objective": "reg:squarederror",
                "random_state": 42,
            }

        elif "SGDRegressor" in self.estimator:
            params = {
                "alpha": trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
                "penalty": trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"]),
                "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
                "learning_rate": trial.suggest_categorical(
                    "learning_rate", ["constant", "optimal", "invscaling", "adaptive"]
                ),
                "early_stopping": trial.suggest_categorical("early_stopping", [True, False]),
                "max_iter": trial.suggest_int("max_iter", 1000, 10000),
                "tol": trial.suggest_float("tol", 1e-4, 1e-1, log=True),
                "random_state": 42,
            }
            if params["penalty"] == "elasticnet":
                params["l1_ratio"] = trial.suggest_float("l1_ratio", 0, 1)
            if params["learning_rate"] in ["constant", "invscaling", "adaptive"]:
                params["eta0"] = trial.suggest_float("eta0", 1e-5, 1e-1)
            if params["learning_rate"] == "invscaling":
                params["power_t"] = trial.suggest_float("power_t", 0.1, 0.5)
            if params["early_stopping"]:
                params["validation_fraction"] = trial.suggest_float("validation_fraction", 0.1, 0.3)
                params["n_iter_no_change"] = trial.suggest_int("n_iter_no_change", 5, 15)
            return params

        elif "KNeighborsRegressor" in self.estimator:
            return {
                "n_neighbors": trial.suggest_int("n_neighbors", 1, 30),
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                "algorithm": trial.suggest_categorical(
                    "algorithm", ["auto", "ball_tree", "kd_tree", "brute"]
                ),
                "p": trial.suggest_int("p", 1, 2),
            }

        elif "DecisionTreeRegressor" in self.estimator:
            return {
                "criterion": trial.suggest_categorical("criterion", ["mse", "friedman_mse", "mae"]),
                "splitter": trial.suggest_categorical("splitter", ["best", "random"]),
                "max_depth": trial.suggest_int("max_depth", 1, 32),
                "min_samples_split": trial.suggest_float("min_samples_split", 0.1, 1),
                "min_samples_leaf": trial.suggest_float("min_samples_leaf", 0.1, 0.5),
                "max_features": trial.suggest_categorical("max_features", ["auto", "sqrt", "log2"]),
            }

        elif "SVR" in self.estimator:
            params = {
                "kernel": trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"]),
                "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
                "C": trial.suggest_float("C", 0.1, 100.0),
                "epsilon": trial.suggest_float("epsilon", 0.01, 1.0),
            }
            if trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"]) == "poly":
                params["degree"] = trial.suggest_int("degree", 1, 5)
            else:
                params["degree"] = 3
            return params

        elif "BayesianRidge" in self.estimator:
            return {
                "n_iter": trial.suggest_int("n_iter", 100, 500),
                "tol": trial.suggest_float("tol", 1e-6, 1e-3, log=True),
                "alpha_1": trial.suggest_float("alpha_1", 1e-6, 1e-3, log=True),
                "alpha_2": trial.suggest_float("alpha_2", 1e-6, 1e-3, log=True),
                "lambda_1": trial.suggest_float("lambda_1", 1e-6, 1e-3, log=True),
                "lambda_2": trial.suggest_float("lambda_2", 1e-6, 1e-3, log=True),
            }

        elif "MLPRegressor" in self.estimator:
            return {
                "hidden_layer_sizes": trial.suggest_categorical(
                    "hidden_layer_sizes",
                    [(50,), (100,), (50, 50), (100, 50), (100, 100)],
                ),
                "activation": trial.suggest_categorical(
                    "activation", ["identity", "logistic", "tanh", "relu"]
                ),
                "solver": trial.suggest_categorical("solver", ["lbfgs", "sgd", "adam"]),
                "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
                "learning_rate": trial.suggest_categorical(
                    "learning_rate", ["constant", "invscaling", "adaptive"]
                ),
                "max_iter": trial.suggest_int("max_iter", 200, 1000),
                "tol": trial.suggest_float("tol", 1e-5, 1e-3, log=True),
                "random_state": 42,
            }

        elif "TweedieRegressor" in self.estimator:
            return {
                "power": trial.suggest_float("power", 1.0, 2.0),
                "alpha": trial.suggest_float("alpha", 0.0, 1.0),
                "link": trial.suggest_categorical("link", ["auto", "identity", "log"]),
                "max_iter": trial.suggest_int("max_iter", 100, 1000),
                "tol": trial.suggest_float("tol", 1e-6, 1e-3, log=True),
                "warm_start": trial.suggest_categorical("warm_start", [True, False]),
            }

        else:
            return {}

    def get_normalization(self, trial) -> Tuple[Any, str]:
        """Suggest a normalizer from the sample space."""
        normalizer_name = trial.suggest_categorical(
            "normalizers",
            ["QuantileUniform", "QuantileNormal", "MinMaxScaler"],
        )

        if normalizer_name == "Yeo-Johnson":
            return PowerTransformer(method="yeo-johnson", standardize=False), normalizer_name
        elif normalizer_name == "Box-Cox":
            return PowerTransformer(method="box-cox"), normalizer_name
        elif normalizer_name == "QuantileUniform":
            return QuantileTransformer(output_distribution="uniform"), normalizer_name
        elif normalizer_name == "QuantileNormal":
            return QuantileTransformer(output_distribution="normal"), normalizer_name
        elif normalizer_name == "Normalizer":
            return Normalizer(norm="max"), normalizer_name
        elif normalizer_name == "MinMaxScaler":
            return MinMaxScaler(feature_range=(0, 1)), normalizer_name
        else:
            return None, normalizer_name

    def get_encoder(self, trial) -> Tuple[Any, str]:
        """Suggest a categorical encoder from category_encoders / feature_engine."""
        all_encoders = [
            "OrdinalEncoder",
            "MeanEncoder",
            "CountFrequencyEncoder",
            "QuantileEncoder",
            "JamesSteinEncoder",
            "LeaveOneOutEncoder",
            "BaseNEncoder",
            "GrayEncoder",
            "CountEncoder",
            "HashingEncoder",
            "HelmertEncoder",
            "SumEncoder",
            "PolynomialEncoder",
            "MEstimateEncoder",
            "GLMMEncoder",
            "BackwardDifferenceEncoder",
        ]

        encoder_name = trial.suggest_categorical("encoders", all_encoders)

        categorical = self._CATEGORICAL_FEATURES

        # MeanEncoder / CountFrequencyEncoder (feature_engine) validate that every
        # column in `variables` is object/category dtype and raise otherwise.
        # DayNumberOnWeek is an int column (WeekdayWeekStatusTransformer: dt.weekday + 1,
        # never cast to category — see src/feature_engineering.py and the interval
        # treatment in PHIK_INTERVAL_COLS), so passing it here made every trial that
        # sampled either of these two encoders fail 100% of the time (see the
        # "Some of the variables are not categorical" rows in dataset/invalid_configs.csv).
        # Excluding it from `variables` leaves it as an unencoded passthrough column
        # instead of crashing; every other encoder here is dtype-agnostic and keeps
        # encoding it as before.
        feature_engine_categorical = [c for c in categorical if c != "DayNumberOnWeek"]

        encoder_map = {
            "BackwardDifferenceEncoder": BackwardDifferenceEncoder(),
            "GrayEncoder": GrayEncoder(),
            "CountEncoder": CountEncoder(),
            "HashingEncoder": _SingleProcessHashingEncoder(),
            "HelmertEncoder": HelmertEncoder(),
            "OneHotEncoder": OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            "OrdinalEncoder": OrdinalEncoder(),
            "SumEncoder": SumEncoder(),
            "PolynomialEncoder": PolynomialEncoder(),
            "BaseNEncoder": BaseNEncoder(),
            "LeaveOneOutEncoder": LeaveOneOutEncoder(),
            "MEstimateEncoder": MEstimateEncoder(),
            "JamesSteinEncoder": JamesSteinEncoder(),
            "GLMMEncoder": GLMMEncoder(),
            "QuantileEncoder": QuantileEncoder(),
            "MeanEncoder": MeanEncoder(variables=feature_engine_categorical),
            "CountFrequencyEncoder": CountFrequencyEncoder(
                encoding_method="count", variables=feature_engine_categorical
            ),
        }

        return encoder_map.get(encoder_name, OrdinalEncoder()), encoder_name

    def get_standardization(self, trial) -> Tuple[Any, str]:
        """Suggest a standardisation scaler."""
        if self.estimator in linear_estimators:
            scaler_name = "StandardScaler"
        elif self.estimator in catboost_estimators:
            scaler_name = trial.suggest_categorical(
                "standardazer", ["StandardScaler", "MaxAbsScaler"]
            )
        else:
            scaler_name = trial.suggest_categorical(
                "standardazer", ["StandardScaler", "MaxAbsScaler", "RobustScaler"]
            )

        if scaler_name == "MaxAbsScaler":
            return MaxAbsScaler(), scaler_name
        elif scaler_name == "StandardScaler":
            return StandardScaler(with_mean=True, with_std=True), scaler_name
        else:
            return RobustScaler(), scaler_name

    def get_feature_selection(self, trial, estimator: object) -> Tuple[Any, str, Any]:
        """Sample a feature selector and its hyperparameters."""
        define_estimator()

        if (
            self.estimator in neighbor_estimators
            or self.estimator in svr_estimator
            or "MLPRegressor" in self.estimator
        ):
            selector_space = ["NoSelector", "SelectKBest", "SequentialFeatureSelector"]
        elif self.estimator in lightgbm_estimators:
            selector_space = ["NoSelector", "SelectKBest", "SelectFromModel"]
        elif self.estimator in xgboost_estimators:
            selector_space = ["NoSelector", "SelectKBest", "RFE", "SelectFromModel"]
        elif "HistGradientBoostingRegressor" == self.estimator:
            # SequentialFeatureSelector is excluded for HGB: backward SFS with
            # cv=3 internal refits of an already early-stopping-guarded HGB made
            # single trials take ~30 min (the last study completed only 17/400
            # trials in ~8.2 h — the selector, not the model, was the bottleneck).
            # RFE/SelectFromModel are also out: both clone the trial's estimator
            # and HGB exposes neither feature_importances_ nor coef_.
            selector_space = ["NoSelector", "SelectKBest"]
        else:
            selector_space = [
                "NoSelector",
                "SelectKBest",
                "RFE",
                "SelectFromModel",
                "SequentialFeatureSelector",
            ]

        # NOTE: selector_space must stay identical across every trial for a given
        # estimator bucket — Optuna's CategoricalDistribution rejects a
        # per-trial-varying candidate list ("does not support dynamic value
        # space"). The NoSelector-for-high-dimensional-preprocessing rule is
        # therefore enforced reactively in objective(), not by narrowing this
        # list based on the already-sampled modeler_name.
        selector_name = trial.suggest_categorical("selectors", selector_space)

        if selector_name == "NoSelector":
            return "passthrough", selector_name, None

        if selector_name == "SelectKBest":
            k_best = trial.suggest_int("kbest_n_features", 6, 22, step=1)
            return SelectKBest(mutual_info_regression, k=k_best), selector_name, k_best

        if selector_name == "RFE":
            rfe_n = trial.suggest_int("rfe_n_features", 6, 18, step=1)
            rfe_step = trial.suggest_float("rfe_step", 0.1, 0.5, step=0.1)
            return (
                RFE(estimator=clone(estimator), n_features_to_select=rfe_n, step=rfe_step),
                selector_name,
                rfe_n,
            )

        if selector_name == "SelectFromModel":
            sfm_threshold = trial.suggest_categorical(
                "sfm_threshold", ["median", "1.25*median", "mean"]
            )
            sfm_max_features = trial.suggest_int("sfm_max_features", 8, 24, step=1)
            return (
                SelectFromModel(
                    estimator=clone(estimator),
                    threshold=sfm_threshold,
                    max_features=sfm_max_features,
                ),
                selector_name,
                sfm_max_features,
            )

        sfs_n = trial.suggest_int("sfs_n_features", 6, 14, step=1)
        return (
            SequentialFeatureSelector(
                estimator=clone(estimator),
                n_features_to_select=sfs_n,
                direction="backward",
                cv=3,
                n_jobs=-1,
            ),
            selector_name,
            sfs_n,
        )

    def objective(self, trial) -> float:
        """Evaluate a single Optuna trial using inner TimeSeriesSplit CV."""
        transformer, transformer_name, num_transformer = self.modeling_transformers(trial)

        model_class = get_regressor(self.estimator)
        param_s = self.get_parameters(self.estimator, trial)
        base_estimator = model_class(**param_s)

        feature_selector, selector_name, n_features_to_select = self.get_feature_selection(
            trial, base_estimator
        )

        modeler_name = trial.params.get("modeler_name")
        encoder_name = trial.params.get("encoders")

        if modeler_name in self._HIGH_DIM_MODELERS and selector_name == "NoSelector":
            reason = f"NoSelector not allowed with high-dimensional preprocessing ({modeler_name})"
            trial.set_user_attr("failed_reason", reason)
            self.register_invalid_config(modeler_name, encoder_name, selector_name, reason=reason)
            return float("inf")

        if self.was_invalid_config(modeler_name, encoder_name, selector_name):
            trial.set_user_attr("failed_reason", "Known invalid/slow configuration — skipped")
            return float("inf")

        cache_dir = mkdtemp()

        pipe = Pipeline(
            steps=[("transformer", transformer)],
            memory=cache_dir,
        )

        estimator = Pipeline(
            steps=[
                ("features", pipe),
                ("imputer", SimpleImputer(strategy="median")),
                ("selector", feature_selector),
                (
                    "regressor",
                    TransformedTargetRegressor(
                        regressor=base_estimator,
                        transformer=clone(num_transformer),
                    ),
                ),
            ]
        )

        try:
            if self.estimator in _EARLY_STOPPING_ESTIMATORS:
                cv_results, err = _cross_validate_with_early_stopping_timeout(
                    estimator,
                    X_train_opt,
                    y_train_opt,
                    ts_cv,
                    timeout=self.trial_timeout,
                )
            else:
                cv_results, err = _cross_validate_with_timeout(
                    estimator,
                    X_train_opt,
                    y_train_opt,
                    ts_cv,
                    ["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"],
                    timeout=self.trial_timeout,
                )
            if err is not None:
                trial.set_user_attr("failed_reason", err)
                if not _is_transient_error(err):
                    self.register_invalid_config(
                        modeler_name, encoder_name, selector_name, reason=err
                    )
                mae = float("inf")
            else:
                mae = -cv_results["test_neg_mean_absolute_error"].mean()

                # Log-only diagnostics: surface the train/val R2 gap per trial via
                # trial.set_user_attr without touching what "best" means — the
                # objective Optuna minimizes stays pure MAE, so prior studies'
                # best_trial/best_params remain comparable. Lets a future notebook
                # cell inspect study.trials_dataframe() for a MAE-vs-gap scatter.
                train_r2 = float(np.mean(cv_results["train_r2"]))
                test_r2 = float(np.mean(cv_results["test_r2"]))
                trial.set_user_attr("train_r2", train_r2)
                trial.set_user_attr("test_r2", test_r2)
                trial.set_user_attr("r2_gap", train_r2 - test_r2)

                if self.estimator in _EARLY_STOPPING_ESTIMATORS:
                    best_iterations = [
                        b for b in cv_results.get("best_iterations", []) if b is not None
                    ]
                    if best_iterations:
                        trial.set_user_attr("best_iterations", best_iterations)
        except Exception as exc:
            reason = str(exc)[:2000]
            trial.set_user_attr("failed_reason", reason)
            if not _is_transient_error(reason):
                self.register_invalid_config(
                    modeler_name, encoder_name, selector_name, reason=reason
                )
            mae = float("inf")
        finally:
            rmtree(cache_dir, ignore_errors=True)
            gc.collect()

        return mae

    def detailed_objective(self, best_trial) -> None:
        """Run the best trial with full evaluation and save artefacts."""
        global pipeline_winner

        transformer, transformer_name, num_transformer = self.modeling_transformers(best_trial)

        model_class = get_regressor(self.estimator)
        param_s = self.get_parameters(self.estimator, best_trial)

        if self.estimator in _EARLY_STOPPING_ESTIMATORS:
            # Use the early-stopping median across the winning trial's CV folds
            # (see objective()/_run_cross_validate_with_early_stopping_in_subprocess)
            # as a fixed n_estimators for the final fit, instead of either the
            # trial's full sampled n_estimators (would undo early stopping's
            # overfitting guard) or re-running early stopping inside the final
            # fit itself (extra complexity/non-determinism for a one-off fit).
            best_iterations = best_trial.user_attrs.get("best_iterations")
            if best_iterations:
                median_iterations = max(int(np.median(best_iterations)), 1)
                logger.info(
                    "[%s] overriding n_estimators=%s with early-stopping median "
                    "best_iteration=%d (from %d CV folds)",
                    self.estimator,
                    param_s.get("n_estimators"),
                    median_iterations,
                    len(best_iterations),
                )
                param_s["n_estimators"] = median_iterations

        base_estimator = model_class(**param_s)

        feature_selector, selector_name, n_features_to_select = self.get_feature_selection(
            best_trial, base_estimator
        )

        if n_features_to_select is None:
            selector_label = selector_name
        else:
            selector_label = f"{selector_name}(k={n_features_to_select})"

        cache_dir = mkdtemp()

        pipe = Pipeline(
            steps=[("transformer", transformer)],
            memory=cache_dir,
        )

        pipeline_winner = Pipeline(
            steps=[
                ("transformer", pipe),
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "selector",
                    clone(feature_selector)
                    if not isinstance(feature_selector, str)
                    else feature_selector,
                ),
                (
                    "regressor",
                    TransformedTargetRegressor(
                        regressor=clone(base_estimator),
                        transformer=clone(num_transformer),
                    ),
                ),
            ]
        )

        save_model_and_metrics_regression(
            description=self.estimator,
            preprocessing=f"{transformer_name} {selector_label}",
            pipe=pipe,
            regressor=base_estimator,
            target_transformer=num_transformer,
            params=best_trial.params,
            feature_selector=None if feature_selector == "passthrough" else feature_selector,
            timeout=self.detailed_timeout,
        )

    def _report_study_health(self, study) -> None:
        """Log a post-study health summary and fail loudly on a fully-broken study.

        The LGBM incident this guards against: a search-space bug made every
        fit raise, so all 400 trials returned ``inf``, the "best" trial was
        trial 0's meaningless defaults, and the failure was only discovered
        later by inspecting invalid_configs.csv. A study where most trials
        fail should announce itself in the notebook cell output — and one
        where *no* trial succeeded must not silently overwrite the saved
        model artifacts with a garbage winner.

        Raises
        ------
        RuntimeError
            If every completed trial returned a non-finite objective value.
        """
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        finite = [t for t in completed if t.value is not None and np.isfinite(t.value)]
        failed = [t for t in completed if t not in finite]

        logger.info(
            "[%s] study health: %d trials completed, %d succeeded, %d failed (inf)",
            self.estimator,
            len(completed),
            len(finite),
            len(failed),
        )
        if failed:
            reason_counts: Dict[str, int] = {}
            for t in failed:
                reason = t.user_attrs.get("failed_reason", "<no failed_reason recorded>")
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            for reason, count in sorted(reason_counts.items(), key=lambda kv: -kv[1])[:5]:
                logger.warning("[%s] %d failed trial(s): %s", self.estimator, count, reason)
        if completed and not finite:
            raise RuntimeError(
                f"[{self.estimator}] every one of the {len(completed)} completed trials "
                f"returned inf — the study found no valid configuration, so the 'best' "
                f"trial is meaningless and no model will be saved. See the failure "
                f"reasons logged above (most frequent first)."
            )

    def optimize(self) -> dict:
        """Run the full Optuna study and return best params.

        Sets module-level ``start_time`` before the study begins.
        After optimisation, calls ``detailed_objective`` on the best trial
        which triggers ``save_model_and_metrics_regression`` and sets
        the module-level ``pipeline_winner`` for downstream use.
        """
        global start_time

        start_time = time.time()
        study = optuna.create_study(
            direction="minimize",
            sampler=self.sampler,
            pruner=optuna.pruners.MedianPruner(),
        )

        study.optimize(self.objective, n_trials=self.trials, timeout=self.study_timeout)
        self.flush_invalid_configs()
        search_elapsed = time.time() - start_time
        logger.info(
            "[%s] search phase finished in %.1fs (study_timeout=%s)",
            self.estimator,
            search_elapsed,
            self.study_timeout,
        )
        self._report_study_health(study)
        fig = optuna.visualization.plot_timeline(study)
        fig.show()
        self.detailed_objective(study.best_trial)
        total_elapsed = time.time() - start_time
        logger.info(
            "[%s] optimize() total wall time %.1fs (search %.1fs + detailed %.1fs, "
            "detailed_timeout=%.1fs)",
            self.estimator,
            total_elapsed,
            search_elapsed,
            total_elapsed - search_elapsed,
            self.detailed_timeout,
        )
        return study.best_params


# ---------------------------------------------------------------------------
# metric_dataframe factory (used at notebook start-up)
# ---------------------------------------------------------------------------


def make_metric_dataframe(csv_path=None) -> pd.DataFrame:
    """Load or create the metric tracking DataFrame.

    Port of the try/except block at the end of cell [137].
    """
    if csv_path is None:
        csv_path = _PROJECT_ROOT / "dataset" / "metric_dataframe.csv"
    try:
        return pd.read_csv(
            csv_path,
            index_col=["Estimator", "Optimization", "Pre-Process Pipeline"],
        )
    except FileNotFoundError:
        metric_columns = [
            "Estimator",
            "Optimization",
            "Pre-Process Pipeline",
            "Train MAE",
            "Train MAE Standard Deviation",
            "Test MAE",
            "Test MAE Standard Deviation",
            "Train MSE",
            "Train MSE Standard Deviation",
            "Test MSE",
            "Test MSE Standard Deviation",
            "Train RMSE",
            "Train RMSE Standard Deviation",
            "Test RMSE",
            "Test RMSE Standard Deviation",
            "Train R2",
            "Train R2 Standard Deviation",
            "Test R2",
            "Test R2 Standard Deviation",
            "Parameters",
        ]
        df = pd.DataFrame(columns=metric_columns)
        df.set_index(["Estimator", "Optimization", "Pre-Process Pipeline"], inplace=True)
        return df
