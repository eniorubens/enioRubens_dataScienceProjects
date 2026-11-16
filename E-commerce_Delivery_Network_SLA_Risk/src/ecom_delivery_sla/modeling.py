"""Baseline modeling helpers for delivery delay and SLA breach."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import statsmodels.api as sm
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from .features import feature_catalog

_MODEL_CATALOG = feature_catalog().query(
    "status == 'available_at_purchase' and use_in_model == True"
)
NUMERIC_FEATURES = _MODEL_CATALOG.loc[
    _MODEL_CATALOG["feature_type"] == "numeric", "feature"
].tolist()
CATEGORICAL_FEATURES = _MODEL_CATALOG.loc[
    _MODEL_CATALOG["feature_type"] == "categorical", "feature"
].tolist()
BASELINE_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _validate_evaluate_splits(
    splits: dict[str, pd.DataFrame],
    evaluate_splits: tuple[str, ...],
    allow_test: bool,
) -> tuple[str, ...]:
    """Validate evaluation partitions and protect the test period."""
    if not evaluate_splits:
        raise ValueError("evaluate_splits must contain at least one split")
    unknown = sorted(set(evaluate_splits) - set(splits))
    if unknown:
        raise KeyError(f"unknown evaluation splits: {unknown}")
    if "test" in evaluate_splits and not allow_test:
        raise ValueError(
            "test evaluation is blocked by default; pass allow_test=True "
            "only during the final evaluation stage"
        )
    return evaluate_splits


def build_modeling_frame(
    candidate_features: pd.DataFrame,
    supervised_orders: pd.DataFrame,
) -> pd.DataFrame:
    """Attach historical targets to the order-level candidate feature table."""
    target_columns = ["order_id", "delivery_delay_days", "sla_breach"]
    # actual_lead_days is a target-derived column kept only for the mandatory
    # lead-time baselines; it is never part of the model feature lists.
    if "actual_lead_days" in supervised_orders:
        target_columns.append("actual_lead_days")
    if candidate_features["order_id"].duplicated().any():
        raise ValueError("candidate_features must contain one row per order_id")
    if supervised_orders["order_id"].duplicated().any():
        raise ValueError("supervised_orders must contain one row per order_id")
    return candidate_features.merge(
        supervised_orders[target_columns],
        on="order_id",
        how="inner",
        validate="one_to_one",
    )


def baseline_feature_catalog() -> pd.DataFrame:
    """Return the baseline feature list split by numeric and categorical type."""
    return pd.DataFrame(
        {
            "feature": BASELINE_FEATURES,
            "feature_type": ["numeric"] * len(NUMERIC_FEATURES)
            + ["categorical"] * len(CATEGORICAL_FEATURES),
        }
    )


def make_preprocessor(
    scaler: str | None = None,
    numeric_features: list[str] | None = None,
) -> ColumnTransformer:
    """Build the shared preprocessing step used by tournaments and Optuna.

    Parameters
    ----------
    scaler:
        ``None`` for no scaling (tree models), ``"standard"`` or ``"robust"``
        for linear models. Imputers are always fitted inside the pipeline, so
        statistics come from the training partition only.
    numeric_features:
        Optional override of the numeric column list (for ablations). ``None``
        keeps the catalog-driven :data:`NUMERIC_FEATURES`.
    """
    numeric_columns = (
        NUMERIC_FEATURES if numeric_features is None else list(numeric_features)
    )
    scalers = {"standard": StandardScaler, "robust": RobustScaler}
    if scaler is not None and scaler not in scalers:
        raise ValueError(f"unsupported scaler: {scaler}")
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scaler is not None:
        numeric_steps.append(("scaler", scalers[scaler]()))
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numeric_columns),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _make_preprocessor(
    numeric_features: list[str] | None = None,
) -> ColumnTransformer:
    """Backward-compatible alias for the unscaled shared preprocessor."""
    return make_preprocessor(None, numeric_features=numeric_features)


def calibrated_classifier(estimator: object, cv: int = 3) -> CalibratedClassifierCV:
    """Wrap a classifier with isotonic calibration fitted on training data only.

    This is the single class-imbalance policy: no class weights anywhere;
    probabilities are recalibrated with ``CalibratedClassifierCV`` (isotonic,
    ``cv`` folds) inside the training partition of each fold.
    """
    return CalibratedClassifierCV(estimator, method="isotonic", cv=cv)


def prepare_model_matrices(
    splits: dict[str, pd.DataFrame],
    numeric_features: list[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], ColumnTransformer]:
    """Fit preprocessing on train and transform all temporal partitions.

    ``numeric_features`` optionally overrides the numeric column list; ``None``
    keeps :data:`NUMERIC_FEATURES`.
    """
    numeric_columns = (
        NUMERIC_FEATURES if numeric_features is None else list(numeric_features)
    )
    required = set(numeric_columns) | set(CATEGORICAL_FEATURES)
    for split_name, frame in splits.items():
        missing = sorted(required - set(frame.columns))
        if missing:
            raise KeyError(f"missing baseline features in {split_name}: {missing}")

    preprocessor = _make_preprocessor(numeric_columns)
    train_matrix = preprocessor.fit_transform(splits["train"])
    names = preprocessor.get_feature_names_out()
    matrices = {
        "train": pd.DataFrame(train_matrix, index=splits["train"].index, columns=names)
    }
    for split_name in ("validation", "test"):
        if split_name not in splits:
            continue
        transformed = preprocessor.transform(splits[split_name])
        matrices[split_name] = pd.DataFrame(
            transformed,
            index=splits[split_name].index,
            columns=names,
        )
    return matrices, preprocessor


def fit_ols_regression(X_train: pd.DataFrame, y_train: pd.Series):
    """Fit an OLS regression with an intercept for inferential baseline use."""
    design = sm.add_constant(X_train.astype(float), has_constant="add")
    return sm.OLS(y_train.astype(float), design).fit()


def predict_ols(model: object, features: pd.DataFrame) -> np.ndarray:
    """Predict with a fitted OLS model using the same intercept convention."""
    design = sm.add_constant(features.astype(float), has_constant="add")
    return np.asarray(model.predict(design))


def fit_ridge_regression(X_train: pd.DataFrame, y_train: pd.Series) -> Ridge:
    """Fit a Ridge regression baseline for predictive comparison."""
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    return model


def fit_logistic_classification(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Fit a standardized logistic regression baseline without class weights."""
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(max_iter=2_000, solver="liblinear"),
            ),
        ]
    )
    model.fit(X_train, y_train.astype(int))
    return model


def regression_baseline_predictions(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Return mandatory delay baselines without fitting on evaluation rows."""
    if "actual_lead_days" not in train or "promised_lead_days" not in evaluation:
        raise KeyError("train/evaluation must contain lead-time columns")
    target = train["delivery_delay_days"].astype(float)
    global_actual_median = train["actual_lead_days"].median()
    state_medians = train.groupby("customer_state")["actual_lead_days"].median()
    state_prediction = evaluation["customer_state"].map(state_medians).fillna(
        global_actual_median
    ) - evaluation["promised_lead_days"]
    return {
        "train_mean": np.full(len(evaluation), target.mean()),
        "median_actual_minus_promise": (
            global_actual_median - evaluation["promised_lead_days"]
        ).to_numpy(),
        "state_median_actual_minus_promise": state_prediction.to_numpy(),
    }


def classification_baseline_scores(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Return prevalence, promise-score, and promise-plus-state baselines."""
    prevalence = float(train["sla_breach"].astype(int).mean())
    promise_score = -evaluation["promised_lead_days"].astype(float).to_numpy()
    design = pd.get_dummies(
        pd.concat(
            [
                train[["promised_lead_days", "customer_state"]],
                evaluation[["promised_lead_days", "customer_state"]],
            ]
        ),
        columns=["customer_state"],
        dtype=float,
    )
    train_design = design.iloc[: len(train)]
    evaluation_design = design.iloc[len(train) :]
    model = LogisticRegression(max_iter=2_000, solver="liblinear")
    model.fit(train_design, train["sla_breach"].astype(int))
    return {
        "prevalence": np.full(len(evaluation), prevalence),
        "negative_promised_lead_days": promise_score,
        "logistic_promise_plus_state": model.predict_proba(evaluation_design)[:, 1],
    }


def fit_quantile_regressors(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    quantiles: tuple[float, ...] = (0.5, 0.9),
    random_state: int = 42,
) -> dict[float, HistGradientBoostingRegressor]:
    """Fit P50/P90 quantile regressors for actual lead time."""
    models = {}
    for quantile in quantiles:
        if not 0 < quantile < 1:
            raise ValueError("quantiles must be between 0 and 1")
        model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=quantile,
            max_iter=200,
            learning_rate=0.05,
            random_state=random_state,
        )
        model.fit(X_train, y_train)
        models[quantile] = model
    return models


def build_regression_tournament(random_state: int = 42) -> dict[str, object]:
    """Build the principal regression estimators for the temporal tournament."""
    return {
        "Ridge": Ridge(alpha=1.0),
        "ElasticNet": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    ElasticNet(
                        alpha=0.01,
                        l1_ratio=0.5,
                        max_iter=100_000,
                        tol=1e-4,
                        selection="cyclic",
                    ),
                ),
            ]
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=1,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=250,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=1,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=random_state,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=random_state,
            n_jobs=1,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=1,
            verbosity=-1,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            loss_function="RMSE",
            random_seed=random_state,
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
        ),
    }


def build_classification_tournament(
    y_train: pd.Series,
    random_state: int = 42,
) -> dict[str, object]:
    """Build the principal classification estimators for the temporal tournament.

    The estimators are unweighted; the class-imbalance policy is isotonic
    calibration applied by :func:`calibrated_classifier` when they are fitted
    in :func:`run_classification_tournament`.
    """
    positive = int(y_train.astype(int).sum())
    if positive == 0:
        raise ValueError("y_train must contain at least one positive class")
    return {
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(max_iter=2_000, solver="liblinear"),
                ),
            ]
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=250,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=random_state,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=1,
            verbosity=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=random_state,
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
        ),
    }


def _select_models(
    candidates: dict[str, object],
    names: tuple[str, ...] | None,
) -> dict[str, object]:
    """Restrict a model dictionary to the requested names (``None`` keeps all)."""
    if names is None:
        return candidates
    unknown = sorted(set(names) - set(candidates))
    if unknown:
        raise KeyError(f"unknown models: {unknown}")
    return {name: model for name, model in candidates.items() if name in names}


def run_regression_tournament(
    splits: dict[str, pd.DataFrame],
    random_state: int = 42,
    evaluate_splits: tuple[str, ...] = ("validation",),
    allow_test: bool = False,
    numeric_features: list[str] | None = None,
    models: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fit regression estimators on approved temporal partitions.

    Each row reports ``fit_seconds`` and ``predict_seconds`` for the model.
    ``numeric_features`` overrides the numeric columns and ``models`` restricts
    the estimators that run (``None`` keeps the default behaviour).
    """
    selected_splits = _validate_evaluate_splits(splits, evaluate_splits, allow_test)
    matrices, _ = prepare_model_matrices(splits, numeric_features=numeric_features)
    target = "delivery_delay_days"
    models = _select_models(
        {"OLS": None, **build_regression_tournament(random_state=random_state)},
        models,
    )
    rows: list[dict[str, object]] = []
    for model_name in list(models):
        started = time.perf_counter()
        if model_name == "OLS":
            models[model_name] = fit_ols_regression(
                matrices["train"], splits["train"][target]
            )
        else:
            models[model_name].fit(matrices["train"], splits["train"][target])
        fit_seconds = time.perf_counter() - started
        model = models[model_name]
        for split_name in selected_splits:
            started = time.perf_counter()
            predictions = (
                predict_ols(model, matrices[split_name])
                if model_name == "OLS"
                else model.predict(matrices[split_name])
            )
            predict_seconds = time.perf_counter() - started
            rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    **regression_metrics(splits[split_name][target], predictions),
                    "fit_seconds": fit_seconds,
                    "predict_seconds": predict_seconds,
                }
            )
    return pd.DataFrame(rows), models


def run_classification_tournament(
    splits: dict[str, pd.DataFrame],
    random_state: int = 42,
    evaluate_splits: tuple[str, ...] = ("validation",),
    allow_test: bool = False,
    calibration_cv: int = 3,
    numeric_features: list[str] | None = None,
    models: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fit calibrated classification estimators on approved temporal partitions.

    Every estimator is unweighted and wrapped in isotonic
    ``CalibratedClassifierCV`` fitted on the training partition only.
    ``numeric_features`` overrides the numeric columns and ``models`` restricts
    the estimators that run (``None`` keeps the default behaviour).
    """
    selected_splits = _validate_evaluate_splits(splits, evaluate_splits, allow_test)
    matrices, _ = prepare_model_matrices(splits, numeric_features=numeric_features)
    target = "sla_breach"
    base_models = _select_models(
        build_classification_tournament(
            splits["train"][target], random_state=random_state
        ),
        models,
    )
    fitted_models = {
        name: calibrated_classifier(estimator, cv=calibration_cv)
        for name, estimator in base_models.items()
    }
    rows: list[dict[str, object]] = []
    for model_name, model in fitted_models.items():
        started = time.perf_counter()
        model.fit(matrices["train"], splits["train"][target].astype(int))
        fit_seconds = time.perf_counter() - started
        for split_name in selected_splits:
            started = time.perf_counter()
            probabilities = model.predict_proba(matrices[split_name])[:, 1]
            predict_seconds = time.perf_counter() - started
            rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    **classification_metrics(
                        splits[split_name][target].astype(int), probabilities
                    ),
                    "fit_seconds": fit_seconds,
                    "predict_seconds": predict_seconds,
                }
            )
    return pd.DataFrame(rows), fitted_models


# Baseline scores that are not probabilities: Brier, ECE and thresholded
# metrics are undefined for them and are reported as NaN.
NON_PROBABILITY_BASELINES = frozenset({"negative_promised_lead_days"})
BASELINE_PREFIX = "baseline_"


def _regression_baseline_rows(
    fold: dict[str, pd.DataFrame],
) -> list[dict[str, object]]:
    """Evaluate the mandatory delay baselines on one fold's validation set."""
    predictions = regression_baseline_predictions(fold["train"], fold["validation"])
    return [
        {
            "model": f"{BASELINE_PREFIX}{name}",
            "split": "validation",
            **regression_metrics(fold["validation"]["delivery_delay_days"], values),
            "fit_seconds": np.nan,
            "predict_seconds": np.nan,
        }
        for name, values in predictions.items()
    ]


def _classification_baseline_rows(
    fold: dict[str, pd.DataFrame],
) -> list[dict[str, object]]:
    """Evaluate the mandatory breach baselines on one fold's validation set."""
    scores = classification_baseline_scores(fold["train"], fold["validation"])
    target = fold["validation"]["sla_breach"].astype(int)
    return [
        {
            "model": f"{BASELINE_PREFIX}{name}",
            "split": "validation",
            **classification_metrics(
                target, values, probabilistic=name not in NON_PROBABILITY_BASELINES
            ),
            "fit_seconds": np.nan,
            "predict_seconds": np.nan,
        }
        for name, values in scores.items()
    ]


def run_regression_tournament_folds(
    folds: list[dict[str, pd.DataFrame]],
    random_state: int = 42,
    include_baselines: bool = True,
    numeric_features: list[str] | None = None,
    models: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Run the regression tournament over expanding development folds.

    Baseline rows are added per fold with ``model = "baseline_<name>"``.
    """
    results = []
    all_models = {}
    for fold in folds:
        split_data = {"train": fold["train"], "validation": fold["validation"]}
        fold_results, fold_models = run_regression_tournament(
            split_data,
            random_state=random_state,
            evaluate_splits=("validation",),
            numeric_features=numeric_features,
            models=models,
        )
        if include_baselines:
            fold_results = pd.concat(
                [fold_results, pd.DataFrame(_regression_baseline_rows(fold))],
                ignore_index=True,
            )
        fold_results.insert(0, "fold", fold["fold"])
        results.append(fold_results)
        all_models[fold["fold"]] = fold_models
    return pd.concat(results, ignore_index=True), all_models


def run_classification_tournament_folds(
    folds: list[dict[str, pd.DataFrame]],
    random_state: int = 42,
    include_baselines: bool = True,
    numeric_features: list[str] | None = None,
    models: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Run the classification tournament over expanding development folds.

    Baseline rows are added per fold with ``model = "baseline_<name>"``; the
    non-probabilistic ``negative_promised_lead_days`` has NaN Brier/ECE.
    """
    results = []
    all_models = {}
    for fold in folds:
        split_data = {"train": fold["train"], "validation": fold["validation"]}
        fold_results, fold_models = run_classification_tournament(
            split_data,
            random_state=random_state,
            evaluate_splits=("validation",),
            numeric_features=numeric_features,
            models=models,
        )
        if include_baselines:
            fold_results = pd.concat(
                [fold_results, pd.DataFrame(_classification_baseline_rows(fold))],
                ignore_index=True,
            )
        fold_results.insert(0, "fold", fold["fold"])
        results.append(fold_results)
        all_models[fold["fold"]] = fold_models
    return pd.concat(results, ignore_index=True), all_models


_LOWER_IS_BETTER = {"mae", "rmse"}
_SUMMARY_METRICS = {
    "regression": ("mae", "rmse", "r2"),
    "classification": ("average_precision", "roc_auc", "brier", "ece_10bins"),
}
_PRIMARY_METRIC = {"regression": "rmse", "classification": "average_precision"}
_DECISION_METRICS = {
    "regression": ("mae", "rmse"),
    "classification": ("average_precision",),
}


def tournament_summary(
    results: pd.DataFrame,
    baseline_prefix: str = BASELINE_PREFIX,
) -> pd.DataFrame:
    """Summarize fold results as mean/std, gain over the best baseline and wins.

    For each decision metric (MAE and RMSE for regression, average precision
    for classification) the best baseline is the baseline model with the best
    mean over folds. ``<metric>_gain_vs_baseline`` is the relative improvement
    over that baseline mean (positive is better; lower MAE/RMSE and higher AP
    are better) and ``<metric>_folds_won`` counts the folds where the model
    beats that baseline's value in the same fold. Gains are NaN for the
    baselines themselves.
    """
    task = "regression" if "rmse" in results.columns else "classification"
    metrics = [m for m in _SUMMARY_METRICS[task] if m in results.columns]
    grouped = results.groupby("model", sort=False)
    summary = pd.DataFrame({"n_folds": grouped["fold"].nunique()})
    for metric in metrics:
        summary[f"{metric}_mean"] = grouped[metric].mean()
        summary[f"{metric}_std"] = grouped[metric].std()
    for column in ("fit_seconds", "predict_seconds"):
        if column in results.columns:
            summary[f"{column}_mean"] = grouped[column].mean()
    summary["is_baseline"] = [
        name.startswith(baseline_prefix) for name in summary.index
    ]
    baseline_names = [name for name in summary.index if name.startswith(baseline_prefix)]
    for metric in _DECISION_METRICS[task]:
        lower = metric in _LOWER_IS_BETTER
        gains = pd.Series(np.nan, index=summary.index)
        wins = pd.Series(np.nan, index=summary.index)
        best = pd.Series("", index=summary.index, dtype=object)
        if baseline_names:
            means = summary.loc[baseline_names, f"{metric}_mean"]
            best_name = means.idxmin() if lower else means.idxmax()
            reference = results.loc[results["model"] == best_name].set_index("fold")[metric]
            reference_mean = float(reference.mean())
            for name in summary.index[~summary["is_baseline"]]:
                values = results.loc[results["model"] == name].set_index("fold")[metric]
                aligned = reference.reindex(values.index)
                difference = aligned - values if lower else values - aligned
                gains[name] = (
                    (reference_mean - values.mean()) / abs(reference_mean)
                    if lower
                    else (values.mean() - reference_mean) / abs(reference_mean)
                )
                wins[name] = int((difference > 0).sum())
                best[name] = best_name
        summary[f"{metric}_gain_vs_baseline"] = gains
        summary[f"{metric}_folds_won"] = wins
        summary[f"{metric}_best_baseline"] = best
    primary = _PRIMARY_METRIC[task]
    return summary.sort_values(
        f"{primary}_mean", ascending=primary in _LOWER_IS_BETTER
    ).reset_index()


def regression_metrics(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    """Calculate regression metrics for a delay prediction."""
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "r2": float(r2_score(y_true, predictions)),
    }


def classification_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float = 0.5,
    probabilistic: bool = True,
) -> dict[str, float]:
    """Calculate threshold-free and thresholded classification metrics.

    With ``probabilistic=False`` the scores are only a ranking (not
    probabilities in [0, 1]); Brier, ECE and balanced accuracy at a 0.5
    threshold are then reported as NaN.
    """
    predictions = (probabilities >= threshold).astype(int)
    y_array = np.asarray(y_true, dtype=int)
    probability_array = np.asarray(probabilities, dtype=float)
    brier = (
        brier_score_loss(y_array, probability_array) if probabilistic else np.nan
    )
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0 if probabilistic else np.nan
    for lower, upper in zip(bins[:-1] if probabilistic else [], bins[1:], strict=False):
        mask = (
            (probability_array >= lower)
            & (probability_array <= upper if upper == 1 else probability_array < upper)
        )
        if mask.any():
            ece += mask.mean() * abs(
                probability_array[mask].mean() - y_array[mask].mean()
            )

    def top_k_metrics(fraction: float) -> tuple[float, float]:
        k = max(1, int(np.ceil(len(probability_array) * fraction)))
        top = np.argsort(-probability_array, kind="stable")[:k]
        true_positive = y_array[top].sum()
        return float(true_positive / k), float(true_positive / max(y_array.sum(), 1))

    precision_5, recall_5 = top_k_metrics(0.05)
    precision_10, recall_10 = top_k_metrics(0.10)
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(y_true, predictions))
            if probabilistic
            else float("nan")
        ),
        "brier": float(brier),
        "ece_10bins": float(ece),
        "precision_at_5pct": precision_5,
        "recall_at_5pct": recall_5,
        "precision_at_10pct": precision_10,
        "recall_at_10pct": recall_10,
    }


_ABLATION_REFERENCE_FEATURE = "distance_mean_km"
_ABLATION_MIN_SUBGROUP = 30
_ABLATION_METRICS = {
    "regression": ("mae", "rmse", "r2"),
    "classification": (
        "roc_auc",
        "average_precision",
        "balanced_accuracy",
        "brier",
        "ece_10bins",
        "precision_at_5pct",
        "recall_at_5pct",
        "precision_at_10pct",
        "recall_at_10pct",
    ),
}
_ABLATION_PRIMARY = {"regression": "mae", "classification": "average_precision"}
MULTI_SELLER_PREFIX = "multi_seller_"


def distance_ablation_variants() -> dict[str, list[str]]:
    """Return the numeric feature lists of the distance ablation variants.

    Every variant starts from the current :data:`NUMERIC_FEATURES` and replaces
    ``distance_mean_km`` in place: ``mean`` keeps it, ``max`` uses
    ``distance_max_km`` and ``max_plus_range`` uses ``distance_max_km`` plus
    ``distance_range_km`` (maximum minus minimum distance).
    """
    position = NUMERIC_FEATURES.index(_ABLATION_REFERENCE_FEATURE)

    def replace(replacement: list[str]) -> list[str]:
        return (
            NUMERIC_FEATURES[:position] + replacement + NUMERIC_FEATURES[position + 1 :]
        )

    return {
        "mean": list(NUMERIC_FEATURES),
        "max": replace(["distance_max_km"]),
        "max_plus_range": replace(["distance_max_km", "distance_range_km"]),
    }


def _multi_seller_metrics(
    task: str,
    y_true: pd.Series,
    predictions: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """Metrics on the multi-seller subgroup, NaN when it is too small or one-class."""
    names = _ABLATION_METRICS[task]
    valid = int(mask.sum()) >= _ABLATION_MIN_SUBGROUP
    if task == "classification" and valid:
        valid = y_true[mask].astype(int).nunique() == 2
    if not valid:
        return {f"{MULTI_SELLER_PREFIX}{name}": np.nan for name in names}
    if task == "regression":
        values = regression_metrics(y_true[mask], predictions[mask])
    else:
        values = classification_metrics(y_true[mask].astype(int), predictions[mask])
    return {f"{MULTI_SELLER_PREFIX}{name}": values[name] for name in names}


def run_distance_ablation(
    folds: list[dict[str, pd.DataFrame]],
    task: str,
    variants: dict[str, list[str]] | None = None,
    models: tuple[str, ...] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare distance feature variants on expanding folds.

    Parameters
    ----------
    folds:
        Output of ``expanding_window_folds`` (train/validation per fold).
    task:
        ``"regression"`` (``delivery_delay_days``) or ``"classification"``
        (``sla_breach``).
    variants:
        Mapping of variant name to the numeric feature list. ``None`` uses
        :func:`distance_ablation_variants` (``mean``, ``max``,
        ``max_plus_range``).
    models:
        Optional restriction of the estimators that run (baselines are not
        part of the ablation).
    random_state:
        Seed forwarded to the estimators.

    Returns
    -------
    pandas.DataFrame
        One row per ``(variant, fold, model)`` with the usual validation
        metrics, ``multi_seller_n`` (validation orders with
        ``seller_count > 1``) and the same metrics restricted to that subgroup
        under the ``multi_seller_`` prefix. The subgroup metrics are NaN when
        it has fewer than 30 orders or, in classification, a single class.
    """
    if task not in _ABLATION_METRICS:
        raise ValueError(f"unsupported task: {task}")
    variants = distance_ablation_variants() if variants is None else variants
    runner = (
        run_regression_tournament
        if task == "regression"
        else run_classification_tournament
    )
    target = "delivery_delay_days" if task == "regression" else "sla_breach"
    rows: list[dict[str, object]] = []
    for variant, numeric_features in variants.items():
        columns = list(numeric_features)
        for fold in folds:
            split_data = {"train": fold["train"], "validation": fold["validation"]}
            fold_results, fitted = runner(
                split_data,
                random_state=random_state,
                evaluate_splits=("validation",),
                numeric_features=columns,
                models=models,
            )
            matrices, _ = prepare_model_matrices(split_data, numeric_features=columns)
            validation = fold["validation"]
            mask = (validation["seller_count"] > 1).to_numpy()
            for record in fold_results.to_dict("records"):
                model = fitted[record["model"]]
                if task == "classification":
                    predictions = model.predict_proba(matrices["validation"])[:, 1]
                elif record["model"] == "OLS":
                    predictions = predict_ols(model, matrices["validation"])
                else:
                    predictions = model.predict(matrices["validation"])
                rows.append(
                    {
                        "variant": variant,
                        "fold": fold["fold"],
                        **record,
                        "multi_seller_n": int(mask.sum()),
                        **_multi_seller_metrics(
                            task, validation[target], np.asarray(predictions), mask
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _reference_comparison(
    values: pd.DataFrame,
    metric: str,
    reference: str,
    lower: bool,
) -> pd.DataFrame:
    """Per (variant, model): mean, std, relative delta vs reference, folds won."""
    rows = []
    for (variant, model), group in values.groupby(["variant", "model"], sort=False):
        series = group.set_index("fold")[metric]
        row = {
            "variant": variant,
            "model": model,
            f"{metric}_mean": series.mean(),
            f"{metric}_std": series.std(),
            f"{metric}_delta_vs_{reference}": np.nan,
            f"{metric}_folds_won": np.nan,
        }
        reference_rows = values.loc[
            (values["variant"] == reference) & (values["model"] == model)
        ]
        if variant != reference and len(reference_rows):
            ref = reference_rows.set_index("fold")[metric].reindex(series.index)
            paired = pd.concat([series, ref], axis=1, keys=["v", "r"]).dropna()
            if len(paired):
                ref_mean = paired["r"].mean()
                var_mean = paired["v"].mean()
                row[f"{metric}_delta_vs_{reference}"] = (
                    (ref_mean - var_mean) / abs(ref_mean)
                    if lower
                    else (var_mean - ref_mean) / abs(ref_mean)
                )
                better = (
                    paired["r"] - paired["v"] if lower else paired["v"] - paired["r"]
                )
                row[f"{metric}_folds_won"] = int((better > 0).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def ablation_summary(
    results: pd.DataFrame,
    reference: str = "mean",
    task: str | None = None,
) -> pd.DataFrame:
    """Summarize :func:`run_distance_ablation` per (variant, model).

    The primary metric is average precision for classification and MAE for
    regression. For the whole validation set and for the multi-seller subgroup
    the table reports the mean and standard deviation across folds, the
    relative delta of the mean versus the reference variant of the same model
    (positive is better, as in :func:`tournament_summary`) and the number of
    folds in which the variant beats the reference. Folds where either value is
    NaN (small or one-class subgroup) are ignored in the subgroup columns, and
    ``multi_seller_n_folds`` counts the folds with a valid subgroup metric. The
    reference variant has NaN delta and folds won.
    """
    if task is None:
        task = "regression" if "rmse" in results.columns else "classification"
    if task not in _ABLATION_PRIMARY:
        raise ValueError(f"unsupported task: {task}")
    if reference not in set(results["variant"]):
        raise KeyError(f"reference variant not in results: {reference}")
    metric = _ABLATION_PRIMARY[task]
    lower = metric in _LOWER_IS_BETTER
    overall = _reference_comparison(results, metric, reference, lower)
    subgroup_column = f"{MULTI_SELLER_PREFIX}{metric}"
    subgroup = _reference_comparison(
        results[["variant", "model", "fold", subgroup_column]].rename(
            columns={subgroup_column: metric}
        ),
        metric,
        reference,
        lower,
    ).rename(
        columns=lambda c: f"{MULTI_SELLER_PREFIX}{c}" if c.startswith(metric) else c
    )
    counts = (
        results.groupby(["variant", "model"], sort=False)
        .agg(
            n_folds=("fold", "nunique"),
            multi_seller_n_mean=("multi_seller_n", "mean"),
            multi_seller_n_folds=(subgroup_column, "count"),
        )
        .reset_index()
    )
    return counts.merge(overall, on=["variant", "model"]).merge(
        subgroup, on=["variant", "model"]
    )
