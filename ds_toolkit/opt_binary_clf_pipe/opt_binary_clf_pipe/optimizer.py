"""
optimizer.py
------------
Optuna-based pipeline optimiser and threshold-tuning routine.
"""
from __future__ import annotations

import gc
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp
from typing import Any

import numpy as np
import optuna
import pandas as pd
from feature_engine.encoding import (
    CountFrequencyEncoder,
    DecisionTreeEncoder,
    MeanEncoder,
)
from imblearn.over_sampling import SMOTE
from optuna.samplers import TPESampler
from pandas.api.types import is_string_dtype
from sklearn import ensemble, linear_model, neural_network, tree
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    SequentialFeatureSelector,
    mutual_info_classif,
)
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

import category_encoders

from .callbacks import EarlyStoppingCallback
from .estimators import define_estimators, get_estimator
from .persistence import ModelStore
from .pipeline_builder import build_full_pipeline, build_num_transformer, build_preprocessor
from .scoring import (
    get_classification_scoring,
    get_objective_scoring,
)


# ---------------------------------------------------------------------------
# Default hyperparameter search spaces
# ---------------------------------------------------------------------------

def _default_param_space(estimator_name: str, random_seed: int, balanced: bool) -> dict | None:
    """
    Return the default Optuna parameter space for *estimator_name*.

    Returns ``None`` when the estimator is not explicitly listed
    (empty params → default constructor).
    """
    return None  # populated by Optimizer.get_parameters via trial


# ---------------------------------------------------------------------------
# Threshold decision helper
# ---------------------------------------------------------------------------

def apply_threshold_decision(
    estimator_name: str,
    pipeline_name: str,
    optimization_name: str,
    model: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    cv: Any,
    model_store: ModelStore,
    metric_df: pd.DataFrame,
    estimator_params: dict | None = None,
    decision_metric: str = "Recall_macro",
    thresholds: np.ndarray | None = None,
    base_scores: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """
    Tune the decision threshold and persist the selected experiment row.

    Threshold search uses out-of-fold training probabilities to avoid leakage.
    Timing metrics are inherited from the upstream cross-validation scores.

    Parameters
    ----------
    estimator_name : str
        Estimator label.
    pipeline_name : str
        Human-readable pipeline description.
    optimization_name : str
        Optimisation strategy label.
    model : fitted pipeline
        Already-fitted sklearn-compatible pipeline.
    X_train, y_train : training data
    X_test, y_test : test data
    cv : cross-validation splitter
    model_store : ModelStore
        Storage backend for metrics and serialized pipelines.
    metric_df : pd.DataFrame
        Accumulated metrics DataFrame.
    estimator_params : dict | None
        Hyperparameters from the best trial.
    decision_metric : str, default="Recall_macro"
        Column used to select the best threshold.
    thresholds : np.ndarray | None
        Custom threshold grid.  Defaults to ``np.linspace(0, 1, 101)``.
    base_scores : dict | None
        Cross-validation timing scores to inherit fit/score times from.

    Returns
    -------
    pd.DataFrame
        Updated metrics DataFrame.
    """
    if thresholds is None:
        thresholds = np.linspace(0, 1, 101)

    y_proba_oof = cross_val_predict(
        model, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]

    rows = []
    for thr in thresholds:
        y_pred = (y_proba_oof >= thr).astype(int)
        rows.append(
            {
                "Threshold": float(thr),
                "Recall_macro": recall_score(y_train, y_pred, average="macro", zero_division=0),
                "Precision_macro": precision_score(y_train, y_pred, average="macro", zero_division=0),
                "F1_macro": f1_score(y_train, y_pred, average="macro", zero_division=0),
                "Balanced Accuracy": balanced_accuracy_score(y_train, y_pred),
            }
        )

    df_thr = pd.DataFrame(rows)
    best_idx = df_thr[decision_metric].idxmax()
    best_row = df_thr.loc[best_idx]
    best_threshold = float(best_row["Threshold"])

    candidates = df_thr.loc[np.isclose(df_thr["Threshold"], 0.5)]
    baseline_row = (
        candidates.iloc[0]
        if not candidates.empty
        else df_thr.loc[(df_thr["Threshold"] - 0.5).abs().idxmin()]
    )

    improvement = best_row[decision_metric] - baseline_row[decision_metric]
    use_threshold = improvement > 0.002
    final_threshold = best_threshold if use_threshold else 0.5
    data_model_tag = (
        f"{optimization_name} + Threshold tuning" if use_threshold else optimization_name
    )

    y_test_score = model.predict_proba(X_test)[:, 1]
    y_train_pred = (y_proba_oof >= final_threshold).astype(int)
    y_test_pred = (y_test_score >= final_threshold).astype(int)

    if base_scores is None:
        fit_time = score_time = np.nan
    else:
        fit_time = float(np.mean(base_scores["fit_time"]))
        score_time = float(np.mean(base_scores["score_time"]))

    scores = {
        "train_roc_auc": [roc_auc_score(y_train, y_proba_oof)],
        "test_roc_auc": [roc_auc_score(y_test, y_test_score)],
        "train_balanced_accuracy": [balanced_accuracy_score(y_train, y_train_pred)],
        "test_balanced_accuracy": [balanced_accuracy_score(y_test, y_test_pred)],
        "train_recall_macro": [recall_score(y_train, y_train_pred, average="macro", zero_division=0)],
        "test_recall_macro": [recall_score(y_test, y_test_pred, average="macro", zero_division=0)],
        "train_precision_macro": [precision_score(y_train, y_train_pred, average="macro", zero_division=0)],
        "test_precision_macro": [precision_score(y_test, y_test_pred, average="macro", zero_division=0)],
        "train_f1_macro": [f1_score(y_train, y_train_pred, average="macro", zero_division=0)],
        "test_f1_macro": [f1_score(y_test, y_test_pred, average="macro", zero_division=0)],
        "fit_time": [fit_time],
        "score_time": [score_time],
    }

    threshold_params = dict(estimator_params or {})
    threshold_params["threshold"] = final_threshold

    return model_store.save(
        description=estimator_name,
        data_model=data_model_tag,
        encoder=f"{pipeline_name} | thr={final_threshold:.3f}",
        pipeline_obj=model,
        scores=scores,
        params=threshold_params,
        metric_df=metric_df,
    )


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class Optimizer:
    """
    Optimise a full classification pipeline with Optuna.

    Parameters
    ----------
    estimator_name : str
        Classifier class name (e.g. ``"RandomForestClassifier"``).
    balanced : bool
        When ``True``, class-weight balancing is used inside the estimator
        instead of over-sampling.
    train_features : pd.DataFrame
        Training feature matrix.
    train_labels : np.ndarray
        Training target vector.
    cv : cross-validation splitter
        Any sklearn-compatible CV object.
    random_seed : int
        Reproducibility seed.
    x_reference : pd.DataFrame
        Reference DataFrame used for column-dtype inspection (categorical
        detection).
    model_store : ModelStore
        Storage backend for metrics and serialized pipelines.
    ratio_min : float | None, default=None
        Lower bound for ``scale_pos_weight`` (XGBClassifier only).
    ratio_max : float | None, default=None
        Upper bound for ``scale_pos_weight`` (XGBClassifier only).
    resampler : str, default=""
        Fixed over-sampler name (overrides Optuna search).
    sampling_method : str, default=""
        ``"Over"`` to include over-sampling in the search space.
    trials : int, default=50
        Maximum number of Optuna trials.
    early_stopping_patience : int, default=20
        Patience for :class:`~opt_binary_clf_pipe.callbacks.EarlyStoppingCallback`.
    early_stopping_min_delta : float, default=1e-4
        Min-delta for early stopping.
    timeout : int | None, default=300
        Time budget in seconds (``None`` disables timeout).
    test_features : pd.DataFrame | None, default=None
        Test feature matrix (used in ``detailed_objective``).
    test_labels : np.ndarray | None, default=None
        Test target vector.
    metric_df : pd.DataFrame | None, default=None
        Accumulated metrics DataFrame.
    invalid_configs_path : str, default="./dataset/invalid_configs.csv"
        Path for the invalid-configurations cache.
    param_space : dict | None, default=None
        Custom hyperparameter search space that **overrides** the built-in
        defaults for *estimator_name*.  Pass a callable
        ``(trial, balanced, random_seed) -> dict`` or a plain ``dict``
        with static values.
    """

    def __init__(
        self,
        estimator_name: str,
        balanced: bool,
        train_features: pd.DataFrame,
        train_labels: np.ndarray,
        cv: Any,
        random_seed: int,
        x_reference: pd.DataFrame,
        model_store: ModelStore,
        ratio_min: float | None = None,
        ratio_max: float | None = None,
        resampler: str = "",
        sampling_method: str = "",
        trials: int = 50,
        early_stopping_patience: int = 20,
        early_stopping_min_delta: float = 1e-4,
        timeout: int | None = 300,
        test_features: pd.DataFrame | None = None,
        test_labels: np.ndarray | None = None,
        metric_df: pd.DataFrame | None = None,
        invalid_configs_path: str = "./dataset/invalid_configs.csv",
        param_space: Any | None = None,
    ) -> None:
        self.estimator_name = estimator_name
        self.balanced = balanced
        self.train_features = train_features
        self.train_labels = train_labels
        self.cv = cv
        self.random_seed = random_seed
        self.x_reference = x_reference
        self.model_store = model_store
        self.resampler = resampler
        self.sampling_method = sampling_method
        self.trials = trials
        self.sampler = TPESampler(seed=random_seed)
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.timeout = timeout
        self.test_features = test_features
        self.test_labels = test_labels
        self.metric_df = metric_df
        self.param_space = param_space
        self.invalid_path = Path(invalid_configs_path)

        if ratio_min is not None and ratio_max is not None:
            self.ratio_min = min(float(ratio_min), float(ratio_max))
            self.ratio_max = max(float(ratio_min), float(ratio_max))
        else:
            self.ratio_min = ratio_min
            self.ratio_max = ratio_max

        (
            self.linear_estimators,
            self.ensemble_estimators,
            self.tree_estimators,
            self.neural_estimators,
            self.xgb_estimators,
            self.lgbm_estimators,
            self.catboost_estimators,
            self.extra_estimators,
            self.available_estimators,
        ) = define_estimators()

        self.scoring_full = get_classification_scoring()
        self.scoring_objective = get_objective_scoring()

        cols = ["estimator", "encoder", "scaler", "normalizer", "selector", "sampler"]
        self.invalid_df = (
            pd.read_csv(self.invalid_path)
            if self.invalid_path.exists()
            else pd.DataFrame(columns=cols)
        )

    # ------------------------------------------------------------------
    # Configuration signature helpers
    # ------------------------------------------------------------------

    def build_config_signature(
        self,
        encoder_name: str | None,
        scaler_name: str | None,
        normalizer_name: str | None,
        selector_name: str | None,
        sampler_name: str | None,
    ) -> dict[str, str]:
        """Return a canonical dict identifying a pipeline configuration."""
        return {
            "estimator": self.estimator_name,
            "encoder": encoder_name or "None",
            "scaler": scaler_name or "None",
            "normalizer": normalizer_name or "None",
            "selector": selector_name or "None",
            "sampler": sampler_name or "None",
        }

    def was_invalid_config(self, encoder_name, scaler_name, normalizer_name,
                        selector_name, sampler_name) -> bool:
        """Return ``True`` if this configuration was previously flagged as invalid."""
        if self.invalid_df.empty:
            return False
        sig = self.build_config_signature(
            encoder_name, scaler_name, normalizer_name, selector_name, sampler_name
        )
        mask = (
            (self.invalid_df["estimator"] == sig["estimator"])
            & (self.invalid_df["encoder"] == sig["encoder"])
            & (self.invalid_df["scaler"] == sig["scaler"])
            & (self.invalid_df["normalizer"] == sig["normalizer"])
            & (self.invalid_df["selector"] == sig["selector"])
            & (self.invalid_df["sampler"] == sig["sampler"])
        )
        return bool(mask.any())

    def register_invalid_config(self, encoder_name, scaler_name, normalizer_name,
                                selector_name, sampler_name, reason: str = "") -> None:
        """Add a configuration to the in-memory invalid cache."""
        sig = self.build_config_signature(
            encoder_name, scaler_name, normalizer_name, selector_name, sampler_name
        )
        sig["reason"] = reason
        self.invalid_df = pd.concat(
            [self.invalid_df, pd.DataFrame([sig])], ignore_index=True
        ).drop_duplicates(
            subset=["estimator", "encoder", "scaler", "normalizer", "selector", "sampler"]
        )

    def flush_invalid_configs(self) -> None:
        """Persist the invalid-configurations cache to disk."""
        self.invalid_path.parent.mkdir(parents=True, exist_ok=True)
        self.invalid_df.to_csv(self.invalid_path, index=False)

    def is_invalid_pipeline(self, encoder_name, scaler_name, normalizer_name,
                            selector_name, sampler_name) -> bool:
        """Apply known static invalid-pipeline rules (currently CatBoost only)."""
        if self.estimator_name == "CatBoostClassifier":
            return any([
                encoder_name not in ["NoEncoding", "NoCategoricalEncoder"],
                scaler_name not in ["NoScaling", None],
                normalizer_name not in ["NoNormalization", None],
                selector_name in ["RFE", "SequentialFeatureSelector"],
            ])
        return False

    # ------------------------------------------------------------------
    # Component samplers
    # ------------------------------------------------------------------

    def get_parameters(self, trial: optuna.Trial) -> dict:
        """
        Return estimator hyperparameters for *trial*.

        If a custom ``param_space`` was provided to the constructor, it takes
        priority over the built-in defaults.
        """
        # Custom override
        if self.param_space is not None:
            if callable(self.param_space):
                return self.param_space(trial, self.balanced, self.random_seed)
            return dict(self.param_space)

        # Built-in defaults
        params: dict = {}

        if self.estimator_name == "SGDClassifier":
            params = {
                "max_iter": trial.suggest_int("max_iter", 300, 2000, log=True),
                "alpha": trial.suggest_float("alpha", 0.01, 1.0, log=True),
                "learning_rate": trial.suggest_categorical(
                    "learning_rate", ["optimal", "constant", "invscaling"]
                ),
                "loss": trial.suggest_categorical("loss", ["log_loss", "modified_huber"]),
                "penalty": trial.suggest_categorical("penalty", ["l2", "l1"]),
                "eta0": trial.suggest_int("eta0", 1, 10, step=1),
                "early_stopping": True,
                "random_state": self.random_seed,
            }
            if self.balanced:
                params["class_weight"] = "balanced"

        elif self.estimator_name == "RandomForestClassifier":
            params = {
                "criterion": trial.suggest_categorical("criterion", ["entropy", "gini"]),
                "max_depth": trial.suggest_int("max_depth", 5, 8, step=1),
                "max_features": trial.suggest_int("max_features", 7, 14, step=1),
                "n_estimators": trial.suggest_int("n_estimators", 140, 300, log=True),
                "min_samples_split": trial.suggest_int("min_samples_split", 20, 70, log=True),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 16, step=1),
                "random_state": self.random_seed,
            }
            if self.balanced:
                params["class_weight"] = trial.suggest_categorical(
                    "class_weight", [{0: 0.265778, 1: 0.734222}, "balanced"]
                )

        elif self.estimator_name == "XGBClassifier":
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.1, 1.0, log=True),
                "max_depth": trial.suggest_int("max_depth", 1, 6, step=1),
                "subsample": trial.suggest_float("subsample", 0.1, 1.0, log=True),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0, log=True),
            }
            if (
                self.balanced
                and self.ratio_min is not None
                and self.ratio_max is not None
            ):
                params["scale_pos_weight"] = trial.suggest_float(
                    "scale_pos_weight",
                    float(self.ratio_min),
                    float(self.ratio_max),
                    log=True,
                )

        elif self.estimator_name == "LGBMClassifier":
            '''
                Estes parâmetros foram testados, mas não trouxeram melhoria significativa 
                no recall_macro, então foram comentados para retornar a configuração 
                antiga abaixo. 
                
                params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 12, step=1),
                "num_leaves": trial.suggest_int("num_leaves", 20, 300, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "random_state": self.random_seed,
                "verbose": -1,
            }'''
            
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "boosting_type": "gbdt",
                "force_col_wise": True,
                "num_leaves": trial.suggest_int("num_leaves", 2, 256, step=1),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.1, log=True
                ),
                "min_child_samples": trial.suggest_int(
                    "min_child_samples", 5, 100, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.1, 1.0, log=True),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.1, 1.0, log=True
                ),
                "reg_alpha": trial.suggest_float(
                    "reg_alpha", 1e-9, 100.0, log=True
                ),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda", 1e-9, 100.0, log=True
                ),
                "max_depth": trial.suggest_int("max_depth", -1, 8, step=1),
                "verbosity": -1,
                "random_state": self.random_seed,
            }
            if self.balanced:
                params["class_weight"] = "balanced"

        elif self.estimator_name == "CatBoostClassifier":
            params = {
                "iterations": trial.suggest_int("iterations", 100, 1000, log=True),
                "depth": trial.suggest_int("depth", 4, 10, step=1),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
                "random_state": self.random_seed,
                "verbose": 0,
            }
            if self.balanced:
                params["auto_class_weights"] = "Balanced"

        elif self.estimator_name == "LogisticRegression":
            params = {
                "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
                "max_iter": trial.suggest_int("max_iter", 200, 2000, log=True),
                "solver": trial.suggest_categorical(
                    "solver", ["lbfgs", "saga", "liblinear"]
                ),
                "random_state": self.random_seed,
            }
            if self.balanced:
                params["class_weight"] = "balanced"

        elif self.estimator_name in ("DecisionTreeClassifier", "ExtraTreeClassifier"):
            params = {
                "max_depth": trial.suggest_int("max_depth", 2, 20, step=1),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 377, log=True),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 377, log=True),
                "random_state": self.random_seed,
            }
            if self.balanced:
                params["class_weight"] = "balanced"

        elif self.estimator_name == "ExtraTreesClassifier":
            params = {
                "max_depth": trial.suggest_int("max_depth", 2, 20, step=1),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 377, log=True),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 377, log=True),
                "random_state": self.random_seed,
            }

        elif self.estimator_name == "SVC":
            params = {
                "C": trial.suggest_float("C", 1e-2, 100.0, log=True),
                "kernel": trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"]),
                "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
                "probability": True,
                "random_state": self.random_seed,
            }
            if self.balanced:
                params["class_weight"] = "balanced"

        elif self.estimator_name == "LinearDiscriminantAnalysis":
            solver = trial.suggest_categorical("solver", ["svd", "lsqr", "eigen"])
            params = {"solver": solver}
            if solver in ("lsqr", "eigen"):
                params["shrinkage"] = trial.suggest_categorical("shrinkage", [None, "auto"])

        elif self.estimator_name == "GaussianNB":
            params = {
                "var_smoothing": trial.suggest_float("var_smoothing", 1e-12, 1e-6, log=True),
            }

        elif self.estimator_name == "BalancedRandomForestClassifier":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 12, step=1),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 50, log=True),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20, step=1),
                "random_state": self.random_seed,
            }

        return params

    def get_standardization(self, trial: optuna.Trial) -> tuple[object | None, str | None]:
        """Sample and return a scaler from the Optuna search space."""
        name = trial.suggest_categorical(
            "standardizer", ["MaxAbsScaler", "StandardScaler", "RobustScaler", None]
        )
        mapping = {
            "MaxAbsScaler": MaxAbsScaler(),
            "StandardScaler": StandardScaler(),
            "RobustScaler": RobustScaler(),
        }
        return mapping.get(name), name

    def get_normalization(self, trial: optuna.Trial) -> tuple[object | None, str | None]:
        """Sample and return a normalizer from the Optuna search space."""
        name = trial.suggest_categorical(
            "normalizer",
            [
                "PowerTransformer", "QuantileTransformer", "QuantileTransformerN",
                "MinMaxScaler", "Normalizer", None,
            ],
        )
        mapping = {
            "PowerTransformer": PowerTransformer(),
            "QuantileTransformer": QuantileTransformer(output_distribution="uniform"),
            "QuantileTransformerN": QuantileTransformer(output_distribution="normal"),
            "Normalizer": Normalizer(),
            "MinMaxScaler": MinMaxScaler(feature_range=(0, 1)),
        }
        return mapping.get(name), name

    def get_feature_selection(
        self, trial: optuna.Trial, estimator: object
    ) -> tuple[object, str, int]:
        """Sample and return a feature selector from the Optuna search space."""
        if self.estimator_name == "CatBoostClassifier":
            return "passthrough", "NoFeatureSelection", {}

        name = trial.suggest_categorical(
            "selector",
            ["SelectKBest", "RFE", "SelectFromModel", "SequentialFeatureSelector"],
        )
        n = trial.suggest_int("n_features_to_select", 6, 13, step=1)

        if name == "SelectKBest":
            return SelectKBest(score_func=mutual_info_classif, k=n), name, n
        if name == "RFE":
            return RFE(estimator=estimator, n_features_to_select=n, step=1), name, n
        if name == "SelectFromModel":
            return SelectFromModel(estimator=estimator), name, n
        return (
            SequentialFeatureSelector(
                estimator=estimator, n_features_to_select="auto",
                cv=3, tol=None, direction="backward",
            ),
            name, n,
        )

    def get_encoders(self, trial: optuna.Trial) -> tuple[object, str]:
        """Sample and return a categorical encoder from the Optuna search space."""
        categorical = [
            c for c in self.x_reference.columns
            if is_string_dtype(self.x_reference[c])
        ]

        if not categorical:
            return "passthrough", "NoCategoricalEncoder"

        all_encoders = [
            "OrdinalEncoder", "OneHotEncoder", "BinaryEncoder", "BaseNEncoder",
            "HelmertEncoder", "SumEncoder", "PolynomialEncoder", "MeanEncoder",
            "CountFrequencyEncoder", "LeaveOneOutEncoder", "TargetEncoder",
            "CatBoostEncoder", "WOEEncoder", "MEstimateEncoder", "JamesSteinEncoder",
            "GLMMEncoder", "QuantileEncoder", "DecisionTreeEncoder",
        ]
        name = trial.suggest_categorical("encoder", all_encoders)

        simple_map = {
            "OrdinalEncoder": category_encoders.OrdinalEncoder,
            "OneHotEncoder": category_encoders.OneHotEncoder,
            "BinaryEncoder": category_encoders.BinaryEncoder,
            "BaseNEncoder": category_encoders.BaseNEncoder,
            "HelmertEncoder": category_encoders.HelmertEncoder,
            "SumEncoder": category_encoders.SumEncoder,
            "PolynomialEncoder": category_encoders.PolynomialEncoder,
            "LeaveOneOutEncoder": category_encoders.LeaveOneOutEncoder,
            "TargetEncoder": category_encoders.TargetEncoder,
            "CatBoostEncoder": category_encoders.CatBoostEncoder,
            "WOEEncoder": category_encoders.WOEEncoder,
            "MEstimateEncoder": category_encoders.MEstimateEncoder,
            "JamesSteinEncoder": category_encoders.JamesSteinEncoder,
            "GLMMEncoder": category_encoders.GLMMEncoder,
            "QuantileEncoder": category_encoders.QuantileEncoder,
        }

        if name in simple_map:
            return simple_map[name](), name
        if name == "MeanEncoder":
            return MeanEncoder(variables=categorical), name
        if name == "CountFrequencyEncoder":
            return CountFrequencyEncoder(encoding_method="count", variables=categorical), name
        if name == "DecisionTreeEncoder":
            return (
                DecisionTreeEncoder(
                    encoding_method="arbitrary", cv=3, scoring="recall_macro",
                    param_grid={"max_depth": [1, 2, 3, 4, 5, 6]},
                    regression=False, variables=categorical,
                ),
                name,
            )
        raise ValueError(f"Unsupported encoder: {name}")

    def get_sampler(self, trial: optuna.Trial) -> tuple[object | None, str | None]:
        """Sample and return an over-sampler."""
        sampler_name: str | None = None

        if self.resampler:
            sampler_name = self.resampler
        elif self.sampling_method == "Over":
            sampler_name = trial.suggest_categorical("resampler", ["SMOTE"])
        elif self.sampling_method in {"Under", "Combination"}:
            raise ValueError(f"{self.sampling_method}-sampling is disabled for automatic runs.")
        elif self.sampling_method:
            raise ValueError(f"Unsupported sampling method: {self.sampling_method}")

        if sampler_name is None:
            return None, None
        if sampler_name == "SMOTE":
            return SMOTE(random_state=self.random_seed, sampling_strategy="minority"), sampler_name
        raise ValueError(f"Unsupported resampler: {sampler_name}")

    # ------------------------------------------------------------------
    # Objective functions
    # ------------------------------------------------------------------

    def objective(self, trial: optuna.Trial) -> float:
        """Fast Optuna objective — optimises recall_macro only."""
        encoder, encoder_name = self.get_encoders(trial)
        sampler_, sampler_name = self.get_sampler(trial)
        scaler, scaler_name = self.get_standardization(trial)
        normalizer, normalizer_name = self.get_normalization(trial)

        num_transformer = build_num_transformer(scaler, normalizer)
        preprocessor = build_preprocessor(num_transformer)
        params = self.get_parameters(trial)

        estimator_cls = get_estimator(
            self.estimator_name,
            self.linear_estimators, self.ensemble_estimators,
            self.tree_estimators, self.neural_estimators,
            self.xgb_estimators, self.lgbm_estimators, self.catboost_estimators,
            self.extra_estimators,
        )
        model = estimator_cls(**params)
        feature_selector, selector_name, _ = self.get_feature_selection(trial, estimator_cls(**params))

        if self.is_invalid_pipeline(encoder_name, scaler_name, normalizer_name, selector_name, sampler_name):
            self.register_invalid_config(encoder_name, scaler_name, normalizer_name, selector_name, sampler_name, "Known invalid pipeline rule")
            raise optuna.exceptions.TrialPruned()

        if self.was_invalid_config(encoder_name, scaler_name, normalizer_name, selector_name, sampler_name):
            raise optuna.exceptions.TrialPruned()

        cache_dir = mkdtemp()
        try:
            pipe = build_full_pipeline(encoder, preprocessor, sampler_, feature_selector, model, cache_dir, self.balanced)
            scores = cross_validate(
                pipe, self.train_features, self.train_labels,
                cv=self.cv, return_train_score=False,
                scoring=self.scoring_objective, return_estimator=False,
                verbose=0, n_jobs=-1,
            )
            score = float(np.nanmean(scores["test_recall_macro"]))
            if not np.isfinite(score):
                self.register_invalid_config(encoder_name, scaler_name, normalizer_name, selector_name, sampler_name, "Non-finite recall_macro")
                raise optuna.exceptions.TrialPruned()
            return score

        except Exception as err:
            self.register_invalid_config(encoder_name, scaler_name, normalizer_name, selector_name, sampler_name, str(err)[:300])
            raise optuna.exceptions.TrialPruned()

        finally:
            rmtree(cache_dir, ignore_errors=True)
            gc.collect()

    def detailed_objective(self, trial: optuna.Trial) -> None:
        """Refit and log the best trial with the full metric set."""
        encoder, encoder_name = self.get_encoders(trial)
        sampler_, sampler_name = self.get_sampler(trial)
        scaler, scaler_name = self.get_standardization(trial)
        normalizer, normalizer_name = self.get_normalization(trial)

        num_transformer = build_num_transformer(scaler, normalizer)
        preprocessor = build_preprocessor(num_transformer)
        params = self.get_parameters(trial)

        estimator_cls = get_estimator(
            self.estimator_name,
            self.linear_estimators, self.ensemble_estimators,
            self.tree_estimators, self.neural_estimators,
            self.xgb_estimators, self.lgbm_estimators, self.catboost_estimators,
            self.extra_estimators,
        )
        model = estimator_cls(**params)
        feature_selector, selector_name, n_features = self.get_feature_selection(trial, estimator_cls(**params))

        cache_dir = mkdtemp()
        try:
            pipe = build_full_pipeline(encoder, preprocessor, sampler_, feature_selector, model, cache_dir, self.balanced)
            scores = cross_validate(
                pipe, self.train_features, self.train_labels,
                cv=self.cv, return_train_score=True,
                scoring=self.scoring_full, return_estimator=True,
                verbose=0, n_jobs=-1,
            )
            pipe.fit(self.train_features, self.train_labels)
        finally:
            rmtree(cache_dir, ignore_errors=True)
            gc.collect()

        parts = [
            p for p in [
                encoder_name, normalizer_name, scaler_name,
                f"{selector_name} {n_features}" if selector_name else None,
                sampler_name,
            ]
            if p is not None
        ]
        pipeline_name = " ".join(str(p).strip() for p in parts)

        opt_name = "Recall Macro Opt " + ("Balanced" if self.balanced else f"{self.sampling_method}-sampling")

        self.metric_df = apply_threshold_decision(
            estimator_name=self.estimator_name,
            pipeline_name=pipeline_name,
            optimization_name=opt_name,
            model=pipe,
            X_train=self.train_features,
            y_train=self.train_labels,
            X_test=self.test_features,
            y_test=self.test_labels,
            cv=self.cv,
            model_store=self.model_store,
            metric_df=self.metric_df,
            estimator_params=params,
            base_scores=scores,
        )

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def optimize(self) -> pd.DataFrame | None:
        """
        Run Optuna optimisation and persist the best trial.

        Returns
        -------
        pd.DataFrame | None
            Updated metrics DataFrame, or ``None`` if no trials completed.
        """
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(direction="maximize", sampler=self.sampler)
        early_stopping = EarlyStoppingCallback(
            patience=self.early_stopping_patience,
            min_delta=self.early_stopping_min_delta,
        )
        study.optimize(
            self.objective,
            n_trials=self.trials,
            timeout=self.timeout,
            callbacks=[early_stopping],
        )

        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        self.flush_invalid_configs()

        if not completed:
            print(f"[WARN] No completed trials for {self.estimator_name}.")
            return None

        self.detailed_objective(study.best_trial)
        return self.metric_df
