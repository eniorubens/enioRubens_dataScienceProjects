"""Dynamic, estimator-conditioned modeling pipeline — the project's authorial contract.

This module is the single, shared statement of the architecture first written
in :class:`src.optimizer.RegressionOptimizer`: **the preprocessing is not
fixed in advance**. For every Optuna trial the search jointly chooses a
feature *representation strategy* (``modeler_name``), a categorical encoder, a
scaler, an optional normalizer, a feature selector, the estimator's own
hyperparameters and, independently of all of those, a target transformer. The
menu of representation strategies offered to a trial depends on the *family*
of the estimator being optimized, because a periodic-spline expansion or a
kernel interaction map means something entirely different to a penalised
linear model than it does to a gradient-boosted tree.

The v3 reference implementation lives in ``RegressionOptimizer``
(``modeling_transformers``, ``get_normalization``, ``get_encoder``,
``get_standardization``, ``get_feature_selection``, ``get_parameters``) and is
deliberately left untouched as the historical record. What is ported here is
the *contract*, adapted to v4:

* every trainable step (imputer, normalizer, scaler, encoder, selector,
  target transformer) lives inside the pipeline, so it is refit from scratch
  on each temporal fold's training window;
* imputation happens inside the numeric or categorical branch it belongs to,
  never as a single global imputer bolted onto the ColumnTransformer's output;
* the target transformer is sampled from its own search space and reaches the
  model only through ``TransformedTargetRegressor`` — it is never the feature
  scaler wearing a second hat;
* nothing is read from module-level globals: every dependency arrives as an
  argument.

Three v3 behaviours are deliberately *not* reproduced; each is a defect rather
than a design choice, and each is called out at its definition below: the
``Sin_Cos`` / ``Time_steps_as_categories`` / ``Periodic_Spline`` branches
dropped every weather column, the ``Pairwise_Interactions`` weekday indicator
tested the wrong column, and the Nystroem branch imputed globally after
encoding.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from category_encoders import (
    BaseNEncoder,
    CountEncoder,
    JamesSteinEncoder,
    MEstimateEncoder,
    OrdinalEncoder,
    QuantileEncoder,
)
from feature_engine.encoding import CountFrequencyEncoder, MeanEncoder
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    SequentialFeatureSelector,
    f_regression,
    mutual_info_regression,
)
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MaxAbsScaler,
    MinMaxScaler,
    PolynomialFeatures,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from xgboost import XGBRegressor

from src.feature_engineering import ElapsedHoursTransformer, build_preprocessing_pipeline
from src.periodic_features import CosTransformer, PeriodicSplineTransformer, SinTransformer
from src.trend import RobustTrendResidualRegressor

# ---------------------------------------------------------------------------
# v4 candidate feature lists
# ---------------------------------------------------------------------------
#
# Supersede RegressionOptimizer._NUMERICAL_FEATURES/_CATEGORICAL_FEATURES:
# those v3 lists predate Ground Temp, Solar Radiation, Visibility, Sunshine,
# Cloud Cover and Rush_Period. ``Year`` is deliberately absent — every
# expanding-CV fold trains on a longer and more recent span than the previous
# one, so ``Year`` would let a model key on the historical growth curve itself
# instead of on weather/calendar signal. ``is_anomalous_2020`` is a
# retrospective regime label computed from demand and stays audit-only.

NUMERICAL_FEATURES: List[str] = [
    "Temperature(C)",
    "Dew point temperature(C)",
    "Ground Temp(C)",
    "Humidity(%)",
    "Solar Radiation (MJ/m2)",
    "Wind speed (m/s)",
    "Visibility (10m)",
    "Sunshine (hr)",
    "Cloud Cover (oktas)",
    "Month",
    "Hour",
    "Elapsed_Hours",
]

CATEGORICAL_FEATURES: List[str] = [
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
    "Sunshine Cat",
    "Cloud Cover Cat",
]

# Sunshine/Cloud Cover keep both a continuous and a categorical
# representation among the candidates; the redundancy is deliberate — the
# feature selectors are part of the search space and decide what survives.

# The two calendar columns that the periodic strategies re-encode. Whenever a
# modeler expands them cyclically they are removed from the plain numeric
# branch, so the same information never enters twice under two encodings.
TIME_COLUMNS: Tuple[str, ...] = ("Month", "Hour")

WEEK_STATUS_COLUMN = "WeekStatus"


# ---------------------------------------------------------------------------
# Estimator families
# ---------------------------------------------------------------------------

FAMILY_BASELINE = "baseline"
FAMILY_LINEAR = "linear"
FAMILY_TREE_ENSEMBLE = "tree_ensemble"
FAMILY_HIST_GRADIENT_BOOSTING = "hist_gradient_boosting"
FAMILY_XGBOOST = "xgboost"
FAMILY_LIGHTGBM = "lightgbm"
FAMILY_CATBOOST = "catboost"

ESTIMATOR_CLASSES: Dict[str, Any] = {
    "DummyRegressor": DummyRegressor,
    "Ridge": Ridge,
    "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
    "XGBRegressor": XGBRegressor,
    "LGBMRegressor": LGBMRegressor,
    "RandomForestRegressor": RandomForestRegressor,
    "CatBoostRegressor": CatBoostRegressor,
}

ESTIMATOR_FAMILIES: Dict[str, str] = {
    "DummyRegressor": FAMILY_BASELINE,
    "Ridge": FAMILY_LINEAR,
    "HistGradientBoostingRegressor": FAMILY_HIST_GRADIENT_BOOSTING,
    "XGBRegressor": FAMILY_XGBOOST,
    "LGBMRegressor": FAMILY_LIGHTGBM,
    "RandomForestRegressor": FAMILY_TREE_ENSEMBLE,
    "CatBoostRegressor": FAMILY_CATBOOST,
}


def estimator_family(estimator_name: str) -> str:
    """Return the search-space family of ``estimator_name``.

    Raises
    ------
    ValueError
        If the estimator has no declared family — a new estimator must be
        classified explicitly rather than silently inheriting a default space.
    """
    try:
        return ESTIMATOR_FAMILIES[estimator_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown estimator '{estimator_name}'. Declare its family in "
            f"ESTIMATOR_FAMILIES. Known: {sorted(ESTIMATOR_FAMILIES)}"
        ) from exc


# ---------------------------------------------------------------------------
# modeler_name — the representation strategies, per family
# ---------------------------------------------------------------------------

LINEAR_MODELING = "linear_modeling"
SIN_COS = "Sin_Cos"
TIME_STEPS_AS_CATEGORIES = "Time_steps_as_categories"
PERIODIC_SPLINE = "Periodic_Spline"
PAIRWISE_INTERACTIONS = "Pairwise_Interactions"
INTERACTIONS_WITH_KERNELS = "Interactions_with_Kernels"
NORMALIZERS = "Normalizers"
POLYNOMIAL = "Polynomial"

ALL_MODELERS: Tuple[str, ...] = (
    LINEAR_MODELING,
    NORMALIZERS,
    POLYNOMIAL,
    SIN_COS,
    TIME_STEPS_AS_CATEGORIES,
    PERIODIC_SPLINE,
    PAIRWISE_INTERACTIONS,
    INTERACTIONS_WITH_KERNELS,
)

# The four strategies that differ only in *how time is represented* — raw
# ordinal columns, sinusoids, categorical levels, or periodic splines. This is
# the whole of what a tree ensemble can meaningfully choose between: trees are
# invariant to any monotone per-column rescaling (so the Normalizers branch is
# a no-op for them modulo floating point) and they already discover feature
# interactions through successive splits (so Polynomial, Pairwise_Interactions
# and the Nystroem kernel map buy them dimensionality, not information).
_TIME_REPRESENTATIONS: Tuple[str, ...] = (
    LINEAR_MODELING,
    SIN_COS,
    TIME_STEPS_AS_CATEGORIES,
    PERIODIC_SPLINE,
)

MODELER_SPACES: Dict[str, Tuple[str, ...]] = {
    # A constant predictor has no representation to choose; the single-element
    # space keeps the categorical distribution well defined for the study.
    FAMILY_BASELINE: (LINEAR_MODELING,),
    # Penalised linear models are the family that actually needs a
    # representation search: periodicity, normalisation and explicit
    # interaction terms are the only way a linear predictor can express what a
    # tree gets for free from its splits.
    FAMILY_LINEAR: ALL_MODELERS,
    FAMILY_TREE_ENSEMBLE: _TIME_REPRESENTATIONS,
    FAMILY_HIST_GRADIENT_BOOSTING: _TIME_REPRESENTATIONS,
    FAMILY_XGBOOST: _TIME_REPRESENTATIONS,
    FAMILY_LIGHTGBM: _TIME_REPRESENTATIONS,
    FAMILY_CATBOOST: _TIME_REPRESENTATIONS,
}

# Strategies that multiply the column count (Nystroem components, degree-2
# polynomials, hour x week-status interaction terms). Pairing one of these
# with "no feature selection at all" is rejected as a structurally invalid
# combination — see validate_combination.
HIGH_DIMENSIONAL_MODELERS = frozenset(
    {INTERACTIONS_WITH_KERNELS, POLYNOMIAL, PAIRWISE_INTERACTIONS}
)

SEARCH_PROFILE_BROAD = "broad"
SEARCH_PROFILE_REFINED = "refined"
SEARCH_PROFILES = (SEARCH_PROFILE_BROAD, SEARCH_PROFILE_REFINED)

TARGET_STRATEGY_DIRECT = "direct"
TARGET_STRATEGY_ROBUST_TREND = "robust_trend_residual"
TARGET_STRATEGY_AUTO = "auto"
TARGET_STRATEGIES = (
    TARGET_STRATEGY_DIRECT,
    TARGET_STRATEGY_ROBUST_TREND,
    TARGET_STRATEGY_AUTO,
)


def modeler_space(
    estimator_name: str,
    search_profile: str = SEARCH_PROFILE_BROAD,
) -> List[str]:
    """Return the effective ``modeler_name`` candidates for one study."""
    candidates = MODELER_SPACES[estimator_family(estimator_name)]
    if search_profile == SEARCH_PROFILE_REFINED:
        candidates = _REFINED_MODELER_SPACES.get(estimator_name, candidates)
    return list(candidates)


def suggest_modeler_name(
    trial,
    estimator_name: str,
    search_profile: str = SEARCH_PROFILE_BROAD,
) -> str:
    """Sample the representation strategy for this trial.

    The candidate list depends only on the estimator's family, which is fixed
    for the whole study, so the categorical distribution registered under
    ``"modeler_name"`` never changes between trials of the same study.
    """
    candidates = modeler_space(estimator_name, search_profile)
    return trial.suggest_categorical("modeler_name", candidates)


# ---------------------------------------------------------------------------
# Normalizers, encoders, standardizers, target transformers
# ---------------------------------------------------------------------------

# Faithful port of the space RegressionOptimizer.get_normalization actually
# sampled. Its Yeo-Johnson / Box-Cox / row-Normalizer branches were
# unreachable (never listed among the candidates) and are not reproduced;
# Yeo-Johnson remains available where it is meaningful, on the target.
NORMALIZER_SPACE: Tuple[str, ...] = ("QuantileUniform", "QuantileNormal", "MinMaxScaler")

ENCODER_SPACE: Tuple[str, ...] = (
    "OrdinalEncoder",
    "MeanEncoder",
    "CountFrequencyEncoder",
    "CountEncoder",
    "BaseNEncoder",
    "JamesSteinEncoder",
    "MEstimateEncoder",
    "QuantileEncoder",
)

TARGET_TRANSFORM_SPACE: Tuple[str, ...] = (
    "none",
    "standard",
    "robust",
    "yeo_johnson",
    "log1p",
)

# The refined profile is empirical, not a replacement for the dynamic
# architecture.  It is the second search stage inferred from the first 400-trial
# experiment: the representation and encoding choices that remained competitive
# are still optimized jointly with the estimator, while branches that never
# appeared among the best trials no longer consume most of the wall-clock budget.
_REFINED_MODELER_SPACES: Dict[str, Tuple[str, ...]] = {
    "HistGradientBoostingRegressor": (PERIODIC_SPLINE,),
    "XGBRegressor": (LINEAR_MODELING,),
    "LGBMRegressor": (LINEAR_MODELING,),
    "CatBoostRegressor": (PERIODIC_SPLINE,),
    "RandomForestRegressor": (LINEAR_MODELING, PERIODIC_SPLINE),
}

_REFINED_ENCODER_SPACES: Dict[str, Tuple[str, ...]] = {
    "HistGradientBoostingRegressor": ("MeanEncoder", "CountFrequencyEncoder"),
    "XGBRegressor": ("JamesSteinEncoder", "CountFrequencyEncoder"),
    "LGBMRegressor": ("CountFrequencyEncoder", "QuantileEncoder"),
    "CatBoostRegressor": ("OrdinalEncoder", "MeanEncoder", "CountFrequencyEncoder"),
    "RandomForestRegressor": ("OrdinalEncoder", "CountFrequencyEncoder"),
}


def encoder_space(
    estimator_name: str,
    search_profile: str = SEARCH_PROFILE_BROAD,
) -> List[str]:
    """Return the effective categorical-encoder candidates for one study."""
    candidates = ENCODER_SPACE
    if search_profile == SEARCH_PROFILE_REFINED:
        candidates = _REFINED_ENCODER_SPACES.get(estimator_name, candidates)
    return list(candidates)


def get_normalization(trial) -> Tuple[Any, str]:
    """Sample a distribution-shaping normalizer for the numeric branch."""
    name = trial.suggest_categorical("normalizer", list(NORMALIZER_SPACE))
    if name == "QuantileUniform":
        return QuantileTransformer(output_distribution="uniform", random_state=42), name
    if name == "QuantileNormal":
        return QuantileTransformer(output_distribution="normal", random_state=42), name
    return MinMaxScaler(feature_range=(0, 1)), name


def suggest_encoder_name(
    trial,
    estimator_name: Optional[str] = None,
    search_profile: str = SEARCH_PROFILE_BROAD,
) -> str:
    """Sample the categorical encoder *name*.

    Only the name is sampled here: the concrete encoder is built later by
    :func:`build_encoder`, because some strategies (notably
    ``Time_steps_as_categories``) route extra columns through the categorical
    branch, and the feature_engine encoders need the exact column list in
    ``variables=``. Splitting name from instance keeps the Optuna call order —
    and therefore the registered distribution — identical for every trial.
    """
    candidates = (
        list(ENCODER_SPACE)
        if estimator_name is None
        else encoder_space(estimator_name, search_profile)
    )
    return trial.suggest_categorical("encoder", candidates)


def build_encoder(name: str, variables: Sequence[str]) -> Any:
    """Instantiate the encoder ``name`` for the given categorical columns.

    The two feature_engine encoders are built with ``unseen="encode"``. Their
    default is to emit NaN for a category absent from the training fold, and a
    temporal split guarantees such categories: a rainfall or cloud-cover level
    that never occurred before the fold boundary appears for the first time in
    the validation block. The v3 pipeline absorbed those NaNs with a global
    ``SimpleImputer`` bolted onto the ColumnTransformer's output; here the
    fallback stays inside the categorical branch, computed from the training
    fold alone — the target mean for MeanEncoder, a zero count for
    CountFrequencyEncoder. The category_encoders classes already behave this
    way through their own ``handle_unknown="value"`` default.
    """
    columns = list(variables)
    if name == "MeanEncoder":
        return MeanEncoder(variables=columns, unseen="encode")
    if name == "CountFrequencyEncoder":
        return CountFrequencyEncoder(encoding_method="count", variables=columns, unseen="encode")
    if name == "CountEncoder":
        return CountEncoder(cols=columns)
    if name == "BaseNEncoder":
        return BaseNEncoder(cols=columns)
    if name == "JamesSteinEncoder":
        return JamesSteinEncoder(cols=columns)
    if name == "MEstimateEncoder":
        return MEstimateEncoder(cols=columns)
    if name == "QuantileEncoder":
        return QuantileEncoder(cols=columns)
    return OrdinalEncoder(cols=columns)


def get_standardization(
    trial,
    estimator_name: str,
    search_profile: str = SEARCH_PROFILE_BROAD,
) -> Tuple[Any, str]:
    """Sample the numeric-branch scaler, conditioned on the estimator family.

    Ported from ``RegressionOptimizer.get_standardization``: a penalised
    linear model is pinned to ``StandardScaler`` (its penalty is only
    comparable across coefficients on a common scale, so leaving that to
    chance wastes trials), CatBoost is offered the two scalers that preserve
    its ordered-boosting statistics, and everything else gets all three.
    """
    family = estimator_family(estimator_name)
    if family == FAMILY_LINEAR:
        return StandardScaler(with_mean=True, with_std=True), "StandardScaler"
    if search_profile == SEARCH_PROFILE_REFINED:
        # A monotone column-wise rescaling cannot change a tree split.  Keeping
        # the step as passthrough removes a redundant Optuna dimension without
        # fixing any of the representation or encoder decisions.
        return "passthrough", "passthrough"
    if family == FAMILY_CATBOOST:
        name = trial.suggest_categorical("standardizer", ["StandardScaler", "MaxAbsScaler"])
    else:
        name = trial.suggest_categorical(
            "standardizer", ["StandardScaler", "MaxAbsScaler", "RobustScaler"]
        )
    if name == "MaxAbsScaler":
        return MaxAbsScaler(), name
    if name == "RobustScaler":
        return RobustScaler(), name
    return StandardScaler(with_mean=True, with_std=True), name


def get_target_transformer(trial) -> Tuple[Optional[Any], str]:
    """Sample the target transformer, independent of every feature transform.

    Applied exclusively inside ``TransformedTargetRegressor(transformer=...)``
    and never shared with the feature scaler from :func:`get_standardization`,
    which is what made the v3 ``max_label`` normalisation impossible to reason
    about: there, one object rescaled both sides of the fit.
    """
    name = trial.suggest_categorical("target_transform", list(TARGET_TRANSFORM_SPACE))
    if name == "none":
        return None, name
    if name == "standard":
        return StandardScaler(), name
    if name == "robust":
        return RobustScaler(), name
    if name == "yeo_johnson":
        return PowerTransformer(method="yeo-johnson"), name
    return FunctionTransformer(func=np.log1p, inverse_func=np.expm1), name


# ---------------------------------------------------------------------------
# Picklable module-level helpers
# ---------------------------------------------------------------------------
#
# Trials are cross-validated inside a spawned subprocess, so anything embedded
# in a pipeline must be importable by name — no lambdas, no bound methods.


def as_object_dtype(frame: pd.DataFrame) -> pd.DataFrame:
    """Cast every column to object dtype while preserving real NaNs.

    Unlike ``.astype(str)`` this never turns a missing value into the literal
    string ``"nan"``; it only strips the strict numeric/category dtype that
    would otherwise reject the ``"Missing"`` constant-fill token (for example
    ``DayNumberOnWeek`` is int64) or trip MeanEncoder's dtype check.
    """
    return frame.astype("object")


def weekday_indicator(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a 0/1 float frame marking working days.

    Reads ``WeekStatus`` (values ``"Weekday"``/``"Weekend"``), not
    ``Weekday`` — the v3 helper ``optimizer._is_weekday_frame`` tested
    ``df["Weekday"].eq("Weekday")``, but ``WeekdayWeekStatusTransformer``
    fills ``Weekday`` with day *names* (``"Monday"``, ...), so that comparison
    was constantly false and the v3 ``Pairwise_Interactions`` branch built its
    interaction terms against an all-zero column.
    """
    return frame[WEEK_STATUS_COLUMN].eq("Weekday").astype(float).to_frame()


class KernelExpansion(BaseEstimator, TransformerMixin):
    """Nystroem polynomial-kernel map applied on top of an inner preprocessor.

    v4 adaptation of ``periodic_features.CustomPreprocessorWithNystroem``. The
    v3 version rebuilt its own encoder/spline ColumnTransformer internally and
    then ran a single global ``SimpleImputer`` over the encoded matrix before
    the kernel map. Here the inner preprocessor is injected already carrying
    per-branch imputation, so no imputer ever sees the concatenated output.

    ``n_components`` defaults to 100 rather than v3's 300: the v4 development
    set is roughly ten times longer (about 81k hourly rows against 8.7k), and
    the downstream mutual-information selector's cost grows with the product
    of rows and columns.

    Parameters
    ----------
    preprocessor:
        Any fitted-on-demand transformer producing a dense numeric matrix.
    n_components, degree, random_state:
        Passed straight through to :class:`sklearn.kernel_approximation.Nystroem`.
    """

    def __init__(
        self,
        preprocessor: Any,
        n_components: int = 100,
        degree: int = 2,
        random_state: int = 0,
    ) -> None:
        self.preprocessor = preprocessor
        self.n_components = n_components
        self.degree = degree
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y=None) -> "KernelExpansion":
        """Fit the inner preprocessor and the Nystroem map on the training fold."""
        self.preprocessor_ = clone(self.preprocessor)
        transformed = self.preprocessor_.fit_transform(X, y)
        self.nystroem_ = Nystroem(
            kernel="poly",
            degree=self.degree,
            n_components=self.n_components,
            random_state=self.random_state,
        )
        self.nystroem_.fit(np.asarray(transformed, dtype=float))
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply the fitted inner preprocessor, then the fitted kernel map."""
        transformed = self.preprocessor_.transform(X)
        return self.nystroem_.transform(np.asarray(transformed, dtype=float))


# ---------------------------------------------------------------------------
# Branch builders — imputation always lives inside its own branch
# ---------------------------------------------------------------------------


def _numeric_branch(scaler, normalizer=None, polynomial: bool = False) -> Pipeline:
    """Median imputation, then optional normalizer, scaler and polynomial expansion."""
    steps: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if normalizer is not None:
        steps.append(("normalizer", normalizer))
    steps.append(("scaler", scaler))
    if polynomial:
        steps.append(("polynomial", PolynomialFeatures(degree=2, include_bias=False)))
    return Pipeline(steps=steps).set_output(transform="pandas")


def _categorical_branch(encoder) -> Pipeline:
    """Object cast, then constant ``"Missing"`` imputation, then the encoder.

    Only the two leading steps force pandas output: the encoder needs a named
    DataFrame on the way in, but its own return type is left alone so any
    encoder can be plugged in regardless of whether it implements
    ``get_feature_names_out``.
    """
    as_object = FunctionTransformer(as_object_dtype, feature_names_out="one-to-one").set_output(
        transform="pandas"
    )
    imputer = SimpleImputer(strategy="constant", fill_value="Missing").set_output(
        transform="pandas"
    )
    return Pipeline(steps=[("as_object", as_object), ("imputer", imputer), ("encoder", encoder)])


def _cyclic_branch(transformer) -> Pipeline:
    """Median imputation, then a periodic encoding of a single time column."""
    return Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("cyclic", transformer)])


def _without_time_columns(numeric_features: Sequence[str]) -> List[str]:
    """Drop Month/Hour from the plain numeric branch when they are re-encoded."""
    return [column for column in numeric_features if column not in TIME_COLUMNS]


def _periodic_column_transformer(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    scaler,
    encoder_name: str,
) -> ColumnTransformer:
    """Weather/numeric branch plus categorical branch plus periodic splines on Month/Hour."""
    plain_numeric = _without_time_columns(numeric_features)
    numeric_scaler = scaler if scaler == "passthrough" else clone(scaler)
    return ColumnTransformer(
        transformers=[
            ("numeric", _numeric_branch(numeric_scaler), plain_numeric),
            (
                "categorical",
                _categorical_branch(build_encoder(encoder_name, categorical_features)),
                list(categorical_features),
            ),
            ("cyclic_month", _cyclic_branch(PeriodicSplineTransformer(12, n_splines=6)), ["Month"]),
            ("cyclic_hour", _cyclic_branch(PeriodicSplineTransformer(24, n_splines=12)), ["Hour"]),
        ],
        remainder="drop",
    )


# ---------------------------------------------------------------------------
# The eight representation strategies
# ---------------------------------------------------------------------------


def build_modeling_transformer(
    modeler_name: str,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    scaler,
    encoder_name: str,
    normalizer=None,
    kernel_components: int = 100,
) -> Any:
    """Build the feature-representation transformer named by ``modeler_name``.

    v4 correction shared by ``Sin_Cos``, ``Time_steps_as_categories`` and
    ``Periodic_Spline``: the v3 versions declared ``remainder="drop"`` while
    listing only the categorical columns and the re-encoded Month/Hour, which
    silently discarded temperature, humidity, radiation and every other
    weather measurement. Here the remaining numeric columns keep flowing
    through their own imputed-and-scaled branch, and only the two calendar
    columns actually being re-encoded are removed from it.

    Parameters
    ----------
    modeler_name:
        One of :data:`ALL_MODELERS`.
    numeric_features, categorical_features:
        Candidate column names, as they exist after feature engineering.
    scaler:
        Unfitted scaler from :func:`get_standardization`.
    encoder_name:
        Name from :func:`suggest_encoder_name`; instantiated per branch so the
        feature_engine encoders receive the correct ``variables=``.
    normalizer:
        Unfitted normalizer from :func:`get_normalization`. Required by
        ``Normalizers`` and ``Polynomial``, ignored by the others.
    kernel_components:
        Nystroem component count for ``Interactions_with_Kernels``.
    """
    numeric_features = list(numeric_features)
    categorical_features = list(categorical_features)
    plain_numeric = _without_time_columns(numeric_features)

    if modeler_name == LINEAR_MODELING:
        return ColumnTransformer(
            transformers=[
                ("numeric", _numeric_branch(scaler), numeric_features),
                (
                    "categorical",
                    _categorical_branch(build_encoder(encoder_name, categorical_features)),
                    categorical_features,
                ),
            ],
            remainder="drop",
        )

    if modeler_name == NORMALIZERS:
        return ColumnTransformer(
            transformers=[
                ("numeric", _numeric_branch(scaler, normalizer=normalizer), numeric_features),
                (
                    "categorical",
                    _categorical_branch(build_encoder(encoder_name, categorical_features)),
                    categorical_features,
                ),
            ],
            remainder="drop",
        )

    if modeler_name == POLYNOMIAL:
        return ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    _numeric_branch(scaler, normalizer=normalizer, polynomial=True),
                    numeric_features,
                ),
                (
                    "categorical",
                    _categorical_branch(build_encoder(encoder_name, categorical_features)),
                    categorical_features,
                ),
            ],
            remainder="drop",
        )

    if modeler_name == SIN_COS:
        return ColumnTransformer(
            transformers=[
                ("numeric", _numeric_branch(scaler), plain_numeric),
                (
                    "categorical",
                    _categorical_branch(build_encoder(encoder_name, categorical_features)),
                    categorical_features,
                ),
                ("month_sin", _cyclic_branch(SinTransformer(12)), ["Month"]),
                ("month_cos", _cyclic_branch(CosTransformer(12)), ["Month"]),
                ("hour_sin", _cyclic_branch(SinTransformer(24)), ["Hour"]),
                ("hour_cos", _cyclic_branch(CosTransformer(24)), ["Hour"]),
            ],
            remainder="drop",
        )

    if modeler_name == TIME_STEPS_AS_CATEGORIES:
        time_as_categories = categorical_features + list(TIME_COLUMNS)
        return ColumnTransformer(
            transformers=[
                ("numeric", _numeric_branch(scaler), plain_numeric),
                (
                    "categorical",
                    _categorical_branch(build_encoder(encoder_name, time_as_categories)),
                    time_as_categories,
                ),
            ],
            remainder="drop",
        )

    if modeler_name == PERIODIC_SPLINE:
        return _periodic_column_transformer(
            numeric_features, categorical_features, scaler, encoder_name
        )

    if modeler_name == PAIRWISE_INTERACTIONS:
        interaction = make_pipeline(
            ColumnTransformer(
                transformers=[
                    (
                        "cyclic_hour",
                        _cyclic_branch(PeriodicSplineTransformer(24, n_splines=8)),
                        ["Hour"],
                    ),
                    (
                        "is_weekday",
                        FunctionTransformer(weekday_indicator),
                        [WEEK_STATUS_COLUMN],
                    ),
                ],
                remainder="drop",
            ),
            PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
        )
        return FeatureUnion(
            [
                (
                    "marginal",
                    _periodic_column_transformer(
                        numeric_features, categorical_features, scaler, encoder_name
                    ),
                ),
                ("interactions", interaction),
            ]
        )

    if modeler_name == INTERACTIONS_WITH_KERNELS:
        return KernelExpansion(
            preprocessor=_periodic_column_transformer(
                numeric_features, categorical_features, scaler, encoder_name
            ),
            n_components=kernel_components,
        )

    raise ValueError(f"Unknown modeler_name '{modeler_name}'. Known: {sorted(ALL_MODELERS)}")


# ---------------------------------------------------------------------------
# Estimator hyperparameter spaces
# ---------------------------------------------------------------------------

# Boosting ceiling for the two estimators whose per-fold budget is discovered
# by temporal early stopping (see ``src.temporal_optimizer``). It is a fixed
# high number rather than a sampled one, and that is the point: whatever the
# search sampled for ``n_estimators`` would become the ceiling early stopping
# could never exceed, so a low sample silently truncates the fit and the
# resulting "best iteration" reports where the budget ran out rather than where
# the validation loss stopped improving. LightGBM showed this most starkly —
# its space never sampled ``n_estimators`` at all, leaving the library default
# of 100 as an invisible cap that one fold reached at iteration 98. Pinning a
# ceiling far above any plausible stopping point hands the decision to the
# early-stopping rule, and removes a search dimension that was never free.
BOOSTING_CEILING = 2000
BOOSTING_BUDGET_TEMPORAL = "temporal_early_stopping"
BOOSTING_BUDGET_FIXED = "fixed_iterations"
BOOSTING_BUDGET_STRATEGIES: Tuple[str, ...] = (
    BOOSTING_BUDGET_TEMPORAL,
    BOOSTING_BUDGET_FIXED,
)

# A fold whose discovered budget reaches this share of the ceiling is recorded
# as truncated: early stopping may have been stopped by the ceiling rather than
# by the data, and the number cannot be read as a converged budget.
BOOSTING_CAP_RATIO = 0.9


def get_parameters(
    estimator_name: str,
    trial,
    search_profile: str = SEARCH_PROFILE_BROAD,
) -> dict:
    """Sample the estimator's own hyperparameters.

    ``HistGradientBoostingRegressor.early_stopping`` is pinned to ``False``:
    sklearn carves its internal validation slice with a random, non-temporal
    split, which would let rows from either side of a fold boundary train the
    stopping rule this optimizer otherwise enforces by timestamp. Disabling it
    is the simpler permitted option.

    Under the refined profile, XGBoost, LightGBM and CatBoost choose one of two
    explicit budget contracts. ``temporal_early_stopping`` pins a high ceiling
    and delegates the budget to a temporal tail inside each training fold;
    ``fixed_iterations`` samples the number of trees as a genuine
    hyperparameter and never opens the early-stopping carve. Keeping the
    contracts explicit prevents a sampled value from being mistaken for an
    accidental early-stopping ceiling.

    ``CatBoostRegressor`` is created on CPU: the v3 space forced
    ``task_type="GPU"`` and misspelled the class name, so that branch could
    never run at all.
    """
    if estimator_name == "DummyRegressor":
        return {"strategy": trial.suggest_categorical("strategy", ["mean", "median"])}

    if estimator_name == "Ridge":
        return {"alpha": trial.suggest_float("alpha", 1e-6, 25.0, log=True), "random_state": 42}

    if estimator_name == "HistGradientBoostingRegressor":
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_iter": trial.suggest_int("max_iter", 50, 500, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 60),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 1.0, log=True),
            "early_stopping": False,
            "random_state": 42,
        }
        if search_profile == SEARCH_PROFILE_REFINED:
            params["loss"] = trial.suggest_categorical(
                "loss_function", ["squared_error", "absolute_error"]
            )
        return params

    if estimator_name == "XGBRegressor":
        objective = "reg:squarederror"
        boosting_budget_strategy = BOOSTING_BUDGET_TEMPORAL
        if search_profile == SEARCH_PROFILE_REFINED:
            objective = trial.suggest_categorical(
                "loss_function",
                ["reg:squarederror", "reg:absoluteerror", "reg:pseudohubererror"],
            )
            boosting_budget_strategy = trial.suggest_categorical(
                "boosting_budget_strategy",
                list(BOOSTING_BUDGET_STRATEGIES),
            )
        n_estimators = (
            trial.suggest_int("fixed_iterations", 100, 600)
            if boosting_budget_strategy == BOOSTING_BUDGET_FIXED
            else BOOSTING_CEILING
        )
        return {
            "n_estimators": n_estimators,
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 20.0, log=True),
            "objective": objective,
            "eval_metric": "mae",
            "random_state": 42,
        }

    if estimator_name == "LGBMRegressor":
        objective = "regression"
        boosting_budget_strategy = BOOSTING_BUDGET_TEMPORAL
        if search_profile == SEARCH_PROFILE_REFINED:
            objective = trial.suggest_categorical(
                "loss_function", ["regression", "regression_l1", "huber"]
            )
            boosting_budget_strategy = trial.suggest_categorical(
                "boosting_budget_strategy",
                list(BOOSTING_BUDGET_STRATEGIES),
            )
        n_estimators = (
            trial.suggest_int("fixed_iterations", 100, 600)
            if boosting_budget_strategy == BOOSTING_BUDGET_FIXED
            else BOOSTING_CEILING
        )
        return {
            "objective": objective,
            "metric": "l1",
            "boosting_type": "gbdt",
            "force_col_wise": True,
            "n_estimators": n_estimators,
            "num_leaves": trial.suggest_int("num_leaves", 7, 255),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "max_depth": trial.suggest_int("max_depth", -1, 8),
            "verbosity": -1,
            "random_state": 42,
        }

    if estimator_name == "RandomForestRegressor":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 25),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 1.0]),
            "random_state": 42,
            "n_jobs": -1,
        }

    if estimator_name == "CatBoostRegressor":
        loss_function = "RMSE"
        boosting_budget_strategy = BOOSTING_BUDGET_TEMPORAL
        if search_profile == SEARCH_PROFILE_REFINED:
            loss_function = trial.suggest_categorical("loss_function", ["RMSE", "MAE"])
            boosting_budget_strategy = trial.suggest_categorical(
                "boosting_budget_strategy",
                list(BOOSTING_BUDGET_STRATEGIES),
            )
            iterations = (
                trial.suggest_int("fixed_iterations", 100, 600)
                if boosting_budget_strategy == BOOSTING_BUDGET_FIXED
                else BOOSTING_CEILING
            )
            # The best pre-v5 configuration used depth=10. Keeping that value
            # reachable lets the focused refinement test the old configuration
            # under the current CV protocol instead of silently excluding it.
            depth_high = 10
            l2_low, l2_high = 1e-1, 30.0
        else:
            iterations = trial.suggest_int("iterations", 100, 600)
            depth_high = 10
            l2_low, l2_high = 1e-2, 10.0
        return {
            "iterations": iterations,
            "depth": trial.suggest_int("depth", 4, depth_high),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "random_strength": trial.suggest_int("random_strength", 0, 100),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", l2_low, l2_high, log=True),
            "border_count": trial.suggest_int("border_count", 1, 255),
            "loss_function": loss_function,
            "eval_metric": "MAE",
            "allow_writing_files": False,
            "verbose": False,
            "task_type": "CPU",
            "random_state": 42,
        }

    raise ValueError(f"Unknown estimator: {estimator_name}")


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

_SELECTOR_SPACES: Dict[str, Tuple[str, ...]] = {
    # No representation and nothing to select for a constant predictor.
    FAMILY_BASELINE: ("NoSelector",),
    # Ridge exposes coef_, so the wrapper selectors are all usable.
    FAMILY_LINEAR: (
        "NoSelector",
        "SelectKBest",
        "RFE",
        "SelectFromModel",
        "SequentialFeatureSelector",
    ),
    # HistGradientBoostingRegressor exposes neither coef_ nor
    # feature_importances_, so RFE and SelectFromModel — which clone the trial
    # estimator and read one of those — cannot be offered. Sequential
    # selection is excluded on cost grounds carried over from v3, where
    # backward SFS around an HGB dominated the entire trial budget.
    FAMILY_HIST_GRADIENT_BOOSTING: ("NoSelector", "SelectKBest"),
    FAMILY_TREE_ENSEMBLE: ("NoSelector", "SelectKBest", "RFE", "SelectFromModel"),
    FAMILY_XGBOOST: ("NoSelector", "SelectKBest", "RFE", "SelectFromModel"),
    FAMILY_LIGHTGBM: ("NoSelector", "SelectKBest", "SelectFromModel"),
    FAMILY_CATBOOST: ("NoSelector", "SelectKBest", "SelectFromModel"),
}

_SCORE_FUNCS = {"mutual_info": mutual_info_regression, "f_regression": f_regression}


def selector_space(
    estimator_name: str,
    search_profile: str = SEARCH_PROFILE_BROAD,
) -> List[str]:
    """Return the effective feature selectors available to ``estimator_name``.

    The list depends only on the estimator family, never on the already
    sampled ``modeler_name`` — Optuna rejects a categorical distribution whose
    candidate list varies between trials of a study. Combinations that are
    invalid only for a particular representation are rejected afterwards, by
    :func:`validate_combination`.
    """
    candidates = _SELECTOR_SPACES[estimator_family(estimator_name)]
    if (
        search_profile == SEARCH_PROFILE_REFINED
        and estimator_family(estimator_name) != FAMILY_LINEAR
    ):
        candidates = ("NoSelector",)
    return list(candidates)


def get_feature_selection(
    trial,
    estimator_name: str,
    base_estimator,
    sfs_n_splits: int = 3,
    sfs_gap: int = 48,
    search_profile: str = SEARCH_PROFILE_BROAD,
) -> Tuple[Any, str]:
    """Sample a feature selector and its hyperparameters.

    ``SequentialFeatureSelector`` receives a ``TimeSeriesSplit`` with the same
    48-hour gap used by the outer validation. The v3 version passed ``cv=3``,
    which sklearn expands into a plain ``KFold`` — random, shuffled-in-time
    inner folds evaluating a selection whose whole purpose is to generalise
    forward.
    """
    candidates = selector_space(estimator_name, search_profile)
    name = trial.suggest_categorical("selector", candidates)

    if name == "NoSelector":
        return "passthrough", name

    if name == "SelectKBest":
        k = trial.suggest_int("kbest_k", 5, 20)
        score_func = trial.suggest_categorical("kbest_score_func", ["mutual_info", "f_regression"])
        return SelectKBest(_SCORE_FUNCS[score_func], k=k), name

    if name == "RFE":
        n_features = trial.suggest_int("rfe_n_features", 5, 20)
        step = trial.suggest_float("rfe_step", 0.1, 0.5, step=0.1)
        return (
            RFE(estimator=clone(base_estimator), n_features_to_select=n_features, step=step),
            name,
        )

    if name == "SelectFromModel":
        threshold = trial.suggest_categorical("sfm_threshold", ["median", "1.25*median", "mean"])
        max_features = trial.suggest_int("sfm_max_features", 5, 20)
        return (
            SelectFromModel(
                estimator=clone(base_estimator),
                threshold=threshold,
                max_features=max_features,
            ),
            name,
        )

    n_features = trial.suggest_int("sfs_n_features", 5, 15)
    return (
        SequentialFeatureSelector(
            estimator=clone(base_estimator),
            n_features_to_select=n_features,
            direction="backward",
            cv=TimeSeriesSplit(n_splits=sfs_n_splits, gap=sfs_gap),
            n_jobs=-1,
        ),
        name,
    )


# ---------------------------------------------------------------------------
# The assembled dynamic pipeline
# ---------------------------------------------------------------------------


@dataclass
class PipelineSpec:
    """Every choice Optuna made for one trial's pipeline, as plain strings.

    Doubles as the record written to MLflow tags and to the hand-off manifest,
    so a frozen candidate can be described — and rebuilt — without unpickling
    the estimator.
    """

    estimator: str
    family: str
    modeler_name: str
    encoder: str
    scaler: str
    selector: str
    target_transform: str
    boosting_budget_strategy: Optional[str] = None
    search_profile: str = SEARCH_PROFILE_BROAD
    normalizer: Optional[str] = None
    n_features_selected: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_tags(self) -> Dict[str, str]:
        """Flatten to a string-valued mapping suitable for MLflow tags."""
        payload = asdict(self)
        payload.pop("extra", None)
        return {key: str(value) for key, value in payload.items() if value is not None}


def validate_combination(spec: PipelineSpec) -> Optional[str]:
    """Return a rejection reason for a structurally invalid combination, else ``None``.

    This is where ``modeler_name``-dependent restrictions live. Encoding them
    here rather than by narrowing :func:`selector_space` is what keeps every
    Optuna categorical distribution constant across the trials of a study.
    """
    if spec.modeler_name in HIGH_DIMENSIONAL_MODELERS and spec.selector == "NoSelector":
        return (
            f"NoSelector is not allowed with the high-dimensional representation "
            f"'{spec.modeler_name}'"
        )
    return None


def build_dynamic_pipeline(
    trial,
    estimator_name: str,
    numeric_features: Sequence[str] = tuple(NUMERICAL_FEATURES),
    categorical_features: Sequence[str] = tuple(CATEGORICAL_FEATURES),
    estimator_classes: Mapping[str, Any] = ESTIMATOR_CLASSES,
    sfs_n_splits: int = 3,
    sfs_gap: int = 48,
    kernel_components: int = 100,
    search_profile: str = SEARCH_PROFILE_BROAD,
    target_strategy: str = TARGET_STRATEGY_DIRECT,
) -> Tuple[Any, PipelineSpec]:
    """Assemble one trial's complete pipeline and describe it.

    The conceptual order is fixed and every trainable step sits inside it, so
    a single ``fit`` on a fold's training window is enough to refit the whole
    thing::

        feature engineering (target-free)
        -> Elapsed_Hours
        -> dynamic preprocessing conditioned on the estimator
        (imputation inside each branch)
        -> feature selection conditioned on the estimator
        -> TransformedTargetRegressor(estimator, target transformer)

    Parameters are always sampled in the same order — representation, encoder,
    scaler, normalizer (only for the two strategies that use one), estimator
    hyperparameters, selector, target transformer — so a study's parameter
    distributions stay stable and a completed trial can be replayed by
    handing its ``FrozenTrial`` back to this function.

    Returns
    -------
    tuple
        The unfitted pipeline and the :class:`PipelineSpec` describing it.
    """
    if search_profile not in SEARCH_PROFILES:
        raise ValueError(f"search_profile must be one of {SEARCH_PROFILES}.")
    if target_strategy not in TARGET_STRATEGIES:
        raise ValueError(f"target_strategy must be one of {TARGET_STRATEGIES}.")

    family = estimator_family(estimator_name)

    modeler_name = suggest_modeler_name(trial, estimator_name, search_profile)
    encoder_name = suggest_encoder_name(trial, estimator_name, search_profile)
    scaler, scaler_name = get_standardization(trial, estimator_name, search_profile)

    normalizer = None
    normalizer_name = None
    if modeler_name in (NORMALIZERS, POLYNOMIAL):
        normalizer, normalizer_name = get_normalization(trial)

    params = get_parameters(estimator_name, trial, search_profile)
    base_estimator = estimator_classes[estimator_name](**params)

    selector, selector_name = get_feature_selection(
        trial,
        estimator_name,
        base_estimator,
        sfs_n_splits=sfs_n_splits,
        sfs_gap=sfs_gap,
        search_profile=search_profile,
    )
    resolved_target_strategy = target_strategy
    if target_strategy == TARGET_STRATEGY_AUTO:
        resolved_target_strategy = trial.suggest_categorical(
            "target_strategy",
            [TARGET_STRATEGY_DIRECT, TARGET_STRATEGY_ROBUST_TREND],
        )

    trend_damping = None
    if resolved_target_strategy == TARGET_STRATEGY_ROBUST_TREND:
        target_transformer, target_name = None, TARGET_STRATEGY_ROBUST_TREND
        trend_damping = trial.suggest_float("trend_extrapolation_damping", 0.0, 1.0, step=0.25)
    elif search_profile == SEARCH_PROFILE_REFINED:
        if estimator_name == "CatBoostRegressor":
            # CatBoost is the only refined estimator for which the raw target
            # is reintroduced. Its ordered boosting and robust losses can model
            # the count scale directly, and the old raw-target configuration is
            # a specifically evidenced hypothesis rather than a global search
            # expansion.
            target_name = trial.suggest_categorical("target_transform", ["none", "log1p"])
            target_transformer = (
                None
                if target_name == "none"
                else FunctionTransformer(func=np.log1p, inverse_func=np.expm1)
            )
        else:
            target_transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1)
            target_name = "log1p"
    else:
        target_transformer, target_name = get_target_transformer(trial)

    modeling_transformer = build_modeling_transformer(
        modeler_name,
        numeric_features,
        categorical_features,
        scaler,
        encoder_name,
        normalizer=normalizer,
        kernel_components=kernel_components,
    )

    core_pipeline = Pipeline(
        steps=[
            ("feature_engineering", build_preprocessing_pipeline()),
            ("elapsed_hours", ElapsedHoursTransformer()),
            ("modeling", modeling_transformer),
            ("selector", selector),
            (
                "regressor",
                TransformedTargetRegressor(
                    regressor=base_estimator, transformer=target_transformer
                ),
            ),
        ]
    )
    pipeline = (
        RobustTrendResidualRegressor(
            estimator=core_pipeline,
            extrapolation_damping=trend_damping,
        )
        if resolved_target_strategy == TARGET_STRATEGY_ROBUST_TREND
        else core_pipeline
    )

    spec = PipelineSpec(
        estimator=estimator_name,
        family=family,
        modeler_name=modeler_name,
        encoder=encoder_name,
        scaler=scaler_name,
        selector=selector_name,
        target_transform=target_name,
        boosting_budget_strategy=(
            getattr(trial, "params", {}).get(
                "boosting_budget_strategy",
                BOOSTING_BUDGET_TEMPORAL,
            )
            if estimator_name in ("XGBRegressor", "LGBMRegressor", "CatBoostRegressor")
            else None
        ),
        search_profile=search_profile,
        normalizer=normalizer_name,
    )
    return pipeline, spec


def count_output_features(pipeline: Any, X: pd.DataFrame, n_rows: int = 50) -> Optional[int]:
    """Return how many columns reach the estimator, or ``None`` if unavailable.

    Pushes a short slice of ``X`` through every fitted step except the
    regressor. Used only for reporting the selected-feature count; a failure
    here must never break a run, so any exception yields ``None``.
    """
    try:
        core = getattr(pipeline, "estimator_", pipeline)
        transformed = core[:-1].transform(X.head(n_rows))
        return int(np.asarray(transformed).shape[1])
    except Exception:  # pragma: no cover - diagnostics only
        return None
