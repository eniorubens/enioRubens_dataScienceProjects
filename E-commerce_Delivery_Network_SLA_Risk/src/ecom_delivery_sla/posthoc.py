"""Post-hoc F2 evaluation: simulated deployment with an update on 2018-08-01 (notebook 10).

The F2 evaluation is **post-hoc and exploratory** (P-06 in
``docs/REGISTRO-DE-DECISOES.md``, plan in ``docs/PLANO-F2-POS-HOC.md``). It never
replaces the pre-registered F1 evaluation. The simulated operation updates the
model on :data:`F2_UPDATE_DATE` using only the labels already known on that
date (orders delivered strictly before it) and is scored on the purchases of
August 2018 with a daily block bootstrap.

Closed list of variants, all reported:

``A``
    The frozen F1 model, not retrained.
``B``
    The F1 configuration retrained on every label known at the update date.
``C``
    ``B`` with exponential recency weights (half-life of 90 days).
``D``
    ``B`` with a recent isotonic recalibration (the classifier is fitted on
    labels delivered before ``update - 30 days`` and calibrated on the labels
    delivered in the last 30 days).
``E``
    ``B`` plus the as-of feature ``promise_vs_recent_median``.

Variants ``C``, ``D`` and ``E`` must first pass a non-inferiority gate on the
development folds; failed variants are not scored in August. The delivery date
is only used to decide which labels are known; it is never a feature.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from .final_evaluation import (
    CALIBRATION_CV,
    CANDIDATE_MODEL,
    FINAL_RANDOM_STATE,
    FINAL_TRAIN_END,
    PRIMARY_VERSION,
    REFERENCE_BASELINE,
    TIMESTAMP_COLUMN,
    _paired_row,
    _run_bootstrap,
    _run_frozen,
    _summarize_bootstrap,
    _top_k_table,
    _validate_version,
    reliability_table,
)
from .modeling import (
    BASELINE_PREFIX,
    CATEGORICAL_FEATURES,
    NON_PROBABILITY_BASELINES,
    NUMERIC_FEATURES,
    build_classification_tournament,
    calibrated_classifier,
    classification_baseline_scores,
    classification_metrics,
    make_preprocessor,
)
from .run_cache import _default_cache_root
from .temporal import expanding_window_folds

F2_UPDATE_DATE = "2018-08-01"
"""Instant of the simulated model update (labels known strictly before it)."""

F2_EVAL_START = "2018-08-01"
"""Inclusive start of the evaluated purchases."""

F2_EVAL_END = "2018-09-01"
"""Exclusive end of the evaluated purchases."""

F1_TRAIN_END = FINAL_TRAIN_END
"""Exclusive end of the F1 training window (reused from the F1 evaluation)."""

DEV_END = "2018-07-01"
"""Exclusive end of the development period used by the gate."""

RECENCY_HALF_LIFE_DAYS = 90
RECALIBRATION_WINDOW_DAYS = 30
PROMISE_LOOKBACK_DAYS = 30
GATE_AP_RATIO = 0.95
GATE_ECE_MARGIN = 0.01
GATE_FIRST_MONTH = "2017-10"
GATE_LAST_MONTH = "2018-06"
GATE_EMBARGO_DAYS = 30

PROMISE_FEATURE = "promise_vs_recent_median"
DELIVERY_COLUMN = "order_delivered_customer_date"
STATE_COLUMN = "customer_state"
PROMISE_COLUMN = "promised_lead_days"

REFERENCE_NAME = "F1_config"
"""Label of the F1-configuration reference in the gate tables."""

VARIANT_NAMES = {
    "A": "A_f1_frozen",
    "B": "B_retrain",
    "C": "C_recency",
    "D": "D_recalibration",
    "E": "E_promise_feature",
}
GATED_VARIANTS = ("C", "D", "E")
BASELINE_RULE = f"{BASELINE_PREFIX}negative_promised_lead_days"
STATUS_EVALUATED = "avaliada"
STATUS_BASELINE = "baseline"
STATUS_FAILED = "reprovada no gate"
READING = "exploratório"
"""Fixed reading of every F2 result: exploratory, never confirmatory."""


# --------------------------------------------------------------------------
# Known labels, weights and recalibration windows
# --------------------------------------------------------------------------
def attach_delivery_dates(
    modeling_frame: pd.DataFrame,
    supervised_orders: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return a copy of the frame with the delivery date, used only to decide label availability.

    Parameters
    ----------
    modeling_frame:
        Supervised order-level frame. If it already has
        ``order_delivered_customer_date`` that column is kept.
    supervised_orders:
        Supervised orders (one row per ``order_id``) providing the delivery
        date. Required when the frame lacks the column.

    Returns
    -------
    pandas.DataFrame
        Copy with a parsed ``order_delivered_customer_date`` column. The model
        feature lists never include it, so it cannot leak into a model.

    Raises
    ------
    KeyError
        If the delivery date is available from neither argument.
    ValueError
        If an order has no delivery date or ``supervised_orders`` repeats an order.
    """
    result = modeling_frame.copy()
    if DELIVERY_COLUMN in result.columns:
        result[DELIVERY_COLUMN] = pd.to_datetime(result[DELIVERY_COLUMN], errors="coerce")
    else:
        if supervised_orders is None or DELIVERY_COLUMN not in supervised_orders.columns:
            raise KeyError(
                f"{DELIVERY_COLUMN} is required to know which labels exist at the update date; "
                "pass supervised_orders or a modeling frame that contains it"
            )
        if supervised_orders["order_id"].duplicated().any():
            raise ValueError("supervised_orders must contain one row per order_id")
        dates = pd.to_datetime(supervised_orders.set_index("order_id")[DELIVERY_COLUMN], errors="coerce")
        result[DELIVERY_COLUMN] = result["order_id"].map(dates)
    if result[DELIVERY_COLUMN].isna().any():
        raise ValueError("every supervised order must have a delivery date")
    return result


def known_label_mask(
    frame: pd.DataFrame,
    as_of: str | pd.Timestamp,
    *,
    timestamp_col: str = TIMESTAMP_COLUMN,
    delivery_col: str = DELIVERY_COLUMN,
) -> pd.Series:
    """Flag the orders whose label is known at ``as_of``.

    A label is known when the order was delivered strictly before ``as_of``
    (and purchased before it). Orders delivered on or after ``as_of`` have no
    label yet, even if their purchase is old.

    Parameters
    ----------
    frame:
        Frame with the purchase timestamp and the delivery date. Use
        :func:`attach_delivery_dates` if the delivery date is in another table.
    as_of:
        Instant of the update.
    timestamp_col, delivery_col:
        Column names.

    Returns
    -------
    pandas.Series
        Boolean mask aligned with ``frame``.

    Raises
    ------
    KeyError
        If a required column is missing.
    """
    for column in (timestamp_col, delivery_col):
        if column not in frame.columns:
            raise KeyError(f"column not found: {column}")
    cutoff = pd.Timestamp(as_of)
    delivered = pd.to_datetime(frame[delivery_col], errors="coerce")
    purchased = pd.to_datetime(frame[timestamp_col], errors="coerce")
    return (delivered < cutoff) & (purchased < cutoff)


def recency_weights(
    purchase_timestamp: pd.Series | pd.DatetimeIndex,
    update_date: str | pd.Timestamp,
    half_life_days: float = RECENCY_HALF_LIFE_DAYS,
) -> np.ndarray:
    """Return ``0.5 ** (age_days / half_life_days)`` with age = update date - purchase.

    Parameters
    ----------
    purchase_timestamp:
        Purchase timestamps, all before ``update_date``.
    update_date:
        Instant of the update.
    half_life_days:
        Age, in days, at which the weight halves.

    Returns
    -------
    numpy.ndarray
        Weights in ``(0, 1]``.

    Raises
    ------
    ValueError
        If a purchase is after the update date or the half-life is not positive.
    """
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    stamps = pd.to_datetime(pd.Series(np.asarray(purchase_timestamp)))
    age_days = (pd.Timestamp(update_date) - stamps).dt.total_seconds().to_numpy() / 86_400.0
    if (age_days < 0).any():
        raise ValueError("purchases after the update date cannot be weighted")
    return 0.5 ** (age_days / float(half_life_days))


def recalibration_split(
    known: pd.DataFrame,
    update_date: str | pd.Timestamp,
    window_days: float = RECALIBRATION_WINDOW_DAYS,
    *,
    time_col: str = DELIVERY_COLUMN,
) -> tuple[pd.Series, pd.Series]:
    """Split known labels into the classifier-fit set and the recent calibration set.

    Parameters
    ----------
    known:
        Labels already known at the update date.
    update_date:
        Instant of the update.
    window_days:
        Length of the recent window.
    time_col:
        Column that defines recency: the delivery date at the real update
        (labels are known when delivered) or the purchase timestamp inside the
        development folds, which have no delivery criterion.

    Returns
    -------
    tuple
        ``(fit_mask, calibration_mask)``. Fit holds ``time < update - window``;
        calibration holds ``update - window <= time < update``.
    """
    update = pd.Timestamp(update_date)
    boundary = update - pd.Timedelta(days=window_days)
    time = pd.to_datetime(known[time_col])
    return time < boundary, (time >= boundary) & (time < update)


# --------------------------------------------------------------------------
# promise_vs_recent_median
# --------------------------------------------------------------------------
def promise_vs_recent_median(
    frame: pd.DataFrame,
    window_days: float = PROMISE_LOOKBACK_DAYS,
    *,
    timestamp_col: str = TIMESTAMP_COLUMN,
    state_col: str = STATE_COLUMN,
    promise_col: str = PROMISE_COLUMN,
) -> pd.Series:
    """Ratio of an order's promise to the recent median promise of the same state.

    For an order bought at ``t`` the denominator is the median of
    ``promised_lead_days`` over purchases with timestamp in
    ``[t - window_days, t)`` (strictly earlier, never the order itself nor
    purchases with the same timestamp) and the same ``customer_state``. When the
    state has no purchase in the window, the global median of the same window is
    used; when that is also unavailable (or not positive), the value is NaN and
    the pipeline imputer fills it. No label is used. The computation is
    vectorized with time-based rolling windows (one per state, not per order).

    Parameters
    ----------
    frame:
        Frame with purchase timestamp, destination state and promised lead time.
    window_days:
        Length of the look-back window.
    timestamp_col, state_col, promise_col:
        Column names.

    Returns
    -------
    pandas.Series
        Feature named ``promise_vs_recent_median`` aligned with ``frame.index``.

    Raises
    ------
    KeyError
        If a required column is missing.
    ValueError
        If a purchase timestamp is missing.
    """
    for column in (timestamp_col, state_col, promise_col):
        if column not in frame.columns:
            raise KeyError(f"column not found: {column}")
    stamps = pd.to_datetime(frame[timestamp_col], errors="coerce")
    if stamps.isna().any():
        raise ValueError(f"timestamp column contains invalid or missing values: {timestamp_col}")
    order = np.argsort(stamps.to_numpy(), kind="stable")
    index = pd.DatetimeIndex(stamps.to_numpy()[order])
    promise = pd.Series(frame[promise_col].astype(float).to_numpy()[order], index=index)
    states = frame[state_col].to_numpy()[order]
    window = f"{int(window_days)}D" if float(window_days).is_integer() else pd.Timedelta(days=window_days)

    def rolling_median(values: pd.Series) -> np.ndarray:
        return values.rolling(window, closed="left").median().to_numpy()

    global_median = rolling_median(promise)
    state_median = np.full(len(promise), np.nan)
    codes, uniques = pd.factorize(pd.Series(states), use_na_sentinel=True)
    for code in range(len(uniques)):
        positions = np.flatnonzero(codes == code)
        state_median[positions] = rolling_median(promise.iloc[positions])
    median = np.where(np.isnan(state_median), global_median, state_median)
    sorted_promise = promise.to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(median > 0, sorted_promise / median, np.nan)
    values = np.empty(len(frame))
    values[order] = ratio
    return pd.Series(values, index=frame.index, name=PROMISE_FEATURE)


# --------------------------------------------------------------------------
# Model fitting
# --------------------------------------------------------------------------
def _require_columns(frame: pd.DataFrame, numeric_features: list[str], name: str) -> None:
    """Raise ``KeyError`` when model features are missing from ``frame``."""
    missing = sorted((set(numeric_features) | set(CATEGORICAL_FEATURES)) - set(frame.columns))
    if missing:
        raise KeyError(f"missing model features in {name}: {missing}")


def _catboost(y: pd.Series, catboost_params: dict | None, random_state: int) -> object:
    """Build the tournament CatBoost (no class weights), with optional overrides."""
    estimator = build_classification_tournament(y, random_state=random_state)[CANDIDATE_MODEL]
    if catboost_params:
        estimator.set_params(**catboost_params)
    return estimator


def _fit_calibrated_model(
    train: pd.DataFrame,
    *,
    numeric_features: list[str],
    sample_weight: np.ndarray | None = None,
    catboost_params: dict | None = None,
    random_state: int = FINAL_RANDOM_STATE,
    estimator: object | None = None,
) -> Pipeline:
    """Fit the F1 configuration (preprocessor + isotonic ``cv=3`` CatBoost) on ``train``.

    ``sample_weight`` is forwarded through :class:`CalibratedClassifierCV`, which
    hands it to the base estimator and to the isotonic calibrator.
    """
    _require_columns(train, numeric_features, "train")
    y = train["sla_breach"].astype(int)
    preprocessor = make_preprocessor(numeric_features=numeric_features)
    preprocessor.set_output(transform="pandas")
    matrix = preprocessor.fit_transform(train)
    base = estimator if estimator is not None else _catboost(y, catboost_params, random_state)
    classifier = calibrated_classifier(base, cv=CALIBRATION_CV)
    if sample_weight is None:
        classifier.fit(matrix, y)
    else:
        classifier.fit(matrix, y, sample_weight=np.asarray(sample_weight, dtype=float))
    return Pipeline([("preprocessor", preprocessor), ("calibrated_catboost", classifier)])


def _fit_recalibrated_model(
    fit_set: pd.DataFrame,
    calibration_set: pd.DataFrame,
    *,
    numeric_features: list[str],
    catboost_params: dict | None = None,
    random_state: int = FINAL_RANDOM_STATE,
) -> CalibratedClassifierCV:
    """Fit CatBoost on ``fit_set`` and isotonic calibration only on ``calibration_set``.

    The preprocessor is fitted on ``fit_set`` only. The fitted pipeline is
    frozen (``FrozenEstimator``) so the calibrator never refits it.
    """
    _require_columns(fit_set, numeric_features, "fit set")
    _require_columns(calibration_set, numeric_features, "calibration set")
    y_fit = fit_set["sla_breach"].astype(int)
    y_cal = calibration_set["sla_breach"].astype(int)
    if y_cal.nunique() < 2:
        raise ValueError("the recent calibration window must contain both classes")
    preprocessor = make_preprocessor(numeric_features=numeric_features)
    preprocessor.set_output(transform="pandas")
    matrix = preprocessor.fit_transform(fit_set)
    booster = _catboost(y_fit, catboost_params, random_state)
    booster.fit(matrix, y_fit)
    fitted = Pipeline([("preprocessor", preprocessor), ("catboost", booster)])
    calibrator = CalibratedClassifierCV(FrozenEstimator(fitted), method="isotonic")
    calibrator.fit(calibration_set, y_cal)
    return calibrator


def _positive_probability(model: object, frame: pd.DataFrame) -> np.ndarray:
    """Return ``P(sla_breach = 1)`` from a fitted model."""
    return np.asarray(model.predict_proba(frame)[:, 1], dtype=float)


# --------------------------------------------------------------------------
# Development gate
# --------------------------------------------------------------------------
def _fold_update(fold_name: str, embargo_days: int) -> pd.Timestamp:
    """Return the fold's training end: validation start minus the embargo."""
    return pd.Period(fold_name, freq="M").start_time - pd.Timedelta(days=embargo_days)


def _gate_fold_scores(
    fold: Mapping[str, pd.DataFrame | str],
    update: pd.Timestamp,
    catboost_params: dict | None,
    random_state: int,
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """Score one fold with the reference and the three mechanisms."""
    train, validation = fold["train"], fold["validation"]
    base_features = list(NUMERIC_FEATURES)
    options = {"catboost_params": catboost_params, "random_state": random_state}
    scores: dict[str, np.ndarray] = {}
    sizes: dict[str, dict] = {}

    reference = _fit_calibrated_model(train, numeric_features=base_features, **options)
    scores[REFERENCE_NAME] = _positive_probability(reference, validation)
    sizes[REFERENCE_NAME] = {"n_fit": len(train), "n_calibration": 0}

    weights = recency_weights(train[TIMESTAMP_COLUMN], update)
    model_c = _fit_calibrated_model(train, numeric_features=base_features, sample_weight=weights, **options)
    scores[VARIANT_NAMES["C"]] = _positive_probability(model_c, validation)
    sizes[VARIANT_NAMES["C"]] = {"n_fit": len(train), "n_calibration": 0}

    # Folds have no delivery-date criterion: the recent window is the last
    # RECALIBRATION_WINDOW_DAYS days of training purchases.
    fit_mask, cal_mask = recalibration_split(train, update, time_col=TIMESTAMP_COLUMN)
    model_d = _fit_recalibrated_model(
        train.loc[fit_mask], train.loc[cal_mask], numeric_features=base_features, **options
    )
    scores[VARIANT_NAMES["D"]] = _positive_probability(model_d, validation)
    sizes[VARIANT_NAMES["D"]] = {"n_fit": int(fit_mask.sum()), "n_calibration": int(cal_mask.sum())}

    model_e = _fit_calibrated_model(train, numeric_features=[*base_features, PROMISE_FEATURE], **options)
    scores[VARIANT_NAMES["E"]] = _positive_probability(model_e, validation)
    sizes[VARIANT_NAMES["E"]] = {"n_fit": len(train), "n_calibration": 0}
    return scores, sizes


def gate_summary_from_folds(
    gate_folds: pd.DataFrame,
    ap_ratio: float | None = None,
    ece_margin: float | None = None,
) -> pd.DataFrame:
    """Apply the non-inferiority gate to per-fold metrics.

    A variant passes when its mean AP is at least ``ap_ratio`` times the mean AP
    of the reference **and** its mean ECE is at most the reference mean ECE plus
    ``ece_margin``.

    Parameters
    ----------
    gate_folds:
        Table with ``fold``, ``variant``, ``average_precision`` and ``ece_10bins``,
        containing the reference rows (``variant == "F1_config"``).
    ap_ratio, ece_margin:
        Thresholds. ``None`` uses :data:`GATE_AP_RATIO` and :data:`GATE_ECE_MARGIN`.

    Returns
    -------
    pandas.DataFrame
        One row per non-reference variant with the means, the AP ratio, the ECE
        difference, both checks, ``passed`` and ``status``.
    """
    ap_ratio = GATE_AP_RATIO if ap_ratio is None else ap_ratio
    ece_margin = GATE_ECE_MARGIN if ece_margin is None else ece_margin
    means = gate_folds.groupby("variant", sort=False)[["average_precision", "ece_10bins"]].mean()
    counts = gate_folds.groupby("variant", sort=False)["fold"].nunique()
    if REFERENCE_NAME not in means.index:
        raise KeyError("the gate table has no reference rows")
    ref_ap = float(means.loc[REFERENCE_NAME, "average_precision"])
    ref_ece = float(means.loc[REFERENCE_NAME, "ece_10bins"])
    rows = []
    for variant in means.index:
        if variant == REFERENCE_NAME:
            continue
        mean_ap = float(means.loc[variant, "average_precision"])
        mean_ece = float(means.loc[variant, "ece_10bins"])
        passes_ap = bool(mean_ap >= ap_ratio * ref_ap - 1e-12)
        passes_ece = bool(mean_ece <= ref_ece + ece_margin + 1e-12)
        passed = passes_ap and passes_ece
        rows.append(
            {
                "variant": variant,
                "n_folds": int(counts.loc[variant]),
                "mean_ap": mean_ap,
                "reference_mean_ap": ref_ap,
                "ap_ratio": mean_ap / ref_ap if ref_ap else np.nan,
                "ap_ratio_threshold": ap_ratio,
                "passes_ap": passes_ap,
                "mean_ece": mean_ece,
                "reference_mean_ece": ref_ece,
                "ece_difference": mean_ece - ref_ece,
                "ece_margin": ece_margin,
                "passes_ece": passes_ece,
                "passed": passed,
                "status": STATUS_EVALUATED if passed else STATUS_FAILED,
            }
        )
    return pd.DataFrame(rows)


def run_development_gate(
    modeling_frame: pd.DataFrame,
    *,
    catboost_params: dict | None = None,
    random_state: int = FINAL_RANDOM_STATE,
    first_val_month: str = GATE_FIRST_MONTH,
    last_val_month: str = GATE_LAST_MONTH,
    embargo_days: int = GATE_EMBARGO_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the non-inferiority gate of variants C, D and E on the development folds.

    Each mechanism is applied in the expanding-window folds (default 2017-10 to
    2018-06, 30-day embargo) and compared with the F1 configuration (identical to
    variant B) in the same fold. In a fold the "update date" is the end of the
    training window (validation start minus the embargo): the recency age of C is
    relative to it and the recalibration window of D is the last 30 days of
    training **purchases**, because folds have no delivery-date criterion (an
    adaptation of the real update, which uses delivery dates). E computes the
    promise feature as-of. Only purchases before ``2018-07-01`` are used; the
    August evaluation window is never touched.

    Parameters
    ----------
    modeling_frame:
        Supervised order-level frame.
    catboost_params:
        Optional CatBoost overrides (tests only).
    random_state:
        CatBoost seed.
    first_val_month, last_val_month, embargo_days:
        Fold definition, as in the tournament.

    Returns
    -------
    tuple
        ``(gate_folds, gate_summary)``; see :func:`gate_summary_from_folds`.
    """
    stamps = pd.to_datetime(modeling_frame[TIMESTAMP_COLUMN])
    development = modeling_frame.loc[stamps < pd.Timestamp(DEV_END)].copy()
    development[PROMISE_FEATURE] = promise_vs_recent_median(development)
    folds = expanding_window_folds(
        development,
        first_val_month=first_val_month,
        last_val_month=last_val_month,
        embargo_days=embargo_days,
        test_start=DEV_END,
    )
    rows = []
    for fold in folds:
        update = _fold_update(fold["fold"], embargo_days)
        scores, sizes = _gate_fold_scores(fold, update, catboost_params, random_state)
        y_validation = fold["validation"]["sla_breach"].astype(int).to_numpy()
        for name, probabilities in scores.items():
            metrics = classification_metrics(y_validation, probabilities)
            rows.append(
                {
                    "fold": fold["fold"],
                    "variant": name,
                    "update_date": update,
                    "n_validation": len(y_validation),
                    **sizes[name],
                    "average_precision": metrics["average_precision"],
                    "ece_10bins": metrics["ece_10bins"],
                    "brier": metrics["brier"],
                }
            )
    gate_folds = pd.DataFrame(rows)
    return gate_folds, gate_summary_from_folds(gate_folds)


# --------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------
def daily_block_key(timestamps: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """Return ``YYYY-MM-DD`` labels of the purchase day, the F2 bootstrap block."""
    return pd.DatetimeIndex(pd.to_datetime(timestamps)).strftime("%Y-%m-%d").to_numpy()


def _population_row(variant: str, role: str, frame: pd.DataFrame) -> dict:
    """Describe one set of orders (size, prevalence, purchase and delivery span)."""
    breaches = int(frame["sla_breach"].astype(int).sum())
    purchase = pd.to_datetime(frame[TIMESTAMP_COLUMN])
    delivery = pd.to_datetime(frame[DELIVERY_COLUMN])
    return {
        "variant": variant,
        "role": role,
        "orders": len(frame),
        "breaches": breaches,
        "prevalence": breaches / len(frame) if len(frame) else np.nan,
        "first_purchase": purchase.min(),
        "last_purchase": purchase.max(),
        "first_delivery": delivery.min(),
        "last_delivery": delivery.max(),
    }


def label_availability_table(
    frame: pd.DataFrame,
    update_date: str | pd.Timestamp = F2_UPDATE_DATE,
    months: tuple[str, ...] = ("2018-06", "2018-07"),
) -> pd.DataFrame:
    """Show how many orders of each purchase month had a known label at the update date.

    Slow deliveries are, by construction, the ones whose label is still missing,
    so the known labels of the most recent months are biased towards fast (and
    therefore non-breaching) orders. The table exposes that bias.

    Parameters
    ----------
    frame:
        Frame with delivery dates (see :func:`attach_delivery_dates`).
    update_date:
        Instant of the update.
    months:
        Purchase months (``YYYY-MM``) to describe.

    Returns
    -------
    pandas.DataFrame
        One row per month with totals, known labels, share known and the
        prevalence among known and not yet known orders.
    """
    known = known_label_mask(frame, update_date)
    month = pd.to_datetime(frame[TIMESTAMP_COLUMN]).dt.to_period("M").astype(str)
    breach = frame["sla_breach"].astype(int)
    rows = []
    for name in months:
        in_month = month == name
        total = int(in_month.sum())
        known_n = int((in_month & known).sum())
        unknown_n = total - known_n
        rows.append(
            {
                "purchase_month": name,
                "as_of": str(pd.Timestamp(update_date).date()),
                "orders": total,
                "known_labels": known_n,
                "share_known": known_n / total if total else np.nan,
                "prevalence_known": float(breach[in_month & known].mean()) if known_n else np.nan,
                "unknown_labels": unknown_n,
                "prevalence_unknown": float(breach[in_month & ~known].mean()) if unknown_n else np.nan,
            }
        )
    return pd.DataFrame(rows)


def promise_regime_table(
    modeling_frame: pd.DataFrame,
    timestamp_col: str = TIMESTAMP_COLUMN,
) -> pd.DataFrame:
    """Describe, by purchase month, whether the promise rule worked in development.

    The F2 hypotheses stem from the promise regime seen in the F1 test. This
    table checks that the regime is absent from the development period: the AUC
    of the rule "negative promised lead time" is computed with labels **only for
    purchases before** :data:`DEV_END` (2018-07-01). For later months only the
    order count and the median promise (a feature, no label) are reported; their
    labels are never read.

    Parameters
    ----------
    modeling_frame:
        Supervised order-level frame.
    timestamp_col:
        Purchase timestamp column.

    Returns
    -------
    pandas.DataFrame
        One row per purchase month with ``orders``, ``promise_median``,
        ``labels_used``, ``breach_prevalence`` and ``auc_negative_promise``
        (NaN where labels are not used or a month has a single class).
    """
    stamps = pd.to_datetime(modeling_frame[timestamp_col])
    months = stamps.dt.to_period("M")
    rows = []
    for month in sorted(months.unique()):
        in_month = months == month
        part = modeling_frame.loc[in_month]
        development = month.end_time < pd.Timestamp(DEV_END)
        row = {
            "purchase_month": str(month),
            "orders": len(part),
            "promise_median": float(part[PROMISE_COLUMN].median()),
            "labels_used": bool(development),
            "breach_prevalence": np.nan,
            "auc_negative_promise": np.nan,
        }
        if development:
            y = part["sla_breach"].astype(int)
            row["breach_prevalence"] = float(y.mean())
            valid = part[PROMISE_COLUMN].notna()
            if y[valid].nunique() == 2:
                row["auc_negative_promise"] = float(roc_auc_score(y[valid], -part.loc[valid, PROMISE_COLUMN]))
        rows.append(row)
    return pd.DataFrame(rows)


def _file_sha256(path: Path) -> str | None:
    """Return the SHA-256 of a file, or ``None`` if it does not exist."""
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paired_table(
    boot: object,
    evaluated: list[str],
    failed: list[str],
    references: tuple[str, ...],
) -> pd.DataFrame:
    """Build the exploratory paired AP differences (no confirmatory label)."""
    rows = []
    for model in evaluated:
        for reference in references:
            row = _paired_row(boot, model, reference, "average_precision")
            difference = row["difference"]
            sign = "zero" if difference == 0 else ("positive" if difference > 0 else "negative")
            low, high = row["ci_low"], row["ci_high"]
            excludes = bool(low > 0 or high < 0) if not (np.isnan(low) or np.isnan(high)) else False
            rows.append(
                {
                    **row,
                    "difference_sign": sign,
                    "ci_excludes_zero": excludes,
                    "reading": READING,
                    "status": STATUS_EVALUATED,
                }
            )
    for model in failed:
        for reference in references:
            rows.append(
                {
                    "model": model,
                    "reference": reference,
                    "metric": "average_precision",
                    **{
                        key: np.nan
                        for key in (
                            "estimate_model",
                            "estimate_reference",
                            "difference",
                            "ci_low",
                            "ci_high",
                            "n_boot",
                            "n_valid",
                            "n_discarded",
                        )
                    },
                    "difference_sign": "not evaluated",
                    "ci_excludes_zero": False,
                    "reading": READING,
                    "status": STATUS_FAILED,
                }
            )
    return pd.DataFrame(rows)


def _compute_posthoc_tables(
    frame: pd.DataFrame,
    *,
    version: str,
    fingerprint: str,
    f1_model_path: Path,
    f1_model_sha256: str | None,
    n_boot: int,
    seed: int,
    catboost_params: dict | None,
) -> dict[str, pd.DataFrame]:
    """Run the gate, then score August with the approved variants only."""
    update = pd.Timestamp(F2_UPDATE_DATE)
    # 1. Development gate: nothing from August is read before this finishes.
    gate_folds, gate_summary = run_development_gate(frame, catboost_params=catboost_params, random_state=FINAL_RANDOM_STATE)
    passed = {
        code: bool(gate_summary.loc[gate_summary["variant"] == VARIANT_NAMES[code], "passed"].any())
        for code in GATED_VARIANTS
    }

    # 2. Populations.
    frame = frame.copy()
    frame[PROMISE_FEATURE] = promise_vs_recent_median(frame)
    stamps = pd.to_datetime(frame[TIMESTAMP_COLUMN])
    known = frame.loc[known_label_mask(frame, update)].copy()
    evaluation = frame.loc[(stamps >= pd.Timestamp(F2_EVAL_START)) & (stamps < pd.Timestamp(F2_EVAL_END))].copy()
    if known.empty or evaluation.empty:
        raise ValueError("known labels and the evaluation window must be non-empty")
    if (pd.to_datetime(known[DELIVERY_COLUMN]) >= update).any():
        raise ValueError("a label delivered on or after the update date reached the training set")
    y_eval = evaluation["sla_breach"].astype(int).to_numpy()

    options = {"catboost_params": catboost_params, "random_state": FINAL_RANDOM_STATE}
    base_features = list(NUMERIC_FEATURES)
    scores: dict[str, np.ndarray] = {}
    populations = []

    # 3. Variants. A: frozen F1 model (loaded, never retrained).
    frozen = joblib.load(f1_model_path)
    scores[VARIANT_NAMES["A"]] = _positive_probability(frozen, evaluation)
    f1_train = frame.loc[stamps < pd.Timestamp(F1_TRAIN_END)]
    populations.append(_population_row(VARIANT_NAMES["A"], "train (F1, frozen)", f1_train))

    model_b = _fit_calibrated_model(known, numeric_features=base_features, **options)
    scores[VARIANT_NAMES["B"]] = _positive_probability(model_b, evaluation)
    populations.append(_population_row(VARIANT_NAMES["B"], "train (known labels)", known))

    if passed["C"]:
        weights = recency_weights(known[TIMESTAMP_COLUMN], update)
        model_c = _fit_calibrated_model(known, numeric_features=base_features, sample_weight=weights, **options)
        scores[VARIANT_NAMES["C"]] = _positive_probability(model_c, evaluation)
        populations.append(_population_row(VARIANT_NAMES["C"], "train (known labels)", known))
    if passed["D"]:
        fit_mask, cal_mask = recalibration_split(known, update)
        model_d = _fit_recalibrated_model(known.loc[fit_mask], known.loc[cal_mask], numeric_features=base_features, **options)
        scores[VARIANT_NAMES["D"]] = _positive_probability(model_d, evaluation)
        populations.append(_population_row(VARIANT_NAMES["D"], "fit (delivered before update - 30d)", known.loc[fit_mask]))
        populations.append(_population_row(VARIANT_NAMES["D"], "calibration (delivered in last 30d)", known.loc[cal_mask]))
    if passed["E"]:
        model_e = _fit_calibrated_model(known, numeric_features=[*base_features, PROMISE_FEATURE], **options)
        scores[VARIANT_NAMES["E"]] = _positive_probability(model_e, evaluation)
        populations.append(_population_row(VARIANT_NAMES["E"], "train (known labels)", known))

    # Baselines refitted on the same known labels as B.
    baselines = classification_baseline_scores(known, evaluation)
    populations.append(_population_row("baselines", "train (known labels)", known))
    populations.append(_population_row("all", "evaluation (August purchases)", evaluation))
    variant_models = list(scores)
    scores.update({f"{BASELINE_PREFIX}{name}": values for name, values in baselines.items()})

    # 4. Daily block bootstrap, every model on the same resamples.
    blocks = daily_block_key(evaluation[TIMESTAMP_COLUMN])
    boot = _run_bootstrap(y_eval, scores, blocks, n_boot, seed, None)
    metrics = _summarize_bootstrap(boot)
    status = {name: STATUS_EVALUATED for name in variant_models}
    status.update({name: STATUS_BASELINE for name in scores if name.startswith(BASELINE_PREFIX)})
    metrics["status"] = metrics["model"].map(status)

    failed = [VARIANT_NAMES[code] for code in GATED_VARIANTS if not passed[code]]
    if failed:
        placeholder = pd.DataFrame(
            [
                {"model": name, "metric": "average_precision", "status": STATUS_FAILED}
                for name in failed
            ]
        )
        metrics = pd.concat([metrics, placeholder], ignore_index=True)

    probability_models = [
        name for name in scores if name.removeprefix(BASELINE_PREFIX) not in NON_PROBABILITY_BASELINES
    ]
    reliability = pd.concat(
        [reliability_table(y_eval, scores[name]).assign(model=name) for name in probability_models],
        ignore_index=True,
    )[["model", "bin", "bin_lower", "bin_upper", "n", "mean_predicted", "observed_rate"]]

    paired = _paired_table(boot, variant_models, failed, (REFERENCE_BASELINE, BASELINE_RULE))

    availability = label_availability_table(frame, update)
    created_at = datetime.now().isoformat(timespec="seconds")
    provenance = pd.DataFrame(
        [
            {
                "version": version,
                "post_hoc": True,
                "reading": READING,
                "production_eligible": False,
                "created_at": created_at,
                "fingerprint": fingerprint,
                "update_date": F2_UPDATE_DATE,
                "eval_start": F2_EVAL_START,
                "eval_end_exclusive": F2_EVAL_END,
                "known_label_orders": len(known),
                "eval_orders": len(evaluation),
                "eval_blocks": int(len(np.unique(blocks))),
                "gate_passed": ",".join(code for code in GATED_VARIANTS if passed[code]),
                "gate_failed": ",".join(code for code in GATED_VARIANTS if not passed[code]),
                "n_boot": n_boot,
                "seed": seed,
                "f1_model_path": str(f1_model_path),
                "f1_model_sha256": f1_model_sha256,
            }
        ]
    )
    variants = pd.DataFrame(
        [
            {
                "code": code,
                "variant": name,
                "gated": code in GATED_VARIANTS,
                "scored": name in scores,
                "status": status.get(name, STATUS_FAILED),
            }
            for code, name in VARIANT_NAMES.items()
        ]
    )
    return {
        "provenance": provenance,
        "variants": variants,
        "populations": pd.DataFrame(populations),
        "gate_folds": gate_folds,
        "gate_summary": gate_summary,
        "classification_metrics": metrics,
        "paired_difference": paired,
        "reliability": reliability,
        "top_k": _top_k_table(metrics.dropna(subset=["estimate"])),
        "label_availability": availability,
    }


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def run_posthoc_evaluation(
    modeling_frame: pd.DataFrame,
    *,
    supervised_orders: pd.DataFrame | None = None,
    allow_test: bool = False,
    version: str = "F2",
    cache_root: Path | None = None,
    f1_model_path: Path,
    n_boot: int = 1_000,
    seed: int = 42,
    catboost_params: dict | None = None,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Run the post-hoc F2 evaluation, or load its frozen result.

    The order inside one execution is fixed: first the development gate of
    variants C, D and E (no August data is read), then the August evaluation of
    the approved variants only. The result of a version is computed once, with
    the same frozen semantics, run name and ledger as F1
    (``<cache_root>/final_evaluation/<version>/`` and ``LEDGER.jsonl``): if the
    version directory exists it is loaded without recomputing and a warning
    reports a fingerprint divergence. Results are always marked ``post_hoc`` and
    read as exploratory; ``production_eligible`` is always ``False``. No model is
    saved to disk.

    Parameters
    ----------
    modeling_frame:
        Supervised order-level frame with features, ``sla_breach`` and the
        purchase timestamp.
    supervised_orders:
        Supervised orders providing ``order_delivered_customer_date`` by
        ``order_id`` (used only to decide which labels are known). Optional if
        ``modeling_frame`` already has the column.
    allow_test:
        Must be ``True``; opens the August 2018 evaluation window.
    version:
        Run version; must differ from ``"F1"``.
    cache_root:
        Base directory of the run cache. Defaults to ``<project>/outputs/runs``.
    f1_model_path:
        Path of the frozen F1 model (``model.joblib``) used by variant A.
    n_boot:
        Bootstrap resamples (daily blocks).
    seed:
        Bootstrap seed.
    catboost_params:
        Optional CatBoost overrides (tests only).

    Returns
    -------
    tuple
        ``(tables, metadata)`` with tables ``provenance``, ``variants``,
        ``populations``, ``gate_folds``, ``gate_summary``,
        ``classification_metrics``, ``paired_difference``, ``reliability``,
        ``top_k`` and ``label_availability``. ``metadata["source"]`` is
        ``"loaded"`` or ``"computed"`` and ``metadata["post_hoc"]`` is ``True``.

    Raises
    ------
    ValueError
        If ``allow_test`` is not ``True``, ``version`` is ``"F1"`` or invalid.
    FileNotFoundError
        If the F1 model is missing when the evaluation must be computed.
    """
    if allow_test is not True:
        raise ValueError(
            "the post-hoc evaluation opens the August 2018 window; pass allow_test=True "
            "only in the F2 post-hoc evaluation"
        )
    _validate_version(version)
    if version == PRIMARY_VERSION:
        raise ValueError("the post-hoc evaluation cannot use the F1 version; use a new version such as 'F2'")
    root = Path(cache_root) if cache_root is not None else _default_cache_root()
    model_path = Path(f1_model_path)
    model_sha = _file_sha256(model_path)
    frame = attach_delivery_dates(modeling_frame, supervised_orders)

    config = {
        "run_version": version,
        "post_hoc": True,
        "kind": "posthoc_f2",
        "update_date": F2_UPDATE_DATE,
        "eval_start": F2_EVAL_START,
        "eval_end_exclusive": F2_EVAL_END,
        "f1_train_end_exclusive": F1_TRAIN_END,
        "variants": VARIANT_NAMES,
        "recency_half_life_days": RECENCY_HALF_LIFE_DAYS,
        "recalibration_window_days": RECALIBRATION_WINDOW_DAYS,
        "promise_lookback_days": PROMISE_LOOKBACK_DAYS,
        "gate_ap_ratio": GATE_AP_RATIO,
        "gate_ece_margin": GATE_ECE_MARGIN,
        "gate_folds": [GATE_FIRST_MONTH, GATE_LAST_MONTH, GATE_EMBARGO_DAYS],
        "bootstrap_blocks": "daily",
        "n_boot": n_boot,
        "seed": seed,
        "random_state": FINAL_RANDOM_STATE,
        "calibration_cv": CALIBRATION_CV,
        "catboost_params": catboost_params or {},
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "f1_model_sha256": model_sha,
    }
    frames = {"modeling_frame": modeling_frame}
    if supervised_orders is not None and DELIVERY_COLUMN in supervised_orders.columns:
        frames["delivery_dates"] = supervised_orders[["order_id", DELIVERY_COLUMN]].reset_index(drop=True)

    def guard() -> None:
        if model_sha is None:
            raise FileNotFoundError(f"frozen F1 model not found: {model_path}")

    def compute(fingerprint: str) -> dict[str, pd.DataFrame]:
        return _compute_posthoc_tables(
            frame,
            version=version,
            fingerprint=fingerprint,
            f1_model_path=model_path,
            f1_model_sha256=model_sha,
            n_boot=n_boot,
            seed=seed,
            catboost_params=catboost_params,
        )

    def ledger(tables: dict[str, pd.DataFrame], meta: dict) -> dict:
        provenance = tables["provenance"].iloc[0]
        return {
            "version": version,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "fingerprint": meta["fingerprint"],
            "train_orders": int(provenance["known_label_orders"]),
            "test_orders": int(provenance["eval_orders"]),
            "post_hoc": True,
            "n_boot": n_boot,
            "seed": seed,
            "gate_passed": str(provenance["gate_passed"]),
        }

    return _run_frozen(
        version=version,
        post_hoc=True,
        root=root,
        config=config,
        frames=frames,
        compute=compute,
        ledger_record=ledger,
        before_compute=guard,
        stacklevel=3,
    )

