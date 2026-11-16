"""Single pre-registered final evaluation (notebook 09).

The period from 2018-07-01 to 2018-08-31 is meant to be opened once, with the
contract approved as P-05 in ``docs/REGISTRO-DE-DECISOES.md``. This module holds
the whole logic: the fixed windows, the model fitted exactly as in the
tournament, the weekly block bootstrap, the pre-registered reading rule and the
single-use cache with an append-only ledger.

The functions never select models or tune anything; they measure and document.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
import warnings
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .drift import feature_psi_table
from .modeling import (
    BASELINE_PREFIX,
    CATEGORICAL_FEATURES,
    NON_PROBABILITY_BASELINES,
    NUMERIC_FEATURES,
    build_classification_tournament,
    build_regression_tournament,
    calibrated_classifier,
    classification_baseline_scores,
    classification_metrics,
    make_preprocessor,
    regression_baseline_predictions,
    regression_metrics,
)
from .network import NETWORK_FEATURE_COLUMNS
from .run_cache import _default_cache_root, cached_run, load_run_unchecked, run_fingerprint
from .temporal import right_censoring_sensitivity

FINAL_TRAIN_END = "2018-06-01"
"""Exclusive end of the training window: purchases strictly before this date."""

FINAL_TEST_START = "2018-07-01"
"""Inclusive start of the test window."""

FINAL_TEST_END = "2018-09-01"
"""Exclusive end of the test window: purchases strictly before this date."""

FINAL_RUN_NAME = "final_evaluation"
PRIMARY_VERSION = "F1"
LEDGER_NAME = "LEDGER.jsonl"
CANDIDATE_MODEL = "CatBoost"
REFERENCE_BASELINE = f"{BASELINE_PREFIX}logistic_promise_plus_state"
FINAL_RANDOM_STATE = 42
CALIBRATION_CV = 3
RELIABILITY_BINS = 10
TIMESTAMP_COLUMN = "order_purchase_timestamp"
BOOTSTRAP_METRICS = (
    "average_precision",
    "roc_auc",
    "brier",
    "ece_10bins",
    "precision_at_5pct",
    "recall_at_5pct",
    "precision_at_10pct",
    "recall_at_10pct",
)
OUTCOME_LABELS = {
    "confirmado": "confirmed",
    "não confirmado": "not confirmed",
    "inconclusivo": "inconclusive",
}
"""Pre-registered outcome labels (Portuguese key, English translation)."""

_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


# --------------------------------------------------------------------------
# Windows
# --------------------------------------------------------------------------
def split_final_windows(
    modeling_frame: pd.DataFrame,
    timestamp_col: str = TIMESTAMP_COLUMN,
) -> dict[str, pd.DataFrame]:
    """Split the modeling frame into the pre-registered train and test windows.

    Parameters
    ----------
    modeling_frame:
        Supervised order-level frame (one row per order).
    timestamp_col:
        Purchase timestamp column.

    Returns
    -------
    dict
        ``{"train": ..., "test": ...}``. Train holds purchases strictly before
        :data:`FINAL_TRAIN_END`; test holds purchases in
        ``[FINAL_TEST_START, FINAL_TEST_END)``. Purchases in between (June
        2018, the embargo) belong to neither partition.

    Raises
    ------
    KeyError
        If the timestamp column is missing.
    ValueError
        If a timestamp is missing or invalid, or if a window is empty.
    """
    if timestamp_col not in modeling_frame.columns:
        raise KeyError(f"timestamp column not found: {timestamp_col}")
    timestamp = pd.to_datetime(modeling_frame[timestamp_col], errors="coerce")
    if timestamp.isna().any():
        raise ValueError(f"timestamp column contains invalid or missing values: {timestamp_col}")
    train = modeling_frame.loc[timestamp < pd.Timestamp(FINAL_TRAIN_END)].copy()
    test = modeling_frame.loc[
        (timestamp >= pd.Timestamp(FINAL_TEST_START)) & (timestamp < pd.Timestamp(FINAL_TEST_END))
    ].copy()
    if train.empty or test.empty:
        raise ValueError("final evaluation windows must be non-empty")
    train_time = pd.to_datetime(train[timestamp_col])
    test_time = pd.to_datetime(test[timestamp_col])
    if not (train_time < pd.Timestamp(FINAL_TRAIN_END)).all():
        raise ValueError("training window contains purchases on or after the train end")
    if not (
        (test_time >= pd.Timestamp(FINAL_TEST_START)) & (test_time < pd.Timestamp(FINAL_TEST_END))
    ).all():
        raise ValueError("test window contains purchases outside the pre-registered interval")
    return {"train": train, "test": test}


def iso_week_key(timestamps: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """Return ``YYYY-Www`` ISO-week labels for purchase timestamps."""
    iso = pd.DatetimeIndex(pd.to_datetime(timestamps)).isocalendar()
    return (iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)).to_numpy()


# --------------------------------------------------------------------------
# Reading rule and reliability
# --------------------------------------------------------------------------
def classify_preregistered_outcome(diff: float, ci_low: float, ci_high: float) -> str:
    """Apply the pre-registered reading rule to the paired AP difference.

    Parameters
    ----------
    diff:
        Point difference AP(CatBoost) minus AP(reference baseline).
    ci_low, ci_high:
        Bounds of the 95% interval of the difference.

    Returns
    -------
    str
        ``"confirmado"`` if ``diff > 0`` and ``ci_low > 0``;
        ``"não confirmado"`` if ``diff < 0`` and ``ci_high < 0``;
        ``"inconclusivo"`` otherwise (including undefined values).
    """
    if any(pd.isna(value) for value in (diff, ci_low, ci_high)):
        return "inconclusivo"
    if diff > 0 and ci_low > 0:
        return "confirmado"
    if diff < 0 and ci_high < 0:
        return "não confirmado"
    return "inconclusivo"


def reliability_table(
    y_true: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
    bins: int = RELIABILITY_BINS,
) -> pd.DataFrame:
    """Return the reliability (calibration) curve in equal-width bins.

    Parameters
    ----------
    y_true:
        Binary outcomes.
    probabilities:
        Predicted probabilities in ``[0, 1]``.
    bins:
        Number of equal-width bins. The last bin includes 1.0, as in the ECE.

    Returns
    -------
    pandas.DataFrame
        One row per bin with ``n``, ``mean_predicted`` and ``observed_rate``
        (NaN in empty bins).
    """
    if bins < 1:
        raise ValueError("bins must be at least 1")
    y = np.asarray(y_true, dtype=int)
    prob = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for number, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True), start=1):
        mask = (prob >= lower) & (prob <= upper if number == bins else prob < upper)
        count = int(mask.sum())
        rows.append(
            {
                "bin": number,
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "n": count,
                "mean_predicted": float(prob[mask].mean()) if count else np.nan,
                "observed_rate": float(y[mask].mean()) if count else np.nan,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Weekly block bootstrap
# --------------------------------------------------------------------------
class _Bootstrap(NamedTuple):
    """Raw output of the weekly block bootstrap."""

    point: dict[str, dict[str, float]]
    replicates: dict[tuple[str, str], np.ndarray]
    n_boot: int
    n_discarded: int


def _week_blocks(week_key: pd.Series | np.ndarray) -> list[np.ndarray]:
    """Return the row positions of each week (one block per distinct key)."""
    codes, _ = pd.factorize(np.asarray(week_key), sort=True)
    if (codes < 0).any():
        raise ValueError("week_key must not contain missing values")
    order = np.argsort(codes, kind="stable")
    boundaries = np.cumsum(np.bincount(codes))[:-1]
    return np.split(order, boundaries)


def _block_resamples(blocks: list[np.ndarray], n_boot: int, seed: int):
    """Yield ``(picked_weeks, row_positions)`` for each bootstrap resample."""
    rng = np.random.default_rng(seed)
    n_weeks = len(blocks)
    for _ in range(n_boot):
        picks = rng.integers(0, n_weeks, size=n_weeks)
        yield picks, np.concatenate([blocks[i] for i in picks])


def _make_evaluator(
    metric_fns: Mapping[str, Callable[[np.ndarray, np.ndarray], float]] | None,
) -> Callable[[str, np.ndarray, np.ndarray], dict[str, float]]:
    """Build ``evaluate(model, y, scores) -> {metric: value}``."""
    if metric_fns is not None:
        functions = dict(metric_fns)

        def custom(model: str, y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
            del model
            return {name: float(fn(y, scores)) for name, fn in functions.items()}

        return custom

    def default(model: str, y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
        probabilistic = model.removeprefix(BASELINE_PREFIX) not in NON_PROBABILITY_BASELINES
        values = classification_metrics(y, scores, probabilistic=probabilistic)
        return {name: float(values[name]) for name in BOOTSTRAP_METRICS}

    return default


def _run_bootstrap(
    y_true: pd.Series | np.ndarray,
    scores_by_model: Mapping[str, pd.Series | np.ndarray],
    week_key: pd.Series | np.ndarray,
    n_boot: int,
    seed: int,
    metric_fns: Mapping[str, Callable[[np.ndarray, np.ndarray], float]] | None,
) -> _Bootstrap:
    """Resample whole blocks and evaluate every model on the same resample.

    ``week_key`` is any block label (ISO week in F1, purchase day in F2).
    """
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    if not scores_by_model:
        raise ValueError("scores_by_model must not be empty")
    y = np.asarray(y_true, dtype=int)
    scores = {name: np.asarray(values, dtype=float) for name, values in scores_by_model.items()}
    weeks = np.asarray(week_key)
    if any(len(v) != len(y) for v in scores.values()) or len(weeks) != len(y):
        raise ValueError("y_true, scores and week_key must have the same length")
    evaluate = _make_evaluator(metric_fns)
    point = {name: evaluate(name, y, values) for name, values in scores.items()}
    collected: dict[tuple[str, str], list[float]] = {
        (name, metric): [] for name, metrics in point.items() for metric in metrics
    }
    discarded = 0
    for _, rows in _block_resamples(_week_blocks(weeks), n_boot, seed):
        y_boot = y[rows]
        if y_boot.min() == y_boot.max():
            discarded += 1
            continue
        for name, values in scores.items():
            for metric, value in evaluate(name, y_boot, values[rows]).items():
                collected[(name, metric)].append(value)
    replicates = {key: np.asarray(values, dtype=float) for key, values in collected.items()}
    return _Bootstrap(point, replicates, n_boot, discarded)


def _interval(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Percentile interval, NaN when there is no finite replicate."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    tail = (1.0 - level) / 2.0 * 100.0
    low, high = np.percentile(finite, [tail, 100.0 - tail])
    return float(low), float(high)


def _summarize_bootstrap(boot: _Bootstrap) -> pd.DataFrame:
    """Build the point estimate and percentile interval per (model, metric)."""
    rows = []
    for (model, metric), values in boot.replicates.items():
        low, high = _interval(values)
        rows.append(
            {
                "model": model,
                "metric": metric,
                "estimate": boot.point[model][metric],
                "ci_low": low,
                "ci_high": high,
                "n_boot": boot.n_boot,
                "n_valid": boot.n_boot - boot.n_discarded,
                "n_discarded": boot.n_discarded,
            }
        )
    return pd.DataFrame(rows)


def _paired_row(
    boot: _Bootstrap,
    model: str,
    reference: str,
    metric: str,
) -> dict:
    """Return the paired difference ``metric(model) - metric(reference)`` as a dict.

    The row has no reading label, so that the pre-registered rule (F1) and the
    purely exploratory description (F2) can share the same computation.
    """
    for name in (model, reference):
        if (name, metric) not in boot.replicates:
            raise KeyError(f"unknown model/metric: {name}/{metric}")
    difference = boot.point[model][metric] - boot.point[reference][metric]
    replicates = boot.replicates[(model, metric)] - boot.replicates[(reference, metric)]
    low, high = _interval(replicates)
    return {
        "model": model,
        "reference": reference,
        "metric": metric,
        "estimate_model": boot.point[model][metric],
        "estimate_reference": boot.point[reference][metric],
        "difference": difference,
        "ci_low": low,
        "ci_high": high,
        "n_boot": boot.n_boot,
        "n_valid": boot.n_boot - boot.n_discarded,
        "n_discarded": boot.n_discarded,
    }


def _paired_difference(
    boot: _Bootstrap,
    model: str,
    reference: str,
    metric: str,
) -> pd.DataFrame:
    """Build the one-row paired difference ``metric(model) - metric(reference)``."""
    row = _paired_row(boot, model, reference, metric)
    outcome = classify_preregistered_outcome(row["difference"], row["ci_low"], row["ci_high"])
    return pd.DataFrame([{**row, "outcome": outcome, "outcome_en": OUTCOME_LABELS[outcome]}])


def weekly_block_bootstrap(
    y_true: pd.Series | np.ndarray,
    scores_by_model: Mapping[str, pd.Series | np.ndarray],
    week_key: pd.Series | np.ndarray,
    n_boot: int = 1_000,
    seed: int = 42,
    metric_fns: Mapping[str, Callable[[np.ndarray, np.ndarray], float]] | None = None,
) -> pd.DataFrame:
    """Estimate metrics with a 95% percentile interval from a weekly block bootstrap.

    Whole weeks are resampled with replacement, the sample is rebuilt and every
    model is scored on the same resample. Resamples with a single class are
    discarded and counted (not redrawn).

    Parameters
    ----------
    y_true:
        Binary outcomes.
    scores_by_model:
        Mapping of model name to scores aligned with ``y_true``. Names starting
        with ``baseline_`` whose remainder is in ``NON_PROBABILITY_BASELINES``
        get NaN Brier and ECE, as in the tournament.
    week_key:
        Block label of each row (for example the ISO week of the purchase).
    n_boot:
        Number of resamples requested.
    seed:
        Seed of the random generator.
    metric_fns:
        Optional mapping ``name -> fn(y, scores)``. ``None`` uses
        :data:`BOOTSTRAP_METRICS` from ``classification_metrics``.

    Returns
    -------
    pandas.DataFrame
        One row per (model, metric) with ``estimate`` (full sample),
        ``ci_low``, ``ci_high``, ``n_boot``, ``n_valid`` and ``n_discarded``.
    """
    return _summarize_bootstrap(_run_bootstrap(y_true, scores_by_model, week_key, n_boot, seed, metric_fns))


def paired_metric_difference(
    y_true: pd.Series | np.ndarray,
    scores_by_model: Mapping[str, pd.Series | np.ndarray],
    week_key: pd.Series | np.ndarray,
    model: str = CANDIDATE_MODEL,
    reference: str = REFERENCE_BASELINE,
    metric: str = "average_precision",
    n_boot: int = 1_000,
    seed: int = 42,
    metric_fns: Mapping[str, Callable[[np.ndarray, np.ndarray], float]] | None = None,
) -> pd.DataFrame:
    """Return the paired bootstrap difference ``metric(model) - metric(reference)``.

    The resamples are the same as in :func:`weekly_block_bootstrap` for equal
    ``seed`` and ``week_key``. The row carries the pre-registered reading
    (``outcome``) from :func:`classify_preregistered_outcome`.
    """
    boot = _run_bootstrap(y_true, scores_by_model, week_key, n_boot, seed, metric_fns)
    return _paired_difference(boot, model, reference, metric)


# --------------------------------------------------------------------------
# Final evaluation
# --------------------------------------------------------------------------
def _top_k_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Reshape the top 5% and 10% precision/recall rows of the bootstrap summary."""
    rows = []
    indexed = summary.set_index(["model", "metric"])
    for model in summary["model"].unique():
        for percent in (5, 10):
            precision = indexed.loc[(model, f"precision_at_{percent}pct")]
            recall = indexed.loc[(model, f"recall_at_{percent}pct")]
            rows.append(
                {
                    "model": model,
                    "top_fraction": percent / 100.0,
                    "precision": precision["estimate"],
                    "precision_ci_low": precision["ci_low"],
                    "precision_ci_high": precision["ci_high"],
                    "recall": recall["estimate"],
                    "recall_ci_low": recall["ci_low"],
                    "recall_ci_high": recall["ci_high"],
                }
            )
    return pd.DataFrame(rows)


def _psi_highlight(feature: str) -> str:
    """Label the features the pre-registration asks to highlight."""
    if feature == "promised_lead_days":
        return "promise"
    if feature in NETWORK_FEATURE_COLUMNS and "rate" in feature:
        return "network_rate"
    return "other"


def _monthly_prevalence(windows: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Orders, breaches and prevalence by purchase month in train and test."""
    parts = []
    for split, frame in windows.items():
        month = pd.to_datetime(frame[TIMESTAMP_COLUMN]).dt.to_period("M").astype(str)
        grouped = frame["sla_breach"].astype(int).groupby(month.to_numpy())
        table = pd.DataFrame({"orders": grouped.size(), "breaches": grouped.sum()})
        table["prevalence"] = table["breaches"] / table["orders"]
        table.insert(0, "split", split)
        parts.append(table.rename_axis("purchase_month").reset_index())
    return pd.concat(parts, ignore_index=True)


def _populations(windows: dict[str, pd.DataFrame], test_weeks: np.ndarray) -> pd.DataFrame:
    """Sizes, prevalence and dates of the two windows."""
    bounds = {
        "train": (None, FINAL_TRAIN_END),
        "test": (FINAL_TEST_START, FINAL_TEST_END),
    }
    rows = []
    for split, frame in windows.items():
        timestamp = pd.to_datetime(frame[TIMESTAMP_COLUMN])
        breaches = int(frame["sla_breach"].astype(int).sum())
        rows.append(
            {
                "split": split,
                "window_start": bounds[split][0],
                "window_end_exclusive": bounds[split][1],
                "orders": len(frame),
                "breaches": breaches,
                "prevalence": breaches / len(frame),
                "first_purchase": timestamp.min(),
                "last_purchase": timestamp.max(),
                "weeks": int(len(np.unique(test_weeks))) if split == "test" else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _save_final_model(
    pipeline: Pipeline,
    target_dir: Path,
    metadata: dict,
) -> None:
    """Persist the fitted pipeline and its metadata atomically; never overwrite."""
    if target_dir.exists():
        raise FileExistsError(f"final model directory already exists: {target_dir}")
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = target_dir.parent / f".{target_dir.name}.tmp-{uuid.uuid4().hex[:8]}"
    tmp_dir.mkdir()
    try:
        joblib.dump(pipeline, tmp_dir / "model.joblib")
        (tmp_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp_dir.rename(target_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def _compute_final_tables(
    modeling_frame: pd.DataFrame,
    orders: pd.DataFrame | None,
    *,
    version: str,
    post_hoc: bool,
    fingerprint: str,
    n_boot: int,
    seed: int,
    catboost_params: dict | None,
    model_target: Path,
) -> dict[str, pd.DataFrame]:
    """Fit on the train window, score the test window and build every table."""
    windows = split_final_windows(modeling_frame)
    train, test = windows["train"], windows["test"]
    y_train = train["sla_breach"].astype(int)
    y_test = test["sla_breach"].astype(int).to_numpy()

    required = set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES)
    for name, part in windows.items():
        missing = sorted(required - set(part.columns))
        if missing:
            raise KeyError(f"missing model features in {name}: {missing}")
    # Same shared preprocessing as the tournament (``prepare_model_matrices``),
    # with pandas output so the saved pipeline keeps feature names end to end.
    preprocessor = make_preprocessor()
    preprocessor.set_output(transform="pandas")
    matrices = {
        "train": preprocessor.fit_transform(train),
        "test": preprocessor.transform(test),
    }
    estimator = build_classification_tournament(y_train, random_state=FINAL_RANDOM_STATE)[CANDIDATE_MODEL]
    if catboost_params:
        estimator.set_params(**catboost_params)
    classifier = calibrated_classifier(estimator, cv=CALIBRATION_CV)
    classifier.fit(matrices["train"], y_train)
    probabilities = classifier.predict_proba(matrices["test"])[:, 1]

    baselines = classification_baseline_scores(train, test)
    scores = {CANDIDATE_MODEL: probabilities}
    scores.update({f"{BASELINE_PREFIX}{name}": values for name, values in baselines.items()})

    test_weeks = iso_week_key(test[TIMESTAMP_COLUMN])
    boot = _run_bootstrap(y_test, scores, test_weeks, n_boot, seed, None)
    metrics = _summarize_bootstrap(boot)

    probability_models = [name for name in scores if name.removeprefix(BASELINE_PREFIX) not in NON_PROBABILITY_BASELINES]
    reliability = pd.concat(
        [
            reliability_table(y_test, scores[name]).assign(model=name)
            for name in probability_models
        ],
        ignore_index=True,
    )[["model", "bin", "bin_lower", "bin_upper", "n", "mean_predicted", "observed_rate"]]

    delay_train = train["delivery_delay_days"].astype(float)
    regressor = build_regression_tournament(random_state=FINAL_RANDOM_STATE)[CANDIDATE_MODEL]
    if catboost_params:
        regressor.set_params(**{k: v for k, v in catboost_params.items() if k == "iterations"})
    regressor.fit(matrices["train"], delay_train)
    delay_predictions = {
        f"{BASELINE_PREFIX}{name}": values
        for name, values in regression_baseline_predictions(train, test).items()
    }
    delay_predictions[CANDIDATE_MODEL] = np.asarray(regressor.predict(matrices["test"]))
    delay_test = test["delivery_delay_days"].astype(float)
    regression = pd.DataFrame(
        [
            {
                "model": name,
                "mae": regression_metrics(delay_test, values)["mae"],
                "rmse": regression_metrics(delay_test, values)["rmse"],
                "n_test": len(test),
            }
            for name, values in delay_predictions.items()
        ]
    )

    psi = feature_psi_table(train, test, columns=list(NUMERIC_FEATURES))
    psi["highlight"] = psi["feature"].map(_psi_highlight)

    populations = _populations(windows, test_weeks)
    created_at = datetime.now().isoformat(timespec="seconds")
    provenance = pd.DataFrame(
        [
            {
                "version": version,
                "post_hoc": post_hoc,
                "created_at": created_at,
                "fingerprint": fingerprint,
                "train_orders": len(train),
                "test_orders": len(test),
                "train_end_exclusive": FINAL_TRAIN_END,
                "test_start": FINAL_TEST_START,
                "test_end_exclusive": FINAL_TEST_END,
                "n_boot": n_boot,
                "seed": seed,
                "model_path": str(model_target / "model.joblib"),
            }
        ]
    )
    tables = {
        "provenance": provenance,
        "populations": populations,
        "classification_metrics": metrics,
        "paired_difference": _paired_difference(boot, CANDIDATE_MODEL, REFERENCE_BASELINE, "average_precision"),
        "reliability": reliability,
        "top_k": _top_k_table(metrics),
        "regression": regression,
        "psi": psi,
        "monthly_prevalence": _monthly_prevalence(windows),
    }
    if orders is not None:
        tables["censoring"] = right_censoring_sensitivity(orders, FINAL_TEST_START, FINAL_TEST_END)

    pipeline = Pipeline([("preprocessor", preprocessor), ("calibrated_catboost", classifier)])
    _save_final_model(
        pipeline,
        model_target,
        {
            "version": version,
            "post_hoc": post_hoc,
            "created_at": created_at,
            "fingerprint": fingerprint,
            "model": CANDIDATE_MODEL,
            "random_state": FINAL_RANDOM_STATE,
            "calibration": f"isotonic, cv={CALIBRATION_CV}",
            "catboost_params": catboost_params or {},
            "numeric_features": list(NUMERIC_FEATURES),
            "categorical_features": list(CATEGORICAL_FEATURES),
            "train_orders": len(train),
            "test_orders": len(test),
            "train_end_exclusive": FINAL_TRAIN_END,
            "production_eligible": False,
        },
    )
    return tables


def _append_ledger(path: Path, record: dict) -> None:
    """Append one JSON line to the append-only ledger and flush it to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _validate_version(version: object) -> str:
    """Return ``version`` if it is a valid run name, else raise ``ValueError``."""
    if not isinstance(version, str) or not _VERSION_PATTERN.match(version):
        raise ValueError(f"invalid version: {version!r}")
    return version


def _run_frozen(
    *,
    version: str,
    post_hoc: bool,
    root: Path,
    config: dict,
    frames: dict[str, pd.DataFrame],
    compute: Callable[[str], dict[str, pd.DataFrame]],
    ledger_record: Callable[[dict[str, pd.DataFrame], dict], dict],
    before_compute: Callable[[], None] | None = None,
    stacklevel: int = 3,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Load a frozen ``final_evaluation/<version>`` run, or compute it once.

    Shared by the F1 evaluation and the F2 post-hoc evaluation (T-09 semantics).
    If the version directory exists it is always loaded, with a warning when the
    stored fingerprint differs from the current one; otherwise ``compute`` runs
    inside :func:`cached_run` and one line is appended to the ledger.

    Parameters
    ----------
    version:
        Run version (second directory level).
    post_hoc:
        Whether the version is post-hoc.
    root:
        Cache root.
    config:
        Configuration included in the fingerprint.
    frames:
        Input frames included in the fingerprint.
    compute:
        Callable receiving the current fingerprint and returning the tables.
    ledger_record:
        Callable ``(tables, metadata) -> dict`` building the ledger line.
    before_compute:
        Optional guard run only when the version is about to be computed.
    stacklevel:
        Stack level of the divergence warning, so it points to the caller's caller.

    Returns
    -------
    tuple
        ``(tables, metadata)``.
    """
    run_dir = root / FINAL_RUN_NAME / version
    fingerprint = run_fingerprint(config, frames)

    if run_dir.exists():
        tables, meta = load_run_unchecked(FINAL_RUN_NAME, version, root)
        diverged = meta.get("fingerprint") != fingerprint
        if diverged:
            warnings.warn(
                f"Final evaluation {version} already exists and is frozen: loading "
                f"{run_dir} (stored fingerprint {str(meta.get('fingerprint'))[:8]}) although "
                f"the current fingerprint is {fingerprint[:8]} (configuration, data or code "
                "changed). Nothing was recomputed; use a new version for a post-hoc run.",
                UserWarning,
                stacklevel=stacklevel,
            )
        stored_config = meta.get("config", {})
        return tables, {
            **meta,
            "post_hoc": bool(stored_config.get("post_hoc", post_hoc)),
            "current_fingerprint": fingerprint,
            "fingerprint_diverged": diverged,
        }

    if before_compute is not None:
        before_compute()

    tables, meta = cached_run(
        FINAL_RUN_NAME,
        version,
        config,
        lambda: compute(fingerprint),
        frames=frames,
        mode="auto",
        cache_root=root,
    )
    _append_ledger(root / FINAL_RUN_NAME / LEDGER_NAME, ledger_record(tables, meta))
    return tables, {**meta, "post_hoc": post_hoc, "current_fingerprint": fingerprint, "fingerprint_diverged": False}


def run_final_evaluation(
    modeling_frame: pd.DataFrame,
    *,
    allow_test: bool = False,
    version: str = PRIMARY_VERSION,
    cache_root: Path | None = None,
    model_dir: Path | None = None,
    n_boot: int = 1_000,
    seed: int = 42,
    orders: pd.DataFrame | None = None,
    catboost_params: dict | None = None,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Run the single pre-registered final evaluation, or load its frozen result.

    The result of a version is computed only once. When
    ``<cache_root>/final_evaluation/<version>/`` already exists, it is always
    loaded, even if the current fingerprint (configuration, data or package
    code) differs; a warning reports the divergence and nothing is recomputed.
    A new evaluation requires a new ``version``; versions other than ``"F1"``
    are marked ``post_hoc``. Every computation appends one line to the
    append-only ``<cache_root>/final_evaluation/LEDGER.jsonl``.

    Parameters
    ----------
    modeling_frame:
        Supervised order-level frame with features, ``sla_breach``,
        ``delivery_delay_days``, ``actual_lead_days`` and the purchase time.
    allow_test:
        Must be ``True``; this is the only call allowed to open the test period.
    version:
        Run version (``"F1"`` for the pre-registered evaluation).
    cache_root:
        Base directory of the run cache. Defaults to ``<project>/outputs/runs``.
    model_dir:
        Base directory of the saved models; the pipeline goes to
        ``model_dir/<version>/model.joblib`` with ``metadata.json``. Defaults to
        ``<cache_root>/../models/catboost_final``.
    n_boot:
        Bootstrap resamples (1,000 in the pre-registration).
    seed:
        Bootstrap seed (42 in the pre-registration).
    orders:
        Optional raw orders table for the descriptive right-censoring table.
    catboost_params:
        Optional CatBoost overrides (for tests only; the pre-registration uses
        the tournament defaults, so leave ``None``).

    Returns
    -------
    tuple
        ``(tables, metadata)``. ``metadata["source"]`` is ``"loaded"`` or
        ``"computed"``; ``metadata["post_hoc"]`` marks non-F1 versions.

    Raises
    ------
    ValueError
        If ``allow_test`` is not ``True`` or ``version`` is not a valid name.
    """
    if allow_test is not True:
        raise ValueError(
            "the final evaluation opens the test period; pass allow_test=True "
            "only in the single pre-registered final evaluation"
        )
    _validate_version(version)
    post_hoc = version != PRIMARY_VERSION
    root = Path(cache_root) if cache_root is not None else _default_cache_root()
    models_root = Path(model_dir) if model_dir is not None else root.parent / "models" / "catboost_final"
    config = {
        "run_version": version,
        "post_hoc": post_hoc,
        "train_end_exclusive": FINAL_TRAIN_END,
        "test_start": FINAL_TEST_START,
        "test_end_exclusive": FINAL_TEST_END,
        "n_boot": n_boot,
        "seed": seed,
        "random_state": FINAL_RANDOM_STATE,
        "calibration_cv": CALIBRATION_CV,
        "catboost_params": catboost_params or {},
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
    }
    frames = {"modeling_frame": modeling_frame}
    if orders is not None:
        frames["orders"] = orders
    model_target = models_root / version

    def guard() -> None:
        if model_target.exists():
            raise FileExistsError(f"final model directory already exists: {model_target}")

    def compute(fingerprint: str) -> dict[str, pd.DataFrame]:
        return _compute_final_tables(
            modeling_frame,
            orders,
            version=version,
            post_hoc=post_hoc,
            fingerprint=fingerprint,
            n_boot=n_boot,
            seed=seed,
            catboost_params=catboost_params,
            model_target=model_target,
        )

    def ledger(tables: dict[str, pd.DataFrame], meta: dict) -> dict:
        provenance = tables["provenance"].iloc[0]
        return {
            "version": version,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "fingerprint": meta["fingerprint"],
            "train_orders": int(provenance["train_orders"]),
            "test_orders": int(provenance["test_orders"]),
            "post_hoc": post_hoc,
            "n_boot": n_boot,
            "seed": seed,
        }

    return _run_frozen(
        version=version,
        post_hoc=post_hoc,
        root=root,
        config=config,
        frames=frames,
        compute=compute,
        ledger_record=ledger,
        before_compute=guard,
        stacklevel=3,
    )
