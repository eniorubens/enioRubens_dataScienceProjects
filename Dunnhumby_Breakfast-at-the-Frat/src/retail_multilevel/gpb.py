"""GPBoost challenger of protocol P1: tree ensemble plus grouped random effects, tuned on internal folds.

The model is ``LOG_UNITS = F(x) + b_store + b_upc + b_pair + e`` with ``F`` a boosted tree ensemble on the
F3 covariates (:data:`FEATURES`) and independent Gaussian grouped random effects for ``STORE_NUM``, ``UPC``
and the pair (``STORE_NUM:UPC``), fitted by ``gpboost``. Groups that are absent from the fit window get a zero
random effect at prediction time (gpboost default for grouped effects), so :func:`predict` returns the tree
part plus the effects of the groups that were seen.

Tuning (Optuna TPE) uses only the two internal folds of P1 (:data:`FOLDS`); no row with ``WEEK >= 104`` is
ever fitted or scored (:func:`assert_no_final_window`). The objective is the mean over folds of the
per-pair-mean MAE of ``LOG_UNITS`` on seen pairs (:func:`pair_mean_mae`).
"""

from __future__ import annotations

import time

import gpboost as gpb
import numpy as np
import optuna
import pandas as pd

from retail_multilevel.features import FOURIER_TERMS, add_model_features, pair_means, upc_means
from retail_multilevel.split import FIRST_TEST_WEEK

FEATURES = [
    "LOG_DISC", "LOG_BASE_DEV", "LOG_BASE_MEAN", "FEATURE", "DISPLAY", "TPR_ONLY", "TREND", *FOURIER_TERMS(3),
]  # fmt: skip
GROUP_COLS = ["STORE_NUM", "UPC", "PAIR_ID"]
TARGET = "LOG_UNITS"
N_FOURIER = 3

# (name, first fit week, last fit week, first validation week, last validation week), all inclusive.
FOLDS: list[tuple[str, int, int, int, int]] = [("A", 0, 51, 52, 77), ("B", 0, 77, 78, 103)]
HISTORY_BINS = [("<26", 0, 25), ("26-77", 26, 77), (">=78", 78, 10**6)]

N_TRIALS = 50
SEED = 20261124
SEARCH_SPACE = {
    "learning_rate": {"low": 0.01, "high": 0.3, "log": True},
    "num_leaves": {"low": 8, "high": 128, "log": True, "type": "int"},
    "min_data_in_leaf": {"low": 20, "high": 500, "log": True, "type": "int"},
    "lambda_l2": {"low": 1e-3, "high": 10.0, "log": True},
    "feature_fraction": {"low": 0.6, "high": 1.0, "log": False},
    "num_boost_round": {"low": 50, "high": 1000, "log": False, "type": "int"},
}
NUM_THREADS = 8


def assert_no_final_window(df: pd.DataFrame, first_test_week: int = FIRST_TEST_WEEK) -> None:
    """Raise if ``df`` holds a row of the final window (``WEEK >= first_test_week``)."""
    if len(df) and int(df["WEEK"].max()) >= first_test_week:
        raise ValueError(f"rows with WEEK >= {first_test_week} must never be fitted or scored in tuning")


def fold_windows(panel: pd.DataFrame, fold: tuple[str, int, int, int, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the ``(fit, validation)`` ``IN_MODEL`` rows of an internal fold, with disjoint week ranges.

    Parameters
    ----------
    panel : pandas.DataFrame
        Panel with ``IN_MODEL`` and ``WEEK``.
    fold : tuple
        ``(name, first fit week, last fit week, first validation week, last validation week)``.

    Returns
    -------
    tuple of pandas.DataFrame
    """
    _, f0, f1, v0, v1 = fold
    if not f1 < v0:
        raise ValueError("validation weeks must come after the fit weeks")
    df = panel.loc[panel["IN_MODEL"]]
    fit = df.loc[df["WEEK"].between(f0, f1)]
    val = df.loc[df["WEEK"].between(v0, v1)]
    assert_no_final_window(fit)
    assert_no_final_window(val)
    return fit, val


def pair_mean_mae(df: pd.DataFrame, pred: np.ndarray | pd.Series, target: str = TARGET) -> float:
    """Mean over pairs of the pair MAE (equal weight per pair), the primary metric of P1.

    Parameters
    ----------
    df : pandas.DataFrame
        Rows with ``PAIR_ID`` and ``target``.
    pred : array-like
        Predictions aligned with ``df``.
    target : str
        Observed column.

    Returns
    -------
    float
        ``NaN`` for an empty frame.
    """
    err = pd.Series(np.abs(df[target].to_numpy(float) - np.asarray(pred, float)), index=df.index)
    return float(err.groupby(df["PAIR_ID"]).mean().mean()) if len(df) else float("nan")


def _group_array(df: pd.DataFrame) -> np.ndarray:
    return df[GROUP_COLS].astype(str).to_numpy()


def fit_model(fit_df: pd.DataFrame, params: dict, num_boost_round: int, seed: int = SEED) -> gpb.Booster:
    """Fit the grouped-random-effects boosting model on ``fit_df`` (features already added).

    Parameters
    ----------
    fit_df : pandas.DataFrame
        Fit-window rows with :data:`FEATURES`, :data:`GROUP_COLS` and ``LOG_UNITS``.
    params : dict
        LightGBM-style parameters (``learning_rate``, ``num_leaves``, ...).
    num_boost_round : int
        Number of boosting rounds (fixed, no early stopping).
    seed : int
        Seed of the tree learner.

    Returns
    -------
    gpboost.Booster
    """
    gp = gpb.GPModel(group_data=_group_array(fit_df), likelihood="gaussian")
    data = gpb.Dataset(fit_df[FEATURES].to_numpy(float), fit_df[TARGET].to_numpy(float))
    p = {"objective": "regression_l2", "verbose": -1, "seed": seed, "num_threads": NUM_THREADS, **params}
    return gpb.train(params=p, train_set=data, num_boost_round=int(num_boost_round), gp_model=gp)


def predict(booster: gpb.Booster, df: pd.DataFrame) -> np.ndarray:
    """Predict ``LOG_UNITS``: trees plus the random effects of seen groups (unseen groups add 0)."""
    out = booster.predict(
        data=df[FEATURES].to_numpy(float), group_data_pred=_group_array(df), predict_var=False, pred_latent=False
    )
    return np.asarray(out["response_mean"], float)


def prepare_folds(panel: pd.DataFrame) -> list[dict]:
    """Build, per internal fold, the fit frame, the covered validation frame and the seen-pair mask.

    Features are learned on the fit window only; validation rows without an admissible covariate
    (``MEAN_SOURCE == "none"``) are dropped.

    Parameters
    ----------
    panel : pandas.DataFrame
        Panel rows (``IN_MODEL`` and ``WEEK < 104``).

    Returns
    -------
    list of dict
        Keys ``name, fit, val, seen, history`` (``history``: distinct fit-window weeks per pair).
    """
    out = []
    for fold in FOLDS:
        fit, val = fold_windows(panel, fold)
        means, umeans = pair_means(fit), upc_means(fit)
        fit = add_model_features(fit, means, umeans, N_FOURIER)
        val = add_model_features(val, means, umeans, N_FOURIER)
        val = val.loc[val["MEAN_SOURCE"] != "none"]
        seen = val["PAIR_ID"].isin(set(fit["PAIR_ID"]))
        hist = fit.groupby("PAIR_ID")["WEEK"].nunique()
        out.append({"name": fold[0], "fit": fit, "val": val, "seen": seen, "history": hist})
    return out


def subsample_fit(fit: pd.DataFrame, frac: float, seed: int = SEED) -> pd.DataFrame:
    """Draw a fixed subsample of whole pairs of the fit window (tuning speed-up); ``frac >= 1`` returns ``fit``."""
    if frac >= 1:
        return fit
    pairs = fit["PAIR_ID"].drop_duplicates().sample(frac=frac, random_state=seed)
    return fit.loc[fit["PAIR_ID"].isin(set(pairs))]


def objective_value(folds: list[dict], params: dict, num_boost_round: int, fit_frac: float = 1.0) -> float:
    """Mean over folds of the per-pair-mean MAE on seen pairs, for one parameter set."""
    vals = []
    for f in folds:
        booster = fit_model(subsample_fit(f["fit"], fit_frac), params, num_boost_round)
        v = f["val"].loc[f["seen"]]
        vals.append(pair_mean_mae(v, predict(booster, v)))
    return float(np.mean(vals))


def suggest(trial: optuna.Trial) -> tuple[dict, int]:
    """Sample ``(params, num_boost_round)`` from :data:`SEARCH_SPACE`."""
    s = SEARCH_SPACE
    lr, nl, md, l2, ff, nr = (s[k] for k in (
        "learning_rate", "num_leaves", "min_data_in_leaf", "lambda_l2", "feature_fraction", "num_boost_round"))  # fmt: skip
    params = {
        "learning_rate": trial.suggest_float("learning_rate", lr["low"], lr["high"], log=True),
        "num_leaves": trial.suggest_int("num_leaves", nl["low"], nl["high"], log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", md["low"], md["high"], log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", l2["low"], l2["high"], log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", ff["low"], ff["high"]),
    }
    rounds = trial.suggest_int("num_boost_round", nr["low"], nr["high"])
    return params, rounds


def tune(
    folds: list[dict], n_trials: int = N_TRIALS, seed: int = SEED, fit_frac: float = 1.0, max_rounds: int | None = None
) -> optuna.Study:
    """Run the Optuna TPE search (minimising the mean per-pair-mean MAE over the internal folds).

    Parameters
    ----------
    folds : list of dict
        Output of :func:`prepare_folds`.
    n_trials : int
        Number of trials.
    seed : int
        Sampler seed.
    fit_frac : float
        Fraction of fit-window pairs used in tuning fits (1 = all).
    max_rounds : int, optional
        Cap on ``num_boost_round`` during tuning.

    Returns
    -------
    optuna.Study
    """

    def objective(trial: optuna.Trial) -> float:
        params, rounds = suggest(trial)
        if max_rounds is not None:
            rounds = min(rounds, max_rounds)
        t0 = time.perf_counter()
        value = objective_value(folds, params, rounds, fit_frac)
        trial.set_user_attr("seconds", time.perf_counter() - t0)
        return value

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study


def _metrics(v: pd.DataFrame, pred: np.ndarray) -> dict:
    if not len(v):
        return {"n_pairs": 0, "n_rows": 0, "mae_pair_mean": np.nan, "mae_row": np.nan, "rmse": np.nan}
    e = v[TARGET].to_numpy(float) - pred
    return {
        "n_pairs": int(v["PAIR_ID"].nunique()),
        "n_rows": int(len(v)),
        "mae_pair_mean": pair_mean_mae(v, pred),
        "mae_row": float(np.abs(e).mean()),
        "rmse": float(np.sqrt((e**2).mean())),
    }


def fold_metrics(f: dict, params: dict, num_boost_round: int, fit_frac: float = 1.0) -> list[dict]:
    """Metrics of one fold by population (seen, unseen) and by fit-window history stratum of seen pairs.

    Parameters
    ----------
    f : dict
        One element of :func:`prepare_folds`.
    params : dict
        Tree parameters.
    num_boost_round : int
        Boosting rounds.
    fit_frac : float
        Fit subsample fraction (as in tuning).

    Returns
    -------
    list of dict
        Keys ``fold, population, stratum, n_pairs, n_rows, mae_pair_mean, mae_row, rmse``.
    """
    booster = fit_model(subsample_fit(f["fit"], fit_frac), params, num_boost_round)
    val = f["val"]
    pred = pd.Series(predict(booster, val), index=val.index)
    rows = []
    for pop, mask in (("seen", f["seen"]), ("unseen", ~f["seen"])):
        rows.append({"fold": f["name"], "population": pop, "stratum": "all", **_metrics(val[mask], pred[mask].to_numpy())})
    nweeks = val["PAIR_ID"].map(f["history"])
    for label, lo, hi in HISTORY_BINS:
        m = f["seen"] & nweeks.between(lo, hi)
        rows.append({"fold": f["name"], "population": "seen", "stratum": label, **_metrics(val[m], pred[m].to_numpy())})
    return rows
