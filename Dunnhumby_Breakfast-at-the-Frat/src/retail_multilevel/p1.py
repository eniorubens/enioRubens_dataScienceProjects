"""Protocol P1: internal-fold selection of the random part (decision register, "Protocolo P1" and E1-E6).

Only the two internal folds are handled here (fit and validation windows inside weeks 0-103):
fold A fits on weeks 0-51 and validates on 52-77, fold B fits on 0-77 and validates on 78-103. Nothing
in this module may touch a row with ``WEEK >= 104``; :func:`prepare_fold` refuses such rows.

Candidates (fixed part F3 in all of them, all fitted by REML on the fit window):

- ``C0``: ``(1|STORE_NUM) + (1|UPC) + (1|STORE_NUM:UPC)``
- ``C1``: C0 + ``(0 + LOG_DISC|UPC) + (0 + LOG_DISC|STORE_NUM)``
- ``C2``: ``(1 + LOG_DISC|UPC) + (1 + LOG_DISC|STORE_NUM) + (1|STORE_NUM:UPC)``
- ``C3``: last accepted + ``(0 + LOG_DISC|STORE_NUM:UPC)``
- ``C4``: last accepted + ``(0 + FEATURE|UPC) + (0 + DISPLAY|UPC)``

Each candidate is compared with the *last accepted* model (starting at C0). It advances only when it
converges without singularity in both folds, lowers the primary metric in both folds and the mean over
folds of the relative reduction is at least 0.5%. The first failure stops the sequence; later
candidates are not fitted. The primary metric is the MAE of ``LOG_UNITS`` per pair, averaged with equal
weight over pairs.

References without a selection role: pooled OLS with the F3 fixed part and a per-pair OLS
(``LOG_DISC + FEATURE + DISPLAY + TPR_ONLY``) fitted only for pairs with at least ``MIN_WEEKS_OLS``
weeks in the fit window. The metric and OLS functions are pure (numpy/pandas); only
:func:`fit_predict_lmer` needs R.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import pandas as pd

from retail_multilevel.features import add_model_features, pair_means, upc_means
from retail_multilevel.fixedeffects import DEFAULT_FE_CONTROL, FE_MODELS, RANDOM
from retail_multilevel.split import FIRST_TEST_WEEK

FIXED_FORMULA = FE_MODELS[-1][1]
FIXED_TERMS = [t.strip() for t in FIXED_FORMULA.split("~")[1].split("+") if t.strip() != "1"]
OLS_TERMS = ["LOG_DISC", "FEATURE", "DISPLAY", "TPR_ONLY"]
MIN_WEEKS_OLS = 8
MIN_REL_GAIN = 0.005
RETRY_CONTROL = 'lmerControl(optimizer="nloptwrap")'
SINGULAR_MSG = "singular fit"

FOLDS: dict[str, dict[str, tuple[int, int]]] = {
    "A": {"fit": (0, 51), "val": (52, 77)},
    "B": {"fit": (0, 77), "val": (78, 103)},
}

# (name, kind, text): "add" appends ``text`` to the last accepted random part, "replace" uses it whole.
CANDIDATES: list[tuple[str, str, str]] = [
    ("C1", "add", "(0 + LOG_DISC | UPC) + (0 + LOG_DISC | STORE_NUM)"),
    ("C2", "replace", "(1 + LOG_DISC | UPC) + (1 + LOG_DISC | STORE_NUM) + (1 | STORE_NUM:UPC)"),
    ("C3", "add", "(0 + LOG_DISC | STORE_NUM:UPC)"),
    ("C4", "add", "(0 + FEATURE | UPC) + (0 + DISPLAY | UPC)"),
]
BASE_MODEL = "C0"
OLS_POOLED = "OLS_pooled"
OLS_PAIR = "OLS_pair"

POP_SEEN = "a_seen"
POP_OLS = "b_ols_covered"
POP_UNSEEN = "c_unseen"
STRATA = ["<26", "26-77", ">=78"]
STRATUM_ALL = "all"

SELECTION_COLS = [
    "step", "candidate", "compared_to", "mae_fold_A", "mae_fold_A_incumbent", "mae_fold_B", "mae_fold_B_incumbent",
    "rel_gain_A", "rel_gain_B", "mean_rel_gain", "converged_ok", "decision", "reason",
]  # fmt: skip
METRIC_COLS = [
    "model", "fold", "population", "stratum", "n_pairs", "n_rows", "mae_pair_mean", "mae_row", "rmse",
]  # fmt: skip
CONV_COLS = [
    "model", "fold", "n_messages", "messages", "singular", "optimizer", "seconds", "npar", "nobs",
]  # fmt: skip


def protocol_config() -> dict:
    """Return the full protocol settings, for the run configuration (and its fingerprint)."""
    return {
        "step": "p1_folds",
        "folds": {k: {"fit": list(v["fit"]), "val": list(v["val"])} for k, v in FOLDS.items()},
        "fixed_formula": FIXED_FORMULA,
        "base_model": {"name": BASE_MODEL, "random": RANDOM},
        "candidates": [list(c) for c in CANDIDATES],
        "reml": True,
        "control": DEFAULT_FE_CONTROL,
        "retry_control": RETRY_CONTROL,
        "min_rel_gain": MIN_REL_GAIN,
        "rule": "advance only if converged without singularity in both folds, lower pair-mean MAE in both folds "
        "and mean relative reduction >= min_rel_gain; first failure stops the sequence",
        "prediction": "lme4 predict(allow.new.levels=TRUE); unseen pairs get pair effect 0",
        "population": "IN_MODEL rows; seen = MEAN_SOURCE pair, unseen = upc, none = not predictable",
        "pair_ols": {
            "terms": OLS_TERMS,
            "min_weeks": MIN_WEEKS_OLS,
            "coverage": "term without variation or dropped by collinearity is not estimated; a validation row is "
            "not covered if it has a nonzero value on a not-estimated term or the pair has no OLS",
        },
        "strata": STRATA,
        "weekly_diff": "per validation week, mean over rows of |err candidate| - |err incumbent| (population a)",
        "primary_metric": "MAE of LOG_UNITS per pair, equal weight per pair",
        "max_week_allowed": FIRST_TEST_WEEK - 1,
    }


# --------------------------------------------------------------------------------------------------------
# Data preparation
# --------------------------------------------------------------------------------------------------------
def prepare_fold(
    rows: pd.DataFrame, fit_weeks: tuple[int, int], val_weeks: tuple[int, int], max_week: int = FIRST_TEST_WEEK - 1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``rows`` into fit and validation windows and add the model features of each.

    Pair means and UPC means are learned on the fit window only and applied to both windows.

    Parameters
    ----------
    rows : pandas.DataFrame
        ``IN_MODEL`` rows with the panel columns used by :func:`add_model_features`.
    fit_weeks, val_weeks : tuple of int
        Inclusive ``(first, last)`` week of each window.
    max_week : int
        Highest week that may be used; any row of the windows above it raises (final-window guard).

    Returns
    -------
    tuple of pandas.DataFrame
        ``(fit_df, val_df)`` with features, reset index.

    Raises
    ------
    ValueError
        If a window goes beyond ``max_week`` or the windows overlap.
    """
    if max(fit_weeks[1], val_weeks[1]) > max_week:
        raise ValueError(f"windows {fit_weeks}/{val_weeks} go beyond the allowed week {max_week}")
    if fit_weeks[1] >= val_weeks[0]:
        raise ValueError("the validation window must start after the fit window")
    week = rows["WEEK"]
    fit = rows.loc[(week >= fit_weeks[0]) & (week <= fit_weeks[1])]
    val = rows.loc[(week >= val_weeks[0]) & (week <= val_weeks[1])]
    if len(fit) == 0 or len(val) == 0:
        raise ValueError("empty fit or validation window")
    fit_f = add_model_features(fit, pair_means(fit), upc_means(fit)).reset_index(drop=True)
    val_f = add_model_features(val, pair_means(fit), upc_means(fit)).reset_index(drop=True)
    return fit_f, val_f


def history_stratum(history: pd.Series | np.ndarray) -> pd.Series:
    """Label the fit-window history (weeks of the pair) as ``<26``, ``26-77`` or ``>=78``."""
    h = pd.Series(np.asarray(history))
    out = np.where(h < 26, STRATA[0], np.where(h < 78, STRATA[1], STRATA[2]))
    return pd.Series(out, index=h.index, dtype="object")


# --------------------------------------------------------------------------------------------------------
# Reference models in numpy
# --------------------------------------------------------------------------------------------------------
def pooled_ols_predict(fit_df: pd.DataFrame, val_df: pd.DataFrame, terms: list[str] | None = None) -> np.ndarray:
    """Pooled OLS with the F3 fixed part, fitted on ``fit_df`` and predicted on ``val_df``.

    Rows with ``MEAN_SOURCE == "none"`` (no admissible covariate) get ``NaN``.

    Returns
    -------
    numpy.ndarray
        Predictions aligned with ``val_df``.
    """
    terms = FIXED_TERMS if terms is None else terms
    x = np.column_stack([np.ones(len(fit_df)), fit_df[terms].to_numpy(dtype=float)])
    beta, *_ = np.linalg.lstsq(x, fit_df["LOG_UNITS"].to_numpy(dtype=float), rcond=None)
    xv = np.column_stack([np.ones(len(val_df)), val_df[terms].to_numpy(dtype=float)])
    pred = xv @ beta
    pred[(val_df["MEAN_SOURCE"] == "none").to_numpy()] = np.nan
    return pred


def pair_ols_fit(
    fit_df: pd.DataFrame, terms: list[str] | None = None, min_weeks: int = MIN_WEEKS_OLS
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Fit one OLS per pair (``LOG_UNITS`` on the terms) on the fit window.

    Only pairs with at least ``min_weeks`` rows (weeks) are fitted. Terms are screened in the given
    order: a term without variation in the pair's fit window, or one that does not raise the rank of the
    design (collinearity), is *not estimated* and gets coefficient 0 (the intercept is always kept).

    Returns
    -------
    tuple
        ``(info, coef, est)``. ``info`` has one row per pair of the fit window (``PAIR_ID, n_weeks,
        has_ols``). ``coef`` is a ``(n_pairs, 1 + len(terms))`` array (intercept first) and ``est`` a
        boolean array of the same shape saying which coefficients were estimated (all ``False`` without OLS).
    """
    terms = OLS_TERMS if terms is None else terms
    groups = fit_df.groupby("PAIR_ID", sort=True).indices
    ids = list(groups)
    k = len(terms)
    coef = np.zeros((len(ids), k + 1))
    est = np.zeros((len(ids), k + 1), dtype=bool)
    n_weeks = np.zeros(len(ids), dtype=int)
    y_all = fit_df["LOG_UNITS"].to_numpy(dtype=float)
    x_all = fit_df[terms].to_numpy(dtype=float)
    for i, pid in enumerate(ids):
        idx = groups[pid]
        n_weeks[i] = len(idx)
        if len(idx) < min_weeks:
            continue
        xs = x_all[idx]
        cols = [np.ones(len(idx))]
        keep = [0]
        for j in range(k):
            col = xs[:, j]
            if np.ptp(col) <= 1e-12:
                continue
            trial = np.column_stack([*cols, col])
            if np.linalg.matrix_rank(trial) <= len(cols):
                continue
            cols.append(col)
            keep.append(j + 1)
        design = np.column_stack(cols)
        beta, *_ = np.linalg.lstsq(design, y_all[idx], rcond=None)
        coef[i, keep] = beta
        est[i, keep] = True
    info = pd.DataFrame({"PAIR_ID": ids, "n_weeks": n_weeks, "has_ols": est[:, 0]})
    return info, coef, est


def pair_ols_predict(
    fit_df: pd.DataFrame, val_df: pd.DataFrame, terms: list[str] | None = None, min_weeks: int = MIN_WEEKS_OLS
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Per-pair OLS predictions and the per-term coverage rule (amendment E2).

    A validation row is *not covered* when its pair has no OLS (absent from the fit window or with fewer
    than ``min_weeks`` weeks) or when it has a nonzero value on a term that was not estimated for the pair
    (no variation in the pair's fit window, or dropped by collinearity).

    Returns
    -------
    tuple
        ``(pred, covered, info)``: ``pred`` (``NaN`` where not covered) and ``covered`` aligned with
        ``val_df``, and ``info``, the per-pair table of :func:`pair_ols_fit` plus one boolean column
        ``est_<term>`` per term.
    """
    terms = OLS_TERMS if terms is None else terms
    info, coef, est = pair_ols_fit(fit_df, terms, min_weeks)
    for j, t in enumerate(terms):
        info[f"est_{t}"] = est[:, j + 1]
    pos = pd.Index(info["PAIR_ID"]).get_indexer(val_df["PAIR_ID"])
    has = (pos >= 0) & np.where(pos >= 0, info["has_ols"].to_numpy()[np.maximum(pos, 0)], False)
    safe = np.maximum(pos, 0)
    xv = val_df[terms].to_numpy(dtype=float)
    bad = ((xv != 0) & ~est[safe, 1:]).any(axis=1)
    covered = has & ~bad
    pred = coef[safe, 0] + (coef[safe, 1:] * xv).sum(axis=1)
    pred = np.where(covered, pred, np.nan)
    return pred, covered, info


# --------------------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------------------
def pair_mean_mae(abs_err: pd.Series | np.ndarray, pair: pd.Series | np.ndarray) -> float:
    """MAE per pair, then the mean over pairs with equal weight (the primary metric)."""
    s = pd.Series(np.asarray(abs_err, dtype=float))
    return float(s.groupby(np.asarray(pair)).mean().mean())


def _metric_row(model: str, fold: str, pop: str, stratum: str, y: np.ndarray, pred: np.ndarray, pair: np.ndarray):
    if len(y) == 0:
        return [model, fold, pop, stratum, 0, 0, np.nan, np.nan, np.nan]
    err = pred - y
    ae = np.abs(err)
    return [
        model, fold, pop, stratum, int(pd.Series(pair).nunique()), int(len(y)),
        pair_mean_mae(ae, pair), float(ae.mean()), float(np.sqrt((err**2).mean())),
    ]  # fmt: skip


def eval_frame(val_df: pd.DataFrame, fit_df: pd.DataFrame, ols_covered: np.ndarray) -> pd.DataFrame:
    """Build the evaluation frame of a fold: one row per validation row, no predictions yet.

    Columns: ``PAIR_ID, WEEK, y, POP`` (``seen``/``unseen``/``none``), ``HISTORY`` (weeks of the pair in the
    fit window), ``STRATUM`` and ``COVERED`` (per-pair OLS coverage, only true on seen pairs).
    """
    hist = fit_df.groupby("PAIR_ID").size()
    pop = val_df["MEAN_SOURCE"].map({"pair": "seen", "upc": "unseen", "none": "none"}).to_numpy()
    ev = pd.DataFrame(
        {
            "PAIR_ID": val_df["PAIR_ID"].to_numpy(),
            "WEEK": val_df["WEEK"].to_numpy(),
            "y": val_df["LOG_UNITS"].to_numpy(dtype=float),
            "POP": pop,
            "HISTORY": val_df["PAIR_ID"].map(hist).fillna(0).to_numpy(dtype=int),
        }
    )
    ev["STRATUM"] = history_stratum(ev["HISTORY"]).to_numpy()
    ev["COVERED"] = np.asarray(ols_covered, dtype=bool) & (ev["POP"] == "seen").to_numpy()
    return ev


def metric_table(ev: pd.DataFrame, preds: dict[str, np.ndarray], fold: str) -> pd.DataFrame:
    """Metrics of every model in the three populations and the history strata.

    Populations: ``a_seen`` (seen pairs; all models except the per-pair OLS), ``b_ols_covered`` (seen rows
    covered by the per-pair OLS; all models including it, on identical rows) and ``c_unseen`` (unseen
    pairs; multilevel models and pooled OLS). Strata (``<26``, ``26-77``, ``>=78`` weeks in the fit
    window) are reported within a and b; empty strata appear with zero counts and ``NaN`` metrics.

    Parameters
    ----------
    ev : pandas.DataFrame
        Output of :func:`eval_frame`.
    preds : dict
        Model name to predictions aligned with ``ev``; ``NaN`` is allowed only outside the population.
    fold : str
        Fold label.

    Returns
    -------
    pandas.DataFrame
        Columns ``model, fold, population, stratum, n_pairs, n_rows, mae_pair_mean, mae_row, rmse``.

    Raises
    ------
    ValueError
        If a model has ``NaN`` predictions inside a population it is evaluated on.
    """
    pops = {
        POP_SEEN: (ev["POP"] == "seen").to_numpy(),
        POP_OLS: ((ev["POP"] == "seen") & ev["COVERED"]).to_numpy(),
        POP_UNSEEN: (ev["POP"] == "unseen").to_numpy(),
    }
    rows = []
    for pop, mask in pops.items():
        for model, pred in preds.items():
            if model == OLS_PAIR and pop != POP_OLS:
                continue
            if model != OLS_PAIR and pop == POP_OLS and OLS_PAIR not in preds:
                continue
            pred = np.asarray(pred, dtype=float)
            if np.isnan(pred[mask]).any():
                raise ValueError(f"{model} has NaN predictions inside population {pop}")
            strata = [STRATUM_ALL] + (STRATA if pop != POP_UNSEEN else [])
            for st in strata:
                m = mask if st == STRATUM_ALL else mask & (ev["STRATUM"] == st).to_numpy()
                rows.append(_metric_row(model, fold, pop, st, ev["y"].to_numpy()[m], pred[m], ev["PAIR_ID"].to_numpy()[m]))
    return pd.DataFrame(rows, columns=METRIC_COLS)


def coverage_row(ev: pd.DataFrame, info: pd.DataFrame, fold: str) -> dict:
    """Coverage of the per-pair OLS (E2), in rows and pairs, plus the unseen and non-predictable counts."""
    seen = ev["POP"] == "seen"
    n_seen_rows = int(seen.sum())
    n_seen_pairs = int(ev.loc[seen, "PAIR_ID"].nunique())
    cov = ev["COVERED"]
    n_cov_pairs = int(ev.loc[cov, "PAIR_ID"].nunique())
    ols_pairs = set(info.loc[info["has_ols"], "PAIR_ID"])
    seen_pairs = set(ev.loc[seen, "PAIR_ID"])
    row = {
        "fold": fold,
        "n_seen_pairs": n_seen_pairs,
        "n_seen_rows": n_seen_rows,
        "n_seen_pairs_with_ols": len(seen_pairs & ols_pairs),
        "n_covered_pairs": n_cov_pairs,
        "n_covered_rows": int(cov.sum()),
        "pct_pairs_covered": 100 * n_cov_pairs / n_seen_pairs if n_seen_pairs else np.nan,
        "pct_rows_covered": 100 * int(cov.sum()) / n_seen_rows if n_seen_rows else np.nan,
        "n_unseen_pairs": int(ev.loc[ev["POP"] == "unseen", "PAIR_ID"].nunique()),
        "n_unseen_rows": int((ev["POP"] == "unseen").sum()),
        "n_none_rows": int((ev["POP"] == "none").sum()),
    }
    sub = info[info["PAIR_ID"].isin(seen_pairs) & info["has_ols"]]
    for c in [c for c in info.columns if c.startswith("est_") and c != "est_"]:
        row[f"n_pairs_not_estimated_{c[4:]}"] = int((~sub[c]).sum())
    return row


def weekly_diff(ev: pd.DataFrame, cand: np.ndarray, inc: np.ndarray, fold: str, names: tuple[str, str]) -> pd.DataFrame:
    """Weekly series of the error difference between a candidate and the incumbent (population a).

    For each validation week: ``mae_candidate`` and ``mae_incumbent`` are the means over the seen rows of
    the week of the absolute error (each pair has at most one row per week, so this is the mean over
    pairs), and ``mae_diff = mae_candidate - mae_incumbent`` (negative = candidate better).
    """
    m = (ev["POP"] == "seen").to_numpy()
    d = pd.DataFrame(
        {
            "WEEK": ev["WEEK"].to_numpy()[m],
            "ae_c": np.abs(np.asarray(cand)[m] - ev["y"].to_numpy()[m]),
            "ae_i": np.abs(np.asarray(inc)[m] - ev["y"].to_numpy()[m]),
        }
    )
    g = d.groupby("WEEK").agg(n_rows=("ae_c", "size"), mae_candidate=("ae_c", "mean"), mae_incumbent=("ae_i", "mean"))
    g["mae_diff"] = g["mae_candidate"] - g["mae_incumbent"]
    g = g.reset_index()
    g.insert(0, "compared_to", names[1])
    g.insert(0, "candidate", names[0])
    g.insert(0, "fold", fold)
    return g


# --------------------------------------------------------------------------------------------------------
# Selection rule
# --------------------------------------------------------------------------------------------------------
def rel_gain(mae_candidate: float, mae_incumbent: float) -> float:
    """Relative MAE reduction ``(incumbent - candidate) / incumbent`` (positive = candidate better)."""
    return (mae_incumbent - mae_candidate) / mae_incumbent


def decide_step(
    step: int,
    candidate: str,
    incumbent: str,
    mae: dict[str, tuple[float, float]],
    converged: dict[str, bool],
    min_gain: float = MIN_REL_GAIN,
) -> dict:
    """Apply the selection rule of one comparison (E3).

    Parameters
    ----------
    step : int
        Position in the sequence.
    candidate, incumbent : str
        Names of the candidate and of the last accepted model.
    mae : dict
        Fold label (``"A"``, ``"B"``) to ``(mae_candidate, mae_incumbent)``, the primary metric in
        population a.
    converged : dict
        Fold label to whether the candidate converged without singularity.
    min_gain : float
        Minimum mean relative reduction (0.005 = 0.5%).

    Returns
    -------
    dict
        One row of the selection log (``SELECTION_COLS``); ``decision`` is ``"accepted"`` or ``"rejected"``.
    """
    gains = {}
    for f in ("A", "B"):
        c, i = mae[f]
        gains[f] = rel_gain(c, i) if np.isfinite(c) and np.isfinite(i) else np.nan
    ok = bool(converged["A"] and converged["B"])
    mean_gain = float(np.mean([gains["A"], gains["B"]])) if all(np.isfinite(list(gains.values()))) else np.nan
    fails = []
    for f in ("A", "B"):
        if not converged[f]:
            fails.append(f"fold {f}: not converged or singular")
    for f in ("A", "B"):
        if not (np.isfinite(gains[f]) and mae[f][0] < mae[f][1]):
            g = gains[f]
            fails.append(f"fold {f}: MAE not lower ({'n/a' if not np.isfinite(g) else f'{100 * g:.3f}%'})")
    if np.isfinite(mean_gain) and mean_gain < min_gain:
        fails.append(f"mean relative gain {100 * mean_gain:.3f}% < {100 * min_gain:.1f}%")
    accepted = ok and not fails
    reason = (
        f"converged in both folds, MAE lower in both folds, mean relative gain {100 * mean_gain:.3f}% >= "
        f"{100 * min_gain:.1f}%"
        if accepted
        else "; ".join(fails)
    )
    return {
        "step": step,
        "candidate": candidate,
        "compared_to": incumbent,
        "mae_fold_A": mae["A"][0],
        "mae_fold_A_incumbent": mae["A"][1],
        "mae_fold_B": mae["B"][0],
        "mae_fold_B_incumbent": mae["B"][1],
        "rel_gain_A": gains["A"],
        "rel_gain_B": gains["B"],
        "mean_rel_gain": mean_gain,
        "converged_ok": ok,
        "decision": "accepted" if accepted else "rejected",
        "reason": reason,
    }


def not_fitted_row(step: int, candidate: str, incumbent: str) -> dict:
    """Selection-log row of a candidate that was not fitted because the sequence stopped."""
    row = dict.fromkeys(SELECTION_COLS, np.nan)
    row.update(
        step=step, candidate=candidate, compared_to=incumbent, converged_ok=False,
        decision="not fitted", reason="not fitted: sequence stopped",
    )  # fmt: skip
    return row


def candidate_random_part(kind: str, text: str, last_accepted_random: str) -> str:
    """Random part of a candidate: ``add`` appends to the last accepted part, ``replace`` uses ``text``."""
    if kind == "add":
        return f"{last_accepted_random} + {text}"
    if kind == "replace":
        return text
    raise ValueError(f"unknown kind {kind!r}")


# --------------------------------------------------------------------------------------------------------
# Multilevel fit and prediction (needs R)
# --------------------------------------------------------------------------------------------------------
def _conv_messages(messages: list[str]) -> list[str]:
    return [m for m in messages if m != SINGULAR_MSG]


def predict_lmer(r_model, newdata: pd.DataFrame) -> np.ndarray:
    """Predict with ``predict(m, newdata, allow.new.levels = TRUE)`` (new levels get random effect 0)."""
    from retail_multilevel.multilevel import to_r_frame
    from retail_multilevel.rbridge import get_robjects

    ro = get_robjects()
    cols = ["STORE_NUM", "UPC", *FIXED_TERMS]
    r_new = to_r_frame(newdata[cols].reset_index(drop=True), ["STORE_NUM", "UPC"])
    res = ro.r("function(m, nd) as.numeric(predict(m, newdata = nd, allow.new.levels = TRUE))")(r_model, r_new)
    return np.asarray(res, dtype=float)


def fit_predict_lmer(
    fit_df: pd.DataFrame, val_df: pd.DataFrame, random_part: str, name: str, fold: str
) -> tuple[np.ndarray | None, dict]:
    """Fit ``LOG_UNITS ~ F3 + random_part`` by REML on the fit window and predict the validation rows.

    The first fit uses the default bobyqa control. If lme4 reports convergence messages, it is refitted
    once with ``nloptwrap``; remaining messages or a singular fit make the model *not ok*. Rows with
    ``MEAN_SOURCE == "none"`` are not predicted (``NaN``).

    Returns
    -------
    tuple
        ``(pred, conv)``: predictions aligned with ``val_df`` (``None`` if the fit raised an R error) and
        a dict with the keys of ``CONV_COLS`` plus ``converged_ok``.
    """
    from retail_multilevel.multilevel import fit_lmer, to_r_frame
    from retail_multilevel.rbridge import get_robjects

    formula = f"{FIXED_FORMULA} + {random_part}"
    cols = ["LOG_UNITS", "STORE_NUM", "UPC", *FIXED_TERMS]
    r_fit = to_r_frame(fit_df[cols], ["STORE_NUM", "UPC"])
    t0 = time.perf_counter()
    optimizer, first_msgs = "bobyqa", []
    try:
        fit = fit_lmer(r_fit, formula, name, reml=True, control=DEFAULT_FE_CONTROL)
        first_msgs = list(fit.messages)
        if _conv_messages(fit.messages):
            optimizer = "nloptwrap"
            fit = fit_lmer(r_fit, formula, name, reml=True, control=RETRY_CONTROL)
    except Exception as exc:  # noqa: BLE001  (an R error is recorded as a failed fit)
        conv = {"model": name, "fold": fold, "n_messages": 1, "messages": f"R error: {exc}"[:500], "singular": False,
                "optimizer": optimizer, "seconds": time.perf_counter() - t0, "npar": np.nan, "nobs": len(fit_df),
                "converged_ok": False}  # fmt: skip
        return None, conv
    singular = SINGULAR_MSG in fit.messages
    conv_msgs = _conv_messages(fit.messages)
    msgs = list(fit.messages)
    if optimizer == "nloptwrap":
        msgs = [f"[first attempt, bobyqa] {m}" for m in _conv_messages(first_msgs)] + msgs
    conv = {
        "model": name, "fold": fold, "n_messages": len(conv_msgs), "messages": " | ".join(msgs),
        "singular": bool(singular), "optimizer": optimizer, "seconds": time.perf_counter() - t0,
        "npar": fit.npar, "nobs": fit.nobs, "converged_ok": bool(not conv_msgs and not singular),
    }  # fmt: skip
    pred = np.full(len(val_df), np.nan)
    ok = (val_df["MEAN_SOURCE"] != "none").to_numpy()
    pred[ok] = predict_lmer(fit.r_model, val_df.loc[ok])
    del fit
    get_robjects().r("invisible(gc())")
    return pred, conv


# --------------------------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------------------------
FitPredict = Callable[[pd.DataFrame, pd.DataFrame, str, str, str], tuple["np.ndarray | None", dict]]


def run_p1(
    rows: pd.DataFrame,
    folds: dict[str, dict[str, tuple[int, int]]] | None = None,
    fit_predict: FitPredict | None = None,
    min_weeks_ols: int = MIN_WEEKS_OLS,
    min_gain: float = MIN_REL_GAIN,
    log: Callable[[str], None] = lambda s: None,
) -> dict:
    """Run the P1 internal-fold selection.

    Parameters
    ----------
    rows : pandas.DataFrame
        ``IN_MODEL`` panel rows (any rows beyond week 103 are ignored by the fold windows and the guard).
    folds : dict, optional
        Fold definitions as in ``FOLDS`` (must contain ``"A"`` and ``"B"``).
    fit_predict : callable, optional
        ``fit_predict(fit_df, val_df, random_part, name, fold) -> (pred or None, conv dict)``; defaults to
        :func:`fit_predict_lmer`. Injectable so that the orchestration can be tested without R.
    min_weeks_ols : int
        Minimum fit-window weeks for the per-pair OLS.
    min_gain : float
        Minimum mean relative reduction to accept a candidate.
    log : callable
        Progress sink.

    Returns
    -------
    dict
        ``selection_log, convergence, metrics, coverage, weekly_diff`` (DataFrames) and ``selected``
        (``{"model", "random_part", "fixed"}``).
    """
    folds = FOLDS if folds is None else folds
    fit_predict = fit_predict_lmer if fit_predict is None else fit_predict
    data, ev, preds, coverage = {}, {}, {}, []
    for f, spec in folds.items():
        fit_df, val_df = prepare_fold(rows, spec["fit"], spec["val"])
        pp, cov, info = pair_ols_predict(fit_df, val_df, min_weeks=min_weeks_ols)
        ev[f] = eval_frame(val_df, fit_df, cov)
        preds[f] = {OLS_POOLED: pooled_ols_predict(fit_df, val_df), OLS_PAIR: pp}
        coverage.append(coverage_row(ev[f], info, f))
        data[f] = (fit_df, val_df)
        log(f"fold {f}: fit {len(fit_df)} rows, validation {len(val_df)} rows")

    convergence, sel_rows, wk = [], [], []

    def fit_all(name: str, random_part: str) -> dict[str, dict]:
        out = {}
        for f in folds:
            pred, conv = fit_predict(data[f][0], data[f][1], random_part, name, f)
            convergence.append({k: conv.get(k) for k in CONV_COLS})
            if pred is not None:
                preds[f][name] = pred
            out[f] = conv
            log(f"{name} fold {f}: ok={conv['converged_ok']} {conv['seconds']:.0f}s")
        return out

    def mae_seen(f: str, name: str) -> float:
        if name not in preds[f]:
            return np.nan
        mask = (ev[f]["POP"] == "seen").to_numpy()
        ae = np.abs(preds[f][name][mask] - ev[f]["y"].to_numpy()[mask])
        return pair_mean_mae(ae, ev[f]["PAIR_ID"].to_numpy()[mask])

    fit_all(BASE_MODEL, RANDOM)
    if BASE_MODEL not in preds["A"] or BASE_MODEL not in preds["B"]:
        raise RuntimeError("the base model C0 failed to fit")
    accepted, accepted_random = BASE_MODEL, RANDOM
    stopped = False
    for step, (name, kind, text) in enumerate(CANDIDATES, start=1):
        if stopped:
            sel_rows.append(not_fitted_row(step, name, accepted))
            continue
        random_part = candidate_random_part(kind, text, accepted_random)
        conv = fit_all(name, random_part)
        ok = {f: bool(conv[f]["converged_ok"]) and name in preds[f] for f in folds}
        mae = {f: (mae_seen(f, name), mae_seen(f, accepted)) for f in folds}
        row = decide_step(step, name, accepted, mae, ok, min_gain)
        sel_rows.append(row)
        for f in folds:
            if name in preds[f]:
                wk.append(weekly_diff(ev[f], preds[f][name], preds[f][accepted], f, (name, accepted)))
        log(f"{name} vs {accepted}: {row['decision']} ({row['reason']})")
        if row["decision"] == "accepted":
            accepted, accepted_random = name, random_part
        else:
            stopped = True

    metrics = pd.concat([metric_table(ev[f], preds[f], f) for f in folds], ignore_index=True)
    return {
        "selection_log": pd.DataFrame(sel_rows, columns=SELECTION_COLS),
        "convergence": pd.DataFrame(convergence, columns=CONV_COLS),
        "metrics": metrics,
        "coverage": pd.DataFrame(coverage),
        "weekly_diff": pd.concat(wk, ignore_index=True) if wk else pd.DataFrame(),
        "selected": {"model": accepted, "random_part": accepted_random, "fixed": FIXED_FORMULA},
    }
