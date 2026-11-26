"""Reader-facing interpretation of the fixed effects of a log-units model.

The response is ``log(units)``, so a coefficient is a log-change. This module turns coefficients
(and their 95% intervals) into percentage effects on units and assembles the tables used by
notebook 03, including the joint discount-and-promotion scenarios. Visible labels come from
:mod:`retail_multilevel.i18n`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from retail_multilevel.features import YEAR_WEEKS
from retail_multilevel.i18n import MODEL_ORDER, STRATUM_ORDER, TEXTS, format_p, model_label, stratum_label, t
from retail_multilevel.runs import Run

PRICE_CHANGE = 0.10
_COEF_COLS = ("estimate", "ci_low", "ci_high")


def pct_effect_of_price_change(coef, change):
    """Relative change in units when the price is multiplied by ``1 + change``.

    For a log-log model, ``units`` scales by ``(1 + change) ** coef``; the effect is that minus 1.

    Parameters
    ----------
    coef : float or array-like
        Elasticity (coefficient of the log price term).
    change : float
        Relative price change (``-0.10`` is a 10% cut, ``0.10`` a 10% rise).

    Returns
    -------
    float or numpy.ndarray
        Relative change in units as a fraction (``0.194`` is +19.4%).
    """
    return (1.0 + change) ** np.asarray(coef, dtype=float) - 1.0


def pct_effect_of_flag(coef):
    """Relative change in units when a 0/1 flag (or one unit of a linear term) is switched on.

    Parameters
    ----------
    coef : float or array-like
        Coefficient on the log scale.

    Returns
    -------
    float or numpy.ndarray
        ``exp(coef) - 1`` as a fraction.
    """
    return np.expm1(coef)


# (i18n key, term, kind, argument)
_EFFECTS: list[tuple[str, str, str, float]] = [
    ("effect_disc10", "LOG_DISC", "price", -PRICE_CHANGE),
    ("effect_base10", "LOG_BASE_DEV", "price", PRICE_CHANGE),
    ("effect_feature", "FEATURE", "flag", 0.0),
    ("effect_display", "DISPLAY", "flag", 0.0),
    ("effect_tpr", "TPR_ONLY", "flag", 0.0),
    ("effect_trend", "TREND", "flag", 0.0),
]


def _transform(values, kind: str, arg: float):
    return pct_effect_of_price_change(values, arg) if kind == "price" else pct_effect_of_flag(values)


def effects_table(fixef: pd.DataFrame, lang: str) -> pd.DataFrame:
    """Business-readable effects on units, with 95% intervals, in percent.

    Each interval is obtained by passing the two confidence limits of the coefficient through the
    same monotone transformation as the point estimate (the limits are then ordered).

    Parameters
    ----------
    fixef : pandas.DataFrame
        Columns ``term, estimate, ci_low, ci_high`` (see ``fixedeffects.fixef_with_ci``).
    lang : str
        ``'en'`` or ``'pt'``.

    Returns
    -------
    pandas.DataFrame
        One row per effect (10% deeper discount, 10% higher regular price, feature, display, TPR,
        trend per year) with the estimate and the interval limits, in percent.
    """
    by_term = fixef.set_index("term")
    cols = [t(k, lang) for k in ("effects_col_effect", "effects_col_estimate", "effects_col_low", "effects_col_high")]
    rows = []
    for key, term, kind, arg in _EFFECTS:
        est, lo, hi = (float(_transform(by_term.loc[term, c], kind, arg)) * 100 for c in _COEF_COLS)
        rows.append([t(key, lang), est, min(lo, hi), max(lo, hi)])
    return pd.DataFrame(rows, columns=cols)


def method_comparison_table(ols: pd.DataFrame, fixef: pd.DataFrame, label: str = "multilevel") -> pd.DataFrame:
    """Stack single-level OLS estimates and the multilevel estimates in one long table.

    Parameters
    ----------
    ols : pandas.DataFrame
        Columns ``model, term, estimate, std_error`` (one block per OLS variant).
    fixef : pandas.DataFrame
        Columns ``term, estimate, std_error`` of the multilevel model.
    label : str, default "multilevel"
        Method name given to the multilevel rows.

    Returns
    -------
    pandas.DataFrame
        Columns ``method, term, estimate, std_error``.
    """
    cols = ["term", "estimate", "std_error"]
    a = ols.rename(columns={"model": "method"})[["method", *cols]]
    b = fixef[cols].assign(method=label)[["method", *cols]]
    return pd.concat([a, b], ignore_index=True)


def seasonal_profile(fixef: pd.DataFrame, n_fourier: int = 3) -> pd.DataFrame:
    """Seasonal curve implied by the Fourier coefficients, centred on its annual mean.

    Parameters
    ----------
    fixef : pandas.DataFrame
        Columns ``term, estimate`` including ``SIN_k`` and ``COS_k`` for ``k = 1..n_fourier``.
    n_fourier : int, default 3
        Number of harmonics.

    Returns
    -------
    pandas.DataFrame
        Columns ``week_of_year`` (1 to 52) and ``effect_pct``: the effect on units, in percent,
        relative to the average week of the year.
    """
    beta = fixef.set_index("term")["estimate"]
    weeks = np.arange(1, 53, dtype=float)
    curve = np.zeros_like(weeks)
    for k in range(1, n_fourier + 1):
        angle = 2 * np.pi * k * weeks / YEAR_WEEKS
        curve += beta[f"SIN_{k}"] * np.sin(angle) + beta[f"COS_{k}"] * np.cos(angle)
    return pd.DataFrame({"week_of_year": weeks.astype(int), "effect_pct": np.expm1(curve - curve.mean()) * 100})


# (i18n key, discount fraction, feature, display, TPR)
Scenario = tuple[str, float, int, int, int]

DEFAULT_SCENARIOS: list[Scenario] = [
    ("scenario_regular", 0.0, 0, 0, 0),
    ("scenario_tpr10", 0.10, 0, 0, 1),
    ("scenario_tpr20", 0.20, 0, 0, 1),
    ("scenario_feature0", 0.0, 1, 0, 0),
    ("scenario_feature20", 0.20, 1, 0, 0),
    ("scenario_display0", 0.0, 0, 1, 0),
    ("scenario_display20", 0.20, 0, 1, 0),
    ("scenario_feature_display25", 0.25, 1, 1, 0),
]
_SCENARIO_TERMS = ["LOG_DISC", "FEATURE", "DISPLAY", "TPR_ONLY"]


def _scenario_design(scenarios: list[Scenario]) -> np.ndarray:
    """Return the ``(n, 4)`` matrix of ``LOG_DISC, FEATURE, DISPLAY, TPR_ONLY`` values of the scenarios."""
    rows = []
    for name, discount, feature, display, tpr in scenarios:
        if not 0.0 <= discount < 1.0:
            raise ValueError(f"discount of scenario {name!r} must be in [0, 1), got {discount}")
        rows.append([np.log1p(-discount), feature, display, tpr])
    return np.asarray(rows, dtype=float).reshape(-1, len(_SCENARIO_TERMS))


def scenario_effects(fixef: pd.DataFrame, scenarios: list[Scenario], cov: pd.DataFrame | None = None) -> pd.DataFrame:
    """Predicted change in units of joint discount-and-promotion scenarios, relative to a regular week.

    A regular week has no discount and no promotion. For a scenario with discount fraction ``d`` the
    log-scale contrast is ``b_disc * log(1 - d) + b_feature * feature + b_display * display +
    b_tpr * tpr``, and the effect is its exponential minus one. The contrast holds all other covariates
    fixed and is a conditional association under the additive log-log specification, not a causal effect.

    The coefficient intervals of ``fixef`` cannot be combined (they ignore the covariances), so no interval
    is reported unless the covariance matrix of the fixed effects is given.

    Parameters
    ----------
    fixef : pandas.DataFrame
        Columns ``term, estimate`` with ``LOG_DISC, FEATURE, DISPLAY, TPR_ONLY``.
    scenarios : list of tuple
        ``(name, discount_fraction, feature, display, tpr)`` with the three promotion arguments in {0, 1}.
    cov : pandas.DataFrame, optional
        Covariance matrix of the fixed effects, indexed and labelled by term. When given, the output also has
        Wald limits ``ci_low`` and ``ci_high`` of the linear combination (delta method), in percent.

    Returns
    -------
    pandas.DataFrame
        Columns ``scenario, discount, feature, display, tpr, effect_pct`` (and ``ci_low, ci_high`` with ``cov``).
    """
    beta = fixef.set_index("term")["estimate"].loc[_SCENARIO_TERMS].to_numpy(dtype=float)
    design = _scenario_design(scenarios)
    eta = design @ beta
    out = pd.DataFrame(scenarios, columns=["scenario", "discount", "feature", "display", "tpr"])
    out["effect_pct"] = np.expm1(eta) * 100
    if cov is not None:
        v = cov.loc[_SCENARIO_TERMS, _SCENARIO_TERMS].to_numpy(dtype=float)
        sd = np.sqrt(np.einsum("ij,jk,ik->i", design, v, design))
        out["ci_low"] = np.expm1(eta - 1.96 * sd) * 100
        out["ci_high"] = np.expm1(eta + 1.96 * sd) * 100
    return out


def scenario_contrast(fixef: pd.DataFrame, vcov: pd.DataFrame, scenarios: list[Scenario], z: float = 1.96) -> pd.DataFrame:
    """Log-scale contrast of each scenario against a regular week, with the joint-covariance standard error.

    For each scenario the contrast is the linear combination ``x' beta`` with ``x = [log(1 - d), feature,
    display, tpr]`` on ``LOG_DISC, FEATURE, DISPLAY, TPR_ONLY`` (zero on every other term), so it holds
    the base price and all other controls constant. Its standard error is ``sqrt(x' V x)`` with ``V`` the
    joint covariance of the fixed effects; marginal coefficient limits are never combined. The effect is
    ``100 * expm1(dlog)`` and the interval ``100 * expm1(dlog -/+ z * se)`` is pointwise: a modelled
    sales ratio on the back-transformed scale, not a causal effect. A regular week gives exactly 0.

    Parameters
    ----------
    fixef : pandas.DataFrame
        Columns ``term, estimate``.
    vcov : pandas.DataFrame
        Covariance matrix of the fixed effects, indexed and labelled by term (it may hold more terms).
    scenarios : list of tuple
        ``(name, discount_fraction, feature, display, tpr)``. Promotion flags must be in {0, 1} and a TPR
        scenario cannot combine with feature or display (``TPR_ONLY`` is exclusive).
    z : float, default 1.96
        Normal quantile of the pointwise interval.

    Returns
    -------
    pandas.DataFrame
        Columns ``scenario, discount, feature, display, tpr, dlog, se, effect_pct, ci_low_pct, ci_high_pct``.

    Raises
    ------
    ValueError
        If a flag is not 0/1, a TPR scenario also has feature or display, or the discount is not in [0, 1).
    """
    for name, _, feature, display, tpr in scenarios:
        if any(v not in (0, 1) for v in (feature, display, tpr)):
            raise ValueError(f"promotion flags of scenario {name!r} must be 0 or 1")
        if tpr and (feature or display):
            raise ValueError(f"scenario {name!r} combines TPR_ONLY with feature or display (TPR_ONLY is exclusive)")
    beta = fixef.set_index("term")["estimate"].loc[_SCENARIO_TERMS].to_numpy(dtype=float)
    design = _scenario_design(scenarios)
    v = vcov.loc[_SCENARIO_TERMS, _SCENARIO_TERMS].to_numpy(dtype=float)
    dlog = design @ beta
    se = np.sqrt(np.einsum("ij,jk,ik->i", design, v, design))
    out = pd.DataFrame(scenarios, columns=["scenario", "discount", "feature", "display", "tpr"])
    out["dlog"] = dlog
    out["se"] = se
    out["effect_pct"] = np.expm1(dlog) * 100
    out["ci_low_pct"] = np.expm1(dlog - z * se) * 100
    out["ci_high_pct"] = np.expm1(dlog + z * se) * 100
    return out


def contrast_between(
    fixef: pd.DataFrame, vcov: pd.DataFrame, a: Scenario, b: Scenario, z: float = 1.96
) -> dict[str, float]:
    """Difference ``contrast(b) - contrast(a)`` on the log scale, with its joint SE and a pointwise ratio interval.

    Parameters
    ----------
    fixef, vcov : pandas.DataFrame
        As in :func:`scenario_contrast`.
    a, b : tuple
        Scenarios ``(name, discount_fraction, feature, display, tpr)``.
    z : float
        Normal quantile.

    Returns
    -------
    dict
        ``dlog, se, ratio, ratio_low, ratio_high`` where ``ratio = exp(dlog)`` is the modelled sales ratio of
        ``b`` relative to ``a`` and the limits are ``exp(dlog -/+ z * se)``.
    """
    xa, xb = _scenario_design([a, b])
    x = xb - xa
    beta = fixef.set_index("term")["estimate"].loc[_SCENARIO_TERMS].to_numpy(dtype=float)
    v = vcov.loc[_SCENARIO_TERMS, _SCENARIO_TERMS].to_numpy(dtype=float)
    dlog = float(x @ beta)
    se = float(np.sqrt(x @ v @ x))
    return {
        "dlog": dlog,
        "se": se,
        "ratio": float(np.exp(dlog)),
        "ratio_low": float(np.exp(dlog - z * se)),
        "ratio_high": float(np.exp(dlog + z * se)),
    }


def scenario_support(df: pd.DataFrame, scenarios: list[Scenario], tol: float = 0.03) -> pd.DataFrame:
    """Count the rows that realise each scenario, to show whether it lies inside the observed data.

    A row realises a scenario when its promotion flags equal the scenario's exactly and its discount
    ``1 - PRICE / BASE_PRICE`` lies within ``tol`` of the scenario's discount.

    Parameters
    ----------
    df : pandas.DataFrame
        Rows to count (e.g. the training sample), with columns ``DISCOUNT, FEATURE, DISPLAY, TPR_ONLY`` and,
        optionally, ``PAIR_ID``.
    scenarios : list of tuple
        ``(name, discount_fraction, feature, display, tpr)``.
    tol : float, default 0.03
        Half-width of the discount window, as a fraction of the base price.

    Returns
    -------
    pandas.DataFrame
        Columns ``scenario, n_rows, share`` (fraction of the rows of ``df``) and, with ``PAIR_ID``, ``n_pairs``.
    """
    disc = df["DISCOUNT"].to_numpy(dtype=float)
    feat, disp, tpr_flag = (df[c].to_numpy().astype(bool) for c in ("FEATURE", "DISPLAY", "TPR_ONLY"))
    rows = []
    for name, discount, feature, display, tpr in scenarios:
        mask = (
            (feat == bool(feature))
            & (disp == bool(display))
            & (tpr_flag == bool(tpr))
            & (np.abs(disc - discount) <= tol)
        )
        row = {"scenario": name, "n_rows": int(mask.sum()), "share": float(mask.sum()) / max(len(df), 1)}
        if "PAIR_ID" in df.columns:
            row["n_pairs"] = int(df.loc[mask, "PAIR_ID"].nunique())
        rows.append(row)
    return pd.DataFrame(rows)


def scenario_table(effects: pd.DataFrame, support: pd.DataFrame, lang: str) -> pd.DataFrame:
    """Join :func:`scenario_effects` and :func:`scenario_support` into one localised table.

    Parameters
    ----------
    effects, support : pandas.DataFrame
        Outputs of :func:`scenario_effects` and :func:`scenario_support` for the same scenarios.
    lang : str
        ``'en'`` or ``'pt'``; scenario names that are i18n keys are translated, others are kept.

    Returns
    -------
    pandas.DataFrame
        Scenario, discount (%), effect on units (%), rows in the data and share of rows (%).
    """
    merged = effects.merge(support, on="scenario", how="left")
    keys = (
        "scenario_col_name",
        "scenario_col_discount",
        "scenario_col_effect",
        "scenario_col_rows",
        "scenario_col_share",
    )
    cols = [t(k, lang) for k in keys]
    names = [t(s, lang) if s in TEXTS else s for s in merged["scenario"]]
    return pd.DataFrame(
        {
            cols[0]: names,
            cols[1]: merged["discount"] * 100,
            cols[2]: merged["effect_pct"],
            cols[3]: merged["n_rows"],
            cols[4]: merged["share"] * 100,
        }
    )


_DISCOUNT_BANDS = [-np.inf, 1e-9, 0.05, 0.10, 0.20, 0.30, np.inf]
_BAND_KEYS = ["band_none", "band_0_5", "band_5_10", "band_10_20", "band_20_30", "band_30_plus"]
_PROMO_KINDS = [
    ("none", "kind_none"),
    ("tpr", "kind_tpr"),
    ("feature", "kind_feature"),
    ("display", "kind_display"),
    ("feature_display", "kind_feature_display"),
]


def promo_by_discount_band(df: pd.DataFrame, lang: str) -> pd.DataFrame:
    """Cross-tabulate promotion type by discount band (row counts), to show the overlap in the data.

    Parameters
    ----------
    df : pandas.DataFrame
        Columns ``DISCOUNT, FEATURE, DISPLAY, TPR_ONLY``.
    lang : str
        ``'en'`` or ``'pt'``.

    Returns
    -------
    pandas.DataFrame
        Rows are promotion types (none, TPR only, feature only, display only, feature and display), columns
        are discount bands (no cut, 0-5%, 5-10%, 10-20%, 20-30%, above 30%), cells are row counts.
    """
    feat, disp, tpr = (df[c].to_numpy().astype(bool) for c in ("FEATURE", "DISPLAY", "TPR_ONLY"))
    kind = np.select([tpr, feat & disp, feat, disp], ["tpr", "feature_display", "feature", "display"], "none")
    band = pd.cut(df["DISCOUNT"].to_numpy(dtype=float), _DISCOUNT_BANDS, labels=_BAND_KEYS)
    tab = pd.crosstab(pd.Series(kind, name="kind"), pd.Series(band, name="band"), dropna=False)
    tab = tab.reindex(index=[k for k, _ in _PROMO_KINDS], columns=_BAND_KEYS, fill_value=0)
    tab.index = [t(key, lang) for _, key in _PROMO_KINDS]
    tab.columns = [t(k, lang) for k in _BAND_KEYS]
    tab.index.name = None
    tab.columns.name = None
    return tab


def compare_estimates(
    fixef: pd.DataFrame, other: pd.DataFrame, se_col: str = "std_error", term_map: dict[str, str] | None = None
) -> pd.DataFrame:
    """Put the F3 estimates side by side with those of a sensitivity analysis.

    Parameters
    ----------
    fixef : pandas.DataFrame
        Columns ``term, estimate, std_error`` of the multilevel model.
    other : pandas.DataFrame
        Columns ``term, estimate`` and ``se_col`` of the sensitivity fit.
    se_col : str, default "std_error"
        Standard-error column of ``other`` (e.g. ``se_cluster`` for pair-clustered errors).
    term_map : dict, optional
        Renames the terms of ``other`` to those of ``fixef`` (e.g. ``{"LOG_BASE_PRICE": "LOG_BASE_DEV"}``).

    Returns
    -------
    pandas.DataFrame
        One row per term of ``other`` found in ``fixef``, in the order of ``other``: ``term, estimate,
        std_error`` (multilevel), ``estimate_other, se_other``, ``difference`` (other minus multilevel) and
        ``se_ratio`` (``se_other / std_error``).
    """
    oth = other[["term", "estimate", se_col]].rename(columns={"estimate": "estimate_other", se_col: "se_other"})
    if term_map:
        oth["term"] = oth["term"].replace(term_map)
    out = oth.merge(fixef[["term", "estimate", "std_error"]], on="term", how="inner")
    out["difference"] = out["estimate_other"] - out["estimate"]
    out["se_ratio"] = out["se_other"] / out["std_error"]
    return out[["term", "estimate", "std_error", "estimate_other", "se_other", "difference", "se_ratio"]]


# --- protocol P1 (notebook 04) ----------------------------------------------------------
_METRIC_KEYS = {"mae_pair_mean": "metric_mae_pair", "mae_row": "metric_mae_row", "rmse": "metric_rmse"}


def _ordered_models(models: list[str] | None, present) -> list[str]:
    present = set(present)
    if models is not None:
        return [m for m in models if m in present]
    return [m for m in MODEL_ORDER if m in present] + sorted(present - set(MODEL_ORDER))


def _fold_text(fold: str, lang: str) -> str:
    return t("fold_final", lang) if str(fold) == "final" else str(fold)


def relative_difference(
    metrics: pd.DataFrame,
    population: str,
    stratum: str,
    model: str,
    reference: str,
    value: str = "mae_pair_mean",
    fold: str | None = None,
) -> float:
    """Return ``100 * (model - reference) / reference`` of ``value`` in one population and stratum.

    Parameters
    ----------
    metrics : pandas.DataFrame
        The ``metrics`` table of a P1 run.
    population, stratum : str
        Selected rows (for example ``'a_seen'`` and ``'all'``).
    model, reference : str
        Model codes; negative results mean ``model`` has the lower error.
    value : str, default 'mae_pair_mean'
        Metric column.
    fold : str, optional
        Fold to select; required when the table has more than one.

    Returns
    -------
    float
        Relative difference in percent.
    """
    sub = metrics[(metrics["population"] == population) & (metrics["stratum"] == stratum)]
    if fold is not None:
        sub = sub[sub["fold"] == fold]
    elif sub["fold"].nunique() > 1:
        raise ValueError("the table has several folds; pass fold")
    series = sub.set_index("model")[value]
    return float(100 * (series[model] - series[reference]) / series[reference])


def metric_table(
    metrics: pd.DataFrame,
    population: str,
    lang: str,
    value: str = "mae_pair_mean",
    models: list[str] | None = None,
    fold: str | None = None,
) -> pd.DataFrame:
    """Return one metric by history stratum and model for a population (a P1 reader table).

    Parameters
    ----------
    metrics : pandas.DataFrame
        The ``metrics`` table of a P1 run (``model``, ``fold``, ``population``, ``stratum``,
        ``n_pairs``, ``n_rows`` and the metric columns).
    population : str
        ``'a_seen'``, ``'b_ols_covered'`` or ``'c_unseen'``.
    lang : str
        ``'en'`` or ``'pt'``.
    value : str, default 'mae_pair_mean'
        Metric column shown in the model columns.
    models : list of str, optional
        Models to show, in order; by default every model present in the canonical order.
    fold : str, optional
        Restrict to one fold; when omitted and several folds exist, a fold column is added.

    Returns
    -------
    pandas.DataFrame
        One row per (fold,) stratum with pairs, rows and one column per model. Empty strata
        (no pairs) are dropped.
    """
    sub = metrics[(metrics["population"] == population) & (metrics["n_pairs"] > 0)]
    if fold is not None:
        sub = sub[sub["fold"] == fold]
    show_fold = sub["fold"].nunique() > 1
    names = _ordered_models(models, sub["model"])
    order = {s: i for i, s in enumerate(STRATUM_ORDER)}
    rows = []
    for (f, s), part in sub.groupby(["fold", "stratum"], sort=False):
        by_model = part.set_index("model")
        row = {}
        if show_fold:
            row[t("col_fold", lang)] = _fold_text(f, lang)
        row[t("col_stratum", lang)] = stratum_label(s, lang)
        first = by_model.iloc[0]
        row[t("col_pairs", lang)] = int(first["n_pairs"])
        row[t("col_rows", lang)] = int(first["n_rows"])
        for m in names:
            row[model_label(m, lang)] = float(by_model.loc[m, value]) if m in by_model.index else np.nan
        rows.append((str(f), order.get(s, 9), row))
    rows.sort(key=lambda r: (r[0], r[1]))
    return pd.DataFrame([r[2] for r in rows])


def model_comparison(
    metrics: pd.DataFrame,
    lang: str,
    specs: list[tuple[str, str, str]],
    models: list[str],
    rel: tuple[str, str] | None = None,
    fold: str | None = None,
) -> pd.DataFrame:
    """Return selected (population, stratum, metric) cells side by side for several models.

    Parameters
    ----------
    metrics : pandas.DataFrame
        The ``metrics`` table of a P1 run.
    lang : str
        ``'en'`` or ``'pt'``.
    specs : list of tuple
        ``(population, stratum, metric_column)`` triples; each becomes one row.
    models : list of str
        Model codes, one column each.
    rel : tuple of str, optional
        ``(model, reference)``: adds the relative difference in percent of the first against the second.
    fold : str, optional
        Fold to select when the table has several.

    Returns
    -------
    pandas.DataFrame
        One row per spec; the first column describes the cell.
    """
    sub = metrics if fold is None else metrics[metrics["fold"] == fold]
    rows = []
    for population, stratum, value in specs:
        part = sub[(sub["population"] == population) & (sub["stratum"] == stratum)].set_index("model")
        label = " · ".join(
            [t("pop_" + population, lang), stratum_label(stratum, lang), t(_METRIC_KEYS[value], lang)]
        )
        row = {t("col_measure", lang): label}
        for m in models:
            row[model_label(m, lang)] = float(part.loc[m, value])
        if rel is not None:
            num, den = float(part.loc[rel[0], value]), float(part.loc[rel[1], value])
            row[t("col_difference_pct", lang)] = 100 * (num - den) / den
        rows.append(row)
    return pd.DataFrame(rows)


def selection_table(selection_log: pd.DataFrame, lang: str) -> pd.DataFrame:
    """Return the sequential selection log of the internal folds as a localised table.

    Parameters
    ----------
    selection_log : pandas.DataFrame
        The ``selection_log`` table of the ``p1_folds`` run (relative gains are fractions).
    lang : str
        ``'en'`` or ``'pt'``.

    Returns
    -------
    pandas.DataFrame
        One row per candidate; MAEs of both folds, gains in percent and the decision.
    """
    s = selection_log
    out = pd.DataFrame(
        {
            t("col_candidate", lang): s["candidate"],
            t("sel_compared_to", lang): s["compared_to"],
            t("sel_mae_a", lang): s["mae_fold_A"],
            t("sel_mae_a_inc", lang): s["mae_fold_A_incumbent"],
            t("sel_mae_b", lang): s["mae_fold_B"],
            t("sel_mae_b_inc", lang): s["mae_fold_B_incumbent"],
            t("sel_gain_a", lang): 100 * s["rel_gain_A"],
            t("sel_gain_b", lang): 100 * s["rel_gain_B"],
            t("sel_gain_mean", lang): 100 * s["mean_rel_gain"],
            t("sel_decision", lang): [t("decision_" + str(d).replace(" ", "_"), lang) for d in s["decision"]],
        }
    )
    return out.reset_index(drop=True)


def weekly_summary_table(weekly_diff: pd.DataFrame, lang: str) -> pd.DataFrame:
    """Summarise a weekly error-difference series per fold, candidate and reference.

    Parameters
    ----------
    weekly_diff : pandas.DataFrame
        Columns ``fold``, ``candidate``, ``compared_to``, ``WEEK`` and ``mae_diff`` (candidate
        error minus reference error; negative favours the candidate).
    lang : str
        ``'en'`` or ``'pt'``.

    Returns
    -------
    pandas.DataFrame
        Weeks, weeks in which the candidate is better, mean, minimum and maximum difference.
    """
    rows = []
    for (fold, cand, ref), part in weekly_diff.groupby(["fold", "candidate", "compared_to"], sort=False):
        rows.append(
            {
                t("col_fold", lang): _fold_text(fold, lang),
                t("col_candidate", lang): model_label(cand, lang),
                t("col_reference", lang): model_label(ref, lang),
                t("wk_weeks", lang): int(len(part)),
                t("wk_better", lang): int((part["mae_diff"] < 0).sum()),
                t("wk_mean", lang): float(part["mae_diff"].mean()),
                t("wk_min", lang): float(part["mae_diff"].min()),
                t("wk_max", lang): float(part["mae_diff"].max()),
            }
        )
    return pd.DataFrame(rows)


def gpboost_fold_table(metrics: pd.DataFrame, fold_metrics: pd.DataFrame, lang: str) -> pd.DataFrame:
    """Compare C0 with the tuned GPBoost on the internal folds (seen pairs, per-pair MAE).

    Parameters
    ----------
    metrics : pandas.DataFrame
        The ``metrics`` table of the ``p1_folds`` run (C0 rows are used).
    fold_metrics : pandas.DataFrame
        The ``fold_metrics`` table of the ``gpboost_tuning`` run (``fold``, ``population``,
        ``stratum``, ``mae_pair_mean``).
    lang : str
        ``'en'`` or ``'pt'``.

    Returns
    -------
    pandas.DataFrame
        One row per fold with both MAEs and the relative difference of GPBoost in percent.
    """
    c0 = metrics[(metrics["model"] == "C0") & (metrics["population"] == "a_seen") & (metrics["stratum"] == "all")]
    gp = fold_metrics[(fold_metrics["population"] == "seen") & (fold_metrics["stratum"] == "all")]
    merged = c0.merge(gp, on="fold", suffixes=("_c0", "_gpb"))
    return pd.DataFrame(
        {
            t("col_fold", lang): merged["fold"],
            t("gpb_col_c0", lang): merged["mae_pair_mean_c0"],
            t("gpb_col_gpb", lang): merged["mae_pair_mean_gpb"],
            t("gpb_col_diff", lang): 100 * (merged["mae_pair_mean_gpb"] / merged["mae_pair_mean_c0"] - 1),
        }
    )


def coverage_table(coverage: pd.DataFrame, lang: str) -> pd.DataFrame:
    """Return the coverage of the per-pair OLS (and the unseen pairs) as a localised table.

    Parameters
    ----------
    coverage : pandas.DataFrame
        The ``coverage`` table of a P1 run: one row per fold or window, with the counts of seen
        pairs and rows, of pairs with a local OLS and of covered pairs and rows (and their
        percentages), of unseen pairs and rows, and of rows without any history.
    lang : str
        ``'en'`` or ``'pt'``.

    Returns
    -------
    pandas.DataFrame
        One row per fold or window with the counts and percentages under localised headers.
    """
    columns = [
        ("fold", "col_fold"),
        ("n_seen_pairs", "cov_seen_pairs"),
        ("n_seen_rows", "cov_seen_rows"),
        ("n_seen_pairs_with_ols", "cov_pairs_with_ols"),
        ("n_covered_pairs", "cov_covered_pairs"),
        ("n_covered_rows", "cov_covered_rows"),
        ("pct_pairs_covered", "cov_pct_pairs"),
        ("pct_rows_covered", "cov_pct_rows"),
        ("n_unseen_pairs", "cov_unseen_pairs"),
        ("n_unseen_rows", "cov_unseen_rows"),
        ("n_none_rows", "cov_none_rows"),
    ]
    out = pd.DataFrame({t(key, lang): coverage[col] for col, key in columns})
    out[t("col_fold", lang)] = [_fold_text(f, lang) for f in coverage["fold"]]
    return out.reset_index(drop=True)


def term_gap_table(coverage: pd.DataFrame, lang: str) -> pd.DataFrame:
    """Return, per term, the number of seen pairs for which the per-pair OLS has no estimate.

    A term with no variation in the pair's fit window, or dropped by collinearity, counts as not
    estimated (amendment E2); a validation row is uncovered when it needs such a term.

    Parameters
    ----------
    coverage : pandas.DataFrame
        The ``coverage`` table of a P1 run, with the columns ``n_pairs_not_estimated_<TERM>``.
    lang : str
        ``'en'`` or ``'pt'``.

    Returns
    -------
    pandas.DataFrame
        One row per fold or window, one column per term (named after the term).
    """
    prefix = "n_pairs_not_estimated_"
    terms = [c[len(prefix):] for c in coverage.columns if c.startswith(prefix)]
    out = pd.DataFrame({t("col_fold", lang): [_fold_text(f, lang) for f in coverage["fold"]]})
    for term in terms:
        out[term] = coverage[prefix + term].to_numpy()
    return out.reset_index(drop=True)


def discount_slope_lr_table(run: Run, lang: str) -> pd.DataFrame:
    """Present the saved training-only likelihood-ratio test and slope standard deviations."""
    summary = run.load_json("summary")
    draws = run.load_table("bootstrap")
    varcomp = run.load_table("varcomp")
    slope_sd = varcomp.loc[
        varcomp["model"].eq("C1") & varcomp["term"].eq("LOG_DISC"), ["group", "sd"]
    ].set_index("group", verify_integrity=True)["sd"]
    max_simulated = draws.loc[draws["valid"].eq(True), "lr"].max()
    unavailable = t("p_unavailable", lang)
    p_value = summary["p_value"]
    measurements = [
        ("slope_lr_observed", f"{summary['observed_lr']:.3f}"),
        ("slope_lr_max_simulated", f"{max_simulated:.3f}" if pd.notna(max_simulated) else unavailable),
        ("slope_lr_count", f"{summary['valid']}/{summary['planned']}"),
        ("slope_lr_exceedances", str(summary["exceedances"])),
        ("slope_lr_p", format_p(p_value) if p_value is not None else unavailable),
        ("slope_lr_store_sd", f"{slope_sd.loc['STORE_NUM']:.3f}"),
        ("slope_lr_upc_sd", f"{slope_sd.loc['UPC']:.3f}"),
    ]
    return pd.DataFrame(
        {
            t("slope_lr_measure", lang): [t(key, lang) for key, _ in measurements],
            t("slope_lr_value", lang): [value.replace(".", ",") if lang == "pt" else value for _, value in measurements],
        }
    )
