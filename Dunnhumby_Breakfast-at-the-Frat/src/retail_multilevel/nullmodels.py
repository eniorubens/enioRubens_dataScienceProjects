"""Null (intercept-only) models: variance decomposition of ``LOG_UNITS`` across stores and products.

Step-up sequence M0..M4 with the same fixed part (``y ~ 1``), all REML, so that the likelihoods
are comparable in likelihood-ratio tests. Fitting needs R (lme4 through rpy2); the statsmodels
cross-check validates lme4's store-only model on the same data.
"""

from __future__ import annotations

import pandas as pd

from retail_multilevel.multilevel import (
    LmerFit,
    fit_lm,
    fit_lmer,
    fits_table,
    icc,
    lr_table,
    ranef_frame,
    to_r_frame,
)

NULL_MODELS: list[tuple[str, str]] = [
    ("M0_ols", "LOG_UNITS ~ 1"),
    ("M1_store", "LOG_UNITS ~ 1 + (1 | STORE_NUM)"),
    ("M2_product", "LOG_UNITS ~ 1 + (1 | UPC)"),
    ("M3_crossed", "LOG_UNITS ~ 1 + (1 | STORE_NUM) + (1 | UPC)"),
    ("M4_crossed_pair", "LOG_UNITS ~ 1 + (1 | STORE_NUM) + (1 | UPC) + (1 | STORE_NUM:UPC)"),
]

LR_PAIRS: list[tuple[str, str]] = [
    ("M0_ols", "M1_store"),
    ("M0_ols", "M2_product"),
    ("M1_store", "M3_crossed"),
    ("M2_product", "M3_crossed"),
    ("M3_crossed", "M4_crossed_pair"),
]

DEFAULT_CONTROL = 'lmerControl(optimizer="bobyqa")'


def _stack(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = [d.assign(model=name) for name, d in frames.items()]
    out = pd.concat(parts, ignore_index=True)
    return out[["model", *[c for c in out.columns if c != "model"]]]


def _formula(formula: str, y: str, store: str, product: str) -> str:
    return formula.replace("LOG_UNITS", y).replace("STORE_NUM", store).replace("UPC", product)


def fit_null_models(
    df: pd.DataFrame,
    y: str = "LOG_UNITS",
    store: str = "STORE_NUM",
    product: str = "UPC",
    control: str | None = DEFAULT_CONTROL,
) -> dict[str, pd.DataFrame]:
    """Fit the five null models (M0 OLS, M1 store, M2 product, M3 crossed, M4 crossed + pair).

    Parameters
    ----------
    df : pandas.DataFrame
        Data with columns ``y``, ``store`` and ``product``.
    y, store, product : str
        Column names; the formulas of :data:`NULL_MODELS` are rewritten with them.
    control : str, optional
        R expression for ``lmerControl`` (default bobyqa).

    Returns
    -------
    dict of pandas.DataFrame
        ``fits``, ``varcomp``, ``icc``, ``lr``, ``ranef_store``, ``ranef_product``, ``ranef_pair``
        and ``fixef`` (stacked tables carry a leading ``model`` column).
    """
    r_df = to_r_frame(df[[y, store, product]], [store, product])
    fits: list[LmerFit] = []
    for name, formula in NULL_MODELS:
        formula = _formula(formula, y, store, product)
        if name == "M0_ols":
            fits.append(fit_lm(r_df, formula, name))
        else:
            fits.append(fit_lmer(r_df, formula, name, reml=True, control=control))
    by_name = {f.name: f for f in fits}
    mixed = {n: f for n, f in by_name.items() if n != "M0_ols"}
    m4 = by_name["M4_crossed_pair"]
    return {
        "fits": fits_table(fits),
        "varcomp": _stack({n: f.varcomp for n, f in by_name.items()}),
        "icc": _stack({n: icc(f.varcomp) for n, f in mixed.items()}),
        "lr": lr_table(fits, LR_PAIRS),
        "ranef_store": ranef_frame(m4, store),
        "ranef_product": ranef_frame(m4, product),
        "ranef_pair": ranef_frame(m4, f"{store}:{product}"),
        "fixef": _stack({n: f.fixef for n, f in by_name.items()}),
    }


def statsmodels_crosscheck(df: pd.DataFrame, y: str = "LOG_UNITS", group: str = "STORE_NUM") -> pd.DataFrame:
    """Fit the store-only random-intercept model with statsmodels MixedLM (REML).

    Returns
    -------
    pandas.DataFrame
        Columns ``engine, component, variance`` with the group variance and the residual variance.
    """
    import statsmodels.formula.api as smf

    data = df[[y, group]].copy()
    data[group] = data[group].astype(str)
    res = smf.mixedlm(f"{y} ~ 1", data=data, groups=data[group]).fit(reml=True)
    return pd.DataFrame(
        [
            ("statsmodels", group, float(res.cov_re.iloc[0, 0])),
            ("statsmodels", "Residual", float(res.scale)),
        ],
        columns=["engine", "component", "variance"],
    )


def crosscheck_table(sm: pd.DataFrame, varcomp: pd.DataFrame) -> pd.DataFrame:
    """Compare the statsmodels variances with lme4's ``M1_store`` ones.

    Parameters
    ----------
    sm : pandas.DataFrame
        Output of :func:`statsmodels_crosscheck`.
    varcomp : pandas.DataFrame
        The stacked ``varcomp`` table of :func:`fit_null_models` (leading ``model`` column).

    Returns
    -------
    pandas.DataFrame
        Columns ``component, statsmodels, lme4, rel_diff`` (relative to lme4).
    """
    m1 = varcomp[varcomp["model"] == "M1_store"]
    m1 = m1[(m1["term"] == "(Intercept)") | (m1["group"] == "Residual")]
    lme4 = m1.rename(columns={"group": "component", "variance": "lme4"})[["component", "lme4"]]
    sm_ = sm.rename(columns={"variance": "statsmodels"})[["component", "statsmodels"]]
    out = sm_.merge(lme4, on="component", how="inner")
    out["rel_diff"] = (out["statsmodels"] - out["lme4"]).abs() / out["lme4"]
    return out[["component", "statsmodels", "lme4", "rel_diff"]]
