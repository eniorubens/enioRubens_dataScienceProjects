"""Linear mixed models (lme4 through rpy2): fitting wrappers, ICC, likelihood-ratio tests, BLUPs.

rpy2 is imported lazily (always through :func:`retail_multilevel.rbridge.get_robjects`), so this
module imports fine where R is not available; only the fitting/extraction functions need R.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import pandas as pd
from scipy.stats import chi2

_VARCOMP_COLS = ["group", "term", "variance", "sd"]
_FIXEF_COLS = ["term", "estimate", "std_error"]


@dataclass
class LmerFit:
    """Summary of a fitted (mixed) linear model plus a handle to the R object."""

    name: str
    formula: str
    reml: bool
    nobs: int
    npar: int
    loglik: float
    aic: float
    bic: float
    varcomp: pd.DataFrame
    fixef: pd.DataFrame
    messages: list[str] = field(default_factory=list)
    r_model: object = field(default=None, repr=False)
    seconds: float = 0.0

    def summary_row(self) -> dict:
        """Return a flat dict of the headline statistics of the fit."""
        return {
            "name": self.name,
            "formula": self.formula,
            "reml": self.reml,
            "nobs": self.nobs,
            "npar": self.npar,
            "loglik": self.loglik,
            "aic": self.aic,
            "bic": self.bic,
            "n_messages": len(self.messages),
            "seconds": self.seconds,
        }


def _ro():
    from retail_multilevel.rbridge import get_robjects

    return get_robjects()


def to_r_frame(df: pd.DataFrame, factors: Sequence[str]):
    """Convert a pandas frame to an R data.frame, turning ``factors`` into R factors.

    Parameters
    ----------
    df : pandas.DataFrame
        Source data.
    factors : sequence of str
        Columns (present in ``df``) to convert to R factors; they are cast to ``str`` first.

    Returns
    -------
    rpy2 data.frame
    """
    ro = _ro()
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    missing = [c for c in factors if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not in df: {missing}")
    work = df.copy()
    for col in factors:
        work[col] = work[col].astype(str)
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df = ro.conversion.get_conversion().py2rpy(work)
    if factors:
        r_df = ro.r("function(d, cols) { for (c in cols) d[[c]] <- factor(d[[c]]); d }")(
            r_df, ro.StrVector(list(factors))
        )
    return r_df


def _varcomp_from_r(ro, model) -> pd.DataFrame:
    res = ro.r(
        """function(m) {
          v <- as.data.frame(VarCorr(m))
          v <- v[is.na(v$var2), , drop = FALSE]
          term <- ifelse(is.na(v$var1), "", v$var1)
          list(group = as.character(v$grp), term = term, variance = v$vcov, sd = v$sdcor)
        }"""
    )(model)
    d = {k: list(res.rx2(k)) for k in _VARCOMP_COLS}
    return pd.DataFrame(d)[_VARCOMP_COLS].astype({"variance": float, "sd": float})


def _fixef_from_r(ro, model) -> pd.DataFrame:
    res = ro.r(
        """function(m) {
          cf <- coef(summary(m))
          list(term = rownames(cf), estimate = cf[, 1], std_error = cf[, 2])
        }"""
    )(model)
    d = {k: list(res.rx2(k)) for k in _FIXEF_COLS}
    return pd.DataFrame(d)[_FIXEF_COLS].astype({"estimate": float, "std_error": float})


def fit_lmer(r_df, formula: str, name: str, reml: bool = True, control: str | None = None) -> LmerFit:
    """Fit a linear mixed model with ``lme4::lmer``.

    Parameters
    ----------
    r_df : rpy2 data.frame
        Data, e.g. from :func:`to_r_frame`.
    formula : str
        lme4 formula.
    name : str
        Label for the fit.
    reml : bool
        REML (default) or ML.
    control : str, optional
        R expression such as ``'lmerControl(optimizer="bobyqa")'`` passed as ``control``.

    Returns
    -------
    LmerFit
    """
    ro = _ro()
    from rpy2.robjects.packages import importr

    importr("lme4")
    fitter = ro.r(
        """function(formula, data, reml, control) {
          ctrl <- if (is.null(control)) lme4::lmerControl() else eval(parse(text = control), asNamespace("lme4"))
          lme4::lmer(as.formula(formula), data = data, REML = reml, control = ctrl)
        }"""
    )
    t0 = time.perf_counter()
    model = fitter(formula, r_df, bool(reml), ro.NULL if control is None else control)
    seconds = time.perf_counter() - t0

    stats = ro.r(
        """function(m) {
          ll <- logLik(m)
          msgs <- m@optinfo$conv$lme4$messages
          if (is.null(msgs)) msgs <- character(0)
          if (isSingular(m)) msgs <- c(msgs, "singular fit")
          list(nobs = nobs(m), npar = attr(ll, "df"), loglik = as.numeric(ll),
               aic = AIC(m), bic = BIC(m), msgs = as.character(msgs))
        }"""
    )(model)
    return LmerFit(
        name=name,
        formula=formula,
        reml=bool(reml),
        nobs=int(stats.rx2("nobs")[0]),
        npar=int(stats.rx2("npar")[0]),
        loglik=float(stats.rx2("loglik")[0]),
        aic=float(stats.rx2("aic")[0]),
        bic=float(stats.rx2("bic")[0]),
        varcomp=_varcomp_from_r(ro, model),
        fixef=_fixef_from_r(ro, model),
        messages=[str(s) for s in stats.rx2("msgs")],
        r_model=model,
        seconds=seconds,
    )


def fit_lm(r_df, formula: str, name: str, reml: bool = True) -> LmerFit:
    """Fit the single-level OLS counterpart (``stats::lm``) packaged as an :class:`LmerFit`.

    The log-likelihood is ``logLik(m, REML=reml)`` so the fit can enter likelihood-ratio tests
    against mixed models fitted with the same ``reml`` flag. AIC/BIC are recomputed from it.
    """
    ro = _ro()
    fitter = ro.r("function(formula, data) stats::lm(as.formula(formula), data = data)")
    t0 = time.perf_counter()
    model = fitter(formula, r_df)
    seconds = time.perf_counter() - t0
    stats = ro.r(
        """function(m, reml) {
          ll <- logLik(m, REML = reml)
          list(nobs = nobs(m), npar = attr(ll, "df"), loglik = as.numeric(ll), sigma = summary(m)$sigma)
        }"""
    )(model, bool(reml))
    nobs = int(stats.rx2("nobs")[0])
    npar = int(stats.rx2("npar")[0])
    ll = float(stats.rx2("loglik")[0])
    sigma = float(stats.rx2("sigma")[0])
    varcomp = pd.DataFrame({"group": ["Residual"], "term": [""], "variance": [sigma**2], "sd": [sigma]})
    return LmerFit(
        name=name,
        formula=formula,
        reml=bool(reml),
        nobs=nobs,
        npar=npar,
        loglik=ll,
        aic=-2 * ll + 2 * npar,
        bic=-2 * ll + math.log(nobs) * npar,
        varcomp=varcomp[_VARCOMP_COLS],
        fixef=_fixef_from_r(ro, model),
        messages=[],
        r_model=model,
        seconds=seconds,
    )


def icc(varcomp: pd.DataFrame) -> pd.DataFrame:
    """Variance shares of the random-intercept components plus the residual.

    The intraclass correlation is defined for random-intercept components only: rows whose
    ``term`` is not ``"(Intercept)"`` (random slopes) are ignored, as is anything but the
    Residual row. ``share`` is each variance divided by the sum of the kept variances.

    Parameters
    ----------
    varcomp : pandas.DataFrame
        Columns ``group, term, variance, sd`` as in :attr:`LmerFit.varcomp`.

    Returns
    -------
    pandas.DataFrame
        Columns ``group, variance, share``.
    """
    keep = varcomp[(varcomp["term"] == "(Intercept)") | (varcomp["group"] == "Residual")]
    out = keep[["group", "variance"]].reset_index(drop=True).copy()
    out["share"] = out["variance"] / out["variance"].sum()
    return out


def lr_test(restricted: LmerFit, full: LmerFit, boundary: bool = True) -> dict:
    """Likelihood-ratio test of a restricted model against a fuller one.

    ``stat = 2 * (full.loglik - restricted.loglik)`` (clipped at 0) and ``df = full.npar -
    restricted.npar``. With ``boundary=True`` the p-value accounts for variance components being
    tested on the boundary of the parameter space: for ``df == 1`` a 50:50 mixture of chi2(0) and
    chi2(1) (Self & Liang, 1987), i.e. ``0.5 * sf(stat, 1)``; for ``df > 1`` the common
    approximation ``0.5 * sf(stat, df - 1) + 0.5 * sf(stat, df)``. Otherwise the naive chi2(df)
    p-value is used.

    Comparing REML likelihoods is only valid when the fixed effects are identical in both models
    (the caller's responsibility); use ML fits (``reml=False``) when the fixed effects differ.

    Returns
    -------
    dict
        Keys ``restricted, full, stat, df, p_value, p_naive``.

    Raises
    ------
    ValueError
        If the ``reml`` flags or ``nobs`` differ, or ``df < 1``.
    """
    if restricted.reml != full.reml:
        raise ValueError("Cannot compare a REML fit with an ML fit.")
    if restricted.nobs != full.nobs:
        raise ValueError(f"Different number of observations ({restricted.nobs} vs {full.nobs}).")
    df = full.npar - restricted.npar
    if df < 1:
        raise ValueError(f"Full model must have more parameters than the restricted one (df={df}).")
    stat = max(0.0, 2 * (full.loglik - restricted.loglik))
    p_naive = float(chi2.sf(stat, df))
    if boundary and df == 1:
        p_value = 0.5 * float(chi2.sf(stat, 1))
    elif boundary:
        p_value = 0.5 * float(chi2.sf(stat, df - 1)) + 0.5 * float(chi2.sf(stat, df))
    else:
        p_value = p_naive
    return {
        "restricted": restricted.name,
        "full": full.name,
        "stat": stat,
        "df": df,
        "p_value": p_value,
        "p_naive": p_naive,
    }


def lr_table(fits: Sequence[LmerFit], pairs: Sequence[tuple[str, str]]) -> pd.DataFrame:
    """Run :func:`lr_test` for each ``(restricted_name, full_name)`` pair of ``fits``.

    Returns
    -------
    pandas.DataFrame
        Columns ``restricted, full, stat, df, p_value, p_naive``.
    """
    by_name = {f.name: f for f in fits}
    rows = [lr_test(by_name[r], by_name[f]) for r, f in pairs]
    return pd.DataFrame(rows, columns=["restricted", "full", "stat", "df", "p_value", "p_naive"])


def ranef_frame(fit: LmerFit, group: str) -> pd.DataFrame:
    """Conditional modes (BLUPs) and conditional SDs of one grouping factor.

    Returns
    -------
    pandas.DataFrame
        Columns ``level, term, blup, cond_sd``.
    """
    ro = _ro()
    res = ro.r(
        """function(m, g) {
          d <- as.data.frame(ranef(m, condVar = TRUE))
          d <- d[as.character(d$grpvar) == g, , drop = FALSE]
          list(level = as.character(d$grp), term = as.character(d$term),
               blup = d$condval, cond_sd = d$condsd)
        }"""
    )(fit.r_model, group)
    d = {k: list(res.rx2(k)) for k in ("level", "term", "blup", "cond_sd")}
    return pd.DataFrame(d).astype({"level": str, "blup": float, "cond_sd": float})


def fits_table(fits: Sequence[LmerFit]) -> pd.DataFrame:
    """Stack :meth:`LmerFit.summary_row` of each fit into a frame."""
    return pd.DataFrame([f.summary_row() for f in fits])
