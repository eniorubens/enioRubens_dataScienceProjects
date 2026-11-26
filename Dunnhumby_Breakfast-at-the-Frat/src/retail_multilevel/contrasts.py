"""Planned interaction tests: ``LOG_DISC`` slope differences by segment, private label and category.

Each test is a two-way fixed-effects regression (pair and week effects, by alternating projections as in
:mod:`retail_multilevel.sensitivity`) of ``LOG_UNITS`` on the five within-pair controls plus the
interactions ``LOG_DISC x 1[level == L]`` for every non-reference level ``L`` of one moderator. The joint
null "all slope differences are zero" is tested by a Wald statistic on the interaction coefficients with the
full two-way (pair and week) clustered covariance (Cameron, Gelbach & Miller, 2011), referred to
``F(q, min(G_pair, G_week) - 1)`` (an approximation). A test is declared unavailable, with a reason, when the
within design loses rank or the ``q x q`` covariance block is not positive definite or is ill-conditioned.
The family of the three tests is adjusted by Holm (1979).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
from scipy.stats import t as t_dist

from retail_multilevel.sensitivity import FE_REGRESSORS, fe_fit_arrays
from retail_multilevel.split import FIRST_TEST_WEEK

BASE_REGRESSORS = list(FE_REGRESSORS)
VARIATION_TOL = 1e-12  # a pair has discount variation when the range of LOG_DISC exceeds this
COND_MAX = 1e12  # the q x q covariance block is declared ill-conditioned above this condition number


@dataclass(frozen=True)
class ContrastSpec:
    """Pre-registered definition of one planned interaction test.

    Attributes
    ----------
    name : str
        Test identifier (``segment``, ``private_label`` or ``category``).
    column : str
        Moderator column in the panel.
    reference : object
        Reference level (a value of ``column``).
    drop_ambiguous : bool
        Whether rows flagged ``SEG_AMBIGUOUS`` are excluded (segment test only).
    """

    name: str
    column: str
    reference: object
    drop_ambiguous: bool = False


SPECS = [
    ContrastSpec("segment", "SEGMENT", "MAINSTREAM", drop_ambiguous=True),
    ContrastSpec("private_label", "PRIVATE_LABEL", False),
    ContrastSpec("category", "CATEGORY", "BAG SNACKS"),
]


def assert_training_only(df: pd.DataFrame, first_test_week: int = FIRST_TEST_WEEK) -> None:
    """Raise ``ValueError`` if any row belongs to the test period (``WEEK >= first_test_week``)."""
    if len(df) and int(df["WEEK"].max()) >= first_test_week:
        raise ValueError(f"rows from week {int(df['WEEK'].max())} found; training uses WEEK < {first_test_week}")


def prepare_frame(train: pd.DataFrame) -> pd.DataFrame:
    """Add ``LOG_DISC = LOG_PRICE - LOG_BASE_PRICE`` and integer promotion flags to a training frame."""
    assert_training_only(train)
    out = train.copy()
    out["LOG_DISC"] = out["LOG_PRICE"] - out["LOG_BASE_PRICE"]
    for col in ("FEATURE", "DISPLAY", "TPR_ONLY"):
        out[col] = out[col].astype(int)
    return out


def build_population(df: pd.DataFrame, spec: ContrastSpec) -> tuple[pd.DataFrame, dict]:
    """Select the rows of one test and count every exclusion.

    Parameters
    ----------
    df : pandas.DataFrame
        Training rows.
    spec : ContrastSpec
        The test definition.

    Returns
    -------
    tuple
        The population frame and a dict with ``test``, ``n_rows_input``, ``n_rows``, ``n_pairs``,
        ``n_excluded`` and ``reason`` (a text listing each exclusion cause and its count).
    """
    assert_training_only(df)
    keep = pd.Series(True, index=df.index)
    reasons = []
    if spec.drop_ambiguous:
        amb = df["SEG_AMBIGUOUS"].fillna(False).astype(bool)
        reasons.append(f"SEG_AMBIGUOUS={int(amb.sum())}")
        keep &= ~amb
    missing = df[spec.column].isna()
    reasons.append(f"missing {spec.column}={int((missing & keep).sum())}")
    keep &= ~missing
    pop = df.loc[keep]
    info = {
        "test": spec.name,
        "n_rows_input": int(len(df)),
        "n_rows": int(len(pop)),
        "n_pairs": int(pop["PAIR_ID"].nunique()),
        "n_excluded": int(len(df) - len(pop)),
        "reason": "; ".join(reasons),
    }
    return pop, info


def moderator_levels(pop: pd.DataFrame, column: str, reference) -> list:
    """Observed levels of the moderator, sorted, with the reference required to be among them."""
    levels = sorted(pop[column].unique().tolist())
    if reference not in levels:
        raise ValueError(f"reference {reference!r} not observed in {column}: {levels}")
    return levels


def interaction_name(column: str, level) -> str:
    """Name of the interaction term ``LOG_DISC x 1[column == level]``."""
    return f"LOG_DISC:{column}[{level}]"


def build_design(
    pop: pd.DataFrame, column: str, reference, base: list[str] | None = None
) -> tuple[np.ndarray, list[str], list[str], list]:
    """Regressor matrix with the base controls and one interaction per non-reference level.

    Returns
    -------
    tuple
        The matrix (``n x (5 + q)``), the term names, the names of the ``q`` interaction terms and the
        non-reference levels (same order).
    """
    base = list(BASE_REGRESSORS if base is None else base)
    levels = moderator_levels(pop, column, reference)
    others = [lv for lv in levels if lv != reference]
    disc = pop["LOG_DISC"].to_numpy(dtype=float)
    cols = [pop[c].to_numpy(dtype=float) for c in base]
    inter_names = []
    for lv in others:
        cols.append(disc * (pop[column].to_numpy() == lv))
        inter_names.append(interaction_name(column, lv))
    return np.column_stack(cols), [*base, *inter_names], inter_names, others


def level_support(pop: pd.DataFrame, column: str) -> pd.DataFrame:
    """Support by level: rows, pairs, pooled within-pair sd of ``LOG_DISC`` and pairs with variation."""
    rows = []
    for level, g in pop.groupby(column, sort=True):
        by_pair = g.groupby("PAIR_ID")["LOG_DISC"]
        dev = g["LOG_DISC"] - by_pair.transform("mean")
        n_pairs = by_pair.ngroups
        ptp = by_pair.max() - by_pair.min()
        dof = len(g) - n_pairs
        rows.append(
            {
                "level": level,
                "n_rows": int(len(g)),
                "n_pairs": int(n_pairs),
                "within_pair_sd_log_disc": float(np.sqrt((dev**2).sum() / dof)) if dof > 0 else float("nan"),
                "n_pairs_with_variation": int((ptp > VARIATION_TOL).sum()),
            }
        )
    return pd.DataFrame(rows)


def _combo_se(cov: np.ndarray, vec: np.ndarray) -> float:
    nz = np.flatnonzero(vec)  # only the terms in the combination (others may be NaN when dropped)
    return float(np.sqrt(max(vec[nz] @ cov[np.ix_(nz, nz)] @ vec[nz], 0.0)))


def wald_joint(delta: np.ndarray, cov_block: np.ndarray, g_pair: int, g_week: int) -> dict:
    """Joint Wald test ``delta' V^-1 delta`` with the ``F(q, min(G) - 1)`` reference.

    Parameters
    ----------
    delta : numpy.ndarray
        The ``q`` interaction coefficients.
    cov_block : numpy.ndarray
        Their ``q x q`` two-way covariance.
    g_pair, g_week : int
        Numbers of pair and week clusters.

    Returns
    -------
    dict
        ``q, W, F, df1, df2, p_raw, available, reason, cond_number, min_eig``. ``available`` is False (and the
        statistic NaN) if the block has non-finite entries, a non-positive smallest eigenvalue, a condition
        number above :data:`COND_MAX` or ``df2 < 1``.
    """
    delta = np.asarray(delta, dtype=float)
    q = len(delta)
    df2 = int(min(g_pair, g_week) - 1)
    out = {"q": q, "W": np.nan, "F": np.nan, "df1": q, "df2": df2, "p_raw": np.nan, "available": False,
           "reason": "", "cond_number": np.nan, "min_eig": np.nan}
    if not (np.isfinite(delta).all() and np.isfinite(cov_block).all()):
        out["reason"] = "interaction coefficients or covariance not estimable (rank-deficient design)"
        return out
    eig = np.linalg.eigvalsh((cov_block + cov_block.T) / 2)
    out["min_eig"] = float(eig.min())
    out["cond_number"] = float(eig.max() / eig.min()) if eig.min() > 0 else float("inf")
    if eig.min() <= 0:
        out["reason"] = "two-way covariance block is not positive definite"
    elif out["cond_number"] > COND_MAX:
        out["reason"] = f"covariance block ill-conditioned (cond > {COND_MAX:.0e})"
    elif df2 < 1:
        out["reason"] = "fewer than two clusters in a dimension"
    else:
        w = float(delta @ np.linalg.solve(cov_block, delta))
        out.update(W=w, F=w / q, p_raw=float(f_dist.sf(w / q, q, df2)), available=True)
    return out


def fit_contrast(pop: pd.DataFrame, spec: ContrastSpec, tol: float = 1e-10) -> dict:
    """Fit the restricted and interaction models of one test and compute slopes and the joint Wald test.

    Parameters
    ----------
    pop : pandas.DataFrame
        Population from :func:`build_population`, prepared with :func:`prepare_frame`.
    spec : ContrastSpec
        The test definition.
    tol : float
        Convergence tolerance of the demeaning.

    Returns
    -------
    dict
        ``coefficients`` (test, model, term, estimate, se_pair, se_week, se_twoway), ``slopes`` (per level,
        pointwise intervals), ``support``, ``wald`` (a dict, see :func:`wald_joint`, plus diagnostics),
        and ``levels``.
    """
    y = pop["LOG_UNITS"].to_numpy(dtype=float)
    pair, week = pop["PAIR_ID"].to_numpy(), pop["WEEK"].to_numpy()
    x, terms, inter_names, others = build_design(pop, spec.column, spec.reference)
    levels = moderator_levels(pop, spec.column, spec.reference)
    k0 = len(BASE_REGRESSORS)
    restricted = fe_fit_arrays(y, x[:, :k0], pair, week, tol=tol, rank_check=True)
    full = fe_fit_arrays(y, x, pair, week, tol=tol, rank_check=True)

    def coef_rows(fit: dict, names: list[str], model: str) -> list[dict]:
        se = {key: np.sqrt(np.diag(fit[f"cov_{key}"])) for key in ("pair", "week", "twoway")}
        return [{"test": spec.name, "model": model, "term": t, "estimate": float(fit["beta"][i]),
                 "se_pair": float(se["pair"][i]), "se_week": float(se["week"][i]),
                 "se_twoway": float(se["twoway"][i])} for i, t in enumerate(names)]

    coefficients = pd.DataFrame(
        coef_rows(restricted, terms[:k0], "restricted") + coef_rows(full, terms, "interaction")
    )
    g_pair, g_week = full["n_pairs"], full["n_weeks"]
    t_crit = float(t_dist.ppf(0.975, min(g_pair, g_week) - 1))
    k_all = len(terms)
    i_disc = terms.index("LOG_DISC")
    beta, cov2, cov_p, cov_w = full["beta"], full["cov_twoway"], full["cov_pair"], full["cov_week"]
    slope_rows = []
    for lv in levels:
        vec = np.zeros(k_all)
        vec[i_disc] = 1.0
        diff = np.zeros(k_all)
        if lv != spec.reference:
            j = terms.index(interaction_name(spec.column, lv))
            vec[j] = 1.0
            diff[j] = 1.0
        slope = float(vec[np.flatnonzero(vec)] @ beta[np.flatnonzero(vec)])
        se2 = _combo_se(cov2, vec)
        d = float(diff[np.flatnonzero(diff)] @ beta[np.flatnonzero(diff)])
        se_d = _combo_se(cov2, diff)
        slope_rows.append(
            {"test": spec.name, "level": lv, "is_reference": lv == spec.reference, "slope": slope,
             "se_pair": _combo_se(cov_p, vec), "se_week": _combo_se(cov_w, vec), "se_twoway": se2,
             "ci_low": slope - t_crit * se2, "ci_high": slope + t_crit * se2, "interval": "pointwise",
             "diff_vs_ref": d, "se_diff": se_d, "diff_ci_low": d - t_crit * se_d,
             "diff_ci_high": d + t_crit * se_d}
        )
    idx = [terms.index(n) for n in inter_names]
    wald = wald_joint(beta[idx], cov2[np.ix_(idx, idx)], g_pair, g_week)
    if not wald["available"] and full["rank_deficient"]:
        zero = [terms[i] for i in full["dropped"]]
        wald["reason"] = (
            f"design rank-deficient (no within-pair variation: {zero})" if zero
            else f"design rank {full['design_rank']} < {len(terms)}"
        )
    wald.update(
        test=spec.name, moderator=spec.column, reference=str(spec.reference), G_pair=g_pair, G_week=g_week,
        design_rank=full["design_rank"], n_terms=len(terms), rank_deficient=full["rank_deficient"],
        twoway_psd_fixed=full["psd_fixed"], design_cond=full["design_cond"], nobs=full["nobs"],
        nobs_restricted=restricted["nobs"],
    )
    support = level_support(pop, spec.column)
    support.insert(0, "test", spec.name)
    return {"coefficients": coefficients, "slopes": pd.DataFrame(slope_rows), "support": support,
            "wald": wald, "levels": levels}


def holm(pvals, available, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Holm (1979) step-down adjustment over a fixed family.

    Unavailable positions enter with ``p = 1`` (the family keeps its size) and are reported as NaN adjusted
    p-values that never reject.

    Parameters
    ----------
    pvals : array-like
        Raw p-values (entries at unavailable positions are ignored).
    available : array-like of bool
        Whether each test is available.
    alpha : float
        Family-wise level.

    Returns
    -------
    tuple
        Adjusted p-values (NaN where unavailable) and the boolean rejection flags.
    """
    p = np.asarray(pvals, dtype=float).copy()
    avail = np.asarray(available, dtype=bool)
    m = len(p)
    p[~avail] = 1.0
    order = np.argsort(p, kind="stable")
    adj_sorted = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    adj = np.empty(m)
    adj[order] = adj_sorted
    reject = (adj <= alpha) & avail
    adj[~avail] = np.nan
    return adj, reject
