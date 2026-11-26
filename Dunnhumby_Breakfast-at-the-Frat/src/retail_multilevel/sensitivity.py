"""Within-pair sensitivity: two-way fixed effects (pair and week) by alternating demeaning.

The audit asks for the "intra-pair" reading of discount depth and promotion flags. A regression with
one effect per store x UPC pair and one per week is estimated without building dummy matrices: the
outcome and the regressors are projected off both sets of effects by alternating means (the method of
alternating projections), and OLS is run on the residualised data (Frisch-Waugh-Lovell). Standard
errors are clustered by pair (CR1) and, in two ways, by pair and week (Cameron-Gelbach-Miller).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import t as t_dist

FE_REGRESSORS = ["LOG_DISC", "LOG_BASE_PRICE", "FEATURE", "DISPLAY", "TPR_ONLY"]


def _group_codes(values) -> tuple[np.ndarray, int]:
    codes, uniques = pd.factorize(pd.Series(values), sort=False)
    return codes.astype(np.int64), len(uniques)


def demean_two_way(
    x: np.ndarray, pair: np.ndarray, week: np.ndarray, tol: float = 1e-10, max_iter: int = 10_000
) -> tuple[np.ndarray, int]:
    """Remove pair and week means from the columns of ``x`` by alternating projections.

    Parameters
    ----------
    x : numpy.ndarray
        Array of shape ``(n,)`` or ``(n, k)``.
    pair, week : numpy.ndarray
        Group labels (any hashable dtype) of length ``n``.
    tol : float
        Stop when the largest absolute change of any element in a full sweep is below ``tol``.
    max_iter : int
        Maximum number of sweeps; a ``RuntimeError`` is raised when it is reached.

    Returns
    -------
    tuple
        The demeaned array (same shape as ``x``) and the number of sweeps.
    """
    arr = np.asarray(x, dtype=float)
    out = arr.reshape(len(arr), -1).copy()
    p, n_p = _group_codes(pair)
    w, n_w = _group_codes(week)
    cnt_p = np.bincount(p, minlength=n_p).astype(float)
    cnt_w = np.bincount(w, minlength=n_w).astype(float)
    for it in range(1, max_iter + 1):
        prev = out.copy()
        for codes, cnt, n_g in ((p, cnt_p, n_p), (w, cnt_w, n_w)):
            for j in range(out.shape[1]):
                out[:, j] -= (np.bincount(codes, weights=out[:, j], minlength=n_g) / cnt)[codes]
        if np.abs(out - prev).max() < tol:
            return out.reshape(arr.shape), it
    raise RuntimeError(f"alternating projections did not converge in {max_iter} sweeps")


def fe_fit_arrays(
    y: np.ndarray,
    x: np.ndarray,
    pair: np.ndarray,
    week: np.ndarray,
    tol: float = 1e-10,
    rank_check: bool = False,
) -> dict:
    """Two-way fixed-effects OLS on arrays, returning the coefficients and the full covariance matrices.

    This is the engine of :func:`two_way_fe_ols`: the same alternating-projection demeaning and the same
    pair, week and two-way (Cameron-Gelbach-Miller) covariances, exposed as full ``k x k`` matrices.

    Parameters
    ----------
    y : numpy.ndarray
        Outcome, shape ``(n,)``.
    x : numpy.ndarray
        Regressors, shape ``(n, k)`` (no constant: the effects absorb it).
    pair, week : numpy.ndarray
        Group labels of the two sets of effects; the pair groups are also the one-way clusters.
    tol : float
        Convergence tolerance of :func:`demean_two_way`.
    rank_check : bool
        When True the rank of the demeaned design is checked: columns with (numerically) zero within
        variation are dropped from the fit (their estimates and covariances are NaN) and a pseudo-inverse is
        used if the remaining columns are still collinear. When False (the default) the plain inverse is used.

    Returns
    -------
    dict
        ``beta`` (k,), ``cov_pair``, ``cov_week``, ``cov_twoway`` (psd-fixed), ``cov_twoway_raw`` (before the
        eigenvalue fix), ``psd_fixed``, ``n_sweeps``, ``nobs``, ``n_pairs``, ``n_weeks``, ``design_rank``,
        ``rank_deficient``, ``dropped`` (indices of zero-variation columns), ``design_cond`` (condition number
        of the column-standardised cross-product) and ``design_min_eig``.
    """
    xy = np.column_stack([np.asarray(y, dtype=float), np.asarray(x, dtype=float)])
    dm, sweeps = demean_two_way(xy, pair, week, tol=tol)
    yd, xd_all = dm[:, 0], dm[:, 1:]
    n, k_all = xd_all.shape
    g_codes, n_g = _group_codes(pair)
    w_codes, n_weeks = _group_codes(week)
    diag = (xd_all**2).sum(axis=0)
    keep = np.flatnonzero(diag > 1e-9) if rank_check else np.arange(k_all)
    dropped = np.setdiff1d(np.arange(k_all), keep)
    xd = xd_all[:, keep]
    k = xd.shape[1]
    xtx = xd.T @ xd
    scale = 1.0 / np.sqrt(np.diag(xtx))
    corr_eig = np.linalg.eigvalsh(xtx * scale[:, None] * scale[None, :]) if rank_check else np.array([np.nan])
    design_rank = int((corr_eig > 1e-10).sum()) if rank_check else k
    rank_deficient = bool(len(dropped) > 0 or (rank_check and design_rank < k))
    xtx_inv = np.linalg.pinv(xtx) if (rank_check and design_rank < k) else np.linalg.inv(xtx)
    beta_k = xtx_inv @ (xd.T @ yd)
    u = yd - xd @ beta_k
    s_row = xd * u[:, None]
    scores = np.zeros((n_g, k))
    scores_w = np.zeros((n_weeks, k))
    for j in range(k):
        scores[:, j] = np.bincount(g_codes, weights=s_row[:, j], minlength=n_g)
        scores_w[:, j] = np.bincount(w_codes, weights=s_row[:, j], minlength=n_weeks)
    meat_p, meat_w, meat_r = scores.T @ scores, scores_w.T @ scores_w, s_row.T @ s_row
    dof = n - k - (n_weeks - 1)
    base = (n - 1) / dof
    c_p = (n_g / (n_g - 1)) * base
    c_w = (n_weeks / (n_weeks - 1)) * base
    c_r = (n / (n - 1)) * base
    cov = c_p * xtx_inv @ meat_p @ xtx_inv
    cov_w = c_w * xtx_inv @ meat_w @ xtx_inv
    meat_two = c_p * meat_p + c_w * meat_w - c_r * meat_r
    cov_two_raw = xtx_inv @ meat_two @ xtx_inv
    cov_two_raw = (cov_two_raw + cov_two_raw.T) / 2
    eigval, eigvec = np.linalg.eigh(cov_two_raw)
    psd_fixed = bool(eigval.min() < 0)
    cov_two = (eigvec * np.clip(eigval, 0.0, None)) @ eigvec.T if psd_fixed else cov_two_raw

    def embed(mat: np.ndarray) -> np.ndarray:
        out = np.full((k_all, k_all), np.nan)
        out[np.ix_(keep, keep)] = mat
        return out

    beta = np.full(k_all, np.nan)
    beta[keep] = beta_k
    full = k == k_all
    return {
        "beta": beta,
        "cov_pair": cov if full else embed(cov),
        "cov_week": cov_w if full else embed(cov_w),
        "cov_twoway": cov_two if full else embed(cov_two),
        "cov_twoway_raw": cov_two_raw if full else embed(cov_two_raw),
        "psd_fixed": psd_fixed,
        "n_sweeps": sweeps,
        "nobs": n,
        "n_pairs": n_g,
        "n_weeks": n_weeks,
        "design_rank": design_rank,
        "rank_deficient": rank_deficient,
        "dropped": dropped,
        "design_cond": float(corr_eig.max() / corr_eig.min()) if rank_check and corr_eig.min() > 0 else float("inf"),
        "design_min_eig": float(corr_eig.min()),
    }


def two_way_fe_ols(
    df: pd.DataFrame,
    y: str = "LOG_UNITS",
    regressors: list[str] | None = None,
    pair: str = "PAIR_ID",
    week: str = "WEEK",
    tol: float = 1e-10,
) -> tuple[pd.DataFrame, dict]:
    """OLS with pair and week fixed effects and pair-clustered (CR1) standard errors.

    The variance is ``c (X'X)^-1 [sum_g X_g' u_g u_g' X_g] (X'X)^-1`` on the demeaned data, with the CR1
    factor ``c = G/(G-1) * (N-1)/(N-K-W+1)``: ``G`` pair clusters, ``N`` rows, ``K`` slopes and ``W`` weeks.
    The pair effects are nested in the clusters and cost no degrees of freedom; the ``W-1`` week effects are
    not nested and are counted (the ``reghdfe`` convention). Normal critical values (1.96) are used for the
    pair-clustered interval.

    Two-way clustering (Cameron-Gelbach-Miller) uses the meat ``c_p M_pair + c_w M_week - c_r M_row`` where
    each ``M`` is the sum over clusters of the outer product of the cluster score sums; the intersection
    cluster (pair x week) is the row, so ``M_row`` is the HC0 meat. Each component gets its own CR1-type
    factor ``c_g = G/(G-1) * (N-1)/(N-K-W+1)`` with ``G`` = pairs, weeks or rows. If the resulting covariance
    is not positive semidefinite, negative eigenvalues are set to zero (``twoway_psd_fixed`` is then True).
    The two-way interval uses a ``t`` critical value with ``min(G_pair, G_week) - 1`` degrees of freedom.
    ``se_week`` is the one-way week-clustered error.

    Parameters
    ----------
    df : pandas.DataFrame
        Rows with ``y``, the regressors and the two group columns.
    y : str
        Outcome column.
    regressors : list of str, optional
        Defaults to :data:`FE_REGRESSORS`.
    pair, week : str
        Group columns of the two sets of effects; clusters are the ``pair`` groups.
    tol : float
        Convergence tolerance of :func:`demean_two_way`.

    Returns
    -------
    tuple
        A table with ``term, estimate, se_cluster, ci_low, ci_high, se_week, se_twoway, ci_low_twoway,
        ci_high_twoway, nobs, n_pairs, n_weeks`` and a dict with ``n_sweeps``, ``twoway_psd_fixed``,
        ``n_clusters_pair``, ``n_clusters_week``, ``n_clusters_cell`` and ``t_crit_twoway``.
    """
    regs = list(FE_REGRESSORS if regressors is None else regressors)
    cols = [y, *regs]
    data = df[cols].to_numpy(dtype=float)
    fit = fe_fit_arrays(data[:, 0], data[:, 1:], df[pair].to_numpy(), df[week].to_numpy(), tol=tol)
    beta, n, n_g, n_weeks = fit["beta"], fit["nobs"], fit["n_pairs"], fit["n_weeks"]
    cov, cov_w, cov_two = fit["cov_pair"], fit["cov_week"], fit["cov_twoway"]
    psd_fixed, sweeps = fit["psd_fixed"], fit["n_sweeps"]
    se = np.sqrt(np.diag(cov))
    se_w = np.sqrt(np.diag(cov_w))
    se_two = np.sqrt(np.diag(cov_two))
    z = norm.ppf(0.975)
    t_crit = t_dist.ppf(0.975, min(n_g, n_weeks) - 1)
    table = pd.DataFrame(
        {
            "term": regs,
            "estimate": beta,
            "se_cluster": se,
            "ci_low": beta - z * se,
            "ci_high": beta + z * se,
            "se_week": se_w,
            "se_twoway": se_two,
            "ci_low_twoway": beta - t_crit * se_two,
            "ci_high_twoway": beta + t_crit * se_two,
            "nobs": n,
            "n_pairs": n_g,
            "n_weeks": n_weeks,
        }
    )
    return table, {"n_sweeps": sweeps, "twoway_psd_fixed": psd_fixed, "n_clusters_pair": n_g,
                   "n_clusters_week": n_weeks, "n_clusters_cell": n, "t_crit_twoway": float(t_crit)}
