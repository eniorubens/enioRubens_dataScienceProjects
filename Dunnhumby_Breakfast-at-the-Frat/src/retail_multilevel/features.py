"""Model features for the fixed effects of price and promotion (decisions D12 and D13).

The log price is split into the depth of the temporary discount (``LOG_DISC``), the within-pair
deviation of the regular price (``LOG_BASE_DEV``) and the pair mean of the regular price
(``LOG_BASE_MEAN``, a Mundlak term). Seasonality enters as Fourier terms of the week of the year
plus a linear trend in years. Everything that is learned from data (pair means, UPC means) is
learned on a *fit window* (the rows passed to :func:`pair_means` and :func:`upc_means`), never on
the rows being scored. A pair absent from the fit window falls back to the mean of the same UPC in
the other stores of the window; a UPC absent from the window has no admissible covariate (D14 and
the external-audit revision).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

YEAR_WEEKS = 52.18
PROMO_FLAGS = ["FEATURE", "DISPLAY", "TPR_ONLY"]


def FOURIER_TERMS(n: int = 3) -> list[str]:  # noqa: N802
    """Names of the Fourier columns ``SIN_1, COS_1, ..., SIN_n, COS_n`` (in that order).

    Parameters
    ----------
    n : int
        Number of harmonics.

    Returns
    -------
    list of str
    """
    return [f"{fn}_{k}" for k in range(1, n + 1) for fn in ("SIN", "COS")]


def pair_means(fit_df: pd.DataFrame) -> pd.DataFrame:
    """Mean of ``LOG_BASE_PRICE`` per ``PAIR_ID`` on the fit-window rows passed.

    Parameters
    ----------
    fit_df : pandas.DataFrame
        Fit-window rows with ``PAIR_ID`` and ``LOG_BASE_PRICE``.

    Returns
    -------
    pandas.DataFrame
        Columns ``PAIR_ID, LOG_BASE_MEAN``.
    """
    return fit_df.groupby("PAIR_ID", sort=True)["LOG_BASE_PRICE"].mean().rename("LOG_BASE_MEAN").reset_index()


def upc_means(fit_df: pd.DataFrame) -> pd.DataFrame:
    """Mean of ``LOG_BASE_PRICE`` per ``UPC`` over all stores of the fit window.

    Used as the covariate fallback for pairs that are absent from the fit window.

    Parameters
    ----------
    fit_df : pandas.DataFrame
        Fit-window rows with ``UPC`` and ``LOG_BASE_PRICE``.

    Returns
    -------
    pandas.DataFrame
        Columns ``UPC, LOG_BASE_MEAN``.
    """
    return fit_df.groupby("UPC", sort=True)["LOG_BASE_PRICE"].mean().rename("LOG_BASE_MEAN").reset_index()


def add_model_features(
    df: pd.DataFrame, means: pd.DataFrame, upc_means: pd.DataFrame | None = None, n_fourier: int = 3
) -> pd.DataFrame:
    """Return a copy of ``df`` with the price, trend and seasonality features.

    Added columns: ``LOG_DISC = LOG_PRICE - LOG_BASE_PRICE``; ``LOG_BASE_MEAN``; ``MEAN_SOURCE``;
    ``LOG_BASE_DEV = LOG_BASE_PRICE - LOG_BASE_MEAN``; ``TREND = WEEK / 52`` (years since the start);
    and ``SIN_k, COS_k = sin/cos(2 pi k WEEK_OF_YEAR / 52.18)`` for ``k = 1..n_fourier``.
    ``FEATURE``, ``DISPLAY`` and ``TPR_ONLY`` are kept as 0/1 integers.

    ``LOG_BASE_MEAN`` follows a three-step rule, using only fit-window information:

    - ``MEAN_SOURCE = "pair"``: the pair is in the fit window, so its own mean is used;
    - ``"upc"``: the pair is absent but the UPC is present, so the UPC mean over the other stores
      of the fit window is used (requires ``upc_means``);
    - ``"none"``: neither is available; ``LOG_BASE_MEAN`` and ``LOG_BASE_DEV`` are ``NaN``.

    Rows with ``MEAN_SOURCE == "none"`` have no admissible covariate and are **not predictable**:
    callers must exclude them from predictions and count them as non-covered in any evaluation.

    Parameters
    ----------
    df : pandas.DataFrame
        Rows with ``PAIR_ID, LOG_PRICE, LOG_BASE_PRICE, WEEK, WEEK_OF_YEAR`` (and ``UPC`` when
        ``upc_means`` is given) and the promo flags.
    means : pandas.DataFrame
        Output of :func:`pair_means` on the fit window.
    upc_means : pandas.DataFrame, optional
        Output of the module-level :func:`upc_means` on the same fit window. Without it, pairs
        absent from the fit window are ``"none"``.
    n_fourier : int
        Number of Fourier harmonics.

    Returns
    -------
    pandas.DataFrame
    """
    out = df.copy()
    out["LOG_DISC"] = out["LOG_PRICE"] - out["LOG_BASE_PRICE"]
    pair_map = means.set_index("PAIR_ID")["LOG_BASE_MEAN"]
    pair_val = out["PAIR_ID"].map(pair_map).astype(float)
    if upc_means is not None:
        upc_val = out["UPC"].map(upc_means.set_index("UPC")["LOG_BASE_MEAN"]).astype(float)
    else:
        upc_val = pd.Series(np.nan, index=out.index)
    source = np.where(pair_val.notna(), "pair", np.where(upc_val.notna(), "upc", "none"))
    out["MEAN_SOURCE"] = pd.Series(source, index=out.index, dtype="object")
    out["LOG_BASE_MEAN"] = pair_val.where(pair_val.notna(), upc_val).astype(float)
    out["LOG_BASE_DEV"] = out["LOG_BASE_PRICE"] - out["LOG_BASE_MEAN"]
    out["TREND"] = out["WEEK"].astype(float) / 52.0
    woy = out["WEEK_OF_YEAR"].to_numpy(dtype=float)
    for k in range(1, n_fourier + 1):
        angle = 2 * np.pi * k * woy / YEAR_WEEKS
        out[f"SIN_{k}"] = np.sin(angle)
        out[f"COS_{k}"] = np.cos(angle)
    for col in PROMO_FLAGS:
        out[col] = out[col].astype(int)
    return out
