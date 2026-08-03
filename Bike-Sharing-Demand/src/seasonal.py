"""Seasonal / de-trending helpers shared across the EDA and analysis notebooks.

The 2015-2024 dataset grows ~110x over its span (business growth, not weather),
so any analysis of the weather/seasonal signal must first remove that trend.
Two utilities are provided:

- ``meteorological_year`` groups December with the following Jan/Feb so a winter
  is never split at the arbitrary Jan 1 calendar boundary.
- ``demand_index_moving_average`` de-trends demand into a multiplicative index
  via ratio-to-moving-average, continuously (no Jan 1 discontinuity).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.i18n import resolve_lang as _resolve_lang


def meteorological_year(dates: pd.Series) -> pd.Series:
    """Return the meteorological year (Dec->Nov) for each timestamp.

    December is assigned to the following year so a winter (Dec-Jan-Feb) stays
    in a single bucket, unlike the calendar year whose Jan 1 boundary cuts the
    middle of winter.

    Parameters
    ----------
    dates:
        Datetime-like Series (or anything ``pd.to_datetime`` accepts).

    Returns
    -------
    pd.Series
        Integer meteorological year, index-aligned with ``dates``.
    """
    dates = pd.to_datetime(dates)
    return dates.dt.year + (dates.dt.month == 12).astype(int)


def demand_index_moving_average(
    df: pd.DataFrame,
    target: str = "Rented Bike Count",
    date_col: str = "Date",
    anomaly_mask: Optional[pd.Series] = None,
    window: int = 365,
    min_periods: int = 180,
) -> pd.Series:
    """De-trend demand into a multiplicative index via ratio-to-moving-average.

    The growth trend is removed continuously (no Jan 1 discontinuity) by
    dividing each hour's demand by a centred rolling mean of the *daily* mean
    demand. A ``window``-day window spans one full seasonal cycle, so the
    baseline tracks the local level (business growth) while the returned index
    retains the seasonal/weather signal, oscillating around 1.

    Aggregating to calendar day first is essential: ``date_col`` here is the
    hourly timestamp, so a rolling window applied to it directly would span
    ``window`` *hours*, not days, and would flatten the seasonal signal.

    Rows flagged by ``anomaly_mask`` are excluded from the baseline estimate
    (so an anomalous period does not drag the trend down) but still receive an
    index value.

    Parameters
    ----------
    df:
        Frame containing ``target`` and ``date_col``.
    target:
        Demand column to de-trend.
    date_col:
        Hourly (or daily) timestamp column used to derive the calendar day.
    anomaly_mask:
        Optional boolean mask; True rows are excluded from the baseline.
    window:
        Rolling window length in days (one year by default).
    min_periods:
        Minimum days required in a window for the baseline to be defined.

    Returns
    -------
    pd.Series
        Multiplicative demand index aligned with ``df.index``; NaN at the
        series edges where the centred window has fewer than ``min_periods``
        days.
    """
    day = pd.to_datetime(df[date_col]).dt.normalize()

    daily = pd.DataFrame({"day": day.to_numpy(), "demand": df[target].to_numpy()})
    if anomaly_mask is not None:
        daily["anom"] = np.asarray(anomaly_mask, dtype=bool)
    else:
        daily["anom"] = False

    daily_mean = daily.groupby("day").agg(demand=("demand", "mean"), anom=("anom", "any"))
    demand_for_base = daily_mean["demand"].where(~daily_mean["anom"])
    baseline = demand_for_base.rolling(window, center=True, min_periods=min_periods).mean()

    base_per_row = day.map(baseline)
    return df[target] / base_per_row


# ---------------------------------------------------------------------------
# 2020 anomaly diagnosis (integrity gate) — shared by notebooks 01, 02 and 03
# ---------------------------------------------------------------------------


@dataclass
class AnomalyDiagnosis:
    """Result of the reproducible 2020-style demand-anomaly gate.

    Each month of ``anomaly_year`` is compared with the geometric interpolation
    between the same month of the two ``ref_years``. Months whose demand falls
    below ``threshold`` of that expectation are flagged. The geometric mean is
    used because the system's growth is multiplicative.

    Attributes
    ----------
    monthly_mean:
        Mean demand with month as the index and year as columns.
    expected:
        Geometric interpolation of the expected level for ``anomaly_year``.
    ratio:
        ``anomaly_year`` demand divided by ``expected`` (per month).
    anomalous_months:
        Months flagged below ``threshold``.
    threshold, anomaly_year, ref_years:
        Rule parameters, kept for reporting and plotting.
    """

    monthly_mean: pd.DataFrame
    expected: pd.Series
    ratio: pd.Series
    anomalous_months: List[int]
    threshold: float
    anomaly_year: int
    ref_years: Tuple[int, int]


def _years_months(
    df: pd.DataFrame, date_col: str, year_col: str, month_col: str
) -> Tuple[pd.Series, pd.Series]:
    """Return (year, month) Series, deriving them from ``date_col`` if absent."""
    stamps = None
    if year_col in df.columns:
        years = df[year_col]
    else:
        stamps = pd.to_datetime(df[date_col])
        years = stamps.dt.year
    if month_col in df.columns:
        months = df[month_col]
    else:
        stamps = pd.to_datetime(df[date_col]) if stamps is None else stamps
        months = stamps.dt.month
    return years, months


def anomaly_diagnosis(
    df: pd.DataFrame,
    target: str = "Rented Bike Count",
    *,
    date_col: str = "DateTime",
    year_col: str = "Year",
    month_col: str = "Month",
    anomaly_year: int = 2020,
    ref_years: Tuple[int, int] = (2019, 2021),
    threshold: float = 0.60,
) -> AnomalyDiagnosis:
    """Run the geometric-interpolation anomaly gate (integrity check).

    ``year_col``/``month_col`` are used when present; otherwise both are derived
    from ``date_col``. The rule and its parameters are analytic invariants and
    are not changed by this refactor.
    """
    years, months = _years_months(df, date_col, year_col, month_col)
    years = years.rename("__year")
    months = months.rename("__month")
    monthly = df.groupby([years, months])[target].mean().unstack("__year")
    monthly.index.name = "Month"

    lo, hi = ref_years
    expected = np.sqrt(monthly[lo] * monthly[hi])
    ratio = monthly[anomaly_year] / expected
    anomalous_months = ratio[ratio < threshold].index.tolist()
    return AnomalyDiagnosis(
        monthly_mean=monthly,
        expected=expected,
        ratio=ratio,
        anomalous_months=anomalous_months,
        threshold=threshold,
        anomaly_year=anomaly_year,
        ref_years=ref_years,
    )


def anomaly_mask(
    df: pd.DataFrame,
    diagnosis: AnomalyDiagnosis,
    *,
    date_col: str = "DateTime",
    year_col: str = "Year",
    month_col: str = "Month",
) -> pd.Series:
    """Boolean mask (aligned to ``df.index``) of the flagged anomaly window.

    True where the row belongs to ``anomaly_year`` and one of the flagged
    ``anomalous_months`` — the same data-driven rule as the diagnosis, not a
    hand-picked date range.
    """
    years, months = _years_months(df, date_col, year_col, month_col)
    return years.eq(diagnosis.anomaly_year) & months.isin(diagnosis.anomalous_months)


def anomaly_report_table(diagnosis: AnomalyDiagnosis, lang=None) -> pd.DataFrame:
    """Build the display-ready (localized) monthly anomaly report."""
    lang = _resolve_lang(lang)
    labels = lang(
        {
            "mean": "Média",
            "geo_ref": "Referência geométrica",
            "ratio": "Razão",
            "reference": "referência",
            "flagged": "Sinalizado",
            "month": "Mês",
        }
    )
    lo, hi = diagnosis.ref_years
    ay = diagnosis.anomaly_year
    table = pd.DataFrame(
        {
            f'{labels["mean"]} {lo}': diagnosis.monthly_mean[lo],
            f'{labels["mean"]} {ay}': diagnosis.monthly_mean[ay],
            f'{labels["mean"]} {hi}': diagnosis.monthly_mean[hi],
            labels["geo_ref"]: diagnosis.expected,
            f'{labels["ratio"]} {ay} / {labels["reference"]}': diagnosis.ratio,
            labels["flagged"]: diagnosis.ratio < diagnosis.threshold,
        }
    )
    table.index.name = labels["month"]
    return table


def plot_anomaly_diagnosis(diagnosis: AnomalyDiagnosis, lang=None) -> Tuple[plt.Figure, np.ndarray]:
    """Plot the monthly reference lines and the ratio-vs-threshold panel.

    Returns ``(fig, axes)``; the caller decides when to show it.
    """
    lang = _resolve_lang(lang)
    labels = lang(
        {
            "monthly_title": "Demanda mensal média — {lo} a {hi}",
            "month": "Mês",
            "per_hour": "Bicicletas por hora",
            "ratio_title": "Demanda de {ay} em relação à referência",
            "ratio": "Razão",
            "threshold": "Limiar = {thr:.2f}",
        }
    )
    lo, hi = diagnosis.ref_years
    ay = diagnosis.anomaly_year
    monthly = diagnosis.monthly_mean

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
    for year, style in [(lo, "--"), (ay, "-"), (hi, ":")]:
        axes[0].plot(
            monthly.index,
            monthly[year],
            style,
            marker="o",
            markersize=3,
            linewidth=1.2,
            label=str(year),
        )
    for month in diagnosis.anomalous_months:
        axes[0].axvspan(month - 0.5, month + 0.5, color="red", alpha=0.08)
    axes[0].set(
        title=labels["monthly_title"].format(lo=lo, hi=hi),
        xlabel=labels["month"],
        ylabel=labels["per_hour"],
        xticks=range(1, 13),
    )
    axes[0].legend()

    axes[1].plot(diagnosis.ratio.index, diagnosis.ratio, color="crimson", marker="o", linewidth=1.2)
    axes[1].axhline(
        diagnosis.threshold,
        color="gray",
        linestyle="--",
        label=labels["threshold"].format(thr=diagnosis.threshold),
    )
    axes[1].set(
        title=labels["ratio_title"].format(ay=ay),
        xlabel=labels["month"],
        ylabel=labels["ratio"],
        xticks=range(1, 13),
    )
    axes[1].legend()
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Detrended-index analysis frame and diagnostics (notebook 02)
# ---------------------------------------------------------------------------


def build_analysis_frame(
    full: pd.DataFrame,
    demand_index: pd.Series,
    target: str = "Rented Bike Count",
    *,
    drop_cols: Sequence[str] = ("Date", "Year", "met_year", "Functioning Day"),
    meta_cols: Sequence[str] = ("Year", "Month", "met_year"),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build the multivariate analysis frame from the detrended index.

    The target column is replaced by ``demand_index``; edge rows where the
    centred baseline is undefined (index NaN) are dropped; non-feature / leakage
    columns are removed. Returns ``(mv_df, meta)`` where ``meta`` keeps the raw
    count and calendar keys for the rows retained.
    """
    keep = demand_index.notna()
    raw_count = full[target]
    mv = full.copy()
    mv[target] = demand_index
    mv = mv.loc[keep].copy()
    meta = full.loc[keep, list(meta_cols)].assign(raw_count=raw_count[keep])
    mv = mv.drop(columns=list(drop_cols), errors="ignore")
    return mv, meta


def time_index_correlations(
    full: pd.DataFrame, demand_index: pd.Series, target: str = "Rented Bike Count"
) -> Tuple[float, float]:
    """Return (corr(time, index), corr(time, raw demand)) — trend-removal check."""
    corr_idx = np.corrcoef(np.arange(len(full))[demand_index.notna()], demand_index.dropna())[0, 1]
    corr_raw = np.corrcoef(np.arange(len(full)), full[target])[0, 1]
    return corr_idx, corr_raw


def plot_baseline_and_index(
    full: pd.DataFrame,
    demand_index: pd.Series,
    target: str = "Rented Bike Count",
    *,
    date_col: str = "Date",
    anomaly_col: str = "is_anomalous_2020",
    lang=None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot the smooth daily baseline and the resulting detrended index.

    The baseline excludes the flagged anomaly days from its estimate. Minor
    ticks are pinned to the start of each meteorological season (Mar/Jun/Sep/Dec)
    while the yearly major ticks are kept; the grid stays off by design.
    """
    lang = _resolve_lang(lang)
    labels = lang(
        {
            "daily_demand": "demanda diária",
            "baseline": "linha de base centrada de 365 dias (tendência)",
            "top_title": "Demanda diária bruta vs linha de base contínua de crescimento",
            "per_hour": "aluguéis/hora",
            "index_title": (
                "Índice de demanda = demanda / linha de base "
                "(tendência removida, sinal sazonal preservado)"
            ),
            "index": "índice",
        }
    )
    day = full[date_col].dt.normalize()
    daily_demand = full.groupby(day)[target].mean()
    baseline_daily = (
        full.assign(d=day, m=full[anomaly_col].astype(bool))
        .groupby("d")
        .apply(lambda g: np.nan if g["m"].any() else g[target].mean())
        .rolling(365, center=True, min_periods=180)
        .mean()
    )

    fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    daily_demand.plot(
        ax=axes[0], color="steelblue", alpha=0.5, label=labels["daily_demand"], lw=0.5
    )
    baseline_daily.plot(ax=axes[0], color="crimson", lw=1, label=labels["baseline"])
    axes[0].set(title=labels["top_title"], ylabel=labels["per_hour"])
    axes[0].legend()

    full.assign(idx=demand_index).groupby(day)["idx"].mean().plot(
        ax=axes[1], color="seagreen", alpha=0.7, lw=0.5
    )
    axes[1].axhline(1.0, ls="--", color="gray", lw=1)
    axes[1].set(title=labels["index_title"], ylabel=labels["index"], xlabel="")

    for ax in axes:
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[3, 6, 9, 12]))

    fig.tight_layout()
    return fig, axes


def seasonal_shape_by_met_year(
    full: pd.DataFrame,
    demand_index: pd.Series,
    *,
    met_year_col: str = "met_year",
    month_col: str = "Month",
) -> pd.DataFrame:
    """Mean index per (meteorological year × month), ordered Dec→Nov."""
    monthly = (
        full.assign(idx=demand_index)
        .dropna(subset=["idx"])
        .groupby([met_year_col, month_col])["idx"]
        .mean()
        .unstack(met_year_col)
    )
    met_order = [12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    return monthly.reindex(met_order)


def plot_seasonal_shape_by_met_year(
    full: pd.DataFrame,
    demand_index: pd.Series,
    *,
    min_months: int = 6,
    lang=None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Overlay each meteorological year's seasonal index profile (Dec→Nov)."""
    lang = _resolve_lang(lang)
    labels = lang(
        {
            "months": [
                "Dez",
                "Jan",
                "Fev",
                "Mar",
                "Abr",
                "Mai",
                "Jun",
                "Jul",
                "Ago",
                "Set",
                "Out",
                "Nov",
            ],
            "title": "Forma sazonal do índice de demanda por ano meteorológico (Dez→Nov)",
            "ylabel": "índice médio",
            "xlabel": "mês (ordem meteorológica)",
            "legend_title": "ano met.",
        }
    )
    monthly_idx = seasonal_shape_by_met_year(full, demand_index)
    met_order = list(monthly_idx.index)

    fig, ax = plt.subplots(figsize=(14, 5))
    for my in monthly_idx.columns:
        col = monthly_idx[my]
        if col.notna().sum() >= min_months:
            ax.plot(
                range(len(met_order)),
                col.values,
                marker="o",
                alpha=0.7,
                label=str(int(my)),
                lw=1,
                markersize=2,
            )
    ax.set_xticks(range(len(met_order)))
    ax.set_xticklabels(labels["months"])
    ax.axhline(1.0, ls="--", color="gray", lw=2)
    ax.axvspan(-0.3, 2.3, color="steelblue", alpha=0.06)  # winter block
    ax.set(title=labels["title"], ylabel=labels["ylabel"], xlabel=labels["xlabel"])
    ax.legend(title=labels["legend_title"], ncol=2, fontsize=8)
    fig.tight_layout()
    return fig, ax
