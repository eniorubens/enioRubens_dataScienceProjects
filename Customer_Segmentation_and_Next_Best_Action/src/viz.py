"""
viz.py
------
Corporate plotting theme and reusable chart functions.

Every user-visible string (axis labels, titles, subtitles, annotations)
passes through ``t()`` so the same function renders correctly in EN and PT
without duplication.

Usage
-----
    from src.viz import set_corporate_theme, hist, barh, scatter, boxplot
    from src.multilang import set_language

    set_language("pt")          # switch once; all charts follow
    set_corporate_theme()
    hist(rfm["Recency"], t("Recency Distribution"), ...)
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
import seaborn as sns

from src.multilang import t

# ── Colour palette ─────────────────────────────────────────────────────────
PALETTE: list[str] = [
    "#005F73",
    "#0A9396",
    "#94D2BD",
    "#EE9B00",
    "#CA6702",
    "#9B2226",
]


# ── Number formatters ──────────────────────────────────────────────────────

def fmt(value: Any, decimals: int = 0) -> str:
    """Format *value* as a locale-neutral number string.

    Uses period thousands separator and comma decimal separator (BR/EU style)
    when the active language is ``"pt"``; dot-separated otherwise.

    Parameters
    ----------
    value : numeric or NA
    decimals : int
        Number of decimal places.
    """
    from src.multilang import get_language

    if pd.isna(value):
        return ""
    text = f"{value:,.{decimals}f}"
    if get_language() == "pt":
        return text.replace(",", "X").replace(".", ",").replace("X", ".")
    return text


def compact(value: float) -> str:
    """Shorten large numbers for chart annotations (e.g. 1 500 000 → '1.5M')."""
    from src.multilang import get_language

    value = float(value)
    lang = get_language()

    if abs(value) >= 1_000_000:
        label = f"{value / 1_000_000:.1f}M"
    elif abs(value) >= 1_000:
        suffix = "mil" if lang == "pt" else "K"
        label = f"{fmt(value / 1_000, 1)}{suffix}"
    else:
        label = fmt(value, 0 if abs(value) >= 10 else 1)
    return label


# ── Theme ──────────────────────────────────────────────────────────────────

def set_corporate_theme() -> None:
    """Apply a clean, editorial chart theme to all subsequent matplotlib/seaborn plots."""
    sns.set_theme(style="ticks")

    plt.rcParams.update({
        "figure.dpi":            96,
        "figure.figsize":        (13.33, 6.5),
        "figure.facecolor":      "white",
        "figure.titlesize":      22,
        "figure.titleweight":    "bold",
        "axes.titlesize":        22,
        "axes.titleweight":      "bold",
        "axes.titlepad":         20,
        "axes.labelsize":        12,
        "axes.labelweight":      "bold",
        "axes.spines.right":     False,
        "axes.spines.left":      False,
        "axes.spines.top":       False,
        "axes.grid":             True,
        "axes.grid.axis":        "y",
        "grid.alpha":            0.45,
        "grid.linewidth":        1.0,
        "ytick.left":            False,
        "legend.title_fontsize": 14,
        "legend.fontsize":       12,
        "legend.frameon":        True,
        "legend.framealpha":     1,
        "legend.fancybox":       True,
        "legend.facecolor":      "white",
        "legend.edgecolor":      "gray",
        "legend.borderpad":      0.6,
        "lines.linewidth":       3,
        "lines.markersize":      10,
    })


# ── Internal helpers ───────────────────────────────────────────────────────

def _add_titles(ax: plt.Axes, title: str, subtitle: str | None = None) -> None:
    ax.set_title(t(title), loc="left", fontsize=15, fontweight="bold", pad=22)
    if subtitle:
        ax.text(
            0, 1.02, t(subtitle),
            transform=ax.transAxes,
            ha="left", va="bottom",
            fontsize=10, color="#52606D",
        )


def _set_compact_axis(ax: plt.Axes, axis: str = "x") -> None:
    formatter = mticker.FuncFormatter(lambda v, _: compact(v))
    if axis == "x":
        ax.xaxis.set_major_formatter(formatter)
    else:
        ax.yaxis.set_major_formatter(formatter)


# ── Public chart functions ─────────────────────────────────────────────────

def hist(
    series: pd.Series,
    title: str,
    subtitle: str,
    xlabel: str,
    color: str,
    bins: int = 30,
    q: float | None = None,
) -> None:
    """Histogram with optional percentile clip.

    Parameters
    ----------
    series : pd.Series
        Numeric data to plot.
    title : str
        Chart title (EN; translated automatically).
    subtitle : str
        Chart subtitle (EN; translated automatically).
    xlabel : str
        X-axis label (EN; translated automatically).
    color : str
        Bar fill colour (hex).
    bins : int
        Number of histogram bins.
    q : float, optional
        If provided, clip at this quantile for display readability.
    """
    values = pd.to_numeric(series, errors="coerce").dropna()
    if q is not None:
        values = values.clip(upper=values.quantile(q))

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.hist(values, bins=bins, color=color, alpha=0.88, edgecolor="white")
    _add_titles(ax, title, subtitle)
    ax.set_xlabel(t(xlabel))
    ax.set_ylabel(t("Customers"))
    ax.grid(axis="y", alpha=0.8)
    _set_compact_axis(ax, "x")
    _set_compact_axis(ax, "y")
    plt.tight_layout()
    plt.show()


def barh(
    df: pd.DataFrame,
    label: str,
    value: str,
    title: str,
    subtitle: str,
    color: str = "#005F73",
    color_col: str | None = None,
    cmap: dict | None = None,
) -> None:
    """Horizontal bar chart with inline value annotations."""
    data = df.reset_index(drop=True)
    colors: Any = color
    if color_col and cmap:
        colors = data[color_col].map(cmap).fillna(color)  # type: ignore[arg-type]

    fig_height = max(4.2, 0.42 * len(data) + 1.4)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    ax.barh(data[label].astype(str), data[value], color=colors, alpha=0.9)
    _add_titles(ax, title, subtitle)
    ax.set_xlabel(t(value))
    ax.grid(axis="x", alpha=0.8)
    _set_compact_axis(ax, "x")

    for idx, val in enumerate(data[value]):
        ax.text(val, idx, f" {compact(val)}", va="center", fontsize=10, color="#323F4B")

    plt.tight_layout()
    plt.show()


def lineplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    subtitle: str,
    color: str,
    zero: bool = False,
) -> None:
    """Line plot — used for elbow / silhouette curves."""
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.plot(df[x], df[y], marker="o", linewidth=2.4, color=color)
    _add_titles(ax, title, subtitle)
    ax.set_xlabel(t("Number of clusters"))
    ax.set_ylabel(t(y))
    if zero:
        ax.set_ylim(bottom=0)
    ax.grid(alpha=0.8)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    _set_compact_axis(ax, "y")
    plt.tight_layout()
    plt.show()


def scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    subtitle: str,
    cmap: dict,
    logx: bool = False,
    logy: bool = False,
) -> None:
    """Scatter plot coloured by cluster, with segment labels in legend."""
    fig, ax = plt.subplots(figsize=(10.8, 5.6))

    for cluster in sorted(df["Cluster"].unique()):
        group = df.loc[df["Cluster"] == cluster]
        segment = t(str(group["SegmentName"].iloc[0]))
        ax.scatter(
            group[x], group[y],
            s=22, alpha=0.55,
            color=cmap[cluster],
            label=f"C{cluster} – {segment}",
        )

    _add_titles(ax, title, subtitle)
    ax.set_xlabel(t(x))
    ax.set_ylabel(t(y))
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.7)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    _set_compact_axis(ax, "x")
    _set_compact_axis(ax, "y")
    plt.tight_layout()
    plt.show()


def boxplot(
    df: pd.DataFrame,
    group: str,
    value: str,
    title: str,
    subtitle: str,
    cmap: dict,
) -> None:
    """Box plot per cluster group, without outlier fliers."""
    groups = sorted(df[group].unique())
    values = [df.loc[df[group] == cluster, value] for cluster in groups]

    fig, ax = plt.subplots(figsize=(9.8, 5.0))
    boxplot_kwargs: dict[str, Any] = {"patch_artist": True, "showfliers": False}

    try:
        box = ax.boxplot(
            values,
            tick_labels=[f"C{c}" for c in groups],
            **boxplot_kwargs,
        )
    except TypeError:
        box = ax.boxplot(
            values,
            labels=[f"C{c}" for c in groups],
            **boxplot_kwargs,
        )

    colors = [cmap.get(c, PALETTE[0]) for c in groups]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    _add_titles(ax, title, subtitle)
    ax.set_xlabel(t(group))
    ax.set_ylabel(t(value))
    ax.grid(axis="y", alpha=0.7)
    plt.tight_layout()
    plt.show()


def budget_curves(
    budget_curve_df: pd.DataFrame,
    budget_total: float,
    optimal_budget_point: pd.DataFrame,
) -> None:
    """Two-panel chart: Budget vs Incremental Profit and Budget vs ROI."""
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.8))

    axes[0].plot(
        budget_curve_df["Budget"],
        budget_curve_df["IncrementalProfit"],
        marker="o", markersize=4, linewidth=2.2, color=PALETTE[0],
    )
    axes[0].axvline(budget_total, color="#52606D", linestyle="--", linewidth=1.4)
    axes[0].text(
        budget_total,
        budget_curve_df["IncrementalProfit"].max(),
        f" {t('current budget')}",
        rotation=90, va="top", ha="left", fontsize=10, color="#52606D",
    )
    _add_titles(
        axes[0],
        "Budget vs Incremental Profit",
        "Total accumulated return from prescriptive margin-based selection.",
    )
    axes[0].set_xlabel(t("Available budget ($)"))
    axes[0].set_ylabel(t("Incremental profit ($)"))
    axes[0].grid(alpha=0.8)
    _set_compact_axis(axes[0], "x")
    _set_compact_axis(axes[0], "y")

    axes[1].plot(
        budget_curve_df["Budget"],
        budget_curve_df["ROI"],
        marker="o", markersize=4, linewidth=2.2, color=PALETTE[3],
    )
    axes[1].axvline(budget_total, color="#52606D", linestyle="--", linewidth=1.4)
    if not optimal_budget_point.empty:
        opt = optimal_budget_point.iloc[0]
        axes[1].scatter(opt["Budget"], opt["ROI"], s=95, color=PALETTE[5], zorder=3)
        axes[1].annotate(
            t("optimal point"),
            xy=(opt["Budget"], opt["ROI"]),
            xytext=(8, 12), textcoords="offset points",
            fontsize=10, color="#323F4B",
        )
    _add_titles(
        axes[1],
        "Budget vs ROI",
        "Average investment efficiency as coverage increases.",
    )
    axes[1].set_xlabel(t("Available budget ($)"))
    axes[1].set_ylabel(t("ROI"))
    axes[1].grid(alpha=0.8)
    _set_compact_axis(axes[1], "x")
    axes[1].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{fmt(v, 1)}x")
    )

    plt.tight_layout()
    plt.show()
