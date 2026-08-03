"""Outlier detection and rainfall/snowfall event analysis.

Faithful port of notebook cells:
  [53] rainfall_event, plot_rainfall
  [62] iqr_outliers, plot_outliers
  [87] plot_snowfall_by_season
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.i18n import resolve_lang as _resolve_lang
from src.plotting import set_graph_parameters


# ---------------------------------------------------------------------------
# iqr_outliers
# ---------------------------------------------------------------------------


def iqr_outliers(data: pd.Series) -> pd.Series:
    """Detect IQR-based outliers in a Series.

    Parameters
    ----------
    data:
        Numeric Series to inspect.

    Returns
    -------
    pd.Series
        Subset of ``data`` containing values below or above the IQR fences.
    """
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_fence = q1 - (1.5 * iqr)
    upper_fence = q3 + (1.5 * iqr)
    return data[(data > upper_fence) | (data < lower_fence)]


def plot_outliers(
    outliers: pd.Series,
    data: pd.Series,
    temperature: pd.Series,
    ax: plt.Axes,
    method: str = "KNN",
    halignment: str = "right",
    valignment: str = "bottom",
    labels: bool = False,
    all_data: bool = False,
    label_shift: tuple = (0.1, 55),
    lang=None,
) -> plt.Axes:
    """Plot a time series with IQR outliers highlighted.

    Parameters
    ----------
    outliers:
        Subset of ``data`` that are outliers (output of ``iqr_outliers``).
    data:
        Full time series of bike counts.
    temperature:
        Temperature Series aligned to ``data`` for annotation.
    ax:
        Matplotlib Axes to draw on.
    method:
        Label used in the subplot title.
    labels:
        If True, annotate individual outlier points with temperature labels.
    """
    lang = _resolve_lang(lang)
    plot_labels = lang(
        {
            "xlabel": "Date",
            "ylabel": "# of Rented Bikes",
            "rented_bikes": "Rented Bikes",
            "outliers_legend": "Outliers",
        }
    )
    ax_1 = data.plot(alpha=0.6, linewidth=1.3, ax=ax)

    outlier_temps = temperature.loc[outliers.index]
    min_temp = outlier_temps.min()
    max_temp = outlier_temps.max()

    if labels:
        data.loc[outliers.index].plot(ax=ax_1, style="rx", linewidth=0.7)

        for idx in outliers.index:
            outlier_value = data.loc[idx]
            temp_value = temperature.loc[idx]
            label_y = outlier_value + label_shift[1]
            ax_1.text(
                idx,
                label_y,
                f"{temp_value:.1f} deg C",
                horizontalalignment=halignment,
                verticalalignment=valignment,
                rotation=75,
                fontsize=8,
            )

        ax_1.text(
            1.03,
            0.65,
            f"Min Temp: {min_temp:.1f} deg C\nMax Temp: {max_temp:.1f} deg C",
            transform=ax_1.transAxes,
            fontsize="small",
            verticalalignment="top",
            horizontalalignment="left",
        )
    else:
        data.loc[outliers.index].plot(ax=ax_1, style="rx", linewidth=0.7)

    ax_1.set_title(f"{method}")
    ax_1.set_xlabel(plot_labels["xlabel"])
    ax_1.set_ylabel(plot_labels["ylabel"])
    ax_1.legend(
        [plot_labels["rented_bikes"], plot_labels["outliers_legend"]],
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.3,
    )
    return ax_1


def plot_iqr_outliers_by_season(
    dataframe: pd.DataFrame,
    lang=None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot IQR outliers for All / Winter / Spring / Summer / Autumn seasons."""
    lang = _resolve_lang(lang)
    season_labels = lang(
        {
            "suptitle": "Rented Bike Outliers",
            "all_season": "All Season",
            "winter_season": "Winter Season",
            "spring_season": "Spring Season",
            "summer_season": "Summer Season",
            "autumn_season": "Autumn Season",
        }
    )
    with plt.style.context(["default"]):
        set_graph_parameters()
        fig, ax = plt.subplots(nrows=5, figsize=(20, 22))

        df2 = dataframe.copy(deep=True)
        df2.set_index(keys="DateTime", inplace=True)

        for i, (season_label, mask) in enumerate(
            [
                (season_labels["all_season"], pd.Series([True] * len(df2), index=df2.index)),
                (season_labels["winter_season"], df2["Seasons"] == "Winter"),
                (season_labels["spring_season"], df2["Seasons"] == "Spring"),
                (season_labels["summer_season"], df2["Seasons"] == "Summer"),
                (season_labels["autumn_season"], df2["Seasons"] == "Autumn"),
            ]
        ):
            tx = df2.loc[mask, "Rented Bike Count"]
            temperature = df2.loc[mask, "Temperature(C)"]
            outliers = iqr_outliers(tx)
            plot_outliers(
                outliers=outliers,
                data=tx,
                temperature=temperature,
                ax=ax[i],
                labels=False,
                method=season_label,
                lang=lang,
            )

        fig.suptitle(season_labels["suptitle"], fontsize=18, y=1)
        plt.tight_layout()

    return fig, ax


def _hourly_timestamp(
    dataframe: pd.DataFrame,
    datetime_col: str = "DateTime",
    hour_col: str = "Hour",
) -> pd.Series:
    """Return one timestamp per hourly row without mutating ``dataframe``."""
    timestamp = pd.to_datetime(dataframe[datetime_col])
    if hour_col in dataframe.columns and timestamp.dt.hour.eq(0).all():
        timestamp = timestamp + pd.to_timedelta(dataframe[hour_col], unit="h")
    return timestamp


def iqr_outlier_summary_by_season_year(
    dataframe: pd.DataFrame,
    target_col: str = "Rented Bike Count",
    datetime_col: str = "DateTime",
    hour_col: str = "Hour",
    season_col: str = "Seasons",
    iqr_multiplier: float = 1.5,
) -> pd.DataFrame:
    """Summarize IQR outliers within each meteorological-year/season group.

    Fences are estimated separately for every group so long-term changes in
    demand level do not cause recent, otherwise typical values to be compared
    with the scale of early years. An ``All`` row is also reported for each
    meteorological year.
    """
    df = dataframe.copy(deep=True)
    df["_timestamp"] = _hourly_timestamp(df, datetime_col, hour_col)
    df["_meteorological_year"] = df["_timestamp"].dt.year + df["_timestamp"].dt.month.eq(12).astype(
        int
    )

    rows = []
    season_order = ["Winter", "Spring", "Summer", "Autumn"]
    for year, year_df in df.groupby("_meteorological_year", sort=True):
        scopes = [("All", year_df)]
        scopes.extend(
            (season, year_df.loc[year_df[season_col].eq(season)]) for season in season_order
        )
        for season, group in scopes:
            values = group[target_col].dropna().astype(float)
            if values.empty:
                continue
            q1, median, q3 = values.quantile([0.25, 0.50, 0.75])
            iqr = q3 - q1
            lower_fence = q1 - iqr_multiplier * iqr
            upper_fence = q3 + iqr_multiplier * iqr
            outlier_mask = values.lt(lower_fence) | values.gt(upper_fence)
            rows.append(
                {
                    "Meteorological year": int(year),
                    "Season": season,
                    "N": len(values),
                    "Q1": q1,
                    "Median": median,
                    "Q3": q3,
                    "Lower fence": lower_fence,
                    "Upper fence": upper_fence,
                    "Outliers": int(outlier_mask.sum()),
                    "Outlier rate (%)": 100 * outlier_mask.mean(),
                    "Maximum": values.max(),
                }
            )

    return pd.DataFrame(rows)


def plot_iqr_outliers_by_season_year(
    dataframe: pd.DataFrame,
    meteorological_year_value: int,
    target_col: str = "Rented Bike Count",
    datetime_col: str = "DateTime",
    hour_col: str = "Hour",
    season_col: str = "Seasons",
    iqr_multiplier: float = 1.5,
    lang=None,
    labels: Optional[Dict[str, str]] = None,
) -> Tuple[plt.Figure, np.ndarray, pd.DataFrame]:
    """Plot one meteorological year in All/Winter/Spring/Summer/Autumn panels."""
    lang = _resolve_lang(lang)
    plot_labels = lang(
        {
            "suptitle": "Outliers da demanda — ano meteorológico {year}",
            "all": "Ano completo",
            "Winter": "Inverno",
            "Spring": "Primavera",
            "Summer": "Verão",
            "Autumn": "Outono",
            "xlabel": "Data",
            "ylabel": "Bicicletas alugadas",
            "series": "Demanda horária",
            "outliers": "Outliers pelo IQR",
            "upper_fence": "Limite superior do IQR",
            "no_obs": "sem observações",
            "count_word": "outliers",
        }
    )
    if labels is not None:
        plot_labels.update(labels)

    df = dataframe.copy(deep=True)
    df["_timestamp"] = _hourly_timestamp(df, datetime_col, hour_col)
    df["_meteorological_year"] = df["_timestamp"].dt.year + df["_timestamp"].dt.month.eq(12).astype(
        int
    )
    year_source = df.loc[df["_meteorological_year"].eq(int(meteorological_year_value))].copy()
    year_df = year_source.sort_values("_timestamp").set_index("_timestamp")
    if year_df.empty:
        raise ValueError(
            f"No observations found for meteorological year {meteorological_year_value}."
        )

    summary = iqr_outlier_summary_by_season_year(
        year_source.drop(columns=["_timestamp", "_meteorological_year"]),
        target_col=target_col,
        datetime_col=datetime_col,
        hour_col=hour_col,
        season_col=season_col,
        iqr_multiplier=iqr_multiplier,
    )
    summary = summary.loc[
        summary["Meteorological year"].eq(int(meteorological_year_value))
    ].reset_index(drop=True)

    panels = [
        ("All", plot_labels["all"], pd.Series(True, index=year_df.index)),
        *[
            (season, plot_labels[season], year_df[season_col].eq(season))
            for season in ["Winter", "Spring", "Summer", "Autumn"]
        ],
    ]

    with plt.style.context(["default"]):
        set_graph_parameters()
        fig, axes = plt.subplots(nrows=5, figsize=(18, 17), constrained_layout=True)
        for ax, (season_key, title, mask) in zip(axes, panels):
            group = year_df.loc[mask]
            if group.empty:
                ax.set_title(f"{title} — {plot_labels['no_obs']}")
                ax.set_axis_off()
                continue
            values = group[target_col].astype(float)
            row = summary.loc[summary["Season"].eq(season_key)].iloc[0]
            outlier_mask = values.lt(row["Lower fence"]) | values.gt(row["Upper fence"])

            ax.plot(
                values.index,
                values.values,
                color="#4C72B0",
                alpha=0.60,
                linewidth=0.75,
                label=plot_labels["series"],
            )
            ax.scatter(
                values.index[outlier_mask],
                values.loc[outlier_mask],
                color="#C44E52",
                marker="x",
                s=26,
                linewidths=1.0,
                label=plot_labels["outliers"],
                zorder=3,
            )
            ax.axhline(
                row["Upper fence"],
                color="#DD8452",
                linestyle="--",
                linewidth=1.0,
                label=plot_labels["upper_fence"],
            )
            ax.set_title(
                f"{title} — {int(row['Outliers'])} {plot_labels['count_word']} "
                f"({row['Outlier rate (%)']:.2f}%)"
            )
            ax.set_ylabel(plot_labels["ylabel"])
            ax.set_xlabel(plot_labels["xlabel"])
            locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
            ax.legend(loc="upper left", fontsize=8, ncols=3)

        fig.suptitle(
            plot_labels["suptitle"].format(year=int(meteorological_year_value)),
            fontsize=17,
        )

    return fig, axes, summary


# ---------------------------------------------------------------------------
# rainfall_event
# ---------------------------------------------------------------------------


def rainfall_event(data: pd.Series) -> pd.Series:
    """Return only observations where rainfall > 0 (i.e. actual rainfall events)."""
    return data[data > 0]


def plot_rainfall(
    outliers: pd.Series,
    data: pd.Series,
    temperature: pd.Series,
    ax: plt.Axes,
    method: str = "KNN",
    halignment: str = "right",
    valignment: str = "bottom",
    labels: bool = False,
    all_data: bool = False,
    label_shift: tuple = (0.1, 55),
    lang=None,
    event_label: Optional[str] = None,
) -> plt.Axes:
    """Plot a time series with weather events highlighted.

    Parameters
    ----------
    outliers:
        Weather event indices (output of ``rainfall_event``).
    data:
        Full time series of bike counts.
    temperature:
        Temperature Series aligned to ``data``.
    ax:
        Matplotlib Axes to draw on.
    method:
        Season label for the subplot title.
    labels:
        If True, show temperature annotations and min/max outside the graph.
    event_label:
        Legend label for the event markers.  Defaults to ``"Rainfall"`` (or
        its translated equivalent) when ``None``.
    """
    lang = _resolve_lang(lang)
    plot_labels = lang(
        {
            "xlabel": "Date",
            "ylabel": "# of Rented Bikes",
            "rented_bikes": "Rented Bikes",
            "rainfall_legend": "Rainfall",
        }
    )
    legend_event = event_label if event_label is not None else plot_labels["rainfall_legend"]
    ax_1 = data.plot(alpha=0.6, linewidth=1.3, ax=ax)

    outlier_temps = temperature.loc[outliers.index]
    min_temp = outlier_temps.min()
    max_temp = outlier_temps.max()

    if labels:
        data.loc[outliers.index].plot(ax=ax_1, style="rx", linewidth=0.7)

        ax_1.text(
            1.03,
            0.65,
            f"Min Temp: {min_temp:.1f} deg C\nMax Temp: {max_temp:.1f} deg C",
            transform=ax_1.transAxes,
            fontsize="small",
            verticalalignment="top",
            horizontalalignment="left",
        )
    else:
        data.loc[outliers.index].plot(
            ax=ax_1, style="rx", linewidth=0.6, markerfacecolor="royalblue"
        )

    ax_1.set_title(f"{method}")
    ax_1.set_xlabel(plot_labels["xlabel"])
    ax_1.set_ylabel(plot_labels["ylabel"])
    ax_1.legend(
        [plot_labels["rented_bikes"], legend_event],
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.3,
    )
    return ax_1


def plot_rainfall_by_season(
    dataframe: pd.DataFrame,
    lang=None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot rainfall events vs bike count for Winter / Spring / Summer / Autumn."""
    lang = _resolve_lang(lang)
    season_labels = lang(
        {
            "suptitle": "Rented Bike under Rainfall",
            "winter_season": "Winter Season",
            "spring_season": "Spring Season",
            "summer_season": "Summer Season",
            "autumn_season": "Autumn Season",
        }
    )
    season_label_map = {
        "Winter": season_labels["winter_season"],
        "Spring": season_labels["spring_season"],
        "Summer": season_labels["summer_season"],
        "Autumn": season_labels["autumn_season"],
    }
    with plt.style.context(["default"]):
        set_graph_parameters()
        fig, ax = plt.subplots(nrows=4, figsize=(20, 22))

        for i, season in enumerate(["Winter", "Spring", "Summer", "Autumn"]):
            mask = dataframe["Seasons"] == season
            tx = dataframe.loc[mask, "Rented Bike Count"]
            temperature = dataframe.loc[mask, "Temperature(C)"]
            rainfall_evt = rainfall_event(dataframe.loc[mask, "Rainfall(mm)"])
            plot_rainfall(
                outliers=rainfall_evt,
                data=tx,
                temperature=temperature,
                ax=ax[i],
                labels=False,
                method=season_label_map[season],
                lang=lang,
            )

        fig.suptitle(season_labels["suptitle"], fontsize=18, y=1)
        plt.tight_layout()

    return fig, ax


# ---------------------------------------------------------------------------
# plot_snowfall_by_season
# ---------------------------------------------------------------------------


def plot_snowfall_by_season(
    dataframe: pd.DataFrame,
    lang=None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot snowfall events vs bike count for Winter / Spring / Summer / Autumn.

    Mirrors :func:`plot_rainfall_by_season` but uses the ``Snowfall (cm)``
    column to identify weather events.  Snowfall is concentrated in Winter;
    the remaining subplots are kept for completeness and season-level context.

    Parameters
    ----------
    dataframe:
        DataFrame containing at least the columns ``Seasons``,
        ``Rented Bike Count``, ``Temperature(C)``, and ``Snowfall (cm)``.
    lang:
        Optional :class:`~src.i18n.LangMap` instance for label translation.
        Defaults to English when ``None``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the four subplots.
    ax : numpy.ndarray
        Array of four :class:`matplotlib.axes.Axes` objects, one per season.

    Examples
    --------
    >>> fig, ax = plot_snowfall_by_season(transformed_df, lang=lang)
    >>> plt.show()
    """
    lang = _resolve_lang(lang)
    season_labels = lang(
        {
            "suptitle": "Rented Bike under Snowfall",
            "winter_season": "Winter Season",
            "spring_season": "Spring Season",
            "summer_season": "Summer Season",
            "autumn_season": "Autumn Season",
        }
    )
    snowfall_label = lang({"snowfall_legend": "Snowfall"})["snowfall_legend"]
    season_label_map = {
        "Winter": season_labels["winter_season"],
        "Spring": season_labels["spring_season"],
        "Summer": season_labels["summer_season"],
        "Autumn": season_labels["autumn_season"],
    }
    with plt.style.context(["default"]):
        set_graph_parameters()
        fig, ax = plt.subplots(nrows=4, figsize=(20, 22))

        for i, season in enumerate(["Winter", "Spring", "Summer", "Autumn"]):
            mask = dataframe["Seasons"] == season
            tx = dataframe.loc[mask, "Rented Bike Count"]
            temperature = dataframe.loc[mask, "Temperature(C)"]
            snowfall_evt = rainfall_event(dataframe.loc[mask, "Snowfall (cm)"])
            plot_rainfall(
                outliers=snowfall_evt,
                data=tx,
                temperature=temperature,
                ax=ax[i],
                labels=False,
                method=season_label_map[season],
                lang=lang,
                event_label=snowfall_label,
            )

        fig.suptitle(season_labels["suptitle"], fontsize=18, y=1)
        plt.tight_layout()

    return fig, ax


def complete_meteorological_years(summary: pd.DataFrame, min_hours: int = 4380) -> list:
    """Meteorological years whose 'All' scope has at least ``min_hours`` rows.

    Uses the summary from :func:`iqr_outlier_summary_by_season_year`; excludes
    partial edge years (e.g. the 2015 fragment) from the display matrix.
    """
    return summary.loc[
        summary["Season"].eq("All") & summary["N"].ge(min_hours),
        "Meteorological year",
    ].tolist()


def outlier_rate_matrix(summary: pd.DataFrame, complete_years: list, lang=None) -> pd.DataFrame:
    """Pivot outlier rates into a (meteorological year × season) matrix.

    Internal season keys are mapped to localized display names for the columns
    only; the analytic ``summary`` contract is untouched.
    """
    lang = _resolve_lang(lang)
    season_names = lang(
        {
            "All": "Ano completo",
            "Winter": "Inverno",
            "Spring": "Primavera",
            "Summer": "Verão",
            "Autumn": "Outono",
        }
    )
    column_order = [season_names[k] for k in ["All", "Winter", "Spring", "Summer", "Autumn"]]
    return (
        summary.loc[summary["Meteorological year"].isin(complete_years)]
        .assign(_season=lambda d: d["Season"].map(season_names))
        .pivot(index="Meteorological year", columns="_season", values="Outlier rate (%)")
        .reindex(columns=column_order)
    )
