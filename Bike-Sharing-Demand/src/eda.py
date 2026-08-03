"""EDA and distribution plot functions.

Faithful port of notebook cells:
  [17] skewness_measure
  [17] histogram_with_kde
  [18] histogram_with_kde_by_season
  [19] decompose_series
  [19] plot_label
  [20] distribution_on_target
  [21] distribution_by_season_on_weekday
  [21] barplot_by_season_on_weekday
  [23] plot_pointplot
  [25] distribution_pointplot
  [42] plot_boxplot_comparation

All functions return ``fig`` (or ``fig, axes``) instead of calling
``plt.show()`` internally (display concern handled by notebooks, §4).
The in-place mutation ``dataframe['Rented Bike Count'] = ...astype(float)``
inside ``distribution_on_target`` is preserved exactly per §4.

Functions that produce output for the user accept an optional ``lang``
parameter (``LangMap`` instance) for multilang support per §5.2.
"""

from __future__ import annotations

from typing import Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

from src.i18n import localize_table, resolve_lang as _resolve_lang
from src.plotting import set_graph_parameters


# ---------------------------------------------------------------------------
# skewness_measure
# ---------------------------------------------------------------------------


def skewness_measure(
    data: pd.DataFrame,
    x: str,
    detail: str,
) -> tuple[str, str, str]:
    """Measure and interpret skewness and kurtosis of a dataframe variable.

    Parameters
    ----------
    data:
        DataFrame to be measured.
    x:
        The variable column name.
    detail:
        Secondary label (e.g. season name) used in the returned text.

    Returns
    -------
    direction : str
        "Left", "Right", or "Neutral".
    intensity : str
        "is fairly symmetrical", "is moderately", or "is highly".
    text : str
        Descriptive text about the skewness and kurtosis.
    """
    skewness = data[x].skew()
    kurtosis = data[x].kurt()
    mean = data[x].mean()

    intensity_thresholds = [
        (-0.5, 0.5, "is fairly symmetrical"),
        (-1, 1, "is moderately"),
        (-float("inf"), -1, "is highly"),
        (1, float("inf"), "is highly"),
    ]

    kurtosis_thresholds = [
        (-float("inf"), 0, "Platykurtic"),
        (0, mean, "Mesokurtic"),
        (mean, float("inf"), "Leptokurtic"),
    ]

    def get_intensity(s):
        for threshold_min, threshold_max, label in intensity_thresholds:
            if threshold_min <= abs(s) <= threshold_max:
                return label
        return "unknown"

    def get_kurtosis(k):
        for threshold_min, threshold_max, label in kurtosis_thresholds:
            if threshold_min <= k <= threshold_max:
                return label
        return "unknown"

    direction = "Neutral" if abs(skewness) < 1e-6 else ("Left" if skewness < 0 else "Right")
    intensity = get_intensity(skewness)
    mess_kurtosis = get_kurtosis(kurtosis)

    text = (
        f"Skewness for {x} when Season is {detail}:\n"
        f"    Skew: {skewness:.4f}\n"
        f"    Kurtosis: {kurtosis:.4f}\n"
        f"    {intensity} {direction} Skewed and {mess_kurtosis}.\n"
    )

    return direction, intensity, text


# ---------------------------------------------------------------------------
# histogram_with_kde
# ---------------------------------------------------------------------------
def histogram_with_kde(
    dataframe: pd.DataFrame,
    text1: str,
    lang=None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Histogram with KDE for 'Rented Bike Count' and skewness interpretation text."""
    with plt.style.context(["default"]):
        set_graph_parameters()
        # Initialize plot
        fig, ax = plt.subplots()  # Adjust the figure size figsize=(10, 6)

        sns.histplot(
            dataframe["Rented Bike Count"],
            kde=True,
            color="darkblue",
        )
        plt.title("Number of Rented Bike by Hour")  # Adjust title font size

        # Adjust the position of the text
        x_position = dataframe["Rented Bike Count"].max() * 0.5
        y_position = ax.get_ylim()[1] * 0.9

        plt.text(
            x_position,
            y_position,
            text1,
            fontsize=10,
            color="black",
            ha="center",
            va="top",
            wrap=True,
        )  # Adjust text properties

        plt.tight_layout()  # Adjusts subplot params for better layout

        return fig, ax


# ---------------------------------------------------------------------------
# histogram_with_kde_by_season
# ---------------------------------------------------------------------------
def histogram_with_kde_by_season(
    dataframe: pd.DataFrame,
    text1: str,
    lang=None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Histogram with KDE for 'Rented Bike Count' by Season.
    """

    with plt.style.context(["default"]):
        set_graph_parameters()
        # Init plot
        fig, ax = plt.subplots()

        # Plot
        ax = sns.kdeplot(
            dataframe[(dataframe["Seasons"] == "Winter")]["Rented Bike Count"],
            color="lightblue",
            fill=True,
        )

        ax = sns.kdeplot(
            dataframe[(dataframe["Seasons"] == "Spring")]["Rented Bike Count"],
            color="orange",
            fill=True,
            ax=ax,
        )

        ax = sns.kdeplot(
            dataframe[(dataframe["Seasons"] == "Summer")]["Rented Bike Count"],
            color="red",
            fill=True,
            ax=ax,
        )

        ax = sns.kdeplot(
            dataframe[(dataframe["Seasons"] == "Autumn")]["Rented Bike Count"],
            color="cyan",
            fill=True,
            ax=ax,
        )

        ax.legend(["Winter", "Spring", "Summer", "Autumn"], loc="upper right")
        plt.title("Number of Rented bike by Season")
        return fig, ax


# ---------------------------------------------------------------------------
# plot_label
# ---------------------------------------------------------------------------
def plot_label(dataset: pd.DataFrame, title=""):
    with plt.style.context(["default"]):
        set_graph_parameters()
        # Plot the values of the airline_decomposed DataFrame
        ax = dataset.plot(figsize=(20, 3), fontsize=15, linewidth=0.7)

        # Specify axis labels
        ax.set_xlabel("Date", fontsize=15)
        plt.legend(fontsize=15)
        plt.title(title)
        # plt.show()


# ---------------------------------------------------------------------------
# decompose_series
# ---------------------------------------------------------------------------
def decompose_series(dataset: pd.Series, date_index: pd.Series = None):
    if date_index is not None:
        dataset = dataset.copy()
        dataset.index = date_index.values
    # period=24: one day of hourly observations as the seasonal cycle
    decomposition = sm.tsa.seasonal_decompose(dataset, period=24)

    trend = decomposition.trend
    seasonal = decomposition._seasonal

    decomposed = pd.concat([trend, seasonal], axis=1)
    return decomposed


# ---------------------------------------------------------------------------
# distribution_on_target
# ---------------------------------------------------------------------------


def distribution_on_target(
    dataframe: pd.DataFrame,
    lang=None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Box plots of 'Rented Bike Count' against various categorical features.

    ⚠️ Preserves in-place mutation:
    ``dataframe['Rented Bike Count'] = dataframe['Rented Bike Count'].astype(float)``
    exactly as in the source notebook (§4).
    """
    lang = _resolve_lang(lang)
    labels = lang(
        {
            "box_count": "Box Plot On Count",
            "box_season": "Box Plot On Count by Season",
            "box_hour": "Box Plot On Count by Hour Of The Day",
            "hour_xlabel": "Hour Of The Day",
            "box_working_day": "Box Plot On Count by Working Day",
            "working_day_xlabel": "Working Day",
            "box_weekday": "Rented Bike by Weekday",
            "box_holiday": "Rented Bike by Holiday",
        }
    )
    dataframe["Rented Bike Count"] = dataframe["Rented Bike Count"].astype(float)
    with plt.style.context(["default"]):
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 8), sharey=True)

        axes[0][0].boxplot(
            dataframe["Rented Bike Count"],
            flierprops=dict(marker="o", markersize=2),
            widths=0.7,
        )
        axes[0][0].set(title=labels["box_count"])

        seasons = dataframe["Seasons"].unique()
        axes[0][1].boxplot(
            [dataframe["Rented Bike Count"][dataframe["Seasons"] == season] for season in seasons],
            flierprops=dict(marker="o", markersize=2),
            widths=0.7,
        )
        axes[0][1].set(title=labels["box_season"], xticklabels=seasons)

        hours = sorted(dataframe["Hour"].unique())
        axes[0][2].boxplot(
            [dataframe["Rented Bike Count"][dataframe["Hour"] == hour] for hour in hours],
            flierprops=dict(marker="o", markersize=2),
            widths=0.7,
        )
        axes[0][2].set(
            title=labels["box_hour"],
            xticklabels=hours,
            xlabel=labels["hour_xlabel"],
        )
        axes[0][2].tick_params(axis="x", rotation=45)

        functioning_days = dataframe["Functioning Day"].unique()
        axes[1][1].boxplot(
            [
                dataframe["Rented Bike Count"][dataframe["Functioning Day"] == day]
                for day in functioning_days
            ],
            flierprops=dict(marker="o", markersize=2),
            widths=0.7,
        )
        axes[1][1].set(
            title=labels["box_working_day"],
            xticklabels=functioning_days,
            xlabel=labels["working_day_xlabel"],
        )

        weekday_order = [
            "Sunday",
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
        ]
        axes[1][0].boxplot(
            [dataframe["Rented Bike Count"][dataframe["Weekday"] == day] for day in weekday_order],
            flierprops=dict(marker="o", markersize=2),
            widths=0.7,
        )
        axes[1][0].set(title=labels["box_weekday"], xticklabels=weekday_order)
        axes[1][0].tick_params(axis="x", rotation=45)

        holidays = dataframe["Holiday"].unique()
        axes[1][2].boxplot(
            [
                dataframe["Rented Bike Count"][dataframe["Holiday"] == holiday]
                for holiday in holidays
            ],
            flierprops=dict(marker="o", markersize=2),
            widths=0.7,
        )
        axes[1][2].set(title=labels["box_holiday"], xticklabels=holidays)
        axes[1][2].tick_params(axis="x", rotation=45)

        plt.title("")
        plt.tight_layout()

    return fig, axes


# ---------------------------------------------------------------------------
# distribution_by_season_on_weekday
# ---------------------------------------------------------------------------


def distribution_by_season_on_weekday(
    dataframe: pd.DataFrame,
    lang=None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Multi-season weekday comparison box plots."""
    lang = _resolve_lang(lang)
    labels = lang(
        {
            "title": "Rented Bike on Weekday by Season",
            "legend_title": "Seasons",
        }
    )
    with plt.style.context(["default"]):
        weekday_order = [
            "Sunday",
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
        ]
        season_order = ["Winter", "Spring", "Summer", "Autumn"]
        num_seasons = len(season_order)

        fig, ax = plt.subplots(figsize=(10, 6))
        width = 0.15
        colors = ["pink", "lightblue", "lightgreen", "yellow"]

        for i, (season, color) in enumerate(zip(season_order, colors)):
            positions = np.arange(len(weekday_order)) - (num_seasons / 2 - i) * width
            bplot = ax.boxplot(
                [
                    dataframe["Rented Bike Count"][
                        (dataframe["Weekday"] == day) & (dataframe["Seasons"] == season)
                    ]
                    for day in weekday_order
                ],
                positions=positions,
                widths=width,
                patch_artist=True,
                flierprops=dict(marker="o", markersize=2, color="black"),
            )
            for box in bplot["boxes"]:
                box.set_facecolor(color)

        ax.set_xticks(np.arange(len(weekday_order)))
        ax.set_xticklabels(weekday_order)
        ax.set_title(labels["title"])

        handles = [plt.Line2D([0], [0], color=color, lw=4) for color in colors]
        ax.legend(
            handles,
            season_order,
            title=labels["legend_title"],
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )

        plt.tight_layout()

    return fig, ax


# ---------------------------------------------------------------------------
# barplot_by_season_on_weekday
# ---------------------------------------------------------------------------
def barplot_by_season_on_weekday(
    dataframe: pd.DataFrame,
    lang=None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Bar plot of mean bike count by weekday and season."""
    lang = _resolve_lang(lang)
    with plt.style.context(["default"]):
        set_graph_parameters()

        # Ensure 'Weekday' is ordered; season colouring comes from the hue below.
        weekdays = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

        # Aggregate the data
        grouped_data = (
            dataframe.groupby(["Weekday", "Seasons"], observed=False)["Rented Bike Count"]
            .sum()
            .reset_index()
        )

        # Map weekday names to integer positions so matplotlib receives numeric x values
        # (day names are date-parseable and would trigger a matplotlib category warning)
        weekday_to_pos = {day: i for i, day in enumerate(weekdays)}
        grouped_data["_weekday_pos"] = grouped_data["Weekday"].map(weekday_to_pos)

        # Plotting
        fig, ax = plt.subplots(figsize=(16, 6))
        ax = sns.barplot(
            x="_weekday_pos",
            y="Rented Bike Count",
            hue="Seasons",
            data=grouped_data,
            order=list(range(len(weekdays))),
            native_scale=True,
            linewidth=0.7,  # Lines width
            saturation=0.95,  # Fill color saturation
            width=0.6,
            dodge=True,
        )

        ax.set_xticks(range(len(weekdays)))
        ax.set_xticklabels(weekdays)
        plt.title("Sum of Rented Bike Count by Weekday on Season")
        plt.xlabel("Weekday")
        plt.ylabel("Rented Bike Count")

        # Add a horizontal line at y = 200000
        plt.axhline(200000, color="orange", linestyle="--")

        # plt.xticks(rotation=45)
        plt.legend(title="Seasons")
        # ax.legend(loc='upper left', ncols=4)
        # Adjust legend placement
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
        # plt.show()

    """with plt.style.context(["default"]):
        figure, ax = plt.subplots()
        order = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        seasons = dataframe["Seasons"].unique()
        colors = plt.cm.Set2(np.linspace(0, 1, len(seasons)))

        for color, season in zip(colors, seasons):
            means = (
                dataframe[dataframe["Seasons"] == season]
                .groupby("Weekday", observed=False)["Rented Bike Count"]
                .mean()
                .reindex(order)
            )
            ax.bar(order, means, color=color, label=season)
            ax.tick_params(axis="x", rotation=45)

        ax.set_title(labels["title"])
        ax.legend(title=labels["legend_title"], bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()"""

    return fig, ax


# ---------------------------------------------------------------------------
# plot_pointplot
# ---------------------------------------------------------------------------
def plot_pointplot(
    dataframe: pd.DataFrame,
    lang=None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Line plot of mean bike count by weekday and season."""
    lang = _resolve_lang(lang)
    labels = lang(
        {
            "title": "Mean of Rented Bikes on Weekday by Season",
            "legend_title": "Seasons",
        }
    )
    with plt.style.context(["default"]):
        set_graph_parameters()
        figure, ax = plt.subplots()
        order = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        positions = list(range(len(order)))
        seasons = dataframe["Seasons"].unique()
        colors = plt.cm.Set2(np.linspace(0, 1, len(seasons)))

        for color, season in zip(colors, seasons):
            means = (
                dataframe[dataframe["Seasons"] == season]
                .groupby("Weekday", observed=False)["Rented Bike Count"]
                .mean()
                .reindex(order)
            )
            ax.plot(positions, means.values, marker="o", color=color, label=season)
            ax.tick_params(axis="x", rotation=45)

        ax.set_xticks(positions)
        ax.set_xticklabels(order)
        ax.set_title(labels["title"])
        ax.legend(title=labels["legend_title"], bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()

    return figure, ax


# ---------------------------------------------------------------------------
# distribution_pointplot
# ---------------------------------------------------------------------------


def distribution_pointplot(
    dataframe: pd.DataFrame,
    lang=None,
) -> Tuple[plt.Figure, np.ndarray]:
    """4-subplot point plots by Functioning Day / Holiday / Weekday / Season."""
    lang = _resolve_lang(lang)
    labels = lang(
        {
            "title_functioning": "Rented bike by Functioning Day",
            "title_holiday": "Rented bike by Holiday",
            "title_weekday": "Rented bike by Weekday",
            "title_seasons": "Rented bike by Functioning Seasons",
            "suptitle": "Rented Bike Hourly Distribution",
        }
    )
    with plt.style.context(["default"]):
        set_graph_parameters()
        figure, axes = plt.subplots(nrows=4, figsize=(18, 10))
        sns.set(rc={"lines.linewidth": 0.7})

        weekday_order = [
            "Sunday",
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
        ]
        season_order = ["Winter", "Spring", "Summer", "Autumn"]

        def set2_palette(column=None, order=None):
            n_colors = len(order) if order is not None else dataframe[column].nunique(dropna=True)
            return sns.color_palette("Set2", n_colors=n_colors)

        sns.pointplot(
            x="Hour",
            y="Rented Bike Count",
            data=dataframe,
            hue="Functioning Day",
            palette=set2_palette("Functioning Day"),
            native_scale=True,
            ax=axes[0],
        )
        axes[0].legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.2)
        axes[0].set(title=labels["title_functioning"])

        sns.pointplot(
            x="Hour",
            y="Rented Bike Count",
            data=dataframe,
            hue="Holiday",
            palette=set2_palette("Holiday"),
            native_scale=True,
            ax=axes[1],
        )
        axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.2)
        axes[1].set(title=labels["title_holiday"])

        sns.pointplot(
            x="Hour",
            y="Rented Bike Count",
            data=dataframe,
            hue="Weekday",
            hue_order=weekday_order,
            palette=set2_palette(order=weekday_order),
            native_scale=True,
            ax=axes[2],
        )
        axes[2].legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.2)
        axes[2].set(title=labels["title_weekday"])

        sns.pointplot(
            x="Hour",
            y="Rented Bike Count",
            data=dataframe,
            hue="Seasons",
            hue_order=season_order,
            palette=set2_palette(order=season_order),
            native_scale=True,
            ax=axes[3],
        )
        axes[3].legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.2)
        axes[3].set(title=labels["title_seasons"])

        plt.suptitle(labels["suptitle"])
        plt.tight_layout()

    return figure, axes


# ---------------------------------------------------------------------------
# distribution_all_others
# ---------------------------------------------------------------------------


def distribution_all_others(dataframe, variables):
    set_graph_parameters()

    if "DateTime" in dataframe.columns:
        dataframe = dataframe.set_index("DateTime")

    # Number of variables
    num_vars = len(variables)

    # Create a figure with custom subplot size
    fig = plt.figure(figsize=(22, 34))
    gs = gridspec.GridSpec(num_vars, 2, width_ratios=[5, 1])

    # Loop through the variables
    for i, var in enumerate(variables):
        # Plotting the line plot
        ax = plt.subplot(gs[i, 0])
        dataframe[var].rolling(window=52).mean().plot(ax=ax, linewidth=0.7)
        ax.set_title(f"{var} over Time")
        ax.grid(True)

        # Plotting the histogram
        ax_hist = plt.subplot(gs[i, 1], sharey=ax)
        # sns.histplot(dataframe[var], ax=ax_hist, vertical=True) #
        ax_hist.hist(dataframe[var], orientation="horizontal", density=True)
        ax_hist.grid(True)

    plt.suptitle("Others Variables Plot Analysis", y=1)
    plt.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# plot_boxplot_comparation
# ---------------------------------------------------------------------------


def plot_boxplot_comparation(
    data: pd.DataFrame,
    title: str = "",
    lang=None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Box plot comparison for rainy/snowy conditions vs non-rainy/non-snowy."""
    lang = _resolve_lang(lang)
    labels = lang(
        {
            "xlabel": "Groups",
            "ylabel": "Bike Count",
        }
    )
    with plt.style.context(["default"]):
        set_graph_parameters()
        fig, ax = plt.subplots(figsize=(12, 3))

        sns.boxplot(
            x="Condition",
            y="Bike Count",
            data=data,
            showmeans=True,
            fliersize=2,
            linewidth=0.7,
            palette="hls",
            hue="Condition",
            ax=ax,
        )

        ax.set_title(title)
        ax.set_xlabel(labels["xlabel"])
        ax.set_ylabel(labels["ylabel"])

    return fig, ax


# ---------------------------------------------------------------------------
# Notebook 01 — integrity, coverage, missingness, temporal profile, growth,
# weather quality. Pure report builders return display-ready DataFrames;
# plot builders return (fig, axes) and never call plt.show().
# ---------------------------------------------------------------------------

_SEASON_ORDER = ["Winter", "Spring", "Summer", "Autumn"]

_SEASON_VALUE_LABELS = {
    "Winter": "Inverno",
    "Spring": "Primavera",
    "Summer": "Verão",
    "Autumn": "Outono",
}

_PREVIEW_VALUE_LABELS = {
    **_SEASON_VALUE_LABELS,
    "No Holiday": "Sem feriado",
    "Holiday": "Feriado",
    "Yes": "Sim",
    "No": "Não",
}


def _season_labels(lang) -> dict:
    """Localized display names for the four seasons (internal keys unchanged)."""
    return lang(_SEASON_VALUE_LABELS)


def localize_preview(dataframe: pd.DataFrame, lang=None) -> pd.DataFrame:
    """Cópia de exibição com valores de Seasons/Holiday/Functioning Day em PT-BR.

    Usada nos previews iniciais dos notebooks; o dataframe original e o schema
    interno (nomes de colunas, valores usados por outras funções) não mudam.
    """
    lang = _resolve_lang(lang)
    return localize_table(
        dataframe,
        lang,
        columns={},
        value_columns=("Seasons", "Holiday", "Functioning Day"),
        value_labels=_PREVIEW_VALUE_LABELS,
    )


def schema_summary(df: pd.DataFrame, lang=None) -> pd.DataFrame:
    """Per-column schema table: dtype, non-null, missing and distinct counts."""
    lang = _resolve_lang(lang)
    labels = lang(
        {
            "dtype": "Tipo",
            "non_null": "Não nulos",
            "missing": "Ausentes",
            "distinct": "Valores únicos",
        }
    )
    return pd.DataFrame(
        {
            labels["dtype"]: df.dtypes.astype(str),
            labels["non_null"]: df.notna().sum(),
            labels["missing"]: df.isna().sum(),
            labels["distinct"]: df.nunique(dropna=True),
        }
    )


def temporal_coverage(
    df: pd.DataFrame, timestamp: pd.Series, lang=None
) -> Tuple[pd.DataFrame, dict]:
    """Observed vs expected hourly coverage per year.

    Returns the per-year coverage table plus a ``stats`` dict with the totals
    used by the notebook's summary line (observed rows, expected hours, missing
    hours and duplicate count).
    """
    lang = _resolve_lang(lang)
    labels = lang(
        {
            "observed": "Horas observadas",
            "expected": "Horas esperadas no período coberto",
            "missing": "Horas ausentes",
            "coverage": "Cobertura (%)",
        }
    )
    full_range = pd.date_range(timestamp.min(), timestamp.max(), freq="h")
    missing_timestamps = full_range.difference(timestamp)

    observed_by_year = timestamp.dt.year.value_counts().sort_index()
    expected_by_year = pd.Series(full_range.year).value_counts().sort_index()
    coverage = (
        pd.DataFrame({labels["observed"]: observed_by_year, labels["expected"]: expected_by_year})
        .fillna(0)
        .astype(int)
    )
    coverage[labels["missing"]] = coverage[labels["expected"]] - coverage[labels["observed"]]
    coverage[labels["coverage"]] = 100 * coverage[labels["observed"]] / coverage[labels["expected"]]

    stats = {
        "observed_rows": len(df),
        "expected_hours": len(full_range),
        "missing_hours": len(missing_timestamps),
        "duplicate_count": int(timestamp.duplicated().sum()),
        "coverage_col": labels["coverage"],
    }
    return coverage, stats


def missing_values_summary(df: pd.DataFrame, lang=None) -> Tuple[pd.DataFrame, dict]:
    """Per-column missing-value table plus row/total counts."""
    lang = _resolve_lang(lang)
    labels = lang(
        {
            "dtype": "Tipo",
            "missing": "Valores ausentes",
            "pct": "Percentual ausente (%)",
        }
    )
    summary = pd.DataFrame(
        {
            labels["dtype"]: df.dtypes.astype(str),
            labels["missing"]: df.isna().sum(),
            labels["pct"]: df.isna().mean() * 100,
        }
    ).sort_values([labels["missing"], labels["pct"]], ascending=False)

    stats = {
        "total_missing": int(df.isna().sum().sum()),
        "rows_with_missing": int(df.isna().any(axis=1).sum()),
        "n_rows": len(df),
        "missing_col": labels["missing"],
        "pct_col": labels["pct"],
    }
    return summary, stats


def observed_period_summary(
    df: pd.DataFrame, timestamp: pd.Series, lang=None
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Period overview and consecutive-gap runs.

    Returns ``(period_table, gap_runs, stats)``. ``gap_runs`` groups the missing
    hourly timestamps into consecutive intervals (start, end, length), sorted by
    length; it is empty when no hour is missing.
    """
    lang = _resolve_lang(lang)
    metric = lang(
        {
            "start": "Timestamp inicial",
            "end": "Timestamp final",
            "observed_rows": "Linhas observadas",
            "unique_hours": "Timestamps horários únicos",
            "expected_hours": "Timestamps horários esperados",
            "missing_hours": "Timestamps horários ausentes",
            "dup_hours": "Timestamps horários duplicados",
            "observed_days": "Dias-calendário observados",
            "years": "Anos-calendário cobertos",
            "metric": "Métrica",
            "value": "Valor",
            "run_start": "Início",
            "run_end": "Fim",
            "run_hours": "Horas",
        }
    )
    ts = timestamp.sort_values()
    period_start, period_end = ts.min(), ts.max()
    observed_hours = len(ts)
    expected_hours = int(((period_end - period_start) / pd.Timedelta(hours=1)) + 1)
    missing_hour_count = expected_hours - ts.nunique()

    period_table = pd.DataFrame(
        {
            metric["metric"]: [
                metric["start"],
                metric["end"],
                metric["observed_rows"],
                metric["unique_hours"],
                metric["expected_hours"],
                metric["missing_hours"],
                metric["dup_hours"],
                metric["observed_days"],
                metric["years"],
            ],
            metric["value"]: [
                period_start,
                period_end,
                observed_hours,
                ts.nunique(),
                expected_hours,
                missing_hour_count,
                int(ts.duplicated().sum()),
                ts.dt.normalize().nunique(),
                ", ".join(map(str, sorted(ts.dt.year.unique()))),
            ],
        }
    )

    full_range = pd.date_range(period_start, period_end, freq="h")
    missing_series = pd.Series(full_range.difference(timestamp), name=metric["run_end"])
    if len(missing_series):
        run_id = missing_series.diff().ne(pd.Timedelta(hours=1)).cumsum()
        gap_runs = (
            missing_series.to_frame()
            .groupby(run_id)
            .agg(
                **{
                    metric["run_start"]: (metric["run_end"], "min"),
                    metric["run_end"]: (metric["run_end"], "max"),
                    metric["run_hours"]: (metric["run_end"], "size"),
                }
            )
            .sort_values([metric["run_hours"], metric["run_start"]], ascending=[False, True])
            .reset_index(drop=True)
        )
    else:
        gap_runs = pd.DataFrame(
            columns=[metric["run_start"], metric["run_end"], metric["run_hours"]]
        )

    stats = {
        "missing_hour_count": missing_hour_count,
        "n_runs": len(gap_runs),
        "max_run_hours": int(gap_runs[metric["run_hours"]].max()) if len(gap_runs) else 0,
        "hours_col": metric["run_hours"],
    }
    return period_table, gap_runs, stats


def daily_hourly_profile(
    df: pd.DataFrame,
    timestamp: pd.Series,
    target: str = "Rented Bike Count",
    lang=None,
) -> dict:
    """Compute the raw-demand temporal profile (calc only, no plotting).

    Returns a dict with the daily mean series, its 30-day rolling median, the
    per-season hourly median/quartile profile and a localized season summary
    table. Consumed by :func:`plot_daily_and_hourly`.
    """
    lang = _resolve_lang(lang)
    season_labels = _season_labels(lang)
    cols = lang(
        {
            "season": "Estação",
            "n": "N",
            "mean": "Média",
            "median": "Mediana",
            "peak_hour": "Hora de pico mediano",
            "peak_median": "Mediana no pico",
        }
    )
    frame = df.assign(_ts=timestamp.to_numpy())
    daily_demand = frame.groupby(frame["_ts"].dt.normalize())[target].mean()
    daily_median_30d = daily_demand.rolling(30, min_periods=15).median()

    hourly_profile = (
        frame.groupby(["Seasons", "Hour"], observed=False)[target]
        .agg(Mediana="median", Q1=lambda s: s.quantile(0.25), Q3=lambda s: s.quantile(0.75))
        .reset_index()
    )

    rows = []
    for season in _SEASON_ORDER:
        group = frame.loc[frame["Seasons"].eq(season), target]
        profile = hourly_profile.loc[hourly_profile["Seasons"].eq(season)].set_index("Hour")[
            "Mediana"
        ]
        rows.append(
            {
                cols["season"]: season_labels[season],
                cols["n"]: len(group),
                cols["mean"]: group.mean(),
                cols["median"]: group.median(),
                cols["peak_hour"]: int(profile.idxmax()),
                cols["peak_median"]: profile.max(),
            }
        )

    return {
        "daily_demand": daily_demand,
        "daily_median_30d": daily_median_30d,
        "hourly_profile": hourly_profile,
        "season_summary": pd.DataFrame(rows),
        "season_labels": season_labels,
    }


def plot_daily_and_hourly(profile: dict, lang=None) -> Tuple[plt.Figure, np.ndarray]:
    """Plot the daily-level series and the per-season hourly-median profile."""
    lang = _resolve_lang(lang)
    labels = lang(
        {
            "daily_title": "Demanda diária bruta e mudança de nível",
            "date": "Data",
            "per_hour": "Bicicletas por hora",
            "daily_mean": "Média diária",
            "rolling_median": "Mediana móvel de 30 dias",
            "hourly_title": "Perfil horário mediano por estação — demanda bruta",
            "hour": "Hora do dia",
            "median_rented": "Mediana de bicicletas alugadas",
        }
    )
    season_labels = profile["season_labels"]
    hourly_profile = profile["hourly_profile"]

    fig, axes = plt.subplots(2, 1, figsize=(15, 9))
    profile["daily_demand"].plot(
        ax=axes[0], color="steelblue", alpha=0.35, linewidth=0.5, label=labels["daily_mean"]
    )
    profile["daily_median_30d"].plot(
        ax=axes[0], color="crimson", linewidth=1.3, label=labels["rolling_median"]
    )
    axes[0].set(title=labels["daily_title"], xlabel=labels["date"], ylabel=labels["per_hour"])
    axes[0].legend()

    for season in _SEASON_ORDER:
        prof = hourly_profile.loc[hourly_profile["Seasons"].eq(season)]
        axes[1].plot(
            prof["Hour"],
            prof["Mediana"],
            marker="o",
            markersize=3,
            linewidth=1.2,
            label=season_labels[season],
        )
    axes[1].set(
        title=labels["hourly_title"],
        xlabel=labels["hour"],
        ylabel=labels["median_rented"],
        xticks=range(0, 24, 2),
    )
    axes[1].legend(ncols=4)
    fig.tight_layout()
    return fig, axes


def yearly_growth_summary(
    df: pd.DataFrame, target: str = "Rented Bike Count", lang=None
) -> pd.DataFrame:
    """Per-year demand aggregates with year-over-year mean change and coverage."""
    lang = _resolve_lang(lang)
    labels = lang(
        {
            "observed": "Horas observadas",
            "mean": "Média horária",
            "median": "Mediana horária",
            "max": "Máximo horário",
            "total": "Total de aluguéis",
            "yoy": "Variação anual da média (%)",
            "coverage": "Cobertura",
            "partial": "Parcial",
            "full": "Ano completo",
        }
    )
    years = pd.to_datetime(df["DateTime"]).dt.year
    yearly = df.groupby(years)[target].agg(["count", "mean", "median", "max", "sum"])
    yearly.columns = [
        labels["observed"],
        labels["mean"],
        labels["median"],
        labels["max"],
        labels["total"],
    ]
    yearly[labels["yoy"]] = yearly[labels["mean"]].pct_change() * 100
    yearly[labels["coverage"]] = np.where(yearly.index == 2015, labels["partial"], labels["full"])
    return yearly


def plot_yearly_growth(yearly: pd.DataFrame, lang=None) -> Tuple[plt.Figure, np.ndarray]:
    """Plot yearly mean demand on linear and logarithmic scales."""
    lang = _resolve_lang(lang)
    labels = lang(
        {
            "mean_col": "Média horária",
            "linear_title": "Demanda horária média por ano",
            "log_title": "Demanda horária média por ano — escala logarítmica",
            "year": "Ano-calendário",
            "per_hour": "Média de bicicletas por hora",
            "partial": "2015 parcial",
        }
    )
    mean_col = labels["mean_col"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    yearly[mean_col].plot(ax=axes[0], marker="o", linewidth=1.2, title=labels["linear_title"])
    yearly[mean_col].plot(
        ax=axes[1], marker="o", logy=True, linewidth=1.2, title=labels["log_title"]
    )
    for ax in axes:
        ax.set(xlabel=labels["year"], ylabel=labels["per_hour"])
        ax.axvspan(2014.75, 2015.25, color="gray", alpha=0.12, label=labels["partial"])
    axes[0].legend()
    fig.tight_layout()
    return fig, axes


def impute_numeric_median(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Median-impute numeric columns (mirrors the modelling pipeline's imputer).

    Returns a new frame (no in-place mutation) plus the per-column NaN counts
    that were present before imputation.
    """
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    na_before = out[num_cols].isna().sum()
    na_before = na_before[na_before > 0]
    out[na_before.index] = out[na_before.index].fillna(out[na_before.index].median())
    return out, na_before


def weather_missing_summary(df: pd.DataFrame, weather_cols: list, lang=None) -> pd.DataFrame:
    """Missing-value counts and percentages for the weather columns."""
    lang = _resolve_lang(lang)
    labels = lang({"missing": "Valores ausentes", "pct": "Percentual ausente (%)"})
    return pd.DataFrame(
        {
            labels["missing"]: df[weather_cols].isna().sum(),
            labels["pct"]: 100 * df[weather_cols].isna().mean(),
        }
    ).sort_values(labels["missing"], ascending=False)


def plot_weather_missing_by_year(
    df: pd.DataFrame, weather_cols: list, lang=None
) -> Tuple[plt.Figure, plt.Axes]:
    """Heatmap of missing weather values by variable and calendar year."""
    lang = _resolve_lang(lang)
    labels = lang(
        {
            "title": "Valores meteorológicos ausentes por variável e ano",
            "year": "Ano-calendário",
            "variable": "Variável",
        }
    )
    missing_by_year = (
        df.assign(_year=pd.to_datetime(df["DateTime"]).dt.year)
        .groupby("_year")[weather_cols]
        .agg(lambda s: int(s.isna().sum()))
    )
    fig, ax = plt.subplots(figsize=(13, 4.8))
    sns.heatmap(missing_by_year.T, cmap="Reds", linewidths=0.3, annot=True, fmt="g", ax=ax)
    ax.set(title=labels["title"], xlabel=labels["year"], ylabel=labels["variable"])
    fig.tight_layout()
    return fig, ax
