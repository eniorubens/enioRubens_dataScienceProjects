"""Chart helpers shared by the English and Portuguese notebooks.

Every function that writes text on a chart takes ``lang`` (``"en"`` or
``"pt"``); the calculations do not depend on it.
"""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_curve,
)

from marketing_campaign.catalog import MESSAGES, SEGMENT_LABELS
from marketing_campaign.i18n import localize, t, translate_text

try:
    import seaborn as sns
except ImportError:  # seaborn is optional
    sns = None

try:
    from IPython.display import display
except ImportError:  # plain Python session
    def display(value: object) -> None:
        """Print ``value`` when IPython is not installed."""
        print(value)


def _texts(lang: str) -> dict[str, str]:
    """Return the chart messages of one language, without their prefix.

    Parameters
    ----------
    lang : {"en", "pt"}
        Language of the messages.

    Returns
    -------
    dict of str to str
        The ``plots.*`` messages of the catalog keyed by their short name.
    """
    prefix = "plots."
    return {
        key[len(prefix):]: value
        for key, value in MESSAGES[lang].items()
        if key.startswith(prefix)
    }


def set_corporate_theme() -> None:
    """Apply the plotting and display style used in the report.

    The theme is inspired by editorial and financial charts and favours
    readability: bold titles, light horizontal grids, no side spines and
    wide figures. It also widens the pandas display limits.
    """
    if sns is not None:
        sns.set_theme(style="ticks")

    plt.rcParams.update(
        {
            "figure.dpi": 96,
            "figure.figsize": (13.33, 6.5),
            "figure.titlesize": 22,
            "figure.titleweight": "bold",
            "axes.titlesize": 22,
            "axes.titleweight": "bold",
            "axes.titlepad": 20,
            "axes.labelsize": 12,
            "axes.labelweight": "bold",
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.top": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.45,
            "grid.linewidth": 1.0,
            "ytick.left": False,
            "legend.title_fontsize": 14,
            "legend.fontsize": 12,
            "legend.frameon": True,
            "legend.framealpha": 1,
            "legend.fancybox": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "gray",
            "legend.borderpad": 0.6,
            "lines.linewidth": 3,
            "lines.markersize": 10,
        }
    )
    pd.set_option("display.max_rows", 3000)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 2000)
    pd.options.display.max_colwidth = 1000


def add_titles(ax: Any, title: str, subtitle: str | None = None) -> None:
    """Write a left-aligned title and an optional subtitle on an axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to decorate.
    title : str
        Main title.
    subtitle : str, optional
        Grey line written just above the plot area.
    """
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", pad=22)
    if subtitle:
        ax.text(
            0,
            1.02,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            color="#52606D",
        )


def _critical_region(
    ax: Any,
    x: np.ndarray,
    y: np.ndarray,
    bounds: list[float],
    sides: list[str],
    text: dict[str, str],
    label_z: float,
) -> None:
    """Shade the tails of the normal curve and mark the critical values.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis holding the curve.
    x, y : numpy.ndarray
        Points of the standard normal density.
    bounds : list of float
        Critical values to mark.
    sides : list of {"left", "right"}
        Tail shaded for each bound.
    text : dict
        Labels of the chosen language.
    label_z : float
        Critical value written in the legend.
    """
    color = "#C62828"
    for position, (bound, side) in enumerate(zip(bounds, sides)):
        tail = x <= bound if side == "left" else x >= bound
        first = position == 0
        ax.fill_between(
            x[tail],
            y[tail],
            color=color,
            alpha=0.35,
            label=text["critical_region"] if first else None,
        )
        ax.axvline(
            bound,
            color=color,
            linestyle="--",
            linewidth=1.8,
            label=text["z_critical"].format(z=label_z) if first else None,
        )


def plot_z_test_distribution(
    z_stat: float,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    title: str | None = None,
    p_value: float | None = None,
    h0_text: str | None = None,
    h1_text: str | None = None,
    lang: str = "en",
) -> None:
    """Plot the standard normal curve to interpret a z-test.

    The critical regions, the observed statistic, the p-value and both
    hypotheses are drawn, and a short text summary is printed.

    Parameters
    ----------
    z_stat : float
        Observed z statistic.
    alpha : float, default=0.05
        Significance level.
    alternative : {"two-sided", "greater", "less"}, default="two-sided"
        Alternative hypothesis of the test.
    title : str, optional
        Chart title. Defaults to a generic one.
    p_value : float, optional
        P-value of the test. Computed from ``z_stat`` when omitted.
    h0_text, h1_text : str, optional
        Text of the null and alternative hypotheses.
    lang : {"en", "pt"}, default="en"
        Language of the chart texts.

    Raises
    ------
    ValueError
        If ``alternative`` is not one of the supported values.
    """
    from scipy import stats

    text = _texts(lang)
    alternative = alternative.lower()
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError(text["alternative_error"])

    if p_value is None:
        if alternative == "two-sided":
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        elif alternative == "greater":
            p_value = 1 - stats.norm.cdf(z_stat)
        else:
            p_value = stats.norm.cdf(z_stat)
    p_value = float(p_value)

    x = np.linspace(-4, 4, 1_000)
    y = stats.norm.pdf(x)
    fig, ax = plt.subplots(figsize=(14, 5))
    observed_color = "#2E7D32"
    ax.plot(x, y, color="#1A1A1A", linewidth=2.2, label=text["normal"])

    if alternative == "two-sided":
        z_crit = stats.norm.ppf(1 - alpha / 2)
        _critical_region(
            ax, x, y, [-z_crit, z_crit], ["left", "right"], text, z_crit
        )
    elif alternative == "greater":
        z_crit = stats.norm.ppf(1 - alpha)
        _critical_region(ax, x, y, [z_crit], ["right"], text, z_crit)
    else:
        z_crit = stats.norm.ppf(alpha)
        _critical_region(ax, x, y, [z_crit], ["left"], text, z_crit)
    critical_label = f"{z_crit:.2f}"

    x_min, x_max = -4, 4
    z_plot = float(np.clip(z_stat, x_min + 0.03, x_max - 0.03))
    ax.axvline(
        z_plot,
        color=observed_color,
        linestyle="--",
        linewidth=2.4,
        label=text["z_observed"].format(z=z_stat),
    )
    if not np.isclose(z_plot, z_stat):
        outside_left = z_stat < x_min
        ax.annotate(
            text["outside_scale"].format(z=z_stat),
            xy=(z_plot, 0.055),
            xytext=(-3.55 if outside_left else 3.55, 0.16),
            ha="left" if outside_left else "right",
            va="center",
            fontsize=10,
            color=observed_color,
            arrowprops={
                "arrowstyle": "->",
                "color": observed_color,
                "linewidth": 1.6,
            },
        )

    decision = (
        text["reject"] if p_value < alpha else text["fail_to_reject"]
    )
    p_value_label = (
        "p-value < 0.001" if p_value < 0.001 else f"p-value = {p_value:.4f}"
    )
    annotation = (
        f"Z = {z_stat:.2f}\n{p_value_label}\nα = {alpha:.2f}\n{decision}"
    )
    ax.text(
        0.03,
        0.95,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "#404040",
            "alpha": 0.94,
        },
    )

    hypotheses = f"{h0_text or text['h0']}\n{h1_text or text['h1']}"
    ax.text(
        0.5,
        -0.23,
        hypotheses,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        color="#303030",
    )

    add_titles(ax, title or text["z_title"], subtitle=text["z_subtitle"])
    ax.set_xlabel(text["z_statistic"], fontweight="bold")
    ax.set_ylabel(text["density"], fontweight="bold")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, y.max() * 1.20)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False)
    plt.subplots_adjust(bottom=0.25)
    plt.show()
    plt.close(fig)

    print("\n" + "=" * 70)
    print(text["z_result"])
    print("=" * 70)
    print(text["z_line"].format(z=z_stat))
    print(text["critical_value"].format(value=critical_label))
    print(f"P-value: {p_value:.6f}")
    print(text["conclusion"])
    if p_value < alpha:
        print(text["significant"])
    else:
        print(text["not_significant"])


def get_transformed_feature_names(
    pipeline: Any, expected_size: int | None = None
) -> np.ndarray:
    """Recover the names of the transformed features of a pipeline.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Fitted pipeline with a ``preprocessor`` step.
    expected_size : int, optional
        Number of features the model uses. If the preprocessor reports a
        different number of names, generic names are returned.

    Returns
    -------
    numpy.ndarray
        The feature names, or ``feature_0``, ``feature_1``... when they
        cannot be recovered.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        names = preprocessor.get_feature_names_out()
    except Exception:
        names = np.array([], dtype=object)
    if expected_size is not None and len(names) != expected_size:
        return np.array([f"feature_{i}" for i in range(expected_size)])
    return names


def plot_retention_diagnostics(
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    roc_auc: float,
    lang: str = "en",
) -> None:
    """Show the confusion matrix and the ROC curve of the model.

    Parameters
    ----------
    y_test : pd.Series
        True retention (1 retained, 0 not retained).
    y_pred : numpy.ndarray
        Predicted classes.
    y_pred_proba : numpy.ndarray
        Predicted probability of the retained class.
    roc_auc : float
        Area under the ROC curve; ``NaN`` when it is not defined.
    lang : {"en", "pt"}, default="en"
        Language of the chart texts.
    """
    text = _texts(lang)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
    ConfusionMatrixDisplay(
        matrix,
        display_labels=[
            t("model.not_retained", lang),
            t("model.retained", lang),
        ],
    ).plot(ax=axes[0], cmap="Blues", values_format="d", colorbar=False)
    add_titles(
        axes[0], text["confusion_title"], subtitle=text["confusion_subtitle"]
    )

    if pd.notna(roc_auc):
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        axes[1].plot(
            fpr, tpr, color="#2E86AB", lw=2, label=f"ROC AUC={roc_auc:.3f}"
        )
        axes[1].legend(loc="lower right")
    else:
        axes[1].text(0.5, 0.5, text["roc_unavailable"], ha="center")
    axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    axes[1].set_xlabel(text["fpr"], fontweight="bold")
    axes[1].set_ylabel(text["tpr"], fontweight="bold")
    add_titles(axes[1], text["roc_title"], subtitle=text["roc_subtitle"])

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_model_signal_summary(
    pipeline: Any,
    estimator_name: str,
    top_n: int = 10,
    lang: str = "en",
) -> None:
    """Show the feature importances or coefficients of the model.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Fitted pipeline with ``preprocessor`` and ``classifier`` steps.
    estimator_name : str
        Name used in the chart title.
    top_n : int, default=10
        Number of features shown.
    lang : {"en", "pt"}, default="en"
        Language of the chart and table texts.
    """
    text = _texts(lang)
    classifier = pipeline.named_steps["classifier"]

    if hasattr(classifier, "feature_importances_"):
        values = np.asarray(classifier.feature_importances_)
        signal = pd.DataFrame(
            {
                "feature": get_transformed_feature_names(
                    pipeline, len(values)
                ),
                "importance": values,
            }
        ).sort_values("importance", ascending=False)
        display(signal.head(top_n))
        signal.head(top_n).sort_values("importance").plot.barh(
            x="feature", y="importance", legend=False
        )
        add_titles(
            plt.gca(),
            text["importance_title"].format(n=top_n, name=estimator_name),
            subtitle=text["importance_subtitle"],
        )
        plt.xlabel(text["importance_axis"])
        plt.tight_layout()
        plt.show()
        plt.close()
        return

    if isinstance(classifier, LogisticRegression) and hasattr(
        classifier, "coef_"
    ):
        coefficients = np.asarray(classifier.coef_).ravel()
        signal = pd.DataFrame(
            {
                "feature": get_transformed_feature_names(
                    pipeline, len(coefficients)
                ),
                "coefficient": coefficients,
                "absolute_coefficient": np.abs(coefficients),
            }
        ).sort_values("absolute_coefficient", ascending=False)
        display(localize(signal.head(top_n), lang))
        signal.head(top_n).sort_values("absolute_coefficient").plot.barh(
            x="feature", y="absolute_coefficient", legend=False
        )
        add_titles(
            plt.gca(),
            text["coefficient_title"].format(n=top_n, name=estimator_name),
            subtitle=text["coefficient_subtitle"],
        )
        plt.xlabel(text["coefficient_axis"])
        plt.tight_layout()
        plt.show()
        plt.close()
        return

    print(text["no_signal"].format(name=estimator_name))


def plot_retention_experiment_comparison(
    results_df: pd.DataFrame, lang: str = "en"
) -> None:
    """Compare ROC AUC and not-retained recall across the scenarios.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of
        :func:`marketing_campaign.retention_model.run_retention_experiments`
        (English or localised), with ``experiment``, ``roc_auc``,
        ``recall_not_retained`` and ``f1_not_retained``.
    lang : {"en", "pt"}, default="en"
        Language of the chart texts.
    """
    text = _texts(lang)
    if results_df.empty:
        print(text["no_experiments"])
        return

    ordered = results_df.sort_values(
        ["roc_auc", "recall_not_retained", "f1_not_retained"],
        ascending=True,
    )
    fig_height = max(6, 0.35 * len(ordered))
    fig, axes = plt.subplots(
        1, 2, figsize=(16, fig_height), sharey=True
    )

    axes[0].barh(ordered["experiment"], ordered["roc_auc"], color="#2E86AB")
    add_titles(axes[0], text["auc_title"], subtitle=text["auc_subtitle"])
    axes[0].set_xlabel("ROC AUC")

    axes[1].barh(
        ordered["experiment"],
        ordered["recall_not_retained"],
        color="#06A77D",
    )
    add_titles(
        axes[1], text["recall_title"], subtitle=text["recall_subtitle"]
    )
    axes[1].set_xlabel(text["recall_axis"])

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_k_diagnostics(k_diagnostics: pd.DataFrame, lang: str = "en") -> None:
    """Draw the elbow and silhouette curves of the K-Means search.

    Parameters
    ----------
    k_diagnostics : pd.DataFrame
        Output of :func:`marketing_campaign.segmentation.evaluate_k`.
    lang : {"en", "pt"}, default="en"
        Language of the chart texts.
    """
    subtitle = t("chart.rfm.subtitle", lang)
    _, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(
        k_diagnostics["k"],
        k_diagnostics["inertia"],
        marker=".",
        linewidth=0.5,
        color="#0667A7",
    )
    add_titles(axes[0], "Elbow: K-Means RFM", subtitle=subtitle)
    axes[0].set_xlabel("k")
    axes[0].set_ylabel(t("chart.k.inertia", lang))
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        k_diagnostics["k"],
        k_diagnostics["silhouette"],
        marker=".",
        linewidth=0.5,
        color="#06A77D",
    )
    add_titles(axes[1], "Silhouette: K-Means RFM", subtitle=subtitle)
    axes[1].set_xlabel("k")
    axes[1].set_ylabel(t("chart.k.silhouette", lang))
    axes[1].grid(alpha=0.25)

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_rfm_boxplots(rfm: pd.DataFrame, lang: str = "en") -> None:
    """Draw recency, frequency and monetary boxplots by segment.

    Parameters
    ----------
    rfm : pd.DataFrame
        Output of :func:`marketing_campaign.segmentation.assign_segments`
        built with the segment names of ``lang``.
    lang : {"en", "pt"}, default="en"
        Language of the chart texts and of the segment names.
    """
    order = SEGMENT_LABELS[lang]
    colors = ["#06A77D", "#2E86AB", "#A23B72"]
    _, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, metric in zip(axes, ["Recency", "Frequency", "Monetary"]):
        data = [rfm.loc[rfm["Segment"] == name, metric] for name in order]
        boxes = ax.boxplot(
            data, tick_labels=order, patch_artist=True, showfliers=False
        )
        for patch, color in zip(boxes["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        add_titles(
            ax,
            t("chart.rfm.box_title", lang, metric=metric),
            subtitle=t("chart.rfm.subtitle", lang),
        )
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_ltv_by_segment(
    by_segment: pd.DataFrame, global_ltv: float, lang: str = "en"
) -> None:
    """Draw the mean historical value of each segment.

    Parameters
    ----------
    by_segment : pd.DataFrame
        ``by_segment`` table of
        :class:`marketing_campaign.economics.LtvSummary`.
    global_ltv : float
        Mean value of all customers, drawn as a reference line.
    lang : {"en", "pt"}, default="en"
        Language of the chart texts.
    """
    _, ax = plt.subplots(figsize=(10, 5))
    ordered = by_segment.sort_values("ltv_mean", ascending=True)
    ax.barh(ordered["segment"], ordered["ltv_mean"], color="#2E86AB")
    ax.axvline(
        global_ltv,
        color="#D62828",
        linestyle="--",
        linewidth=0.5,
        label=t("chart.ltv.global", lang, value=global_ltv),
    )
    ax.set_xlabel(t("chart.ltv.xlabel", lang), fontweight="bold")
    add_titles(
        ax,
        t("chart.ltv.title", lang),
        subtitle=t("chart.ltv.subtitle", lang),
    )
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_exploration(df: pd.DataFrame, lang: str = "en") -> None:
    """Draw the distribution of conversion, retention, channel and variant.

    Parameters
    ----------
    df : pd.DataFrame
        User-level base. Each panel is drawn only when its column exists.
    lang : {"en", "pt"}, default="en"
        Language of the chart texts.
    """
    _, axes = plt.subplots(2, 2, figsize=(15, 10))

    def panel(position, column, kind, colors, name, xlabel=False):
        """Draw one distribution if its column is available."""
        if column not in df.columns:
            return
        ax = axes[position]
        df[column].value_counts().plot(
            kind=kind, ax=ax, color=colors
        )
        add_titles(
            ax,
            t(f"chart.explore.{name}.title", lang),
            subtitle=t(f"chart.explore.{name}.subtitle", lang),
        )
        if xlabel:
            ax.set_xlabel(t(f"chart.explore.{name}.xlabel", lang))

    panel((0, 0), "converted_flag", "bar", ["#2E86AB", "#A23B72"],
          "conversion", xlabel=True)
    panel((0, 1), "retained_flag", "bar", ["#F18F01", "#C73E1D"],
          "retention", xlabel=True)
    panel((1, 0), "marketing_channel", "barh", "#2E86AB", "channel")
    panel((1, 1), "variant", "bar", ["#06A77D", "#D62828"], "variant")

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_funnel(funnel: pd.DataFrame, lang: str = "en") -> None:
    """Draw the funnel from impacted to retained users.

    Parameters
    ----------
    funnel : pd.DataFrame
        Output of :func:`marketing_campaign.funnel.build_funnel`
        (English); the first three values are the stage counts.
    lang : {"en", "pt"}, default="en"
        Language of the chart texts.
    """
    stages = [
        t("chart.funnel.impacted", lang),
        t("chart.funnel.converted", lang),
        t("chart.funnel.retained", lang),
    ]
    counts = funnel.iloc[:3, 1].astype(int).tolist()
    colors = ["#2E86AB", "#A23B72", "#06A77D"]

    _, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(stages, counts, color=colors, alpha=0.8, edgecolor="black")
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(count):,}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )
    ax.set_ylabel(t("chart.funnel.ylabel", lang), fontweight="bold")
    add_titles(
        ax,
        t("chart.funnel.title", lang),
        subtitle=t("chart.funnel.subtitle", lang),
    )
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_channel_performance(channels: pd.DataFrame, lang: str = "en") -> None:
    """Draw the conversion and retention rate of each channel.

    Parameters
    ----------
    channels : pd.DataFrame
        Output of
        :func:`marketing_campaign.funnel.analyze_channel_performance`.
    lang : {"en", "pt"}, default="en"
        Language of the chart texts.
    """
    _, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, color, name, column in (
        (axes[0], "#2E86AB", "conversion", "conversion_rate_%"),
        (axes[1], "#06A77D", "retention", "retention_rate_%"),
    ):
        ordered = channels.sort_values(column, ascending=True)
        ax.barh(ordered["marketing_channel"], ordered[column], color=color)
        ax.set_xlabel(
            t(f"chart.channel.{name}.xlabel", lang), fontweight="bold"
        )
        add_titles(
            ax,
            t(f"chart.channel.{name}.title", lang),
            subtitle=t(f"chart.channel.{name}.subtitle", lang),
        )
        ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_language_performance(
    summary: pd.DataFrame, lang: str = "en"
) -> None:
    """Draw conversion and retention for correct and wrong language.

    Parameters
    ----------
    summary : pd.DataFrame
        Output of
        :func:`marketing_campaign.diagnostics.analyze_language_performance`
        (English labels).
    lang : {"en", "pt"}, default="en"
        Language of the chart texts and of the bar labels.
    """
    ordered = (
        summary.set_index("label")
        .reindex(["Correct Language", "Wrong Language"])
        .fillna(0)
        .reset_index()
    )
    ordered["label"] = ordered["label"].map(
        lambda value: translate_text(value, lang)
    )
    colors = ["#06A77D", "#D62828"]
    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, name, column in (
        (axes[0], "conversion", "conversion_rate_%"),
        (axes[1], "retention", "retention_rate_%"),
    ):
        ax.bar(ordered["label"], ordered[column], color=colors)
        ax.set_ylabel(t("chart.language.rate", lang), fontweight="bold")
        add_titles(
            ax,
            t(f"chart.language.{name}.title", lang),
            subtitle=t(f"chart.language.{name}.subtitle", lang),
        )
        top = max(100, ordered[column].max() * 1.15) if name == (
            "conversion"
        ) else 105
        ax.set_ylim(0, top)
        for position, value in enumerate(ordered[column]):
            ax.text(
                position,
                value + 1,
                f"{value:.1f}%",
                ha="center",
                fontweight="bold",
            )
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_power_curve(
    sample_sizes: list[int] | np.ndarray,
    powers: list[float],
    sample_per_group: int,
    lang: str = "en",
) -> None:
    """Draw the simulated power against the sample size.

    Parameters
    ----------
    sample_sizes : list of int or numpy.ndarray
        Users per group.
    powers : list of float
        Simulated power for each size, from
        :func:`marketing_campaign.experiment.simulate_power_curve`.
    sample_per_group : int
        Sample size of the experiment, drawn as a reference line.
    lang : {"en", "pt"}, default="en"
        Language of the chart texts.
    """
    _, ax = plt.subplots(figsize=(12, 5))
    ax.plot(sample_sizes, powers, color="#2E86AB", linewidth=2.5)
    ax.axhline(
        0.80,
        color="#D62828",
        linestyle="--",
        linewidth=1.5,
        label=t("chart.power.minimum", lang),
    )
    ax.axvline(
        sample_per_group,
        color="#06A77D",
        linestyle="--",
        linewidth=1.5,
        label=t("chart.power.current", lang, n=sample_per_group),
    )
    ax.fill_between(
        sample_sizes,
        powers,
        0.80,
        where=[power >= 0.80 for power in powers],
        alpha=0.15,
        color="#06A77D",
    )
    ax.set_xlabel(t("chart.power.xlabel", lang), fontweight="bold")
    ax.set_ylabel(t("chart.power.ylabel", lang), fontweight="bold")
    add_titles(
        ax,
        t("chart.power.title", lang),
        subtitle=t("chart.power.subtitle", lang),
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_variant_rates(df: pd.DataFrame, lang: str = "en") -> None:
    """Draw the observed conversion and retention rate of each variant.

    Parameters
    ----------
    df : pd.DataFrame
        User-level base with ``variant``, ``converted_flag`` and
        ``retained_flag``.
    lang : {"en", "pt"}, default="en"
        Language of the chart texts.
    """
    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    conversion = df.groupby("variant")["converted_flag"].agg(["sum", "count"])
    conversion["rate"] = conversion["sum"] / conversion["count"] * 100
    retention = (
        df[df["converted_flag"] == 1]
        .groupby("variant")["retained_flag"]
        .agg(["sum", "count"])
    )
    retention["rate"] = retention["sum"] / retention["count"] * 100

    for ax, stats, name, colors, share, headroom in (
        (axes[0], conversion, "conversion", ["#2E86AB", "#A23B72"],
         0.06, 2.8),
        (axes[1], retention, "retention", ["#06A77D", "#D62828"],
         0.03, 2.5),
    ):
        ax.bar(stats.index, stats["rate"], color=colors[: len(stats)])
        ax.set_ylabel(t("chart.variant.rate", lang), fontweight="bold")
        add_titles(
            ax,
            t(f"chart.variant.{name}.title", lang),
            subtitle=t(f"chart.variant.{name}.subtitle", lang),
        )
        top = float(stats["rate"].max())
        offset = max(top * share, 0.8)
        ax.set_ylim(0, top + offset * headroom)
        for position, value in enumerate(stats["rate"]):
            ax.text(
                position,
                value + offset,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_roi(roi: pd.DataFrame, lang: str = "en") -> None:
    """Draw the incremental ROI and the estimated profit by channel.

    Parameters
    ----------
    roi : pd.DataFrame
        Output of :func:`marketing_campaign.economics.simulate_roi`
        (English names).
    lang : {"en", "pt"}, default="en"
        Language of the chart texts.
    """
    _, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, column, name in (
        (axes[0], "roi_incremental_%", "incremental"),
        (axes[1], "estimated_profit", "profit"),
    ):
        ordered = roi.sort_values(column, ascending=True)
        ax.barh(
            ordered["marketing_channel"],
            ordered[column],
            color=[
                "#06A77D" if value > 0 else "#D62828"
                for value in ordered[column]
            ],
        )
        ax.set_xlabel(t(f"chart.roi.{name}.xlabel", lang), fontweight="bold")
        add_titles(
            ax,
            t(f"chart.roi.{name}.title", lang),
            subtitle=t(f"chart.roi.{name}.subtitle", lang),
        )
        ax.axvline(x=0, color="black", linestyle="--")
        ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_sample_adequacy(
    variant_counts: pd.Series, required_per_group: int, lang: str = "en"
) -> None:
    """Compare the users of each variant with the power-analysis need.

    Parameters
    ----------
    variant_counts : pd.Series
        Users per variant.
    required_per_group : int
        Users needed per group; variants below it are drawn in red.
    lang : {"en", "pt"}, default="en"
        Language of the chart texts.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    positions = range(len(variant_counts))
    bars = ax.bar(
        positions,
        variant_counts.values,
        color="#2E86AB",
        label=t("chart.sample.available", lang),
        zorder=3,
    )
    ax.axhline(
        required_per_group,
        color="#D62828",
        linestyle="--",
        linewidth=2,
        label=t("chart.sample.required", lang, n=required_per_group),
        zorder=4,
    )
    for bar, count in zip(bars, variant_counts.values):
        if required_per_group > 0 and int(count) < required_per_group:
            bar.set_color("#D62828")
            bar.set_alpha(0.7)
    for bar, count in zip(bars, variant_counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(variant_counts.values) * 0.02,
            f"{int(count):,}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )
    ax.set_xticks(list(positions))
    ax.set_xticklabels(variant_counts.index.tolist(), fontsize=11)
    ax.set_ylabel(t("chart.sample.ylabel", lang), fontweight="bold")
    add_titles(
        ax,
        t("chart.sample.title", lang),
        subtitle=t("chart.sample.subtitle", lang),
    )
    ax.legend(loc="center", frameon=False)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_false_positive_bias(
    comparison: pd.DataFrame, column: str, lang: str = "en"
) -> None:
    """Compare the mix of a variable in the base and in the false positives.

    Parameters
    ----------
    comparison : pd.DataFrame
        Output of
        :func:`marketing_campaign.diagnostics.compare_false_positives`
        (English names).
    column : str
        Name of the compared variable, used in the title.
    lang : {"en", "pt"}, default="en"
        Language of the chart texts.
    """
    ordered = comparison.sort_values("diff_pp", ascending=True)
    positions = np.arange(len(ordered))
    bar_height = 0.38

    _, ax = plt.subplots(figsize=(10, max(4, len(ordered) * 0.45)))
    ax.barh(
        positions - bar_height / 2,
        ordered["base_pct"],
        height=bar_height,
        color="#2E86AB",
        label=t("chart.bias.base", lang),
    )
    ax.barh(
        positions + bar_height / 2,
        ordered["false_positive_pct"],
        height=bar_height,
        color="#D62828",
        label=t("chart.bias.false_positives", lang),
    )
    ax.set_yticks(positions)
    ax.set_yticklabels(ordered["category"].astype(str))
    ax.set_xlabel(t("chart.bias.xlabel", lang), fontweight="bold")
    add_titles(
        ax,
        t("chart.bias.title", lang, column=column),
        subtitle=t("chart.bias.subtitle", lang),
    )
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.show()
    plt.close()
