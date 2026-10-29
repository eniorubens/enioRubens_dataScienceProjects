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

from marketing_campaign.i18n import localize

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


TEXT = {
    "en": {
        "alternative_error": (
            "alternative should be 'two-sided', 'greater' or 'less'."
        ),
        "normal": "Standard normal distribution",
        "critical_region": "Critical region",
        "z_critical": "Z critical = {z:.2f}",
        "z_observed": "Z observed = {z:.2f}",
        "outside_scale": "Z observed = {z:.2f}\noutside the scale [-4, 4]",
        "reject": "Reject H0",
        "fail_to_reject": "Fail to reject H0",
        "h0": "H0: The conversion rates are equal",
        "h1": "H1: The conversion rates are different",
        "z_title": "Standard Normal Distribution for Z-Test",
        "z_subtitle": (
            "Visualization of the observed Z statistic and critical "
            "regions for interpreting the hypothesis test"
        ),
        "z_statistic": "Z Statistic",
        "density": "Density",
        "z_result": "Z-TEST RESULT",
        "z_line": "Z Statistic: {z:.4f}",
        "critical_value": "Critical Value: {value}",
        "conclusion": "\nConclusion:",
        "significant": (
            "There are statistically significant evidence to reject H0."
        ),
        "not_significant": "There is not enough evidence to reject H0.",
        "classes": ["Not Retained", "Retained"],
        "confusion_title": "Confusion Matrix",
        "confusion_subtitle": (
            "Comparison between actual and predicted retention classes"
        ),
        "roc_unavailable": "ROC AUC unavailable",
        "fpr": "False Positive Rate",
        "tpr": "True Positive Rate",
        "roc_title": "ROC Curve",
        "roc_subtitle": (
            "Model ability to distinguish retained vs not-retained users"
        ),
        "no_experiments": "No experiments available for visualization.",
        "auc_title": "ROC AUC by Experiment",
        "auc_subtitle": (
            "Comparison of predictive performance across experiments"
        ),
        "recall_title": "Not Retained Recall by Experiment",
        "recall_subtitle": "Sensitivity in identifying non-retained users",
        "recall_axis": "Recall - Not Retained",
        "importance_title": "Top {n} Feature Importances - {name}",
        "importance_subtitle": (
            "Most relevant features used by the model for retention "
            "prediction"
        ),
        "importance_axis": "Importance",
        "coefficient_title": "Top {n} Absolute Coefficients - {name}",
        "coefficient_subtitle": (
            "Features most strongly associated with retention in logistic "
            "regression, regardless of direction"
        ),
        "coefficient_axis": "Absolute coefficient",
        "no_signal": (
            "{name} does not expose direct feature importances or "
            "coefficients."
        ),
    },
    "pt": {
        "alternative_error": (
            "alternative deve ser 'two-sided', 'greater' ou 'less'."
        ),
        "normal": "Distribuição normal padrão",
        "critical_region": "Região crítica",
        "z_critical": "Z crítico = {z:.2f}",
        "z_observed": "Z observado = {z:.2f}",
        "outside_scale": "Z observado = {z:.2f}\nfora da escala [-4, 4]",
        "reject": "Rejeita H0",
        "fail_to_reject": "Não rejeita H0",
        "h0": "H0: As proporções de conversão são iguais",
        "h1": "H1: As proporções de conversão são diferentes",
        "z_title": "Distribuição Normal Padrão do Teste Z",
        "z_subtitle": (
            "Visualização da estatística Z observada e regiões críticas "
            "para interpretação do teste de hipóteses"
        ),
        "z_statistic": "Estatística Z",
        "density": "Densidade",
        "z_result": "RESULTADO DO TESTE Z",
        "z_line": "Estatística Z: {z:.4f}",
        "critical_value": "Valor crítico: {value}",
        "conclusion": "\nConclusão:",
        "significant": (
            "Há evidências estatisticamente significativas para rejeitar H0."
        ),
        "not_significant": "Não há evidências suficientes para rejeitar H0.",
        "classes": ["Não Retido", "Retido"],
        "confusion_title": "Matriz de Confusão",
        "confusion_subtitle": (
            "Comparação entre as classes reais e previstas para retenção"
        ),
        "roc_unavailable": "ROC AUC indisponível",
        "fpr": "Taxa de Falsos Positivos",
        "tpr": "Taxa de Verdadeiros Positivos",
        "roc_title": "Curva ROC",
        "roc_subtitle": (
            "Avaliação da capacidade do modelo de distinguir entre "
            "usuários retidos e não retidos"
        ),
        "no_experiments": "Nenhum experimento disponível para visualização.",
        "auc_title": "ROC AUC por Experimento",
        "auc_subtitle": (
            "Comparação da capacidade preditiva entre os experimentos"
        ),
        "recall_title": "Recall de Não Retido por Experimento",
        "recall_subtitle": (
            "Sensibilidade na identificação de usuários não retidos"
        ),
        "recall_axis": "Recall - Não Retido",
        "importance_title": "Top {n} Importâncias - {name}",
        "importance_subtitle": (
            "Principais features que o modelo considera para prever "
            "retenção"
        ),
        "importance_axis": "Importância",
        "coefficient_title": "Top {n} Coeficientes Absolutos - {name}",
        "coefficient_subtitle": (
            "Principais features que o modelo de regressão logística "
            "associa com retenção, considerando a magnitude do efeito "
            "independentemente do sinal"
        ),
        "coefficient_axis": "Coeficiente absoluto",
        "no_signal": (
            "{name} nao expõe importâncias ou coeficientes diretos."
        ),
    },
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

    text = TEXT[lang]
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
    text = TEXT[lang]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
    ConfusionMatrixDisplay(
        matrix, display_labels=text["classes"]
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
    text = TEXT[lang]
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
    text = TEXT[lang]
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
