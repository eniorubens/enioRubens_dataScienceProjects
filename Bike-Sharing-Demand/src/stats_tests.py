"""Statistical hypothesis tests.

Faithful port of notebook cells:
  [32] Compute Variance Inflation Factor (VIF) for a given set of features
  [35] Mann-Whitney U, Kruskal-Wallis, Spearman (critical-region functions)
  [36] Chi-square feature selection + critical-region plot
  [38] ANOVA / f_classif feature selection
  [TS-98] f_regression feature selection (Time Series notebook)
  [42] Shapiro-Wilk + Anderson-Darling normality tests
  [46] A/B test rainfall/snowfall (mannwhitney_ab_decision + plot)
  [52] Kruskal-Wallis weather categories

All plot functions return ``fig`` (or ``fig, axes``) per §4 convention.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import (
    anderson,
    chi2 as chi2_dist,
    f as f_dist,
    mannwhitneyu,
    norm,
    shapiro,
    t as t_dist,
    chi2_contingency,
)
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.i18n import localize_table, resolve_lang as _resolve_lang
from src.plotting import set_graph_parameters


# ---------------------------------------------------------------------------
# Shared internal helpers (cells [35])
# ---------------------------------------------------------------------------


def _two_tailed_decision(p_value: float, alpha: float = 0.05) -> str:
    return "Reject H0" if p_value < alpha else "Fail to reject H0"


def _right_tailed_decision(p_value: float, alpha: float = 0.05) -> str:
    return "Reject H0" if p_value < alpha else "Fail to reject H0"


def _mannwhitney_z_result(
    group_a,
    group_b,
    label: str,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Compute Mann-Whitney U with normal-approximation Z statistic."""
    group_a = pd.Series(group_a).dropna()
    group_b = pd.Series(group_b).dropna()
    u_stat, p_value = stats.mannwhitneyu(group_a, group_b, alternative="two-sided")
    n_a, n_b = len(group_a), len(group_b)
    mean_u = n_a * n_b / 2
    std_u = np.sqrt(n_a * n_b * (n_a + n_b + 1) / 12)
    z_observed = (u_stat - mean_u) / std_u
    rank_biserial = 2 * u_stat / (n_a * n_b) - 1
    z_critical = norm.ppf(1 - alpha / 2)
    return {
        "Test": label,
        "N": f"{n_a} / {n_b}",
        "Group medians": f"{group_a.median():.3f} / {group_b.median():.3f}",
        "Reference": "Normal approximation",
        "Observed statistic": z_observed,
        "Raw statistic": u_stat,
        "Effect measure": "rank-biserial r",
        "Effect size": rank_biserial,
        "Critical left": -z_critical,
        "Critical right": z_critical,
        "p-value": p_value,
        "Decision": _two_tailed_decision(p_value, alpha),
    }


def _kruskal_result(
    groups: list,
    label: str,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Compute Kruskal-Wallis H statistic with Chi-square reference."""
    groups = [pd.Series(group).dropna() for group in groups]
    h_stat, p_value = stats.kruskal(*groups)
    df = len(groups) - 1
    n_total = sum(len(group) for group in groups)
    epsilon_squared = (h_stat - len(groups) + 1) / (n_total - len(groups))
    critical = chi2_dist.ppf(1 - alpha, df)
    return {
        "Test": label,
        "N": str(n_total),
        "Group medians": " / ".join(f"{group.median():.3f}" for group in groups),
        "Reference": f"Chi-square(df={df})",
        "Observed statistic": h_stat,
        "Raw statistic": h_stat,
        "Effect measure": "epsilon-squared",
        "Effect size": epsilon_squared,
        "Critical left": np.nan,
        "Critical right": critical,
        "p-value": p_value,
        "Decision": _right_tailed_decision(p_value, alpha),
    }


def _spearman_result(
    x,
    y,
    label: str,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Compute Spearman correlation with t-distribution reference."""
    rho, p_value = stats.spearmanr(x, y)
    n = pd.concat([pd.Series(x), pd.Series(y)], axis=1).dropna().shape[0]
    df = n - 2
    t_observed = rho * np.sqrt(df / (1 - rho**2))
    critical = t_dist.ppf(1 - alpha / 2, df)
    return {
        "Test": label,
        "N": str(n),
        "Group medians": np.nan,
        "Reference": f"t(df={df})",
        "Observed statistic": t_observed,
        "Raw statistic": rho,
        "Effect measure": "Spearman rho",
        "Effect size": rho,
        "Critical left": -critical,
        "Critical right": critical,
        "p-value": p_value,
        "Decision": _two_tailed_decision(p_value, alpha),
    }


# ---------------------------------------------------------------------------
# Seasonal hypothesis tests (cell [35])
# ---------------------------------------------------------------------------


def seasonal_hypothesis_tests(
    dataframe: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run Mann-Whitney (seasonal pairs), Kruskal-Wallis (all seasons), and Spearman (temp).

    Returns
    -------
    pd.DataFrame
        Results table including sample sizes, group medians, effect sizes,
        raw p-values, Holm-adjusted p-values, and both decisions.
    """
    results = [
        _mannwhitney_z_result(
            dataframe[dataframe["Seasons"] == "Spring"]["Rented Bike Count"],
            dataframe[dataframe["Seasons"] == "Winter"]["Rented Bike Count"],
            "Mann-Whitney: Spring vs Winter",
            alpha,
        ),
        _mannwhitney_z_result(
            dataframe[dataframe["Seasons"] == "Summer"]["Rented Bike Count"],
            dataframe[dataframe["Seasons"] == "Autumn"]["Rented Bike Count"],
            "Mann-Whitney: Summer vs Autumn",
            alpha,
        ),
        _mannwhitney_z_result(
            dataframe[dataframe["Seasons"] == "Summer"]["Rented Bike Count"],
            dataframe[dataframe["Seasons"] == "Winter"]["Rented Bike Count"],
            "Mann-Whitney: Summer vs Winter",
            alpha,
        ),
        _kruskal_result(
            [
                dataframe[dataframe["Seasons"] == season]["Rented Bike Count"]
                for season in ["Spring", "Summer", "Autumn", "Winter"]
            ],
            "Kruskal-Wallis: All seasons",
            alpha,
        ),
        _spearman_result(
            dataframe["Rented Bike Count"],
            dataframe["Temperature(C)"],
            "Spearman: Demand vs Temperature",
            alpha,
        ),
        _mannwhitney_z_result(
            dataframe[dataframe["Holiday"] == "Holiday"]["Rented Bike Count"],
            dataframe[dataframe["Holiday"] == "No Holiday"]["Rented Bike Count"],
            "Mann-Whitney: Holiday vs Non-Holiday",
            alpha,
        ),
    ]
    results_df = pd.DataFrame(results)
    p_values = results_df["p-value"].to_numpy(dtype=float)
    valid = np.isfinite(p_values)
    adjusted_p_values = np.full(len(results_df), np.nan, dtype=float)
    if valid.any():
        _, adjusted_p_values[valid], _, _ = multipletests(
            p_values[valid],
            alpha=alpha,
            method="holm",
        )
    results_df["Holm p-value"] = adjusted_p_values
    holm_decision = np.full(len(results_df), "Not evaluated", dtype=object)
    holm_decision[valid] = np.where(
        adjusted_p_values[valid] < alpha,
        "Reject H0",
        "Fail to reject H0",
    )
    results_df["Holm Decision"] = holm_decision
    return results_df


def _plot_two_tailed_reference(
    ax: plt.Axes, row: pd.Series, distribution: str = "normal", lang=None
) -> None:
    """Draw critical region for two-tailed tests (normal or t distribution)."""
    lang = _resolve_lang(lang)
    axis_labels = lang(
        {
            "xlabel": "Estatística padronizada sob H0",
            "ylabel": "Densidade",
            "h0_ref": "Distribuição de referência sob H0",
            "critical_area": "Área crítica",
            "critical_value": "Valor crítico",
            "observed": "Observado",
            "outside": " (fora da escala)",
            "pvalue": "valor-p",
            "decision": "Decisão",
        }
    )
    x = np.linspace(-4.5, 4.5, 1200)
    if distribution == "t":
        df_text = row["Reference"].split("df=")[1].rstrip(")")
        y = t_dist.pdf(x, int(df_text))
    else:
        y = norm.pdf(x)

    left = row["Critical left"]
    right = row["Critical right"]
    ax.plot(x, y, color="steelblue", linewidth=2, label=axis_labels["h0_ref"])
    ax.fill_between(
        x[x <= left], y[x <= left], color="tomato", alpha=0.35, label=axis_labels["critical_area"]
    )
    ax.fill_between(x[x >= right], y[x >= right], color="tomato", alpha=0.35)
    ax.axvline(
        left, color="firebrick", linestyle="--", linewidth=1.3, label=axis_labels["critical_value"]
    )
    ax.axvline(right, color="firebrick", linestyle="--", linewidth=1.3)

    observed = row["Observed statistic"]
    observed_for_plot = np.clip(observed, x.min(), x.max())
    observed_label = f"{axis_labels['observed']} = {observed:.2f}"
    if observed_for_plot != observed:
        observed_label += axis_labels["outside"]
    ax.axvline(observed_for_plot, color="black", linewidth=2, label=observed_label)
    ax.scatter([observed_for_plot], [np.interp(observed_for_plot, x, y)], color="black", zorder=5)
    ax.set_title(row["Test"])
    ax.set_xlabel(axis_labels["xlabel"])
    ax.set_ylabel(axis_labels["ylabel"])
    ax.text(
        0.03,
        0.93,
        f"{axis_labels['pvalue']} = {row['p-value']:.2e}\n"
        f"{axis_labels['decision']}: {row['Decision']}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black"),
    )
    # ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")


def _plot_right_tailed_reference(ax: plt.Axes, row: pd.Series, lang=None) -> None:
    """Draw critical region for right-tailed Chi-square / Kruskal tests."""
    lang = _resolve_lang(lang)
    axis_labels = lang(
        {
            "ylabel": "Densidade",
            "h0_ref": "Distribuição de referência sob H0",
            "critical_area": "Área crítica",
            "critical": "Crítico",
            "observed": "Observado",
            "outside": " (fora da escala)",
            "pvalue": "valor-p",
            "decision": "Decisão",
        }
    )
    degrees_freedom = int(row["Reference"].split("df=")[1].rstrip(")"))
    critical = row["Critical right"]
    x_max = max(critical * 3, chi2_dist.ppf(0.999, degrees_freedom) * 1.2)
    x = np.linspace(0, x_max, 1200)
    y = chi2_dist.pdf(x, degrees_freedom)

    ax.plot(x, y, color="steelblue", linewidth=2, label=axis_labels["h0_ref"])
    ax.fill_between(
        x[x >= critical],
        y[x >= critical],
        color="tomato",
        alpha=0.35,
        label=axis_labels["critical_area"],
    )
    ax.axvline(
        critical,
        color="firebrick",
        linestyle="--",
        linewidth=1.3,
        label=f"{axis_labels['critical']} = {critical:.2f}",
    )

    observed = row["Observed statistic"]
    observed_for_plot = min(observed, x.max())
    observed_label = f"{axis_labels['observed']} = {observed:.2f}"
    if observed_for_plot != observed:
        observed_label += axis_labels["outside"]
    ax.axvline(observed_for_plot, color="black", linewidth=2, label=observed_label)
    ax.scatter([observed_for_plot], [np.interp(observed_for_plot, x, y)], color="black", zorder=5)
    ax.set_title(row["Test"])
    ax.set_xlabel(row["Reference"])
    ax.set_ylabel(axis_labels["ylabel"])
    ax.text(
        0.03,
        0.93,
        f"{axis_labels['pvalue']} = {row['p-value']:.2e}\n"
        f"{axis_labels['decision']}: {row['Decision']}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black"),
    )
    # ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")


def plot_seasonal_hypothesis_tests(
    results_df: pd.DataFrame,
    lang=None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot critical regions and observed statistics for all seasonal hypothesis tests."""
    lang = _resolve_lang(lang)
    labels = lang({"suptitle": "Testes de hipótese: regiões críticas e estatísticas observadas"})
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes = axes.ravel()
    for ax, (_, row) in zip(axes, results_df.iterrows()):
        if row["Reference"].startswith("Chi-square"):
            _plot_right_tailed_reference(ax, row, lang=lang)
        elif row["Reference"].startswith("t("):
            _plot_two_tailed_reference(ax, row, distribution="t", lang=lang)
        else:
            _plot_two_tailed_reference(ax, row, distribution="normal", lang=lang)

    fig.suptitle(
        labels["suptitle"],
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Chi-square feature selection (cell [36])
# ---------------------------------------------------------------------------


def chi_square_feature_selection(
    dataframe: pd.DataFrame,
    target_col: str = "Rented Bike Count",
    alpha: float = 0.05,
    lang=None,
) -> Tuple[pd.DataFrame, plt.Figure, plt.Axes]:
    """Test independence between categorical features and demand quantiles.

    The target-derived demand band is local to this test and must not
    persist in ``dataframe`` (leakage guard, §6).

    Returns
    -------
    feature_scores : pd.DataFrame
        Pearson Chi-square statistics, p-values, degrees of freedom,
        Cramer's V effect sizes, expected-count diagnostics, and decisions.
    fig, ax : matplotlib figure/axes
    """
    dataframe = dataframe.drop(columns=["Rented Bike Count Binned"], errors="ignore")

    categorical_features = dataframe.select_dtypes(include=["category", "object"]).columns.tolist()
    target = pd.qcut(
        dataframe[target_col],
        q=5,
        labels=[f"Q{i}" for i in range(1, 6)],
    )

    assert "Rented Bike Count Binned" not in dataframe.columns

    results = []
    for feature in categorical_features:
        values = dataframe[feature].astype("string").fillna("<missing>")
        contingency = pd.crosstab(values, target, dropna=False)
        chi_square, p_value, degrees_freedom, expected = chi2_contingency(
            contingency,
            correction=False,
        )
        n_observations = contingency.to_numpy().sum()
        min_dimension = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
        cramers_v = (
            np.sqrt(chi_square / (n_observations * min_dimension))
            if min_dimension > 0 and n_observations > 0
            else np.nan
        )
        results.append(
            {
                "feature": feature,
                "Chi Squared Score": chi_square,
                "p-value": p_value,
                "df": degrees_freedom,
                "Cramer's V": cramers_v,
                "Min expected count": expected.min(),
                "Expected cells < 5 (%)": 100 * np.mean(expected < 5),
            }
        )

    feature_scores = pd.DataFrame(results).set_index("feature")
    feature_scores["Decision"] = np.where(
        feature_scores["p-value"] < alpha,
        "Reject H0",
        "Fail to reject H0",
    )
    _, adjusted_p_values, _, _ = multipletests(
        feature_scores["p-value"].to_numpy(),
        alpha=alpha,
        method="holm",
    )
    feature_scores["Holm p-value"] = adjusted_p_values
    feature_scores["Holm Decision"] = np.where(
        feature_scores["Holm p-value"] < alpha,
        "Reject H0",
        "Fail to reject H0",
    )
    feature_scores = feature_scores.sort_values("Cramer's V", ascending=False)

    lang = _resolve_lang(lang)
    labels = lang(
        {
            "suptitle": "Teste Chi² de independência",
            "title": "Associação categórica com os quantis de demanda (V de Cramér)",
        }
    )
    colors = ["coral", "cornflowerblue"]
    with plt.style.context(["default"]):
        set_graph_parameters()
        fig, ax = plt.subplots()
        import seaborn as sns

        sns.heatmap(
            feature_scores[["Cramer's V"]],
            annot=True,
            cmap=colors,
            linewidths=0.4,
            fmt=".3f",
            ax=ax,
        )
        plt.suptitle(
            labels["suptitle"],
            fontsize="large",
            x=0.139,
            y=0.95,
            ha="right",
            fontweight="bold",
            style="italic",
        )
        plt.title(labels["title"], x=0.065, y=0.97, ha="right", fontsize="small", style="italic")

    return feature_scores, fig, ax


# ---------------------------------------------------------------------------
# ANOVA feature selection (cell [38])
# ---------------------------------------------------------------------------


def anova_feature_selection(
    dataframe: pd.DataFrame,
    numerical_features: List[str],
    target: pd.Series,
    alpha: float = 0.05,
    lang=None,
) -> Tuple[pd.DataFrame, plt.Figure, plt.Axes]:
    """Run SelectKBest(f_classif) on numerical features and plot scores.

    Returns
    -------
    feature_scores : pd.DataFrame
    fig, ax : matplotlib figure/axes
    """
    features = dataframe.loc[:, numerical_features]

    best_features = SelectKBest(score_func=f_classif, k="all")
    fit = best_features.fit(features, target)

    anova_df1 = target.nunique() - 1
    anova_df2 = len(target) - target.nunique()

    feature_scores = pd.DataFrame(
        data=fit.scores_,
        index=list(features.columns),
        columns=["ANOVA Score"],
    )
    feature_scores["p-value"] = fit.pvalues_
    feature_scores["df1"] = anova_df1
    feature_scores["df2"] = anova_df2
    feature_scores["Eta squared"] = (
        feature_scores["ANOVA Score"]
        * anova_df1
        / (feature_scores["ANOVA Score"] * anova_df1 + anova_df2)
    )
    feature_scores["Decision"] = np.where(
        feature_scores["p-value"] < alpha,
        "Reject H0",
        "Fail to reject H0",
    )
    _, adjusted_p_values, _, _ = multipletests(
        feature_scores["p-value"].to_numpy(),
        alpha=alpha,
        method="holm",
    )
    feature_scores["Holm p-value"] = adjusted_p_values
    feature_scores["Holm Decision"] = np.where(
        feature_scores["Holm p-value"] < alpha,
        "Reject H0",
        "Fail to reject H0",
    )
    feature_scores = feature_scores.sort_values("Eta squared", ascending=False)

    lang = _resolve_lang(lang)
    labels = lang(
        {
            "suptitle": "Testes ANOVA",
            "title": "Separação numérica entre quantis de demanda (η²)",
        }
    )
    colors = ["coral", "cornflowerblue"]
    with plt.style.context(["default"]):
        set_graph_parameters()
        fig, ax = plt.subplots()
        import seaborn as sns

        sns.heatmap(
            feature_scores[["Eta squared"]],
            annot=True,
            cmap=colors,
            linewidths=0.4,
            fmt=".2f",
            ax=ax,
        )
        plt.suptitle(
            labels["suptitle"],
            fontsize="large",
            x=0.093,
            y=0.95,
            ha="right",
            fontweight="bold",
            style="italic",
        )
        plt.title(labels["title"], x=0.058, y=0.97, ha="right", fontsize="small", style="italic")

    return feature_scores, fig, ax


def plot_anova_critical_region(
    feature_scores: pd.DataFrame,
    alpha: float = 0.05,
    top_n: int = 5,
    lang=None,
) -> Tuple[pd.DataFrame, plt.Figure, plt.Axes]:
    """Plot F-distribution critical region with top observed ANOVA scores.

    Parameters
    ----------
    feature_scores : pd.DataFrame
        Output of ``anova_feature_selection()``. Must contain columns
        ``'ANOVA Score'``, ``'p-value'``, ``'df1'``, and ``'df2'``.
    alpha : float
        Significance level (default 0.05).
    top_n : int
        Number of top features to annotate on the plot (default 5).
    lang : optional
        Language map for i18n labels.

    Returns
    -------
    hypothesis_df : pd.DataFrame
        Hypothesis test summary table with Decision column.
    fig, ax : matplotlib figure/axes
    """
    anova_df1 = int(feature_scores["df1"].iloc[0])
    anova_df2 = int(feature_scores["df2"].iloc[0])
    anova_critical = f_dist.ppf(1 - alpha, anova_df1, anova_df2)

    hypothesis_df = pd.DataFrame(
        {
            "Feature": feature_scores.index,
            "F observed": feature_scores["ANOVA Score"].values,
            "p-value": feature_scores["p-value"].values,
            "df1": anova_df1,
            "df2": anova_df2,
            "Critical value": anova_critical,
        }
    )
    hypothesis_df["Decision"] = np.where(
        hypothesis_df["p-value"] < alpha,
        "Reject H0",
        "Fail to reject H0",
    )
    hypothesis_df = hypothesis_df.sort_values("F observed", ascending=False).reset_index(drop=True)

    lang = _resolve_lang(lang)
    labels = lang(
        {
            "title": "Seleção ANOVA: região crítica e maiores estatísticas observadas",
            "xlabel": "Estatística F sob H0",
            "ylabel": "Densidade",
            "critical_area": "Área crítica",
            "critical": "Crítico",
            "outside": " (fora da escala)",
        }
    )

    x_max = max(anova_critical * 5, f_dist.ppf(0.999, anova_df1, anova_df2) * 1.2)
    x = np.linspace(0, x_max, 1200)
    y = f_dist.pdf(x, anova_df1, anova_df2)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(x, y, color="steelblue", linewidth=2, label=f"H0: F(df1={anova_df1}, df2={anova_df2})")
    ax.fill_between(
        x[x >= anova_critical],
        y[x >= anova_critical],
        color="tomato",
        alpha=0.35,
        label=labels["critical_area"],
    )
    ax.axvline(
        anova_critical,
        color="firebrick",
        linestyle="--",
        linewidth=1.5,
        label=f"{labels['critical']} = {anova_critical:.2f}",
    )

    for _, row in hypothesis_df.head(top_n).iterrows():
        observed = row["F observed"]
        observed_for_plot = min(observed, x.max())
        line_label = f"{row['Feature']} = {observed:.1f}"
        if observed_for_plot != observed:
            line_label += labels["outside"]
        ax.axvline(observed_for_plot, linewidth=1.5, alpha=0.85, label=line_label)

    ax.set_title(labels["title"])
    ax.set_xlabel(labels["xlabel"])
    ax.set_ylabel(labels["ylabel"])
    # ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()

    return hypothesis_df, fig, ax


# ---------------------------------------------------------------------------
# F-regression feature selection (Time Series notebook, cell [TS-98])
# ---------------------------------------------------------------------------


def f_regression_feature_selection(
    dataframe: pd.DataFrame,
    target_col: str = "Rented Bike Count",
    categorical_cols: Optional[List[str]] = None,
    drop_cols: Optional[List[str]] = None,
    alpha: float = 0.05,
    top_n: Optional[int] = None,
    lang=None,
) -> Tuple[pd.DataFrame, plt.Figure, plt.Axes]:
    """Run SelectKBest(f_regression) on numerical + one-hot encoded features and plot scores.

    Categorical columns are one-hot encoded with ``pd.get_dummies(drop_first=True)``
    before computing F-statistics. Each retained column is tested separately.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input DataFrame (feature-engineered).
    target_col : str
        Name of the continuous target column (default ``'Rented Bike Count'``).
    categorical_cols : list[str] or None
        Columns to one-hot encode before running the test.  When None, all
        ``object`` and ``category`` dtype columns are used automatically.
    drop_cols : list[str] or None
        Non-feature columns to remove before fitting (default ``['Date', 'DateTime']``).
    alpha : float
        Significance level for the ``Decision`` column (default 0.05).
    top_n : int or None
        Number of top features to display in the plot.  When None, all features
        are shown.  The returned DataFrame always contains all features.
    lang : optional
        Language map for i18n labels.

    Returns
    -------
    feature_scores : pd.DataFrame
        Indexed by feature name; columns: ``F Score``, ``p-value``,
        ``Univariate R2``, ``Relative F share (%)``, and decisions. Sorted by
        univariate R-squared descending.
    fig, ax : matplotlib figure/axes
    """
    if drop_cols is None:
        drop_cols = ["Date", "DateTime"]

    df = dataframe.drop(columns=[c for c in drop_cols if c in dataframe.columns])

    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)

    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    X = df_encoded.drop(columns=[target_col], errors="ignore")
    y = df_encoded[target_col]

    selector = SelectKBest(score_func=f_regression, k="all")
    selector.fit(X, y)

    relative_f_share = (selector.scores_ / selector.scores_.sum()) * 100
    univariate_r_squared = selector.scores_ / (selector.scores_ + len(y) - 2)

    feature_scores = pd.DataFrame(
        {
            "F Score": selector.scores_,
            "p-value": selector.pvalues_,
            "Univariate R2": univariate_r_squared,
            "Relative F share (%)": relative_f_share,
        },
        index=X.columns,
    )
    feature_scores["Decision"] = np.where(
        feature_scores["p-value"] < alpha,
        "Reject H0",
        "Fail to reject H0",
    )
    _, adjusted_p_values, _, _ = multipletests(
        feature_scores["p-value"].to_numpy(),
        alpha=alpha,
        method="holm",
    )
    feature_scores["Holm p-value"] = adjusted_p_values
    feature_scores["Holm Decision"] = np.where(
        feature_scores["Holm p-value"] < alpha,
        "Reject H0",
        "Fail to reject H0",
    )
    feature_scores = feature_scores.sort_values("Univariate R2", ascending=False)

    lang = _resolve_lang(lang)
    labels = lang(
        {
            "suptitle": "Seleção de features por F-regression",
            "title": "Associação linear univariada com o alvo de regressão",
            "xlabel": "R² univariado",
        }
    )

    n_plot = len(feature_scores) if top_n is None else min(int(top_n), len(feature_scores))
    chart_df = feature_scores.head(n_plot).sort_values("Univariate R2", ascending=True)
    bar_colors = np.where(chart_df["Holm Decision"] == "Reject H0", "cornflowerblue", "coral")
    with plt.style.context(["default"]):
        set_graph_parameters()
        fig, ax = plt.subplots(figsize=(10, max(4, len(chart_df) * 0.3)))
        ax.barh(chart_df.index, chart_df["Univariate R2"], color=bar_colors)
        ax.set_xlabel(labels["xlabel"])
        plt.suptitle(labels["suptitle"], fontsize="large", fontweight="bold", style="italic")
        plt.title(labels["title"], fontsize="small", style="italic")
        plt.tight_layout()

    return feature_scores, fig, ax


# ---------------------------------------------------------------------------
# Shapiro-Wilk + Anderson-Darling normality tests (cell [42])
# ---------------------------------------------------------------------------

_SHAPIRO_MAX_N = 5000


def _shapiro_safe_sample(values, max_n: int = _SHAPIRO_MAX_N, random_state: int = 42):
    """Sample values for Shapiro-Wilk (max 5000 for valid p-values)."""
    clean_values = pd.Series(values).dropna().astype(float)
    if len(clean_values) > max_n:
        return clean_values.sample(n=max_n, random_state=random_state), len(clean_values)
    return clean_values, len(clean_values)


def _anderson_critical_value(anderson_result, alpha_percent: float = 5.0):
    """Extract critical value at ``alpha_percent`` from Anderson-Darling result."""
    significance_levels = np.asarray(anderson_result.significance_level, dtype=float)
    critical_values = np.asarray(anderson_result.critical_values, dtype=float)
    idx = int(np.argmin(np.abs(significance_levels - alpha_percent)))
    return critical_values[idx], significance_levels[idx]


def normality_tests(
    normality_samples: Dict[str, pd.Series],
    alpha: float = 0.05,
    lang=None,
) -> Tuple[pd.DataFrame, plt.Figure, np.ndarray]:
    """Run Shapiro-Wilk and Anderson-Darling normality tests on each group.

    Parameters
    ----------
    normality_samples:
        Dict mapping group name → Series of numeric values.
    alpha:
        Significance level.

    Returns
    -------
    results_df : pd.DataFrame
    fig, axes : matplotlib figure/axes
    """
    normality_results = []
    for group_name, values in normality_samples.items():
        clean_values = pd.Series(values).dropna().astype(float)
        shapiro_values, original_n = _shapiro_safe_sample(clean_values)
        shapiro_statistic, shapiro_p_value = shapiro(shapiro_values)

        anderson_result = anderson(clean_values, dist="norm")
        ad_critical_5pct, ad_significance_level = _anderson_critical_value(
            anderson_result, alpha_percent=5.0
        )

        normality_results.append(
            {
                "Group": group_name,
                "N": original_n,
                "Median": clean_values.median(),
                "IQR": clean_values.quantile(0.75) - clean_values.quantile(0.25),
                "Skewness": clean_values.skew(),
                "Excess kurtosis": clean_values.kurt(),
                "Shapiro N Used": len(shapiro_values),
                "Shapiro W Statistic": shapiro_statistic,
                "Shapiro p-value": shapiro_p_value,
                "Shapiro Decision": "Reject H0"
                if shapiro_p_value <= alpha
                else "Fail to reject H0",
                "Anderson-Darling Statistic": anderson_result.statistic,
                "AD Critical Value 5%": ad_critical_5pct,
                "AD Significance Level Used (%)": ad_significance_level,
                "Anderson-Darling Decision": (
                    "Reject H0"
                    if anderson_result.statistic > ad_critical_5pct
                    else "Fail to reject H0"
                ),
            }
        )

    results_df = pd.DataFrame(normality_results)

    lang = _resolve_lang(lang)
    axis_labels = lang(
        {
            "shapiro_title": "Região de decisão do teste de normalidade de Shapiro-Wilk",
            "shapiro_xlabel": "valor-p",
            "shapiro_ylabel": "Densidade de referência",
            "anderson_title": "Região de decisão do teste de normalidade de Anderson-Darling",
            "anderson_xlabel": "Estatística de Anderson-Darling",
            "h0_ref": "Referência do valor-p Uniforme(0, 1) sob H0",
            "critical_region": "Região crítica: p <= {alpha}",
            "alpha": "alfa = {alpha}",
            "observed_ad": "Estatística AD observada",
            "crit_5pct": "Valor crítico de 5%",
        }
    )
    x_max = 0.10
    x = np.linspace(0, x_max, 500)
    y = np.ones_like(x)

    with plt.style.context(["default"]):
        set_graph_parameters()
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        axes[0].plot(x, y, color="steelblue", lw=2, label=axis_labels["h0_ref"])
        axes[0].fill_between(
            x,
            0,
            y,
            where=x <= alpha,
            color="tab:red",
            alpha=0.25,
            label=axis_labels["critical_region"].format(alpha=alpha),
        )
        axes[0].axvline(
            alpha,
            color="tab:red",
            linestyle="--",
            lw=2,
            label=axis_labels["alpha"].format(alpha=alpha),
        )

        marker_y = 1.05
        for _, row in results_df.iterrows():
            observed_p = min(max(row["Shapiro p-value"], 0), x_max)
            axes[0].axvline(observed_p, color="black", linestyle=":", alpha=0.75)
            axes[0].scatter(
                observed_p,
                marker_y,
                s=90,
                zorder=3,
                label=f"{row['Group']} p={row['Shapiro p-value']:.2e}",
            )
            marker_y += 0.08

        axes[0].set_xlim(-0.002, x_max)
        axes[0].set_ylim(0, 1.45)
        axes[0].set_title(axis_labels["shapiro_title"])
        axes[0].set_xlabel(axis_labels["shapiro_xlabel"])
        axes[0].set_ylabel(axis_labels["shapiro_ylabel"])
        axes[0].legend(loc="upper right", fontsize=8)

        y_positions = np.arange(len(results_df))
        colors = np.where(
            results_df["Anderson-Darling Decision"].eq("Reject H0"),
            "tab:red",
            "tab:green",
        )
        axes[1].barh(
            y_positions,
            results_df["Anderson-Darling Statistic"],
            color=colors,
            alpha=0.75,
            label=axis_labels["observed_ad"],
        )
        for y_pos, (_, row) in zip(y_positions, results_df.iterrows()):
            axes[1].plot(
                row["AD Critical Value 5%"],
                y_pos,
                marker="|",
                markersize=18,
                markeredgewidth=2,
                color="black",
                label=axis_labels["crit_5pct"] if y_pos == 0 else None,
            )

        axes[1].set_yticks(y_positions)
        axes[1].set_yticklabels(results_df["Group"])
        axes[1].set_title(axis_labels["anderson_title"])
        axes[1].set_xlabel(axis_labels["anderson_xlabel"])
        axes[1].invert_yaxis()
        axes[1].legend(loc="lower right", fontsize=8)

        plt.tight_layout()

    return results_df, fig, axes


# ---------------------------------------------------------------------------
# A/B test rainfall/snowfall (cell [46])
# ---------------------------------------------------------------------------


def mannwhitney_ab_decision(
    treatment,
    control,
    label: str,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Return Mann-Whitney U decision metrics and normal-approximation coordinates."""
    treatment = pd.Series(treatment).dropna()
    control = pd.Series(control).dropna()
    u_stat, p_value = mannwhitneyu(treatment, control, alternative="two-sided")

    n_treatment = len(treatment)
    n_control = len(control)
    mean_u = n_treatment * n_control / 2
    std_u = np.sqrt(n_treatment * n_control * (n_treatment + n_control + 1) / 12)
    z_observed = (u_stat - mean_u) / std_u

    z_critical = norm.ppf(1 - alpha / 2)
    u_critical_low = mean_u - z_critical * std_u
    u_critical_high = mean_u + z_critical * std_u

    decision = "Reject H0" if p_value < alpha else "Fail to reject H0"

    return {
        "Test": label,
        "Treatment N": n_treatment,
        "Control N": n_control,
        "U observed": u_stat,
        "U critical low": u_critical_low,
        "U critical high": u_critical_high,
        "z observed": z_observed,
        "z critical left": -z_critical,
        "z critical right": z_critical,
        "p-value": p_value,
        "alpha": alpha,
        "Decision": decision,
    }


def plot_mannwhitney_critical_regions(
    results_df: pd.DataFrame,
    alpha: float = 0.05,
    lang=None,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Plot two-tailed critical regions under the H0 normal approximation."""
    lang = _resolve_lang(lang)
    axis_labels = lang(
        {
            "xlabel": "Aproximação normal padrão sob H0",
            "ylabel": "Densidade",
            "suptitle": "Teste U de Mann-Whitney: regiões críticas e estatística observada",
            "h0_dist": "Distribuição sob H0",
            "critical_area": "Área crítica",
            "critical_z": "z crítico = ±{z:.2f}",
            "observed_z": "Observado z = {z:.2f}",
            "outside": " (fora da escala do gráfico)",
            "u_observed": "U observado",
            "pvalue": "valor-p",
            "decision": "Decisão",
        }
    )
    z_critical = norm.ppf(1 - alpha / 2)
    x = np.linspace(-4.5, 4.5, 1200)
    y = norm.pdf(x)

    fig, axes = plt.subplots(1, len(results_df), figsize=(16, 5), sharey=True)
    if len(results_df) == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, results_df.iterrows()):
        ax.plot(x, y, color="steelblue", linewidth=2, label=axis_labels["h0_dist"])

        left_region = x <= -z_critical
        right_region = x >= z_critical
        ax.fill_between(
            x[left_region],
            y[left_region],
            color="tomato",
            alpha=0.35,
            label=axis_labels["critical_area"],
        )
        ax.fill_between(x[right_region], y[right_region], color="tomato", alpha=0.35)

        ax.axvline(
            -z_critical,
            color="firebrick",
            linestyle="--",
            linewidth=1.4,
            label=axis_labels["critical_z"].format(z=z_critical),
        )
        ax.axvline(z_critical, color="firebrick", linestyle="--", linewidth=1.4)

        z_observed = row["z observed"]
        z_for_plot = np.clip(z_observed, x.min(), x.max())
        observed_label = axis_labels["observed_z"].format(z=z_observed)
        if z_for_plot != z_observed:
            observed_label += axis_labels["outside"]

        ax.axvline(z_for_plot, color="black", linewidth=2.0, label=observed_label)
        ax.scatter([z_for_plot], [norm.pdf(z_for_plot)], color="black", s=50, zorder=5)

        y_text = 0.34
        x_text = -4.25 if z_for_plot < 0 else 0.85
        ax.text(
            x_text,
            y_text,
            f"{axis_labels['u_observed']} = {row['U observed']:.1f}\n"
            f"{axis_labels['pvalue']} = {row['p-value']:.2e}\n"
            f"{axis_labels['decision']}: {row['Decision']}",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black"),
        )

        ax.set_title(row["Test"])
        ax.set_xlabel(axis_labels["xlabel"])
        ax.set_ylabel(axis_labels["ylabel"])
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(0, 0.43)
        # ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        axis_labels["suptitle"],
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Kruskal-Wallis weather categories (cell [52])
# ---------------------------------------------------------------------------


def kruskal_weather_test(
    dataframe: pd.DataFrame,
    alpha: float = 0.05,
    lang=None,
) -> Tuple[pd.DataFrame, plt.Figure, np.ndarray]:
    """Run Kruskal-Wallis tests for Rainfall and Snowfall category groups.

    Returns
    -------
    results_df : pd.DataFrame
    fig, axes : matplotlib figure/axes
    """
    from scipy.stats import kruskal

    lang = _resolve_lang(lang)
    test_labels = lang(
        {
            "rain": "Categorias de chuva",
            "snow": "Categorias de neve",
            "median": "mediana",
        }
    )
    results = []
    for label, group_col in [
        (test_labels["rain"], "Rainfall Cat"),
        (test_labels["snow"], "Snowfall Cat"),
    ]:
        grouped_values = []
        group_summaries = []
        for group_name, group in dataframe.groupby(group_col, observed=False):
            values = group["Rented Bike Count"].dropna().astype(float)
            if len(values) == 0:
                continue
            grouped_values.append(values.values)
            group_summaries.append(
                f"{group_name}: n={len(values):,}, "
                f"{test_labels['median']}={values.median():.3f}"
            )

        h_stat, p_value = kruskal(*grouped_values)
        degrees_freedom = len(grouped_values) - 1
        total_n = sum(len(values) for values in grouped_values)
        epsilon_squared = (
            max(0.0, (h_stat - len(grouped_values) + 1) / (total_n - len(grouped_values)))
            if total_n > len(grouped_values)
            else np.nan
        )
        critical = chi2_dist.ppf(1 - alpha, degrees_freedom)
        results.append(
            {
                "Test": label,
                "N": total_n,
                "Groups": len(grouped_values),
                "Group summary": " | ".join(group_summaries),
                "df": degrees_freedom,
                "H observed": h_stat,
                "Epsilon squared": epsilon_squared,
                "Critical value": critical,
                "p-value": p_value,
                "Decision": "Reject H0" if p_value < alpha else "Fail to reject H0",
            }
        )

    axis_labels = lang(
        {
            "xlabel": "H de Kruskal-Wallis sob H0",
            "ylabel": "Densidade",
            "suptitle": (
                "Categorias meteorológicas (Kruskal-Wallis): "
                "regiões críticas e estatísticas observadas"
            ),
            "critical_area": "Área crítica",
            "critical": "Crítico",
            "observed": "Observado",
            "outside": " (fora da escala)",
            "pvalue": "valor-p",
            "decision": "Decisão",
        }
    )
    results_df = pd.DataFrame(results)
    valid_p_values = results_df["p-value"].notna()
    adjusted_p_values = np.full(len(results_df), np.nan)
    if valid_p_values.any():
        _, adjusted_p_values[valid_p_values], _, _ = multipletests(
            results_df.loc[valid_p_values, "p-value"],
            alpha=alpha,
            method="holm",
        )
    results_df["Holm p-value"] = adjusted_p_values
    results_df["Holm Decision"] = np.where(
        results_df["Holm p-value"].isna(),
        "Not evaluated",
        np.where(
            results_df["Holm p-value"] < alpha,
            "Reject H0",
            "Fail to reject H0",
        ),
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=False)
    for ax, (_, row) in zip(axes, results_df.iterrows()):
        degrees_freedom = int(row["df"])
        critical = row["Critical value"]
        x_max = max(critical * 3, chi2_dist.ppf(0.999, degrees_freedom) * 1.2)
        x = np.linspace(0, x_max, 1200)
        y = chi2_dist.pdf(x, degrees_freedom)

        ax.plot(x, y, color="steelblue", linewidth=2, label=f"H0: Chi²(df={degrees_freedom})")
        ax.fill_between(
            x[x >= critical],
            y[x >= critical],
            color="tomato",
            alpha=0.35,
            label=axis_labels["critical_area"],
        )
        ax.axvline(
            critical,
            color="firebrick",
            linestyle="--",
            linewidth=1.5,
            label=f"{axis_labels['critical']} = {critical:.2f}",
        )

        observed = row["H observed"]
        observed_for_plot = min(observed, x.max())
        observed_label = f"{axis_labels['observed']} H = {observed:.2f}"
        if observed_for_plot != observed:
            observed_label += axis_labels["outside"]
        ax.axvline(observed_for_plot, color="black", linewidth=2, label=observed_label)
        ax.scatter(
            [observed_for_plot], [np.interp(observed_for_plot, x, y)], color="black", zorder=5
        )
        ax.text(
            0.03,
            0.93,
            f"Holm {axis_labels['pvalue']} = {row['Holm p-value']:.2e}\n"
            f"Holm {axis_labels['decision'].lower()}: {row['Holm Decision']}\n"
            f"ε² = {row['Epsilon squared']:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black"),
        )
        ax.set_title(row["Test"])
        ax.set_xlabel(axis_labels["xlabel"])
        ax.set_ylabel(axis_labels["ylabel"])
        # ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        axis_labels["suptitle"],
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    return results_df, fig, axes


# ---------------------------------------------------------------------------
# check_vif (cell [32])
# ---------------------------------------------------------------------------


def check_vif(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor (VIF) for a given set of features.

    A constant column is added internally to account for the intercept,
    following the standard OLS assumption. It is not included in the output.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the features to be evaluated.
    features : list[str]
        List of column names for which VIF is to be calculated.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['Feature', 'VIF'] sorted by VIF descending.

    Raises
    ------
    ValueError
        If any feature is not found in the DataFrame.
    """
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Features not found in DataFrame: {missing}")

    subset = df[features].dropna().copy()
    subset.insert(0, "_intercept", 1.0)

    vif_data = (
        pd.DataFrame(
            {
                "Feature": features,
                "VIF": [
                    variance_inflation_factor(subset.values, idx + 1)  # +1 skips intercept
                    for idx, _ in enumerate(features)
                ],
            }
        )
        .sort_values("VIF", ascending=False)
        .reset_index(drop=True)
    )

    return vif_data


def plot_chi_square_critical_region(
    feature_scores: pd.DataFrame,
    alpha: float = 0.05,
    top_n: int = 5,
    lang=None,
) -> Tuple[pd.DataFrame, plt.Figure, plt.Axes]:
    """Compare each Chi-square statistic with its feature-specific critical value.

    Parameters
    ----------
    feature_scores : pd.DataFrame
        Output of ``chi_square_feature_selection()``. Must contain columns
        ``'Chi Squared Score'``, ``'p-value'``, and ``'df'``.
    alpha : float
        Significance level (default 0.05).
    top_n : int
        Number of top features to annotate on the plot (default 5).
    lang : optional
        Language map for i18n labels.

    Returns
    -------
    hypothesis_df : pd.DataFrame
        Hypothesis test summary table with Decision column.
    fig, ax : matplotlib figure/axes
    """
    hypothesis_df = pd.DataFrame(
        {
            "Feature": feature_scores.index,
            "Chi-square observed": feature_scores["Chi Squared Score"].values,
            "p-value": feature_scores["p-value"].values,
            "df": feature_scores["df"].astype(int).values,
            "Cramer's V": feature_scores["Cramer's V"].values,
        }
    )
    hypothesis_df["Critical value"] = chi2_dist.ppf(
        1 - alpha,
        hypothesis_df["df"],
    )
    hypothesis_df["Observed / critical"] = (
        hypothesis_df["Chi-square observed"] / hypothesis_df["Critical value"]
    )
    hypothesis_df["Decision"] = feature_scores["Decision"].values
    hypothesis_df["Holm p-value"] = feature_scores["Holm p-value"].values
    hypothesis_df["Holm Decision"] = feature_scores["Holm Decision"].values
    hypothesis_df = hypothesis_df.sort_values("Cramer's V", ascending=False).reset_index(drop=True)

    lang = _resolve_lang(lang)
    labels = lang(
        {
            "title": "Triagem Chi²: estatística observada relativa ao seu valor crítico",
            "xlabel": "Chi² observado / valor crítico específico da variável",
            "ylabel": "Variável",
            "observed_eq_critical": "Observado = valor crítico",
        }
    )

    chart_df = hypothesis_df.head(top_n).sort_values("Observed / critical", ascending=True)
    colors = np.where(chart_df["Holm Decision"].eq("Reject H0"), "cornflowerblue", "coral")
    fig, ax = plt.subplots(figsize=(13, max(4, len(chart_df) * 0.55)))
    ax.barh(chart_df["Feature"], chart_df["Observed / critical"], color=colors)
    ax.axvline(
        1.0,
        color="firebrick",
        linestyle="--",
        linewidth=1.5,
        label=labels["observed_eq_critical"],
    )

    ax.set_title(labels["title"])
    ax.set_xlabel(labels["xlabel"])
    ax.set_ylabel(labels["ylabel"])
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()

    return hypothesis_df, fig, ax


# ---------------------------------------------------------------------------
# Compact report builders (notebook 02). Column names are kept as the internal
# statistical contract (Cramer's V, Eta squared, Holm Decision, …) so that any
# code/tests referencing this schema keep working. ``localize_report()`` below
# produces a display-only copy with PT-BR headers and decisions for `display`/
# `print` in the notebooks — the internal contract itself is never renamed.
# ---------------------------------------------------------------------------

_SEASONAL_REPORT_COLS = [
    "Test",
    "N",
    "Group medians",
    "Effect measure",
    "Effect size",
    "p-value",
    "Holm p-value",
    "Holm Decision",
]
_CHI_REPORT_COLS = [
    "Cramer's V",
    "Chi Squared Score",
    "df",
    "Min expected count",
    "Expected cells < 5 (%)",
    "Holm p-value",
    "Holm Decision",
]
_ANOVA_REPORT_COLS = ["Eta squared", "ANOVA Score", "df1", "df2", "Holm p-value", "Holm Decision"]
_F_REGRESSION_REPORT_COLS = [
    "Univariate R2",
    "F Score",
    "Relative F share (%)",
    "Holm p-value",
    "Holm Decision",
]


def seasonal_report_table(seasonal_results: pd.DataFrame) -> pd.DataFrame:
    """Compact seasonal-tests report: magnitude + multiplicity-adjusted inference."""
    return seasonal_results.loc[:, _SEASONAL_REPORT_COLS]


def chi_square_report(chi_scores: pd.DataFrame) -> pd.DataFrame:
    """Chi-square report ordered by Cramer's V (descending)."""
    return chi_scores[_CHI_REPORT_COLS].sort_values("Cramer's V", ascending=False)


def anova_report(anova_scores: pd.DataFrame) -> pd.DataFrame:
    """ANOVA report ordered by eta-squared (descending)."""
    return anova_scores[_ANOVA_REPORT_COLS].sort_values("Eta squared", ascending=False)


def f_regression_report(f_scores: pd.DataFrame) -> pd.DataFrame:
    """F-regression report (already ordered by univariate R2)."""
    return f_scores[_F_REGRESSION_REPORT_COLS]


# ---------------------------------------------------------------------------
# Presentation layer: PT-BR display copies of the tables above. The internal
# English schema (columns consumed by the functions/tests in this module)
# is never renamed in place; only the copy returned here is.
# ---------------------------------------------------------------------------

_DECISION_VALUE_LABELS = {
    "Reject H0": "Rejeita H₀",
    "Fail to reject H0": "Não rejeita H₀",
    "Not evaluated": "Não avaliado",
}

# Fixed test-name/reference strings produced by seasonal_hypothesis_tests() and
# _mannwhitney_z_result(); other callers (mannwhitney_ab_decision,
# kruskal_weather_test) already build their "Test" label in PT via lang(), so
# those values simply pass through .fillna() unchanged below.
_VALUE_TEXT_LABELS = {
    "Mann-Whitney: Spring vs Winter": "Mann-Whitney: Primavera vs Inverno",
    "Mann-Whitney: Summer vs Autumn": "Mann-Whitney: Verão vs Outono",
    "Mann-Whitney: Summer vs Winter": "Mann-Whitney: Verão vs Inverno",
    "Kruskal-Wallis: All seasons": "Kruskal-Wallis: Todas as estações",
    "Spearman: Demand vs Temperature": "Spearman: Demanda vs Temperatura",
    "Mann-Whitney: Holiday vs Non-Holiday": "Mann-Whitney: Feriado vs Não feriado",
    "Normal approximation": "Aproximação normal",
}

_DECISION_COLUMNS = ("Decision", "Holm Decision", "Shapiro Decision", "Anderson-Darling Decision")
_VALUE_LOCALIZED_COLUMNS = _DECISION_COLUMNS + ("Test", "Reference")
_VALUE_LABELS = {**_DECISION_VALUE_LABELS, **_VALUE_TEXT_LABELS}

_STATS_COLUMN_LABELS = {
    "Test": "Teste",
    "Group medians": "Medianas dos grupos",
    "Reference": "Referência",
    "Observed statistic": "Estatística observada",
    "Raw statistic": "Estatística bruta",
    "Effect measure": "Medida de efeito",
    "Effect size": "Tamanho do efeito",
    "Critical left": "Crítico esquerdo",
    "Critical right": "Crítico direito",
    "p-value": "valor-p",
    "Decision": "Decisão",
    "Holm p-value": "valor-p de Holm",
    "Holm Decision": "Decisão de Holm",
    "feature": "Variável",
    "Feature": "Variável",
    "Chi Squared Score": "Estatística Qui-quadrado",
    "Cramer's V": "V de Cramér",
    "Min expected count": "Contagem mínima esperada",
    "Expected cells < 5 (%)": "Células esperadas < 5% (%)",
    "ANOVA Score": "Estatística ANOVA",
    "Eta squared": "Eta ao quadrado (η²)",
    "F Score": "Estatística F",
    "Univariate R2": "R² univariado",
    "Relative F share (%)": "Participação relativa de F (%)",
    "Group": "Grupo",
    "Median": "Mediana",
    "Skewness": "Assimetria",
    "Excess kurtosis": "Curtose em excesso",
    "Shapiro N Used": "N usado (Shapiro)",
    "Shapiro W Statistic": "Estatística W (Shapiro)",
    "Shapiro p-value": "valor-p (Shapiro)",
    "Shapiro Decision": "Decisão (Shapiro)",
    "Anderson-Darling Statistic": "Estatística de Anderson-Darling",
    "AD Critical Value 5%": "Valor crítico AD (5%)",
    "AD Significance Level Used (%)": "Nível de significância AD usado (%)",
    "Anderson-Darling Decision": "Decisão (Anderson-Darling)",
    "Groups": "Grupos",
    "Group summary": "Resumo dos grupos",
    "H observed": "H observado",
    "Epsilon squared": "Épsilon ao quadrado",
    "Critical value": "Valor crítico",
    "Chi-square observed": "Qui-quadrado observado",
    "Observed / critical": "Observado / crítico",
    "F observed": "F observado",
    "Treatment N": "N tratamento",
    "Control N": "N controle",
    "U observed": "U observado",
    "U critical low": "U crítico inferior",
    "U critical high": "U crítico superior",
    "z observed": "z observado",
    "z critical left": "z crítico esquerdo",
    "z critical right": "z crítico direito",
    "alpha": "alfa",
}


def univariate_selection_summary(
    chi_report: pd.DataFrame,
    anova_report: pd.DataFrame,
    f_report: pd.DataFrame,
    lang=None,
    top_n: int = 15,
) -> str:
    """Bloco de texto único (Chi², ANOVA, F-regression) para impressão no notebook.

    Recebe os relatórios compactos (schema interno estável, em inglês) e devolve
    um único texto já formatado e localizado em PT-BR — incluindo a subseção de
    features de F-regression não retidas pela correção de Holm. Substitui o
    bloco de ``print()``/formatação que antes vivia direto na célula do notebook.
    """
    lang = _resolve_lang(lang)
    headers = lang(
        {
            "chi": "CHI-SQUARE — features categóricas",
            "anova": "ANOVA — features numéricas",
            "freg": "F-REGRESSION — top {n} por R² univariado",
            "freg_not": "F-REGRESSION — features não retidas por Holm",
        }
    )
    chi_block = localize_report(
        chi_report[["Cramer's V", "df", "Min expected count", "Holm p-value", "Holm Decision"]],
        lang,
    ).to_string(
        formatters={
            "V de Cramér": "{:.3f}".format,
            "Contagem mínima esperada": "{:.1f}".format,
            "valor-p de Holm": "{:.3e}".format,
        }
    )
    anova_block = localize_report(
        anova_report[["Eta squared", "ANOVA Score", "Holm p-value", "Holm Decision"]],
        lang,
    ).to_string(
        formatters={
            "Eta ao quadrado (η²)": "{:.3f}".format,
            "Estatística ANOVA": "{:.1f}".format,
            "valor-p de Holm": "{:.3e}".format,
        }
    )
    freg_head = localize_report(f_report.head(top_n), lang).to_string(
        formatters={
            "R² univariado": "{:.3f}".format,
            "Participação relativa de F (%)": "{:.2f}".format,
            "valor-p de Holm": "{:.3e}".format,
        }
    )
    freg_not_retained = localize_report(
        f_report.loc[
            f_report["Holm Decision"].ne("Reject H0"),
            ["Univariate R2", "Holm p-value", "Holm Decision"],
        ],
        lang,
    ).to_string(
        formatters={
            "R² univariado": "{:.6f}".format,
            "valor-p de Holm": "{:.3e}".format,
        }
    )
    return (
        f"{headers['chi']}\n{chi_block}\n\n"
        f"{headers['anova']}\n{anova_block}\n\n"
        f"{headers['freg'].format(n=top_n)}\n{freg_head}\n\n"
        f"{headers['freg_not']}\n{freg_not_retained}"
    )


def localize_report(report: pd.DataFrame, lang=None) -> pd.DataFrame:
    """Cópia do relatório com cabeçalhos e decisões localizados para exibição.

    Aplica-se a qualquer tabela produzida por este módulo (testes sazonais,
    Chi², ANOVA, F-regression, normalidade, Kruskal-Wallis, Mann-Whitney):
    apenas as colunas presentes em ``report`` são renomeadas/mapeadas, então
    a mesma função serve tanto para a tabela completa quanto para os
    relatórios compactos (``*_report``). Valores numéricos não são alterados.
    """
    lang = _resolve_lang(lang)
    return localize_table(
        report,
        lang,
        columns=_STATS_COLUMN_LABELS,
        value_columns=_VALUE_LOCALIZED_COLUMNS,
        value_labels=_VALUE_LABELS,
    )
