"""Chart and table helpers for the final report notebook (presentation only).

The helpers here only reshape tables that were already computed and stored by
the training package, and draw them. Nothing is trained or scored, no file is
read and ``ecom_delivery_sla`` is never imported, so editing this module leaves
the source fingerprint of the run cache untouched. Labels are kept in this
module (Portuguese and English) so the report does not depend on the shared
label catalogue.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TEMPLATE = "plotly_white"
TRAIN_COLOR = "#2563EB"
TEST_COLOR = "#D97706"
LINE_COLOR = "#111827"
IDEAL_COLOR = "#6B7280"

REPORT_LABELS: dict[str, dict[str, dict[str, str]]] = {
    "pt": {
        "reliability": {
            "title": "Curva de confiabilidade (10 faixas)",
            "x": "Probabilidade média prevista",
            "y": "Taxa observada",
            "ideal": "calibração perfeita",
            "orders": "pedidos na faixa",
        },
        "prevalence": {
            "title": "Pedidos e prevalência de violação por mês de compra",
            "month": "Mês de compra",
            "orders": "Pedidos",
            "prevalence": "Prevalência de violação",
            "train": "Treino",
            "test": "Teste",
        },
        "optuna": {
            "classification": "validation_average_precision",
            "regression": "validation_rmse",
        },
        "evidence": {
            "development": "Desenvolvimento (folds de 2017-10 a 2018-06)",
            "f1": "F1 (oficial)",
            "f2": "F2 (pós-hoc, exploratório)",
            "ap_mean": "AP média do {model}",
            "ap_best_baseline": "AP média da melhor baseline ({model})",
            "gain": "Ganho relativo de AP sobre a melhor baseline",
            "ap": "AP: {model}",
            "diff": "Diferença pareada de AP: {model} menos {reference}",
            "folds_won": "{won} de {total} folds vencidos",
            "excludes_zero": "IC exclui zero",
            "includes_zero": "IC inclui zero",
        },
    },
    "en": {
        "reliability": {
            "title": "Reliability curve (10 bins)",
            "x": "Mean predicted probability",
            "y": "Observed rate",
            "ideal": "perfect calibration",
            "orders": "orders in the bin",
        },
        "prevalence": {
            "title": "Orders and breach prevalence by purchase month",
            "month": "Purchase month",
            "orders": "Orders",
            "prevalence": "Breach prevalence",
            "train": "Train",
            "test": "Test",
        },
        "optuna": {
            "classification": "validation_average_precision",
            "regression": "validation_rmse",
        },
        "evidence": {
            "development": "Development (folds from 2017-10 to 2018-06)",
            "f1": "F1 (official)",
            "f2": "F2 (post-hoc, exploratory)",
            "ap_mean": "Mean AP of {model}",
            "ap_best_baseline": "Mean AP of the best baseline ({model})",
            "gain": "Relative AP gain over the best baseline",
            "ap": "AP: {model}",
            "diff": "Paired AP difference: {model} minus {reference}",
            "folds_won": "{won} of {total} folds won",
            "excludes_zero": "CI excludes zero",
            "includes_zero": "CI includes zero",
        },
    },
}


def get_report_labels(language: str = "pt") -> dict[str, dict[str, str]]:
    """Return the report labels for ``language``.

    Parameters
    ----------
    language : {"pt", "en"}, default "pt"
        Label language.

    Returns
    -------
    dict
        Nested mapping of label groups to label strings.

    Raises
    ------
    ValueError
        If ``language`` is not ``"pt"`` or ``"en"``.
    """
    if language not in REPORT_LABELS:
        raise ValueError("language must be 'pt' or 'en'")
    return REPORT_LABELS[language]


def plot_reliability_curves(
    reliability: pd.DataFrame,
    language: str = "pt",
    title: str | None = None,
    model_col: str = "model",
) -> go.Figure:
    """Plot one reliability curve per model against the diagonal.

    Parameters
    ----------
    reliability : pandas.DataFrame
        Columns ``model``, ``mean_predicted`` and ``observed_rate``; an optional
        ``n`` column (orders in the bin) appears in the hover. Bins without a
        value simply leave a gap in the line.
    language : {"pt", "en"}, default "pt"
        Label language.
    title : str, optional
        Overrides the default title (for example to name the evaluation).
    model_col : str, default "model"
        Name of the model column.

    Returns
    -------
    plotly.graph_objects.Figure
        One line-and-marker trace per model plus a dashed ideal diagonal.
    """
    text = get_report_labels(language)["reliability"]
    figure = go.Figure()
    for model, part in reliability.groupby(model_col, sort=False):
        counts = part["n"].to_numpy() if "n" in part.columns else np.full(len(part), np.nan)
        figure.add_trace(
            go.Scatter(
                x=part["mean_predicted"],
                y=part["observed_rate"],
                mode="lines+markers",
                name=str(model),
                customdata=counts,
                hovertemplate=(
                    f"{model}<br>{text['x']}: %{{x:.3f}}<br>{text['y']}: %{{y:.3f}}"
                    f"<br>{text['orders']}: %{{customdata:,.0f}}<extra></extra>"
                ),
            )
        )
    figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"dash": "dash", "color": IDEAL_COLOR},
            name=text["ideal"],
            hoverinfo="skip",
        )
    )
    figure.update_layout(
        title=title or text["title"],
        template=TEMPLATE,
        xaxis={"title": text["x"], "rangemode": "tozero"},
        yaxis={"title": text["y"], "rangemode": "tozero"},
        height=480,
        margin={"l": 60, "r": 40, "t": 80, "b": 60},
    )
    return figure


def plot_monthly_prevalence(monthly: pd.DataFrame, language: str = "pt") -> go.Figure:
    """Plot monthly order counts (bars, by split) and breach prevalence (line).

    Parameters
    ----------
    monthly : pandas.DataFrame
        Columns ``purchase_month``, ``split`` (``"train"`` or ``"test"``),
        ``orders`` and ``prevalence``.
    language : {"pt", "en"}, default "pt"
        Label language.

    Returns
    -------
    plotly.graph_objects.Figure
        One bar trace per split on the primary axis and a prevalence line on a
        secondary axis; months absent from the table (for example the month
        excluded by the embargo) are absent from the chart.
    """
    text = get_report_labels(language)["prevalence"]
    months = monthly["purchase_month"].astype(str)
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    for split, color in (("train", TRAIN_COLOR), ("test", TEST_COLOR)):
        part = monthly[monthly["split"] == split]
        figure.add_trace(
            go.Bar(
                x=part["purchase_month"].astype(str),
                y=part["orders"],
                name=text[split],
                marker={"color": color},
                hovertemplate="%{x}<br>" + text["orders"] + ": %{y:,.0f}<extra></extra>",
            ),
            secondary_y=False,
        )
    figure.add_trace(
        go.Scatter(
            x=months,
            y=monthly["prevalence"],
            mode="lines+markers",
            name=text["prevalence"],
            line={"color": LINE_COLOR},
            hovertemplate="%{x}<br>" + text["prevalence"] + ": %{y:.1%}<extra></extra>",
        ),
        secondary_y=True,
    )
    figure.update_layout(
        title=text["title"],
        template=TEMPLATE,
        xaxis={"title": text["month"], "type": "category"},
        barmode="stack",
        height=460,
        margin={"l": 60, "r": 60, "t": 80, "b": 80},
    )
    figure.update_yaxes(title_text=text["orders"], secondary_y=False)
    figure.update_yaxes(title_text=text["prevalence"], tickformat=".0%", rangemode="tozero", secondary_y=True)
    return figure


def optuna_study_table(summaries: Iterable[object], version: str = "V4_folds") -> pd.DataFrame:
    """Tabulate stored Optuna study summaries of one version.

    The function only reads attributes of the summaries (for example the objects
    returned by ``optuna.get_all_study_summaries``); it never imports Optuna and
    never runs a study. Study names are expected as ``<task>_<model>_<version>``.

    Parameters
    ----------
    summaries : iterable
        Objects with ``study_name``, ``n_trials``, ``direction`` (with a
        ``name`` attribute) and ``best_trial`` (``None`` or an object with
        ``value``).
    version : str, default "V4_folds"
        Study version suffix to keep; other versions are ignored.

    Returns
    -------
    pandas.DataFrame
        Columns ``task``, ``model``, ``trials``, ``metric``, ``direction``,
        ``best_value`` and ``rank`` (1 is best within the task), sorted by task
        and rank. Empty (with those columns) when no study matches.
    """
    suffix = f"_{version}"
    metrics = REPORT_LABELS["pt"]["optuna"]
    rows = []
    for summary in summaries:
        name = str(summary.study_name)
        if not name.endswith(suffix):
            continue
        task, _, model = name[: -len(suffix)].partition("_")
        best = summary.best_trial
        rows.append(
            {
                "task": task,
                "model": model,
                "trials": int(summary.n_trials),
                "metric": metrics.get(task, ""),
                "direction": str(getattr(summary.direction, "name", summary.direction)).lower(),
                "best_value": float(best.value) if best is not None and best.value is not None else np.nan,
            }
        )
    columns = ["task", "model", "trials", "metric", "direction", "best_value", "rank"]
    if not rows:
        return pd.DataFrame(columns=columns)
    table = pd.DataFrame(rows)
    table["rank"] = np.nan
    for _, part in table.groupby("task"):
        minimize = part["direction"].iloc[0] == "minimize"
        table.loc[part.index, "rank"] = part["best_value"].rank(method="min", ascending=minimize)
    return table.sort_values(["task", "rank", "model"]).reset_index(drop=True)[columns]


def evidence_table(
    classification_summary: pd.DataFrame,
    f1_metrics: pd.DataFrame,
    f1_paired: pd.DataFrame,
    f2_metrics: pd.DataFrame,
    f2_paired: pd.DataFrame,
    language: str = "pt",
    candidate: str = "CatBoost",
    reference: str = "baseline_logistic_promise_plus_state",
) -> pd.DataFrame:
    """Gather the key stored estimates of the three stages in one table.

    Parameters
    ----------
    classification_summary : pandas.DataFrame
        Tournament summary (``tournament_summary`` of the classification
        results) with ``model``, ``n_folds``, ``average_precision_mean``,
        ``average_precision_gain_vs_baseline``, ``average_precision_folds_won``
        and ``average_precision_best_baseline``.
    f1_metrics, f2_metrics : pandas.DataFrame
        Classification metrics with ``model``, ``metric``, ``estimate``,
        ``ci_low`` and ``ci_high``.
    f1_paired, f2_paired : pandas.DataFrame
        Paired differences with ``model``, ``reference``, ``difference``,
        ``ci_low`` and ``ci_high``. The F1 table carries ``outcome`` and
        ``outcome_en``; the F2 table carries ``reading``.
    language : {"pt", "en"}, default "pt"
        Label language.
    candidate : str, default "CatBoost"
        Model of interest in the tournament and in F1.
    reference : str, default "baseline_logistic_promise_plus_state"
        Reference of the paired differences.

    Returns
    -------
    pandas.DataFrame
        Columns ``stage``, ``item``, ``estimate``, ``ci_low``, ``ci_high`` and
        ``note``, one row per quantity. The inputs are not modified.
    """
    text = get_report_labels(language)["evidence"]
    rows: list[dict[str, object]] = []

    def add(stage: str, item: str, estimate: float, low: float = np.nan, high: float = np.nan, note: str = ""):
        rows.append(
            {
                "stage": text[stage],
                "item": item,
                "estimate": float(estimate),
                "ci_low": float(low),
                "ci_high": float(high),
                "note": note,
            }
        )

    def interval_note(low: float, high: float) -> str:
        return text["excludes_zero"] if low > 0 or high < 0 else text["includes_zero"]

    summary = classification_summary.set_index("model")
    row = summary.loc[candidate]
    best = str(row["average_precision_best_baseline"])
    add("development", text["ap_mean"].format(model=candidate), row["average_precision_mean"])
    add("development", text["ap_best_baseline"].format(model=best), summary.loc[best, "average_precision_mean"])
    add(
        "development",
        text["gain"],
        row["average_precision_gain_vs_baseline"],
        note=text["folds_won"].format(
            won=int(row["average_precision_folds_won"]), total=int(row["n_folds"])
        ),
    )

    outcome_col = "outcome_en" if language == "en" else "outcome"
    f1_ap = f1_metrics[f1_metrics["metric"] == "average_precision"].drop_duplicates(subset=["model"])
    for _, item in f1_ap.iterrows():
        add("f1", text["ap"].format(model=item["model"]), item["estimate"], item["ci_low"], item["ci_high"])
    for _, item in f1_paired[f1_paired["reference"] == reference].iterrows():
        note = str(item[outcome_col]) if outcome_col in item.index else interval_note(item["ci_low"], item["ci_high"])
        add(
            "f1",
            text["diff"].format(model=item["model"], reference=reference),
            item["difference"],
            item["ci_low"],
            item["ci_high"],
            note,
        )

    f2_ap = f2_metrics[f2_metrics["metric"] == "average_precision"].drop_duplicates(subset=["model"])
    for _, item in f2_ap.iterrows():
        note = str(item["status"]) if pd.isna(item["estimate"]) and "status" in item.index else ""
        add("f2", text["ap"].format(model=item["model"]), item["estimate"], item["ci_low"], item["ci_high"], note)
    for _, item in f2_paired[f2_paired["reference"] == reference].iterrows():
        if pd.isna(item["difference"]):
            note = str(item["status"]) if "status" in item.index else ""
        else:
            note = interval_note(item["ci_low"], item["ci_high"])
            if "reading" in item.index:
                note = f"{item['reading']}; {note}"
        add(
            "f2",
            text["diff"].format(model=item["model"], reference=reference),
            item["difference"],
            item["ci_low"],
            item["ci_high"],
            note,
        )
    return pd.DataFrame(rows)
