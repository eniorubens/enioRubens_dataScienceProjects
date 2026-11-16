"""Plotly figures built from already-computed result tables.

Every function takes DataFrames produced elsewhere (tables shown by the
notebooks), returns a :class:`plotly.graph_objects.Figure` and never mutates
its inputs. Nothing here trains, scores or reads files.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .labels import get_labels

TRAIN_COLOR = "#2563EB"
VALIDATION_COLOR = "#0F766E"
NEUTRAL_COLOR = "#9CA3AF"
WARM_COLOR = "#F59E0B"
ACCENT_COLORS = ["#2563EB", "#F59E0B", "#0F766E", "#DC2626", "#7C3AED", "#DB2777", "#475569"]
REFERENCE_COLOR = "#111827"
PSI_FLOOR = 1e-4
PSI_MODERATE = 0.10
PSI_LARGE = 0.25
TEMPLATE = "plotly_white"


def _humanize(name: str) -> str:
    """Turn a column name such as ``average_precision`` into readable text."""
    return name.replace("_", " ")


def _pct(value: float) -> str:
    """Format a fraction as a percentage, or an empty string for NaN."""
    return "" if pd.isna(value) else f"{value:.1%}"


def plot_expanding_folds(summary: pd.DataFrame, language: str = "pt") -> go.Figure:
    """Plot expanding-window folds as a timeline plus a prevalence panel.

    The left panel draws, per fold, the train span, the embargo gap between
    the end of training and the start of validation, and the validation span.
    The right panel compares train and validation prevalence. Folds are listed
    chronologically from top to bottom.

    Parameters
    ----------
    summary : pandas.DataFrame
        Output of ``expanding_fold_summary`` with columns ``fold``,
        ``train_start``, ``train_end``, ``validation_start``, ``validation_end``,
        ``train_orders``, ``validation_orders``, ``train_prevalence`` and
        ``validation_prevalence``.
    language : str, default "pt"
        ``"pt"`` or ``"en"``.

    Returns
    -------
    plotly.graph_objects.Figure
        Two-panel figure sharing the fold axis.
    """
    text = get_labels(language)["folds"]
    frame = summary.sort_values("validation_start").reset_index(drop=True)
    names = [f"{text['fold']} {fold}" for fold in frame["fold"]]
    figure = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        column_widths=[0.62, 0.38],
        horizontal_spacing=0.09,
        subplot_titles=(text["timeline_panel"], text["prevalence_panel"]),
    )
    legend_seen: set[str] = set()

    def add_span(name: str, key: str, start: object, end: object, color: str, dash: str, hover: str, label: str):
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        middle = start + (end - start) / 2
        figure.add_trace(
            go.Scatter(
                x=[start, middle, end],
                y=[name] * 3,
                mode="lines+text" if label else "lines",
                line={"color": color, "width": 16, "dash": dash},
                text=["", label, ""],
                textposition="middle center",
                textfont={"color": "white", "size": 11},
                hovertemplate=hover + "<br>%{x|%Y-%m-%d}<extra></extra>",
                name=text[key],
                legendgroup=key,
                showlegend=key not in legend_seen,
            ),
            row=1,
            col=1,
        )
        legend_seen.add(key)

    for name, row in zip(names, frame.itertuples(index=False), strict=True):
        train_orders = f"{int(row.train_orders):,} {text['orders']}"
        validation_orders = f"{int(row.validation_orders):,} {text['orders']}"
        add_span(
            name, "train", row.train_start, row.train_end, TRAIN_COLOR, "solid",
            f"{text['train']}: {train_orders}", train_orders,
        )
        add_span(name, "embargo", row.train_end, row.validation_start, NEUTRAL_COLOR, "dot", text["embargo"], "")
        add_span(
            name, "validation", row.validation_start, row.validation_end, VALIDATION_COLOR, "solid",
            f"{text['validation']}: {validation_orders}", "",
        )
    for key, column, color in (
        ("train", "train_prevalence", TRAIN_COLOR),
        ("validation", "validation_prevalence", VALIDATION_COLOR),
    ):
        figure.add_trace(
            go.Bar(
                x=frame[column].astype(float),
                y=names,
                orientation="h",
                marker={"color": color},
                text=[_pct(value) for value in frame[column]],
                textposition="outside",
                hovertemplate=f"{text[key]}: %{{x:.2%}}<extra></extra>",
                name=text[key],
                legendgroup=key,
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    figure.update_layout(
        title=text["title"],
        template=TEMPLATE,
        barmode="group",
        height=max(420, 90 + 46 * len(names)),
        margin={"l": 60, "r": 40, "t": 90, "b": 60},
        legend={"orientation": "h", "y": -0.12},
    )
    figure.update_yaxes(categoryorder="array", categoryarray=names, autorange="reversed")
    figure.update_xaxes(title_text=text["date"], type="date", nticks=5, tickformat="%b %Y", row=1, col=1)
    top = float(max(frame["train_prevalence"].max(), frame["validation_prevalence"].max(), 0.0)) if len(frame) else 0.0
    figure.update_xaxes(
        title_text=text["prevalence"], tickformat=".0%", range=[0, max(top * 1.3, 0.01)], nticks=4, row=1, col=2
    )
    return figure


def plot_psi(
    psi_table: pd.DataFrame,
    top_n: int | None = None,
    neutral_features: Iterable[str] = (),
    highlight_col: str | None = None,
    language: str = "pt",
    feature_col: str = "feature",
    psi_col: str = "psi",
) -> go.Figure:
    """Plot the population stability index per feature on a log axis.

    Bars are sorted from the largest PSI at the top. Dashed vertical lines mark
    the conventional 0.10 and 0.25 reference values. Features listed in
    ``neutral_features`` are drawn in grey, because their PSI is high by
    construction (calendar variables), and get their own legend entry.

    Parameters
    ----------
    psi_table : pandas.DataFrame
        Table with one row per feature (columns ``feature`` and ``psi``).
    top_n : int, optional
        Keep only the ``top_n`` largest PSI values; ``None`` shows all rows.
    neutral_features : iterable of str, default ()
        Feature names drawn in the neutral grey.
    highlight_col : str, optional
        Column whose value ``"other"`` means "not highlighted"; any other value
        (for example ``"promise"`` or ``"network_rate"``) gets its own colour.
        Neutral features take precedence over highlights.
    language : str, default "pt"
        ``"pt"`` or ``"en"``.
    feature_col, psi_col : str
        Names of the feature and PSI columns.

    Returns
    -------
    plotly.graph_objects.Figure
        Horizontal bar chart with a log-scaled x axis. PSI values below
        ``1e-4`` are drawn at ``1e-4``; hover shows the exact value.
    """
    text = get_labels(language)["psi"]
    neutral = set(neutral_features)
    frame = psi_table.dropna(subset=[psi_col]).sort_values(psi_col, ascending=False)
    if top_n is not None:
        frame = frame.head(int(top_n))
    frame = frame.reset_index(drop=True)

    groups = []
    for _, row in frame.iterrows():
        if row[feature_col] in neutral:
            groups.append("neutral")
        elif highlight_col is not None and str(row[highlight_col]) != "other":
            groups.append(str(row[highlight_col]))
        else:
            groups.append("other")
    group_series = pd.Series(groups, index=frame.index, dtype=object)
    colors = {"neutral": NEUTRAL_COLOR, "other": TRAIN_COLOR}
    extra = [name for name in dict.fromkeys(groups) if name not in colors]
    for index, name in enumerate(extra):
        colors[name] = ACCENT_COLORS[(index + 1) % len(ACCENT_COLORS)]
    figure = go.Figure()
    for name in ["other", *extra, "neutral"]:
        part = frame[group_series == name]
        if part.empty:
            continue
        figure.add_trace(
            go.Bar(
                x=part[psi_col].astype(float).clip(lower=PSI_FLOOR),
                y=part[feature_col],
                orientation="h",
                marker={"color": colors[name]},
                customdata=part[psi_col].astype(float),
                hovertemplate="%{y}<br>PSI = %{customdata:.4f}<extra></extra>",
                name=text.get(name, name),
            )
        )
    features = frame[feature_col].tolist()
    if features:
        for value, key in ((PSI_MODERATE, "moderate"), (PSI_LARGE, "large")):
            figure.add_trace(
                go.Scatter(
                    x=[value, value],
                    y=[features[-1], features[0]],
                    mode="lines",
                    line={"color": REFERENCE_COLOR, "dash": "dash" if key == "moderate" else "dot", "width": 1.5},
                    name=f"{text['reference']} {text[key]}",
                    hoverinfo="skip",
                )
            )
    figure.update_layout(
        title=text["title"],
        template=TEMPLATE,
        xaxis={"title": text["xaxis"], "type": "log"},
        yaxis={
            "title": text["yaxis"],
            "categoryorder": "array",
            "categoryarray": features,
            "autorange": "reversed",
            "automargin": True,
        },
        height=max(380, 120 + 24 * len(features)),
        margin={"l": 60, "r": 40, "t": 80, "b": 60},
        legend={"orientation": "h", "y": -0.15},
    )
    return figure


def plot_metric_intervals(
    metrics: pd.DataFrame,
    metric: str = "average_precision",
    paired: pd.DataFrame | None = None,
    language: str = "pt",
    model_col: str = "model",
) -> go.Figure:
    """Plot point estimates with 95% intervals and, optionally, paired differences.

    The left panel shows the estimate of ``metric`` with its confidence interval
    for each model. When ``paired`` is given, a right panel shows the paired
    differences with their intervals against a vertical zero line; each
    reference gets its own colour and marker, offset vertically so both stay
    readable. Rows with a missing estimate or difference (for example models
    that did not pass a gate) are skipped.

    Parameters
    ----------
    metrics : pandas.DataFrame
        Columns ``model``, ``estimate``, ``ci_low`` and ``ci_high``; an optional
        ``metric`` column is used to select ``metric``.
    metric : str, default "average_precision"
        Metric to display.
    paired : pandas.DataFrame, optional
        Columns ``model``, ``reference``, ``difference``, ``ci_low`` and
        ``ci_high`` (optional ``metric`` column).
    language : str, default "pt"
        ``"pt"`` or ``"en"``.
    model_col : str, default "model"
        Name of the model column in both tables.

    Returns
    -------
    plotly.graph_objects.Figure
        One or two panels sharing the model axis.
    """
    text = get_labels(language)["intervals"]
    estimates = metrics
    if "metric" in estimates.columns:
        estimates = estimates[estimates["metric"] == metric]
    estimates = estimates.dropna(subset=["estimate"]).drop_duplicates(subset=[model_col])
    models = estimates[model_col].tolist()

    differences = None
    if paired is not None:
        differences = paired
        if "metric" in differences.columns:
            differences = differences[differences["metric"] == metric]
        differences = differences.dropna(subset=["difference"])
        models += [name for name in dict.fromkeys(differences[model_col]) if name not in models]
    position = {name: index for index, name in enumerate(models)}

    two_panels = differences is not None
    metric_label = get_labels(language)["metric_names"].get(metric, _humanize(metric))
    figure = make_subplots(
        rows=1,
        cols=2 if two_panels else 1,
        shared_yaxes=True,
        horizontal_spacing=0.08,
        subplot_titles=(text["estimate_panel"], text["paired_panel"]) if two_panels else None,
    )
    figure.add_trace(
        go.Scatter(
            x=estimates["estimate"].astype(float),
            y=[position[name] for name in estimates[model_col]],
            mode="markers",
            marker={"color": TRAIN_COLOR, "size": 9},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": (estimates["ci_high"] - estimates["estimate"]).astype(float),
                "arrayminus": (estimates["estimate"] - estimates["ci_low"]).astype(float),
                "color": TRAIN_COLOR,
            },
            text=estimates[model_col],
            hovertemplate="%{text}<br>%{x:.4f}<extra></extra>",
            name=metric_label,
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    if differences is not None:
        references = list(dict.fromkeys(differences["reference"]))
        symbols = ["diamond", "square", "triangle-up", "circle"]
        for index, reference in enumerate(references):
            part = differences[differences["reference"] == reference]
            offset = (index - (len(references) - 1) / 2) * 0.25
            color = ACCENT_COLORS[(index + 1) % len(ACCENT_COLORS)]
            figure.add_trace(
                go.Scatter(
                    x=part["difference"].astype(float),
                    y=[position[name] + offset for name in part[model_col]],
                    mode="markers",
                    marker={"color": color, "size": 9, "symbol": symbols[index % len(symbols)]},
                    error_x={
                        "type": "data",
                        "symmetric": False,
                        "array": (part["ci_high"] - part["difference"]).astype(float),
                        "arrayminus": (part["difference"] - part["ci_low"]).astype(float),
                        "color": color,
                    },
                    text=part[model_col],
                    hovertemplate="%{text}<br>" + f"{text['reference']}: {reference}" + "<br>%{x:.4f}<extra></extra>",
                    name=f"{text['reference']}: {reference}",
                ),
                row=1,
                col=2,
            )
        figure.add_vline(x=0, line={"color": REFERENCE_COLOR, "dash": "dash", "width": 1}, row=1, col=2)
    figure.update_layout(
        title=text["title_paired"] if two_panels else text["title_single"],
        template=TEMPLATE,
        height=max(360, 140 + 56 * len(models)),
        margin={"l": 60, "r": 40, "t": 90, "b": 60},
        legend={"orientation": "h", "y": -0.18},
    )
    figure.update_yaxes(
        tickvals=list(range(len(models))),
        ticktext=models,
        autorange="reversed",
        automargin=True,
    )
    figure.update_yaxes(title_text=text["model"], col=1)
    figure.update_xaxes(title_text=metric_label, col=1)
    if two_panels:
        figure.update_xaxes(title_text=f"{text['difference']} ({metric_label})", col=2)
    return figure


def plot_metric_by_fold(
    frame: pd.DataFrame,
    fold_col: str,
    model_col: str,
    metric_col: str,
    reference: str | None = None,
    language: str = "pt",
    title: str | None = None,
    metric_label: str | None = None,
    models: Iterable[str] | None = None,
    lower_is_better: bool = False,
) -> go.Figure:
    """Plot one metric across folds, with one line per model or variant.

    Parameters
    ----------
    frame : pandas.DataFrame
        Long table with one row per (fold, model).
    fold_col, model_col, metric_col : str
        Column names for the fold, the model/variant and the metric.
    reference : str, optional
        Value of ``model_col`` drawn as a dashed dark line (for example
        ``"F1_config"`` or the best baseline). It must be present in ``frame``.
    language : str, default "pt"
        ``"pt"`` or ``"en"``.
    title : str, optional
        Overrides the default title.
    metric_label : str, optional
        Axis label; defaults to the humanized ``metric_col``.
    models : iterable of str, optional
        Restrict the chart to these values of ``model_col`` (for example the
        candidate models plus the best baseline); ``None`` draws every value.
        ``reference`` is always kept when given.
    lower_is_better : bool, default False
        Append a "lower is better" note to the axis label (for errors such as
        MAE).

    Returns
    -------
    plotly.graph_objects.Figure
        Line chart with folds on the x axis.

    Raises
    ------
    KeyError
        If ``reference`` is not a value of ``model_col``.
    """
    text = get_labels(language)["by_fold"]
    available = list(dict.fromkeys(frame[model_col]))
    if reference is not None and reference not in available:
        raise KeyError(f"reference not found in {model_col!r}: {reference}")
    if models is None:
        model_names = available
    else:
        wanted = set(models)
        if reference is not None:
            wanted.add(reference)
        model_names = [name for name in available if name in wanted]
        frame = frame[frame[model_col].isin(model_names)]
    models = model_names
    label = metric_label or get_labels(language)["metric_names"].get(metric_col, _humanize(metric_col))
    axis_label = f"{label} ({text['lower_is_better']})" if lower_is_better else label
    figure = go.Figure()
    color_index = 0
    ordered = [name for name in models if name != reference] + ([reference] if reference is not None else [])
    for model in ordered:
        part = frame[frame[model_col] == model].sort_values(fold_col)
        is_reference = reference is not None and model == reference
        color = REFERENCE_COLOR
        if not is_reference:
            color = ACCENT_COLORS[color_index % len(ACCENT_COLORS)]
            color_index += 1
        figure.add_trace(
            go.Scatter(
                x=part[fold_col],
                y=part[metric_col].astype(float),
                mode="lines+markers",
                line={"color": color, "dash": "dash" if is_reference else "solid"},
                marker={"size": 7},
                name=f"{model} ({text['reference']})" if is_reference else str(model),
                hovertemplate=f"{model}<br>{text['fold']} %{{x}}: %{{y:.4f}}<extra></extra>",
            )
        )
    folds = sorted(frame[fold_col].unique())
    figure.update_layout(
        title=title or text["title"].format(metric=label),
        template=TEMPLATE,
        xaxis={"title": text["fold"], "tickmode": "array", "tickvals": folds, "ticktext": [str(f) for f in folds]},
        yaxis={"title": axis_label},
        height=460,
        margin={"l": 60, "r": 40, "t": 80, "b": 60},
        legend={"orientation": "h", "y": -0.2},
    )
    return figure


def plot_promise_regime(table: pd.DataFrame, language: str = "pt") -> go.Figure:
    """Plot the promised lead time (bars) and the promise-rule AUC (line) by month.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of ``promise_regime_table`` with columns ``purchase_month``,
        ``promise_median`` and ``auc_negative_promise``. Months with a missing
        AUC simply have no line point.
    language : str, default "pt"
        ``"pt"`` or ``"en"``.

    Returns
    -------
    plotly.graph_objects.Figure
        Bars on the primary axis, AUC line on a secondary axis and a dashed
        reference at 0.5.
    """
    text = get_labels(language)["regime"]
    months = table["purchase_month"].astype(str).tolist()
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Bar(
            x=months,
            y=table["promise_median"].astype(float),
            marker={"color": TRAIN_COLOR, "opacity": 0.55},
            name=text["promise_median"],
            hovertemplate="%{x}<br>%{y:.1f}<extra></extra>",
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=months,
            y=table["auc_negative_promise"].astype(float),
            mode="lines+markers",
            line={"color": WARM_COLOR},
            connectgaps=False,
            name=text["auc"],
            hovertemplate="%{x}<br>AUC = %{y:.3f}<extra></extra>",
        ),
        secondary_y=True,
    )
    if months:
        figure.add_trace(
            go.Scatter(
                x=[months[0], months[-1]],
                y=[0.5, 0.5],
                mode="lines",
                line={"color": REFERENCE_COLOR, "dash": "dash", "width": 1},
                name=text["chance"],
                hoverinfo="skip",
            ),
            secondary_y=True,
        )
    figure.update_layout(
        title=text["title"],
        template=TEMPLATE,
        xaxis={"title": text["month"], "type": "category"},
        height=460,
        margin={"l": 60, "r": 60, "t": 80, "b": 60},
        legend={"orientation": "h", "y": -0.25},
    )
    figure.update_yaxes(title_text=text["promise_median"], rangemode="tozero", secondary_y=False)
    figure.update_yaxes(title_text="AUC", range=[0, 1], showgrid=False, secondary_y=True)
    return figure


def plot_label_availability(table: pd.DataFrame, language: str = "pt") -> go.Figure:
    """Plot stacked bars of known and unknown labels by purchase month.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of ``label_availability_table`` with columns ``purchase_month``,
        ``known_labels``, ``unknown_labels``, ``prevalence_known`` and
        ``prevalence_unknown``. The prevalence of each group appears in the bar
        text and in the hover.
    language : str, default "pt"
        ``"pt"`` or ``"en"``.

    Returns
    -------
    plotly.graph_objects.Figure
        Stacked bar chart with one trace per label group.
    """
    text = get_labels(language)["availability"]
    months = table["purchase_month"].astype(str).tolist()
    figure = go.Figure()
    for key, count, prevalence, color in (
        ("known", "known_labels", "prevalence_known", TRAIN_COLOR),
        ("unknown", "unknown_labels", "prevalence_unknown", WARM_COLOR),
    ):
        figure.add_trace(
            go.Bar(
                x=months,
                y=table[count].astype(float),
                marker={"color": color},
                text=[f"{text['prevalence']}: {_pct(value)}" if not pd.isna(value) else "" for value in table[prevalence]],
                textposition="auto",
                customdata=[_pct(value) for value in table[prevalence]],
                hovertemplate=(
                    "%{x}<br>%{y:,.0f} " + text["orders"].lower()
                    + f"<br>{text['prevalence']}: " + "%{customdata}<extra></extra>"
                ),
                name=text[key],
            )
        )
    figure.update_layout(
        title=text["title"],
        template=TEMPLATE,
        barmode="stack",
        xaxis={"title": text["month"], "type": "category"},
        yaxis={"title": text["orders"]},
        height=460,
        margin={"l": 60, "r": 40, "t": 80, "b": 60},
        legend={"orientation": "h", "y": -0.2},
    )
    return figure


def plot_gain_over_baseline(
    summary: pd.DataFrame,
    metric: str = "average_precision",
    threshold: float | None = None,
    language: str = "pt",
    model_col: str = "model",
    fold_results: pd.DataFrame | None = None,
    fold_col: str = "fold",
) -> go.Figure:
    """Plot the mean relative gain of each model over the best baseline.

    Baselines have no gain (NaN) and are skipped. The bar height is always the
    summary's ``<metric>_gain_vs_baseline`` (positive is better, also for the
    lower-is-better metrics ``mae`` and ``rmse``).

    When ``fold_results`` is given, the error bars show the standard deviation
    across folds of the per-fold gain against the best baseline of the *same*
    fold: ``(model - base) / base`` for higher-is-better metrics and
    ``(base - model) / base`` for ``mae``/``rmse``. The mean of those per-fold
    gains can differ slightly from the summary gain (mean of ratios versus
    ratio of means); the bar keeps the summary value and only the spread comes
    from the per-fold gains. Without ``fold_results`` no error bars are drawn,
    because the spread of the absolute metric is not the spread of the gain.

    Each bar is annotated with the folds won (for example ``8/9``), taken from
    ``<metric>_folds_won`` when present and otherwise computed from
    ``fold_results``.

    Parameters
    ----------
    summary : pandas.DataFrame
        Output of ``tournament_summary`` (needs ``<metric>_gain_vs_baseline``
        and, with ``fold_results``, ``<metric>_best_baseline``).
    metric : str, default "average_precision"
        Decision metric (``"mae"`` and ``"rmse"`` also work for regression).
    threshold : float, optional
        Gain criterion as a fraction (``0.05`` is 5%), drawn as a dashed line.
    language : str, default "pt"
        ``"pt"`` or ``"en"``.
    model_col : str, default "model"
        Name of the model column (in both tables).
    fold_results : pandas.DataFrame, optional
        Per-fold results (one row per fold and model, with a ``metric`` column).
    fold_col : str, default "fold"
        Name of the fold column in ``fold_results``.

    Returns
    -------
    plotly.graph_objects.Figure
        Bar chart of the gain in percent, sorted from largest to smallest.
    """
    text = get_labels(language)["gain"]
    gain_col = f"{metric}_gain_vs_baseline"
    best_col = f"{metric}_best_baseline"
    won_col = f"{metric}_folds_won"
    frame = summary.dropna(subset=[gain_col]).sort_values(gain_col, ascending=False).reset_index(drop=True)
    gains = frame[gain_col].astype(float) * 100
    lower = metric in ("mae", "rmse")

    fold_gain: dict[str, pd.Series] = {}
    if fold_results is not None and best_col in frame.columns:
        values = fold_results.pivot_table(index=fold_col, columns=model_col, values=metric, aggfunc="first")
        for name, best in zip(frame[model_col], frame[best_col], strict=True):
            if name in values.columns and best in values.columns:
                base = values[best]
                diff = base - values[name] if lower else values[name] - base
                fold_gain[name] = (diff / base.abs()).dropna()

    error = None
    if fold_gain:
        error = [float(fold_gain[n].std()) * 100 if n in fold_gain else float("nan") for n in frame[model_col]]

    labels = [""] * len(frame)
    for index, name in enumerate(frame[model_col]):
        won = None
        if won_col in frame.columns and not pd.isna(frame.loc[index, won_col]):
            won = int(frame.loc[index, won_col])
        elif name in fold_gain:
            won = int((fold_gain[name] > 0).sum())
        total = None
        if name in fold_gain:
            total = len(fold_gain[name])
        elif "n_folds" in frame.columns and not pd.isna(frame.loc[index, "n_folds"]):
            total = int(frame.loc[index, "n_folds"])
        if won is not None:
            labels[index] = f"{won}/{total}" if total is not None else str(won)
    figure = go.Figure(
        go.Bar(
            x=frame[model_col],
            y=gains,
            marker={"color": TRAIN_COLOR},
            error_y={"type": "data", "array": error, "color": REFERENCE_COLOR} if error is not None else None,
            text=labels,
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}<br>%{y:.2f}%<br>" + text["folds_won"] + ": %{text}<extra></extra>",
            name=_humanize(metric),
            showlegend=False,
        )
    )
    if threshold is not None:
        figure.add_hline(
            y=float(threshold) * 100,
            line={"color": REFERENCE_COLOR, "dash": "dash", "width": 1},
            annotation_text=f"{text['threshold']}: {float(threshold):.0%}",
            annotation_position="top left",
        )
    figure.update_layout(
        title=f"{text['title']} ({_humanize(metric)})",
        template=TEMPLATE,
        xaxis={"title": text["xaxis"]},
        yaxis={"title": text["yaxis"], "zeroline": True},
        height=460,
        margin={"l": 60, "r": 40, "t": 80, "b": 80},
    )
    return figure


def plot_ablation(
    summary: pd.DataFrame,
    metric: str | None = None,
    subgroup: bool = False,
    language: str = "pt",
) -> go.Figure:
    """Plot the primary metric by ablation variant, with spread across folds.

    Parameters
    ----------
    summary : pandas.DataFrame
        Output of ``ablation_summary`` (columns ``variant``, ``model``,
        ``<metric>_mean``, ``<metric>_std`` and optionally the delta column).
    metric : str, optional
        Metric name; defaults to ``average_precision`` or ``mae``, whichever
        has a ``_mean`` column in ``summary``.
    subgroup : bool, default False
        Use the multi-seller subgroup columns (prefix ``multi_seller_``).
    language : str, default "pt"
        ``"pt"`` or ``"en"``.

    Returns
    -------
    plotly.graph_objects.Figure
        Grouped bars (one trace per model) with standard-deviation error bars.

    Raises
    ------
    KeyError
        If no matching ``_mean`` column is found.
    """
    text = get_labels(language)["ablation"]
    prefix = "multi_seller_" if subgroup else ""
    if metric is None:
        candidates = [m for m in ("average_precision", "mae") if f"{prefix}{m}_mean" in summary.columns]
        if not candidates:
            raise KeyError("no <metric>_mean column found in the ablation summary")
        metric = candidates[0]
    mean_col, std_col = f"{prefix}{metric}_mean", f"{prefix}{metric}_std"
    delta_cols = [c for c in summary.columns if c.startswith(f"{prefix}{metric}_delta_vs_")]
    frame = summary.dropna(subset=[mean_col])
    variants = list(dict.fromkeys(frame["variant"]))
    figure = go.Figure()
    for index, (model, part) in enumerate(frame.groupby("model", sort=False)):
        custom = part[delta_cols[0]].map(_pct) if delta_cols else [""] * len(part)
        error = (
            {"type": "data", "array": part[std_col].astype(float), "color": REFERENCE_COLOR}
            if std_col in part.columns
            else None
        )
        figure.add_trace(
            go.Bar(
                x=part["variant"],
                y=part[mean_col].astype(float),
                error_y=error,
                marker={"color": ACCENT_COLORS[index % len(ACCENT_COLORS)]},
                customdata=custom,
                hovertemplate="%{x}<br>%{y:.4f}<br>delta: %{customdata}<extra>" + str(model) + "</extra>",
                name=str(model),
            )
        )
    figure.update_layout(
        title=f"{text['title']} ({_humanize(metric)})",
        template=TEMPLATE,
        barmode="group",
        xaxis={"title": text["xaxis"], "categoryorder": "array", "categoryarray": variants},
        yaxis={"title": f"{text['yaxis']} ({_humanize(metric)})"},
        height=460,
        margin={"l": 60, "r": 40, "t": 80, "b": 80},
        legend={"title": {"text": text["model"]}},
    )
    return figure
