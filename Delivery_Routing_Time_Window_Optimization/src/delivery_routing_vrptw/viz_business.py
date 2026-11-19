"""Plots of the business scenarios (layer B; labels come from :mod:`labels`)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .business import BASE_TOLERANCE, fleet_class, tolerance_text
from .io import FAMILIES
from .labels import get_labels
from .route_compare import REALLOC_METRICS, reallocation_summary
from .viz_final import FAMILY_COLORS

MARKERS = dict(zip(FAMILIES, ("o", "s", "^", "D", "v", "P"), strict=True))


def _family_lines(ax, wide: pd.DataFrame, x_labels: list[str]) -> None:
    x = np.arange(len(x_labels))
    for fam in wide.index:
        ax.plot(x, wide.loc[fam].to_numpy(dtype=float), marker=MARKERS[fam], color=FAMILY_COLORS[fam], label=fam)
    ax.set_xticks(x, x_labels)
    ax.axhline(0.0, color="0.4", lw=0.8, ls="--")


def plot_window_effect(summary: pd.DataFrame, lang: str = "en") -> Figure:
    """Draw the window-widening effect by factor and family: vehicles saved and distance change.

    Parameters
    ----------
    summary : pandas.DataFrame
        Output of :func:`delivery_routing_vrptw.business.window_results_summary`.
    lang : str
        Label language.

    Returns
    -------
    matplotlib.figure.Figure
        Panel (a): mean vehicles saved against the factor; panel (b): mean distance change (%) against the
        factor-1.0 plan of the same instance, **only for the instances whose fleet stayed equal** (distance
        is not comparable across different fleets), one line per family.
    """
    text = get_labels(lang)["plots"]
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for ax, column, title, ylabel in (
        (ax_a, "vehicles_saved", text["window_veh_title"], text["window_veh_y"]),
        (ax_b, "distance_change_pct", text["window_dist_title"], text["window_dist_y"]),
    ):
        data = summary if column == "vehicles_saved" else summary[summary["vehicles_saved"].map(fleet_class) == "equal"]
        wide = data.pivot_table(index="family", columns="factor", values=column, aggfunc="mean")
        wide = wide.reindex(index=[f for f in FAMILIES if f in summary["family"].unique()],
                            columns=sorted(summary["factor"].unique()))
        _family_lines(ax, wide, [f"{c:g}×" for c in wide.columns])
        ax.set_xlabel(text["window_x"])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    ax_b.legend(title=text["family_x"], fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    return fig


def plot_early_effect(summary: pd.DataFrame, lang: str = "en") -> Figure:
    """Draw the early-delivery effect: duration under both departure rules, then duration and fleet by family.

    Parameters
    ----------
    summary : pandas.DataFrame
        Output of :func:`delivery_routing_vrptw.business.early_results_summary`.
    lang : str
        Label language.

    Returns
    -------
    matplotlib.figure.Figure
        Panel (a): mean change in route duration with departures at 0 and with optimal departures; panel (b):
        change in duration (optimal departures) per family; both only over the instances whose fleet stayed
        equal to the baseline plan. Panel (c): mean vehicles saved per family (all instances).
    """
    text = get_labels(lang)["plots"]
    sub = summary[summary["tolerance_label"] != BASE_TOLERANCE]
    order = sub.drop_duplicates("tolerance_label").sort_values("tolerance")["tolerance_label"].tolist()
    equal = sub[sub["vehicles_saved"].map(fleet_class) == "equal"]
    names = [tolerance_text(t, lang) for t in order]
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), layout="constrained")
    ax_a, ax_b, ax_c = axes
    for column, label, color, marker in (
        ("duration_pct", text["early_depart0"], "#D55E00", "o"),
        ("duration_min_pct", text["early_optimal"], "#0072B2", "s"),
    ):
        mean = equal.groupby("tolerance_label")[column].mean().reindex(order)
        ax_a.plot(x, mean.to_numpy(dtype=float), marker=marker, color=color, label=label)
    ax_a.set_xticks(x, names)
    ax_a.axhline(0.0, color="0.4", lw=0.8, ls="--")
    ax_a.legend(fontsize=8)
    ax_a.set_title(text["early_a_title"])
    ax_a.set_ylabel(text["early_dur_y"])
    for ax, column, title, ylabel in (
        (ax_b, "duration_min_pct", text["early_b_title"], text["early_dur_y"]),
        (ax_c, "vehicles_saved", text["early_c_title"], text["early_veh_y"]),
    ):
        data = sub if column == "vehicles_saved" else equal
        wide = data.pivot_table(index="family", columns="tolerance_label", values=column, aggfunc="mean")
        wide = wide.reindex(index=[f for f in FAMILIES if f in sub["family"].unique()], columns=order)
        _family_lines(ax, wide, names)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
    ax_c.legend(title=text["family_x"], fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    for ax in axes:
        ax.set_xlabel(text["early_x"])
    return fig


def plot_reallocation(table: pd.DataFrame, lang: str = "en") -> Figure:
    """Draw the plan-to-plan reallocation by tolerance, next to the run-to-run noise floor.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`delivery_routing_vrptw.route_compare.reallocation_table` (including ``noise`` rows).
    lang : str
        Label language.

    Returns
    -------
    matplotlib.figure.Figure
        Grouped bars (medians, in percent): customers reassigned, pairs kept together and arcs kept, one group
        per tolerance with the noise floor last.
    """
    text = get_labels(lang)["plots"]
    summ = reallocation_summary(table)
    metrics = [m for m in REALLOC_METRICS if m != "pair_agreement"]
    colors = {"customers_reassigned_share": "#D55E00", "pairs_kept": "#0072B2", "arcs_kept": "#009E73"}
    fig, ax = plt.subplots(figsize=(8.5, 4.2), layout="constrained")
    x = np.arange(len(summ))
    width = 0.26
    for k, metric in enumerate(metrics):
        values = 100 * summ[metric].to_numpy(dtype=float)
        bars = ax.bar(x + (k - 1) * width, values, width, color=colors[metric], label=text[f"realloc_{metric}"])
        ax.bar_label(bars, fmt="%.0f", fontsize=7, padding=1)
    ax.set_xticks(x, [tolerance_text(t, lang) for t in summ["tolerance_label"]])
    ax.set_ylim(0, 112)
    ax.set_xlabel(text["early_x"])
    ax.set_ylabel(text["realloc_y"])
    ax.set_title(text["realloc_title"])
    ax.legend(fontsize=8, ncol=3, loc="upper center")
    return fig


def plot_warm_start(levels: pd.DataFrame, lang: str = "en") -> Figure:
    """Draw the mean vehicles saved per level with a cold start (v1) and with a warm start between levels (v2).

    Parameters
    ----------
    levels : pandas.DataFrame
        Output of :func:`delivery_routing_vrptw.v2_view.layerb_levels` (``scenario``, ``level``,
        ``v1_vehicles_saved_mean`` and ``v2_vehicles_saved_mean``).
    lang : str
        Label language.

    Returns
    -------
    matplotlib.figure.Figure
        One panel per scenario; paired bars per level, v1 against v2.
    """
    text = get_labels(lang)["plots"]
    scenarios = list(dict.fromkeys(levels["scenario"]))
    fig, axes = plt.subplots(1, len(scenarios), figsize=(5.2 * len(scenarios), 4.0), layout="constrained")
    for ax, name in zip(np.atleast_1d(axes), scenarios, strict=True):
        part = levels[levels["scenario"] == name].reset_index(drop=True)
        x = np.arange(len(part))
        for i, (col, label, color) in enumerate((
            ("v1_vehicles_saved_mean", text["warm_v1"], "#7a7a7a"), ("v2_vehicles_saved_mean", text["warm_v2"], "#1b6ca8"),
        )):
            bars = ax.bar(x + (i - 0.5) * 0.38, part[col], 0.38, color=color, label=label)
            ax.bar_label(bars, fmt="%.2f", fontsize=8)
        ax.set_xticks(x, part["level"])
        ax.set_title(name)
        ax.set_ylabel(text["warm_y"])
    np.atleast_1d(axes)[0].legend(fontsize=8)
    return fig
