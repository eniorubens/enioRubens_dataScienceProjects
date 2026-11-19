"""Plots of phase C (robustness and operational scenarios; labels come from :mod:`labels`)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from . import phase_c
from .labels import get_labels
from .phase_c_view import LEVEL_PRIORITY, METHODS, SCENARIO_ORDER, SLACK_ORDER

METHOD_COLORS = {"ortools": "#1b6ca8", "nearest_neighbor_tw": "#e08a1e", "greedy_deadline": "#7a7a7a"}


def plot_robustness(summary: pd.DataFrame, slack_lateness: pd.DataFrame, lang: str = "en") -> Figure:
    """Draw how fixed plans fail under delays: late customers by scenario and late rate by nominal slack.

    Parameters
    ----------
    summary : pandas.DataFrame
        ``summary`` of :func:`delivery_routing_vrptw.results.load_robustness` (the ``overall`` scope is used).
    slack_lateness : pandas.DataFrame
        ``slack_lateness`` of the same loader (the ``travel_0.10`` scenario is used).
    lang : str
        Label language.

    Returns
    -------
    matplotlib.figure.Figure
        Panel (a): share of customers served after the end of their window, by shock scenario and method;
        panel (b): late rate of a customer against the nominal slack bin of the customer, by method.
    """
    text, names = get_labels(lang)["plots"], get_labels(lang)["methods"]
    phase = get_labels(lang)["phase_c"]
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    overall = summary[summary["scope"] == "overall"]
    x = np.arange(len(SCENARIO_ORDER))
    width = 0.27
    for i, method in enumerate(METHODS):
        part = overall[overall["method"] == method].set_index("scenario").reindex(list(SCENARIO_ORDER))
        ax_a.bar(x + (i - 1) * width, 100.0 * part["late_customer_share"], width, color=METHOD_COLORS[method],
                 label=names[method])
    ax_a.set_xticks(x, [phase[f"scenario_short_{s}"] for s in SCENARIO_ORDER], fontsize=8)
    ax_a.set_ylabel(text["robust_late_y"])
    ax_a.set_title(text["robust_a_title"])
    ax_a.legend(fontsize=8)
    bins = slack_lateness[slack_lateness["scenario"] == "travel_0.10"]
    for method in METHODS:
        part = bins[bins["method"] == method].set_index("slack_bin").reindex(list(SLACK_ORDER))
        ax_b.plot(np.arange(len(SLACK_ORDER)), 100.0 * part["late_rate"], marker="o", color=METHOD_COLORS[method],
                  label=names[method])
    ax_b.set_xticks(np.arange(len(SLACK_ORDER)), list(SLACK_ORDER), fontsize=8)
    ax_b.set_xlabel(text["robust_slack_x"])
    ax_b.set_ylabel(text["robust_rate_y"])
    ax_b.set_title(text["robust_b_title"])
    return fig


def plot_fleet_priority(cells: pd.DataFrame, ref: pd.DataFrame, lang: str = "en") -> Figure:
    """Draw the customers left unserved with a limited fleet, with and without priority.

    Parameters
    ----------
    cells : pandas.DataFrame
        Cells of the C2.2 block.
    ref : pandas.DataFrame
        Reference plans (:func:`delivery_routing_vrptw.phase_c_view.reference_plans`).
    lang : str
        Label language.

    Returns
    -------
    matplotlib.figure.Figure
        Panel (a): customers dropped over the 14 instances; panel (b): share of the priority customers dropped,
        for each fleet limit (K minus 1 and K minus 2), uniform penalty against priority penalty.
    """
    text, phase = get_labels(lang)["plots"], get_labels(lang)["phase_c"]
    agg = phase_c.fleet_priority_aggregate(phase_c.fleet_priority_summary(cells, ref))
    agg = agg[agg["variant"].isin(["uniform", "priority"])]
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.5, 4.2), layout="constrained")
    x = np.arange(len(LEVEL_PRIORITY))
    for ax, column, title, ylabel, scale in (
        (ax_a, "dropped_total", text["fleet_a_title"], text["fleet_a_y"], 1.0),
        (ax_b, "priority_dropped_share", text["fleet_b_title"], text["fleet_b_y"], 100.0),
    ):
        for i, variant in enumerate(("uniform", "priority")):
            vals = [float(agg[(agg["level"] == lv) & (agg["variant"] == variant)][column].iloc[0]) * scale
                    for lv in LEVEL_PRIORITY]
            bars = ax.bar(x + (i - 0.5) * 0.36, vals, 0.36, label=phase[f"variant_{variant}"],
                          color=("#7a7a7a", "#1b6ca8")[i])
            ax.bar_label(bars, fmt="%.0f" if column == "dropped_total" else "%.1f", fontsize=8)
        ax.set_xticks(x, [lv.replace("K-", "K − ") for lv in LEVEL_PRIORITY])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    ax_a.legend(fontsize=8)
    return fig


def plot_reoptimisation(removal: pd.DataFrame, insertion: pd.DataFrame, lang: str = "en") -> Figure:
    """Draw how much of the plan is kept after removing or inserting 10% of the customers.

    Parameters
    ----------
    removal, insertion : pandas.DataFrame
        Cells of the two C3 blocks.
    lang : str
        Label language.

    Returns
    -------
    matplotlib.figure.Figure
        Panel (a): median share of common customers that change vehicle; panel (b): median share of arcs kept;
        bars for warm start and cold start (the representative replicate of each instance) and the median noise
        floor (two cold replicates of the same scenario; its distribution is in
        :func:`delivery_routing_vrptw.phase_c_view.reopt_noise_table`).
    """
    text, phase = get_labels(lang)["plots"], get_labels(lang)["phase_c"]
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.5, 4.2), layout="constrained")
    aggs = {}
    for key, cells in (("removal", removal), ("insertion", insertion)):
        aggs[key] = phase_c.reopt_aggregate_representative(cells, phase_c.noise_table(cells)).set_index("variant")
    colors = {"warm": "#1b6ca8", "cold": "#e08a1e", "noise": "#9a9a9a"}
    x = np.arange(2)
    for ax, column, ylabel in (
        (ax_a, "realloc_customers_reassigned_share", text["reopt_a_y"]),
        (ax_b, "realloc_arcs_kept", text["reopt_b_y"]),
    ):
        for i, variant in enumerate(("warm", "cold", "noise")):
            vals = [100.0 * float(aggs[k].loc[variant, column]) for k in ("removal", "insertion")]
            bars = ax.bar(x + (i - 1) * 0.27, vals, 0.27, color=colors[variant], label=phase[f"variant_{variant}"])
            ax.bar_label(bars, fmt="%.1f", fontsize=8)
        ax.set_xticks(x, [phase["event_removal"], phase["event_insertion"]])
        ax.set_ylabel(ylabel)
    ax_a.set_title(text["reopt_a_title"])
    ax_b.set_title(text["reopt_b_title"])
    ax_a.legend(fontsize=8)
    return fig
