"""Plots for baseline and tournament results (labels come from :mod:`labels`)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .distance import exact
from .io import Instance
from .labels import get_labels
from .verify import verify
from .viz import plot_routes

METAHEURISTIC_ORDER = ("GUIDED_LOCAL_SEARCH", "TABU_SEARCH", "SIMULATED_ANNEALING")
FIRST_SOLUTION_ORDER = ("PARALLEL_CHEAPEST_INSERTION", "PATH_CHEAPEST_ARC", "SAVINGS")
VERTICAL_STEP = 0.03
"""Vertical offset step (vehicles) between the evenly spaced offsets of the instances in the budget plot."""


def plot_construction_comparison(comparison: pd.DataFrame, lang: str = "en") -> Figure:
    """Draw two bar panels for the constructions: mean Δ vehicles and median distance excess.

    Parameters
    ----------
    comparison : pandas.DataFrame
        Output of :func:`delivery_routing_vrptw.analysis.construction_comparison`.
    lang : str
        Label language.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    labels = get_labels(lang)
    text = labels["plots"]
    short = [labels["short_methods"][k] for k in comparison["construction"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    panels = (("mean_delta_vehicles", text["constr_dv_title"], "#0072B2"),
              ("median_gap_all", text["constr_gap_title"], "#D55E00"))
    for ax, (column, title, color) in zip(axes, panels, strict=True):
        values = comparison[column].to_numpy(dtype=float)
        bars = ax.bar(short, values, color=color, alpha=0.85)
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
        ax.set_title(title)
        ax.set_ylim(0, float(np.nanmax(values)) * 1.15 if np.isfinite(values).any() else 1)
        ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    return fig


def plot_rank_heatmap(ranking: pd.DataFrame, lang: str = "en") -> Figure:
    """Draw the mean rank by first solution (rows) and metaheuristic (columns), one panel per fleet strategy.

    Parameters
    ----------
    ranking : pandas.DataFrame
        Ranking with columns ``fleet_strategy``, ``first_solution``, ``metaheuristic`` and ``mean_rank``.
    lang : str
        Label language.

    Returns
    -------
    matplotlib.figure.Figure
        The figure with a shared colour scale.
    """
    text = get_labels(lang)["plots"]
    fleets = list(dict.fromkeys(ranking["fleet_strategy"]))
    rows = [f for f in FIRST_SOLUTION_ORDER if f in set(ranking["first_solution"])]
    cols = [m for m in METAHEURISTIC_ORDER if m in set(ranking["metaheuristic"])]
    vmin, vmax = float(ranking["mean_rank"].min()), float(ranking["mean_rank"].max())
    fig, axes = plt.subplots(1, len(fleets), figsize=(5.2 * len(fleets), 4.4), squeeze=False,
                             layout="constrained")
    image = None
    for ax, fleet in zip(axes.ravel(), fleets, strict=True):
        sub = ranking[ranking["fleet_strategy"] == fleet]
        grid = sub.pivot(index="first_solution", columns="metaheuristic", values="mean_rank").reindex(
            index=rows, columns=cols)
        image = ax.imshow(grid.to_numpy(dtype=float), cmap="viridis_r", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(cols)), [c.replace("_", "\n") for c in cols], fontsize=8)
        first_panel = ax is axes.ravel()[0]
        ax.set_yticks(range(len(rows)), [r.replace("_", "\n") for r in rows] if first_panel else [],
                      fontsize=8)
        for i in range(len(rows)):
            for j in range(len(cols)):
                value = grid.iloc[i, j]
                if pd.isna(value):
                    continue
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if (value - vmin) / max(vmax - vmin, 1e-9) > 0.5 else "black")
        ax.set_title(text["rank_title"].format(fleet=fleet))
        ax.set_xlabel(text["rank_x"])
        if ax is axes.ravel()[0]:
            ax.set_ylabel(text["rank_y"])
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.85, label=text["rank_cbar"])
    return fig


def plot_route_panels(
    panels: Sequence[tuple[Instance, str, list[list[int]]]], lang: str = "en"
) -> Figure:
    """Draw route maps side by side; titles show the verified vehicles and distance.

    Parameters
    ----------
    panels : sequence of (Instance, str, list of list of int)
        One entry per panel: the instance, a short label and the routes.
    lang : str
        Label language.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    text = get_labels(lang)["plots"]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 5.6), squeeze=False)
    for ax, (inst, label, routes) in zip(axes.ravel(), panels, strict=True):
        report = verify(inst, routes, exact(inst))
        title = text["panel_title"].format(name=inst.name, label=label, vehicles=report.vehicles,
                                           distance=report.distance)
        plot_routes(inst, routes, ax=ax, title=title)
        ax.title.set_fontsize(8)
    fig.tight_layout()
    return fig


def plot_time_curve(df: pd.DataFrame, lang: str = "en") -> Figure:
    """Draw the time-budget sensitivity: mean Δ vehicles and vehicles per R/RC instance.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :func:`delivery_routing_vrptw.results.load_time_sensitivity` (columns ``instance``,
        ``family``, ``budget_s``, ``vehicles``, ``ref_vehicles``, ``delta_vehicles``).
    lang : str
        Label language.

    Returns
    -------
    matplotlib.figure.Figure
        Panel (a): mean Δ vehicles against the budget (log-scale x axis). Panel (b): vehicles against the
        budget, one line per R/RC instance, each with a dashed level at the SINTEF reference. Instances
        are drawn with a small deterministic vertical offset (evenly spaced multiples of ``VERTICAL_STEP`` centred on 0, assigned with a stride so neighbouring
        instances in the legend get distant offsets; at most about 0.14 vehicles) plus distinct markers and line styles, so coinciding lines stay visible;
        the true values are integers (stated in the axis label).
    """
    text = get_labels(lang)["plots"]
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    mean = df.groupby("budget_s")["delta_vehicles"].mean().sort_index()
    ax_a.plot(mean.index, mean.to_numpy(dtype=float), marker="o", color="#0072B2")
    ax_a.set_xscale("log")
    ax_a.set_xticks(list(mean.index), [f"{b:g}" for b in mean.index])
    ax_a.minorticks_off()
    ax_a.set_xlabel(text["budget_x"])
    ax_a.set_ylabel(text["budget_dv_y"])
    ax_a.set_title(text["budget_dv_title"])
    ax_a.set_ylim(bottom=0)
    sub = df[df["family"].str.startswith("R")]
    names = sorted(sub["instance"].unique())
    colors = plt.get_cmap("tab10")(np.arange(len(names)) % 10)
    markers = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "*")
    styles = ("-", "-.", ":")
    stride = next(s for s in range(3, 3 + len(names)) if math.gcd(s, len(names)) == 1)
    for k, (name, color) in enumerate(zip(names, colors, strict=True)):
        one = sub[sub["instance"] == name].sort_values("budget_s")
        ref = int(one["ref_vehicles"].iloc[0])
        shift = (((k * stride) % len(names)) - (len(names) - 1) / 2) * VERTICAL_STEP
        ax_b.plot(one["budget_s"], one["vehicles"] + shift, marker=markers[k % len(markers)], ms=4,
                  ls=styles[k % len(styles)], color=color, label=text["budget_ref"].format(name=name, ref=ref))
        ax_b.plot(one["budget_s"], np.full(len(one), ref + shift), color=color, ls="--", lw=0.8, alpha=0.6)
    ax_b.set_xscale("log")
    budgets = sorted(sub["budget_s"].unique())
    ax_b.set_xticks(budgets, [f"{b:g}" for b in budgets])
    ax_b.minorticks_off()
    ax_b.set_xlabel(text["budget_x"])
    ax_b.set_ylabel(text["budget_inst_y"])
    ax_b.set_title(text["budget_inst_title"])
    ax_b.legend(fontsize=7, ncol=2, loc="center left", bbox_to_anchor=(1.01, 0.5))
    return fig
