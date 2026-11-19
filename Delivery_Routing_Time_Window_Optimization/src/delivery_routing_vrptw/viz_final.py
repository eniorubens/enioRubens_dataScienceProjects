"""Plots of the final evaluation (labels come from :mod:`labels`)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .evaluation import ORTOOLS
from .final_eval import delta_counts, fleet_equal_table
from .io import FAMILIES
from .labels import get_labels

DELTA_COLORS = {"<0": "#0072B2", "0": "#009E73", "1": "#E69F00", "2": "#D55E00", "3+": "#CC79A7"}
FAMILY_COLORS = dict(zip(FAMILIES, ("#0072B2", "#56B4E9", "#D55E00", "#E69F00", "#009E73", "#CC79A7"), strict=True))


def plot_delta_distribution(table: pd.DataFrame, lang: str = "en") -> Figure:
    """Draw the number of instances per family, stacked by Δ vehicles (median replicate).

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`delivery_routing_vrptw.evaluation.instance_table`.
    lang : str
        Label language.

    Returns
    -------
    matplotlib.figure.Figure
        Stacked bars, one per family; classes with no instance are omitted from the legend.
    """
    text = get_labels(lang)["plots"]
    counts = delta_counts(table)
    fig, ax = plt.subplots(figsize=(7.5, 4.2), layout="constrained")
    bottom = np.zeros(len(counts))
    for cls in counts.columns:
        values = counts[cls].to_numpy(dtype=float)
        if not values.any():
            continue
        bars = ax.bar(counts.index, values, bottom=bottom, color=DELTA_COLORS[cls],
                      label=text["delta_class"].format(delta=cls if cls in ("<0", "0", "3+") else f"+{cls}"))
        ax.bar_label(bars, labels=[f"{int(v)}" if v else "" for v in values], label_type="center", fontsize=8,
                     color="white")
        bottom += values
    ax.set_xlabel(text["family_x"])
    ax.set_ylabel(text["instances_y"])
    ax.set_title(text["delta_dist_title"])
    ax.legend(title=text["delta_legend"], fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return fig


def plot_gap_fleet_equal(table: pd.DataFrame, lang: str = "en") -> Figure:
    """Draw the distance gap of the instances whose fleet equals the reference, coloured by family.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`delivery_routing_vrptw.evaluation.instance_table`.
    lang : str
        Label language.

    Returns
    -------
    matplotlib.figure.Figure
        One bar per fleet-equal instance (sorted by family and name); zero-height bars are instances that
        reproduce the reference distance to the cent and are marked with a dot. Instances above the reference
        fleet have no gap.
    """
    text = get_labels(lang)["plots"]
    eq = fleet_equal_table(table, lang)
    fig, ax = plt.subplots(figsize=(11, 4.2), layout="constrained")
    x = np.arange(len(eq))
    colors = [FAMILY_COLORS[f] for f in eq["family"]]
    ax.bar(x, eq["gap_distance_pct"].to_numpy(dtype=float), color=colors)
    ax.set_xticks(x, eq["instance"], rotation=70, fontsize=8)
    ax.axhline(0.0, color="0.3", lw=0.8)
    repro = (eq["reproduces"] == get_labels(lang)["yesno"]["yes"]).to_numpy()
    handles = [Patch(color=FAMILY_COLORS[f], label=f) for f in FAMILIES if (eq["family"] == f).any()]
    if repro.any():
        ax.plot(x[repro], np.zeros(int(repro.sum())), "o", color="0.15", ms=5, clip_on=False)
        handles.append(Line2D([], [], marker="o", ls="", color="0.15", label=text["gap_reproduces"]))
    ax.legend(handles=handles, title=text["family_x"], fontsize=8, ncol=1, loc="upper center",
              bbox_to_anchor=(0.45, 1.0))
    ax.set_ylabel(text["gap_y"])
    ax.set_title(text["gap_equal_title"])
    return fig


def plot_time_metrics(agg: pd.DataFrame, lang: str = "en") -> Figure:
    """Draw how much of the OR-Tools route duration is waiting, and distance versus duration per method.

    Parameters
    ----------
    agg : pandas.DataFrame
        Output of :func:`delivery_routing_vrptw.final_eval.time_aggregate`.
    lang : str
        Label language.

    Returns
    -------
    matplotlib.figure.Figure
        Panel (a): per family, median share of waiting in the OR-Tools route duration with every vehicle
        leaving at 0 and with the optimal departure. Panel (b): per method, median per-instance distance and
        duration (optimal departure) as a percentage of the OR-Tools plan of the same instance.
    """
    text = get_labels(lang)
    plots, methods = text["plots"], text["methods"]
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    fam = agg[(agg["method"] == ORTOOLS) & (agg["scope"] != "overall")]
    x = np.arange(len(fam))
    for k, (col, label, color) in enumerate((
        ("wait_share_pct", plots["early_depart0"], "#D55E00"),
        ("wait_share_min_pct", plots["early_optimal"], "#0072B2"),
    )):
        bars = ax_a.bar(x + (k - 0.5) * 0.38, fam[col].to_numpy(dtype=float), 0.38, color=color, label=label)
        ax_a.bar_label(bars, fmt="%.0f", fontsize=7, padding=1)
    ax_a.set_xticks(x, fam["scope"])
    ax_a.set_xlabel(plots["family_x"])
    ax_a.set_ylabel(plots["time_share_y"])
    ax_a.set_title(plots["time_a_title"])
    ax_a.legend(fontsize=8)
    over = agg[agg["scope"] == "overall"]
    xm = np.arange(len(over))
    for k, (col, label, color) in enumerate((
        ("ratio_distance_pct", plots["time_distance"], "#009E73"),
        ("ratio_duration_min_pct", plots["time_duration_min"], "#0072B2"),
    )):
        bars = ax_b.bar(xm + (k - 0.5) * 0.38, over[col].to_numpy(dtype=float), 0.38, color=color, label=label)
        ax_b.bar_label(bars, fmt="%.0f", fontsize=7, padding=1)
    ax_b.axhline(100.0, color="0.4", lw=0.8, ls="--")
    ax_b.set_xticks(xm, [methods[m] for m in over["method"]], fontsize=8)
    ax_b.set_ylabel(plots["time_ratio_y"])
    ax_b.set_title(plots["time_b_title"])
    ax_b.legend(fontsize=8)
    return fig


def plot_version_comparison(by_instance: pd.DataFrame, lang: str = "en") -> Figure:
    """Draw the distance difference between v2 and v1 for every instance, with the noise band.

    Parameters
    ----------
    by_instance : pandas.DataFrame
        First output of :func:`delivery_routing_vrptw.evaluation.compare_versions`.
    lang : str
        Label language.

    Returns
    -------
    matplotlib.figure.Figure
        One point per instance with equal fleets in both versions (distance difference of v2 relative to v1, in
        percent, coloured by family), the band of the noise threshold, and a marker at the bottom for the
        instances whose fleet changed (their distance is not comparable).
    """
    from .evaluation import NOISE_PCT

    text = get_labels(lang)["plots"]
    d = by_instance.sort_values(["family", "instance"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, 4.2), layout="constrained")
    x = np.arange(len(d))
    ax.axhspan(-NOISE_PCT, NOISE_PCT, color="0.88", zorder=0, label=text["version_noise"].format(pct=NOISE_PCT))
    ax.axhline(0.0, color="0.4", lw=0.8)
    for fam in FAMILIES:
        part = d[(d["family"] == fam) & d["fleet_equal_versions"].astype(bool)]
        ax.scatter(part.index, part["distance_diff_pct"], color=FAMILY_COLORS[fam], s=28, label=fam, zorder=3)
    changed = d[d["delta_vehicles_v2_minus_v1"] != 0]
    low = float(d["distance_diff_pct"].min()) - 1.0
    ax.scatter(changed.index, [low] * len(changed), marker="v", color="black", s=60, zorder=4,
               label=text["version_fleet_changed"])
    ax.set_xticks(x, d["instance"], rotation=90, fontsize=7)
    ax.set_ylabel(text["version_y"])
    ax.set_title(text["version_title"])
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    return fig
