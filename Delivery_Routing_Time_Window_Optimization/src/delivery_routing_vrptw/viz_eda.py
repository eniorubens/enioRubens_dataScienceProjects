"""Plots for the exploratory analysis of instances (labels come from :mod:`labels`)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .io import FAMILIES, Instance
from .labels import get_labels
from .viz import plot_time_windows

# Colour-blind-safe palette (Okabe-Ito), one colour per family.
FAMILY_COLORS = {
    "C1": "#0072B2", "C2": "#56B4E9", "R1": "#D55E00", "R2": "#E69F00", "RC1": "#009E73", "RC2": "#CC79A7",
}


def _families_present(df: pd.DataFrame) -> list[str]:
    return [f for f in FAMILIES if f in set(df["family"])]


def plot_family_layouts(
    instances_by_name: Mapping[str, Instance], names: Sequence[str], lang: str = "en"
) -> Figure:
    """Draw one map per instance: customers coloured by window start as a share of the horizon.

    Parameters
    ----------
    instances_by_name : mapping
        Instances keyed by name.
    names : sequence of str
        Instances to draw, in order (three panels per row).
    lang : str
        Label language.

    Returns
    -------
    matplotlib.figure.Figure
        The figure. The colour is ``ready / horizon`` with the common limits 0 and 1 in every panel, so a single
        colour bar is valid for all of them (the horizons differ across families); the depot is a red square.
    """
    text = get_labels(lang)["plots"]
    ncols = min(3, len(names))
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.4 * nrows), squeeze=False)
    scatter = None
    for ax, name in zip(axes.ravel(), names, strict=False):
        inst = instances_by_name[name]
        scatter = ax.scatter(inst.x[1:], inst.y[1:], c=inst.ready[1:] / inst.horizon, cmap="viridis", s=22,
                             vmin=0.0, vmax=1.0)
        ax.scatter([inst.x[0]], [inst.y[0]], marker="s", s=80, color="crimson", zorder=3,
                   label=text["depot"])
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(text["layout_title"].format(name=name, family=inst.family))
        ax.set_xlabel(text["x"])
        ax.set_ylabel(text["y"])
    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)
    axes.ravel()[0].legend(loc="upper left", fontsize=8)
    if scatter is not None:
        fig.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.8, label=text["layout_colorbar"])
    return fig


def plot_window_comparison(
    instances_by_name: Mapping[str, Instance], names: Sequence[str], lang: str = "en"
) -> Figure:
    """Draw :func:`delivery_routing_vrptw.viz.plot_time_windows` side by side for several instances."""
    fig, axes = plt.subplots(1, len(names), figsize=(5.5 * len(names), 5), squeeze=False)
    for ax, name in zip(axes.ravel(), names, strict=True):
        plot_time_windows(instances_by_name[name], ax=ax, title=name, lang=lang)
    fig.tight_layout()
    return fig


def plot_window_widths(window_df: pd.DataFrame, lang: str = "en", ax: Axes | None = None) -> Axes:
    """Box plot of the mean window width over the horizon, one box per family."""
    text = get_labels(lang)["plots"]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    fams = _families_present(window_df)
    data = [window_df.loc[window_df["family"] == f, "width_share_horizon"].to_numpy() for f in fams]
    box = ax.boxplot(data, tick_labels=fams, patch_artist=True)
    for patch, fam in zip(box["boxes"], fams, strict=True):
        patch.set_facecolor(FAMILY_COLORS[fam])
        patch.set_alpha(0.7)
    ax.set_title(text["widths_title"])
    ax.set_xlabel(text["widths_x"])
    ax.set_ylabel(text["widths_y"])
    ax.set_ylim(bottom=0)
    return ax


def plot_fleet_bounds(fleet_df: pd.DataFrame, lang: str = "en", ax: Axes | None = None) -> Axes:
    """Scatter of the capacity lower bound against the SINTEF vehicles, with the identity line."""
    text = get_labels(lang)["plots"]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    top = max(fleet_df["capacity_lb"].max(), fleet_df["ref_vehicles"].max()) + 1
    ax.plot([0, top], [0, top], "--", color="0.5", lw=1, label=text["fleet_diag"])
    for k, fam in enumerate(_families_present(fleet_df)):
        sub = fleet_df[fleet_df["family"] == fam]
        offset = (k - 2.5) * 0.07  # small deterministic shift so overlapping families stay visible
        ax.scatter(sub["capacity_lb"] + offset, sub["ref_vehicles"], s=34, alpha=0.8,
                   color=FAMILY_COLORS[fam], label=fam)
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_title(text["fleet_title"])
    ax.set_xlabel(text["fleet_x"])
    ax.set_ylabel(text["fleet_y"])
    ax.legend(fontsize=8)
    return ax


def plot_compatibility(density_df: pd.DataFrame, lang: str = "en", ax: Axes | None = None) -> Axes:
    """Per-instance compatibility density by family (points) with the family median (bar)."""
    text = get_labels(lang)["plots"]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    fams = _families_present(density_df)
    for k, fam in enumerate(fams):
        vals = density_df.loc[density_df["family"] == fam, "density"].to_numpy()
        jitter = np.linspace(-0.2, 0.2, len(vals)) if len(vals) > 1 else np.zeros(1)
        ax.scatter(k + jitter, vals, s=22, alpha=0.8, color=FAMILY_COLORS[fam])
        ax.hlines(np.median(vals), k - 0.3, k + 0.3, color="black", lw=1.6)
    ax.set_xticks(range(len(fams)), fams)
    ax.set_title(text["compat_title"])
    ax.set_xlabel(text["compat_x"])
    ax.set_ylabel(text["compat_y"])
    ax.set_ylim(0, 1)
    return ax


def plot_convention_tradeoff(conv_df: pd.DataFrame, lang: str = "en", ax: Axes | None = None) -> Axes:
    """Scatter of the vehicle difference SINTEF minus CVRPLIB (x) against the extra SINTEF distance in percent (y)."""
    text = get_labels(lang)["plots"]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.8))
    for k, fam in enumerate(_families_present(conv_df)):
        sub = conv_df[conv_df["family"] == fam]
        offset = (k - 2.5) * 0.05
        ax.scatter(-sub["delta_vehicles"] + offset, sub["distance_diff_pct"], s=34, alpha=0.8,
                   color=FAMILY_COLORS[fam], label=fam)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_title(text["conv_title"])
    ax.set_xlabel(text["conv_x"])
    ax.set_ylabel(text["conv_y"])
    ax.legend(fontsize=8)
    return ax
