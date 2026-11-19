"""Matplotlib helpers for routes and time windows."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from .io import Instance
from .labels import get_labels


def plot_routes(
    instance: Instance, routes: list[list[int]], ax: Axes | None = None, title: str | None = None
) -> Axes:
    """Plot the depot, the customers and one polyline per route.

    Parameters
    ----------
    instance : Instance
        The instance.
    routes : list of list of int
        Customer ids per route, without the depot.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    title : str, optional
        Axes title.

    Returns
    -------
    matplotlib.axes.Axes
        The axes used.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    colors = plt.get_cmap("tab20").colors
    for k, route in enumerate(r for r in routes if r):
        path = [0, *route, 0]
        ax.plot(instance.x[path], instance.y[path], "-", lw=1.2, color=colors[k % len(colors)], zorder=1)
    ax.scatter(instance.x[1:], instance.y[1:], s=18, color="0.25", zorder=2, label="customers")
    ax.scatter([instance.x[0]], [instance.y[0]], marker="s", s=90, color="crimson", zorder=3, label="depot")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title is not None:
        ax.set_title(title)
    return ax


def plot_time_windows(
    instance: Instance, ax: Axes | None = None, title: str | None = None, lang: str | None = None
) -> Axes:
    """Plot customer time windows as horizontal bars sorted by ready time.

    Parameters
    ----------
    instance : Instance
        The instance.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    title : str, optional
        Axes title; when ``lang`` is given and ``title`` is omitted, a translated title is used.
    lang : str, optional
        Label language (``"pt"`` or ``"en"``) for the axis labels; plain English if omitted.

    Returns
    -------
    matplotlib.axes.Axes
        The axes used.
    """
    plots = get_labels(lang or "en")["plots"]
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    order = np.lexsort((np.arange(1, len(instance.x)), instance.ready[1:]))
    ready = instance.ready[1:][order]
    width = instance.due[1:][order] - ready
    ax.barh(np.arange(len(order)), width, left=ready, height=0.8, color="steelblue")
    ax.set_xlabel(plots["time_x"] if lang else "time")
    ax.set_ylabel(plots["time_y"] if lang else "customers (sorted by ready time)")
    ax.set_xlim(0, instance.horizon)
    if title is not None:
        ax.set_title(title)
    elif lang:
        ax.set_title(plots["time_title"].format(name=instance.name))
    return ax


def setup_notebook_figures(dpi: int = 100) -> bool:
    """Make inline notebook figures sharp (retina, 2x pixels) and consistent.

    Sets the inline backend format to ``"retina"`` and a default figure ``dpi`` / tight bounding box. Outside
    IPython (or without ``matplotlib-inline``) the format call is skipped, so the function never raises.

    Parameters
    ----------
    dpi : int
        Default figure dpi (retina doubles the saved pixels).

    Returns
    -------
    bool
        ``True`` if the retina format was applied, ``False`` when running outside IPython.
    """
    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["savefig.bbox"] = "tight"
    try:
        from matplotlib_inline.backend_inline import set_matplotlib_formats
    except ImportError:
        return False
    try:
        from IPython import get_ipython

        if get_ipython() is None:
            return False
        set_matplotlib_formats("retina")
    except Exception:  # noqa: BLE001 - never break a notebook over a display preference
        return False
    return True
