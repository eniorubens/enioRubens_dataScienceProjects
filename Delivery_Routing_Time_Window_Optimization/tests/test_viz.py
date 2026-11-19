import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from delivery_routing_vrptw import plot_routes, plot_time_windows


def test_plot_routes(inst):
    ax = plot_routes(inst, [[1, 2], [3, 5], [], [4]], title="t")
    assert isinstance(ax, Axes) and ax.get_title() == "t"
    _, own = plt.subplots()
    assert plot_routes(inst, [[1]], ax=own) is own
    plt.close("all")


def test_plot_time_windows(inst):
    ax = plot_time_windows(inst)
    assert isinstance(ax, Axes)
    assert len(ax.patches) == 5
    _, own = plt.subplots()
    assert plot_time_windows(inst, ax=own) is own
    plt.close("all")


def test_plot_time_windows_localized(inst):
    ax = plot_time_windows(inst, lang="pt")
    assert ax.get_xlabel() == "Tempo" and "R101" in ax.get_title()
    assert plot_time_windows(inst, title="x", lang="en").get_title() == "x"
    plt.close("all")


def test_setup_notebook_figures_outside_ipython():
    from delivery_routing_vrptw import setup_notebook_figures

    assert setup_notebook_figures() is False
    assert plt.rcParams["figure.dpi"] == 100
