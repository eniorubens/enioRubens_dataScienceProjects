"""Shared plotting utilities and global graph parameters.

Port of notebook cell [3] (exec_count=3, code_index=1).
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


def set_graph_parameters() -> None:
    """Configure matplotlib/seaborn visual defaults for the project."""
    plt.rcParams["axes.grid"] = True
    plt.rcParams["patch.force_edgecolor"] = True

    sns.set_theme(style="ticks", context="notebook")
    mpl.rcParams["figure.figsize"] = (16, 6)

    # Title
    mpl.rcParams["figure.titlesize"] = 16
    mpl.rcParams["figure.titleweight"] = "bold"
    mpl.rcParams["axes.titlesize"] = 16
    mpl.rcParams["axes.titleweight"] = "bold"
    mpl.rcParams["axes.titlepad"] = 16

    # Axes labels
    mpl.rcParams["axes.labelsize"] = 16
    mpl.rcParams["axes.labelweight"] = "bold"

    # Grid and ticks
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.left"] = False
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.grid"] = True
    mpl.rcParams["axes.grid.axis"] = "y"
    mpl.rcParams["ytick.left"] = False
    mpl.rcParams["axes.formatter.useoffset"] = True

    # Legend
    mpl.rcParams["legend.facecolor"] = "w"
    mpl.rcParams["legend.title_fontsize"] = 14
    mpl.rcParams["legend.fontsize"] = 12
    mpl.rcParams["legend.frameon"] = True
    mpl.rcParams["legend.framealpha"] = 1
    mpl.rcParams["legend.fancybox"] = True
    mpl.rcParams["legend.facecolor"] = "white"
    mpl.rcParams["legend.edgecolor"] = "black"
    mpl.rcParams["legend.borderpad"] = 0.8

    # Other
    mpl.rcParams["lines.linewidth"] = 3
    mpl.rcParams["lines.markersize"] = 10
