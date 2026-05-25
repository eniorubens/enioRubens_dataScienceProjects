"""
utils.py — Shared plotting helpers and logging utilities for the CLTV project.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import pandas as pd


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger configured with a standard project format.

    Retorna um logger configurado com o formato padrão do projeto.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def format_currency_axis(
    ax: plt.Axes,
    axis: str = "y",
    currency: str = "GBP",
) -> None:
    """
    Format a Matplotlib axis as compact currency values (e.g., GBP 1,234).

    Formata um eixo do Matplotlib como valores monetários compactos (ex: GBP 1.234).

    Parameters
    ----------
    ax : plt.Axes
    axis : 'x' or 'y'
    currency : ISO currency code prefix
    """
    formatter = plt.FuncFormatter(lambda value, _: f"{currency} {value:,.0f}")
    if axis == "y":
        ax.yaxis.set_major_formatter(formatter)
    else:
        ax.xaxis.set_major_formatter(formatter)


def add_chart_labels(
    ax: plt.Axes,
    title: str,
    subtitle: str,
    xlabel: str,
    ylabel: str,
) -> None:
    """
    Apply consistent title + subtitle labels to a Matplotlib Axes.

    Aplica título e subtítulo consistentes a um objeto Axes do Matplotlib.
    """
    ax.set_title(f"{title}\n{subtitle}", loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def log_dataframe_summary(
    df: pd.DataFrame,
    name: str,
    logger: logging.Logger,
) -> None:
    """
    Log the shape, columns, and basic descriptive statistics of a DataFrame.

    Registra shape, colunas e estatísticas básicas de um DataFrame no logger.
    """
    logger.info("%s — shape: %s", name, df.shape)
    logger.info("%s — columns: %s", name, list(df.columns))
    try:
        stats = df.describe(include="all").loc[["count", "mean", "std"]].to_string()
        logger.debug("%s — stats:\n%s", name, stats)
    except Exception:
        pass
