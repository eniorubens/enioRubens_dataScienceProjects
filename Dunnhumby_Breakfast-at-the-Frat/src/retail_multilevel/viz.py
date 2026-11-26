"""Plotting helpers for the notebooks.

Every visible string (titles, axis labels, legend entries, category labels) comes from
:mod:`retail_multilevel.i18n`, so EN and PT notebooks share identical code. Each plot function
returns the :class:`matplotlib.figure.Figure` and never calls ``plt.show()``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import PercentFormatter

from retail_multilevel.i18n import (
    MODEL_ORDER,
    STRATUM_ORDER,
    contrast_label,
    level_label,
    model_label,
    scenario_label,
    stratum_label,
    t,
)

PALETTE: dict[str, str] = {
    "train": "#1f77b4",
    "test": "#d95f02",
    "neutral": "#4c78a8",
    "residual": "#b0b0b0",
    "line": "#444444",
    "grid": "#dddddd",
}
GROUP_COLORS: list[str] = ["#1f77b4", "#d95f02", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
PROMO_COLORS: dict[str, str] = {
    "FEATURE": "#1f77b4",
    "DISPLAY": "#d95f02",
    "TPR_ONLY": "#2ca02c",
    "ANY_PROMO": "#9467bd",
    "BELOW_BASE": "#8c564b",
}
PROMO_ORDER = ["FEATURE", "DISPLAY", "TPR_ONLY", "ANY_PROMO", "BELOW_BASE"]

_GROUP_KEYS: dict[str, str] = {
    "STORE_NUM": "group_store",
    "UPC": "group_product",
    "PAIR_ID": "group_pair",
    "STORE_NUM:UPC": "group_pair",
    "UPC:STORE_NUM": "group_pair",
    "Residual": "group_residual",
}
_PROMO_KEYS: dict[str, str] = {code: f"promo_{code.lower()}" for code in PROMO_ORDER}

_TITLE_SIZE = 13
_LABEL_SIZE = 11
_TICK_SIZE = 10


def group_label(group: str, lang: str) -> str:
    """Translate a grouping name (any known spelling); unknown names are returned unchanged."""
    key = _GROUP_KEYS.get(str(group))
    return t(key, lang) if key else group


def promo_label(code: str, lang: str) -> str:
    """Translate a promotion code (FEATURE, DISPLAY, ...); unknown codes are returned unchanged."""
    key = _PROMO_KEYS.get(str(code))
    return t(key, lang) if key else code


EXPORT_DPI = 300  # retina-equivalent density for exported PNGs (2x the 150 dpi screen baseline)


def use_retina() -> bool:
    """Render inline notebook figures as retina (2x density) PNGs.

    Call once in a notebook's setup cell. Outside an IPython kernel (scripts, tests) it does
    nothing and returns ``False``.

    Returns
    -------
    bool
        ``True`` when the inline backend was switched to the retina format.
    """
    try:
        from IPython import get_ipython
        from matplotlib_inline.backend_inline import set_matplotlib_formats
    except ImportError:
        return False
    if get_ipython() is None:
        return False
    set_matplotlib_formats("retina")
    return True


def save_figure(fig: Figure, path: str | Path, dpi: int = EXPORT_DPI) -> Path:
    """Save ``fig`` as a PNG at ``dpi`` (retina-equivalent by default), creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def _prepare(ax: Axes | None, figsize: tuple[float, float]) -> tuple[Figure, Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        return fig, ax
    return ax.figure, ax


def _style(ax: Axes, grid_axis: str = "y") -> None:
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=PALETTE["grid"], linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=_TICK_SIZE)


def _set_texts(ax: Axes, title: str, xlabel: str | None = None, ylabel: str | None = None) -> None:
    ax.set_title(title, fontsize=_TITLE_SIZE)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=_LABEL_SIZE)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=_LABEL_SIZE)


def _flat_label(label: object) -> str:
    if isinstance(label, tuple):
        return " / ".join(str(x) for x in label)
    return str(label)


def plot_panel_structure(lang: str) -> Figure:
    """Show observation weeks nested within a pair and the crossed store and UPC groups."""
    fig, ax = plt.subplots(figsize=(10, 4.7))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set(xlim=(0, 1000), ylim=(0, 470))
    ax.set_axis_off()

    store = t("panel_structure_store", lang)
    pair = t("panel_structure_pair_node", lang)
    week = t("panel_structure_week_node", lang)
    groups = [store.format(code="B"), "UPC A", store.format(code="A"), "UPC B"]
    pairs = [pair.format(store=s, upc=p) for s, p in [("B", "A"), ("A", "A"), ("A", "B")]]
    weeks = [week.format(time=time) for time in ["t", "t + 1"]]
    group_centers = [346, 518, 690, 862]
    pair_centers = [432, 604, 776]
    week_centers = [518, 690]
    band_specs = [
        (350, "panel_structure_crossed", "#f4f6f8"),
        (220, "panel_structure_pair", "#eef4fa"),
        (90, "panel_structure_week", "#f4f6f8"),
    ]
    for y, label, color in band_specs:
        ax.add_patch(Rectangle((24, y - 38), 952, 76, facecolor=color, edgecolor="none"))
        ax.text(44, y, t(label, lang), va="center", fontsize=10.5, color="#304557", linespacing=1.4)

    for left, center, right in zip(group_centers[:-1], pair_centers, group_centers[1:], strict=True):
        ax.plot([left + 20, left + 20, center - 30, center - 30], [326, 298, 298, 244], color="#637383", lw=1.15)
        ax.plot([right - 20, right - 20, center + 30, center + 30], [326, 298, 298, 244], color="#637383", lw=1.15)
    for start, end in zip([574, 634], week_centers, strict=True):
        ax.plot([start, start, end, end], [196, 166, 166, 114], color=PALETTE["neutral"], lw=1.5)

    nodes = [
        *zip(group_centers, [350] * 4, groups, strict=True),
        *zip(pair_centers, [220] * 3, pairs, strict=True),
        *zip(week_centers, [90] * 2, weeks, strict=True),
    ]
    for x, y, label in nodes:
        selected = (x, y) == (604, 220)
        ax.add_patch(
            FancyBboxPatch(
                (x - 76, y - 24),
                152,
                48,
                boxstyle="round,pad=0,rounding_size=6",
                facecolor="#e0ecf7" if selected else "white",
                edgecolor="#27567d" if selected else "#738294",
                linewidth=1.65 if selected else 1,
            )
        )
        ax.text(x, y, label, ha="center", va="center", fontsize=10.3, color="#17314a", weight="bold")
    ax.text(24, 437, t("panel_structure_title", lang), weight="bold", fontsize=16, color="#17314a", va="top")
    ax.text(24, 23, t("panel_structure_note", lang), fontsize=9, color="#43566a", va="center")
    return fig


def plot_weekly_units(df: pd.DataFrame, lang: str, ax: Axes | None = None) -> Figure:
    """Line chart of total weekly units, train and test in different colors.

    Parameters
    ----------
    df : pandas.DataFrame
        Columns ``WEEK_END_DATE``, ``WEEK``, ``UNITS``, ``IS_TEST``.
    lang : str
        ``'en'`` or ``'pt'``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created when ``None``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _prepare(ax, (9, 4))
    data = df.sort_values("WEEK_END_DATE").reset_index(drop=True)
    is_test = data["IS_TEST"].to_numpy(dtype=bool)
    train = data[~is_test]
    test = data[is_test]
    ax.plot(
        train["WEEK_END_DATE"], train["UNITS"], color=PALETTE["train"], linewidth=1.8, label=t("legend_train", lang)
    )
    if len(test):
        first = int(np.argmax(is_test))
        # Include the last train point so the two segments touch.
        mask = is_test.copy()
        mask[max(first - 1, 0)] = True
        test_line = data[mask]
        ax.plot(
            test_line["WEEK_END_DATE"],
            test_line["UNITS"],
            color=PALETTE["test"],
            linewidth=1.8,
            label=t("legend_test", lang),
        )
        ax.axvline(test["WEEK_END_DATE"].iloc[0], color=PALETTE["line"], linestyle="--", linewidth=1)
    _style(ax)
    _set_texts(ax, t("weekly_units_title", lang), t("weekly_units_xlabel", lang), t("weekly_units_ylabel", lang))
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=_TICK_SIZE)
    return fig


def plot_promo_rates(df: pd.DataFrame, lang: str, ax: Axes | None = None) -> Figure:
    """Draw grouped horizontal bars with the share of weeks under each promotion flag.

    Parameters
    ----------
    df : pandas.DataFrame
        Columns ``FEATURE``, ``DISPLAY``, ``TPR_ONLY``, ``ANY_PROMO``, ``BELOW_BASE``, ``n``;
        the index holds the groups (a ``MultiIndex`` is flattened with ``" / "``).
    lang : str
        ``'en'`` or ``'pt'``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_groups = len(df)
    fig, ax = _prepare(ax, (9, 1.8 + 0.9 * n_groups + 0.6 * len(PROMO_ORDER) * n_groups / 3))
    labels = [t("all_rows", lang) if _flat_label(i) == "all" else _flat_label(i) for i in df.index]
    height = 0.8 / len(PROMO_ORDER)
    positions = np.arange(n_groups)
    for k, code in enumerate(PROMO_ORDER):
        offset = (k - (len(PROMO_ORDER) - 1) / 2) * height
        ax.barh(
            positions + offset,
            df[code].to_numpy(dtype=float),
            height=height,
            color=PROMO_COLORS[code],
            label=promo_label(code, lang),
        )
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    _style(ax, "x")
    _set_texts(ax, t("promo_rates_title", lang), t("promo_rates_xlabel", lang))
    ax.legend(frameon=False, fontsize=_TICK_SIZE, loc="best")
    return fig


def plot_variance_shares(
    df: pd.DataFrame,
    lang: str,
    ax: Axes | None = None,
    title_key: str = "variance_share_title",
    share_col: str = "share",
) -> Figure:
    """Horizontal bars of variance shares per group, labeled with percentages.

    Parameters
    ----------
    df : pandas.DataFrame
        Columns ``group`` and the share column (0-1).
    lang : str
        ``'en'`` or ``'pt'``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    title_key : str, default "variance_share_title"
        i18n key of the title.
    share_col : str, default "share"
        Name of the column holding the shares.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _prepare(ax, (8, 1.6 + 0.6 * len(df)))
    labels = [group_label(g, lang) for g in df["group"]]
    shares = df[share_col].to_numpy(dtype=float)
    colors = [PALETTE["residual"] if g == "Residual" else PALETTE["neutral"] for g in df["group"]]
    bars = ax.barh(labels, shares, color=colors)
    ax.bar_label(bars, labels=[f"{s:.1%}" for s in shares], padding=3, fontsize=_TICK_SIZE)
    ax.invert_yaxis()
    ax.set_xlim(0, max(1.0, float(shares.max(initial=0)) * 1.1) if len(shares) else 1.0)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    _style(ax, "x")
    _set_texts(ax, t(title_key, lang), t("variance_share_xlabel", lang))
    return fig


def plot_icc_by_model(df: pd.DataFrame, lang: str, ax: Axes | None = None) -> Figure:
    """Stacked horizontal bars of variance shares, one bar per model.

    Parameters
    ----------
    df : pandas.DataFrame
        Columns ``model``, ``group``, ``share``.
    lang : str
        ``'en'`` or ``'pt'``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.figure.Figure
    """
    models = list(dict.fromkeys(df["model"]))
    groups = list(dict.fromkeys(df["group"]))
    fig, ax = _prepare(ax, (9, 1.8 + 0.7 * len(models)))
    palette = iter(GROUP_COLORS * 3)
    colors = {g: PALETTE["residual"] if g == "Residual" else next(palette) for g in groups}
    pivot = df.pivot_table(index="model", columns="group", values="share", aggfunc="sum").reindex(models)
    left = np.zeros(len(models))
    for g in groups:
        values = pivot[g].fillna(0).to_numpy(dtype=float)
        ax.barh(models, values, left=left, color=colors[g], label=group_label(g, lang), height=0.6)
        left += values
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    _style(ax, "x")
    _set_texts(ax, t("icc_by_model_title", lang), t("icc_by_model_xlabel", lang))
    ax.legend(frameon=False, fontsize=_TICK_SIZE, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=len(groups))
    return fig


def plot_pair_coverage(df: pd.DataFrame, lang: str, ax: Axes | None = None) -> Figure:
    """Histogram of training weeks per store x product pair with a median line.

    Parameters
    ----------
    df : pandas.DataFrame
        Columns ``PAIR_ID``, ``STORE_NUM``, ``UPC``, ``n_weeks_train``, ``n_weeks_test``.
    lang : str
        ``'en'`` or ``'pt'``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _prepare(ax, (8, 4))
    values = df["n_weeks_train"].to_numpy(dtype=float)
    ax.hist(values, bins=30, color=PALETTE["train"], edgecolor="white")
    median = float(np.median(values)) if len(values) else float("nan")
    ax.axvline(median, color=PALETTE["test"], linestyle="--", linewidth=1.5)
    ax.annotate(
        t("pair_coverage_median", lang).format(value=median),
        xy=(median, 1.0),
        xycoords=ax.get_xaxis_transform(),
        xytext=(5, -5),
        textcoords="offset points",
        va="top",
        fontsize=_TICK_SIZE,
        color=PALETTE["test"],
    )
    _style(ax)
    _set_texts(ax, t("pair_coverage_title", lang), t("pair_coverage_xlabel", lang), t("pair_coverage_ylabel", lang))
    return fig


def plot_raw_promo_lift(df: pd.DataFrame, lang: str, ax: Axes | None = None) -> Figure:
    """Horizontal bars of the raw (descriptive) promotion lift, shown as a percentage.

    Parameters
    ----------
    df : pandas.DataFrame
        Index named ``promo`` (FEATURE, DISPLAY, TPR_ONLY); columns ``lift_log``, ``lift_pct`` (fraction, ``exp(lift_log) - 1``), ``n_weeks``.
    lang : str
        ``'en'`` or ``'pt'``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _prepare(ax, (8, 1.8 + 0.7 * len(df)))
    labels = [promo_label(p, lang) for p in df.index]
    values = df["lift_pct"].to_numpy(dtype=float)
    colors = [PROMO_COLORS.get(str(p), PALETTE["neutral"]) for p in df.index]
    bars = ax.barh(labels, values, color=colors)
    ax.bar_label(bars, labels=[f"{v:.0%}" for v in values], padding=3, fontsize=_TICK_SIZE)
    ax.invert_yaxis()
    ax.axvline(0, color=PALETTE["line"], linewidth=0.8)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    _style(ax, "x")
    _set_texts(ax, t("raw_lift_title", lang), t("raw_lift_xlabel", lang))
    ax.annotate(
        t("raw_lift_caption", lang),
        xy=(0, -0.22),
        xycoords="axes fraction",
        fontsize=_TICK_SIZE - 1,
        style="italic",
        color=PALETTE["line"],
    )
    return fig


def plot_caterpillar(
    df: pd.DataFrame,
    lang: str,
    group: str,
    ax: Axes | None = None,
    z: float = 1.96,
    max_levels: int | None = None,
    labels: dict[str, str] | None = None,
) -> Figure:
    """Caterpillar plot of random-effect predictions (BLUPs) with ``z`` x conditional-sd intervals.

    Parameters
    ----------
    df : pandas.DataFrame
        Columns ``level``, ``term``, ``blup`` and ``cond_sd``.
    lang : str
        ``'en'`` or ``'pt'``.
    group : str
        Grouping name (``STORE_NUM``, ``UPC``, ...), translated in the title.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    z : float, default 1.96
        Half-width of the interval in conditional standard deviations (1.96 gives 95%).
    max_levels : int, optional
        Keep only this many levels, the ones with the most extreme BLUPs.
    labels : dict, optional
        Maps ``str(level)`` to a display name (e.g. product description).

    Returns
    -------
    matplotlib.figure.Figure
    """
    data = df.sort_values("blup").reset_index(drop=True)
    if max_levels is not None and len(data) > max_levels:
        keep = data["blup"].abs().nlargest(max_levels).index
        data = data.loc[sorted(keep)].reset_index(drop=True)
    n = len(data)
    fig, ax = _prepare(ax, (8, max(3.5, min(0.2 * n + 1.5, 11))))
    pos = np.arange(n)
    ax.errorbar(
        data["blup"],
        pos,
        xerr=z * data["cond_sd"],
        fmt="o",
        markersize=3,
        color=PALETTE["neutral"],
        ecolor=PALETTE["residual"],
        elinewidth=1,
    )
    ax.axvline(0, color=PALETTE["line"], linestyle="--", linewidth=1)
    if n <= 60:
        names = [(labels or {}).get(str(lv), str(lv)) for lv in data["level"]]
        ax.set_yticks(pos, names, fontsize=_TICK_SIZE - 3 if n > 30 else _TICK_SIZE)
    else:
        ax.set_yticks([])
    _style(ax, "x")
    _set_texts(ax, t("caterpillar_title", lang).format(group=group_label(group, lang)), t("caterpillar_xlabel", lang))
    return fig


_TERM_KEYS: dict[str, str] = {
    "(Intercept)": "term_intercept",
    "LOG_DISC": "term_log_disc",
    "LOG_BASE_DEV": "term_log_base_dev",
    "LOG_BASE_MEAN": "term_log_base_mean",
    "TREND": "term_trend",
}
_FOURIER_PREFIXES = ("SIN_", "COS_")
COMPARISON_TERMS = ["LOG_DISC", "LOG_BASE_DEV", "FEATURE", "DISPLAY"]
_METHOD_KEYS: dict[str, str] = {
    "ols_pooled": "method_ols_pooled",
    "ols_upc_fe": "method_ols_upc_fe",
    "multilevel": "method_multilevel",
}
_METHOD_COLORS: list[str] = ["#8c8c8c", "#d95f02", "#1f77b4"]


def term_label(term: str, lang: str) -> str:
    """Translate a model term (``LOG_DISC``, ``FEATURE``, ``SIN_2``, ...); unknown terms are returned unchanged."""
    term = str(term)
    if term in _TERM_KEYS:
        return t(_TERM_KEYS[term], lang)
    if term in _PROMO_KEYS:
        return promo_label(term, lang)
    if term.startswith(_FOURIER_PREFIXES):
        key = "term_sin" if term.startswith("SIN_") else "term_cos"
        return f"{t(key, lang)} {term.split('_', 1)[1]}"
    return term


def _is_fourier(term: str) -> bool:
    return str(term).startswith(_FOURIER_PREFIXES)


def plot_coefficients(fixef: pd.DataFrame, lang: str, terms: list[str] | None = None, ax: Axes | None = None) -> Figure:
    """Forest plot of fixed effects with 95% intervals.

    Parameters
    ----------
    fixef : pandas.DataFrame
        Columns ``term, estimate, ci_low, ci_high``.
    lang : str
        ``'en'`` or ``'pt'``.
    terms : list of str, optional
        Terms to draw, in order. By default the intercept and the Fourier terms are hidden.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.figure.Figure
    """
    data = fixef.set_index("term")
    if terms is None:
        terms = [x for x in data.index if x != "(Intercept)" and not _is_fourier(x)]
    data = data.loc[terms]
    n = len(data)
    fig, ax = _prepare(ax, (9, 1.8 + 0.6 * n))
    pos = np.arange(n)
    est = data["estimate"].to_numpy(dtype=float)
    lo = data["ci_low"].to_numpy(dtype=float)
    hi = data["ci_high"].to_numpy(dtype=float)
    ax.errorbar(
        est, pos, xerr=[est - lo, hi - est], fmt="o", color=PALETTE["neutral"], ecolor=PALETTE["neutral"],
        elinewidth=2, capsize=3,
    )  # fmt: skip
    for p, e, h in zip(pos, est, hi, strict=True):
        ax.annotate(f"{e:+.3f}", xy=(h, p), xytext=(6, 0), textcoords="offset points", va="center", fontsize=_TICK_SIZE)
    ax.axvline(0, color=PALETTE["line"], linestyle="--", linewidth=1)
    ax.set_yticks(pos, [term_label(x, lang) for x in data.index])
    ax.invert_yaxis()
    ax.margins(x=0.12)
    _style(ax, "x")
    _set_texts(ax, t("coef_title", lang), t("coef_xlabel", lang))
    return fig


def plot_pseudo_r2(df: pd.DataFrame, lang: str, ax: Axes | None = None) -> Figure:
    """Horizontal bars of the variance reduction per level (negative values allowed).

    Parameters
    ----------
    df : pandas.DataFrame
        Columns ``group`` and ``reduction`` (fraction; see ``fixedeffects.pseudo_r2``).
    lang : str
        ``'en'`` or ``'pt'``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _prepare(ax, (8, 1.6 + 0.6 * len(df)))
    values = df["reduction"].to_numpy(dtype=float)
    colors = [PALETTE["neutral"] if v >= 0 else PALETTE["residual"] for v in values]
    bars = ax.barh([group_label(g, lang) for g in df["group"]], values, color=colors)
    ax.bar_label(bars, labels=[f"{v:.1%}" for v in values], padding=3, fontsize=_TICK_SIZE)
    ax.invert_yaxis()
    ax.axvline(0, color=PALETTE["line"], linewidth=0.8)
    ax.margins(x=0.15)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    _style(ax, "x")
    _set_texts(ax, t("pseudo_r2_title", lang), t("pseudo_r2_xlabel", lang))
    return fig


def plot_method_comparison(
    df: pd.DataFrame, lang: str, terms: list[str] | None = None, ax: Axes | None = None, z: float = 1.96
) -> Figure:
    """Compare coefficients across estimators (point +- ``z`` standard errors), one row per term.

    Parameters
    ----------
    df : pandas.DataFrame
        Columns ``method, term, estimate, std_error``.
    lang : str
        ``'en'`` or ``'pt'``.
    terms : list of str, optional
        Terms to draw; defaults to the discount, regular-price, feature and display terms present.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    z : float, default 1.96
        Half-width of the interval in standard errors.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if terms is None:
        terms = [x for x in COMPARISON_TERMS if x in set(df["term"])]
    methods = list(dict.fromkeys(df["method"]))
    fig, ax = _prepare(ax, (9, 2.2 + 0.9 * len(terms)))
    height = 0.8 / len(methods)
    pos = np.arange(len(terms))
    for k, method in enumerate(methods):
        sub = df[df["method"] == method].set_index("term").loc[terms]
        offset = (k - (len(methods) - 1) / 2) * height
        label = t(_METHOD_KEYS[method], lang) if method in _METHOD_KEYS else str(method)
        ax.errorbar(
            sub["estimate"].to_numpy(dtype=float), pos + offset, xerr=z * sub["std_error"].to_numpy(dtype=float),
            fmt="o", color=_METHOD_COLORS[k % len(_METHOD_COLORS)], elinewidth=2, capsize=2, label=label,
        )  # fmt: skip
    ax.axvline(0, color=PALETTE["line"], linestyle="--", linewidth=1)
    ax.set_yticks(pos, [term_label(x, lang) for x in terms])
    ax.invert_yaxis()
    _style(ax, "x")
    _set_texts(ax, t("method_comparison_title", lang), t("method_comparison_xlabel", lang))
    ax.legend(frameon=False, fontsize=_TICK_SIZE, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncols=len(methods))
    return fig


def plot_seasonality(df: pd.DataFrame, lang: str, ax: Axes | None = None) -> Figure:
    """Line chart of the seasonal profile (effect on units, in percent, by week of the year).

    Parameters
    ----------
    df : pandas.DataFrame
        Columns ``week_of_year`` and ``effect_pct`` (see ``interpret.seasonal_profile``).
    lang : str
        ``'en'`` or ``'pt'``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _prepare(ax, (9, 4))
    ax.plot(df["week_of_year"], df["effect_pct"], color=PALETTE["train"], linewidth=2)
    ax.axhline(0, color=PALETTE["line"], linestyle="--", linewidth=1)
    ax.yaxis.set_major_formatter(PercentFormatter(100.0, decimals=0))
    _style(ax)
    _set_texts(ax, t("seasonality_title", lang), t("seasonality_xlabel", lang), t("seasonality_ylabel", lang))
    return fig


def plot_residual_lag_corr(df: pd.DataFrame, lang: str, ax: Axes | None = None) -> Figure:
    """Bar chart of the within-pair residual correlation by lag.

    Parameters
    ----------
    df : pandas.DataFrame
        Columns ``lag`` and ``corr`` (see ``diagnostics.lag_correlations``).
    lang : str
        ``'en'`` or ``'pt'``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _prepare(ax, (6, 4))
    ax.bar(df["lag"].astype(int), df["corr"], color=PALETTE["neutral"], width=0.6)
    ax.axhline(0, color=PALETTE["line"], linestyle="--", linewidth=1)
    ax.set_xticks(df["lag"].astype(int))
    _style(ax)
    _set_texts(ax, t("resid_lag_title", lang), t("resid_lag_xlabel", lang), t("resid_lag_ylabel", lang))
    return fig


def plot_week_shocks(df: pd.DataFrame, lang: str, ax: Axes | None = None) -> Figure:
    """Line chart of the mean residual by week with a +-1.96 sd band expected under independence.

    Parameters
    ----------
    df : pandas.DataFrame
        Columns ``WEEK``, ``mean_resid`` and ``expected_sd`` (see ``diagnostics.week_shocks``).
    lang : str
        ``'en'`` or ``'pt'``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _prepare(ax, (9, 4))
    band = 1.96 * df["expected_sd"].to_numpy(dtype=float)
    ax.fill_between(df["WEEK"], -band, band, color=PALETTE["residual"], alpha=0.5, label=t("week_shocks_band", lang))
    ax.plot(df["WEEK"], df["mean_resid"], color=PALETTE["train"], linewidth=1.8)
    ax.axhline(0, color=PALETTE["line"], linestyle="--", linewidth=1)
    ax.legend(frameon=False, fontsize=_TICK_SIZE, loc="best")
    _style(ax)
    _set_texts(ax, t("week_shocks_title", lang), t("week_shocks_xlabel", lang), t("week_shocks_ylabel", lang))
    return fig


_MODEL_COLORS: dict[str, str] = {
    "C0": "#1f77b4",
    "C1": "#9467bd",
    "GPBoost": "#2ca02c",
    "OLS_pair": "#d95f02",
    "OLS_pooled": "#8c8c8c",
}


def plot_fold_mae(metrics: pd.DataFrame, lang: str, ax: Axes | None = None, population: str = "a_seen") -> Figure:
    """Draw grouped bars of the per-pair MAE, one group per internal fold and history stratum.

    Parameters
    ----------
    metrics : pandas.DataFrame
        Columns ``model``, ``fold``, ``population``, ``stratum``, ``n_pairs`` and ``mae_pair_mean``
        (the ``metrics`` table of the ``p1_folds`` run). Empty strata (no pairs) are skipped.
    lang : str
        ``'en'`` or ``'pt'``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    population : str, default 'a_seen'
        Population shown.

    Returns
    -------
    matplotlib.figure.Figure
    """
    data = metrics[(metrics["population"] == population) & (metrics["n_pairs"] > 0)].dropna(subset=["mae_pair_mean"])
    models = [m for m in MODEL_ORDER if m in set(data["model"])]
    strata = {s: i for i, s in enumerate(STRATUM_ORDER)}
    groups = sorted(
        {(f, s) for f, s in zip(data["fold"], data["stratum"], strict=True)},
        key=lambda g: (str(g[0]), strata.get(g[1], 9)),
    )
    fig, ax = _prepare(ax, (10, 4.5))
    x = np.arange(len(groups), dtype=float)
    width = 0.8 / max(len(models), 1)
    lookup = data.set_index(["model", "fold", "stratum"])["mae_pair_mean"]
    for i, model in enumerate(models):
        values = [lookup.get((model, f, s), np.nan) for f, s in groups]
        offset = (i - (len(models) - 1) / 2) * width
        ax.bar(
            x + offset,
            values,
            width,
            color=_MODEL_COLORS.get(model, PALETTE["neutral"]),
            label=model_label(model, lang),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t('fold_label', lang).format(fold=f)}\n{stratum_label(s, lang)}" for f, s in groups])
    ax.set_ylim(0, float(data["mae_pair_mean"].max()) * 1.2)
    ax.legend(frameon=False, fontsize=_TICK_SIZE, loc="upper center", ncol=max(len(models), 1))
    _style(ax)
    _set_texts(ax, t("fold_mae_title", lang), None, t("fold_mae_ylabel", lang))
    return fig


def plot_weekly_error_diff(weekly_diff: pd.DataFrame, lang: str, ax: Axes | None = None) -> Figure:
    """Draw the weekly error difference between C0 and each reference, with a zero line.

    Parameters
    ----------
    weekly_diff : pandas.DataFrame
        Columns ``WEEK``, ``compared_to`` and ``mae_diff`` (the ``weekly_diff`` table of the
        ``p1_final`` run). ``mae_diff`` is the error of C0 minus the error of the reference, so
        negative values favour C0. One line is drawn per reference.
    lang : str
        ``'en'`` or ``'pt'``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _prepare(ax, (9, 4))
    for reference, part in weekly_diff.groupby("compared_to", sort=False):
        part = part.sort_values("WEEK")
        ax.plot(
            part["WEEK"],
            part["mae_diff"],
            color=_MODEL_COLORS.get(str(reference), PALETTE["neutral"]),
            linewidth=1.8,
            label=model_label(str(reference), lang),
        )
    ax.axhline(0, color=PALETTE["line"], linestyle="--", linewidth=1)
    ax.legend(frameon=False, fontsize=_TICK_SIZE, loc="best")
    _style(ax)
    _set_texts(ax, t("weekly_diff_title", lang), t("weekly_diff_xlabel", lang), t("weekly_diff_ylabel", lang))
    return fig


def plot_moderator_slopes(slopes: pd.DataFrame, lang: str, tests: list[str] | None = None) -> Figure:
    """Draw the discount slope of every moderator level with pointwise 95% intervals, one panel per test.

    Parameters
    ----------
    slopes : pandas.DataFrame
        Columns ``test, level, is_reference, slope, ci_low, ci_high`` (the ``slopes`` table of the
        ``p2_contrasts`` run). Private-label levels (``"False"``/``"True"``) are shown as brand labels and
        the reference level is drawn hollow. The intervals are pointwise, not adjusted for multiplicity.
    lang : str
        ``'en'`` or ``'pt'``.
    tests : list of str, optional
        Tests to draw, in order. Defaults to the order of appearance in ``slopes``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    tests = list(dict.fromkeys(slopes["test"])) if tests is None else tests
    counts = [int((slopes["test"] == name).sum()) for name in tests]
    fig, axes = plt.subplots(
        len(tests), 1, sharex=True, figsize=(8, 1.4 + 0.45 * sum(counts)), height_ratios=counts, layout="constrained",
        squeeze=False,
    )  # fmt: skip
    lo_all, hi_all = float(slopes["ci_low"].min()), float(slopes["ci_high"].max())
    for ax, name in zip(axes[:, 0], tests, strict=True):
        part = slopes[slopes["test"] == name].reset_index(drop=True)
        pos = np.arange(len(part))
        est = part["slope"].to_numpy(dtype=float)
        lo, hi = part["ci_low"].to_numpy(dtype=float), part["ci_high"].to_numpy(dtype=float)
        ax.errorbar(est, pos, xerr=[est - lo, hi - est], fmt="none", ecolor=PALETTE["neutral"], elinewidth=2, capsize=3)
        for p, e, is_ref in zip(pos, est, part["is_reference"], strict=True):
            ax.plot(
                e,
                p,
                "o",
                markersize=7,
                color=PALETTE["neutral"],
                markerfacecolor="white" if is_ref else PALETTE["neutral"],
            )
        names = [level_label(name, lv, lang) for lv in part["level"]]
        names = [
            f"{n} ({t('slopes_reference', lang)})" if r else n for n, r in zip(names, part["is_reference"], strict=True)
        ]
        ax.set_yticks(pos, names)
        ax.set_ylim(len(part) - 0.5, -0.5)
        ax.set_title(contrast_label(name, lang), fontsize=_LABEL_SIZE, loc="left")
        _style(ax, "x")
    pad = 0.05 * (hi_all - lo_all)
    axes[-1, 0].set_xlim(lo_all - pad, hi_all + pad)
    axes[-1, 0].set_xlabel(t("slopes_xlabel", lang), fontsize=_LABEL_SIZE)
    fig.suptitle(t("slopes_title", lang), fontsize=_TITLE_SIZE)
    return fig


_CATWEEK_MODEL_KEYS: dict[str, str] = {"F3_ref": "model_f3_ref", "F3_catweek": "model_f3_catweek"}
_CATWEEK_COLORS: dict[str, str] = {"F3_ref": "#8c8c8c", "F3_catweek": "#1f77b4"}


def plot_catweek_compare(
    fixef_compare: pd.DataFrame, lang: str, terms: list[str] | None = None, z: float = 1.96
) -> Figure:
    """Compare fixed effects of the reference F3 and of F3 with a category x week intercept.

    Parameters
    ----------
    fixef_compare : pandas.DataFrame
        Columns ``term, est_ref, se_ref, est_catweek, se_catweek`` (the ``fixef_compare`` table of the
        ``p2_catweek`` run). Intervals are ``estimate +- z * SE`` with the model-based standard errors.
    lang : str
        ``'en'`` or ``'pt'``.
    terms : list of str, optional
        Terms to draw, in order; defaults to all rows.
    z : float, default 1.96
        Normal quantile of the interval.

    Returns
    -------
    matplotlib.figure.Figure
    """
    data = fixef_compare.set_index("term")
    terms = list(data.index) if terms is None else terms
    data = data.loc[terms]
    n = len(data)
    fig, ax = _prepare(None, (9, 1.6 + 0.55 * n))
    pos = np.arange(n, dtype=float)
    series = ((-0.15, "F3_ref", "est_ref", "se_ref"), (0.15, "F3_catweek", "est_catweek", "se_catweek"))
    for offset, model, est_col, se_col in series:
        est = data[est_col].to_numpy(dtype=float)
        ax.errorbar(
            est, pos + offset, xerr=z * data[se_col].to_numpy(dtype=float), fmt="o", markersize=5,
            color=_CATWEEK_COLORS[model], ecolor=_CATWEEK_COLORS[model], elinewidth=1.8, capsize=3,
            label=t(_CATWEEK_MODEL_KEYS[model], lang),
        )  # fmt: skip
    ax.axvline(0, color=PALETTE["line"], linestyle="--", linewidth=1)
    ax.set_yticks(pos, [term_label(x, lang) for x in data.index])
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=_TICK_SIZE, loc="best")
    _style(ax, "x")
    _set_texts(ax, t("catweek_compare_title", lang), t("catweek_compare_xlabel", lang))
    return fig


def plot_lag_corr_compare(lag_corr: pd.DataFrame, lang: str, ax: Axes | None = None) -> Figure:
    """Draw grouped bars of the same-pair residual correlation by lag, one bar per model.

    Parameters
    ----------
    lag_corr : pandas.DataFrame
        Columns ``model``, ``lag`` and ``corr`` (the ``lag_corr`` table of the ``p2_catweek`` run).
    lang : str
        ``'en'`` or ``'pt'``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _prepare(ax, (7, 4))
    models = list(dict.fromkeys(lag_corr["model"]))
    lags = sorted(lag_corr["lag"].unique())
    width = 0.8 / max(len(models), 1)
    for i, model in enumerate(models):
        part = lag_corr[lag_corr["model"] == model].set_index("lag")["corr"]
        offset = (i - (len(models) - 1) / 2) * width
        name = t(_CATWEEK_MODEL_KEYS[model], lang) if model in _CATWEEK_MODEL_KEYS else str(model)
        ax.bar(
            np.arange(len(lags)) + offset, [part.get(lag, np.nan) for lag in lags], width,
            color=_CATWEEK_COLORS.get(model, PALETTE["neutral"]), label=name,
        )  # fmt: skip
    ax.set_xticks(np.arange(len(lags)), [str(int(lag)) for lag in lags])
    ax.axhline(0, color=PALETTE["line"], linestyle="--", linewidth=1)
    ax.legend(frameon=False, fontsize=_TICK_SIZE, loc="best")
    _style(ax)
    _set_texts(ax, t("lag_compare_title", lang), t("resid_lag_xlabel", lang), t("resid_lag_ylabel", lang))
    return fig


def plot_scenario_effects(scenarios: pd.DataFrame, lang: str) -> Figure:
    """Draw the scenario effects with pointwise intervals beside the observed support of each scenario.

    Parameters
    ----------
    scenarios : pandas.DataFrame
        Columns ``scenario, effect_pct, ci_low_pct, ci_high_pct, share, n_rows, n_rows_fallback`` and
        ``outside_support`` (the ``scenarios`` table of the ``p2_scenarios`` run). Scenario names that are
        i18n keys are translated. A scenario outside the observed support is drawn with a hollow marker
        and a note instead of a share; it stays in the chart.
    lang : str
        ``'en'`` or ``'pt'``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    data = scenarios.reset_index(drop=True)
    n = len(data)
    fig, (ax_eff, ax_sup) = plt.subplots(
        1, 2, sharey=True, figsize=(11, 1.8 + 0.5 * n), width_ratios=[3, 2], layout="constrained"
    )
    pos = np.arange(n)
    est = data["effect_pct"].to_numpy(dtype=float)
    lo, hi = data["ci_low_pct"].to_numpy(dtype=float), data["ci_high_pct"].to_numpy(dtype=float)
    outside = data["outside_support"].to_numpy(dtype=bool)
    ax_eff.errorbar(est, pos, xerr=[est - lo, hi - est], fmt="none", ecolor=PALETTE["neutral"], elinewidth=2, capsize=3)
    for p, e, out in zip(pos, est, outside, strict=True):
        face = "white" if out else PALETTE["neutral"]
        ax_eff.plot(e, p, "o", markersize=7, color=PALETTE["neutral"], markerfacecolor=face)
        ax_eff.annotate(
            f"{e:+.1f}%", xy=(e, p), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=_TICK_SIZE - 1
        )
    ax_eff.axvline(0, color=PALETTE["line"], linestyle="--", linewidth=1)
    ax_eff.set_yticks(pos, [scenario_label(s, lang) for s in data["scenario"]])
    ax_eff.set_ylim(n - 0.5, -0.5)
    ax_eff.margins(x=0.08)
    _style(ax_eff, "x")
    _set_texts(ax_eff, t("scen_effects_title", lang), t("scen_effects_xlabel", lang))
    share = data["share"].to_numpy(dtype=float) * 100
    n_fallback = data["n_rows_fallback"].to_numpy(dtype=int)
    ax_sup.barh(pos, share, color=PALETTE["train"], height=0.6)
    for p, s_, out, n_fb in zip(pos, share, outside, n_fallback, strict=True):
        text = t("scen_outside", lang) if out else f"{s_:.1f}%"
        if n_fb > 0 and not out:
            text += f" ({t('scen_fallback', lang).format(n=n_fb)})"
        ax_sup.annotate(
            text, xy=(s_, p), xytext=(4, 0), textcoords="offset points", va="center", fontsize=_TICK_SIZE - 1
        )
    ax_sup.set_xscale("log")
    positive = share[share > 0]
    ax_sup.set_xlim(min(0.1, positive.min() / 2) if positive.size else 0.1, max(10.0, share.max() * 60))
    _style(ax_sup, "x")
    _set_texts(ax_sup, t("scen_support_title", lang), t("scen_support_xlabel", lang))
    return fig


def plot_pair_slopes(slopes: pd.DataFrame, lang: str, ax: Axes | None = None, n_bins: int = 50) -> Figure:
    """Histogram of the per-pair OLS discount slopes, stacked by history stratum, with a line at zero.

    The horizontal range covers the 1st to the 99th percentile of the slopes; the pairs beyond it are
    counted in a note instead of being drawn.

    Parameters
    ----------
    slopes : pandas.DataFrame
        Columns ``slope`` (``NaN`` rows are ignored) and ``stratum`` (``<26``, ``26-77``, ``>=78``), as in the
        ``slopes`` table of the ``pair_ols_slopes`` run.
    lang : str
        ``'en'`` or ``'pt'``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    n_bins : int
        Number of histogram bins over the drawn range.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _prepare(ax, (9, 4.5))
    data = slopes.loc[slopes["slope"].notna()]
    values = data["slope"].to_numpy(dtype=float)
    strata = [x for x in STRATUM_ORDER if x != "all"]
    if values.size:
        lo, hi = (float(v) for v in np.quantile(values, [0.01, 0.99]))
        lo, hi = min(lo, -1e-9), max(hi, 1e-9)
        inside = (values >= lo) & (values <= hi)
        bins = np.linspace(lo, hi, n_bins + 1)
        groups = [data.loc[data["stratum"] == x, "slope"].to_numpy(dtype=float) for x in strata]
        ax.hist(
            [g[(g >= lo) & (g <= hi)] for g in groups],
            bins=bins,
            stacked=True,
            color=GROUP_COLORS[: len(strata)],
            edgecolor="white",
            linewidth=0.4,
            label=[f"{stratum_label(x, lang)} (n = {len(g)})" for x, g in zip(strata, groups, strict=True)],
        )
        n_out = int((~inside).sum())
        if n_out:
            ax.annotate(
                t("pair_slopes_outside", lang).format(n=n_out),
                xy=(1.0, 1.0),
                xycoords="axes fraction",
                xytext=(-5, -5),
                textcoords="offset points",
                ha="right",
                va="top",
                fontsize=_TICK_SIZE,
            )
    ax.axvline(0, color=PALETTE["line"], linestyle="--", linewidth=1.5)
    ax.legend(frameon=False, fontsize=_TICK_SIZE, loc="upper left")
    _style(ax)
    _set_texts(ax, t("pair_slopes_title", lang), t("pair_slopes_xlabel", lang), t("pair_slopes_ylabel", lang))
    return fig
