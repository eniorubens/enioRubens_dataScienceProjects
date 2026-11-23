"""Graficos reutilizados pelos notebooks PT-BR e EN-US.

Cada funcao recebe tabelas ja calculadas e o idioma, e devolve uma figura
``matplotlib``. Nenhuma funcao calcula taxas ou ajusta modelos.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from freq_sinistros.labels import text

COLORS = {"global": "#6b7280", "poisson": "#2563eb", "nb2": "#d97706",
          "poisson_free_exposure": "#059669"}


def _finalize_figure(fig: Figure) -> Figure:
    """Aplique as regras visuais comuns e ajuste o layout da figura."""
    for ax in fig.axes:
        title = ax.get_title()
        if title:
            ax.set_title(f"{title[:1].upper()}{title[1:]}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def plot_claimnb(dist: pd.DataFrame, lang: str) -> Figure:
    """Barras de linhas por valor de ``ClaimNb`` (escala logaritmica)."""
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.bar(dist["claimnb"].astype(str), dist["rows"], color="#2563eb")
    ax.set_yscale("log")
    ax.set_xlabel(text("claimnb_x", lang))
    ax.set_ylabel(text("rows", lang))
    ax.set_title(text("claimnb_title", lang))
    for i, (r, s) in enumerate(zip(dist["rows"], dist["share_of_valid_rows"],
                                   strict=True)):
        ax.annotate(f"{100 * s:.2f}%", (i, r), ha="center", va="bottom",
                    fontsize=7)
    return _finalize_figure(fig)


def plot_exposure(hist: pd.DataFrame, lang: str) -> Figure:
    """Histograma de ``Exposure`` com a barra de ``= 1`` destacada."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    body = hist[hist["kind"] != "exactly 1"]
    one = hist[hist["kind"] == "exactly 1"].iloc[0]
    ax.bar(body["upper"] - 0.025, body["rows"], width=0.045,
           color="#2563eb")
    ax.bar([1.0], [one["rows"]], width=0.03, color="#d97706",
           label=text("exactly_one", lang))
    ax.set_xlabel(text("exposure_x", lang))
    ax.set_ylabel(text("rows", lang))
    ax.set_title(text("exposure_title", lang))
    ax.legend()
    return _finalize_figure(fig)


def plot_rates(table: pd.DataFrame, lang: str, label_key: str,
               rate_col: str = "rate_per_100") -> Figure:
    """Barras da taxa por 100 apolices-ano; baixo suporte hachurado.

    Args:
        table: tabela com ``segment``, ``low_support`` e a coluna de taxa.
        lang: ``pt_br`` ou ``en_us``.
        label_key: chave de ``labels.TEXT`` para titulo e eixo x.
        rate_col: coluna da taxa.
    """
    label = text(label_key, lang)
    fig, ax = plt.subplots(figsize=(max(6.0, 0.45 * len(table) + 3), 3.8))
    low = table["low_support"].to_numpy(dtype=bool)
    x = np.arange(len(table))
    ax.bar(x[~low], table.loc[~low, rate_col], color="#2563eb",
           label=text("ok_support", lang))
    if low.any():
        ax.bar(x[low], table.loc[low, rate_col].fillna(0), color="#d1d5db",
               hatch="//", edgecolor="#6b7280",
               label=text("low_support", lang))
    ax.set_xticks(x)
    ax.set_xticklabels(table["segment"], rotation=45, ha="right")
    ax.set_ylabel(text("rate100", lang))
    ax.set_xlabel(label)
    ax.set_title(f"{text('rate100', lang)} - {label}")
    ax.legend()
    return _finalize_figure(fig)


def plot_short_check(bands: pd.DataFrame, lang: str) -> Figure:
    """Razao observado/esperado por faixa de exposicao (escala log)."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(bands["band"], bands["ratio_observed_expected"], color="#d97706")
    ax.axhline(1.0, color="#111827", lw=1)
    ax.set_yscale("log")
    ax.set_ylabel(text("short_y", lang))
    ax.set_xlabel(text("band", lang))
    ax.set_title(text("short_title", lang))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return _finalize_figure(fig)


def plot_calibration_bands(cal: pd.DataFrame, lang: str,
                           candidates: Sequence[str] = ("global", "poisson",
                                                        "nb2")) -> Figure:
    """Diferenca de taxas (prevista - observada) por faixa de exposicao."""
    sub = cal[(cal["family"] == "exposure_band")
              & cal["candidate"].isin(candidates)]
    bands = list(dict.fromkeys(sub["segment"]))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    width = 0.8 / max(len(candidates), 1)
    for k, cand in enumerate(candidates):
        part = sub[sub["candidate"] == cand].set_index("segment").reindex(
            bands)
        x = np.arange(len(bands)) + k * width
        diff = part["rate_difference_pred_minus_obs"]
        err = np.vstack([diff - part["ci_low"], part["ci_high"] - diff])
        err = np.where(np.isfinite(err), err, 0)
        ax.bar(x, diff, width=width, color=COLORS.get(cand, "#374151"),
               label=cand, yerr=err, capsize=2, error_kw={"lw": 0.8})
    ax.axhline(0, color="#111827", lw=1)
    ax.set_xticks(np.arange(len(bands)) + width)
    ax.set_xticklabels(bands, rotation=45, ha="right")
    ax.set_ylabel(text("pred_minus_obs", lang))
    ax.set_xlabel(text("band", lang))
    ax.set_title(text("cal_title", lang))
    ax.legend()
    return _finalize_figure(fig)


def plot_forest(intervals: pd.DataFrame, lang: str,
                indicators: list[str]) -> Figure:
    """Estimativa e IC 95% de indicadores pareados."""
    sub = intervals.set_index("indicator").reindex(indicators)
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(indicators) + 1.5))
    y = np.arange(len(indicators))
    est = sub["estimate"].to_numpy(dtype="float64")
    lo = est - sub["ci_low"].to_numpy(dtype="float64")
    hi = sub["ci_high"].to_numpy(dtype="float64") - est
    ax.errorbar(est, y, xerr=np.vstack([lo, hi]), fmt="o",
                color="#2563eb", capsize=3)
    ax.axvline(0, color="#111827", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(indicators, fontsize=8)
    ax.set_title(text("forest_title", lang))
    return _finalize_figure(fig)


def plot_cost(fits: pd.DataFrame, lang: str) -> Figure:
    """Tempo de parede dos ajustes principais por degrau e candidato."""
    main = fits[fits["context"] == "main"]
    pivot = main.pivot(index="degree", columns="candidate",
                       values="wall_s")[["global", "poisson", "nb2"]]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    pivot.plot.bar(ax=ax, color=[COLORS[c] for c in pivot.columns])
    ax.set_ylabel(text("seconds", lang))
    ax.set_xlabel(text("degree", lang))
    ax.set_title(text("cost_title", lang))
    plt.setp(ax.get_xticklabels(), rotation=0)
    return _finalize_figure(fig)


def plot_zero_diff(zeros: pd.DataFrame, lang: str,
                   candidates: Sequence[str] = ("global", "poisson", "nb2")
                   ) -> Figure:
    """Zeros previstos menos observados por faixa de exposicao."""
    sub = zeros[zeros["candidate"].isin(candidates)
                & (zeros["band"] != "overall")]
    bands = list(dict.fromkeys(sub["band"]))
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    width = 0.8 / max(len(candidates), 1)
    for k, cand in enumerate(candidates):
        part = sub[sub["candidate"] == cand].set_index("band").reindex(
            bands)
        diff = part["diff_pred_minus_obs"]
        err = np.vstack([diff - part["ci_low"], part["ci_high"] - diff])
        err = np.where(np.isfinite(err), err, 0)
        ax.bar(np.arange(len(bands)) + k * width, diff, width=width,
               color=COLORS.get(cand, "#374151"), label=cand, yerr=err,
               capsize=2, error_kw={"lw": 0.8})
    ax.axhline(0, color="#111827", lw=1)
    ax.set_xticks(np.arange(len(bands)) + width)
    ax.set_xticklabels(bands, rotation=45, ha="right")
    ax.set_title(text("zero_title", lang))
    ax.set_xlabel(text("band", lang))
    ax.legend()
    return _finalize_figure(fig)


def show(fig: Figure) -> None:
    """Exiba a figura no notebook e a feche em seguida."""
    from IPython.display import display

    display(fig)
    plt.close(fig)


def plot_exposure_rates(bands: pd.DataFrame, lang: str) -> Figure:
    """Taxa observada e previstas (offset fixo e coeficiente livre)."""
    sub = bands[bands["band"] != "total"]
    x = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    series = [("observed_rate_per_100", text("observed", lang), "#111827"),
              ("predicted_rate_poisson", text("poisson_offset", lang),
               COLORS["poisson"]),
              ("predicted_rate_poisson_free_exposure",
               text("poisson_free", lang),
               COLORS["poisson_free_exposure"])]
    width = 0.27
    for k, (col, label, color) in enumerate(series):
        ax.bar(x + (k - 1) * width, sub[col], width=width, label=label,
               color=color)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["band"], rotation=45, ha="right")
    ax.set_ylabel(text("rate100", lang))
    ax.set_xlabel(text("band", lang))
    ax.set_title(text("rates_title", lang))
    ax.legend()
    return _finalize_figure(fig)


def plot_stability(decision: pd.DataFrame, lang: str) -> Figure:
    """Diferenca de desvio Poisson - taxa global por degrau, com IC."""
    sub = decision[decision["population"] == "validation"]
    est = sub["d_dev_poisson_minus_global"].to_numpy(dtype="float64")
    lo = est - sub["d_dev_poisson_minus_global_lo"].to_numpy(dtype="float64")
    hi = sub["d_dev_poisson_minus_global_hi"].to_numpy(dtype="float64") - est
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.errorbar(est, np.arange(len(sub)), xerr=np.vstack([lo, hi]),
                fmt="o", color=COLORS["poisson"], capsize=3)
    ax.axvline(0, color="#111827", lw=1)
    ax.set_yticks(np.arange(len(sub)))
    ax.set_yticklabels(sub["degree"].astype(str))
    ax.set_xlabel(text("dev_diff", lang))
    ax.set_ylabel(text("degree", lang))
    ax.set_title(text("stability_title", lang))
    return _finalize_figure(fig)


def plot_panel(panel: pd.DataFrame, lang: str, family: str,
               candidate: str = "poisson") -> Figure:
    """Diferenca prevista - observada por segmento, com IC e suporte."""
    sub = panel[(panel["family"] == family)
                & (panel["candidate"] == candidate)].reset_index(drop=True)
    diff = sub["pred_minus_obs"].to_numpy(dtype="float64")
    lo = np.where(np.isfinite(sub["ci_low"]), diff - sub["ci_low"], 0.0)
    hi = np.where(np.isfinite(sub["ci_high"]), sub["ci_high"] - diff, 0.0)
    low = sub["low_support"].to_numpy(dtype=bool)
    fig, ax = plt.subplots(figsize=(7.5, max(3.2, 0.28 * len(sub) + 1.5)))
    y = np.arange(len(sub))
    ax.barh(y[~low], diff[~low], xerr=np.vstack([lo[~low], hi[~low]]),
            color=COLORS.get(candidate, "#2563eb"), capsize=2,
            label=text("ok_support", lang))
    if low.any():
        ax.barh(y[low], diff[low], color="#d1d5db", hatch="//",
                edgecolor="#6b7280", label=text("low_support", lang))
    ax.axvline(0, color="#111827", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["segment"])
    ax.invert_yaxis()
    ax.set_xlabel(text("pred_minus_obs", lang))
    ax.set_title(f"{text('panel_title', lang)} ({candidate})")
    ax.legend()
    return _finalize_figure(fig)


def plot_validation_vs_test(comparison: pd.DataFrame, lang: str) -> Figure:
    """Diferenca prevista - observada por faixa: validacao e teste."""
    x = np.arange(len(comparison))
    width = 0.38
    diff = comparison["test_diff"].to_numpy(dtype="float64")
    lo = np.where(np.isfinite(comparison["test_ci_low"]),
                  diff - comparison["test_ci_low"], 0.0)
    hi = np.where(np.isfinite(comparison["test_ci_high"]),
                  comparison["test_ci_high"] - diff, 0.0)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(x - width / 2, comparison["validation_diff"], width,
           color="#6b7280", label=text("validation", lang))
    ax.bar(x + width / 2, diff, width, yerr=np.vstack([lo, hi]),
           color=COLORS["poisson"], capsize=2, label=text("test", lang))
    ax.axhline(0, color="#111827", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(comparison["band"], rotation=45, ha="right")
    ax.set_ylabel(text("pred_minus_obs", lang))
    ax.set_xlabel(text("band", lang))
    ax.set_title(text("val_vs_test_title", lang))
    ax.legend()
    return _finalize_figure(fig)


def plot_paired_difference(intervals: pd.DataFrame, lang: str) -> Figure:
    """Diferenca pareada de desvio (Poisson - global) nos dois bootstraps."""
    key = "d_dev|poisson-global"
    rows = [("test", text("bootstrap_main", lang)),
            ("test_cluster_bootstrap", text("bootstrap_profile", lang))]
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    for k, (pop, _) in enumerate(rows):
        r = intervals[(intervals["population"] == pop)
                      & (intervals["indicator"] == key)].iloc[0]
        ax.errorbar(r["estimate"], k, xerr=[[r["estimate"] - r["ci_low"]],
                                            [r["ci_high"] - r["estimate"]]],
                    fmt="o", color=COLORS["poisson"], capsize=3)
    ax.axvline(0, color="#111827", lw=1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([r[1] for r in rows])
    ax.set_xlabel(text("dev_diff", lang))
    ax.set_title(text("paired_title", lang))
    return _finalize_figure(fig)
