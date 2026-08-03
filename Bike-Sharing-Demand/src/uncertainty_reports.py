"""Reports and charts for notebook 06 uncertainty experiments."""

from __future__ import annotations

from typing import Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.i18n import localize_table, resolve_lang
from src.model_selection_reports import (
    environment_report as _environment_report,
    fold_audit_report as _selection_fold_audit_report,
    plot_fold_audit as _selection_plot_fold_audit,
)
from src.uncertainty_experiments import (
    OperationalResidualContract,
    UncertaintyExperimentConfig,
    UncertaintyExperimentResults,
    frozen_artifact_hashes,
    probabilistic_fold_metrics as _derive_probabilistic_fold_metrics,
    scale_diagnostics as _derive_scale_diagnostics,
)
from src.utils import public_path

_BASELINE_COLOR = "#9e9e9e"
_CHAMPION_COLOR = "#f58518"
_CANDIDATE_COLOR = "#4c78a8"
_PROB_COLOR = "#54a24b"

_VALUE_LABELS = {
    "E0": "E0",
    "E1": "E1",
    "E2": "E2",
    "E3": "E3",
    "E4": "E4",
    "selection": "selecao",
    "stress": "estresse",
    "Winter": "Inverno",
    "Spring": "Primavera",
    "Summer": "Verao",
    "Autumn": "Outono",
    "No Rain": "Sem chuva",
    "Light Rain": "Chuva leve",
    "Moderate Rain": "Chuva moderada",
    "Heavy Rain": "Chuva forte",
    "Non-Rush": "Fora do rush",
    "Morning Rush": "Rush da manha",
    "Evening Rush": "Rush da tarde",
    True: "Sim",
    False: "Nao",
}

_EXPERIMENT_LINE_STYLES = {
    "E0": {"color": _CHAMPION_COLOR, "linewidth": 2.8, "linestyle": "-"},
    "E1": {"color": "#72b7b2", "linewidth": 1.8, "linestyle": "-"},
    "E2": {"color": _PROB_COLOR, "linewidth": 1.8, "linestyle": "-"},
    "E3": {"color": _CANDIDATE_COLOR, "linewidth": 2.0, "linestyle": "-"},
    "E4": {"color": "#b279a2", "linewidth": 2.0, "linestyle": "--"},
}


def _selection_fold_metrics(results: UncertaintyExperimentResults) -> pd.DataFrame:
    return results.fold_metrics.loc[results.fold_metrics["fold_role"].eq("selection")].copy()


def _probabilistic_fold_frame(results: UncertaintyExperimentResults) -> pd.DataFrame:
    frame = getattr(results, "probabilistic_fold_metrics", None)
    if frame is None:
        return _derive_probabilistic_fold_metrics(results.predictions, results.config)
    return frame.copy()


def _scale_frame(results: UncertaintyExperimentResults) -> pd.DataFrame:
    frame = getattr(results, "scale_diagnostics", None)
    if frame is None:
        return _derive_scale_diagnostics(results.predictions)
    return frame.copy()


def _localized_value(value):
    return _VALUE_LABELS.get(value, str(value))


def _key_value_frame(lang, rows: List[tuple]) -> pd.DataFrame:
    labels = lang({"item": "Item", "value": "Valor"})
    return pd.DataFrame([{labels["item"]: label, labels["value"]: value} for label, value in rows])


def environment_report(lang=None) -> pd.DataFrame:
    return _environment_report(lang=lang)


def protocol_report(config: UncertaintyExperimentConfig, lang=None) -> pd.DataFrame:
    lang = resolve_lang(lang)
    labels = lang(
        {
            "mode": "Modo de execucao",
            "claim": "Natureza da conclusao",
            "holdout": "Uso do holdout final",
            "post": "Uso de dezembro de 2024",
            "cv": "Protocolo temporal",
            "folds": "Folds normais no modo smoke",
            "iterations": "Iteracoes por CatBoost no smoke",
            "contract": "Contrato operacional residual",
        }
    )
    contract = OperationalResidualContract()
    return _key_value_frame(
        lang,
        [
            (labels["mode"], config.run_mode),
            (labels["claim"], "candidato sucessor experimental"),
            (labels["holdout"], "nao utilizado"),
            (labels["post"], "nao utilizado"),
            (labels["cv"], "ForwardMeteorologicalYearSplit sobre desenvolvimento"),
            (labels["folds"], config.smoke_fold_limit if config.run_mode == "smoke" else "todos"),
            (
                labels["iterations"],
                config.smoke_iterations if config.run_mode == "smoke" else "manifesto",
            ),
            (labels["contract"], contract.forecast_horizon),
        ],
    )


def artifact_hash_report(paths: Optional[List[Any]] = None, lang=None) -> pd.DataFrame:
    lang = resolve_lang(lang)
    frame = frozen_artifact_hashes(tuple(paths or ()))
    if frame.empty:
        return frame
    frame["path"] = frame["path"].map(public_path)
    return localize_table(
        frame,
        lang,
        {"path": "Artefato", "sha256": "SHA256"},
    )


def development_report(results_or_development: Any, lang=None) -> pd.DataFrame:
    lang = resolve_lang(lang)
    development = getattr(results_or_development, "development", results_or_development)
    holdout = development.holdout
    labels = lang(
        {
            "dev": "Periodo de desenvolvimento",
            "rows": "Linhas de desenvolvimento",
            "sealed": "Holdout selado",
            "post": "Periodo posterior descartado",
            "dataset": "Fingerprint do dataset",
            "regime": "Fingerprint do regime",
            "eligible": "Linhas elegiveis",
            "excluded": "Linhas excluidas",
        }
    )
    return _key_value_frame(
        lang,
        [
            (labels["dev"], f"{holdout.dev_start:%d/%m/%Y} a {holdout.dev_end:%d/%m/%Y}"),
            (labels["rows"], len(development.X_dev)),
            (labels["sealed"], "sim, sem retorno de linhas"),
            (
                labels["post"],
                f"{holdout.post_holdout_start:%d/%m/%Y} a {holdout.post_holdout_end:%d/%m/%Y}",
            ),
            (labels["dataset"], development.fingerprint),
            (labels["regime"], development.regime_fingerprint),
            (labels["eligible"], int(development.train_eligible_mask.sum())),
            (labels["excluded"], int((~development.train_eligible_mask).sum())),
        ],
    )


def fold_audit_report(results_or_development: Any, lang=None) -> pd.DataFrame:
    development = getattr(results_or_development, "development", results_or_development)
    return _selection_fold_audit_report(development, lang=lang)


def plot_fold_audit(results_or_development: Any, lang=None):
    development = getattr(results_or_development, "development", results_or_development)
    return _selection_plot_fold_audit(development, lang=lang)


def experiment_spec_report(results_or_specs: Any, lang=None) -> pd.DataFrame:
    lang = resolve_lang(lang)
    specs = getattr(results_or_specs, "specs", results_or_specs)
    frame = pd.DataFrame([spec.__dict__ for spec in specs])
    return localize_table(
        frame,
        lang,
        {
            "experiment_id": "Experimento",
            "label": "Especificacao",
            "point_model": "Modelo pontual",
            "uses_hour_of_week": "Usa hora da semana",
            "uses_weather_interactions": "Usa interacoes meteorologicas",
            "probabilistic_loss": "Usa RMSEWithUncertainty",
            "residual_scale_model": "Usa escala residual",
            "point_prediction": "Previsao pontual",
            "status": "Status",
            "notes": "Notas",
        },
        value_columns=(
            "uses_hour_of_week",
            "uses_weather_interactions",
            "probabilistic_loss",
            "residual_scale_model",
        ),
        value_labels=_VALUE_LABELS,
    )


def point_metrics_report(results: UncertaintyExperimentResults, lang=None) -> pd.DataFrame:
    lang = resolve_lang(lang)
    frame = results.aggregate_metrics.copy()
    return localize_table(
        frame,
        lang,
        {
            "experiment_id": "Experimento",
            "cv_mae_mean": "MAE medio",
            "cv_mae_weighted": "MAE ponderado",
            "cv_rmse_mean": "RMSE medio",
            "cv_r2_mean": "R2 medio",
            "cv_r2_median": "R2 mediano",
            "cv_r2_weighted": "R2 ponderado",
            "cv_wape_mean": "WAPE medio",
            "cv_mean_bias": "Bias medio",
            "cv_mean_abs_fold_bias": "Media do bias absoluto por fold",
            "cv_mae_std": "Desvio-padrao do MAE",
        },
    )


def plot_point_metrics(results: UncertaintyExperimentResults, lang=None):
    lang = resolve_lang(lang)
    labels = lang({"title": "Comparacao pontual por validacao temporal", "x": "MAE ponderado"})
    frame = results.aggregate_metrics.sort_values("cv_mae_weighted", ascending=True)
    colors = [
        _BASELINE_COLOR
        if exp == "E0"
        else _CHAMPION_COLOR
        if exp == frame.iloc[0]["experiment_id"]
        else _CANDIDATE_COLOR
        for exp in frame["experiment_id"]
    ]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(frame["experiment_id"], frame["cv_mae_weighted"], color=colors)
    ax.invert_yaxis()
    ax.set_title(labels["title"])
    ax.set_xlabel(labels["x"])
    for idx, value in enumerate(frame["cv_mae_weighted"]):
        ax.text(value, idx, f" {value:,.0f}", va="center")
    return fig


def plot_point_metrics_by_fold(results: UncertaintyExperimentResults, lang=None):
    """Plot point MAE and R2 by normal selection fold."""
    lang = resolve_lang(lang)
    labels = lang(
        {
            "title": "Desempenho pontual por fold normal",
            "mae": "MAE",
            "r2": "R2",
            "year": "Ano meteorologico",
            "note": "E4 usa as mesmas previsoes pontuais de E3",
        }
    )
    frame = _selection_fold_metrics(results).sort_values(["experiment_id", "test_year"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    for experiment_id, group in frame.groupby("experiment_id", sort=False):
        style = _EXPERIMENT_LINE_STYLES.get(
            experiment_id,
            {"linewidth": 1.8, "linestyle": "-", "color": _CANDIDATE_COLOR},
        )
        label = f"{experiment_id} ({labels['note']})" if experiment_id == "E4" else experiment_id
        axes[0].plot(
            group["test_year"],
            group["selection_mae"],
            marker="o",
            label=label,
            **style,
        )
        axes[1].plot(
            group["test_year"],
            group["selection_r2"],
            marker="o",
            label=label,
            **style,
        )
    axes[0].set_title(labels["mae"])
    axes[1].set_title(labels["r2"])
    for ax in axes:
        ax.set_xlabel(labels["year"])
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(labels["mae"])
    axes[1].set_ylabel(labels["r2"])
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle(labels["title"])
    fig.tight_layout()
    return fig


def fold_metrics_report(results: UncertaintyExperimentResults, lang=None) -> pd.DataFrame:
    lang = resolve_lang(lang)
    columns = [
        "experiment_id",
        "fold",
        "test_year",
        "n_train",
        "n_train_excluded",
        "n_test",
        "n_selection_test",
        "selection_mae",
        "selection_rmse",
        "selection_r2",
        "selection_wape",
        "selection_mean_bias",
    ]
    frame = results.fold_metrics.loc[
        results.fold_metrics["fold_role"].eq("selection"), columns
    ].copy()
    return localize_table(
        frame,
        lang,
        {
            "experiment_id": "Experimento",
            "fold": "Fold",
            "test_year": "Ano meteorologico",
            "n_train": "Linhas de treino",
            "n_train_excluded": "Linhas de treino excluidas",
            "n_test": "Linhas de teste",
            "n_selection_test": "Linhas normais pontuadas",
            "selection_mae": "MAE de selecao",
            "selection_rmse": "RMSE de selecao",
            "selection_r2": "R2 de selecao",
            "selection_wape": "WAPE de selecao",
            "selection_mean_bias": "Bias de selecao",
        },
    )


def stress_metrics_report(results: UncertaintyExperimentResults, lang=None) -> pd.DataFrame:
    """Keep the excluded 2020 regime visible without mixing it into selection."""
    lang = resolve_lang(lang)
    columns = [
        "experiment_id",
        "fold",
        "test_year",
        "n_test",
        "mae",
        "rmse",
        "r2",
        "wape",
        "mean_bias",
    ]
    frame = results.fold_metrics.loc[results.fold_metrics["fold_role"].eq("stress"), columns].copy()
    return localize_table(
        frame,
        lang,
        {
            "experiment_id": "Experimento",
            "fold": "Fold",
            "test_year": "Ano meteorologico",
            "n_test": "Linhas do diagnostico",
            "mae": "MAE de estresse",
            "rmse": "RMSE de estresse",
            "r2": "R2 de estresse",
            "wape": "WAPE de estresse",
            "mean_bias": "Bias de estresse",
        },
    )


def residual_dependence_report(results: UncertaintyExperimentResults, lang=None) -> pd.DataFrame:
    lang = resolve_lang(lang)
    return localize_table(
        results.residual_metrics,
        lang,
        {
            "experiment_id": "Experimento",
            "n": "Observacoes",
            "residual_acf_lag_1": "ACF residual lag 1",
            "residual_acf_lag_24": "ACF residual lag 24",
            "residual_acf_lag_168": "ACF residual lag 168",
            "squared_residual_acf_lag_1": "ACF residual quadratico lag 1",
            "squared_residual_acf_lag_24": "ACF residual quadratico lag 24",
            "squared_residual_acf_lag_168": "ACF residual quadratico lag 168",
            "arch_per_obs_lag_24": "ARCH por observacao lag 24",
            "arch_per_obs_lag_168": "ARCH por observacao lag 168",
        },
    )


def plot_residual_dependence(results: UncertaintyExperimentResults, lang=None):
    lang = resolve_lang(lang)
    labels = lang(
        {"title": "Persistencia residual por experimento", "x": "Experimento", "y": "ACF"}
    )
    frame = results.residual_metrics
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(frame))
    width = 0.24
    for offset, lag in zip((-width, 0, width), (1, 24, 168)):
        ax.bar(x + offset, frame[f"residual_acf_lag_{lag}"], width=width, label=f"lag {lag}")
    ax.set_xticks(x)
    ax.set_xticklabels(frame["experiment_id"])
    ax.set_title(labels["title"])
    ax.set_xlabel(labels["x"])
    ax.set_ylabel(labels["y"])
    ax.legend()
    return fig


def plot_residual_diagnostics_heatmap(results: UncertaintyExperimentResults, lang=None):
    """Display ACF and ARCH residual diagnostics in one annotated matrix."""
    lang = resolve_lang(lang)
    labels = lang(
        {
            "title": "Diagnosticos residuais por experimento",
            "subtitle": "Valores proximos de zero indicam menor persistencia residual",
            "experiment": "Experimento",
        }
    )
    columns = [
        ("residual_acf_lag_1", "ACF res. 1h"),
        ("residual_acf_lag_24", "ACF res. 24h"),
        ("residual_acf_lag_168", "ACF res. 168h"),
        ("squared_residual_acf_lag_1", "ACF res.^2 1h"),
        ("squared_residual_acf_lag_24", "ACF res.^2 24h"),
        ("squared_residual_acf_lag_168", "ACF res.^2 168h"),
        ("arch_per_obs_lag_24", "ARCH/obs 24h"),
        ("arch_per_obs_lag_168", "ARCH/obs 168h"),
    ]
    frame = results.residual_metrics.set_index("experiment_id")
    matrix = frame[[column for column, _ in columns]].astype(float)
    fig, ax = plt.subplots(figsize=(12, 4.6))
    values = matrix.to_numpy()
    finite = values[np.isfinite(values)]
    limit = max(float(np.nanmax(np.abs(finite))) if finite.size else 1.0, 1e-6)
    image = ax.imshow(values, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels([label for _, label in columns], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_ylabel(labels["experiment"])
    ax.set_title(f"{labels['title']}\n{labels['subtitle']}")
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            text = "" if not np.isfinite(value) else f"{value:.3f}"
            ax.text(col_idx, row_idx, text, ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, shrink=0.78)
    fig.tight_layout()
    return fig


def probabilistic_metrics_report(results: UncertaintyExperimentResults, lang=None) -> pd.DataFrame:
    lang = resolve_lang(lang)
    return localize_table(
        results.probabilistic_metrics,
        lang,
        {
            "experiment_id": "Experimento",
            "n": "Observacoes",
            "negative_log_likelihood": "Negative log-likelihood",
            "mean_interval_width_90": "Largura media do intervalo 90%",
        },
    )


def interval_metrics_report(results: UncertaintyExperimentResults, lang=None) -> pd.DataFrame:
    lang = resolve_lang(lang)
    return localize_table(
        results.interval_metrics,
        lang,
        {
            "experiment_id": "Experimento",
            "coverage": "Cobertura nominal",
            "empirical_coverage": "Cobertura observada",
            "coverage_error": "Erro de calibracao",
            "mean_width": "Largura media",
            "winkler_score": "Winkler score",
        },
    )


def probabilistic_fold_metrics_report(
    results: UncertaintyExperimentResults,
    lang=None,
) -> pd.DataFrame:
    lang = resolve_lang(lang)
    frame = _probabilistic_fold_frame(results)
    return localize_table(
        frame,
        lang,
        {
            "experiment_id": "Experimento",
            "fold": "Fold",
            "test_year": "Ano meteorologico",
            "coverage": "Cobertura nominal",
            "empirical_coverage": "Cobertura observada",
            "coverage_error": "Erro de calibracao",
            "mean_width": "Largura media",
            "winkler_score": "Winkler score",
            "n": "Observacoes",
        },
    )


def scale_diagnostics_report(results: UncertaintyExperimentResults, lang=None) -> pd.DataFrame:
    lang = resolve_lang(lang)
    frame = _scale_frame(results)
    return localize_table(
        frame,
        lang,
        {
            "experiment_id": "Experimento",
            "fold": "Fold",
            "test_year": "Ano meteorologico",
            "n": "Observacoes",
            "fallback_rate": "Taxa de fallback",
            "floor_rate": "Taxa no piso",
            "p10": "P10",
            "p25": "P25",
            "median": "Mediana",
            "p75": "P75",
            "p90": "P90",
        },
    )


def plot_coverage_calibration(results: UncertaintyExperimentResults, lang=None):
    lang = resolve_lang(lang)
    labels = lang(
        {
            "title": "Cobertura nominal versus cobertura observada",
            "x": "Cobertura nominal",
            "y": "Cobertura observada",
        }
    )
    frame = results.interval_metrics
    fig, ax = plt.subplots(figsize=(6, 5))
    for experiment_id, group in frame.groupby("experiment_id", sort=False):
        ax.plot(group["coverage"], group["empirical_coverage"], marker="o", label=experiment_id)
    ax.plot([0, 1], [0, 1], color="#9e9e9e", linestyle="--")
    ax.set_title(labels["title"])
    ax.set_xlabel(labels["x"])
    ax.set_ylabel(labels["y"])
    ax.legend()
    return fig


def plot_probabilistic_metrics_by_fold(
    results: UncertaintyExperimentResults,
    coverage: float = 0.90,
    lang=None,
):
    """Plot probabilistic calibration, width and Winkler score by normal fold."""
    lang = resolve_lang(lang)
    labels = lang(
        {
            "title": "Calibracao probabilistica por fold normal",
            "coverage": "Cobertura observada",
            "width": "Largura media",
            "winkler": "Winkler score",
            "year": "Ano meteorologico",
            "nominal": "Cobertura nominal",
        }
    )
    frame = _probabilistic_fold_frame(results)
    frame = frame.loc[np.isclose(frame["coverage"].astype(float), float(coverage))].copy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    metric_columns = [
        ("empirical_coverage", labels["coverage"]),
        ("mean_width", labels["width"]),
        ("winkler_score", labels["winkler"]),
    ]
    for experiment_id, group in frame.groupby("experiment_id", sort=False):
        style = _EXPERIMENT_LINE_STYLES.get(
            experiment_id,
            {"linewidth": 1.8, "linestyle": "-", "color": _CANDIDATE_COLOR},
        )
        for ax, (column, title) in zip(axes, metric_columns):
            ax.plot(group["test_year"], group[column], marker="o", label=experiment_id, **style)
            ax.set_title(title)
            ax.set_xlabel(labels["year"])
            ax.grid(alpha=0.25)
    axes[0].axhline(float(coverage), color=_BASELINE_COLOR, linestyle="--", label=labels["nominal"])
    axes[0].set_ylim(0.0, 1.05)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(labels["title"])
    fig.tight_layout()
    return fig


def plot_scale_diagnostics(results: UncertaintyExperimentResults, lang=None):
    """Plot E4 scale multiplier quantiles and fallback/floor rates."""
    lang = resolve_lang(lang)
    labels = lang(
        {
            "title": "Diagnostico do multiplicador de escala do E4",
            "scale": "Scale multiplier",
            "rate": "Taxa",
            "year": "Ano meteorologico",
            "floor": "Piso 0,25",
            "fallback": "Fallback",
            "floor_rate": "No piso",
        }
    )
    frame = _scale_frame(results).sort_values("test_year")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=False)
    years = frame["test_year"].to_numpy()
    axes[0].plot(years, frame["median"], marker="o", color=_CHAMPION_COLOR, label="Mediana")
    axes[0].fill_between(
        years,
        frame["p10"].astype(float),
        frame["p90"].astype(float),
        color=_CHAMPION_COLOR,
        alpha=0.18,
        label="P10-P90",
    )
    axes[0].axhline(0.25, color=_BASELINE_COLOR, linestyle="--", label=labels["floor"])
    axes[0].set_title(labels["scale"])
    axes[0].set_xlabel(labels["year"])
    axes[0].set_ylabel(labels["scale"])
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(alpha=0.25)

    x = np.arange(len(frame))
    width = 0.36
    axes[1].bar(x - width / 2, frame["floor_rate"], width=width, label=labels["floor_rate"])
    axes[1].bar(x + width / 2, frame["fallback_rate"], width=width, label=labels["fallback"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(frame["test_year"])
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title(labels["rate"])
    axes[1].set_xlabel(labels["year"])
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle(labels["title"])
    fig.tight_layout()
    return fig


def interval_width_report(results: UncertaintyExperimentResults, lang=None) -> pd.DataFrame:
    lang = resolve_lang(lang)
    frame = (
        results.segment_metrics.loc[
            results.segment_metrics["segment"].eq("predicted_demand_decile")
            & results.segment_metrics["mean_width_90"].notna()
        ]
        .copy()
        .sort_values(["experiment_id", "segment_value"])
    )
    return localize_table(
        frame,
        lang,
        {
            "experiment_id": "Experimento",
            "segment_value": "Decil da demanda prevista",
            "n": "Observacoes",
            "mae": "MAE",
            "coverage_90": "Cobertura 90%",
            "mean_width_90": "Largura media 90%",
        },
    )


def _representative_interval_windows(results: UncertaintyExperimentResults) -> pd.DataFrame:
    predictions = results.predictions.copy()
    normal = predictions.loc[
        predictions["selection_eligible"].fillna(False) & predictions["fold_role"].eq("selection")
    ].copy()
    if normal.empty:
        return pd.DataFrame()
    latest_year = int(normal["test_year"].max())
    e0 = normal.loc[
        normal["experiment_id"].eq("E0") & normal["test_year"].eq(latest_year),
        ["timestamp", "y_pred"],
    ].rename(columns={"y_pred": "e0_prediction"})
    e4 = normal.loc[normal["experiment_id"].eq("E4") & normal["test_year"].eq(latest_year)].copy()
    required = {"timestamp", "Seasons", "y_true", "demand_median", "lower_90", "upper_90"}
    if e4.empty or not required.issubset(e4.columns):
        return pd.DataFrame()
    frame = e4.merge(e0, on="timestamp", how="left")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    windows = []
    for season in ("Winter", "Spring", "Summer", "Autumn"):
        season_frame = frame.loc[frame["Seasons"].eq(season)].sort_values("timestamp").copy()
        if season_frame.empty:
            continue
        season_frame["week_start"] = season_frame["timestamp"].dt.to_period("W-SUN").dt.start_time
        chosen = None
        for _, week in season_frame.groupby("week_start", sort=True):
            if week["timestamp"].nunique() >= 168:
                chosen = week
                break
        if chosen is None:
            chosen = season_frame.head(168)
        windows.append(chosen.assign(representative_season=season))
    if not windows:
        return pd.DataFrame()
    return pd.concat(windows, ignore_index=True)


def plot_representative_interval_windows(results: UncertaintyExperimentResults, lang=None):
    """Plot observed demand, E0 point forecast and E4 intervals for four fixed weeks."""
    lang = resolve_lang(lang)
    labels = lang(
        {
            "title": "Semanas representativas no fold normal mais recente",
            "observed": "Demanda observada",
            "e0": "Previsao E0",
            "median": "Mediana E4",
            "interval": "Intervalo 90% E4",
            "demand": "Bicicletas/hora",
        }
    )
    frame = _representative_interval_windows(results)
    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharey=True)
    axes = axes.ravel()
    for ax, season in zip(axes, seasons):
        window = frame.loc[frame["representative_season"].eq(season)].sort_values("timestamp")
        ax.set_title(_localized_value(season))
        if window.empty:
            ax.text(0.5, 0.5, "sem janela completa", ha="center", va="center")
            ax.set_axis_off()
            continue
        ax.fill_between(
            window["timestamp"],
            window["lower_90"].astype(float),
            window["upper_90"].astype(float),
            color=_PROB_COLOR,
            alpha=0.16,
            label=labels["interval"],
        )
        ax.plot(
            window["timestamp"],
            window["y_true"],
            color="#222222",
            linewidth=1.5,
            label=labels["observed"],
        )
        ax.plot(
            window["timestamp"],
            window["e0_prediction"],
            color=_CHAMPION_COLOR,
            linewidth=1.4,
            label=labels["e0"],
        )
        ax.plot(
            window["timestamp"],
            window["demand_median"],
            color=_CANDIDATE_COLOR,
            linewidth=1.2,
            label=labels["median"],
        )
        ax.tick_params(axis="x", rotation=25)
        ax.grid(alpha=0.22)
    axes[0].set_ylabel(labels["demand"])
    axes[2].set_ylabel(labels["demand"])
    handles, legend_labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, legend_labels, loc="lower center", ncol=4)
    fig.suptitle(labels["title"])
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    return fig


def segment_metrics_report(
    results: UncertaintyExperimentResults,
    segment: str = "Rush_Period",
    lang=None,
) -> pd.DataFrame:
    lang = resolve_lang(lang)
    frame = results.segment_metrics.loc[results.segment_metrics["segment"].eq(segment)].copy()
    return localize_table(
        frame,
        lang,
        {
            "experiment_id": "Experimento",
            "segment": "Segmento",
            "segment_value": "Valor",
            "n": "Observacoes",
            "mae": "MAE",
            "r2": "R2",
            "mean_bias": "Bias medio",
            "coverage_90": "Cobertura 90%",
            "mean_width_90": "Largura media 90%",
        },
    )


def plot_segment_coverage(results: UncertaintyExperimentResults, lang=None):
    """Plot E4 90% coverage error by operational segment."""
    lang = resolve_lang(lang)
    labels = lang(
        {
            "title": "Erro de cobertura de 90% do E4 por segmento",
            "x": "Cobertura observada - 90%",
            "Seasons": "Estacao",
            "Rush_Period": "Periodo de rush",
            "Rainfall Cat": "Chuva",
            "predicted_demand_decile": "Decil da demanda prevista",
        }
    )
    wanted = ["Seasons", "Rush_Period", "Rainfall Cat", "predicted_demand_decile"]
    frame = results.segment_metrics.loc[
        results.segment_metrics["experiment_id"].eq("E4")
        & results.segment_metrics["segment"].isin(wanted)
        & results.segment_metrics["coverage_90"].notna()
    ].copy()
    frame["coverage_error"] = frame["coverage_90"].astype(float) - 0.90
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    axes = axes.ravel()
    for ax, segment in zip(axes, wanted):
        segment_frame = frame.loc[frame["segment"].eq(segment)].copy()
        if segment_frame.empty:
            ax.text(0.5, 0.5, "sem dados", ha="center", va="center")
            ax.set_axis_off()
            continue
        segment_frame["label"] = segment_frame["segment_value"].map(_localized_value)
        segment_frame = segment_frame.sort_values("coverage_error")
        colors = np.where(segment_frame["coverage_error"] < 0, "#e45756", _PROB_COLOR)
        ax.barh(segment_frame["label"], segment_frame["coverage_error"], color=colors)
        ax.axvline(0.0, color=_BASELINE_COLOR, linestyle="--")
        ax.set_title(labels[segment])
        ax.set_xlabel(labels["x"])
        ax.grid(axis="x", alpha=0.22)
    fig.suptitle(labels["title"])
    fig.tight_layout()
    return fig


def friday_18_report(results: UncertaintyExperimentResults, lang=None) -> pd.DataFrame:
    lang = resolve_lang(lang)
    return localize_table(
        results.friday_18_metrics,
        lang,
        {
            "experiment_id": "Experimento",
            "segment": "Segmento",
            "n": "Observacoes",
            "mae": "MAE",
            "rmse": "RMSE",
            "r2": "R2",
            "mean_bias": "Bias medio",
        },
    )


def ablation_report(results: UncertaintyExperimentResults, lang=None) -> pd.DataFrame:
    lang = resolve_lang(lang)
    metrics = results.aggregate_metrics.set_index("experiment_id")
    rows = []
    for experiment_id, baseline in (("E1", "E0"), ("E2", "E0"), ("E3", "E0"), ("E4", "E3")):
        if experiment_id not in metrics.index or baseline not in metrics.index:
            continue
        rows.append(
            {
                "experiment_id": experiment_id,
                "reference": baseline,
                "delta_mae_weighted": float(
                    metrics.loc[experiment_id, "cv_mae_weighted"]
                    - metrics.loc[baseline, "cv_mae_weighted"]
                ),
                "delta_r2_weighted": float(
                    metrics.loc[experiment_id, "cv_r2_weighted"]
                    - metrics.loc[baseline, "cv_r2_weighted"]
                ),
            }
        )
    return localize_table(
        pd.DataFrame(rows),
        lang,
        {
            "experiment_id": "Experimento",
            "reference": "Referencia",
            "delta_mae_weighted": "Delta MAE ponderado",
            "delta_r2_weighted": "Delta R2 ponderado",
        },
    )


def successor_message(results: UncertaintyExperimentResults, lang=None) -> str:
    lang = resolve_lang(lang)
    if results.aggregate_metrics.empty:
        return lang({"m": "Nenhum resultado experimental foi produzido."})["m"]
    best = results.aggregate_metrics.sort_values("cv_mae_weighted").iloc[0]
    if results.is_smoke:
        return lang(
            {
                "m": (
                    "O modo smoke validou a infraestrutura, mas nao define ranking. "
                    f"O menor MAE ponderado provisório apareceu em {best['experiment_id']}."
                )
            }
        )["m"]
    if best["experiment_id"] == "E0":
        return lang(
            {
                "m": (
                    "Nenhum sucessor pontual foi identificado. O Champion E0 foi "
                    "preservado, enquanto o E4 permaneceu apenas como candidato "
                    "experimental para modelagem da incerteza, ainda dependente de "
                    "calibracao adicional."
                )
            }
        )["m"]
    return lang(
        {
            "m": (
                f"{best['experiment_id']} pode ser congelado como candidato sucessor "
                "experimental, pendente de confirmacao em nova janela independente."
            )
        }
    )["m"]


def artifact_report(results: UncertaintyExperimentResults, lang=None) -> pd.DataFrame:
    lang = resolve_lang(lang)
    frame = pd.DataFrame(
        [
            {"artifact": name, "path": public_path(path)}
            for name, path in sorted(results.artifacts.items())
        ]
    )
    return localize_table(frame, lang, {"artifact": "Artefato", "path": "Caminho"})


def synthesis_report(results: UncertaintyExperimentResults, lang=None) -> str:
    lang = resolve_lang(lang)
    message = successor_message(results, lang=lang)
    metrics = results.aggregate_metrics.set_index("experiment_id")
    e0 = metrics.loc["E0"] if "E0" in metrics.index else None
    e4 = metrics.loc["E4"] if "E4" in metrics.index else None
    intervals = results.interval_metrics
    e4_90 = intervals.loc[
        intervals["experiment_id"].eq("E4") & np.isclose(intervals["coverage"], 0.90)
    ]
    if e0 is not None:
        point_text = (
            f"O E0 permaneceu como Champion pontual, com MAE ponderado "
            f"{e0['cv_mae_weighted']:,.3f}, R2 ponderado {e0['cv_r2_weighted']:.3f} "
            f"e R2 medio {e0['cv_r2_mean']:.3f}."
        )
    else:
        point_text = "O Champion pontual nao foi encontrado na tabela agregada."
    if e4 is not None and not e4_90.empty:
        row = e4_90.iloc[0]
        uncertainty_text = (
            "O E4 nao alterou a previsao pontual de E3. Sua contribuicao ficou "
            f"restrita a incerteza, com cobertura de 90% em "
            f"{row['empirical_coverage']:.3f}, largura media "
            f"{row['mean_width']:,.1f} e Winkler {row['winkler_score']:,.1f}."
        )
    else:
        uncertainty_text = "A camada probabilistica E4 nao foi encontrada na tabela intervalar."
    return lang(
        {
            "m": (
                f"{point_text} E1, E2 e E3 nao melhoraram o MAE ou o R2 sob os "
                f"folds normais de selecao. {uncertainty_text} A persistencia "
                "residual nao foi removida; portanto, a calibracao conformal "
                "temporal sobre E0 e/ou E4 foi registrada como proxima hipotese "
                "experimental para um futuro Notebook 07. O holdout nao foi "
                f"reaberto nesta etapa. {message}"
            )
        }
    )["m"]
