"""Presentation layer for notebook 07 temporal conformal calibration."""

from __future__ import annotations

from typing import Any, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.conformal_uncertainty import ConformalCalibrationResults
from src.i18n import localize_table, make_lang, resolve_lang as _project_resolve_lang
from src.model_selection_reports import environment_report as _environment_report
from src.utils import public_path


def _key_value_frame(lang, rows: List[tuple]) -> pd.DataFrame:
    labels = lang({"item": "Item", "value": "Valor"})
    return pd.DataFrame([{labels["item"]: label, labels["value"]: value} for label, value in rows])


def resolve_lang(lang):
    """Accept the project LangMap and the short language code used by tests."""
    if isinstance(lang, str):
        return make_lang(lang)
    return _project_resolve_lang(lang)


def _coverage_mask(frame: pd.DataFrame, coverage: float) -> pd.Series:
    return pd.Series(np.isclose(frame["coverage"], coverage), index=frame.index)


def _candidate_colors(candidate_ids: Iterable[str]) -> dict:
    identifiers = list(dict.fromkeys(candidate_ids))
    palette = plt.get_cmap("tab20")(np.linspace(0.0, 0.95, max(len(identifiers), 1)))
    return dict(zip(identifiers, palette))


def environment_report(lang=None) -> pd.DataFrame:
    return _environment_report(lang=lang)


def protocol_report(config_or_results: Any, lang=None) -> pd.DataFrame:
    lang = resolve_lang(lang)
    config = getattr(config_or_results, "config", config_or_results)
    labels = lang(
        {
            "mode": "Modo de execução",
            "claim": "Natureza da conclusão",
            "holdout": "Uso do holdout final",
            "source": "Fonte das previsões",
            "point": "Centro dos intervalos",
            "warmup": "Warm-up por fold",
            "reset": "Reinicialização do estado",
            "coverages": "Coberturas avaliadas",
            "stress": "Política para 2020",
            "bootstrap": "Bootstrap temporal",
        }
    )
    return _key_value_frame(
        lang,
        [
            (labels["mode"], config.run_mode),
            (labels["claim"], "candidato experimental para a camada de incerteza"),
            (labels["holdout"], "não utilizado"),
            (labels["source"], "previsões OOF congeladas do Notebook 06"),
            (labels["point"], config.point_experiment_id),
            (labels["warmup"], f"{config.warmup_hours} horas"),
            (labels["reset"], "a cada fold e mudança de regime"),
            (labels["coverages"], ", ".join(f"{x:.0%}" for x in config.interval_coverages)),
            (labels["stress"], "relatório separado, fora do ranking"),
            (
                labels["bootstrap"],
                f"{config.bootstrap_repetitions} repetições; blocos de "
                f"{config.bootstrap_block_hours} horas",
            ),
        ],
    )


def source_provenance_report(results: ConformalCalibrationResults, lang=None) -> pd.DataFrame:
    lang = resolve_lang(lang)
    manifest = results.source_manifest
    labels = lang(
        {
            "version": "Versão do código de origem",
            "mode": "Modo do artefato de origem",
            "dataset": "Fingerprint do dataset",
            "regime": "Fingerprint do regime",
            "cv": "Validação temporal de origem",
        }
    )
    return _key_value_frame(
        lang,
        [
            (labels["version"], manifest.get("code_version")),
            (labels["mode"], manifest.get("run_mode")),
            (labels["dataset"], manifest.get("dataset_fingerprint")),
            (labels["regime"], manifest.get("regime_fingerprint")),
            (labels["cv"], manifest.get("cv_strategy_version")),
        ],
    )


def input_audit_report(results: ConformalCalibrationResults, lang=None) -> pd.DataFrame:
    lang = resolve_lang(lang)
    frame = results.input_audit.copy()
    frame["path"] = frame["path"].map(public_path)
    return localize_table(
        frame,
        lang,
        {
            "artifact": "Artefato",
            "path": "Caminho",
            "bytes": "Bytes",
            "sha256": "SHA-256",
        },
    )


def method_spec_report(results_or_specs: Any, lang=None) -> pd.DataFrame:
    lang = resolve_lang(lang)
    specs = getattr(results_or_specs, "specs", results_or_specs)
    frame = pd.DataFrame([spec.__dict__ for spec in specs])
    return localize_table(
        frame,
        lang,
        {
            "candidate_id": "Candidato",
            "method_id": "Método",
            "label": "Especificação",
            "calibration_window": "Janela de calibração",
            "nonconformity": "Não conformidade",
            "adaptive_alpha": "Alpha adaptativo",
            "uses_e4_scale": "Usa escala E4",
            "status": "Status",
            "notes": "Notas",
        },
    )


def warmup_report(results: ConformalCalibrationResults, lang=None) -> pd.DataFrame:
    lang = resolve_lang(lang)
    frame = results.fold_metrics[
        [
            "candidate_id",
            "coverage",
            "fold",
            "test_year",
            "fold_role",
            "n_total",
            "n_warmup",
            "n_scored",
            "maximum_calibration_size",
        ]
    ].copy()
    return localize_table(
        frame,
        lang,
        {
            "candidate_id": "Candidato",
            "coverage": "Cobertura nominal",
            "fold": "Fold",
            "test_year": "Ano meteorológico",
            "fold_role": "Papel do fold",
            "n_total": "Observações totais",
            "n_warmup": "Observações em warm-up",
            "n_scored": "Observações pontuadas",
            "maximum_calibration_size": "Histórico máximo",
        },
    )


def aggregate_metrics_report(results: ConformalCalibrationResults, lang=None) -> pd.DataFrame:
    lang = resolve_lang(lang)
    frame = results.aggregate_metrics.copy()
    columns = [
        "candidate_id",
        "method_id",
        "coverage",
        "n_folds_calibrated",
        "empirical_coverage_weighted",
        "coverage_error_weighted",
        "mean_width_weighted",
        "winkler_score_weighted",
        "lower_miss_rate_weighted",
        "upper_miss_rate_weighted",
        "fallback_rate_weighted",
    ]
    return localize_table(
        frame[columns],
        lang,
        {
            "candidate_id": "Candidato",
            "method_id": "Método",
            "coverage": "Cobertura nominal",
            "n_folds_calibrated": "Folds calibrados",
            "empirical_coverage_weighted": "Cobertura observada",
            "coverage_error_weighted": "Erro de cobertura",
            "mean_width_weighted": "Largura média",
            "winkler_score_weighted": "Winkler score",
            "lower_miss_rate_weighted": "Violações inferiores",
            "upper_miss_rate_weighted": "Violações superiores",
            "fallback_rate_weighted": "Taxa de fallback",
        },
    )


def fold_metrics_report(results: ConformalCalibrationResults, lang=None) -> pd.DataFrame:
    lang = resolve_lang(lang)
    frame = results.fold_metrics.loc[results.fold_metrics["fold_role"].eq("selection")].copy()
    columns = [
        "candidate_id",
        "coverage",
        "fold",
        "test_year",
        "n_scored",
        "empirical_coverage",
        "coverage_error",
        "mean_width",
        "winkler_score",
        "fallback_rate",
        "worst_miss_streak",
    ]
    return localize_table(
        frame[columns],
        lang,
        {
            "candidate_id": "Candidato",
            "coverage": "Cobertura nominal",
            "fold": "Fold",
            "test_year": "Ano meteorológico",
            "n_scored": "Observações pontuadas",
            "empirical_coverage": "Cobertura observada",
            "coverage_error": "Erro de cobertura",
            "mean_width": "Largura média",
            "winkler_score": "Winkler score",
            "fallback_rate": "Taxa de fallback",
            "worst_miss_streak": "Maior sequência descoberta",
        },
    )


def stress_metrics_report(results: ConformalCalibrationResults, lang=None) -> pd.DataFrame:
    lang = resolve_lang(lang)
    frame = results.stress_metrics.copy()
    columns = [
        "candidate_id",
        "coverage",
        "test_year",
        "n_scored",
        "empirical_coverage",
        "coverage_error",
        "mean_width",
        "winkler_score",
    ]
    return localize_table(
        frame[columns] if not frame.empty else pd.DataFrame(columns=columns),
        lang,
        {
            "candidate_id": "Candidato",
            "coverage": "Cobertura nominal",
            "test_year": "Ano de stress",
            "n_scored": "Observações pontuadas",
            "empirical_coverage": "Cobertura de stress",
            "coverage_error": "Erro de cobertura",
            "mean_width": "Largura média",
            "winkler_score": "Winkler score",
        },
    )


def scale_diagnostics_report(results: ConformalCalibrationResults, lang=None) -> pd.DataFrame:
    lang = resolve_lang(lang)
    return localize_table(
        results.scale_diagnostics,
        lang,
        {
            "scope": "Escopo",
            "scale_decile": "Decil da escala",
            "n": "Observações",
            "mean_predicted_scale": "Escala média prevista",
            "mean_absolute_error": "Erro absoluto médio E0",
            "spearman_scale_abs_error": "Correlação de Spearman",
            "monotonic_growth_rate": "Taxa de crescimento monotônico",
        },
    )


def sensitivity_report(results: ConformalCalibrationResults, method_id: str, lang=None):
    lang = resolve_lang(lang)
    frame = results.aggregate_metrics.loc[
        results.aggregate_metrics["method_id"].eq(method_id)
        & _coverage_mask(results.aggregate_metrics, results.config.primary_coverage)
    ].copy()
    return localize_table(
        frame[
            [
                "candidate_id",
                "empirical_coverage_weighted",
                "coverage_error_weighted",
                "mean_width_weighted",
                "winkler_score_weighted",
            ]
        ],
        lang,
        {
            "candidate_id": "Variante",
            "empirical_coverage_weighted": "Cobertura observada",
            "coverage_error_weighted": "Erro de cobertura",
            "mean_width_weighted": "Largura média",
            "winkler_score_weighted": "Winkler score",
        },
    )


def segment_metrics_report(
    results: ConformalCalibrationResults, alerts_only: bool = False, lang=None
) -> pd.DataFrame:
    lang = resolve_lang(lang)
    frame = results.segment_metrics.copy()
    if alerts_only:
        frame = frame.loc[frame["segment_alert"]]
    return localize_table(
        frame,
        lang,
        {
            "candidate_id": "Candidato",
            "segment_type": "Tipo de segmento",
            "segment_value": "Segmento",
            "n": "Observações",
            "empirical_coverage": "Cobertura observada",
            "coverage_error": "Erro de cobertura",
            "mean_width": "Largura média",
            "winkler_score": "Winkler score",
            "fallback_rate": "Taxa de fallback",
            "segment_alert": "Alerta de subcobertura",
        },
    )


def bootstrap_report(results: ConformalCalibrationResults, lang=None) -> pd.DataFrame:
    lang = resolve_lang(lang)
    return localize_table(
        results.bootstrap_metrics,
        lang,
        {
            "candidate_id": "Candidato",
            "coverage": "Cobertura nominal",
            "observed_coverage": "Cobertura observada",
            "ci_lower": "IC 95% inferior",
            "ci_upper": "IC 95% superior",
            "bootstrap_repetitions": "Repetições",
            "block_hours": "Tamanho do bloco (horas)",
        },
    )


def decision_report(results: ConformalCalibrationResults, lang=None) -> pd.DataFrame:
    lang = resolve_lang(lang)
    return localize_table(
        results.decision_table,
        lang,
        {
            "candidate_id": "Candidato",
            "method_id": "Método",
            "empirical_coverage": "Cobertura observada",
            "coverage_error": "Erro de cobertura",
            "worst_fold_coverage": "Pior cobertura por fold",
            "winkler_score": "Winkler score",
            "mean_width": "Largura média",
            "coverage_std_between_folds": "Desvio entre folds",
            "fallback_rate": "Taxa de fallback",
            "segment_alerts": "Alertas de segmento",
            "coverage_gate": "Gate de cobertura",
            "experimental_rank": "Ranking experimental",
        },
    )


def plot_coverage_calibration(results: ConformalCalibrationResults, lang=None):
    lang = resolve_lang(lang)
    labels = lang(
        {
            "title": "Cobertura nominal versus observada",
            "x": "Cobertura nominal",
            "y": "Cobertura observada",
        }
    )
    frame = results.aggregate_metrics.copy()
    colors = _candidate_colors(frame["candidate_id"])
    fig, ax = plt.subplots(figsize=(9, 6))
    for candidate_id, group in frame.groupby("candidate_id", sort=False):
        group = group.sort_values("coverage")
        ax.plot(
            group["coverage"],
            group["empirical_coverage_weighted"],
            marker=".",
            linewidth=1.0,
            label=candidate_id,
            color=colors[candidate_id],
        )
    ax.plot([0.75, 1.0], [0.75, 1.0], color="black", linestyle="--", linewidth=1)
    ax.set(title=labels["title"], xlabel=labels["x"], ylabel=labels["y"])
    ax.legend(ncol=3, fontsize=8)
    ax.grid(alpha=0.2)
    return fig


def plot_fold_coverage_heatmap(results: ConformalCalibrationResults, lang=None):
    lang = resolve_lang(lang)
    labels = lang(
        {
            "title": "Erro de cobertura por candidato e fold normal",
            "x": "Ano meteorológico",
            "y": "Candidato",
        }
    )
    frame = results.fold_metrics.loc[
        results.fold_metrics["fold_role"].eq("selection")
        & _coverage_mask(results.fold_metrics, results.config.primary_coverage)
    ]
    pivot = frame.pivot(index="candidate_id", columns="test_year", values="coverage_error")
    fig, ax = plt.subplots(figsize=(9, max(4, len(pivot) * 0.38)))
    image = ax.imshow(pivot.to_numpy(), cmap="RdBu", vmin=-0.2, vmax=0.2, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    for row_index in range(len(pivot.index)):
        for column_index in range(len(pivot.columns)):
            value = pivot.iloc[row_index, column_index]
            if np.isfinite(value):
                ax.text(column_index, row_index, f"{value:+.1%}", ha="center", va="center")
    ax.set(title=labels["title"], xlabel=labels["x"], ylabel=labels["y"])
    fig.colorbar(image, ax=ax, label="erro de cobertura")
    fig.tight_layout()
    return fig


def plot_coverage_width_pareto(results: ConformalCalibrationResults, lang=None):
    lang = resolve_lang(lang)
    labels = lang(
        {
            "title": "Fronteira entre calibração e largura",
            "x": "Erro absoluto de cobertura",
            "y": "Largura média",
        }
    )
    frame = results.aggregate_metrics.loc[
        _coverage_mask(results.aggregate_metrics, results.config.primary_coverage)
    ].copy()
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = _candidate_colors(frame["candidate_id"])
    for row in frame.itertuples(index=False):
        ax.scatter(
            abs(row.coverage_error_weighted),
            row.mean_width_weighted,
            s=30,
            color=colors[row.candidate_id],
        )
        ax.annotate(
            row.candidate_id,
            (abs(row.coverage_error_weighted), row.mean_width_weighted),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axvline(results.config.coverage_tolerance, color="black", linestyle="--", linewidth=1)
    ax.set(title=labels["title"], xlabel=labels["x"], ylabel=labels["y"])
    ax.grid(alpha=0.2)
    return fig


def plot_scale_diagnostics(results: ConformalCalibrationResults, lang=None):
    lang = resolve_lang(lang)
    labels = lang(
        {
            "title": "Erro do E0 por decil da escala prevista pelo E4",
            "x": "Decil da escala",
            "y": "Erro absoluto médio",
        }
    )
    frame = results.scale_diagnostics.loc[results.scale_diagnostics["scope"].eq("overall")]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(frame["scale_decile"], frame["mean_absolute_error"], marker=".", linewidth=1.0)
    ax.set(title=labels["title"], xlabel=labels["x"], ylabel=labels["y"])
    ax.grid(alpha=0.2)
    return fig


def plot_rolling_coverage(results: ConformalCalibrationResults, lang=None):
    lang = resolve_lang(lang)
    labels = lang(
        {
            "title": "Cobertura móvel no fold normal mais recente",
            "x": "Tempo",
            "y": "Cobertura móvel",
        }
    )
    frame = results.rolling_coverage.loc[results.rolling_coverage["fold_role"].eq("selection")]
    if frame.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.text(0.5, 0.5, "histórico insuficiente para cobertura móvel", ha="center")
        ax.set_axis_off()
        return fig
    latest_year = int(frame["test_year"].max())
    frame = frame.loc[frame["test_year"].eq(latest_year)]
    preferred_window = 720 if frame["window_hours"].eq(720).any() else 168
    frame = frame.loc[frame["window_hours"].eq(preferred_window)]
    colors = _candidate_colors(frame["candidate_id"])
    fig, ax = plt.subplots(figsize=(12, 5))
    for candidate_id, group in frame.groupby("candidate_id", sort=False):
        ax.plot(
            group["timestamp"],
            group["rolling_coverage"],
            label=candidate_id,
            color=colors[candidate_id],
            linewidth=1,
        )
    ax.axhline(results.config.primary_coverage, color="black", linestyle="--")
    ax.set(title=labels["title"], xlabel=labels["x"], ylabel=labels["y"], ylim=(0, 1.02))
    ax.legend(ncol=3, fontsize=8)
    ax.grid(alpha=0.2)
    return fig


def plot_aci_alpha_trajectory(results: ConformalCalibrationResults, lang=None):
    lang = resolve_lang(lang)
    labels = lang(
        {
            "title": "Trajetória prequential do alpha adaptativo",
            "x": "Tempo",
            "y": "Alpha utilizado",
        }
    )
    trace = results.aci_alpha_trace.loc[
        np.isclose(results.aci_alpha_trace["coverage"], results.config.primary_coverage)
        & results.aci_alpha_trace["fold_role"].eq("selection")
    ]
    latest_year = int(trace["test_year"].max())
    trace = trace.loc[trace["test_year"].eq(latest_year)]
    colors = _candidate_colors(trace["candidate_id"])
    fig, ax = plt.subplots(figsize=(12, 5))
    for candidate_id, group in trace.groupby("candidate_id", sort=False):
        ax.plot(
            group["timestamp"],
            group["alpha_before"],
            label=candidate_id,
            color=colors[candidate_id],
            linewidth=1,
        )
    ax.axhline(1.0 - results.config.primary_coverage, color="black", linestyle="--")
    ax.set(title=labels["title"], xlabel=labels["x"], ylabel=labels["y"])
    ax.legend(ncol=3, fontsize=8)
    ax.grid(alpha=0.2)
    return fig


def plot_segment_coverage(results: ConformalCalibrationResults, lang=None):
    lang = resolve_lang(lang)
    labels = lang(
        {
            "title": "Cobertura por regime operacional",
            "x": "Cobertura observada",
            "y": "Segmento",
        }
    )
    frame = results.segment_metrics.loc[
        results.segment_metrics["segment_type"].isin(["Rush_Period", "Rainfall Cat"])
    ].copy()
    top_candidates = results.decision_table.head(4)["candidate_id"]
    frame = frame.loc[frame["candidate_id"].isin(top_candidates)]
    frame["label"] = frame["segment_type"] + ": " + frame["segment_value"]
    pivot = frame.pivot_table(
        index="label", columns="candidate_id", values="empirical_coverage", aggfunc="first"
    )
    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot) * 0.38)))
    image = ax.imshow(pivot.to_numpy(), cmap="RdYlGn", vmin=0.6, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=25)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set(title=labels["title"], xlabel=labels["x"], ylabel=labels["y"])
    fig.colorbar(image, ax=ax, label="cobertura")
    fig.tight_layout()
    return fig


def _representative_window(results: ConformalCalibrationResults) -> pd.DataFrame:
    frame = results.predictions.loc[
        results.predictions["fold_role"].eq("selection")
        & _coverage_mask(results.predictions, results.config.primary_coverage)
        & results.predictions["interval_available"]
    ].copy()
    latest_year = int(frame["test_year"].max())
    frame = frame.loc[frame["test_year"].eq(latest_year)].sort_values("timestamp")
    start = frame["timestamp"].min()
    return frame.loc[frame["timestamp"] < start + pd.Timedelta(days=7)]


def plot_representative_interval_windows(results: ConformalCalibrationResults, lang=None):
    lang = resolve_lang(lang)
    labels = lang(
        {
            "title": "Semana representativa no fold normal mais recente",
            "observed": "Demanda observada",
            "point": "Previsão E0",
            "interval": "Intervalo",
        }
    )
    window = _representative_window(results)
    preferred = list(results.decision_table.head(4)["candidate_id"])
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True)
    for ax, candidate_id in zip(axes.ravel(), preferred):
        group = window.loc[window["candidate_id"].eq(candidate_id)]
        ax.fill_between(group["timestamp"], group["lower"], group["upper"], alpha=0.2)
        ax.plot(
            group["timestamp"],
            group["y_true"],
            color="black",
            label=labels["observed"],
            linewidth=1.0,
        )
        ax.plot(
            group["timestamp"],
            group["y_pred"],
            color="#f58518",
            label=labels["point"],
            linewidth=1.0,
        )
        ax.set_title(candidate_id)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(alpha=0.2)
    handles, legend_labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=3)
    fig.suptitle(labels["title"])
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    return fig


def calibration_message(results: ConformalCalibrationResults, lang=None) -> str:
    lang = resolve_lang(lang)
    if results.is_smoke:
        return lang(
            {
                "m": (
                    "A execução smoke validou a infraestrutura prequential, a persistência e "
                    "os relatórios. Seus números não definem ranking nem candidato definitivo."
                )
            }
        )["m"]
    feasible = results.decision_table.loc[results.decision_table["coverage_gate"]]
    if feasible.empty:
        return lang(
            {
                "m": (
                    "Nenhuma camada de incerteza foi considerada operacionalmente calibrada. "
                    "Todos os resultados permanecem como hipóteses experimentais."
                )
            }
        )["m"]
    best = feasible.sort_values("experimental_rank").iloc[0]
    return lang(
        {
            "m": (
                f"O candidato {best['candidate_id']} foi o calibrador experimental mais "
                f"defensável: cobertura {best['empirical_coverage']:.1%}, erro "
                f"{best['coverage_error']:+.2%}, largura média {best['mean_width']:,.1f} e "
                f"Winkler score {best['winkler_score']:,.1f}. A conclusão ainda exige uma "
                "nova janela independente antes de qualquer adoção operacional."
            )
        }
    )["m"]


def synthesis_report(results: ConformalCalibrationResults, lang=None) -> str:
    lang = resolve_lang(lang)
    scale = results.scale_diagnostics.loc[results.scale_diagnostics["scope"].eq("overall")]
    spearman = scale["spearman_scale_abs_error"].iloc[0] if not scale.empty else np.nan
    mode_text = (
        "Esta execução foi realizada em modo smoke e não autoriza ranking definitivo. "
        if results.is_smoke
        else "A execução full percorreu todos os folds declarados. "
    )
    return lang(
        {
            "m": (
                f"{mode_text}O Champion pontual E0 foi preservado e somente os intervalos foram "
                "recalibrados. Em cada fold, as primeiras 168 horas foram tratadas como "
                "warm-up; a observação corrente atualizou apenas a previsão seguinte, e o "
                "estado foi reiniciado antes de cada nova janela temporal. O regime de 2020 "
                "foi mantido fora do ranking. "
                f"A escala do E4 apresentou correlação de Spearman {spearman:.3f} com o erro "
                "absoluto do E0, o que quantifica sua utilidade de ordenação sem pressupor "
                "calibração. "
                f"{calibration_message(results, lang=lang)} O holdout selado não foi reaberto, "
                "nenhum estimator foi reajustado e nenhum sucessor pontual foi proposto."
            )
        }
    )["m"]


def artifact_report(results: ConformalCalibrationResults, lang=None) -> pd.DataFrame:
    lang = resolve_lang(lang)
    frame = pd.DataFrame(
        [
            {"artifact": name, "path": public_path(path)}
            for name, path in sorted(results.artifacts.items())
        ]
    )
    return localize_table(frame, lang, {"artifact": "Artefato", "path": "Caminho"})
