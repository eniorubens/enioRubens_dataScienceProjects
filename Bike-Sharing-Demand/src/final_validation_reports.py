"""Reports and charts for the final-holdout validation notebook (05).

Every function takes the objects produced by :mod:`src.final_validation` and
returns something displayable — a localized DataFrame, a matplotlib figure or a
narrative string. The numbers were already computed there; this layer only
shapes them for reading. The localization boundary is respected: internal
column names stay English and stable, and only the returned display copy is
translated through :func:`src.i18n.localize_table`.

Two sign conventions are surfaced deliberately and repeatedly, because they are
opposite and easy to invert: ``bias`` is ``mean(y_pred - y_true)`` (positive
means over-estimation), while the per-row ``residual`` plotted here is
``y_true - y_pred`` (positive means under-estimation).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.environment import TRACKED_PACKAGES, describe_environment, package_versions
from src.final_validation import (
    CONFIRMED,
    FINAL_VALIDATION_CODE_VERSION,
    CandidateHoldoutEvaluation,
    FinalValidationPlan,
    FinalValidationResults,
    ShapCandidateExplanation,
    autocorrelation_function,
    champion_error_profiles,
    champion_residual_diagnostic_frame,
    champion_residual_transformation_diagnostics,
    champion_residual_transformation_frame,
    champion_residual_triage,
    champion_rolling_residual_diagnostics,
    heteroscedasticity_diagnostics,
    partial_autocorrelation_function,
    residual_diagnostics,
)
from src.i18n import localize_table, resolve_lang
from src.temporal_optimizer import CODE_VERSION, CV_STRATEGY_VERSION
from src.utils import public_path

_CHAMPION_COLOR = "#f58518"
_CHALLENGER_COLORS = ["#4c78a8", "#54a24b", "#b279a2"]
_REFERENCE_COLOR = "#9e9e9e"

# Canonical PT labels for the categorical values that appear in segment reports.
_VALUE_LABELS = {
    "champion": "Champion",
    "challenger": "Challenger",
    "Winter": "Inverno",
    "Spring": "Primavera",
    "Summer": "Verão",
    "Autumn": "Outono",
    "Weekday": "Dia útil",
    "Weekend": "Fim de semana",
    "Holiday": "Feriado",
    "No Holiday": "Sem feriado",
    "Yes": "Sim",
    "No": "Não",
    "extreme_cold": "Frio extremo",
    "extreme_hot": "Calor extremo",
    "other": "Demais",
    CONFIRMED: "Champion confirmado",
    "champion_not_confirmed": "Champion não confirmado",
    "raw": "Original",
    "globally_debiased": "Desenviesado globalmente",
    "calendar_demeaned": "Sem padrão semanal médio",
    "level_standardized": "Padronizado por nível",
    True: "Sim",
    False: "Não",
}

_WEEKDAY_LABELS = {
    0: "Segunda",
    1: "Terça",
    2: "Quarta",
    3: "Quinta",
    4: "Sexta",
    5: "Sábado",
    6: "Domingo",
}


def _key_value_frame(lang, rows: List[tuple]) -> pd.DataFrame:
    """Build a two-column label/value report from already-canonical PT labels."""
    labels = lang({"item": "Item", "value": "Valor"})
    return pd.DataFrame([{labels["item"]: label, labels["value"]: value} for label, value in rows])


def _candidate_color(index: int, role: str) -> str:
    """Colour a candidate: the champion in its own hue, challengers cycling."""
    if role == "champion":
        return _CHAMPION_COLOR
    return _CHALLENGER_COLORS[index % len(_CHALLENGER_COLORS)]


def _format_p_value(value: Any) -> Any:
    """Display p-values without turning numerical underflow into a literal zero."""
    if pd.isna(value):
        return value
    number = float(value)
    if number == 0.0:
        return "abaixo da precisão numérica"
    if abs(number) < 0.001:
        return f"{number:.2e}"
    return round(number, 6)


def _format_p_value_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Format p-value columns in a display copy only."""
    display = frame.copy()
    for column in ("p_value", "adjusted_p_value"):
        if column in display.columns:
            display[column] = display[column].map(_format_p_value)
    return display


# ---------------------------------------------------------------------------
# 1. Pre-registered protocol
# ---------------------------------------------------------------------------


def protocol_report(results_or_config: Any, lang=None) -> pd.DataFrame:
    """State the confirmation rule as it was fixed before the holdout was opened."""
    lang = resolve_lang(lang)
    config = getattr(results_or_config, "config", results_or_config)
    labels = lang(
        {
            "champion": "Champion pré-registrado",
            "rule_mae": "Regra de MAE",
            "rule_r2": "Regra de R²",
            "mae_ratio": "Tolerância de MAE (razão)",
            "r2_margin": "Margem de R²",
            "decision_confirmed": "Decisão se ambas satisfeitas",
            "decision_not": "Decisão caso contrário",
            "no_reopen": "Reabertura da busca",
        }
    )
    return _key_value_frame(
        lang,
        [
            (labels["champion"], "CatBoostRegressor"),
            (
                labels["rule_mae"],
                lang({"m": "MAE do champion ≤ 1,05 × menor MAE entre os três"})["m"],
            ),
            (
                labels["rule_r2"],
                lang({"r": "R² do champion ≥ maior R² entre os três − 0,02"})["r"],
            ),
            (labels["mae_ratio"], config.confirm_mae_ratio),
            (labels["r2_margin"], config.confirm_r2_margin),
            (labels["decision_confirmed"], lang({"c": "champion confirmado"})["c"]),
            (labels["decision_not"], lang({"n": "champion não confirmado"})["n"]),
            (
                labels["no_reopen"],
                lang({"x": "não ocorre; uma nova janela temporal seria necessária"})["x"],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# 2. Environment and provenance
# ---------------------------------------------------------------------------


def provenance_report(plan_or_results: Any, lang=None) -> pd.DataFrame:
    """Report the interpreter, library versions and the provenance fingerprints."""
    lang = resolve_lang(lang)
    environment = describe_environment()
    versions = package_versions(TRACKED_PACKAGES)
    manifest = getattr(plan_or_results, "manifest", {})
    labels = lang(
        {
            "environment": "Ambiente",
            "executable": "Interpretador",
            "python": "Versão do Python",
            "env_fp": "Fingerprint do ambiente",
            "dataset_fp": "Fingerprint do dataset",
            "regime_fp": "Fingerprint do regime",
            "cv_version": "Versão da estratégia de CV",
            "selection_code": "Versão do código de seleção",
            "final_code": "Versão do código de validação final",
        }
    )
    rows = [
        (labels["environment"], environment["environment_name"]),
        (
            labels["executable"],
            f"{environment['environment_name']}/{Path(environment['python_executable']).name}",
        ),
        (labels["python"], environment["python_version"]),
        (labels["env_fp"], environment["environment_fingerprint"]),
        (labels["dataset_fp"], manifest.get("dataset_fingerprint")),
        (labels["regime_fp"], manifest.get("regime_fingerprint")),
        (labels["cv_version"], CV_STRATEGY_VERSION),
        (labels["selection_code"], CODE_VERSION),
        (labels["final_code"], FINAL_VALIDATION_CODE_VERSION),
    ]
    rows.extend((f"{name}", version) for name, version in versions.items())
    return _key_value_frame(lang, rows)


# ---------------------------------------------------------------------------
# 3. Frozen candidates
# ---------------------------------------------------------------------------


def candidates_report(plan: FinalValidationPlan, lang=None) -> pd.DataFrame:
    """Describe the three verified frozen candidates and their CV standing."""
    lang = resolve_lang(lang)
    rows = []
    for candidate in plan.candidates:
        rows.append(
            {
                "role": candidate.role,
                "estimator": candidate.estimator,
                "run_id": candidate.run_id,
                "cv_mae_mean": candidate.cv_metrics.get("cv_mae_mean"),
                "cv_mae_weighted": candidate.cv_metrics.get("cv_mae_weighted"),
                "cv_r2_mean": candidate.cv_metrics.get("cv_r2_mean"),
                "artifact_sha256": candidate.artifact_sha256[:12],
                "provenance_verified": True,
            }
        )
    frame = pd.DataFrame(rows)
    return localize_table(
        frame,
        lang,
        columns={
            "role": "Papel",
            "estimator": "Estimator",
            "run_id": "Run ID de origem",
            "cv_mae_mean": "MAE médio (CV)",
            "cv_mae_weighted": "MAE ponderado (CV)",
            "cv_r2_mean": "R² médio (CV)",
            "artifact_sha256": "SHA-256 do artefato",
            "provenance_verified": "Proveniência confirmada",
        },
        value_columns=["role"],
        value_labels=_VALUE_LABELS,
    )


# ---------------------------------------------------------------------------
# 4. Holdout seal
# ---------------------------------------------------------------------------


def holdout_seal_report(results: FinalValidationResults, lang=None) -> pd.DataFrame:
    """Summarise the sealed holdout window, the development split and the discard."""
    lang = resolve_lang(lang)
    manifest = results.manifest
    data = results.data
    labels = lang(
        {
            "holdout_window": "Janela do holdout",
            "holdout_rows": "Observações no holdout",
            "dev_window": "Janela de desenvolvimento",
            "dev_rows": "Observações de desenvolvimento",
            "post_rows": "Horas de dez/2024 descartadas",
            "dataset_fp": "Fingerprint do dataset",
            "holdout_fp": "Fingerprint do holdout",
            "regime_fp": "Fingerprint do regime",
            "source": "Origem dos resultados",
        }
    )
    if data is not None:
        window = f"{data.holdout_start.date()} — {data.holdout_end.date()}"
        holdout_rows = data.n_holdout_rows
        dev_window = f"{data.dev_start} — {data.dev_end}"
        dev_rows = data.n_dev_rows
        post_rows = data.n_post_holdout_rows
        dataset_fp = data.dataset_fingerprint
        holdout_fp = data.holdout_fingerprint
        regime_fp = data.regime_fingerprint
    else:
        holdout = manifest.get("holdout", {})
        window = f"{holdout.get('start')} — {holdout.get('end')}"
        holdout_rows = holdout.get("n_rows")
        dev_window = lang({"c": "carregado do cache"})["c"]
        dev_rows = None
        post_rows = None
        dataset_fp = manifest.get("dataset_fingerprint")
        holdout_fp = None
        regime_fp = manifest.get("regime_fingerprint")
    source = (
        lang({"c": "resultados reutilizados do cache"})["c"]
        if results.loaded_from_cache
        else lang({"f": "holdout aberto uma única vez"})["f"]
    )
    return _key_value_frame(
        lang,
        [
            (labels["holdout_window"], window),
            (labels["holdout_rows"], holdout_rows),
            (labels["dev_window"], dev_window),
            (labels["dev_rows"], dev_rows),
            (labels["post_rows"], post_rows),
            (labels["dataset_fp"], dataset_fp),
            (labels["holdout_fp"], holdout_fp),
            (labels["regime_fp"], regime_fp),
            (labels["source"], source),
        ],
    )


# ---------------------------------------------------------------------------
# 5. Final metrics and comparison
# ---------------------------------------------------------------------------


def metrics_report(results: FinalValidationResults, lang=None) -> pd.DataFrame:
    """Full per-candidate holdout metrics, one row per candidate."""
    lang = resolve_lang(lang)
    rows = []
    for evaluation in results.evaluations:
        row = {"role": evaluation.role, "estimator": evaluation.estimator}
        row.update(evaluation.metrics)
        rows.append(row)
    frame = pd.DataFrame(rows)
    columns = {
        "role": "Papel",
        "estimator": "Estimator",
        "holdout_mae": "MAE",
        "holdout_rmse": "RMSE",
        "holdout_r2": "R²",
        "holdout_wape": "WAPE",
        "holdout_median_abs_error": "Erro absoluto mediano",
        "holdout_mean_bias": "Bias médio (ŷ−y)",
        "holdout_mean_abs_residual": "Média do |resíduo|",
        "holdout_abs_error_q90": "|Erro| q90",
        "holdout_abs_error_q95": "|Erro| q95",
        "holdout_abs_error_q99": "|Erro| q99",
        "mae_holdout_minus_cv": "MAE holdout − CV",
        "r2_holdout_minus_cv": "R² holdout − CV",
    }
    keep = [column for column in columns if column in frame.columns]
    return localize_table(
        frame[keep], lang, columns=columns, value_columns=["role"], value_labels=_VALUE_LABELS
    )


def comparison_report(results: FinalValidationResults, lang=None) -> pd.DataFrame:
    """Headline holdout-versus-CV comparison, sorted by ascending holdout MAE."""
    lang = resolve_lang(lang)
    frame = results.comparison.sort_values("holdout_mae").reset_index(drop=True)
    return localize_table(
        frame,
        lang,
        columns={
            "role": "Papel",
            "estimator": "Estimator",
            "run_id": "Run ID",
            "holdout_mae": "MAE (holdout)",
            "holdout_rmse": "RMSE (holdout)",
            "holdout_r2": "R² (holdout)",
            "holdout_wape": "WAPE (holdout)",
            "holdout_median_abs_error": "Erro absoluto mediano",
            "holdout_mean_bias": "Bias médio (ŷ−y)",
            "cv_mae_mean": "MAE médio (CV)",
            "cv_rmse_mean": "RMSE médio (CV)",
            "cv_r2_mean": "R² médio (CV)",
            "mae_holdout_minus_cv": "MAE holdout − CV",
        },
        value_columns=["role"],
        value_labels=_VALUE_LABELS,
    )


def plot_comparison(results: FinalValidationResults, lang=None) -> plt.Figure:
    """Bar charts of holdout MAE and R² per candidate, with the CV MAE overlaid."""
    lang = resolve_lang(lang)
    text = lang(
        {
            "title": "Comparação no holdout temporal",
            "mae": "MAE (bicicletas/hora)",
            "r2": "R²",
            "cv": "MAE médio (CV)",
            "estimator": "Estimator",
        }
    )
    evaluations = list(results.evaluations)
    names = [item.estimator for item in evaluations]
    holdout_mae = [item.metrics["holdout_mae"] for item in evaluations]
    holdout_r2 = [item.metrics["holdout_r2"] for item in evaluations]
    cv_mae = [item.cv_metrics.get("cv_mae_mean", np.nan) for item in evaluations]
    colors = [_candidate_color(index, item.role) for index, item in enumerate(evaluations)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    positions = np.arange(len(names))
    axes[0].bar(positions, holdout_mae, color=colors)
    axes[0].scatter(
        positions, cv_mae, color=_REFERENCE_COLOR, marker="D", zorder=3, label=text["cv"]
    )
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(names, rotation=20, ha="right")
    axes[0].set_ylabel(text["mae"])
    axes[0].legend()
    axes[1].bar(positions, holdout_r2, color=colors)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(names, rotation=20, ha="right")
    axes[1].set_ylabel(text["r2"])
    fig.suptitle(text["title"])
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Confirmation
# ---------------------------------------------------------------------------


def confirmation_report(results: FinalValidationResults, lang=None) -> pd.DataFrame:
    """Tabulate the pre-registered decision and the numbers that produced it."""
    lang = resolve_lang(lang)
    confirmation = results.confirmation
    labels = lang(
        {
            "decision": "Decisão",
            "champion_mae": "MAE do champion",
            "champion_r2": "R² do champion",
            "best_mae": "Menor MAE entre os três",
            "best_r2": "Maior R² entre os três",
            "mae_threshold": "Limiar de MAE (1,05 × menor)",
            "r2_threshold": "Limiar de R² (maior − 0,02)",
            "mae_ok": "Condição de MAE satisfeita",
            "r2_ok": "Condição de R² satisfeita",
            "best_holdout": "Melhor desempenho no holdout",
        }
    )
    decision_label = _VALUE_LABELS.get(confirmation["decision"], confirmation["decision"])
    return _key_value_frame(
        lang,
        [
            (labels["decision"], lang({"d": decision_label})["d"]),
            (labels["champion_mae"], round(confirmation["champion_holdout_mae"], 3)),
            (labels["champion_r2"], round(confirmation["champion_holdout_r2"], 4)),
            (labels["best_mae"], round(confirmation["best_holdout_mae"], 3)),
            (labels["best_r2"], round(confirmation["best_holdout_r2"], 4)),
            (labels["mae_threshold"], round(confirmation["mae_threshold"], 3)),
            (labels["r2_threshold"], round(confirmation["r2_threshold"], 4)),
            (labels["mae_ok"], confirmation["mae_condition_met"]),
            (labels["r2_ok"], confirmation["r2_condition_met"]),
            (labels["best_holdout"], confirmation["best_holdout_estimator"]),
        ],
    )


def confirmation_message(results: FinalValidationResults, lang=None) -> str:
    """A value-dependent narrative of the confirmation outcome, produced at run time.

    Written here rather than as a static markdown insight because it depends on
    the real holdout numbers, which do not exist until the notebook is executed.
    """
    lang = resolve_lang(lang)
    confirmation = results.confirmation
    champion = confirmation["champion_estimator"]
    best = confirmation["best_holdout_estimator"]
    mae_gap = confirmation["champion_mae_gap_to_best"]
    if confirmation["decision"] == CONFIRMED:
        template = lang(
            {
                "m": (
                    "O champion pré-registrado {champion} foi confirmado no holdout temporal. "
                    "Ambas as condições pré-registradas foram satisfeitas, e a diferença de MAE "
                    "para o melhor candidato ({best}) é de {gap:.2f} bicicletas/hora, uma "
                    "distância pequena que não caracteriza superioridade substantiva de "
                    "nenhum concorrente."
                )
            }
        )["m"]
    else:
        template = lang(
            {
                "m": (
                    "O champion pré-registrado {champion} não foi confirmado no holdout. "
                    "O melhor desempenho observado no holdout coube a {best}, com diferença de "
                    "MAE de {gap:.2f} bicicletas/hora. O manifesto do notebook 04 não é alterado, "
                    "a busca não é reaberta e nenhum modelo é retreinado; uma nova janela "
                    "temporal independente seria necessária para confirmar uma troca definitiva."
                )
            }
        )["m"]
    return template.format(champion=champion, best=best, gap=abs(mae_gap))


# ---------------------------------------------------------------------------
# 7. Temporal residuals
# ---------------------------------------------------------------------------


def residual_diagnostics_report(results: FinalValidationResults, lang=None) -> pd.DataFrame:
    """Scalar residual diagnostics per candidate — not a binary normality verdict."""
    lang = resolve_lang(lang)
    rows = []
    for evaluation in results.evaluations:
        diagnostics = residual_diagnostics(evaluation.residuals.to_numpy())
        row = {"role": evaluation.role, "estimator": evaluation.estimator}
        row.update(diagnostics)
        rows.append(row)
    frame = pd.DataFrame(rows)
    return localize_table(
        frame,
        lang,
        columns={
            "role": "Papel",
            "estimator": "Estimator",
            "mean": "Média do resíduo",
            "std": "Desvio do resíduo",
            "skew": "Assimetria",
            "kurtosis": "Curtose",
            "durbin_watson": "Durbin–Watson",
            "autocorr_lag_1": "Autocorr. lag 1",
            "autocorr_lag_24": "Autocorr. lag 24",
            "autocorr_lag_168": "Autocorr. lag 168",
        },
        value_columns=["role"],
        value_labels=_VALUE_LABELS,
    )


def heteroscedasticity_report(results: FinalValidationResults, lang=None) -> pd.DataFrame:
    """Formal heteroscedasticity diagnostics, localized for notebook display."""
    lang = resolve_lang(lang)
    frame = _format_p_value_columns(heteroscedasticity_diagnostics(results))
    value_labels = {
        **_VALUE_LABELS,
        "ok": "Aplicavel",
        "not_applicable": "Nao aplicavel",
        True: "Sim",
        False: "Nao",
    }
    return localize_table(
        frame,
        lang,
        columns={
            "role": "Papel",
            "estimator": "Estimator",
            "test": "Teste",
            "null_hypothesis": "Hipotese nula",
            "statistic": "Estatistica",
            "p_value": "p-valor bruto",
            "adjusted_p_value": "p-valor ajustado (Holm)",
            "alpha": "Alfa",
            "evidence_of_heteroscedasticity": "Evidencia de heterocedasticidade",
            "n_observations": "Observacoes",
            "diagnostic_specification": "Especificacao diagnostica",
            "limitations": "Limitacoes",
            "status": "Status",
            "reason": "Motivo",
        },
        value_columns=["role", "status", "evidence_of_heteroscedasticity"],
        value_labels=value_labels,
    )


def _champion_evaluation(results: FinalValidationResults) -> CandidateHoldoutEvaluation:
    """The pre-registered champion's holdout evaluation."""
    return results.champion_evaluation


def plot_temporal_residuals(results: FinalValidationResults, lang=None) -> plt.Figure:
    """Champion residuals over time, rolling error, and observed vs predicted (recent)."""
    lang = resolve_lang(lang)
    text = lang(
        {
            "title": "Resíduos temporais do champion",
            "residual": "Resíduo (y − ŷ)",
            "time": "Tempo",
            "rolling": "Média móvel (168 h)",
            "abs_roll": "Média móvel do |erro| (168 h)",
            "observed": "Observado",
            "predicted": "Previsto",
            "recent": "Janela recente (últimas 336 h)",
            "count": "Bicicletas/hora",
        }
    )
    evaluation = _champion_evaluation(results)
    residuals = evaluation.residuals
    timestamps = residuals.index
    abs_error = residuals.abs()
    predictions = pd.Series(evaluation.predictions, index=timestamps)
    y_true = predictions + residuals

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    axes[0].plot(timestamps, residuals.to_numpy(), color=_CHAMPION_COLOR, linewidth=0.6)
    axes[0].axhline(0.0, color=_REFERENCE_COLOR, linewidth=1)
    axes[0].set_ylabel(text["residual"])
    axes[0].set_title(text["title"])

    axes[1].plot(
        timestamps,
        residuals.rolling(168, min_periods=24).mean().to_numpy(),
        color=_CHAMPION_COLOR,
        label=text["rolling"],
    )
    axes[1].plot(
        timestamps,
        abs_error.rolling(168, min_periods=24).mean().to_numpy(),
        color=_CHALLENGER_COLORS[0],
        label=text["abs_roll"],
    )
    axes[1].axhline(0.0, color=_REFERENCE_COLOR, linewidth=1)
    axes[1].set_ylabel(text["residual"])
    axes[1].legend()

    recent = min(336, len(residuals))
    axes[2].plot(
        timestamps[-recent:],
        y_true.to_numpy()[-recent:],
        color=_REFERENCE_COLOR,
        label=text["observed"],
    )
    axes[2].plot(
        timestamps[-recent:],
        predictions.to_numpy()[-recent:],
        color=_CHAMPION_COLOR,
        label=text["predicted"],
    )
    axes[2].set_ylabel(text["count"])
    axes[2].set_xlabel(text["recent"])
    axes[2].legend()
    fig.tight_layout()
    return fig


def plot_residual_structure(results: FinalValidationResults, lang=None) -> plt.Figure:
    """Champion residual distribution, QQ plot, residual-vs-fitted, ACF, PACF, calibration."""
    lang = resolve_lang(lang)
    text = lang(
        {
            "title": "Estrutura dos resíduos do champion",
            "hist": "Distribuição dos resíduos",
            "residual": "Resíduo (y − ŷ)",
            "qq": "QQ plot dos resíduos",
            "theoretical": "Quantis teóricos",
            "sample": "Quantis amostrais",
            "vs_fitted": "Resíduo vs. previsto",
            "fitted": "Valor previsto",
            "acf": "Autocorrelação (ACF)",
            "pacf": "Autocorrelação parcial (PACF)",
            "lag": "Defasagem (horas)",
            "calibration": "Calibração por decis de previsão",
            "mean_pred": "Previsão média",
            "mean_obs": "Observado médio",
        }
    )
    evaluation = _champion_evaluation(results)
    residuals = evaluation.residuals.to_numpy()
    predictions = np.asarray(evaluation.predictions, dtype=float)
    y_true = predictions + residuals

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes[0, 0].hist(residuals, bins=60, color=_CHAMPION_COLOR)
    axes[0, 0].set_title(text["hist"])
    axes[0, 0].set_xlabel(text["residual"])

    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title(text["qq"])
    axes[0, 1].set_xlabel(text["theoretical"])
    axes[0, 1].set_ylabel(text["sample"])

    axes[0, 2].scatter(predictions, residuals, s=4, alpha=0.3, color=_CHAMPION_COLOR)
    axes[0, 2].axhline(0.0, color=_REFERENCE_COLOR, linewidth=1)
    axes[0, 2].set_title(text["vs_fitted"])
    axes[0, 2].set_xlabel(text["fitted"])
    axes[0, 2].set_ylabel(text["residual"])

    n_lags = 48
    acf = autocorrelation_function(residuals, n_lags)
    pacf = partial_autocorrelation_function(residuals, n_lags)
    axes[1, 0].bar(range(len(acf)), acf, color=_CHALLENGER_COLORS[0])
    axes[1, 0].set_title(text["acf"])
    axes[1, 0].set_xlabel(text["lag"])
    axes[1, 1].bar(range(len(pacf)), pacf, color=_CHALLENGER_COLORS[1])
    axes[1, 1].set_title(text["pacf"])
    axes[1, 1].set_xlabel(text["lag"])

    frame = pd.DataFrame({"pred": predictions, "obs": y_true})
    frame["decile"] = pd.qcut(frame["pred"], q=10, duplicates="drop")
    calibration = frame.groupby("decile", observed=True).mean()
    axes[1, 2].plot(calibration["pred"], calibration["obs"], marker="o", color=_CHAMPION_COLOR)
    lims = [
        float(min(calibration["pred"].min(), calibration["obs"].min())),
        float(max(calibration["pred"].max(), calibration["obs"].max())),
    ]
    axes[1, 2].plot(lims, lims, color=_REFERENCE_COLOR, linestyle="--")
    axes[1, 2].set_title(text["calibration"])
    axes[1, 2].set_xlabel(text["mean_pred"])
    axes[1, 2].set_ylabel(text["mean_obs"])
    fig.suptitle(text["title"])
    fig.tight_layout()
    return fig


def residual_triage_report(results: FinalValidationResults, lang=None) -> pd.DataFrame:
    """Main champion triage: bias, serial dependence, level scale and ARCH."""
    lang = resolve_lang(lang)
    frame = champion_residual_triage(results)
    return localize_table(
        frame,
        lang,
        columns={
            "estimator": "Estimator",
            "arch_lag": "Lag ARCH",
            "mae": "MAE",
            "bias_mean": "Bias médio (ŷ-y)",
            "bias_abs_to_mae_ratio": "|bias| / MAE",
            "durbin_watson": "Durbin-Watson",
            "autocorr_lag_1": "Autocorr. lag 1",
            "autocorr_lag_24": "Autocorr. lag 24",
            "autocorr_lag_168": "Autocorr. lag 168",
            "predicted_level_residual_std_ratio": "Razão desvio residual por nível previsto",
            "predicted_level_mae_ratio": "Razão MAE por nível previsto",
            "raw_arch_statistic": "ARCH original",
            "standardized_arch_statistic": "ARCH padronizado",
            "arch_statistic_reduction_after_standardization": "Redução ARCH após padronização",
            "raw_arch_statistic_per_observation": "ARCH/n original",
            "standardized_arch_statistic_per_observation": "ARCH/n padronizado",
            "evidence_of_arch_after_standardization": "ARCH persiste após padronização",
            "diagnostic_only": "Uso diagnóstico",
        },
        value_columns=["evidence_of_arch_after_standardization", "diagnostic_only"],
        value_labels=_VALUE_LABELS,
    )


def residual_profile_report(
    results: FinalValidationResults,
    view: str = "month",
    lang=None,
) -> pd.DataFrame:
    """Champion segmented residual profile for one view."""
    lang = resolve_lang(lang)
    profiles = champion_error_profiles(results)
    if view not in profiles:
        valid = ", ".join(sorted(profiles))
        raise ValueError(f"unknown residual profile view {view!r}; expected one of {valid}")
    frame = profiles[view].copy()
    if view == "weekday":
        frame["segment"] = frame["segment"].map(_WEEKDAY_LABELS)
    return localize_table(
        frame,
        lang,
        columns={
            "view": "Dimensão",
            "segment": "Segmento",
            "n": "Observações",
            "observed_mean": "Demanda média observada",
            "predicted_mean": "Previsão média",
            "bias_mean": "Bias médio (ŷ-y)",
            "residual_mean": "Resíduo médio (y-ŷ)",
            "mae": "MAE",
            "rmse": "RMSE",
            "residual_std": "Desvio residual",
            "overestimation_share": "Proporção de superestimações",
            "underestimation_share": "Proporção de subestimações",
        },
        value_columns=["segment"],
        value_labels=_VALUE_LABELS,
    )


def plot_residual_triage(results: FinalValidationResults, lang=None) -> plt.Figure:
    """Rolling bias/errors, weekday-hour bias heatmap and demand-level scale."""
    lang = resolve_lang(lang)
    text = lang(
        {
            "title": "Triagem residual do champion no holdout",
            "rolling_bias": "Bias móvel (168 h)",
            "rolling_error": "Erro móvel (168 h)",
            "bias": "Bias (ŷ-y)",
            "mae": "MAE",
            "rmse": "RMSE",
            "heatmap": "Bias médio por dia da semana e hora",
            "hour": "Hora",
            "weekday": "Dia da semana",
            "level": "Erro por decil da previsão",
            "decile": "Decil da previsão",
            "residual_std": "Desvio residual",
        }
    )
    frame = champion_residual_diagnostic_frame(results)
    rolling = champion_rolling_residual_diagnostics(results)
    profile = champion_error_profiles(results)["predicted_demand_decile"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    axes[0, 0].plot(
        rolling["timestamp"],
        rolling["bias_rolling"],
        color=_CHAMPION_COLOR,
        linewidth=1.0,
    )
    axes[0, 0].axhline(0.0, color=_REFERENCE_COLOR, linewidth=1)
    axes[0, 0].set_title(text["rolling_bias"])
    axes[0, 0].set_ylabel(text["bias"])

    axes[0, 1].plot(
        rolling["timestamp"],
        rolling["mae_rolling"],
        color=_CHAMPION_COLOR,
        label=text["mae"],
    )
    axes[0, 1].plot(
        rolling["timestamp"],
        rolling["rmse_rolling"],
        color=_CHALLENGER_COLORS[0],
        label=text["rmse"],
    )
    axes[0, 1].set_title(text["rolling_error"])
    axes[0, 1].legend()

    heatmap = frame.pivot_table(
        index="weekday",
        columns="hour",
        values="bias",
        aggfunc="mean",
        observed=True,
    ).reindex(index=range(7), columns=range(24))
    image = axes[1, 0].imshow(heatmap.to_numpy(), aspect="auto", cmap="coolwarm")
    axes[1, 0].set_title(text["heatmap"])
    axes[1, 0].set_xlabel(text["hour"])
    axes[1, 0].set_ylabel(text["weekday"])
    axes[1, 0].set_yticks(range(7))
    axes[1, 0].set_yticklabels([_WEEKDAY_LABELS[index] for index in range(7)])
    fig.colorbar(image, ax=axes[1, 0], fraction=0.046, pad=0.04)

    axes[1, 1].plot(
        profile["segment"].astype(str),
        profile["mae"],
        marker="o",
        color=_CHAMPION_COLOR,
        label=text["mae"],
    )
    axes[1, 1].plot(
        profile["segment"].astype(str),
        profile["residual_std"],
        marker="o",
        color=_CHALLENGER_COLORS[0],
        label=text["residual_std"],
    )
    axes[1, 1].set_title(text["level"])
    axes[1, 1].set_xlabel(text["decile"])
    axes[1, 1].legend()
    fig.suptitle(text["title"])
    fig.tight_layout()
    return fig


def residual_transformation_report(results: FinalValidationResults, lang=None) -> pd.DataFrame:
    """ARCH and ACF diagnostics after descriptive residual transformations."""
    lang = resolve_lang(lang)
    frame = _format_p_value_columns(champion_residual_transformation_diagnostics(results))
    columns = {
        "residual_version": "Versão do resíduo",
        "arch_lag": "Lag ARCH",
        "n_observations": "Observações",
        "n_effective": "Observações efetivas",
        "mean": "Média",
        "std": "Desvio",
        "durbin_watson": "Durbin-Watson",
        "autocorr_lag_1": "ACF lag 1",
        "autocorr_lag_24": "ACF lag 24",
        "autocorr_lag_168": "ACF lag 168",
        "squared_autocorr_lag_1": "ACF² lag 1",
        "squared_autocorr_lag_24": "ACF² lag 24",
        "squared_autocorr_lag_168": "ACF² lag 168",
        "arch_statistic": "ARCH LM",
        "arch_statistic_per_observation": "ARCH LM/n",
        "p_value": "p-valor bruto",
        "adjusted_p_value": "p-valor ajustado (Holm)",
        "evidence_of_arch": "Evidência ARCH",
        "arch_statistic_reduction_vs_raw": "Redução ARCH vs. original",
        "autocorr_lag_24_reduction_vs_raw": "Redução ACF lag 24",
        "autocorr_lag_168_reduction_vs_raw": "Redução ACF lag 168",
        "squared_autocorr_lag_24_reduction_vs_raw": "Redução ACF² lag 24",
        "squared_autocorr_lag_168_reduction_vs_raw": "Redução ACF² lag 168",
        "status": "Status",
        "reason": "Motivo",
        "diagnostic_note": "Nota diagnóstica",
    }
    keep = [column for column in columns if column in frame.columns]
    return localize_table(
        frame[keep],
        lang,
        columns=columns,
        value_columns=["residual_version", "evidence_of_arch", "status"],
        value_labels={**_VALUE_LABELS, "ok": "Aplicável", "not_applicable": "Não aplicável"},
    )


def plot_residual_transformation_acf(
    results: FinalValidationResults,
    n_lags: int = 168,
    lang=None,
) -> plt.Figure:
    """Compare residual and squared-residual ACF after diagnostic transformations."""
    lang = resolve_lang(lang)
    text = lang(
        {
            "title": "Persistência residual antes e depois das transformações diagnósticas",
            "residual": "ACF dos resíduos",
            "squared": "ACF dos resíduos ao quadrado",
            "lag": "Defasagem (horas)",
            "acf": "Autocorrelação",
            "confidence": "Limite descritivo ±1,96/√n",
        }
    )
    transformed = champion_residual_transformation_frame(results)
    versions = ["raw", "calendar_demeaned", "level_standardized"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for version in versions:
        values = transformed.loc[
            transformed["residual_version"] == version, "diagnostic_residual"
        ].to_numpy(dtype=float)
        acf = autocorrelation_function(values, n_lags)
        squared_acf = autocorrelation_function(values**2, n_lags)
        label = _VALUE_LABELS.get(version, version)
        axes[0].plot(range(n_lags + 1), acf, linewidth=1.2, label=label)
        axes[1].plot(range(n_lags + 1), squared_acf, linewidth=1.2, label=label)

    n = len(transformed[transformed["residual_version"] == "raw"])
    limit = 1.96 / np.sqrt(n)
    for ax, title in zip(axes, [text["residual"], text["squared"]]):
        ax.axhline(0.0, color=_REFERENCE_COLOR, linewidth=1)
        ax.axhline(limit, color=_REFERENCE_COLOR, linestyle="--", linewidth=0.8)
        ax.axhline(-limit, color=_REFERENCE_COLOR, linestyle="--", linewidth=0.8)
        for lag in (24, 168):
            if lag <= n_lags:
                ax.axvline(lag, color=_REFERENCE_COLOR, linestyle=":", linewidth=0.9)
        ax.set_title(title)
        ax.set_xlabel(text["lag"])
        ax.set_ylabel(text["acf"])
        ax.legend()
    fig.suptitle(text["title"])
    fig.tight_layout()
    return fig


def _format_percent(value: float) -> str:
    """Compact percentage for narrative text."""
    if not np.isfinite(value):
        return "não mensurável"
    return f"{100 * value:.1f}%"


def residual_handoff_message(results: FinalValidationResults, lang=None) -> str:
    """Dynamic residual handoff for the next model version."""
    lang = resolve_lang(lang)
    triage = champion_residual_triage(results)
    diagnostics = champion_residual_transformation_diagnostics(results)
    first = triage.iloc[0]
    bias = float(first["bias_mean"])
    direction = (
        lang({"over": "superestimação"})["over"]
        if bias > 0
        else lang({"under": "subestimação"})["under"]
    )
    bias_ratio = _format_percent(float(first["bias_abs_to_mae_ratio"]))

    calendar = diagnostics[diagnostics["residual_version"] == "calendar_demeaned"]
    standardized = diagnostics[diagnostics["residual_version"] == "level_standardized"]
    weekly_reduction = float(calendar["autocorr_lag_168_reduction_vs_raw"].mean())
    arch_reduction = float(standardized["arch_statistic_reduction_vs_raw"].mean())
    arch_remains = bool(standardized["evidence_of_arch"].fillna(False).any())
    arch_text = (
        lang({"yes": "permaneceu"})["yes"]
        if arch_remains
        else lang({"no": "não permaneceu de forma material"})["no"]
    )

    return lang(
        {
            "m": (
                "A triagem residual indica deslocamento predominante para {direction}; "
                "o |bias| representa {bias_ratio} do MAE do champion. Após a retirada "
                "do padrão médio semanal, a autocorrelação no lag 168 foi reduzida em "
                "{weekly_reduction}; após a padronização por nível previsto, o indicador "
                "ARCH médio foi reduzido em {arch_reduction}, mas a evidência de "
                "agrupamento condicional {arch_text}. O CatBoost continua confirmado "
                "apenas como melhor previsão pontual sob o protocolo vigente; intervalos "
                "sob hipótese IID ainda não foram validados."
            )
        }
    )["m"].format(
        direction=direction,
        bias_ratio=bias_ratio,
        weekly_reduction=_format_percent(weekly_reduction),
        arch_reduction=_format_percent(arch_reduction),
        arch_text=arch_text,
    )


# ---------------------------------------------------------------------------
# 8. Residuals by operational condition
# ---------------------------------------------------------------------------

# Segmentation views highlighted in the notebook; every view is still persisted.
_CONDITION_VIEWS = ("season", "hour", "temperature_band", "demand_quintile")


def condition_metrics_report(
    results: FinalValidationResults, view: str = "season", lang=None
) -> pd.DataFrame:
    """One segmentation's per-segment MAE for every candidate."""
    lang = resolve_lang(lang)
    frame = results.segmented[view].copy()
    return localize_table(
        frame,
        lang,
        columns={
            "segment": "Segmento",
            "estimator": "Estimator",
            "role": "Papel",
            "n": "Observações",
            "mae": "MAE",
            "rmse": "RMSE",
            "wape": "WAPE",
            "mean_bias": "Bias médio (ŷ−y)",
        },
        value_columns=["role", "segment"],
        value_labels=_VALUE_LABELS,
    )


def plot_condition_metrics(results: FinalValidationResults, lang=None) -> plt.Figure:
    """Champion MAE across the highlighted operational-condition segmentations."""
    lang = resolve_lang(lang)
    text = lang(
        {
            "title": "MAE do champion por condição operacional",
            "season": "Estação",
            "hour": "Hora",
            "temperature_band": "Faixa de temperatura",
            "demand_quintile": "Quintil de demanda observada",
            "mae": "MAE",
        }
    )
    champion = _champion_evaluation(results)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, view in zip(axes.ravel(), _CONDITION_VIEWS):
        frame = results.segmented.get(view)
        if frame is None or frame.empty:
            ax.axis("off")
            continue
        subset = frame[frame["estimator"] == champion.estimator]
        ax.bar(subset["segment"].astype(str), subset["mae"], color=_CHAMPION_COLOR)
        ax.set_title(text.get(view, view))
        ax.set_ylabel(text["mae"])
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle(text["title"])
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 9-11. SHAP
# ---------------------------------------------------------------------------


def shap_methodology_report(lang=None) -> pd.DataFrame:
    """State what the SHAP decomposition explains and the additivity contract."""
    lang = resolve_lang(lang)
    labels = lang(
        {
            "decomposition": "Decomposição da previsão",
            "explains": "O que os valores SHAP explicam",
            "scale": "Escala de aditividade",
            "not_additive": "Não aditivo diretamente em",
            "identity_1": "Identidade verificada (1)",
            "identity_2": "Identidade verificada (2)",
            "tolerance": "Tolerância numérica",
            "limitation": "Limitação de interpretação",
        }
    )
    return _key_value_frame(
        lang,
        [
            (
                labels["decomposition"],
                lang(
                    {
                        "d": (
                            "log1p(previsão) = baseline temporal robusta + "
                            "previsão do modelo de resíduos"
                        )
                    }
                )["d"],
            ),
            (
                labels["explains"],
                lang({"e": "a previsão do modelo de resíduos na escala logarítmica residual"})["e"],
            ),
            (labels["scale"], lang({"s": "escala logarítmica residual"})["s"]),
            (
                labels["not_additive"],
                lang({"n": "bicicletas/hora, por causa da transformação expm1"})["n"],
            ),
            (
                labels["identity_1"],
                lang({"i": "expected_value + Σ SHAP ≈ previsão do estimator núcleo"})["i"],
            ),
            (
                labels["identity_2"],
                lang({"j": "clip(expm1(baseline + resíduo), 0) ≈ previsão do pipeline externo"})[
                    "j"
                ],
            ),
            (labels["tolerance"], "rtol=1e-6, atol=1e-6"),
            (
                labels["limitation"],
                lang(
                    {
                        "l": (
                            "os valores explicam o comportamento do modelo, não relações "
                            "causais; features correlacionadas podem compartilhar ou "
                            "redistribuir atribuições"
                        )
                    }
                )["l"],
            ),
        ],
    )


def shap_additivity_report(
    explanations: Sequence[ShapCandidateExplanation], lang=None
) -> pd.DataFrame:
    """Report the worst additivity and reconstruction error observed per candidate."""
    lang = resolve_lang(lang)
    rows = [
        {
            "role": explanation.role,
            "estimator": explanation.estimator,
            "n_sample": len(explanation.sample_positions),
            "n_features": len(explanation.feature_names),
            "additivity_max_error": explanation.additivity_max_error,
            "reconstruction_max_error": explanation.reconstruction_max_error,
        }
        for explanation in explanations
    ]
    frame = pd.DataFrame(rows)
    return localize_table(
        frame,
        lang,
        columns={
            "role": "Papel",
            "estimator": "Estimator",
            "n_sample": "Observações na amostra",
            "n_features": "Features transformadas",
            "additivity_max_error": "Erro máx. de aditividade",
            "reconstruction_max_error": "Erro máx. de reconstrução",
        },
        value_columns=["role"],
        value_labels=_VALUE_LABELS,
    )


def _explanation_by_role(
    explanations: Sequence[ShapCandidateExplanation], role: str = "champion"
) -> ShapCandidateExplanation:
    """Return the explanation for a given role."""
    return next(item for item in explanations if item.role == role)


def shap_grouped_report(
    explanations: Sequence[ShapCandidateExplanation], top_n: int = 15, lang=None
) -> pd.DataFrame:
    """Champion grouped (conceptual) SHAP importance, ranked."""
    lang = resolve_lang(lang)
    explanation = _explanation_by_role(explanations, "champion")
    frame = explanation.grouped_importance.head(top_n).copy()
    return localize_table(
        frame,
        lang,
        columns={"feature": "Feature conceitual", "mean_abs_shap": "|SHAP| médio (agrupado)"},
    )


def shap_detailed_report(
    explanations: Sequence[ShapCandidateExplanation], top_n: int = 20, lang=None
) -> pd.DataFrame:
    """Champion detailed (transformed-feature) SHAP importance, ranked."""
    lang = resolve_lang(lang)
    explanation = _explanation_by_role(explanations, "champion")
    frame = explanation.detailed_importance.head(top_n).copy()
    return localize_table(
        frame,
        lang,
        columns={"feature": "Feature transformada", "mean_abs_shap": "|SHAP| médio"},
    )


def shap_feature_comparison_report(
    explanations: Sequence[ShapCandidateExplanation], top_n: int = 10, lang=None
) -> pd.DataFrame:
    """Compare the principal conceptual features across the three candidates."""
    lang = resolve_lang(lang)
    merged: Optional[pd.DataFrame] = None
    for explanation in explanations:
        grouped = explanation.grouped_importance.rename(
            columns={"mean_abs_shap": explanation.estimator}
        )
        merged = grouped if merged is None else merged.merge(grouped, on="feature", how="outer")
    top_features = (
        _explanation_by_role(explanations, "champion")
        .grouped_importance["feature"]
        .head(top_n)
        .tolist()
    )
    merged = merged[merged["feature"].isin(top_features)].reset_index(drop=True)
    return localize_table(merged, lang, columns={"feature": "Feature conceitual"})


def plot_shap_summary(
    explanations: Sequence[ShapCandidateExplanation], top_n: int = 12, lang=None
) -> plt.Figure:
    """Beeswarm and grouped bar of the SHAP attributions for each candidate."""
    lang = resolve_lang(lang)
    text = lang(
        {
            "title": "Importância SHAP por candidato",
            "shap": "Valor SHAP (resíduo log)",
            "grouped": "|SHAP| médio agrupado",
            "beeswarm": "Dispersão (beeswarm)",
        }
    )
    n = len(explanations)
    fig, axes = plt.subplots(n, 2, figsize=(14, 3.6 * n))
    if n == 1:
        axes = axes.reshape(1, 2)
    for row, explanation in enumerate(explanations):
        _beeswarm(axes[row, 0], explanation, top_n)
        axes[row, 0].set_title(f"{explanation.estimator} — {text['beeswarm']}")
        axes[row, 0].set_xlabel(text["shap"])
        grouped = explanation.grouped_importance.head(top_n).iloc[::-1]
        axes[row, 1].barh(
            grouped["feature"].astype(str),
            grouped["mean_abs_shap"],
            color=_candidate_color(row, explanation.role),
        )
        axes[row, 1].set_title(f"{explanation.estimator} — {text['grouped']}")
    fig.suptitle(text["title"])
    fig.tight_layout()
    return fig


def _beeswarm(ax, explanation: ShapCandidateExplanation, top_n: int) -> None:
    """Draw a simple beeswarm of the top transformed features, coloured by value."""
    order = explanation.detailed_importance["feature"].head(top_n).tolist()
    name_to_index = {name: index for index, name in enumerate(explanation.feature_names)}
    rng = np.random.default_rng(0)
    for position, name in enumerate(reversed(order)):
        column = name_to_index[name]
        values = explanation.shap_values[:, column]
        feature_values = explanation.matrix[:, column]
        finite = np.isfinite(feature_values)
        norm = np.zeros_like(feature_values)
        if finite.any() and np.ptp(feature_values[finite]) > 0:
            low, high = feature_values[finite].min(), feature_values[finite].max()
            norm = (feature_values - low) / (high - low)
        jitter = rng.uniform(-0.25, 0.25, size=len(values))
        ax.scatter(values, position + jitter, c=norm, cmap="coolwarm", s=6, alpha=0.6)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(list(reversed(order)))
    ax.axvline(0.0, color=_REFERENCE_COLOR, linewidth=1)


def plot_shap_local(
    results: FinalValidationResults,
    explanations: Sequence[ShapCandidateExplanation],
    top_n: int = 12,
    lang=None,
) -> plt.Figure:
    """Waterfall-style local explanations for the champion (median, under, over).

    The bars are contributions to the log-residual, not to bicycles per hour;
    the title states this so the scale is not read as demand.
    """
    lang = resolve_lang(lang)
    text = lang(
        {
            "title": "Explicações locais do champion (contribuições ao resíduo logarítmico)",
            "median": "Erro absoluto mediano",
            "under": "Maior subestimação",
            "over": "Maior superestimação",
            "contribution": "Contribuição SHAP (resíduo log)",
        }
    )
    if results.decision == CONFIRMED:
        explanation = _explanation_by_role(explanations, "champion")
    else:
        best_role = "champion"
        best_run = results.confirmation["best_holdout_run_id"]
        for item in explanations:
            if item.run_id == best_run:
                best_role = item.role
        explanation = _explanation_by_role(explanations, best_role)

    order = ["median_abs_error", "largest_underestimation", "largest_overestimation"]
    titles = [text["median"], text["under"], text["over"]]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, key, title in zip(axes, order, titles):
        local_position = explanation.local_examples.get(key, 0)
        _waterfall(ax, explanation, local_position, top_n)
        ax.set_title(title)
        ax.set_xlabel(text["contribution"])
    fig.suptitle(text["title"])
    fig.tight_layout()
    return fig


def _waterfall(ax, explanation: ShapCandidateExplanation, row_position: int, top_n: int) -> None:
    """Horizontal bar of the largest SHAP contributions for one observation."""
    values = explanation.shap_values[row_position]
    order = np.argsort(np.abs(values))[::-1][:top_n]
    names = [explanation.feature_names[index] for index in order]
    contributions = values[order]
    colors = [_CHAMPION_COLOR if value >= 0 else _CHALLENGER_COLORS[0] for value in contributions]
    ax.barh(range(len(order))[::-1], contributions, color=colors)
    ax.set_yticks(range(len(order))[::-1])
    ax.set_yticklabels(names)
    ax.axvline(0.0, color=_REFERENCE_COLOR, linewidth=1)


# ---------------------------------------------------------------------------
# 12-13. Artifacts and synthesis
# ---------------------------------------------------------------------------


def artifacts_report(results: FinalValidationResults, lang=None) -> pd.DataFrame:
    """List the persisted artifacts and the MLflow run identifiers."""
    lang = resolve_lang(lang)
    labels = lang(
        {
            "runtime_root": "Diretório de runtime",
            "manifest": "Manifesto final",
            "parent_run": "Run pai (MLflow)",
            "child_runs": "Runs filhas (MLflow)",
            "experiment": "Experimento MLflow",
            "cache": "Origem dos resultados",
            "decision": "Decisão registrada",
        }
    )
    source = (
        lang({"c": "reutilizados do cache"})["c"]
        if results.loaded_from_cache
        else lang({"f": "calculados nesta execução"})["f"]
    )
    return _key_value_frame(
        lang,
        [
            (labels["runtime_root"], public_path(results.config.runtime_root)),
            (labels["manifest"], public_path(results.final_manifest_path)),
            (labels["experiment"], results.config.experiment_name),
            (labels["parent_run"], results.parent_run_id),
            (labels["child_runs"], ", ".join(results.child_run_ids.values()) or None),
            (labels["cache"], source),
            (
                labels["decision"],
                lang({"d": _VALUE_LABELS.get(results.decision, results.decision)})["d"],
            ),
        ],
    )


def synthesis_report(results: FinalValidationResults, lang=None) -> pd.DataFrame:
    """Final synthesis: decision, champion holdout standing and closest challenger."""
    lang = resolve_lang(lang)
    confirmation = results.confirmation
    champion = results.champion_evaluation
    triage = champion_residual_triage(results)
    residual_summary = triage.iloc[0]
    labels = lang(
        {
            "decision": "Decisão final",
            "champion": "Champion",
            "champion_mae": "MAE do champion no holdout",
            "champion_r2": "R² do champion no holdout",
            "best": "Melhor no holdout",
            "gap": "Diferença de MAE para o melhor",
            "bias_ratio": "|bias| / MAE do champion",
            "dw": "Durbin-Watson residual",
            "iid_intervals": "Intervalos IID validados",
            "independent": "Nova janela independente exigida",
        }
    )
    return _key_value_frame(
        lang,
        [
            (
                labels["decision"],
                lang({"d": _VALUE_LABELS.get(results.decision, results.decision)})["d"],
            ),
            (labels["champion"], champion.estimator),
            (labels["champion_mae"], round(champion.metrics["holdout_mae"], 3)),
            (labels["champion_r2"], round(champion.metrics["holdout_r2"], 4)),
            (labels["best"], confirmation["best_holdout_estimator"]),
            (labels["gap"], round(abs(confirmation["champion_mae_gap_to_best"]), 3)),
            (labels["bias_ratio"], round(float(residual_summary["bias_abs_to_mae_ratio"]), 4)),
            (labels["dw"], round(float(residual_summary["durbin_watson"]), 4)),
            (labels["iid_intervals"], False),
            (labels["independent"], confirmation["independent_window_required_for_switch"]),
        ],
    )


def handoff_message(results: FinalValidationResults, lang=None) -> str:
    """A closing narrative summarising the confirmatory outcome and its limits."""
    lang = resolve_lang(lang)
    if results.decision == CONFIRMED:
        base = lang(
            {
                "m": (
                    "A validação final confirma o champion pré-registrado sob o protocolo "
                    "confirmatório, medido uma única vez no holdout temporal selado. Os "
                    "resultados foram persistidos e registrados em experimento MLflow próprio, "
                    "sem qualquer modificação nas runs de seleção. "
                )
            }
        )["m"]
    else:
        base = lang(
            {
                "m": (
                    "A validação final não confirma o champion pré-registrado no holdout "
                    "temporal. A proximidade entre os candidatos é discutida nos relatórios; "
                    "uma troca definitiva exigiria uma nova janela temporal independente. O "
                    "manifesto do notebook 04 permanece inalterado e nenhum modelo é "
                    "retreinado. "
                )
            }
        )["m"]
    return base + residual_handoff_message(results, lang=lang)
