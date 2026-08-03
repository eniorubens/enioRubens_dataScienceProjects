"""PT-BR presentation layer for notebook 08 operational replay."""

from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

from src.i18n import localize_table, make_lang, resolve_lang as _resolve_lang
from src.model_selection_reports import environment_report as _environment_report
from src.operational_demo import OperationalDemoConfig, OperationalReplayResult
from src.utils import public_path


def resolve_lang(lang):
    if isinstance(lang, str):
        return make_lang(lang)
    return _resolve_lang(lang)


def environment_report(lang=None) -> pd.DataFrame:
    return _environment_report(lang=lang)


def _key_value_frame(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["Item", "Valor"])


def _format_number(value: float, decimals: int = 0) -> str:
    rendered = f"{value:,.{decimals}f}"
    return rendered.replace(",", "_").replace(".", ",").replace("_", ".")


def protocol_report(config: OperationalDemoConfig, lang=None) -> pd.DataFrame:
    resolve_lang(lang)
    return _key_value_frame(
        [
            ("Natureza", "replay histórico de uma previsão OOF congelada"),
            ("Uso do holdout final", "não utilizado"),
            ("Candidato de incerteza", config.candidate_id),
            ("Cobertura nominal", f"{config.coverage:.0%}"),
            ("Semente do sorteio", config.random_state),
            (
                "Capacidade operacional simulada",
                f"{_format_number(config.planned_capacity)} aluguéis/hora",
            ),
            ("Reajuste de modelos", "nenhum"),
        ]
    )


def source_audit_report(results: OperationalReplayResult, lang=None) -> pd.DataFrame:
    frame = results.source_audit.copy()
    frame["path"] = frame["path"].map(public_path)
    return localize_table(
        frame,
        resolve_lang(lang),
        {
            "artifact": "Artefato",
            "path": "Caminho",
            "bytes": "Bytes",
            "sha256": "SHA-256",
        },
    )


def selection_report(results: OperationalReplayResult, lang=None) -> pd.DataFrame:
    resolve_lang(lang)
    return _key_value_frame(
        [
            ("Timestamp sorteado", results.timestamp.strftime("%d/%m/%Y %H:%M")),
            ("Fold temporal", results.fold),
            ("Ano meteorológico", results.test_year),
            ("Linhas elegíveis", _format_number(results.eligible_rows)),
            (
                "Histórico do calibrador",
                f"{_format_number(results.calibration_size)} horas",
            ),
            ("Alpha adaptativo usado", _format_number(results.alpha_used, 3)),
        ]
    )


def profile_report(results: OperationalReplayResult, lang=None) -> pd.DataFrame:
    resolve_lang(lang)
    profile = results.profile
    weekday_map = {
        "Monday": "segunda-feira",
        "Tuesday": "terça-feira",
        "Wednesday": "quarta-feira",
        "Thursday": "quinta-feira",
        "Friday": "sexta-feira",
        "Saturday": "sábado",
        "Sunday": "domingo",
    }
    value_maps: Dict[str, Dict[str, str]] = {
        "Rush_Period": {
            "Morning Rush": "pico da manhã",
            "Evening Rush": "pico da tarde",
            "Non-Rush": "fora do pico",
        },
        "Seasons": {
            "Winter": "inverno",
            "Spring": "primavera",
            "Summer": "verão",
            "Autumn": "outono",
        },
        "Rainfall Cat": {
            "No Rain": "sem chuva",
            "Light Rain": "chuva leve",
            "Moderate Rain": "chuva moderada",
            "Heavy Rain": "chuva forte",
        },
        "Holiday": {"Holiday": "feriado", "No Holiday": "não é feriado"},
        "Functioning Day": {"Yes": "sim", "No": "não"},
    }
    rows = [
        ("Data e hora", pd.Timestamp(profile["timestamp"]).strftime("%d/%m/%Y %H:%M")),
        ("Dia da semana", weekday_map.get(profile["weekday"], profile["weekday"])),
        ("Hora", int(profile["hour"])),
        ("Período", value_maps["Rush_Period"].get(profile["Rush_Period"], profile["Rush_Period"])),
        ("Estação", value_maps["Seasons"].get(profile["Seasons"], profile["Seasons"])),
        ("Chuva", value_maps["Rainfall Cat"].get(profile["Rainfall Cat"], profile["Rainfall Cat"])),
        ("Temperatura", f"{float(profile['Temperature(C)']):.1f} °C"),
        ("Umidade", f"{float(profile['Humidity(%)']):.0f}%"),
        ("Velocidade do vento", f"{float(profile['Wind speed (m/s)']):.1f} m/s"),
        ("Visibilidade", f"{float(profile['Visibility (10m)']) * 10:.0f} m"),
        (
            "Radiação solar",
            f"{float(profile['Solar Radiation (MJ/m2)']):.2f} MJ/m²",
        ),
        ("Precipitação", f"{float(profile['Rainfall(mm)']):.1f} mm"),
        ("Neve", f"{float(profile['Snowfall (cm)']):.1f} cm"),
        ("Feriado", value_maps["Holiday"].get(profile["Holiday"], profile["Holiday"])),
        (
            "Sistema em funcionamento",
            value_maps["Functioning Day"].get(
                profile["Functioning Day"], profile["Functioning Day"]
            ),
        ),
    ]
    return _key_value_frame(rows)


def forecast_report(results: OperationalReplayResult, lang=None) -> pd.DataFrame:
    resolve_lang(lang)
    return _key_value_frame(
        [
            (
                "Previsão pontual E0",
                f"{_format_number(results.point_prediction)} aluguéis/hora",
            ),
            (
                "Limite inferior de 90%",
                f"{_format_number(results.lower)} aluguéis/hora",
            ),
            (
                "Limite superior de 90%",
                f"{_format_number(results.upper)} aluguéis/hora",
            ),
            (
                "Largura do intervalo",
                f"{_format_number(results.interval_width)} aluguéis/hora",
            ),
        ]
    )


def _decision_label(code: str) -> str:
    labels = {
        "critical_shortage": "CAPACIDADE CRITICAMENTE INSUFICIENTE",
        "reinforcement_recommended": "REFORÇO RECOMENDADO",
        "attention_zone": "ZONA DE ATENÇÃO",
        "capacity_compatible": "CAPACIDADE COMPATÍVEL",
    }
    return labels[code]


def decision_report(results: OperationalReplayResult, lang=None) -> pd.DataFrame:
    resolve_lang(lang)
    return _key_value_frame(
        [
            (
                "Capacidade operacional simulada",
                f"{_format_number(results.planned_capacity)} aluguéis/hora",
            ),
            ("Decisão", _decision_label(results.decision_code)),
            (
                "Capacidade adicional até a previsão central",
                f"{_format_number(results.additional_capacity_to_point)} aluguéis/hora",
            ),
            (
                "Reserva adicional até o limite superior",
                f"{_format_number(results.additional_capacity_to_upper)} aluguéis/hora",
            ),
        ]
    )


def audit_report(results: OperationalReplayResult, lang=None) -> pd.DataFrame:
    resolve_lang(lang)
    return _key_value_frame(
        [
            (
                "Demanda posteriormente observada",
                _format_number(results.actual_demand),
            ),
            (
                "Erro absoluto da previsão central",
                _format_number(results.absolute_error),
            ),
            ("Valor real contido no intervalo", "sim" if results.covered else "não"),
            (
                "Capacidade simulada teria atendido a demanda",
                "sim" if results.planned_capacity >= results.actual_demand else "não",
            ),
        ]
    )


def plot_operational_forecast(results: OperationalReplayResult, lang=None):
    resolve_lang(lang)
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.hlines(0, results.lower, results.upper, color="#4C78A8", linewidth=12, alpha=0.35)
    ax.scatter(
        results.point_prediction,
        0,
        s=130,
        color="#1F4E79",
        label="Previsão E0",
        zorder=3,
    )
    actual_color = "#2E8B57" if results.covered else "#C0392B"
    ax.scatter(
        results.actual_demand,
        0,
        s=130,
        marker="D",
        color=actual_color,
        label="Demanda observada",
        zorder=3,
    )
    ax.axvline(
        results.planned_capacity,
        color="#F28E2B",
        linewidth=2.5,
        linestyle="--",
        label="Capacidade simulada",
    )
    ax.annotate(
        f"Intervalo de 90%: {_format_number(results.lower)}–" f"{_format_number(results.upper)}",
        ((results.lower + results.upper) / 2, 0.08),
        ha="center",
        va="bottom",
        fontsize=10,
    )
    maximum = max(results.upper, results.planned_capacity, results.actual_demand)
    ax.set_xlim(0, maximum * 1.08 if maximum > 0 else 1)
    ax.set_ylim(-0.35, 0.35)
    ax.set_yticks([])
    ax.set_xlabel("Demanda agregada (aluguéis por hora)")
    ax.set_title("Replay operacional: previsão, incerteza e capacidade")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return fig


def synthesis_report(results: OperationalReplayResult, lang=None) -> str:
    resolve_lang(lang)
    coverage_text = "foi contida" if results.covered else "não foi contida"
    capacity_text = (
        "teria atendido"
        if results.planned_capacity >= results.actual_demand
        else "não teria atendido"
    )
    return (
        f"Para {results.timestamp:%d/%m/%Y às %H:%M}, o E0 previu "
        f"{_format_number(results.point_prediction)} aluguéis/hora e o U4b produziu \n"
        f"o intervalo [{_format_number(results.lower)}, "
        f"{_format_number(results.upper)}]. A capacidade operacional simulada de \n"
        f"{_format_number(results.planned_capacity)} foi classificada como "
        f"{_decision_label(results.decision_code)}. Depois da revelação, a demanda \n"
        f"observada de {_format_number(results.actual_demand)} {coverage_text} no "
        f"intervalo e a "
        f"capacidade simulada {capacity_text} o volume realizado."
    )
