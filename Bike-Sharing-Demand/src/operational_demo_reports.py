"""Bilingual presentation layer for notebook 08 operational replay."""

from __future__ import annotations

from typing import Dict, Iterable

import matplotlib.pyplot as plt
import pandas as pd

from src.i18n import localize_table, make_lang, resolve_lang as _project_resolve_lang
from src.model_selection_reports import environment_report as _environment_report
from src.operational_demo import OperationalDemoConfig, OperationalReplayResult
from src.utils import public_path


def resolve_lang(lang):
    """Accept a project LangMap or the short language code used by tests."""
    if isinstance(lang, str):
        return make_lang(lang)
    return _project_resolve_lang(lang)


def _key_value_frame(lang, rows: Iterable[tuple]) -> pd.DataFrame:
    labels = lang({"item": "Item", "value": "Valor"})
    return pd.DataFrame([{labels["item"]: label, labels["value"]: value} for label, value in rows])


def _format_number(value: float, decimals: int = 0, lang=None) -> str:
    rendered = f"{value:,.{decimals}f}"
    if getattr(lang, "target", "pt") == "en":
        return rendered
    return rendered.replace(",", "_").replace(".", ",").replace("_", ".")


def _date_format(lang) -> str:
    return lang({"format": "%d/%m/%Y %H:%M"})["format"]


def environment_report(lang=None) -> pd.DataFrame:
    return _environment_report(lang=lang)


def protocol_report(config: OperationalDemoConfig, lang=None) -> pd.DataFrame:
    lang = resolve_lang(lang)
    labels = lang(
        {
            "nature": "Natureza",
            "holdout": "Uso do holdout final",
            "candidate": "Candidato de incerteza",
            "coverage": "Cobertura nominal",
            "seed": "Semente do sorteio",
            "capacity": "Capacidade operacional simulada",
            "refit": "Reajuste de modelos",
        }
    )
    values = lang(
        {
            "nature": "replay histórico de uma previsão OOF congelada",
            "holdout": "não utilizado",
            "capacity": "{capacity} aluguéis/hora",
            "refit": "nenhum",
        }
    )
    return _key_value_frame(
        lang,
        [
            (labels["nature"], values["nature"]),
            (labels["holdout"], values["holdout"]),
            (labels["candidate"], config.candidate_id),
            (labels["coverage"], f"{config.coverage:.0%}"),
            (labels["seed"], config.random_state),
            (
                labels["capacity"],
                values["capacity"].format(
                    capacity=_format_number(config.planned_capacity, lang=lang)
                ),
            ),
            (labels["refit"], values["refit"]),
        ],
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
    lang = resolve_lang(lang)
    labels = lang(
        {
            "timestamp": "Timestamp sorteado",
            "fold": "Fold temporal",
            "year": "Ano meteorológico",
            "rows": "Linhas elegíveis",
            "history": "Histórico do calibrador",
            "alpha": "Alpha adaptativo usado",
        }
    )
    values = lang({"hours": "{hours} horas"})
    return _key_value_frame(
        lang,
        [
            (labels["timestamp"], results.timestamp.strftime(_date_format(lang))),
            (labels["fold"], results.fold),
            (labels["year"], results.test_year),
            (labels["rows"], _format_number(results.eligible_rows, lang=lang)),
            (
                labels["history"],
                values["hours"].format(hours=_format_number(results.calibration_size, lang=lang)),
            ),
            (labels["alpha"], _format_number(results.alpha_used, 3, lang=lang)),
        ],
    )


def profile_report(results: OperationalReplayResult, lang=None) -> pd.DataFrame:
    lang = resolve_lang(lang)
    profile = results.profile
    labels = lang(
        {
            "timestamp": "Data e hora",
            "weekday": "Dia da semana",
            "hour": "Hora",
            "period": "Período",
            "season": "Estação",
            "rain": "Chuva",
            "temperature": "Temperatura",
            "humidity": "Umidade",
            "wind": "Velocidade do vento",
            "visibility": "Visibilidade",
            "solar": "Radiação solar",
            "precipitation": "Precipitação",
            "snow": "Neve",
            "holiday": "Feriado",
            "functioning": "Sistema em funcionamento",
        }
    )
    values = lang(
        {
            "monday": "segunda-feira",
            "tuesday": "terça-feira",
            "wednesday": "quarta-feira",
            "thursday": "quinta-feira",
            "friday": "sexta-feira",
            "saturday": "sábado",
            "sunday": "domingo",
            "morning_rush": "pico da manhã",
            "evening_rush": "pico da tarde",
            "non_rush": "fora do pico",
            "winter": "inverno",
            "spring": "primavera",
            "summer": "verão",
            "autumn": "outono",
            "no_rain": "sem chuva",
            "light_rain": "chuva leve",
            "moderate_rain": "chuva moderada",
            "heavy_rain": "chuva forte",
            "holiday": "feriado",
            "not_holiday": "não é feriado",
            "yes": "sim",
            "no": "não",
        }
    )
    weekday_map = {
        "Monday": values["monday"],
        "Tuesday": values["tuesday"],
        "Wednesday": values["wednesday"],
        "Thursday": values["thursday"],
        "Friday": values["friday"],
        "Saturday": values["saturday"],
        "Sunday": values["sunday"],
    }
    value_maps: Dict[str, Dict[str, str]] = {
        "Rush_Period": {
            "Morning Rush": values["morning_rush"],
            "Evening Rush": values["evening_rush"],
            "Non-Rush": values["non_rush"],
        },
        "Seasons": {
            "Winter": values["winter"],
            "Spring": values["spring"],
            "Summer": values["summer"],
            "Autumn": values["autumn"],
        },
        "Rainfall Cat": {
            "No Rain": values["no_rain"],
            "Light Rain": values["light_rain"],
            "Moderate Rain": values["moderate_rain"],
            "Heavy Rain": values["heavy_rain"],
        },
        "Holiday": {
            "Holiday": values["holiday"],
            "No Holiday": values["not_holiday"],
        },
        "Functioning Day": {"Yes": values["yes"], "No": values["no"]},
    }
    rows = [
        (
            labels["timestamp"],
            pd.Timestamp(profile["timestamp"]).strftime(_date_format(lang)),
        ),
        (
            labels["weekday"],
            weekday_map.get(profile["weekday"], profile["weekday"]),
        ),
        (labels["hour"], int(profile["hour"])),
        (
            labels["period"],
            value_maps["Rush_Period"].get(profile["Rush_Period"], profile["Rush_Period"]),
        ),
        (
            labels["season"],
            value_maps["Seasons"].get(profile["Seasons"], profile["Seasons"]),
        ),
        (
            labels["rain"],
            value_maps["Rainfall Cat"].get(profile["Rainfall Cat"], profile["Rainfall Cat"]),
        ),
        (labels["temperature"], f"{float(profile['Temperature(C)']):.1f} °C"),
        (labels["humidity"], f"{float(profile['Humidity(%)']):.0f}%"),
        (labels["wind"], f"{float(profile['Wind speed (m/s)']):.1f} m/s"),
        (labels["visibility"], f"{float(profile['Visibility (10m)']) * 10:.0f} m"),
        (
            labels["solar"],
            f"{float(profile['Solar Radiation (MJ/m2)']):.2f} MJ/m²",
        ),
        (labels["precipitation"], f"{float(profile['Rainfall(mm)']):.1f} mm"),
        (labels["snow"], f"{float(profile['Snowfall (cm)']):.1f} cm"),
        (
            labels["holiday"],
            value_maps["Holiday"].get(profile["Holiday"], profile["Holiday"]),
        ),
        (
            labels["functioning"],
            value_maps["Functioning Day"].get(
                profile["Functioning Day"], profile["Functioning Day"]
            ),
        ),
    ]
    return _key_value_frame(lang, rows)


def forecast_report(results: OperationalReplayResult, lang=None) -> pd.DataFrame:
    lang = resolve_lang(lang)
    labels = lang(
        {
            "point": "Previsão pontual E0",
            "lower": "Limite inferior de 90%",
            "upper": "Limite superior de 90%",
            "width": "Largura do intervalo",
        }
    )
    value_template = lang({"rentals": "{value} aluguéis/hora"})["rentals"]
    rows = []
    for key, value in (
        ("point", results.point_prediction),
        ("lower", results.lower),
        ("upper", results.upper),
        ("width", results.interval_width),
    ):
        rows.append(
            (
                labels[key],
                value_template.format(value=_format_number(value, lang=lang)),
            )
        )
    return _key_value_frame(lang, rows)


def _decision_label(code: str, lang) -> str:
    labels = lang(
        {
            "critical_shortage": "CAPACIDADE CRITICAMENTE INSUFICIENTE",
            "reinforcement_recommended": "REFOR\u00c7O RECOMENDADO",
            "attention_zone": "ZONA DE ATEN\u00c7\u00c3O",
            "capacity_compatible": "CAPACIDADE COMPATÍVEL",
        }
    )
    return labels[code]


def decision_report(results: OperationalReplayResult, lang=None) -> pd.DataFrame:
    lang = resolve_lang(lang)
    labels = lang(
        {
            "capacity": "Capacidade operacional simulada",
            "decision": "Decisão",
            "point_gap": "Capacidade adicional até a previsão central",
            "upper_gap": "Reserva adicional até o limite superior",
        }
    )
    value_template = lang({"rentals": "{value} aluguéis/hora"})["rentals"]

    def rentals(value: float) -> str:
        return value_template.format(value=_format_number(value, lang=lang))

    return _key_value_frame(
        lang,
        [
            (labels["capacity"], rentals(results.planned_capacity)),
            (labels["decision"], _decision_label(results.decision_code, lang)),
            (labels["point_gap"], rentals(results.additional_capacity_to_point)),
            (labels["upper_gap"], rentals(results.additional_capacity_to_upper)),
        ],
    )


def audit_report(results: OperationalReplayResult, lang=None) -> pd.DataFrame:
    lang = resolve_lang(lang)
    labels = lang(
        {
            "actual": "Demanda posteriormente observada",
            "error": "Erro absoluto da previsão central",
            "covered": "Valor real contido no intervalo",
            "served": "Capacidade simulada teria atendido a demanda",
        }
    )
    values = lang({"yes": "sim", "no": "não"})
    return _key_value_frame(
        lang,
        [
            (labels["actual"], _format_number(results.actual_demand, lang=lang)),
            (labels["error"], _format_number(results.absolute_error, lang=lang)),
            (labels["covered"], values["yes"] if results.covered else values["no"]),
            (
                labels["served"],
                values["yes"]
                if results.planned_capacity >= results.actual_demand
                else values["no"],
            ),
        ],
    )


def plot_operational_forecast(results: OperationalReplayResult, lang=None):
    lang = resolve_lang(lang)
    labels = lang(
        {
            "point": "Previsão E0",
            "actual": "Demanda observada",
            "capacity": "Capacidade simulada",
            "interval": "Intervalo de 90%: {lower}-{upper}",
            "xlabel": "Demanda agregada (aluguéis por hora)",
            "title": "Replay operacional: previsão, incerteza e capacidade",
        }
    )
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.hlines(
        0,
        results.lower,
        results.upper,
        color="#4C78A8",
        linewidth=12,
        alpha=0.35,
    )
    ax.scatter(
        results.point_prediction,
        0,
        s=130,
        color="#1F4E79",
        label=labels["point"],
        zorder=3,
    )
    actual_color = "#2E8B57" if results.covered else "#C0392B"
    ax.scatter(
        results.actual_demand,
        0,
        s=130,
        marker="D",
        color=actual_color,
        label=labels["actual"],
        zorder=3,
    )
    ax.axvline(
        results.planned_capacity,
        color="#F28E2B",
        linewidth=2.5,
        linestyle="--",
        label=labels["capacity"],
    )
    ax.annotate(
        labels["interval"].format(
            lower=_format_number(results.lower, lang=lang),
            upper=_format_number(results.upper, lang=lang),
        ),
        ((results.lower + results.upper) / 2, 0.08),
        ha="center",
        va="bottom",
        fontsize=10,
    )
    maximum = max(results.upper, results.planned_capacity, results.actual_demand)
    ax.set_xlim(0, maximum * 1.08 if maximum > 0 else 1)
    ax.set_ylim(-0.35, 0.35)
    ax.set_yticks([])
    ax.set_xlabel(labels["xlabel"])
    ax.set_title(labels["title"])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return fig


def synthesis_report(results: OperationalReplayResult, lang=None) -> str:
    lang = resolve_lang(lang)
    phrases = lang(
        {
            "covered": "foi contida",
            "not_covered": "não foi contida",
            "served": "teria atendido",
            "not_served": "não teria atendido",
        }
    )
    template = lang(
        {
            "summary": (
                "Para {timestamp}, o E0 previu {point} aluguéis/hora e o U4b "
                "produziu o intervalo [{lower}, {upper}]. A capacidade operacional "
                "simulada de {capacity} foi classificada como {decision}. Depois da "
                "revelação, a demanda observada de {actual} {coverage_text} no intervalo "
                "e a capacidade simulada {capacity_text} o volume realizado."
            )
        }
    )["summary"]
    return template.format(
        timestamp=results.timestamp.strftime(_date_format(lang)),
        point=_format_number(results.point_prediction, lang=lang),
        lower=_format_number(results.lower, lang=lang),
        upper=_format_number(results.upper, lang=lang),
        capacity=_format_number(results.planned_capacity, lang=lang),
        decision=_decision_label(results.decision_code, lang),
        actual=_format_number(results.actual_demand, lang=lang),
        coverage_text=(phrases["covered"] if results.covered else phrases["not_covered"]),
        capacity_text=(
            phrases["served"]
            if results.planned_capacity >= results.actual_demand
            else phrases["not_served"]
        ),
    )
