"""Geração reutilizável de gráficos SVG para métricas por threshold."""

from __future__ import annotations

from html import escape

import pandas as pd


def build_threshold_metric_svgs(
    data: pd.DataFrame,
    metric_labels: dict[str, str],
    model_label: str,
) -> dict[str, str]:
    """Retorna um SVG acessível para cada métrica dependente do threshold.

    A função não exibe nem salva arquivos: devolve strings SVG para que o
    notebook escolha como apresentá-las. Isso mantém a construção visual em
    ``src`` e deixa a camada narrativa responsável apenas pela exibição.
    """
    required_columns = {"threshold", *metric_labels}
    missing = required_columns.difference(data.columns)
    if missing:
        raise ValueError(f"Missing threshold-plot columns: {sorted(missing)}")
    if data.empty:
        raise ValueError("Cannot plot threshold metrics from an empty table")

    width, height = 720, 500
    left, right, top, bottom = 78, 24, 38, 120
    x_values = data["threshold"].astype(float).tolist()
    x_min, x_max = min(x_values), max(x_values)
    if x_min == x_max:
        raise ValueError("Threshold values must contain more than one point")

    def x_scale(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (width - left - right)

    def y_scale(value: float) -> float:
        return top + (1 - value) * (height - top - bottom)

    safe_model = escape(model_label)
    svgs = {}
    for metric, label in metric_labels.items():
        safe_label = escape(label)
        y_values = data[metric].astype(float).tolist()
        points = " ".join(
            f"{x_scale(x):.1f},{y_scale(y):.1f}"
            for x, y in zip(x_values, y_values)
        )
        x_ticks = "".join(
            f'<line x1="{x_scale(x):.1f}" y1="{top}" '
            f'x2="{x_scale(x):.1f}" y2="{height-bottom}" '
            'stroke="#d9dee7" stroke-dasharray="2,4"/>'
            f'<text x="{x_scale(x):.1f}" y="{height-bottom+18}" '
            f'text-anchor="start" font-size="10" '
            f'transform="rotate(90 {x_scale(x):.1f} {height-bottom+18})">'
            f'{x:.2f}</text>'
            for x in x_values
        )
        y_ticks = "".join(
            f'<line x1="{left}" y1="{y_scale(y):.1f}" '
            f'x2="{width-right}" y2="{y_scale(y):.1f}" '
            'stroke="#d9dee7" stroke-dasharray="2,4"/>'
            f'<text x="{left-12}" y="{y_scale(y)+4:.1f}" '
            f'text-anchor="end" font-size="12">{y:.1f}</text>'
            for y in [0, 0.2, 0.4, 0.6, 0.8, 1]
        )
        markers = "".join(
            f'<circle cx="{x_scale(x):.1f}" cy="{y_scale(y):.1f}" '
            f'r="4" fill="#2f5597"><title>Threshold {x:.2f}: '
            f'{safe_label} {y:.3f}</title></circle>'
            for x, y in zip(x_values, y_values)
        )
        title = f"{label} em função do threshold — {model_label}"
        safe_title = escape(title)
        svgs[metric] = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
            f'aria-label="{safe_title}"><title>{safe_title}</title>'
            f'<rect width="100%" height="100%" fill="white"/>'
            f'<text x="{width/2}" y="22" text-anchor="middle" '
            f'font-size="16" font-weight="600">{safe_title}</text>'
            f"{x_ticks}{y_ticks}"
            f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" '
            f'y2="{height-bottom}" stroke="#555"/>'
            f'<line x1="{left}" y1="{top}" x2="{left}" '
            f'y2="{height-bottom}" stroke="#555"/>'
            f'<polyline points="{points}" fill="none" stroke="#2f5597" '
            f'stroke-width="2.5"/>{markers}'
            f'<text x="{width/2}" y="{height-14}" text-anchor="middle" '
            f'font-size="13">Threshold</text>'
            f'<text x="18" y="{height/2}" text-anchor="middle" '
            f'font-size="13" transform="rotate(90 18 {height/2})">'
            f"{safe_label}</text></svg>"
        )
    return svgs
