"""Gráficos SVG reutilizáveis para taxas de auditoria com intervalo de Wilson."""

from __future__ import annotations

from html import escape
import math

import pandas as pd


AUDIT_COLUMNS = ("sex", "race")
INTERSECTION_COLUMN = "sex_race"
LOW_SUPPORT_THRESHOLD = 30


def _safe_text(value: object) -> str:
    """Converte texto para uso seguro dentro de SVG."""
    return escape(str(value))


def _panel_svg(
    data: pd.DataFrame,
    metric: str,
    metric_label: str,
    title: str,
    model_label: str,
    audit_column: str,
    width: int = 900,
) -> str:
    """Constrói um forest plot horizontal para um atributo de auditoria."""
    panel = data.loc[
        data["audit_column"].eq(audit_column) & data["metric"].eq(metric)
    ].copy()
    panel = panel.sort_values("group", kind="stable").reset_index(drop=True)
    row_height = 38
    left, right, top, bottom = 250, 34, 58, 62
    height = max(330, top + bottom + row_height * len(panel))
    plot_width = width - left - right

    def x_scale(value: float) -> float:
        return left + value * plot_width

    def y_scale(index: int) -> float:
        return top + row_height * (index + 0.5)

    grid = "".join(
        f'<line x1="{x_scale(value):.1f}" y1="{top}" '
        f'x2="{x_scale(value):.1f}" y2="{height-bottom}" '
        'stroke="#d9dee7" stroke-dasharray="2,4"/>'
        f'<text x="{x_scale(value):.1f}" y="{height-bottom+22}" '
        f'text-anchor="middle" font-size="12">{value:.1f}</text>'
        for value in [0, 0.2, 0.4, 0.6, 0.8, 1]
    )
    marks = []
    labels = []
    for index, row in panel.iterrows():
        y = y_scale(index)
        group = _safe_text(row["group"])
        low_support = str(row.get("low_support", False)).lower() == "true"
        marker = " †" if low_support else ""
        labels.append(
            f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" '
            f'font-size="12">{group}{marker}</text>'
        )
        estimate = float(row["estimate"]) if pd.notna(row["estimate"]) else math.nan
        ci_low = float(row["ci_low"]) if pd.notna(row["ci_low"]) else math.nan
        ci_high = float(row["ci_high"]) if pd.notna(row["ci_high"]) else math.nan
        if not math.isfinite(estimate):
            continue
        color = "#c55a11" if low_support else "#2f5597"
        if math.isfinite(ci_low) and math.isfinite(ci_high):
            marks.append(
                f'<line x1="{x_scale(ci_low):.1f}" y1="{y:.1f}" '
                f'x2="{x_scale(ci_high):.1f}" y2="{y:.1f}" '
                f'stroke="{color}" stroke-width="2"/>'
                f'<line x1="{x_scale(ci_low):.1f}" y1="{y-6:.1f}" '
                f'x2="{x_scale(ci_low):.1f}" y2="{y+6:.1f}" '
                f'stroke="{color}" stroke-width="2"/>'
                f'<line x1="{x_scale(ci_high):.1f}" y1="{y-6:.1f}" '
                f'x2="{x_scale(ci_high):.1f}" y2="{y+6:.1f}" '
                f'stroke="{color}" stroke-width="2"/>'
            )
        marks.append(
            f'<circle cx="{x_scale(estimate):.1f}" cy="{y:.1f}" r="5" '
            f'fill="{color}"><title>{group}: estimate={estimate:.3f}; '
            f'95% Wilson CI=[{ci_low:.3f}, {ci_high:.3f}]; '
            f'n={int(row["count"])}; denominator={int(row["denominator"])}</title></circle>'
        )
    support_note = (
        '<text x="20" y="20" font-size="12">† denominator &lt; '
        f"{LOW_SUPPORT_THRESHOLD}</text>"
        if panel.get("low_support", pd.Series(dtype=bool)).astype(str).str.lower().eq("true").any()
        else ""
    )
    safe_title = _safe_text(title)
    safe_metric = _safe_text(metric_label)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{safe_title}"><title>{safe_title}</title>'
        f'<rect width="100%" height="100%" fill="white"/>'
        f'<text x="{width/2}" y="24" text-anchor="middle" '
        f'font-size="16" font-weight="600">{safe_title}</text>'
        f"{support_note}{grid}"
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" '
        f'y2="{height-bottom}" stroke="#555"/>'
        f'<line x1="{left}" y1="{top}" x2="{left}" '
        f'y2="{height-bottom}" stroke="#555"/>'
        f'{"".join(labels)}{"".join(marks)}'
        f'<text x="{width/2}" y="{height-14}" text-anchor="middle" '
        f'font-size="13">{safe_metric} — proportion (0–1)</text>'
        f'<text x="18" y="{height/2}" text-anchor="middle" '
        f'font-size="13" transform="rotate(-90 18 {height/2})">'
        f'{_safe_text(audit_column)}</text></svg>'
    )


def build_group_audit_svgs(
    data: pd.DataFrame,
    metric_labels: dict[str, str],
    model_name: str,
    threshold: float,
) -> dict[str, dict[str, str]]:
    """Retorna gráficos marginais e interseccionais por métrica.

    O retorno tem duas chaves: ``marginal`` para sexo/raça e
    ``intersection`` para ``sex_race``. Cada bloco contém um SVG por métrica.
    """
    required = {
        "model",
        "threshold",
        "audit_column",
        "metric",
        "group",
        "count",
        "denominator",
        "estimate",
        "ci_low",
        "ci_high",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing group-audit columns: {sorted(missing)}")
    filtered = data.loc[
        data["model"].eq(model_name) & data["threshold"].eq(threshold)
    ].copy()
    if filtered.empty:
        raise ValueError("No group-audit rows match model and threshold")
    marginal = {}
    intersection = {}
    for metric, label in metric_labels.items():
        marginal[metric] = _panel_svg(
            filtered,
            metric,
            label,
            f"{label} por sexo e raça — {model_name}, threshold={threshold:.2f}",
            model_name,
            "sex",
        )
        # A segunda marginalidade usa a mesma função e o mesmo contrato; o
        # notebook exibirá os dois SVGs lado a lado por métrica.
        marginal[f"race_{metric}"] = _panel_svg(
            filtered,
            metric,
            label,
            f"{label} por raça — {model_name}, threshold={threshold:.2f}",
            model_name,
            "race",
        )
        intersection[metric] = _panel_svg(
            filtered,
            metric,
            label,
            f"{label} por sexo × raça — {model_name}, threshold={threshold:.2f}",
            model_name,
            "sex_race",
        )
    return {"marginal": marginal, "intersection": intersection}
