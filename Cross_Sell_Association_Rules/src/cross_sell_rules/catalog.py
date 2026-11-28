"""Bilingual text catalog for the cross-sell analysis notebooks."""
from __future__ import annotations

from typing import Final


CATALOG: Final[dict[str, dict[str, str]]] = {
    "cooccurrence_report": {
        "en": (
            "Sausage with whole milk         : {sausage_milk}\n"
            "Whole milk with pastry          : {milk_pastry}\n"
            "Sausage with hygiene articles   : {sausage_hygiene}"
        ),
        "pt": (
            "Salsicha com leite integral    : {sausage_milk}\n"
            "Leite integral com pastelaria  : {milk_pastry}\n"
            "Salsicha com artigos de higiene: {sausage_hygiene}"
        ),
    },
    "dataset_summary": {
        "en": (
            "Our dataset has:\n"
            "  — {occurrences} total item occurrences  "
            "(individual records in the CSV file)\n"
            "  — {transactions} unique transactions   "
            "(shopping baskets, grouped by Member_number × Date)\n"
            "  — {items} distinct items in the catalog\n\n"
            "With {items} distinct items, the itemset space grows "
            "combinatorially:\n"
            "  - 1 itemset        : the null set, with no items.\n"
            "  - {items} itemsets        : a single item.\n"
            "  - Approx. {duo}  itemsets: in pairs.\n"
            "  - Nearly  {trio} itemsets: in triplets.\n"
            "  - Nearly  {quartet} itemsets: in quartets."
        ),
        "pt": (
            "Nosso dataset possui:\n"
            "  — {occurrences} ocorrências totais de itens  "
            "(registros individuais no arquivo CSV)\n"
            "  — {transactions} transações únicas       "
            "(cestas de compra, agrupadas por Member_number × Date)\n"
            "  — {items} itens distintos no catálogo\n\n"
            "Com {items} itens distintos, o espaço de itemsets cresce "
            "combinatoriamente:\n"
            "  - 1 itemset        : o conjunto nulo, sem itens.\n"
            "  - {items} itemsets        : um item apenas.\n"
            "  - Aprox. {duo}  itemsets: em duplas.\n"
            "  - Quase  {trio} itemsets: em trios.\n"
            "  - Quase  {quartet} itemsets: em quartetos."
        ),
    },
    "total_item_occurrences": {
        "en": "Total item occurrences",
        "pt": "Ocorrências totais de itens",
    },
    "distinct_items": {
        "en": "Distinct items",
        "pt": "Itens distintos",
    },
    "support_heatmap": {
        "en": "Antecedents and Consequents — Heatmap by Support",
        "pt": "Antecedentes e Consequentes — Mapa de Calor por Suporte",
    },
    "lift_heatmap": {
        "en": "Antecedents and Consequents — Heatmap by Lift",
        "pt": "Antecedentes e Consequentes — Mapa de Calor por Lift",
    },
    "support_confidence_scatter": {
        "en": "Support vs Confidence — Scatter Plot by Lift",
        "pt": "Suporte vs Confiança — Gráfico de Dispersão por Lift",
    },
    "parallel_coordinates": {
        "en": "Parallel Coordinates",
        "pt": "Coordenadas Paralelas",
    },
}


def translate(key: str, language: str) -> str:
    """Return the catalog text for ``key`` and a supported language."""
    if language not in {"en", "pt"}:
        raise ValueError("language must be 'en' or 'pt'")
    try:
        return CATALOG[key][language]
    except KeyError as exc:
        raise KeyError(f"unknown catalog key: {key}") from exc
