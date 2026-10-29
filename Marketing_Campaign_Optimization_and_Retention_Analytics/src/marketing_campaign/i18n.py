"""Portuguese labels for the tables and texts produced by the analysis.

The package computes everything with English names. The Portuguese
notebook asks for the same results and localises them here, so both
notebooks share one implementation.
"""
from __future__ import annotations

import unicodedata

import pandas as pd

SUPPORTED_LANGUAGES = ("en", "pt")

SEGMENT_LABELS = {
    "en": ["High Value", "Medium Value", "Low Value"],
    "pt": ["Alto Valor", "Médio Valor", "Baixo Valor"],
}

COLUMNS_PT = {
    "Column": "Coluna",
    "% Missing": "% Faltantes",
    "Metric": "Métrica",
    "Value": "Valor",
    "Parameter": "Parametro",
    "impacted_users": "impactados",
    "conversions": "conversoes",
    "retained": "retidos",
    "cancellations": "cancelamentos",
    "Variant_A": "Variante_A",
    "Variant_B": "Variante_B",
    "Significance": "Significante",
    "total_cost": "custo_total",
    "estimated_revenue": "receita_estimada",
    "incremental_revenue": "receita_incremental",
    "estimated_profit": "lucro_estimado",
    "majority_segment": "segmento_majoritario",
    "ltv_reference": "ltv_referencia",
    "coefficient": "coeficiente",
    "absolute_coefficient": "coeficiente_abs",
    "quantity": "quantidade",
    "percentage": "percentual",
    "category": "categoria",
    "base_count": "qtd_base",
    "base_pct": "perc_base",
    "false_positive_count": "qtd_false_positive",
    "false_positive_pct": "perc_false_positive",
    "interpretation": "interpretacao",
}

VALUES_PT = {
    "Impacted Users": "Usuários Impactados",
    "Conversions": "Conversões",
    "Retained (among converted)": "Retidos (entre convertidos)",
    "Conversion Rate (%)": "Taxa de Conversão (%)",
    "Retention Rate (%)": "Taxa de Retenção (%)",
    "High Value": "Alto Valor",
    "Medium Value": "Médio Valor",
    "Low Value": "Baixo Valor",
    "Not segmented": "Nao segmentado",
    "Unidentified segment": "Segmento nao identificado",
    "YES": "SIM",
    "NO": "NÃO",
    "Correct Language": "Idioma Correto",
    "Wrong Language": "Idioma Incorreto",
    "Not informed": "Não informado",
    "Over-represented in false positives": (
        "Super-representado nos falsos positivos"
    ),
    "Under-represented in false positives": (
        "Sub-representado nos falsos positivos"
    ),
    "Proportional distribution": "Distribuicao proporcional",
    "Loyalty Campaign": "Campanha de Fidelização",
    "Promotional Offer": "Oferta Promocional",
    "Win-Back Campaign or No Action": "Campanha de Win-Back ou Nao Agir",
    "No Action - Negative Incremental Profit": (
        "Nao Agir - Lucro Incremental Negativo"
    ),
    "High-value customers - maintain engagement and reduce churn.": (
        "Clientes de alto valor - manter engajamento e reduzir churn."
    ),
    "Customers with potential - increase conversion and average ticket.": (
        "Clientes com potencial - aumentar conversão e ticket medio."
    ),
    "Low ROI - evaluate whether reactivation cost is justified.": (
        "ROI baixo - avaliar se custo de reativação se justifica."
    ),
}
FIX_LANGUAGE_PREFIX = {
    "en": "Fix Language -> then ",
    "pt": "Corrigir Idioma -> depois ",
}
VALUE_COLUMNS = {
    "Metric",
    "Segment",
    "majority_segment",
    "Significance",
    "label",
    "interpretation",
    "recommended_action",
    "strategy_rationale",
}

_CANONICAL_SEGMENTS = {
    "high value": "High Value",
    "alto valor": "High Value",
    "medium value": "Medium Value",
    "medio valor": "Medium Value",
    "low value": "Low Value",
    "baixo valor": "Low Value",
}


def _fold(text: str) -> str:
    """Lower-case a text and strip its accents.

    Parameters
    ----------
    text : str
        Any text.

    Returns
    -------
    str
        The text in lower case without diacritics.
    """
    decomposed = unicodedata.normalize("NFKD", text.strip().lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def canonical_segment(name: object) -> str | None:
    """Return the English name of a segment given in any supported language.

    Parameters
    ----------
    name : object
        Segment name such as ``"Alto Valor"``, ``"medio valor"`` or
        ``"Low Value"``.

    Returns
    -------
    str or None
        ``"High Value"``, ``"Medium Value"`` or ``"Low Value"``, or
        ``None`` when the name is not a known segment.
    """
    if not isinstance(name, str):
        return None
    return _CANONICAL_SEGMENTS.get(_fold(name))


def translate_text(text: object, lang: str = "pt") -> object:
    """Translate one known text, leaving anything else unchanged.

    Parameters
    ----------
    text : object
        A label, segment, action or rationale produced by the package.
    lang : {"en", "pt"}, default="pt"
        Target language.

    Returns
    -------
    object
        The translated text; ``text`` itself when ``lang`` is ``"en"`` or
        the text is unknown. Actions that start with the language-fix
        prefix keep the prefix and translate the action that follows.
    """
    if lang == "en" or not isinstance(text, str):
        return text
    prefix = FIX_LANGUAGE_PREFIX["en"]
    if text.startswith(prefix):
        action = text[len(prefix):]
        return FIX_LANGUAGE_PREFIX[lang] + str(translate_text(action, lang))
    return VALUES_PT.get(text, text)


def localize(df: pd.DataFrame, lang: str = "pt") -> pd.DataFrame:
    """Translate the column names and known labels of a result table.

    Parameters
    ----------
    df : pd.DataFrame
        A table produced by the package, with English names.
    lang : {"en", "pt"}, default="pt"
        Target language.

    Returns
    -------
    pd.DataFrame
        A copy of ``df``. For ``"pt"`` the columns listed in
        :data:`COLUMNS_PT` are renamed and the texts in the columns of
        :data:`VALUE_COLUMNS` are translated; other columns and values
        are untouched.

    Raises
    ------
    ValueError
        If ``lang`` is not supported.
    """
    if lang not in SUPPORTED_LANGUAGES:
        raise ValueError(f"lang must be one of {SUPPORTED_LANGUAGES}.")
    localized = df.copy()
    if lang == "en":
        return localized
    for column in VALUE_COLUMNS.intersection(localized.columns):
        localized[column] = localized[column].map(
            lambda value: translate_text(value, lang)
        )
    return localized.rename(columns=COLUMNS_PT)
