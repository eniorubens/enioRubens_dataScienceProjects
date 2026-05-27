"""
multilang.py
------------
Lightweight internationalisation for chart labels, axis titles, print()
and display() outputs.

Design principles
-----------------
- English is the canonical source language. All strings are written in EN.
- Portuguese (PT) translations are stored in a static dictionary: no runtime
  API calls, no latency, no network dependency during notebook execution.
- If a string is missing from the PT dictionary the EN original is returned,
  so missing translations degrade gracefully instead of raising exceptions.
- The active language is set once (via set_language / LANG env var) and
  respected by every module that imports `t()`.

Usage
-----
    from src.multilang import t, set_language

    set_language("pt")          # switch to Portuguese
    ax.set_ylabel(t("Customers"))       # → "Clientes"
    ax.set_xlabel(t("Days since last purchase"))  # → "Dias desde a última compra"

Adding translations
-------------------
Add a new key-value pair to the PT dict below. Key = EN string,
value = PT translation. Keys are case-sensitive.
"""

from __future__ import annotations

import os

# ── Supported languages ───────────────────────────────────────────────────
SUPPORTED: set[str] = {"en", "pt"}

# Module-level active language (default: English)
_LANG: str = os.environ.get("NOTEBOOK_LANG", "en").lower()


def set_language(lang: str) -> None:
    """Set the active language for all subsequent t() calls.

    Parameters
    ----------
    lang : str
        Language code. Supported values: ``"en"``, ``"pt"``.

    Raises
    ------
    ValueError
        If ``lang`` is not in the supported set.
    """
    global _LANG
    lang = lang.lower()
    if lang not in SUPPORTED:
        raise ValueError(f"Language '{lang}' not supported. Choose from {SUPPORTED}.")
    _LANG = lang


def get_language() -> str:
    """Return the currently active language code."""
    return _LANG


def t(text: str | dict[str, str]) -> str | dict[str, str]:
    """Translate *text* from English to the active language.

    Accepted inputs:
    - ``str``: returns a translated string (or the original if missing).
    - ``dict[str, str]``: returns a new dict with translated values.

    Dict support keeps compatibility with ``LangMap``-style usage from
    older notebooks while preserving the lightweight local translation flow.
    """
    if isinstance(text, dict):
        return {key: t(value) for key, value in text.items()}

    if not isinstance(text, str):
        raise TypeError("t() accepts str or dict[str, str].")

    if _LANG == "en":
        return text
    return _TRANSLATIONS.get(_LANG, {}).get(text, text)


# ── Translation dictionaries ───────────────────────────────────────────────
# Format: _TRANSLATIONS[lang_code][english_string] = translated_string
_TRANSLATIONS: dict[str, dict[str, str]] = {
    "pt": {
        # ── Axis labels ───────────────────────────────────────────────────
        "Customers":                        "Clientes",
        "Revenue":                          "Receita",
        "Number of clusters":               "Número de clusters",
        "Days since last purchase":         "Dias desde a última compra",
        "Purchase frequency":               "Frequência de compra",
        "Total spend":                      "Valor total gasto",
        "Average ticket":                   "Ticket médio",
        "Total quantity":                   "Quantidade total",
        "Unique products":                  "Produtos únicos",
        "Recency":                          "Recência",
        "Frequency":                        "Frequência",
        "Monetary":                         "Monetário",
        "Recency (days)":                   "Recência (dias)",
        "Frequency (invoices)":             "Frequência (faturas)",
        "Monetary ($)":                     "Monetário ($)",
        "inertia":                          "Inércia",
        "silhouette_score":                 "Score de Silhueta",
        "Probability":                      "Probabilidade",
        "Expected profit ($)":              "Lucro esperado ($)",
        "Incremental profit ($)":           "Lucro incremental ($)",
        "Available budget ($)":             "Budget disponível ($)",
        "ROI":                              "ROI",
        "Repurchase probability":           "Probabilidade de recompra",
        "Mean predicted probability (Positive class: 1)":
            "Probabilidade média prevista (Classe positiva: 1)",
        "Fraction of positives (Positive class: 1)":
            "Fração de positivos (Classe positiva: 1)",
        "Segment":                          "Segmento",
        "Action":                           "Ação",
        "Incentive cost ($)":               "Custo do incentivo ($)",
        "Cluster":                          "Cluster",
        "TotalRevenue":                     "Receita Total",
        "AvgProb":                          "Probabilidade Média",
        "TotalCost":                        "Custo Total",
        "BudgetSaved":                      "Orçamento Economizado",
        "TotalIncrementalProfit":           "Lucro Incremental Total",

        # ── Chart titles ──────────────────────────────────────────────────
        "Recency Distribution":             "Distribuição de Recência",
        "Frequency Distribution":           "Distribuição de Frequência",
        "Monetary Distribution":            "Distribuição Monetária",
        "Average Ticket Distribution":      "Distribuição do Ticket Médio",
        "Top 10 Customers by Monetary":     "Top 10 Clientes por Monetário",
        "Elbow Method":                     "Método do Cotovelo",
        "Silhouette Score by k":            "Silhouette Score por k",
        "Frequency vs Monetary by Cluster Color":
            "Frequência vs Monetário por Cor do Cluster",
        "Recency vs Monetary by Cluster Color":
            "Recência vs Monetário por Cor do Cluster",
        "Monetary Boxplot by Cluster":      "Boxplot de Monetário por Cluster",
        "Recency Boxplot by Cluster":       "Boxplot de Recência por Cluster",
        "Revenue by Country":               "Receita por País",
        "Revenue by Segment":               "Receita por Segmento",
        "Customers by Segment":             "Clientes por Segmento",
        "Repurchase Probability by Segment":"Probabilidade de Recompra por Segmento",
        "Budget Allocation by Segment":     "Alocação de Orçamento por Segmento",
        "Budget Saved by Do-Nothing Decision":
            "Orçamento Economizado pela Decisão de Não Agir",
        "Total Incremental Profit by Action Type":
            "Lucro Incremental Total por Tipo de Ação",
        "Best Recommended Action by Temporal Cluster":
            "Melhor Ação Recomendada por Cluster Temporal",
        "Repurchase Probability vs Monetary":
            "Probabilidade de Recompra vs Monetário",
        "Budget vs Incremental Profit":     "Budget vs Lucro Incremental",
        "Budget vs ROI":                    "Budget vs ROI",
        "Calibration Curve":                "Curva de Calibração",
        "Without calibration":              "Sem calibração",
        "Calibrated":                       "Calibrado",
        "Perfectly calibrated":             "Perfeitamente calibrado",
        "ROC Curve":                        "Curva ROC",

        # ── Chart subtitles ───────────────────────────────────────────────
        "Inertia drop as k increases.":
            "Queda da inércia conforme k aumenta.",
        "Silhouette peaks at the optimal k.":
            "Silhouette atinge o pico no k ótimo.",
        "Revenue concentration across behavioural groups.":
            "Concentração de receita por grupos comportamentais.",
        "Revenue concentration across top countries.":
            "Concentração de receita entre os principais países.",
        "Customers with highest accumulated revenue.":
            "Clientes com maior receita acumulada.",
        "Axes in log scale to reduce outlier effect.":
            "Eixos em escala log para reduzir o efeito de outliers.",
        "Monetary in log scale; lower Recency means more recent customer.":
            "Monetário em escala log; menor Recência indica cliente mais recente.",
        "Median, quartiles and whiskers using 1.5 IQR criterion.":
            "Mediana, quartis e whiskers usando critério de 1,5 IQR.",
        "Lower values indicate more recent customers.":
            "Valores menores indicam clientes mais recentes.",
        "Size of each behavioural segment.":
            "Tamanho de cada segmento comportamental.",
        "Clusters computed only from pre-cutoff data.":
            "Clusters calculados apenas com dados pré-corte.",
        "Distribution clipped at 99th percentile for readability.":
            "Distribuição truncada no percentil 99 para legibilidade.",
        "Total cost of incentives selected by the greedy heuristic.":
            "Custo total dos incentivos selecionados pela heurística gulosa.",
        "Estimated savings against a uniform campaign applied to the entire base.":
            "Economia estimada em relação a uma campanha uniforme aplicada a toda a base.",
        "Difference between best action and No Action, based on uplift hypotheses.":
            "Diferença entre a melhor ação e No Action, com base em hipóteses de uplift.",
        "Total accumulated return from prescriptive margin-based selection.":
            "Retorno acumulado total pela seleção prescritiva baseada em margem.",
        "Average investment efficiency as coverage increases.":
            "Eficiência média do investimento conforme a cobertura aumenta.",
        "Fraction of population with elbow-method guidance.":
            "Fração da população com guia pelo método do cotovelo.",

        # ── Table / display column names ──────────────────────────────────
        "Step":                             "Etapa",
        "Before":                           "Antes",
        "After":                            "Depois",
        "Removed":                          "Removidos",
        "% removed":                        "% removido",
        "Model":                            "Modelo",
        "Sample":                           "Amostra",
        "Segment Name":                     "Nome do Segmento",
        "Customers (n)":                    "Clientes (n)",
        "Revenue Share":                    "Participação na Receita",
        "Avg Recency":                      "Recência Média",
        "Avg Frequency":                    "Frequência Média",
        "Avg Monetary":                     "Monetário Médio",
        "Recommended Action":               "Ação Recomendada",
        "Expected Profit":                  "Lucro Esperado",
        "Incremental Profit":               "Lucro Incremental",
        "Priority Rank":                    "Rank de Prioridade",
        "Budget Percentile":                "Percentil do Budget",

        # ── Print / narrative strings ─────────────────────────────────────
        "Reference date: {}":              "Data de referência: {}",
        "RFM table: {} customers":          "Tabela RFM: {} clientes",
        "Clean dataset: {} rows and {} columns":
            "Dataset limpo: {} linhas e {} colunas",
        "Unique customers: {}":             "Clientes únicos: {}",
        "Total revenue: {}":               "Receita total: {}",
        "Export generated: {}":            "Exportação gerada: {}",
        "Customers exported: {}":          "Clientes exportados: {}",
        "Segments covered: {}":            "Segmentos cobertos: {}",
        "**Number of clusters chosen: {selected_k}.**\n\nThe highest silhouette was observed at `k={best_k}`. `k={selected_k}` was chosen because it offers more actionable segments for Next Best Action, separating high-value customers, loyal customers, occasional buyers, low-value customers, inactive customers and at-risk high-value customers. The decision prioritizes a balance between statistical quality, interpretability and business utility.":
            "**Número de clusters escolhido: {selected_k}.**\n\nA maior silhueta foi observada em `k={best_k}`. `k={selected_k}` foi escolhido porque oferece segmentos mais acionáveis para Next Best Action, separando clientes de alto valor, clientes fiéis, compradores ocasionais, clientes de baixo valor, clientes inativos e clientes de alto valor em risco. A decisão prioriza equilíbrio entre qualidade estatística, interpretabilidade e utilidade de negócio.",
        "optimal point":                   "ponto ótimo",
        "current budget":                  "budget atual",

        # ── Prescriptive recommendation templates ─────────────────────────
        "preserve margin; incentive does not increase expected profit.":
            "preservar margem; incentivo não aumenta o lucro esperado.",
        "Maximum retention priority with aggressive campaign.":
            "Máxima prioridade de retenção com campanha agressiva.",
        "Prioritize relationship with {}.":
            "Priorizar relacionamento com {}.",
        "Review cost before aggressive campaign for low-value customer.":
            "Revisar custo antes de campanha agressiva para cliente de baixo valor.",
        "Test controlled reactivation with {}.":
            "Testar reativação controlada com {}.",
        "Execute {} based on positive expected profit.":
            "Executar {} com base em lucro esperado positivo.",

        # ── Segment names ─────────────────────────────────────────────────
        "Champions":            "Campeões",
        "Loyal Customers":      "Clientes Fiéis",
        "High Value at Risk":   "Alto Valor em Risco",
        "Occasional Buyers":    "Compradores Ocasionais",
        "Low Value":            "Baixo Valor",
        "Inactive Customers":   "Clientes Inativos",
    }
}
