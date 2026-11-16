"""Portuguese and English labels for the presentation charts."""

from __future__ import annotations

LABELS: dict[str, dict[str, dict[str, str]]] = {
    "pt": {
        "folds": {
            "title": "Folds expanding-window: linha do tempo e prevalência",
            "fold": "Fold",
            "train": "Treino",
            "embargo": "Embargo",
            "validation": "Validação",
            "orders": "pedidos",
            "date": "Data de compra",
            "prevalence": "Prevalência de violação de SLA",
            "prevalence_panel": "Prevalência por fold",
            "timeline_panel": "Períodos por fold",
        },
        "psi": {
            "title": "Deriva das features (PSI)",
            "xaxis": "PSI (escala log)",
            "yaxis": "Feature",
            "neutral": "calendário (PSI alto por construção)",
            "moderate": "0,10",
            "reference": "Referência",
            "large": "0,25",
            "other": "demais features",
            "promise": "prazo prometido",
            "network_rate": "taxas históricas da rede",
        },
        "intervals": {
            "title_single": "Estimativa e IC 95% por modelo",
            "title_paired": "Estimativa com IC 95% e diferença pareada",
            "estimate_panel": "Estimativa e IC 95%",
            "paired_panel": "Diferença pareada",
            "difference": "Diferença",
            "reference": "Referência",
            "model": "Modelo",
            "zero": "sem diferença",
        },
        "metric_names": {"average_precision": "Precisão média (AP)", "roc_auc": "AUC ROC", "mae": "MAE (dias)"},
        "by_fold": {
            "title": "{metric} por fold",
            "fold": "Fold",
            "model": "Modelo",
            "reference": "referência",
            "lower_is_better": "menor é melhor",
        },
        "regime": {
            "title": "Regime da promessa por mês de compra",
            "month": "Mês de compra",
            "promise_median": "Mediana do prazo prometido (dias)",
            "auc": "AUC da regra do prazo prometido negativo",
            "chance": "acaso (AUC = 0,5)",
        },
        "availability": {
            "title": "Rótulos conhecidos e desconhecidos na data de atualização",
            "month": "Mês de compra",
            "orders": "Pedidos",
            "known": "Rótulo conhecido",
            "unknown": "Rótulo desconhecido",
            "prevalence": "prevalência",
        },
        "gain": {
            "title": "Ganho médio sobre a melhor baseline",
            "xaxis": "Modelo",
            "yaxis": "Ganho relativo (%)",
            "threshold": "critério",
            "folds_won": "folds vencidos",
        },
        "ablation": {
            "title": "Ablação por variante",
            "xaxis": "Variante",
            "yaxis": "Média entre folds",
            "model": "Modelo",
        },
        "routes": {
            "title_network": "Rede de rotas entre UFs: volume e taxa de atraso",
            "title_matrix": "Taxa de atraso por rota (principais UFs de origem; células com menos de {min_cell} pares em branco)",
            "title_cities": "As {top} maiores rotas entre cidades, com IC 95% de Wilson",
            "colorbar": "Taxa de atraso",
            "min_volume": "Volume mínimo:",
            "node": "UF",
            "origin": "Origem",
            "destination": "Destino",
            "route": "Rota (origem → destino)",
            "orders": "Pares pedido–vendedor",
            "breaches": "Violações de SLA",
            "rate": "Taxa de atraso",
            "ci": "IC 95%",
            "share": "Participação no volume total",
            "distance": "Mediana da distância Haversine",
            "promised": "Mediana do prazo prometido",
            "actual": "Mediana do prazo realizado",
            "days": "dias",
            "sent": "Pares enviados",
            "received": "Pares recebidos",
            "rate_out": "Taxa de atraso de saída",
            "rate_in": "Taxa de atraso de entrada",
            "loop": "Pares dentro da própria UF",
            "sellers": "Vendedores",
            "customers": "Clientes",
            "out_degree": "Grau de saída (rotas)",
            "in_degree": "Grau de entrada (rotas)",
            "out_strength": "Força de saída (pares)",
            "in_strength": "Força de entrada (pares)",
        },
    },
    "en": {
        "folds": {
            "title": "Expanding-window folds: timeline and prevalence",
            "fold": "Fold",
            "train": "Train",
            "embargo": "Embargo",
            "validation": "Validation",
            "orders": "orders",
            "date": "Purchase date",
            "prevalence": "SLA breach prevalence",
            "prevalence_panel": "Prevalence by fold",
            "timeline_panel": "Periods by fold",
        },
        "psi": {
            "title": "Feature drift (PSI)",
            "xaxis": "PSI (log scale)",
            "yaxis": "Feature",
            "neutral": "calendar (high PSI by construction)",
            "moderate": "0.10",
            "reference": "Reference",
            "large": "0.25",
            "other": "other features",
            "promise": "promised lead time",
            "network_rate": "historical network rates",
        },
        "intervals": {
            "title_single": "Estimate and 95% CI by model",
            "title_paired": "Estimate with 95% CI and paired difference",
            "estimate_panel": "Estimate and 95% CI",
            "paired_panel": "Paired difference",
            "difference": "Difference",
            "reference": "Reference",
            "model": "Model",
            "zero": "no difference",
        },
        "metric_names": {"average_precision": "Average precision (AP)", "roc_auc": "ROC AUC", "mae": "MAE (days)"},
        "by_fold": {
            "title": "{metric} by fold",
            "fold": "Fold",
            "model": "Model",
            "reference": "reference",
            "lower_is_better": "lower is better",
        },
        "regime": {
            "title": "Promise regime by purchase month",
            "month": "Purchase month",
            "promise_median": "Median promised lead time (days)",
            "auc": "AUC of the negative promised lead time rule",
            "chance": "chance (AUC = 0.5)",
        },
        "availability": {
            "title": "Known and unknown labels at the update date",
            "month": "Purchase month",
            "orders": "Orders",
            "known": "Known label",
            "unknown": "Unknown label",
            "prevalence": "prevalence",
        },
        "gain": {
            "title": "Mean gain over the best baseline",
            "xaxis": "Model",
            "yaxis": "Relative gain (%)",
            "threshold": "criterion",
            "folds_won": "folds won",
        },
        "ablation": {
            "title": "Ablation by variant",
            "xaxis": "Variant",
            "yaxis": "Mean across folds",
            "model": "Model",
        },
        "routes": {
            "title_network": "State route network: volume and breach rate",
            "title_matrix": "Breach rate by route (top origin states; cells with fewer than {min_cell} pairs left blank)",
            "title_cities": "The {top} largest city-to-city routes, with 95% Wilson CI",
            "colorbar": "Breach rate",
            "min_volume": "Minimum volume:",
            "node": "State",
            "origin": "Origin",
            "destination": "Destination",
            "route": "Route (origin → destination)",
            "orders": "Order–seller pairs",
            "breaches": "SLA breaches",
            "rate": "Breach rate",
            "ci": "95% CI",
            "share": "Share of total volume",
            "distance": "Median Haversine distance",
            "promised": "Median promised lead time",
            "actual": "Median actual lead time",
            "days": "days",
            "sent": "Pairs sent",
            "received": "Pairs received",
            "rate_out": "Outbound breach rate",
            "rate_in": "Inbound breach rate",
            "loop": "Pairs within the same state",
            "sellers": "Sellers",
            "customers": "Customers",
            "out_degree": "Out-degree (routes)",
            "in_degree": "In-degree (routes)",
            "out_strength": "Out-strength (pairs)",
            "in_strength": "In-strength (pairs)",
        },
    },
}


def get_labels(language: str = "pt") -> dict[str, dict[str, str]]:
    """Return the label dictionary for a language.

    Parameters
    ----------
    language : str
        ``"pt"`` or ``"en"``.

    Returns
    -------
    dict
        Nested label dictionary, one section per chart.

    Raises
    ------
    ValueError
        If the language is not supported.
    """
    if language not in LABELS:
        raise ValueError(f"unsupported language: {language!r}; use 'pt' or 'en'")
    return LABELS[language]
