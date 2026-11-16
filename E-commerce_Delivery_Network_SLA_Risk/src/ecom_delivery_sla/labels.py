"""Language labels shared by the Portuguese and English notebooks."""

from __future__ import annotations

LABELS = {
    "pt": {
        "inventory": {"orders": "pedidos", "customers": "clientes", "items": "itens", "products": "produtos", "payments": "pagamentos", "reviews": "avaliações", "sellers": "vendedores", "geolocation": "geolocalização", "categories": "tradução de categorias"},
        "target_metrics": [
            "Pedidos supervisionáveis", "Violações de SLA", "Taxa de violação",
            "Mediana do atraso (dias)", "P95 do atraso (dias)", "P99 do atraso (dias)",
            "Atraso > 30 dias", "Antecipação < -30 dias",
        ],
        "lead_measures": ["Prazo prometido", "Prazo realizado"],
        "columns": {"metric": "métrica", "value": "valor", "measure": "medida", "median": "mediana_dias", "p95": "p95_dias", "month": "mês_da_compra", "orders": "pedidos", "breaches": "violacoes", "median_delay": "atraso_mediano", "p95_delay": "atraso_p95", "breach_rate": "taxa_violacao", "state": "estado_cliente", "category": "categoria_modal", "rule": "regra", "cases": "casos"},
        "plots": {"delay": "Distribuição do atraso em relação à data estimada", "lead": "Atraso observado e prazo prometido", "monthly": "Taxa mensal de violação por mês da compra", "state": "Taxa de violação por estado do cliente", "category": "Categorias com maior taxa descritiva de violação"},
        "axes": {"delay": "Atraso (dias)", "orders": "Pedidos", "promised": "Prazo prometido desde a compra (dias)", "relative_delay": "Atraso em relação ao SLA (dias)", "breach": "Violação de SLA", "month": "Mês da compra", "state": "Estado", "category": "Categoria modal"},
        "anomaly_rules": ["Entrega ao cliente antes do despacho", "Despacho ausente", "Aprovação antes da compra", "Entrega ao cliente antes da compra"],
        "geo": {"distance_band": "faixa de distância (km)", "breach_rate": "taxa de violação", "distance_title": "Taxa de violação por distância aproximada", "composition": "composição do pedido", "category": "composição por categorias", "items_title": "Taxa de violação por quantidade de itens", "products_title": "Taxa de violação por produtos distintos", "category_title": "Taxa de violação por variedade de categorias", "sellers_title": "Taxa de violação por quantidade de vendedores"},
        "temporal": {"title": "Divisão cronológica para modelagem", "xaxis": "Data da compra", "train": "Treino", "validation": "Validação", "test": "Teste"},
    },
    "en": {
        "inventory": {"orders": "orders", "customers": "customers", "items": "items", "products": "products", "payments": "payments", "reviews": "reviews", "sellers": "sellers", "geolocation": "geolocation", "categories": "category translation"},
        "target_metrics": [
            "Supervised orders", "SLA breaches", "Breach rate",
            "Median delay (days)", "P95 delay (days)", "P99 delay (days)",
            "Delay > 30 days", "Early delivery < -30 days",
        ],
        "lead_measures": ["Promised lead time", "Actual lead time"],
        "columns": {"metric": "metric", "value": "value", "measure": "measure", "median": "median_days", "p95": "p95_days", "month": "purchase_month", "orders": "orders", "breaches": "breaches", "median_delay": "median_delay", "p95_delay": "p95_delay", "breach_rate": "breach_rate", "state": "customer_state", "category": "dominant_category", "rule": "rule", "cases": "cases"},
        "plots": {"delay": "Delivery delay relative to estimated date", "lead": "Observed delay and promised lead time", "monthly": "Monthly SLA breach rate by purchase month", "state": "Breach rate by customer state", "category": "Categories with the highest descriptive breach rate"},
        "axes": {"delay": "Delay (days)", "orders": "Orders", "promised": "Promised lead time from purchase (days)", "relative_delay": "Delay relative to SLA (days)", "breach": "SLA breach", "month": "Purchase month", "state": "State", "category": "Dominant category"},
        "anomaly_rules": ["Customer delivery before carrier handoff", "Carrier handoff missing", "Approval before purchase", "Customer delivery before purchase"],
        "geo": {"distance_band": "approximate distance band (km)", "breach_rate": "breach rate", "distance_title": "Breach rate by approximate distance", "composition": "order composition", "category": "category composition", "items_title": "Breach rate by item count", "products_title": "Breach rate by distinct products", "category_title": "Breach rate by category variety", "sellers_title": "Breach rate by seller count"},
        "temporal": {"title": "Chronological modeling split", "xaxis": "Purchase date", "train": "Training", "validation": "Validation", "test": "Test"},
    },
}


def get_labels(language: str = "pt") -> dict:
    """Return labels for ``pt`` or ``en``."""
    normalized = language.lower().replace("-br", "").replace("-us", "")
    if normalized not in LABELS:
        raise ValueError("language must be 'pt' or 'en'")
    return LABELS[normalized]
