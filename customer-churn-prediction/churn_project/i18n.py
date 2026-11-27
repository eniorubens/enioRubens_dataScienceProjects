"""Local, deterministic translations used by the churn project."""

from __future__ import annotations

import os
from typing import Mapping


SUPPORTED_LANGUAGES = {"en", "pt"}
_LANGUAGE = os.environ.get("CHURN_LANGUAGE", "en").lower()

_PT: dict[str, str] = {
    "Insight: even a modest monthly churn rate compounds into substantial annual revenue loss.":
        "Insight: mesmo uma taxa de churn mensal modesta se acumula em uma perda de receita anual substancial.",
    "Number of Customers in Each Churn Class": "Número de Clientes em Cada Classe de Churn",
    "Class distribution of the target variable": "Distribuição de classes da variável alvo",
    "the churn class is smaller, so model evaluation should prioritize recall and F1-score over accuracy alone.":
        "a classe de churn é menor, portanto a avaliação do modelo deve priorizar recall e F1-score em vez de apenas acurácia.",
    "Pairplot of Numerical Features by Contract Type": "Pairplot das Features Numéricas por Tipo de Contrato",
    "Distribution and relationships across key numerical variables":
        "Distribuição e relações entre as principais variáveis numéricas",
    "Insight: tenure and total charges show strong linear structure, indicating accumulated billing behavior.":
        "Insight: tenure e total de cobranças mostram forte estrutura linear, indicando comportamento de cobrança acumulada.",
    "Gender Distribution": "Distribuição por Gênero",
    "Customer base is evenly distributed across genders":
        "A base de clientes está distribuída de forma equilibrada entre os gêneros",
    "Insight: gender appears balanced and is unlikely to explain churn on its own.":
        "Insight: o gênero parece equilibrado e provavelmente não explica o churn por si só.",
    "Customers with Dependents and/or Partners": "Clientes com Dependentes e/ou Parceiros",
    "Customer distribution by household composition": "Distribuição de clientes por composição familiar",
    "Insight: partner status is more common than dependent status, and the overlap between both is only partial.":
        "Insight: ter parceiro é mais comum do que ter dependentes, e a sobreposição entre ambos é apenas parcial.",
    "Number of Customers by Tenure": "Número de Clientes por Tenure",
    "Customer retention duration distribution": "Distribuição da duração de retenção dos clientes",
    "Insight: tenure shows concentration at both early and long customer lifecycles, indicating heterogeneous retention profiles.":
        "Insight: o tenure mostra concentração tanto nos primeiros quanto nos longos ciclos de vida dos clientes, indicando perfis de retenção heterogêneos.",
    "Number of Customers by Contract Type": "Número de Clientes por Tipo de Contrato",
    "Customer count by contract category": "Contagem de clientes por categoria de contrato",
    "Insight: month-to-month contracts dominate the customer base, while one-year and two-year contracts are less frequent.":
        "Insight: contratos mensais dominam a base de clientes, enquanto contratos de um e dois anos são menos frequentes.",
    "Tenure Distribution by Contract Type": "Distribuição de Tenure por Tipo de Contrato",
    "Customer tenure segmented by contract structure": "Tenure dos clientes segmentado pela estrutura contratual",
    "Insight: longer contract terms are associated with longer customer lifecycles and stronger retention.":
        "Insight: contratos de prazo mais longo estão associados a ciclos de vida mais longos e maior retenção.",
    "Services Distribution": "Distribuição de Serviços",
    "Customer adoption across subscribed services": "Adoção de clientes entre os serviços assinados",
    "Insight: basic connectivity services are widely adopted, while add-on services show more selective uptake.":
        "Insight: serviços básicos de conectividade são amplamente adotados, enquanto serviços adicionais mostram adoção mais seletiva.",
    "Pearson Correlation": "Correlação de Pearson",
    "Linear relationships among numerical variables": "Relações lineares entre variáveis numéricas",
    "Insight: Pearson correlation is useful for numerical variables, but it does not capture associations across categorical features.":
        "Insight: a correlação de Pearson é útil para variáveis numéricas, mas não captura associações entre features categóricas.",
    "Telecom PhiK Correlation": "Correlação PhiK - Telecom",
    "Association structure across numerical and categorical variables":
        "Estrutura de associação entre variáveis numéricas e categóricas",
    "Insight: PhiK captures association strength across mixed-type variables and is more suitable than Pearson for this dataset.":
        "Insight: o PhiK captura a força da associação em variáveis de tipo misto e é mais adequado que o Pearson para este dataset.",
    "Significance of the Coefficients": "Significância dos Coeficientes",
    "Statistical significance of observed associations": "Significância estatística das associações observadas",
    "Insight: correlation magnitude and statistical significance should be interpreted jointly.":
        "Insight: a magnitude da correlação e a significância estatística devem ser interpretadas em conjunto.",
    "Complete Dataset": "Dataset Completo",
    "Overall churn class distribution": "Distribuição geral das classes de churn",
    "Train/Test Split Validation": "Validação da Divisão Treino/Teste",
    "Class distribution preserved after stratified split":
        "Distribuição de classes preservada após divisão estratificada",
    "Monthly Charges": "Cobranças mensais",
    "Total Charges": "Total",
    "Senior Citizen": "Idoso",
    "Tenure": "Tenure",
    "Contract": "Contrato",
    "LogisticRegression Baseline's Confusion Matrix":
        "Matriz de Confusão do Baseline LogisticRegression",
    "Decision Function 0.46's Confusion Matrix":
        "Matriz de Confusão com Função de Decisão 0.46",
    "Winner": "Vencedor",
    "Encoder": "Codificador",
    "Test Recall": "Recall de Teste",
    "Winner: ": "Vencedor: ",
    "Confusion Matrix": "Matriz de Confusão",
    "Prediction breakdown:": "Distribuição das previsões:",
    "Correct": "Correto",
    "FN (Missed Churner)": "FN (Churner Perdido)",
    "FP (False Alarm)": "FP (Falso Alarme)",
    "Threshold = 0.50": "Limiar = 0,50",
    "True Churners: TP vs FN": "Churners Reais: VP vs FN",
    "Non-Churners: TN vs FP": "Não-Churners: VN vs FP",
    "Predicted Churn Probability": "Probabilidade Prevista de Churn",
    "Count": "Contagem",
    "Churn Probability Distribution by Prediction Outcome":
        "Distribuição de Probabilidade de Churn por Tipo de Resultado",
    "FN cluster near 0.40–0.49 (just below threshold); FP concentrate just above 0.50":
        "FN agrupados próximos de 0,40–0,49 (logo abaixo do limiar); FP concentrados logo acima de 0,50",
    "Telco Kaggle Dataset (test split)": "Dataset Telco Kaggle (partição de teste)",
    "Median Feature Profile: FN vs TP (True Churners)":
        "Perfil Mediano de Features: FN vs TP (Churners Reais)",
    "FN (Missed)": "FN (Perdido)",
    "TP (Caught)": "TP (Capturado)",
    "Contract Distribution in FN vs FP": "Distribuição de Contratos em FN vs FP",
    "Key finding: FN customers tend to have higher tenure and lower monthly charges":
        "Principal achado: clientes FN tendem a ter maior tenure e menores cobranças mensais",
    "than typical churners — they resemble non-churners on surface features, making":
        "do que churners típicos — eles se parecem com não-churners nas features superficiais, tornando-os",
    "them the hardest cases for any churn model to identify.":
        "os casos mais difíceis de identificar para qualquer modelo de churn.",
}


def set_language(language: str) -> None:
    """Set the deterministic language used by project modules."""
    global _LANGUAGE
    normalized = language.lower()
    if normalized not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")
    _LANGUAGE = normalized


def get_language() -> str:
    return _LANGUAGE


def t(text: str | Mapping[str, str]) -> str | dict[str, str]:
    """Translate a string or the values of a string mapping."""
    if isinstance(text, Mapping):
        return {key: t(value) for key, value in text.items()}  # type: ignore[dict-item]
    if not isinstance(text, str):
        raise TypeError("t() accepts str or a mapping of strings")
    if _LANGUAGE == "en":
        return text
    return _PT.get(text, text)
