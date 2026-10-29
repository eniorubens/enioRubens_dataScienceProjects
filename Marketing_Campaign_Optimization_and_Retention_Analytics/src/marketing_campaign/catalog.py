"""Texts of the analysis in each supported language.

This module only holds data: translations of table columns and labels,
segment names and the messages written by the notebooks and charts. The
functions that apply them live in :mod:`marketing_campaign.i18n`.
"""
from __future__ import annotations

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
    "Variant": "Variante",
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
VALUES_PT.update(
    {
        "Accuracy": "Acurácia",
        "Precision": "Precisão",
        "Precision - Not Retained": "Precisão - Não Retido",
        "Recall - Not Retained": "Recall - Não Retido",
        "F1 - Not Retained": "F1 - Não Retido",
        "False Positives": "Falsos Positivos",
        "False Negatives": "Falsos Negativos",
        "with converted": "com converted",
        "without converted": "sem converted",
        "without SMOTE": "sem SMOTE",
        "Missing": "Missing",
    }
)
FIX_LANGUAGE_PREFIX = {
    "en": "Fix Language -> then ",
    "pt": "Corrigir Idioma -> depois ",
}
VALUE_COLUMNS = {
    "Metric",
    "experiment",
    "Segment",
    "majority_segment",
    "Significance",
    "label",
    "interpretation",
    "recommended_action",
    "strategy_rationale",
}

CANONICAL_SEGMENTS = {
    "high value": "High Value",
    "alto valor": "High Value",
    "medium value": "Medium Value",
    "medio valor": "Medium Value",
    "low value": "Low Value",
    "baixo valor": "Low Value",
}


_SPAN = '<span style="color:steelblue; text-align: left">'


def _md(*lines: str) -> str:
    """Join the lines of a Markdown message."""
    return "\n".join(lines)


def _title(text: str) -> list[str]:
    """Return the lines of a coloured section title."""
    return ["", _SPAN, "", f"### {text}", "", "</span>", ""]


# Each message is written once as (English, Portuguese), so a key can never
# exist in one language only. Placeholders use ``str.format`` fields.
_PAIRS = {
    # setup and data -----------------------------------------------------
    "setup.seaborn_missing": (
        "Seaborn unavailable; using only matplotlib.",
        "Seaborn indisponível; usando apenas matplotlib.",
    ),
    "setup.imports_done": (
        "✓ All imports completed successfully.",
        "✓ Todos os imports realizados com sucesso",
    ),
    "setup.constants_done": (
        "✓ Business constants defined",
        "✓ Constantes de negócio definidas",
    ),
    "data.loaded": (
        "✓ Dataset loaded: {rows} rows, {columns} columns",
        "✓ Dataset carregado: {rows} linhas, {columns} colunas",
    ),
    "inspect.title": ("DATA INSPECTION", "INSPEÇÃO DE DADOS"),
    "inspect.dimensions": ("\n Dimensions: {shape}", "\n Dimensões: {shape}"),
    "inspect.first_rows": ("\n First rows:", "\n Primeiras linhas:"),
    "inspect.dtypes": ("\n Data types:", "\n Tipos de dados:"),
    "inspect.describe": (
        "\n Descriptive statistics:",
        "\n Estatísticas descritivas:",
    ),
    "inspect.missing": (
        "\n  Missing values:",
        "\n  Valores ausentes:",
    ),
    "clean.done": (
        "✓ Data cleaned and prepared successfully (NumPy standard)",
        "✓ Dados limpos e preparados com sucesso (Padrão NumPy)",
    ),
    "clean.exposure_base": (
        "✓ Exposure base after cleaning: {shape}",
        "✓ Base de exposições após limpeza: {shape}",
    ),
    # segmentation and value --------------------------------------------
    "seg.canonical_base": (
        "Canonical base: {users:,} stable users; {switchers:,} users who "
        "switched variants excluded from the A/B test.",
        "Base canônica: {users:,} usuários estáveis; {switchers:,} "
        "usuários com troca de variante excluídos do A/B.",
    ),
    "seg.done": (
        "C1: Proxy-based RFM segmentation created and integrated into "
        "df_clean.",
        "C1: Segmentação RFM por proxy criada a partir das exposições e "
        "integrada à base canônica de usuários.",
    ),
    "chart.rfm.subtitle": (
        "Distribution of RFM proxies by segment",
        "Distribuição dos proxies RFM por segmento",
    ),
    "chart.k.inertia": ("Inertia", "Inercia"),
    "chart.k.silhouette": ("Silhouette score", "Silhouette score"),
    "chart.rfm.box_title": ("{metric} by segment", "{metric} por segmento"),
    "ltv.done": (
        "C2: LTV per segment calculated and available at segment_ltv_map.",
        "C2: LTV por segmento calculado e disponível em segment_ltv_map.",
    ),
    "chart.ltv.global": (
        "Global average LTV: \\$ {value:,.2f}",
        "LTV medio global: \\$ {value:,.2f}",
    ),
    "chart.ltv.xlabel": (
        "Average historical LTV ($)",
        "LTV histórico medio ($)",
    ),
    "chart.ltv.title": (
        "Average Customer Lifetime Value by Segment",
        "Customer Lifetime Value medio por segmento",
    ),
    "chart.ltv.subtitle": (
        "Historical and calculated LTV per customer as estimated revenue "
        "minus impact cost, projected by segment.",
        "LTV histórico e calculado por cliente como receita estimada menos "
        "custo de impacto, projetado por segmento",
    ),
    # exploration ---------------------------------------------------------
    "explore.title": ("DESCRIPTIVE EXPLORATION", "EXPLORAÇÃO DESCRITIVA"),
    "explore.done": (
        "✓ Exploration charts created",
        "✓ Gráficos de exploração criados",
    ),
    "chart.explore.conversion.title": (
        "Conversion Distribution",
        "Distribuição de Conversão",
    ),
    "chart.explore.conversion.subtitle": (
        "Distribution of converted vs. non-converted customers",
        "Distribuição dos clientes convertidos vs não convertidos",
    ),
    "chart.explore.conversion.xlabel": ("Converted", "Convertido"),
    "chart.explore.retention.title": (
        "Retention Distribution",
        "Distribuição de Retenção",
    ),
    "chart.explore.retention.subtitle": (
        "Distribution of customer retention after conversion",
        "Distribuição de retenção dos clientes após conversão",
    ),
    "chart.explore.retention.xlabel": ("Retained", "Retido"),
    "chart.explore.channel.title": ("Volume by Channel", "Volume por Canal"),
    "chart.explore.channel.subtitle": (
        "Distribution of marketing channels",
        "Distribuição dos canais de marketing",
    ),
    "chart.explore.variant.title": (
        "Distribution by A/B Variant",
        "Distribuição por Variante A/B",
    ),
    "chart.explore.variant.subtitle": (
        "Distribution of customers between A and B variants",
        "Distribuição dos clientes entre as variantes A e B",
    ),
    # funnel and channels -----------------------------------------------
    "funnel.title": ("FUNNEL ANALYSIS", "ANÁLISE DE FUNIL"),
    "funnel.done": ("✓ Funnel built", "✓ Funil construído"),
    "chart.funnel.impacted": ("Impacted", "Impactados"),
    "chart.funnel.converted": ("Converted", "Convertidos"),
    "chart.funnel.retained": ("Retained", "Retidos"),
    "chart.funnel.ylabel": ("Number of Users", "Quantidade de Usuários"),
    "chart.funnel.title": (
        "Funnel: From Impact to Retention",
        "Funil: Do Impacto à Retenção",
    ),
    "chart.funnel.subtitle": (
        "Marketing funnel visualization from impacted users to retained "
        "users",
        "Visualização do funil de marketing desde os usuários impactados "
        "até os retidos",
    ),
    "channel.title": (
        "CHANNEL PERFORMANCE ANALYSIS",
        "ANÁLISE POR CANAL",
    ),
    "channel.done": (
        "✓ Channel analysis completed",
        "✓ Análise por canal concluída",
    ),
    "chart.channel.conversion.xlabel": (
        "Conversion Rate (%)",
        "Taxa de Conversão (%)",
    ),
    "chart.channel.conversion.title": (
        "Conversion Rate by Channel",
        "Taxa de Conversão por Canal",
    ),
    "chart.channel.conversion.subtitle": (
        "Conversion rate calculated as conversions divided by the number "
        "of users impacted, per channel.",
        "Taxa de conversão calculada usuários de conversões dividido pelo "
        "número de usuários impactados, por canal",
    ),
    "chart.channel.retention.xlabel": (
        "Retention Rate (%)",
        "Taxa de Retenção (%)",
    ),
    "chart.channel.retention.title": (
        "Retention Rate by Channel",
        "Taxa de Retenção por Canal",
    ),
    "chart.channel.retention.subtitle": (
        "Retention rate calculated as retained users after conversion "
        "divided by total conversions, by channel",
        "Taxa de retenção calculada entre usuários retidos após conversão "
        "dividido pelo número de conversões, por canal",
    ),
    # language ------------------------------------------------------------
    "language.title": (
        "PERSONALIZATION ANALYSIS: CORRECT LANGUAGE",
        "ANÁLISE DE PERSONALIZAÇÃO: IDIOMA CORRETO",
    ),
    "language.done": (
        "Language analysis completed.",
        "Análise de idioma concluída.",
    ),
    "language.validation": (
        "Validation of retention by language: min={low:.2f}% | "
        "max={high:.2f}%",
        "Validação da retenção por idioma: min={low:.2f}% | "
        "max={high:.2f}%",
    ),
    "language.alert": (
        "ALERT: retention rate above 100%.",
        "ALERTA: existe taxa de retenção acima de 100%.",
    ),
    "chart.language.rate": ("Rate (%)", "Taxa (%)"),
    "chart.language.conversion.title": (
        "Conversion Rate by Language",
        "Taxa de Conversão por Idioma",
    ),
    "chart.language.conversion.subtitle": (
        "Conversion rate calculated as the number of converted users "
        "divided by the number of users impacted.",
        "Taxa de conversão calculada entre usuários convertidos dividido "
        "por usuários impactados",
    ),
    "chart.language.retention.title": (
        "Retention Rate by Language",
        "Taxa de Retenção por Idioma",
    ),
    "chart.language.retention.subtitle": (
        "Retention rate calculated as retained users after conversion "
        "divided by conversions",
        "Taxa de retenção calculada entre usuários retidos após conversão "
        "dividido por conversões",
    ),
    # power ---------------------------------------------------------------
    "chart.power.minimum": ("power minimum (80%)", "Poder minimo (80%)"),
    "chart.power.current": (
        "current sample by group ({n:,})",
        "Amostra atual por grupo ({n:,})",
    ),
    "chart.power.xlabel": (
        "Sample size per group",
        "Tamanho da amostra por grupo",
    ),
    "chart.power.ylabel": (
        "empirical power (Monte Carlo)",
        "Poder empirico (Monte Carlo)",
    ),
    "chart.power.title": (
        "Power curve: how many users per group are required?",
        "Curva de poder: quantos usuarios precisamos por grupo?",
    ),
    "chart.power.subtitle": (
        "Monte Carlo simulation to estimate the power of the experiment "
        "based on simulated conversions.",
        "Simulação Monte Carlo para estimar o poder do experimento com "
        "base em conversões simuladas",
    ),
    "power.result": (
        "\nEmpirical power with {n:,} per group: {power:.1%}",
        "\nPoder empírico com {n:,} por grupo: {power:.1%}",
    ),
    # A/B test ------------------------------------------------------------
    "ab.title": (
        "A/B TEST - PROPORTIONS Z-TEST",
        "TESTE A/B - PROPORÇÕES Z-TEST",
    ),
    "ab.conversion_title": (
        "\n A/B TEST - CONVERSION",
        "\n O TESTE A/B - CONVERSÃO",
    ),
    "ab.z_title": (
        "Proportions Z-Test: {a} vs {b}",
        "Teste Z de Proporções: {a} vs {b}",
    ),
    "ab.h0": (
        "H0: The conversion ratios are equal.",
        "H0: As proporções de conversão são iguais",
    ),
    "ab.h1": (
        "H1: The conversion ratios are different.",
        "H1: As proporções de conversão são diferentes",
    ),
    "ab.effect_title": (
        "EFFECT SIZE - conversion",
        "TAMANHO DO EFEITO - CONVERSÃO",
    ),
    "ab.rate": (
        "Rate {variant}: {value:.2f}%",
        "Taxa {variant}: {value:.2f}%",
    ),
    "ab.difference": (
        "Absolute difference ({b} - {a}): {value:.2f} p.p.",
        "Diferença absoluta ({b} - {a}): {value:.2f} p.p.",
    ),
    "ab.uplift": (
        "Percentage Uplift: {value:.2f}%",
        "Uplift percentual: {value:.2f}%",
    ),
    "ab.interval": (
        "IC 95% of difference: {value} p.p.",
        "IC 95% da diferença: {value} p.p.",
    ),
    "ab.done": ("A/B test completed", "Teste A/B concluído"),
    "chart.variant.rate": ("Rate (%)", "Taxa (%)"),
    "chart.variant.conversion.title": (
        "Conversion Rate by Variant",
        "Taxa de Conversão por Variante",
    ),
    "chart.variant.conversion.subtitle": (
        "Comparison of observed conversion rates by variant of the A/B "
        "test.",
        "Comparação das taxas de conversão observadas por variante do "
        "teste A/B",
    ),
    "chart.variant.retention.title": (
        "Retention Rate by Variant",
        "Taxa de Retenção por Variante",
    ),
    "chart.variant.retention.subtitle": (
        "Comparison of observed retention rates among converted users by "
        "A/B test variant.",
        "Comparação das taxas de retenção observadas entre os usuários "
        "convertidos por variante do teste A/B",
    ),
    # ROI -----------------------------------------------------------------
    "roi.title": ("ROI SIMULATION", "SIMULACAO DE ROI"),
    "roi.baseline": (
        "\nEstimated organic baseline: {pct:.0f}% of the current "
        "conversion rate.",
        "\nBaseline organico estimado: {pct:.0f}% da taxa de conversão "
        "atual.",
    ),
    "roi.note": (
        "Incremental ROI = gain attributable to the campaign beyond what "
        "would occur organically.",
        "ROI incremental = ganho atribuivel a campanha alem do que "
        "ocorreria organicamente.",
    ),
    "roi.done": ("ROI simulation completed", "Simulacao de ROI concluída"),
    "chart.roi.incremental.xlabel": (
        "ROI incremental (%)",
        "ROI incremental (%)",
    ),
    "chart.roi.incremental.title": (
        "Incremental ROI by Channel",
        "ROI Incremental por Canal",
    ),
    "chart.roi.incremental.subtitle": (
        "Comparison of the estimated incremental ROI per channel, "
        "considering additional revenue beyond the organic baseline.",
        "Comparação do ROI incremental estimado por canal, considerando "
        "receita adicional além do baseline orgânico",
    ),
    "chart.roi.profit.xlabel": ("Profit ($)", "Lucro ($)"),
    "chart.roi.profit.title": (
        "Estimated Profit by Channel",
        "Lucro Estimado por Canal",
    ),
    "chart.roi.profit.subtitle": (
        "Comparison of estimated profit by channel, considering total "
        "revenue minus total cost",
        "Comparação do lucro estimado por canal, considerando receita "
        "total menos custo total",
    ),
    "plots.alternative_error": (
        "alternative should be 'two-sided', 'greater' or 'less'.",
        "alternative deve ser 'two-sided', 'greater' ou 'less'.",
    ),
    "plots.normal": (
        'Standard normal distribution',
        'Distribuição normal padrão',
    ),
    "plots.critical_region": (
        'Critical region',
        'Região crítica',
    ),
    "plots.z_critical": (
        'Z critical = {z:.2f}',
        'Z crítico = {z:.2f}',
    ),
    "plots.z_observed": (
        'Z observed = {z:.2f}',
        'Z observado = {z:.2f}',
    ),
    "plots.outside_scale": (
        'Z observed = {z:.2f}\noutside the scale [-4, 4]',
        'Z observado = {z:.2f}\nfora da escala [-4, 4]',
    ),
    "plots.reject": (
        'Reject H0',
        'Rejeita H0',
    ),
    "plots.fail_to_reject": (
        'Fail to reject H0',
        'Não rejeita H0',
    ),
    "plots.h0": (
        'H0: The conversion rates are equal',
        'H0: As proporções de conversão são iguais',
    ),
    "plots.h1": (
        'H1: The conversion rates are different',
        'H1: As proporções de conversão são diferentes',
    ),
    "plots.z_title": (
        'Standard Normal Distribution for Z-Test',
        'Distribuição Normal Padrão do Teste Z',
    ),
    "plots.z_subtitle": (
        'Visualization of the observed Z statistic and critical regions for '
        'interpreting the hypothesis test',
        'Visualização da estatística Z observada e regiões críticas para '
        'interpretação do teste de hipóteses',
    ),
    "plots.z_statistic": (
        'Z Statistic',
        'Estatística Z',
    ),
    "plots.density": (
        'Density',
        'Densidade',
    ),
    "plots.z_result": (
        'Z-TEST RESULT',
        'RESULTADO DO TESTE Z',
    ),
    "plots.z_line": (
        'Z Statistic: {z:.4f}',
        'Estatística Z: {z:.4f}',
    ),
    "plots.critical_value": (
        'Critical Value: {value}',
        'Valor crítico: {value}',
    ),
    "plots.conclusion": (
        '\nConclusion:',
        '\nConclusão:',
    ),
    "plots.significant": (
        'There are statistically significant evidence to reject H0.',
        'Há evidências estatisticamente significativas para rejeitar H0.',
    ),
    "plots.not_significant": (
        'There is not enough evidence to reject H0.',
        'Não há evidências suficientes para rejeitar H0.',
    ),
    "plots.confusion_title": (
        'Confusion Matrix',
        'Matriz de Confusão',
    ),
    "plots.confusion_subtitle": (
        'Comparison between actual and predicted retention classes',
        'Comparação entre as classes reais e previstas para retenção',
    ),
    "plots.roc_unavailable": (
        'ROC AUC unavailable',
        'ROC AUC indisponível',
    ),
    "plots.fpr": (
        'False Positive Rate',
        'Taxa de Falsos Positivos',
    ),
    "plots.tpr": (
        'True Positive Rate',
        'Taxa de Verdadeiros Positivos',
    ),
    "plots.roc_title": (
        'ROC Curve',
        'Curva ROC',
    ),
    "plots.roc_subtitle": (
        'Model ability to distinguish retained vs not-retained users',
        'Avaliação da capacidade do modelo de distinguir entre usuários '
        'retidos e não retidos',
    ),
    "plots.no_experiments": (
        'No experiments available for visualization.',
        'Nenhum experimento disponível para visualização.',
    ),
    "plots.auc_title": (
        'ROC AUC by Experiment',
        'ROC AUC por Experimento',
    ),
    "plots.auc_subtitle": (
        'Comparison of predictive performance across experiments',
        'Comparação da capacidade preditiva entre os experimentos',
    ),
    "plots.recall_title": (
        'Not Retained Recall by Experiment',
        'Recall de Não Retido por Experimento',
    ),
    "plots.recall_subtitle": (
        'Sensitivity in identifying non-retained users',
        'Sensibilidade na identificação de usuários não retidos',
    ),
    "plots.recall_axis": (
        'Recall - Not Retained',
        'Recall - Não Retido',
    ),
    "plots.importance_title": (
        'Top {n} Feature Importances - {name}',
        'Top {n} Importâncias - {name}',
    ),
    "plots.importance_subtitle": (
        'Most relevant features used by the model for retention prediction',
        'Principais features que o modelo considera para prever retenção',
    ),
    "plots.importance_axis": (
        'Importance',
        'Importância',
    ),
    "plots.coefficient_title": (
        'Top {n} Absolute Coefficients - {name}',
        'Top {n} Coeficientes Absolutos - {name}',
    ),
    "plots.coefficient_subtitle": (
        'Features most strongly associated with retention in logistic '
        'regression, regardless of direction',
        'Principais features que o modelo de regressão logística associa com '
        'retenção, considerando a magnitude do efeito independentemente do '
        'sinal',
    ),
    "plots.coefficient_axis": (
        'Absolute coefficient',
        'Coeficiente absoluto',
    ),
    "plots.no_signal": (
        '{name} does not expose direct feature importances or coefficients.',
        '{name} nao expõe importâncias ou coeficientes diretos.',
    ),
    # experiment design table --------------------------------------------
    "table.parameter": ("Parameter", "Parametro"),
    "table.value": ("Value", "Valor"),
    "design.h0_label": ("Null hypothesis (H0)", "Hipotese nula (H0)"),
    "design.h1_label": (
        "Alternative hypothesis (H1)",
        "Hipotese alternativa (H1)",
    ),
    "design.lift_label": ("Minimum lift", "Minimum lift minimo"),
    "design.baseline_label": (
        "Baseline conversion rate",
        "Taxa de conversao baseline",
    ),
    "design.treatment_label": (
        "Expected treatment rate",
        "Taxa esperada no tratamento",
    ),
    "design.cohen_label": ("Cohen h", "Cohen h"),
    "design.alpha_label": ("Alpha (Type I error)", "Alpha (Type I error)"),
    "design.power_label": (
        "Power (1 - Type II error)",
        "Power (1 - Type II error)",
    ),
    "design.sample_label": ("Sample size per group", "Sample size por grupo"),
    "design.runtime_label": ("Estimated runtime", "Runtime estimado"),
    "design.guardrail_label": ("Guardrail metric", "Guardrail metric"),
    "design.h0": (
        "conversion(control) = conversion(personalization)",
        "Conversao(controle) = Conversao(personalizacao)",
    ),
    "design.h1": (
        "conversion(personalization) != conversion(control)",
        "Conversao(personalizacao) != Conversao(controle)",
    ),
    "design.users": ("{n:,} users", "{n:,} usuarios"),
    "design.runtime": (
        "{days} days (min. 7 for day-of-week effect)",
        "{days} dias (min. 7 para day-of-week effect)",
    ),
    "design.guardrail": (
        "Average revenue per conversion must not fall vs control",
        "Receita media por convertido nao deve cair vs controle",
    ),
    "design.adequate": (
        "Sufficient sample ({available:,} available vs {required:,} "
        "required)",
        "Amostra suficiente ({available:,} disponíveis vs {required:,} "
        "necessários)",
    ),
    "design.inadequate": (
        "Insufficient sample ({available:,} available, necessary "
        "{required:,} required) - see Monte Carlo section",
        "`Amostra insuficiente ({available:,} disponíveis, necessários "
        "{required:,}) - ver Seção Monte Carlo`",
    ),
    # Monte Carlo summary -------------------------------------------------
    "power.baseline_label": (
        "Conversion baseline (control)",
        "Conversão baseline (controle)",
    ),
    "power.expected_label": (
        "Expected conversion (treatment)",
        "Conversão esperada (tratamento)",
    ),
    "power.lift_label": ("Simulated minimum lift", "Minimum lift simulado"),
    "power.sample_label": ("Sample size per group", "Tamanho por grupo"),
    "power.simulations_label": (
        "Number of simulations",
        "Numero de simulacoes",
    ),
    "power.power_label": (
        "Estimated empirical power",
        "Poder empirico estimado",
    ),
    "power.interpretation_label": ("Interpretation", "Interpretacao"),
    "power.adequate": ("Adequate (>= 80%)", "Adequado (>= 80%)"),
    "power.inadequate": (
        "Insufficient - increase MINIMUM_LIFT_PP or sample size",
        "Insuficiente - aumentar MINIMUM_LIFT_PP ou amostra",
    ),
    # sample adequacy by variant -----------------------------------------
    "adequacy.variant": ("Variant", "Variante"),
    "adequacy.available": ("Available users", "Usuarios disponiveis"),
    "adequacy.required": (
        "Required (power analysis)",
        "Necessario (power anal.)",
    ),
    "adequacy.sufficient": ("Sufficient?", "Suficiente?"),
    "adequacy.power": ("Estimated power", "Poder estimado"),
    "adequacy.gap": ("Gap (users)", "Gap (usuarios)"),
    "adequacy.yes": ("✓ Yes", "✓ Sim"),
    "adequacy.no": ("✗ No", "✗ Nao"),
    "adequacy.na": ("N/A", "N/D"),
    "pretest.title": ("  PRE-TEST DECISION", "  DECISAO PRE-TESTE"),
    "pretest.not_run": (
        "  ⚠ Power analysis was not executed - missing "
        "required_n_per_group.",
        "  ⚠ Power analysis nao executado - required_n_per_group ausente.",
    ),
    "pretest.run_cell": (
        "    Run Cell 27 before proceeding.",
        "    Execute a Celula 27 antes de prosseguir.",
    ),
    "pretest.sufficient": (
        "  ✓ Sufficient sample across all variants.",
        "  ✓ Amostra suficiente em todas as variantes.",
    ),
    "pretest.smallest_ok": (
        "    Smallest group: {smallest:,} users (required: {required:,}).",
        "    Menor grupo: {smallest:,} usuarios (necessario: {required:,}).",
    ),
    "pretest.power": (
        "    Empirical power (Monte Carlo): {power:.1%}",
        "    Poder empirico (Monte Carlo): {power:.1%}",
    ),
    "pretest.proceed": (
        "    Proceed with run_ab_test(df_clean).",
        "    Prosseguir com run_ab_test(df_clean).",
    ),
    "pretest.insufficient": (
        "  ✗ Insufficient sample.",
        "  ✗ Amostra insuficiente.",
    ),
    "pretest.smallest_bad": (
        "    Smallest group: {smallest:,} users (required: {required:,}, "
        "deficit: {deficit:,}).",
        "    Menor grupo: {smallest:,} usuarios (necessario: "
        "{required:,}, deficit: {deficit:,}).",
    ),
    "pretest.estimated_power": (
        "    Estimated power with current sample: ~{available:.1%} "
        "(target: {target:.1%})",
        "    Poder estimado com amostra atual: ~{available:.1%} "
        "(meta: {target:.1%})",
    ),
    "pretest.runtime": (
        "    Estimated time to reach required sample: {days} days.",
        "    Tempo estimado para atingir amostra: {days} dias.",
    ),
    "pretest.options": ("  Options:", "  Opcoes:"),
    "pretest.option_1": (
        "    1. Wait for more data (recommended for production).",
        "    1. Aguardar mais dados (recomendado para producao).",
    ),
    "pretest.option_2": (
        "    2. Increase MINIMUM_LIFT_PP to reduce required_n.",
        "    2. Aumentar MINIMUM_LIFT_PP para reduzir required_n.",
    ),
    "pretest.option_3": (
        "    3. Proceed acknowledging reduced statistical power "
        "(exploratory only).",
        "    3. Prosseguir ciente do poder reduzido (apenas "
        "exploratorio).",
    ),
    "pretest.warning": (
        "Real sample ({smallest:,}) is smaller than required_n_per_group "
        "({required:,}). Test power may be lower than the Monte Carlo "
        "estimate ({power:.1%}).",
        "Amostra real ({smallest:,}) menor que required_n_per_group "
        "({required:,}). Poder do teste pode ser inferior ao estimado "
        "pelo Monte Carlo ({power:.1%}).",
    ),
    "chart.sample.available": ("Available", "Disponível"),
    "chart.sample.required": ("Required ({n:,})", "Necessário ({n:,})"),
    "chart.sample.ylabel": ("Users", "Usuarios"),
    "chart.sample.title": (
        "Available Sample by Variant vs Power Analysis Requirement",
        "Amostra disponível por variante vs requisito do power analysis",
    ),
    "chart.sample.subtitle": (
        "Comparison between available users per variant and the "
        "estimated sample required by the power analysis to detect the "
        "minimum lift with adequate statistical power",
        "Comparação entre número de usuários disponíveis por variante e "
        "o número necessário estimado pelo power analysis para detectar "
        "o lift mínimo com poder adequado",
    ),
    # retention model ----------------------------------------------------
    "model.running": (
        "Running experiment: {label}",
        "Executando experimento: {label}",
    ),
    "model.no_experiments": (
        "No retention experiments were successfully executed.",
        "Nenhum experimento de retenção foi executado com sucesso.",
    ),
    "model.best": (
        "\nBest scenario according to the defined ranking:",
        "\nMelhor cenário pela ordenação definida:",
    ),
    "model.configuration": (
        "CONFIGURATION: Estimator={estimator} | conversion={conversion} | "
        "Resampling={resampling}",
        "CONFIGURAÇÃO: Estimador={estimator} | Conversão={conversion} | "
        "Resampling={resampling}",
    ),
    "model.target_distribution": (
        "Target class distribution:",
        "Distribuição da classe alvo:",
    ),
    "model.not_retained": ("Not Retained", "Não Retido"),
    "model.retained": ("Retained", "Retido"),
    # classification errors and bias --------------------------------------
    "errors.summary": (
        "Classification error summary",
        "Resumo dos erros de classificação",
    ),
    "errors.false_positives": (
        "False positives: {count}",
        "Falsos positivos: {count}",
    ),
    "errors.false_negatives": (
        "False negatives: {count}",
        "Falsos negativos: {count}",
    ),
    "errors.none": (
        "No false positives to analyze in the test set.",
        "Não há falsos positivos para analisar no conjunto de teste.",
    ),
    "errors.distribution": (
        "\nFalse positive distribution by variable",
        "\nDistribuição dos falsos positivos por variável",
    ),
    "bias.title": (
        "False positive over-representation analysis",
        "Analise de super-representação nos falsos positivos",
    ),
    "bias.total": (
        "Total dataset: {count} records",
        "Base total: {count} registros",
    ),
    "bias.false_positives": (
        "False positives: {count} records",
        "Falsos positivos: {count} registros",
    ),
    "bias.comparison": (
        "\nPercentage comparison for {column}",
        "\nComparacao percentual para {column}",
    ),
    "bias.over": (
        "{category} represents {fp:.1f}% of false positives, but only "
        "{base:.1f}% of the full dataset, indicating a possible "
        "concentration of errors in {column}.",
        "{category} representa {fp:.1f}% dos falsos positivos, mas apenas "
        "{base:.1f}% da base, indicando possível concentração de erros "
        "em {column}.",
    ),
    "bias.within": (
        "Largest difference in {column}: {category} represents {fp:.1f}% "
        "of false positives and {base:.1f}% of the full dataset "
        "({diff:.1f} p.p.), without significant over-representation under "
        "the 10 p.p. threshold.",
        "Maior diferença em {column}: {category} representa {fp:.1f}% dos "
        "falsos positivos e {base:.1f}% da base ({diff:.1f} p.p.), sem "
        "super-representação relevante pelo limiar de 10 p.p.",
    ),
    "chart.bias.base": ("Full dataset", "Base total"),
    "chart.bias.false_positives": ("False positives", "Falsos positivos"),
    "chart.bias.xlabel": ("Percentage (%)", "Percentual (%)"),
    "chart.bias.title": (
        "Full dataset vs False Positives: {column}",
        "Base total vs Falsos Positivos: {column}",
    ),
    "chart.bias.subtitle": (
        "Comparison of the percentage distribution between the full "
        "dataset and false positives for each category",
        "Comparação da distribuição percentual entre a base completa e os "
        "falsos positivos para cada categoria",
    ),
    # scoring, recommendations and export --------------------------------
    "scoring.done": (
        "Expected value computed for all users",
        "Valor esperado computado para todos os usuarios",
    ),
    "recs.title": (
        "OPERATIONAL RECOMMENDATIONS",
        "RECOMENDAÇÕES OPERACIONAIS",
    ),
    "recs.distribution": (
        "\nAction distribution:\n{counts}",
        "\nDistribuicao de acoes:\n{counts}",
    ),
    "recs.top": (
        "\nTop 10 users by expected value:",
        "\nTop 10 usuarios por valor esperado:",
    ),
    "recs.done": ("Recommendations generated", "Recomendações geradas"),
    "recs.missing": (
        "Columns missing from recommendations: {missing}",
        "Colunas ausentes em recommendations: {missing}",
    ),
    "export.title": ("RESULT EXPORT", "EXPORTAÇÃO DE RESULTADOS"),
    "export.excel": ("Excel exported: {path}", "Excel exportado: {path}"),
    "export.no_openpyxl": (
        "openpyxl unavailable; results exported to CSV.",
        "openpyxl indisponível; resultados exportados em CSV.",
    ),
    "export.csv_dir": (
        "Exported CSV directory: {path}",
        "Diretorio CSV exportado: {path}",
    ),
    "export.crm": ("CRM CSV exported: {path}", "CSV CRM exportado: {path}"),
    # insights: funnel, channels, language -------------------------------
    "insight.funnel": (
        _md(
            *_title("Funnel Insights"),
            "**What the data shows:** {conv:.1f}% of impacted users convert "
            "and {ret:.1f}% of converts remain. The biggest loss occurs in "
            "conversion, not retention.",
            "",
            "**Why it matters:** The main bottleneck is qualified "
            "acquisition, not post-conversion retention. Focusing efforts "
            "here has greater leverage.",
            "",
            "**Recommended action:** Prioritize budget optimization in "
            "channels and variants that increase initial conversion.",
            "",
            "**Estimated impact:** Each +1 p.p. in conversion generates "
            "approximately \\$ {impact:,.0f} in additional revenue on the "
            "current base.",
        ),
        _md(
            *_title("Insights do Funil"),
            "**O que os dados mostram:** {conv:.1f}% dos usuários "
            "impactados convertem e {ret:.1f}% dos convertidos permanecem. "
            "A maior perda ocorre na conversão, não na retenção.",
            "",
            "**Por que importa:** O gargalo principal é aquisição "
            "qualificada, não retenção pós-conversão. Concentrar esforços "
            "aqui tem maior alavancagem.",
            "",
            "**Ação recomendada:** Priorizar budget de otimização em "
            "canais e variantes que aumentam conversão inicial.",
            "",
            "**Impacto estimado:** Cada +1 p.p. na conversão gera "
            "aproximadamente \\$ {impact:,.0f} em receita adicional sobre "
            "a base atual.",
        ),
    ),
    "channel.action_dominates": (
        "Email leads in both dimensions (conversion: {email_conv:.1f}% vs "
        "{house_conv:.1f}%; retention: {email_ret:.1f}% vs "
        "{house_ret:.1f}%). **Action**: Scale email budget. House Ads does "
        "not justify budget reallocation while email dominates both "
        "metrics.",
        "Email lidera em ambas as dimensões (conversão: {email_conv:.1f}% "
        "vs {house_conv:.1f}%; retenção: {email_ret:.1f}% vs "
        "{house_ret:.1f}%). Ação: escalar budget de Email. House Ads não "
        "justifica realocação de budget enquanto Email dominar ambas as "
        "métricas.",
    ),
    "channel.realloc_dominates": (
        "from House Ads to Email",
        "de House Ads para Email",
    ),
    "channel.action_tradeoff": (
        "There's a real tradeoff: {best_conv} leads in conversion "
        "({best_conv_rate:.1f}%) and {best_ret} leads in retention "
        "({best_ret_rate:.1f}%). **Action**: Use {best_conv} for "
        "acquisition and {best_ret} for convert retention.",
        "Há tradeoff real: {best_conv} lidera conversão "
        "({best_conv_rate:.1f}%) e {best_ret} lidera retenção "
        "({best_ret_rate:.1f}%). Ação: usar {best_conv} para aquisição e "
        "{best_ret} para retenção de convertidos.",
    ),
    "channel.realloc_tradeoff": (
        "from {best_conv} to {best_ret} in retention campaigns",
        "de {best_conv} para {best_ret} em campanhas de retenção",
    ),
    "channel.action_house": (
        "House Ads leads on both dimensions. Review data; this is an "
        "uncommon result.",
        "House Ads lidera em ambas as dimensões. Revisar dados - "
        "resultado incomum.",
    ),
    "channel.realloc_house": (
        "review data before recommending",
        "verificar dados antes de recomendar",
    ),
    "channel.action2_dominates": (
        "**Focus budget on Email** - dominant channel in conversion "
        "({email_conv:.1f}%) and retention ({email_ret:.1f}%). Reduce "
        "allocation to channels with both metrics below average. Do not "
        "redirect to House Ads while Email dominates both dimensions.",
        "**Concentrar budget em Email** - canal dominante em conversão "
        "({email_conv:.1f}%) e retenção ({email_ret:.1f}%). Reduzir "
        "alocação em canais com ambas as métricas abaixo da média. Não "
        "redirecionar para House Ads enquanto Email dominar ambas as "
        "dimensões.",
    ),
    "channel.action2_tradeoff": (
        "Separate channels by objective: {best_conv} for acquisition, "
        "{best_ret} for retention. Allocate a percentage of the budget "
        "proportional to the expected value of each stage of the funnel.",
        "**Separar canais por objetivo:** {best_conv} para aquisição, "
        "{best_ret} para retenção. Alocar percentual do budget "
        "proporcional ao valor esperado de cada etapa do funil.",
    ),
    "channel.action2_default": (
        "Review channel performance data before recommending "
        "reallocation.",
        "Revisar dados de performance de canal antes de recomendar "
        "realocação.",
    ),
    "channel.finding_dominates": (
        "Email dominates conversion ({email_conv:.1f}%) and retention "
        "({email_ret:.1f}%).",
        "Email domina conversão ({email_conv:.1f}%) e retenção "
        "({email_ret:.1f}%).",
    ),
    "channel.finding_tradeoff": (
        "Real tradeoff: {best_conv} leads conversion and {best_ret} leads "
        "retention.",
        "Tradeoff real: {best_conv} lidera conversão e {best_ret} lidera "
        "retenção.",
    ),
    "channel.finding_default": (
        "No clear dominance between Email and House Ads; review by "
        "channel.",
        "Sem dominância clara entre Email e House Ads; revisar por canal.",
    ),
    "insight.channel": (
        _md(
            *_title("Channel Insights"),
            "**What they show:** {action}",
            "",
            "**Why it matters:** The channel with the highest conversion "
            "and highest retention simultaneously should receive budget "
            "priority - there is no benefit in diversifying to channels "
            "with inferior performance in both dimensions.",
            "",
            "**Recommended action:** Concentrate the next campaign on the "
            "dominant channel.",
            "",
            "Review current allocation: Is there budget in channels with "
            "a conversion rate below {mean_conv:.1f}% (overall average)?",
            "",
            "**Estimated impact:** Budget reallocation {realloc} can "
            "increase incremental profit by ~\\$ {ltv_impact:,.0f} per "
            "user for each additional retained user (conservative "
            "estimate: 5% gain in average LTV).",
        ),
        _md(
            *_title("Insights por Canal"),
            "**O que mostram:** {action}",
            "",
            "**Por que importa:** Canal com maior conversão e maior "
            "retenção simultaneamente deve receber prioridade de budget - "
            "não há benefício em diversificar para canais com performance "
            "inferior em ambas as dimensões.",
            "",
            "**Ação recomendada:** Concentrar próxima campanha no canal "
            "dominante.",
            "",
            "Revisar alocação atual: há budget em canais com "
            "conversion_rate abaixo de {mean_conv:.1f}% (média geral)?",
            "",
            "**Impacto estimado:** Realocação de budget {realloc} pode "
            "aumentar lucro incremental em ~\\$ {ltv_impact:,.0f} por "
            "usuário adicional retido (estimativa conservadora de 5% de "
            "ganho no LTV médio).",
        ),
    ),
    "language.lift_note": (
        "Measured conversion lift = 0 p.p., but wrong-language exposure "
        "still has an estimated satisfaction cost of \\$ {value:.0f} "
        "(5% LTV loss across {users} users).",
        "Lift de conversao medido = 0 p.p., mas exposicao no idioma "
        "errado ainda tem custo estimado de satisfacao de \\$ {value:.0f} "
        "(perda de 5% de LTV em {users} usuarios).",
    ),
    "insight.language": (
        _md(
            *_title("Language Personalization Insights"),
            "> **What it shows:** users receiving the correct language "
            "convert at {correct_conv:.1f}%",
            "> versus {wrong_conv:.1f}% when exposed to the wrong "
            "language - a difference of {lift_conv:.1f} p.p.",
            "> There are {n_wrong:,} users ({pct_wrong:.1f}% of the "
            "dataset) being impacted in the wrong language.",
            ">",
            "> **Why it matters:** incorrect language delivery affects "
            "both immediate conversion",
            "> and long-term retention. Even small conversion "
            "differences may generate meaningful",
            "> impact on customer lifetime value (LTV).",
            ">",
            "> **Recommended action:** implement automatic language "
            "detection based on",
            "> `language_preferred` before serving campaigns.",
            ">",
            "> **Estimated impact:** correcting language delivery for "
            "{n_wrong:,.0f} users could generate",
            "> approximately \\$ {total:,.0f}  in additional estimated "
            "revenue",
            "> (conversion: \\$ {impact_conv:,.0f} + retention/LTV: "
            "\\$ {impact_ret:,.0f}).",
        ),
        _md(
            *_title("Insights de Personalização por Idioma"),
            "> **O que mostram:** usuarios no idioma correto convertem "
            "{correct_conv:.1f}%",
            "> vs {wrong_conv:.1f}% no idioma errado - diferenca de "
            "{lift_conv:.1f} p.p.",
            "> Ha {n_wrong:,} usuarios ({pct_wrong:.1f}% da base) sendo "
            "impactados no idioma errado.",
            ">",
            "> **Por que importa:** idioma errado afeta a conversão "
            "imediata e a retenção de longo prazo.",
            "> Mesmo com diferenca pequena em conversão, o impacto no "
            "LTV pode ser relevante.",
            ">",
            "> **Ação recomendada:** implementar detecção automática de "
            "idioma baseada em",
            "> `language_preferred` antes de servir campanhas.",
            ">",
            "> **Impacto estimado:** corrigir idioma para os {n_wrong:,} "
            "usuarios pode gerar",
            "> aproximadamente \\$ {total:,.0f} em receita adicional "
            "estimada",
            "> (conversão: \\$ {impact_conv:,.0f} + retenção/LTV: "
            "\\$ {impact_ret:,.0f}).",
        ),
    ),
    # insights: A/B test ---------------------------------------------------
    "insight.ab_design": (
        _md(
            *_title("A/B Test Insight: personalization vs control"),
            "The experiment design was defined above. The hypothesis: "
            "customized campaigns generate higher conversion rates than "
            "the control group with statistical significance and "
            "sufficient magnitude (>= {lift:.1f} p.p. of lift) to justify "
            "operational cost.",
        ),
        _md(
            *_title("Insight Teste A/B: personalização vs controle"),
            "O design do experimento foi definido acima. A hipótese: "
            "campanhas personalizadas geram conversão superior ao "
            "controle com significância estatística e magnitude "
            "suficiente (>= {lift:.1f} p.p. de lift) para justificar "
            "custo operacional.",
        ),
    ),
    "insight.z_test": (
        _md(
            *_title("Inferential Insight from Z-Test"),
            "> **What it shows:** The Z-test evaluates whether the "
            "personalization variant",
            "> produces a statistically significant difference compared "
            "to the control group.",
            ">",
            "> **Why it matters:** Statistical significance does not "
            "necessarily imply",
            "> practical relevance. It is also necessary to verify "
            "whether the lift magnitude",
            "> (>= {lift:.1f} p.p.) justifies the operational cost.",
            ">",
            "> **Recommended action:** If p < 0.05 and lift >= "
            "{lift:.1f} p.p.,",
            "> launch the campaign; otherwise, revisit the experimental "
            "design or increase",
            "> the sample size.",
            ">",
            "> **Business impact:** Each additional percentage point of "
            "lift may generate",
            "> approximately \\$ {value:,.0f}, considering the current "
            "user base.",
        ),
        _md(
            *_title("Insight inferencial do teste Z"),
            "> **O que mostram:** O teste Z avalia se a variante "
            "personalização gera diferença estatisticamente significativa "
            "em relação ao controle.",
            ">",
            "> **Por que importa:** Diferença estatística != relevância "
            "prática. Precisamos também verificar se a magnitude do lift "
            "(>= {lift:.1f} p.p.) justifica custo operacional.",
            ">",
            "> **Ação recomendada:** Se p < 0.05 e lift >= {lift:.1f} "
            "p.p., lançar campanha; senão, retornar ao design ou aumentar "
            "sample size.",
            ">",
            "> **Impacto:** Cada p.p. de lift adicional pode gerar "
            "\\$ {value:,.0f} considerando tamanho da base.",
        ),
    ),
    # insights: ROI -------------------------------------------------------
    "roi.alert": (
        "ALERT: {channel} has an incremental ROI of {roi:.0f}% with only "
        "\\$ {cost:,.0f} of total cost. A high ROI on a low-volume "
        "channel may indicate an underestimated cost or an insufficient "
        "sample - validate before scaling.",
        "ALERTA: {channel} tem ROI incremental de {roi:.0f}% com apenas "
        "\\$ {cost:,.0f} de custo total. ROI alto em canal de baixo "
        "volume pode indicar custo subestimado ou amostra insuficiente - "
        "validar antes de escalar.",
    ),
    "insight.roi": (
        _md(
            *_title("Insights on ROI and Expected Value"),
            "> **What it shows:** {best} has the highest incremental ROI, "
            "while",
            "> {worst} has the lowest among channels with relevant volume",
            "> (>= 5% of total budget).",
            ">",
            "> **Why it matters:** Large ROI differences across channels "
            "indicate",
            "> inefficiencies in the current budget allocation, "
            "suggesting that part",
            "> of the investment may be generating returns below its "
            "potential.",
            ">",
            "> **Recommended action:** Reallocate {fraction:.0f}% of the "
            "budget",
            "> from {worst} to {best}.",
            ">",
            "> **Estimated impact:** Reallocating \\$ {moved:,.0f}",
            "> (10% of the total budget of \\$ {budget:,.0f})",
            "> may generate approximately \\$ {extra:,.0f}",
            "> in additional estimated profit, increasing overall ROI "
            "from",
            "> {roi_now:.1f}% to approximately {roi_new:.1f}%",
            "> (+{gain:.1f} p.p.).",
            ">",
            "> Additional savings of up to \\$ {savings:,.0f} may also be "
            "achieved",
            "> by reducing investment in channels with negative "
            "incremental ROI.",
        ),
        _md(
            *_title("Insights sobre ROI e Valor Esperado"),
            "> **O que mostram:** {best} tem o maior ROI incremental; "
            "{worst} o menor",
            "> entre canais com volume relevante (>= 5% do budget total).",
            ">",
            "> **Por que importa:** Canais com ROI muito diferente "
            "indicam ineficiência",
            "> na alocação atual - parte do budget pode estar gerando "
            "retorno abaixo do potencial.",
            ">",
            "> **Ação recomendada:** Realocar {fraction:.0f}% do "
            "orçamento de",
            "> {worst} para {best}.",
            ">",
            "> **Impacto estimado:** Realocação de \\$ {moved:,.0f} "
            "(10% do budget total de",
            "> \\$ {budget:,.0f}) gera ~\\$ {extra:,.0f} em lucro "
            "adicional estimado,",
            "> aumentando ROI global de {roi_now:.1f}% para "
            "~{roi_new:.1f}%",
            "> (+{gain:.1f} p.p.). Economia adicional de até "
            "\\$ {savings:,.0f}",
            "> cortando canais com ROI incremental negativo.",
        ),
    ),
    # insights: model, errors, bias --------------------------------------
    "insight.model": (
        _md(
            "",
            "> **Selected model:** {estimator} | {with_flag} "
            "converted_flag | without SMOTE",
            ">",
            "> **Metrics:** ROC AUC = {roc_auc:.3f} | Not Retained Recall "
            "= {recall:.3f} | F1-score Not Retained = {f1:.3f}",
            ">",
            "> **Operational implication:** The model identifies "
            "{recall:.0%} of users",
            "> who would not remain retained. This enables proactive "
            "preventive campaigns for this group",
            "> before churn occurs, with a precision of {precision:.0%}",
            "> (proportion of correct alerts).",
            ">",
            "> **Limitation:** Prediction is not causality. Results "
            "should be validated with a dedicated",
            "> retention A/B experiment before scaling preventive "
            "campaigns.",
        ),
        _md(
            "",
            "> **Modelo selecionado:** {estimator} | {with_flag} "
            "converted_flag | sem SMOTE",
            ">",
            "> **Métricas:** ROC AUC = {roc_auc:.3f} | Recall Não Retido "
            "= {recall:.3f} | F1 Não Retido = {f1:.3f}",
            ">",
            "> **Implicação operacional:** O modelo identifica "
            "{recall:.0%} dos usuários",
            "> que não permaneceriam. Isso permite antecipar campanhas "
            "preventivas para esse grupo",
            "> antes do cancelamento, com precisão de {precision:.0%} "
            "(proporção de alertas corretos).",
            ">",
            "> **Limitação:** Predição não é causalidade. Validar com "
            "experimento A/B dedicado",
            "> a retenção antes de escalar campanhas preventivas.",
        ),
    ),
    "model.with_converted": ("with", "com"),
    "model.without_converted": ("without", "sem"),
    "insight.errors": (
        _md(
            *_title("Error Analysis Insights"),
            ">**What the data shows:** The model generated {fp:,} false "
            "positives ({pct:.1f}% of converted users). These users were "
            "predicted as retained but were not retained.",
            ">**Why this matters:** Each false positive represents "
            "wasted budget on users who do not benefit from the "
            "campaign. Reducing FP improves retention ROI.",
            ">**Recommended action:** Run granular analysis to identify "
            "channels, segments, or periods with higher FP. Adjust model "
            "threshold or improve features.",
            ">**Impact:** Reducing FP by 20% saves \\$ {savings:,.0f} "
            "without losing predictive capacity in recall.",
        ),
        _md(
            *_title("Insights da Análise de Erros"),
            ">**O que mostram:** O modelo gerou {fp:,} falsos positivos "
            "({pct:.1f}% dos convertidos). Estes usuários foram preditos "
            "como retidos mas não retiveram.",
            ">**Por que importa:** Cada falso positivo representa budget "
            "desperdiçado em usuário que não se beneficia da campanha. "
            "Reduzir FP melhora ROI de retenção.",
            ">**Ação recomendada:** Análise granular: quais canais, "
            "segmentos ou períodos geram mais FP? Ajustar threshold do "
            "modelo ou aprimorar features.",
            ">**Impacto:** Reduzir FP em 20% economiza "
            "\\$ {savings:,.0f} sem perder capacidade preditiva no "
            "recall.",
        ),
    ),
    "insight.bias_action": (
        _md(
            "",
            "> **Action derived from bias diagnosis:**",
            ">",
            "> **{category}** is over-represented in false positives",
            "> ({fp:.1f}% of errors vs {base:.1f}% in the base,",
            "> difference of {diff:.1f} p.p.).",
            ">",
            "> This means the model overestimates retention probability "
            "for",
            "> users in this channel. **Action:** apply a more "
            "conservative classification",
            "> threshold for this channel - use `prob_retention >= 0.65` "
            "instead of",
            "> `0.50` when prioritizing preventive campaigns for users "
            "of",
            "> **{category}**. Reassess after 30 days of real data.",
        ),
        _md(
            "",
            "> **Ação derivada do diagnóstico de viés:**",
            ">",
            "> **{category}** está super-representado nos falsos "
            "positivos",
            "> ({fp:.1f}% dos erros vs {base:.1f}% na base,",
            "> diferença de {diff:.1f} p.p.).",
            ">",
            "> Isso significa que o modelo superestima probabilidade de "
            "retenção para",
            "> usuários desse canal. **Ação:** aplicar threshold de "
            "classificação mais",
            "> conservador para esse canal - usar `prob_retention >= "
            "0.65` em vez de",
            "> `0.50` ao priorizar campanhas preventivas para usuários "
            "de",
            "> **{category}**. Reavaliar após 30 dias de dados reais.",
        ),
    ),
    "bias.no_channel": (
        "No channel with over-representation > 10 p.p. in false "
        "positives.",
        "Nenhum canal com super-representação > 10 p.p. nos falsos "
        "positivos.",
    ),
    "insight.bias_summary": (
        _md(
            *_title("False Positive Bias Insights"),
            ">**What the data shows:** False positives concentrate in "
            "{segment} (channel={segment}), suggesting model bias in this "
            "segment.",
            ">**Why this matters:** If the model systematically fails in "
            "one segment, recommendations for that group become "
            "inefficient and waste resources.",
            ">**Recommended action:** Train a dedicated model for the "
            "biased segment, increase class weight, or collect more "
            "representative data.",
            ">**Impact:** Fixing bias can improve precision by "
            "~{fix:.1f}% for the affected segment, saving "
            "\\$ {savings:,.0f} in campaign waste.",
        ),
        _md(
            *_title("Insights de Viés nos Falsos Positivos"),
            ">**O que mostram:** Falsos positivos concentram-se em "
            "{segment} (channel={segment}), sugerindo viés do modelo para "
            "esse segmento.",
            ">**Por que importa:** Se o modelo sistematicamente erra em "
            "um segmento, as recomendações para esse grupo serão "
            "ineficientes, desperdiçando recursos nele.",
            ">**Ação recomendada:** Treinar modelo separado para "
            "segmento enviesado, aumentar weight dessa classe, ou coletar "
            "mais dados representativos.",
            ">**Impacto:** Corrigir viés pode melhorar precision em "
            "~{fix:.1f}% para o segmento afetado, economizando "
            "\\$ {savings:,.0f} em campaign waste.",
        ),
    ),
    "summary.not_available": ("N/D", "N/D"),
    # executive summary ---------------------------------------------------
    "summary.crm_ok": ("generated", "gerado"),
    "summary.crm_missing": (
        "not generated (run export cell)",
        "nao gerado (executar celula de export)",
    ),
    "summary.pipeline_done": (
        "FULL PIPELINE EXECUTED SUCCESSFULLY",
        "PIPELINE COMPLETO EXECUTADO COM SUCESSO",
    ),
    "summary.report": (
        _md(
            "",
            _SPAN,
            "",
            "EXECUTIVE SUMMARY",
            "",
            "</span>",
            "",
            "1. Canonical user base: {users} users, {columns} columns; "
            "exposure base: {exposures} rows; variant switchers "
            "excluded: {switchers}",
            "2. RFM segments: {segments} segments",
            "3. Global Conversion Rate: {conversion:.2f}%",
            "4. Global Retention Rate: {retention:.2f}%",
            "",
            "LTV BY SEGMENT",
            "    - {high_name}: \\$ {high_ltv:,.2f}",
            "    - {low_name}: \\$ {low_ltv:,.2f}",
            "    - Global average LTV: \\$ {global_ltv:,.2f}",
            "",
            "DESIGN A/B",
            "    - Sample size per group: {required_n}",
            "    - Estimated runtime: {runtime} days",
            "    - Minimum lift: {lift:.1f} p.p.",
            "",
            "ROI SIMULATION",
            "    - Maximum incremental ROI: {max_roi:.2f}%",
            "    - Estimated Total Profit: \\$ {profit:.2f}",
            "",
            "PREDICTIVE MODEL",
            "    - ROC AUC: {roc_auc}",
            "    - Not Retained Recall: {recall}",
            "",
            "OUTPUTS",
            "    - Excel exported: {excel}",
            "    - CSV CRM: {crm} ({crm_status})",
            "    - Recommendations generated: {records} records",
            "",
        ),
        _md(
            "",
            _SPAN,
            "",
            "RESUMO EXECUTIVO",
            "",
            "</span>",
            "",
            "1. Base canônica: {users} usuários, {columns} colunas; "
            "base de exposições: {exposures} linhas; trocadores de "
            "variante excluídos: {switchers}",
            "2. Segmentos RFM: {segments} segmentos",
            "3. Taxa de Conversão Global: {conversion:.2f}%",
            "4. Taxa de Retenção Global: {retention:.2f}%",
            "",
            "LTV POR SEGMENTO",
            "    - {high_name}: \\$ {high_ltv:,.2f}",
            "    - {low_name}: \\$ {low_ltv:,.2f}",
            "    - LTV medio global: \\$ {global_ltv:,.2f}",
            "",
            "DESIGN A/B",
            "    - Sample size por grupo: {required_n}",
            "    - Runtime estimado: {runtime} dias",
            "    - Lift minimo: {lift:.1f} p.p.",
            "",
            "SIMULACAO DE ROI",
            "    - ROI incremental maximo: {max_roi:.2f}%",
            "    - Lucro Total Estimado: \\$ {profit:.2f}",
            "",
            "MODELO PREDITIVO",
            "    - ROC AUC: {roc_auc}",
            "    - Recall Nao Retido: {recall}",
            "",
            "OUTPUTS",
            "    - Excel exportado: {excel}",
            "    - CSV CRM: {crm} ({crm_status})",
            "    - Recomendações geradas: {records} registros",
            "",
        ),
    ),
    "decisions.realloc": (
        "Reallocating 10% of the budget from **{worst}** to **{best}** "
        "generates ~\\$ {extra:,.0f} in estimated additional profit "
        "(+{gain:.1f} pp in global ROI).",
        "Realocar 10% do budget de **{worst}** para **{best}** gera "
        "~\\$ {extra:,.0f} em lucro adicional estimado (+{gain:.1f} p.p. "
        "no ROI global).",
    ),
    "decisions.report": (
        _md(
            "",
            _SPAN,
            "",
            "## Executive Summary - Recommended Decisions",
            "",
            "</span>",
            "",
            "### What the data revealed",
            "",
            "| Dimension | Key Finding | Confidence |",
            "|---|---|---|",
            "| Funnel | Bottleneck is in conversion, not in retention "
            "| High |",
            "| Most efficient channel | {channel_finding} | High |",
            "| Personalization | correct language improves conversion "
            "and expected retention | High |",
            "| Most valuable segment | {high_name}: average LTV "
            "\\$ {high_ltv:,.2f} vs \\$ {low_ltv:,.2f} ({low_name}) "
            "| Medium |",
            "| Churn risk | Model identifies {detected:.1f}% of "
            "not-retained users | Medium |",
            "",
            "### 3 priority actions",
            "",
            "**1. Implement automatic language detection** (immediate "
            "impact, low cost)  ",
            "Fixing language for {n_wrong:,} users currently served in "
            "the wrong language generates ~\\$ {impact:,.0f} in "
            "estimated impact (conversion + retention/LTV).",
            "",
            "**2. Channel optimization based on real performance** "
            "(medium impact, 30-60 days)  ",
            "{action2}",
            "",
            "**3. Prioritize execution by incremental profit** (high "
            "impact, 30-90 days)  ",
            "{high_name} -> loyalty (max \\$ 15/customer). "
            "Medium Value -> promotional offer (max \\$ 8/customer). "
            "{low_name} -> exclude from paid campaigns or minimum "
            "win-back (max \\$ 3/customer). Do not act where "
            "incremental_profit <= 0.",
            "",
            "{realloc}",
            "",
            "### Immediate next step",
            "",
            "Export `crm_next_best_action.csv` generated in this "
            "notebook and import into the CRM. Prioritize the "
            "{n_action:,} customers with incremental_profit > 0 in the "
            "next campaign round. Reassess in 30 days with real "
            "conversion data.",
            "",
        ),
        _md(
            "",
            _SPAN,
            "",
            "## Sumário executivo - decisões recomendadas",
            "",
            "</span>",
            "",
            "### O que os dados revelaram",
            "",
            "| Dimensao | Achado principal | Confianca |",
            "|---|---|---|",
            "| Funil | Gargalo esta na conversão, nao na retenção "
            "| Alta |",
            "| Canal mais eficiente | {channel_finding} | Alta |",
            "| Personalização | Idioma correto melhora conversão e "
            "retenção esperada | Alta |",
            "| Segmento mais valioso | {high_name}: LTV medio de "
            "\\$ {high_ltv:,.2f} vs \\$ {low_ltv:,.2f} ({low_name}) "
            "| Media |",
            "| Risco de churn | Modelo identifica {detected:.1f}% dos "
            "nao-retidos | Media |",
            "",
            "### 3 ações prioritárias",
            "",
            "**1. Implementar detecção automática de idioma** (impacto "
            "imediato, custo baixo)  ",
            "Corrigir o idioma dos {n_wrong:,} usuarios atualmente "
            "mal-servidos gera ~\\$ {impact:,.0f} de impacto estimado "
            "(conversao + retencao/LTV).",
            "",
            "**2. Otimização de canal com base em performance real** "
            "(impacto medio, 30-60 dias)  ",
            "{action2}",
            "",
            "**3. Priorizar execução por lucro incremental** (impacto "
            "alto, 30-90 dias)  ",
            "{high_name} -> fidelização (max \\$ 15/cliente). "
            "Medio Valor -> oferta promocional (max \\$ 8/cliente). "
            "{low_name} -> excluir de campanhas pagas ou win-back minimo "
            "(max \\$ 3/cliente). Nao agir onde incremental_profit <= 0.",
            "",
            "{realloc}",
            "",
            "### Próximo passo imediato",
            "",
            "Exportar `crm_next_best_action.csv` gerado neste notebook "
            "e importar no CRM. Priorizar os {n_action:,} clientes com "
            "incremental_profit > 0 na proxima rodada de campanhas. "
            "Reavaliar em 30 dias com dados reais de conversao.",
            "",
        ),
    ),
}

# --- end of message pairs (add new ones above this line) ---

MESSAGES = {
    "en": {key: pair[0] for key, pair in _PAIRS.items()},
    "pt": {key: pair[1] for key, pair in _PAIRS.items()},
}
