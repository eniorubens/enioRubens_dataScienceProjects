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
}

# --- end of message pairs (add new ones above this line) ---

MESSAGES = {
    "en": {key: pair[0] for key, pair in _PAIRS.items()},
    "pt": {key: pair[1] for key, pair in _PAIRS.items()},
}
