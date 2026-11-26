"""Bilingual (EN/PT) texts for notebooks and charts.

The EN and PT notebooks run identical code; only ``LANG`` changes. Every title, label
and Markdown string comes from :data:`TEXTS` through :func:`t`.
"""

from __future__ import annotations

LANGS = ("en", "pt")

TEXTS: dict[str, dict[str, str]] = {
    "project_title": {
        "en": "Retail pricing and promotion with multilevel models",
        "pt": "Precificação e promoção no varejo com modelos multinível",
    },
    "data_credit": {
        "en": "Data: dunnhumby Source Files – Breakfast at the Frat",
        "pt": "Dados: dunnhumby Source Files – Breakfast at the Frat",
    },
    # --- group and promo labels -------------------------------------------------------
    "group_store": {"en": "Store", "pt": "Loja"},
    "group_product": {"en": "Product", "pt": "Produto"},
    "group_pair": {"en": "Store × product", "pt": "Loja × produto"},
    "group_residual": {"en": "Residual", "pt": "Resíduo"},
    "promo_feature": {"en": "Feature (flyer)", "pt": "Encarte"},
    "promo_display": {"en": "In-store display", "pt": "Display na loja"},
    "promo_tpr_only": {
        "en": "Temporary price reduction (TPR)",
        "pt": "Redução temporária de preço (TPR)",
    },
    "promo_any_promo": {"en": "Any promotion", "pt": "Qualquer promoção"},
    "promo_below_base": {"en": "Price below base", "pt": "Preço abaixo do base"},
    # --- charts -----------------------------------------------------------------------
    "legend_train": {"en": "Train", "pt": "Treino"},
    "legend_test": {"en": "Test", "pt": "Teste"},
    "all_rows": {"en": "All", "pt": "Todos"},
    "weekly_units_title": {"en": "Total units sold per week", "pt": "Total de unidades vendidas por semana"},
    "weekly_units_xlabel": {"en": "Week", "pt": "Semana"},
    "weekly_units_ylabel": {"en": "Units", "pt": "Unidades"},
    "promo_rates_title": {"en": "Promotion incidence", "pt": "Incidência de promoções"},
    "promo_rates_xlabel": {"en": "Share of observations", "pt": "Participação das observações"},
    "variance_share_title": {
        "en": "Share of variance in log(units)",
        "pt": "Participação na variância de log(unidades)",
    },
    "variance_share_xlabel": {"en": "Share of variance", "pt": "Participação na variância"},
    "panel_structure_title": {"en": "How the sales panel is grouped", "pt": "Como as vendas estão agrupadas no painel"},
    "panel_structure_crossed": {
        "en": "Crossed groups\nStore and product",
        "pt": "Agrupamentos cruzados\nLoja e produto",
    },
    "panel_structure_pair": {
        "en": "Level 2 · Longitudinal unit\nStore × UPC pair",
        "pt": "Nível 2 · Unidade longitudinal\nPar loja × UPC",
    },
    "panel_structure_week": {
        "en": "Level 1 · Weekly observations\nRepeated within the pair",
        "pt": "Nível 1 · Observações semanais\nRepetidas dentro do par",
    },
    "panel_structure_store": {"en": "Store {code}", "pt": "Loja {code}"},
    "panel_structure_pair_node": {"en": "Store {store} × UPC {upc}", "pt": "Loja {store} × UPC {upc}"},
    "panel_structure_week_node": {"en": "Week {time}", "pt": "Semana {time}"},
    "panel_structure_note": {
        "en": "Illustrative example. Weekly observations are shown for the highlighted pair.",
        "pt": "Exemplo esquemático. As observações semanais aparecem para o par destacado.",
    },
    "icc_by_model_title": {
        "en": "Variance decomposition by model",
        "pt": "Decomposição da variância por modelo",
    },
    "icc_by_model_xlabel": {"en": "Share of variance", "pt": "Participação na variância"},
    "pair_coverage_title": {
        "en": "Training weeks per store × product pair",
        "pt": "Semanas de treino por par loja × produto",
    },
    "pair_coverage_xlabel": {"en": "Training weeks", "pt": "Semanas de treino"},
    "pair_coverage_ylabel": {"en": "Number of pairs", "pt": "Número de pares"},
    "pair_coverage_median": {"en": "Median: {value:.0f} weeks", "pt": "Mediana: {value:.0f} semanas"},
    "pair_slopes_title": {
        "en": "Per-pair OLS discount slopes (training weeks)",
        "pt": "Inclinações do desconto no OLS por par (semanas de treino)",
    },
    "pair_slopes_xlabel": {
        "en": "Discount slope (LOG_DISC coefficient)",
        "pt": "Inclinação do desconto (coeficiente de LOG_DISC)",
    },
    "pair_slopes_ylabel": {"en": "Number of pairs", "pt": "Número de pares"},
    "pair_slopes_outside": {
        "en": "{n} pairs beyond the 1st-99th percentile range not drawn",
        "pt": "{n} pares fora do intervalo entre os percentis 1 e 99 não desenhados",
    },
    "caterpillar_title": {
        "en": "Random intercepts (BLUP) with 95% intervals: {group}",
        "pt": "Interceptos aleatórios (BLUP) com intervalos de 95%: {group}",
    },
    "caterpillar_xlabel": {"en": "Predicted deviation in log(units)", "pt": "Desvio previsto em log(unidades)"},
    "corr_same_pair": {
        "en": "Correlation between two weeks of the same pair",
        "pt": "Correlação entre duas semanas do mesmo par",
    },
    "corr_same_store": {
        "en": "Correlation between two products in the same store",
        "pt": "Correlação entre dois produtos na mesma loja",
    },
    "segment_ambiguous": {"en": "Ambiguous", "pt": "Ambíguo"},
    "brand_national": {"en": "National brand", "pt": "Marca nacional"},
    "brand_private": {"en": "Private label", "pt": "Marca própria"},
    "n_manufacturers": {"en": "Manufacturers (incl. private label)", "pt": "Fabricantes (incl. marca própria)"},
    "raw_lift_title": {"en": "Raw promotion lift in units", "pt": "Lift bruto de promoção em unidades"},
    "raw_lift_xlabel": {"en": "Lift in units (%)", "pt": "Lift em unidades (%)"},
    "raw_lift_caption": {
        "en": "Descriptive: within-pair, confounded with price cuts",
        "pt": "Descritivo: dentro do par, confundido com redução de preço",
    },
    # --- fixed effects (notebook 03) --------------------------------------------------
    "term_intercept": {"en": "Intercept", "pt": "Intercepto"},
    "term_log_disc": {"en": "Discount depth (log price / base)", "pt": "Profundidade do desconto (log preço / base)"},
    "term_log_base_dev": {
        "en": "Regular price, within pair (log)",
        "pt": "Preço regular, dentro do par (log)",
    },
    "term_log_base_mean": {"en": "Regular price, pair mean (log)", "pt": "Preço regular, média do par (log)"},
    "term_trend": {"en": "Trend (per year)", "pt": "Tendência (por ano)"},
    "term_sin": {"en": "Seasonality: sine", "pt": "Sazonalidade: seno"},
    "term_cos": {"en": "Seasonality: cosine", "pt": "Sazonalidade: cosseno"},
    "coef_title": {
        "en": "Fixed effects on log(units), 95% intervals",
        "pt": "Efeitos fixos em log(unidades), intervalos de 95%",
    },
    "coef_xlabel": {"en": "Coefficient (log scale)", "pt": "Coeficiente (escala log)"},
    "pseudo_r2_title": {
        "en": "Variance reduction by price and promotion, per level",
        "pt": "Redução da variância por preço e promoção, por nível",
    },
    "pseudo_r2_xlabel": {
        "en": "Reduction relative to the null model",
        "pt": "Redução em relação ao modelo nulo",
    },
    "method_comparison_title": {
        "en": "Same specification, three estimators (95% intervals)",
        "pt": "Mesma especificação, três estimadores (intervalos de 95%)",
    },
    "method_comparison_xlabel": {"en": "Coefficient (log scale)", "pt": "Coeficiente (escala log)"},
    "method_ols_pooled": {"en": "Pooled OLS", "pt": "OLS agrupado"},
    "method_ols_upc_fe": {"en": "OLS with product fixed effects", "pt": "OLS com efeitos fixos de produto"},
    "method_multilevel": {
        "en": "Multilevel (crossed random intercepts)",
        "pt": "Multinível (interceptos aleatórios cruzados)",
    },
    "seasonality_title": {
        "en": "Seasonal profile implied by the Fourier terms",
        "pt": "Perfil sazonal implícito nos termos de Fourier",
    },
    "seasonality_xlabel": {"en": "Week of the year", "pt": "Semana do ano"},
    "seasonality_ylabel": {
        "en": "Effect on units vs. average week (%)",
        "pt": "Efeito nas unidades vs. semana média (%)",
    },
    "effects_col_effect": {"en": "Effect", "pt": "Efeito"},
    "effects_col_estimate": {"en": "Change in units (%)", "pt": "Variação nas unidades (%)"},
    "effects_col_low": {"en": "95% CI, lower (%)", "pt": "IC de 95%, inferior (%)"},
    "effects_col_high": {"en": "95% CI, upper (%)", "pt": "IC de 95%, superior (%)"},
    "effect_disc10": {"en": "10% deeper discount", "pt": "Desconto 10% mais profundo"},
    "effect_base10": {"en": "10% higher regular price", "pt": "Preço regular 10% maior"},
    "effect_feature": {"en": "Feature (flyer), beyond the price cut", "pt": "Encarte, além da redução de preço"},
    "effect_display": {
        "en": "In-store display, beyond the price cut",
        "pt": "Display na loja, além da redução de preço",
    },
    "effect_tpr": {"en": "TPR only, beyond the price cut", "pt": "Apenas TPR, além da redução de preço"},
    "effect_trend": {"en": "One more year (trend)", "pt": "Um ano a mais (tendência)"},
    "resid_lag_title": {
        "en": "Residual correlation within a pair, by lag",
        "pt": "Correlação dos resíduos dentro do par, por defasagem",
    },
    "resid_lag_xlabel": {"en": "Lag (weeks)", "pt": "Defasagem (semanas)"},
    "resid_lag_ylabel": {"en": "Correlation", "pt": "Correlação"},
    "week_shocks_title": {
        "en": "Mean residual by week (training weeks)",
        "pt": "Resíduo médio por semana (semanas de treino)",
    },
    "week_shocks_xlabel": {"en": "Week", "pt": "Semana"},
    "week_shocks_ylabel": {"en": "Mean residual (log units)", "pt": "Resíduo médio (log de unidades)"},
    "week_shocks_band": {
        "en": "95% band if rows were independent",
        "pt": "Faixa de 95% se as linhas fossem independentes",
    },
    # --- scenarios and promotion support (notebook 03) --------------------------------
    "scenario_regular": {
        "en": "Regular week (no discount, no promotion)",
        "pt": "Semana regular (sem desconto nem promoção)",
    },
    "scenario_tpr10": {"en": "TPR with 10% discount", "pt": "TPR com 10% de desconto"},
    "scenario_tpr20": {"en": "TPR with 20% discount", "pt": "TPR com 20% de desconto"},
    "scenario_feature0": {"en": "Feature, no discount", "pt": "Encarte, sem desconto"},
    "scenario_feature20": {"en": "Feature with 20% discount", "pt": "Encarte com 20% de desconto"},
    "scenario_display0": {"en": "Display, no discount", "pt": "Display, sem desconto"},
    "scenario_display20": {"en": "Display with 20% discount", "pt": "Display com 20% de desconto"},
    "scenario_feature_display25": {
        "en": "Feature and display with 25% discount",
        "pt": "Encarte e display com 25% de desconto",
    },
    "scenario_col_name": {"en": "Scenario", "pt": "Cenário"},
    "scenario_col_discount": {"en": "Discount (%)", "pt": "Desconto (%)"},
    "scenario_col_effect": {
        "en": "Change in units vs. regular week (%)",
        "pt": "Variação nas unidades vs. semana regular (%)",
    },
    "scenario_col_rows": {"en": "Training rows realising it", "pt": "Linhas de treino que o realizam"},
    "scenario_col_share": {"en": "Share of training rows (%)", "pt": "Participação nas linhas de treino (%)"},
    "kind_none": {"en": "No promotion", "pt": "Sem promoção"},
    "kind_tpr": {"en": "TPR only", "pt": "Apenas TPR"},
    "kind_feature": {"en": "Feature only", "pt": "Apenas encarte"},
    "kind_display": {"en": "Display only", "pt": "Apenas display"},
    "kind_feature_display": {"en": "Feature and display", "pt": "Encarte e display"},
    "band_none": {"en": "No price cut", "pt": "Sem redução de preço"},
    "band_0_5": {"en": "0-5%", "pt": "0-5%"},
    "band_5_10": {"en": "5-10%", "pt": "5-10%"},
    "band_10_20": {"en": "10-20%", "pt": "10-20%"},
    "band_20_30": {"en": "20-30%", "pt": "20-30%"},
    "band_30_plus": {"en": "Above 30%", "pt": "Acima de 30%"},
    # --- protocol P1 (notebook 04) ----------------------------------------------------
    "model_c0": {"en": "C0", "pt": "C0"},
    "model_c1": {"en": "C1", "pt": "C1"},
    "model_ols_pooled": {"en": "Pooled OLS", "pt": "OLS agrupado"},
    "model_ols_pair": {"en": "Per-pair OLS", "pt": "OLS por par"},
    "model_gpboost": {"en": "GPBoost", "pt": "GPBoost"},
    "stratum_lt26": {"en": "< 26 weeks", "pt": "< 26 semanas"},
    "stratum_26_77": {"en": "26-77 weeks", "pt": "26-77 semanas"},
    "stratum_ge78": {"en": "≥ 78 weeks", "pt": "≥ 78 semanas"},
    "pop_a_seen": {"en": "(a) Seen pairs", "pt": "(a) Pares vistos"},
    "pop_b_ols_covered": {"en": "(b) Rows covered by per-pair OLS", "pt": "(b) Linhas cobertas pelo OLS por par"},
    "pop_c_unseen": {"en": "(c) Unseen pairs", "pt": "(c) Pares inéditos"},
    "col_model": {"en": "Model", "pt": "Modelo"},
    "col_fold": {"en": "Fold", "pt": "Dobra"},
    "col_stratum": {"en": "History stratum", "pt": "Estrato de histórico"},
    "col_pairs": {"en": "Pairs", "pt": "Pares"},
    "col_rows": {"en": "Rows", "pt": "Linhas"},
    "col_candidate": {"en": "Candidate", "pt": "Candidato"},
    "col_reference": {"en": "Reference", "pt": "Referência"},
    "col_measure": {"en": "Population, stratum and metric", "pt": "População, estrato e métrica"},
    "col_difference_pct": {"en": "Difference (%)", "pt": "Diferença (%)"},
    "fold_label": {"en": "Fold {fold}", "pt": "Dobra {fold}"},
    "fold_final": {"en": "Final window", "pt": "Janela final"},
    "metric_mae_pair": {"en": "MAE per pair", "pt": "MAE por par"},
    "metric_mae_row": {"en": "MAE per row", "pt": "MAE por linha"},
    "metric_rmse": {"en": "RMSE", "pt": "RMSE"},
    "sel_compared_to": {"en": "Compared with", "pt": "Comparado com"},
    "sel_mae_a": {"en": "MAE, fold A", "pt": "MAE, dobra A"},
    "sel_mae_a_inc": {"en": "MAE, fold A, incumbent", "pt": "MAE, dobra A, modelo aceito"},
    "sel_mae_b": {"en": "MAE, fold B", "pt": "MAE, dobra B"},
    "sel_mae_b_inc": {"en": "MAE, fold B, incumbent", "pt": "MAE, dobra B, modelo aceito"},
    "sel_gain_a": {"en": "Gain, fold A (%)", "pt": "Ganho, dobra A (%)"},
    "sel_gain_b": {"en": "Gain, fold B (%)", "pt": "Ganho, dobra B (%)"},
    "sel_gain_mean": {"en": "Mean gain (%)", "pt": "Ganho médio (%)"},
    "sel_decision": {"en": "Decision", "pt": "Decisão"},
    "decision_accepted": {"en": "accepted", "pt": "aceito"},
    "decision_rejected": {"en": "rejected", "pt": "rejeitado"},
    "decision_not_fitted": {"en": "not fitted", "pt": "não ajustado"},
    "wk_weeks": {"en": "Weeks", "pt": "Semanas"},
    "wk_better": {"en": "Weeks candidate is better", "pt": "Semanas em que o candidato é melhor"},
    "wk_mean": {"en": "Mean difference", "pt": "Diferença média"},
    "wk_min": {"en": "Minimum difference", "pt": "Diferença mínima"},
    "wk_max": {"en": "Maximum difference", "pt": "Diferença máxima"},
    "gpb_col_c0": {"en": "C0, per-pair MAE", "pt": "C0, MAE por par"},
    "gpb_col_gpb": {"en": "GPBoost, per-pair MAE", "pt": "GPBoost, MAE por par"},
    "gpb_col_diff": {"en": "GPBoost vs. C0 (%)", "pt": "GPBoost vs. C0 (%)"},
    "cov_seen_pairs": {"en": "Seen pairs", "pt": "Pares vistos"},
    "cov_seen_rows": {"en": "Seen rows", "pt": "Linhas vistas"},
    "cov_pairs_with_ols": {"en": "Pairs with a local OLS", "pt": "Pares com OLS local"},
    "cov_covered_pairs": {"en": "Pairs covered", "pt": "Pares cobertos"},
    "cov_covered_rows": {"en": "Rows covered", "pt": "Linhas cobertas"},
    "cov_pct_pairs": {"en": "Pairs covered (%)", "pt": "Pares cobertos (%)"},
    "cov_pct_rows": {"en": "Rows covered (%)", "pt": "Linhas cobertas (%)"},
    "cov_unseen_pairs": {"en": "Unseen pairs", "pt": "Pares inéditos"},
    "cov_unseen_rows": {"en": "Unseen rows", "pt": "Linhas de pares inéditos"},
    "cov_none_rows": {"en": "Rows without UPC history", "pt": "Linhas sem histórico do UPC"},
    "fold_mae_title": {
        "en": "Per-pair MAE by internal fold and history stratum (seen pairs)",
        "pt": "MAE por par em cada dobra interna e estrato de histórico (pares vistos)",
    },
    "fold_mae_ylabel": {"en": "MAE of log units (mean over pairs)", "pt": "MAE de log-unidades (média entre pares)"},
    "weekly_diff_title": {
        "en": "Weekly error difference, C0 minus reference",
        "pt": "Diferença semanal de erro, C0 menos a referência",
    },
    "weekly_diff_xlabel": {"en": "Week", "pt": "Semana"},
    "weekly_diff_ylabel": {
        "en": "Change in mean absolute error (negative = C0 better)",
        "pt": "Variação do erro absoluto médio (negativo = C0 melhor)",
    },
    # --- notebooks 05 and 06: contrasts, category x week sensitivity, scenarios -----------
    "contrast_test_segment": {"en": "Segment", "pt": "Segmento"},
    "contrast_test_private_label": {"en": "Brand type", "pt": "Tipo de marca"},
    "contrast_test_category": {"en": "Category", "pt": "Categoria"},
    "slopes_title": {
        "en": "Discount slope by level (pointwise 95% intervals)",
        "pt": "Inclinação do desconto por nível (intervalos pontuais de 95%)",
    },
    "slopes_xlabel": {
        "en": "Slope of log(units) on LOG_DISC (two-way clustered)",
        "pt": "Inclinação de log(unidades) em LOG_DISC (agrupada em duas vias)",
    },
    "slopes_reference": {"en": "reference", "pt": "referência"},
    "model_f3_ref": {"en": "F3 (reference)", "pt": "F3 (referência)"},
    "model_f3_catweek": {"en": "F3 + category × week", "pt": "F3 + categoria × semana"},
    "catweek_compare_title": {
        "en": "Fixed effects without and with a category × week intercept",
        "pt": "Efeitos fixos sem e com intercepto de categoria × semana",
    },
    "catweek_compare_xlabel": {
        "en": "Coefficient (interval: model-based standard error)",
        "pt": "Coeficiente (intervalo: erro-padrão do modelo)",
    },
    "lag_compare_title": {
        "en": "Same-pair residual correlation by calendar lag",
        "pt": "Correlação residual no mesmo par por defasagem de calendário",
    },
    "scen_effects_title": {
        "en": "Modelled change in sales (pointwise 95% interval)",
        "pt": "Variação modelada nas vendas (intervalo pontual de 95%)",
    },
    "scen_effects_xlabel": {"en": "Change in sales (%)", "pt": "Variação nas vendas (%)"},
    "scen_support_title": {"en": "Observed support", "pt": "Suporte observado"},
    "scen_support_xlabel": {
        "en": "Share of training rows (%, log scale)",
        "pt": "Participação nas linhas de treino (%, escala log)",
    },
    "scen_outside": {"en": "outside observed support", "pt": "fora do suporte observado"},
    "scen_fallback": {"en": "{n} with fallback base price", "pt": "{n} com preço-base de fallback"},
    "p_unavailable": {"en": "unavailable", "pt": "indisponível"},
    "slope_lr_measure": {"en": "Measure", "pt": "Medida"},
    "slope_lr_value": {"en": "Value", "pt": "Valor"},
    "slope_lr_observed": {"en": "Observed LR (C0 vs. C1)", "pt": "RV observada (C0 × C1)"},
    "slope_lr_max_simulated": {"en": "Largest simulated LR under C0", "pt": "Maior RV simulada sob C0"},
    "slope_lr_count": {"en": "Valid / planned replicates", "pt": "Réplicas válidas / planejadas"},
    "slope_lr_exceedances": {
        "en": "Simulated LR at least as large as observed",
        "pt": "RV simuladas iguais ou maiores que a observada",
    },
    "slope_lr_p": {"en": "Monte Carlo p-value", "pt": "Valor-p Monte Carlo"},
    "slope_lr_store_sd": {
        "en": "Discount slope SD by store (C1)",
        "pt": "Desvio-padrão da inclinação de desconto por loja (C1)",
    },
    "slope_lr_upc_sd": {
        "en": "Discount slope SD by UPC (C1)",
        "pt": "Desvio-padrão da inclinação de desconto por UPC (C1)",
    },
    "group_catweek": {"en": "Category × week", "pt": "Categoria × semana"},
}

# Column labels of the tables of notebooks 05 and 06, as (en, pt); keys are ``f_<column>``.
_FIELDS: dict[str, tuple[str, str]] = {
    "test": ("Test", "Teste"),
    "level": ("Level", "Nível"),
    "moderator": ("Moderator", "Moderador"),
    "reference": ("Reference", "Referência"),
    "n_rows_input": ("Training rows", "Linhas de treino"),
    "n_rows": ("Rows", "Linhas"),
    "n_pairs": ("Pairs", "Pares"),
    "n_excluded": ("Excluded rows", "Linhas excluídas"),
    "reason": ("Exclusion reason", "Motivo da exclusão"),
    "within_pair_sd_log_disc": ("Within-pair SD of LOG_DISC", "DP intrapar de LOG_DISC"),
    "n_pairs_with_variation": ("Pairs with discount variation", "Pares com variação de desconto"),
    "is_reference": ("Reference level", "Nível de referência"),
    "slope": ("Slope", "Inclinação"),
    "se_pair": ("SE by pair", "EP por par"),
    "se_week": ("SE by week", "EP por semana"),
    "se_twoway": ("SE two-way", "EP em duas vias"),
    "ci_low": ("Lower 95% (pointwise)", "Inferior 95% (pontual)"),
    "ci_high": ("Upper 95% (pointwise)", "Superior 95% (pontual)"),
    "diff_vs_ref": ("Difference vs. reference", "Diferença vs. referência"),
    "se_diff": ("SE of difference", "EP da diferença"),
    "diff_ci_low": ("Difference, lower 95%", "Diferença, inferior 95%"),
    "diff_ci_high": ("Difference, upper 95%", "Diferença, superior 95%"),
    "estimate": ("Estimate", "Estimativa"),
    "q": ("Restrictions (q)", "Restrições (q)"),
    "W": ("Wald statistic", "Estatística de Wald"),
    "F": ("F = W / q", "F = W / q"),
    "df1": ("Numerator df", "GL do numerador"),
    "df2": ("Denominator df", "GL do denominador"),
    "G_pair": ("Pair clusters", "Clusters de par"),
    "G_week": ("Week clusters", "Clusters de semana"),
    "p_raw": ("Raw p-value", "Valor-p bruto"),
    "p_holm": ("Holm-adjusted p-value", "Valor-p ajustado (Holm)"),
    "reject_5pct": ("Rejects at 5%", "Rejeita a 5%"),
    "available": ("Available", "Disponível"),
    "design_rank": ("Design rank", "Posto do desenho"),
    "n_terms": ("Design columns", "Colunas do desenho"),
    "rank_deficient": ("Rank deficient", "Posto deficiente"),
    "cond_number": ("Condition number of the q × q block", "Número de condição do bloco q × q"),
    "min_eig": ("Smallest eigenvalue of the block", "Menor autovalor do bloco"),
    "twoway_psd_fixed": ("PSD correction applied", "Correção PSD aplicada"),
    "design_cond": ("Design condition number", "Número de condição do desenho"),
    "nobs": ("Observations", "Observações"),
    "nobs_restricted": ("Observations, restricted model", "Observações, modelo restrito"),
    "n_stores": ("Stores", "Lojas"),
    "n_upcs": ("Products", "Produtos"),
    "n_categories": ("Categories", "Categorias"),
    "n_catweek_levels": ("Category × week levels", "Níveis de categoria × semana"),
    "n_weeks": ("Weeks", "Semanas"),
    "max_week": ("Last week", "Última semana"),
    "n_missing_category": ("Rows without category", "Linhas sem categoria"),
    "model": ("Model", "Modelo"),
    "optimizer": ("Optimizer", "Otimizador"),
    "messages": ("Convergence messages", "Mensagens de convergência"),
    "singular": ("Singular fit", "Ajuste singular"),
    "seconds": ("Seconds", "Segundos"),
    "npar": ("Parameters", "Parâmetros"),
    "loglik_reml": ("REML log-likelihood", "Log-verossimilhança REML"),
    "used": ("Used in the comparison", "Usado na comparação"),
    "term": ("Term", "Termo"),
    "est_ref": ("Estimate, F3", "Estimativa, F3"),
    "se_ref": ("SE, F3", "EP, F3"),
    "est_catweek": ("Estimate, with category × week", "Estimativa, com categoria × semana"),
    "se_catweek": ("SE, with category × week", "EP, com categoria × semana"),
    "diff": ("Difference", "Diferença"),
    "se_ratio": ("SE ratio", "Razão dos EP"),
    "group": ("Grouping", "Agrupamento"),
    "variance": ("Variance", "Variância"),
    "sd": ("Standard deviation", "Desvio-padrão"),
    "share": ("Share", "Participação"),
    "lag": ("Lag (weeks)", "Defasagem (semanas)"),
    "corr": ("Correlation", "Correlação"),
    "n": ("Pairs of rows", "Pares de linhas"),
    "sigma2": ("Residual variance", "Variância residual"),
    "var_week_means": ("Variance of weekly mean residuals", "Variância dos resíduos médios semanais"),
    "var_expected_independent": ("Expected if independent", "Esperada sob independência"),
    "variance_ratio": ("Ratio to independence", "Razão frente à independência"),
    "share_week": ("Share of variance, week", "Participação da variância, semana"),
    "share_week_expected": ("Expected share, week", "Participação esperada, semana"),
    "autocorr_week_means": ("Lag-1 autocorrelation of weekly means", "Autocorrelação de lag 1 das médias semanais"),
    "share_category_week": ("Share of variance, category × week", "Participação da variância, categoria × semana"),
    "share_category_week_expected": ("Expected share, category × week", "Participação esperada, categoria × semana"),
    "scenario": ("Scenario", "Cenário"),
    "discount": ("Discount (%)", "Desconto (%)"),
    "feature": ("Feature", "Encarte"),
    "display": ("Display", "Display"),
    "tpr": ("TPR", "TPR"),
    "dlog": ("Log difference", "Diferença em log"),
    "se": ("SE (joint covariance)", "EP (covariância conjunta)"),
    "effect_pct": ("Modelled change in sales (%)", "Variação modelada nas vendas (%)"),
    "ci_low_pct": ("Lower 95% (%)", "Inferior 95% (%)"),
    "ci_high_pct": ("Upper 95% (%)", "Superior 95% (%)"),
    "share_pct": ("Share of training rows (%)", "Participação nas linhas de treino (%)"),
    "n_rows_observed_base": ("Rows, observed base price", "Linhas, preço-base observado"),
    "n_pairs_observed_base": ("Pairs, observed base price", "Pares, preço-base observado"),
    "share_observed_base": ("Share, observed base price (%)", "Participação, preço-base observado (%)"),
    "n_rows_fallback": ("Rows with fallback base price", "Linhas com preço-base de fallback"),
    "outside_support": ("Outside observed support", "Fora do suporte observado"),
    "stored_estimate": ("Stored estimate (notebook 03)", "Estimativa gravada (notebook 03)"),
    "refit_estimate": ("Refit estimate", "Estimativa do reajuste"),
    "abs_diff": ("Absolute difference", "Diferença absoluta"),
    "effect_nb03_pct": ("Change from stored coefficients (%)", "Variação a partir dos coeficientes gravados (%)"),
    "effect_diff_pp": ("Max. absolute effect difference (pp)", "Diferença absoluta máxima nos efeitos (p.p.)"),
    "coef_max_abs_diff": ("Max. absolute coefficient difference", "Diferença absoluta máxima nos coeficientes"),
    "stat": ("LR statistic", "Estatística LR"),
    "df": ("Degrees of freedom", "Graus de liberdade"),
    "p_value": ("p-value (50:50 mixture, descriptive)", "Valor-p (mistura 50:50, descritivo)"),
    "ratio": ("Modelled sales ratio, display vs. TPR", "Razão modelada de vendas, display vs. TPR"),
    "ratio_low": ("Ratio, lower 95%", "Razão, inferior 95%"),
    "ratio_high": ("Ratio, upper 95%", "Razão, superior 95%"),
    "common_slope": ("Common slope of F3 (LOG_DISC)", "Inclinação comum do F3 (LOG_DISC)"),
    "sd_over_common": ("SD / |common slope|", "DP / |inclinação comum|"),
    "stratum": ("History stratum", "Estrato de histórico"),
    "n_positive": ("Pairs with slope > 0", "Pares com inclinação > 0"),
    "share_positive_pct": ("Share with slope > 0 (%)", "Participação com inclinação > 0 (%)"),
    "q05": ("5th percentile", "Percentil 5"),
    "q25": ("25th percentile", "Percentil 25"),
    "median": ("Median", "Mediana"),
    "q75": ("75th percentile", "Percentil 75"),
    "q95": ("95th percentile", "Percentil 95"),
}
TEXTS.update({f"f_{col}": {"en": en, "pt": pt} for col, (en, pt) in _FIELDS.items()})


STRATUM_ORDER = ("all", "<26", "26-77", ">=78")
MODEL_ORDER = ("C0", "C1", "GPBoost", "OLS_pair", "OLS_pooled")
_STRATUM_KEYS = {"all": "all_rows", "<26": "stratum_lt26", "26-77": "stratum_26_77", ">=78": "stratum_ge78"}
_MODEL_KEYS = {
    "C0": "model_c0",
    "C1": "model_c1",
    "GPBoost": "model_gpboost",
    "OLS_pair": "model_ols_pair",
    "OLS_pooled": "model_ols_pooled",
}


def t(key: str, lang: str) -> str:
    """Return the text ``key`` in ``lang`` ('en' or 'pt')."""
    if lang not in LANGS:
        raise ValueError(f"lang must be one of {LANGS}, got {lang!r}")
    return TEXTS[key][lang]


def stratum_label(stratum: str, lang: str) -> str:
    """Translate a history stratum code (``all``, ``<26``, ``26-77``, ``>=78``); unknown codes are kept."""
    key = _STRATUM_KEYS.get(str(stratum))
    return t(key, lang) if key else str(stratum)


def model_label(model: str, lang: str) -> str:
    """Translate a model code of the P1 runs (``C0``, ``OLS_pair``, ...); unknown codes are kept."""
    key = _MODEL_KEYS.get(str(model))
    return t(key, lang) if key else str(model)


_CONTRAST_TESTS = ("segment", "private_label", "category")


def format_p(p: float) -> str:
    """Format a p-value for display: scientific when small, and never as a bare zero.

    Values that underflow to zero (below ``1e-300``) are shown as ``"< 1e-300"``.
    """
    if p != p:  # NaN
        return "n/a"
    if p < 1e-300:
        return "< 1e-300"
    if p < 1e-3:
        return f"{p:.2e}"
    return f"{p:.4f}"


def contrast_label(test: str, lang: str) -> str:
    """Localised name of a planned contrast (``segment``, ``private_label``, ``category``); unknown names are kept."""
    return t(f"contrast_test_{test}", lang) if str(test) in _CONTRAST_TESTS else str(test)


def level_label(test: str, level: object, lang: str) -> str:
    """Localised label of a moderator level; private-label levels (``False``/``True``) become brand labels."""
    if str(test) == "private_label":
        return t("brand_private" if str(level) == "True" else "brand_national", lang)
    return str(level)


def scenario_label(name: str, lang: str) -> str:
    """Localised scenario name when ``name`` is an i18n key, otherwise the name itself."""
    return t(name, lang) if name in TEXTS else str(name)


def localize_columns(df, lang: str):
    """Return ``df`` with the columns that have an ``f_<column>`` label renamed to ``lang``; others are kept."""
    return df.rename(columns={c: t(f"f_{c}", lang) for c in df.columns if f"f_{c}" in TEXTS})
