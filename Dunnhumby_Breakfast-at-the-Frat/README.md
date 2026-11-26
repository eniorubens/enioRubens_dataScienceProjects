# Retail pricing and promotion with multilevel models

[Português](README.pt-BR.md) · [Executive report](site/en.html)

> **Status:** protocol P1 (model selection and final evaluation) and the training-only analyses of notebooks 05 and 06 are concluded. Open and not executed items are listed at the end of this README.

## Business question

How strongly are **price** and **promotion** (feature/flyer, in-store display and temporary price reduction, TPR) associated with unit sales **across stores and products**, and how do comparable pricing and promotion scenarios rank, given the observed support for each? Margin, promotion cost, ROI and causal price recommendations are out of scope.

The central argument is **partial pooling (shrinkage)**. Elasticities estimated store by store with OLS are noisy and sometimes positive, which makes no economic sense. A multilevel (mixed-effects) model stabilises them by borrowing strength from the whole panel. The project tests this claim with evidence, including out-of-sample.

## Approach (summary)

- Response: `log(UNITS)` (linear mixed model). The planned negative binomial GLMM robustness check was not run.
- Steps actually run: null models and ICCs with boundary-corrected likelihood-ratio tests → fixed effects (discount depth, base price split into within-pair deviation and pair mean, feature, display, TPR, Fourier seasonality and trend) with crossed random intercepts (store, product, store×product) → random discount slopes by store and product (C1), judged by out-of-sample error in protocol P1 and later by a parametric-bootstrap likelihood-ratio test in training → planned interactions of the discount slope with store segment, private label and category (Holm) → category × week shock sensitivity → scenarios. Promotion-lift slopes, store-size interactions and covariance-structure extensions were not fitted.
- Comparison: separate OLS per store×product vs pooled OLS vs multilevel (BLUP), plus a GPBoost challenger. Model selection uses two internal folds inside the first 104 weeks. The final comparison uses the last 52 weeks, which are a **retrospective temporal evaluation window with prior exploratory exposure** (decision D11, revised). The models of P1's formal evaluation were fitted on the first 104 weeks. The brief's exploratory OLS used the whole panel. The comparison followed protocol P1, registered before any candidate was fitted. **P1 is concluded.** The selected specification, the GPBoost parameters and the hashes of the code and panel were frozen **before the formal scoring** of the final window, which was then scored once. Predictions are conditional on the realised price and promotion of each week.
- Engine: `lme4` (R, installed inside the conda env from conda-forge) called from Python through `rpy2`; `statsmodels.MixedLM` cross-checks the store-only null model.
- Weeks without a row (zero sales or stock-out cannot be told apart) are left out of the model, so estimates are conditional on recorded sales. The planned sensitivity on pairs with a complete 156-week panel was not run.

## How to get the data

The data are **not** included in this repository (dunnhumby licence).

1. Register and download *Breakfast at the Frat: A Time Series Analysis* from https://www.dunnhumby.com/source-files/.
2. Put `dunnhumby - Breakfast at the Frat.xlsx` (and the user guide PDF) into `dataset/`.

Data: dunnhumby Source Files – Breakfast at the Frat.

## Reproduction

~~~
conda env create -f environment.yml
conda activate retail-multilevel
python scripts/ingest.py        # Excel -> data/interim/*.parquet (~1 min, once)
python scripts/build_panel.py   # cleaning rules -> data/processed/panel.parquet
python scripts/validate_data.py # data-quality contract on the real panel (local only)
pytest                          # synthetic fixtures only; this is what CI runs
~~~

Models are fitted by scripts, once per experiment, into `outputs/runs/`. The PT and EN notebooks only read those stored runs and never refit.

On Windows, the conda env ships its own R; `retail_multilevel.rbridge` points `R_HOME` to it even if a system R is installed. Cleaning rules and their row counts are shown in notebook 01.

## Notebook map

| Stage | PT-BR | EN-US | What it covers |
| :--- | :---: | :---: | :--- |
| 01 — Panel and exploration | [PT-BR](notebooks/pt_br/01_eda_painel.ipynb) | [EN-US](notebooks/en_us/01_panel_eda.ipynb) | business briefing, cleaning rules, temporal split, descriptive EDA, discount slopes estimated pair by pair |
| 02 — Null model | [PT-BR](notebooks/pt_br/02_modelo_nulo.ipynb) | [EN-US](notebooks/en_us/02_null_model.ipynb) | null models, ICCs, likelihood-ratio tests with boundary correction |
| 03 — Fixed effects | [PT-BR](notebooks/pt_br/03_efeitos_fixos.ipynb) | [EN-US](notebooks/en_us/03_fixed_effects.ipynb) | price and promotion, scenarios, sensitivities, residual dependence |
| 04 — Protocol P1 | [PT-BR](notebooks/pt_br/04_protocolo_p1.ipynb) | [EN-US](notebooks/en_us/04_protocol_p1.ipynb) | pre-registered protocol, random slopes, single final evaluation, parametric-bootstrap test of the slopes |
| 05 — Contrasts and sensitivity | [PT-BR](notebooks/pt_br/05_contrastes_e_sensibilidade.ipynb) | [EN-US](notebooks/en_us/05_contrasts_and_sensitivity.ipynb) | planned contrasts of the discount slope (segment, private label, category) with Holm, category × week shocks |
| 06 — Pricing and promotion scenarios | [PT-BR](notebooks/pt_br/06_cenarios_preco_promocao.ipynb) | [EN-US](notebooks/en_us/06_pricing_promotion_scenarios.ipynb) | eight scenarios with joint covariance, observed support, dispersion of the slope across products |

## Main results (protocol P1)

In the internal folds, adding random discount slopes by store and product (C1) improved pairs with less than 26 weeks of history in both folds, but failed the pre-registered global rule. The intercept-only multilevel model (C0, with a common discount slope) was therefore selected. The final window evaluated C0, so the question of random slopes for short pairs remains **open**.

In the final window, the multilevel model clearly beat pooled OLS and per-store×product OLS. C0 and the GPBoost challenger performed closely, and the leader depends on the metric. GPBoost had a per-pair MAE 0.47% lower on seen pairs (0.4520 against 0.4541). C0 had a lower row MAE (0.4446 against 0.4478) and a lower error on unseen pairs (0.626 against 0.688). These differences are observed; no uncertainty analysis was run for them. For short pairs, the result depends on the population and on the weighting; see notebook 04 and the decision log.

## Contrasts and scenarios (notebooks 05 and 06, training weeks only)

After Holm, the three planned global tests rejected a common discount slope across store segments, between private label and manufacturer brands, and across categories. Slopes ranged from about −1.1 in BAG SNACKS to −2.4 in FROZEN PIZZA. When a category × week random intercept was added, the common weekly shocks were absorbed and the price and promotion coefficients barely moved. In the scenarios, display at a 20% discount was associated with sales 119% above a regular week, and TPR at 20% with sales 40% above. The modelled ratio between the two was 1.57. All eight scenarios have observed support in training, although promotional ones cover only 0.3% to 3% of the rows.

## Random discount slopes in training (notebook 04, section 9)

A complementary joint test, run after P1 on training weeks only, compared C0 and C1 by maximum likelihood and calibrated the likelihood ratio by parametric bootstrap under C0 (39 replicates). The observed statistic was 12,974 against a largest simulated value of 5.18, so the Monte Carlo p-value was 0.025, the finest resolution of the design: the hypothesis that both slope variances are zero was rejected. C1 estimates a discount-slope standard deviation of 0.88 across products and 0.27 across stores. C1 fits the training data much better, yet it did not lower the out-of-sample error consistently, so C0 remains the selected model.

Coefficients are conditional associations in weeks with recorded transactions, not causal effects. Still deferred, pending approval: scoring C1 on the final window and the long pair bootstrap.

## Open and not executed items

Pending approval: scoring C1 on the final window, as a later exploration, and the long pair bootstrap for the uncertainty of the difference in predictive error. Also open: a general rule for weeks with abnormally low sales (possible stock-outs); week 61 was audited and kept.

Planned but not executed, with no published result depending on them: the negative binomial GLMM robustness check; the sensitivities on pairs with a complete 156-week panel, without imputed base-price rows and with the `OUTLIER_UPV` rows (only the sensitivity without fallback rows was run); the comparison of per-pair OLS slopes with multilevel slopes, since C0 has no per-pair slope (the per-pair OLS slopes themselves are in notebook 01); extensions C2 to C4 of protocol P1, stopped by its rule after C1 was rejected; and store-size interactions and covariance-structure extensions.

## Licence

Code under the MIT licence. The dunnhumby data are subject to dunnhumby's own terms and are not redistributed.
