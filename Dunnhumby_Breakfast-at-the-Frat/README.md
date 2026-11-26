# Retail pricing and promotion with multilevel models

[Português](README.pt-BR.md) · [Executive report](site/en.html)

> Model selection and final evaluation under protocol P1 are complete, as are the training-only analyses in notebooks 05 and 06.

## Business question

How do **price** and **promotion** relate to unit sales across stores and products? The study compares feature advertising, in-store displays and temporary price reductions (TPR). It also compares pricing and promotion scenarios and checks how often similar conditions occurred in the data. Margin, promotion cost, ROI and causal price recommendations are outside the scope of the study.

A separate ordinary least squares (OLS) regression for each store–product pair can yield unstable discount elasticities, including positive estimates. With **partial pooling**, a multilevel model draws on the full panel to estimate differences across stores and products. The study tests whether that improves out-of-sample predictions and whether allowing discount slopes to vary by store and product improves them further.

## Approach (summary)

- The linear mixed model uses `log(UNITS)` as its response. The analysis starts with null models and intraclass correlations (ICCs), using likelihood-ratio tests adjusted for parameters at the boundary.
- The fixed effects include discount depth, feature, display, TPR, Fourier seasonality and trend. Base price enters as the pair mean and the deviation from that mean. The model includes crossed random intercepts for store, product and store × product.
- P1 tested random discount slopes by store and product (C1) using out-of-sample error. A follow-up likelihood-ratio test used a parametric bootstrap on the training weeks. Other training analyses tested planned discount-slope interactions with store segment, private label and category using Holm's correction, checked sensitivity to category × week shocks and compared pricing and promotion scenarios.
- The prediction comparison includes OLS by store–product pair, pooled OLS, a multilevel model with BLUPs and GPBoost. P1 set the rules before any candidate was fitted. Model selection used two internal folds within the first 104 weeks. The models in the final comparison were fitted on those 104 weeks.
- The last 52 weeks form a **retrospective evaluation window**. The exploratory OLS in the project brief had already used the whole panel, as noted in revised decision D11. Before scoring the final window once, the project froze the selected specification, GPBoost parameters and the code and panel hashes. Predictions use the price and promotion observed each week.
- Python calls `lme4` through `rpy2`. R is installed from conda-forge in the conda environment. `statsmodels.MixedLM` provides a cross-check for the store-only null model.
- Weeks without a record can reflect zero sales or a stockout. The model uses only weeks with recorded sales.

## How to get the data

Reproduction uses the original dunnhumby spreadsheet and its user guide.

1. Register and download *Breakfast at the Frat: A Time Series Analysis* from https://www.dunnhumby.com/source-files/.
2. Put `dunnhumby - Breakfast at the Frat.xlsx` (and the user guide PDF) into `dataset/`.

*Breakfast at the Frat* is part of the dunnhumby Source Files collection.

## Reproduction

~~~
conda env create -f environment.yml
conda activate retail-multilevel
python scripts/ingest.py        # Excel -> data/interim/*.parquet (~1 min, once)
python scripts/build_panel.py   # cleaning rules -> data/processed/panel.parquet
python scripts/validate_data.py # data-quality contract on the real panel (local only)
pytest                          # synthetic fixtures only; this is what CI runs
~~~

The scripts fit the models once per experiment and save the runs in `outputs/runs/`. Both notebook editions read those saved runs.

On Windows, `retail_multilevel.rbridge` sets `R_HOME` to the R installation in the conda environment, even if R is also installed on the system. Notebook 01 shows the cleaning rules and the row counts for each rule.

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

In both internal folds, C1 reduced prediction error for pairs with fewer than 26 weeks of training history. Across eligible pairs, C1 fell short of the pre-registered selection rule. P1 selected C0, which has random intercepts and a common discount slope. C0 was the multilevel specification evaluated in the final window. Whether random slopes would help pairs with limited history in that window remains open.

In the final window, C0 had lower overall error than pooled OLS and OLS by store–product pair. Among pairs seen in training, GPBoost had a 0.47% lower mean absolute error (MAE) in log units sold when each pair counted equally (0.4520 versus C0's 0.4541). When each row counted equally, C0 had the lower MAE (0.4446 versus 0.4478). C0 also had the lower per-pair MAE among pairs unseen in training (0.626 versus 0.688). The uncertainty around the C0–GPBoost differences remains unestimated. For pairs with limited history, notebook 04 and the decision log show how the result varies with the comparison population and weighting.

## Contrasts and scenarios (notebooks 05 and 06, training weeks only)

After Holm's correction, each of the three planned global tests rejected a common discount slope. The tests covered store segment, private label versus manufacturer brand and product category. The category estimates ranged from about −1.1 for BAG SNACKS to −2.4 for FROZEN PIZZA. Adding a category × week random intercept accounted for shared weekly shocks, with little change in the price and promotion coefficients.

At a 20% discount, the model estimated unit sales 119% above a regular week for display and 40% above for TPR. The estimated ratio of display sales to TPR sales was 1.57. All eight scenarios had comparable observations in training. Each promotional scenario covered 0.3% to 3% of training rows.

## Random discount slopes in training (notebook 04, section 9)

A follow-up test after P1 compared C0 and C1 on the training weeks. It fitted both models by maximum likelihood and calibrated the likelihood-ratio statistic with 39 parametric bootstrap replicates under C0. The observed statistic was 12,974 and the largest simulated statistic was 5.18. The Monte Carlo p-value of 0.025 is the smallest the design can resolve. Under the C0 residual structure, the joint test rejects the hypothesis that both slope variances are zero. The residual autocorrelation documented in notebook 03 limits that inference.

C1 estimated a discount-slope standard deviation of 0.88 across products and 0.27 across stores. The p-value applies to the joint hypothesis about the two variances. C1 fit the training data better. The P1 selection was based on out-of-sample error, where C1's gains were inconsistent. The coefficients describe associations conditional on recorded sales. Estimating causal effects would require a causal design.

## Open and not executed items

Two analyses are pending approval. One would score C1 on the final window as a follow-up exploration. The other would use a long pair bootstrap to estimate uncertainty in the difference in predictive error. The audit kept week 61. A general rule for abnormally low sales that may indicate stockouts is still open.

The planned negative binomial GLMM robustness check and three sensitivity analyses were left open. The sensitivity analyses would restrict the data to pairs with a complete 156-week panel, exclude rows with imputed base prices, or include `OUTLIER_UPV` rows. The sensitivity analysis that ran excluded fallback rows. Notebook 01 reports the OLS slopes by pair. Comparing them with multilevel slopes by pair would require a specification with pair-specific slopes. P1 stopped at C1 under its selection rule, before extensions C2 to C4. Promotion-lift slopes, store-size interactions and covariance-structure extensions were left unfitted.

## License

The code is under the MIT License. The data are subject to dunnhumby's terms and are not redistributed in this repository.
