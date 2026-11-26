# E-commerce delivery network and SLA risk (Olist)

This project investigates delivery delay and the risk of breaching the promised delivery date (SLA) for orders from the Brazilian Olist marketplace. Reusable code lives in the ecom_delivery_sla package, route-network and presentation charts live in ecom_delivery_sla_viz, and the analysis is conducted through bilingual Portuguese and English notebooks.

The project is exploratory: production_eligible = false. No model is considered eligible for operational use.

## Objective and research question

The central question is whether, at purchase time, the risk that an order will arrive after the promised date can be predicted with material improvement over simple baselines, using only information available at that moment: order characteristics, distance, calendar variables, and the historical seller-to-destination network. The binary target uses calendar-day delivery status. Continuous delay in fractional days is treated descriptively and inferentially. The OLS association is not interpreted as causal.

## Data

The source is the public *Brazilian E-Commerce Public Dataset by Olist*, published on Kaggle (https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). Raw data are not versioned, and the download is manual because Kaggle requires credentials. Save brazilian-ecommerce.zip under data/raw/olist/. Provenance and the expected SHA-256 are recorded in data/raw/olist/README.md.

The ZIP integrity is checked and the nine CSV files are extracted to data/raw/olist/csv/:

~~~
python scripts/verify_data.py
python scripts/verify_data.py --extract
~~~

## Reproduction

Create the conda environment from environment.yml:

~~~
conda env create -f environment.yml
conda activate ecom-delivery-sla
python scripts/verify_data.py --extract
~~~

Run notebooks 01 → 11 in order. The Portuguese notebooks compute and store expensive results, while the English notebooks load the corresponding caches. The final evaluation (F1) opens the test period only once, and notebook 08 reuses the Optuna studies stored in SQLite.

CI installs the development dependencies, runs lint, contract tests, PT/EN parity checks, and exports the final report. It does not execute notebooks or build the site because raw CSV files are not versioned.

~~~
pytest
ruff check src tests scripts
~~~

## Notebook map

| No. | Portuguese notebook | English notebook |
|---:|---|---|
| 01 | 01_apresentacao_e_pre_analise_olist.ipynb | 01_apresentacao_e_pre_analise_olist.ipynb |
| 02 | 02_eda_atraso_sla_olist.ipynb | 02_eda_atraso_sla_olist.ipynb |
| 03 | 03_eda_geografia_e_composicao_pedido.ipynb | 03_eda_geography_and_order_composition.ipynb |
| 04 | 04_tabela_analitica_candidata.ipynb | 04_candidate_analytical_table.ipynb |
| 05 | 05_contrato_temporal_e_split.ipynb | 05_temporal_contract_and_split.ipynb |
| 06 | 06_modeling_baseline_ols_sklearn.ipynb | 06_modeling_baseline_ols_sklearn.ipynb |
| 07 | 07_tournament_estimators.ipynb | 07_estimator_tournament.ipynb |
| 08 | 08_otimizacao_optuna.ipynb | 08_optuna_optimization.ipynb |
| 09 | 09_avaliacao_final.ipynb | 09_final_evaluation.ipynb |
| 10 | 10_pos_hoc_f2.ipynb | 10_post_hoc_f2.ipynb |
| 11 | 11_relatorio_final.ipynb | 11_final_report.ipynb |

Notebook 11 is the final report. It reads stored results only, does not retrain models or reopen the test period, and is exported to HTML under reports/.

## Executive report

The bilingual static report with interactive charts is available at site/index.html (Portuguese) and site/en.html (English). Rebuild it with:

~~~
python scripts/build_site.py
~~~

Executed notebooks record machine-specific paths in their outputs (warnings and provenance tables). Before publishing, they are replaced with neutral placeholders such as PROJECT_ROOT and CONDA_ENV:

~~~
python scripts/anonymize_local_paths.py
~~~

## Results

The development tournament uses nine temporal folds with a 30-day embargo. CatBoost was the only classification candidate under the preregistered gate, with mean AP of 0.175 versus 0.154 for the best baseline. No regression model was a candidate.

The official final evaluation was inconclusive: CatBoost achieved AP 0.082 versus 0.084 for the logistic baseline, with a paired difference of −0.002 and a 95% CI from −0.032 to +0.064. The complementary F2 analysis is exploratory, and no variant outperformed the references.

## Limitations

The Olist dataset covers one marketplace and a short period ending in 2018. Right censoring was handled by excluding orders without a mature outcome. The test period is exhausted for further evaluations, and delay regression remains descriptive and inferential, with no candidate model. Therefore, production_eligible = false remains explicit.

## Documentation

The docs/ directory contains the approved decisions, data contracts, route-graph plan, modular architecture, work sequence, and technical review.
