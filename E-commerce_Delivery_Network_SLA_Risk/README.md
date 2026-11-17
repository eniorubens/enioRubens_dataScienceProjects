# E-commerce delivery network and SLA risk (Olist)

This project investigates delivery delay and the risk of breaching the promised delivery date (SLA) for orders from the Brazilian Olist marketplace. Reusable code lives in ecom_delivery_sla, while route-network and presentation charts live in ecom_delivery_sla_viz.

The project is **exploratory**: production_eligible = false. No model is considered eligible for operational use.

## Objective

The central question is whether, at purchase time, the risk that an order will arrive after the promised date can be predicted with material improvement over simple baselines, using order characteristics, distance, calendar variables, and the historical seller-to-destination network. The binary target uses calendar-day status. Continuous delay is treated descriptively and inferentially, and the OLS association is not interpreted as causal.

## Data and reproduction

The source is the public *Brazilian E-Commerce Public Dataset by Olist*, published on Kaggle (https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). Raw data are not versioned. Save brazilian-ecommerce.zip under data/raw/olist/:

~~~
conda env create -f environment.yml
conda activate ecom-delivery-sla
python scripts/verify_data.py --extract
~~~

Run notebooks 01 → 11 in order. Portuguese notebooks compute expensive results and English notebooks load the corresponding caches. The final evaluation opens the test period once, and notebook 08 reuses Optuna studies stored in SQLite.

~~~
pytest
ruff check src tests scripts
~~~

## Report and results

The executive report is available at site/index.html (Portuguese) and site/en.html (English). The temporal tournament found CatBoost as the only classification candidate under the preregistered gate, but the official final evaluation was **inconclusive**. The F2 analysis is exploratory, and no model is considered eligible for operational use.

## Limitations and documentation

The Olist dataset covers one marketplace and ends in 2018. The test period is exhausted for further evaluations, and production_eligible = false remains explicit. Full documentation is in docs/.
