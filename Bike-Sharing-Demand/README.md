# Hourly Bike-Sharing Demand Forecasting in Seoul

**Language:** English | [Português (Brasil)](README.pt-BR.md)

This project models hourly demand for Seoul's public bike-sharing system using
approximately nine years of observations (2015-2024) enriched with
meteorological variables. Its objective is to estimate demand under normal
operating conditions through a modular, testable, and auditable architecture,
from exploratory analysis to an uncertainty-aware operational decision
demonstration.

English is the primary language of this portfolio page. Internal identifiers
remain in English, while visible notebook content is localized through the
`multilang` layer. The complete PT-BR and EN-US notebook editions were reviewed
and execution-validated independently while preserving identical analytical
code and numerical results.

## Central Modeling Decision

An extraordinary mobility regime shift was identified in the 2020 data and
could not be explained by the available meteorological variables. Because this
shock is outside the objective of forecasting demand under normal operating
conditions, calendar year 2020 was excluded from model fitting and from the
primary selection metric.

The source dataset was not truncated. The exclusion was represented by an
auditable `normal_operations` regime mask, while meteorological year 2020 was
preserved as a stress diagnostic. The anomaly therefore remains visible without
dominating model selection.

## Temporal Validation Design

Cross-validation uses expanding folds defined by meteorological year, from
December through November, keeping each winter within the same evaluation
window. Primary selection was calculated on meteorological years 2019, 2021,
2022, and 2023, with greater weight assigned to recent folds. The 2020 fold was
used exclusively for stress testing.

The final holdout covers December 2023 through November 2024 and contains 8,784
hourly observations. It was materialized by a single audited function and opened
once in Notebook 05. December 2024 was discarded so that it could influence
neither development nor final confirmation.

## Notebook Workflow

The project consists of eight notebooks, each available in Brazilian Portuguese
and US English:

| Notebook | PT-BR | EN-US | Purpose |
|---|---|---|---|
| 01 - EDA 2015-2024 | [PT-BR](notebooks/pt-BR/01_Seoul_Bike_2015-2024_EDA.ipynb) | [EN-US](notebooks/en-US/01_Seoul_Bike_2015-2024_EDA.ipynb) | Descriptive |
| 02 - Multivariate analysis | [PT-BR](notebooks/pt-BR/02_Seoul_Bike_Multivariate_Analysis.ipynb) | [EN-US](notebooks/en-US/02_Seoul_Bike_Multivariate_Analysis.ipynb) | Descriptive |
| 03 - Feature engineering | [PT-BR](notebooks/pt-BR/03_Feature_Engineering_EDA.ipynb) | [EN-US](notebooks/en-US/03_Feature_Engineering_EDA.ipynb) | Descriptive |
| 04 - Model selection | [PT-BR](notebooks/pt-BR/04_Seoul_Bike_Model_Selection.ipynb) | [EN-US](notebooks/en-US/04_Seoul_Bike_Model_Selection.ipynb) | Selection |
| 05 - Final validation | [PT-BR](notebooks/pt-BR/05_Seoul_Bike_Final_Validation.ipynb) | [EN-US](notebooks/en-US/05_Seoul_Bike_Final_Validation.ipynb) | Confirmatory |
| 06 - Residuals and uncertainty | [PT-BR](notebooks/pt-BR/06_Seoul_Bike_Residual_and_Uncertainty_Experiments.ipynb) | [EN-US](notebooks/en-US/06_Seoul_Bike_Residual_and_Uncertainty_Experiments.ipynb) | Experimental |
| 07 - Conformal calibration | [PT-BR](notebooks/pt-BR/07_Seoul_Bike_Temporal_Conformal_Calibration.ipynb) | [EN-US](notebooks/en-US/07_Seoul_Bike_Temporal_Conformal_Calibration.ipynb) | Experimental |
| 08 - Operational demonstration | [PT-BR](notebooks/pt-BR/08_Seoul_Bike_Operational_Forecast_Demo.ipynb) | [EN-US](notebooks/en-US/08_Seoul_Bike_Operational_Forecast_Demo.ipynb) | Demonstration |

Notebooks 06 and 07 do not constitute a second final validation. The holdout
was not reopened, no point candidate was refitted, and the confirmed Champion
was not replaced. Notebook 08 uses a historical OOF observation to demonstrate
the system without converting it into new confirmatory evidence.

## Dynamic Architecture and Confirmed Champion

In Notebook 04, distinct estimator families are optimized under the same
temporal geometry. Within every trial, preprocessing, encoding, feature
selection, target strategy, and estimator hyperparameters are selected jointly
inside the pipeline. This dynamic composition is the project's central modeling
architecture.

`CatBoostRegressor` was pre-registered as the Champion before the holdout was
opened. In Notebook 05, it was compared with two frozen challengers through a
previously declared decision rule. The Champion would be confirmed if its MAE
were at most 1.05 times the best MAE and its R² were no more than 0.02 below the
best R².

| Holdout metric (n=8,784) | CatBoost Champion | HistGradientBoosting | Random Forest |
|---|---:|---:|---:|
| MAE | 1,118.1 | 1,424.9 | 1,593.4 |
| RMSE | 1,605.7 | 2,069.7 | 2,322.2 |
| R² | 0.839 | 0.733 | 0.664 |
| WAPE | 23.1% | 29.4% | 32.9% |
| Median absolute error | 768.3 | 893.4 | 988.0 |

CatBoost was simultaneously the pre-registered candidate and the best empirical
model on the sealed holdout.

## Residual and Uncertainty Experiments

Notebook 06 compared the reproducible E0 baseline, temporal and meteorological
interactions, `RMSEWithUncertainty`, and a prequential residual-scale layer. No
point successor was identified: E0 retained a recency-weighted MAE of 840.165,
a weighted R² of 0.857, and a mean R² of 0.839. E4 was retained only as an
experimental scale producer, with 81.007% coverage, an average width of 4,501.5
bikes/hour, and a Winkler score of 6,530.4 at the 90% target.

Notebook 07 recalibrated that scale without refitting the point estimator. The
normalized adaptive calibrator `U4b_g0p01` was selected as the experimental
candidate for the uncertainty layer.

| Result at 90% nominal coverage | E4 | U4b_g0p01 | Improvement |
|---|---:|---:|---:|
| Observed coverage | 81.007% | 90.146% | +9.139 p.p. |
| Absolute distance from target | 8.993 p.p. | 0.146 p.p. | -98.4% |
| Average width | 4,501.5 | 3,351.0 | -25.6% |
| Winkler score | 6,530.4 | 4,836.9 | -25.9% |

Across the four selection folds, coverage ranged from 90.024% to 90.242%. A
500-repetition temporal bootstrap produced an interval from 89.709% to 90.591%,
which includes the nominal target. The improvement was restricted to prediction
intervals; E0's MAE, RMSE, and R² were unchanged.

Conditional-coverage alerts remained for the morning rush, evening rush, very
high demand, and Fridays at 6:00 p.m. U4b was therefore classified as an
experimental candidate rather than a production-ready component.

## Operational Demonstration

Notebook 08 simulates a decision before actual demand is revealed. For the
observation selected at 7:00 a.m. on October 22, 2022, the following outputs were
produced:

| Element | Result |
|---|---:|
| E0 point forecast | 1,384 rentals |
| 90% U4b interval | 0 to 6,732 |
| Simulated capacity | 4,000 rentals/hour |
| Prior decision | Attention zone |
| Reserve up to the upper bound | 2,732 rentals/hour |
| Revealed demand | 2,469 rentals |
| Point forecast absolute error | 1,085 rentals |

Actual demand remained within the interval and below the simulated capacity.
The replay therefore demonstrates how a point forecast and its uncertainty can
be converted into an explicit decision without claiming that the example
replaces online validation.

## Project Structure

| Path | Role |
|---|---|
| `src/` | Data access, EDA, feature engineering, CV, pipelines, tracking, validation, uncertainty, and reports |
| `notebooks/pt-BR/` | Canonical Portuguese edition of the eight-stage analytical workflow |
| `notebooks/en-US/` | Equivalent, reviewed, and execution-validated English edition |
| `tests/` | Leakage, temporal split, pipeline, report, and notebook-structure tests |
| `dataset/` | Public dataset, attribution, and local artifacts ignored by Git |
| `mlruns/` | Local MLflow tracking generated during experiments and not versioned |
| `environment.yml` / `requirements.txt` | Tested and pinned dependencies |

## Conventions

Bias is defined as `mean(y_pred - y_true)`; positive values indicate
overestimation. Residuals used in diagnostics are defined as
`y_true - y_pred`; positive values indicate underestimation.

Python variables, columns, and identifiers remain in English. Narrative text,
titles, `print()` output, tables, and charts are displayed in PT-BR or EN-US
through the `multilang` module.

## Installation and Execution

This project is published within the following structure:

```text
enioRubens_dataScienceProjects/
├── Bike-Sharing-Demand/
└── ds_toolkit/
    └── multilang/
```

The relative path is used because `multilang`, already published in the same
monorepository, remains a sibling dependency of this project.

```bash
cd Bike-Sharing-Demand
conda env create -f environment.yml
conda activate Bike-Sharing
python -m ipykernel install --user --name bike-sharing --display-name "Python (Bike-Sharing)"
python -m pytest -q
```

The notebooks must be executed in order to reproduce the complete workflow.
Large runtime-specific artifacts are not versioned: Notebook 06 consumes the
frozen candidates produced by Notebook 04; Notebook 07 consumes the OOF
predictions from Notebook 06; and Notebook 08 consumes the manifest and
conformal predictions produced by Notebook 07.

On Windows, the local MLflow server can be started with:

```bat
start_mlflow.bat
```

## Limitations and Next Steps

Final metrics come from a single twelve-month holdout. Any future Champion
replacement would require a new independent temporal window. The conformal
layer also requires stronger conditional coverage in high-demand regimes before
it can be promoted to production.

Planned next steps include regime-conditional calibration, joint freezing and
versioning of E0, the E4 scale, and the U4b state for online inference, and
incorporation of a new public temporal window.

## Author & Methodology Notes

**Author:** Enio Rubens<br>
**Role:** Data Science & Analytics<br>

### Development Approach

This project demonstrates effective collaboration between human expertise and
AI-assisted tools, including OpenAI Codex and other generative AI coding
assistants:

- **AI-Assisted:** Code review and optimization, modularization, test
  scaffolding, documentation structuring, hypothesis exploration, and
  analytical validation support
- **Human-Driven:** Business problem framing, predictive objective definition,
  methodological decisions, temporal validation design, regime-change
  treatment, result interpretation, and strategic recommendations

All AI-assisted contributions were reviewed, tested, and integrated by the
author. The reported metrics were produced by executing the versioned notebooks
and modules. Final responsibility for the analytical decisions, conclusions,
and published content remains entirely with the author.

---

## Citation Format

```bibtex
@misc{rubens2026seoulbike,
  author = {Rubens, Enio},
  title = {Seoul Bike Sharing Demand: Temporal Forecasting and Conformal Uncertainty},
  year = {2026},
  url = {https://github.com/eniorubens/enioRubens_dataScienceProjects/tree/main/Bike-Sharing-Demand},
  note = {End-to-End Data Science Portfolio Project}
}
```

---

## License & Acknowledgments

**Code:** Distributed under the [MIT License](LICENSE).

**Dataset:** [Seoul Public Bike Usage](https://data.seoul.go.kr/dataList/OA-15182/F/1/datasetView.do)
and meteorological observations from the
[Korea Meteorological Administration](https://data.kma.go.kr), accessed through
the [consolidated Kaggle copy](https://www.kaggle.com/datasets/lnoahl/seoul-bike-sharing-dataset).
The source data remain subject to
[KOGL Type 1](https://www.kogl.or.kr/info/license.do) attribution requirements;
see [dataset/DATASET_README.md](dataset/DATASET_README.md).

**Libraries:** scikit-learn, CatBoost, XGBoost, LightGBM, Optuna, MLflow,
feature-engine, SHAP, Pandas, NumPy, SciPy, Statsmodels, Matplotlib, and Seaborn
development communities.

**Research:** Cited work by Micci-Barreca; Baak et al.; Lundberg & Lee;
Prokhorenkova et al.; Gneiting & Raftery; Gibbs & Candès; and Barber et al.

---

**Last Updated:** August 2026<br>
**Project Status:** Complete & Portfolio-Ready<br>
**Model Performance:** MAE 1,118.1 bikes/hour | R² 0.839 | WAPE 23.1% |
90.146% conformal coverage

---

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn 1.3.2](https://img.shields.io/badge/scikit--learn-1.3.2-orange.svg)](https://scikit-learn.org/)
[![CatBoost 1.2.2](https://img.shields.io/badge/CatBoost-1.2.2-yellow.svg)](https://catboost.ai/)
[![Optuna 3.5.0](https://img.shields.io/badge/Optuna-3.5.0-blueviolet.svg)](https://optuna.org/)
[![MLflow 2.10.0](https://img.shields.io/badge/MLflow-tracking-blue.svg)](https://mlflow.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

---

*This portfolio project showcases end-to-end data science expertise in temporal
machine learning, uncertainty quantification, analytical validation, and
operational decision support.*
