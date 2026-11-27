# Marketing campaign uplift modeling: Hillstrom

Executive report: [Português](https://eniorubens.github.io/enioRubens_dataScienceProjects/Marketing_Promotion_Campaign_Uplift_Modelling/) · [English](https://eniorubens.github.io/enioRubens_dataScienceProjects/Marketing_Promotion_Campaign_Uplift_Modelling/en.html)

Language: English | [Português (Brasil)](README.pt-BR.md)

This project studies who should receive a marketing treatment because the treatment changes their outcome, separating causal effect from baseline response probability. The randomized Hillstrom MineThatData e-mail experiment provides the basis to compare response-targeting, CATE estimators, uplift trees, causal forests, and budget-constrained contact policies.

The analysis forms a modular, tested, and bilingual workflow with a pre-registered sealed-test evaluation. Its primary conclusion is conservative: the primary confirmatory hypothesis was not supported, and the resulting policies remain restricted to study and pilot design without direct deployment approval.

## Scope and evidence

Can support: causal-targeting research and design of a prospective randomized pilot.

Does not support: automatic policy deployment, validated ROI, or retrospective superiority claims.

Engineering surface: modular `src/` components, bilingual notebook workflow, persisted artifacts, and automated tests; no production serving API is exposed.

Governance layer: the [executive report](https://eniorubens.github.io/enioRubens_dataScienceProjects/Marketing_Promotion_Campaign_Uplift_Modelling/) preserves the sealed-test result and the `not_ready_for_direct_deployment` boundary.

## Executive result

The pre-registered primary comparison was the Qini AUC difference between the `UpliftTree` and the response-targeting baseline on a sealed test set of 12,800 customers.

| Evidence | Result | Interpretation |
|---|---:|---|
| `UpliftTree - response-targeting baseline` | -0.0088, 95% CI [-0.0492, 0.0302] | No confirmatory advantage |
| `X+Tree(depth=4)` absolute Qini AUC | 0.0470, 95% CI [0.0180, 0.0749] | Strongest absolute sealed-test ranking, without retrospective promotion |
| Visit-spend rank alignment | Spearman 0.0941 | Targeting for visits does not reliably transfer to spend |
| Final decision boundary | `not_ready_for_direct_deployment` | Restricted to learning and prospective pilot design |

The post-confirmatory heterogeneity, policy, and ROI sections are exploratory. They preserve the null primary result without reopening model selection.

## Business question

A response model ranks customers by the probability of an observed outcome, which can spend budget on customers who would have responded without an e-mail. Uplift modeling estimates the conditional treatment effect:

$$
\tau(x) = \mathbb{E}[Y(1) - Y(0) \mid X=x]
$$

The operational question becomes which customers to contact, with which campaign, under a contact budget and explicit economic assumptions.

## Dataset and outcomes

The [Hillstrom MineThatData E-Mail Analytics Challenge](http://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html) contains 64,000 customers randomized across three arms.

| Arm | Customers | Description |
|---|---:|---|
| No E-Mail | 21,306 | Control |
| Mens E-Mail | 21,307 | Men's merchandise campaign |
| Womens E-Mail | 21,387 | Women's merchandise campaign |

The dataset has eight pre-treatment customer features, one treatment column, and three outcomes:

| Outcome | Role | Observed prevalence or scale |
|---|---|---|
| `visit` | Primary | 14.7% visited the site |
| `conversion` | Secondary | 0.9% purchased; 578 events |
| `spend` | Secondary | Revenue proxy with a large mass at zero |

The primary outcome is `visit` because `conversion` is too rare for stable subgroup CATE estimation at this sample size. The project tests whether a ranking optimized for top-of-funnel engagement transfers to conversion or revenue.

## Experimental contract

The original three-arm randomized controlled trial supports causal identification under the randomization assumptions verified in Notebook 01. Development uses a deterministic 60%/20%/20% train, validation, and sealed test split, stratified to preserve treatment and outcome rates, pooling both e-mail arms against `No E-Mail` for binary-treatment development.

Notebook 05 evaluates frozen candidates once on the sealed test set with 2,000 bootstrap repetitions. Sections S7-S9 consume persisted outputs without refitting the sealed candidates, repeating the sealed predictions, or selecting a retrospective champion. Interpretation uses quantile profiles and a shallow surrogate for communication, without using SHAP or claiming individual causal explanations.

## Notebook workflow

Eight notebook pairs implement nine analytical sections. The Portuguese edition (PT-BR) serves as the canonical reference, and the English version (EN-US) remains editorially equivalent using the same code and persisted analytical state.

| Notebook | PT-BR | EN-US | Evidence role |
|---|---|---|---|
| 01. Framing and EDA | [PT-BR](notebooks/pt-BR/01_Framing_EDA_PT.ipynb) | [EN-US](notebooks/en-US/01_Framing_EDA_EN.ipynb) | Descriptive and identification checks |
| 02. Response-targeting baseline | [PT-BR](notebooks/pt-BR/02_Baseline_Propensity_PT.ipynb) | [EN-US](notebooks/en-US/02_Baseline_Propensity_EN.ipynb) | Development baseline |
| 03. Meta-learners | [PT-BR](notebooks/pt-BR/03_Meta_Learners_PT.ipynb) | [EN-US](notebooks/en-US/03_Meta_Learners_EN.ipynb) | Candidate development |
| 04. Causal forest and uplift trees | [PT-BR](notebooks/pt-BR/04_Causal_Forest_Uplift_Trees_PT.ipynb) | [EN-US](notebooks/en-US/04_Causal_Forest_Uplift_Trees_EN.ipynb) | Candidate development and pre-registration |
| 05. Sealed-test evaluation | [PT-BR](notebooks/pt-BR/05_Evaluation_Sealed_Test_PT.ipynb) | [EN-US](notebooks/en-US/05_Evaluation_Sealed_Test_EN.ipynb) | Confirmatory |
| 06. Heterogeneity and uplift funnel | [PT-BR](notebooks/pt-BR/06_Heterogeneity_Uplift_Funnel_PT.ipynb) | [EN-US](notebooks/en-US/06_Heterogeneity_Uplift_Funnel_EN.ipynb) | Post-confirmatory exploratory |
| 07. Policy learning and ROI | [PT-BR](notebooks/pt-BR/07_Policy_Learning_ROI_PT.ipynb) | [EN-US](notebooks/en-US/07_Policy_Learning_ROI_EN.ipynb) | Post-confirmatory exploratory |
| 08. Robustness and limitations | [PT-BR](notebooks/pt-BR/08_Robustness_Limitations_PT.ipynb) | [EN-US](notebooks/en-US/08_Robustness_Limitations_EN.ipynb) | Evidence synthesis and decision boundary |

## Methods

The candidate set covers distinct approaches: a response-targeting baseline trained only on treated observations, S-, T-, X-, and R-meta-learners with shared preprocessing, `econml.CausalForestDML` for orthogonalized heterogeneous effects, and `causalml.UpliftTreeClassifier` for direct uplift splits.

Evaluation covers Qini AUC, uplift AUC, uplift at 30%, repeated holdouts, random-ranking envelopes, and paired bootstrap intervals. Heterogeneity analysis uses quantile profiles, GATES-style summaries, uplift funnels, and cross-outcome rank correlations, while the decision stage covers binary contact policies, three-arm treatment assignment, IPW evaluation, budget constraints, and replaceable margin and cost scenarios.

## Post-confirmatory findings

The exploratory stages add context without altering the confirmatory result. `X+Tree(depth=4)` produced the highest absolute sealed-test Qini AUC, though its difference from the response-targeting baseline included zero. Visit and spend rankings showed low alignment (Spearman 0.0941), indicating that engagement uplift fails as a reliable proxy for economic uplift.

Across most budgets, policy confidence intervals included zero, and budget points represent correlated views of the same validation sample rather than independent experiments. ROI values represent point-estimate scenarios based on observed spend as a revenue proxy and illustrative gross margins and e-mail costs, without constituting a validated business case or propagating effect uncertainty.

## Decision boundary

Allowed uses are restricted to learning, hypothesis generation, designing a prospective randomized pilot, and illustrative simulation with replaceable assumptions. Prohibited interpretations include claiming confirmatory superiority for an uplift candidate, automatically deploying the learned policy, retrospectively selecting the best model or budget, and treating multiple budget points as independent tests.

A defensible next step is running a powered randomized pilot with a policy frozen before launch, a pre-specified primary outcome, real cost and margin inputs, and monitoring for fatigue, deliverability, opt-outs, complaints, and capacity.

## Software architecture

| Path | Responsibility |
|---|---|
| `src/data.py`, `src/features.py`, `src/splits.py` | Dataset contract, features, deterministic split, and sealed indices |
| `src/learners.py` | Baseline, meta-learners, causal forest, and uplift-tree training |
| `src/evaluation.py` | Uplift metrics, repeated holdouts, bootstrap, and random envelopes |
| `src/reports.py` | Heterogeneity, quantile profiles, funnel diagnostics, and surrogate summaries |
| `src/policy.py`, `src/policy_reports.py` | Policy construction, IPW evaluation, budgets, ROI, and displays |
| `src/robustness_reports.py` | S9 evidence register, limitations, decision boundary, and source manifest |
| `src/i18n.py`, `src/i18n_catalogs/` | Offline PT-BR and EN-US presentation layer |
| `tests/test_suite.py`, `tests/test_notebooks.py` | Data, leakage, model, metric, artifact, localization, and bilingual notebook-parity tests |
| `artifacts/s6` to `artifacts/s9` | Frozen confirmatory result and reporting-only downstream evidence |

## Data setup

The third-party dataset is not redistributed because the original publication omits an explicit reuse license. Download, filename, and checksum instructions are listed in [`dataset/README.md`](dataset/README.md) and must be followed before running notebooks or tests.

## Installation

The Conda environment expects this project and `ds_toolkit` to be sibling directories, matching the portfolio monorepository layout:

```text
projects-root/
├── Marketing_Promotion_Campaign_Uplift_Modelling/
└── ds_toolkit/
    └── multilang/
```

Create the environment and register the kernel:

```bash
mamba env create -f environment.yml
mamba activate uplift
python -m ipykernel install --user --name uplift --display-name "Python (uplift)"
```

The environment pins OpenBLAS on Windows because the MKL resolution tested for this stack caused native crashes when importing CausalML, EconML, and SHAP.

## Validation

Run the complete test suite from the project root:

```bash
python -m pytest -q -p no:cacheprovider
python -m ruff check src tests
```

The suite contains 157 tests covering data integrity, split reproducibility, fit boundaries, artifact hashes, metrics, localization, reports, PT-BR/EN-US notebook structure, and identical executable code between the bilingual pairs.

For review, Notebook 05 and `artifacts/s6/` are treated as frozen evidence. Re-running the downstream reporting notebooks is supported, whereas opening a new sealed-test experiment would constitute a separate study.

## References

- Radcliffe, N. and Surry, P. (2011), [Real-World Uplift Modelling with Significance-Based Uplift Trees](https://stochasticsolutions.com/pdf/sig-based-up-trees.pdf).
- Künzel, S. R. et al. (2019), [Metalearners for Estimating Heterogeneous Treatment Effects Using Machine Learning](https://doi.org/10.1073/pnas.1804597116).
- Nie, X. and Wager, S. (2021), [Quasi-Oracle Estimation of Heterogeneous Treatment Effects](https://doi.org/10.1093/biomet/asaa076).
- Athey, S., Tibshirani, J. and Wager, S. (2019), [Generalized Random Forests](https://doi.org/10.1214/18-AOS1709).
- Zhao, Y., Fang, X. and Simchi-Levi, D. (2017), [Uplift Modeling with Multiple Treatments and General Response Types](https://doi.org/10.1137/1.9781611974973.66).

## Author and responsibility

Author: Enio Rubens<br>
Role: Data Science and Analytics

AI coding assistants supported modularization, translation, test scaffolding, documentation, and review. Business framing, methodological decisions, result interpretation, and final approval remained human-led, with all published claims under the author's responsibility.

---

## License and acknowledgments

Code: distributed under the [MIT License](LICENSE).

Dataset: [Hillstrom MineThatData E-Mail Analytics Challenge](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html), published by Kevin Hillstrom. Data are obtained directly from the publisher and are not redistributed because the original source omits an explicit reuse license; see [dataset/README.md](dataset/README.md).

Libraries: scikit-learn, CausalML, EconML, scikit-uplift, XGBoost, LightGBM, Pandas, NumPy, SciPy, Statsmodels, Matplotlib, Seaborn, Plotly, Graphviz, Optuna, MLflow, and pytest development communities.

Research: cited work by Radcliffe & Surry; Künzel et al.; Nie & Wager; Athey, Tibshirani & Wager; and Zhao, Fang & Simchi-Levi.

---

Last updated: August 2026<br>
Project status: complete and portfolio-ready<br>
Confirmatory result: Δ Qini AUC -0.0088 | 95% CI [-0.0492, 0.0302] | no superiority established<br>
Decision boundary: not ready for direct deployment

---

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn 1.6+](https://img.shields.io/badge/scikit--learn-1.6%2B-orange.svg)](https://scikit-learn.org/)
[![CausalML 0.15.5](https://img.shields.io/badge/CausalML-0.15.5-008B8B.svg)](https://causalml.readthedocs.io/)
[![EconML 0.16.0](https://img.shields.io/badge/EconML-0.16.0-5C2D91.svg)](https://www.pywhy.org/EconML/)
[![scikit-uplift](https://img.shields.io/badge/scikit--uplift-metrics-brightgreen.svg)](https://www.uplift-modeling.com/)
[![pytest](https://img.shields.io/badge/pytest-154%20passed-brightgreen.svg)](https://pytest.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Bilingual notebooks](https://img.shields.io/badge/notebooks-PT--BR%20%7C%20EN--US-blueviolet.svg)](notebooks/README.md)

---

*This portfolio project implements end-to-end causal machine learning, confirmatory evaluation, heterogeneous treatment-effect analysis, and budget-constrained policy learning.*
