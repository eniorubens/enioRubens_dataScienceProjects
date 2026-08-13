# Marketing Campaign Uplift Modeling: Hillstrom

**Language:** English | [Português (Brasil)](README.pt-BR.md)

This project studies **who should receive a marketing treatment because the
treatment changes their outcome**, rather than who is simply most likely to
respond. It uses the randomized Hillstrom MineThatData e-mail experiment to
compare response-targeting, CATE estimators, uplift trees, causal forests, and
budget-constrained contact policies.

The analysis is organized as a modular, tested, and bilingual workflow with a
pre-registered sealed-test evaluation. Its main conclusion is deliberately
conservative: the primary confirmatory hypothesis was not supported, and the
resulting policies are **not ready for direct deployment**.

## Executive Result

The pre-registered primary comparison was the Qini AUC difference between the
`UpliftTree` and the response-targeting baseline on a sealed test set of 12,800
customers.

| Evidence | Result | Interpretation |
|---|---:|---|
| `UpliftTree - response-targeting baseline` | -0.0088, 95% CI [-0.0492, 0.0302] | No confirmatory advantage |
| `X+Tree(depth=4)` absolute Qini AUC | 0.0470, 95% CI [0.0180, 0.0749] | Strongest absolute sealed-test ranking, but not a retrospectively promotable winner |
| Visit-spend rank alignment | Spearman 0.0941 | Targeting for visits does not reliably transfer to spend |
| Final decision boundary | `not_ready_for_direct_deployment` | Use for learning and prospective pilot design only |

The post-confirmatory heterogeneity, policy, and ROI sections are exploratory.
They preserve the null primary result and do not reopen model selection.

## Business Question

A response model ranks customers by the probability of an observed outcome.
That ranking can spend budget on customers who would have responded without an
e-mail. Uplift modeling instead estimates the conditional treatment effect:

$$
\tau(x) = \mathbb{E}[Y(1) - Y(0) \mid X=x]
$$

The operational question becomes: **which customers should be contacted, with
which campaign, under a contact budget and explicit economic assumptions?**

## Dataset and Outcomes

The [Hillstrom MineThatData E-Mail Analytics Challenge](http://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html)
contains 64,000 customers randomized across three arms.

| Arm | Customers | Description |
|---|---:|---|
| No E-Mail | 21,306 | Control |
| Mens E-Mail | 21,307 | Men's merchandise campaign |
| Womens E-Mail | 21,387 | Women's merchandise campaign |

The dataset has eight pre-treatment customer features, one treatment column,
and three outcomes:

| Outcome | Role | Observed prevalence / scale |
|---|---|---|
| `visit` | Primary | 14.7% visited the site |
| `conversion` | Secondary | 0.9% purchased; only 578 events |
| `spend` | Secondary | Revenue proxy with a large mass at zero |

`visit` is primary because `conversion` is too rare for stable subgroup CATE
estimation at this sample size. The project explicitly tests whether a ranking
optimized for top-of-funnel engagement transfers to conversion or revenue.

## Experimental Contract

- The original three-arm RCT supports causal identification under the
  randomization assumptions checked in Notebook 01.
- Development uses a deterministic 60%/20%/20% train, validation, and sealed
  test split, stratified to preserve treatment and outcome rates.
- Binary-treatment development pools both e-mail arms against `No E-Mail`.
- Notebook 05 evaluates frozen candidates once on the sealed test set with
  2,000 bootstrap repetitions.
- S7-S9 consume persisted outputs. They do not refit the sealed candidates,
  repeat the sealed predictions, or select a retrospective champion.
- Interpretation uses quantile profiles and a shallow surrogate for
  communication; it does not use SHAP and does not claim individual causal
  explanations.

## Notebook Workflow

Eight notebook pairs implement nine analytical sections. PT-BR is the
canonical edition; EN-US is editorially equivalent and uses the same code and
persisted analytical state.

| Notebook | PT-BR | EN-US | Evidence role |
|---|---|---|---|
| 01 - Framing and EDA | [PT-BR](notebooks/pt-BR/01_Framing_EDA_PT.ipynb) | [EN-US](notebooks/en-US/01_Framing_EDA_EN.ipynb) | Descriptive and identification checks |
| 02 - Response-targeting baseline | [PT-BR](notebooks/pt-BR/02_Baseline_Propensity_PT.ipynb) | [EN-US](notebooks/en-US/02_Baseline_Propensity_EN.ipynb) | Development baseline |
| 03 - Meta-learners | [PT-BR](notebooks/pt-BR/03_Meta_Learners_PT.ipynb) | [EN-US](notebooks/en-US/03_Meta_Learners_EN.ipynb) | Candidate development |
| 04 - Causal forest and uplift trees | [PT-BR](notebooks/pt-BR/04_Causal_Forest_Uplift_Trees_PT.ipynb) | [EN-US](notebooks/en-US/04_Causal_Forest_Uplift_Trees_EN.ipynb) | Candidate development and pre-registration |
| 05 - Sealed-test evaluation | [PT-BR](notebooks/pt-BR/05_Evaluation_Sealed_Test_PT.ipynb) | [EN-US](notebooks/en-US/05_Evaluation_Sealed_Test_EN.ipynb) | Confirmatory |
| 06 - Heterogeneity and uplift funnel | [PT-BR](notebooks/pt-BR/06_Heterogeneity_Uplift_Funnel_PT.ipynb) | [EN-US](notebooks/en-US/06_Heterogeneity_Uplift_Funnel_EN.ipynb) | Post-confirmatory exploratory |
| 07 - Policy learning and ROI | [PT-BR](notebooks/pt-BR/07_Policy_Learning_ROI_PT.ipynb) | [EN-US](notebooks/en-US/07_Policy_Learning_ROI_EN.ipynb) | Post-confirmatory exploratory |
| 08 - Robustness and limitations | [PT-BR](notebooks/pt-BR/08_Robustness_Limitations_PT.ipynb) | [EN-US](notebooks/en-US/08_Robustness_Limitations_EN.ipynb) | Evidence synthesis and decision boundary |

## Methods

The candidate set covers deliberately different approaches:

- Response-targeting baseline trained only on treated observations.
- S-, T-, X-, and R-meta-learners with shared preprocessing.
- `econml.CausalForestDML` for orthogonalized heterogeneous effects.
- `causalml.UpliftTreeClassifier` for direct uplift splits.
- Qini AUC, uplift AUC, uplift at 30%, repeated holdouts, random-ranking
  envelopes, and paired bootstrap intervals.
- Quantile profiles, GATES-style summaries, uplift funnels, and cross-outcome
  rank correlations.
- Binary contact policies, three-arm treatment assignment, IPW evaluation,
  budget constraints, and replaceable margin/cost scenarios.

## Post-Confirmatory Findings

The exploratory stages add context without changing the confirmatory result:

- `X+Tree(depth=4)` produced the strongest absolute sealed-test Qini AUC, but
  its difference from the response-targeting baseline still included zero.
- Visit and spend rankings had low alignment (Spearman 0.0941), so engagement
  uplift is not a reliable proxy for economic uplift.
- Across budget levels, most policy confidence intervals included zero. Budget
  points are correlated views of the same validation sample, not independent
  experiments.
- ROI values are point-estimate scenarios based on observed spend as a revenue
  proxy and illustrative gross margins and e-mail costs. They are not a
  validated business case and do not propagate effect uncertainty.

## Decision Boundary

Allowed uses:

- Learning and hypothesis generation.
- Designing a prospective randomized pilot.
- Illustrative simulation with visibly replaceable assumptions.

Prohibited interpretations:

- Claiming confirmatory superiority for an uplift candidate.
- Automatically deploying the learned policy.
- Retrospectively selecting the best model or budget.
- Treating multiple budget points as independent tests.

A defensible next step is a powered randomized pilot with a policy frozen
before launch, a pre-specified primary outcome, real cost and margin inputs,
and monitoring for fatigue, deliverability, opt-outs, complaints, and capacity.

## Software Architecture

| Path | Responsibility |
|---|---|
| `src/data.py`, `src/features.py`, `src/splits.py` | Dataset contract, features, deterministic split, and sealed indices |
| `src/learners.py` | Baseline, meta-learners, causal forest, and uplift-tree training |
| `src/evaluation.py` | Uplift metrics, repeated holdouts, bootstrap, and random envelopes |
| `src/reports.py` | Heterogeneity, quantile profiles, funnel diagnostics, and surrogate summaries |
| `src/policy.py`, `src/policy_reports.py` | Policy construction, IPW evaluation, budgets, ROI, and displays |
| `src/robustness_reports.py` | S9 evidence register, limitations, decision boundary, and source manifest |
| `src/i18n.py`, `src/i18n_catalogs/` | Offline PT-BR/EN-US presentation layer |
| `tests/test_suite.py` | Data, leakage, model, metric, artifact, localization, and notebook-contract tests |
| `artifacts/s6` to `artifacts/s9` | Frozen confirmatory result and reporting-only downstream evidence |

## Data Setup

The third-party dataset is not redistributed because the original publication
does not state an explicit reuse license. Follow the download, filename, and
checksum instructions in [`dataset/README.md`](dataset/README.md) before
running notebooks or tests.

## Installation

The Conda environment expects this project and `ds_toolkit` to be sibling
directories, matching the portfolio monorepository layout:

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

The environment pins OpenBLAS on Windows because the MKL resolution tested for
this stack caused native crashes when importing CausalML, EconML, and SHAP.

## Validation

Run the complete test module from the project root:

```bash
python -m pytest tests/test_suite.py -q -p no:cacheprovider
python -m ruff check src tests
```

The suite contains 154 tests covering data integrity, split reproducibility,
fit boundaries, artifact hashes, metrics, localization, reports, and PT-BR /
EN-US notebook structure.

For review, treat Notebook 05 and `artifacts/s6/` as frozen evidence. Re-running
the downstream reporting notebooks is supported, but opening a new sealed-test
experiment would be a different study.

## Scientific References

- Radcliffe, N. and Surry, P. (2011), [Real-World Uplift Modelling with Significance-Based Uplift Trees](https://stochasticsolutions.com/pdf/sig-based-up-trees.pdf).
- Künzel, S. R. et al. (2019), [Metalearners for Estimating Heterogeneous Treatment Effects Using Machine Learning](https://doi.org/10.1073/pnas.1804597116).
- Nie, X. and Wager, S. (2021), [Quasi-Oracle Estimation of Heterogeneous Treatment Effects](https://doi.org/10.1093/biomet/asaa076).
- Athey, S., Tibshirani, J. and Wager, S. (2019), [Generalized Random Forests](https://doi.org/10.1214/18-AOS1709).
- Zhao, Y., Fang, X. and Simchi-Levi, D. (2017), [Uplift Modeling with Multiple Treatments and General Response Types](https://doi.org/10.1137/1.9781611974973.66).

## Author and Responsibility

**Author:** Enio Rubens<br>
**Role:** Data Science and Analytics

AI coding assistants supported modularization, translation, test scaffolding,
documentation, and review. Business framing, methodological decisions, result
interpretation, and final approval remained human-led. All published claims are
the author's responsibility.

---

## License & Acknowledgments

**Code:** Distributed under the [MIT License](LICENSE).

**Dataset:** [Hillstrom MineThatData E-Mail Analytics Challenge](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html),
published by Kevin Hillstrom. The dataset is obtained directly from the
publisher and is not redistributed because the source does not state an
explicit reuse license; see [dataset/README.md](dataset/README.md).

**Libraries:** scikit-learn, CausalML, EconML, scikit-uplift, XGBoost,
LightGBM, Pandas, NumPy, SciPy, Statsmodels, Matplotlib, Seaborn, Plotly,
Graphviz, Optuna, MLflow, and pytest development communities.

**Research:** Cited work by Radcliffe & Surry; Künzel et al.; Nie & Wager;
Athey, Tibshirani & Wager; and Zhao, Fang & Simchi-Levi.

---

**Last Updated:** August 2026<br>
**Project Status:** Complete & Portfolio-Ready<br>
**Confirmatory Result:** Δ Qini AUC -0.0088 | 95% CI [-0.0492, 0.0302] |
no superiority established<br>
**Decision Boundary:** Not ready for direct deployment

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

*This portfolio project showcases end-to-end causal machine learning, honest
confirmatory evaluation, heterogeneous treatment-effect analysis, and
budget-constrained policy learning.*
