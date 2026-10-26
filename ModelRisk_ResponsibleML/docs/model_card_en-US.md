# Model Card — UCI Adult Responsible ML

## Status

This document describes an exploratory portfolio model. The model is not production-eligible (`production_eligible=false`) and must not be used for real decisions about people.

## Model summary

The selected model is a `HistGradientBoostingClassifier` with preprocessing encapsulated in a pipeline. `sex` and `race` are excluded from predictors, but proxy features remain available. `fnlwgt` is used as a predictor, not as a sample weight.

The selected fine-tuned configuration is:

| Hyperparameter | Value |
|---|---:|
| `max_iter` | 250 |
| `learning_rate` | 0.05 |
| `max_leaf_nodes` | 15 |
| `min_samples_leaf` | 30 |

The final threshold is `0.31`. It was selected after estimator tuning and sigmoid calibration, using the portfolio exception documented in the selection contract.

## Intended use and prohibited uses

The intended use is to demonstrate a reproducible protocol for model risk, calibration, threshold selection, and fairness auditing on a historical dataset. The model must not be used for credit, employment, benefits, screening, or any automated decision about real people.

## Data

The dataset is UCI Adult, derived from 1994 census data, with 48,842 rows. The official training set contains 32,561 rows and the official test set contains 16,281 rows. The target is income `>50K` versus `<=50K`.

The dataset contains missing categorical values and duplicates recorded in the data contract. We adopt the exact predictor signature, excluding the target, as the operational identity while retaining source rows and avoiding silent deduplication. The final diagnostic found 26 repeated predictor signatures between train and test, involving 28 training rows and 26 test rows; three signatures have discordant targets. This overlap limits interpretation of the test as fully independent evidence.

Source: [UCI Adult Dataset](https://archive.ics.uci.edu/dataset/2/adult), DOI `10.24432/C5XW20`.

## Training and selection

Fine-tuning evaluated six HistGradientBoosting configurations using group-aware OOF predictions inside the `fit` block. The selection order was PR-AUC, ROC-AUC, and Brier. The gain over the fixed baseline was small: PR-AUC increased from 0.818356 to 0.818842 in the tuning artifact.

After tuning, the model was fitted in `fit`, probabilities were calibrated in `calibration`, and the threshold was selected in `selection`. The official test did not participate in those decisions.

## Official test performance

The official test was used once, with the model, calibrator, and threshold frozen.

| Metric | Result |
|---|---:|
| ROC-AUC | 0.9233 |
| PR-AUC | 0.8170 |
| Brier | 0.0903 |
| Precision | 0.6330 |
| Recall / TPR | 0.8138 |
| F1 | 0.7121 |
| Selection rate | 0.3037 |

These values describe the evaluated snapshot and do not guarantee future performance.

## Group audit

The reported intervals are 95% Wilson intervals. On the test set, the main marginal differences by sex were:

| Group | Selection rate | TPR | FPR | PPV |
|---|---:|---:|---:|---:|
| Female | 0.134 | 0.753 | 0.059 | 0.611 |
| Male | 0.388 | 0.825 | 0.201 | 0.637 |

Smaller racial categories have reduced TPR and PPV denominators, with `low_support=true` and wider intervals. Observed differences are descriptive; they do not demonstrate causality, discriminatory intent, or absence of bias.

## Limitations

The dataset represents a specific historical population and context. Directly excluding `sex` and `race` does not remove proxies because features such as `relationship`, `marital_status`, `education`, `occupation`, and `native_country` remain available. Fine-tuning used a small, non-exhaustive search. Global and group metrics come from one final partition and do not capture all repeated-split variance.

The overlap of predictor signatures between train and test, including discordant targets, weakens the strength of any independence claim about the test. A sensitivity analysis excluding the 26 overlapping test rows retained 16,255 rows and produced ROC-AUC 0.9232, PR-AUC 0.8170, Brier 0.0903, precision 0.6326, recall 0.8135 and F1 0.7118, very close to the primary benchmark. The result therefore remains exploratory and is not production-eligible.

## Reproducibility and artifacts

Contracts and outputs are stored in:

- `data/processed/tuning/`: fine-tuning metrics and contract;
- `data/processed/selection/`: selection metrics, audit, and contract;
- `data/processed/final_evaluation/`: final test metrics, audit, and contract;
- `src/hyperparameter_tuning.py`: estimator tuning;
- `src/threshold_selection.py`: calibration and threshold selection;
- `src/final_evaluation.py`: single final evaluation;
- `en-US/05a_hyperparameter_tuning.ipynb`, `en-US/05_model_threshold_selection.ipynb`, and `en-US/06_final_test_evaluation.ipynb`: executed narrative.

## Governance decision

The model may be presented as the exploratory winner of this portfolio under the approved gate exception. It must not be presented as production-ready, externally validated, causally fair, or suitable for deployment.
