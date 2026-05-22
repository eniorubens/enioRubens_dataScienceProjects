# churn_project

Modular Python package extracted from the Customer Churn Prediction notebook.  
Provides reusable utilities for data loading, feature analysis, threshold evaluation, and corporate-style visualisation.

```bash
pip install -e .          # from the project root
```

```python
import churn_project      # top-level re-exports everything
```

---

## Modules

### `data.py` — Data Loading and Splitting

```python
from churn_project.data import read_telecom_data, split_telecom_dataset, compute_class_ratio
```

| Function | Description |
|----------|-------------|
| `read_data(path)` | Minimal CSV reader (convenience wrapper) |
| `read_telecom_data(filepath)` | Read the Telco churn dataset |
| `split_telecom_dataset(df, ...)` | Stratified train/test split; returns dict with `train_features`, `test_features`, `train_labels`, `test_labels`, `X`, `y` |
| `compute_class_ratio(labels)` | Negative-to-positive class ratio (useful for `scale_pos_weight`) |

**`split_telecom_dataset` key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_col` | `"Churn"` | Target column |
| `id_col` | `"customerID"` | Identifier column to drop |
| `test_size` | `0.2` | Test fraction |
| `random_state` | `42` | Reproducibility seed |
| `drop_cols` | `None` | Additional columns to exclude |

---

### `features.py` — Feature Analysis and Selection

```python
from churn_project.features import (
    prepare_feature_sets,
    encode_categorical_features,
    plot_chi_squared_feature_selection,
    plot_anova_feature_selection,
    build_phik_significance_df,
    filter_relevant_relationships,
    show_skewness,
)
```

| Function | Description |
|----------|-------------|
| `prepare_feature_sets(df, target_col, id_cols)` | Returns `(data, categorical_features, numerical_features)` |
| `encode_categorical_features(df, categorical_features)` | LabelEncoder applied to specified columns |
| `plot_chi_squared_feature_selection(df, ...)` | Chi-Squared scores for categorical features (heatmap) |
| `plot_anova_feature_selection(df, ...)` | ANOVA F-scores for numerical features (heatmap) |
| `build_phik_significance_df(df, drop_cols)` | Pairwise PhiK + significance as a ranked DataFrame |
| `filter_relevant_relationships(df, phik_threshold, significance_threshold)` | Filter significant variable pairs |
| `show_skewness(data, x, detail)` | Print skewness / kurtosis description for a column |

---

### `evaluation.py` — Threshold Tuning and Metrics

```python
from churn_project.evaluation import (
    find_intersection_point,
    plot_metrics,
    save_threshold_metrics,
    highlight_greaterthan,
    highlight_row,
)
```

| Function | Description |
|----------|-------------|
| `find_intersection_point(df_metric)` | Finds the threshold where metric curves minimise variance (optimal balance point) |
| `plot_metrics(df_metric, show_intersection)` | Line plot of all metrics vs threshold with annotated optimal point |
| `save_threshold_metrics(df_metric, thresholds, metric_df, ...)` | Appends threshold-tuned rows to the experiment tracking CSV |
| `highlight_greaterthan(s, threshold, column)` | Pandas Styler helper — highlights max value in a column |
| `highlight_row(row)` | Pandas Styler helper — highlights rows at key threshold values |

**`save_threshold_metrics` key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `baseline_estimator_name` | `"LogisticRegression"` | Estimator to inherit timing from |
| `w_recall` | `7.0` | Business weight for Recall |
| `w_precision` | `3.0` | Business weight for Precision |
| `metric_path` | `"./dataset/metric_dataframe.csv"` | Output CSV path |

---

### `viz.py` — Corporate Visualisation Library

```python
from churn_project.viz import (
    set_corporate_theme,
    plot_importance,
    plot_churn_distribution,
    plot_annual_churn_impact,
    # ... see full list below
)
```

#### Theme

| Function | Description |
|----------|-------------|
| `set_corporate_theme()` | Apply editorial/financial chart style (seaborn ticks + custom rcParams) |
| `add_corporate_header(fig, title, subtitle)` | Clean top-left title block |
| `add_corporate_footer(fig, text, data_source, method)` | Bottom-left italic footer |
| `format_corporate_axes(ax, ...)` | Spine / grid / tick formatter (3 modes) |
| `add_finance_header(fig, title, subtitle)` | Compact finance-style italic header |

#### EDA Charts

| Function | Description |
|----------|-------------|
| `plot_churn_distribution(df, ...)` | Churn class percentages |
| `plot_gender_distribution(df, ...)` | Gender split |
| `plot_senior_distribution(df, ...)` | Senior vs non-senior |
| `plot_household_composition(df, ...)` | Partner / dependents stacked bars |
| `plot_tenure_distribution(df, ...)` | Tenure histogram + KDE |
| `plot_contract_distribution(df, ...)` | Customer count by contract type |
| `plot_tenure_by_contract(df, ...)` | Tenure histograms faceted by contract |
| `plot_services_distribution(df, ...)` | 3×3 grid of service adoption bars |
| `plot_pairplot_corporate(df, ...)` | Seaborn pairplot in corporate style |

#### Correlation Charts

| Function | Description |
|----------|-------------|
| `plot_pearson_correlation(df, ...)` | Clustermap of numerical Pearson correlations |
| `plot_phik_correlation(df, ...)` | PhiK correlation matrix (mixed types) |
| `plot_phik_significance(df, ...)` | Statistical significance of PhiK coefficients |

#### Bivariate / Churn Charts

| Function | Description |
|----------|-------------|
| `plot_churn_vs_tenure(df, ...)` | Boxplot + PhiK: tenure by churn |
| `plot_churn_vs_contract(df, ...)` | Stacked bars + PhiK: churn by contract |
| `plot_churn_vs_monthly_charges(df)` | KDE + skewness/kurtosis/PhiK: charges by churn |
| `plot_target_distribution(df, feature, ...)` | Pie chart of a binary target |
| `plot_target_distribution_split(train_df, test_df, ...)` | Side-by-side train/test pie charts |

#### Business / Model Charts

| Function | Description |
|----------|-------------|
| `plot_annual_churn_impact(annual_revenue, churn_rate, ...)` | Cumulative revenue loss curve |
| `plot_importance(model, title, train_X, train_y, test_X, test_y, scoring)` | Permutation importance box plots (train + test) sorted by test importance |

**`plot_importance` note:** uses `recall_macro` scorer by default, consistent with the primary optimisation objective. Sort order is always by test-set importance, so the chart reflects generalisation, not training fit.

---

## Feature Importance Insight

Permutation importance analysis on the winning **BalancedRandomForestClassifier** reveals:

- **Test set**: `Contract` is the dominant feature, followed by `TotalCharges`, `InternetService`, `tenure`.
- **Train set**: `Contract` and `TotalCharges` show nearly identical importance — both anchor the model's discriminative power during training.

The `Contract → TotalCharges` divergence between train and test is consistent with `Contract` being the stronger **generalizable** churn driver, while `TotalCharges` is heavily leveraged during fitting.

---

## Constant

`viz.RANDOM_SEED = 738` — used for all permutation importance computations; matches the winning estimator's `random_state`.
