# opt_binary_clf_pipe

Reusable ML modeling library for binary classification projects.

Encapsulates the full experiment loop: baseline training, Optuna-based
pipeline optimisation, threshold tuning, and metric persistence.

---

## Installation (local, editable)

```bash
pip install -e "path"
```

Add to each project's `requirements.txt`:

```
opt_binary_clf_pipe @ file:///c:path
```

---

## Quick start

```python
from opt_binary_clf_pipe import train_all_models, CsvModelStore

store = CsvModelStore(
    metric_path="./dataset/metric_dataframe.csv",
    model_dir="./models/",
    w_recall=7,
    w_precision=3,
    w_time=0,
)

metric_df = train_all_models(
    train_features=X_train,
    train_labels=y_train,
    test_features=X_test,
    test_labels=y_test,
    cv=cv,
    x_reference=X_reference,
    metric_df=metric_df,
    random_seed=42,
    model_store=store,
    trials=50,
)
```

---

## Architecture

```
opt_binary_clf_pipe/
├── __init__.py          Public API
├── estimators.py        Classifier registry (define_estimators, get_estimator)
├── callbacks.py         EarlyStoppingCallback for Optuna
├── scoring.py           Metrics, business score, threshold helpers
├── persistence.py       ModelStore interface + CsvModelStore backend
├── pipeline_builder.py  Pipeline construction helpers
├── optimizer.py         Optimizer class + apply_threshold_decision
└── trainer.py           run_baseline_estimator, train_all_models
```

### Storage backends

Today: **CSV + gzip-pickle**

```python
from opt_binary_clf_pipe import CsvModelStore
store = CsvModelStore(metric_path="./dataset/metrics.csv", model_dir="./models/")
```

Future: **MLflow** (same interface, one line change in the notebook)

```python
from opt_binary_clf_pipe import MlflowModelStore          # coming in v2.0
store = MlflowModelStore(experiment_name="churn-v2")
```

### Custom hyperparameter space

Override built-in defaults for a specific project:

```python
def my_rf_space(trial, balanced, random_seed):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "random_state": random_seed,
    }
    if balanced:
        params["class_weight"] = "balanced"
    return params

metric_df = train_all_models(
    ...,
    estimators_to_run=["RandomForestClassifier"],
    param_space=my_rf_space,
)
```

---

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/ -v --cov=opt_binary_clf_pipe
```

---

## Upgrading to MLflow (planned v2.0)

1. Implement `MlflowModelStore(ModelStore)` in `persistence.py`
2. Replace `CsvModelStore(...)` with `MlflowModelStore(...)` in the notebook
3. No other changes needed
