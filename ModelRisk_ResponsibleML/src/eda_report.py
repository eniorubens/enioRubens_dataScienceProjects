"""Tabelas reutilizáveis para a EDA do UCI Adult.

As funções deste módulo calculam estatísticas descritivas sem treinar modelos
e sem usar o conjunto de teste para decisões de limiar.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from adult_data import COLUMNS, TARGET


NUMERIC_COLUMNS = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

CATEGORICAL_COLUMNS = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

PREDICTOR_CATEGORICAL_COLUMNS = [
    column for column in CATEGORICAL_COLUMNS if column not in {"race", "sex"}
]

RARE_CATEGORY_THRESHOLD = 100
PROXY_COMPOSITION_GAP_THRESHOLD = 0.20


def _feature_overview(frame: pd.DataFrame) -> pd.DataFrame:
    """Resume tipo, cardinalidade e ausência de cada feature."""
    rows = []
    for column in COLUMNS[:-1]:
        series = frame[column]
        rows.append(
            {
                "feature": column,
                "dtype": str(series.dtype),
                "n_unique": int(series.nunique(dropna=True)),
                "missing_count": int(series.isna().sum()),
                "missing_rate": float(series.isna().mean()),
                "role": "audit" if column in {"sex", "race"} else "predictor_candidate",
            }
        )
    return pd.DataFrame(rows)


def _categorical_target_rates(frame: pd.DataFrame) -> pd.DataFrame:
    """Calcula volume e taxa de renda alta por categoria."""
    work = frame.copy()
    work["income_rate"] = (work[TARGET] == ">50K").astype(int)
    tables = []
    for column in CATEGORICAL_COLUMNS:
        grouped = (
            work.groupby(column, dropna=False)["income_rate"]
            .agg(count="size", positive_rate="mean")
            .reset_index()
            .rename(columns={column: "category"})
        )
        grouped.insert(0, "feature", column)
        tables.append(grouped)
    return pd.concat(tables, ignore_index=True)


def _numeric_by_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Compara estatísticas numéricas por classe do alvo."""
    rows = []
    for column in NUMERIC_COLUMNS:
        for target_value, group in frame.groupby(TARGET, dropna=False):
            values = group[column].dropna()
            rows.append(
                {
                    "feature": column,
                    "income": str(target_value),
                    "count": int(values.size),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "q25": float(values.quantile(0.25)),
                    "q75": float(values.quantile(0.75)),
                }
            )
    return pd.DataFrame(rows)


def _split_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    """Compara alvo e ausência entre os splits oficiais."""
    rows = []
    for split, group in frame.groupby("split", dropna=False):
        rows.append(
            {
                "split": str(split),
                "rows": int(len(group)),
                "positive_rate": float((group[TARGET] == ">50K").mean()),
                "missing_cells": int(group[COLUMNS[:-1]].isna().sum().sum()),
                "missing_rate_cells": float(group[COLUMNS[:-1]].isna().mean().mean()),
                "duplicate_rows": int(group.duplicated().sum()),
            }
        )
    return pd.DataFrame(rows)


def _conditional_target_rates(frame: pd.DataFrame) -> pd.DataFrame:
    """Compara taxas do alvo dentro de estratos observáveis.

    Esta é uma análise descritiva de padronização estratificada, não uma
    estimativa causal. Os estratos são analisados separadamente para evitar
    que células muito pequenas da combinação entre todas as covariáveis
    produzam taxas instáveis.
    """
    work = frame.copy()
    work["positive"] = (work[TARGET] == ">50K").astype(int)
    work["education_band"] = pd.cut(
        work["education_num"],
        bins=[0, 7, 9, 12, 16],
        labels=["basic", "high_school", "college", "advanced"],
        include_lowest=True,
    )
    work["hours_band"] = pd.cut(
        work["hours_per_week"],
        bins=[-1, 20, 35, 45, 60, float("inf")],
        labels=["0-20", "21-35", "36-45", "46-60", "61+"],
    )
    rows = []
    for control in ["education_band", "occupation", "hours_band"]:
        for audit_column in ["sex", "race"]:
            grouped = (
                work.groupby([control, audit_column], dropna=False, observed=False)["positive"]
                .agg(count="size", positive_rate="mean")
                .reset_index()
            )
            grouped["control"] = control
            grouped["audit_column"] = audit_column
            grouped = grouped.rename(columns={control: "stratum", audit_column: "audit_group"})
            grouped["stratum"] = grouped["stratum"].astype("string").fillna("(missing)")
            grouped["audit_group"] = grouped["audit_group"].astype("string").fillna("(missing)")
            gaps = grouped.groupby("stratum", dropna=False)["positive_rate"].transform(
                lambda values: values.max() - values.min()
            )
            grouped["within_stratum_rate_gap"] = gaps
            rows.append(
                grouped[
                    [
                        "control",
                        "stratum",
                        "audit_column",
                        "audit_group",
                        "count",
                        "positive_rate",
                        "within_stratum_rate_gap",
                    ]
                ]
            )
    return pd.concat(rows, ignore_index=True)


def _category_risk_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    """Quantifica categorias raras e composição por atributos de auditoria.

    A sinalização de proxy é uma triagem exploratória baseada em diferença de
    composição de 20 pontos percentuais; ela não prova causalidade nem deve
    ser interpretada como evidência definitiva de proxy.
    """
    rows = []
    total_rows = len(frame)
    for feature in PREDICTOR_CATEGORICAL_COLUMNS:
        values = frame[feature].astype("string").fillna("(missing)")
        for category, category_frame in frame.assign(_category=values).groupby(
            "_category", dropna=False, observed=False
        ):
            category_mask = values == category
            category_count = int(category_mask.sum())
            row = {
                "feature": feature,
                "category": str(category),
                "count": category_count,
                "sample_share": category_count / total_rows,
                "is_rare": category_count < RARE_CATEGORY_THRESHOLD,
            }
            for audit_column in ["sex", "race"]:
                overall = frame[audit_column].value_counts(normalize=True, dropna=False)
                within = frame.loc[category_mask, audit_column].value_counts(
                    normalize=True, dropna=False
                )
                composition = within.reindex(overall.index, fill_value=0.0)
                gap = float((composition - overall).abs().max()) if len(composition) else 0.0
                dominant = str(composition.idxmax()) if len(composition) else "(none)"
                row[f"{audit_column}_dominant_group"] = dominant
                row[f"{audit_column}_max_composition_gap"] = gap
                row[f"{audit_column}_proxy_flag"] = gap >= PROXY_COMPOSITION_GAP_THRESHOLD
            rows.append(row)
    return pd.DataFrame(rows)


def _split_distribution_drift(frame: pd.DataFrame) -> pd.DataFrame:
    """Resume diferenças de composição entre treino e teste oficiais."""
    train = frame.loc[frame["split"] == "train_official"]
    test = frame.loc[frame["split"] == "test_official"]
    rows = []
    for feature in COLUMNS[:-1]:
        if feature in CATEGORICAL_COLUMNS:
            train_rates = train[feature].astype("string").fillna("(missing)").value_counts(
                normalize=True
            )
            test_rates = test[feature].astype("string").fillna("(missing)").value_counts(
                normalize=True
            )
            categories = train_rates.index.union(test_rates.index)
            differences = (train_rates.reindex(categories, fill_value=0.0) - test_rates.reindex(
                categories, fill_value=0.0
            )).abs()
            rows.append(
                {
                    "feature": feature,
                    "comparison": "categorical",
                    "metric": "total_variation_distance",
                    "value": float(differences.sum() / 2),
                    "max_category_rate_gap": float(differences.max()),
                }
            )
        else:
            train_values = train[feature].astype(float)
            test_values = test[feature].astype(float)
            pooled_std = float(pd.concat([train_values, test_values]).std())
            mean_gap = float(train_values.mean() - test_values.mean())
            rows.append(
                {
                    "feature": feature,
                    "comparison": "numeric",
                    "metric": "standardized_mean_difference",
                    "value": mean_gap / pooled_std if pooled_std else 0.0,
                    "max_category_rate_gap": float("nan"),
                }
            )
    return pd.DataFrame(rows)


def write_eda_outputs(frame: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    """Gera todas as tabelas da EDA inicial e retorna seus caminhos."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "feature_overview": output_dir / "feature_overview.csv",
        "categorical_target_rates": output_dir / "categorical_target_rates.csv",
        "numeric_by_target": output_dir / "numeric_by_target.csv",
        "split_comparison": output_dir / "split_comparison.csv",
        "conditional_target_rates": output_dir / "conditional_target_rates.csv",
        "category_risk_diagnostics": output_dir / "category_risk_diagnostics.csv",
        "split_distribution_drift": output_dir / "split_distribution_drift.csv",
    }
    _feature_overview(frame).to_csv(outputs["feature_overview"], index=False, encoding="utf-8")
    _categorical_target_rates(frame).to_csv(outputs["categorical_target_rates"], index=False, encoding="utf-8")
    _numeric_by_target(frame).to_csv(outputs["numeric_by_target"], index=False, encoding="utf-8")
    _split_comparison(frame).to_csv(outputs["split_comparison"], index=False, encoding="utf-8")
    _conditional_target_rates(frame).to_csv(
        outputs["conditional_target_rates"], index=False, encoding="utf-8"
    )
    _category_risk_diagnostics(frame).to_csv(
        outputs["category_risk_diagnostics"], index=False, encoding="utf-8"
    )
    _split_distribution_drift(frame).to_csv(
        outputs["split_distribution_drift"], index=False, encoding="utf-8"
    )
    return outputs
