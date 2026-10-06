from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold


def load_official_rul(data_dir: str | Path, dataset: str) -> pd.Series:
    """Carrega os RULs oficiais associados às trajetórias de teste."""
    path = Path(data_dir) / f"RUL_{dataset}.txt"
    values = pd.read_csv(path, sep=r"\s+", header=None, names=["rul"])
    return values["rul"].astype(float)


def last_observation_per_unit(test_df: pd.DataFrame) -> pd.DataFrame:
    """Retorna a última observação disponível de cada motor de teste."""
    ordered = test_df.sort_values(["unit", "cycle"])
    return ordered.groupby("unit", sort=True, as_index=False).tail(1).reset_index(drop=True)


def evaluate_official_test(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    official_rul: pd.Series,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Treina no conjunto completo e avalia o RUL oficial do teste FD001."""
    last_test = last_observation_per_unit(test_df)
    if len(last_test) != len(official_rul):
        raise ValueError(
            "O número de motores de teste não coincide com o arquivo oficial de RUL."
        )

    model.fit(train_df[features], train_df["rul"])
    prediction = model.predict(last_test[features])
    actual = official_rul.to_numpy(dtype=float)
    error = prediction - actual

    results = pd.DataFrame({
        "unit": last_test["unit"].to_numpy(),
        "last_cycle": last_test["cycle"].to_numpy(),
        "rul_real": actual,
        "rul_predito": prediction,
        "erro": error,
        "erro_absoluto": np.abs(error),
    })
    metrics = {
        "mae": float(mean_absolute_error(actual, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(actual, prediction))),
        "erro_medio": float(np.mean(error)),
        "mae_mediano_motor": float(results["erro_absoluto"].median()),
    }
    return results, metrics


def summarize_signed_error(results: pd.DataFrame) -> pd.DataFrame:
    """Resume a direção e a concentração dos erros por motor."""
    error = results["erro"]
    summary = {
        "erro_medio": float(error.mean()),
        "erro_mediano": float(error.median()),
        "percentual_superestimacao": float((error > 0).mean() * 100),
        "percentual_subestimacao": float((error < 0).mean() * 100),
        "percentual_exato": float((error == 0).mean() * 100),
        "erro_p10": float(error.quantile(0.10)),
        "erro_p90": float(error.quantile(0.90)),
        "maior_superestimacao": float(error.max()),
        "maior_subestimacao": float(error.min()),
        "motores": int(len(results)),
    }
    return pd.DataFrame([summary])


def grouped_oof_predictions(
    df: pd.DataFrame,
    features: list[str],
    estimators: dict[str, object],
    target: str = "rul",
    n_splits: int = 5,
) -> pd.DataFrame:
    """Gera previsões out-of-fold mantendo cada motor inteiro em um fold.

    O estimador é clonado e ajustado separadamente em cada partição. Se ele
    contiver seleção de atributos em um Pipeline, a seleção é recalculada
    somente com os motores usados no treinamento daquela partição.
    """
    required = {"unit", "cycle", target}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Colunas ausentes para avaliação agrupada: {sorted(missing)}")

    missing_features = set(features).difference(df.columns)
    if missing_features:
        raise KeyError(f"Atributos ausentes para avaliação: {sorted(missing_features)}")

    splitter = GroupKFold(n_splits=n_splits)
    X = df[features]
    y = df[target]
    groups = df["unit"]
    records: list[pd.DataFrame] = []

    for model_name, estimator in estimators.items():
        for fold, (train_index, validation_index) in enumerate(
            splitter.split(X, y, groups),
            start=1,
        ):
            fitted = clone(estimator)
            fitted.fit(X.iloc[train_index], y.iloc[train_index])

            validation = df.iloc[validation_index]
            prediction = np.asarray(fitted.predict(validation[features]), dtype=float)
            actual = validation[target].to_numpy(dtype=float)
            model_records = pd.DataFrame({
                "modelo": model_name,
                "fold": fold,
                "unit": validation["unit"].to_numpy(),
                "cycle": validation["cycle"].to_numpy(),
                "rul_real": actual,
                "rul_predito": prediction,
            })
            model_records["erro"] = model_records["rul_predito"] - model_records["rul_real"]
            model_records["erro_absoluto"] = model_records["erro"].abs()
            model_records["erro_quadratico"] = model_records["erro"] ** 2
            records.append(model_records)

    if not records:
        raise ValueError("Nenhuma previsão foi gerada.")
    return pd.concat(records, ignore_index=True)


def summarize_grouped_predictions(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Resume métricas por motor, por fold e no conjunto out-of-fold."""
    required = {
        "modelo", "fold", "unit", "erro", "erro_absoluto", "erro_quadratico",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise KeyError(f"Colunas ausentes nas previsões: {sorted(missing)}")

    per_unit = (
        predictions.groupby(["modelo", "fold", "unit"], as_index=False)
        .agg(
            mae_motor=("erro_absoluto", "mean"),
            rmse_motor=("erro_quadratico", lambda values: float(np.sqrt(values.mean()))),
            erro_medio_motor=("erro", "mean"),
            observacoes=("erro", "size"),
        )
    )

    fold_summary = (
        per_unit.groupby(["modelo", "fold"], as_index=False)
        .agg(
            mae_medio_motor=("mae_motor", "mean"),
            rmse_medio_motor=("rmse_motor", "mean"),
            erro_medio_motor=("erro_medio_motor", "mean"),
            pior_mae_motor=("mae_motor", "max"),
        )
    )

    rows: list[dict[str, float | int | str]] = []
    for model_name, model_predictions in predictions.groupby("modelo"):
        model_units = per_unit[per_unit["modelo"] == model_name]
        errors = model_predictions["erro"]
        rows.append({
            "modelo": model_name,
            "mae_medio_motor": float(model_units["mae_motor"].mean()),
            "mae_desvio_motor": float(model_units["mae_motor"].std()),
            "rmse_medio_motor": float(model_units["rmse_motor"].mean()),
            "rmse_desvio_motor": float(model_units["rmse_motor"].std()),
            "erro_medio_motor": float(model_units["erro_medio_motor"].mean()),
            "erro_desvio_motor": float(model_units["erro_medio_motor"].std()),
            "mae_por_observacao": float(model_predictions["erro_absoluto"].mean()),
            "rmse_por_observacao": float(np.sqrt(model_predictions["erro_quadratico"].mean())),
            "percentual_superestimacao": float((errors > 0).mean() * 100),
            "percentual_subestimacao": float((errors < 0).mean() * 100),
            "pior_mae_motor": float(model_units["mae_motor"].max()),
            "motores": int(model_units["unit"].nunique()),
            "observacoes": int(len(model_predictions)),
        })

    summary = (
        pd.DataFrame(rows)
        .sort_values("mae_medio_motor")
        .reset_index(drop=True)
    )
    return summary, per_unit, fold_summary


def paired_bootstrap_mae_comparison(
    results_by_model: dict[str, pd.DataFrame],
    n_bootstrap: int = 5_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compara MAEs em pares usando os mesmos motores e bootstrap pareado.

    A diferença é calculada como ``MAE_modelo_a - MAE_modelo_b``. Valores
    negativos favorecem o primeiro modelo. O intervalo de confiança é
    descritivo e não substitui a avaliação definida para o conjunto de teste.
    """
    if len(results_by_model) < 2:
        raise ValueError("São necessários pelo menos dois modelos para comparação.")

    required = {"unit", "erro_absoluto"}
    for model_name, results in results_by_model.items():
        missing = required.difference(results.columns)
        if missing:
            raise KeyError(
                f"Colunas ausentes nos resultados de {model_name}: {sorted(missing)}"
            )

    model_names = list(results_by_model)
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, float | int | str]] = []

    for index, model_a in enumerate(model_names):
        left = results_by_model[model_a][["unit", "erro_absoluto"]].rename(
            columns={"erro_absoluto": "erro_a"}
        )
        for model_b in model_names[index + 1:]:
            right = results_by_model[model_b][["unit", "erro_absoluto"]].rename(
                columns={"erro_absoluto": "erro_b"}
            )
            paired = left.merge(right, on="unit", how="inner", validate="one_to_one")
            if paired.empty:
                raise ValueError(f"Nenhum motor comum entre {model_a} e {model_b}.")

            differences = (paired["erro_a"] - paired["erro_b"]).to_numpy(dtype=float)
            sample_index = rng.integers(
                0,
                len(differences),
                size=(n_bootstrap, len(differences)),
            )
            bootstrap_means = differences[sample_index].mean(axis=1)
            rows.append({
                "modelo_a": model_a,
                "modelo_b": model_b,
                "diferenca_mae_a_menos_b": float(differences.mean()),
                "ic95_inferior": float(np.quantile(bootstrap_means, 0.025)),
                "ic95_superior": float(np.quantile(bootstrap_means, 0.975)),
                "vitorias_modelo_a": int((differences < 0).sum()),
                "vitorias_modelo_b": int((differences > 0).sum()),
                "empates": int((differences == 0).sum()),
                "motores_comuns": int(len(differences)),
            })

    return pd.DataFrame(rows)
