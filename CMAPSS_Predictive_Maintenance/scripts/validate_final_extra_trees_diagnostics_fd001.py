"""Valida a robustez dos padrões diagnósticos do Extra Trees no FD001."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cmapss_eda import load_split
from cmapss_error_analysis import (
    enrich_error_profile,
    enrich_observable_error_profile,
)
from cmapss_evaluation import grouped_oof_predictions
from cmapss_modeling import add_rul_label, candidate_rul_features, nonconstant_features
from cmapss_optuna import build_pipeline_from_params
from cmapss_temporal_features import add_causal_temporal_features, temporal_feature_names


DATA_DIR = PROJECT_ROOT / "data" / "raw" / "dataset"
STORAGE_DIR = PROJECT_ROOT / "outputs" / "optuna_temporal"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
STUDY_NAME = "fd001_temporal_205_extra_trees_v1"


def enrich_for_diagnostics(frame: pd.DataFrame, group_column: str = "unit") -> pd.DataFrame:
    """Aplica as mesmas categorias diagnósticas usadas no teste oficial."""

    renamed = frame.rename(
        columns={
            group_column: "group",
            "rul_real": "y_true",
            "rul_predito": "y_predito",
        }
    )
    enriched = enrich_error_profile(
        renamed,
        rul_column="y_true",
        prediction_column="y_predito",
        group_column="group",
        cycle_column="cycle",
    )
    enriched = enrich_observable_error_profile(
        enriched,
        prediction_column="y_predito",
        cycle_column="cycle",
    )
    return enriched.rename(columns={"group": group_column, "y_true": "rul_real", "y_predito": "rul_predito"})


def unit_category_metrics(frame: pd.DataFrame, category: str) -> pd.DataFrame:
    """Agrega primeiro por motor para não tratar linhas temporais como independentes."""

    return (
        frame.dropna(subset=[category])
        .groupby([category, "unit"], observed=True)
        .agg(
            erro_medio_unidade=("erro", "mean"),
            mae_unidade=("erro_absoluto", "mean"),
            superestimacao_unidade=("erro", lambda values: float((values > 0).mean() * 100)),
            subestimacao_unidade=("erro", lambda values: float((values < 0).mean() * 100)),
        )
        .reset_index()
    )


def bootstrap_category_metrics(
    frame: pd.DataFrame,
    category: str,
    source: str,
    n_bootstrap: int = 2_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Calcula médias e IC95% reamostrando motores dentro de cada categoria."""

    unit_metrics = unit_category_metrics(frame, category)
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, float | int | str]] = []
    metrics = [
        "erro_medio_unidade",
        "mae_unidade",
        "superestimacao_unidade",
        "subestimacao_unidade",
    ]

    for label, group in unit_metrics.groupby(category, observed=True, sort=False):
        values = group[metrics].to_numpy(dtype=float)
        n_units = len(values)
        sampled = rng.integers(0, n_units, size=(n_bootstrap, n_units))
        bootstrap_means = values[sampled].mean(axis=1)
        row: dict[str, float | int | str] = {
            "origem": source,
            "categoria": str(label),
            "motores": int(n_units),
        }
        for index, metric in enumerate(metrics):
            base_name = metric.removesuffix("_unidade")
            row[f"{base_name}_medio"] = float(values[:, index].mean())
            row[f"{base_name}_ic95_inferior"] = float(np.quantile(bootstrap_means[:, index], 0.025))
            row[f"{base_name}_ic95_superior"] = float(np.quantile(bootstrap_means[:, index], 0.975))
        rows.append(row)

    return pd.DataFrame(rows)


def distribution_summary(frame: pd.DataFrame, source: str) -> dict[str, float | int | str]:
    errors = frame["erro"]
    return {
        "origem": source,
        "motores_ou_observacoes": int(len(frame)),
        "superestimados": int((errors > 0).sum()),
        "subestimados": int((errors < 0).sum()),
        "percentual_superestimacao": float((errors > 0).mean() * 100),
        "percentual_subestimacao": float((errors < 0).mean() * 100),
        "erro_medio": float(errors.mean()),
        "mediana_erro": float(errors.median()),
        "magnitude_media_superestimacao": float(errors[errors > 0].mean()),
        "magnitude_media_subestimacao": float(errors[errors < 0].abs().mean()),
    }


def markdown_table(frame: pd.DataFrame, decimals: int = 2) -> str:
    formatted = frame.copy()
    numeric = formatted.select_dtypes(include="number").columns
    formatted[numeric] = formatted[numeric].round(decimals)
    rows = [[str(column) for column in formatted.columns], ["---"] * len(formatted.columns)]
    rows.extend([["" if pd.isna(value) else str(value) for value in row] for row in formatted.itertuples(index=False, name=None)])
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train = add_rul_label(load_split(DATA_DIR, "train", "FD001"))
    features_base = nonconstant_features(train, candidate_rul_features())
    temporal_sources = [feature for feature in features_base if feature.startswith("s")]
    temporal_windows = (5, 10, 20, 40, 80)
    train_temporal = add_causal_temporal_features(
        train, sensors=temporal_sources, windows=temporal_windows
    )
    features_temporal = features_base + temporal_feature_names(temporal_sources, temporal_windows)
    if len(features_temporal) != 205:
        raise AssertionError(f"Esperados 205 atributos; encontrados {len(features_temporal)}")

    db_path = STORAGE_DIR / f"{STUDY_NAME}.db"
    study = optuna.load_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{db_path.as_posix()}",
    )
    estimator = build_pipeline_from_params(
        model_name="Extra Trees",
        params=study.best_params,
        n_features=len(features_temporal),
        random_state=42,
        search_version="v1",
    )

    oof = grouped_oof_predictions(
        train_temporal,
        features=features_temporal,
        estimators={"Extra Trees": estimator},
        n_splits=5,
    )
    oof = enrich_for_diagnostics(oof)

    official_path = REPORT_DIR / "extra-trees-erros-por-motor-fd001.csv"
    official = pd.read_csv(official_path).rename(columns={"group": "unit", "cycle": "cycle"})
    oof_unit_distribution = (
        oof.groupby("unit", as_index=False)
        .agg(erro=("erro", "mean"))
    )
    oof_unit_distribution["erro_absoluto"] = oof_unit_distribution["erro"].abs()

    distribution = pd.DataFrame([
        distribution_summary(official, "Teste oficial"),
        distribution_summary(oof_unit_distribution, "OOF desenvolvimento (média por motor)"),
    ])
    rul_validation = pd.concat([
        bootstrap_category_metrics(official, "faixa_rul", "Teste oficial"),
        bootstrap_category_metrics(oof, "faixa_rul", "OOF desenvolvimento"),
    ], ignore_index=True)
    cycle_validation = pd.concat([
        bootstrap_category_metrics(official, "faixa_cycle", "Teste oficial"),
        bootstrap_category_metrics(oof, "faixa_cycle", "OOF desenvolvimento"),
    ], ignore_index=True)

    oof_path = REPORT_DIR / "extra-trees-oof-erros-fd001.csv"
    distribution_path = REPORT_DIR / "extra-trees-validacao-distribuicao-fd001.csv"
    rul_path = REPORT_DIR / "extra-trees-validacao-associacao-rul-fd001.csv"
    cycle_path = REPORT_DIR / "extra-trees-validacao-associacao-ciclo-fd001.csv"
    oof.to_csv(oof_path, index=False)
    distribution.to_csv(distribution_path, index=False)
    rul_validation.to_csv(rul_path, index=False)
    cycle_validation.to_csv(cycle_path, index=False)

    report_path = REPORT_DIR / "validacao-associacoes-diagnosticas-extra-trees-fd001.md"
    report = f"""# Validação das associações diagnósticas — Extra Trees — FD001

## Protocolo

Esta etapa não ajusta o modelo e não usa o teste oficial para seleção. O Extra Trees mantém os hiperparâmetros Optuna congelados. A comparação é entre:

- **Teste oficial:** uma previsão final por cada um dos 100 motores do teste;
- **OOF desenvolvimento:** previsões out-of-fold com cinco partições por motor no conjunto de treino.

Os intervalos de confiança foram calculados por bootstrap reamostrando motores, e não linhas, para respeitar a dependência temporal dentro de cada trajetória.

## Distribuição geral

{markdown_table(distribution)}

## Associação com a faixa de RUL

As associações por RUL são diagnósticas. O RUL real não foi usado como entrada do modelo.

{markdown_table(rul_validation)}

## Associação com o ciclo observado

No teste oficial, o ciclo observado é o último ciclo disponível para cada motor. No OOF, há várias observações temporais por motor; por isso, a agregação e o bootstrap foram feitos primeiro no nível do motor.

{markdown_table(cycle_validation)}

## Interpretação

O padrão deve ser considerado robusto apenas quando a direção aparece tanto no teste oficial quanto no OOF e os intervalos de confiança não forem excessivamente largos. A distribuição geral do OOF foi resumida pela média do erro de cada motor, para ficar comparável ao teste oficial. Faixas com poucos motores devem permanecer como evidência fraca, mesmo quando a média parecer grande.

Os arquivos completos estão em:

- [Previsões OOF](extra-trees-oof-erros-fd001.csv)
- [Distribuição geral](extra-trees-validacao-distribuicao-fd001.csv)
- [Validação por RUL](extra-trees-validacao-associacao-rul-fd001.csv)
- [Validação por ciclo](extra-trees-validacao-associacao-ciclo-fd001.csv)
"""
    report_path.write_text(report, encoding="utf-8")

    print(f"Relatório salvo em: {report_path}")
    print(f"OOF salvo em: {oof_path}")
    print("\nDistribuição:")
    print(distribution.round(2).to_string(index=False))
    print("\nValidação por RUL:")
    print(rul_validation.round(2).to_string(index=False))
    print("\nValidação por ciclo:")
    print(cycle_validation.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
