"""Documenta a decisão final e diagnostica os erros do Extra Trees no FD001."""

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
    summarize_error_profile,
)
from cmapss_evaluation import evaluate_official_test
from cmapss_modeling import add_rul_label, candidate_rul_features, nonconstant_features
from cmapss_optuna import build_pipeline_from_params
from cmapss_temporal_features import add_causal_temporal_features, temporal_feature_names


DATA_DIR = PROJECT_ROOT / "data" / "raw" / "dataset"
STORAGE_DIR = PROJECT_ROOT / "outputs" / "optuna_temporal"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
STUDY_NAME = "fd001_temporal_205_extra_trees_v1"


def format_table(frame: pd.DataFrame, decimals: int = 2) -> str:
    """Converte uma tabela numérica em Markdown sem expor índices."""

    formatted = frame.copy()
    numeric = formatted.select_dtypes(include="number").columns
    formatted[numeric] = formatted[numeric].round(decimals)
    headers = [str(column) for column in formatted.columns]
    separator = ["---"] * len(headers)
    rows = [headers, separator]
    for values in formatted.itertuples(index=False, name=None):
        rows.append(["" if pd.isna(value) else str(value) for value in values])
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train = add_rul_label(load_split(DATA_DIR, "train", "FD001"))
    test = load_split(DATA_DIR, "test", "FD001")

    features_base = nonconstant_features(train, candidate_rul_features())
    temporal_sources = [feature for feature in features_base if feature.startswith("s")]
    temporal_windows = (5, 10, 20, 40, 80)

    train_temporal = add_causal_temporal_features(
        train, sensors=temporal_sources, windows=temporal_windows
    )
    test_temporal = add_causal_temporal_features(
        test, sensors=temporal_sources, windows=temporal_windows
    )
    features_temporal = features_base + temporal_feature_names(
        temporal_sources, temporal_windows
    )
    if len(features_temporal) != 205:
        raise AssertionError(f"Esperados 205 atributos temporais; encontrados {len(features_temporal)}")

    db_path = STORAGE_DIR / f"{STUDY_NAME}.db"
    study = optuna.load_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{db_path.as_posix()}",
    )
    model = build_pipeline_from_params(
        model_name="Extra Trees",
        params=study.best_params,
        n_features=len(features_temporal),
        random_state=42,
        search_version="v1",
    )

    official_rul = __import__("cmapss_evaluation").load_official_rul(DATA_DIR, "FD001")
    results, metrics = evaluate_official_test(
        model=model,
        train_df=train_temporal,
        test_df=test_temporal,
        features=features_temporal,
        official_rul=official_rul,
    )

    profile = results.rename(
        columns={
            "unit": "group",
            "last_cycle": "cycle",
            "rul_real": "y_true",
            "rul_predito": "y_predito",
        }
    )
    profile = enrich_error_profile(
        profile,
        rul_column="y_true",
        prediction_column="y_predito",
        group_column="group",
        cycle_column="cycle",
    )
    profile = enrich_observable_error_profile(
        profile,
        prediction_column="y_predito",
        cycle_column="cycle",
    )

    motor_columns = [
        "group",
        "cycle",
        "y_true",
        "y_predito",
        "erro",
        "erro_absoluto",
        "faixa_rul",
        "faixa_cycle",
        "faixa_rul_predito",
    ]
    motor_profile = profile[motor_columns].sort_values(
        "erro_absoluto", ascending=False
    )

    rul_summary = summarize_error_profile(profile, "faixa_rul", group_column="group")
    cycle_summary = summarize_error_profile(profile, "faixa_cycle", group_column="group")

    profile_path = REPORT_DIR / "extra-trees-erros-por-motor-fd001.csv"
    rul_path = REPORT_DIR / "extra-trees-erros-por-rul-fd001.csv"
    cycle_path = REPORT_DIR / "extra-trees-erros-por-ciclo-fd001.csv"
    motor_profile.to_csv(profile_path, index=False)
    rul_summary.to_csv(rul_path, index=False)
    cycle_summary.to_csv(cycle_path, index=False)

    comparison = pd.DataFrame(
        [
            {"modelo": "Extra Trees", "mae": 16.23, "rmse": 23.80, "erro_medio": 10.08},
            {"modelo": "XGBoost", "mae": 17.29, "rmse": 25.15, "erro_medio": 9.31},
            {"modelo": "HistGradientBoosting", "mae": 17.77, "rmse": 27.01, "erro_medio": 8.83},
        ]
    )

    best = motor_profile.head(10)
    worst = motor_profile.head(1).iloc[0]
    report_path = REPORT_DIR / "decisao-final-extra-trees-fd001.md"
    report = f"""# Decisão final e análise de erros — FD001

## Decisão metodológica

O benchmark oficial foi congelado antes desta análise. Os três candidatos receberam os mesmos 205 atributos temporais, foram treinados com o mesmo protocolo e avaliados nos mesmos 100 motores do conjunto oficial de teste. Nenhum hiperparâmetro foi ajustado usando esse conjunto.

O Extra Trees foi escolhido como modelo final do FD001 porque apresentou o menor MAE e o menor RMSE no benchmark oficial:

{format_table(comparison)}

Os valores acima são o registro congelado do notebook `21_benchmark_oficial_finalistas_atributos_temporais_rul_fd001.ipynb`. A análise abaixo foi recalculada com o estudo Optuna congelado do Extra Trees (`{STUDY_NAME}`), usando os mesmos 205 atributos.

## Métricas oficiais do Extra Trees

- MAE: **{metrics['mae']:.2f} ciclos**
- RMSE: **{metrics['rmse']:.2f} ciclos**
- Erro médio assinado: **{metrics['erro_medio']:.2f} ciclos**
- Mediana do erro absoluto por motor: **{metrics['mae_mediano_motor']:.2f} ciclos**
- Motores avaliados: **{len(profile)}**

O erro médio positivo indica tendência de **superestimação do RUL**: o modelo prevê, em média, mais vida útil do que a observada. Essa direção deve ser considerada na interpretação operacional.

## Erros por motor

O maior erro absoluto ocorreu no motor **{int(worst['group'])}**, no ciclo observado **{int(worst['cycle'])}**, com RUL real de **{worst['y_true']:.2f}**, RUL previsto de **{worst['y_predito']:.2f}** e erro assinado de **{worst['erro']:.2f} ciclos**.

Os dez maiores erros absolutos estão no arquivo [extra-trees-erros-por-motor-fd001.csv](extra-trees-erros-por-motor-fd001.csv). A tabela abaixo mostra os casos mais críticos:

{format_table(best)}

## Erros por faixa de RUL real

Esta divisão usa o RUL real somente para diagnóstico retrospectivo e medição do desempenho. Ele não foi usado como atributo de entrada.

{format_table(rul_summary)}

O padrão esperado aparece claramente: o erro tende a crescer nas faixas de RUL mais altas. Nelas, o modelo precisa estimar uma vida útil longa a partir de sinais ainda relativamente distantes da falha, o que torna a tarefa mais incerta.

## Erros por ciclo observado

No benchmark oficial há uma observação final por motor. Portanto, `cycle` representa o último ciclo disponível antes da previsão para aquele motor, e não todos os ciclos da trajetória.

{format_table(cycle_summary)}

Essa análise é útil para verificar se motores observados em diferentes durações de operação apresentam regimes de erro distintos. A concentração de superestimação nas primeiras faixas é uma associação observada no teste oficial, mas deve ser tratada como exploratória até ser comparada com o OOF. Ela não deve ser confundida com uma feature de faixa de ciclo: `faixa_cycle` foi criada apenas depois da previsão, para diagnóstico.

## Limites e próximo uso

- A avaliação oficial usa os 100 motores do teste FD001 uma única vez.
- As faixas de RUL e ciclo são estratificações diagnósticas; não são entradas do modelo.
- O benchmark não deve ser reutilizado para escolher hiperparâmetros ou criar novas features.
- Qualquer nova melhoria deve ser avaliada em uma nova divisão de desenvolvimento, preservando este benchmark como resultado final selado.

Arquivos detalhados:

- [Erros por motor](extra-trees-erros-por-motor-fd001.csv)
- [Erros por faixa de RUL](extra-trees-erros-por-rul-fd001.csv)
- [Erros por faixa de ciclo](extra-trees-erros-por-ciclo-fd001.csv)
"""
    report_path.write_text(report, encoding="utf-8")

    print(f"Relatório salvo em: {report_path}")
    print(f"Perfil por motor salvo em: {profile_path}")
    print(f"MAE={metrics['mae']:.4f} | RMSE={metrics['rmse']:.4f} | erro_medio={metrics['erro_medio']:.4f}")
    print("\nResumo por faixa de RUL:")
    print(rul_summary.round(2).to_string(index=False))
    print("\nResumo por faixa de ciclo:")
    print(cycle_summary.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
