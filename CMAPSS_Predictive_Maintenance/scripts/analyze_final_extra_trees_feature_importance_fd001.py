"""Analisa a importância dos atributos do Extra Trees final no FD001."""

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
from cmapss_explainability import permutation_importance_by_engine
from cmapss_modeling import add_rul_label, candidate_rul_features, nonconstant_features
from cmapss_optuna import build_pipeline_from_params
from cmapss_temporal_features import add_causal_temporal_features, temporal_feature_names


DATA_DIR = PROJECT_ROOT / "data" / "raw" / "dataset"
STORAGE_DIR = PROJECT_ROOT / "outputs" / "optuna_temporal"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
STUDY_NAME = "fd001_temporal_205_extra_trees_v1"


def feature_metadata(feature: str) -> dict[str, str | int]:
    if feature.endswith("_delta"):
        return {"familia": "Delta", "tipo": "Delta", "janela": "Atual"}
    if "_media_" in feature:
        return {"familia": "Média móvel", "tipo": "Média", "janela": int(feature.rsplit("_", 1)[-1])}
    if "_desvio_" in feature:
        return {"familia": "Desvio-padrão móvel", "tipo": "Desvio-padrão", "janela": int(feature.rsplit("_", 1)[-1])}
    return {"familia": "Base", "tipo": "Base", "janela": "Atual"}


def markdown_table(frame: pd.DataFrame, decimals: int = 4) -> str:
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

    study = optuna.load_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{(STORAGE_DIR / (STUDY_NAME + '.db')).as_posix()}",
    )
    model = build_pipeline_from_params(
        model_name="Extra Trees",
        params=study.best_params,
        n_features=len(features_temporal),
        random_state=42,
        search_version="v1",
    )

    model.fit(train_temporal[features_temporal], train_temporal["rul"])
    selector = model.named_steps["feature_selection"]
    estimator = model.named_steps["estimator"]
    if hasattr(selector, "get_support"):
        selected_mask = selector.get_support()
    else:
        selected_mask = np.ones(len(features_temporal), dtype=bool)

    native = pd.DataFrame({"feature": features_temporal})
    native["selecionada_optuna"] = selected_mask
    native["importancia_nativa"] = 0.0
    native.loc[selected_mask, "importancia_nativa"] = estimator.feature_importances_

    metadata = pd.DataFrame([feature_metadata(feature) for feature in features_temporal])
    native = pd.concat([native, metadata], axis=1)
    native = native.sort_values("importancia_nativa", ascending=False).reset_index(drop=True)

    permutation_path = REPORT_DIR / "extra-trees-importancia-permutacao-fd001.csv"
    if permutation_path.exists():
        permutation = pd.read_csv(
            permutation_path,
            usecols=[
                "feature",
                "importance_mean",
                "importance_std_between_folds",
                "repeats_std_mean",
            ],
        )
    else:
        permutation = permutation_importance_by_engine(
            train_temporal,
            features=features_temporal,
            model=model,
            n_splits=5,
            n_repeats=3,
            random_state=42,
        )
    permutation = permutation.merge(metadata.assign(feature=features_temporal), on="feature", how="left")
    permutation["selecionada_optuna"] = permutation["feature"].isin(native.loc[native["selecionada_optuna"], "feature"])
    permutation = permutation.sort_values("importance_mean", ascending=False).reset_index(drop=True)

    family_native = (
        native.groupby(["familia", "tipo"], as_index=False)
        .agg(
            atributos=("feature", "size"),
            selecionados=("selecionada_optuna", "sum"),
            importancia_nativa_total=("importancia_nativa", "sum"),
        )
        .sort_values("importancia_nativa_total", ascending=False)
    )
    family_permutation = (
        permutation.groupby(["familia", "tipo"], as_index=False)
        .agg(
            atributos=("feature", "size"),
            selecionados=("selecionada_optuna", "sum"),
            importancia_permutacao_media=("importance_mean", "mean"),
            importancia_permutacao_abs_media=("importance_mean", lambda values: float(np.abs(values).mean())),
            atributos_importancia_positiva=("importance_mean", lambda values: int((values > 0).sum())),
        )
        .sort_values("importancia_permutacao_media", ascending=False)
    )

    native_path = REPORT_DIR / "extra-trees-importancia-nativa-fd001.csv"
    family_native_path = REPORT_DIR / "extra-trees-importancia-familias-nativa-fd001.csv"
    family_permutation_path = REPORT_DIR / "extra-trees-importancia-familias-permutacao-fd001.csv"
    native.to_csv(native_path, index=False)
    permutation.to_csv(permutation_path, index=False)
    family_native.to_csv(family_native_path, index=False)
    family_permutation.to_csv(family_permutation_path, index=False)

    report_path = REPORT_DIR / "importancia-features-extra-trees-fd001.md"
    report = f"""# Importância dos atributos — Extra Trees final — FD001

## Protocolo

O modelo usado é o Extra Trees final, reconstruído com o estudo Optuna congelado `{STUDY_NAME}` e treinado nos 205 atributos temporais. A seleção do Optuna manteve **{int(selected_mask.sum())} atributos**.

Foram calculadas duas medidas:

- **Importância nativa:** redução média de impureza usada pelas árvores. É descritiva e pode distribuir importância entre atributos correlacionados.
- **Importância por permutação:** aumento do MAE quando o atributo é embaralhado. Foi calculada em cinco folds por motor, com três repetições, usando somente o desenvolvimento OOF.

O conjunto oficial de teste não foi usado para calcular a importância.

## Maiores importâncias nativas

{markdown_table(native.head(20)[['feature', 'familia', 'tipo', 'janela', 'selecionada_optuna', 'importancia_nativa']])}

## Maiores importâncias por permutação

Uma importância positiva indica que a permutação piorou o MAE. Valores negativos sugerem redundância, instabilidade ou que a permutação acidentalmente melhorou a previsão naquele conjunto de validação.

{markdown_table(permutation.head(20)[['feature', 'familia', 'tipo', 'janela', 'selecionada_optuna', 'importance_mean', 'importance_std_between_folds']])}

## Importância por família

### Importância nativa

{markdown_table(family_native)}

### Importância por permutação

{markdown_table(family_permutation)}

## Limitações

Importância não significa causalidade. Atributos temporais da mesma variável e de janelas próximas são correlacionados; por isso, a contribuição pode ser compartilhada entre eles. A importância por permutação deve ser interpretada junto com a estabilidade entre folds e não como uma prova de que uma variável isolada determina o RUL.

Arquivos detalhados:

- [Importância nativa](extra-trees-importancia-nativa-fd001.csv)
- [Importância por permutação](extra-trees-importancia-permutacao-fd001.csv)
- [Famílias — importância nativa](extra-trees-importancia-familias-nativa-fd001.csv)
- [Famílias — importância por permutação](extra-trees-importancia-familias-permutacao-fd001.csv)
"""
    report_path.write_text(report, encoding="utf-8")

    print(f"Relatório salvo em: {report_path}")
    print(f"Atributos selecionados: {int(selected_mask.sum())} de {len(features_temporal)}")
    print("\nTop 15 por permutação:")
    print(permutation.head(15).round(4).to_string(index=False))
    print("\nFamílias por permutação:")
    print(family_permutation.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
