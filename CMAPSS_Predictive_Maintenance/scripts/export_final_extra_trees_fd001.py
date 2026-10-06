"""Exporta o pipeline congelado FD001 e verifica serialização e causalidade."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import joblib
import numpy as np
import optuna
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cmapss_eda import load_split
from cmapss_modeling import add_rul_label, candidate_rul_features, nonconstant_features
from cmapss_temporal_features import add_causal_temporal_features, temporal_feature_names
from cmapss_optuna import build_pipeline_from_params
from cmapss_final_inference import predict_history, load_final_bundle

OUT = ROOT / "outputs" / "models" / "extra_trees_fd001_v1"
STUDY = "fd001_temporal_205_extra_trees_v1"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(f"Entrega já existe; preservada: {OUT}")
    train = add_rul_label(load_split(ROOT / "data" / "raw" / "dataset", "train", "FD001"))
    base = nonconstant_features(train, candidate_rul_features())
    sources = [f for f in base if f.startswith("s")]
    windows = (5, 10, 20, 40, 80)
    features = base + temporal_feature_names(sources, windows)
    assert len(features) == 205
    db = ROOT / "outputs" / "optuna_temporal" / f"{STUDY}.db"
    study = optuna.load_study(study_name=STUDY, storage=f"sqlite:///{db.as_posix()}")
    model = build_pipeline_from_params(
        model_name="Extra Trees", params=study.best_params,
        n_features=len(features), random_state=42, search_version="v1")
    temporal = add_causal_temporal_features(train, sources, windows)
    print("Ajuste final com hiperparâmetros congelados; 100 motores e 205 atributos.", flush=True)
    model.fit(temporal[features], temporal.rul)
    selected = [f for f, keep in zip(features, model.named_steps["feature_selection"].get_support()) if keep]
    bundle = dict(schema_version=1, dataset="FD001", pipeline=model, base_features=base,
                  temporal_sources=sources, windows=list(windows), features=features)
    sample = train.loc[(train.unit == train.unit.min()) & (train.cycle <= 80), ["unit", *base]].copy()
    before = predict_history(sample, bundle)
    np.testing.assert_allclose(before.rul_predito, model.predict(
        temporal.loc[(temporal.unit == sample.unit.iloc[0]) & (temporal.cycle == 80), features]))
    # Uma extensão posterior do histórico não pode modificar previsões anteriores.
    extended = train.loc[train.unit == sample.unit.iloc[0], ["unit", *base]]
    all_predictions = predict_history(extended, bundle, latest_only=False)
    prefix_predictions = predict_history(sample, bundle, latest_only=False)
    np.testing.assert_allclose(prefix_predictions.rul_predito,
                               all_predictions.loc[all_predictions.cycle <= 80, "rul_predito"])
    # Contratos inválidos devem falhar antes de produzir previsões.
    for invalid in [sample.iloc[1:], pd.concat([sample, sample.iloc[:1]])]:
        try:
            predict_history(invalid, bundle)
        except ValueError:
            pass
        else:
            raise AssertionError("Contrato inválido aceito")
    OUT.mkdir(parents=True)
    artifact = OUT / "model.joblib"
    joblib.dump(bundle, artifact, compress=3)
    loaded = load_final_bundle(artifact)
    np.testing.assert_allclose(before.rul_predito, predict_history(sample, loaded).rul_predito)
    sample.to_csv(OUT / "historico_exemplo.csv", index=False)
    before.to_csv(OUT / "previsao_exemplo.csv", index=False)
    source_paths = ["src/cmapss_temporal_features.py", "src/cmapss_final_inference.py", "src/cmapss_optuna.py"]
    metadata = dict(schema_version=1, dataset="FD001", estimator="Extra Trees",
                    study=STUDY, best_trial=study.best_trial.number, parameters=study.best_params,
                    training_rows=len(train), training_engines=int(train.unit.nunique()),
                    base_features=base, temporal_sources=sources, windows=list(windows),
                    features=features, selected_features=selected,
                    python=platform.python_version(),
                    packages={name: importlib.metadata.version(name) for name in
                              ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "optuna", "xgboost", "catboost"]},
                    sha256_model=sha256(artifact), sha256_optuna_db=sha256(db),
                    sha256_training_file=sha256(ROOT / "data" / "raw" / "dataset" / "train_FD001.txt"),
                    sha256_sources={name: sha256(ROOT / name) for name in source_paths},
                    checks=["serialização", "equivalência com engenharia de treino", "invariância ao futuro",
                            "rejeição de histórico incompleto", "rejeição de ciclos duplicados"],
                    limitations="FD001 simulado; alertas e intervalos exploratórios; artefato contém previsão pontual.",
                    recorded_benchmark={"mae":16.23, "rmse":23.80, "signed_error":10.08,
                                        "note":"Registro do benchmark anterior; não recalculado na exportação."})
    (OUT / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exportado: {artifact}\nSelecionados: {len(selected)}\nVerificações concluídas.")


if __name__ == "__main__":
    main()

