"""Executa um estudo Optuna temporal para um único estimador."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import optuna


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cmapss_eda import load_split
from cmapss_modeling import add_rul_label, candidate_rul_features, nonconstant_features
from cmapss_optuna import optimize_model, study_summary
from cmapss_temporal_features import add_causal_temporal_features, temporal_feature_names


FINALISTS = ("Extra Trees", "XGBoost", "HistGradientBoosting")


def build_temporal_training_data():
    data_dir = PROJECT_ROOT / "data" / "raw" / "dataset"
    train = add_rul_label(load_split(data_dir, "train", "FD001"))
    features_base = nonconstant_features(train, candidate_rul_features())
    temporal_sources = [feature for feature in features_base if feature.startswith("s")]
    temporal_windows = (5, 10, 20, 40, 80)
    train_temporal = add_causal_temporal_features(
        train,
        sensors=temporal_sources,
        windows=temporal_windows,
    )
    features_temporal = features_base + temporal_feature_names(
        temporal_sources,
        temporal_windows,
    )
    if len(features_temporal) != 205:
        raise RuntimeError(
            f"Representação temporal inesperada: {len(features_temporal)} atributos; "
            "eram esperados 205."
        )
    return train_temporal, features_temporal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Otimiza um único estimador com 205 atributos temporais."
    )
    parser.add_argument("--model", choices=FINALISTS, required=True)
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--splits", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    optuna.logging.set_verbosity(optuna.logging.INFO)

    train_temporal, features_temporal = build_temporal_training_data()
    storage_dir = PROJECT_ROOT / "outputs" / "optuna_temporal"
    print(
        f"Início: {args.model} | trials-alvo={args.trials} | "
        f"folds={args.splits} | atributos={len(features_temporal)}",
        flush=True,
    )
    print(
        "O estudo será retomado do SQLite se já houver trials concluídos.",
        flush=True,
    )

    study = optimize_model(
        train_temporal,
        features=features_temporal,
        model_name=args.model,
        n_trials=args.trials,
        n_splits=args.splits,
        random_state=42,
        storage_dir=storage_dir,
        search_version="v1",
        study_id="fd001_temporal_205",
        show_progress_bar=True,
        log_trials=True,
    )

    summary = study_summary({args.model: study})
    print("\nResumo do estudo:", flush=True)
    print(summary[["modelo", "mae_cv_motor", "trials", "melhor_trial", "status"]].to_string(index=False))
    print("\nMelhores parâmetros:", flush=True)
    for name, value in study.best_params.items():
        print(f"  {name}={value}", flush=True)


if __name__ == "__main__":
    main()
