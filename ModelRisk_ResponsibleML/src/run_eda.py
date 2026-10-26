"""Executa a EDA inicial do UCI Adult e salva artefatos auditáveis."""

from pathlib import Path

import pandas as pd

from adult_data import load_adult, write_quality_outputs


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "adult_uci"
RAW_ZIP = ROOT / "data" / "raw" / "adult.zip"
OUTPUT_DIR = ROOT / "data" / "processed" / "eda_initial"


def main() -> None:
    """Gera tabelas, gráficos mínimos e o contrato inicial do dataset."""
    frame = load_adult(RAW_DIR)
    profile = write_quality_outputs(frame, RAW_ZIP, OUTPUT_DIR)
    frame.to_parquet(OUTPUT_DIR / "adult_combined.parquet", index=False)
    frame["income_rate"] = (frame["income"] == ">50K").astype(int)
    frame.groupby("sex", dropna=False)["income_rate"].agg(["count", "mean"]).reset_index().to_csv(
        OUTPUT_DIR / "income_rate_by_sex.csv", index=False, encoding="utf-8"
    )
    frame.groupby("race", dropna=False)["income_rate"].agg(["count", "mean"]).reset_index().to_csv(
        OUTPUT_DIR / "income_rate_by_race.csv", index=False, encoding="utf-8"
    )
    frame.groupby(["sex", "race"], dropna=False)["income_rate"].agg(["count", "mean"]).reset_index().to_csv(
        OUTPUT_DIR / "income_rate_by_sex_race.csv", index=False, encoding="utf-8"
    )
    frame[["age", "education_num", "hours_per_week", "capital_gain", "capital_loss"]].describe().transpose().to_csv(
        OUTPUT_DIR / "numeric_summary.csv", encoding="utf-8"
    )

    print(profile)
    print(frame.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
