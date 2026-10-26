"""Carregamento e perfil estrutural do UCI Adult.

Este módulo mantém a leitura dos arquivos brutos separada dos notebooks,
permitindo que a narrativa de EDA seja curta e reproduzível.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

AUDIT_COLUMNS = ["sex", "race"]
TARGET = "income"
EXPECTED_TARGETS = {"<=50K", ">50K"}
EXPECTED_SPLITS = {"train_official", "test_official"}


def _read_split(path: Path, split: str) -> pd.DataFrame:
    """Lê um arquivo Adult e acrescenta a identificação do split."""
    frame = pd.read_csv(
        path,
        names=COLUMNS,
        header=None,
        skipinitialspace=True,
        na_values=["?"],
        comment="|",
    )
    frame["split"] = split
    frame[TARGET] = frame[TARGET].astype("string").str.rstrip(".")
    return frame


def load_adult(raw_dir: str | Path) -> pd.DataFrame:
    """Carrega os splits oficiais e retorna um único DataFrame tipado."""
    raw_dir = Path(raw_dir)
    data = _read_split(raw_dir / "adult.data", "train_official")
    test = _read_split(raw_dir / "adult.test", "test_official")
    frame = pd.concat([data, test], ignore_index=True)
    for column in ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex", "native_country", TARGET, "split"]:
        frame[column] = frame[column].astype("string")
    validate_contract(frame)
    return frame


def validate_contract(frame: pd.DataFrame) -> None:
    """Falha cedo quando o arquivo carregado viola o contrato semântico."""
    expected_columns = COLUMNS + ["split"]
    if list(frame.columns) != expected_columns:
        raise ValueError(f"Unexpected Adult schema: {list(frame.columns)}")
    if frame[TARGET].isna().any():
        raise ValueError("Target contains missing values")
    target_values = set(frame[TARGET].dropna().astype(str).unique())
    if not target_values.issubset(EXPECTED_TARGETS):
        raise ValueError(f"Unexpected target labels: {sorted(target_values - EXPECTED_TARGETS)}")
    split_values = set(frame["split"].dropna().astype(str).unique())
    if split_values != EXPECTED_SPLITS:
        raise ValueError(f"Unexpected split labels: {sorted(split_values)}")


def sha256(path: str | Path) -> str:
    """Calcula o fingerprint SHA-256 de um arquivo."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quality_profile(frame: pd.DataFrame) -> dict:
    """Produz métricas estruturais e de qualidade para o contrato inicial."""
    missing = frame.isna().sum().sort_values(ascending=False)
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "feature_columns": COLUMNS[:-1],
        "target": TARGET,
        "audit_columns": AUDIT_COLUMNS,
        "duplicates": int(frame.duplicated().sum()),
        "duplicates_without_split": int(frame.drop(columns=["split"]).duplicated().sum()),
        "missing_by_column": {key: int(value) for key, value in missing.items() if value},
        "target_counts": {str(key): int(value) for key, value in frame[TARGET].value_counts(dropna=False).items()},
        "split_counts": {str(key): int(value) for key, value in frame["split"].value_counts(dropna=False).items()},
        "audit_cardinality": {column: int(frame[column].nunique(dropna=True)) for column in AUDIT_COLUMNS},
        "duplicate_policy": "source rows retained; exact predictor-signature overlap is diagnosed; no silent deduplication",
        "duplicate_policy_status": "adopted",
        "duplicate_identity": "exact_model_predictor_signature_with_missing_marker; target_excluded",
    }


def write_quality_outputs(frame: pd.DataFrame, raw_zip: str | Path, output_dir: str | Path) -> dict:
    """Salva o resumo do contrato, distribuições e tabelas de auditoria."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = quality_profile(frame)
    profile["raw_zip_sha256"] = sha256(raw_zip)
    profile["dataset_source"] = "UCI Machine Learning Repository, Adult, dataset id 2"
    profile["dataset_doi"] = "10.24432/C5XW20"
    (output_dir / "data_contract.json").write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")

    frame.describe(include="all").transpose().to_csv(output_dir / "descriptive_profile.csv", encoding="utf-8")
    frame.groupby(["sex", TARGET], dropna=False).size().rename("count").reset_index().to_csv(output_dir / "target_by_sex.csv", index=False, encoding="utf-8")
    frame.groupby(["race", TARGET], dropna=False).size().rename("count").reset_index().to_csv(output_dir / "target_by_race.csv", index=False, encoding="utf-8")
    return profile
