"""Chave de perfil, coincidencia de perfis e particao agrupada (etapa 2).

A chave representa o conjunto de **atributos observados** das oito
covariaveis candidatas do modelo principal. A identidade de apolice,
contrato ou condutor e indeterminada pela fonte. Nada aqui usa ``ClaimNb``.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

PROFILE_COLUMNS = ["Area", "VehPower", "VehAge", "DrivAge", "VehBrand",
                   "VehGas", "Density", "Region"]
MISSING_TOKEN = "__MISSING__"
SEPARATOR = "|"
GROUP_PREFIX = "freMTPL2freq-v1-etapa2-grupo|"
GROUP_MODULUS = 10_000
# Mesma razao 70:15 do desenvolvimento (7000/8500 = 82,35%).
GROUP_TRAIN_END = 8_235


def _encode_column(series: pd.Series) -> pd.Series:
    """Codifique uma coluna de forma estavel e sem ambiguidade.

    Numericos integrais viram inteiros em texto; nao integrais usam
    ``repr``; categoricos usam o texto do nivel; ausentes viram
    ``MISSING_TOKEN``.
    """
    if isinstance(series.dtype, pd.CategoricalDtype) or (
            series.dtype == object) or pd.api.types.is_string_dtype(series):
        text = series.astype("object").where(series.notna(), MISSING_TOKEN)
        return text.astype(str)
    values = series.to_numpy(dtype="float64")
    out = np.full(len(values), MISSING_TOKEN, dtype=object)
    finite = np.isfinite(values)
    integral = finite & (values == np.floor(values))
    out[integral] = values[integral].astype("int64").astype(str)
    other = finite & ~integral
    out[other] = [repr(float(v)) for v in values[other]]
    return pd.Series(out, index=series.index)


def profile_keys(frame: pd.DataFrame,
                 columns: list[str] | None = None) -> pd.Series:
    """Construa a chave de perfil de cada linha.

    Args:
        frame: tabela com as covariaveis.
        columns: colunas da chave (padrao: as oito candidatas).

    Returns:
        Serie de texto ``col1|col2|...`` na ordem de ``columns``.
    """
    columns = columns or PROFILE_COLUMNS
    parts = [_encode_column(frame[c]) for c in columns]
    key = parts[0]
    for part in parts[1:]:
        key = key + SEPARATOR + part
    return key


def group_bucket(key: str) -> int:
    """Bucket 0-9999 de uma chave de perfil (SHA-256 com prefixo fixo)."""
    digest = hashlib.sha256(
        f"{GROUP_PREFIX}{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % GROUP_MODULUS


def grouped_assignments(frame: pd.DataFrame,
                        dev_mask: pd.Series) -> pd.DataFrame:
    """Particione o desenvolvimento por chave de perfil.

    Linhas com a mesma chave ficam sempre na mesma particao agrupada
    (``group_train`` ou ``group_validation``). Linhas de teste nunca
    entram: ``dev_mask`` seleciona so treino e validacao.

    Args:
        frame: tabela com ``IDpol`` e as covariaveis.
        dev_mask: mascara booleana do conjunto de desenvolvimento.

    Returns:
        Tabela ``IDpol``, ``grouped_partition`` ordenada por ``IDpol``.
    """
    dev = frame.loc[dev_mask]
    keys = profile_keys(dev)
    unique = keys.unique()
    lookup = {k: ("group_train" if group_bucket(k) < GROUP_TRAIN_END
                  else "group_validation") for k in unique}
    out = pd.DataFrame({"IDpol": dev["IDpol"].astype("int64").to_numpy(),
                        "grouped_partition": keys.map(lookup).to_numpy()})
    return out.sort_values("IDpol", kind="stable").reset_index(drop=True)


def profile_coincidence(train: pd.DataFrame,
                        validation: pd.DataFrame) -> dict:
    """Quantifique a coincidencia de perfis entre treino e validacao.

    Usa apenas as chaves e a exposicao de validacao (nenhum desfecho).

    Args:
        train: linhas de treino.
        validation: linhas de validacao.

    Returns:
        Resumo escalar para JSON.
    """
    k_train = profile_keys(train)
    k_val = profile_keys(validation)
    train_set = set(k_train.unique())
    in_train = k_val.isin(train_set)
    val_unique = set(k_val.unique())
    exposure = validation["Exposure"].to_numpy(dtype="float64")
    train_sizes = k_train.value_counts()
    return {
        "key_columns": PROFILE_COLUMNS,
        "missing_policy": f"ausente codificado como {MISSING_TOKEN}",
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "train_distinct_profiles": int(len(train_set)),
        "validation_distinct_profiles": int(len(val_unique)),
        "validation_rows_with_profile_in_train": int(in_train.sum()),
        "validation_rows_with_profile_in_train_share": float(
            in_train.mean()),
        "validation_exposure_with_profile_in_train_share": float(
            exposure[in_train.to_numpy()].sum() / exposure.sum()),
        "validation_distinct_profiles_in_train": int(
            len(val_unique & train_set)),
        "validation_distinct_profiles_in_train_share": float(
            len(val_unique & train_set) / len(val_unique)),
        "train_rows_in_profiles_shared_within_train": int(
            (train_sizes[train_sizes > 1]).sum()),
        "train_rows_in_profiles_shared_within_train_share": float(
            train_sizes[train_sizes > 1].sum() / len(train)),
        "interpretation": "coincidencia de atributos observados; a "
                          "identidade de apolice ou condutor e "
                          "indeterminada",
    }


def grouped_summary(frame: pd.DataFrame,
                    assignments: pd.DataFrame) -> dict:
    """Resuma a particao agrupada do desenvolvimento e verifique-a.

    Args:
        frame: tabela completa (so ``IDpol`` e covariaveis sao usadas).
        assignments: saida de :func:`grouped_assignments`.

    Returns:
        Tamanhos, sobreposicao de chaves (deve ser 0) e hash.
    """
    merged = assignments.merge(frame[["IDpol"] + PROFILE_COLUMNS],
                               on="IDpol", how="left")
    merged["key"] = profile_keys(merged)
    sets = {p: set(merged.loc[merged["grouped_partition"] == p, "key"])
            for p in ("group_train", "group_validation")}
    counts = assignments["grouped_partition"].value_counts()
    payload = assignments.to_csv(index=False, lineterminator="\n").encode()
    return {
        "dev_rows": int(len(assignments)),
        "group_train_rows": int(counts.get("group_train", 0)),
        "group_validation_rows": int(counts.get("group_validation", 0)),
        "group_train_share": float(counts.get("group_train", 0)
                                   / len(assignments)),
        "key_overlap_between_groups": int(
            len(sets["group_train"] & sets["group_validation"])),
        "distinct_keys": int(merged["key"].nunique()),
        "assignment_sha256": hashlib.sha256(payload).hexdigest(),
        "rule": f"bucket(sha256('{GROUP_PREFIX}'+key)[:8]) % "
                f"{GROUP_MODULUS} < {GROUP_TRAIN_END} -> group_train",
    }
