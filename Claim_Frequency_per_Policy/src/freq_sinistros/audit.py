"""Auditoria descritiva do freMTPL2freq (etapa 1).

Todas as funcoes de calculo sao puras: recebem a tabela lida do bruto e
devolvem agregados. Nenhuma linha e corrigida, excluida, deduplicada ou
imputada; as regras de limite (``ClaimNb = 4`` e ``Exposure = 1``) sao
aplicadas apenas de forma agregada, para quantificar o efeito.

Definicoes usadas em todo o modulo:

* ``ClaimNb`` valido: finito, inteiro e ``>= 0``.
* ``Exposure`` valida: finita e ``> 0`` (anos).
* Subconjunto de taxa: linhas com ambas validas. A taxa por 100
  apolices-ano e ``100 * sum(ClaimNb) / sum(Exposure)`` nesse subconjunto.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from freq_sinistros import __version__
from freq_sinistros.acquire import hash_file
from freq_sinistros.arff import ReadResult, read_arff
from freq_sinistros.labels import COLUMN_DICTIONARY, describe_column

CLAIM_CAP = 4
EXPOSURE_CAP = 1.0
SHORT_EXPOSURE_LIMITS = (0.02, 0.08)  # ~7 dias e ~1 mes, em anos
CLAIM_CLASSES = ["0", "1", "2", "3", "4", ">4"]
BAND_LABELS = [
    "(0, 0.02]", "(0.02, 0.08]", "(0.08, 0.25]", "(0.25, 0.50]",
    "(0.50, 0.75]", "(0.75, 1)", "= 1", "> 1",
]
INVALID_BAND = "exposicao_invalida"
CATEGORICAL_COLUMNS = ["Area", "VehBrand", "VehGas", "Region", "VehPower"]
QUANTILES = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]
SOURCE_COUNTS = {
    "openml_qualities_NumberOfInstances": 678013,
    "scikit_learn_load_mtpl2_docstring": 678013,
    "openml_description_policies": 677991,
    "casdatasets_description_policies": 677991,
    "casdatasets_dim_freMTPL2freq_rows": 677991,
}


def safe_ratio(numerator: float, denominator: float) -> float:
    """Divida com denominador nulo ou ausente devolvendo ``NaN``.

    Um numerador zero com denominador positivo resulta em zero (por
    exemplo, taxa absoluta zero quando ha exposicao e nenhum sinistro).

    Args:
        numerator: numerador.
        denominator: denominador; zero, ``NaN`` ou infinito dao ``NaN``.

    Returns:
        O quociente ou ``NaN`` se nao estiver definido.
    """
    if denominator is None or not np.isfinite(denominator) or (
            denominator == 0) or not np.isfinite(numerator):
        return float("nan")
    return float(numerator) / float(denominator)


# --------------------------------------------------------------- mascaras
def claim_valid_mask(claim: pd.Series) -> pd.Series:
    """Marca as contagens validas (finitas, inteiras e nao negativas)."""
    values = claim.to_numpy(dtype="float64")
    finite = np.isfinite(values)
    safe = np.where(finite, values, 0.0)
    ok = finite & (safe >= 0) & (safe == np.floor(safe))
    return pd.Series(ok, index=claim.index)


def exposure_valid_mask(exposure: pd.Series) -> pd.Series:
    """Marca as exposicoes validas (finitas e estritamente positivas)."""
    values = exposure.to_numpy(dtype="float64")
    return pd.Series(np.isfinite(values) & (values > 0),
                     index=exposure.index)


def assign_band(exposure: pd.Series) -> pd.Series:
    """Atribui cada exposicao valida a uma faixa descritiva fixa.

    Faixas nao sobrepostas, com limite superior fechado, exceto
    ``(0.75, 1)`` (aberto em 1), ``= 1`` e ``> 1``. Exposicoes invalidas
    recebem ``INVALID_BAND``.
    """
    values = exposure.to_numpy(dtype="float64")
    valid = np.isfinite(values) & (values > 0)
    v = np.where(valid, values, 0.5)
    conditions = [
        v <= 0.02, v <= 0.08, v <= 0.25, v <= 0.50, v <= 0.75, v < 1.0,
        v == 1.0,
    ]
    codes = np.select(conditions, BAND_LABELS[:7], default=BAND_LABELS[7])
    out = np.where(valid, codes, INVALID_BAND)
    return pd.Series(out, index=exposure.index)


def claim_class(claim: pd.Series) -> pd.Series:
    """Classifica contagens validas em 0, 1, 2, 3, 4 e >4."""
    values = claim.to_numpy(dtype="float64")
    labels = np.where(values > CLAIM_CAP, ">4",
                      np.where(np.isfinite(values), values, -1)
                      .astype("int64").astype(str))
    return pd.Series(labels, index=claim.index)


# ----------------------------------------------------------- validade
def validity_counts(series: pd.Series, diag: dict, kind: str) -> dict:
    """Conta as categorias de invalidade de uma coluna numerica.

    Args:
        series: coluna lida (``float64``; ausentes e nao numericos em NaN).
        diag: contagens do leitor (``missing`` e ``non_numeric``).
        kind: ``claim`` (inteira ``>= 0``) ou ``exposure`` (``> 0``).

    Returns:
        Dicionario com contagens mutuamente exclusivas e ``valid``.
    """
    values = series.to_numpy(dtype="float64")
    n_nan = int(np.isnan(values).sum())
    n_inf = int(np.isinf(values).sum())
    finite = np.isfinite(values)
    safe = np.where(finite, values, 0.0)
    out = {
        "rows": int(len(values)),
        "missing": int(diag["missing"]),
        "non_numeric": int(diag["non_numeric"]),
        "non_finite": n_inf + n_nan - diag["missing"] - diag["non_numeric"],
        "negative": int((finite & (safe < 0)).sum()),
    }
    if kind == "claim":
        out["fractional"] = int(
            (finite & (safe >= 0) & (safe != np.floor(safe))).sum())
        out["valid"] = int(claim_valid_mask(series).sum())
        out["zero_valid"] = int((finite & (safe == 0)).sum())
    else:
        out["zero"] = int((finite & (safe == 0)).sum())
        out["valid"] = int(exposure_valid_mask(series).sum())
    return out


# ------------------------------------------------------------ esquema
def schema_summary(
    result: ReadResult, openml_features: dict | None = None
) -> pd.DataFrame:
    """Resume esquema, tipos, faltantes, dominio e divergencias por coluna.

    Args:
        result: leitura tipada do bruto.
        openml_features: JSON de ``data/features`` do OpenML (opcional).

    Returns:
        Uma linha por coluna.
    """
    frame, diag = result.frame, result.diagnostics
    openml = {}
    if openml_features:
        for item in openml_features["data_features"]["feature"]:
            openml[item["name"]] = item
    rows = []
    for attr in result.header.attributes:
        col = frame[attr.name]
        info = describe_column(attr.name)
        numeric = attr.arff_type == "numeric"
        counts = diag["numeric"].get(attr.name, {})
        missing = counts.get("missing", 0) + diag["nominal_missing"].get(
            attr.name, 0) + diag["string_missing"].get(attr.name, 0)
        notes = []
        omd = openml.get(attr.name)
        if omd and omd["data_type"] != (
                "nominal" if attr.arff_type == "nominal" else
                attr.arff_type):
            notes.append(f"tipo OpenML {omd['data_type']} != ARFF "
                         f"{attr.arff_type}")
        if omd and attr.arff_type == "nominal":
            declared = omd.get("nominal_value", [])
            if len(declared) != len(set(declared)):
                notes.append("OpenML lista os niveis nominais em duplicata")
            if set(declared) != set(attr.levels or []):
                notes.append("niveis OpenML != ARFF")
        row = {
            "column": attr.name,
            "arff_type": attr.arff_type,
            "openml_type": omd["data_type"] if omd else "",
            "read_dtype": str(col.dtype),
            "role": info["role"],
            "unit_or_category": info["unit"],
            "rows": int(len(col)),
            "missing": int(missing),
            "non_numeric": int(counts.get("non_numeric", 0)),
            "out_of_domain": int(diag["out_of_domain"].get(attr.name, 0)),
            "distinct": int(col.nunique(dropna=True)),
            "min": float(col.min()) if numeric else None,
            "max": float(col.max()) if numeric else None,
            "mean": float(col.mean()) if numeric else None,
            "declared_levels": len(attr.levels) if attr.levels else None,
            "notes": "; ".join(notes),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def categorical_levels(frame: pd.DataFrame) -> pd.DataFrame:
    """Frequencia de niveis das colunas categoricas e de ``VehPower``."""
    parts = []
    total = len(frame)
    for name in CATEGORICAL_COLUMNS:
        if name not in frame:
            continue
        counts = frame[name].value_counts(dropna=False).sort_index()
        parts.append(pd.DataFrame({
            "column": name, "level": counts.index.astype(str),
            "rows": counts.to_numpy(), "share": counts.to_numpy() / total,
        }))
    return pd.concat(parts, ignore_index=True)


def numeric_quantiles(frame: pd.DataFrame) -> pd.DataFrame:
    """Minimo, quantis e maximo das colunas numericas."""
    rows = []
    for name in frame.columns:
        if str(frame[name].dtype) == "category":
            continue
        col = frame[name].astype("float64")
        row = {"column": name, "min": col.min()}
        for q in QUANTILES:
            row[f"p{q * 100:g}"] = col.quantile(q)
        row["max"] = col.max()
        rows.append(row)
    return pd.DataFrame(rows)


def dictionary_checks(frame: pd.DataFrame) -> dict:
    """Confronta dominios observados com o dicionario CASdatasets."""
    out = {}
    if "BonusMalus" in frame:
        bm = frame["BonusMalus"]
        out["BonusMalus_outside_50_350"] = int(((bm < 50) | (bm > 350))
                                               .sum())
        out["BonusMalus_min"] = float(bm.min())
        out["BonusMalus_max"] = float(bm.max())
        out["BonusMalus_eq_50"] = int((bm == 50).sum())
    if "DrivAge" in frame:
        out["DrivAge_below_18"] = int((frame["DrivAge"] < 18).sum())
        out["DrivAge_min"] = float(frame["DrivAge"].min())
        out["DrivAge_max"] = float(frame["DrivAge"].max())
        out["DrivAge_ge_100"] = int((frame["DrivAge"] >= 100).sum())
    if "VehAge" in frame:
        out["VehAge_negative"] = int((frame["VehAge"] < 0).sum())
        out["VehAge_gt_30"] = int((frame["VehAge"] > 30).sum())
        out["VehAge_max"] = float(frame["VehAge"].max())
        out["VehAge_ge_100"] = int((frame["VehAge"] >= 100).sum())
    if "Density" in frame:
        out["Density_nonpositive"] = int((frame["Density"] <= 0).sum())
    return out


# ------------------------------------------------- identificador/unidade
def id_audit(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Audita ``IDpol``: unicidade, repeticoes, duplicatas e conflitos.

    Returns:
        Resumo escalar e a tabela das linhas com ID repetido.
    """
    ids = frame["IDpol"]
    counts = ids.value_counts()
    repeated = counts[counts > 1]
    in_repeated = ids.isin(repeated.index)
    full_dup = frame.duplicated(keep=False)
    conflict_groups = 0
    if len(repeated):
        sub = frame[in_repeated]
        nun = sub.drop(columns="IDpol").astype(str).groupby(
            sub["IDpol"]).nunique(dropna=False)
        conflict_groups = int((nun.max(axis=1) > 1).sum())
    summary = {
        "rows": int(len(frame)),
        "ids_present": int(ids.notna().sum()),
        "ids_distinct": int(ids.nunique()),
        "ids_repeated": int(len(repeated)),
        "rows_in_repeated_ids": int(in_repeated.sum()),
        "rows_excess_over_distinct": int(len(frame) - ids.nunique()),
        "fully_duplicated_rows_incl_id": int(full_dup.sum()),
        "repeated_id_groups_with_conflicting_values": conflict_groups,
        "id_min": int(ids.min()), "id_max": int(ids.max()),
    }
    return summary, frame[in_repeated].sort_values("IDpol")


def profile_groups(frame: pd.DataFrame) -> dict:
    """Quantifica grupos de atributos observados coincidentes.

    Conta registros com ``IDpol`` distintos que compartilham todos os
    valores das covariaveis. O resultado descreve a coincidencia de
    atributos; a identidade de apolice, contrato ou condutor e
    indeterminada pela fonte.
    """
    covariates = [c for c in frame.columns
                  if c not in ("IDpol", "ClaimNb", "Exposure")]
    key = frame.groupby(covariates, observed=True, sort=False).ngroup()
    size = key.map(key.value_counts())
    exp_sum = frame["Exposure"].groupby(key).transform("sum")
    multi = size > 1
    sizes = key.value_counts()
    return {
        "covariates_used": covariates,
        "distinct_profiles": int(sizes.size),
        "profiles_with_more_than_one_row": int((sizes > 1).sum()),
        "rows_in_shared_profiles": int(multi.sum()),
        "share_rows_in_shared_profiles": float(multi.mean()),
        "rows_in_profiles_with_exposure_sum_gt_1": int(
            (multi & (exp_sum > 1.0 + 1e-9)).sum()),
        "max_profile_size": int(sizes.max()),
        "profile_size_distribution": {
            str(k): int(v) for k, v in
            sizes.value_counts().sort_index().head(15).items()},
    }


# ------------------------------------------------------------- ClaimNb
def claimnb_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    """Distribuicao da contagem valida: linhas, sinistros e proporcoes."""
    valid = claim_valid_mask(frame["ClaimNb"])
    claims = frame.loc[valid, "ClaimNb"].astype("int64")
    grouped = claims.value_counts().sort_index()
    n_valid = int(valid.sum())
    total = int(claims.sum())
    out = pd.DataFrame({"claimnb": grouped.index.astype(int),
                        "rows": grouped.to_numpy()})
    out["share_of_valid_rows"] = out["rows"] / n_valid
    out["claims"] = out["claimnb"] * out["rows"]
    out["share_of_claims"] = out["claims"] / total if total else np.nan
    return out


def claimnb_summary(frame: pd.DataFrame, diag: dict) -> dict:
    """Resumo da contagem: invalidos, momentos e cauda.

    Media e variancia usam as linhas validas como denominador; a variancia
    e reportada com ``ddof=1`` (amostral) e ``ddof=0`` (populacional).
    """
    series = frame["ClaimNb"]
    counts = validity_counts(series, diag, "claim")
    valid = claim_valid_mask(series)
    y = series[valid].to_numpy(dtype="float64")
    n = int(y.size)
    total = float(y.sum())
    mean = total / n if n else float("nan")
    var1 = float(np.var(y, ddof=1)) if n > 1 else float("nan")
    var0 = float(np.var(y, ddof=0)) if n else float("nan")
    tail = y > CLAIM_CAP
    return {
        "validity": counts,
        "valid_rows": n,
        "zeros": int((y == 0).sum()),
        "zeros_share": float((y == 0).mean()) if n else float("nan"),
        "positives": int((y > 0).sum()),
        "positives_share": float((y > 0).mean()) if n else float("nan"),
        "total_claims": total,
        "mean": mean,
        "variance_ddof1": var1,
        "variance_ddof0": var0,
        "variance_to_mean_ratio_ddof1": var1 / mean if mean else None,
        "max": float(y.max()) if n else None,
        "denominator": "linhas com ClaimNb valido",
        "rows_gt_4": int(tail.sum()),
        "rows_gt_4_share_of_valid": float(tail.mean()) if n else None,
        "claims_in_rows_gt_4": float(y[tail].sum()),
        "claims_in_rows_gt_4_share": float(y[tail].sum() / total)
        if total else None,
        "claims_excess_over_4": float(np.maximum(y - CLAIM_CAP, 0).sum()),
    }


# ------------------------------------------------------------ Exposure
def exposure_summary(frame: pd.DataFrame, diag: dict) -> dict:
    """Resumo da exposicao: invalidos, quantis, caudas e totais."""
    series = frame["Exposure"]
    counts = validity_counts(series, diag, "exposure")
    valid = exposure_valid_mask(series)
    claim_ok = claim_valid_mask(frame["ClaimNb"])
    e = series[valid].to_numpy(dtype="float64")
    out = {
        "validity": counts,
        "valid_rows": int(e.size),
        "sum_positive_finite": float(e.sum()),
        "min_positive": float(e.min()) if e.size else None,
        "max": float(e.max()) if e.size else None,
        "quantiles": {f"p{q * 100:g}": float(np.quantile(e, q))
                      for q in QUANTILES} if e.size else {},
        "unit": "anos (dicionario CASdatasets)",
        "thresholds_declared": {
            "short_le": list(SHORT_EXPOSURE_LIMITS),
            "long_gt": EXPOSURE_CAP,
        },
    }

    def tail(mask: pd.Series) -> dict:
        both = mask & valid & claim_ok
        claims = frame.loc[both, "ClaimNb"]
        return {
            "rows": int((mask & valid).sum()),
            "exposure": float(series[mask & valid].sum()),
            "rows_claim_valid": int(both.sum()),
            "claims": float(claims.sum()),
            "zeros": int((claims == 0).sum()),
            "positives": int((claims > 0).sum()),
        }

    for limit in SHORT_EXPOSURE_LIMITS:
        out[f"short_le_{limit}"] = tail(series <= limit)
    out["long_gt_1"] = tail(series > EXPOSURE_CAP)
    out["equal_1"] = tail(series == EXPOSURE_CAP)
    invalid = ~valid
    out["invalid_rows_detail"] = {
        "rows": int(invalid.sum()),
        "exposure_defined_sum": float(
            series[invalid & np.isfinite(series)].sum()),
        "claims_in_rows_with_valid_claimnb": float(
            frame.loc[invalid & claim_ok, "ClaimNb"].sum()),
    }
    return out


def band_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Tabela por faixa de exposicao, com linha de exposicao invalida.

    Cada linha do bruto cai em exatamente uma faixa. As colunas
    ``*_rate_subset`` valem apenas para linhas com ``ClaimNb`` e
    ``Exposure`` validos; a taxa e ``100 * claims / exposure``.
    """
    band = assign_band(frame["Exposure"])
    claim_ok = claim_valid_mask(frame["ClaimNb"])
    exp_ok = exposure_valid_mask(frame["Exposure"])
    subset = claim_ok & exp_ok
    rows = []
    for label in BAND_LABELS + [INVALID_BAND]:
        in_band = band == label
        sub = in_band & subset
        ok_claims = frame.loc[in_band & claim_ok, "ClaimNb"]
        sub_claims = frame.loc[sub, "ClaimNb"]
        exposure_sub = float(frame.loc[sub, "Exposure"].sum())
        claims_sub = float(sub_claims.sum())
        rows.append({
            "band": label,
            "rows": int(in_band.sum()),
            "rows_claim_valid": int((in_band & claim_ok).sum()),
            "claims_claim_valid": float(ok_claims.sum()),
            "exposure_valid": float(frame.loc[in_band & exp_ok,
                                              "Exposure"].sum()),
            "rows_rate_subset": int(sub.sum()),
            "exposure_rate_subset": exposure_sub,
            "claims_rate_subset": claims_sub,
            "zeros_rate_subset": int((sub_claims == 0).sum()),
            "zero_share": float((sub_claims == 0).mean())
            if sub.any() else np.nan,
            "rate_per_100_policy_years": safe_ratio(
                100 * claims_sub, exposure_sub),
        })
    table = pd.DataFrame(rows)
    total = {c: table[c].sum() for c in table.columns if c not in (
        "band", "zero_share", "rate_per_100_policy_years")}
    total["band"] = "total"
    total["zero_share"] = (total["zeros_rate_subset"]
                           / total["rows_rate_subset"]
                           if total["rows_rate_subset"] else np.nan)
    total["rate_per_100_policy_years"] = safe_ratio(
        100 * total["claims_rate_subset"], total["exposure_rate_subset"])
    return pd.concat([table, pd.DataFrame([total])], ignore_index=True)


def exposure_by_claim_class(frame: pd.DataFrame) -> pd.DataFrame:
    """Cruza faixa de exposicao e classe de ``ClaimNb`` (subconjunto)."""
    subset = claim_valid_mask(frame["ClaimNb"]) & exposure_valid_mask(
        frame["Exposure"])
    sub = frame[subset]
    cross = pd.DataFrame({
        "band": assign_band(sub["Exposure"]),
        "claim_class": claim_class(sub["ClaimNb"]),
        "claims": sub["ClaimNb"], "exposure": sub["Exposure"],
    })
    grouped = cross.groupby(["band", "claim_class"]).agg(
        rows=("claims", "size"), claims=("claims", "sum"),
        exposure=("exposure", "sum")).reset_index()
    full = pd.MultiIndex.from_product(
        [BAND_LABELS, CLAIM_CLASSES], names=["band", "claim_class"])
    return (grouped.set_index(["band", "claim_class"])
            .reindex(full, fill_value=0).reset_index())


def clip_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    """Efeito agregado dos limites didaticos ``ClaimNb<=4``, ``Exposure<=1``.

    Regra: ``ClaimNb' = min(ClaimNb, 4)`` e ``Exposure' = min(Exposure, 1)``,
    sem remover linhas. Os totais ajustados sao os brutos menos as
    excedencias ``sum(max(ClaimNb - 4, 0))`` e ``sum(max(Exposure - 1, 0))``,
    no subconjunto de taxa. O bruto nao e alterado.
    """
    subset = claim_valid_mask(frame["ClaimNb"]) & exposure_valid_mask(
        frame["Exposure"])
    c = frame.loc[subset, "ClaimNb"].to_numpy(dtype="float64")
    e = frame.loc[subset, "Exposure"].to_numpy(dtype="float64")
    claims, expo = float(c.sum()), float(e.sum())
    ex_c = float(np.maximum(c - CLAIM_CAP, 0).sum())
    ex_e = float(np.maximum(e - EXPOSURE_CAP, 0).sum())
    rows_c = int((c > CLAIM_CAP).sum())
    rows_e = int((e > EXPOSURE_CAP).sum())
    scenarios = [
        ("bruto", 0, claims, expo),
        ("ClaimNb<=4", rows_c, claims - ex_c, expo),
        ("Exposure<=1", rows_e, claims, expo - ex_e),
        ("ClaimNb<=4 e Exposure<=1", int(((c > CLAIM_CAP)
                                          | (e > EXPOSURE_CAP)).sum()),
         claims - ex_c, expo - ex_e),
    ]
    base_rate = safe_ratio(100 * claims, expo)
    rows = []
    for name, affected, cl, ex in scenarios:
        rate = safe_ratio(100 * cl, ex)
        rows.append({
            "scenario": name, "rows_in_scope": int(subset.sum()),
            "rows_affected": affected, "total_claims": cl,
            "total_exposure": ex,
            "rate_per_100_policy_years": rate,
            "claims_excess_removed": claims - cl,
            "exposure_excess_removed": expo - ex,
            "claims_change_pct": safe_ratio(100 * (cl - claims), claims),
            "exposure_change_pct": safe_ratio(100 * (ex - expo), expo),
            "rate_change_pct": safe_ratio(100 * (rate - base_rate),
                                          base_rate),
        })
    return pd.DataFrame(rows)


def source_count_comparison(n_rows: int, n_ids: int) -> pd.DataFrame:
    """Compara a contagem observada com as contagens citadas."""
    rows = [{"source": "arquivo_baixado_linhas", "count": n_rows,
             "diff_vs_observed_rows": 0},
            {"source": "arquivo_baixado_ids_distintos", "count": n_ids,
             "diff_vs_observed_rows": n_ids - n_rows}]
    for name, count in SOURCE_COUNTS.items():
        rows.append({"source": name, "count": count,
                     "diff_vs_observed_rows": count - n_rows})
    return pd.DataFrame(rows)


def reconciliation(frame: pd.DataFrame, bands: pd.DataFrame) -> pd.DataFrame:
    """Reconcilia totais da tabela de faixas com o resumo geral do bruto."""
    claim_ok = claim_valid_mask(frame["ClaimNb"])
    exp_ok = exposure_valid_mask(frame["Exposure"])
    subset = claim_ok & exp_ok
    total = bands[bands["band"] == "total"].iloc[0]
    checks = [
        ("linhas_brutas", len(frame), total["rows"]),
        ("linhas_claimnb_valido", int(claim_ok.sum()),
         total["rows_claim_valid"]),
        ("sinistros_claimnb_valido",
         float(frame.loc[claim_ok, "ClaimNb"].sum()),
         total["claims_claim_valid"]),
        ("exposicao_valida", float(frame.loc[exp_ok, "Exposure"].sum()),
         total["exposure_valid"]),
        ("linhas_subconjunto_taxa", int(subset.sum()),
         total["rows_rate_subset"]),
        ("sinistros_subconjunto_taxa",
         float(frame.loc[subset, "ClaimNb"].sum()),
         total["claims_rate_subset"]),
        ("exposicao_subconjunto_taxa",
         float(frame.loc[subset, "Exposure"].sum()),
         total["exposure_rate_subset"]),
        ("linhas_fora_subconjunto", int((~subset).sum()),
         total["rows"] - total["rows_rate_subset"]),
    ]
    return pd.DataFrame([{
        "item": name, "raw_summary": float(raw), "band_table": float(tab),
        "abs_diff": abs(float(raw) - float(tab)),
        "ok": bool(np.isclose(raw, tab, rtol=0, atol=1e-6)),
    } for name, raw, tab in checks])


def covariate_contract(frame: pd.DataFrame) -> pd.DataFrame:
    """Tabela papel/unidade/disponibilidade por coluna, em PT-BR e EN-US."""
    rows = []
    for name in frame.columns:
        pt = describe_column(name, "pt_br")
        en = describe_column(name, "en_us")
        rows.append({
            "column": name,
            "role_pt": pt["role"], "unit_pt": pt["unit"],
            "availability_pt": pt["availability"],
            "role_en": en["role"], "unit_en": en["unit"],
            "availability_en": en["availability"],
            "source": "CASdatasets freMTPL (dicionario); OpenML 41214",
        })
    return pd.DataFrame(rows)


def structure_checks(columns: list[str]) -> dict:
    """Procura datas por apolice ou identificador de unidade superior."""
    keys = ("date", "data", "year", "ano", "start", "end", "inicio",
            "fim", "contract", "policyholder", "client", "customer",
            "holder", "group")
    hits = [c for c in columns
            if c != "IDpol" and any(k in c.lower() for k in keys)]
    return {"columns": columns, "date_or_group_like_columns": hits,
            "has_date_columns": False, "has_higher_unit_id": False}


# --------------------------------------------------------- orquestracao
def audit_frame(
    result: ReadResult, openml_features: dict | None = None
) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Executa todas as verificacoes sobre a tabela lida.

    Args:
        result: saida de :func:`freq_sinistros.arff.read_arff`.
        openml_features: JSON de features do OpenML (opcional).

    Returns:
        Resumo escalar (JSON) e dicionario de tabelas por nome de arquivo.
    """
    frame, diag = result.frame, result.diagnostics
    id_summary, id_rows = id_audit(frame)
    bands = band_table(frame)
    summary = {
        "audit_version": __version__,
        "production_eligible": False,
        "dimensions": {"rows": int(len(frame)),
                       "columns": int(frame.shape[1]),
                       "column_names": list(frame.columns)},
        "read_diagnostics": {
            k: v for k, v in diag.items()},
        "identifier": {
            **id_summary,
            "read_noncanonical_id_tokens": diag["id_noncanonical_tokens"],
            "read_id_roundtrip_mismatch": diag["id_roundtrip_mismatch"],
            "read_id_non_integer_or_unsafe":
                diag["id_non_integer_or_unsafe"],
            "read_id_dtype": str(frame["IDpol"].dtype),
            "profile_groups": profile_groups(frame),
        },
        "claimnb": claimnb_summary(frame, diag["numeric"]["ClaimNb"]),
        "exposure": exposure_summary(frame, diag["numeric"]["Exposure"]),
        "dictionary_checks": dictionary_checks(frame),
        "structure": structure_checks(list(frame.columns)),
        "rate_subset": {
            "definition": "ClaimNb valido e Exposure valida (>0, finita)",
            "rows": int((claim_valid_mask(frame["ClaimNb"])
                         & exposure_valid_mask(frame["Exposure"])).sum()),
        },
    }
    rate_row = bands[bands["band"] == "total"].iloc[0]
    summary["rate_subset"].update({
        "claims": float(rate_row["claims_rate_subset"]),
        "exposure": float(rate_row["exposure_rate_subset"]),
        "rate_per_100_policy_years": float(
            rate_row["rate_per_100_policy_years"]),
        "rows_outside_subset": int(len(frame) - rate_row["rows_rate_subset"]),
    })
    tables = {
        "schema_summary": schema_summary(result, openml_features),
        "categorical_levels": categorical_levels(frame),
        "numeric_quantiles": numeric_quantiles(frame),
        "claimnb_distribution": claimnb_distribution(frame),
        "exposure_bands": bands,
        "exposure_x_claimnb": exposure_by_claim_class(frame),
        "clip_sensitivity": clip_sensitivity(frame),
        "source_count_comparison": source_count_comparison(
            len(frame), int(frame["IDpol"].nunique())),
        "reconciliation": reconciliation(frame, bands),
        "covariate_contract": covariate_contract(frame),
        "id_repeats": id_rows,
    }
    return summary, tables


# ------------------------------------------------------------- escrita
def _jsonable(value):
    """Converta tipos numpy/pandas em tipos nativos serializaveis."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_outputs(outputs: Path, summary: dict,
                  tables: dict[str, pd.DataFrame]) -> list[Path]:
    """Grava resumo JSON e tabelas CSV de forma deterministica.

    Args:
        outputs: pasta de saida.
        summary: resumo escalar.
        tables: tabelas por nome.

    Returns:
        Caminhos dos arquivos escritos.
    """
    outputs.mkdir(parents=True, exist_ok=True)
    written = []
    path = outputs / "audit_summary.json"
    path.write_text(json.dumps(_jsonable(summary), indent=2,
                               ensure_ascii=False, sort_keys=True),
                    encoding="utf-8")
    written.append(path)
    for name, table in tables.items():
        path = outputs / f"{name}.csv"
        table.to_csv(path, index=False, encoding="utf-8",
                     lineterminator="\n", float_format="%.12g")
        written.append(path)
    return written


def _peak_mb() -> float:
    """Pico de memoria residente do processo, em MiB."""
    import psutil

    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", None) or info.rss
    return peak / 2**20


def run_audit(root: Path) -> int:
    """Audita o ARFF local, sem rede, e grava os resultados.

    Verifica tamanho e SHA-256 do bruto contra o manifesto antes de ler.
    Mede tempo de parede e pico de memoria por estagio.

    Args:
        root: raiz do projeto.

    Returns:
        0 em sucesso; 2 se o bruto nao conferir com o manifesto.
    """
    root = Path(root)
    outputs = root / "outputs" / "etapa_01"
    manifest = json.loads((outputs / "manifest.json").read_text("utf-8"))
    raw = root / manifest["artifact"]["local_path"]
    stages = []

    def stage(name, started):
        stages.append({"stage": name,
                       "wall_s": round(time.perf_counter() - started, 3),
                       "peak_rss_mib_cumulative": round(_peak_mb(), 1)})

    t0 = time.perf_counter()
    digest = hash_file(raw)
    stage("verify_hash", t0)
    if (digest["sha256"] != manifest["artifact"]["sha256"]
            or digest["size_bytes"] != manifest["artifact"]["size_bytes"]):
        print("ERRO: bruto diverge do manifesto", digest)
        return 2
    t0 = time.perf_counter()
    result = read_arff(raw)
    stage("read_arff_chunked", t0)
    features_path = outputs / "openml_metadata.json"
    features = None
    if features_path.exists():
        features = json.loads(features_path.read_text("utf-8"))["features"]
    t0 = time.perf_counter()
    summary, tables = audit_frame(result, features)
    stage("compute_audit", t0)
    t0 = time.perf_counter()
    summary["raw_file"] = {
        "path": manifest["artifact"]["local_path"],
        "size_bytes": digest["size_bytes"], "sha256": digest["sha256"],
        "manifest_match": True,
    }
    write_outputs(outputs, summary, tables)
    stage("write_outputs", t0)
    import psutil

    cost = {
        "note": "pico = peak_wset do processo de auditoria (Windows), "
                "acumulado ao fim de cada estagio; CPU, sem GPU.",
        "stages": stages,
        "total_wall_s": round(sum(s["wall_s"] for s in stages), 3),
        "peak_rss_mib": round(_peak_mb(), 1),
        "download_cost_from_manifest": {
            "wall_s": manifest["artifact"]["download_elapsed_s"],
            "bytes": manifest["artifact"]["size_bytes"],
        },
        "environment": {
            "python": sys.version, "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__,
            "psutil": psutil.__version__,
            "cpu_logical": psutil.cpu_count(),
            "ram_total_gib": round(psutil.virtual_memory().total / 2**30,
                                   1),
        },
    }
    (outputs / "cost.json").write_text(
        json.dumps(cost, indent=2), encoding="utf-8")
    print(json.dumps(cost["stages"], indent=1))
    print("peak_rss_mib:", cost["peak_rss_mib"])
    return 0


__all__ = ["audit_frame", "run_audit", "COLUMN_DICTIONARY"]
