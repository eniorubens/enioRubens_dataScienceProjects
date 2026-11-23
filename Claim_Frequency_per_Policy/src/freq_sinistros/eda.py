"""EDA descritiva restrita ao treino (etapa 3).

Nenhuma funcao recebe validacao ou teste. As associacoes sao descritivas:
taxas por 100 apolices-ano calculadas como ``100 * sum(ClaimNb) /
sum(Exposure)`` no grupo, com denominadores explicitos e sinalizacao de
baixo suporte.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from freq_sinistros import audit, models, prep, segments
from freq_sinistros.audit import safe_ratio
from freq_sinistros.protocol import nested_train_sample

DRIVAGE_LABELS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65-74",
                  "75+"]
DRIVAGE_LOWER = [18, 25, 35, 45, 55, 65, 75]
DRIVAGE_INVALID = "idade_ausente_ou_invalida"
REFERENCE_MIN_EXPOSURE = 0.75
SHORT_LIMIT = 0.08
HIST_STEP = 0.05


def driver_age_band(age: pd.Series) -> pd.Series:
    """Faixas de idade do condutor (limite inferior inclusivo)."""
    v = age.to_numpy(dtype="float64")
    valid = np.isfinite(v) & (v >= DRIVAGE_LOWER[0])
    idx = np.searchsorted(DRIVAGE_LOWER, np.where(valid, v, 18.0),
                          side="right") - 1
    labels = np.array(DRIVAGE_LABELS, dtype=object)[idx]
    return pd.Series(np.where(valid, labels, DRIVAGE_INVALID),
                     index=age.index)


def rate_table(frame: pd.DataFrame, segment: pd.Series,
               order: list[str] | None = None) -> pd.DataFrame:
    """Tabela de frequencia por segmento, com suporte sinalizado.

    Args:
        frame: linhas de treino.
        segment: rotulo do segmento de cada linha.
        order: ordem desejada dos segmentos (os demais vao ao final).

    Returns:
        Registros, exposicao, sinistros, zeros, taxa por 100 apolices-ano
        (``100 * claims / exposure``) e ``low_support``.
    """
    data = pd.DataFrame({"segment": segment.to_numpy(),
                         "y": frame["ClaimNb"].to_numpy(dtype="float64"),
                         "e": frame["Exposure"].to_numpy(dtype="float64")})
    names = list(order or []) + sorted(set(data["segment"])
                                       - set(order or []))
    rows = []
    for name in names:
        part = data[data["segment"] == name]
        rows.append({
            "segment": name, "rows": int(len(part)),
            "exposure": float(part["e"].sum()),
            "claims": float(part["y"].sum()),
            "zeros": int((part["y"] == 0).sum()),
            "zero_share": safe_ratio((part["y"] == 0).sum(), len(part)),
            "rate_per_100": safe_ratio(100 * part["y"].sum(),
                                       part["e"].sum()),
            "low_support": segments.low_support(len(part),
                                                part["e"].sum()),
        })
    return pd.DataFrame(rows)


def band_rate_table(bands: pd.DataFrame) -> pd.DataFrame:
    """Adapte a tabela de faixas de exposicao ao formato de ``rate_table``.

    Remove as linhas ``total`` e de exposicao invalida e recalcula o
    sinal de baixo suporte pelos limites aprovados.
    """
    keep = bands[~bands["band"].isin(["total", audit.INVALID_BAND])]
    out = pd.DataFrame({
        "segment": keep["band"].to_numpy(),
        "rows": keep["rows"].to_numpy(),
        "exposure": keep["exposure_valid"].to_numpy(),
        "claims": keep["claims_claim_valid"].to_numpy(),
        "rate_per_100": keep["rate_per_100_policy_years"].to_numpy()})
    out["low_support"] = [segments.low_support(r, e) for r, e in zip(
        out["rows"], out["exposure"], strict=True)]
    return out


def exposure_histogram(train: pd.DataFrame) -> pd.DataFrame:
    """Contagem de linhas por faixa de 0,05 ano, mais o valor ``= 1``."""
    e = train["Exposure"].to_numpy(dtype="float64")
    exactly_one = e == 1.0
    edges = np.arange(0.0, 2.0 + HIST_STEP, HIST_STEP)
    counts, _ = np.histogram(e[~exactly_one], bins=edges)
    out = pd.DataFrame({"lower": edges[:-1], "upper": edges[1:],
                        "rows": counts})
    out["kind"] = "interval (lower, upper]"
    extra = pd.DataFrame({"lower": [1.0], "upper": [1.0],
                          "rows": [int(exactly_one.sum())],
                          "kind": "exactly 1"})
    return pd.concat([out, extra], ignore_index=True)


def extreme_cases(train: pd.DataFrame) -> dict:
    """Casos extremos do treino (somente contagens agregadas e a cauda)."""
    claims = train["ClaimNb"]
    tail = train[claims > 4][["ClaimNb", "Exposure", "VehAge", "DrivAge"]]
    return {
        "claimnb_gt_4_rows": int(len(tail)),
        "claimnb_gt_4_claims": float(tail["ClaimNb"].sum()),
        "claimnb_gt_4_table": tail.sort_values(
            ["ClaimNb", "Exposure"], ascending=False).reset_index(
            drop=True).to_dict(orient="records"),
        "exposure_gt_1_rows": int((train["Exposure"] > 1).sum()),
        "exposure_le_0.02_rows": int((train["Exposure"] <= 0.02).sum()),
        "drivage_ge_100_rows": int((train["DrivAge"] >= 100).sum()),
        "vehage_ge_100_rows": int((train["VehAge"] >= 100).sum()),
        "bonusmalus_eq_50_share": float((train["BonusMalus"] == 50).mean()),
    }


def short_exposure_check(train: pd.DataFrame) -> tuple[pd.DataFrame,
                                                       pd.DataFrame, dict]:
    """Verifique se a taxa alta das exposicoes curtas persiste em perfis.

    Estratos: faixa de idade do condutor x faixa de idade do veiculo (35).
    A taxa de referencia de cada estrato usa as linhas com
    ``Exposure >= 0,75`` e so vale com suporte (>= 500 linhas e >= 25
    apolices-ano de referencia). Para cada faixa de exposicao: sinistros
    observados contra esperados ``sum(Exposure_i * taxa_ref(estrato_i))`` nas
    linhas de estratos com suporte; razao O/E.

    Args:
        train: linhas de treino.

    Returns:
        Tabela por faixa de exposicao, tabela por estrato (grupo curto
        ``<= 0,08`` contra referencia) e resumo.
    """
    stratum = (driver_age_band(train["DrivAge"]) + "|"
               + segments.vehicle_age_band(train["VehAge"]))
    e = train["Exposure"].to_numpy(dtype="float64")
    y = train["ClaimNb"].to_numpy(dtype="float64")
    df = pd.DataFrame({"stratum": stratum.to_numpy(), "e": e, "y": y,
                       "band": audit.assign_band(train["Exposure"])
                       .to_numpy()})
    ref = df[df["e"] >= REFERENCE_MIN_EXPOSURE]
    g = ref.groupby("stratum").agg(rows=("y", "size"), y=("y", "sum"),
                                   e=("e", "sum"))
    g["rate"] = g["y"] / g["e"]
    g["supported"] = (g["rows"] >= segments.LOW_SUPPORT_ROWS) & (
        g["e"] >= segments.LOW_SUPPORT_EXPOSURE)
    rate = g["rate"].where(g["supported"])
    df["ref_rate"] = df["stratum"].map(rate)
    used = df[df["ref_rate"].notna()].copy()
    used["expected"] = used["e"] * used["ref_rate"]
    rows = []
    for band in audit.BAND_LABELS:
        part = used[used["band"] == band]
        obs, exp = part["y"].sum(), part["expected"].sum()
        rows.append({
            "band": band, "rows": int(len(part)),
            "exposure": float(part["e"].sum()), "observed": float(obs),
            "expected_ref_rate": float(exp),
            "ratio_observed_expected": safe_ratio(obs, exp),
            "low_support": segments.low_support(len(part),
                                                part["e"].sum()),
        })
    bands = pd.DataFrame(rows)
    short = df[df["e"] <= SHORT_LIMIT].groupby("stratum").agg(
        short_rows=("y", "size"), short_claims=("y", "sum"),
        short_exposure=("e", "sum"))
    strata = short.join(g[["rows", "e", "rate", "supported"]].rename(
        columns={"rows": "ref_rows", "e": "ref_exposure",
                 "rate": "ref_rate"}), how="left")
    strata["short_rate"] = strata["short_claims"] / strata["short_exposure"]
    strata["short_supported"] = (
        (strata["short_rows"] >= segments.LOW_SUPPORT_ROWS)
        & (strata["short_exposure"] >= segments.LOW_SUPPORT_EXPOSURE))
    strata["both_supported"] = strata["short_supported"] & strata[
        "supported"].fillna(False).astype(bool)
    strata["ratio_short_over_ref"] = strata["short_rate"] / strata[
        "ref_rate"]
    strata = strata.reset_index().rename(columns={"index": "stratum"})
    ok = strata[strata["both_supported"]]
    summary = {
        "strata_total": int(len(g)),
        "strata_with_reference_support": int(g["supported"].sum()),
        "short_limit_exposure_le": SHORT_LIMIT,
        "reference_exposure_ge": REFERENCE_MIN_EXPOSURE,
        "strata_supported_in_both": int(len(ok)),
        "strata_short_rate_above_reference": int(
            (ok["ratio_short_over_ref"] > 1).sum()),
        "median_ratio_short_over_ref": float(
            ok["ratio_short_over_ref"].median()) if len(ok) else None,
        "min_ratio_short_over_ref": float(
            ok["ratio_short_over_ref"].min()) if len(ok) else None,
        "pooled_short_ratio_observed_expected": safe_ratio(
            used[used["e"] <= SHORT_LIMIT]["y"].sum(),
            used[used["e"] <= SHORT_LIMIT]["expected"].sum()),
    }
    return bands, strata, summary


def offset_assumption_check(train: pd.DataFrame, n: int = 100_000) -> dict:
    """Coeficiente livre de ``log(Exposure)`` (diagnostico, nao candidato).

    Ajusta um Poisson no treino (amostra aninhada de ``n`` IDs) com as oito
    covariaveis e ``log(Exposure)`` como covariavel. O offset aprovado
    pressupoe coeficiente 1: ``E[Y] proporcional a Exposure``.

    Args:
        train: linhas de treino.
        n: tamanho da amostra aninhada.

    Returns:
        Coeficiente, erro padrao ingenuo e intervalo aproximado.
    """
    ids = set(nested_train_sample(train["IDpol"], n))
    sample = train[train["IDpol"].isin(ids)]
    pre = prep.Preprocessor().fit(sample)
    matrix, _ = pre.transform(sample)
    design = np.hstack([matrix, np.log(sample["Exposure"].to_numpy())
                        .reshape(-1, 1)])
    y = sample["ClaimNb"].to_numpy(dtype="float64")
    fit = models.fit_poisson(design, y, np.ones(len(y)),
                             pre.columns + ["log_Exposure"])
    coef = float(fit.beta[-1])
    se = float(fit.extras["std_errors"][-1])
    return {"sample_rows": int(len(sample)), "converged": fit.converged,
            "coefficient_log_exposure": coef, "naive_se": se,
            "approx_95_low": coef - 1.96 * se,
            "approx_95_high": coef + 1.96 * se,
            "assumed_by_offset": 1.0}


def unknown_level_check(train: pd.DataFrame, n: int = 100_000) -> dict:
    """Niveis desconhecidos em dados de treino nao usados no ajuste.

    Ajusta o pre-processador na amostra aninhada de ``n`` IDs e transforma
    as demais linhas **de treino**, contando niveis desconhecidos e a
    presenca de ``OTHER``.
    """
    ids = set(nested_train_sample(train["IDpol"], n))
    in_sample = train["IDpol"].isin(ids)
    pre = prep.Preprocessor().fit(train[in_sample])
    matrix, report = pre.transform(train[~in_sample])
    return {
        "fit_rows": int(in_sample.sum()),
        "heldout_train_rows": int((~in_sample).sum()),
        "unknown_levels_heldout": report["unknown_levels"],
        "pooled_levels_in_fit": pre.pooled,
        "has_other": pre.has_other,
        "design_all_finite": bool(np.isfinite(matrix).all()),
    }


def compute_eda(train: pd.DataFrame) -> tuple[dict[str, pd.DataFrame],
                                              dict]:
    """Calcule todas as tabelas e resumos da EDA, so com o treino.

    Args:
        train: linhas de treino.

    Returns:
        Tabelas por nome e resumo JSON.
    """
    bands = audit.band_table(train)
    tables = {
        "eda_claimnb_distribution": audit.claimnb_distribution(train),
        "eda_exposure_bands": bands,
        "eda_exposure_histogram": exposure_histogram(train),
        "eda_rate_vehicle_age": rate_table(
            train, segments.vehicle_age_band(train["VehAge"]),
            segments.VEHAGE_LABELS),
        "eda_rate_driver_age": rate_table(
            train, driver_age_band(train["DrivAge"]), DRIVAGE_LABELS),
        "eda_rate_region": rate_table(
            train, segments.region_segment(train["Region"])),
        "eda_rate_area": rate_table(train, train["Area"].astype(str)),
    }
    short_bands, strata, short_summary = short_exposure_check(train)
    tables["eda_short_exposure_bands"] = short_bands
    tables["eda_short_exposure_strata"] = strata
    summary = {
        "train_rows": int(len(train)),
        "train_exposure": float(train["Exposure"].sum()),
        "train_claims": float(train["ClaimNb"].sum()),
        "train_zeros": int((train["ClaimNb"] == 0).sum()),
        "claimnb": audit.claimnb_summary(
            train, {"missing": 0, "non_numeric": 0}),
        "extremes": extreme_cases(train),
        "short_exposure_check": short_summary,
        "offset_check": offset_assumption_check(train),
        "unknown_level_check": unknown_level_check(train),
    }
    return tables, summary
