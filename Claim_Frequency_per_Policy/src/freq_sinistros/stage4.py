"""Etapa 4: decisao de modelo, diagnostico de zeros e congelamento.

Usa so treino e validacao. Os ajustes sao refeitos de forma deterministica
com o mesmo codigo e a mesma amostra da etapa 3 e conferidos contra os
resultados aprovados; os intervalos vem de ``outputs/etapa_03``. Nenhum ZIP
ou ZINB e ajustado nesta rodada.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from freq_sinistros import (
    __version__,
    artifacts,
    audit,
    compare,
    freeze,
    models,
    prep,
    segments,
)
from freq_sinistros.audit import _peak_mb, safe_ratio
from freq_sinistros.data import load_partitions
from freq_sinistros.protocol import amendment_stage3
from freq_sinistros.stage3 import write_json, write_table

OUT3 = Path("outputs") / "etapa_03"
OUT4 = Path("outputs") / "etapa_04"
FULL = "train_full"
CANDIDATES = ("global", "poisson", "nb2", "poisson_free_exposure")
BANDS = audit.BAND_LABELS
REPRO_TOL = {"rtol": 1e-9, "atol": 1e-10}


# ----------------------------------------------------------- reajuste
def refit(train: pd.DataFrame, val: pd.DataFrame) -> dict:
    """Refaca, de forma deterministica, os ajustes do degrau ``train_full``.

    Args:
        train: treino inteiro.
        val: validacao.

    Returns:
        Pre-processador, ajustes, previsoes, o avaliador ``EvalSet`` com
        os quatro candidatos e o coeficiente livre de ``log(Exposure)``.
    """
    pre = prep.Preprocessor().fit(train)
    fits, _, matrix = compare.fit_candidates(train, pre,
                                             {"context": "stage4"})
    design = np.hstack([matrix, np.log(train["Exposure"].to_numpy())
                        .reshape(-1, 1)])
    free = models.fit_poisson(design, train["ClaimNb"].to_numpy(
        dtype="float64"), np.ones(len(train)), pre.columns + ["log_Exposure"])
    del matrix, design
    mu, report = compare.predict_all(fits, pre, val)
    mat_v, _ = pre.transform(val)
    mu["poisson_free_exposure"] = free.predict(
        np.hstack([mat_v, np.log(val["Exposure"].to_numpy())
                   .reshape(-1, 1)]), np.ones(len(val)))
    ev = compare.EvalSet(val, mu, {"nb2": fits["nb2"].alpha}, [])
    return {"pre": pre, "fits": fits, "free": free, "mu": mu,
            "report": report, "ev": ev}


def reproduction_check(ev: compare.EvalSet, free_coef: float,
                       out3: dict, coefficients: pd.DataFrame,
                       beta: np.ndarray) -> dict:
    """Confira o reajuste contra as saidas aprovadas da etapa 3."""
    met = out3["validation_metrics"]
    rows, worst = [], 0.0
    for cand in CANDIDATES:
        pop = ("validation_free_exposure_diagnostic"
               if cand == "poisson_free_exposure" else "validation")
        ref = met[(met["population"] == pop)
                  & (met["degree"].astype(str) == FULL)
                  & (met["subset"] == "all_rows")
                  & (met["candidate"] == cand)]
        if cand in ("global", "poisson", "nb2"):
            ref = ref.iloc[:1]
        ref = ref.iloc[0]
        mine = {
            "mean_poisson_deviance": float(ev.dev[cand].mean()),
            "mean_loglik": float(ev.ll[cand].mean()),
            "predicted_claims": float(ev.mu[cand].sum()),
            "zeros_predicted": float(ev.p0[cand].sum())}
        for key, value in mine.items():
            diff = abs(value - float(ref[key]))
            worst = max(worst, diff / max(abs(float(ref[key])), 1e-12))
            rows.append({"candidate": cand, "metric": key,
                         "reproduced": value, "stage3": float(ref[key]),
                         "abs_diff": diff})
    coef_diff = float(np.abs(
        coefficients["poisson_coef"].to_numpy() - beta).max())
    free_ref = float(met["free_log_exposure_coef"].dropna().iloc[0])
    return {"metrics": rows, "max_rel_diff_metrics": worst,
            "poisson_coefficients_max_abs_diff_vs_stage3_csv": coef_diff,
            "free_exposure_coefficient": {
                "reproduced": free_coef, "stage3": free_ref,
                "abs_diff": abs(free_coef - free_ref)},
            "tolerance": REPRO_TOL,
            "ok": bool(worst <= REPRO_TOL["rtol"] * 10
                       and coef_diff <= 1e-9
                       and abs(free_coef - free_ref) <= 1e-9)}


# ----------------------------------------------------------- decisao
def _interval(iv: pd.DataFrame, degree: str, population: str,
              key: str) -> tuple[float, float, float]:
    """Estimativa e limites de um indicador (NaN se nao informado)."""
    sub = iv[(iv["degree"].astype(str) == degree)
             & (iv["population"] == population)
             & (iv["indicator"] == key)]
    if sub.empty:
        return (np.nan, np.nan, np.nan)
    row = sub.iloc[0]
    if not bool(row["interval_reported"]):
        return (float(row["estimate"]), np.nan, np.nan)
    return (float(row["estimate"]), float(row["ci_low"]),
            float(row["ci_high"]))


def decision_table(out3: dict) -> pd.DataFrame:
    """Evidencia da regra de escolha por degrau e na particao agrupada.

    Cada linha traz desvios medios, diferencas pareadas com IC, erros
    absolutos de taxa global e o resultado das condicoes do parecer.
    """
    met = out3["validation_metrics"]
    met = met[met["subset"] == "all_rows"]
    iv = out3["intervals"]
    rules = artifacts.rules_table(out3).set_index(["degree", "population"])
    rows = []
    keys = [("100000", "validation"), ("200000", "validation"),
            (FULL, "validation"), (FULL, "grouped_validation")]
    for degree, pop in keys:
        sub = met[(met["degree"].astype(str) == degree)
                  & (met["population"] == pop)].set_index("candidate")
        if pop == "grouped_validation":
            sub = sub.loc[["global", "poisson", "nb2"]]
        ipop = pop
        d_pg = _interval(iv, degree, ipop, "d_dev|poisson-global")
        d_np = _interval(iv, degree, ipop, "d_ll|nb2-poisson")
        d_dnp = _interval(iv, degree, ipop, "d_dev|nb2-poisson")
        d_ng = _interval(iv, degree, ipop, "d_dev|nb2-global")
        rule = rules.loc[(degree, pop)]
        rows.append({
            "degree": degree, "population": pop,
            "dev_global": sub.loc["global", "mean_poisson_deviance"],
            "dev_poisson": sub.loc["poisson", "mean_poisson_deviance"],
            "dev_nb2": sub.loc["nb2", "mean_poisson_deviance"],
            "d_dev_poisson_minus_global": d_pg[0],
            "d_dev_poisson_minus_global_lo": d_pg[1],
            "d_dev_poisson_minus_global_hi": d_pg[2],
            "abs_rate_err_global": sub.loc["global", "abs_rate_error"],
            "abs_rate_err_poisson": sub.loc["poisson", "abs_rate_error"],
            "abs_rate_err_nb2": sub.loc["nb2", "abs_rate_error"],
            "d_ll_nb2_minus_poisson": d_np[0],
            "d_ll_nb2_minus_poisson_lo": d_np[1],
            "d_ll_nb2_minus_poisson_hi": d_np[2],
            "d_dev_nb2_minus_poisson": d_dnp[0],
            "d_dev_nb2_minus_poisson_lo": d_dnp[1],
            "d_dev_nb2_minus_poisson_hi": d_dnp[2],
            "d_dev_nb2_minus_global": d_ng[0],
            "poisson_over_global": bool(rule["poisson_over_global"]),
            "nb2_over_poisson": bool(rule["nb2_over_poisson"]),
            "preferred_by_rules": rule["preferred_by_rules"]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------- exposicao
def exposure_band_table(ev: compare.EvalSet, out3: dict) -> pd.DataFrame:
    """Taxas observada e previstas por faixa de exposicao (validacao).

    As taxas sao ``100 * soma(mu) / soma(Exposure)`` do grupo, calculadas
    das previsoes diretas. Os IC vem do bootstrap pareado da etapa 3.
    """
    iv = out3["intervals"]
    band = segments.segment_columns(ev.frame)["exposure_band"].to_numpy()
    rows = []
    for label in BANDS:
        m = band == label
        if not m.any():
            continue
        e, y = float(ev.e[m].sum()), float(ev.y[m].sum())
        row = {"band": label, "rows": int(m.sum()), "exposure": e,
               "observed_claims": y,
               "observed_rate_per_100": safe_ratio(100 * y, e),
               "low_support": segments.low_support(int(m.sum()), e)}
        for cand in CANDIDATES:
            mu = float(ev.mu[cand][m].sum())
            pop = ("validation_free_exposure_diagnostic"
                   if cand == "poisson_free_exposure" else "validation")
            est = _interval(iv, FULL, pop,
                            f"ratediff|{cand}|exposure_band|{label}")
            row[f"predicted_rate_{cand}"] = safe_ratio(100 * mu, e)
            row[f"diff_{cand}"] = safe_ratio(100 * (mu - y), e)
            row[f"diff_{cand}_lo"] = est[1]
            row[f"diff_{cand}_hi"] = est[2]
        rows.append(row)
    total = {"band": "total", "rows": int(ev.n),
             "exposure": float(ev.e.sum()),
             "observed_claims": float(ev.y.sum()),
             "observed_rate_per_100": 100 * ev.y.sum() / ev.e.sum(),
             "low_support": False}
    for cand in CANDIDATES:
        pop = ("validation_free_exposure_diagnostic"
               if cand == "poisson_free_exposure" else "validation")
        est = _interval(iv, FULL, pop, f"ratediff|{cand}|overall")
        total[f"predicted_rate_{cand}"] = (
            100 * ev.mu[cand].sum() / ev.e.sum())
        total[f"diff_{cand}"] = 100 * (
            ev.mu[cand].sum() - ev.y.sum()) / ev.e.sum()
        total[f"diff_{cand}_lo"], total[f"diff_{cand}_hi"] = est[1], est[2]
    rows.append(total)
    return pd.DataFrame(rows)


# --------------------------------------------------------------- zeros
def zero_diagnostic(ev: compare.EvalSet,
                    out3: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Zeros observados e previstos por candidato, total e faixa.

    ``P(Y_i = 0)`` e ``exp(-mu_i)`` nos candidatos Poisson (inclusive a
    taxa global e o diagnostico livre) e a expressao NB2 com a dispersao
    estimada. Os IC vem do bootstrap da etapa 3.

    Returns:
        Tabela de diagnostico e tabela de reconciliacao das faixas com o
        total.
    """
    iv = out3["intervals"]
    band = segments.segment_columns(ev.frame)["exposure_band"].to_numpy()
    rows = []
    for cand in CANDIDATES:
        pop = ("validation_free_exposure_diagnostic"
               if cand == "poisson_free_exposure" else "validation")
        for label in ["overall"] + BANDS:
            m = (np.ones(ev.n, bool) if label == "overall"
                 else band == label)
            if not m.any():
                continue
            key = (f"zerosdiff|{cand}|overall" if label == "overall"
                   else f"zerosdiff|{cand}|exposure_band|{label}")
            est = _interval(iv, FULL, pop, key)
            obs, pred = float(ev.zero[m].sum()), float(ev.p0[cand][m].sum())
            e = float(ev.e[m].sum())
            low = (False if label == "overall"
                   else segments.low_support(int(m.sum()), e))
            rows.append({
                "candidate": cand, "band": label, "rows": int(m.sum()),
                "exposure": e, "zeros_observed": obs,
                "zeros_predicted": pred,
                "diff_pred_minus_obs": pred - obs,
                "ci_low": est[1], "ci_high": est[2],
                "low_support": low,
                "zero_share_observed": obs / m.sum(),
                "zero_share_predicted": pred / m.sum()})
    table = pd.DataFrame(rows)
    recon = []
    for cand, group in table.groupby("candidate", sort=False):
        total = group[group["band"] == "overall"].iloc[0]
        bands = group[group["band"] != "overall"]
        for col in ("rows", "exposure", "zeros_observed",
                    "zeros_predicted"):
            value = float(bands[col].sum())
            recon.append({"candidate": cand, "indicator": col,
                          "bands_sum": value, "total": float(total[col]),
                          "ok": bool(np.isclose(value, total[col],
                                                rtol=0, atol=1e-6))})
    return table, pd.DataFrame(recon)


def zero_gate(table: pd.DataFrame) -> dict:
    """Aplique o criterio de inflacao de zeros da etapa 2.

    A parte numerica exige IC 95% de ``previstos - observados``
    inteiramente abaixo de zero no total e em ao menos duas faixas com
    suporte. A hipotese plausivel para um segundo processo e requisito
    separado e nao e avaliada aqui.

    Returns:
        Resultado por candidato.
    """
    out = {}
    for cand, group in table.groupby("candidate", sort=False):
        total = group[group["band"] == "overall"].iloc[0]
        bands = group[(group["band"] != "overall") & ~group["low_support"]]
        below = bands[bands["ci_high"] < 0]
        out[cand] = {
            "total_diff": float(total["diff_pred_minus_obs"]),
            "total_ci": [float(total["ci_low"]), float(total["ci_high"])],
            "total_ci_below_zero": bool(total["ci_high"] < 0),
            "bands_with_support_ci_below_zero": below["band"].tolist(),
            "numeric_requirement_met": bool(
                total["ci_high"] < 0 and len(below) >= 2)}
    return out


def mean_vs_zero_table(ev: compare.EvalSet) -> pd.DataFrame:
    """Erros nas medias e diferencas de zeros por faixa (diagnostico).

    Mostra, para o Poisson com offset e para o diagnostico livre, que erros
    nas contagens por duracao coexistem com diferencas de zeros. A coluna
    ``zeros_diff_if_band_rescaled`` reescala ``mu`` por faixa para somar as
    contagens observadas e recalcula ``sum exp(-mu)``: e apenas um
    diagnostico, nunca usado para taxas ou previsoes.
    """
    band = segments.segment_columns(ev.frame)["exposure_band"].to_numpy()
    rows = []
    for cand in ("poisson", "poisson_free_exposure"):
        for label in BANDS:
            m = band == label
            if not m.any():
                continue
            y, mu = float(ev.y[m].sum()), float(ev.mu[cand][m].sum())
            scaled = ev.mu[cand][m] * (y / mu)
            obs0 = float(ev.zero[m].sum())
            rows.append({
                "candidate": cand, "band": label, "rows": int(m.sum()),
                "claims_observed": y, "claims_predicted": mu,
                "claims_diff_pred_minus_obs": mu - y,
                "zeros_observed": obs0,
                "zeros_predicted": float(ev.p0[cand][m].sum()),
                "zeros_diff_pred_minus_obs": float(
                    ev.p0[cand][m].sum() - obs0),
                "zeros_diff_if_band_rescaled": float(
                    np.exp(-scaled).sum() - obs0)})
    table = pd.DataFrame(rows)
    totals = []
    for cand, group in table.groupby("candidate", sort=False):
        total = {c: group[c].sum() for c in table.columns
                 if c not in ("candidate", "band")}
        totals.append({"candidate": cand, "band": "total", **total})
    return pd.concat([table, pd.DataFrame(totals)], ignore_index=True)


# ------------------------------------------------------------- roteiro
def stage5_script(frozen_path: str) -> dict:
    """Roteiro congelado da avaliacao final (etapa 5)."""
    return {
        "production_eligible": False,
        "frozen_model": frozen_path,
        "model": freeze.MODEL_NAME, "reference": freeze.REFERENCE_NAME,
        "preconditions": [
            "aceite do Codex sobre a etapa 4",
            "verificar SHA-256 do ARFF, das atribuicoes, do artefato "
            "congelado e dos arquivos de origem (verify-freeze)",
            "nenhum reajuste, reaprendizado do pre-processamento ou "
            "mudanca de covariaveis apos o aceite"],
        "test_access": "uma unica execucao sobre a particao de teste, "
                       "apos o aceite; resultados nao retroalimentam o "
                       "modelo",
        "primary_metric": "desvio medio de Poisson por registro, "
                          "previsoes diretas; mesmas linhas para o "
                          "Poisson e a taxa global",
        "paired_comparison": "diferenca de desvio Poisson - taxa global, "
                             "IC 95% por bootstrap pareado por IDpol",
        "calibration": "totais previstos e observados e diferenca prevista "
                       "- observada por 100 apolices-ano, com a exposicao "
                       "total de cada grupo; sem recalibracao",
        "uncertainty": {"replicas": 1000, "seed": 20261121,
                        "sensitivity": "bootstrap por chave de perfil"},
        "support": {"rows_lt": segments.LOW_SUPPORT_ROWS,
                    "exposure_years_lt": segments.LOW_SUPPORT_EXPOSURE,
                    "rule": "sinalizar e mostrar; sem IC abaixo do limite"},
        "segmentations": list(segments.FAMILIES),
        "exposure_bands": BANDS + ["exposicao_invalida"],
        "shortest_band_visible": "(0, 0.02]",
        "panel_columns": [
            "segmento", "registros", "apolices-ano", "sinistros observados",
            "sinistros previstos", "taxa observada por 100",
            "taxa prevista por 100", "prevista - observada", "IC 95%",
            "baixo suporte"],
        "reconciliation": "registros, exposicao, sinistros observados e "
                          "previstos de cada segmentacao com o total do "
                          "teste; total do teste com 101.859 registros "
                          "conferido contra a etapa 2",
        "limitations_to_state": [
            "o offset fixo subestima as exposicoes muito curtas e "
            "superestima as longas",
            "resultados internos ao benchmark",
            "BonusMalus fora do modelo",
            "sem ZIP/ZINB"],
        "outputs": "outputs/etapa_05",
    }


# ------------------------------------------------------------- execucao
def run_stage4(root: Path) -> int:
    """Execute o diagnostico, a decisao e o congelamento.

    Efeitos colaterais: le o ARFF local, grava ``outputs/etapa_04/*``.

    Args:
        root: raiz do projeto.

    Returns:
        0 em sucesso; 1 se o reajuste nao reproduzir a etapa 3.
    """
    root = Path(root)
    out = root / OUT4
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()
    parts, provenance = load_partitions(root, keep=("train", "validation"))
    train, val = parts["train"], parts["validation"]
    out3 = artifacts.load_outputs(root)
    origin = artifacts.verify_origin(root)
    fitted = refit(train, val)
    ev, fits, pre = fitted["ev"], fitted["fits"], fitted["pre"]
    coefficients = pd.read_csv(root / OUT3 / "coefficients_ntrain_full.csv")
    repro = reproduction_check(ev, float(fitted["free"].beta[-1]), out3,
                               coefficients, fits["poisson"].beta)
    write_json(out / "reproduction_check.json", {**repro, "origin": origin})

    decision = decision_table(out3)
    bands = exposure_band_table(ev, out3)
    zeros, zero_recon = zero_diagnostic(ev, out3)
    gate = zero_gate(zeros)
    mean_zero = mean_vs_zero_table(ev)
    for name, table in (("model_decision", decision),
                        ("exposure_band_rates", bands),
                        ("zero_diagnostics", zeros),
                        ("zero_reconciliation", zero_recon),
                        ("mean_vs_zero_by_band", mean_zero)):
        write_table(out / f"{name}.csv", table)
    write_json(out / "zero_gate.json", {
        "criterion": amendment_stage3()["decisions_from_stage2_opinion"][
            "zero_inflation_gate"],
        "reference_model": "poisson",
        "by_candidate": gate,
        "plausible_second_process_hypothesis": {
            "documented": False,
            "contract_evidence": (
                "O dicionario define ClaimNb como numero de sinistros no "
                "periodo de exposicao e Exposure como duracao em anos; nao "
                "descreve mecanismo de apolices sem risco, nao ha datas nem "
                "identificador de unidade superior, e o instante de "
                "medicao de BonusMalus e desconhecido. Nenhuma hipotese de "
                "segundo processo e sustentada pelo contrato."),
            "diagnostic_reading": (
                "As diferencas de zeros acompanham os erros nas contagens "
                "por duracao (mean_vs_zero_by_band.csv)")},
        "decision": "encerrar com os modelos de contagem ja avaliados; "
                    "nenhum ZIP ou ZINB ajustado nesta rodada"})

    frozen = _build_frozen(root, pre, fits, train, val, provenance, repro)
    write_json(out / "frozen_model.json", frozen)
    write_json(out / "stage5_script.json", stage5_script(
        "outputs/etapa_04/frozen_model.json"))
    write_json(out / "decision.json", {
        "model": freeze.MODEL_NAME, "reference": freeze.REFERENCE_NAME,
        "rule_result_main_full": decision[
            (decision["degree"] == FULL)
            & (decision["population"] == "validation")].iloc[0][
            ["poisson_over_global", "nb2_over_poisson",
             "preferred_by_rules"]].to_dict(),
        "grouped_partition_sensitivity_preferred": decision[
            decision["population"] == "grouped_validation"].iloc[0][
            "preferred_by_rules"],
        "zero_inflation": "nao autorizado",
        "exposure_treatment": "offset fixo congelado; coeficiente livre "
                              "permanece diagnostico",
        "production_eligible": False})
    import psutil

    write_json(out / "cost.json", {
        "total_wall_s": round(time.perf_counter() - t_all, 3),
        "peak_rss_mib": round(_peak_mb(), 1),
        "python": sys.version, "platform": platform.platform(),
        "psutil": psutil.__version__, "package_version": __version__})
    print("reproducao ok:", repro["ok"], "| zeros gate (poisson):",
          gate["poisson"]["numeric_requirement_met"], "| livre:",
          gate["poisson_free_exposure"]["numeric_requirement_met"])
    return 0 if repro["ok"] else 1


def _build_frozen(root, pre, fits, train, val, provenance, repro) -> dict:
    """Monte o artefato congelado do Poisson e da taxa global."""
    out3 = root / OUT3
    out2 = root / "outputs" / "etapa_02"
    poisson, glob = fits["poisson"], fits["global"]
    mu = {"poisson": poisson.predict(pre.transform(val)[0],
                                     val["Exposure"].to_numpy()),
          "global": glob.predict(None, val["Exposure"].to_numpy())}
    reference = freeze.aggregate_reference(val, mu)
    sample_hash = compare.ids_hash(train["IDpol"])
    config3 = json.loads((out3 / "comparison_config.json").read_text(
        "utf-8"))
    expected = [d for d in config3["ladder"] if d["degree"] == FULL][0]
    return {
        "production_eligible": False,
        "model": freeze.MODEL_NAME, "reference": freeze.REFERENCE_NAME,
        "decision_basis": "regra aprovada; maior degrau viavel; particao "
                          "principal por IDpol",
        "estimator": {
            "family": "Poisson", "link": "log",
            "offset": "log(Exposure), coeficiente fixo 1",
            "exposure_treatment": "Exposure e dado do cenario; "
                                  "mu = Exposure * exp(x'beta)",
            "columns": list(pre.columns),
            "coefficients": [float(b) for b in poisson.beta],
            "n_fit": int(len(train)), "converged": bool(poisson.converged),
            "iterations": int(poisson.iterations),
            "loglik_fit": float(poisson.loglik),
            "bonus_malus_included": False,
            "coefficient_precision": "repr de float64 (17 digitos), exato "
                                     "na ida e na volta"},
        "global_rate": {"rate": float(glob.rate),
                        "definition": "soma(ClaimNb)/soma(Exposure) no "
                                      "mesmo treino",
                        "n_fit": int(len(train))},
        "preprocessor": pre.summary(),
        "unknown_and_missing_policy": {
            "unknown_or_rare_category": "OTHER (ou nivel de referencia se "
                                        "nao houver OTHER)",
            "missing_categorical": "__MISSING__ tratado como desconhecido",
            "missing_numeric": "mediana do ajuste",
            "invalid_claimnb_exposure_or_mu": "interrompem a avaliacao"},
        "training_sample": {"degree": FULL, "rows": int(len(train)),
                            "ids_sha256": sample_hash,
                            "matches_stage3_ladder": bool(
                                sample_hash == expected["sample_ids_sha256"])},
        "hashes": {
            "arff_sha256": provenance["arff_sha256"],
            "assignments_sha256": provenance["assignment_sha256"],
            "partition_config_sha256": freeze.file_hash(
                out2 / "partition_config.json"),
            "protocol_config_sha256": provenance["protocol_sha256"],
            "stage3_protocol_amendment_sha256": freeze.file_hash(
                out3 / "protocol_amendment.json"),
            "stage3_comparison_config_sha256": freeze.file_hash(
                out3 / "comparison_config.json"),
            "stage3_validation_metrics_sha256": freeze.file_hash(
                out3 / "validation_metrics.csv"),
            "stage3_intervals_sha256": freeze.file_hash(
                out3 / "intervals.csv"),
            "stage3_coefficients_sha256": freeze.file_hash(
                out3 / "coefficients_ntrain_full.csv"),
            "stage3_preprocessing_sha256": freeze.file_hash(
                out3 / "preprocessing_ntrain_full.json")},
        "code": {"modules_sha256": freeze.code_hashes(root),
                 **freeze.environment_record()},
        "protocol": {"document": "reports/etapa_02_protocolo.md (secoes "
                                 "1-11)",
                     "bootstrap": {"replicas": 1000, "seed": 20261121}},
        "reproduction_of_stage3": {
            "ok": repro["ok"],
            "max_rel_diff_metrics": repro["max_rel_diff_metrics"],
            "coefficients_max_abs_diff": repro[
                "poisson_coefficients_max_abs_diff_vs_stage3_csv"]},
        "verification": {"tolerance": {"rtol": freeze.RTOL,
                                       "atol": freeze.ATOL},
                         "validation_reference": reference},
    }


def verify_freeze(root: Path) -> int:
    """Restaure o artefato e reproduza a validacao aprovada, sem reajustar.

    Nao reaprende o pre-processamento nem reajusta o estimador: usa so o
    estado persistido. Confere hashes de origem, agregados congelados e as
    metricas aprovadas da etapa 3.

    Args:
        root: raiz do projeto.

    Returns:
        0 se tudo conferir; 1 caso contrario.
    """
    root = Path(root)
    frozen = json.loads((root / OUT4 / "frozen_model.json").read_text(
        "utf-8"))
    parts, provenance = load_partitions(root, keep=("validation",))
    val = parts["validation"]
    hashes = frozen["hashes"]
    checks = {
        "arff_sha256": provenance["arff_sha256"] == hashes["arff_sha256"],
        "assignments_sha256": provenance["assignment_sha256"]
        == hashes["assignments_sha256"]}
    for key, rel in (("stage3_protocol_amendment_sha256",
                      "protocol_amendment.json"),
                     ("stage3_comparison_config_sha256",
                      "comparison_config.json"),
                     ("stage3_validation_metrics_sha256",
                      "validation_metrics.csv"),
                     ("stage3_intervals_sha256", "intervals.csv"),
                     ("stage3_coefficients_sha256",
                      "coefficients_ntrain_full.csv"),
                     ("stage3_preprocessing_sha256",
                      "preprocessing_ntrain_full.json")):
        checks[key] = freeze.file_hash(root / OUT3 / rel) == hashes[key]
    mu = freeze.predict_frozen(frozen, val)
    now = freeze.aggregate_reference(val, mu)
    frozen_cmp = freeze.compare_references(
        now, frozen["verification"]["validation_reference"])
    met = pd.read_csv(root / OUT3 / "validation_metrics.csv")
    met = met[(met["population"] == "validation")
              & (met["degree"].astype(str) == FULL)
              & (met["subset"] == "all_rows")]
    approved = {}
    for cand in ("global", "poisson"):
        row = met[met["candidate"] == cand].iloc[0]
        approved[cand] = {
            k: abs(now["candidates"][cand][k] - float(row[k]))
            for k in ("mean_poisson_deviance", "mean_loglik",
                      "predicted_claims", "rate_diff_pred_minus_obs",
                      "zeros_predicted")}
    approved_ok = all(
        np.isclose(now["candidates"][c][k], float(
            met[met["candidate"] == c].iloc[0][k]),
            rtol=freeze.RTOL, atol=freeze.ATOL)
        for c in approved for k in approved[c])
    code_same = (freeze.code_hashes(root)
                 == frozen["code"]["modules_sha256"])
    ok = all(checks.values()) and frozen_cmp["ok"] and approved_ok
    result = {"hash_checks": checks, "frozen_reference": frozen_cmp,
              "approved_stage3_abs_diff": approved,
              "approved_stage3_ok": bool(approved_ok),
              "code_modules_unchanged": bool(code_same),
              "refit_performed": False, "preprocessor_relearned": False,
              "test_used": False, "ok": bool(ok)}
    write_json(root / OUT4 / "verify_freeze.json", result)
    print("verify-freeze ok:", ok, "| max_abs_diff:",
          frozen_cmp["max_abs_diff"], "| code unchanged:", code_same)
    return 0 if ok else 1
