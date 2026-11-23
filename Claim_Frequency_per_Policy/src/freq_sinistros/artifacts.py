"""Leitura verificada dos artefatos da etapa 3 para os notebooks.

O notebook de comparacao nao repete os ajustes: ele carrega as saidas de
``freq_sinistros compare`` apos conferir a origem (hash do ARFF, das
atribuicoes, da emenda ao protocolo e da configuracao da comparacao).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from freq_sinistros.labels import (
    LANGUAGES,
    describe_column,
    describe_column_meaning,
)

OUTPUT_TABLES = ["fits", "validation_metrics", "intervals",
                 "calibration_by_segment", "zeros_preliminary",
                 "reconciliation", "dispersion_diagnostics"]

COLUMN_NAMES = {
    "pt_br": {
        "rows": "registros", "exposure": "exposição",
        "claims": "sinistros", "zeros": "zeros", "zero_share":
            "proporção de zeros", "rate_per_100": "taxa por 100 ap.-ano",
        "low_support": "baixo suporte", "segment": "segmento",
        "candidate": "candidato", "degree": "degrau",
        "converged": "convergiu", "wall_s": "tempo (s)",
        "peak_rss_mib": "pico RAM (MiB)", "iterations": "iterações",
        "mean_poisson_deviance": "desvio médio de Poisson",
        "mean_loglik": "log-verossimilhança média",
        "rate_diff_pred_minus_obs": "taxa prevista − observada",
        "observed_claims": "sinistros observados",
        "predicted_claims": "sinistros previstos",
        "band": "faixa", "estimate": "estimativa", "ci_low": "IC inf.",
        "ci_high": "IC sup.", "indicator": "indicador",
    },
    "en_us": {
        "rows": "records", "exposure": "exposure", "claims": "claims",
        "zeros": "zeros", "zero_share": "zero share",
        "rate_per_100": "rate per 100 policy-years",
        "low_support": "low support", "segment": "segment",
        "candidate": "candidate", "degree": "step",
        "converged": "converged", "wall_s": "time (s)",
        "peak_rss_mib": "peak RAM (MiB)", "iterations": "iterations",
        "mean_poisson_deviance": "mean Poisson deviance",
        "mean_loglik": "mean log-likelihood",
        "rate_diff_pred_minus_obs": "predicted − observed rate",
        "observed_claims": "observed claims",
        "predicted_claims": "predicted claims",
        "band": "band", "estimate": "estimate", "ci_low": "CI low",
        "ci_high": "CI high", "indicator": "indicator",
    },
}


def project_root(start: Path | None = None) -> Path:
    """Suba a partir de ``start`` ate achar ``pyproject.toml``.

    Raises:
        FileNotFoundError: se nao houver raiz de projeto acima.
    """
    here = Path(start or Path.cwd()).resolve()
    for folder in [here, *here.parents]:
        if (folder / "pyproject.toml").exists():
            return folder
    raise FileNotFoundError("pyproject.toml nao encontrado")


def localize(table: pd.DataFrame, language: str) -> pd.DataFrame:
    """Renomeie colunas conhecidas para o idioma do notebook.

    Raises:
        ValueError: se o idioma nao for suportado.
    """
    if language not in LANGUAGES:
        raise ValueError(f"idioma nao suportado: {language}")
    return table.rename(columns=COLUMN_NAMES[language])


def variable_dictionary_table(
        columns: list[str] | pd.Index, language: str) -> pd.DataFrame:
    """Apresente o dicionario bilingue das variaveis de um conjunto.

    Args:
        columns: nomes das variaveis, na ordem em que devem ser exibidas.
        language: ``pt_br`` ou ``en_us``.

    Returns:
        Tabela com nome, papel, unidade ou categorias e disponibilidade.

    Raises:
        ValueError: se o idioma nao for suportado.
    """
    if language not in LANGUAGES:
        raise ValueError(f"idioma nao suportado: {language}")
    headers = (
        ["variável", "papel / significado", "unidade / categorias",
         "disponibilidade"]
        if language == "pt_br"
        else ["variable", "role / meaning", "unit / categories",
              "availability"]
    )
    rows = []
    for name in columns:
        description = describe_column(str(name), language)
        meaning = describe_column_meaning(str(name), language)
        role_and_meaning = description["role"]
        if meaning != role_and_meaning:
            role_and_meaning = f"{role_and_meaning}: {meaning}"
        rows.append([name, role_and_meaning, description["unit"],
                     description["availability"]])
    return pd.DataFrame(rows, columns=headers)


def _sha256(path: Path) -> str:
    """SHA-256 de um arquivo."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_origin(root: Path) -> dict[str, Any]:
    """Confira a origem das saidas contra os hashes registrados.

    Verifica: SHA-256 do ARFF contra o manifesto da etapa 1, atribuicoes
    contra a configuracao da etapa 2 e contra a comparacao, e a emenda ao
    protocolo contra o hash gravado na configuracao da comparacao.

    Args:
        root: raiz do projeto.

    Returns:
        Dicionario com cada verificacao (``ok``) e o resultado agregado.

    Raises:
        FileNotFoundError: se faltar algum artefato de origem.
    """
    root = Path(root)
    manifest = json.loads((root / "outputs/etapa_01/manifest.json")
                          .read_text("utf-8"))
    part = json.loads((root / "outputs/etapa_02/partition_config.json")
                      .read_text("utf-8"))
    comp = json.loads((root / "outputs/etapa_03/comparison_config.json")
                      .read_text("utf-8"))
    arff = root / manifest["artifact"]["local_path"]
    checks = {
        "arff_sha256_matches_stage1_manifest":
            _sha256(arff) == manifest["artifact"]["sha256"],
        "arff_sha256_matches_comparison":
            comp["provenance"]["arff_sha256"]
            == manifest["artifact"]["sha256"],
        "assignments_match_stage2_config":
            comp["provenance"]["assignment_sha256"]
            == part["assignment_file"]["sha256_of_ordered_assignments"],
        "assignment_file_hash_matches":
            _sha256(root / part["assignment_file"]["path"])
            == part["assignment_file"]["sha256_of_ordered_assignments"],
        "protocol_amendment_matches_comparison":
            _sha256(root / "outputs/etapa_03/protocol_amendment.json")
            == comp["amendment_sha256"],
        "stage2_validation_totals_match":
            bool(comp["stage2_validation_totals_match"]),
        "test_partition_not_returned":
            "test" not in comp["provenance"]["sizes"],
    }
    return {"checks": checks, "all_ok": all(checks.values()),
            "arff_sha256": manifest["artifact"]["sha256"],
            "assignment_sha256":
                part["assignment_file"]["sha256_of_ordered_assignments"],
            "partition_rule": part["rule"]}


def load_outputs(root: Path) -> dict[str, Any]:
    """Carregue as tabelas e a configuracao da comparacao.

    Args:
        root: raiz do projeto.

    Returns:
        Tabelas por nome e a chave ``config``.
    """
    base = Path(root) / "outputs" / "etapa_03"
    out = {name: pd.read_csv(base / f"{name}.csv")
           for name in OUTPUT_TABLES}
    out["config"] = json.loads(
        (base / "comparison_config.json").read_text("utf-8"))
    return out


_PROVENANCE_LABELS = {
    "pt_br": ["SHA-256 do ARFF", "SHA-256 das atribuições", "regra de "
              "partição", "partição usada", "registros de treino",
              "unidade de observação", "registro de cobertura (IDpol)"],
    "en_us": ["ARFF SHA-256", "assignments SHA-256", "partition rule",
              "partition used", "training records", "observation unit",
              "coverage record (IDpol)"],
}


def provenance_table(provenance: dict[str, Any], n_rows: int,
                     language: str) -> pd.DataFrame:
    """Tabela item/valor com a procedencia verificada de uma leitura.

    Args:
        provenance: saida de :func:`freq_sinistros.data.load_partitions`.
        n_rows: linhas da particao usada.
        language: ``pt_br`` ou ``en_us``.
    """
    names = _PROVENANCE_LABELS[language]
    values = [provenance["arff_sha256"], provenance["assignment_sha256"],
              provenance["partition_rule"], "train", f"{n_rows:,}",
              names[6], ""]
    items = names[:6]
    return pd.DataFrame({"item": items, "value": values[:6]})


def coverage_table(summary: dict[str, Any], language: str) -> pd.DataFrame:
    """Niveis desconhecidos em linhas de treino fora da amostra de ajuste."""
    check = summary["unknown_level_check"]
    head = ["covariável", "linhas mapeadas para OTHER ou referência",
            "níveis agrupados no ajuste"] if language == "pt_br" else [
        "covariate", "rows mapped to OTHER or reference",
        "levels pooled in the fit"]
    rows = []
    for col, n in check["unknown_levels_heldout"].items():
        rows.append({head[0]: col, head[1]: n,
                     head[2]: ", ".join(check["pooled_levels_in_fit"][col])
                     or "-"})
    return pd.DataFrame(rows)


def extremes_table(summary: dict[str, Any], language: str) -> pd.DataFrame:
    """Casos extremos do treino em formato item/valor."""
    ex = summary["extremes"]
    pt = language == "pt_br"
    rows = [
        ("registros com ClaimNb > 4" if pt else "records with ClaimNb > 4",
         ex["claimnb_gt_4_rows"]),
        ("sinistros nesses registros" if pt else "claims in those records",
         ex["claimnb_gt_4_claims"]),
        ("registros com Exposure > 1" if pt else "records with Exposure > 1",
         ex["exposure_gt_1_rows"]),
        ("registros com Exposure ≤ 0,02" if pt
         else "records with Exposure ≤ 0.02", ex["exposure_le_0.02_rows"]),
        ("registros com DrivAge = 100" if pt
         else "records with DrivAge = 100", ex["drivage_ge_100_rows"]),
        ("registros com VehAge = 100" if pt
         else "records with VehAge = 100", ex["vehage_ge_100_rows"]),
    ]
    return pd.DataFrame(rows, columns=["item", "value"])


MAIN_POP = "validation"
KEY_INDICATORS = ["d_dev|poisson-global", "d_dev|nb2-poisson",
                  "d_dev|nb2-global", "d_ll|poisson-global",
                  "d_ll|nb2-poisson", "d_ll|nb2-global"]


def select_intervals(intervals: pd.DataFrame, degree: str, population: str,
                     keys: list[str]) -> pd.DataFrame:
    """Intervalos de um degrau e populacao, para os indicadores ``keys``."""
    sub = intervals[(intervals["degree"].astype(str) == degree)
                    & (intervals["population"] == population)
                    & intervals["indicator"].isin(keys)]
    return sub.drop_duplicates("indicator")


def main_metrics(out: dict[str, Any],
                 degree: str | None = None) -> pd.DataFrame:
    """Metricas de validacao dos tres candidatos (todos os degraus)."""
    met = out["validation_metrics"]
    met = met[(met["population"] == MAIN_POP)
              & (met["subset"] == "all_rows")]
    if degree is not None:
        met = met[met["degree"].astype(str) == degree]
    cols = ["degree", "candidate", "rows", "exposure", "observed_claims",
            "predicted_claims", "observed_rate_per_100",
            "predicted_rate_per_100", "rate_diff_pred_minus_obs",
            "mean_poisson_deviance", "mean_loglik"]
    return met[cols].reset_index(drop=True)


def rules_table(out: dict[str, Any]) -> pd.DataFrame:
    """Aplique as regras de escolha a cada degrau e a particao agrupada.

    Resultado provisorio: a decisao de modelo pertence a etapa 4.
    """
    from freq_sinistros.compare import evaluate_rules

    rows = []
    met, iv = out["validation_metrics"], out["intervals"]
    for (degree, pop), group in met[
            met["population"].isin([MAIN_POP, "grouped_validation"])
            & (met["subset"] == "all_rows")].groupby(
            ["degree", "population"], sort=False):
        group = group[group["candidate"].isin(
            ["global", "poisson", "nb2"])]
        ints = iv[(iv["degree"].astype(str) == str(degree))
                  & (iv["population"] == pop)]
        if "sensitivity" in ints and pop == "grouped_validation":
            ints = ints[ints["sensitivity"] == "grouped_partition"]
        result = evaluate_rules(group, ints)
        rows.append({
            "degree": degree, "population": pop,
            "poisson_over_global": result["poisson_over_global"],
            "nb2_over_poisson": result["nb2_over_poisson_conditions"],
            "preferred_by_rules": result["preferred_by_rules"]})
    return pd.DataFrame(rows)


def calibration_pivot(out: dict[str, Any], family: str,
                      degree: str) -> pd.DataFrame:
    """Diferenca prevista - observada por segmento e candidato."""
    cal = out["calibration_by_segment"]
    cal = cal[(cal["population"] == MAIN_POP)
              & (cal["degree"].astype(str) == degree)
              & (cal["family"] == family)
              & cal["candidate"].isin(["global", "poisson", "nb2"])]
    base = cal[cal["candidate"] == "poisson"].set_index("segment")
    table = cal.pivot(index="segment", columns="candidate",
                      values="rate_difference_pred_minus_obs")
    table.insert(0, "rows", base["rows"])
    table.insert(1, "observed_rate_per_100", base["observed_rate_per_100"])
    table["low_support"] = base["low_support"]
    return table.reset_index()


def sensitivity_table(out: dict[str, Any]) -> pd.DataFrame:
    """Diferencas pareadas das sensibilidades, com IC 95%."""
    iv = out["intervals"]
    iv = iv[iv["population"] != MAIN_POP]
    rows = iv[iv["indicator"].str.startswith(("d_dev|", "d_ll|"))]
    return rows[["population", "indicator", "estimate", "ci_low",
                 "ci_high"]].reset_index(drop=True)


def reconciliation_summary(out: dict[str, Any]) -> pd.DataFrame:
    """Resumo da reconciliacao: verificacoes e falhas por populacao."""
    rec = out["reconciliation"]
    return rec.groupby("population").agg(
        checks=("ok", "size"), all_ok=("ok", "all")).reset_index()


def zero_table(out: dict[str, Any], degree: str, population: str = MAIN_POP,
               candidates: tuple[str, ...] = ("global", "poisson", "nb2")
               ) -> pd.DataFrame:
    """Zeros observados e previstos (resumo preliminar) de uma populacao.

    Args:
        out: saida de :func:`load_outputs`.
        degree: degrau de ajuste.
        population: populacao de avaliacao (padrao: validacao principal).
        candidates: candidatos a listar; use ``("poisson_free_exposure",)``
            com a populacao ``validation_free_exposure_diagnostic`` para o
            diagnostico com ``log(Exposure)`` livre.
    """
    z = out["zeros_preliminary"]
    z = z[(z["population"] == population)
          & (z["degree"].astype(str) == degree)
          & z["candidate"].isin(candidates)]
    return z.reset_index(drop=True)


def population_totals(out: dict[str, Any]) -> pd.DataFrame:
    """Totais por populacao de avaliacao e resultado da reconciliacao.

    Cada populacao tem seus proprios denominadores: validacao principal,
    validacao agrupada e as sensibilidades que reusam a validacao principal.
    O recorte descritivo sem a faixa muito curta entra como ultima linha,
    sem reconciliacao por segmento.
    """
    rec = out["reconciliation"]
    rows = []
    for pop, group in rec.groupby("population", sort=False):
        total = group[group["family"] == group["family"].iloc[0]]
        total = total[total["candidate"] == total["candidate"].iloc[0]]
        values = total.groupby("indicator")["total"].first()
        rows.append({
            "population": pop, "records": float(values["rows"]),
            "exposure": float(values["exposure"]),
            "observed_claims": float(values["observed_claims"]),
            "candidates": int(group["candidate"].nunique()),
            "checks": int(len(group)), "all_ok": bool(group["ok"].all()),
            "reconciled_by_segment": True})
    met = out["validation_metrics"]
    sub = met[met["subset"] == "exposure_gt_0.02_descriptive"]
    if len(sub):
        first = sub.iloc[0]
        rows.append({
            "population": "validation (Exposure > 0.02, descriptive)",
            "records": float(first["rows"]),
            "exposure": float(first["exposure"]),
            "observed_claims": float(first["observed_claims"]),
            "candidates": int(sub["candidate"].nunique()),
            "checks": 0, "all_ok": None,
            "reconciled_by_segment": False})
    return pd.DataFrame(rows)


STAGE4_TABLES = ["model_decision", "exposure_band_rates", "zero_diagnostics",
                 "zero_reconciliation", "mean_vs_zero_by_band"]
STAGE4_JSON = ["frozen_model", "zero_gate", "decision", "stage5_script",
               "verify_freeze", "reproduction_check"]


def load_stage4(root: Path) -> dict[str, Any]:
    """Carregue tabelas e JSON da etapa 4 (``outputs/etapa_04``)."""
    base = Path(root) / "outputs" / "etapa_04"
    out = {name: pd.read_csv(base / f"{name}.csv")
           for name in STAGE4_TABLES}
    for name in STAGE4_JSON:
        out[name] = json.loads((base / f"{name}.json").read_text("utf-8"))
    return out


def verify_stage4(root: Path) -> dict[str, Any]:
    """Confira o artefato congelado contra os arquivos de origem atuais.

    Verifica os hashes registrados no congelamento (ARFF, atribuicoes e
    arquivos da etapa 3) e o resultado de ``verify-freeze`` e do reajuste.

    Args:
        root: raiz do projeto.

    Returns:
        Verificacoes individuais e ``all_ok``.
    """
    root = Path(root)
    s4 = load_stage4(root)
    frozen = s4["frozen_model"]
    hashes = frozen["hashes"]
    out3, out2 = root / "outputs/etapa_03", root / "outputs/etapa_02"
    pairs = {
        "partition_config": (out2 / "partition_config.json",
                             "partition_config_sha256"),
        "protocol_config": (out2 / "protocol_config.json",
                            "protocol_config_sha256"),
        "stage3_amendment": (out3 / "protocol_amendment.json",
                             "stage3_protocol_amendment_sha256"),
        "stage3_comparison_config": (out3 / "comparison_config.json",
                                     "stage3_comparison_config_sha256"),
        "stage3_validation_metrics": (out3 / "validation_metrics.csv",
                                      "stage3_validation_metrics_sha256"),
        "stage3_coefficients": (out3 / "coefficients_ntrain_full.csv",
                                "stage3_coefficients_sha256"),
    }
    checks = {name: _sha256(path) == hashes[key]
              for name, (path, key) in pairs.items()}
    checks["arff"] = verify_origin(root)["arff_sha256"] == hashes[
        "arff_sha256"]
    checks["verify_freeze_ok"] = bool(s4["verify_freeze"]["ok"])
    checks["reproduction_of_stage3_ok"] = bool(
        s4["reproduction_check"]["ok"])
    checks["training_sample_matches_stage3"] = bool(
        frozen["training_sample"]["matches_stage3_ladder"])
    checks["test_not_used"] = not s4["verify_freeze"]["test_used"]
    return {"checks": checks, "all_ok": all(checks.values())}


def zero_reconciliation_ok(s4: dict[str, Any]) -> pd.Series:
    """Resultado da reconciliacao das faixas com o total, por candidato."""
    rec = s4["zero_reconciliation"]
    return rec.groupby("candidate")["ok"].all()


def zero_gate_table(s4: dict[str, Any]) -> pd.DataFrame:
    """Criterio numerico de zeros da etapa 2, por candidato."""
    table = pd.DataFrame(s4["zero_gate"]["by_candidate"]).T
    table["bands_with_support_ci_below_zero"] = table[
        "bands_with_support_ci_below_zero"].apply(", ".join)
    return table


def second_process_table(s4: dict[str, Any]) -> pd.Series:
    """Evidencia do contrato sobre um segundo processo e a decisao."""
    gate = s4["zero_gate"]
    hypothesis = gate["plausible_second_process_hypothesis"]
    return pd.Series({
        "criterion": gate["criterion"],
        "hypothesis_documented": hypothesis["documented"],
        "contract_evidence": hypothesis["contract_evidence"],
        "decision": gate["decision"]})


STAGE5_TABLES = ["test_metrics", "test_intervals", "test_calibration",
                 "test_reconciliation", "validation_vs_test_bands", "panel"]
STAGE5_JSON = ["official_run", "integrity_gate", "conclusion",
               "test_population"]


def load_stage5(root: Path) -> dict[str, Any]:
    """Carregue os agregados da avaliacao oficial (``outputs/etapa_05``)."""
    base = Path(root) / "outputs" / "etapa_05"
    out = {name: pd.read_csv(base / f"{name}.csv")
           for name in STAGE5_TABLES}
    for name in STAGE5_JSON:
        out[name] = json.loads((base / f"{name}.json").read_text("utf-8"))
    return out


def verify_stage5(root: Path) -> dict[str, Any]:
    """Confira offline os agregados da avaliacao oficial.

    Nao le o ARFF nem o teste: confere o registro, os hashes dos arquivos
    listados, o modelo e o roteiro contra os hashes do registro, a
    reconciliacao e o numero de registros do teste.

    Args:
        root: raiz do projeto.

    Returns:
        Verificacoes individuais e ``all_ok``.
    """
    root = Path(root)
    base = root / "outputs" / "etapa_05"
    out = load_stage5(root)
    record = out["official_run"]
    sums = {}
    for line in (base / "SHA256SUMS_etapa05.txt").read_text(
            "ascii").splitlines():
        digest, name = line.split(" *", 1)
        sums[name] = digest
    config = json.loads((root / "outputs/etapa_02/partition_config.json")
                        .read_text("utf-8"))
    n_test = next(r["ids"] for r in config["sizes"]
                  if r["partition"] == "test")
    checks = {
        "status_completed": record["status"] == "completed",
        "files_match_checksums": all(
            _sha256(base / name) == digest for name, digest in sums.items()),
        "frozen_model_matches_record": _sha256(
            root / "outputs/etapa_04/frozen_model.json")
        == record["frozen_model_sha256"],
        "script_matches_record": _sha256(
            root / "outputs/etapa_04/stage5_script.json")
        == record["stage5_script_sha256"],
        "integrity_gate_ok": bool(out["integrity_gate"]["ok"]),
        "test_rows_match_stage2": record["test_rows"] == n_test
        == out["test_population"]["rows"],
        "reconciliation_ok": bool(out["test_reconciliation"]["ok"].all()),
        "single_official_run_lock": (base / "official_run.lock").exists(),
    }
    return {"checks": checks, "all_ok": all(checks.values())}


_PANEL_PT = {"segment": "segmento", "records": "registros",
             "policy_years": "apólices-ano",
             "observed_claims": "sinistros observados",
             "predicted_claims": "sinistros previstos",
             "observed_rate_per_100": "taxa observada por 100",
             "predicted_rate_per_100": "taxa prevista por 100",
             "pred_minus_obs": "prevista − observada",
             "ci_low": "IC inf.", "ci_high": "IC sup.",
             "low_support": "baixo suporte"}
_PANEL_EN = {"segment": "segment", "records": "records",
             "policy_years": "policy-years",
             "observed_claims": "observed claims",
             "predicted_claims": "predicted claims",
             "observed_rate_per_100": "observed rate per 100",
             "predicted_rate_per_100": "predicted rate per 100",
             "pred_minus_obs": "predicted − observed",
             "ci_low": "CI low", "ci_high": "CI high",
             "low_support": "low support"}


def panel_view(out5: dict[str, Any], family: str, candidate: str,
               language: str) -> pd.DataFrame:
    """Painel de uma segmentacao e candidato, com colunas localizadas."""
    panel = out5["panel"]
    sub = panel[(panel["family"] == family)
                & (panel["candidate"] == candidate)]
    sub = sub.drop(columns=["family", "candidate"]).reset_index(drop=True)
    return sub.rename(columns=_PANEL_PT if language == "pt_br"
                      else _PANEL_EN)


def metrics_view(out5: dict[str, Any], language: str) -> pd.DataFrame:
    """Metricas de teste dos dois candidatos, com colunas localizadas."""
    cols = ["candidate", "rows", "exposure", "observed_claims",
            "predicted_claims", "observed_rate_per_100",
            "predicted_rate_per_100", "rate_diff_pred_minus_obs",
            "mean_poisson_deviance", "mean_loglik"]
    return localize(out5["test_metrics"][cols], language)


def run_summary(out5: dict[str, Any]) -> pd.Series:
    """Resumo do registro da execucao oficial e da populacao de teste."""
    record, pop = out5["official_run"], out5["test_population"]
    approved = out5["integrity_gate"]["approved"]
    return pd.Series({
        "status": record["status"], "test_rows": record["test_rows"],
        "policy_years": pop["exposure"],
        "observed_claims": pop["observed_claims"],
        "observed_zeros": pop["observed_zeros"],
        "wall_s": record["wall_s"], "peak_rss_mib": record["peak_rss_mib"],
        "bootstrap_replicas": record["bootstrap"]["replicas"],
        "bootstrap_seed": record["bootstrap"]["seed"],
        "frozen_model_sha256_approved": approved["frozen_model_sha256"],
        "stage5_script_sha256_approved": approved["stage5_script_sha256"],
        "test_ids_sha256": record["test_ids_sha256"]})


def gate_checks(out5: dict[str, Any]) -> pd.Series:
    """Verificacoes do portao de integridade executado antes do teste."""
    return pd.Series(out5["integrity_gate"]["checks"], name="ok")
