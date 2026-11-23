"""Etapa 5: avaliacao final unica no teste reservado.

O fluxo e: (1) portao de integridade, que confere o modelo, o roteiro, os
hashes de origem e o codigo **antes** de qualquer linha de teste ser lida;
(2) trava de execucao unica; (3) leitura da particao de teste; (4)
avaliacao do Poisson congelado e da taxa global de referencia com o roteiro
aprovado; (5) gravacao dos agregados. Nenhum modulo congelado e alterado:
esta etapa so os reutiliza.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from freq_sinistros import __version__, compare, freeze, metrics, stage4
from freq_sinistros.arff import read_arff
from freq_sinistros.audit import _peak_mb
from freq_sinistros.models import ResourceMonitor
from freq_sinistros.stage3 import write_json, write_table

OUT5 = Path("outputs") / "etapa_05"
LOCK_NAME = "official_run.lock"
CANDIDATES = ("poisson", "global")

# Referencia independente aprovada no parecer da etapa 4 (nao lida de
# nenhum dos arquivos verificados).
APPROVED = {
    "frozen_model_sha256":
        "8ac94b7e59e55becdd0319f05f9d958483ea4042dc0abaad1a2f075994efe80a",
    "stage5_script_sha256":
        "c0365ad5566e32e5099c48bb1d007367e646ce63ee8b040bc95dc38ba13c1ed8",
    "modules": {
        "freeze":
            "ad44566d990037ee8379b11d9b9c93f4f8ad010708ccd4a74b23d9848e9af31c",
        "arff":
            "a24d0fb2dfb3fa5fc0809fc7543bd13f1e5cdfafb4222e815d4197e6fa42a923",
        "profiles":
            "5841136473ff529c489e95960d40a3cef2dca535ca5a9faf9f2affbc4ca6aca3",
    },
}


class IntegrityError(RuntimeError):
    """Alguma verificacao de integridade falhou antes do teste."""


class AlreadyEvaluatedError(RuntimeError):
    """A avaliacao oficial ja foi iniciada: nova execucao recusada."""


@dataclass
class GateResult:
    """Resultado do portao de integridade (necessario para ler o teste).

    Attributes:
        ok: todas as verificacoes passaram.
        checks: resultado de cada verificacao.
        frozen: modelo congelado carregado.
        expected_test_rows: tamanho da particao de teste da etapa 2.
    """

    ok: bool
    checks: dict
    frozen: dict = field(default_factory=dict)
    expected_test_rows: int = 0


def file_sha(path: Path) -> str:
    """SHA-256 de um arquivo."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module_hashes(package_dir: Path, names: list[str]) -> dict:
    """SHA-256 dos modulos ``names`` em ``package_dir``."""
    return {n: file_sha(Path(package_dir) / f"{n}.py") for n in names}


def default_package_dir() -> Path:
    """Pasta do pacote efetivamente importado."""
    return Path(__file__).resolve().parent


# ----------------------------------------------------------------- portao
def integrity_gate(root: Path, approved: dict | None = None,
                   package_dir: Path | None = None) -> GateResult:
    """Confira a integridade; nao le nenhuma linha de teste.

    Verifica: SHA-256 do modelo congelado e do roteiro contra os valores
    aprovados; ARFF, atribuicoes, configuracoes de particao e protocolo e
    arquivos de origem da etapa 3 contra os hashes do modelo; modulos
    registrados no modelo e tres modulos complementares contra os valores
    aprovados; e ``verify-freeze`` (codigo de saida 0, agregados dentro da
    tolerancia, ``code_modules_unchanged``).

    Args:
        root: raiz do projeto.
        approved: referencias aprovadas (padrao: ``APPROVED``).
        package_dir: pasta dos modulos (padrao: o pacote importado).

    Returns:
        ``GateResult`` com ``ok`` verdadeiro.

    Raises:
        IntegrityError: se qualquer verificacao falhar (nenhuma linha de
            teste foi lida).
    """
    root = Path(root)
    approved = approved or APPROVED
    package_dir = Path(package_dir) if package_dir else default_package_dir()
    out4 = root / stage4.OUT4
    out3 = root / stage4.OUT3
    out2 = root / "outputs" / "etapa_02"
    checks: dict[str, bool] = {}
    frozen_path, script_path = (out4 / "frozen_model.json",
                                out4 / "stage5_script.json")
    checks["frozen_model_sha256_approved"] = (
        file_sha(frozen_path) == approved["frozen_model_sha256"])
    checks["stage5_script_sha256_approved"] = (
        file_sha(script_path) == approved["stage5_script_sha256"])
    frozen = json.loads(frozen_path.read_text("utf-8"))
    hashes = frozen["hashes"]
    manifest = json.loads((root / "outputs/etapa_01/manifest.json")
                          .read_text("utf-8"))
    part_cfg = json.loads((out2 / "partition_config.json").read_text(
        "utf-8"))
    raw = root / manifest["artifact"]["local_path"]
    assign = root / part_cfg["assignment_file"]["path"]
    checks["arff_sha256"] = (
        file_sha(raw) == hashes["arff_sha256"]
        == manifest["artifact"]["sha256"])
    checks["assignments_sha256"] = (
        file_sha(assign) == hashes["assignments_sha256"]
        == part_cfg["assignment_file"]["sha256_of_ordered_assignments"])
    checks["partition_config_sha256"] = (
        file_sha(out2 / "partition_config.json")
        == hashes["partition_config_sha256"])
    checks["protocol_config_sha256"] = (
        file_sha(out2 / "protocol_config.json")
        == hashes["protocol_config_sha256"])
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
        checks[key] = file_sha(out3 / rel) == hashes[key]
    registered = frozen["code"]["modules_sha256"]
    now = module_hashes(package_dir, list(registered))
    checks["code_modules_match_frozen"] = now == registered
    extra = module_hashes(package_dir, list(approved["modules"]))
    for name, expected in approved["modules"].items():
        checks[f"module_{name}_approved"] = extra[name] == expected
    checks["production_eligible_false"] = (
        frozen["production_eligible"] is False)
    if not any(not v for v in checks.values()):
        code = stage4.verify_freeze(root)
        result = json.loads((out4 / "verify_freeze.json").read_text("utf-8"))
        checks["verify_freeze_exit_0"] = code == 0
        checks["verify_freeze_code_modules_unchanged"] = bool(
            result["code_modules_unchanged"])
        checks["verify_freeze_ok"] = bool(result["ok"])
    expected = next(r["ids"] for r in part_cfg["sizes"]
                    if r["partition"] == "test")
    ok = all(checks.values())
    gate = GateResult(ok, checks, frozen, int(expected))
    (root / OUT5).mkdir(parents=True, exist_ok=True)
    write_json(root / OUT5 / "integrity_gate.json",
               {"ok": ok, "checks": checks,
                "approved": {k: v for k, v in approved.items()},
                "test_rows_read": 0})
    if not ok:
        failed = [k for k, v in checks.items() if not v]
        raise IntegrityError(f"verificacoes falharam: {failed}")
    return gate


# --------------------------------------------------------------- teste
def _load_test_rows(root: Path, gate: GateResult) -> pd.DataFrame:
    """Leia a particao de teste; so e chamada apos o portao e a trava.

    Raises:
        IntegrityError: sem portao valido ou com tamanho divergente.
    """
    if not isinstance(gate, GateResult) or not gate.ok:
        raise IntegrityError("portao de integridade ausente ou reprovado")
    root = Path(root)
    manifest = json.loads((root / "outputs/etapa_01/manifest.json")
                          .read_text("utf-8"))
    part_cfg = json.loads((root / "outputs/etapa_02/partition_config.json")
                          .read_text("utf-8"))
    frame = read_arff(root / manifest["artifact"]["local_path"]).frame
    assignments = pd.read_csv(root / part_cfg["assignment_file"]["path"])
    part = assignments.set_index("IDpol")["partition"].reindex(
        frame["IDpol"]).to_numpy()
    test = frame[part == "test"].reset_index(drop=True)
    if len(test) != gate.expected_test_rows:
        raise IntegrityError(
            f"teste com {len(test)} linhas; esperado "
            f"{gate.expected_test_rows}")
    return test


def ids_sha256(ids: pd.Series) -> str:
    """SHA-256 dos ``IDpol`` ordenados (um por linha)."""
    return compare.ids_hash(ids)


# ------------------------------------------------------------ avaliacao
def evaluate(test: pd.DataFrame, frozen: dict) -> dict:
    """Avalie o Poisson congelado e a taxa global nas linhas de teste.

    Funcao pura sobre as linhas recebidas: restaura so o estado congelado,
    faz previsoes diretas e aplica o roteiro (desvio medio de Poisson,
    diferenca pareada com bootstrap de 1.000 replicas, calibracao por
    segmento, reconciliacao e sensibilidade por perfil).

    Args:
        test: linhas de teste (contagens e exposicoes originais).
        frozen: artefato congelado.

    Returns:
        Tabelas e resumos da avaliacao.
    """
    mu = freeze.predict_frozen(frozen, test)
    report = mu.pop("report")
    ev = compare.EvalSet(test, {c: mu[c] for c in CANDIDATES}, {},
                         [("poisson", "global")])
    intervals = compare.interval_table(ev, "test", "test")
    cluster = compare.interval_table(
        ev, "test", "test_cluster_bootstrap",
        clusters=compare._profile_codes(test))
    met = compare.metric_rows(ev, "test", "test")
    cal = compare.calibration_rows(ev, "test", "test", intervals)
    rec = compare.reconcile(ev, "test", "test")
    return {"ev": ev, "metrics": met, "intervals": intervals,
            "cluster_intervals": cluster, "calibration": cal,
            "reconciliation": rec, "preprocessor_report": report,
            "conclusion": conclusion(met, intervals, cluster)}


def conclusion(met: pd.DataFrame, intervals: pd.DataFrame,
               cluster: pd.DataFrame) -> dict:
    """Classifique o resultado frente a referencia.

    Ganho sustentado se o IC 95% da diferenca pareada de desvio
    Poisson - taxa global ficar inteiramente abaixo de zero; referencia
    melhor se ficar inteiramente acima; caso contrario, inconclusivo.
    """
    def read(frame):
        row = frame[frame["indicator"] == "d_dev|poisson-global"].iloc[0]
        return (float(row["estimate"]), float(row["ci_low"]),
                float(row["ci_high"]))

    est, lo, hi = read(intervals)
    c_est, c_lo, c_hi = read(cluster)

    def verdict(low, high):
        if high < 0:
            return "ganho_sustentado"
        if low > 0:
            return "referencia_melhor"
        return "inconclusivo"

    m = met.set_index("candidate")
    return {
        "primary": {"d_dev_poisson_minus_global": est, "ci_low": lo,
                    "ci_high": hi, "verdict": verdict(lo, hi)},
        "profile_cluster_bootstrap": {
            "d_dev_poisson_minus_global": c_est, "ci_low": c_lo,
            "ci_high": c_hi, "verdict": verdict(c_lo, c_hi)},
        "mean_poisson_deviance": {
            c: float(m.loc[c, "mean_poisson_deviance"]) for c in CANDIDATES},
        "abs_rate_error": {
            c: float(m.loc[c, "abs_rate_error"]) for c in CANDIDATES},
        "note": "conclusao restrita ao benchmark; intervalos condicionados "
                "aos modelos ajustados",
    }


def validation_vs_test(calibration: pd.DataFrame,
                       validation_bands: pd.DataFrame) -> pd.DataFrame:
    """Compare as faixas de exposicao em validacao e no teste (Poisson).

    Args:
        calibration: calibracao do teste (``compare.calibration_rows``).
        validation_bands: ``exposure_band_rates.csv`` da etapa 4.

    Returns:
        Por faixa: diferenca prevista - observada na validacao e no teste,
        IC do teste e se o sinal coincide.
    """
    test = calibration[(calibration["candidate"] == "poisson")
                       & (calibration["family"] == "exposure_band")]
    rows = []
    for r in test.itertuples():
        val = validation_bands[validation_bands["band"] == r.segment]
        v = float(val["diff_poisson"].iloc[0]) if len(val) else np.nan
        rows.append({
            "band": r.segment, "test_rows": int(r.rows),
            "test_observed_rate": r.observed_rate_per_100,
            "test_predicted_rate": r.predicted_rate_per_100,
            "test_diff": r.rate_difference_pred_minus_obs,
            "test_ci_low": r.ci_low, "test_ci_high": r.ci_high,
            "test_low_support": bool(r.low_support),
            "validation_diff": v,
            "same_sign": bool(np.sign(v)
                              == np.sign(r.rate_difference_pred_minus_obs)),
            "validation_diff_inside_test_ci": bool(
                np.isfinite(r.ci_low) and r.ci_low <= v <= r.ci_high)})
    return pd.DataFrame(rows)


# --------------------------------------------------------- execucao oficial
def _write_lock(path: Path, payload: dict) -> None:
    """Crie a trava de forma atomica; falha se ja existir."""
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise AlreadyEvaluatedError(
            f"avaliacao oficial ja iniciada ({path.name})") from error
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True))


def run_official(root: Path, approved: dict | None = None,
                 package_dir: Path | None = None) -> int:
    """Execute a unica avaliacao oficial do teste.

    Ordem: recusa se ja houve execucao; portao de integridade (sem ler o
    teste); trava; leitura do teste; avaliacao; gravacao. Uma falha depois
    da leitura do teste preserva os artefatos e registra o ponto de falha;
    a trava impede nova execucao.

    Args:
        root: raiz do projeto.
        approved: referencias aprovadas (padrao: ``APPROVED``).
        package_dir: pasta dos modulos (padrao: o pacote importado).

    Returns:
        0 em sucesso.

    Raises:
        AlreadyEvaluatedError: se a trava ja existir.
        IntegrityError: se o portao reprovar (nenhuma linha de teste lida).
    """
    root = Path(root)
    out = root / OUT5
    out.mkdir(parents=True, exist_ok=True)
    lock = out / LOCK_NAME
    if lock.exists():
        raise AlreadyEvaluatedError(
            f"avaliacao oficial ja iniciada ({lock.name})")
    gate = integrity_gate(root, approved, package_dir)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _write_lock(lock, {"started_utc": started, "test_rows_expected":
                       gate.expected_test_rows,
                       "frozen_model_sha256": file_sha(
                           root / stage4.OUT4 / "frozen_model.json")})
    stage = "read_test_rows"
    record = {"status": "started", "started_utc": started,
              "package_version": __version__, "test_rows_read": False}
    t0 = time.perf_counter()
    try:
        with ResourceMonitor() as monitor:
            test = _load_test_rows(root, gate)
            record["test_rows_read"] = True
            record["test_rows"] = int(len(test))
            record["test_ids_sha256"] = ids_sha256(test["IDpol"])
            stage = "evaluate"
            result = evaluate(test, gate.frozen)
            stage = "write_outputs"
            validation_bands = pd.read_csv(
                root / stage4.OUT4 / "exposure_band_rates.csv")
            comparison = validation_vs_test(result["calibration"],
                                            validation_bands)
            _write_outputs(out, result, comparison, test)
        record.update({
            "status": "completed",
            "finished_utc": datetime.now(timezone.utc).isoformat(
                timespec="seconds"),
            "wall_s": round(time.perf_counter() - t0, 3),
            "peak_rss_mib": round(monitor.peak_rss_mib, 1),
            "process_peak_rss_mib": round(_peak_mb(), 1),
            "frozen_model_sha256": file_sha(
                root / stage4.OUT4 / "frozen_model.json"),
            "stage5_script_sha256": file_sha(
                root / stage4.OUT4 / "stage5_script.json"),
            "code_sha256": module_hashes(
                package_dir or default_package_dir(),
                ["stage5", "stage4", "compare", "freeze", "metrics",
                 "segments", "prep", "models", "arff", "profiles"]),
            "bootstrap": {"replicas": metrics.BOOTSTRAP_REPLICAS,
                          "seed": metrics.BOOTSTRAP_SEED},
            "reconciliation_all_ok": bool(
                result["reconciliation"]["ok"].all()),
            "environment": {"python": sys.version,
                            "platform": platform.platform(),
                            "numpy": np.__version__,
                            "pandas": pd.__version__}})
        write_json(out / "official_run.json", record)
        write_checksums(out)
        return 0
    except Exception as error:
        record.update({"status": "failed", "failed_at": stage,
                       "error": f"{type(error).__name__}: {error}"[:1500],
                       "traceback": traceback.format_exc()[-3000:]})
        write_json(out / "official_run.json", record)
        raise


def _write_outputs(out: Path, result: dict, comparison: pd.DataFrame,
                   test: pd.DataFrame) -> None:
    """Grave so agregados (nenhuma linha de teste e persistida)."""
    write_table(out / "test_metrics.csv", result["metrics"])
    write_table(out / "test_intervals.csv", pd.concat(
        [result["intervals"], result["cluster_intervals"]],
        ignore_index=True))
    write_table(out / "test_calibration.csv", result["calibration"])
    write_table(out / "test_reconciliation.csv", result["reconciliation"])
    write_table(out / "validation_vs_test_bands.csv", comparison)
    write_table(out / "panel.csv", panel_table(result["calibration"]))
    write_json(out / "conclusion.json", result["conclusion"])
    write_json(out / "test_population.json", {
        "rows": int(len(test)),
        "exposure": float(test["Exposure"].sum()),
        "observed_claims": float(test["ClaimNb"].sum()),
        "observed_zeros": int((test["ClaimNb"] == 0).sum()),
        "ids_sha256": ids_sha256(test["IDpol"]),
        "preprocessor_report": result["preprocessor_report"]})


def panel_table(calibration: pd.DataFrame) -> pd.DataFrame:
    """Painel de calibracao no formato do roteiro (por candidato)."""
    cols = {"candidate": "candidate", "family": "family",
            "segment": "segment", "rows": "records", "exposure":
            "policy_years", "observed_claims": "observed_claims",
            "predicted_claims": "predicted_claims",
            "observed_rate_per_100": "observed_rate_per_100",
            "predicted_rate_per_100": "predicted_rate_per_100",
            "rate_difference_pred_minus_obs": "pred_minus_obs",
            "ci_low": "ci_low", "ci_high": "ci_high",
            "low_support": "low_support"}
    return calibration[list(cols)].rename(columns=cols).reset_index(
        drop=True)


def write_checksums(out: Path) -> None:
    """Grave ``SHA256SUMS_etapa05.txt`` dos arquivos agregados."""
    lines = []
    for path in sorted(Path(out).iterdir()):
        if path.is_file() and path.name not in (
                "SHA256SUMS_etapa05.txt", LOCK_NAME):
            lines.append(f"{file_sha(path)} *{path.name}")
    (Path(out) / "SHA256SUMS_etapa05.txt").write_text(
        "\n".join(lines) + "\n", encoding="ascii")


def check_only(root: Path) -> int:
    """Execute so o portao de integridade (verificacao previa).

    Nao le linhas de teste nem cria a trava.

    Args:
        root: raiz do projeto.

    Returns:
        0 se o portao passar; 1 caso contrario.
    """
    try:
        gate = integrity_gate(Path(root))
    except IntegrityError as error:
        print("FALHA:", error)
        return 1
    print("portao de integridade ok:", all(gate.checks.values()),
          "| verificacoes:", len(gate.checks), "| linhas de teste lidas: 0")
    return 0


def verify_official(root: Path) -> int:
    """Verifique offline os agregados da execucao oficial.

    Confere o registro, os hashes dos arquivos e a reconciliacao, sem ler
    o ARFF nem o teste.

    Args:
        root: raiz do projeto.

    Returns:
        0 se tudo conferir; 1 caso contrario.
    """
    out = Path(root) / OUT5
    record = json.loads((out / "official_run.json").read_text("utf-8"))
    sums = {}
    for line in (out / "SHA256SUMS_etapa05.txt").read_text("ascii"
                                                           ).splitlines():
        digest, name = line.split(" *", 1)
        sums[name] = digest
    checks = {
        "status_completed": record["status"] == "completed",
        "hashes_match": all(file_sha(out / n) == d
                            for n, d in sums.items()),
        "reconciliation_ok": record["reconciliation_all_ok"],
        "all_files_listed": {p.name for p in out.iterdir() if p.is_file()
                             and p.name not in ("SHA256SUMS_etapa05.txt",
                                                LOCK_NAME)} == set(sums)}
    ok = all(checks.values())
    print("verify-stage5 ok:", ok, checks)
    return 0 if ok else 1
