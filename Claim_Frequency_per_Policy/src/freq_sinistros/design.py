"""Orquestracao da etapa 2: particoes, verificacoes e protocolo.

Regra de acesso ao teste: as saidas desta etapa usam do conjunto de teste
somente o identificador e a contagem de IDs. Desfechos e exposicao so sao
lidos para o treino; a validacao contribui com registros e exposicao, nunca
com ``ClaimNb``.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from freq_sinistros import __version__, partition, profiles, segments
from freq_sinistros.acquire import hash_file
from freq_sinistros.arff import read_arff
from freq_sinistros.audit import _jsonable, _peak_mb, safe_ratio
from freq_sinistros.protocol import protocol_config

SPLITS_DIR = Path("data") / "splits"
OUTPUTS_DIR = Path("outputs") / "etapa_02"


def attach_partition(frame: pd.DataFrame,
                     assignments: pd.DataFrame) -> pd.Series:
    """Devolva a particao de cada linha de ``frame`` (alinhada por ID)."""
    lookup = assignments.set_index("IDpol")["partition"]
    return pd.Series(lookup.reindex(frame["IDpol"]).to_numpy(),
                     index=frame.index)


def train_overview(train: pd.DataFrame) -> dict:
    """Totais do treino: registros, exposicao, sinistros e zeros."""
    claims = train["ClaimNb"].to_numpy(dtype="float64")
    exposure = train["Exposure"].to_numpy(dtype="float64")
    rows = int(len(train))
    zeros = int((claims == 0).sum())
    return {
        "rows": rows, "exposure": float(exposure.sum()),
        "claims": float(claims.sum()), "zeros": zeros,
        "zeros_share": zeros / rows if rows else float("nan"),
        "rate_per_100_policy_years": safe_ratio(
            100 * claims.sum(), exposure.sum()),
        "rows_claimnb_gt_4": int((claims > 4).sum()),
        "rows_exposure_gt_1": int((exposure > 1).sum()),
        "rows_exposure_le_0.02": int((exposure <= 0.02).sum()),
    }


def development_outputs(frame: pd.DataFrame,
                        assignments: pd.DataFrame) -> dict:
    """Calcule todas as evidencias que usam dados de desenvolvimento.

    O teste contribui so com a contagem de IDs: alterar ``ClaimNb`` ou
    ``Exposure`` de linhas de teste nao muda nenhuma saida (testado).

    Args:
        frame: tabela completa lida do bruto.
        assignments: particao principal.

    Returns:
        Dicionario com tabelas e resumos.
    """
    part = attach_partition(frame, assignments)
    train = frame[part == "train"]
    validation = frame[part == "validation"]
    dev_mask = part.isin(["train", "validation"])
    seg_train = segments.segment_table(train, with_outcome=True)
    seg_val = segments.segment_table(
        validation.drop(columns=["ClaimNb"]).assign(ClaimNb=np.nan),
        with_outcome=False)
    support = seg_train.rename(columns={
        "rows": "train_rows", "exposure": "train_exposure",
        "claims": "train_claims",
        "observed_rate_per_100": "train_observed_rate_per_100",
        "low_support": "train_low_support"}).merge(
        seg_val.rename(columns={
            "rows": "validation_rows", "exposure": "validation_exposure",
            "low_support": "validation_low_support"}),
        on=["family", "segment"], how="outer")
    support = segments.sort_segments(support)
    support["train_low_support"] = support["train_low_support"].fillna(True)
    support["validation_low_support"] = support[
        "validation_low_support"].fillna(True)
    grouped = profiles.grouped_assignments(frame, dev_mask)
    recon = pd.concat([
        segments.reconcile_segments(seg_train, train, True)
        .assign(partition="train"),
        segments.reconcile_segments(
            seg_val, validation.assign(ClaimNb=np.nan), False)
        .assign(partition="validation")], ignore_index=True)
    return {
        "sizes": partition.partition_sizes(assignments),
        "train_overview": train_overview(train),
        "validation_overview": {
            "rows": int(len(validation)),
            "exposure": float(validation["Exposure"].sum()),
            "note": "sem desfechos nesta etapa"},
        "test_overview": {
            "ids": int((part == "test").sum()),
            "note": "so integridade e tamanho; desfechos reservados"},
        "segment_support": support,
        "segment_reconciliation": recon,
        "profile_coincidence": profiles.profile_coincidence(
            train, validation),
        "grouped_assignments": grouped,
        "grouped_summary": profiles.grouped_summary(frame, grouped),
    }


def _write_json(path: Path, payload: Any) -> None:
    """Grave JSON deterministico (chaves ordenadas, UTF-8)."""
    path.write_text(json.dumps(_jsonable(payload), indent=2,
                               ensure_ascii=False, sort_keys=True),
                    encoding="utf-8")


def _load_verified(root: Path):
    """Verifique o ARFF contra o manifesto da etapa 1 e leia-o.

    Raises:
        ValueError: se tamanho ou SHA-256 divergirem do manifesto.
    """
    manifest = json.loads(
        (root / "outputs" / "etapa_01" / "manifest.json").read_text("utf-8"))
    raw = root / manifest["artifact"]["local_path"]
    digest = hash_file(raw)
    if (digest["sha256"] != manifest["artifact"]["sha256"]
            or digest["size_bytes"] != manifest["artifact"]["size_bytes"]):
        raise ValueError("ARFF diverge do manifesto da etapa 1")
    return manifest, digest, read_arff(raw)


def run_partition(root: Path) -> int:
    """Materialize particoes e evidencias da etapa 2 (sem rede).

    Efeitos colaterais: grava ``data/splits/*.csv`` (atribuicoes locais) e
    ``outputs/etapa_02/*``.

    Args:
        root: raiz do projeto.

    Returns:
        0 em sucesso; 2 se o ARFF divergir do manifesto.
    """
    root = Path(root)
    stages = []
    t0 = time.perf_counter()
    try:
        manifest, digest, result = _load_verified(root)
    except ValueError as error:
        print(f"ERRO: {error}")
        return 2
    stages.append({"stage": "verify_and_read",
                   "wall_s": round(time.perf_counter() - t0, 3),
                   "peak_rss_mib_cumulative": round(_peak_mb(), 1)})
    frame = result.frame
    t0 = time.perf_counter()
    assignments = partition.build_assignments(frame["IDpol"])
    checks = partition.check_partition(assignments, frame["IDpol"])
    digest_main = partition.assignment_hash(assignments)
    shuffled = frame["IDpol"].sample(frac=1.0, random_state=7)
    order_ok = partition.assignment_hash(
        partition.build_assignments(shuffled)) == digest_main
    repeat_ok = partition.assignment_hash(
        partition.build_assignments(frame["IDpol"])) == digest_main
    stages.append({"stage": "partition_and_checks",
                   "wall_s": round(time.perf_counter() - t0, 3),
                   "peak_rss_mib_cumulative": round(_peak_mb(), 1)})
    t0 = time.perf_counter()
    dev = development_outputs(frame, assignments)
    stages.append({"stage": "development_evidence",
                   "wall_s": round(time.perf_counter() - t0, 3),
                   "peak_rss_mib_cumulative": round(_peak_mb(), 1)})
    splits = root / SPLITS_DIR
    out = root / OUTPUTS_DIR
    splits.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    main_path = splits / "partition_assignments.csv"
    main_path.write_bytes(partition.assignment_bytes(assignments))
    grouped_path = splits / "grouped_dev_assignments.csv"
    grouped_path.write_bytes(
        dev["grouped_assignments"].to_csv(index=False,
                                          lineterminator="\n").encode())
    config = {
        "stage": 2, "package_version": __version__,
        "production_eligible": False,
        "arff": {"path": manifest["artifact"]["local_path"],
                 "sha256": digest["sha256"],
                 "size_bytes": digest["size_bytes"],
                 "matches_stage1_manifest": True},
        "rule": partition.RULE_TEXT, "rule_modulus": partition.MODULUS,
        "rule_parameters": {"train_end": partition.TRAIN_END,
                            "validation_end": partition.VALIDATION_END,
                            "prefix": partition.RULE_PREFIX},
        "assignment_file": {
            "path": str(SPLITS_DIR / "partition_assignments.csv").replace(
                "\\", "/"),
            "columns": ["IDpol", "partition"],
            "sha256_of_ordered_assignments": digest_main,
            "rows": int(len(assignments))},
        "sizes": dev["sizes"].to_dict(orient="records"),
        "checks": {**checks, "invariant_to_row_order": bool(order_ok),
                   "repeatable": bool(repeat_ok),
                   "token_1e+05_canonical_id": partition.canonical_id(
                       "1e+05")},
        "grouped_assignment_file": {
            "path": str(SPLITS_DIR / "grouped_dev_assignments.csv").replace(
                "\\", "/"),
            "sha256": hashlib.sha256(grouped_path.read_bytes()).hexdigest()},
    }
    _write_json(out / "partition_config.json", config)
    dev["sizes"].to_csv(out / "partition_sizes.csv", index=False,
                        lineterminator="\n", float_format="%.12g")
    _write_json(out / "overview.json", {
        "train": dev["train_overview"],
        "validation": dev["validation_overview"],
        "test": dev["test_overview"]})
    dev["segment_support"].to_csv(out / "segment_support.csv", index=False,
                                  lineterminator="\n", float_format="%.12g")
    dev["segment_reconciliation"].to_csv(
        out / "segment_reconciliation.csv", index=False,
        lineterminator="\n", float_format="%.12g")
    _write_json(out / "profile_coincidence_dev.json",
                dev["profile_coincidence"])
    _write_json(out / "grouped_partition_summary.json",
                dev["grouped_summary"])
    _write_json(out / "protocol_config.json", protocol_config())
    import psutil

    _write_json(out / "cost.json", {
        "stages": stages,
        "total_wall_s": round(sum(s["wall_s"] for s in stages), 3),
        "peak_rss_mib": round(_peak_mb(), 1),
        "environment": {"python": sys.version,
                        "platform": platform.platform(),
                        "numpy": np.__version__, "pandas": pd.__version__,
                        "psutil": psutil.__version__},
        "note": "peak_wset do processo; CPU; sem rede"})
    sizes = dev["sizes"]
    print(sizes.to_string(index=False))
    print("assignment_sha256:", digest_main)
    print("checks:", checks, "order_ok:", order_ok, "repeat_ok:", repeat_ok)
    print("peak_rss_mib:", round(_peak_mb(), 1))
    ok = (checks["full_coverage"] and checks["disjoint"] and order_ok
          and repeat_ok)
    return 0 if ok else 1


def verify_partition(root: Path) -> int:
    """Recrie as atribuicoes a partir do ARFF e compare com o registrado.

    Compara o hash recalculado com ``partition_config.json`` e com os
    bytes do arquivo local ``partition_assignments.csv``. Nao grava nada.

    Args:
        root: raiz do projeto.

    Returns:
        0 se tudo conferir; 2 se o ARFF divergir; 1 se as atribuicoes
        divergirem.
    """
    root = Path(root)
    config = json.loads(
        (root / OUTPUTS_DIR / "partition_config.json").read_text("utf-8"))
    try:
        _, digest, result = _load_verified(root)
    except ValueError as error:
        print(f"ERRO: {error}")
        return 2
    assignments = partition.build_assignments(result.frame["IDpol"])
    recomputed = partition.assignment_hash(assignments)
    expected = config["assignment_file"]["sha256_of_ordered_assignments"]
    on_disk = hashlib.sha256((root / config["assignment_file"]["path"])
                             .read_bytes()).hexdigest()
    ok = (recomputed == expected == on_disk
          and digest["sha256"] == config["arff"]["sha256"])
    print("hash recalculado:", recomputed)
    print("confere com config e arquivo local:", ok)
    return 0 if ok else 1
