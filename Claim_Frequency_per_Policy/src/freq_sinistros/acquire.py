"""Aquisicao do artefato bruto do OpenML e geracao do manifesto de origem.

O arquivo baixado e preservado byte a byte. Este modulo nunca altera,
reescreve ou decodifica o conteudo do artefato: apenas o grava em disco e
calcula tamanho e somas de verificacao sobre os bytes recebidos.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DATASET_ID = 41214
DATASET_NAME = "freMTPL2freq"
DATASET_VERSION = "1"
METADATA_URL = f"https://www.openml.org/api/v1/json/data/{DATASET_ID}"
FEATURES_URL = f"https://www.openml.org/api/v1/json/data/features/{DATASET_ID}"
QUALITIES_URL = (
    f"https://www.openml.org/api/v1/json/data/qualities/{DATASET_ID}"
)
USER_AGENT = "freq-sinistros-audit/0.1"
CHUNK_BYTES = 1024 * 1024
TIMEOUT_S = 120


class ChecksumMismatchError(RuntimeError):
    """O MD5 dos bytes baixados difere do MD5 declarado pela fonte."""


def _open(url: str):
    """Abre uma URL HTTP(S) e devolve o objeto de resposta.

    Args:
        url: endereco a ser requisitado.

    Raises:
        urllib.error.URLError: em falha de rede ou status HTTP de erro.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(request, timeout=TIMEOUT_S)


def fetch_json(url: str) -> dict:
    """Baixa e decodifica um documento JSON.

    Args:
        url: endereco do documento.

    Returns:
        O documento decodificado.
    """
    with _open(url) as response:
        return json.loads(response.read().decode("utf-8"))


def download_file(
    url: str, destination: Path, expected_md5: str | None = None
) -> dict:
    """Baixa ``url`` para ``destination`` em fluxo, sem alterar os bytes.

    O conteudo e gravado primeiro em um arquivo temporario e so entao
    renomeado, de modo que uma falha nao deixa um artefato parcial. Se
    ``expected_md5`` for dado, o arquivo so chega ao caminho final quando o
    MD5 dos bytes baixados conferir; caso contrario o temporario e removido.

    Args:
        url: endereco do artefato.
        destination: caminho final do arquivo local.
        expected_md5: MD5 declarado pela fonte para esta representacao.

    Returns:
        Dicionario com ``size_bytes``, ``sha256``, ``md5``, ``elapsed_s``,
        ``final_url`` e ``http_headers``.

    Raises:
        FileExistsError: se ``destination`` ja existir (nao sobrescreve).
        ChecksumMismatchError: se o MD5 calculado diferir de
            ``expected_md5``; nada e deixado no caminho final.
    """
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"{destination} ja existe; nao sobrescrito.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    sha, md5, size = hashlib.sha256(), hashlib.md5(), 0
    start = time.perf_counter()
    with _open(url) as response, open(partial, "wb") as handle:
        final_url = response.geturl()
        headers = {
            key: response.headers[key]
            for key in ("Content-Length", "Content-Type", "ETag",
                        "Last-Modified")
            if response.headers.get(key) is not None
        }
        while True:
            block = response.read(CHUNK_BYTES)
            if not block:
                break
            handle.write(block)
            sha.update(block)
            md5.update(block)
            size += len(block)
    if expected_md5 is not None and md5.hexdigest() != expected_md5:
        partial.unlink()
        raise ChecksumMismatchError(
            f"MD5 divergente: declarado {expected_md5}, "
            f"calculado {md5.hexdigest()}")
    partial.replace(destination)
    return {
        "size_bytes": size,
        "sha256": sha.hexdigest(),
        "md5": md5.hexdigest(),
        "elapsed_s": round(time.perf_counter() - start, 3),
        "final_url": final_url,
        "http_headers": headers,
    }


def hash_file(path: Path) -> dict:
    """Calcula tamanho, SHA-256 e MD5 de um arquivo local.

    Args:
        path: arquivo a verificar.

    Returns:
        Dicionario com ``size_bytes``, ``sha256`` e ``md5``.
    """
    sha, md5, size = hashlib.sha256(), hashlib.md5(), 0
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
            sha.update(block)
            md5.update(block)
            size += len(block)
    return {"size_bytes": size, "sha256": sha.hexdigest(),
            "md5": md5.hexdigest()}


def acquire(root: Path, outputs: Path | None = None) -> dict:
    """Executa a aquisicao completa e grava o manifesto de origem.

    Efeitos colaterais: acessa a rede, grava o ARFF em ``data/raw`` e os
    metadados e o manifesto em ``outputs/etapa_01``. O ARFF so e aceito e o
    manifesto so e gravado se o MD5 declarado pelo OpenML conferir.

    Args:
        root: raiz do projeto.
        outputs: pasta de saida do manifesto (padrao ``outputs/etapa_01``).

    Returns:
        O manifesto como dicionario.

    Raises:
        FileExistsError: se o ARFF local ja existir (preservado).
        ChecksumMismatchError: se o MD5 declarado faltar ou divergir.
    """
    root = Path(root)
    outputs = Path(outputs) if outputs else root / "outputs" / "etapa_01"
    raw = root / "data" / "raw" / f"{DATASET_NAME}_v{DATASET_VERSION}.arff"
    if raw.exists():
        raise FileExistsError(f"{raw} ja existe; nao sobrescrito.")
    outputs.mkdir(parents=True, exist_ok=True)
    meta = fetch_json(METADATA_URL)
    features = fetch_json(FEATURES_URL)
    qualities = fetch_json(QUALITIES_URL)
    (outputs / "openml_metadata.json").write_text(
        json.dumps({"data": meta, "features": features,
                    "qualities": qualities}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    desc = meta["data_set_description"]
    url = desc["url"]
    acquired_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    declared = desc.get("md5_checksum")
    if not declared:
        raise ChecksumMismatchError("metadado sem md5_checksum declarado")
    dl = download_file(url, raw, expected_md5=declared)
    manifest = {
        "dataset": {
            "name": desc["name"], "version": desc["version"],
            "dataset_id": int(desc["id"]), "file_id": desc["file_id"],
            "licence": desc["licence"], "format": desc["format"],
            "upload_date": desc.get("upload_date"),
            "status": desc.get("status"),
            "original_data_url": desc.get("original_data_url"),
            "description": desc.get("description"),
        },
        "source": {
            "metadata_url": METADATA_URL,
            "features_url": FEATURES_URL,
            "qualities_url": QUALITIES_URL,
            "download_url_declared": desc["url"],
            "download_url_final": dl["final_url"],
            "parquet_url_declared_not_used": desc.get("parquet_url"),
        },
        "artifact": {
            "representation": "ARFF (rota registrada no metadado 'url')",
            "local_path": str(raw.relative_to(root)).replace("\\", "/"),
            "local_path_absolute": str(raw.resolve()),
            "acquired_at_utc": acquired_at,
            "size_bytes": dl["size_bytes"],
            "sha256": dl["sha256"],
            "md5": dl["md5"],
            "http_headers": dl["http_headers"],
            "download_elapsed_s": dl["elapsed_s"],
        },
        "checksum_verification": {
            "algorithm": "md5",
            "declared_by_openml": declared,
            "computed_on_downloaded_bytes": dl["md5"],
            "match": declared == dl["md5"],
            "note": "O OpenML publica apenas MD5, referente a representacao "
                    "ARFF; o SHA-256 foi calculado localmente.",
        },
        "environment": {
            "python": sys.version, "platform": platform.platform(),
        },
    }
    (outputs / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8")
    return manifest
