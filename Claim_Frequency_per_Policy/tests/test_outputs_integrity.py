"""Confere as listas SHA256SUMS* de ``outputs/`` contra os arquivos reais.

Cada pasta selada deve ter todos os arquivos listados com o hash correto e
nenhum arquivo fora da lista (exceto a propria lista e a trava oficial).
Registros de execucao de notebooks ficam fora das pastas seladas.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
LISTS = sorted(OUTPUTS.glob("etapa_0*/SHA256SUMS*.txt")) if (
    OUTPUTS.exists()) else []
ALLOWED_UNLISTED = {"official_run.lock"}
# Excecoes legitimas por pasta (fora da lista por desenho):
# * ``cost.json`` (etapas 1 e 2): tempo e memoria variam a cada execucao;
#   os relatorios dizem que so ele varia entre duas execucoes.
# * ``manifest.json`` e ``openml_metadata.json`` (etapa 1): produzidos pela
#   aquisicao (``acquire``), nao pela auditoria que gera a lista.
FOLDER_EXCEPTIONS = {
    "etapa_01": {"cost.json", "manifest.json", "openml_metadata.json"},
    "etapa_02": {"cost.json"},
}


def _parse(path: Path) -> dict[str, str]:
    """Le ``<hash> *<nome>``; aceita CRLF ou LF."""
    entries = {}
    for line in path.read_bytes().decode("ascii").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(" *", 1)
        entries[name] = digest
    return entries


def test_outputs_exist_or_skip():
    if not OUTPUTS.exists():
        pytest.skip("outputs/ ausente")
    assert LISTS, "nenhuma lista SHA256SUMS* encontrada"


@pytest.mark.parametrize("sums", LISTS, ids=lambda p: p.parent.name)
def test_sums_list_matches_folder(sums):
    entries = _parse(sums)
    assert entries, f"lista vazia: {sums}"
    folder = sums.parent
    for name, digest in entries.items():
        target = folder / name
        assert target.is_file(), f"arquivo listado ausente: {name}"
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        assert actual == digest, f"hash divergente: {name}"
    present = {p.name for p in folder.iterdir() if p.is_file()}
    unlisted = (present - set(entries) - {sums.name} - ALLOWED_UNLISTED
                - FOLDER_EXCEPTIONS.get(folder.name, set()))
    assert not unlisted, f"arquivos fora da lista: {sorted(unlisted)}"


def test_parse_accepts_crlf_and_lf(tmp_path):
    for eol in (b"\n", b"\r\n"):
        path = tmp_path / "SHA.txt"
        path.write_bytes(b"a" * 64 + b" *x.csv" + eol + b"b" * 64
                         + b" *y.json" + eol)
        assert _parse(path) == {"x.csv": "a" * 64, "y.json": "b" * 64}
