"""Testes de aquisicao (via file://), CLI e auditoria completa sintetica."""

import hashlib
import json

import pytest

from freq_sinistros import acquire
from freq_sinistros.audit import run_audit
from freq_sinistros.cli import build_parser, main
from tests.test_audit import ARFF


def test_download_preserves_bytes_and_hashes(tmp_path):
    source = tmp_path / "src.arff"
    payload = ARFF.encode("utf-8")
    source.write_bytes(payload)
    dest = tmp_path / "raw" / "out.arff"
    info = acquire.download_file(source.as_uri(), dest)
    assert dest.read_bytes() == payload
    assert info["size_bytes"] == len(payload)
    assert info["sha256"] == hashlib.sha256(payload).hexdigest()
    assert info["md5"] == hashlib.md5(payload).hexdigest()
    assert acquire.hash_file(dest)["sha256"] == info["sha256"]
    with pytest.raises(FileExistsError):
        acquire.download_file(source.as_uri(), dest)


def _toy_project(tmp_path, corrupt=False):
    raw = tmp_path / "data" / "raw" / "toy.arff"
    raw.parent.mkdir(parents=True)
    raw.write_text(ARFF, encoding="utf-8")
    digest = acquire.hash_file(raw)
    if corrupt:
        digest["sha256"] = "0" * 64
    out = tmp_path / "outputs" / "etapa_01"
    out.mkdir(parents=True)
    manifest = {"artifact": {
        "local_path": "data/raw/toy.arff", "sha256": digest["sha256"],
        "size_bytes": digest["size_bytes"], "download_elapsed_s": 0.0}}
    (out / "manifest.json").write_text(json.dumps(manifest), "utf-8")
    return out


def test_run_audit_writes_outputs(tmp_path):
    out = _toy_project(tmp_path)
    assert run_audit(tmp_path) == 0
    for name in ("audit_summary.json", "schema_summary.csv", "cost.json",
                 "exposure_bands.csv", "reconciliation.csv"):
        assert (out / name).exists()
    summary = json.loads((out / "audit_summary.json").read_text("utf-8"))
    assert summary["raw_file"]["manifest_match"] is True


def test_run_audit_refuses_hash_mismatch(tmp_path):
    _toy_project(tmp_path, corrupt=True)
    assert run_audit(tmp_path) == 2


def test_cli_parser_and_audit_command(tmp_path):
    assert build_parser().parse_args(["audit"]).command == "audit"
    _toy_project(tmp_path)
    assert main(["--root", str(tmp_path), "audit"]) == 0


def _fake_openml(monkeypatch, source, declared_md5):
    meta = {"data_set_description": {
        "id": "1", "name": "toy", "version": "1", "file_id": "9",
        "licence": "CC0", "format": "ARFF", "url": source.as_uri(),
        "md5_checksum": declared_md5}}
    monkeypatch.setattr(acquire, "fetch_json", lambda url: meta)


def test_acquire_rejects_md5_mismatch_then_accepts_retry(
        tmp_path, monkeypatch, capsys):
    source = tmp_path / "src.arff"
    source.write_bytes(ARFF.encode("utf-8"))
    good = hashlib.md5(ARFF.encode("utf-8")).hexdigest()
    root = tmp_path / "proj"
    raw = root / "data" / "raw" / "freMTPL2freq_v1.arff"
    _fake_openml(monkeypatch, source, "0" * 32)
    with pytest.raises(acquire.ChecksumMismatchError):
        acquire.acquire(root)
    assert not raw.exists()
    assert not list(raw.parent.glob("*.part"))
    assert not (root / "outputs" / "etapa_01" / "manifest.json").exists()
    assert main(["--root", str(root), "acquire"]) == 3
    assert "ERRO" in capsys.readouterr().out
    _fake_openml(monkeypatch, source, good)
    manifest = acquire.acquire(root)
    assert manifest["checksum_verification"]["match"] is True
    assert raw.read_bytes() == ARFF.encode("utf-8")


def test_acquire_preserves_existing_verified_raw(tmp_path, monkeypatch):
    source = tmp_path / "src.arff"
    source.write_bytes(b"outro conteudo")
    root = tmp_path / "proj"
    raw = root / "data" / "raw" / "freMTPL2freq_v1.arff"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(ARFF.encode("utf-8"))
    _fake_openml(monkeypatch, source, hashlib.md5(b"outro conteudo")
                 .hexdigest())
    with pytest.raises(FileExistsError):
        acquire.acquire(root)
    assert raw.read_bytes() == ARFF.encode("utf-8")
    assert main(["--root", str(root), "acquire"]) == 3


def test_acquire_requires_declared_md5(tmp_path, monkeypatch):
    source = tmp_path / "src.arff"
    source.write_bytes(b"x")
    _fake_openml(monkeypatch, source, None)
    with pytest.raises(acquire.ChecksumMismatchError):
        acquire.acquire(tmp_path / "proj")
