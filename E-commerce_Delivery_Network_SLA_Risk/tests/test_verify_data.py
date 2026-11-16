"""Tests for scripts/verify_data.py using a synthetic ZIP (no real data)."""

from __future__ import annotations

import hashlib
import importlib.util
import tempfile
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_data.py"
_spec = importlib.util.spec_from_file_location("verify_data", SCRIPT)
verify_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verify_data)


@pytest.fixture
def tmp_path():
    """Local override: the default pytest temp root is not writable in this Windows setup."""
    with tempfile.TemporaryDirectory(prefix="verify_data_") as directory:
        yield Path(directory)


def make_dataset(tmp_path: Path, *, readme_hash: str | None = None, members=None):
    if members is None:
        members = {name: f"a,b\n1,{i}\n" for i, name in enumerate(verify_data.EXPECTED_CSVS)}
    zip_path = tmp_path / verify_data.ZIP_NAME
    with zipfile.ZipFile(zip_path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    shown = readme_hash if readme_hash is not None else digest.upper()
    (tmp_path / "README.md").write_text(
        f"- Fonte: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce\n- SHA-256: `{shown}`\n",
        encoding="utf-8",
    )
    return zip_path, digest


def test_check_only_passes_with_uppercase_readme_hash(tmp_path, capsys):
    make_dataset(tmp_path)
    assert verify_data.main(["--data-dir", str(tmp_path)]) == 0
    assert "OK" in capsys.readouterr().out
    assert not (tmp_path / "csv").exists()


def test_extract_writes_only_expected_csvs(tmp_path):
    members = {name: "x\n1\n" for name in verify_data.EXPECTED_CSVS}
    members["extra.csv"] = "x\n"
    make_dataset(tmp_path, members=members)
    assert verify_data.main(["--data-dir", str(tmp_path), "--extract"]) == 0
    written = sorted(p.name for p in (tmp_path / "csv").iterdir())
    assert written == sorted(verify_data.EXPECTED_CSVS)


def test_missing_zip_exits_2_and_points_to_kaggle(tmp_path, capsys):
    (tmp_path / "README.md").write_text(
        "SHA-256: `" + "a" * 64 + "`\nhttps://www.kaggle.com/datasets/olistbr/brazilian-ecommerce\n",
        encoding="utf-8",
    )
    assert verify_data.main(["--data-dir", str(tmp_path)]) == 2
    assert "kaggle.com" in capsys.readouterr().err


def test_hash_mismatch_exits_1_without_extracting(tmp_path, capsys):
    make_dataset(tmp_path, readme_hash="0" * 64)
    assert verify_data.main(["--data-dir", str(tmp_path), "--extract"]) == 1
    assert "mismatch" in capsys.readouterr().err
    assert not (tmp_path / "csv").exists()


def test_readme_without_hash_exits_2(tmp_path):
    make_dataset(tmp_path)
    (tmp_path / "README.md").write_text("no hash here\n", encoding="utf-8")
    assert verify_data.main(["--data-dir", str(tmp_path)]) == 2


def test_path_traversal_member_is_refused(tmp_path):
    base = tmp_path / "olist"
    base.mkdir()
    members = {name: "x\n" for name in verify_data.EXPECTED_CSVS}
    members["../evil.csv"] = "x\n"
    make_dataset(base, members=members)
    assert verify_data.main(["--data-dir", str(base), "--extract"]) == 1
    assert not (tmp_path / "evil.csv").exists()
    assert not (base / "csv").exists()


def test_archive_missing_expected_csv_is_refused(tmp_path):
    make_dataset(tmp_path, members={"olist_orders_dataset.csv": "x\n"})
    assert verify_data.main(["--data-dir", str(tmp_path), "--extract"]) == 1


def test_sha256_of_matches_hashlib(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc" * 1000)
    assert verify_data.sha256_of(path) == hashlib.sha256(b"abc" * 1000).hexdigest()
