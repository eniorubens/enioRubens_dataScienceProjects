"""Tests of scripts/fetch_data.py and scripts/make_public_snapshot.py (temporary folders, no network)."""

import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fetch = load("fetch_data")
snap = load("make_public_snapshot")


def test_verify_hash(tmp_path):
    f = tmp_path / "a.bin"
    assert fetch.verify(f, "0" * 64) == "missing"
    f.write_bytes(b"abc")
    good = hashlib.sha256(b"abc").hexdigest()
    assert fetch.verify(f, good) == "ok"
    assert fetch.verify(f, good.upper()) == "ok"
    assert fetch.verify(f, "0" * 64) == "mismatch"


def test_ensure_never_touches_network_when_present_or_check_only(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "ROOT", tmp_path)
    def boom(*a, **k):
        raise AssertionError("network used")
    monkeypatch.setattr(fetch, "download", boom)
    f = tmp_path / "x.7z"
    asset = fetch.Asset("x", f, hashlib.sha256(b"abc").hexdigest(), "http://example.invalid/x")
    assert fetch.ensure(asset, check_only=True) is False  # missing, check-only: no download
    f.write_bytes(b"abc")
    assert fetch.ensure(asset, check_only=False) is True
    f.write_bytes(b"tampered")
    assert fetch.ensure(asset, check_only=False) is False  # mismatch is never overwritten silently


def test_ensure_rejects_bad_download(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(fetch, "ROOT", tmp_path)
    f = tmp_path / "x.7z"
    monkeypatch.setattr(fetch, "download", lambda url, target, timeout=0: target.write_bytes(b"evil"))
    asset = fetch.Asset("x", f, hashlib.sha256(b"abc").hexdigest(), "http://example.invalid/x")
    assert fetch.ensure(asset, check_only=False) is False
    assert not f.exists()


def test_sintef_missing_prints_manual_instructions(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(fetch, "ROOT", tmp_path)
    asset = fetch.Asset("sintef", tmp_path / "solomon-100.zip", "ab" * 32, None)
    assert fetch.ensure(asset, check_only=False) is False
    out = capsys.readouterr().out
    assert "sintef.no/projectweb/top/vrptw/100-customers" in out and "ab" * 32 in out


def test_main_check_only_fails_without_data(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "ROOT", tmp_path)
    monkeypatch.setattr(
        fetch, "ASSETS", tuple(fetch.Asset(a.key, tmp_path / a.path.name, a.sha256, a.url) for a in fetch.ASSETS)
    )
    assert fetch.main(["--check-only"]) == 1


def test_expected_hashes_match_manifests():
    root = SCRIPTS.parent
    text = (root / "data/raw/sintef/MANIFEST.md").read_text(encoding="utf-8") + (
        root / "data/raw/cvrplib_solomon_100/MANIFEST.md"
    ).read_text(encoding="utf-8")
    for asset in fetch.ASSETS:
        assert asset.sha256 in text


def test_is_excluded():
    assert snap.is_excluded("data/raw/sintef/solomon-100.zip")
    assert snap.is_excluded("data/reference/cvrplib_bks_100.csv".replace("/", chr(92)))
    assert not snap.is_excluded("data/reference/sintef_bks_100.csv")
    assert not snap.is_excluded("data/raw/sintef/MANIFEST.md")


def make_tree(root):
    for rel in (*snap.EXCLUDED, "data/reference/sintef_bks_100.csv", "README.md", ".gitignore"):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x", encoding="utf-8")


def test_remove_and_gitignore_and_assert(tmp_path):
    make_tree(tmp_path)
    with pytest.raises(SystemExit):
        snap.assert_clean(tmp_path)
    removed = snap.remove_excluded(tmp_path)
    assert sorted(removed) == sorted(snap.EXCLUDED)
    snap.write_gitignore(tmp_path)
    snap.assert_clean(tmp_path)
    assert (tmp_path / "data/reference/sintef_bks_100.csv").exists()
    ignore = (tmp_path / ".gitignore").read_text(encoding="utf-8")
    assert all(f"/{rel}" in ignore for rel in snap.EXCLUDED)


def test_check_target(tmp_path):
    private = tmp_path / "private"
    private.mkdir()
    with pytest.raises(SystemExit):
        snap.check_target(private / "snap", private.resolve())
    full = tmp_path / "full"
    full.mkdir()
    (full / "f").write_text("x")
    with pytest.raises(SystemExit):
        snap.check_target(full, private.resolve())
    assert snap.check_target(tmp_path / "new", private.resolve()) == (tmp_path / "new").resolve()
    empty = tmp_path / "empty"
    empty.mkdir()
    assert snap.check_target(empty, private.resolve()) == empty.resolve()


def test_end_to_end_with_temporary_repo(tmp_path):
    repo = tmp_path / "repo"
    make_tree(repo)
    (repo / "scripts").mkdir()
    (repo / "scripts" / "new.py").write_text("print(1)\n")  # untracked, not ignored
    env = ["-c", "user.name=T", "-c", "user.email=t@example.com"]
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    subprocess.run(["git", "add", "README.md", ".gitignore", "data"], cwd=repo, check=True)
    subprocess.run(["git", *env, "commit", "-q", "-m", "c"], cwd=repo, check=True)

    out_ref = tmp_path / "out_ref"
    snap.extract_archive("HEAD", out_ref, repo)
    assert (out_ref / "data/raw/sintef/solomon-100.zip").exists() and not (out_ref / "scripts").exists()
    snap.remove_excluded(out_ref)
    snap.assert_clean(out_ref)

    out_wt = tmp_path / "out_wt"
    snap.copy_worktree(out_wt, repo)
    assert (out_wt / "scripts" / "new.py").exists()
    snap.remove_excluded(out_wt)
    snap.write_gitignore(out_wt)
    snap.assert_clean(out_wt)
    snap.init_repo(out_wt, "abc1234", ("T", "t@example.com"))
    tracked = subprocess.run(["git", "ls-files"], cwd=out_wt, capture_output=True, text=True).stdout
    assert "solomon-100.zip" not in tracked and "scripts/new.py" in tracked
