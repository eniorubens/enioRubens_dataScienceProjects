"""Obtain and verify the third-party benchmark data (not redistributed in the public repository).

Usage::

    python scripts/fetch_data.py                 # verify, download what can be downloaded, rebuild the CVRPLIB csv
    python scripts/fetch_data.py --check-only    # verify SHA-256 of the files already present (no network)
    python scripts/fetch_data.py --no-cvrplib-reference

The SINTEF archive cannot be downloaded automatically (Cloudflare); it must be saved manually.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CVRPLIB_BASE = "https://galgos.inf.puc-rio.br/cvrplib/index.php/en/download"
SINTEF_PAGE = "https://www.sintef.no/projectweb/top/vrptw/100-customers/"
USER_AGENT = "delivery-routing-vrptw-fetch"
TIMEOUT_S = 60


@dataclass(frozen=True)
class Asset:
    """A third-party file with its expected hash."""

    key: str
    path: Path
    sha256: str
    url: str | None  # None: manual download only
    extract_to: Path | None = None


RAW = ROOT / "data" / "raw"
ASSETS = (
    Asset(
        "sintef",
        RAW / "sintef" / "solomon-100.zip",
        "8a0a72cbe6b7f8f9988ace4ebde0378ec34943acaaac47f2c408915e41887747",
        None,
    ),
    Asset(
        "cvrplib_instances",
        RAW / "cvrplib_solomon_100" / "Solomon.7z",
        "55ba67b56a47855432f1b80164db7cc87ce3c7c22bc9855f2cc0013c81d92a37",
        f"{CVRPLIB_BASE}/instance-set/22",
        RAW / "cvrplib_solomon_100" / "extracted" / "instances",
    ),
    Asset(
        "cvrplib_solutions",
        RAW / "cvrplib_solomon_100" / "Solomon-solutions.7z",
        "88a5d385c0faf63e8eebc6a1bc755ced85d92575b816d33737b0258df459394e",
        f"{CVRPLIB_BASE}/instance-set-bks/22",
        RAW / "cvrplib_solomon_100" / "extracted" / "bks",
    ),
)


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, expected: str) -> str:
    """Return ``"missing"``, ``"ok"`` or ``"mismatch"`` for a file against its expected SHA-256."""
    if not path.is_file():
        return "missing"
    return "ok" if sha256_of(path) == expected.lower() else "mismatch"


def download(url: str, target: Path, timeout: float = TIMEOUT_S) -> None:
    """Download ``url`` to ``target`` (written to a temporary name first)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    partial = target.with_name(target.name + ".part")
    with urllib.request.urlopen(request, timeout=timeout) as response, open(partial, "wb") as out:
        shutil.copyfileobj(response, out)
    partial.replace(target)


def find_bsdtar() -> str | None:
    """Return a ``tar`` that reads 7z archives (bsdtar/libarchive), or None."""
    candidates = [shutil.which("bsdtar"), shutil.which("tar")]
    system_tar = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "tar.exe"
    if system_tar.is_file():
        candidates.insert(0, str(system_tar))
    for exe in candidates:
        if not exe:
            continue
        try:
            out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=30).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        if "libarchive" in out or "bsdtar" in out:
            return exe
    return None


def extract_7z(archive: Path, destination: Path) -> None:
    """Extract a (hash-verified) 7z archive into ``destination`` with bsdtar/libarchive."""
    exe = find_bsdtar()
    if exe is None:
        raise RuntimeError("no tar with libarchive found (Windows tar.exe or bsdtar); cannot extract .7z")
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run([exe, "-xf", str(archive), "-C", str(destination)], check=True, timeout=300)


def manual_instructions(asset: Asset, status: str) -> str:
    """Return the manual-download message for an asset."""
    return (
        f"SINTEF archive {status}. Download it manually (the site blocks automated clients):\n"
        f"  page:   {SINTEF_PAGE}\n"
        f"  save as: {asset.path}\n"
        f"  sha256: {asset.sha256}"
    )


def ensure(asset: Asset, check_only: bool) -> bool:
    """Make one asset available and verified; return True on success."""
    status = verify(asset.path, asset.sha256)
    if status == "ok":
        print(f"[ok]      {asset.key}: {asset.path.relative_to(ROOT)}")
        return True
    if status == "mismatch":
        print(f"[ERRO]    {asset.key}: SHA-256 differs from the expected value; refusing to use {asset.path}")
        return False
    if asset.url is None:
        print(f"[FALTA]   {manual_instructions(asset, 'is missing')}")
        return False
    if check_only:
        print(f"[FALTA]   {asset.key}: {asset.path.relative_to(ROOT)} (run without --check-only to download)")
        return False
    print(f"[baixando] {asset.key} <- {asset.url}")
    try:
        download(asset.url, asset.path)
    except Exception as exc:  # network errors of any kind
        print(f"[ERRO]    download failed: {exc}\n  Download manually from {asset.url} to {asset.path}")
        return False
    if verify(asset.path, asset.sha256) != "ok":
        print(f"[ERRO]    {asset.key}: downloaded file has an unexpected SHA-256; removing it")
        asset.path.unlink(missing_ok=True)
        return False
    print(f"[ok]      {asset.key}: downloaded and verified")
    return True


def rebuild_cvrplib_reference() -> bool:
    """Extract the verified 7z archives into ``extracted/`` and regenerate the CVRPLIB reference csv."""
    for asset in ASSETS:
        if asset.extract_to is not None:
            try:
                extract_7z(asset.path, asset.extract_to)
            except (RuntimeError, subprocess.SubprocessError, OSError) as exc:
                print(f"[ERRO]    extraction of {asset.path.name} failed: {exc}")
                return False
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_cvrplib_reference.py")], cwd=ROOT)
    return result.returncode == 0


def main(argv: list[str] | None = None) -> int:
    """Run the command line interface; return the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--check-only", action="store_true", help="only verify SHA-256 of present files (no network)"
    )
    parser.add_argument(
        "--no-cvrplib-reference", action="store_true", help="do not regenerate data/reference/cvrplib_bks_100.csv"
    )
    args = parser.parse_args(argv)

    status = {asset.key: ensure(asset, args.check_only) for asset in ASSETS}
    failed = not all(status.values())
    cvrplib_ready = status["cvrplib_instances"] and status["cvrplib_solutions"]
    if cvrplib_ready and not args.check_only and not args.no_cvrplib_reference:
        failed = not rebuild_cvrplib_reference() or failed
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
