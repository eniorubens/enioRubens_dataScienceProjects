"""Verify (and optionally extract) the raw Olist dataset.

The Kaggle download is manual because it requires credentials. This script checks the
SHA-256 of ``data/raw/olist/brazilian-ecommerce.zip`` against the value recorded in
``data/raw/olist/README.md`` and, with ``--extract``, unpacks the nine expected CSVs into
``data/raw/olist/csv/``. Only the Python standard library is used.

Exit codes: 0 success, 1 hash mismatch or invalid archive, 2 missing ZIP or README/hash.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLIST_DIR = ROOT / "data" / "raw" / "olist"
ZIP_NAME = "brazilian-ecommerce.zip"
DEFAULT_URL = "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce"

EXPECTED_CSVS = (
    "olist_customers_dataset.csv",
    "olist_geolocation_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "product_category_name_translation.csv",
)

_HASH_RE = re.compile(r"SHA-?256\W+([0-9a-fA-F]{64})", flags=re.IGNORECASE)
_URL_RE = re.compile(r"https://www\.kaggle\.com/datasets/[\w\-/]+")


class DataError(Exception):
    """Raised with a user-facing message and the intended exit code."""

    def __init__(self, message: str, code: int = 1):
        super().__init__(message)
        self.code = code


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_readme(readme: Path) -> tuple[str, str]:
    """Return the expected lowercase SHA-256 and the Kaggle URL (or a default)."""
    if not readme.is_file():
        raise DataError(f"README not found: {readme}", code=2)
    text = readme.read_text(encoding="utf-8")
    match = _HASH_RE.search(text)
    if match is None:
        raise DataError(f"No SHA-256 value found in {readme}", code=2)
    url = _URL_RE.search(text)
    return match.group(1).lower(), url.group(0) if url else DEFAULT_URL


def verify_zip(zip_path: Path, readme: Path) -> str:
    """Check the ZIP hash against the README; return the digest or raise DataError."""
    expected, url = read_readme(readme)
    if not zip_path.is_file():
        raise DataError(
            f"ZIP not found: {zip_path}\n"
            f"Download it manually from {url} (Kaggle requires a login) and save it at that path.",
            code=2,
        )
    actual = sha256_of(zip_path)
    if actual != expected:
        raise DataError(
            f"SHA-256 mismatch for {zip_path.name}\n  expected: {expected}\n  actual:   {actual}\n"
            f"Download the file again from {url}.",
            code=1,
        )
    return actual


def _is_unsafe(name: str) -> bool:
    normalized = name.replace("\\", "/")
    return normalized.startswith("/") or ".." in normalized.split("/") or ":" in normalized


def extract_csvs(zip_path: Path, target: Path, expected: tuple[str, ...] = EXPECTED_CSVS) -> list[str]:
    """Extract only the expected CSV names, refusing any unsafe member path."""
    try:
        archive = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as error:
        raise DataError(f"{zip_path} is not a valid ZIP archive: {error}", code=1) from error
    extracted = []
    with archive:
        infos = archive.infolist()
        for info in infos:
            if _is_unsafe(info.filename):
                raise DataError(f"Unsafe path in archive refused: {info.filename!r}", code=1)
        members = {info.filename: info for info in infos if not info.is_dir()}
        missing = [name for name in expected if name not in members]
        if missing:
            raise DataError(f"Archive lacks expected CSVs: {', '.join(missing)}", code=1)
        target.mkdir(parents=True, exist_ok=True)
        target = target.resolve()
        for name in expected:
            destination = (target / name).resolve()
            if destination.parent != target:
                raise DataError(f"Unsafe destination refused: {destination}", code=1)
            with archive.open(members[name]) as source, destination.open("wb") as sink:
                while chunk := source.read(1 << 20):
                    sink.write(chunk)
            extracted.append(name)
    return extracted


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Verify the Olist raw ZIP and optionally extract its CSVs.")
    parser.add_argument("--data-dir", type=Path, default=OLIST_DIR, help="Folder with the ZIP and README.md.")
    parser.add_argument("--extract", action="store_true", help="Extract the nine CSVs into <data-dir>/csv/.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return the process exit code."""
    args = build_parser().parse_args(argv)
    data_dir: Path = args.data_dir
    zip_path = data_dir / ZIP_NAME
    try:
        digest = verify_zip(zip_path, data_dir / "README.md")
        print(f"OK: {zip_path.name} SHA-256 matches README ({digest})")
        if args.extract:
            names = extract_csvs(zip_path, data_dir / "csv")
            print(f"Extracted {len(names)} CSVs into {data_dir / 'csv'}")
    except DataError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return error.code
    return 0


if __name__ == "__main__":
    sys.exit(main())
