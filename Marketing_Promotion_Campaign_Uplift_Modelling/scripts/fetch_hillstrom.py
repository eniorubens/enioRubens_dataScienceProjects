"""Fetch the Hillstrom MineThatData CSV and rebuild the exact file used in the analysis.

The dataset is not redistributed (the publication states no reuse license), so
it is downloaded from the publisher. The publisher's file and the copy used in
the analysis hold identical values but are not byte-identical: the publisher
writes ``spend`` as ``0`` where the analysis copy has ``0.0``. The pre-registered
fingerprints in ``artifacts/`` refer to the analysis copy, so this script

1. downloads the publisher file and verifies its SHA-256,
2. re-serializes it with pandas (CRLF line endings), and
3. verifies that the result has the SHA-256 recorded in the pre-registration.

The transport is plain http on purpose: the two hashes are the integrity check.

Usage (from the project root):  python scripts/fetch_hillstrom.py
"""

from __future__ import annotations

import hashlib
import io
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

URL = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
PUBLISHER_SHA256 = "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece"
ANALYSIS_SHA256 = "9bd6dd74628d14b652cef843df5e52c07f9024706dc00ece13855a85c4543b41"
TARGET = Path(__file__).resolve().parents[1] / "dataset" / "Kevin_Hillstrom_MineThatData.csv"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(url: str, attempts: int = 3) -> bytes:
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read()
        except OSError as exc:
            if attempt == attempts:
                raise
            print(f"download failed ({exc}); retrying", file=sys.stderr)
            time.sleep(2 * attempt)
    raise RuntimeError("unreachable")


def main() -> int:
    if TARGET.exists() and sha256(TARGET.read_bytes()) == ANALYSIS_SHA256:
        print(f"{TARGET} already matches the analysis fingerprint")
        return 0

    raw = download(URL)
    if sha256(raw) != PUBLISHER_SHA256:
        print(f"publisher file changed: sha256 {sha256(raw)}", file=sys.stderr)
        return 1

    buffer = io.StringIO(newline="")
    pd.read_csv(io.BytesIO(raw)).to_csv(buffer, index=False, lineterminator="\r\n")
    data = buffer.getvalue().encode("utf-8")
    if sha256(data) != ANALYSIS_SHA256:
        print(f"rebuilt file does not match the analysis fingerprint: {sha256(data)}", file=sys.stderr)
        return 1

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_bytes(data)
    print(f"wrote {TARGET} ({len(data):,} bytes, sha256 {ANALYSIS_SHA256[:12]}...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
