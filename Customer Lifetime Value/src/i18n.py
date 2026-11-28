"""Offline language selector for the bilingual CLTV notebooks."""

from __future__ import annotations

from pathlib import Path

from multilang import LangMap


CATALOG_DIR = Path(__file__).parent / "i18n_catalogs"
SUPPORTED_LANGUAGES = ("en", "pt")


def make_lang(target: str) -> LangMap:
    """Create the project language map backed by the local catalog."""
    if target not in SUPPORTED_LANGUAGES:
        raise ValueError(f"target must be one of {SUPPORTED_LANGUAGES}")
    return LangMap(
        source="en",
        target=target,
        cache_dir=str(CATALOG_DIR),
        offline=True,
    )
