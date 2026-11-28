"""Tests for the bilingual notebook text catalog."""
import pytest

from cross_sell_rules.catalog import CATALOG, translate


def test_catalog_has_both_languages_for_every_key():
    """Every catalog entry is available in English and Portuguese."""
    assert CATALOG
    assert all(set(texts) == {"en", "pt"} for texts in CATALOG.values())


def test_translate_returns_the_requested_language():
    """The selected language changes text without changing analysis code."""
    assert translate("support_heatmap", "en") != translate(
        "support_heatmap", "pt"
    )


def test_translate_rejects_unknown_language():
    """Unsupported languages fail explicitly instead of falling back silently."""
    with pytest.raises(ValueError, match="language"):
        translate("support_heatmap", "es")


def test_translate_rejects_unknown_key():
    """Missing catalog entries are visible during development."""
    with pytest.raises(KeyError, match="unknown catalog key"):
        translate("missing", "en")
