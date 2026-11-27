from __future__ import annotations

import pytest

from churn_project.i18n import get_language, set_language, t


def test_language_catalog_is_deterministic():
    set_language("pt")
    assert t("Complete Dataset") == "Dataset Completo"

    set_language("en")
    assert t("Complete Dataset") == "Complete Dataset"


def test_unknown_text_falls_back_to_source_text():
    set_language("pt")
    assert t("A text not present in the catalog") == "A text not present in the catalog"


def test_mapping_values_are_translated_without_changing_keys():
    set_language("pt")
    assert t({"title": "Monthly Charges"}) == {"title": "Cobranças mensais"}


def test_unsupported_language_is_rejected():
    with pytest.raises(ValueError):
        set_language("es")
    assert get_language() == "pt"
    set_language("en")
