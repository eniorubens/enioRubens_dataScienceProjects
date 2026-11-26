import pytest

from retail_multilevel.i18n import LANGS, TEXTS, t


def test_every_key_has_every_language():
    for key, by_lang in TEXTS.items():
        assert set(by_lang) == set(LANGS), key
        assert all(v.strip() for v in by_lang.values()), key


def test_unknown_language_raises():
    with pytest.raises(ValueError):
        t("project_title", "es")
