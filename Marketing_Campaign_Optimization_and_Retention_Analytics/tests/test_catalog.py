"""Tests for the message catalog and the helpers that print it."""
import string

import pytest

from marketing_campaign import catalog, i18n


def _fields(message: str) -> set[str]:
    """Return the ``str.format`` field names used by a message."""
    return {
        name
        for _, name, _, _ in string.Formatter().parse(message)
        if name is not None
    }


def test_both_languages_define_the_same_messages():
    """A key can never exist in one language only."""
    assert set(catalog.MESSAGES["en"]) == set(catalog.MESSAGES["pt"])


def test_both_languages_use_the_same_placeholders():
    """Each message takes the same values in English and in Portuguese."""
    for key in catalog.MESSAGES["en"]:
        english = _fields(catalog.MESSAGES["en"][key])
        portuguese = _fields(catalog.MESSAGES["pt"][key])
        assert english == portuguese, key


def test_no_message_is_empty():
    """Every message has text in both languages."""
    for lang, messages in catalog.MESSAGES.items():
        assert all(text.strip() for text in messages.values()), lang


@pytest.mark.parametrize(
    ("lang", "expected"),
    [
        ("en", "Canonical base: 6,849 stable users; 460 users who switched "
               "variants excluded from the A/B test."),
        ("pt", "Base canônica: 6,849 usuários estáveis; 460 usuários com "
               "troca de variante excluídos do A/B."),
    ],
)
def test_t_fills_the_fields_of_a_message(lang, expected):
    """Values are formatted into the message of the chosen language."""
    text = i18n.t("seg.canonical_base", lang, users=6849, switchers=460)

    assert text == expected


def test_t_returns_messages_without_fields_untouched():
    """A message with braces-free text needs no values."""
    assert i18n.t("funnel.title", "pt") == "ANÁLISE DE FUNIL"


def test_t_rejects_unknown_languages_and_keys():
    """Both mistakes fail loudly instead of printing a wrong text."""
    with pytest.raises(ValueError, match="lang must be one of"):
        i18n.t("funnel.title", "fr")
    with pytest.raises(KeyError):
        i18n.t("no.such.key", "en")


def test_say_prints_the_message(capsys):
    """``say`` writes the formatted message to the output."""
    i18n.say("data.loaded", "en", rows=10, columns=3)

    assert capsys.readouterr().out == (
        "✓ Dataset loaded: 10 rows, 3 columns\n"
    )


def test_banner_frames_the_title_between_two_rules(capsys):
    """The title sits between two lines of ``=``."""
    i18n.banner("funnel.title", "en", width=10)

    assert capsys.readouterr().out == (
        "\n==========\nFUNNEL ANALYSIS\n==========\n"
    )
