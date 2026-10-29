"""Apply the texts of :mod:`marketing_campaign.catalog` to the results.

The package computes everything with English names. Both notebooks run
the same code and ask for the results in their language here.
"""
from __future__ import annotations

import unicodedata

import pandas as pd

from marketing_campaign.catalog import (
    COLUMNS_PT,
    FIX_LANGUAGE_PREFIX,
    MESSAGES,
    SEGMENT_LABELS,
    SUPPORTED_LANGUAGES,
    VALUE_COLUMNS,
    VALUES_PT,
    CANONICAL_SEGMENTS,
)


def _fold(text: str) -> str:
    """Lower-case a text and strip its accents.

    Parameters
    ----------
    text : str
        Any text.

    Returns
    -------
    str
        The text in lower case without diacritics.
    """
    decomposed = unicodedata.normalize("NFKD", text.strip().lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def canonical_segment(name: object) -> str | None:
    """Return the English name of a segment given in any supported language.

    Parameters
    ----------
    name : object
        Segment name such as ``"Alto Valor"``, ``"medio valor"`` or
        ``"Low Value"``.

    Returns
    -------
    str or None
        ``"High Value"``, ``"Medium Value"`` or ``"Low Value"``, or
        ``None`` when the name is not a known segment.
    """
    if not isinstance(name, str):
        return None
    return CANONICAL_SEGMENTS.get(_fold(name))


def translate_text(text: object, lang: str = "pt") -> object:
    """Translate one known text, leaving anything else unchanged.

    Parameters
    ----------
    text : object
        A label, segment, action or rationale produced by the package.
    lang : {"en", "pt"}, default="pt"
        Target language.

    Returns
    -------
    object
        The translated text; ``text`` itself when ``lang`` is ``"en"`` or
        the text is unknown. Actions that start with the language-fix
        prefix keep the prefix and translate the action that follows.
    """
    if lang == "en" or not isinstance(text, str):
        return text
    prefix = FIX_LANGUAGE_PREFIX["en"]
    if text.startswith(prefix):
        action = text[len(prefix):]
        return FIX_LANGUAGE_PREFIX[lang] + str(translate_text(action, lang))
    if " | " in text:
        return " | ".join(
            str(translate_text(part, lang)) for part in text.split(" | ")
        )
    return VALUES_PT.get(text, text)


def localize(df: pd.DataFrame, lang: str = "pt") -> pd.DataFrame:
    """Translate the column names and known labels of a result table.

    Parameters
    ----------
    df : pd.DataFrame
        A table produced by the package, with English names.
    lang : {"en", "pt"}, default="pt"
        Target language.

    Returns
    -------
    pd.DataFrame
        A copy of ``df``. For ``"pt"`` the columns listed in
        :data:`COLUMNS_PT` are renamed and the texts in the columns of
        :data:`VALUE_COLUMNS` are translated; other columns and values
        are untouched.

    Raises
    ------
    ValueError
        If ``lang`` is not supported.
    """
    if lang not in SUPPORTED_LANGUAGES:
        raise ValueError(f"lang must be one of {SUPPORTED_LANGUAGES}.")
    localized = df.copy()
    if lang == "en":
        return localized
    for column in VALUE_COLUMNS.intersection(localized.columns):
        localized[column] = localized[column].map(
            lambda value: translate_text(value, lang)
        )
    return localized.rename(columns=COLUMNS_PT)


def t(key: str, lang: str = "en", **values: object) -> str:
    """Return a message of the catalog in the requested language.

    Parameters
    ----------
    key : str
        Message identifier, for example ``"funnel.title"``.
    lang : {"en", "pt"}, default="en"
        Target language.
    **values : object
        Values for the ``str.format`` fields of the message.

    Returns
    -------
    str
        The message with its fields filled in.

    Raises
    ------
    ValueError
        If ``lang`` is not supported.
    KeyError
        If ``key`` is not in the catalog.
    """
    if lang not in SUPPORTED_LANGUAGES:
        raise ValueError(f"lang must be one of {SUPPORTED_LANGUAGES}.")
    message = MESSAGES[lang][key]
    return message.format(**values) if values else message


def say(key: str, lang: str = "en", **values: object) -> None:
    """Print a message of the catalog.

    Parameters
    ----------
    key : str
        Message identifier.
    lang : {"en", "pt"}, default="en"
        Target language.
    **values : object
        Values for the fields of the message.
    """
    print(t(key, lang, **values))


def banner(key: str, lang: str = "en", width: int = 80) -> None:
    """Print a message between two lines of ``=``.

    Parameters
    ----------
    key : str
        Message identifier of the title.
    lang : {"en", "pt"}, default="en"
        Target language.
    width : int, default=80
        Length of the lines.
    """
    print("\n" + "=" * width)
    print(t(key, lang))
    print("=" * width)
