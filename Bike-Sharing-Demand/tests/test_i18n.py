"""Tests for the i18n foundation: PT is the canonical language and passthrough."""

from __future__ import annotations

import pandas as pd

import src.i18n as i18n
from src.i18n import BASE_LANG, localize_table, make_lang, resolve_lang


def test_base_lang_is_portuguese():
    assert BASE_LANG == "pt"


def test_make_lang_pt_is_passthrough():
    lang = make_lang("pt")
    assert lang.source == lang.target == "pt"
    payload = {"title": "Título em Português", "x": "Eixo X"}
    # passthrough returns an equal (new) dict, unchanged, with no translation
    out = lang(payload)
    assert out == payload
    assert out is not payload


def test_resolve_lang_none_returns_pt_passthrough(monkeypatch):
    # Guard against any accidental network translation: deep_translator must
    # never be needed for the PT (passthrough) path.
    def _boom(*args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("translation backend must not be invoked for PT")

    monkeypatch.setattr(i18n.LangMap, "_translate_batch", _boom, raising=True)
    lang = resolve_lang(None)
    assert lang.source == lang.target == "pt"
    assert lang({"a": "olá", "b": "mundo"}) == {"a": "olá", "b": "mundo"}


def test_resolve_lang_passes_through_existing_instance():
    lang = make_lang("pt")
    assert resolve_lang(lang) is lang


def test_localize_table_renames_columns_without_mutating_input():
    lang = make_lang("pt")
    internal = pd.DataFrame({"feature": ["a", "b"], "score": [1.0, 2.0]})
    before = internal.copy()

    display_df = localize_table(internal, lang, columns={"feature": "Variável"})

    assert list(display_df.columns) == ["Variável", "score"]
    assert list(display_df["score"]) == [1.0, 2.0]
    pd.testing.assert_frame_equal(internal, before)  # internal schema untouched


def test_localize_table_maps_values_and_preserves_numbers():
    lang = make_lang("pt")
    internal = pd.DataFrame(
        {"Decision": ["Reject H0", "Fail to reject H0"], "p-value": [0.001, 0.9]}
    )
    before = internal.copy()

    display_df = localize_table(
        internal,
        lang,
        columns={"Decision": "Decisão", "p-value": "valor-p"},
        value_columns=["Decision"],
        value_labels={"Reject H0": "Rejeita H0", "Fail to reject H0": "Não rejeita H0"},
    )

    assert list(display_df["Decisão"]) == ["Rejeita H0", "Não rejeita H0"]
    assert list(display_df["valor-p"]) == [0.001, 0.9]  # numbers unchanged
    pd.testing.assert_frame_equal(internal, before)  # internal schema untouched


def test_localize_table_maps_values_on_categorical_column_without_mutating_input():
    lang = make_lang("pt")
    internal = pd.DataFrame({"Seasons": pd.Categorical(["Winter", "Summer", "Winter"])})
    before = internal.copy()

    display_df = localize_table(
        internal,
        lang,
        columns={},
        value_columns=("Seasons",),
        value_labels={"Winter": "Inverno", "Summer": "Verão"},
    )

    assert list(display_df["Seasons"]) == ["Inverno", "Verão", "Inverno"]
    pd.testing.assert_frame_equal(internal, before)  # internal categorical untouched
