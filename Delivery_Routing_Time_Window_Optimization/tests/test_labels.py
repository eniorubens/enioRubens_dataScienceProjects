import pytest

from delivery_routing_vrptw import LABELS, get_labels


def structure(d):
    return {k: set(v) for k, v in d.items()}


def test_same_structure():
    assert structure(LABELS["pt"]) == structure(LABELS["en"])
    for group in ("metrics", "methods", "families", "sets"):
        assert group in LABELS["pt"]
    assert set(LABELS["pt"]["families"]) == {"C1", "C2", "R1", "R2", "RC1", "RC2"}


@pytest.mark.parametrize("alias", ["pt", "pt-br", "PT-BR", "pt_br"])
def test_pt_aliases(alias):
    assert get_labels(alias) is LABELS["pt"]


@pytest.mark.parametrize("alias", ["en", "en-us", "EN-US"])
def test_en_aliases(alias):
    assert get_labels(alias) is LABELS["en"]


def test_bad_language():
    with pytest.raises(ValueError):
        get_labels("fr")


def test_accents():
    assert get_labels("pt")["metrics"]["vehicles"] == "Veículos"
