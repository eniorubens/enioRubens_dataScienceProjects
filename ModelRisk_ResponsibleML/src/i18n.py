"""Shared presentation labels for the bilingual Responsible ML notebooks."""

from __future__ import annotations


SUPPORTED_LANGUAGES = ("en", "pt")

_CATALOG = {
    "en": {
        "table_value": "value",
        "threshold_precision": "Precision",
        "threshold_recall": "Recall / TPR",
        "threshold_f1": "F1",
        "threshold_selection_rate": "Selection rate",
        "group_selection_rate": "Selection rate",
        "group_tpr": "TPR / Recall",
        "group_fpr": "FPR",
        "group_fnr": "FNR",
        "group_ppv": "PPV / precision",
    },
    "pt": {
        "table_value": "valor",
        "threshold_precision": "Precisão",
        "threshold_recall": "Recall / TPR",
        "threshold_f1": "F1",
        "threshold_selection_rate": "Taxa de seleção",
        "group_selection_rate": "Taxa de seleção",
        "group_tpr": "TPR / Recall",
        "group_fpr": "FPR",
        "group_fnr": "FNR",
        "group_ppv": "PPV / Precisão",
    },
}


def get_labels(lang: str) -> dict[str, str]:
    """Return a copy of the presentation labels for ``lang``.

    Parameters
    ----------
    lang : {"en", "pt"}
        Target notebook language.

    Returns
    -------
    dict[str, str]
        Labels shared by the PT-BR and EN-US notebook code.

    Raises
    ------
    ValueError
        If ``lang`` is not supported.
    """
    if lang not in SUPPORTED_LANGUAGES:
        raise ValueError(f"lang must be one of {SUPPORTED_LANGUAGES}")
    return dict(_CATALOG[lang])
