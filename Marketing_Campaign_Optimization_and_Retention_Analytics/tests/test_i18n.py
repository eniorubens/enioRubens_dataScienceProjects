"""Tests for the Portuguese localisation layer."""
import pandas as pd
import pytest

from marketing_campaign import i18n
from marketing_campaign import recommendations as rec
from marketing_campaign.segmentation import assign_segments, build_rfm


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("High Value", "High Value"),
        ("Alto Valor", "High Value"),
        ("alto valor", "High Value"),
        ("Médio Valor", "Medium Value"),
        ("Medio Valor", "Medium Value"),
        ("Baixo Valor", "Low Value"),
        ("Platinum", None),
        (None, None),
        (float("nan"), None),
    ],
)
def test_canonical_segment_accepts_both_languages(name, expected):
    """Any spelling of a known segment maps to its English name."""
    assert i18n.canonical_segment(name) == expected


def test_translate_text_keeps_unknown_text_and_the_language_fix_prefix():
    """Known texts translate; the fix prefix wraps the translated action."""
    assert i18n.translate_text("Loyalty Campaign") == "Campanha de Fidelização"
    assert i18n.translate_text("Something else") == "Something else"
    assert i18n.translate_text(3.5) == 3.5
    assert i18n.translate_text(
        "Fix Language -> then Loyalty Campaign"
    ) == "Corrigir Idioma -> depois Campanha de Fidelização"
    assert i18n.translate_text("Loyalty Campaign", "en") == "Loyalty Campaign"


def test_localize_renames_columns_and_translates_labels():
    """The funnel table gets Portuguese headers and metric names."""
    funnel = pd.DataFrame(
        {
            "Metric": ["Impacted Users", "Conversion Rate (%)"],
            "Value": [10, 5.0],
        }
    )

    portuguese = i18n.localize(funnel)

    assert portuguese.columns.tolist() == ["Métrica", "Valor"]
    assert portuguese["Métrica"].tolist() == [
        "Usuários Impactados",
        "Taxa de Conversão (%)",
    ]
    assert funnel.columns.tolist() == ["Metric", "Value"]  # input untouched


def test_localize_in_english_changes_nothing():
    """English is the package language, so the table is only copied."""
    table = pd.DataFrame({"Segment": ["High Value"], "conversions": [3]})

    result = i18n.localize(table, "en")

    pd.testing.assert_frame_equal(result, table)
    assert result is not table


def test_localize_rejects_unknown_languages():
    """Only the supported languages are accepted."""
    with pytest.raises(ValueError, match="lang must be one of"):
        i18n.localize(pd.DataFrame(), "fr")


def test_real_channel_table_has_the_notebook_headers(result):
    """The Portuguese channel table keeps the headers the notebook uses."""
    columns = i18n.localize(result.channels).columns.tolist()

    assert columns == [
        "marketing_channel",
        "impactados",
        "conversoes",
        "retidos",
        "cancelamentos",
        "conversion_rate_%",
        "retention_rate_%",
    ]


def test_real_roi_table_has_the_notebook_headers(result):
    """The Portuguese ROI table keeps the headers the notebook uses."""
    columns = i18n.localize(result.roi).columns.tolist()

    assert "custo_total" in columns
    assert "lucro_estimado" in columns
    assert "roi_incremental_%" in columns


def test_assign_segments_can_use_portuguese_names(result):
    """The same clusters get Portuguese names, in the same order."""
    rfm = build_rfm(result.exposures)

    english = assign_segments(rfm)
    portuguese = assign_segments(
        rfm, labels=i18n.SEGMENT_LABELS["pt"]
    )

    assert set(portuguese["Segment"]) == {
        "Alto Valor",
        "Médio Valor",
        "Baixo Valor",
    }
    assert portuguese["Segment"].tolist() == [
        i18n.translate_text(name) for name in english["Segment"]
    ]


def test_portuguese_segments_receive_the_same_action():
    """The recommendation is the same for "Alto Valor" and "High Value"."""
    base = {"expected_profit": 5.0, "is_correct_lang": "Yes"}

    english = rec.recommend_action(
        pd.Series({**base, "Segment": "High Value"})
    )
    portuguese = rec.recommend_action(
        pd.Series({**base, "Segment": "Alto Valor"})
    )

    assert english == portuguese == "Loyalty Campaign"


def test_portuguese_recommendations_translate_after_localize():
    """Actions and rationales come out in Portuguese after localize."""
    scored = pd.DataFrame(
        {
            "user_id": ["a", "b"],
            "marketing_channel": ["Email", "Push"],
            "Segment": ["Alto Valor", "Médio Valor"],
            "prob_retention": [0.5, 0.2],
            "expected_profit": [10.0, -1.0],
            "is_correct_lang": ["No", "Yes"],
        }
    )

    recs = i18n.localize(rec.generate_recommendations(scored, 40.0))

    assert recs["recommended_action"].tolist() == [
        "Corrigir Idioma -> depois Campanha de Fidelização",
        "Nao Agir - Lucro Incremental Negativo",
    ]
    assert recs["strategy_rationale"].tolist() == [
        "Clientes de alto valor - manter engajamento e reduzir churn.",
        "Clientes com potencial - aumentar conversão e ticket medio.",
    ]
    assert recs["max_incentive"].tolist() == [15.0, 8.0]
