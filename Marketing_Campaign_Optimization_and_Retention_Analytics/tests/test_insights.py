"""Tests for the insight texts."""
import pandas as pd
import pytest

from marketing_campaign import insights
from marketing_campaign.diagnostics import (
    analyze_false_positive_bias,
    analyze_language_performance,
)


def _channels(rows) -> pd.DataFrame:
    """Build a channel table from (name, conversion %, retention %)."""
    return pd.DataFrame(
        rows,
        columns=["marketing_channel", "conversion_rate_%", "retention_rate_%"],
    )


def test_funnel_insight_quotes_rates_and_the_value_of_a_point(result):
    """Rates come from the funnel and one point of conversion is priced."""
    english = insights.funnel_insight(result.funnel, 6849, "en")
    portuguese = insights.funnel_insight(result.funnel, 6849, "pt")

    assert "10.3% of impacted users convert and 65.9% of converts" in english
    assert "$ 3,082" in english
    assert "10.3% dos usuários impactados convertem" in portuguese
    assert "Insights do Funil" in portuguese


def test_channel_insight_on_real_data_says_email_dominates(result):
    """Email leads both dimensions in the real base."""
    info = insights.channel_insight(
        result.channels, result.ltv.global_ltv, "en"
    )

    assert info.finding == (
        "Email dominates conversion (37.9%) and retention (76.2%)."
    )
    assert info.action.startswith("**Focus budget on Email**")
    assert "from House Ads to Email" in info.markdown


def test_channel_insight_detects_a_tradeoff_and_a_house_ads_lead():
    """Different leaders per dimension are a tradeoff; House Ads can lead."""
    tradeoff = insights.channel_insight(
        _channels([("Email", 30, 50), ("House Ads", 20, 70),
                   ("Push", 10, 40)]),
        10.0,
        "en",
    )
    house = insights.channel_insight(
        _channels([("Email", 10, 50), ("House Ads", 20, 70)]), 10.0, "pt"
    )

    assert "Real tradeoff: Email leads conversion and House Ads leads" in (
        tradeoff.finding
    )
    assert "from Email to House Ads in retention campaigns" in (
        tradeoff.markdown
    )
    assert house.action.startswith("Revisar dados")
    assert "verificar dados antes de recomendar" in house.markdown


def test_language_insight_prices_the_wrong_language_group(result):
    """325 users are served in the wrong language; the impact is positive."""
    summary = analyze_language_performance(result.users)

    info = insights.language_insight(
        summary, result.users, result.ltv.global_ltv, "en"
    )

    assert info.n_wrong == 325
    assert info.impact_total == pytest.approx(412, abs=5)
    assert info.note is None
    assert "There are 325 users (4.7% of the dataset)" in info.markdown


def test_language_insight_falls_back_to_a_five_percent_ltv_loss():
    """Zero measured lift still costs 5% of the LTV of each affected user."""
    users = pd.DataFrame(
        {
            "user_id": list("abcd"),
            "converted_flag": [0, 0, 0, 0],
            "retained_flag": [0, 0, 0, 0],
            "is_correct_lang": ["Yes", "Yes", "No", "No"],
        }
    )

    info = insights.language_insight(None, users, 10.0, "en")

    assert info.n_wrong == 2
    assert info.impact_total == pytest.approx(1.0)
    assert "estimated satisfaction cost of \\$ 1" in info.note


def test_a_b_notes_quote_the_lift_and_the_base():
    """The design note and the z-test guide use the configured lift."""
    design = insights.ab_design_insight(2.0, "pt")
    guide = insights.z_test_insight(2.0, 6849, "en")

    assert ">= 2.0 p.p. de lift" in design
    assert "\\$ 3,082" in guide
    assert "(>= 2.0 p.p.)" in guide


def test_roi_insight_reallocates_from_the_worst_to_the_best_channel(result):
    """Best and worst channels with relevant volume drive the figures."""
    info = insights.roi_insight(result.roi, "en")

    assert info.best == "House Ads"
    assert info.worst == "Instagram"
    assert info.extra_profit == pytest.approx(2669, abs=20)
    assert info.roi_gain_pp == pytest.approx(36.0, abs=0.5)
    assert "Reallocate 10% of the budget" in info.markdown


def test_roi_insight_warns_about_high_roi_on_small_cost():
    """A channel above 500% incremental ROI raises an alert."""
    roi = pd.DataFrame(
        {
            "marketing_channel": ["A", "B", "C"],
            "total_cost": [100.0, 100.0, 100.0],
            "estimated_profit": [10.0, 20.0, 30.0],
            "roi_incremental_%": [-5.0, 100.0, 900.0],
        }
    )

    info = insights.roi_insight(roi, "pt")

    assert len(info.alerts) == 1
    assert info.alerts[0].startswith("ALERTA: C tem ROI incremental de 900%")
    assert (info.best, info.worst) == ("C", "A")


def test_roi_insight_needs_two_channels_with_volume():
    """With a single eligible channel the names are not available."""
    roi = pd.DataFrame(
        {
            "marketing_channel": ["A"],
            "total_cost": [100.0],
            "estimated_profit": [10.0],
            "roi_incremental_%": [5.0],
        }
    )

    info = insights.roi_insight(roi, "en")

    assert (info.best, info.worst, info.extra_profit) == ("N/D", "N/D", 0.0)


def test_model_and_error_insights_fill_the_numbers():
    """The selected model, its metrics and the false-positive saving."""
    best = pd.Series(
        {
            "estimator_name": "LogisticRegression",
            "include_converted": True,
            "roc_auc": 0.956,
            "recall_not_retained": 0.9592,
            "f1_not_retained": 0.53,
            "precision_not_retained": 0.37,
        }
    )

    english = insights.model_insight(best, "en")
    portuguese = insights.model_insight(best, "pt")
    errors = insights.error_insight(2, 703, "en")

    assert "LogisticRegression | with converted_flag" in english
    assert "com converted_flag | sem SMOTE" in portuguese
    assert "ROC AUC = 0.956" in english
    assert "identifies 96% of users" in english
    assert "2 false positives (0.3% of converted users)" in errors


def _bias_tables(diff_high: bool) -> dict:
    """Return channel bias tables with or without a large difference."""
    base = pd.DataFrame({"marketing_channel": ["a"] * 60 + ["b"] * 40})
    errors_big = pd.DataFrame({"marketing_channel": ["a"] * 2 + ["b"] * 8})
    errors_small = pd.DataFrame({"marketing_channel": ["a"] * 6 + ["b"] * 4})
    return analyze_false_positive_bias(
        base, errors_big if diff_high else errors_small, ["marketing_channel"]
    )


def test_bias_action_depends_on_the_size_of_the_difference():
    """Large gap: an action to display; small gap: a plain notice."""
    action, actionable = insights.bias_action(_bias_tables(True), "en")
    notice, plain = insights.bias_action(_bias_tables(False), "pt")

    assert actionable and "**b** is over-represented" in action
    assert not plain
    assert notice.startswith("Nenhum canal com super-representação")
    assert insights.bias_action({}, "en") is None


def test_bias_summary_names_the_most_over_represented_channel():
    """The summary uses the first row and reports N/D without a table."""
    summary = insights.bias_summary(_bias_tables(True), 10, "en")
    empty = insights.bias_summary({}, 10, "pt")

    assert "concentrate in b (channel=b)" in summary
    assert "~40.0%" in summary
    assert "concentram-se em N/D" in empty


@pytest.mark.parametrize(
    ("lang", "needles"),
    [
        ("en", ["6849 users, 21 columns", "RFM segments: 3 segments",
                "- High Value: \\$ 43.98", "(generated)"]),
        ("pt", ["6849 usuários, 21 colunas", "Segmentos RFM: 3 segmentos",
                "- Alto Valor: \\$", "(gerado)"]),
    ],
)
def test_summary_report_lists_the_key_figures(
    result, tmp_path, lang, needles
):
    """Every block of the summary is filled with the pipeline figures."""
    crm = tmp_path / "crm.csv"
    crm.write_text("x")

    text = insights.summary_report(
        users=result.users,
        exposure_rows=len(result.exposures),
        switchers=result.switchers,
        ltv=result.ltv,
        design={"required_n": 3916, "runtime_days": 37,
                "minimum_lift_pp": 2.0},
        roi=result.roi,
        metrics=result.model.metrics,
        outputs={"excel": tmp_path / "out.xlsx", "crm": crm},
        n_recommendations=len(result.recommendations),
        lang=lang,
    )

    for needle in needles:
        assert needle in text, needle
    assert "3916" in text and "37" in text


def test_summary_report_flags_a_missing_crm_file(result, tmp_path):
    """Without the exported file the status says so."""
    text = insights.summary_report(
        users=result.users,
        exposure_rows=1,
        switchers=0,
        ltv=result.ltv,
        design={"required_n": 1, "runtime_days": 1, "minimum_lift_pp": 2.0},
        roi=result.roi,
        metrics=result.model.metrics,
        outputs={"excel": tmp_path / "a.xlsx", "crm": tmp_path / "none.csv"},
        n_recommendations=0,
        lang="en",
    )

    assert "not generated (run export cell)" in text


@pytest.mark.parametrize(
    ("lang", "needles"),
    [
        (
            "en",
            [
                "Fixing language for 325 users currently served in the "
                "wrong language",
                "from **Instagram** to **House Ads**",
                "Prioritize the 580 customers",
                "Model identifies 95.9% of not-retained users",
            ],
        ),
        (
            "pt",
            [
                "Corrigir o idioma dos 325 usuarios atualmente "
                "mal-servidos",
                "de **Instagram** para **House Ads**",
                "Priorizar os 580 clientes",
                "Modelo identifica 95.9% dos nao-retidos",
            ],
        ),
    ],
)
def test_decisions_report_names_the_actions(result, lang, needles):
    """Findings, three actions and the next step use the insight facts."""
    channel = insights.channel_insight(
        result.channels, result.ltv.global_ltv, lang
    )
    language = insights.language_insight(
        analyze_language_performance(result.users),
        result.users,
        result.ltv.global_ltv,
        lang,
    )
    roi = insights.roi_insight(result.roi, lang)

    text = insights.decisions_report(
        result.ltv, result.model.metrics, channel, language, roi, 580, lang
    )

    for needle in needles:
        assert needle in text, needle
