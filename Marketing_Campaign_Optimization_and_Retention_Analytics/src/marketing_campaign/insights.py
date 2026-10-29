"""Insight texts that quote the numbers computed by the analysis.

Each function reads the English result tables, works out the figures the
sentences need and fills the templates of the message catalog, so both
notebooks run the same code and only the language changes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from marketing_campaign.catalog import SEGMENT_LABELS
from marketing_campaign.config import (
    DEFAULT_COST,
    RETENTION_LTV_SHARE,
    REVENUE_PER_CONVERSION,
)
from marketing_campaign.i18n import t
from marketing_campaign.narrative import (
    normalize_language_flag,
    safe_float,
)

LTV_GAIN_SHARE = 0.05
REALLOCATION_FRACTION = 0.10
MIN_VOLUME_SHARE = 0.05
ROI_OUTLIER_PERCENT = 500
BIAS_THRESHOLD_PP = 10.0
FALSE_POSITIVE_REDUCTION = 0.20
LANGUAGE_LABELS = ("Correct Language", "Wrong Language")


def funnel_insight(
    funnel: pd.DataFrame, n_users: int, lang: str = "en"
) -> str:
    """Write the funnel insight.

    Parameters
    ----------
    funnel : pd.DataFrame
        Output of :func:`marketing_campaign.funnel.build_funnel`.
    n_users : int
        Users in the base.
    lang : {"en", "pt"}, default="en"
        Language of the text.

    Returns
    -------
    str
        Markdown with the conversion and retention rates and the value of
        one more percentage point of conversion.
    """
    values = funnel.set_index("Metric")["Value"]
    return t(
        "insight.funnel",
        lang,
        conv=float(values["Conversion Rate (%)"]),
        ret=float(values["Retention Rate (%)"]),
        impact=n_users * 0.01 * REVENUE_PER_CONVERSION,
    )


@dataclass(frozen=True)
class ChannelInsight:
    """Channel insight and the facts reused by the executive summary.

    Attributes
    ----------
    markdown : str
        The insight text.
    action : str
        Second priority action of the executive summary.
    finding : str
        One-line finding for the executive summary table.
    """

    markdown: str
    action: str
    finding: str


def channel_insight(
    channels: pd.DataFrame, global_ltv: float, lang: str = "en"
) -> ChannelInsight:
    """Compare Email and House Ads and write the channel insight.

    Parameters
    ----------
    channels : pd.DataFrame
        Output of
        :func:`marketing_campaign.funnel.analyze_channel_performance`.
    global_ltv : float
        Mean historical value of a customer.
    lang : {"en", "pt"}, default="en"
        Language of the text.

    Returns
    -------
    ChannelInsight
        The insight, the second priority action and the one-line finding.
    """
    by_channel = channels.set_index("marketing_channel")

    def rates(name: str) -> tuple[float, float]:
        """Return the conversion and retention rate of one channel."""
        if name not in by_channel.index:
            return 0.0, 0.0
        row = by_channel.loc[name]
        return float(row["conversion_rate_%"]), float(row["retention_rate_%"])

    email_conv, email_ret = rates("Email")
    house_conv, house_ret = rates("House Ads")
    na = t("summary.not_available", lang)
    if channels.empty:
        best_conv, best_ret, best_conv_rate, best_ret_rate = na, na, 0.0, 0.0
    else:
        best_conv = channels.loc[
            channels["conversion_rate_%"].idxmax(), "marketing_channel"
        ]
        best_ret = channels.loc[
            channels["retention_rate_%"].idxmax(), "marketing_channel"
        ]
        best_conv_rate = float(channels["conversion_rate_%"].max())
        best_ret_rate = float(channels["retention_rate_%"].max())

    email_leads = email_conv >= house_conv and email_ret >= house_ret
    house_leads = house_conv >= email_conv and house_ret >= email_ret
    values = {
        "email_conv": email_conv,
        "email_ret": email_ret,
        "house_conv": house_conv,
        "house_ret": house_ret,
        "best_conv": best_conv,
        "best_ret": best_ret,
        "best_conv_rate": best_conv_rate,
        "best_ret_rate": best_ret_rate,
    }
    if email_leads:
        case = "dominates"
    elif not house_leads:
        case = "tradeoff"
    else:
        case = "house"

    action = t(f"channel.action_{case}", lang, **values)
    realloc = t(f"channel.realloc_{case}", lang, **values)
    action2_key = "default" if case == "house" else case
    finding_key = "default" if case == "house" else case
    markdown = t(
        "insight.channel",
        lang,
        action=action,
        mean_conv=float(channels["conversion_rate_%"].mean()),
        realloc=realloc,
        ltv_impact=global_ltv * LTV_GAIN_SHARE,
    )
    return ChannelInsight(
        markdown=markdown,
        action=t(f"channel.action2_{action2_key}", lang, **values),
        finding=t(f"channel.finding_{finding_key}", lang, **values),
    )


@dataclass(frozen=True)
class LanguageInsight:
    """Language insight and the facts reused by the executive summary.

    Attributes
    ----------
    markdown : str
        The insight text.
    note : str or None
        Extra note printed when the measured lift is zero.
    impact_total : float
        Estimated revenue from fixing the language, in dollars.
    n_wrong : int
        Users served in the wrong language.
    """

    markdown: str
    note: str | None
    impact_total: float
    n_wrong: int


def _language_rates(
    summary: pd.DataFrame | None, users: pd.DataFrame
) -> dict[str, float]:
    """Return conversion, retention and size of the wrong-language group.

    Parameters
    ----------
    summary : pd.DataFrame or None
        Output of
        :func:`marketing_campaign.diagnostics.analyze_language_performance`.
    users : pd.DataFrame
        User-level base, used when ``summary`` has no usable labels.

    Returns
    -------
    dict
        ``correct_conv``, ``wrong_conv``, ``correct_ret``, ``wrong_ret``
        and ``n_wrong``.
    """
    rates = dict.fromkeys(
        ["correct_conv", "wrong_conv", "correct_ret", "wrong_ret"], 0.0
    )
    rates["n_wrong"] = 0
    if summary is not None and not summary.empty:
        by_label = summary.set_index("label")
        if all(label in by_label.index for label in LANGUAGE_LABELS):
            right, wrong = LANGUAGE_LABELS
            rates["correct_conv"] = safe_float(
                by_label.at[right, "conversion_rate_%"]
            )
            rates["wrong_conv"] = safe_float(
                by_label.at[wrong, "conversion_rate_%"]
            )
            rates["correct_ret"] = safe_float(
                by_label.at[right, "retention_rate_%"]
            )
            rates["wrong_ret"] = safe_float(
                by_label.at[wrong, "retention_rate_%"]
            )
            rates["n_wrong"] = int(safe_float(by_label.at[wrong, "user_id"]))
            return rates

    flags = users["is_correct_lang"].apply(normalize_language_flag)
    for flag, prefix in ((1, "correct"), (0, "wrong")):
        group = users[flags == flag]
        converted = group[group["converted_flag"] == 1]
        if len(group):
            rates[f"{prefix}_conv"] = group["converted_flag"].sum() / len(
                group
            ) * 100
        if len(converted):
            rates[f"{prefix}_ret"] = converted["retained_flag"].sum() / len(
                converted
            ) * 100
        if flag == 0:
            rates["n_wrong"] = int(group["user_id"].nunique())
    return rates


def language_insight(
    summary: pd.DataFrame | None,
    users: pd.DataFrame,
    global_ltv: float,
    lang: str = "en",
) -> LanguageInsight:
    """Quantify and describe the cost of serving the wrong language.

    Parameters
    ----------
    summary : pd.DataFrame or None
        Output of
        :func:`marketing_campaign.diagnostics.analyze_language_performance`.
    users : pd.DataFrame
        User-level base.
    global_ltv : float
        Mean historical value of a customer.
    lang : {"en", "pt"}, default="en"
        Language of the text.

    Returns
    -------
    LanguageInsight
        The insight, an optional note, the estimated impact and the number
        of users affected.
    """
    rates = _language_rates(summary, users)
    n_wrong = rates["n_wrong"]
    lift_conv = rates["correct_conv"] - rates["wrong_conv"]
    lift_ret = rates["correct_ret"] - rates["wrong_ret"]
    impact_conv = n_wrong * lift_conv / 100 * REVENUE_PER_CONVERSION
    impact_ret = (
        n_wrong * max(lift_ret, 0) / 100 * global_ltv * RETENTION_LTV_SHARE
    )
    total = impact_conv + impact_ret

    note = None
    if total == 0.0 and n_wrong > 0:
        total = n_wrong * global_ltv * LTV_GAIN_SHARE
        note = t("language.lift_note", lang, value=total, users=n_wrong)

    markdown = t(
        "insight.language",
        lang,
        correct_conv=rates["correct_conv"],
        wrong_conv=rates["wrong_conv"],
        lift_conv=lift_conv,
        n_wrong=n_wrong,
        pct_wrong=n_wrong / max(users["user_id"].nunique(), 1) * 100,
        total=total,
        impact_conv=impact_conv,
        impact_ret=impact_ret,
    )
    return LanguageInsight(markdown, note, total, n_wrong)


def ab_design_insight(minimum_lift_pp: float, lang: str = "en") -> str:
    """Write the note that introduces the A/B experiment.

    Parameters
    ----------
    minimum_lift_pp : float
        Smallest lift worth detecting, in percentage points.
    lang : {"en", "pt"}, default="en"
        Language of the text.

    Returns
    -------
    str
        Markdown with the hypothesis and the lift that justifies it.
    """
    return t("insight.ab_design", lang, lift=minimum_lift_pp)


def z_test_insight(
    minimum_lift_pp: float, n_users: int, lang: str = "en"
) -> str:
    """Write the reading guide of the z-test.

    Parameters
    ----------
    minimum_lift_pp : float
        Smallest lift worth detecting, in percentage points.
    n_users : int
        Users in the base.
    lang : {"en", "pt"}, default="en"
        Language of the text.

    Returns
    -------
    str
        Markdown that explains significance against practical relevance
        and the value of one extra percentage point of lift.
    """
    value = max(n_users, 1) * REVENUE_PER_CONVERSION * 0.01
    return t("insight.z_test", lang, lift=minimum_lift_pp, value=value)


@dataclass(frozen=True)
class RoiInsight:
    """ROI insight and the facts reused by the executive summary.

    Attributes
    ----------
    markdown : str
        The insight text.
    alerts : list of str
        Warnings about channels with a very high ROI on a small cost.
    best : str
        Channel with the highest incremental ROI.
    worst : str
        Channel with the lowest incremental ROI.
    extra_profit : float
        Profit from moving part of the budget from ``worst`` to ``best``.
    roi_gain_pp : float
        Resulting gain of the global ROI, in percentage points.
    """

    markdown: str
    alerts: list[str]
    best: str
    worst: str
    extra_profit: float
    roi_gain_pp: float


def roi_insight(roi: pd.DataFrame, lang: str = "en") -> RoiInsight:
    """Describe the best budget reallocation between channels.

    Parameters
    ----------
    roi : pd.DataFrame
        Output of :func:`marketing_campaign.economics.simulate_roi`.
    lang : {"en", "pt"}, default="en"
        Language of the text.

    Returns
    -------
    RoiInsight
        The insight, the alerts and the figures of the reallocation.
    """
    budget = float(roi["total_cost"].sum())
    threshold = budget * MIN_VOLUME_SHARE
    eligible = roi[roi["total_cost"] >= threshold].copy()

    best = worst = t("summary.not_available", lang)
    moved = extra = now = new = gain = savings = 0.0
    outliers = eligible.iloc[0:0]
    if len(eligible) >= 2:
        top = eligible.loc[eligible["roi_incremental_%"].idxmax()]
        bottom = eligible.loc[eligible["roi_incremental_%"].idxmin()]
        best = str(top["marketing_channel"])
        worst = str(bottom["marketing_channel"])
        moved = budget * REALLOCATION_FRACTION
        extra = moved * (
            float(top["roi_incremental_%"]) / 100
            - float(bottom["roi_incremental_%"]) / 100
        )
        profit = float(roi["estimated_profit"].sum())
        now = profit / max(budget, 1e-9) * 100
        new = (profit + extra) / max(budget, 1e-9) * 100
        gain = new - now
        savings = float(
            roi.loc[roi["roi_incremental_%"] < 0, "total_cost"].sum()
        ) * REALLOCATION_FRACTION
        outliers = eligible[
            eligible["roi_incremental_%"] > ROI_OUTLIER_PERCENT
        ]

    alerts = [
        t(
            "roi.alert",
            lang,
            channel=row["marketing_channel"],
            roi=row["roi_incremental_%"],
            cost=row["total_cost"],
        )
        for _, row in outliers.iterrows()
    ]
    markdown = t(
        "insight.roi",
        lang,
        best=best,
        worst=worst,
        fraction=REALLOCATION_FRACTION * 100,
        moved=moved,
        budget=budget,
        extra=extra,
        roi_now=now,
        roi_new=new,
        gain=gain,
        savings=savings,
    )
    return RoiInsight(markdown, alerts, best, worst, extra, gain)


def model_insight(best_experiment: pd.Series, lang: str = "en") -> str:
    """Describe the selected retention model.

    Parameters
    ----------
    best_experiment : pd.Series
        First row of the ranked experiments table.
    lang : {"en", "pt"}, default="en"
        Language of the text.

    Returns
    -------
    str
        Markdown with the model, its metrics and its operational use.
    """
    flag = (
        "model.with_converted"
        if best_experiment["include_converted"]
        else "model.without_converted"
    )
    return t(
        "insight.model",
        lang,
        estimator=best_experiment["estimator_name"],
        with_flag=t(flag, lang),
        roc_auc=best_experiment["roc_auc"],
        recall=best_experiment["recall_not_retained"],
        f1=best_experiment["f1_not_retained"],
        precision=best_experiment["precision_not_retained"],
    )


def error_insight(
    false_positives: int, converted_total: int, lang: str = "en"
) -> str:
    """Describe the false positives of the retention model.

    Parameters
    ----------
    false_positives : int
        Number of false positives in the test set.
    converted_total : int
        Converted users in the base.
    lang : {"en", "pt"}, default="en"
        Language of the text.

    Returns
    -------
    str
        Markdown with the count, its share of the converted users and the
        saving from reducing it.
    """
    return t(
        "insight.errors",
        lang,
        fp=false_positives,
        pct=false_positives / max(converted_total, 1) * 100,
        savings=false_positives * FALSE_POSITIVE_REDUCTION * DEFAULT_COST,
    )


def bias_action(
    bias_tables: dict[str, pd.DataFrame], lang: str = "en"
) -> tuple[str, bool] | None:
    """Turn the channel bias into an operational action.

    Parameters
    ----------
    bias_tables : dict of str to pd.DataFrame
        Output of
        :func:`marketing_campaign.diagnostics.analyze_false_positive_bias`.
    lang : {"en", "pt"}, default="en"
        Language of the text.

    Returns
    -------
    tuple of (str, bool) or None
        The text and whether it is an action to display (``True``) or a
        plain notice that no channel is over-represented (``False``).
        ``None`` when there is no channel table to read.
    """
    table = bias_tables.get("marketing_channel")
    if table is None or table.empty:
        return None
    top = table.iloc[0]
    if top["diff_pp"] < BIAS_THRESHOLD_PP:
        return t("bias.no_channel", lang), False
    return (
        t(
            "insight.bias_action",
            lang,
            category=top["category"],
            fp=top["false_positive_pct"],
            base=top["base_pct"],
            diff=top["diff_pp"],
        ),
        True,
    )


def bias_summary(
    bias_tables: dict[str, pd.DataFrame],
    false_positives: int,
    lang: str = "en",
) -> str:
    """Summarise where the false positives concentrate.

    Parameters
    ----------
    bias_tables : dict of str to pd.DataFrame
        Output of
        :func:`marketing_campaign.diagnostics.analyze_false_positive_bias`.
    false_positives : int
        Number of false positives in the test set.
    lang : {"en", "pt"}, default="en"
        Language of the text.

    Returns
    -------
    str
        Markdown naming the most over-represented channel and the saving
        from fixing the bias.
    """
    segment = t("summary.not_available", lang)
    fix = savings = 0.0
    table = bias_tables.get("marketing_channel")
    if table is not None and not table.empty:
        top = table.iloc[0]
        segment = str(top["category"])
        fix = float(top["diff_pp"])
        savings = max(fix, 0.0) / 100 * max(false_positives, 0) * DEFAULT_COST
    return t(
        "insight.bias_summary", lang, segment=segment, fix=fix, savings=savings
    )


def _segment_values(
    ltv: Any, lang: str
) -> tuple[str, float, str, float]:
    """Return the names and values of the best and worst segments.

    Parameters
    ----------
    ltv : marketing_campaign.economics.LtvSummary
        Historical customer value.
    lang : {"en", "pt"}
        Language of the segment names.

    Returns
    -------
    tuple
        High segment name and value, low segment name and value.
    """
    labels = SEGMENT_LABELS[lang]
    high_name, low_name = labels[0], labels[2]
    high = float(ltv.segment_ltv_map.get(high_name, ltv.global_ltv * 1.5))
    low = float(ltv.segment_ltv_map.get(low_name, ltv.global_ltv * 0.5))
    return high_name, high, low_name, low


def summary_report(
    users: pd.DataFrame,
    exposure_rows: int,
    switchers: int,
    ltv: Any,
    design: dict[str, Any],
    roi: pd.DataFrame,
    metrics: dict[str, Any],
    outputs: dict[str, Path],
    n_recommendations: int,
    lang: str = "en",
) -> str:
    """Write the executive summary of the whole pipeline.

    Parameters
    ----------
    users : pd.DataFrame
        Segmented user-level base.
    exposure_rows : int
        Rows of the cleaned exposure base.
    switchers : int
        Users excluded for seeing more than one variant.
    ltv : marketing_campaign.economics.LtvSummary
        Historical customer value.
    design : dict
        ``required_n``, ``runtime_days`` and ``minimum_lift_pp`` of the
        experiment.
    roi : pd.DataFrame
        Output of :func:`marketing_campaign.economics.simulate_roi`.
    metrics : dict
        Output of
        :func:`marketing_campaign.retention_model.calculate_retention_metrics`.
    outputs : dict
        ``excel`` and ``crm`` paths of the exports.
    n_recommendations : int
        Rows of the recommendation table.
    lang : {"en", "pt"}, default="en"
        Language of the text.

    Returns
    -------
    str
        Markdown with the key figures of every step.
    """
    high_name, high, low_name, low = _segment_values(ltv, lang)
    converted = users["converted_flag"]
    retained = users.loc[converted == 1, "retained_flag"].sum()
    crm = str(outputs["crm"])
    status = "summary.crm_ok" if os.path.exists(crm) else "summary.crm_missing"
    return t(
        "summary.report",
        lang,
        users=users.shape[0],
        columns=users.shape[1],
        exposures=exposure_rows,
        switchers=switchers,
        segments=users["Segment"].nunique(),
        conversion=converted.mean() * 100,
        retention=retained / max(converted.sum(), 1) * 100,
        high_name=high_name,
        high_ltv=high,
        low_name=low_name,
        low_ltv=low,
        global_ltv=ltv.global_ltv,
        required_n=design["required_n"],
        runtime=design["runtime_days"],
        lift=design["minimum_lift_pp"],
        max_roi=roi["roi_incremental_%"].max(),
        profit=roi["estimated_profit"].sum(),
        roc_auc=metrics["roc_auc"],
        recall=metrics["recall_not_retained"],
        excel=outputs["excel"],
        crm=crm,
        crm_status=t(status, lang),
        records=n_recommendations,
    )


def decisions_report(
    ltv: Any,
    metrics: dict[str, Any],
    channel: ChannelInsight,
    language: LanguageInsight,
    roi: RoiInsight,
    n_action_clients: int,
    lang: str = "en",
) -> str:
    """Write the recommended decisions of the executive summary.

    Parameters
    ----------
    ltv : marketing_campaign.economics.LtvSummary
        Historical customer value.
    metrics : dict
        Retention metrics of the selected model.
    channel : ChannelInsight
        Output of :func:`channel_insight`.
    language : LanguageInsight
        Output of :func:`language_insight`.
    roi : RoiInsight
        Output of :func:`roi_insight`.
    n_action_clients : int
        Customers whose incremental profit is positive.
    lang : {"en", "pt"}, default="en"
        Language of the text.

    Returns
    -------
    str
        Markdown with the findings and the three priority actions.
    """
    high_name, high, low_name, low = _segment_values(ltv, lang)
    realloc = t(
        "decisions.realloc",
        lang,
        worst=roi.worst,
        best=roi.best,
        extra=roi.extra_profit,
        gain=roi.roi_gain_pp,
    )
    return t(
        "decisions.report",
        lang,
        channel_finding=channel.finding,
        high_name=high_name,
        high_ltv=high,
        low_name=low_name,
        low_ltv=low,
        detected=metrics["recall_not_retained"] * 100,
        n_wrong=language.n_wrong,
        impact=language.impact_total,
        action2=channel.action,
        realloc=realloc,
        n_action=n_action_clients,
    )


__all__ = [
    "ChannelInsight",
    "LanguageInsight",
    "RoiInsight",
    "ab_design_insight",
    "bias_action",
    "bias_summary",
    "channel_insight",
    "decisions_report",
    "error_insight",
    "funnel_insight",
    "language_insight",
    "model_insight",
    "roi_insight",
    "summary_report",
    "z_test_insight",
]
