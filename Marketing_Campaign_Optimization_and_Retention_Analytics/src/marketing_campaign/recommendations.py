"""Expected value per user, operational recommendations and exports."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from marketing_campaign.config import (
    COST_PER_IMPRESSION,
    DEFAULT_COST,
    DEFAULT_OUTPUT_DIR,
    NO_ACTION,
    ORGANIC_CONVERSION_RATE,
    SEGMENT_ACTIONS,
)
from marketing_campaign.economics import LtvSummary
from marketing_campaign.i18n import canonical_segment
from marketing_campaign.retention_model import (
    RetentionModel,
    prepare_retention_model_frame,
)

CRM_FILENAME = "crm_next_best_action.csv"
WRONG_LANGUAGE_VALUES = frozenset({"0", "0.0", "no", "n", "false"})
BASE_COLUMNS = [
    "user_id",
    "marketing_channel",
    "variant",
    "language_displayed",
    "language_preferred",
    "is_correct_lang",
    "Segment",
    "converted_flag",
    "retained_flag",
    "prob_retention",
    "expected_profit",
    "cost_per_user",
    "ltv_reference",
]


def score_users(
    df: pd.DataFrame,
    model: RetentionModel,
    ltv: LtvSummary,
    cost_per_impression: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Score every user with its retention probability and expected value.

    The expected value is the retention probability times the reference
    value of the user's segment; the expected profit subtracts the cost
    of the user's channel.

    Parameters
    ----------
    df : pd.DataFrame
        Segmented, user-level base.
    model : RetentionModel
        Fitted model from
        :func:`marketing_campaign.retention_model.build_retention_model`.
    ltv : LtvSummary
        Output of :func:`marketing_campaign.economics.estimate_ltv`.
    cost_per_impression : dict, optional
        Cost of an impression by channel.

    Returns
    -------
    pd.DataFrame
        A copy of ``df`` with ``prob_retention``, ``cost_per_user``,
        ``ltv_reference``, ``expected_value`` and ``expected_profit``.
    """
    costs = cost_per_impression or COST_PER_IMPRESSION
    scored = prepare_retention_model_frame(df)

    features = scored[
        model.categorical_features + model.numeric_features
    ].copy()
    for column in model.numeric_features:
        features[column] = features[column].fillna(features[column].median())
    for column in model.categorical_features:
        features[column] = features[column].fillna("Unknown")

    scored["prob_retention"] = model.pipeline.predict_proba(features)[:, 1]
    scored["cost_per_user"] = (
        scored["marketing_channel"].map(costs).fillna(DEFAULT_COST)
    )
    scored["ltv_reference"] = (
        scored["Segment"].map(ltv.segment_ltv_map).fillna(ltv.global_ltv)
    )
    scored["expected_value"] = (
        scored["prob_retention"] * scored["ltv_reference"]
    )
    scored["expected_profit"] = (
        scored["expected_value"] - scored["cost_per_user"]
    )
    return scored


def is_wrong_language(value: object) -> bool:
    """Tell whether an ``is_correct_lang`` value marks the wrong language.

    The dataset stores the flag as the strings ``"Yes"`` and ``"No"``;
    derived frames use ``1`` and ``0``. Both spellings are understood.

    Parameters
    ----------
    value : object
        Flag such as ``"No"``, ``"yes"``, ``0``, ``1``, ``False`` or a
        missing value.

    Returns
    -------
    bool
        ``True`` only when the flag explicitly says the language shown
        differs from the preferred one. Missing or unknown values are
        not treated as wrong.
    """
    if pd.isna(value):
        return False
    return str(value).strip().lower() in WRONG_LANGUAGE_VALUES


def recommend_action(row: pd.Series) -> str:
    """Choose the action for one scored user.

    A user whose expected profit is not positive gets no action. Users
    flagged with the wrong language (see :func:`is_wrong_language`) are
    sent to a language fix first. Everyone else receives the action of
    their segment.

    Parameters
    ----------
    row : pd.Series
        A scored user with ``Segment``, ``expected_profit`` and
        ``is_correct_lang``.

    Returns
    -------
    str
        The recommended action.

    Notes
    -----
    The flag is read as text or number, so the ``"No"`` values of the
    dataset reach the language-fix branch. The first published version
    compared the flag with ``0`` only and never took that branch.
    """
    segment = canonical_segment(row.get("Segment", "Low Value"))
    config = SEGMENT_ACTIONS[segment or "Low Value"]
    if row.get("expected_profit", 0) <= 0:
        return NO_ACTION
    if is_wrong_language(row.get("is_correct_lang", 1)):
        return f"Fix Language -> then {config['action']}"
    return str(config["action"])


def generate_recommendations(
    scored: pd.DataFrame,
    global_ltv: float,
    organic_conversion_rate: float = ORGANIC_CONVERSION_RATE,
) -> pd.DataFrame:
    """Turn scored users into operational recommendations.

    Parameters
    ----------
    scored : pd.DataFrame
        Output of :func:`score_users`.
    global_ltv : float
        Mean historical value, used to estimate the profit without
        action.
    organic_conversion_rate : float, default=0.20
        Share of the expected value assumed to happen organically.

    Returns
    -------
    pd.DataFrame
        One row per user with the recommended action, its rationale and
        maximum incentive, the expected profit without action and the
        incremental profit of acting, sorted by expected profit.
    """
    columns = [c for c in BASE_COLUMNS if c in scored.columns]
    recommendations = scored[columns].copy()
    if "Segment" not in recommendations.columns:
        recommendations["Segment"] = "Low Value"

    recommendations["recommended_action"] = recommendations.apply(
        recommend_action, axis=1
    )
    segments = recommendations["Segment"].map(canonical_segment)
    recommendations["strategy_rationale"] = segments.map(
        {k: v["rationale"] for k, v in SEGMENT_ACTIONS.items()}
    ).fillna("Unidentified segment")
    recommendations["max_incentive"] = segments.map(
        {k: v["max_incentive"] for k, v in SEGMENT_ACTIONS.items()}
    ).fillna(0)

    recommendations["expected_profit_no_action"] = (
        recommendations["prob_retention"]
        * global_ltv
        * organic_conversion_rate
    )
    recommendations["incremental_profit"] = (
        recommendations["expected_profit"]
        - recommendations["expected_profit_no_action"]
    )
    return recommendations.sort_values("expected_profit", ascending=False)


def export_crm(
    recommendations: pd.DataFrame, output_dir: str | Path | None = None
) -> Path:
    """Write the row-level recommendations for CRM integration.

    The file holds one row per ``user_id``. It is a local artefact and
    must not be published.

    Parameters
    ----------
    recommendations : pd.DataFrame
        Output of :func:`generate_recommendations`.
    output_dir : str or Path, optional
        Destination folder. Defaults to the project ``output`` folder.

    Returns
    -------
    Path
        The CSV file written.
    """
    folder = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / CRM_FILENAME
    recommendations.to_csv(path, index=False)
    return path


def export_results(
    sheets: dict[str, pd.DataFrame], output_dir: str | Path | None = None
) -> Path:
    """Export the result tables to Excel, or to CSV without openpyxl.

    Parameters
    ----------
    sheets : dict of str to pd.DataFrame
        Tables keyed by sheet name. Empty tables are skipped.
    output_dir : str or Path, optional
        Destination folder. Defaults to the project ``output`` folder.

    Returns
    -------
    Path
        The workbook ``marketing_campaign_recommendations.xlsx``, or the
        folder ``marketing_campaign_recommendations_csv`` when openpyxl
        is not installed.
    """
    folder = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    folder.mkdir(parents=True, exist_ok=True)
    tables = {name: table for name, table in sheets.items() if not table.empty}
    workbook = folder / "marketing_campaign_recommendations.xlsx"
    try:
        with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
            for name, table in tables.items():
                table.to_excel(writer, sheet_name=name, index=False)
        return workbook
    except ModuleNotFoundError as error:
        if error.name != "openpyxl":
            raise
    csv_dir = folder / "marketing_campaign_recommendations_csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(csv_dir / f"{name}.csv", index=False)
    return csv_dir
