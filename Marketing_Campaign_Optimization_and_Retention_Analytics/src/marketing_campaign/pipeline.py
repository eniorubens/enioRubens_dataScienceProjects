"""End-to-end run of the analysis, reproducing the notebook results."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from marketing_campaign import (
    data,
    economics,
    experiment,
    funnel,
    recommendations,
    retention_model,
    segmentation,
)


@dataclass(frozen=True)
class PipelineResult:
    """Every table produced by one run of the analysis.

    Attributes
    ----------
    raw_rows : int
        Rows of the raw file.
    exposures : pd.DataFrame
        Cleaned exposures.
    users : pd.DataFrame
        Canonical user-level base, with the RFM segment.
    switchers : int
        Users shown more than one variant, excluded from the base.
    funnel : pd.DataFrame
        Funnel summary.
    channels : pd.DataFrame
        Channel performance.
    ab_test : pd.DataFrame
        A/B test of the variants.
    ltv : economics.LtvSummary
        Historical customer value.
    roi : pd.DataFrame
        Simulated ROI by channel.
    experiments : pd.DataFrame
        Retention scenarios ranked by ROC AUC.
    model : retention_model.RetentionModel
        The selected retention model.
    scored : pd.DataFrame
        Users with retention probability and expected value.
    recommendations : pd.DataFrame
        Row-level operational recommendations.
    """

    raw_rows: int
    exposures: pd.DataFrame
    users: pd.DataFrame
    switchers: int
    funnel: pd.DataFrame
    channels: pd.DataFrame
    ab_test: pd.DataFrame
    ltv: economics.LtvSummary
    roi: pd.DataFrame
    experiments: pd.DataFrame
    model: retention_model.RetentionModel
    scored: pd.DataFrame
    recommendations: pd.DataFrame


def run_pipeline(path: str | Path | None = None) -> PipelineResult:
    """Run the analysis from the raw file to the recommendations.

    The steps follow the notebooks: cleaning, the canonical stable-user
    base, RFM segments, funnel, channels, A/B test, customer value, ROI,
    the retention model and the recommendations.

    Parameters
    ----------
    path : str or Path, optional
        Campaign CSV file. Defaults to the dataset shipped with the
        project.

    Returns
    -------
    PipelineResult
        All intermediate and final tables.
    """
    raw = data.load_data(path)
    exposures = data.clean_data(raw)
    users, switchers = data.aggregate_stable_users(exposures)
    segments = segmentation.assign_segments(segmentation.build_rfm(exposures))
    users = segmentation.attach_segments(users, segments)

    channels = funnel.analyze_channel_performance(users)
    ltv = economics.estimate_ltv(users)
    experiments = retention_model.run_retention_experiments(users)
    best = experiments.iloc[0]
    users = retention_model.prepare_retention_model_frame(users)
    model = retention_model.build_retention_model(
        users, include_converted=bool(best["include_converted"])
    )
    scored = recommendations.score_users(users, model, ltv)

    return PipelineResult(
        raw_rows=len(raw),
        exposures=exposures,
        users=users,
        switchers=switchers,
        funnel=funnel.build_funnel(users),
        channels=channels,
        ab_test=experiment.run_ab_test(users),
        ltv=ltv,
        roi=economics.simulate_roi(channels, users, ltv),
        experiments=experiments,
        model=model,
        scored=scored,
        recommendations=recommendations.generate_recommendations(
            scored, ltv.global_ltv
        ),
    )
