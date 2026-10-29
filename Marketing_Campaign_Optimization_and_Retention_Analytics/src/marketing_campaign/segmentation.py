"""RFM-proxy segmentation built from campaign exposures.

These are proxies, not transactional RFM: recency is the days since the
last exposure, frequency the number of exposures and monetary the revenue
assumed for the conversions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from marketing_campaign.config import (
    RANDOM_STATE,
    REVENUE_PER_CONVERSION,
    SEGMENT_LABELS,
    SELECTED_K,
)


def build_rfm(exposures: pd.DataFrame) -> pd.DataFrame:
    """Compute recency, frequency and monetary proxies per user.

    Parameters
    ----------
    exposures : pd.DataFrame
        Cleaned exposures (one row per user and date served) from
        :func:`marketing_campaign.data.clean_data`.

    Returns
    -------
    pd.DataFrame
        One row per ``user_id`` with ``Recency`` (days), ``Frequency``
        and ``Monetary`` columns.
    """
    frame = exposures.copy()
    frame["date_served"] = pd.to_datetime(
        frame["date_served"], errors="coerce"
    )
    reference_date = frame["date_served"].max()

    recency = (
        frame.groupby("user_id")["date_served"]
        .max()
        .apply(lambda served: (reference_date - served).days)
        .rename("Recency")
        .reset_index()
    )
    frequency = (
        frame.groupby("user_id")["date_served"]
        .count()
        .rename("Frequency")
        .reset_index()
    )
    monetary = (
        frame.groupby("user_id")["converted_flag"]
        .sum()
        .mul(REVENUE_PER_CONVERSION)
        .rename("Monetary")
        .reset_index()
    )

    rfm = recency.merge(frequency, on="user_id").merge(
        monetary, on="user_id"
    )
    rfm["Recency"] = rfm["Recency"].fillna(rfm["Recency"].max() + 1)
    rfm["Frequency"] = rfm["Frequency"].fillna(0)
    rfm["Monetary"] = rfm["Monetary"].fillna(0)
    return rfm


def assign_segments(
    rfm: pd.DataFrame,
    n_clusters: int = SELECTED_K,
    random_state: int = RANDOM_STATE,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Cluster the RFM proxies and name the clusters by monetary value.

    The proxies are log-transformed and standardised before K-Means. The
    cluster with the highest mean ``Monetary`` becomes "High Value", the
    next "Medium Value" and the last "Low Value" (or the names in
    ``labels``).

    Parameters
    ----------
    rfm : pd.DataFrame
        Output of :func:`build_rfm`.
    n_clusters : int, default=3
        Number of clusters. Only the first three have a name, so it must
        not exceed ``len(SEGMENT_LABELS)``.
    random_state : int, default=42
        Seed of K-Means.
    labels : list of str, optional
        Names from the highest to the lowest value, for example
        ``i18n.SEGMENT_LABELS["pt"]``. Defaults to the English names.

    Returns
    -------
    pd.DataFrame
        A copy of ``rfm`` with ``RFM_Cluster`` and ``Segment`` columns.

    Raises
    ------
    ValueError
        If ``n_clusters`` is larger than the number of segment labels.
    """
    names_list = list(labels) if labels else SEGMENT_LABELS
    if n_clusters > len(names_list):
        raise ValueError(
            f"n_clusters must be at most {len(names_list)}."
        )
    scaled = StandardScaler().fit_transform(
        np.log1p(rfm[["Recency", "Frequency", "Monetary"]])
    )
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)

    segmented = rfm.copy()
    segmented["RFM_Cluster"] = model.fit_predict(scaled)

    ranked = (
        segmented.groupby("RFM_Cluster")["Monetary"]
        .mean()
        .sort_values(ascending=False)
    )
    names = {
        cluster: names_list[position]
        for position, cluster in enumerate(ranked.index)
    }
    segmented["Segment"] = segmented["RFM_Cluster"].map(names)
    return segmented


def attach_segments(
    users: pd.DataFrame, segmented_rfm: pd.DataFrame
) -> pd.DataFrame:
    """Add the RFM cluster and segment to the canonical user base.

    Parameters
    ----------
    users : pd.DataFrame
        Canonical user-level base from
        :func:`marketing_campaign.data.aggregate_stable_users`.
    segmented_rfm : pd.DataFrame
        Output of :func:`assign_segments`.

    Returns
    -------
    pd.DataFrame
        ``users`` with ``RFM_Cluster`` and ``Segment`` (left join on
        ``user_id``; any previous values of those columns are replaced).
    """
    return users.drop(
        columns=["RFM_Cluster", "Segment"], errors="ignore"
    ).merge(
        segmented_rfm[["user_id", "RFM_Cluster", "Segment"]],
        on="user_id",
        how="left",
    )
