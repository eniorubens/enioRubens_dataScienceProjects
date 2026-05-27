"""
segmentation.py
---------------
K-Means customer segmentation: transformer, search, naming and validation.

The ``KMeansClusterAdder`` is a scikit-learn-compatible transformer that can
live inside a ``Pipeline``, ensuring K-Means is fitted **only** on training
data and ``predict()`` is used for validation and hold-out sets.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from src.config import N_CLUSTERS, RANDOM_STATE
from src.multilang import t


# ── Low-level helpers ──────────────────────────────────────────────────────

def fit_kmeans(
    x: np.ndarray,
    k: int,
    n_init: int = 10,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, float]:
    """Fit K-Means for a single candidate *k*.

    Parameters
    ----------
    x : np.ndarray
        Scaled feature matrix.
    k : int
        Number of clusters.
    n_init : int
        Number of centroid initialisations.
    random_state : int

    Returns
    -------
    labels : np.ndarray
    inertia : float
    """
    model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    labels = model.fit_predict(x)
    return labels, float(model.inertia_)


def silhouette(
    x: np.ndarray,
    labels: np.ndarray,
    sample: int = 2500,
    random_state: int = RANDOM_STATE,
) -> float:
    """Silhouette score with sub-sampling to keep execution fast."""
    return float(
        silhouette_score(
            x, labels,
            sample_size=min(sample, len(x)),
            random_state=random_state,
        )
    )


def search_k(
    x_scaled: np.ndarray,
    k_range: range = range(2, 11),
    n_init: int = 10,
) -> pd.DataFrame:
    """Grid-search over *k_range* and return inertia + silhouette per k.

    Parameters
    ----------
    x_scaled : np.ndarray
        Pre-scaled cluster input.
    k_range : range
        Candidate values of k to evaluate.
    n_init : int
        K-Means initialisations per k.

    Returns
    -------
    pd.DataFrame
        Columns: ``k``, ``inertia``, ``silhouette_score``.
    """
    rows = []
    for k in k_range:
        labels, inertia = fit_kmeans(x_scaled, k, n_init=n_init)
        rows.append({
            "k": k,
            "inertia": inertia,
            "silhouette_score": silhouette(x_scaled, labels),
        })
    return pd.DataFrame(rows)


# ── Segment naming ─────────────────────────────────────────────────────────

def rank01(s: pd.Series, high: bool = True) -> pd.Series:
    """Rank a series to [0, 1]; highest value → 1 when *high* is ``True``."""
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(0.5, index=s.index)
    norm = (s - mn) / (mx - mn)
    return norm if high else (1 - norm)


def segment_names(profiles: pd.DataFrame) -> dict[int, str]:
    """Assign a business label to each cluster based on RFM profile scores.

    Parameters
    ----------
    profiles : pd.DataFrame
        Cluster-level aggregation with columns
        ``Recency``, ``Frequency``, ``Monetary``.

    Returns
    -------
    dict[int, str]
        Mapping ``cluster_id → segment_name`` (English labels).
    """
    p = profiles.copy()
    p["score_recency"]  = rank01(p["Recency"],  high=False)   # low recency = good
    p["score_frequency"] = rank01(p["Frequency"], high=True)
    p["score_monetary"]  = rank01(p["Monetary"],  high=True)
    p["rfm_score"] = (
        p["score_recency"] * 0.35
        + p["score_frequency"] * 0.35
        + p["score_monetary"] * 0.30
    )
    p = p.sort_values("rfm_score", ascending=False).reset_index(drop=True)

    labels = [
        "Champions",
        "Loyal Customers",
        "High Value at Risk",
        "Occasional Buyers",
        "Low Value",
        "Inactive Customers",
    ]
    n = len(p)
    names = labels[:n] + [f"Segment {i+1}" for i in range(n, max(0, n - len(labels)))]

    cluster_ids = p["Cluster"].tolist()
    return dict(zip(cluster_ids, names))


def validate_segment_names(
    profiles: pd.DataFrame,
    assigned_names: dict[int, str],
) -> None:
    """Warn when expected high-value segments have unexpectedly low monetary values."""
    high_value_segments = {"Champions", "Loyal Customers"}
    median_monetary = profiles["Monetary"].median()

    for cluster_id, name in assigned_names.items():
        if name in high_value_segments:
            cluster_monetary = profiles.loc[
                profiles["Cluster"] == cluster_id, "Monetary"
            ].values
            if len(cluster_monetary) > 0 and cluster_monetary[0] < median_monetary:
                warnings.warn(
                    f"Segment '{name}' (cluster {cluster_id}) has monetary value "
                    f"below the dataset median. Review cluster assignments.",
                    UserWarning,
                    stacklevel=2,
                )


# ── sklearn Transformer ────────────────────────────────────────────────────

class KMeansClusterAdder(BaseEstimator, TransformerMixin):
    """Add a ``PredictiveCluster`` column via K-Means fitted on training data only.

    Designed to sit as the first step in a scikit-learn ``Pipeline`` so that
    the clustering is never contaminated by validation or hold-out samples.

    Parameters
    ----------
    cluster_features : list[str] or None
        Feature columns used for clustering.  When ``None``, falls back to
        the column list seen during ``fit``.
    log_features : list[str] or None
        Subset of *cluster_features* to log-transform before scaling.
    n_clusters : int
        Number of K-Means clusters.
    random_state : int
    n_init : int
        K-Means initialisations.
    """

    def __init__(
        self,
        cluster_features: list[str] | None = None,
        log_features: list[str] | None = None,
        n_clusters: int = N_CLUSTERS,
        random_state: int = RANDOM_STATE,
        n_init: int = 20,
    ) -> None:
        self.cluster_features = cluster_features
        self.log_features = log_features
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init

    # sklearn convention: all parameters set in __init__ only
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "KMeansClusterAdder":
        """Fit scaler and K-Means on *X* (training set only)."""
        x_df = self._as_frame(X, fit=True)
        cluster_input = self._cluster_input(x_df)
        self.cluster_scaler_ = StandardScaler()
        scaled = self.cluster_scaler_.fit_transform(cluster_input)
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
        )
        self.kmeans_.fit(scaled)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Append ``PredictiveCluster`` column to *X*."""
        check_is_fitted(self, ["kmeans_", "cluster_scaler_"])
        x_df = self._as_frame(X, fit=False)
        cluster_input = self._cluster_input(x_df)
        scaled = self.cluster_scaler_.transform(cluster_input)
        x_out = x_df.copy()
        x_out["PredictiveCluster"] = self.kmeans_.predict(scaled).astype(str)
        return x_out

    # ── Private helpers ────────────────────────────────────────────────────

    def _as_frame(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            if fit:
                self.feature_names_in_ = list(X.columns)
            return X.copy()
        return pd.DataFrame(X, columns=self.feature_names_in_)

    def _cluster_input(self, x_df: pd.DataFrame) -> pd.DataFrame:
        features = self.cluster_features or list(x_df.select_dtypes(include="number").columns)
        log_cols = self.log_features or []
        cluster_input = x_df[features].copy()
        for col in log_cols:
            if col in cluster_input.columns:
                cluster_input[col] = np.log1p(cluster_input[col].clip(lower=0))
        return cluster_input
