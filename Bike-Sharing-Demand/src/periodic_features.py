"""Periodic feature transformers extracted from RegressionOptimizer (cell [137]).

Faithful port of:
  PeriodicSplineTransformer
  SinTransformer
  CosTransformer
  DebugTransformer
  CustomPreprocessorWithNystroem

These classes are also imported inside `src/optimizer.py` so that
`RegressionOptimizer.modeling_transformers()` can build the correct pipelines.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import SplineTransformer


# ---------------------------------------------------------------------------
# PeriodicSplineTransformer
# ---------------------------------------------------------------------------


class PeriodicSplineTransformer(BaseEstimator, TransformerMixin):
    """Periodic spline encoding for cyclic features (e.g. hour, month)."""

    def __init__(self, period: int, n_splines: int | None = None, degree: int = 3) -> None:
        self.period = period
        self.n_splines = n_splines if n_splines is not None else period
        self.degree = degree

    def fit(self, X, y=None) -> "PeriodicSplineTransformer":
        return self

    def transform(self, X):
        n_knots = self.n_splines + 1
        spline_transformer = SplineTransformer(
            degree=self.degree,
            n_knots=n_knots,
            knots=np.linspace(0, self.period, n_knots).reshape(n_knots, 1),
            extrapolation="periodic",
            include_bias=True,
        )
        return spline_transformer.fit_transform(X)


# ---------------------------------------------------------------------------
# SinTransformer
# ---------------------------------------------------------------------------


class SinTransformer(BaseEstimator, TransformerMixin):
    """Sine encoding for a periodic feature with the given period."""

    def __init__(self, period: int) -> None:
        self.period = period

    def fit(self, X, y=None) -> "SinTransformer":
        return self

    def transform(self, X):
        return np.sin(X / self.period * 2 * np.pi)


# ---------------------------------------------------------------------------
# CosTransformer
# ---------------------------------------------------------------------------


class CosTransformer(BaseEstimator, TransformerMixin):
    """Cosine encoding for a periodic feature with the given period."""

    def __init__(self, period: int) -> None:
        self.period = period

    def fit(self, X, y=None) -> "CosTransformer":
        return self

    def transform(self, X):
        return np.cos(X / self.period * 2 * np.pi)


# ---------------------------------------------------------------------------
# DebugTransformer
# ---------------------------------------------------------------------------


class DebugTransformer(BaseEstimator, TransformerMixin):
    """Pass-through transformer that prints NaN presence for debugging."""

    def fit(self, X, y=None) -> "DebugTransformer":
        return self

    def transform(self, X):
        import pandas as pd

        if hasattr(X, "isnull"):
            has_nans = X.isnull().any().any()
            print(f"Data contains NaNs: {has_nans}")
            if has_nans:
                print("NaNs found in columns:", X.columns[X.isnull().any()])
        return X


# ---------------------------------------------------------------------------
# CustomPreprocessorWithNystroem
# ---------------------------------------------------------------------------


class CustomPreprocessorWithNystroem(BaseEstimator, TransformerMixin):
    """Cyclic-spline + categorical encoding + Nystroem kernel expansion.

    Applies categorical encoding and periodic spline transforms, then
    expands to an approximate polynomial kernel feature space via Nystroem.
    Some encoders can emit NaN for unseen temporal categories in CV folds —
    the ``kernel_imputer`` fills those before the Nystroem step.
    """

    def __init__(self, encoder_class, categorical_features: list, scaler) -> None:
        self.encoder_class = encoder_class
        self.categorical_features = categorical_features
        self.scaler = scaler
        self.cyclic_spline_transformer = None
        self.kernel_imputer = SimpleImputer(strategy="constant", fill_value=0.0)
        self.nystroem_transformer = Nystroem(
            kernel="poly", degree=2, n_components=300, random_state=0
        )

    def fit(self, X, y=None) -> "CustomPreprocessorWithNystroem":
        self.cyclic_spline_transformer = ColumnTransformer(
            transformers=[
                ("categorical", self.encoder_class, self.categorical_features),
                ("cyclic_month", PeriodicSplineTransformer(12, n_splines=6), ["Month"]),
                ("cyclic_hour", PeriodicSplineTransformer(24, n_splines=12), ["Hour"]),
            ],
            remainder=self.scaler,
        )
        X_transformed = self.cyclic_spline_transformer.fit_transform(X, y)
        X_transformed = self.kernel_imputer.fit_transform(X_transformed)
        self.nystroem_transformer.fit(X_transformed)
        return self

    def transform(self, X):
        X_transformed = self.cyclic_spline_transformer.transform(X)
        X_transformed = self.kernel_imputer.transform(X_transformed)
        return self.nystroem_transformer.transform(X_transformed)
