"""
pipeline_builder.py
-------------------
Reusable factories for sklearn / imbalanced-learn preprocessing pipelines.
"""
from __future__ import annotations

import category_encoders
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.pipeline import make_pipeline as make_pipeline_with_sampler
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.pipeline import Pipeline


def build_num_transformer(
    scaler: object | None,
    normalizer: object | None,
) -> ImbPipeline | str:
    """
    Build a numeric preprocessing sub-pipeline.

    Parameters
    ----------
    scaler : object | None
        Fitted-compatible scaler (e.g. ``StandardScaler``), or ``None``.
    normalizer : object | None
        Fitted-compatible normalizer (e.g. ``PowerTransformer``), or ``None``.

    Returns
    -------
    ImbPipeline | str
        A two-step pipeline of ``(normalizer, scaler)`` when at least one
        transformer is provided, or the string ``"passthrough"`` otherwise.
    """
    steps: list = []

    if normalizer is not None:
        steps.append(("normalizer", normalizer))

    if scaler is not None:
        steps.append(("scaler", scaler))

    if not steps:
        return "passthrough"

    return ImbPipeline(steps=steps)


def build_preprocessor(num_transformer: object) -> ColumnTransformer:
    """
    Build a ``ColumnTransformer`` that applies *num_transformer* to numeric
    columns and passes through all others.

    Parameters
    ----------
    num_transformer : object
        Result of :func:`build_num_transformer` — either a pipeline or
        ``"passthrough"``.

    Returns
    -------
    ColumnTransformer
    """
    return ColumnTransformer(
        transformers=[
            ("num", num_transformer, selector(dtype_exclude=object))
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def build_baseline_preprocessor() -> ColumnTransformer:
    """
    Build the fixed preprocessor used for baseline (non-optimised) models.

    Applies ``OrdinalEncoder`` to all object columns and passes numeric
    columns through unchanged.

    Returns
    -------
    ColumnTransformer
    """
    categorical_transformer = Pipeline(
        steps=[("encoder", category_encoders.OrdinalEncoder())]
    )

    return ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, selector(dtype_include=object))
        ],
        remainder="passthrough",
    )


def build_full_pipeline(
    encoder: object,
    preprocessor: ColumnTransformer,
    sampler: object | None,
    feature_selector: object,
    model: object,
    cache_dir: str,
    balanced: bool,
) -> Pipeline:
    """
    Assemble the full dynamic pipeline from its components.

    Parameters
    ----------
    encoder : object
        Categorical encoder or ``"passthrough"``.
    preprocessor : ColumnTransformer
        Numeric preprocessing transformer.
    sampler : object | None
        Over-sampler (e.g. ``SMOTE``) or ``None``.
    feature_selector : object
        Feature selector or ``"passthrough"``.
    model : object
        Instantiated classifier.
    cache_dir : str
        Path for joblib pipeline caching.
    balanced : bool
        When ``True`` the sampler step is skipped (class weights handle
        imbalance directly inside the estimator).

    Returns
    -------
    Pipeline
        Full imbalanced-learn compatible pipeline.
    """
    components: list = [encoder, preprocessor]

    if not balanced and sampler is not None:
        components.append(sampler)

    components.extend([feature_selector, model])

    return make_pipeline_with_sampler(*components, memory=cache_dir)
