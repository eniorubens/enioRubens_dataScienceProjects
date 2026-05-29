"""
estimators.py
-------------
Registry of all supported classifiers and factory function.
"""
from __future__ import annotations

import catboost
import lightgbm
import xgboost
from imblearn.ensemble import BalancedRandomForestClassifier  # noqa: F401
from sklearn import (
    discriminant_analysis,
    ensemble,
    linear_model,
    naive_bayes,
    neural_network,
    svm,
    tree,
)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
EstimatorGroups = tuple[
    list[str],  # linear
    list[str],  # ensemble
    list[str],  # tree
    list[str],  # neural
    list[str],  # xgb
    list[str],  # lgbm
    list[str],  # catboost
    list[str],  # extra  (svm, lda, bayes, balanced_ensemble)
    list[str],  # all    (used by train_all_models)
]

_EXCLUDED: frozenset[str] = frozenset(
    {
        "StackingClassifier",
        "RidgeClassifierCV",
        "SGDOneClassSVM",
        "VotingClassifier",
        "AdaBoostClassifier",
        "BaggingClassifier",
    }
)


def define_estimators() -> EstimatorGroups:
    """
    Build and return all estimator groups used in experiments.

    Returns
    -------
    EstimatorGroups
        Tuple of (linear, ensemble, tree, neural, xgb, lgbm, catboost, all).
    """
    linear_estimators: list[str] = [
        e for e in set(linear_model.__all__) if "Class" in e
    ]
    linear_estimators.append("LogisticRegression")

    ensemble_estimators: list[str] = [
        e for e in set(ensemble.__all__) if "Class" in e
    ]

    tree_estimators: list[str] = [
        e for e in set(tree.__all__) if "Class" in e
    ]

    neural_estimators: list[str] = [
        e for e in set(neural_network.__all__) if "Class" in e
    ]

    xgb_estimators: list[str] = ["XGBClassifier"]
    lgbm_estimators: list[str] = ["LGBMClassifier"]
    catboost_estimators: list[str] = ["CatBoostClassifier"]

    extra_estimators: list[str] = [
        "SVC",
        "LinearDiscriminantAnalysis",
        "GaussianNB",
        "BalancedRandomForestClassifier",
    ]

    all_estimators: list[str] = (
        linear_estimators + ensemble_estimators + tree_estimators
    )
    all_estimators = [e for e in all_estimators if e not in _EXCLUDED]
    all_estimators.extend(["LogisticRegression", "XGBClassifier", "LGBMClassifier"])
    all_estimators.extend(extra_estimators)

    return (
        linear_estimators,
        ensemble_estimators,
        tree_estimators,
        neural_estimators,
        xgb_estimators,
        lgbm_estimators,
        catboost_estimators,
        extra_estimators,
        all_estimators,
    )


_EXTRA_CLASS_MAP: dict[str, type] = {
    "SVC": svm.SVC,
    "LinearDiscriminantAnalysis": discriminant_analysis.LinearDiscriminantAnalysis,
    "GaussianNB": naive_bayes.GaussianNB,
    "BalancedRandomForestClassifier": BalancedRandomForestClassifier,
}


def get_estimator(
    estimator_name: str,
    linear_estimators: list[str],
    ensemble_estimators: list[str],
    tree_estimators: list[str],
    neural_estimators: list[str],
    xgb_estimators: list[str],
    lgbm_estimators: list[str],
    catboost_estimators: list[str],
    extra_estimators: list[str] | None = None,
) -> type:
    """
    Return the estimator **class** (not instance) from its string name.

    Parameters
    ----------
    estimator_name : str
        Classifier class name (e.g. ``"RandomForestClassifier"``).
    linear_estimators, ensemble_estimators, tree_estimators,
    neural_estimators, xgb_estimators, lgbm_estimators,
    catboost_estimators : list[str]
        Group lists returned by :func:`define_estimators`.
    extra_estimators : list[str] | None
        Optional extra group (SVC, LDA, GaussianNB, BalancedRandomForestClassifier).

    Returns
    -------
    type
        Uninstantiated estimator class.

    Raises
    ------
    ValueError
        If *estimator_name* is not found in any group.
    """
    if estimator_name in linear_estimators:
        return getattr(linear_model, estimator_name)
    if estimator_name in ensemble_estimators:
        return getattr(ensemble, estimator_name)
    if estimator_name in tree_estimators:
        return getattr(tree, estimator_name)
    if estimator_name in xgb_estimators:
        return getattr(xgboost, estimator_name)
    if estimator_name in lgbm_estimators:
        return getattr(lightgbm, estimator_name)
    if estimator_name in catboost_estimators:
        return getattr(catboost, estimator_name)
    if estimator_name in neural_estimators:
        return getattr(neural_network, estimator_name)
    if extra_estimators and estimator_name in extra_estimators:
        return _EXTRA_CLASS_MAP[estimator_name]

    raise ValueError(
        f"Estimator '{estimator_name}' not found in any registered group."
    )
