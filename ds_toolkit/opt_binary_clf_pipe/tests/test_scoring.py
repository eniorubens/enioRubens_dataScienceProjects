"""Regression tests for the cross-validation scorers in ``opt_binary_clf_pipe.scoring``."""
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.svm import LinearSVC

from opt_binary_clf_pipe import get_classification_scoring


@pytest.fixture(scope="module")
def data():
    return make_classification(
        n_samples=800, n_features=12, n_informative=5, weights=[0.8, 0.2], flip_y=0.08, random_state=0
    )


def test_roc_auc_scorer_uses_scores_not_hard_labels(data):
    """The scorer must rank by predicted probability, not by the 0/1 predictions.

    ``make_scorer(roc_auc_score)`` calls ``predict``, so the "AUC" it returned was the
    AUC of a single operating point, which equals the balanced accuracy.
    """
    X, y = data
    model = LogisticRegression(max_iter=1000).fit(X, y)
    scorer = get_classification_scoring()["roc_auc"]

    value = scorer(model, X, y)

    assert value == pytest.approx(roc_auc_score(y, model.predict_proba(X)[:, 1]))
    assert value != pytest.approx(balanced_accuracy_score(y, model.predict(X)))


def test_cross_validated_roc_auc_is_not_the_balanced_accuracy(data):
    X, y = data
    scores = cross_validate(
        LogisticRegression(max_iter=1000),
        X,
        y,
        cv=StratifiedKFold(n_splits=4, shuffle=True, random_state=0),
        scoring=get_classification_scoring(),
        return_train_score=True,
    )

    assert not np.allclose(scores["test_roc_auc"], scores["test_balanced_accuracy"])
    assert not np.allclose(scores["train_roc_auc"], scores["train_balanced_accuracy"])
    assert np.all(scores["test_roc_auc"] > scores["test_balanced_accuracy"])  # AUC >= BA for a sane model


def test_roc_auc_scorer_supports_estimators_without_predict_proba(data):
    """Models exposing only ``decision_function`` (LinearSVC, SGD hinge...) must still be scored."""
    X, y = data
    model = LinearSVC(dual=False).fit(X, y)
    scorer = get_classification_scoring()["roc_auc"]

    assert scorer(model, X, y) == pytest.approx(roc_auc_score(y, model.decision_function(X)))


def test_other_scorers_are_unchanged(data):
    X, y = data
    model = LogisticRegression(max_iter=1000).fit(X, y)
    scoring = get_classification_scoring()

    assert set(scoring) == {"roc_auc", "precision_macro", "recall_macro", "f1_macro", "balanced_accuracy"}
    assert scoring["balanced_accuracy"](model, X, y) == pytest.approx(balanced_accuracy_score(y, model.predict(X)))
