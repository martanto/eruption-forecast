"""Tests to verify scikit-learn-intelex patch is applied to ClassifierModel instances."""

import pytest

# Trigger patch_sklearn() before any sklearn imports
import eruption_forecast.model  # noqa: F401

from eruption_forecast.model.classifier_model import ClassifierModel


def _is_patched(model: object) -> bool:
    """Return True if the model's module originates from sklearnex or daal4py."""
    module = type(model).__module__
    return "sklearnex" in module or "daal4py" in module


# ---------------------------------------------------------------------------
# Global patch
# ---------------------------------------------------------------------------


def test_sklearnex_patch_is_active() -> None:
    """sklearn_is_patched() returns True after the model package is imported."""
    from sklearnex import sklearn_is_patched

    assert sklearn_is_patched(), "scikit-learn-intelex patch was not applied"


# ---------------------------------------------------------------------------
# Classifiers that sklearnex accelerates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("classifier", ["rf", "lite-rf"])
def test_random_forest_is_patched(classifier: str) -> None:
    """RandomForestClassifier returned by ClassifierModel is the sklearnex version."""
    model = ClassifierModel(classifier).model
    assert _is_patched(model), (
        f"Expected sklearnex RandomForestClassifier, got: {type(model).__module__}"
    )


def test_svm_is_patched() -> None:
    """SVC returned by ClassifierModel is the sklearnex version."""
    model = ClassifierModel("svm").model
    assert _is_patched(model), (
        f"Expected sklearnex SVC, got: {type(model).__module__}"
    )


def test_logistic_regression_is_patched() -> None:
    """LogisticRegression returned by ClassifierModel is the sklearnex version."""
    model = ClassifierModel("lr").model
    assert _is_patched(model), (
        f"Expected sklearnex LogisticRegression, got: {type(model).__module__}"
    )


def test_knn_is_patched() -> None:
    """KNeighborsClassifier returned by ClassifierModel is the sklearnex version."""
    model = ClassifierModel("knn").model
    assert _is_patched(model), (
        f"Expected sklearnex KNeighborsClassifier, got: {type(model).__module__}"
    )


# ---------------------------------------------------------------------------
# Classifiers that fall back to standard sklearn (not supported by sklearnex)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("classifier", ["gb", "nn", "nb", "dt"])
def test_unsupported_classifiers_still_instantiate(classifier: str) -> None:
    """Classifiers not accelerated by sklearnex still instantiate without error."""
    model = ClassifierModel(classifier).model
    assert model is not None, f"{classifier} failed to instantiate after patching"


def test_voting_classifier_sub_estimators_rf_is_patched() -> None:
    """The 'rf' sub-estimator inside VotingClassifier is the sklearnex version."""
    voting = ClassifierModel("voting").model
    rf = dict(voting.estimators)["rf"]
    assert _is_patched(rf), (
        f"Expected sklearnex RandomForestClassifier inside VotingClassifier, "
        f"got: {type(rf).__module__}"
    )
