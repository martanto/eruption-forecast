"""Unit tests for ClassifierModel.

Tests cover construction, CV splitter selection, model/grid defaults for every
supported classifier, property accessors, and mutation helpers.  No I/O or
training — ClassifierModel is a pure factory/configuration class.
"""

import pytest
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit, TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from eruption_forecast.model.classifier_model import ClassifierModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALL_CLASSIFIERS = ["svm", "knn", "dt", "rf", "gb", "xgb", "nn", "nb", "lr", "voting", "lite-rf"]

CLASSIFIER_TYPES: dict[str, type] = {
    "svm": SVC,
    "knn": KNeighborsClassifier,
    "dt": DecisionTreeClassifier,
    "rf": RandomForestClassifier,
    "lite-rf": RandomForestClassifier,
    "gb": GradientBoostingClassifier,
    "xgb": XGBClassifier,
    "nn": MLPClassifier,
    "nb": GaussianNB,
    "lr": LogisticRegression,
    "voting": VotingClassifier,
}


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestClassifierModelInit:
    """Test default construction behaviour."""

    def test_stores_classifier_string(self) -> None:
        """Constructor stores the classifier identifier unchanged."""
        clf = ClassifierModel("rf")
        assert clf.classifier == "rf"

    def test_default_random_state_is_none(self) -> None:
        """random_state defaults to None when not provided."""
        clf = ClassifierModel("rf")
        assert clf.random_state is None

    def test_default_cv_strategy_is_shuffle_stratified(self) -> None:
        """cv_strategy defaults to 'shuffle-stratified'."""
        clf = ClassifierModel("rf")
        assert clf.cv_strategy == "shuffle-stratified"

    def test_cv_name_set_on_init(self) -> None:
        """cv_name is derived from the CV splitter class on init."""
        clf = ClassifierModel("rf", cv_strategy="stratified")
        assert clf.cv_name == "StratifiedKFold"

    def test_custom_n_splits_stored(self) -> None:
        """n_splits is stored as provided."""
        clf = ClassifierModel("rf", n_splits=10)
        assert clf.n_splits == 10

    def test_custom_n_jobs_stored(self) -> None:
        """n_jobs is stored as provided."""
        clf = ClassifierModel("rf", n_jobs=4)
        assert clf.n_jobs == 4


# ---------------------------------------------------------------------------
# set_random_state
# ---------------------------------------------------------------------------


class TestSetRandomState:
    """Test set_random_state() validation and chaining."""

    def test_sets_random_state(self) -> None:
        """set_random_state() updates random_state attribute."""
        clf = ClassifierModel("rf")
        clf.set_random_state(42)
        assert clf.random_state == 42

    def test_returns_self_for_chaining(self) -> None:
        """set_random_state() returns the same instance."""
        clf = ClassifierModel("rf")
        result = clf.set_random_state(0)
        assert result is clf

    def test_zero_is_valid(self) -> None:
        """random_state=0 is a valid seed."""
        clf = ClassifierModel("rf")
        clf.set_random_state(0)
        assert clf.random_state == 0

    def test_negative_raises(self) -> None:
        """Negative random_state raises ValueError."""
        clf = ClassifierModel("rf")
        with pytest.raises(ValueError, match="random_state must be >= 0"):
            clf.set_random_state(-1)


# ---------------------------------------------------------------------------
# get_cv_splitter
# ---------------------------------------------------------------------------


class TestGetCvSplitter:
    """Test get_cv_splitter() returns correct CV object for each strategy."""

    def test_shuffle_returns_shuffle_split(self) -> None:
        """'shuffle' strategy returns ShuffleSplit."""
        clf = ClassifierModel("rf", cv_strategy="shuffle")
        assert isinstance(clf.get_cv_splitter(), ShuffleSplit)

    def test_shuffle_stratified_returns_stratified_shuffle_split(self) -> None:
        """'shuffle-stratified' strategy returns StratifiedShuffleSplit."""
        clf = ClassifierModel("rf", cv_strategy="shuffle-stratified")
        assert isinstance(clf.get_cv_splitter(), StratifiedShuffleSplit)

    def test_stratified_returns_stratified_k_fold(self) -> None:
        """'stratified' strategy returns StratifiedKFold."""
        clf = ClassifierModel("rf", cv_strategy="stratified")
        assert isinstance(clf.get_cv_splitter(), StratifiedKFold)

    def test_timeseries_returns_time_series_split(self) -> None:
        """'timeseries' strategy returns TimeSeriesSplit."""
        clf = ClassifierModel("rf", cv_strategy="timeseries")
        assert isinstance(clf.get_cv_splitter(), TimeSeriesSplit)

    def test_override_strategy_via_argument(self) -> None:
        """Passing strategy argument overrides self.cv_strategy."""
        clf = ClassifierModel("rf", cv_strategy="shuffle")
        assert isinstance(clf.get_cv_splitter(strategy="timeseries"), TimeSeriesSplit)

    def test_n_splits_is_propagated(self) -> None:
        """n_splits is passed through to the CV splitter."""
        clf = ClassifierModel("rf", cv_strategy="stratified", n_splits=7)
        cv = clf.get_cv_splitter()
        assert cv.n_splits == 7

    def test_random_state_is_propagated_to_stratified_kfold(self) -> None:
        """random_state is passed to StratifiedKFold."""
        clf = ClassifierModel("rf", cv_strategy="stratified", random_state=99)
        cv = clf.get_cv_splitter()
        assert cv.random_state == 99


# ---------------------------------------------------------------------------
# model property — correct type for every classifier
# ---------------------------------------------------------------------------


class TestModelProperty:
    """Test that model property returns the expected sklearn estimator type."""

    @pytest.mark.parametrize("classifier", ALL_CLASSIFIERS)
    def test_model_returns_correct_type(self, classifier: str) -> None:
        """model property returns an instance of the expected estimator class."""
        clf = ClassifierModel(classifier, random_state=0)
        model = clf.model
        assert isinstance(model, CLASSIFIER_TYPES[classifier])

    def test_svm_has_probability_enabled(self) -> None:
        """SVC model is created with probability=True."""
        clf = ClassifierModel("svm")
        assert clf.model.probability is True

    def test_rf_receives_n_jobs(self) -> None:
        """RandomForestClassifier is created with the configured n_jobs."""
        clf = ClassifierModel("rf", n_jobs=4)
        assert clf.model.n_jobs == 4

    def test_xgb_receives_n_jobs(self) -> None:
        """XGBClassifier is created with the configured n_jobs."""
        clf = ClassifierModel("xgb", n_jobs=4)
        assert clf.model.n_jobs == 4

    def test_voting_contains_rf_and_xgb(self) -> None:
        """VotingClassifier ensemble contains 'rf' and 'xgb' sub-estimators."""
        clf = ClassifierModel("voting")
        estimator_names = [name for name, _ in clf.model.estimators]
        assert "rf" in estimator_names
        assert "xgb" in estimator_names

    def test_model_setter_overrides_default(self) -> None:
        """Assigning to model replaces the default instance."""
        clf = ClassifierModel("rf")
        custom = RandomForestClassifier(n_estimators=999)
        clf.model = custom
        assert clf.model.n_estimators == 999

    def test_model_returns_same_instance_after_set(self) -> None:
        """model property returns the exact object that was set."""
        clf = ClassifierModel("rf")
        custom = RandomForestClassifier()
        clf.model = custom
        assert clf.model is custom


# ---------------------------------------------------------------------------
# grid property — correct keys for every classifier
# ---------------------------------------------------------------------------


class TestGridProperty:
    """Test that grid property returns a non-empty dict for every classifier."""

    @pytest.mark.parametrize("classifier", ALL_CLASSIFIERS)
    def test_grid_is_non_empty(self, classifier: str) -> None:
        """grid property returns a non-empty list or dict for every supported classifier."""
        clf = ClassifierModel(classifier)
        grid = clf.grid
        assert grid  # truthy: non-empty list or dict

    def _all_grid_keys(self, grid) -> set:
        """Collect all keys across list-of-dicts or plain dict grid."""
        if isinstance(grid, dict):
            return set(grid.keys())
        return {k for d in grid for k in d.keys()}

    def test_rf_grid_contains_n_estimators(self) -> None:
        """RF grid includes 'n_estimators' key."""
        clf = ClassifierModel("rf")
        assert "n_estimators" in self._all_grid_keys(clf.grid)

    def test_svm_grid_contains_c(self) -> None:
        """SVM grid includes 'C' key."""
        clf = ClassifierModel("svm")
        assert "C" in self._all_grid_keys(clf.grid)

    def test_xgb_grid_contains_learning_rate(self) -> None:
        """XGB grid includes 'learning_rate' key."""
        clf = ClassifierModel("xgb")
        assert "learning_rate" in self._all_grid_keys(clf.grid)

    def test_lr_grid_contains_penalty(self) -> None:
        """LR grid includes 'penalty' key."""
        clf = ClassifierModel("lr")
        assert "penalty" in self._all_grid_keys(clf.grid)

    def test_grid_setter_overrides_default(self) -> None:
        """Assigning a custom grid replaces the default."""
        clf = ClassifierModel("rf")
        custom_grid = {"n_estimators": [5]}
        clf.grid = custom_grid
        assert clf.grid == custom_grid

    def test_custom_grid_is_returned_unchanged(self) -> None:
        """Custom grid dict is returned by reference after setting."""
        clf = ClassifierModel("rf")
        custom_grid = {"n_estimators": [5, 10], "max_depth": [2]}
        clf.grid = custom_grid
        assert clf.grid is custom_grid


# ---------------------------------------------------------------------------
# name and slug properties
# ---------------------------------------------------------------------------


class TestNameAndSlugProperties:
    """Test name, slug_name, and slug_cv_name properties."""

    def test_name_returns_class_name(self) -> None:
        """name property returns the classifier's class name."""
        clf = ClassifierModel("rf")
        assert clf.name == "RandomForestClassifier"

    def test_lite_rf_name_is_prefixed(self) -> None:
        """'lite-rf' classifier name is prefixed with 'Lite'."""
        clf = ClassifierModel("lite-rf")
        assert clf.name == "LiteRandomForestClassifier"

    def test_slug_name_is_lowercase_hyphenated(self) -> None:
        """slug_name is lowercase with hyphens (no uppercase, no underscores)."""
        clf = ClassifierModel("rf")
        slug = clf.slug_name
        assert slug == slug.lower()
        assert "_" not in slug

    def test_slug_cv_name_is_lowercase_hyphenated(self) -> None:
        """slug_cv_name is lowercase with hyphens."""
        clf = ClassifierModel("rf", cv_strategy="stratified")
        slug = clf.slug_cv_name
        assert slug == slug.lower()
        assert "_" not in slug


# ---------------------------------------------------------------------------
# model_and_grid property
# ---------------------------------------------------------------------------


class TestModelAndGrid:
    """Test model_and_grid convenience property."""

    def test_returns_two_tuple(self) -> None:
        """model_and_grid returns a 2-tuple of (model, grid)."""
        clf = ClassifierModel("rf")
        result = clf.model_and_grid
        assert len(result) == 2

    def test_first_element_is_model(self) -> None:
        """First element of model_and_grid is the classifier instance."""
        clf = ClassifierModel("rf")
        model, _ = clf.model_and_grid
        assert isinstance(model, RandomForestClassifier)

    def test_second_element_is_dict(self) -> None:
        """Second element of model_and_grid is the hyperparameter grid dict."""
        clf = ClassifierModel("rf")
        _, grid = clf.model_and_grid
        assert isinstance(grid, dict)


# ---------------------------------------------------------------------------
# update_model_and_grid
# ---------------------------------------------------------------------------


class TestUpdateModelAndGrid:
    """Test update_model_and_grid() convenience method."""

    def test_updates_model(self) -> None:
        """update_model_and_grid() replaces the model."""
        clf = ClassifierModel("rf")
        custom_model = SVC(probability=True)
        clf.update_model_and_grid(custom_model, {"C": [1]})
        assert clf.model is custom_model

    def test_updates_grid(self) -> None:
        """update_model_and_grid() replaces the grid."""
        clf = ClassifierModel("rf")
        custom_grid = {"C": [0.1, 1]}
        clf.update_model_and_grid(SVC(probability=True), custom_grid)
        assert clf.grid is custom_grid

    def test_returns_self_for_chaining(self) -> None:
        """update_model_and_grid() returns the same instance."""
        clf = ClassifierModel("rf")
        result = clf.update_model_and_grid(RandomForestClassifier(), {"n_estimators": [10]})
        assert result is clf
