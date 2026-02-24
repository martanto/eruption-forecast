"""Unit tests for ClassifierComparator.

Tests cross-classifier comparison using synthetic metrics JSON files and
minimal trained-model registry CSVs generated in temporary directories.
"""

import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from eruption_forecast.model.classifier_comparator import ClassifierComparator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_METRICS_TEMPLATE: dict = {
    "accuracy": 0.80,
    "balanced_accuracy": 0.78,
    "f1_score": 0.75,
    "precision": 0.78,
    "recall": 0.72,
    "sensitivity": 0.72,
    "specificity": 0.84,
    "roc_auc": 0.85,
    "pr_auc": 0.70,
    "average_precision": 0.70,
    "true_positives": 50,
    "true_negatives": 120,
    "false_positives": 10,
    "false_negatives": 20,
    "optimal_threshold": 0.45,
    "f1_at_optimal": 0.76,
    "recall_at_optimal": 0.73,
    "precision_at_optimal": 0.79,
}


def _make_metrics_dir(base: str, name: str, n_seeds: int = 5) -> str:
    """Create a metrics directory with synthetic per-seed JSON files.

    Writes ``n_seeds`` JSON files into ``{base}/{name}/metrics/`` with
    slightly varied metric values so std is non-zero.

    Args:
        base (str): Parent temporary directory path.
        name (str): Classifier name (used as subdirectory).
        n_seeds (int, optional): Number of synthetic seed files. Defaults to 5.

    Returns:
        str: Path to the metrics directory.
    """
    metrics_dir = os.path.join(base, name, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    rng = np.random.default_rng(42 + hash(name) % 100)
    for seed in range(n_seeds):
        record = {k: float(v + rng.uniform(-0.05, 0.05)) for k, v in _METRICS_TEMPLATE.items()}
        record["model_name"] = name
        path = os.path.join(metrics_dir, f"seed_{seed:04d}.json")
        with open(path, "w") as f:
            json.dump(record, f)
    return metrics_dir


def _make_registry_csv(base: str, name: str, n_seeds: int = 5) -> str:
    """Create a minimal trained-model registry CSV alongside a metrics directory.

    Writes a CSV with the columns expected by ``MultiModelEvaluator`` and
    returns its path. Companion metrics JSON files are written by
    ``_make_metrics_dir``.

    Args:
        base (str): Parent temporary directory path.
        name (str): Classifier name.
        n_seeds (int, optional): Number of seed rows. Defaults to 5.

    Returns:
        str: Path to the registry CSV file.
    """
    _make_metrics_dir(base, name, n_seeds)
    classifier_dir = os.path.join(base, name)

    rows = []
    for seed in range(n_seeds):
        rows.append(
            {
                "random_state": seed,
                "significant_features_csv": os.path.join(classifier_dir, f"sig_{seed}.csv"),
                "trained_model_filepath": os.path.join(classifier_dir, f"model_{seed}.pkl"),
                "X_test_filepath": os.path.join(classifier_dir, f"X_{seed}.csv"),
                "y_test_filepath": os.path.join(classifier_dir, f"y_{seed}.csv"),
            }
        )

    df = pd.DataFrame(rows).set_index("random_state")
    csv_path = os.path.join(classifier_dir, f"trained_model_{name}.csv")
    df.to_csv(csv_path)
    return csv_path


def _make_comparator(tmp_path, names=("rf", "xgb"), **kwargs) -> ClassifierComparator:
    """Build a ClassifierComparator backed entirely by synthetic data.

    Creates registry CSVs and metrics JSON files for each classifier in
    ``names`` under ``tmp_path``, then returns a ``ClassifierComparator``
    pointing to those files.

    Args:
        tmp_path (pathlib.Path): Pytest ``tmp_path`` fixture value.
        names (tuple[str, ...], optional): Classifier names. Defaults to
            ``("rf", "xgb")``.
        **kwargs: Extra keyword arguments forwarded to ``ClassifierComparator``.

    Returns:
        ClassifierComparator: Initialised comparator with synthetic data.
    """
    base = str(tmp_path)
    classifiers = {n: _make_registry_csv(base, n) for n in names}
    kwargs.setdefault("output_dir", str(tmp_path / "comparison"))
    return ClassifierComparator(classifiers=classifiers, **kwargs)


# ---------------------------------------------------------------------------
# TestClassifierComparatorInit
# ---------------------------------------------------------------------------

class TestClassifierComparatorInit:
    """Tests for ClassifierComparator.__init__."""

    def test_valid_input_sets_attributes(self, tmp_path):
        """Valid classifier dict sets output_dir, figures_dir, and metrics_dir."""
        comparator = _make_comparator(tmp_path)
        assert os.path.isabs(comparator.output_dir)
        assert comparator.figures_dir.endswith("figures")
        assert comparator.metrics_dir.endswith("metrics")
        assert set(comparator._evaluators.keys()) == {"rf", "xgb"}

    def test_single_metric_string_normalised_to_list(self, tmp_path):
        """A single string metric is stored as a one-element list."""
        comparator = _make_comparator(tmp_path, metrics="roc_auc")
        assert comparator.metrics == ["roc_auc"]

    def test_metrics_list_preserved(self, tmp_path):
        """A list of metrics is stored as-is."""
        comparator = _make_comparator(tmp_path, metrics=["f1_score", "roc_auc"])
        assert comparator.metrics == ["f1_score", "roc_auc"]

    def test_explicit_output_dir_appends_comparison(self, tmp_path):
        """An explicit output_dir always has 'comparison' appended."""
        custom = str(tmp_path / "custom_out")
        comparator = _make_comparator(tmp_path, output_dir=custom)
        assert comparator.output_dir == os.path.join(custom, "comparison")

    def test_from_json_loads_classifiers(self, tmp_path):
        """from_json constructs a comparator identical to the direct dict call."""
        base = str(tmp_path)
        csv_rf = _make_registry_csv(base, "rf")
        csv_xgb = _make_registry_csv(base, "xgb")
        json_path = str(tmp_path / "trained_models.json")
        with open(json_path, "w") as f:
            json.dump({"rf": csv_rf, "xgb": csv_xgb}, f)

        comparator = ClassifierComparator.from_json(
            json_path, output_dir=str(tmp_path / "out")
        )
        assert set(comparator._evaluators.keys()) == {"rf", "xgb"}

    def test_from_json_missing_file_raises(self, tmp_path):
        """from_json raises FileNotFoundError when the JSON path does not exist."""
        with pytest.raises(FileNotFoundError, match="not found"):
            ClassifierComparator.from_json(str(tmp_path / "missing.json"))

    def test_from_json_invalid_content_raises(self, tmp_path):
        """from_json raises ValueError when the JSON is not a non-empty object."""
        json_path = str(tmp_path / "bad.json")
        with open(json_path, "w") as f:
            json.dump([], f)
        with pytest.raises(ValueError, match="non-empty object"):
            ClassifierComparator.from_json(json_path)

    def test_missing_csv_raises_file_not_found(self, tmp_path):
        """FileNotFoundError raised when a CSV path does not exist."""
        with pytest.raises(FileNotFoundError, match="not found"):
            ClassifierComparator(
                classifiers={"rf": str(tmp_path / "nonexistent.csv")},
                output_dir=str(tmp_path),
            )

    def test_empty_dict_raises_value_error(self, tmp_path):
        """ValueError raised when an empty classifiers dict is supplied."""
        with pytest.raises(ValueError, match="empty"):
            ClassifierComparator(classifiers={}, output_dir=str(tmp_path))

    def test_default_output_dir(self, tmp_path, monkeypatch):
        """Default output_dir is cwd/output/comparison when not specified."""
        monkeypatch.chdir(tmp_path)
        base = str(tmp_path)
        csv_path = _make_registry_csv(base, "rf")
        comparator = ClassifierComparator(classifiers={"rf": csv_path})
        assert comparator.output_dir == os.path.join(str(tmp_path), "output", "comparison")


# ---------------------------------------------------------------------------
# TestGetMetricsTable
# ---------------------------------------------------------------------------

class TestGetMetricsTable:
    """Tests for ClassifierComparator.get_metrics_table."""

    def test_returns_dataframe_with_classifier_rows(self, tmp_path):
        """get_metrics_table returns a DataFrame with one row per classifier."""
        comparator = _make_comparator(tmp_path, names=("rf", "xgb", "gb"))
        table = comparator.get_metrics_table()
        assert isinstance(table, pd.DataFrame)
        assert set(table.index) == {"rf", "xgb", "gb"}

    def test_columns_contain_metric_stat_pairs(self, tmp_path):
        """Column names follow the {metric}_{stat} pattern."""
        comparator = _make_comparator(tmp_path)
        table = comparator.get_metrics_table()
        assert "f1_score_mean" in table.columns
        assert "f1_score_std" in table.columns

    def test_mean_values_in_expected_range(self, tmp_path):
        """Mean f1_score values are within the plausible synthetic range."""
        comparator = _make_comparator(tmp_path)
        table = comparator.get_metrics_table()
        for clf in table.index:
            assert 0.0 <= table.loc[clf, "f1_score_mean"] <= 1.0


# ---------------------------------------------------------------------------
# TestGetRanking
# ---------------------------------------------------------------------------

class TestGetRanking:
    """Tests for ClassifierComparator.get_ranking."""

    def test_returns_ranked_dataframe(self, tmp_path):
        """get_ranking returns a DataFrame with a rank column."""
        comparator = _make_comparator(tmp_path, names=("rf", "xgb", "gb"))
        ranked = comparator.get_ranking()
        assert isinstance(ranked, pd.DataFrame)
        assert "rank" in ranked.columns
        assert list(ranked["rank"]) == [1, 2, 3]

    def test_default_metric_is_recall(self, tmp_path):
        """Default ranking metric is recall."""
        comparator = _make_comparator(tmp_path, names=("rf", "xgb"))
        ranked = comparator.get_ranking()
        means = ranked["recall_mean"].values
        assert means[0] >= means[-1]

    def test_custom_metric_ranking(self, tmp_path):
        """Custom metric is used when supplied."""
        comparator = _make_comparator(tmp_path, names=("rf", "xgb"))
        ranked = comparator.get_ranking(metric="roc_auc", by="mean")
        means = ranked["roc_auc_mean"].values
        assert means[0] >= means[-1]

    def test_csv_saved_to_metrics_dir(self, tmp_path):
        """Ranking CSV is written to {output_dir}/metrics/ranking_{metric}.csv."""
        comparator = _make_comparator(tmp_path)
        comparator.get_ranking()
        assert os.path.isfile(os.path.join(comparator.metrics_dir, "ranking_recall.csv"))

    def test_invalid_metric_raises_key_error(self, tmp_path):
        """KeyError raised when the requested metric column is absent."""
        comparator = _make_comparator(tmp_path)
        with pytest.raises(KeyError):
            comparator.get_ranking(metric="nonexistent_metric")


# ---------------------------------------------------------------------------
# TestPlotMetricBar
# ---------------------------------------------------------------------------

class TestPlotMetricBar:
    """Tests for ClassifierComparator.plot_metric_bar."""

    def test_returns_dict_of_figures(self, tmp_path):
        """plot_metric_bar returns a dict mapping metric name to Figure."""
        comparator = _make_comparator(tmp_path)
        figs = comparator.plot_metric_bar(save=False)
        assert isinstance(figs, dict)
        for fig in figs.values():
            assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_saves_one_file_per_metric(self, tmp_path):
        """A separate PNG is saved for each metric."""
        comparator = _make_comparator(tmp_path)
        comparator.plot_metric_bar(metrics=["f1_score", "roc_auc"], save=True)
        assert os.path.isfile(os.path.join(comparator.figures_dir, "metric_bar_f1_score.png"))
        assert os.path.isfile(os.path.join(comparator.figures_dir, "metric_bar_roc_auc.png"))
        plt.close("all")

    def test_single_metric_returns_one_entry(self, tmp_path):
        """A single metric string produces a dict with exactly one entry."""
        comparator = _make_comparator(tmp_path)
        figs = comparator.plot_metric_bar(metrics="f1_score", save=False)
        assert list(figs.keys()) == ["f1_score"]
        plt.close("all")

    def test_custom_filename_applied_to_first_metric(self, tmp_path):
        """Custom filename is used for the first metric only."""
        comparator = _make_comparator(tmp_path)
        comparator.plot_metric_bar(
            metrics=["f1_score", "roc_auc"], save=True, filename="custom_bar.png"
        )
        assert os.path.isfile(os.path.join(comparator.figures_dir, "custom_bar.png"))
        assert os.path.isfile(os.path.join(comparator.figures_dir, "metric_bar_roc_auc.png"))
        plt.close("all")

    def test_multiple_metrics_includes_all_key(self, tmp_path):
        """Multiple metrics produces an 'all' overview figure in the dict."""
        comparator = _make_comparator(tmp_path)
        figs = comparator.plot_metric_bar(metrics=["f1_score", "roc_auc"], save=False)
        assert "all" in figs
        assert isinstance(figs["all"], plt.Figure)
        plt.close("all")

    def test_all_overview_saved_to_figures_dir(self, tmp_path):
        """Combined overview figure is saved as metric_bar_all.png."""
        comparator = _make_comparator(tmp_path)
        comparator.plot_metric_bar(metrics=["f1_score", "roc_auc"], save=True)
        assert os.path.isfile(os.path.join(comparator.figures_dir, "metric_bar_all.png"))
        plt.close("all")

    def test_single_metric_no_all_key(self, tmp_path):
        """A single metric does not produce an 'all' overview figure."""
        comparator = _make_comparator(tmp_path)
        figs = comparator.plot_metric_bar(metrics="f1_score", save=False)
        assert "all" not in figs
        plt.close("all")


# ---------------------------------------------------------------------------
# TestPlotSeedStability
# ---------------------------------------------------------------------------

class TestPlotSeedStability:
    """Tests for ClassifierComparator.plot_seed_stability."""

    def test_returns_dict_of_figures(self, tmp_path):
        """plot_seed_stability returns a dict mapping metric name to Figure."""
        comparator = _make_comparator(tmp_path)
        figs = comparator.plot_seed_stability(save=False)
        assert isinstance(figs, dict)
        for fig in figs.values():
            assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_saves_one_file_per_metric(self, tmp_path):
        """A separate PNG is saved for each metric."""
        comparator = _make_comparator(tmp_path)
        comparator.plot_seed_stability(metrics=["f1_score", "roc_auc"], save=True)
        assert os.path.isfile(os.path.join(comparator.figures_dir, "seed_stability_f1_score.png"))
        assert os.path.isfile(os.path.join(comparator.figures_dir, "seed_stability_roc_auc.png"))
        plt.close("all")

    def test_single_metric_returns_one_entry(self, tmp_path):
        """A single metric string produces a dict with exactly one entry."""
        comparator = _make_comparator(tmp_path)
        figs = comparator.plot_seed_stability(metrics="roc_auc", save=False)
        assert list(figs.keys()) == ["roc_auc"]
        plt.close("all")

    def test_custom_filename_applied_to_first_metric(self, tmp_path):
        """Custom filename is used for the first metric only."""
        comparator = _make_comparator(tmp_path)
        comparator.plot_seed_stability(
            metrics=["f1_score", "roc_auc"], save=True, filename="custom_stability.png"
        )
        assert os.path.isfile(os.path.join(comparator.figures_dir, "custom_stability.png"))
        assert os.path.isfile(os.path.join(comparator.figures_dir, "seed_stability_roc_auc.png"))
        plt.close("all")

    def test_multiple_metrics_includes_all_key(self, tmp_path):
        """Multiple metrics produces an 'all' overview figure in the dict."""
        comparator = _make_comparator(tmp_path)
        figs = comparator.plot_seed_stability(metrics=["f1_score", "roc_auc"], save=False)
        assert "all" in figs
        assert isinstance(figs["all"], plt.Figure)
        plt.close("all")

    def test_all_overview_saved_to_figures_dir(self, tmp_path):
        """Combined overview figure is saved as seed_stability_all.png."""
        comparator = _make_comparator(tmp_path)
        comparator.plot_seed_stability(metrics=["f1_score", "roc_auc"], save=True)
        assert os.path.isfile(os.path.join(comparator.figures_dir, "seed_stability_all.png"))
        plt.close("all")

    def test_single_metric_no_all_key(self, tmp_path):
        """A single metric does not produce an 'all' overview figure."""
        comparator = _make_comparator(tmp_path)
        figs = comparator.plot_seed_stability(metrics="roc_auc", save=False)
        assert "all" not in figs
        plt.close("all")


# ---------------------------------------------------------------------------
# TestPlotComparisonGrid
# ---------------------------------------------------------------------------

class TestPlotComparisonGrid:
    """Tests for ClassifierComparator.plot_comparison_grid."""

    def test_returns_figure(self, tmp_path):
        """plot_comparison_grid returns a matplotlib Figure."""
        comparator = _make_comparator(tmp_path)
        fig = comparator.plot_comparison_grid(save=False)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_saves_to_figures_dir(self, tmp_path):
        """Figure saved to {output_dir}/figures/comparison_grid.png."""
        comparator = _make_comparator(tmp_path)
        comparator.plot_comparison_grid(save=True)
        expected = os.path.join(comparator.figures_dir, "comparison_grid.png")
        assert os.path.isfile(expected)
        plt.close("all")

    def test_grid_shape(self, tmp_path):
        """Figure has n_classifiers rows and n_metrics columns of axes."""
        comparator = _make_comparator(tmp_path, names=("rf", "xgb"))
        metrics = ["f1_score", "roc_auc", "precision"]
        fig = comparator.plot_comparison_grid(metrics=metrics, save=False)
        # 2 classifiers × 3 metrics = 6 axes
        assert len(fig.axes) == 2 * 3
        plt.close("all")

    def test_single_metric(self, tmp_path):
        """plot_comparison_grid works with a single metric string."""
        comparator = _make_comparator(tmp_path)
        fig = comparator.plot_comparison_grid(metrics="f1_score", save=False)
        assert isinstance(fig, plt.Figure)
        plt.close("all")


# ---------------------------------------------------------------------------
# TestPlotRoc
# ---------------------------------------------------------------------------

class TestPlotRoc:
    """Tests for ClassifierComparator.plot_roc."""

    def test_returns_figure_without_registry(self, tmp_path):
        """plot_roc raises ValueError when registry CSV rows lack model files.

        The registry CSV contains paths to non-existent model/test files so
        each seed load will fail gracefully; the figure is still returned with
        only the diagonal chance line drawn.
        """
        comparator = _make_comparator(tmp_path)
        # plot_roc catches per-seed exceptions and skips them, so the figure
        # should still be returned (empty of curves beyond the diagonal).
        fig = comparator.plot_roc(save=False)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_saves_to_figures_dir(self, tmp_path):
        """Figure saved to {output_dir}/figures/comparison_roc.png."""
        comparator = _make_comparator(tmp_path)
        comparator.plot_roc(save=True)
        expected = os.path.join(comparator.figures_dir, "comparison_roc.png")
        assert os.path.isfile(expected)
        plt.close("all")


# ---------------------------------------------------------------------------
# TestPlotAll
# ---------------------------------------------------------------------------

class TestPlotAll:
    """Tests for ClassifierComparator.plot_all."""

    def test_returns_dict_with_expected_keys(self, tmp_path):
        """plot_all returns a dict with all expected keys including comparison_grid."""
        comparator = _make_comparator(tmp_path)
        results = comparator.plot_all()
        assert set(results.keys()) == {
            "metric_bar", "seed_stability", "comparison_grid", "roc", "ranking"
        }
        plt.close("all")

    def test_figure_values_are_correct_types(self, tmp_path):
        """plot_all dict values have correct types per key."""
        comparator = _make_comparator(tmp_path)
        results = comparator.plot_all()
        # metric_bar returns a dict of figures (one per metric)
        assert isinstance(results["metric_bar"], dict)
        for fig in results["metric_bar"].values():
            assert isinstance(fig, plt.Figure)
        # seed_stability returns a dict of figures (one per metric)
        assert isinstance(results["seed_stability"], dict)
        for fig in results["seed_stability"].values():
            assert isinstance(fig, plt.Figure)
        # remaining plot keys return a single Figure
        for key in ("comparison_grid", "roc"):
            assert isinstance(results[key], plt.Figure), f"Expected Figure for '{key}'"
        plt.close("all")

    def test_ranking_value_is_dataframe(self, tmp_path):
        """plot_all dict value for ranking is a DataFrame with rank column."""
        comparator = _make_comparator(tmp_path)
        results = comparator.plot_all()
        assert isinstance(results["ranking"], pd.DataFrame)
        assert "rank" in results["ranking"].columns
        plt.close("all")
