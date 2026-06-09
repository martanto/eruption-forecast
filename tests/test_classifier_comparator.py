"""Unit tests for ClassifierComparator.

Exercises cross-classifier comparison using a lightweight stub that mimics
the ``MetricsEnsemble`` attribute surface the comparator depends on
(``metrics``, ``y_probas``, ``y_true``, ``output_dir``).
"""

import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import matplotlib
import matplotlib.pyplot as plt


matplotlib.use("Agg")

from eruption_forecast.model.classifier_comparator import ClassifierComparator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_METRIC_COLUMNS: list[str] = [
    "accuracy",
    "balanced_accuracy",
    "f1_score",
    "precision",
    "recall",
    "specificity",
    "roc_auc",
    "pr_auc",
    "g_mean",
    "mcc",
]


def _make_metrics_df(name: str, n_seeds: int = 5) -> pd.DataFrame:
    """Build a per-seed metrics DataFrame with slight per-seed variation."""
    rng = np.random.default_rng(42 + hash(name) % 100)
    rows = []
    for seed in range(n_seeds):
        row = {col: float(0.75 + rng.uniform(-0.05, 0.05)) for col in _METRIC_COLUMNS}
        row["random_state"] = seed
        rows.append(row)
    return pd.DataFrame(rows).set_index("random_state")


def _make_y_probas(n_samples: int = 60, n_seeds: int = 5, seed: int = 0) -> np.ndarray:
    """Build a synthetic ``(n_samples, n_seeds)`` probability matrix."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=(n_samples, n_seeds))


def _make_stub_metrics_ensemble(
    tmp_path,
    names: tuple[str, ...] = ("rf", "xgb"),
    n_seeds: int = 5,
    n_samples: int = 60,
) -> SimpleNamespace:
    """Build a stub with the attributes ``ClassifierComparator`` reads."""
    metrics = {name: _make_metrics_df(name, n_seeds) for name in names}
    y_probas = {
        name: _make_y_probas(n_samples, n_seeds, seed=hash(name) % 1000)
        for name in names
    }
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=n_samples)
    return SimpleNamespace(
        metrics=metrics,
        y_probas=y_probas,
        y_true=y_true,
        output_dir=str(tmp_path / "eval"),
    )


def _make_comparator(tmp_path, names=("rf", "xgb"), **kwargs) -> ClassifierComparator:
    """Build a ``ClassifierComparator`` against a stub metrics ensemble."""
    stub = _make_stub_metrics_ensemble(tmp_path, names=names)
    kwargs.setdefault("output_dir", str(tmp_path / "out"))
    return ClassifierComparator(metrics_ensemble=stub, **kwargs)


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestClassifierComparatorInit:
    """Tests for ``ClassifierComparator.__init__``."""

    def test_valid_input_sets_attributes(self, tmp_path):
        comparator = _make_comparator(tmp_path)
        assert comparator.output_dir.endswith("comparison")
        assert comparator.figures_dir.endswith("figures")
        assert comparator.metrics_dir.endswith("metrics")
        assert set(comparator._classifier_names) == {"rf", "xgb"}

    def test_single_metric_string_normalised_to_list(self, tmp_path):
        comparator = _make_comparator(tmp_path, metrics="roc_auc")
        assert comparator.metrics == ["roc_auc"]

    def test_metrics_list_preserved(self, tmp_path):
        comparator = _make_comparator(tmp_path, metrics=["f1_score", "roc_auc"])
        assert comparator.metrics == ["f1_score", "roc_auc"]

    def test_explicit_output_dir_appends_comparison(self, tmp_path):
        custom = str(tmp_path / "custom_out")
        comparator = _make_comparator(tmp_path, output_dir=custom)
        assert comparator.output_dir == os.path.join(custom, "comparison")

    def test_falls_back_to_metrics_ensemble_output_dir(self, tmp_path):
        stub = _make_stub_metrics_ensemble(tmp_path)
        comparator = ClassifierComparator(metrics_ensemble=stub)
        assert comparator.output_dir == os.path.join(stub.output_dir, "comparison")

    def test_empty_metrics_raises_value_error(self, tmp_path):
        stub = SimpleNamespace(
            metrics={},
            y_probas={},
            y_true=np.array([]),
            output_dir=str(tmp_path),
        )
        with pytest.raises(ValueError, match="compute"):
            ClassifierComparator(metrics_ensemble=stub)


# ---------------------------------------------------------------------------
# get_metrics_table
# ---------------------------------------------------------------------------


class TestGetMetricsTable:
    """Tests for ``ClassifierComparator.get_metrics_table``."""

    def test_returns_dataframe_with_classifier_rows(self, tmp_path):
        comparator = _make_comparator(tmp_path, names=("rf", "xgb", "gb"))
        table = comparator.get_metrics_table()
        assert isinstance(table, pd.DataFrame)
        assert set(table.index) == {"rf", "xgb", "gb"}

    def test_columns_contain_metric_stat_pairs(self, tmp_path):
        comparator = _make_comparator(tmp_path)
        table = comparator.get_metrics_table()
        assert "f1_score_mean" in table.columns
        assert "f1_score_std" in table.columns

    def test_mean_values_in_expected_range(self, tmp_path):
        comparator = _make_comparator(tmp_path)
        table = comparator.get_metrics_table()
        for clf in table.index:
            assert 0.0 <= table.loc[clf, "f1_score_mean"] <= 1.0


# ---------------------------------------------------------------------------
# get_ranking
# ---------------------------------------------------------------------------


class TestGetRanking:
    """Tests for ``ClassifierComparator.get_ranking``."""

    def test_returns_ranked_dataframe(self, tmp_path):
        comparator = _make_comparator(tmp_path, names=("rf", "xgb", "gb"))
        ranked = comparator.get_ranking()
        assert isinstance(ranked, pd.DataFrame)
        assert "rank" in ranked.columns
        assert list(ranked["rank"]) == [1, 2, 3]

    def test_default_metric_is_recall(self, tmp_path):
        comparator = _make_comparator(tmp_path, names=("rf", "xgb"))
        ranked = comparator.get_ranking()
        means = ranked["recall_mean"].values
        assert means[0] >= means[-1]

    def test_custom_metric_ranking(self, tmp_path):
        comparator = _make_comparator(tmp_path, names=("rf", "xgb"))
        ranked = comparator.get_ranking(metric="roc_auc", by="mean")
        means = ranked["roc_auc_mean"].values
        assert means[0] >= means[-1]

    def test_csv_saved_to_metrics_dir(self, tmp_path):
        comparator = _make_comparator(tmp_path)
        comparator.get_ranking()
        assert os.path.isfile(os.path.join(comparator.metrics_dir, "ranking_recall.csv"))

    def test_invalid_metric_raises_key_error(self, tmp_path):
        comparator = _make_comparator(tmp_path)
        with pytest.raises(KeyError):
            comparator.get_ranking(metric="nonexistent_metric")


# ---------------------------------------------------------------------------
# plot_metric_bar
# ---------------------------------------------------------------------------


class TestPlotMetricBar:
    """Tests for ``ClassifierComparator.plot_metric_bar``."""

    def test_returns_dict_of_figures(self, tmp_path):
        comparator = _make_comparator(tmp_path)
        figs = comparator.plot_metric_bar(save=False)
        assert isinstance(figs, dict)
        for fig in figs.values():
            assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_saves_one_file_per_metric(self, tmp_path):
        comparator = _make_comparator(tmp_path)
        comparator.plot_metric_bar(metrics=["f1_score", "roc_auc"], save=True)
        assert os.path.isfile(os.path.join(comparator.figures_dir, "metric_bar_f1_score.png"))
        assert os.path.isfile(os.path.join(comparator.figures_dir, "metric_bar_roc_auc.png"))
        plt.close("all")

    def test_single_metric_returns_one_entry(self, tmp_path):
        comparator = _make_comparator(tmp_path)
        figs = comparator.plot_metric_bar(metrics="f1_score", save=False)
        assert list(figs.keys()) == ["f1_score"]
        plt.close("all")

    def test_multiple_metrics_includes_all_key(self, tmp_path):
        comparator = _make_comparator(tmp_path)
        figs = comparator.plot_metric_bar(metrics=["f1_score", "roc_auc"], save=False)
        assert "all" in figs
        assert isinstance(figs["all"], plt.Figure)
        plt.close("all")

    def test_all_overview_saved_to_figures_dir(self, tmp_path):
        comparator = _make_comparator(tmp_path)
        comparator.plot_metric_bar(metrics=["f1_score", "roc_auc"], save=True)
        assert os.path.isfile(os.path.join(comparator.figures_dir, "metric_bar_all.png"))
        plt.close("all")


# ---------------------------------------------------------------------------
# plot_seed_stability
# ---------------------------------------------------------------------------


class TestPlotSeedStability:
    """Tests for ``ClassifierComparator.plot_seed_stability``."""

    def test_returns_dict_of_figures(self, tmp_path):
        comparator = _make_comparator(tmp_path)
        figs = comparator.plot_seed_stability(save=False)
        assert isinstance(figs, dict)
        for fig in figs.values():
            assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_saves_one_file_per_metric(self, tmp_path):
        comparator = _make_comparator(tmp_path)
        comparator.plot_seed_stability(metrics=["f1_score", "roc_auc"], save=True)
        assert os.path.isfile(
            os.path.join(comparator.figures_dir, "seed_stability_f1_score.png")
        )
        assert os.path.isfile(
            os.path.join(comparator.figures_dir, "seed_stability_roc_auc.png")
        )
        plt.close("all")

    def test_multiple_metrics_includes_all_key(self, tmp_path):
        comparator = _make_comparator(tmp_path)
        figs = comparator.plot_seed_stability(metrics=["f1_score", "roc_auc"], save=False)
        assert "all" in figs
        assert isinstance(figs["all"], plt.Figure)
        plt.close("all")


# ---------------------------------------------------------------------------
# plot_comparison_grid
# ---------------------------------------------------------------------------


class TestPlotComparisonGrid:
    """Tests for ``ClassifierComparator.plot_comparison_grid``."""

    def test_returns_figure(self, tmp_path):
        comparator = _make_comparator(tmp_path)
        fig = comparator.plot_comparison_grid(save=False)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_saves_to_figures_dir(self, tmp_path):
        comparator = _make_comparator(tmp_path)
        comparator.plot_comparison_grid(save=True)
        expected = os.path.join(comparator.figures_dir, "comparison_grid.png")
        assert os.path.isfile(expected)
        plt.close("all")

    def test_grid_shape(self, tmp_path):
        comparator = _make_comparator(tmp_path, names=("rf", "xgb"))
        metrics = ["f1_score", "roc_auc", "precision"]
        fig = comparator.plot_comparison_grid(metrics=metrics, save=False)
        assert len(fig.axes) == 2 * 3
        plt.close("all")


# ---------------------------------------------------------------------------
# plot_roc
# ---------------------------------------------------------------------------


class TestPlotRoc:
    """Tests for ``ClassifierComparator.plot_roc``."""

    def test_returns_figure(self, tmp_path):
        comparator = _make_comparator(tmp_path)
        fig = comparator.plot_roc(save=False)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_saves_to_figures_dir(self, tmp_path):
        comparator = _make_comparator(tmp_path)
        comparator.plot_roc(save=True)
        expected = os.path.join(comparator.figures_dir, "comparison_roc.png")
        assert os.path.isfile(expected)
        plt.close("all")


# ---------------------------------------------------------------------------
# plot_all
# ---------------------------------------------------------------------------


class TestPlotAll:
    """Tests for ``ClassifierComparator.plot_all``."""

    def test_returns_dict_with_expected_keys(self, tmp_path):
        comparator = _make_comparator(tmp_path)
        results = comparator.plot_all(parallel=False)
        assert set(results.keys()) == {
            "metric_bar",
            "seed_stability",
            "comparison_grid",
            "roc",
            "pr",
            "ranking",
        }
        plt.close("all")

    def test_ranking_value_is_dataframe(self, tmp_path):
        comparator = _make_comparator(tmp_path)
        results = comparator.plot_all(parallel=False)
        assert isinstance(results["ranking"], pd.DataFrame)
        assert "rank" in results["ranking"].columns
        plt.close("all")
