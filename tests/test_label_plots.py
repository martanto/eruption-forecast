"""Unit tests for label distribution plotting helpers."""

import os

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


class TestPlotLabelDistribution:
    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame({"is_erupted": [0, 0, 0, 1, 1]})

    def test_output_file_exists(self, tmp_path):
        from eruption_forecast.label.label_plots import plot_label_distribution

        filepath = str(tmp_path / "distribution")
        result = plot_label_distribution(self._make_df(), filepath, verbose=False)
        assert os.path.isfile(result)
        assert result == f"{filepath}.png"

    def test_custom_filetype(self, tmp_path):
        from eruption_forecast.label.label_plots import plot_label_distribution

        filepath = str(tmp_path / "distribution")
        result = plot_label_distribution(
            self._make_df(), filepath, filetype="pdf", verbose=False
        )
        assert result.endswith(".pdf")
        assert os.path.isfile(result)

    def test_missing_label_column_raises(self, tmp_path):
        from eruption_forecast.label.label_plots import plot_label_distribution

        df = pd.DataFrame({"wrong_col": [0, 1]})
        with pytest.raises(KeyError):
            plot_label_distribution(
                df,
                str(tmp_path / "distribution"),
                label_column="is_erupted",
                verbose=False,
            )

    def test_figures_closed_after_save(self, tmp_path):
        from eruption_forecast.label.label_plots import plot_label_distribution

        before = len(plt.get_fignums())
        plot_label_distribution(
            self._make_df(), str(tmp_path / "distribution"), verbose=False
        )
        assert len(plt.get_fignums()) == before
