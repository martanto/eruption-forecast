"""Unit tests for label distribution plotting helpers."""

import os

import pandas as pd
import pytest
import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


class TestPlotLabelDistribution:
    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame({"is_erupted": [0, 0, 0, 1, 1]})

    def test_output_file_exists(self, tmp_path):
        from eruption_forecast.plots.label_plots import plot_label_distribution

        filepath = str(tmp_path / "distribution")
        result = plot_label_distribution(self._make_df(), filepath, verbose=False)
        assert os.path.isfile(result)
        assert result == f"{filepath}.png"

    def test_custom_filetype(self, tmp_path):
        from eruption_forecast.plots.label_plots import plot_label_distribution

        filepath = str(tmp_path / "distribution")
        result = plot_label_distribution(
            self._make_df(), filepath, filetype="pdf", verbose=False
        )
        assert result.endswith(".pdf")
        assert os.path.isfile(result)

    def test_missing_label_column_raises(self, tmp_path):
        from eruption_forecast.plots.label_plots import plot_label_distribution

        df = pd.DataFrame({"wrong_col": [0, 1]})
        with pytest.raises(KeyError):
            plot_label_distribution(
                df,
                str(tmp_path / "distribution"),
                label_column="is_erupted",
                verbose=False,
            )

    def test_figures_closed_after_save(self, tmp_path):
        from eruption_forecast.plots.label_plots import plot_label_distribution

        before = len(plt.get_fignums())
        plot_label_distribution(
            self._make_df(), str(tmp_path / "distribution"), verbose=False
        )
        assert len(plt.get_fignums()) == before


class TestPlotLabelDistributionComparison:
    def _make_entries(self) -> list[dict]:
        return [
            {"name": "Scenario A", "df": pd.DataFrame({"is_erupted": [0, 0, 1]})},
            {"name": "Scenario B", "df": pd.DataFrame({"is_erupted": [0, 1, 1]})},
        ]

    def test_x_label_rotation_applied(self, tmp_path, monkeypatch):
        from eruption_forecast.plots import label_plots

        # save_figure closes the figure via plt.close; suppress that so the axes
        # are still inspectable after the call.
        monkeypatch.setattr(plt, "close", lambda *args, **kwargs: None)

        filepath = str(tmp_path / "comparison")
        try:
            result = label_plots.plot_label_distribution_comparison(
                self._make_entries(),
                filepath,
                x_label_rotation=45.0,
                verbose=False,
            )
            assert os.path.isfile(result)

            ax = plt.gcf().axes[0]
            rotations = [tick.get_rotation() for tick in ax.get_xticklabels()]
            assert rotations, "expected at least one X-axis tick label"
            assert all(r == 45.0 for r in rotations)
        finally:
            monkeypatch.undo()
            plt.close("all")
