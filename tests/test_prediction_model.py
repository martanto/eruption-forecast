"""Fast tests for ``PredictionModel`` without tsfresh.

A synthetic features matrix and label CSV stand in for a previous
extraction, so ``load_features()`` and ``forecast()`` run in about a second
against a tiny ``LogisticRegression`` ensemble from ``make_seed_ensemble``.
The full extraction path is covered by the slow ``test_pipeline.py``.
"""

import os

import numpy as np
import pandas as pd
import pytest

from tests.conftest import FEATURE_NAMES, eruption_ramp_tremor_df
from eruption_forecast.model.prediction_model import PredictionModel
from eruption_forecast.ensemble.classifier_ensemble import ClassifierEnsemble


def _write_feature_files(tmp_path, start: str, end: str, step: str = "6h") -> dict[str, str]:
    """Write a features matrix + label CSV on a full-day grid from start to end."""
    index = pd.date_range(start, f"{end} 23:59:59", freq=step, name="datetime")
    labels = pd.DataFrame({"id": range(len(index)), "is_erupted": 0}, index=index)
    rng = np.random.default_rng(0)
    features = pd.DataFrame(
        rng.standard_normal((len(index), len(FEATURE_NAMES))),
        columns=FEATURE_NAMES,
        index=pd.RangeIndex(len(index), name="id"),
    )
    label_csv = tmp_path / "features-label.csv"
    matrix = tmp_path / "features-matrix.parquet"
    labels.to_csv(label_csv)
    features.to_parquet(matrix)
    return {"features_matrix_path": str(matrix), "label_features_csv": str(label_csv)}


@pytest.fixture()
def prediction_model(make_seed_ensemble, tmp_path) -> PredictionModel:
    ensemble = ClassifierEnsemble.from_seed_ensembles(
        {"LogisticRegression": make_seed_ensemble(n_seeds=2)}
    )
    return PredictionModel(
        model=ensemble,
        tremor_data=eruption_ramp_tremor_df(end="2025-01-31"),
        start_date="2025-01-20",
        end_date="2025-01-27",
        output_dir=str(tmp_path / "output"),
    )


class TestInit:
    def test_kind_and_ensemble(self, prediction_model: PredictionModel) -> None:
        assert prediction_model.kind == "prediction"
        assert prediction_model.ClassifierEnsemble.classifiers == ["LogisticRegression"]

    def test_accepts_seed_ensemble(self, make_seed_ensemble, tmp_path) -> None:
        pm = PredictionModel(
            model=make_seed_ensemble(n_seeds=1),
            tremor_data=eruption_ramp_tremor_df(end="2025-01-31"),
            start_date="2025-01-20",
            end_date="2025-01-27",
            output_dir=str(tmp_path),
        )
        assert isinstance(pm.ClassifierEnsemble, ClassifierEnsemble)


class TestLoadFeatures:
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "load_features slices labels with .loc[start_date:end_date] where "
            "end_date is midnight, so the last day's later windows are dropped. "
            "The fresh build_label grid runs to 23:59:59 on end_date."
        ),
    )
    def test_covers_the_whole_last_day(self, prediction_model, tmp_path) -> None:
        files = _write_feature_files(tmp_path, "2025-01-18", "2025-01-31")

        prediction_model.load_features(**files, window_step=6, window_step_unit="hours")

        # 2025-01-20 .. 2025-01-27 at 6 h = 8 days x 4 windows.
        assert len(prediction_model.features_df) == 32

    def test_slices_to_requested_range(self, prediction_model, tmp_path) -> None:
        files = _write_feature_files(tmp_path, "2025-01-18", "2025-01-31")

        prediction_model.load_features(**files, window_step=6, window_step_unit="hours")

        datetimes = prediction_model._labels.index
        assert datetimes.min() == pd.Timestamp("2025-01-20")
        assert datetimes.max().date() == pd.Timestamp("2025-01-27").date()
        assert len(prediction_model.features_df) == len(prediction_model.labels)
        assert prediction_model.window_step == 6

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "check_sampling_consistency returns is_consistent=inconsistent_data.empty, "
            "and DataFrame.empty is True for the column-less frame load_features "
            "passes, so the sampling-rate mismatch is never detected."
        ),
    )
    def test_window_step_mismatch_raises(self, prediction_model, tmp_path) -> None:
        files = _write_feature_files(tmp_path, "2025-01-18", "2025-01-31")
        with pytest.raises(ValueError, match="Sampling-rate mismatch"):
            prediction_model.load_features(**files, window_step=3, window_step_unit="hours")

    def test_range_not_covered_raises(self, prediction_model, tmp_path) -> None:
        files = _write_feature_files(tmp_path, "2025-01-22", "2025-01-31")
        with pytest.raises(ValueError, match="does not fully cover"):
            prediction_model.load_features(**files, window_step=6, window_step_unit="hours")


class TestForecast:
    def test_forecast_on_loaded_features(self, prediction_model, tmp_path) -> None:
        files = _write_feature_files(tmp_path, "2025-01-18", "2025-01-31")
        prediction_model.load_features(**files, window_step=6, window_step_unit="hours")

        results = prediction_model.forecast(plot_pdf=False)

        assert len(results) == len(prediction_model.features_df)
        for column in (
            "consensus_probability",
            "LogisticRegression_probability",
            "LogisticRegression_uncertainty",
        ):
            assert column in results.columns
        assert results["consensus_probability"].between(0, 1).all()
        assert prediction_model.results is results
        assert prediction_model.forecast_plot_path
        assert os.path.isfile(prediction_model.forecast_plot_path)
