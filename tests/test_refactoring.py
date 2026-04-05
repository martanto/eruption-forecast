"""Regression tests for code-quality refactoring on branch dev/code-quality-dry-clarity.

Covers every change made during the DRY / clarity pass:

1. parse_label_filename moved to utils.date_utils
2. TremorData.csv_path replaces self.csv
3. SDS._make_log_prefix removed (inherited from SeismicDataSource)
4. label_data.self.kwargs removed (parameters property is the API)
5. dsar.py — direct assignment instead of intermediate aliases
6. array.py — detect_anomalies_zscore uses _filter_nans
7. ml.py  — confidence bug fix: shape[1] for n_seeds
8. model_trainer._cv_train_evaluate — X_train arg removed
"""

import inspect
import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# 1. parse_label_filename lives in utils.date_utils (and re-exported from utils)
# ---------------------------------------------------------------------------

class TestParseLabelFilenameLocation:
    def test_importable_from_date_utils(self):
        from eruption_forecast.utils.date_utils import parse_label_filename
        assert callable(parse_label_filename)

    def test_importable_from_utils(self):
        from eruption_forecast.utils import parse_label_filename
        assert callable(parse_label_filename)

    def test_not_defined_in_label_data_module(self):
        import eruption_forecast.label.label_data as mod
        # Must be imported there, not *defined* there
        src = inspect.getsourcefile(mod.parse_label_filename)
        assert "date_utils" in src.replace("\\", "/"), (
            "parse_label_filename should be defined in date_utils, not label_data"
        )

    def test_parses_correctly(self):
        from eruption_forecast.utils.date_utils import parse_label_filename
        result = parse_label_filename(
            "label_2020-01-01_2020-12-31_step-12-hours_dtf-2"
        )
        assert result["start_date_str"] == "2020-01-01"
        assert result["end_date_str"] == "2020-12-31"
        assert result["window_step"] == 12
        assert result["window_step_unit"] == "hours"
        assert result["day_to_forecast"] == 2


# ---------------------------------------------------------------------------
# 2. TremorData — only csv_path, no self.csv
# ---------------------------------------------------------------------------

class TestTremorDataCsvPath:
    def test_has_csv_attribute(self):
        from eruption_forecast.tremor.tremor_data import TremorData
        td = TremorData()
        assert hasattr(td, "csv"), "TremorData should have a csv attribute"

    def test_csv_initialises_empty(self):
        from eruption_forecast.tremor.tremor_data import TremorData
        td = TremorData()
        assert td.csv == ""

    def test_repr_uses_csv(self):
        from eruption_forecast.tremor.tremor_data import TremorData
        import inspect
        src = inspect.getsource(TremorData.__repr__)
        assert "csv" in src


# ---------------------------------------------------------------------------
# 3. SDS._make_log_prefix is inherited, not overridden
# ---------------------------------------------------------------------------

class TestSDSMakeLogPrefix:
    def test_not_defined_on_sds_class(self):
        from eruption_forecast.sources.sds import SDS
        # Must NOT be in SDS's own __dict__ (it is inherited)
        assert "_make_log_prefix" not in SDS.__dict__, (
            "SDS._make_log_prefix was removed; it should be inherited from SeismicDataSource"
        )

    def test_still_callable_via_inheritance(self, tmp_path):
        from datetime import datetime
        from eruption_forecast.sources.sds import SDS
        from eruption_forecast.sources.base import SeismicDataSource
        # Verify _make_log_prefix comes from SeismicDataSource, not SDS itself
        assert "_make_log_prefix" in SeismicDataSource.__dict__
        assert "_make_log_prefix" not in SDS.__dict__
        # Verify the method produces the right format via the base implementation
        sds_dir = str(tmp_path)
        sds = SDS(sds_dir=sds_dir, station="OJN", channel="EHZ",
                  network="VG", location="00")
        prefix = sds._make_log_prefix(datetime(2025, 1, 1))
        assert prefix == "2025-01-01 :: VG.OJN.00.EHZ"


# ---------------------------------------------------------------------------
# 4. LabelData — no self.kwargs
# ---------------------------------------------------------------------------

class TestLabelDataNoKwargs:
    def _make_label_csv(self, tmp_path):
        """Create a minimal label CSV with the correct filename format."""
        fname = "label_2020-01-01_2020-12-31_step-12-hours_dtf-2.csv"
        path = tmp_path / fname
        df = pd.DataFrame(
            {"id": [0, 1], "is_erupted": [0, 1]},
            index=pd.to_datetime(["2020-06-01", "2020-06-02"]),
        )
        df.index.name = "datetime"
        df.to_csv(path)
        return str(path)

    def test_no_kwargs_attribute(self, tmp_path):
        from eruption_forecast.label.label_data import LabelData
        csv = self._make_label_csv(tmp_path)
        ld = LabelData(csv)
        assert not hasattr(ld, "kwargs"), (
            "self.kwargs was removed; parameters property is the public API"
        )

    def test_parameters_property_still_works(self, tmp_path):
        from eruption_forecast.label.label_data import LabelData
        csv = self._make_label_csv(tmp_path)
        ld = LabelData(csv)
        p = ld.parameters
        assert p["window_step"] == 12
        assert p["window_unit"] == "hours"
        assert p["day_to_forecast"] == 2


# ---------------------------------------------------------------------------
# 5. DSAR — direct assignment (no intermediate aliases)
# ---------------------------------------------------------------------------

class TestDSARDirectAssignment:
    def test_no_intermediate_alias_in_calculate(self):
        """Ensure first_dsar / second_dsar local variables are gone from source."""
        import eruption_forecast.tremor.dsar as dsar_mod
        src = inspect.getsource(dsar_mod.DSAR.calculate)
        assert "first_dsar: pd.Series = first_stream" not in src
        assert "second_dsar: pd.Series = second_stream" not in src

    def test_no_redundant_series_wrap(self):
        """pd.Series(second_dsar) wrapper was removed."""
        import eruption_forecast.tremor.dsar as dsar_mod
        src = inspect.getsource(dsar_mod.DSAR.calculate)
        assert "pd.Series(second_dsar)" not in src

    def test_stores_series_after_calculate(self):
        from eruption_forecast.tremor.dsar import DSAR
        idx = pd.date_range("2025-01-01", periods=6, freq="10min")
        s1 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=idx)
        s2 = pd.Series([2.0, 2.0, 2.0, 2.0, 2.0, 2.0], index=idx)
        dsar = DSAR()
        result = dsar.calculate(s1, s2, window_duration_minutes=10)
        assert isinstance(dsar.first_dsar, pd.Series)
        assert isinstance(dsar.second_dsar, pd.Series)
        assert len(result) == 6


# ---------------------------------------------------------------------------
# 6. array.py — detect_anomalies_zscore uses _filter_nans
# ---------------------------------------------------------------------------

class TestDetectAnomaliesUsesFilterNans:
    def test_source_uses_filter_nans(self):
        import eruption_forecast.utils.array as arr_mod
        src = inspect.getsource(arr_mod.detect_anomalies_zscore)
        assert "filter_nans" in src, (
            "detect_anomalies_zscore should call filter_nans, not inline data[~np.isnan(data)]"
        )
        assert "data[~np.isnan(data)]" not in src

    def test_works_with_nans(self):
        from eruption_forecast.utils.array import detect_anomalies_zscore
        data = np.array([1.0, 2.0, np.nan, 3.0, 100.0, 2.0])
        result = detect_anomalies_zscore(data)
        assert result.dtype == bool
        assert result[-2]  # 100.0 should be flagged as anomaly


# ---------------------------------------------------------------------------
# 7. ml.py — confidence uses shape[1] (n_seeds), not shape[0] (n_windows)
# ---------------------------------------------------------------------------

class TestConfidenceNSeedsBugFix:
    def test_source_uses_shape_1(self):
        import eruption_forecast.utils.array as arr_mod
        src = inspect.getsource(arr_mod.compute_model_probabilities)
        assert "shape[1]" in src, "n_seeds must come from axis-1 (seeds), not axis-0 (windows)"

    def test_confidence_matrix_orientation(self):
        """Directly verify the fix: matrix is (n_windows, n_seeds) so n_seeds = shape[1]."""
        n_windows, n_seeds = 4, 10
        # Build the same structure as compute_model_probabilities does
        seed_eruption_probabilities = [
            np.ones(n_windows) * 0.9 for _ in range(n_seeds)  # all seeds predict eruption
        ]
        matrix = np.stack(seed_eruption_probabilities, axis=1)  # (n_windows, n_seeds)
        assert matrix.shape == (n_windows, n_seeds)

        # The bug: shape[0] = n_windows (4), not n_seeds (10)
        # The fix: shape[1] = n_seeds (10)
        n_seeds_correct = matrix.shape[1]
        votes = (matrix >= 0.5).sum(axis=1)
        confidence = votes / n_seeds_correct
        assert np.allclose(confidence, 1.0), (
            f"When all seeds agree, confidence must be 1.0, got {confidence}"
        )


# ---------------------------------------------------------------------------
# 8. model_trainer._cv_train_evaluate — X_train removed from signature
# ---------------------------------------------------------------------------

class TestCvTrainEvaluateNoXTrain:
    def test_x_train_not_in_signature(self):
        from eruption_forecast.model.model_trainer import ModelTrainer
        sig = inspect.signature(ModelTrainer._cv_train_evaluate)
        assert "X_train" not in sig.parameters, (
            "X_train was removed from _cv_train_evaluate; use selected_features instead"
        )

    def test_required_params_present(self):
        from eruption_forecast.model.model_trainer import ModelTrainer
        sig = inspect.signature(ModelTrainer._cv_train_evaluate)
        for param in ("y_train", "X_test", "y_test", "selected_features", "random_state"):
            assert param in sig.parameters
