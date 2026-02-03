"""Unit tests for ClassifierModel (Phase 5).

Tests cover initialisation, validation, the validate/create_directories
separation, and the train() pipeline.  All tests use synthetic data and
temporary directories — no real seismic data required.
"""

import os
import tempfile

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from eruption_forecast.features.constants import ERUPTED_COLUMN, ID_COLUMN
from eruption_forecast.model.classifier_model import ClassifierModel


# ---------------------------------------------------------------------------
# Helpers — synthetic data factories
# ---------------------------------------------------------------------------

_FAST_GRID: dict[str, list] = {
    "n_estimators": [10],
    "max_depth": [3],
    "criterion": ["gini"],
    "max_features": ["sqrt"],
}
"""Single-combination grid so each seed does exactly one fit."""


def _write_synthetic_csvs(
    tmp: str, n_features: int = 5, n_samples: int = 30
) -> tuple[str, str]:
    """Write synthetic features and label CSVs.

    30 rows by default; the last 6 rows are class 1 (~20 % imbalance).
    Returns ``(features_csv, label_csv)`` paths.
    """
    rng = np.random.default_rng(42)
    features = pd.DataFrame(
        rng.random((n_samples, n_features)),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    features.index.name = ID_COLUMN
    features_csv = os.path.join(tmp, "features.csv")
    features.to_csv(features_csv)

    labels = pd.DataFrame(
        {
            ID_COLUMN: range(n_samples),
            ERUPTED_COLUMN: [0] * (n_samples - 6) + [1] * 6,
        }
    )
    label_csv = os.path.join(tmp, "labels.csv")
    labels.to_csv(label_csv, index=False)

    return features_csv, label_csv


# ---------------------------------------------------------------------------
# ClassifierModel — initialisation & validation
# ---------------------------------------------------------------------------


class TestClassifierModelInit:
    """Test ClassifierModel construction and validation."""

    def test_valid_initialization(self) -> None:
        """Constructs successfully; features shape is 30 and output_dir exists."""
        with tempfile.TemporaryDirectory() as tmp:
            features_csv, label_csv = _write_synthetic_csvs(tmp)
            model = ClassifierModel(
                features_csv=features_csv,
                label_csv=label_csv,
                output_dir=os.path.join(tmp, "out"),
            )
            assert model.df_features.shape[0] == 30
            assert os.path.isdir(model.output_dir)

    def test_empty_features_raises(self) -> None:
        """ValueError when features CSV has zero rows."""
        with tempfile.TemporaryDirectory() as tmp:
            pd.DataFrame(columns=["feat_0"]).to_csv(
                os.path.join(tmp, "features.csv")
            )
            pd.DataFrame(
                {ID_COLUMN: [0], ERUPTED_COLUMN: [0]}
            ).to_csv(os.path.join(tmp, "labels.csv"), index=False)

            with pytest.raises(ValueError, match="Features cannot be empty"):
                ClassifierModel(
                    features_csv=os.path.join(tmp, "features.csv"),
                    label_csv=os.path.join(tmp, "labels.csv"),
                    output_dir=os.path.join(tmp, "out"),
                )

    def test_empty_labels_raises(self) -> None:
        """ValueError when label CSV has zero rows."""
        with tempfile.TemporaryDirectory() as tmp:
            pd.DataFrame({"feat_0": range(5)}).to_csv(
                os.path.join(tmp, "features.csv")
            )
            pd.DataFrame(columns=[ID_COLUMN, ERUPTED_COLUMN]).to_csv(
                os.path.join(tmp, "labels.csv"), index=False
            )

            with pytest.raises(ValueError, match="Labels cannot be empty"):
                ClassifierModel(
                    features_csv=os.path.join(tmp, "features.csv"),
                    label_csv=os.path.join(tmp, "labels.csv"),
                    output_dir=os.path.join(tmp, "out"),
                )

    def test_mismatched_lengths_raises(self) -> None:
        """ValueError when features and labels row counts differ."""
        with tempfile.TemporaryDirectory() as tmp:
            pd.DataFrame({"feat_0": range(10)}).to_csv(
                os.path.join(tmp, "features.csv")
            )
            pd.DataFrame(
                {ID_COLUMN: range(5), ERUPTED_COLUMN: [0] * 5}
            ).to_csv(os.path.join(tmp, "labels.csv"), index=False)

            with pytest.raises(ValueError, match="do not match"):
                ClassifierModel(
                    features_csv=os.path.join(tmp, "features.csv"),
                    label_csv=os.path.join(tmp, "labels.csv"),
                    output_dir=os.path.join(tmp, "out"),
                )

    def test_output_directories_created(self) -> None:
        """Nested output path is created on init."""
        with tempfile.TemporaryDirectory() as tmp:
            features_csv, label_csv = _write_synthetic_csvs(tmp)
            out = os.path.join(tmp, "deep", "models")
            ClassifierModel(
                features_csv=features_csv,
                label_csv=label_csv,
                output_dir=out,
            )
            assert os.path.isdir(out)


# ---------------------------------------------------------------------------
# ClassifierModel — validate
# ---------------------------------------------------------------------------


class TestClassifierModelValidate:
    """Prove that validate() does not create directories."""

    def test_validate_does_not_create_dirs(self) -> None:
        """Calling validate() after directory removal does not recreate it."""
        with tempfile.TemporaryDirectory() as tmp:
            features_csv, label_csv = _write_synthetic_csvs(tmp)
            out = os.path.join(tmp, "out")
            model = ClassifierModel(
                features_csv=features_csv,
                label_csv=label_csv,
                output_dir=out,
            )
            assert os.path.isdir(out)

            os.rmdir(out)
            model.validate()
            assert not os.path.isdir(out)


# ---------------------------------------------------------------------------
# ClassifierModel — train
# ---------------------------------------------------------------------------


class TestClassifierModelTrain:
    """Test ClassifierModel.train() behaviour."""

    def test_train_saves_models(self) -> None:
        """After train(), exactly total_seed .pkl files exist in output_dir."""
        with tempfile.TemporaryDirectory() as tmp:
            features_csv, label_csv = _write_synthetic_csvs(tmp)
            out = os.path.join(tmp, "out")
            model = ClassifierModel(
                features_csv=features_csv,
                label_csv=label_csv,
                output_dir=out,
            )
            model.train(total_seed=2, grid_params=_FAST_GRID, n_splits=2)

            pkls = [f for f in os.listdir(out) if f.endswith(".pkl")]
            assert len(pkls) == 2

    def test_train_returns_none(self) -> None:
        """train() returns None."""
        with tempfile.TemporaryDirectory() as tmp:
            features_csv, label_csv = _write_synthetic_csvs(tmp)
            out = os.path.join(tmp, "out")
            model = ClassifierModel(
                features_csv=features_csv,
                label_csv=label_csv,
                output_dir=out,
            )
            result = model.train(total_seed=2, grid_params=_FAST_GRID, n_splits=2)
            assert result is None

    def test_train_skip_existing(self) -> None:
        """Second train() with overwrite=False does not touch existing files."""
        with tempfile.TemporaryDirectory() as tmp:
            features_csv, label_csv = _write_synthetic_csvs(tmp)
            out = os.path.join(tmp, "out")
            model = ClassifierModel(
                features_csv=features_csv,
                label_csv=label_csv,
                output_dir=out,
            )
            model.train(total_seed=2, grid_params=_FAST_GRID, n_splits=2)

            pkls = sorted(f for f in os.listdir(out) if f.endswith(".pkl"))
            mtime_before = os.path.getmtime(os.path.join(out, pkls[0]))

            # Second run — should skip every file
            model.train(
                total_seed=2, grid_params=_FAST_GRID, n_splits=2, overwrite=False
            )
            mtime_after = os.path.getmtime(os.path.join(out, pkls[0]))
            assert mtime_before == mtime_after

    def test_train_model_is_loadable(self) -> None:
        """Saved .pkl file loads as a RandomForestClassifier."""
        with tempfile.TemporaryDirectory() as tmp:
            features_csv, label_csv = _write_synthetic_csvs(tmp)
            out = os.path.join(tmp, "out")
            model = ClassifierModel(
                features_csv=features_csv,
                label_csv=label_csv,
                output_dir=out,
            )
            model.train(total_seed=2, grid_params=_FAST_GRID, n_splits=2)

            pkl_files = [f for f in os.listdir(out) if f.endswith(".pkl")]
            loaded = joblib.load(os.path.join(out, pkl_files[0]))
            assert isinstance(loaded, RandomForestClassifier)

    def test_train_model_can_predict(self) -> None:
        """Loaded model can predict on feature-shaped input."""
        with tempfile.TemporaryDirectory() as tmp:
            features_csv, label_csv = _write_synthetic_csvs(tmp)
            out = os.path.join(tmp, "out")
            model = ClassifierModel(
                features_csv=features_csv,
                label_csv=label_csv,
                output_dir=out,
            )
            model.train(total_seed=2, grid_params=_FAST_GRID, n_splits=2)

            pkl_files = [f for f in os.listdir(out) if f.endswith(".pkl")]
            loaded = joblib.load(os.path.join(out, pkl_files[0]))

            X = pd.read_csv(features_csv, index_col=0)
            predictions = loaded.predict(X)
            assert len(predictions) == len(X)


# ---------------------------------------------------------------------------
# ClassifierModel — evaluate
# ---------------------------------------------------------------------------


class TestClassifierModelEvaluate:
    """Test ClassifierModel.evaluate() behaviour."""

    def test_evaluate_returns_dataframe(self) -> None:
        """evaluate() returns a DataFrame with correct structure."""
        with tempfile.TemporaryDirectory() as tmp:
            features_csv, label_csv = _write_synthetic_csvs(tmp, n_samples=50)
            out = os.path.join(tmp, "out")
            model = ClassifierModel(
                features_csv=features_csv,
                label_csv=label_csv,
                output_dir=out,
            )
            model.train(total_seed=2, grid_params=_FAST_GRID, n_splits=2)

            results = model.evaluate(save_results=False)

            # 2 models + MEAN + STD rows
            assert len(results) == 4
            assert "model" in results.columns
            assert "accuracy" in results.columns
            assert "f1" in results.columns
            assert "roc_auc" in results.columns
            assert results.iloc[-2]["model"] == "MEAN"
            assert results.iloc[-1]["model"] == "STD"

    def test_evaluate_saves_csv(self) -> None:
        """evaluate() saves results to CSV when save_results=True."""
        with tempfile.TemporaryDirectory() as tmp:
            features_csv, label_csv = _write_synthetic_csvs(tmp, n_samples=50)
            out = os.path.join(tmp, "out")
            model = ClassifierModel(
                features_csv=features_csv,
                label_csv=label_csv,
                output_dir=out,
            )
            model.train(total_seed=2, grid_params=_FAST_GRID, n_splits=2)

            model.evaluate(save_results=True)

            csv_path = os.path.join(out, "evaluation_results.csv")
            assert os.path.isfile(csv_path)

            loaded = pd.read_csv(csv_path)
            assert len(loaded) == 4

    def test_evaluate_custom_filename(self) -> None:
        """evaluate() uses custom filename when provided."""
        with tempfile.TemporaryDirectory() as tmp:
            features_csv, label_csv = _write_synthetic_csvs(tmp, n_samples=50)
            out = os.path.join(tmp, "out")
            model = ClassifierModel(
                features_csv=features_csv,
                label_csv=label_csv,
                output_dir=out,
            )
            model.train(total_seed=2, grid_params=_FAST_GRID, n_splits=2)

            model.evaluate(save_results=True, output_filename="custom_eval.csv")

            assert os.path.isfile(os.path.join(out, "custom_eval.csv"))

    def test_evaluate_no_models_raises(self) -> None:
        """evaluate() raises ValueError when no models exist."""
        with tempfile.TemporaryDirectory() as tmp:
            features_csv, label_csv = _write_synthetic_csvs(tmp)
            out = os.path.join(tmp, "out")
            model = ClassifierModel(
                features_csv=features_csv,
                label_csv=label_csv,
                output_dir=out,
            )
            # Do NOT train

            with pytest.raises(ValueError, match="No trained models found"):
                model.evaluate()

    def test_evaluate_metrics_in_range(self) -> None:
        """All evaluation metrics are in valid ranges."""
        with tempfile.TemporaryDirectory() as tmp:
            features_csv, label_csv = _write_synthetic_csvs(tmp, n_samples=50)
            out = os.path.join(tmp, "out")
            model = ClassifierModel(
                features_csv=features_csv,
                label_csv=label_csv,
                output_dir=out,
            )
            model.train(total_seed=3, grid_params=_FAST_GRID, n_splits=2)

            results = model.evaluate(save_results=False)

            # Check model rows (exclude MEAN and STD)
            model_rows = results[~results["model"].isin(["MEAN", "STD"])]

            for _, row in model_rows.iterrows():
                assert 0 <= row["accuracy"] <= 1
                assert 0 <= row["precision"] <= 1
                assert 0 <= row["recall"] <= 1
                assert 0 <= row["f1"] <= 1
                assert 0 <= row["roc_auc"] <= 1


# ---------------------------------------------------------------------------
# ClassifierModel — get_classification_report
# ---------------------------------------------------------------------------


class TestClassifierModelClassificationReport:
    """Test ClassifierModel.get_classification_report() behaviour."""

    def test_classification_report_returns_string(self) -> None:
        """get_classification_report() returns a string."""
        with tempfile.TemporaryDirectory() as tmp:
            features_csv, label_csv = _write_synthetic_csvs(tmp, n_samples=50)
            out = os.path.join(tmp, "out")
            model = ClassifierModel(
                features_csv=features_csv,
                label_csv=label_csv,
                output_dir=out,
            )
            model.train(total_seed=2, grid_params=_FAST_GRID, n_splits=2)

            report = model.get_classification_report()

            assert isinstance(report, str)
            assert "precision" in report
            assert "recall" in report
            assert "f1-score" in report
            assert "Not Erupted" in report
            assert "Erupted" in report

    def test_classification_report_no_models_raises(self) -> None:
        """get_classification_report() raises ValueError when no models exist."""
        with tempfile.TemporaryDirectory() as tmp:
            features_csv, label_csv = _write_synthetic_csvs(tmp)
            out = os.path.join(tmp, "out")
            model = ClassifierModel(
                features_csv=features_csv,
                label_csv=label_csv,
                output_dir=out,
            )

            with pytest.raises(ValueError, match="No trained models found"):
                model.get_classification_report()

    def test_classification_report_invalid_index_raises(self) -> None:
        """get_classification_report() raises IndexError for out-of-range index."""
        with tempfile.TemporaryDirectory() as tmp:
            features_csv, label_csv = _write_synthetic_csvs(tmp, n_samples=50)
            out = os.path.join(tmp, "out")
            model = ClassifierModel(
                features_csv=features_csv,
                label_csv=label_csv,
                output_dir=out,
            )
            model.train(total_seed=2, grid_params=_FAST_GRID, n_splits=2)

            with pytest.raises(IndexError, match="out of range"):
                model.get_classification_report(model_index=10)


# ---------------------------------------------------------------------------
# ClassifierModel — get_feature_importances
# ---------------------------------------------------------------------------


class TestClassifierModelFeatureImportances:
    """Test ClassifierModel.get_feature_importances() behaviour."""

    def test_feature_importances_returns_dataframe(self) -> None:
        """get_feature_importances() returns a DataFrame with correct columns."""
        with tempfile.TemporaryDirectory() as tmp:
            features_csv, label_csv = _write_synthetic_csvs(tmp, n_features=5)
            out = os.path.join(tmp, "out")
            model = ClassifierModel(
                features_csv=features_csv,
                label_csv=label_csv,
                output_dir=out,
            )
            model.train(total_seed=2, grid_params=_FAST_GRID, n_splits=2)

            importances = model.get_feature_importances()

            assert isinstance(importances, pd.DataFrame)
            assert "feature" in importances.columns
            assert "importance" in importances.columns
            assert len(importances) == 5

    def test_feature_importances_sorted_descending(self) -> None:
        """get_feature_importances() returns importances sorted descending."""
        with tempfile.TemporaryDirectory() as tmp:
            features_csv, label_csv = _write_synthetic_csvs(tmp, n_features=5)
            out = os.path.join(tmp, "out")
            model = ClassifierModel(
                features_csv=features_csv,
                label_csv=label_csv,
                output_dir=out,
            )
            model.train(total_seed=2, grid_params=_FAST_GRID, n_splits=2)

            importances = model.get_feature_importances()

            values = importances["importance"].tolist()
            assert values == sorted(values, reverse=True)

    def test_feature_importances_top_n(self) -> None:
        """get_feature_importances(top_n=3) returns only top 3 features."""
        with tempfile.TemporaryDirectory() as tmp:
            features_csv, label_csv = _write_synthetic_csvs(tmp, n_features=10)
            out = os.path.join(tmp, "out")
            model = ClassifierModel(
                features_csv=features_csv,
                label_csv=label_csv,
                output_dir=out,
            )
            model.train(total_seed=2, grid_params=_FAST_GRID, n_splits=2)

            importances = model.get_feature_importances(top_n=3)

            assert len(importances) == 3

    def test_feature_importances_no_models_raises(self) -> None:
        """get_feature_importances() raises ValueError when no models exist."""
        with tempfile.TemporaryDirectory() as tmp:
            features_csv, label_csv = _write_synthetic_csvs(tmp)
            out = os.path.join(tmp, "out")
            model = ClassifierModel(
                features_csv=features_csv,
                label_csv=label_csv,
                output_dir=out,
            )

            with pytest.raises(ValueError, match="No trained models found"):
                model.get_feature_importances()
