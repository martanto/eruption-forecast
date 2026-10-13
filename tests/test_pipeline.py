"""End-to-end tests for ``ForecastModel`` and the four stage models.

One module-scoped pipeline runs ``calculate -> train -> predict -> evaluate
-> explain`` on synthetic tremor that ramps up before a 2025-01-20 eruption.
Speed comes from three shortcuts:

* ``seed_tremor_csv`` pre-writes the merged tremor CSV, so ``calculate()``
  loads it instead of processing waveforms.
* ``tiny_grids`` collapses every hyperparameter grid to a single point.
* Two tremor columns, two seeds, two CV splits, top-5 features.

The whole module still takes a couple of minutes, so it is marked ``slow``.
Run it with ``uv run pytest tests/ -m slow``.
"""

import os
import glob

import numpy as np
import pandas as pd
import pytest

from tests.conftest import seed_tremor_csv, eruption_ramp_tremor_df
from eruption_forecast.model.forecast_model import ForecastModel
from eruption_forecast.model.training_model import TrainingModel
from eruption_forecast.model.evaluation_model import EvaluationModel
from eruption_forecast.model.prediction_model import PredictionModel
from eruption_forecast.model.explanation_model import ExplanationModel
from eruption_forecast.ensemble.classifier_ensemble import ClassifierEnsemble


pytestmark = pytest.mark.slow

ERUPTION_DATES = ["2025-01-20"]
CLASSIFIER_NAMES = {"LiteRandomForestClassifier", "GaussianNB"}

CALCULATE_KWARGS = {"start_date": "2025-01-03", "end_date": "2025-01-25"}
TRAIN_KWARGS = {
    "start_date": "2025-01-03",
    "end_date": "2025-01-22",
    "eruption_dates": ERUPTION_DATES,
    "window_step": 6,
    "window_step_unit": "hours",
    "classifiers": ["lite-rf", "nb"],
    "cv_splits": 2,
    "top_n_features": 5,
    "seeds": 2,
    "save_tremor_matrix_per_method": False,
    "plot_features": False,
}
PREDICT_KWARGS = {
    "start_date": "2025-01-18",
    "end_date": "2025-01-25",
    "window_step": 6,
    "window_step_unit": "hours",
}


def _forecast_model(output_dir: str) -> ForecastModel:
    return ForecastModel(
        station="OJN",
        channel="EHZ",
        network="VG",
        location="00",
        day_to_forecast=2,
        output_dir=output_dir,
    )


def _calculate(fm: ForecastModel, sds_dir: str) -> ForecastModel:
    # ``sds_dir`` only has to exist: the seeded CSV short-circuits the run.
    return fm.calculate(**CALCULATE_KWARGS, source="sds", sds_dir=sds_dir)


@pytest.fixture(scope="module")
def workspace(tmp_path_factory, tiny_grids) -> dict[str, str]:
    """Seed the tremor CSV and an empty SDS root."""
    root = tmp_path_factory.mktemp("pipeline")
    output_dir = str(root / "output")
    sds_dir = root / "sds"
    sds_dir.mkdir()
    seed_tremor_csv(output_dir, eruption_ramp_tremor_df())
    return {"output_dir": output_dir, "sds_dir": str(sds_dir)}


@pytest.fixture(scope="module")
def fm(workspace) -> ForecastModel:
    """Run the full chain once for the whole module."""
    model = _forecast_model(workspace["output_dir"])
    (
        _calculate(model, workspace["sds_dir"])
        .train(**TRAIN_KWARGS)
        .predict(**PREDICT_KWARGS)
        .evaluate(model="prediction")
        .explain(model="prediction", max_display=5, dpi=50)
    )
    return model


# ---------------------------------------------------------------------------
# calculate
# ---------------------------------------------------------------------------


class TestCalculate:
    def test_loads_seeded_tremor(self, fm: ForecastModel) -> None:
        assert fm.tremor_df.columns.tolist() == ["rsam_f2", "entropy"]
        # calculate() shifts the start back by day_to_forecast for lead-in.
        assert fm.tremor_start_date == pd.Timestamp("2025-01-01")
        assert fm.tremor_end_date.date() == pd.Timestamp("2025-01-25").date()

    def test_train_before_calculate_raises(self, tmp_path) -> None:
        with pytest.raises(ValueError):
            _forecast_model(str(tmp_path)).train(**TRAIN_KWARGS)


# ---------------------------------------------------------------------------
# TrainingModel
# ---------------------------------------------------------------------------


class TestTrainingModel:
    def test_ensemble_has_every_classifier(self, fm: ForecastModel) -> None:
        tm = fm.TrainingModel
        assert isinstance(tm, TrainingModel)
        assert tm.kind == "training"
        assert set(fm.ClassifierEnsemble.classifiers) == CLASSIFIER_NAMES
        for name in CLASSIFIER_NAMES:
            assert len(fm.ClassifierEnsemble[name]) == TRAIN_KWARGS["seeds"]

    def test_ensemble_persisted_and_reloadable(self, fm: ForecastModel) -> None:
        path = fm.TrainingModel.classifier_ensemble_path
        assert os.path.isfile(path)
        reloaded = ClassifierEnsemble.from_any(path)
        assert set(reloaded.classifiers) == CLASSIFIER_NAMES

    def test_registry_json_per_classifier(self, fm: ForecastModel) -> None:
        classifiers_dir = os.path.join(fm.TrainingModel.training_dir, "classifiers")
        registries = glob.glob(
            os.path.join(classifiers_dir, "*", "*", "trained-model__*.json")
        )
        assert len(registries) == len(CLASSIFIER_NAMES)

    def test_each_seed_uses_top_n_features(self, fm: ForecastModel) -> None:
        for seed in fm.ClassifierEnsemble["GaussianNB"].seeds:
            assert len(seed["feature_names"]) == TRAIN_KWARGS["top_n_features"]

    def test_labels_and_features_align(self, fm: ForecastModel) -> None:
        tm = fm.TrainingModel
        assert len(tm.features_df) == len(tm.labels)
        assert set(tm.labels.unique()) == {0, 1}

    def test_window_ids_carry_datetimes(self, fm: ForecastModel) -> None:
        window_ids = fm.TrainingModel.window_ids
        assert isinstance(window_ids.index, pd.DatetimeIndex)
        assert sorted(window_ids) == sorted(fm.TrainingModel.labels.index)

    def test_writes_config_and_cache(self, fm: ForecastModel) -> None:
        tm = fm.TrainingModel
        assert os.path.isfile(os.path.join(tm.training_dir, "training.config.yaml"))
        assert tm.id is not None
        assert os.path.isfile(
            os.path.join(tm.training_dir, f"{tm.id}.TrainingModel.pkl")
        )

    def test_load_features_skips_tsfresh(self, fm: ForecastModel, tmp_path) -> None:
        tm = fm.TrainingModel
        (label_csv,) = glob.glob(
            os.path.join(tm.training_dir, "features", "*", "features-label_*.csv")
        )
        fresh = TrainingModel(
            tremor_data=fm.tremor_df,
            start_date=TRAIN_KWARGS["start_date"],
            end_date=TRAIN_KWARGS["end_date"],
            classifiers="nb",
            eruption_dates=ERUPTION_DATES,
            output_dir=str(tmp_path),
        )

        fresh.load_features(
            features_matrix_path=tm.features_path, label_features_csv=label_csv
        )

        assert fresh.features_df.shape == tm.features_df.shape
        assert fresh.window_ids.sort_index().equals(tm.window_ids.sort_index())


# ---------------------------------------------------------------------------
# PredictionModel
# ---------------------------------------------------------------------------


class TestPredictionModel:
    def test_results_columns(self, fm: ForecastModel) -> None:
        expected = {"consensus"} | CLASSIFIER_NAMES
        columns = set(fm.results.columns)
        for prefix in expected:
            for suffix in ("probability", "uncertainty", "prediction", "confidence"):
                assert f"{prefix}_{suffix}" in columns

    def test_one_row_per_window(self, fm: ForecastModel) -> None:
        # 2025-01-18 .. 2025-01-25 at 6 hours = 8 days x 4 windows.
        assert len(fm.results) == 32

    def test_probabilities_in_unit_interval(self, fm: ForecastModel) -> None:
        probability = fm.results["consensus_probability"]
        assert probability.between(0, 1).all()

    def test_forecast_rises_before_eruption(self, fm: ForecastModel) -> None:
        # The training ramp is the only signal in the data, so the windows
        # right before the eruption must look more eruptive than the quiet
        # tail of the prediction period.
        probability = fm.results["LiteRandomForestClassifier_probability"]
        index = pd.DatetimeIndex(fm.PredictionModel.labels.index)
        pre_eruption = probability[(index >= "2025-01-18") & (index < "2025-01-20")]
        quiet = probability[index >= "2025-01-23"]
        assert pre_eruption.mean() > quiet.mean()

    def test_training_hash_threaded_through(self, fm: ForecastModel) -> None:
        assert fm.PredictionModel.training_hash == fm.TrainingModel.id
        assert fm.PredictionModel.kind == "prediction"

    def test_artefacts_written(self, fm: ForecastModel) -> None:
        pm = fm.PredictionModel
        assert pm.forecast_plot_path and os.path.isfile(pm.forecast_plot_path)
        results_dir = os.path.join(fm.station_dir, "prediction", "results")
        assert glob.glob(os.path.join(results_dir, "forecast-results_*.csv"))

    def test_identity_depends_on_training_hash(self, fm: ForecastModel) -> None:
        kwargs = {
            "nslc": fm.nslc,
            "tremor_df": fm.tremor_df,
            "start_date": PREDICT_KWARGS["start_date"],
            "end_date": PREDICT_KWARGS["end_date"],
            "window_size": 2,
            "build_label_params": {"window_step": 6, "window_step_unit": "hours"},
            "extract_features_params": {},
        }
        first = PredictionModel.build_identity(training_hash="aaa", **kwargs)
        second = PredictionModel.build_identity(training_hash="bbb", **kwargs)
        assert PredictionModel.compute_hash(first) != PredictionModel.compute_hash(
            second
        )


# ---------------------------------------------------------------------------
# EvaluationModel
# ---------------------------------------------------------------------------


class TestEvaluationModel:
    def test_prediction_mode_results(self, fm: ForecastModel) -> None:
        em = fm.EvaluationModel
        assert em.model_kind == "prediction"
        assert set(fm.evaluation_results) == CLASSIFIER_NAMES
        for metrics in fm.evaluation_results.values():
            assert len(metrics) == TRAIN_KWARGS["seeds"]
            assert "roc_auc" in metrics.columns

    def test_prediction_mode_outputs_namespaced(self, fm: ForecastModel) -> None:
        evaluation_dir = os.path.join(fm.station_dir, "evaluation", "prediction")
        assert os.path.isfile(os.path.join(evaluation_dir, "labels", "y_true.csv"))
        for name in CLASSIFIER_NAMES:
            predictions = os.path.join(
                evaluation_dir, "classifiers", name, "predictions", "y_proba.csv"
            )
            assert os.path.isfile(predictions)

    def test_y_true_marks_eruption_day(self, fm: ForecastModel) -> None:
        y_true = pd.Series(np.asarray(fm.EvaluationModel.y_true))
        # Prediction labels use day_to_forecast=2 plus the eruption day.
        assert 0 < y_true.sum() < len(y_true)

    def test_compare_reuses_metrics_ensemble(self, fm: ForecastModel) -> None:
        em = fm.EvaluationModel
        cached = em.MetricsEnsemble
        comparator = em.compare()
        assert em.MetricsEnsemble is cached
        assert set(comparator.get_ranking().index) == CLASSIFIER_NAMES

    def test_training_mode(self, fm: ForecastModel, tmp_path) -> None:
        em = EvaluationModel(fm.TrainingModel, output_dir=str(tmp_path))
        metrics = em.evaluate(plot_aggregate=False, compare_classifiers=False)
        assert set(metrics) == CLASSIFIER_NAMES
        assert os.path.isdir(os.path.join(str(tmp_path), "evaluation", "training"))

    def test_prediction_mode_requires_eruption_dates(self, fm: ForecastModel) -> None:
        with pytest.raises(ValueError):
            EvaluationModel(fm.PredictionModel)

    def test_from_file_round_trip(self, fm: ForecastModel, tmp_path) -> None:
        path = fm.PredictionModel.save(path=str(tmp_path / "PredictionModel.pkl"))
        em = EvaluationModel.from_file(
            path, eruption_dates=ERUPTION_DATES, output_dir=str(tmp_path)
        )
        assert em.model_kind == "prediction"


# ---------------------------------------------------------------------------
# ExplanationModel
# ---------------------------------------------------------------------------


class TestExplanationModel:
    def test_forecast_model_keeps_explanation_model(self, fm: ForecastModel) -> None:
        assert isinstance(fm.ExplanationModel, ExplanationModel)
        assert fm.ExplanationModel.explanations

    def test_only_tree_classifiers_explained(self, fm: ForecastModel, tmp_path) -> None:
        xm = ExplanationModel(
            fm.PredictionModel, eruption_dates=ERUPTION_DATES, output_dir=str(tmp_path)
        ).explain()
        names = [e.classifier_name for e in xm.explanations]
        assert names == ["LiteRandomForestClassifier"]

    def test_outputs_namespaced(self, fm: ForecastModel) -> None:
        clf_dir = os.path.join(
            fm.station_dir,
            "explanation",
            "prediction",
            "classifiers",
            "LiteRandomForestClassifier",
        )
        assert os.path.isfile(
            os.path.join(clf_dir, "ClassifierExplanation_LiteRandomForestClassifier.pkl")
        )
        assert os.path.isdir(os.path.join(clf_dir, "figures", "aggregate"))

    def test_waterfall_per_eruption(self, fm: ForecastModel) -> None:
        eruption_dir = os.path.join(
            fm.station_dir, "explanation", "prediction", "eruptions", "2025-01-20"
        )
        assert any(name.endswith(".png") for name in os.listdir(eruption_dir))

    def test_prediction_mode_requires_eruption_dates(self, fm: ForecastModel) -> None:
        with pytest.raises(ValueError):
            ExplanationModel(fm.PredictionModel)

    def test_training_mode_uses_upstream_eruption_dates(
        self, fm: ForecastModel, tmp_path
    ) -> None:
        xm = ExplanationModel(fm.TrainingModel, output_dir=str(tmp_path)).explain()
        assert [e.classifier_name for e in xm.explanations] == [
            "LiteRandomForestClassifier"
        ]

    def test_training_mode_waterfall_per_eruption(
        self, fm: ForecastModel, tmp_path
    ) -> None:
        ExplanationModel(fm.TrainingModel, output_dir=str(tmp_path)).explain().plot(
            max_display=5, dpi=50, plot_per_seed=False, plot_aggregate=False
        )
        eruption_dir = tmp_path / "explanation" / "training" / "eruptions" / "2025-01-20"
        assert any(name.endswith(".png") for name in os.listdir(eruption_dir))

    def test_training_mode_waterfall_labels_fallback(
        self, fm: ForecastModel, tmp_path, monkeypatch
    ) -> None:
        # Training pickles cached before ``window_ids`` existed lack it, so
        # the datetimes must come from the ``LabelBuilder`` grid instead.
        expected = fm.TrainingModel.window_ids
        monkeypatch.setattr(fm.TrainingModel, "window_ids", pd.Series())
        xm = ExplanationModel(fm.TrainingModel, output_dir=str(tmp_path))
        labels = xm._waterfall_labels()
        assert isinstance(labels.index, pd.DatetimeIndex)
        assert labels.sort_index().tolist() == expected.sort_index().tolist()


# ---------------------------------------------------------------------------
# Cache and config replay
# ---------------------------------------------------------------------------


def _forbid(monkeypatch, cls: type, method: str) -> None:
    """Make ``cls.method`` fail loudly so a cache miss cannot pass silently."""

    def _fail(*args, **kwargs):
        raise AssertionError(f"{cls.__name__}.{method} ran; expected a cache hit")

    monkeypatch.setattr(cls, method, _fail)


class TestCache:
    def test_train_cache_hit(self, fm: ForecastModel, workspace, monkeypatch) -> None:
        second = _calculate(_forecast_model(workspace["output_dir"]), workspace["sds_dir"])
        _forbid(monkeypatch, TrainingModel, "extract_features")

        second.train(**TRAIN_KWARGS)

        assert second.TrainingModel.id == fm.TrainingModel.id

    def test_predict_cache_hit(self, fm: ForecastModel, workspace, monkeypatch) -> None:
        second = _calculate(_forecast_model(workspace["output_dir"]), workspace["sds_dir"])
        second.train(**TRAIN_KWARGS)
        _forbid(monkeypatch, PredictionModel, "extract_features")

        second.predict(**PREDICT_KWARGS)

        pd.testing.assert_frame_equal(second.results, fm.results)

    def test_config_round_trip(self, fm: ForecastModel) -> None:
        path = fm.save_config()
        replayed = ForecastModel.from_config(path)
        assert replayed._config.train.classifiers == TRAIN_KWARGS["classifiers"]
        assert replayed._config.predict.start_date == PREDICT_KWARGS["start_date"]
        assert replayed._config.evaluate.model == "prediction"

    def test_replay_comes_from_cache(self, fm: ForecastModel, monkeypatch) -> None:
        replayed = ForecastModel.from_config(fm.save_config())

        # Every stage must come from cache: nothing re-extracts features.
        _forbid(monkeypatch, TrainingModel, "extract_features")
        _forbid(monkeypatch, PredictionModel, "extract_features")
        replayed.run()

        pd.testing.assert_frame_equal(replayed.results, fm.results)
