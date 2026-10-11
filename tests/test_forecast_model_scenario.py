"""Tests for ``ForecastModelScenario``.

Two tiny scenarios run against seeded synthetic tremor (see
``conftest.seed_tremor_csv``) with one-point hyperparameter grids. The
second scenario's prediction window sits inside the first one's so the
``use_features_from="files"`` reuse path is exercised. Telegram is faked.
"""

import os
import json
import importlib

import pandas as pd
import pytest

from tests.conftest import seed_tremor_csv, eruption_ramp_tremor_df
from eruption_forecast.model.prediction_model import PredictionModel
from eruption_forecast.model.forecast_model_scenario import (
    Scenario,
    ForecastModelScenario,
)


scenario_module = importlib.import_module(
    "eruption_forecast.model.forecast_model_scenario"
)

ERUPTION_DATES = ["2025-01-20"]
TREMOR_END = "2025-01-31"

SCENARIOS: list[Scenario] = [
    {
        "name": "Scenario 1",
        "description": "Wide prediction window",
        "train_start_date": "2025-01-03",
        "train_end_date": "2025-01-22",
        "prediction_start_date": "2025-01-18",
        "prediction_end_date": "2025-01-31",
        "prediction_window_step": 6,
        "prediction_window_step_unit": "hours",
    },
    {
        "name": "Scenario 2",
        "description": "Narrow prediction window inside scenario 1",
        "train_start_date": "2025-01-03",
        "train_end_date": "2025-01-21",
        "prediction_start_date": "2025-01-20",
        "prediction_end_date": "2025-01-27",
        "prediction_window_step": 6,
        "prediction_window_step_unit": "hours",
    },
]


class _FakeTelegram:
    """Records every document the scenario loop tries to send."""

    sent: list[tuple[str, str]] = []

    def __init__(self, *args, **kwargs) -> None:
        pass

    def send_document(self, path: str, caption: str = "", **kwargs):
        _FakeTelegram.sent.append((path, caption))
        return self


def _scenario_runner(output_dir: str, scenarios=SCENARIOS) -> ForecastModelScenario:
    return ForecastModelScenario(
        station="OJN",
        channel="EHZ",
        network="VG",
        location="00",
        eruption_dates=ERUPTION_DATES,
        scenarios=scenarios,
        day_to_forecast=2,
        output_dir=output_dir,
    )


@pytest.fixture()
def seeded(tmp_path) -> dict[str, str]:
    output_dir = str(tmp_path / "output")
    sds_dir = tmp_path / "sds"
    sds_dir.mkdir()
    # Prediction windows must span at least 7 days, so extend past the
    # pipeline tests' 2025-01-25 cut-off.
    seed_tremor_csv(output_dir, eruption_ramp_tremor_df(end=TREMOR_END))
    return {"output_dir": output_dir, "sds_dir": str(sds_dir)}


def _calculated(seeded) -> ForecastModelScenario:
    return _scenario_runner(seeded["output_dir"]).calculate(
        start_date="2025-01-03",
        end_date=TREMOR_END,
        source="sds",
        sds_dir=seeded["sds_dir"],
    )


class TestValidation:
    def test_run_before_setup_raises(self, seeded) -> None:
        with pytest.raises(RuntimeError):
            _scenario_runner(seeded["output_dir"]).run(
                evaluate_model=False, explain_model=False
            )

    def test_requires_shared_training(self, seeded) -> None:
        runner = _calculated(seeded)
        with pytest.raises(RuntimeError, match="shared_training"):
            runner.run(evaluate_model=False, explain_model=False)

    def test_requires_shared_evaluation_when_evaluating(self, seeded) -> None:
        with pytest.raises(RuntimeError, match="shared_evaluation"):
            _calculated(seeded).run(evaluate_model=True, explain_model=False)

    def test_duplicate_scenario_slugs_raise(self, seeded) -> None:
        duplicate = [SCENARIOS[0], {**SCENARIOS[1], "name": "scenario-1"}]
        runner = _scenario_runner(seeded["output_dir"], scenarios=duplicate)
        with pytest.raises(ValueError):
            runner.validate()

    def test_plot_kwargs_always_carry_eruption_dates(self) -> None:
        scenario = {**SCENARIOS[0], "prediction_plot_kwargs": {"legend_n_cols": 2}}
        kwargs = ForecastModelScenario.build_plot_kwargs(scenario, ERUPTION_DATES)
        assert kwargs == {"legend_n_cols": 2, "eruption_dates": ERUPTION_DATES}


@pytest.mark.slow
def test_run_two_scenarios(seeded, tiny_grids, monkeypatch) -> None:
    _FakeTelegram.sent = []
    monkeypatch.setattr(scenario_module, "TelegramNotification", _FakeTelegram)

    # Scenario 2 must reuse scenario 1's prediction features from disk.
    original_extract = PredictionModel.extract_features
    extract_calls: list[str] = []

    def _tracking_extract(self, *args, **kwargs):
        extract_calls.append(self.start_date_str)
        return original_extract(self, *args, **kwargs)

    monkeypatch.setattr(PredictionModel, "extract_features", _tracking_extract)

    runner = (
        _calculated(seeded)
        .shared_training(
            window_step=6,
            window_step_unit="hours",
            classifiers=["lite-rf"],
            cv_splits=2,
            top_n_features=5,
            seeds=1,
            save_tremor_matrix_per_method=False,
            plot_features=False,
        )
        .shared_prediction(use_features_from="files", plot_pdf=False)
        .shared_evaluation(model="prediction", plot_aggregate=False)
    )

    runner.run(evaluate_model=True, explain_model=False, notification_message="ci")

    manifest = os.path.join(runner.scenarios_dir, "scenarios.json")
    with open(manifest, encoding="utf-8") as f:
        assert [s["name"] for s in json.load(f)] == ["Scenario 1", "Scenario 2"]

    for slug in ("scenario-1", "scenario-2"):
        scenario_dir = os.path.join(runner.scenarios_dir, slug)
        for stage in ("training", "prediction", "evaluation"):
            assert os.path.isdir(os.path.join(scenario_dir, stage)), (slug, stage)
        assert os.path.isfile(
            os.path.join(scenario_dir, "training", f"training.{slug}.config.yaml")
        )

    assert extract_calls == ["2025-01-18"]
    assert len(_FakeTelegram.sent) == 2
    assert _FakeTelegram.sent[1][1] == (
        "[CI] Scenario 2: Narrow prediction window inside scenario 1"
    )
    assert runner.ForecastModel.results is not None
    # Exact row counts for the reuse path live in test_prediction_model.py.
    index = runner.ForecastModel.results.index
    assert index.min() == pd.Timestamp("2025-01-20")
    assert index.max().date() == pd.Timestamp("2025-01-27").date()
