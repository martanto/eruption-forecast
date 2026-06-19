# %%
import os
from typing import Any, Literal, TypedDict, NotRequired

from dotenv import load_dotenv

from eruption_forecast import send_telegram_notification
from eruption_forecast.logger import logger
from eruption_forecast.utils.formatting import slugify
from eruption_forecast.model.forecast_model import ForecastModel


class Scenario(TypedDict):
    name: str
    description: str
    train_start_date: str
    train_end_date: str
    prediction_start_date: str
    prediction_end_date: str
    window_step: int
    window_step_unit: Literal["minutes", "hours"]
    plot_kwargs: NotRequired[dict[str, Any]]


# %%
load_dotenv(override=True)

root_dir = r"D:\Projects\eruption-forecast"
eruption_dates = [
    "2025-03-20",
    "2025-04-10",
    "2025-04-22",
    "2025-05-18",
    "2025-06-17",
    "2025-07-07",
    "2025-08-02",
    "2025-08-18",
]


scenarios: list[Scenario] = [
    {
        "name": "Scenario 1",
        "description": "Training using 1 eruption to forecast eruption 2",
        "train_start_date": "2025-01-01",
        "train_end_date": "2025-03-31",
        "prediction_start_date": "2025-04-01",
        "prediction_end_date": "2025-04-30",
        "window_step": 10,
        "window_step_unit": "minutes",
    },
    {
        "name": "Scenario 2",
        "description": "Training using 1 eruption to forecast eruption 3",
        "train_start_date": "2025-01-01",
        "train_end_date": "2025-03-31",
        "prediction_start_date": "2025-05-01",
        "prediction_end_date": "2025-05-31",
        "window_step": 10,
        "window_step_unit": "minutes",
    },
    {
        "name": "Scenario 3",
        "description": "Training using 1 eruption to forecast eruption 4",
        "train_start_date": "2025-01-01",
        "train_end_date": "2025-03-31",
        "prediction_start_date": "2025-06-01",
        "prediction_end_date": "2025-06-30",
        "window_step": 10,
        "window_step_unit": "minutes",
    },
    {
        "name": "Scenario 5",
        "description": "Training using 1 and 2 eruptions to forecast eruption 3",
        "train_start_date": "2025-01-01",
        "train_end_date": "2025-04-30",
        "prediction_start_date": "2025-05-01",
        "prediction_end_date": "2025-05-31",
        "window_step": 10,
        "window_step_unit": "minutes",
    },
    {
        "name": "Scenario 6",
        "description": "Training using 1, 2 and 3 eruptions to forecast eruption 4",
        "train_start_date": "2025-01-01",
        "train_end_date": "2025-05-31",
        "prediction_start_date": "2025-06-01",
        "prediction_end_date": "2025-06-30",
        "window_step": 10,
        "window_step_unit": "minutes",
    },
    {
        "name": "Scenario 7",
        "description": "Training using 1, 2, 3 and 4 eruption to forecast 5",
        "train_start_date": "2025-01-01",
        "train_end_date": "2025-06-30",
        "prediction_start_date": "2025-07-01",
        "prediction_end_date": "2025-07-13",
        "window_step": 10,
        "window_step_unit": "minutes",
    },
    {
        "name": "Scenario 8",
        "description": "Training using 1, 2, 3, 4 and 5 eruption to forecast 6, 7",
        "train_start_date": "2025-01-01",
        "train_end_date": "2025-07-26",
        "prediction_start_date": "2025-07-27",
        "prediction_end_date": "2025-08-22",
        "window_step": 10,
        "window_step_unit": "minutes",
        "plot_kwargs": {
            "legend_n_cols": 4,
        },
    },
    {
        "name": "Scenario 9",
        "description": "Prediciton using ALL - 6 Hours",
        "train_start_date": "2025-01-01",
        "train_end_date": "2025-08-22",
        "prediction_start_date": "2025-01-01",
        "prediction_end_date": "2025-08-22",
        "window_step": 6,
        "window_step_unit": "hours",
        "plot_kwargs": {
            "rolling_window": "6h",
            "x_days_interval": 14,
            "legend_n_cols": 4,
        },
    },
    {
        "name": "Scenario 10",
        "description": "Prediciton using ALL - 10 minutes",
        "train_start_date": "2025-01-01",
        "train_end_date": "2025-08-22",
        "prediction_start_date": "2025-01-01",
        "prediction_end_date": "2025-08-22",
        "window_step": 10,
        "window_step_unit": "minutes",
        "plot_kwargs": {
            "rolling_window": "6h",
            "x_days_interval": 14,
            "legend_n_cols": 4,
        },
    },
]


def build_plot_kwargs(scenario: Scenario, eruption_dates: list[str]) -> dict[str, Any]:
    plot_kwargs = dict(scenario.get("plot_kwargs", {}))
    plot_kwargs["eruption_dates"] = eruption_dates
    return plot_kwargs


def main(sds_dir: str, n_jobs: int = 2):
    # %%
    fm = ForecastModel(
        network="VG",
        station="OJN",
        location="00",
        channel="EHZ",
        day_to_forecast=2,
        n_jobs=n_jobs,
        verbose=True,
    )
    # %%
    fm.calculate(
        start_date="2025-01-01",
        end_date="2025-12-31",
        source="sds",
        sds_dir=sds_dir,
        methods=["rsam", "dsar", "entropy"],
        remove_tremor_anomalies=False,
        interpolate=True,
        plot_daily=True,
        save_plot=True,
        overwrite_plot=True,
        overwrite=False,
        n_jobs=n_jobs,
        verbose=False,
    )

    # %%
    for scenario in scenarios:
        name = scenario["name"]
        description = scenario["description"]
        plot_kwargs = build_plot_kwargs(scenario, eruption_dates)

        logger.info("=================================")
        logger.info(f"Running {name}: {description}")
        logger.info("=================================")

        output_dir = os.path.join(
            root_dir, "output", fm.nslc, "scenarios", slugify(name)
        )

        fm.train(
            start_date=scenario["train_start_date"],
            end_date=scenario["train_end_date"],
            classifiers=["lite-rf", "rf", "gb", "xgb"],
            eruption_dates=eruption_dates,
            window_step=6,
            window_step_unit="hours",
            label_builder="standard",
            cv_strategy="shuffle-stratified",
            scoring="recall",
            select_tremor_columns=[
                "rsam_f2",
                "rsam_f3",
                "rsam_f4",
                "dsar_f3-f4",
                "entropy",
            ],
            exclude_features=[
                "agg_linear_trend",
                "linear_trend_timewise",
                "length",
                "has_duplicate_max",
                "has_duplicate_min",
                "has_duplicate",
            ],
            seeds=25,
            resample_method="under",
            plot_features=True,
            output_dir=output_dir,
            n_jobs=4,
            n_grids=4,
            verbose=False,
        )

        fm.predict(
            start_date=scenario["prediction_start_date"],
            end_date=scenario["prediction_end_date"],
            window_step=scenario["window_step"],
            window_step_unit=scenario["window_step_unit"],
            save_seed_result=True,
            plot_threshold=0.7,
            output_dir=output_dir,
            use_cache=False,
            verbose=False,
            **plot_kwargs,
        )

        if fm.PredictionModel and fm.PredictionModel.forecast_plot_path:
            send_telegram_notification(
                message=f"{name}: {description}",
                files=[fm.PredictionModel.forecast_plot_path],
                file_caption=f"[LAPTOP] {name}: {description}",
                send_as_document=True,
            )

        fm.evaluate(
            model="prediction",
            eruption_dates=eruption_dates,
            plot_per_seed=True,
            output_dir=output_dir,
        )

        fm.explain(
            model="prediction",
            eruption_dates=eruption_dates,
            save_per_seed=True,
            plot_per_seed=False,
            plot_aggregate=True,
            max_display=20,
            dpi=150,
            output_dir=output_dir,
        )


# %%
if __name__ == "__main__":
    main(sds_dir=r"D:\Data\OJN", n_jobs=8)
