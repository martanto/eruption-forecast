# %%
import os
from typing import Any, Literal, TypedDict, NotRequired

from dotenv import load_dotenv

from eruption_forecast import TelegramNotification
from eruption_forecast.logger import logger
from eruption_forecast.decorators import timer
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

root_dir = r"D:\Anto\Codes\eruption-forecast"
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


scenarios_new: list[Scenario] = [
    {
        "name": "Scenario 1.1",
        "description": "Training 1 eruption. Forecast: 2,3,4,5,6",
        "train_start_date": "2025-01-01",
        "train_end_date": "2025-03-31",
        "prediction_start_date": "2025-01-01",
        "prediction_end_date": "2025-08-22",
        "window_step": 10,
        "window_step_unit": "minutes",
        "plot_kwargs": {
            "rolling_window": "6h",
            "x_days_interval": 14,
            "legend_n_cols": 6,
        },
    },
    {
        "name": "Scenario 2.1",
        "description": "Training 1,2 eruption. Forecast: 3,4,5,6",
        "train_start_date": "2025-01-01",
        "train_end_date": "2025-04-30",
        "prediction_start_date": "2025-01-01",
        "prediction_end_date": "2025-08-22",
        "window_step": 10,
        "window_step_unit": "minutes",
        "plot_kwargs": {
            "rolling_window": "6h",
            "x_days_interval": 14,
            "legend_n_cols": 6,
        },
    },
    {
        "name": "Scenario 3.1",
        "description": "Training 1,2,3 eruption. Forecast: 4,5,6",
        "train_start_date": "2025-01-01",
        "train_end_date": "2025-05-31",
        "prediction_start_date": "2025-01-01",
        "prediction_end_date": "2025-08-22",
        "window_step": 10,
        "window_step_unit": "minutes",
        "plot_kwargs": {
            "rolling_window": "6h",
            "x_days_interval": 14,
            "legend_n_cols": 6,
        },
    },
    {
        "name": "Scenario 4.1",
        "description": "Training 1,2,3,4 eruption. Forecast: 5,6",
        "train_start_date": "2025-01-01",
        "train_end_date": "2025-06-30",
        "prediction_start_date": "2025-01-01",
        "prediction_end_date": "2025-08-22",
        "window_step": 10,
        "window_step_unit": "minutes",
        "plot_kwargs": {
            "rolling_window": "6h",
            "x_days_interval": 14,
            "legend_n_cols": 6,
        },
    },
    {
        "name": "Scenario 5.1",
        "description": "Training 1,2,3,4,5 eruption. Forecast: 6",
        "train_start_date": "2025-01-01",
        "train_end_date": "2025-07-26",
        "prediction_start_date": "2025-01-01",
        "prediction_end_date": "2025-08-22",
        "window_step": 10,
        "window_step_unit": "minutes",
        "plot_kwargs": {
            "rolling_window": "6h",
            "x_days_interval": 14,
            "legend_n_cols": 6,
        },
    },
]


scenarios: list[Scenario] = [
    {
        "name": "Scenario 1",
        "description": "Training 1 eruption. Forecast: 2",
        "train_start_date": "2025-01-01",
        "train_end_date": "2025-03-31",
        "prediction_start_date": "2025-04-01",
        "prediction_end_date": "2025-04-30",
        "window_step": 10,
        "window_step_unit": "minutes",
    },
    {
        "name": "Scenario 2",
        "description": "Training 1,2 eruption. Forecast: 3",
        "train_start_date": "2025-01-01",
        "train_end_date": "2025-04-30",
        "prediction_start_date": "2025-05-01",
        "prediction_end_date": "2025-05-31",
        "window_step": 10,
        "window_step_unit": "minutes",
    },
    {
        "name": "Scenario 3",
        "description": "Training 1,2,3 eruption. Forecast: 4",
        "train_start_date": "2025-01-01",
        "train_end_date": "2025-05-31",
        "prediction_start_date": "2025-06-01",
        "prediction_end_date": "2025-06-30",
        "window_step": 10,
        "window_step_unit": "minutes",
    },
    {
        "name": "Scenario 4",
        "description": "Training 1,2,3,4 eruption. Forecast: 5",
        "train_start_date": "2025-01-01",
        "train_end_date": "2025-06-30",
        "prediction_start_date": "2025-07-01",
        "prediction_end_date": "2025-07-31",
        "window_step": 10,
        "window_step_unit": "minutes",
    },
    {
        "name": "Scenario 5",
        "description": "Training 1,2,3,4,5 eruption. Forecast: 6",
        "train_start_date": "2025-01-01",
        "train_end_date": "2025-07-26",
        "prediction_start_date": "2025-08-01",
        "prediction_end_date": "2025-08-22",
        "window_step": 10,
        "window_step_unit": "minutes",
    },
]


def build_plot_kwargs(scenario: Scenario, eruption_dates: list[str]) -> dict[str, Any]:
    plot_kwargs = dict(scenario.get("plot_kwargs", {}))
    plot_kwargs["eruption_dates"] = eruption_dates
    return plot_kwargs


# %%
@timer("Run Single Prediction")
def predict(
    fm: ForecastModel,
    start_date: str,
    end_date: str,
    name: str,
    description: str,
    window_step=10,
    window_step_unit="minutes",
    save_seed_result=True,
    plot_threshold=0.7,
    use_features_from="all",
    features_matrix_path: str | None = None,
    label_features_csv: str | None = None,
    output_dir: str | None = None,
    use_cache=False,
    verbose=True,
    **plot_kwargs,
) -> ForecastModel:

    fm.predict(
        start_date=start_date,
        end_date=end_date,
        window_step=window_step,
        window_step_unit=window_step_unit,
        save_seed_result=save_seed_result,
        plot_threshold=plot_threshold,
        use_features_from=use_features_from,
        features_matrix_path=features_matrix_path,
        label_features_csv=label_features_csv,
        output_dir=output_dir,
        use_cache=use_cache,
        verbose=verbose,
        **plot_kwargs,
    )

    if fm.PredictionModel and fm.PredictionModel.forecast_plot_path:
        tn = TelegramNotification(verbose=True)
        tn.send_document(
            fm.PredictionModel.forecast_plot_path,
            caption=f"[AGS-WS1-OJN-OLD] {name}: {description}",
        )

    return fm


@timer("Run All Scenarios")
def main(sds_dir: str, n_jobs: int = 2):

    features_matrix_path = None
    label_features_csv = None

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
        minimum_completion_ratio=0.3,
        plot_eruption_dates=eruption_dates,
        plot_rsam_as_log=True,
        plot_rolling_window="2D",
        plot_filter_dsar_value=20,
        plot_overwrite=False,
        overwrite=False,
        n_jobs=n_jobs,
        verbose=True,
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
            root_dir, "output", fm.nslc, "scenarios-old", slugify(name)
        )

        fm.prefix_config = slugify(name)

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
            seeds=100,
            resample_method="under",
            plot_features=True,
            output_dir=output_dir,
            n_jobs=15,
            n_grids=4,
            verbose=False,
        )

        fm = predict(
            fm=fm,
            start_date=scenario["prediction_start_date"],
            end_date=scenario["prediction_end_date"],
            name=name,
            description=description,
            window_step=scenario["window_step"],
            window_step_unit=scenario["window_step_unit"],
            save_seed_result=True,
            plot_threshold=0.7,
            enable_segments_plot=True,
            use_features_from="training",
            features_matrix_path=features_matrix_path,
            label_features_csv=label_features_csv,
            output_dir=output_dir,
            use_cache=False,
            verbose=False,
            **plot_kwargs,
        )

        if features_matrix_path is None and label_features_csv is None:
            logger.info("Using features and labels CSV")
            features_matrix_path = fm.PredictionModel.features_path
            label_features_csv = fm.PredictionModel.labels_csv

        fm.evaluate(
            model="prediction",
            plot_per_seed=True,
            output_dir=output_dir,
            use_cache=True,
            verbose=True,
        )

        fm.explain(
            model="prediction",
            eruption_dates=eruption_dates,
            save_per_seed=True,
            plot_per_seed=False,
            plot_aggregate=True,
            max_display=20,
            dpi=150,
            overwrite=False,
            output_dir=output_dir,
            use_cache=True,
            verbose=True,
        )


# %%
if __name__ == "__main__":
    main(sds_dir=r"D:\Anto\Data\SDS", n_jobs=60)
