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
    "2020-11-29",
    "2020-11-30",
    "2020-12-01",
    "2020-12-02",
    "2021-12-01",
    "2021-12-04",
    "2021-12-05",
    "2021-12-06",
    "2022-12-04",
    "2022-12-05",
]


scenarios_old: list[Scenario] = [
    {
        "name": "Semeru Scenario 1",
        "description": "Training 2020 eruption. Forecast: 2021",
        "train_start_date": "2020-08-10",
        "train_end_date": "2020-12-02",
        "prediction_start_date": "2021-11-29",
        "prediction_end_date": "2021-12-06",
        "window_step": 10,
        "window_step_unit": "minutes",
    },
    {
        "name": "Semeru Scenario 2",
        "description": "Training 2020 eruption. Forecast: 2022",
        "train_start_date": "2020-08-10",
        "train_end_date": "2020-12-02",
        "prediction_start_date": "2022-11-29",
        "prediction_end_date": "2022-12-06",
        "window_step": 10,
        "window_step_unit": "minutes",
    },
    {
        "name": "Semeru Scenario 3",
        "description": "Training 2020 and 2021 eruption. Forecast: 2022",
        "train_start_date": "2020-08-10",
        "train_end_date": "2021-12-06",
        "prediction_start_date": "2022-11-29",
        "prediction_end_date": "2022-12-06",
        "window_step": 10,
        "window_step_unit": "minutes",
    },
]

scenarios: list[Scenario] = [
    {
        "name": "Trained 2020 forecast 2021",
        "description": "Training 2020 eruption. Forecast: 2021",
        "train_start_date": "2020-08-10",
        "train_end_date": "2020-12-02",
        "prediction_start_date": "2021-01-01",
        "prediction_end_date": "2021-12-31",
        "window_step": 10,
        "window_step_unit": "minutes",
    },
    {
        "name": "Trained 2020 forecast 2022",
        "description": "Training 2020 eruption. Forecast: 2022",
        "train_start_date": "2020-08-10",
        "train_end_date": "2020-12-02",
        "prediction_start_date": "2022-01-01",
        "prediction_end_date": "2022-12-31",
        "window_step": 10,
        "window_step_unit": "minutes",
    },
    {
        "name": "Trained 2020 and 2021 forecast 2022",
        "description": "Training 2020 and 2021 eruption. Forecast: 2022",
        "train_start_date": "2020-08-10",
        "train_end_date": "2021-12-06",
        "prediction_start_date": "2022-01-01",
        "prediction_end_date": "2022-12-31",
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
            caption=f"[AGS-WS1] {name}: {description}",
        )

    return fm


@timer("Run All Scenarios")
def main(sds_dir: str, n_jobs: int = 2):

    features_matrix_path = None
    label_features_csv = None

    # %%
    fm = ForecastModel(
        network="VG",
        station="LEKR",
        location="00",
        channel="EHZ",
        day_to_forecast=2,
        n_jobs=n_jobs,
        verbose=True,
    )
    # %%
    fm.calculate(
        start_date="2020-08-01",
        end_date="2022-12-31",
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
            root_dir, "output", fm.nslc, "scenarios", slugify(name)
        )

        fm.prefix_config = slugify(name)

        fm.train(
            start_date=scenario["train_start_date"],
            end_date=scenario["train_end_date"],
            classifiers=["lite-rf", "rf", "gb", "xgb"],
            eruption_dates=eruption_dates,
            window_step=6,
            window_step_unit="hours",
            label_builder="dynamic",
            days_before_eruption=60,
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
            verbose=True,
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
            features_matrix_path=features_matrix_path,
            label_features_csv=label_features_csv,
            output_dir=output_dir,
            use_cache=False,
            verbose=True,
            **plot_kwargs,
        )

        # if features_matrix_path is None and label_features_csv is None:
        #     logger.info("Using features and labels CSV")
        #     features_matrix_path = fm.PredictionModel.features_path
        #     label_features_csv = fm.PredictionModel.labels_csv

        fm.evaluate(
            model="prediction",
            plot_per_seed=True,
            output_dir=output_dir,
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
            verbose=True,
        )


# %%
if __name__ == "__main__":
    main(sds_dir=r"D:\Anto\Data\SDS", n_jobs=60)
