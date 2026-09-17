from dotenv import load_dotenv

from eruption_forecast.model.forecast_model_scenario import (
    Scenario,
    ForecastModelScenario,
)


load_dotenv(override=True)


def main(
    network: str,
    station: str,
    location: str,
    channel: str,
    eruption_dates: list[str],
    scenarios: list[Scenario],
    sds_dir: str,
    day_to_forecast: int,
    n_jobs: int = 1,
    verbose: bool = True,
) -> None:

    # %%
    fms = ForecastModelScenario(
        station=station,
        location=location,
        channel=channel,
        network=network,
        eruption_dates=eruption_dates,
        scenarios=scenarios,
        day_to_forecast=day_to_forecast,
        output_dir="output-test",
        n_jobs=n_jobs,
        verbose=verbose,
    )

    # %%
    fms.calculate(
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
        plot_rsam_as_log=True,
        plot_rolling_window="2D",
        plot_filter_dsar_value=20,
        plot_overwrite=False,
        overwrite=False,
        n_jobs=n_jobs,
        verbose=False,
    )

    # %%
    fms.shared_training(
        window_step=6,
        window_step_unit="hours",
        label_builder="standard",
        # classifiers=["lite-rf", "rf", "gb", "xgb"],
        classifiers=["lite-rf", "rf"],
        cv_strategy="shuffle-stratified",
        cv_splits=5,
        scoring="recall",
        top_n_features=20,
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
        seeds=2,
        resample_method="under",
        plot_features=True,
        n_jobs=2,
        n_grids=4,
        verbose=False,
    )

    # %%
    fms.shared_prediction(
        save_seed_result=True,
        plot_threshold=0.7,
        use_features_from="files",
        enable_segments_plot=False,
        use_cache=True,
        verbose=True,
    )

    # %%
    fms.shared_evaluation(
        model="prediction",
        plot_per_seed=True,
        plot_aggregate=True,
    )

    # %%
    fms.shared_explanation(
        model="prediction",
        save_per_seed=True,
        plot_per_seed=True,
        plot_aggregate=True,
        max_display=20,
        dpi=150,
    )

    # %%
    fms.run()


if __name__ == "__main__":
    scenarios: list[Scenario] = [
        {
            "name": "Scenario 1.2",
            "description": "Training 1 eruption. Forecast: 2,3,4,5,6",
            "train_start_date": "2025-01-01",
            "train_end_date": "2025-03-31",
            "prediction_start_date": "2025-04-01",
            "prediction_end_date": "2025-08-22",
            "prediction_window_step": 10,
            "prediction_window_step_unit": "minutes",
            "prediction_plot_kwargs": {
                "rolling_window": "6h",
                "x_days_interval": 14,
                "legend_n_cols": 6,
            },
        },
        {
            "name": "Scenario 2.2",
            "description": "Training 1,2 eruption. Forecast: 3,4,5,6",
            "train_start_date": "2025-01-01",
            "train_end_date": "2025-04-30",
            "prediction_start_date": "2025-05-01",
            "prediction_end_date": "2025-08-22",
            "prediction_window_step": 10,
            "prediction_window_step_unit": "minutes",
            "prediction_plot_kwargs": {
                "rolling_window": "6h",
                "x_days_interval": 14,
                "legend_n_cols": 6,
            },
        },
        {
            "name": "Scenario 3.2",
            "description": "Training 1,2,3 eruption. Forecast: 4,5,6",
            "train_start_date": "2025-01-01",
            "train_end_date": "2025-05-31",
            "prediction_start_date": "2025-06-01",
            "prediction_end_date": "2025-08-22",
            "prediction_window_step": 10,
            "prediction_window_step_unit": "minutes",
            "prediction_plot_kwargs": {
                "rolling_window": "6h",
                "x_days_interval": 14,
                "legend_n_cols": 6,
            },
        },
        {
            "name": "Scenario 4.2",
            "description": "Training 1,2,3,4 eruption. Forecast: 5,6",
            "train_start_date": "2025-01-01",
            "train_end_date": "2025-06-30",
            "prediction_start_date": "2025-07-01",
            "prediction_end_date": "2025-08-22",
            "prediction_window_step": 10,
            "prediction_window_step_unit": "minutes",
            "prediction_plot_kwargs": {
                "rolling_window": "6h",
                "x_days_interval": 14,
                "legend_n_cols": 6,
            },
        },
        {
            "name": "Scenario 5.2",
            "description": "Training 1,2,3,4,5 eruption. Forecast: 6",
            "train_start_date": "2025-01-01",
            "train_end_date": "2025-07-31",
            "prediction_start_date": "2025-08-01",
            "prediction_end_date": "2025-08-22",
            "prediction_window_step": 10,
            "prediction_window_step_unit": "minutes",
            "prediction_plot_kwargs": {
                "rolling_window": "6h",
                "x_days_interval": 14,
                "legend_n_cols": 6,
            },
        },
    ]

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

    main(
        scenarios=scenarios,
        network="VG",
        station="OJN",
        location="00",
        channel="EHZ",
        eruption_dates=eruption_dates,
        day_to_forecast=2,
        sds_dir=r"D:\Data\OJN",
        n_jobs=8,
        verbose=True,
    )
