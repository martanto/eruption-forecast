from typing import Any

from eruption_forecast import ForecastModel
from eruption_forecast.decorators import timer


@timer("Forecast Model")
def main(use_relevant_features: bool = False):
    sds_dir = r"D:\Data\OJN"

    params: dict[str, Any] = {
        "station": "OJN",
        "channel": "EHZ",
        "start_date": "2025-01-01",
        "end_date": "2025-12-31",
        "window_size": 2,
        "volcano_id": "Lewotobi Laki-laki",
        "verbose": True,
        "debug": False,
    }

    eruptions = [
        "2025-03-20",
        "2025-04-22",
        "2025-05-18",
        "2025-06-17",
        "2025-07-07",
        "2025-08-01",
        "2025-08-17",
    ]

    fm = ForecastModel(
        root_dir=r"D:\Projects\eruption-forecast",
        overwrite=False,
        n_jobs=4,
        **params,
    )

    fm.calculate(
        source="sds",
        sds_dir=sds_dir,
        plot_daily=True,
        save_plot=True,
        remove_outlier_method="maximum",
    ).build_label(
        start_date="2025-01-01",
        end_date="2025-07-27",
        day_to_forecast=2,
        window_step=6,
        window_step_unit="hours",
        eruption_dates=eruptions,
        verbose=True,
    ).extract_features(
        select_tremor_columns=["rsam_f2", "rsam_f3", "rsam_f4", "dsar_f3-f4"],
        save_tremor_matrix_per_method=True,
        save_tremor_matrix_per_id=False,
        exclude_features=[
            "agg_linear_trend",
            "linear_trend_timewise",
            "length",
            "has_duplicate_max",
            "has_duplicate_min",
            "has_duplicate",
        ],
        use_relevant_features=use_relevant_features,
        overwrite=False,
    ).train(
        classifier=["lite-rf", "rf"],
        cv_strategy="stratified",
        random_state=0,
        total_seed=500,
        with_evaluation=False,
        number_of_significant_features=20,
        sampling_strategy=0.75,
        save_all_features=True,
        plot_significant_features=True,
        overwrite=False,
        verbose=True,
    ).forecast(
        start_date="2025-07-28",
        end_date="2025-08-04",
        window_size=2,
        window_step=10,
        window_step_unit="minutes",
    )


if __name__ == "__main__":
    main(use_relevant_features=True)
