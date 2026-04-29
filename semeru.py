import os
from typing import Any

from dotenv import load_dotenv

from eruption_forecast import ForecastModel, send_telegram_notification
from eruption_forecast.logger import logger
from eruption_forecast.decorators import timer, notify


load_dotenv(override=True)
DEBUG = os.environ.get("DEBUG", "false").lower() in ("true", "1", "yes")

ROOT_DIR = r"D:\Projects\eruption-forecast"
N_JOBS = 8
CALCULATE_START_DATE = "2020-08-01"
CALCULATE_END_DATE = "2022-12-31"
CLASSIFIER = ["lite-rf", "rf"] if DEBUG else ["lite-rf", "rf", "gb", "xgb"]
TRAINING_SEEDS = 25 if DEBUG else 500


def main(root_dir=None, output_dir: str | None = None):
    sds_dir = r"D:\Data\LEKR"

    params: dict[str, Any] = {
        "station": "LEKR",
        "channel": "EHZ",
        "network": "VG",
        "location": "00",
        "window_size": 2,
        "volcano_id": "Semeru",
        "n_jobs": N_JOBS,
        "verbose": True,
    }

    eruptions = [
        "2020-11-29",
        "2020-11-30",
        "2020-12-01",
        "2020-12-02",
        "2020-12-23",
        "2020-12-30",
        "2021-12-01",
        "2021-12-04",
        "2021-12-05",
        "2021-12-06",
        # "2021-12-16",
        # "2021-12-18",
        # "2021-12-20",
        # "2021-12-22",
        # "2021-12-31",
        # "2022-11-09",
        # "2022-12-04",
        # "2022-12-05",
    ]

    fm = ForecastModel(root_dir=root_dir or ROOT_DIR, **params)

    fm.calculate(
        start_date=CALCULATE_START_DATE,
        end_date=CALCULATE_END_DATE,
        source="sds",
        sds_dir=sds_dir,
        plot_daily=False,
        save_plot=True,
        remove_outlier_method="maximum",
        verbose=False,
    ).build_label(
        builder="dynamic",
        days_before_eruption=60,
        day_to_forecast=2,
        window_step=6,
        window_step_unit="hours",
        eruption_dates=eruptions,
        include_eruption_date=True,
        verbose=True,
    ).extract_features(
        select_tremor_columns=[
            "rsam_f2",
            "rsam_f3",
            "rsam_f4",
            "dsar_f3-f4",
            "entropy",
        ],
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
        use_relevant_features=True,
        overwrite=False,
    ).train(
        classifier=CLASSIFIER,
        cv_strategy="shuffle-stratified",
        random_state=0,
        total_seed=TRAINING_SEEDS,
        number_of_significant_features=20,
        sampling_strategy=0.75,
        save_all_features=False,
        plot_significant_features=False,
        output_dir=output_dir,
        n_jobs=4 if N_JOBS > 1 else 1,
        grid_search_n_jobs=4,
        resample_method="under",
        with_evaluation=False,
        verbose=True,
    ).forecast(
        start_date="2022-11-28",
        end_date="2022-12-07",
        window_step=10,
        window_step_unit="minutes",
        plot_pdf=True,
        eruption_dates=[
            # "2022-11-09",
            "2022-12-04",
            "2022-12-05",
        ],
        output_dir=output_dir,
        x_days_interval=1,
    )


if __name__ == "__main__":
    main()
