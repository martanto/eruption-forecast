import os

from eruption_forecast import ForecastModel
from eruption_forecast.decorators import timer


@timer("Forecast Model")
def main(use_relevant_features: bool = False):
    sds_dir = r"D:\Data\OJN"

    params = {
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
        overwrite=False,
        n_jobs=4,
        output_dir=os.path.join(os.getcwd(), "output_12h"),
        **params,
    )

    fm.calculate(
        source="sds",
        sds_dir=sds_dir,
        plot_tmp=True,
        save_plot=True,
        remove_outlier_method="maximum",
    ).build_label(
        start_date="2025-01-01",
        end_date="2025-07-24",
        day_to_forecast=2,
        window_step=12,
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
        classifier="rf",
        random_state=0,
        total_seed=500,
        number_of_significant_features=20,
        sampling_strategy=0.75,
        save_all_features=True,
        plot_significant_features=True,
        output_dir=r"D:\Projects\eruption-forecast\output_12h\VG.OJN.00.EHZ\trainings_relevant_features",
        overwrite=False,
        verbose=True,
    )


if __name__ == "__main__":
    main(use_relevant_features=True)
