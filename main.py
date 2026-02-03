from eruption_forecast import ForecastModel


def main():
    sds_dir = r"D:\Data\OJN"

    params = {
        "station": "OJN",
        "channel": "EHZ",
        "start_date": "2025-03-16",
        "end_date": "2025-03-23",
        "window_size": 2,
        "volcano_id": "Lewotobi Laki-laki",
        "verbose": False,
        "debug": False,
        "n_jobs": 1,
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

    fm = ForecastModel(overwrite=False, **params)

    fm.calculate(sds_dir=sds_dir, remove_outlier_method="maximum").build_label(
        day_to_forecast=2,
        window_step=6,
        window_step_unit="hours",
        eruption_dates=eruptions,
        verbose=False,
    ).build_features(overwrite=False, verbose=False).extract_features(
        use_relevant_features=False,
        overwrite=False,
        tremor_columns=["rsam_f2", "rsam_f3", "rsam_f4", "dsar_f3-f4"],
        exclude_features=[
            "agg_linear_trend",
            "linear_trend_timewise",
            "length",
            "has_duplicate_max",
            "has_duplicate_min",
            "has_duplicate",
        ],
        concat_features=True,
        n_jobs=4,
    )


if __name__ == "__main__":
    main()
