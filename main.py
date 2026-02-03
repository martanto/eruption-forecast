from eruption_forecast import ForecastModel
from eruption_forecast.model.build_model import TrainModel

def main():
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
        start_date="2025-01-01",
        end_date="2025-07-24",
        day_to_forecast=2,
        window_step=6,
        window_step_unit="hours",
        eruption_dates=eruptions,
        verbose=True,
    ).build_features(overwrite=False, verbose=True).extract_features(
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

def train():
    features_csv = r"D:\Projects\eruption-forecast\output\VG.OJN.00.EHZ\features\extracted_features_2025-01-01-2025-09-28.csv"
    label_csv = r"D:\Projects\eruption-forecast\output\VG.OJN.00.EHZ\features\label_2025-01-01-2025-09-28.csv"

    train_model = TrainModel(
        features_csv=features_csv,
        label_csv=label_csv,
        output_dir=r"D:\Projects\eruption-forecast\output\VG.OJN.00.EHZ\predictions",
        overwrite=False,
        verbose=False,
        n_jobs=4,
    )

    train_model.train(
        sampling_strategy=0.75,
        random_state=0,
        total_seed=500,
        number_of_significant_features=20,
        save_features=True,
        plot_features=True,
        overwrite=False,
        verbose=False,
    )


if __name__ == "__main__":
    # main()
    train()
