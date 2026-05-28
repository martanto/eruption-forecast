# %%
from eruption_forecast.model.forecast import ForecastModel


def main():
    # %%
    n_jobs = 8
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
        sds_dir=r"D:\Data\OJN",
        methods=["rsam", "dsar", "entropy"],
        remove_tremor_anomalies=False,
        interpolate=True,
        plot_daily=False,
        save_plot=False,
        overwrite_plot=False,
        overwrite=False,
        n_jobs=n_jobs,
        verbose=True,
    )
    # %%
    fm.train(
        start_date="2025-01-01",
        end_date="2025-07-26",
        classifiers=["rf", "lite-rf"],
        eruption_dates=[
            "2025-03-20",
            "2025-04-10",
            "2025-04-22",
            "2025-05-18",
            "2025-06-17",
            "2025-07-07",
            "2025-08-02",
            "2025-08-18",
        ],
        window_step=6,
        window_step_unit="hours",
        label_builder="standard",
        cv_strategy="shuffle-stratified",
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
        seeds=10,
        resample_method="under",
        plot_features=True,
        n_jobs=4,
        n_grids=4,
        verbose=True,
    )
    # %%
    fm.predict(
        start_date="2025-07-27",
        end_date="2025-08-22",
        window_step=10,
        window_step_unit="minutes",
        save_seed_result=True,
        plot_threshold=0.7,
    )


# %%
if __name__ == "__main__":
    main()
