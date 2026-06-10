# %%
from eruption_forecast.model.forecast import ForecastModel


def main():
    # %%
    n_jobs = 8
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
        sds_dir=r"D:\Data\LEKR",
        methods=["rsam", "dsar", "entropy"],
        remove_tremor_anomalies=False,
        interpolate=True,
        plot_daily=True,
        save_plot=True,
        overwrite_plot=True,
        overwrite=False,
        n_jobs=n_jobs,
        verbose=True,
    )
    # %%
    fm.train(
        start_date="2020-08-01",
        end_date="2021-12-06",
        classifiers=["lite-rf", "rf", "gb", "xgb"],
        eruption_dates=[
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
            "2022-12-04",
            "2022-12-05",
        ],
        window_step=6,
        window_step_unit="hours",
        label_builder="dynamic",
        days_before_eruption=60,
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
        seeds=25,
        resample_method="under",
        plot_features=True,
        n_jobs=4,
        n_grids=4,
        verbose=True,
    )
    # %%
    fm.predict(
        start_date="2022-11-29",
        end_date="2022-12-07",
        window_step=10,
        window_step_unit="minutes",
        save_seed_result=True,
        plot_threshold=0.7,
        use_cache=True,
        verbose=True,
    )

    fm.evaluate(model="prediction", plot_per_seed=True)

    # %%
    if fm.EvaluationModel:
        comparator = fm.EvaluationModel.compare()
        comparator.get_ranking()
        comparator.plot_all()


# %%
if __name__ == "__main__":
    main()
