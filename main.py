# %%
from eruption_forecast.decorators import timer, notify
from eruption_forecast.model.forecast_model import ForecastModel


@timer("Run Forecasting")
@notify("Run Forecasting")
def main(sds_dir: str, n_jobs: int = 2):
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
        overwrite_plot=True,
        overwrite=False,
        n_jobs=n_jobs,
        verbose=False,
    )
    # %%
    fm.train(
        start_date="2025-01-01",
        end_date="2025-07-26",
        classifiers=["lite-rf", "rf", "gb", "xgb"],
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
        seeds=25,
        resample_method="under",
        plot_features=True,
        n_jobs=4,
        n_grids=4,
        verbose=False,
    )
    # %%
    fm.predict(
        start_date="2025-07-27",
        end_date="2025-08-22",
        window_step=10,
        window_step_unit="minutes",
        save_seed_result=True,
        plot_threshold=0.7,
        use_cache=False,
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
    main(sds_dir=r"D:\Data\OJN", n_jobs=8)
