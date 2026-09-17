from eruption_forecast.model.prediction_model import PredictionModel
from eruption_forecast.tremor.calculate_tremor import CalculateTremor


def main(
    model: str,
    tremor_start_date: str,
    tremor_end_date: str,
    prediction_start_date: str,
    prediction_end_date: str,
    plot_prediction_start_date: str,
    plot_prediction_end_date: str,
    sds_dir: str,
):
    tremor_df = (
        CalculateTremor(
            start_date=tremor_start_date,
            end_date=tremor_end_date,
            station="RUA3",
            channel="EHZ",
            network="VG",
            location="00",
            methods=["rsam", "dsar", "entropy"],
            remove_tremor_anomalies=False,
            interpolate=True,
            plot_daily=True,
            save_plot=True,
            minimum_completion_ratio=0.3,
            plot_overwrite=False,
            overwrite=False,
            n_jobs=8,
            verbose=True,
        )
        .from_sds(sds_dir=sds_dir)
        .run()
        .df
    )

    pm = PredictionModel(
        model=model,
        tremor_data=tremor_df,
        start_date=prediction_start_date,
        end_date=prediction_end_date,
        window_size=2,
        output_dir=r"D:\Projects\eruption-forecast\output\VG.RUA3.00.EHZ",
        prefix_config="rua-3",
        n_jobs=8,
        verbose=True,
    )

    (
        pm.build_label(
            window_step=10,
            window_step_unit="minutes",
        )
        .extract_features(
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
        )
        .forecast(
            save_seed_result=True,
            plot_threshold=0.7,
            x_days_interval=1,
            plot_start_date=plot_prediction_start_date,
            plot_end_date=plot_prediction_end_date,
            eruption_dates=["2024-04-17"],
        )
    )


if __name__ == "__main__":
    main(
        model=r"D:\Projects\eruption-forecast\output\VG.OJN.00.EHZ\scenarios-old\scenario-5\training\classifiers\ClassifierEnsemble_StratifiedShuffleSplit.pkl",
        tremor_start_date="2024-03-15",
        tremor_end_date="2024-04-30",
        prediction_start_date="2024-04-01",
        prediction_end_date="2024-04-30",
        plot_prediction_start_date="2024-04-11",
        plot_prediction_end_date="2024-04-18",
        sds_dir=r"D:\Data\RUA3",
    )
