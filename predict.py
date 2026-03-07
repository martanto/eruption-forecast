from eruption_forecast.utils.ml import merge_seed_models
from eruption_forecast.decorators.notify import notify
from eruption_forecast.model.model_predictor import ModelPredictor


@notify("Predict Model")
def predict(trained_models: str):
    predictor = ModelPredictor(
        start_date="2025-07-28",
        end_date="2025-08-20",
        trained_models=trained_models,
        output_dir=r"D:\Projects\eruption-forecast\output",
        n_jobs=6,
    )

    df_forecast = predictor.predict_proba(
        tremor_data=r"D:\Projects\eruption-forecast\output\VG.OJN.00.EHZ\tremor\tremor_VG.OJN.00.EHZ_2025-01-01-2025-09-27.csv",  # or a pd.DataFrame
        select_tremor_columns=[
            "rsam_f2",
            "rsam_f3",
            "rsam_f4",
            "dsar_f3-f4",
            "entropy",
        ],
        window_size=2,
        window_step=10,
        window_step_unit="minutes",
        threshold=0.5,
        plot=True,
    )

    df_forecast.to_csv(r"D:\Projects\eruption-forecast\output\anto_test.csv")


if __name__ == "__main__":
    _trained_models = merge_seed_models(
        r"D:\Projects\eruption-forecast\output\VG.OJN.00.EHZ\trainings\model-only\random-forest-classifier\stratified-k-fold\trained_model_RandomForestClassifier-StratifiedKFold_rs-0_ts-500_top-20.csv",
        output_dir=r"D:\Projects\eruption-forecast\output",
    )
    predict(_trained_models)
