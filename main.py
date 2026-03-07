import os
from typing import Any

from dotenv import load_dotenv

from eruption_forecast import ForecastModel
from eruption_forecast.decorators import timer, notify
from eruption_forecast.model.classifier_comparator import ClassifierComparator
from eruption_forecast.model.multi_model_evaluator import MultiModelEvaluator


load_dotenv()
DEBUG = os.environ.get("DEBUG", "false").lower() in ("true", "1", "yes")

ROOT_DIR = r"D:\Projects\eruption-forecast"
SDS_DIR = r"D:\Data\OJN"
N_JOBS = 6
CLASSIFIER = ["lite-rf", "rf"] if DEBUG else ["lite-rf", "rf", "gb", "xgb"]
TRAINING_SEEDS = 5 if DEBUG else 500

ERUPTION_DATES = [
    "2025-03-20",
    "2025-04-22",
    "2025-05-18",
    "2025-06-17",
    "2025-07-07",
    "2025-08-01",
    "2025-08-17",
]

FORECAST_PARAMETERS: dict[str, Any] = {
    "root_dir": ROOT_DIR,
    "start_date": "2025-01-01",  # tremor calculation start date
    "end_date": "2025-12-31",  # tremor calculation end date
    "station": "OJN",
    "channel": "EHZ",
    "network": "VG",
    "location": "00",
    "window_size": 2,  # days to calculate tremor
    "volcano_id": "Lewotobi Laki-laki",  # required
    "n_jobs": N_JOBS,
    "verbose": True,  # show more informations
    "overwrite": False,
    "debug": False,
}

LABEL_PARAMETERS: dict[str, Any] = {
    "day_to_forecast": 2,  # days before eruptions
    "window_step": 6,
    "window_step_unit": "hours",
    "eruption_dates": ERUPTION_DATES,
    "verbose": True,  # show more informations
}

EXTRACT_FEATURES_PARAMETERS: dict[str, Any] = {
    "select_tremor_columns": ["rsam_f2", "rsam_f3", "rsam_f4", "dsar_f3-f4"],
    "save_tremor_matrix_per_method": True,
    "save_tremor_matrix_per_id": False,
    "exclude_features": [
        "agg_linear_trend",
        "linear_trend_timewise",
        "length",
        "has_duplicate_max",
        "has_duplicate_min",
        "has_duplicate",
    ],
    "use_relevant_features": True,
    "overwrite": False,
}

TRAINING_START_DATE = "2025-01-01"
TRAINING_END_DATE = "2025-08-24"
TRAINING_PARAMETERS: dict[str, Any] = {
    "classifier": CLASSIFIER,
    "cv_strategy": "stratified",
    "random_state": 0,
    "total_seed": TRAINING_SEEDS,
    "number_of_significant_features": 20,
    "sampling_strategy": 0.75,
    "save_all_features": True,
    "plot_significant_features": True,
    "n_jobs": 4,
    "grid_search_n_jobs": 1,
    "overwrite": False,
    "plot_shap": False,  # SHAP Explanation plot
    "verbose": True,
}

TRAINING_PREDICTION_START_DATE = "2025-01-01"
TRAINING_PREDICTION_END_DATE = "2025-07-27"
PREDICTION_PARAMETERS: dict[str, Any] = {
    "start_date": "2025-07-28",
    "end_date": "2025-08-20",
    "window_step": 10,
    "window_step_unit": "minutes",
}


def train_and_evaluate(forecast_model: ForecastModel) -> None:
    forecast_model.build_label(
        start_date=TRAINING_START_DATE, end_date=TRAINING_END_DATE, **LABEL_PARAMETERS
    ).extract_features(**EXTRACT_FEATURES_PARAMETERS).train(
        with_evaluation=True, **TRAINING_PARAMETERS
    )

    # Evaluate model per seed
    if not forecast_model.trained_models:
        print(
            "[workflow] Skipping Stage 2: no trained_models available. "
            "Run Stage 2 first or load a ForecastModel with trained_models set."
        )
    else:
        print("[workflow] Stage 3: Evaluating")
        for name, csv_path in forecast_model.trained_models.items():
            print(f"    [workflow] Evaluating '{name}'")
            csv_dir = os.path.dirname(os.path.abspath(csv_path))
            metrics_dir: str | None = os.path.join(csv_dir, "metrics")

            if not os.path.isdir(metrics_dir):
                metrics_dir = None

            evaluator = MultiModelEvaluator(
                trained_model_csv=csv_path,
                metrics_dir=metrics_dir,
                classifier_name=name,
                output_dir=csv_dir,
                plot_shap=False,
            )

            if metrics_dir is not None:
                evaluator.get_aggregate_metrics()

            evaluator.plot_all()

    # Evaluate per classifier
    if len(forecast_model.trained_models) < 2:
        print(
            "[workflow] Skipping Stage 4: comparison requires at least 2 classifiers "
            f"(found {len(forecast_model.trained_models)})."
        )
    else:
        print("[workflow] Stage 4: comparing classifiers")
        comparator = ClassifierComparator(
            classifiers=forecast_model.trained_models,
            output_dir=forecast_model.station_dir,
        )
        comparator.plot_all()


def predict(forecast_model: ForecastModel) -> None:
    forecast_model.build_label(
        start_date=TRAINING_PREDICTION_START_DATE,
        end_date=TRAINING_PREDICTION_END_DATE,
        **LABEL_PARAMETERS,
    ).extract_features(**EXTRACT_FEATURES_PARAMETERS).train(
        with_evaluation=False, **TRAINING_PARAMETERS
    ).forecast(**PREDICTION_PARAMETERS)


@timer("Forecast Model")
@notify("Primer - Workflow")
def main():
    fm = ForecastModel(**FORECAST_PARAMETERS)

    # Calculate Tremor
    fm.calculate(
        source="sds",
        sds_dir=SDS_DIR,
        plot_daily=True,
        save_plot=True,
        remove_outlier_method="maximum",
    )

    # Train and evaluate
    train_and_evaluate(forecast_model=fm)

    # Predict
    predict(forecast_model=fm)


if __name__ == "__main__":
    main()
