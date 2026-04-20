import os
from typing import Any

from dotenv import load_dotenv

from eruption_forecast import ForecastModel, send_telegram_notification
from eruption_forecast.logger import logger
from eruption_forecast.decorators import timer, notify
from eruption_forecast.utils.formatting import slugify
from eruption_forecast.model.classifier_comparator import ClassifierComparator
from eruption_forecast.model.multi_model_evaluator import MultiModelEvaluator


load_dotenv(override=True)
DEBUG = os.environ.get("DEBUG", "false").lower() in ("true", "1", "yes")

ROOT_DIR = r"D:\Projects\eruption-forecast"
SDS_DIR = r"D:\Data\OJN"
N_JOBS = 8
CLASSIFIER = ["lite-rf", "rf"] if DEBUG else ["lite-rf", "rf", "gb", "xgb"]
TRAINING_SEEDS = 25 if DEBUG else 500

ERUPTION_DATES = [
    "2025-03-20",
    "2025-04-10",
    "2025-04-22",
    "2025-05-18",
    "2025-06-17",
    "2025-07-07",
    "2025-08-02",
    "2025-08-18",
]

CALCULATE_START_DATE = "2025-01-01"
CALCULATE_END_DATE = "2025-12-31"

FORECAST_PARAMETERS: dict[str, Any] = {
    "station": "OJN",
    "channel": "EHZ",
    "network": "VG",
    "location": "00",
    "window_size": 2,  # days to calculate tremor
    "volcano_id": "Lewotobi Laki-laki",  # required
    "n_jobs": N_JOBS,
    "verbose": True,  # show more information
    "overwrite": False,
    "debug": False,
}

LABEL_PARAMETERS: dict[str, Any] = {
    "day_to_forecast": 2,  # days before eruptions
    "window_step": 6,
    "window_step_unit": "hours",
    "eruption_dates": ERUPTION_DATES,
    "include_eruption_date": True,
    "verbose": True,  # show more information
}

EXTRACT_FEATURES_PARAMETERS: dict[str, Any] = {
    "select_tremor_columns": ["rsam_f2", "rsam_f3", "rsam_f4", "dsar_f3-f4", "entropy"],
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

EVALUATION_START_DATE = "2025-01-01"
EVALUATION_END_DATE = "2025-07-26"
TRAINING_PARAMETERS: dict[str, Any] = {
    "classifier": CLASSIFIER,
    "cv_strategy": "shuffle-stratified",
    "random_state": 0,
    "total_seed": TRAINING_SEEDS,
    "number_of_significant_features": 20,
    "sampling_strategy": 0.75,
    "resample_method": "under",
    "save_all_features": False,
    "plot_significant_features": False,
    "n_jobs": 4 if N_JOBS > 1 else 1,
    "grid_search_n_jobs": 4,
    "overwrite": False,
    "plot_shap": False,  # SHAP Explanation plot
    "verbose": True,
}

TRAINING_PREDICTION_START_DATE = "2025-01-01"
TRAINING_PREDICTION_END_DATE = "2025-07-26"

PREDICTION_START_DATE = "2025-07-27"
PREDICTION_END_DATE = "2025-08-22"
PREDICTION_PARAMETERS: dict[str, Any] = {
    "window_step": 10,
    "window_step_unit": "minutes",
}


@timer("Run Multiple Scenarios")
@notify("Run Multiple Scenarios")
def scenarios(forecast_model: ForecastModel) -> None:
    _scenarios = [
        # {
        #     "name": "Scenario 1",
        #     "description": "Training using 1 eruption to forecast eruption 1",
        #     "train_start_date": "2025-01-01",
        #     "train_end_date": "2025-03-31",
        #     "prediction_start_date": "2025-04-01",
        #     "prediction_end_date": "2025-04-30",
        # },
        # {
        #     "name": "Scenario 2",
        #     "description": "Training using 1 eruption to forecast eruption 3",
        #     "train_start_date": "2025-01-01",
        #     "train_end_date": "2025-03-31",
        #     "prediction_start_date": "2025-05-01",
        #     "prediction_end_date": "2025-05-31",
        # },
        # {
        #     "name": "Scenario 3",
        #     "description": "Training using 1 eruption to forecast eruption 4",
        #     "train_start_date": "2025-01-01",
        #     "train_end_date": "2025-03-31",
        #     "prediction_start_date": "2025-06-01",
        #     "prediction_end_date": "2025-06-30",
        # },
        # {
        #     "name": "Scenario 5",
        #     "description": "Training using 1 and 2 eruptions to forecast eruption 3",
        #     "train_start_date": "2025-01-01",
        #     "train_end_date": "2025-04-30",
        #     "prediction_start_date": "2025-05-01",
        #     "prediction_end_date": "2025-05-31",
        # },
        # {
        #     "name": "Scenario 6",
        #     "description": "Training using 1, 2 and 3 eruptions to forecast eruption 4",
        #     "train_start_date": "2025-01-01",
        #     "train_end_date": "2025-05-31",
        #     "prediction_start_date": "2025-06-01",
        #     "prediction_end_date": "2025-06-30",
        # },
        # {
        #     "name": "Scenario 7",
        #     "description": "Training using 1, 2, 3 and 4 eruption to forecast 5",
        #     "train_start_date": "2025-01-01",
        #     "train_end_date": "2025-06-30",
        #     "prediction_start_date": "2025-07-01",
        #     "prediction_end_date": "2025-07-13",
        # },
        {
            "name": "Scenario 8",
            "description": "Training using 1, 2, 3, 4 and 5 eruption to forecast 6, 7",
            "train_start_date": "2025-01-01",
            "train_end_date": "2025-07-26",
            "prediction_start_date": "2025-07-27",
            "prediction_end_date": "2025-08-22",
            "prediction_parameter": PREDICTION_PARAMETERS,
            "plot_kwargs": {
                "legend_n_cols": 4,
            },
        },
        # {
        #     "name": "Scenario 9",
        #     "description": "Training using ALL",
        #     "train_start_date": "2025-01-01",
        #     "train_end_date": "2025-08-22",
        #     "prediction_start_date": "2025-01-01",
        #     "prediction_end_date": "2025-08-22",
        #     "prediction_parameter": {
        #         "window_step": LABEL_PARAMETERS["window_step"],
        #         "window_step_unit": LABEL_PARAMETERS["window_step_unit"],
        #     },
        #     "plot_kwargs": {
        #         "rolling_window": "12h",
        #         "x_days_interval": 14,
        #         "legend_n_cols": 4,
        #     },
        # },
        # {
        #     "name": "Scenario 10",
        #     "description": "Training using ALL",
        #     "train_start_date": "2025-01-01",
        #     "train_end_date": "2025-07-26",
        #     "prediction_start_date": "2025-07-27",
        #     "prediction_end_date": "2025-08-22",
        #     "prediction_parameter": {
        #         "window_step": LABEL_PARAMETERS["window_step"],
        #         "window_step_unit": LABEL_PARAMETERS["window_step_unit"],
        #     },
        #     "plot_kwargs": {
        #         "rolling_window": "12h",
        #         "x_days_interval": 2,
        #         "legend_n_cols": 4,
        #     },
        # },
    ]

    for scenario in _scenarios:
        name = scenario["name"]
        prediction_parameter = scenario.get("prediction_parameter", {})
        plot_kwargs = scenario.get("plot_kwargs", {})

        rolling_window = (
            scenario["plot_kwargs"]["rolling_window"]  # ty:ignore[invalid-argument-type]
            if hasattr(plot_kwargs, "rolling_window")
            else "6h"  # ty:ignore[invalid-argument-type]
        )
        window_step, window_step_unit = (
            (
                prediction_parameter["window_step"],  # ty:ignore[invalid-argument-type]
                prediction_parameter["window_step_unit"],  # ty:ignore[invalid-argument-type]
            )
            if len(prediction_parameter) > 0
            else (
                LABEL_PARAMETERS["window_step"],
                LABEL_PARAMETERS["window_step_unit"],
            )
        )
        window = f"{window_step} {window_step_unit}"
        window_train = (
            f"{LABEL_PARAMETERS['window_step']} {LABEL_PARAMETERS['window_step_unit']}"
        )

        title = (
            f"{name} - Plot Rolling Window: {rolling_window}.\n "
            f"Training date: {scenario['train_start_date']} - {scenario['train_end_date']}. "
            f"Training Window Size: {window_train}.\n"
            f"Prediction date: {scenario['prediction_start_date']} - {scenario['prediction_end_date']} "
            f"Prediction Window Size: {window}"
        )

        print(f"\n\n\n[workflow] Scenario {name}: {scenario['description']}\n\n\n")
        output_dir = os.path.join(
            ROOT_DIR,
            "output",
            forecast_model.nslc,
            "scenarios",
            slugify(name),  # ty:ignore[invalid-argument-type]
        )

        plot_forecast_path = predict(
            forecast_model=forecast_model,
            training_start_date=scenario["train_start_date"],
            training_end_date=scenario["train_end_date"],
            prediction_start_date=scenario["prediction_start_date"],
            prediction_end_date=scenario["prediction_end_date"],
            title=title,
            output_dir=output_dir,
            **scenario.get("prediction_parameter", PREDICTION_PARAMETERS),  # ty:ignore[invalid-argument-type]
            **scenario.get("plot_kwargs", {}),  # ty:ignore[invalid-argument-type]
        )

        if plot_forecast_path:
            send_telegram_notification(
                message=f"{scenario['name']}: {scenario['description']}",
                files=[plot_forecast_path],
                file_caption=title,  # ty:ignore[invalid-argument-type]
                send_as_document=True,
            )

    return None


@timer("Evaluation Model")
@notify("Evaluation Model")
def evaluate(forecast_model: ForecastModel) -> None:
    """Run training, evaluation, and comparison stages of the pipeline.

    Trains the model on the full training dataset, then evaluates each
    trained classifier with MultiModelEvaluator and compares all classifiers
    with ClassifierComparator when more than one classifier is trained.

    Args:
        forecast_model (ForecastModel): A configured ForecastModel instance.
    """
    forecast_model.build_label(
        start_date=EVALUATION_START_DATE,
        end_date=EVALUATION_END_DATE,
        **LABEL_PARAMETERS,
    ).extract_features(**EXTRACT_FEATURES_PARAMETERS).train(**TRAINING_PARAMETERS)

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
                plot_shap=TRAINING_PARAMETERS["plot_shap"],
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


@timer("Model Prediction")
@notify("Model Prediction")
def predict(
    forecast_model: ForecastModel,
    training_start_date: str | None = None,
    training_end_date: str | None = None,
    prediction_start_date: str | None = None,
    prediction_end_date: str | None = None,
    output_dir: str | None = None,
    **plot_kwargs,
) -> str | None:
    """Run the prediction (no-evaluation) training and forecast stages.

    Builds labels using LabelBuilder, extracts features, trains on
    the full dataset without evaluation, runs forecast, and generates a report.

    Args:
        forecast_model (ForecastModel): A configured ForecastModel instance.
        training_start_date: str | None: The training start date for prediction.
        training_end_date: str | None: The training end date for prediction.
        prediction_start_date: str | None: The prediction start date for prediction.
        prediction_end_date: str | None: The prediction end date for prediction.
        output_dir: str | None: The output directory for prediction.
    """
    forecast_plot_path = (
        forecast_model.build_label(
            start_date=training_start_date or TRAINING_PREDICTION_START_DATE,
            end_date=training_end_date or TRAINING_PREDICTION_END_DATE,
            **LABEL_PARAMETERS,
        )
        .extract_features(**EXTRACT_FEATURES_PARAMETERS)
        .train(output_dir=output_dir, **TRAINING_PARAMETERS)
        .forecast(
            start_date=prediction_start_date or PREDICTION_START_DATE,
            end_date=prediction_end_date or PREDICTION_END_DATE,
            output_dir=output_dir,
            **plot_kwargs,
        )
        .forecast_plot_path
    )

    return forecast_plot_path


def main(
    root_dir: str | None = None,
    run_prediction: bool = True,
    run_evaluation: bool = False,
    run_scenarios: bool = False,
) -> None:
    """Execute the full eruption forecast workflow.

    Initialises a ForecastModel, calculates tremor data from SDS archive,
    runs the prediction pipeline, and then runs training with evaluation.
    """
    fm = ForecastModel(root_dir=root_dir or ROOT_DIR, **FORECAST_PARAMETERS)

    # Calculate Tremor
    fm.calculate(
        start_date=CALCULATE_START_DATE,
        end_date=CALCULATE_END_DATE,
        source="sds",
        sds_dir=SDS_DIR,
        plot_daily=True,
        save_plot=True,
        remove_outlier_method="maximum",
    )

    # Predict
    if run_prediction:
        predict(forecast_model=fm)

    # Train and evaluate
    if run_evaluation:
        evaluate(forecast_model=fm)

    # Run multiple forecast scenarios
    if run_scenarios:
        scenarios(forecast_model=fm)


if __name__ == "__main__":
    logger.info("Start forecasting..")
    main(
        run_prediction=False,
        run_evaluation=False,
        run_scenarios=True,
    )
    logger.info("Finish forecasting..")
